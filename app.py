import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import uuid
from collections import Counter
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import fitz
from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DEFAULT_CATEGORIES = (
    "Timetable",
    "Syllabus",
    "Course material",
    "Circular",
    "Internship",
    "Placement",
    "Examination",
    "Event",
    "Academic notice",
    "Notice",
    "Other",
)
VECTOR_SIZE = 2048
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
TOKEN_RE = re.compile(r"[a-z0-9]+")
USERNAME_RE = re.compile(r"[a-zA-Z0-9_.-]{3,40}\Z")
NOTICE_CATEGORIES = {
    "Academic notice",
    "Notice",
    "Circular",
    "Examination",
    "Event",
    "Internship",
    "Placement",
}
NOTICE_DATE_RE = re.compile(
    r"\b(?:20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]20\d{2}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\s*,?\s+20\d{2}"
    r"|[A-Za-z]{3,9}\s+\d{1,2},?\s+20\d{2})\b",
    re.IGNORECASE,
)
DATE_CUES = re.compile(
    r"\b(?:deadline|due date|last date|submit by|apply by|on or before|"
    r"registration closes|application closes|closes on|due on|extended to)\b",
    re.IGNORECASE,
)
NOTICE_ACTION_RE = re.compile(
    r"\b(?:submit|register|apply|upload|attend|bring|complete|contact|pay|"
    r"report|verify|fill|carry|prepare|collect|visit|enroll|enrol)\b",
    re.IGNORECASE,
)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "the",
    "to", "was", "what", "when", "where", "which", "who", "with",
}


def _tokens(text):
    return [word for word in TOKEN_RE.findall(text.lower()) if word not in STOP_WORDS]


def _embedding(text):
    counts = Counter(_tokens(text))
    vector = {}
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % VECTOR_SIZE
        vector[index] = vector.get(index, 0) + count
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm:
        vector = {index: value / norm for index, value in vector.items()}
    return vector


def _cosine(left, right):
    return sum(value * right.get(index, 0) for index, value in left.items())


def _chunk_pages(page_texts):
    chunks = []
    current = ""
    first_page = 1
    for page_number, text in page_texts:
        for paragraph in re.split(r"\n{2,}", text):
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if not paragraph:
                continue
            while paragraph:
                available = CHUNK_SIZE - len(current)
                if len(paragraph) <= available:
                    if not current:
                        first_page = page_number
                    current = f"{current} {paragraph}".strip()
                    paragraph = ""
                else:
                    split_at = paragraph.rfind(" ", 0, max(1, available))
                    if split_at < available // 2:
                        split_at = available
                    piece, paragraph = paragraph[:split_at], paragraph[split_at:].strip()
                    if not current:
                        first_page = page_number
                    current = f"{current} {piece}".strip()
                    chunks.append((current, first_page))
                    current = current[-CHUNK_OVERLAP:]
                    first_page = page_number
        if current:
            chunks.append((current, first_page))
            current = ""
    return chunks


def _database():
    if "db" not in g:
        database_path = Path(current_app_config()["DATABASE"])
        database_path.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(database_path)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def current_app_config():
    from flask import current_app

    return current_app.config


def _initialize_database(app):
    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(app.config["DATABASE"])) as db, db:
        db.execute("PRAGMA foreign_keys = ON")
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                original_name TEXT NOT NULL,
                stored_name TEXT NOT NULL UNIQUE,
                uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                semantic_embedding TEXT
            );
            CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks(document_id);
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS notice_analysis (
                document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                deadlines_json TEXT NOT NULL,
                checklist_json TEXT NOT NULL,
                missing_json TEXT NOT NULL,
                analyzed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        document_columns = {
            row[1] for row in db.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "is_notice" not in document_columns:
            db.execute(
                "ALTER TABLE documents ADD COLUMN is_notice INTEGER NOT NULL DEFAULT 0"
            )
        db.execute(
            """
            UPDATE documents SET is_notice = 1
            WHERE category IN (?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(sorted(NOTICE_CATEGORIES)),
        )
        columns = {
            row[1] for row in db.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "semantic_embedding" not in columns:
            db.execute("ALTER TABLE chunks ADD COLUMN semantic_embedding TEXT")


def _admin_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            flash("Please sign in as an administrator to continue.", "error")
            return redirect(url_for("admin_login"))
        return view(*args, **kwargs)

    return wrapped


def _user_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin") and not session.get("user_id"):
            flash("Please sign in to search college documents.", "error")
            return redirect(url_for("user_login", next=request.full_path))
        return view(*args, **kwargs)

    return wrapped


def _safe_redirect_target(target, fallback):
    if not target or "\\" in target:
        return fallback
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/") or parsed.path.startswith("//"):
        return fallback
    return target


def _answer_sentences(query, chunks):
    query_tokens = set(_tokens(query))
    ranked = []
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk["content"]):
            sentence = sentence.strip()
            sentence_tokens = set(_tokens(sentence))
            overlap = len(query_tokens & sentence_tokens)
            if overlap:
                ranked.append((overlap / math.sqrt(max(1, len(sentence_tokens))), sentence))
    ranked.sort(key=lambda result: result[0], reverse=True)
    answer, seen = [], set()
    for _, sentence in ranked:
        key = sentence.casefold()
        if key not in seen:
            answer.append(sentence)
            seen.add(key)
        if len(answer) == 3:
            break
    return " ".join(answer)


def _search(query):
    query_embedding = _embedding(query)
    query_tokens = set(_tokens(query))
    if not query_embedding or not query_tokens:
        return []
    rows = _database().execute(
        """
        SELECT chunks.id, chunks.page_number, chunks.content, chunks.embedding,
               documents.id AS document_id, documents.title, documents.category,
               documents.original_name
        FROM chunks JOIN documents ON documents.id = chunks.document_id
        """
    ).fetchall()
    results = []
    for row in rows:
        embedding = {int(key): value for key, value in json.loads(row["embedding"]).items()}
        similarity = _cosine(query_embedding, embedding)
        words = set(_tokens(row["content"]))
        coverage = len(query_tokens & words) / len(query_tokens)
        score = similarity * 0.75 + coverage * 0.25
        if score >= 0.12:
            results.append({**dict(row), "score": score})
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:5]


class AIServiceError(Exception):
    pass


def _ai_configured(config):
    key = config["AI_API_KEY"].strip()
    if not key or key == "YOUR_API_KEY_HERE":
        return False
    return config["AI_PROVIDER"] == "gemini" or bool(config["AI_BASE_URL"].strip())


def _api_url(config, endpoint):
    base_url = config["AI_BASE_URL"].strip().rstrip("/")
    parsed_url = urlsplit(base_url)
    if (
        parsed_url.scheme != "https"
        and not (
            parsed_url.scheme == "http"
            and parsed_url.hostname in {"localhost", "127.0.0.1", "::1"}
        )
    ) or not parsed_url.netloc or parsed_url.username or parsed_url.password:
        raise AIServiceError(
            "AI endpoint must use HTTPS (HTTP is allowed only for localhost)."
        )
    if not parsed_url.path.rstrip("/").endswith("/v1"):
        base_url += "/v1"
    return f"{base_url}/{endpoint}"


def _post_ai_json(config, endpoint, payload):
    if config["AI_PROVIDER"] == "gemini":
        base_url = (
            config["GEMINI_BASE_URL"].strip().rstrip("/")
            or "https://generativelanguage.googleapis.com/v1beta"
        )
        parsed_url = urlsplit(base_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise AIServiceError("Gemini API endpoint must use HTTPS.")
        if endpoint == "embeddings":
            model = payload["model"].removeprefix("models/")
            url = f"{base_url}/models/{model}:batchEmbedContents"
            body = {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": text}]},
                        "taskType": payload.get("task_type", "RETRIEVAL_DOCUMENT"),
                    }
                    for text in payload["texts"]
                ]
            }
        else:
            model = payload["model"].removeprefix("models/")
            url = f"{base_url}/models/{model}:generateContent"
            messages = payload["messages"]
            system_instruction = next(
                (message["content"] for message in messages if message["role"] == "system"),
                "",
            )
            body = {
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "contents": [
                    {
                        "role": "model" if message["role"] == "assistant" else "user",
                        "parts": [{"text": message["content"]}],
                    }
                    for message in messages
                    if message["role"] != "system"
                ],
                "generationConfig": {"temperature": payload.get("temperature", 0)},
            }
        api_request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "x-goog-api-key": config["AI_API_KEY"].strip(),
                "Content-Type": "application/json",
            },
            method="POST",
        )
    else:
        api_request = Request(
            _api_url(config, endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {config['AI_API_KEY'].strip()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
    try:
        with urlopen(api_request, timeout=35) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code in {401, 403}:
            raise AIServiceError(
                "AI provider rejected the key or model. Check the provider's enabled models."
            ) from None
        raise AIServiceError(
            f"AI provider request failed (HTTP {error.code})."
        ) from None
    except (URLError, TimeoutError):
        raise AIServiceError("Could not connect to the AI provider.") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise AIServiceError("AI provider returned an invalid response.") from None


def _generate_embeddings(texts, config, task_type="RETRIEVAL_DOCUMENT"):
    if not texts:
        return []
    if config["AI_PROVIDER"] == "gemini":
        data = _post_ai_json(
            config,
            "embeddings",
            {
                "model": config["AI_EMBEDDING_MODEL"],
                "texts": texts,
                "task_type": task_type,
            },
        )
        try:
            vectors = [item["values"] for item in data["embeddings"]]
            if len(vectors) != len(texts):
                raise ValueError
            if any(
                not vector
                or any(not isinstance(value, (int, float)) for value in vector)
                for vector in vectors
            ):
                raise ValueError
            return vectors
        except (KeyError, TypeError, ValueError):
            raise AIServiceError("Gemini returned invalid embeddings.") from None
    data = _post_ai_json(
        config,
        "embeddings",
        {"model": config["AI_EMBEDDING_MODEL"], "input": texts},
    )
    try:
        vectors = [None] * len(texts)
        for item in data["data"]:
            index = item["index"]
            vector = item["embedding"]
            if (
                not isinstance(index, int)
                or index < 0
                or index >= len(texts)
                or not isinstance(vector, list)
                or not vector
                or any(not isinstance(value, (int, float)) for value in vector)
            ):
                raise ValueError
            vectors[index] = vector
        if any(vector is None for vector in vectors):
            raise ValueError
        return vectors
    except (KeyError, TypeError, ValueError):
        raise AIServiceError("AI provider returned invalid embeddings.") from None


def _ensure_semantic_embeddings(config):
    rows = _database().execute(
        "SELECT id, content FROM chunks WHERE semantic_embedding IS NULL ORDER BY id"
    ).fetchall()
    for offset in range(0, len(rows), 32):
        batch = rows[offset:offset + 32]
        vectors = _generate_embeddings([row["content"] for row in batch], config)
        _database().executemany(
            "UPDATE chunks SET semantic_embedding = ? WHERE id = ?",
            [
                (json.dumps(vector, separators=(",", ":")), row["id"])
                for row, vector in zip(batch, vectors)
            ],
        )
        _database().commit()


def _semantic_search(query, query_vector):
    query_tokens = set(_tokens(query))
    query_embedding = _embedding(query)
    rows = _database().execute(
        """
        SELECT chunks.id, chunks.page_number, chunks.content, chunks.embedding,
               chunks.semantic_embedding, documents.id AS document_id,
               documents.title, documents.category, documents.original_name
        FROM chunks JOIN documents ON documents.id = chunks.document_id
        WHERE chunks.semantic_embedding IS NOT NULL
        """
    ).fetchall()
    results = []
    for row in rows:
        stored_vector = json.loads(row["semantic_embedding"])
        semantic_score = _cosine(
            {index: value for index, value in enumerate(query_vector)},
            {index: value for index, value in enumerate(stored_vector)},
        )
        sparse_vector = {
            int(key): value for key, value in json.loads(row["embedding"]).items()
        }
        sparse_score = _cosine(query_embedding, sparse_vector)
        words = set(_tokens(row["content"]))
        coverage = len(query_tokens & words) / max(1, len(query_tokens))
        lexical_score = sparse_score * 0.75 + coverage * 0.25
        score = semantic_score * 0.85 + lexical_score * 0.15
        if semantic_score >= 0.22:
            results.append({**dict(row), "score": score})
    results.sort(key=lambda item: item["score"], reverse=True)
    title_matches = {
        result["document_id"]
        for result in results
        if query_tokens & set(_tokens(f"{result['title']} {result['category']}"))
    }
    if title_matches:
        matching_documents = [
            result for result in results
            if result["document_id"] in title_matches
        ]
        if matching_documents and matching_documents[0]["score"] >= results[0]["score"] * 0.55:
            results = matching_documents
    return results[:5]


def _generate_answer(query, chunks, history, config):
    excerpts = "\n\n".join(
        f"Source: {chunk['title']} (page {chunk['page_number']})\n"
        f"{chunk['content'][:1200]}"
        for chunk in chunks[:2]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a college document assistant. Answer from this context only. "
                "If the answer is present, give it and cite source/page. If absent, "
                "say: not found."
            ),
        }
    ]
    for turn in history[-8:]:
        messages.append({"role": turn["role"], "content": turn["content"][:600]})
    messages.append(
        {
            "role": "user",
            "content": f"Question: {query}\n\nRetrieved document excerpts:\n{excerpts}",
        }
    )
    data = _post_ai_json(
        config,
        "chat/completions",
        {
            "model": config["AI_MODEL"],
            "messages": messages,
            "temperature": 0,
        },
    )
    try:
        if config["AI_PROVIDER"] == "gemini":
            answer = "".join(
                part.get("text", "")
                for part in data["candidates"][0]["content"]["parts"]
            ).strip()
        else:
            answer = data["choices"][0]["message"]["content"].strip()
        if not answer:
            raise ValueError
        if (
            answer.casefold().startswith("not found")
            or answer.casefold().startswith("i couldn't find")
            or answer.casefold().startswith("sorry, i couldn't find")
        ):
            return "not found"
        return answer
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        raise AIServiceError("AI provider returned an empty or invalid answer.") from None


def _answer_question(query, history, config):
    notice = None
    results = []
    previous_questions = [
        turn["content"] for turn in history[-8:]
        if turn["role"] == "user"
    ]
    retrieval_query = " ".join(previous_questions[-3:] + [query])
    if _ai_configured(config):
        try:
            _ensure_semantic_embeddings(config)
            query_vector = _generate_embeddings(
                [retrieval_query], config, task_type="RETRIEVAL_QUERY"
            )[0]
            results = _semantic_search(retrieval_query, query_vector)
        except AIServiceError as error:
            notice = f"{error} Using local document search instead."
            results = _search(retrieval_query)
    else:
        notice = "AI is not configured; showing the document-based answer."
        results = _search(retrieval_query)

    if not results:
        return (
            "Sorry, I couldn't find this information in the uploaded college documents.",
            [],
            notice,
            False,
        )

    if _ai_configured(config):
        try:
            answer = _generate_answer(query, results, history, config)
            if answer == "not found":
                return (
                    "Sorry, I couldn't find this information in the uploaded college documents.",
                    [],
                    notice,
                    True,
                )
            return answer, results, notice, True
        except AIServiceError as error:
            notice = f"{error} Showing the document-based answer."
    answer = _answer_sentences(query, results)
    if not answer:
        answer = results[0]["content"][:700].strip()
    return answer, results, notice, False


def _parse_notice_date(value):
    normalized = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", value.strip(), flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).replace(",", "")
    formats = (
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
    )
    for date_format in formats:
        try:
            return datetime.strptime(normalized, date_format).date()
        except ValueError:
            continue
    return None


def _notice_reminder(target_date, today):
    days_until = (target_date - today).days
    if days_until < 0:
        return "This date has passed."
    if days_until == 0:
        return "Reminder: due today."
    if days_until == 1:
        return "Reminder: due tomorrow."
    if days_until <= 7:
        return f"Reminder: due in {days_until} days."
    return f"Upcoming: due in {days_until} days."


def _analyze_notice(chunks, today=None):
    today = today or date.today()
    full_text = "\n".join(chunk["content"] for chunk in chunks)
    dates = {}
    for chunk in chunks:
        content = chunk["content"]
        for match in NOTICE_DATE_RE.finditer(content):
            parsed = _parse_notice_date(match.group())
            if parsed is None:
                continue
            start = max(0, match.start() - 100)
            end = min(len(content), match.end() + 100)
            context = content[start:end].strip()
            is_deadline = bool(DATE_CUES.search(context))
            kind = "Deadline" if is_deadline else "Important date"
            key = (parsed.isoformat(), kind)
            dates.setdefault(
                key,
                {
                    "date": parsed.isoformat(),
                    "display_date": parsed.strftime("%B %-d, %Y")
                    if os.name != "nt"
                    else parsed.strftime("%B %d, %Y").replace(" 0", " "),
                    "kind": kind,
                    "status": "Expired" if parsed < today else "Upcoming",
                    "reminder": _notice_reminder(parsed, today),
                    "page": chunk["page_number"],
                    "context": context,
                },
            )

    checklist = []
    seen_tasks = set()
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk["content"]):
            sentence = re.sub(r"\s+", " ", sentence).strip(" \t-•")
            if (
                12 <= len(sentence) <= 220
                and NOTICE_ACTION_RE.search(sentence)
                and sentence.casefold() not in seen_tasks
            ):
                checklist.append({"task": sentence, "page": chunk["page_number"]})
                seen_tasks.add(sentence.casefold())
                if len(checklist) >= 12:
                    break
        if len(checklist) >= 12:
            break

    fields = {
        "Eligibility": (
            r"\b(?:eligib\w*|who can apply|qualification|qualifications|"
            r"criteria|open to)\b",
        ),
        "Venue or location": (
            r"\b(?:venue|location|address|auditorium|room\s*(?:no\.?|number)?|"
            r"hall\s*(?:no\.?|number)?|campus)\b",
        ),
        "Required documents": (
            r"\b(?:required documents|documents required|documents to be submitted|"
            r"supporting documents|bring the following|"
            r"required certificates|ID proof)\b",
        ),
        "Contact information": (
            r"\b(?:contact|email|e-mail|phone|helpline|coordinator|reach us)\b|@",
        ),
    }
    missing = [
        {"field": label, "message": f"No {label.lower()} found in this notice."}
        for label, patterns in fields.items()
        if not any(re.search(pattern, full_text, re.IGNORECASE) for pattern in patterns)
    ]
    if DATE_CUES.search(full_text) and not any(
        entry["kind"] == "Deadline" for entry in dates.values()
    ):
        missing.append(
            {
                "field": "Deadline date",
                "message": "A deadline is mentioned, but a complete date with year was not found.",
            }
        )

    return {
        "dates": sorted(dates.values(), key=lambda item: item["date"]),
        "checklist": checklist,
        "missing": missing,
    }


def _notices_for_page():
    documents = _database().execute(
        "SELECT id, title, category, original_name, uploaded_at "
        "FROM documents WHERE is_notice = 1 ORDER BY uploaded_at DESC"
    ).fetchall()
    notices = []
    for document in documents:
        chunks = _database().execute(
            "SELECT page_number, content FROM chunks "
            "WHERE document_id = ? ORDER BY id",
            (document["id"],),
        ).fetchall()
        analysis = _analyze_notice(chunks)
        _database().execute(
            """
            INSERT INTO notice_analysis
                (document_id, deadlines_json, checklist_json, missing_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                deadlines_json = excluded.deadlines_json,
                checklist_json = excluded.checklist_json,
                missing_json = excluded.missing_json,
                analyzed_at = CURRENT_TIMESTAMP
            """,
            (
                document["id"],
                json.dumps(analysis["dates"]),
                json.dumps(analysis["checklist"]),
                json.dumps(analysis["missing"]),
            ),
        )
        notices.append({**dict(document), **analysis})
    _database().commit()
    return notices


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("CAMPUSASSIST_SECRET_KEY") or secrets.token_hex(32),
        ADMIN_USERNAME=os.environ.get("CAMPUSASSIST_ADMIN_USERNAME", ""),
        ADMIN_PASSWORD=os.environ.get("CAMPUSASSIST_ADMIN_PASSWORD", ""),
        AI_API_KEY=os.environ.get("CAMPUSASSIST_AI_API_KEY", ""),
        AI_BASE_URL=os.environ.get("CAMPUSASSIST_AI_BASE_URL", ""),
        AI_PROVIDER=os.environ.get("CAMPUSASSIST_AI_PROVIDER", "nexus").lower(),
        GEMINI_BASE_URL=os.environ.get("CAMPUSASSIST_GEMINI_BASE_URL", ""),
        AI_MODEL=os.environ.get("CAMPUSASSIST_AI_MODEL", "nova-micro"),
        AI_EMBEDDING_MODEL=os.environ.get(
            "CAMPUSASSIST_AI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        GEMINI_CHAT_MODEL=os.environ.get(
            "CAMPUSASSIST_GEMINI_CHAT_MODEL", "gemini-2.5-flash"
        ),
        GEMINI_EMBEDDING_MODEL=os.environ.get(
            "CAMPUSASSIST_GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        ),
        DATABASE=str(Path(app.instance_path) / "campusassist.sqlite3"),
        UPLOAD_FOLDER=str(Path(app.instance_path) / "documents"),
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
    )
    if test_config:
        app.config.update(test_config)
    if app.config["AI_PROVIDER"] == "gemini":
        if not test_config or "AI_MODEL" not in test_config:
            app.config["AI_MODEL"] = app.config["GEMINI_CHAT_MODEL"]
        if not test_config or "AI_EMBEDDING_MODEL" not in test_config:
            app.config["AI_EMBEDDING_MODEL"] = app.config["GEMINI_EMBEDDING_MODEL"]
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    _initialize_database(app)

    @app.teardown_appcontext
    def close_database(_error=None):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.errorhandler(413)
    def too_large(_error):
        flash("The PDF is too large. The maximum upload size is 20 MB.", "error")
        return redirect(url_for("admin_dashboard"))

    @app.get("/")
    def home():
        documents = _database().execute(
            "SELECT category, COUNT(*) AS count FROM documents GROUP BY category"
        ).fetchall()
        return render_template("home.html", categories=documents)

    @app.get("/student")
    @_user_required
    def student_dashboard():
        documents = _database().execute(
            "SELECT id, title, category, uploaded_at FROM documents ORDER BY uploaded_at DESC LIMIT 6"
        ).fetchall()
        notices_count = _database().execute(
            "SELECT COUNT(*) FROM documents WHERE is_notice = 1"
        ).fetchone()[0]
        return render_template(
            "student.html", documents=documents, notices_count=notices_count
        )

    @app.get("/notices")
    @_user_required
    def notices():
        return render_template("notices.html", notices=_notices_for_page())

    @app.route("/search", methods=["GET", "POST"])
    @_user_required
    def search():
        query = (request.values.get("q") or "").strip()[:300]
        if request.method == "POST":
            return redirect(url_for("search", q=query))
        if not query:
            return render_template(
                "search_results.html",
                query="",
                results=[],
                answer="",
                ai_notice=None,
                ai_used=False,
            )
        answer, results, ai_notice, ai_used = _answer_question(
            query, [], app.config
        )
        return render_template(
            "search_results.html",
            query=query,
            results=results,
            answer=answer,
            ai_notice=ai_notice,
            ai_used=ai_used,
        )

    @app.get("/chat")
    @_user_required
    def chat():
        return render_template("chat.html")

    @app.post("/api/chat")
    @_user_required
    def chat_api():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON request with a question."), 400
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            return jsonify(error="Enter a question to continue."), 400
        question = question.strip()
        if len(question) > 500:
            return jsonify(error="Questions must be 500 characters or fewer."), 400

        raw_history = payload.get("history", [])
        if not isinstance(raw_history, list):
            return jsonify(error="Chat history must be a list of messages."), 400
        history = []
        for turn in raw_history[-8:]:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                history.append({"role": role, "content": content.strip()[:600]})

        answer, results, notice, ai_used = _answer_question(
            question, history, app.config
        )
        return jsonify(
            answer=answer,
            ai_used=ai_used,
            notice=notice,
            sources=[
                {
                    "title": result["title"],
                    "page": result["page_number"],
                    "excerpt": result["content"][:650],
                    "url": url_for(
                        "document_file", document_id=result["document_id"]
                    ),
                }
                for result in results
            ],
        )

    @app.get("/admin/login")
    def admin_login():
        return render_template(
            "admin_login.html",
            configured=bool(app.config["ADMIN_USERNAME"] and app.config["ADMIN_PASSWORD"]),
        )

    @app.post("/admin/login")
    def admin_login_post():
        username = app.config["ADMIN_USERNAME"]
        password = app.config["ADMIN_PASSWORD"]
        supplied_username = request.form.get("username", "").strip()
        supplied = request.form.get("password", "")
        if not username or not password:
            flash(
                "Admin login is not configured. Set CAMPUSASSIST_ADMIN_USERNAME and CAMPUSASSIST_ADMIN_PASSWORD, then restart.",
                "error",
            )
        elif (
            secrets.compare_digest(supplied_username.casefold(), username.casefold())
            and secrets.compare_digest(supplied, password)
        ):
            session.clear()
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            flash("The administrator username or password is incorrect.", "error")
        return redirect(url_for("admin_login"))

    @app.post("/admin/logout")
    def admin_logout():
        session.clear()
        flash("You have been signed out.", "success")
        return redirect(url_for("home"))

    @app.route("/login", methods=["GET", "POST"])
    def user_login():
        next_url = request.values.get("next", "")
        if request.method == "POST":
            username = request.form.get("username", "").strip().casefold()
            password = request.form.get("password", "")
            user = _database().execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if user and check_password_hash(user["password_hash"], password):
                session.clear()
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                return redirect(
                    _safe_redirect_target(next_url, url_for("student_dashboard"))
                )
            flash("The username or password is incorrect.", "error")
        return render_template("user_login.html", next_url=next_url)

    @app.route("/register", methods=["GET", "POST"])
    def user_register():
        next_url = request.values.get("next", "")
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            confirmation = request.form.get("confirm_password", "")
            if not USERNAME_RE.fullmatch(username):
                flash(
                    "Choose a username with 3-40 letters, numbers, dots, underscores, or hyphens.",
                    "error",
                )
            elif len(password) < 8 or len(password) > 128:
                flash("Your password must be between 8 and 128 characters.", "error")
            elif password != confirmation:
                flash("The passwords do not match.", "error")
            else:
                try:
                    cursor = _database().execute(
                        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                        (username, generate_password_hash(password)),
                    )
                    _database().commit()
                except sqlite3.IntegrityError:
                    flash("That username is already registered.", "error")
                else:
                    session.clear()
                    session["user_id"] = cursor.lastrowid
                    session["username"] = username
                    flash("Your account is ready. You are now signed in.", "success")
                    return redirect(
                        _safe_redirect_target(next_url, url_for("student_dashboard"))
                    )
        return render_template("user_register.html", next_url=next_url)

    @app.post("/logout")
    def user_logout():
        session.clear()
        flash("You have been signed out.", "success")
        return redirect(url_for("home"))

    @app.get("/admin")
    @_admin_required
    def admin_dashboard():
        documents = _database().execute(
            """
            SELECT documents.*, COUNT(chunks.id) AS chunk_count
            FROM documents LEFT JOIN chunks ON chunks.document_id = documents.id
            GROUP BY documents.id ORDER BY documents.uploaded_at DESC
            """
        ).fetchall()
        query = (request.args.get("q") or "").strip()[:200]
        if query:
            documents = [
                document for document in documents
                if query.casefold() in document["title"].casefold()
                or query.casefold() in document["original_name"].casefold()
                or query.casefold() in document["category"].casefold()
            ]
        return render_template(
            "admin_dashboard.html",
            documents=documents,
            query=query,
            categories=DEFAULT_CATEGORIES,
        )

    @app.get("/admin/notices")
    @_admin_required
    def admin_notices():
        documents = _database().execute(
            """
            SELECT documents.*, COUNT(chunks.id) AS chunk_count
            FROM documents LEFT JOIN chunks ON chunks.document_id = documents.id
            WHERE documents.is_notice = 1
            GROUP BY documents.id ORDER BY documents.uploaded_at DESC
            """
        ).fetchall()
        return render_template("admin_notices.html", documents=documents)

    @app.post("/admin/documents")
    @_admin_required
    def upload_document():
        title = request.form.get("title", "").strip()[:160]
        category = request.form.get("category", "").strip()
        uploaded = request.files.get("document")
        if not title or category not in DEFAULT_CATEGORIES or not uploaded or not uploaded.filename:
            flash("Enter a title, choose a category, and select a PDF.", "error")
            return redirect(url_for("admin_dashboard"))
        original_name = secure_filename(uploaded.filename)
        if not original_name.lower().endswith(".pdf"):
            flash("Only PDF documents can be uploaded.", "error")
            return redirect(url_for("admin_dashboard"))

        stored_name = f"{uuid.uuid4().hex}.pdf"
        destination = Path(app.config["UPLOAD_FOLDER"]) / stored_name
        is_notice = (
            request.form.get("is_notice") == "1"
            or category in NOTICE_CATEGORIES
        )
        try:
            uploaded.save(destination)
            with fitz.open(destination) as pdf:
                if pdf.is_encrypted:
                    raise ValueError("This PDF is password-protected.")
                page_texts = [(index + 1, page.get_text()) for index, page in enumerate(pdf)]
            chunks = _chunk_pages(page_texts)
            if not any(_tokens(text) for text, _ in chunks):
                raise ValueError("No searchable text was found. Scanned PDFs need OCR before upload.")
            semantic_embeddings = [None] * len(chunks)
            embedding_notice = None
            if _ai_configured(app.config):
                try:
                    semantic_embeddings = _generate_embeddings(
                        [text for text, _ in chunks], app.config
                    )
                except AIServiceError as error:
                    embedding_notice = (
                        f" The local search index is ready; semantic embeddings "
                        f"could not be generated: {error}"
                    )
            with closing(sqlite3.connect(app.config["DATABASE"])) as db, db:
                db.execute("PRAGMA foreign_keys = ON")
                cursor = db.execute(
                    "INSERT INTO documents (title, category, original_name, stored_name, is_notice) VALUES (?, ?, ?, ?, ?)",
                    (title, category, original_name, stored_name, int(is_notice)),
                )
                db.executemany(
                    "INSERT INTO chunks (document_id, page_number, content, embedding, semantic_embedding) VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            cursor.lastrowid,
                            page,
                            content,
                            json.dumps(_embedding(content)),
                            json.dumps(semantic_embedding, separators=(",", ":"))
                            if semantic_embedding
                            else None,
                        )
                        for (content, page), semantic_embedding
                        in zip(chunks, semantic_embeddings)
                    ],
                )
        except (fitz.FileDataError, fitz.EmptyFileError, ValueError) as error:
            destination.unlink(missing_ok=True)
            flash(f"Could not process this PDF: {error}", "error")
            return redirect(url_for("admin_dashboard"))
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        flash(
            "PDF uploaded and indexed successfully."
            + (embedding_notice or ""),
            "success",
        )
        return redirect(url_for("admin_dashboard"))

    @app.post("/admin/documents/<int:document_id>/delete")
    @_admin_required
    def delete_document(document_id):
        db = _database()
        document = db.execute(
            "SELECT stored_name FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if document is None:
            abort(404)
        db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        db.commit()
        (Path(app.config["UPLOAD_FOLDER"]) / document["stored_name"]).unlink(missing_ok=True)
        flash("Document deleted.", "success")
        return redirect(url_for("admin_dashboard"))

    @app.get("/documents/<int:document_id>/file")
    @_user_required
    def document_file(document_id):
        document = _database().execute(
            "SELECT original_name, stored_name FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if document is None:
            abort(404)
        path = Path(app.config["UPLOAD_FOLDER"]) / document["stored_name"]
        if not path.is_file():
            abort(404)
        return send_file(
            path,
            mimetype="application/pdf",
            as_attachment=request.args.get("download") == "1",
            download_name=document["original_name"],
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
