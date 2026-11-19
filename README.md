# CampusAssist

A lightweight college-document portal built with Flask, SQLite and PyMuPDF. Administrators upload PDFs once; files and their searchable text index are kept in the local `instance/` folder between restarts. Registered users can search extracted passages and open or download the original PDF.

Search retrieves matching PDF passages from the existing SQLite index. The **AI Chat** page and search-results page use Gemini embeddings and grounded Gemini answers when configured. Existing Nexus/OpenAI-compatible credentials are also supported. Scanned/image-only PDFs need OCR before they can be searched.

## Requirements

- Python 3.10 or later
- Windows PowerShell (commands below use PowerShell syntax)

## Install and run

From the project directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
$env:CAMPUSASSIST_ADMIN_PASSWORD = "choose-a-long-private-password"
$env:CAMPUSASSIST_ADMIN_USERNAME = "campusadmin"
$env:CAMPUSASSIST_SECRET_KEY = python -c "import secrets; print(secrets.token_hex(32))"
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Set both admin credentials in the current PowerShell window and use them at **Admin login**. Users can create accounts through **Create account** and then use the visible **Open AI Chat** button or **Notices** page. AI settings are read from `.env`. For direct Gemini access, set `CAMPUSASSIST_AI_PROVIDER=gemini`, use a Google AI Studio key, and configure `CAMPUSASSIST_GEMINI_BASE_URL`; the direct Gemini API uses `generateContent`, `batchEmbedContents`, and `x-goog-api-key`. Never commit `.env` or paste a real API key into source code. A Gemini endpoint setting is kept separate from `CAMPUSASSIST_AI_BASE_URL`, which is used by OpenAI-compatible providers. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, then activate the environment.

Alternatively, run without activating the environment:

```powershell
py -3 -m pip install -r requirements.txt
$env:CAMPUSASSIST_ADMIN_PASSWORD = "choose-a-long-private-password"
$env:CAMPUSASSIST_ADMIN_USERNAME = "campusadmin"
$env:CAMPUSASSIST_SECRET_KEY = python -c "import secrets; print(secrets.token_hex(32))"
py -3 app.py
```

## Use

1. Sign in at **Admin login** using `CAMPUSASSIST_ADMIN_USERNAME` and `CAMPUSASSIST_ADMIN_PASSWORD`, then add a document title, category and PDF.
2. The app extracts text and indexes it immediately. Password-protected PDFs and PDFs without selectable text are rejected with an explanation.
3. Admins publish a PDF notice by choosing a notice category or checking **Publish on the student notice board**. Manage published notices at **Manage notices**.
4. Users create an account, sign in, and use **Open AI Chat**, search, and the **Notices** page. Passwords are stored as secure hashes.
5. AI Chat retrieves document chunks with semantic embeddings, answers only from those sources, retains recent follow-up context for the current chat, shows source/page links, and displays a loading state. If evidence is weak or absent, it uses the required not-found response.
6. The notice board extracts complete dates, marks past/upcoming dates, generates reminder text, extracts action-item checklists, and flags whether eligibility, venue, required documents, or contact details were not detected in the PDF text.
7. Admins can filter or delete documents/notices. Deleting a PDF also removes its chunks and saved notice analysis.

Gemini setup is shown in `.env.example`; model names are configurable. The existing Nexus configuration remains on its compatible provider unless changed. To use Gemini, obtain a Google AI Studio key, set `CAMPUSASSIST_AI_PROVIDER=gemini`, set `CAMPUSASSIST_GEMINI_BASE_URL` to Google's Gemini API URL, replace `CAMPUSASSIST_AI_API_KEY`, and restart. For Nexus, use `CAMPUSASSIST_AI_PROVIDER=nexus` and its compatible endpoint/model settings. Semantic vectors are stored with existing SQLite chunks; legacy chunks are embedded in batches when first needed. If AI is unavailable, local search/answers remain available with a status notice. Retrieved text and recent chat turns are sent to the configured provider.

Notices are identified by their notice category or the publish checkbox. Date extraction only creates calendar reminders from dates with an explicit year. The checklist and missing-detail checks are based on extracted notice text and should be verified against the source PDF.

Uploads are limited to 20 MB. Configure the admin account with `CAMPUSASSIST_ADMIN_USERNAME` and `CAMPUSASSIST_ADMIN_PASSWORD`; no default credentials are built in. The generated `instance/` folder contains the SQLite database, user accounts and uploaded files; back it up to preserve them. This built-in admin login is intended for a trusted, single-admin deployment. For public production hosting, use HTTPS and a production-grade WSGI server.

For local development with Flask's debugger, set `$env:FLASK_DEBUG = "1"` before starting. Do not enable debug mode on a public server.
