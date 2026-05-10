# Tennis Doubles Match Analyzer

> Upload a doubles match video, tag all 4 players by court position, and get AI-powered coaching feedback for every player — powered by **Google Gemini 2.5 Flash**.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange?logo=google)

---

## Features

- **Video upload up to 500MB** via Google Gemini File API
- **Visual court diagram** — tag all 4 players by position before analysis
- **Per-player report cards** — overall rating + 3 strengths + 3 improvements
- **Team comparison** — Team 1 (FL+FR) vs Team 2 (BL+BR) average scores
- **Match history** — all analyses saved to SQLite, viewable anytime
- **Dark UI** — clean, mobile-friendly interface

---

## Project Structure

```
tennis_analyzer/
├── app.py              # Flask routes
├── database.py         # SQLite helpers
├── gemini_client.py    # Gemini File API + generateContent
├── templates/
│   └── index.html      # Single-page web UI
├── requirements.txt
├── Procfile
├── render.yaml
├── runtime.txt
├── .env.example
└── README.md
```

---

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/tennis-match-analyzer.git
cd tennis-match-analyzer

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

python app.py
```

Open **http://localhost:5000**

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | — | Google AI Studio API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model name |
| `DB_PATH` | No | `tennis_matches.db` | SQLite database path |

Get a free Gemini API key at **https://aistudio.google.com/app/apikey**

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo (Render auto-detects `render.yaml`)
4. Add environment variable: `GEMINI_API_KEY` = your key
5. Click **Create Web Service**

> **Note:** Render's free tier has an ephemeral filesystem. The SQLite DB is stored at `/tmp/tennis_matches.db` and will reset on redeploy. For persistent history, upgrade to a paid tier or use a hosted database.

---

## How It Works

1. Video uploaded to `/tmp/tennis_uploads/` on the server
2. Uploaded to Google Gemini File API (handles up to 2GB)
3. Gemini polls until video state is `ACTIVE`
4. `generateContent` called with video + structured prompt
5. File deleted from Gemini File API + local disk
6. Results saved to SQLite and returned to browser

---

## License

MIT
