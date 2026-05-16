# Tennis Ball In/Out Detector

A mobile-first web app that uses your phone camera to record tennis ball landings, extracts frames around the impact point, and sends them to Google Gemini 2.5 Flash for AI-powered in/out analysis.

## How It Works

1. **Record** - Point your phone camera at a ball landing near a court line. Tap the record button to capture up to 3 seconds of video.
2. **Review** - The app extracts 5 evenly-spaced frames from your recording. Review them to confirm the ball landing is visible.
3. **Analyze** - Tap "Analyze Ball Position" to send the frames to Gemini AI, which determines whether the ball landed IN or OUT based on its position relative to the white court lines.
4. **Result** - Get a verdict with confidence percentage and explanation.

## Setup

### Prerequisites

- Python 3.11+
- A Google Gemini API key ([get one free](https://aistudio.google.com/app/apikey))

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the app
python app.py
```

Open http://localhost:5000 on your phone (must be on same network) or use a tunnel for HTTPS (required for camera access on mobile).

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required) | - |
| `GEMINI_MODEL` | Gemini model to use | `gemini-2.5-flash` |
| `DB_PATH` | SQLite database path | `tennis_calls.db` |
| `PORT` | Server port | `5000` |

## Deployment

### Render

The app includes a `render.yaml` for one-click deployment to Render:

1. Connect your GitHub repo to Render
2. Add your `GEMINI_API_KEY` in the Render dashboard
3. Deploy

### Other Platforms

Use the Procfile for Heroku-compatible platforms:

```
web: gunicorn app:app --workers 2 --timeout 300 --bind 0.0.0.0:$PORT
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve the mobile web app |
| GET | `/health` | Health check with config status |
| POST | `/analyze` | Analyze frames (JSON body with `frames` array of base64 JPEGs) |
| GET | `/history` | Get all past call results |
| DELETE | `/call/<id>` | Delete a call from history |

## Tech Stack

- **Backend**: Python Flask + raw urllib for Gemini API
- **Frontend**: Vanilla JavaScript SPA (no frameworks)
- **AI**: Google Gemini 2.5 Flash (vision model)
- **Database**: SQLite
- **Deployment**: Gunicorn on Render

## Running Tests

```bash
python -m pytest tests/ -v
```
