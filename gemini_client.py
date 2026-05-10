"""
Gemini File API + generateContent client.
Uses only Python stdlib (urllib) — no SDK required.
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def _api_key() -> str:
    key = (
        os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("gemini_api_key", "")
    ).strip()
    if not key or key == "your_gemini_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "  • On Render: Dashboard → Environment → add GEMINI_API_KEY\n"
            "  • Locally:   add GEMINI_API_KEY=your_key to .env\n"
            "  • Free key:  https://aistudio.google.com/app/apikey"
        )
    return key


def _model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()


# ---------------------------------------------------------------------------
# File API — upload
# ---------------------------------------------------------------------------

def upload_video(video_path: str) -> str:
    """
    Upload a video file to the Gemini File API.
    Returns the file URI (e.g. 'files/abc123').
    Polls until the file state is ACTIVE.
    """
    api_key = _api_key()
    path = Path(video_path)
    file_size = path.stat().st_size

    # Determine MIME type
    ext = path.suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".m4v": "video/mp4",
    }
    mime_type = mime_map.get(ext, "video/mp4")

    # ── Step 1: initiate resumable upload ─────────────────────
    init_url = (
        f"https://generativelanguage.googleapis.com/upload/v1beta/files"
        f"?uploadType=resumable&key={api_key}"
    )
    metadata = json.dumps({
        "file": {"display_name": path.name}
    }).encode("utf-8")

    init_req = urllib.request.Request(
        init_url,
        data=metadata,
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(init_req, timeout=30) as resp:
        upload_url = resp.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise RuntimeError("Gemini File API did not return an upload URL.")

    # ── Step 2: upload the file bytes ─────────────────────────
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    upload_req = urllib.request.Request(
        upload_url,
        data=video_bytes,
        headers={
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        method="POST",
    )
    with urllib.request.urlopen(upload_req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    file_name = result["file"]["name"]   # e.g. "files/abc123"
    file_uri  = result["file"]["uri"]    # full URI

    # ── Step 3: poll until ACTIVE ──────────────────────────────
    poll_url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{file_name}?key={api_key}"
    )
    for _ in range(40):   # max 120 seconds
        time.sleep(3)
        with urllib.request.urlopen(poll_url, timeout=15) as resp:
            status = json.loads(resp.read().decode("utf-8"))
        state = status.get("state", "")
        if state == "ACTIVE":
            return file_name   # return the name (files/xxx) for later deletion
        if state == "FAILED":
            raise RuntimeError(f"Gemini File API processing failed: {status}")

    raise RuntimeError("Timed out waiting for Gemini to process the video.")


# ---------------------------------------------------------------------------
# generateContent — analyze
# ---------------------------------------------------------------------------

def analyze_match(file_name: str, players: dict) -> dict:
    """
    players: {"FL": "John", "FR": "Mike", "BL": "Sarah", "BR": "Emma"}
    Returns parsed analysis dict.
    """
    api_key = _api_key()
    model   = _model()

    prompt = f"""You are an expert tennis coach analyzing a doubles match video.

Players by court position (viewing from behind the baseline):
- Front Left (FL):  {players.get('FL', 'Player 1')}
- Front Right (FR): {players.get('FR', 'Player 2')}
- Back Left (BL):   {players.get('BL', 'Player 3')}
- Back Right (BR):  {players.get('BR', 'Player 4')}

Team 1 = FL + FR (near side)
Team 2 = BL + BR (far side)

Watch the full video and analyze each player's performance. Return ONLY a JSON object in exactly this format — no markdown, no explanation, just the JSON:

{{
  "players": [
    {{
      "name": "{players.get('FL', 'Player 1')}",
      "position": "FL",
      "overall_rating": 7.5,
      "strengths": ["specific strength 1", "specific strength 2", "specific strength 3"],
      "improvements": ["specific improvement 1", "specific improvement 2", "specific improvement 3"]
    }},
    {{
      "name": "{players.get('FR', 'Player 2')}",
      "position": "FR",
      "overall_rating": 7.0,
      "strengths": ["specific strength 1", "specific strength 2", "specific strength 3"],
      "improvements": ["specific improvement 1", "specific improvement 2", "specific improvement 3"]
    }},
    {{
      "name": "{players.get('BL', 'Player 3')}",
      "position": "BL",
      "overall_rating": 6.5,
      "strengths": ["specific strength 1", "specific strength 2", "specific strength 3"],
      "improvements": ["specific improvement 1", "specific improvement 2", "specific improvement 3"]
    }},
    {{
      "name": "{players.get('BR', 'Player 4')}",
      "position": "BR",
      "overall_rating": 6.0,
      "strengths": ["specific strength 1", "specific strength 2", "specific strength 3"],
      "improvements": ["specific improvement 1", "specific improvement 2", "specific improvement 3"]
    }}
  ],
  "team1_score": 7.2,
  "team2_score": 6.2,
  "match_summary": "2-3 sentence overall match summary covering key moments and team dynamics."
}}

Rate each player out of 10. Be specific and actionable — reference actual shots or moments from the video."""

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"file_data": {"mime_type": "video/mp4", "file_uri": f"https://generativelanguage.googleapis.com/v1beta/{file_name}"}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "response_mime_type": "application/json",
        }
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error {exc.code}: {detail}") from exc

    # Extract text from response
    try:
        raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {result}") from exc

    # Strip markdown fences if present
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini returned invalid JSON: {raw_text[:300]}"
        ) from exc


# ---------------------------------------------------------------------------
# File API — delete
# ---------------------------------------------------------------------------

def delete_file(file_name: str):
    """Delete a file from Gemini File API. file_name = 'files/abc123'"""
    try:
        api_key = _api_key()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"{file_name}?key={api_key}"
        )
        req = urllib.request.Request(url, method="DELETE")
        urllib.request.urlopen(req, timeout=15)
    except Exception:
        pass  # best-effort cleanup
