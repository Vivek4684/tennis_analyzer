"""
Tennis Doubles Match Analyzer — Flask backend
Run locally:  python app.py
Deploy:       gunicorn app:app --workers 2 --timeout 300 --bind 0.0.0.0:$PORT
"""

import os
import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from database import delete_match, get_all_matches, get_match, init_db, save_match
from gemini_client import analyze_match, delete_file, upload_video

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

UPLOAD_DIR = Path(tempfile.gettempdir()) / "tennis_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def allowed_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO


# Initialise DB on startup
with app.app_context():
    init_db()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    key = (
        os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("gemini_api_key", "")
    ).strip()
    key_ok = bool(key) and key != "your_gemini_api_key_here"
    return jsonify({
        "status": "ok",
        "gemini_key_configured": key_ok,
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    })


@app.route("/upload", methods=["POST"])
def upload():
    """Save uploaded video to /tmp and return a temp filename."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_video(f.filename):
        return jsonify({"error": "Unsupported format. Use MP4, MOV, AVI, MKV, or WEBM."}), 400

    uid = uuid.uuid4().hex
    ext = Path(f.filename).suffix.lower()
    save_path = UPLOAD_DIR / f"{uid}{ext}"
    f.save(str(save_path))

    size_mb = round(save_path.stat().st_size / (1024 * 1024), 1)
    return jsonify({
        "filename": f.filename,
        "temp_name": save_path.name,
        "size_mb": size_mb,
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Full pipeline:
      1. Upload video to Gemini File API
      2. Wait for processing
      3. Analyze with Gemini
      4. Delete from File API + local disk
      5. Save to DB
      6. Return results
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required."}), 400

    temp_name = data.get("temp_name")
    players   = data.get("players", {})
    original_filename = data.get("filename", temp_name or "unknown")

    required_positions = {"FL", "FR", "BL", "BR"}
    if not all(players.get(p, "").strip() for p in required_positions):
        return jsonify({"error": "All 4 player names (FL, FR, BL, BR) are required."}), 400

    if not temp_name:
        return jsonify({"error": "temp_name is required."}), 400

    video_path = UPLOAD_DIR / temp_name
    if not video_path.exists():
        return jsonify({"error": "Video file not found. Please re-upload."}), 400

    file_name = None
    try:
        # 1. Upload to Gemini File API
        file_name = upload_video(str(video_path))

        # 2. Analyze
        analysis = analyze_match(file_name, players)

        # 3. Save to DB
        match_id = save_match(original_filename, analysis)
        analysis["match_id"] = match_id

        return jsonify(analysis)

    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    finally:
        # Always clean up — File API and local disk
        if file_name:
            delete_file(file_name)
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.route("/history")
def history():
    return jsonify(get_all_matches())


@app.route("/match/<int:match_id>", methods=["GET"])
def get_match_route(match_id):
    m = get_match(match_id)
    if not m:
        return jsonify({"error": "Match not found."}), 404
    return jsonify(m)


@app.route("/match/<int:match_id>", methods=["DELETE"])
def delete_match_route(match_id):
    delete_match(match_id)
    return jsonify({"deleted": match_id})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
