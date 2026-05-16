"""
Tennis Ball In/Out Detector - Flask backend
Run locally:  python app.py
Deploy:       gunicorn app:app --workers 2 --timeout 300 --bind 0.0.0.0:$PORT
"""

import base64
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from database import delete_call, get_all_calls, init_db, save_call
from gemini_client import analyze_frames

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

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


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts JSON with 'frames' array of base64-encoded JPEG strings.
    Validates frames, sends to Gemini for analysis, saves result.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required."}), 400

    frames = data.get("frames")
    if not frames or not isinstance(frames, list):
        return jsonify({"error": "A 'frames' array is required."}), 400

    if len(frames) < 1 or len(frames) > 10:
        return jsonify({"error": "Must provide between 1 and 10 frames."}), 400

    # Validate base64 encoding
    validated_frames = []
    for i, frame in enumerate(frames):
        if not isinstance(frame, str) or not frame.strip():
            return jsonify({"error": f"Frame {i} is not a valid base64 string."}), 400
        # Strip data URL prefix if present
        clean = frame
        if "," in clean and clean.startswith("data:"):
            clean = clean.split(",", 1)[1]
        try:
            base64.b64decode(clean, validate=True)
        except Exception:
            return jsonify({"error": f"Frame {i} is not valid base64."}), 400
        validated_frames.append(clean)

    try:
        result = analyze_frames(validated_frames)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    # Save to database
    call_id = save_call(
        result_in_pct=result["percentage_in"],
        result_out_pct=result["percentage_out"],
        confidence=result["confidence"],
        explanation=result["explanation"],
        num_frames=len(validated_frames),
    )
    result["id"] = call_id

    return jsonify(result)


@app.route("/history")
def history():
    return jsonify(get_all_calls())


@app.route("/call/<int:call_id>", methods=["DELETE"])
def delete_call_route(call_id):
    delete_call(call_id)
    return jsonify({"deleted": call_id})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
