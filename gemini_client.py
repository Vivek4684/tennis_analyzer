"""
Gemini Vision client for tennis ball in/out analysis.
Sends base64-encoded frame images to Gemini 2.5 Flash via generateContent.
Uses only Python stdlib (urllib) - no SDK required.
"""

import json
import os
import urllib.error
import urllib.request


def _api_key() -> str:
    key = (
        os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("gemini_api_key", "")
    ).strip()
    if not key or key == "your_gemini_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "  - On Render: Dashboard > Environment > add GEMINI_API_KEY\n"
            "  - Locally:   add GEMINI_API_KEY=your_key to .env\n"
            "  - Free key:  https://aistudio.google.com/app/apikey"
        )
    return key


def _model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()


def analyze_frames(frames_base64: list[str]) -> dict:
    """
    Analyze tennis ball landing frames for in/out determination.

    Args:
        frames_base64: List of base64-encoded JPEG image strings (1-10 frames).

    Returns:
        Dict with percentage_in, percentage_out, confidence, explanation.
    """
    api_key = _api_key()
    model = _model()

    prompt = """You are an expert tennis line judge analyzing video frames of a tennis ball landing near a court line.

Examine the provided frames showing a tennis ball making contact with or near the court surface. Determine whether the ball landed IN (touching or inside the white court line) or OUT (entirely outside the white court line).

Consider:
- The position of the ball relative to the white court lines
- Any visible mark or bounce point on the court surface
- The ball's trajectory across multiple frames
- In tennis, a ball touching any part of the line is considered IN

Return ONLY a JSON object in exactly this format - no markdown, no explanation, just the JSON:

{
  "percentage_in": 75.0,
  "percentage_out": 25.0,
  "confidence": "high",
  "explanation": "Brief explanation of why the ball is judged in or out based on its position relative to the court lines."
}

Rules:
- percentage_in + percentage_out must equal 100.0
- confidence must be one of: "high", "medium", "low"
- Be specific in the explanation about what you observe in the frames"""

    # Build parts: text prompt followed by inline image data
    parts = [{"text": prompt}]
    for frame_b64 in frames_base64:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": frame_b64,
            }
        })

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
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
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini returned invalid JSON: {raw_text[:300]}"
        ) from exc

    # Validate required fields
    required = ["percentage_in", "percentage_out", "confidence", "explanation"]
    for field in required:
        if field not in parsed:
            raise RuntimeError(f"Gemini response missing field: {field}")

    # Clamp percentages to [0, 100]
    pct_in = max(0.0, min(100.0, float(parsed["percentage_in"])))
    pct_out = max(0.0, min(100.0, float(parsed["percentage_out"])))

    # Normalize so they sum to 100
    total = pct_in + pct_out
    if total > 0:
        pct_in = round(pct_in / total * 100.0, 2)
        pct_out = round(pct_out / total * 100.0, 2)
    else:
        pct_in = 50.0
        pct_out = 50.0

    # Validate confidence against allowed values
    confidence = str(parsed["confidence"]).lower().strip()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    return {
        "percentage_in": pct_in,
        "percentage_out": pct_out,
        "confidence": confidence,
        "explanation": str(parsed["explanation"]),
    }
