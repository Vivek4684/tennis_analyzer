"""
Gemini Vision client for tennis ball in/out analysis.
Sends base64-encoded frame images to Gemini 2.5 Flash via generateContent.
Uses only Python stdlib (urllib) - no SDK required.

Two-step analysis flow:
1. detect_ball_contact: Identifies which sampled frame is closest to ball-ground contact.
2. analyze_frames: Analyzes frames from a 5-second window around that contact point.
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


def _call_gemini(parts: list, temperature: float = 0.2) -> str:
    """
    Low-level helper: send parts to Gemini generateContent and return raw text.

    Args:
        parts: List of content parts (text and/or inline_data).
        temperature: Sampling temperature.

    Returns:
        Raw text string from Gemini response.
    """
    api_key = _api_key()
    model = _model()

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": temperature,
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

    return raw_text


def detect_ball_contact(frames_base64: list[str]) -> dict:
    """
    Step 1: Identify which frame (by index) is closest to the ball making
    contact with the ground. This acts as an agent that determines whether
    the ball is in the air or near the ground in each frame.

    Args:
        frames_base64: List of base64-encoded JPEG image strings (sampled from
                       the full recording at regular intervals).

    Returns:
        Dict with:
          - contact_frame_index: 0-based index of the frame closest to ground contact
          - ball_state: "air" or "ground" for the identified frame
          - description: brief description of what was observed
    """
    num_frames = len(frames_base64)

    prompt = f"""You are a tennis video analysis agent. You are given {num_frames} frames sampled at regular intervals from a tennis recording (up to 2 minutes long).

Your task: Identify which frame shows the moment closest to the tennis ball making contact with the ground (bouncing on the court surface). Look for:
- The ball transitioning from being in the air to being near or touching the ground
- A visible bounce, mark on the court, or the ball at its lowest point near the surface
- Compression or deformation of the ball at ground contact

For each frame, determine if the ball is "in the air" or "near/on the ground". Then pick the single frame index (0-based) that is closest to the moment of ground contact.

Return ONLY a JSON object in exactly this format:

{{
  "contact_frame_index": 3,
  "ball_state": "ground",
  "description": "Brief description of what you see in the contact frame."
}}

Rules:
- contact_frame_index must be an integer from 0 to {num_frames - 1}
- ball_state must be "ground" (ball near/on surface) or "air" (ball still airborne, if no clear contact is visible)
- If you cannot clearly identify a ground contact moment, pick the frame where the ball appears closest to the court surface"""

    # Build parts: text prompt followed by inline image data
    parts = [{"text": prompt}]
    for frame_b64 in frames_base64:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": frame_b64,
            }
        })

    raw_text = _call_gemini(parts, temperature=0.1)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini returned invalid JSON for contact detection: {raw_text[:300]}"
        ) from exc

    # Validate required fields
    if "contact_frame_index" not in parsed:
        raise RuntimeError("Gemini contact detection response missing field: contact_frame_index")

    index = int(parsed["contact_frame_index"])
    # Clamp to valid range
    index = max(0, min(num_frames - 1, index))

    ball_state = str(parsed.get("ball_state", "ground")).lower().strip()
    if ball_state not in ("air", "ground"):
        ball_state = "ground"

    return {
        "contact_frame_index": index,
        "ball_state": ball_state,
        "description": str(parsed.get("description", "")),
    }


def analyze_frames(frames_base64: list[str], shot_type: str = "rally") -> dict:
    """
    Step 2: Analyze tennis ball landing frames for in/out determination.

    Args:
        frames_base64: List of base64-encoded JPEG image strings (1-10 frames),
                       extracted from a 5-second window around the ball contact point.
        shot_type: Either "serve" or "rally". Determines which court lines are
                   relevant for the in/out call.
                   - "serve": The service box lines (service line, center service
                     line, and singles sideline) define the boundary.
                   - "rally": The baseline and sidelines (singles or doubles
                     depending on match type) define the boundary.

    Returns:
        Dict with percentage_in, percentage_out, confidence, explanation.
    """
    if shot_type == "serve":
        line_context = """This is a SERVICE shot. For a serve to be IN, the ball must land within the correct service box:
- The relevant boundaries are the service line (horizontal line between net and baseline), the center service line (vertical line dividing the two service boxes), and the singles sideline.
- A ball touching any part of these lines is IN.
- A ball landing beyond the service line, outside the singles sideline, or on the wrong side of the center line is OUT."""
    else:
        line_context = """This is a RALLY shot (not a serve). For a rally shot to be IN, the ball must land within the court boundaries:
- The relevant boundaries are the baseline (back line) and the sidelines.
- For singles: the inner sidelines define the court width.
- For doubles: the outer sidelines define the court width.
- A ball touching any part of the baseline or relevant sideline is IN.
- A ball landing beyond the baseline or outside the sideline is OUT."""

    prompt = f"""You are an expert tennis line judge analyzing video frames of a tennis ball landing near a court line.

{line_context}

Examine the provided frames showing a tennis ball making contact with or near the court surface. Determine whether the ball landed IN (touching or inside the relevant white court line) or OUT (entirely outside the relevant white court line).

Consider:
- The position of the ball relative to the white court lines
- Any visible mark or bounce point on the court surface
- The ball's trajectory across multiple frames
- In tennis, a ball touching any part of the line is considered IN

Return ONLY a JSON object in exactly this format - no markdown, no explanation, just the JSON:

{{
  "percentage_in": 75.0,
  "percentage_out": 25.0,
  "confidence": "high",
  "explanation": "Brief explanation of why the ball is judged in or out based on its position relative to the court lines."
}}

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

    raw_text = _call_gemini(parts, temperature=0.2)

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
