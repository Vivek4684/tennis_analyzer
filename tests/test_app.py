"""
Tests for Tennis Ball In/Out Detector.
Uses Flask test client with mocked Gemini API calls.
"""

import base64
import json
import os
import tempfile
from unittest.mock import patch

import pytest

# Use a temp file for test DB (in-memory doesn't work across connections)
_test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_test_db.close()
os.environ["DB_PATH"] = _test_db.name

from app import app
from database import init_db


@pytest.fixture
def client():
    """Create a Flask test client with a fresh database."""
    app.config["TESTING"] = True
    with app.app_context():
        init_db()
    with app.test_client() as c:
        yield c
    # Clean up: remove all data after each test
    import sqlite3
    conn = sqlite3.connect(os.environ["DB_PATH"])
    conn.execute("DELETE FROM calls")
    conn.commit()
    conn.close()


@pytest.fixture
def sample_frame():
    """Return a valid base64-encoded 1x1 JPEG."""
    # Minimal valid JPEG (1x1 pixel, white)
    jpeg_bytes = (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
        b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
        b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=8'
        b'3<.telerik34\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
        b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
        b'\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04'
        b'\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa'
        b'\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n'
        b'\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz'
        b'\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99'
        b'\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7'
        b'\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4'
        b'\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea'
        b'\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01'
        b'\x01\x00\x00?\x00\xfb\xd2\x8a+\xff\xd9'
    )
    return base64.b64encode(jpeg_bytes).decode("ascii")


def mock_gemini_response():
    """Return a mock Gemini analysis result."""
    return {
        "percentage_in": 72.5,
        "percentage_out": 27.5,
        "confidence": "high",
        "explanation": "The ball appears to be touching the outer edge of the baseline."
    }


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "gemini_key_configured" in data

    def test_health_shows_model(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "gemini_model" in data


class TestAnalyzeEndpoint:
    @patch("app.analyze_frames")
    def test_analyze_success(self, mock_analyze, client, sample_frame):
        mock_analyze.return_value = mock_gemini_response()

        resp = client.post("/analyze", json={"frames": [sample_frame]})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["percentage_in"] == 72.5
        assert data["percentage_out"] == 27.5
        assert data["confidence"] == "high"
        assert "explanation" in data
        assert "id" in data

    @patch("app.analyze_frames")
    def test_analyze_with_shot_type_serve(self, mock_analyze, client, sample_frame):
        mock_analyze.return_value = mock_gemini_response()

        resp = client.post("/analyze", json={"frames": [sample_frame], "shot_type": "serve"})
        assert resp.status_code == 200
        mock_analyze.assert_called_once()
        # Verify shot_type was passed
        call_kwargs = mock_analyze.call_args
        assert call_kwargs[1]["shot_type"] == "serve"

    @patch("app.analyze_frames")
    def test_analyze_with_shot_type_rally(self, mock_analyze, client, sample_frame):
        mock_analyze.return_value = mock_gemini_response()

        resp = client.post("/analyze", json={"frames": [sample_frame], "shot_type": "rally"})
        assert resp.status_code == 200
        mock_analyze.assert_called_once()
        call_kwargs = mock_analyze.call_args
        assert call_kwargs[1]["shot_type"] == "rally"

    @patch("app.analyze_frames")
    def test_analyze_invalid_shot_type_defaults_to_rally(self, mock_analyze, client, sample_frame):
        mock_analyze.return_value = mock_gemini_response()

        resp = client.post("/analyze", json={"frames": [sample_frame], "shot_type": "invalid"})
        assert resp.status_code == 200
        mock_analyze.assert_called_once()
        call_kwargs = mock_analyze.call_args
        assert call_kwargs[1]["shot_type"] == "rally"

    def test_analyze_no_frames(self, client):
        resp = client.post("/analyze", json={"frames": []})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_analyze_missing_frames_key(self, client):
        resp = client.post("/analyze", json={})
        assert resp.status_code == 400

    def test_analyze_no_body(self, client):
        resp = client.post("/analyze", content_type="application/json")
        assert resp.status_code == 400

    def test_analyze_too_many_frames(self, client, sample_frame):
        frames = [sample_frame] * 11
        resp = client.post("/analyze", json={"frames": frames})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "10" in data["error"]

    @patch("app.analyze_frames")
    def test_analyze_with_data_url_prefix(self, mock_analyze, client, sample_frame):
        mock_analyze.return_value = mock_gemini_response()
        frame_with_prefix = "data:image/jpeg;base64," + sample_frame

        resp = client.post("/analyze", json={"frames": [frame_with_prefix]})
        assert resp.status_code == 200

    @patch("app.analyze_frames")
    def test_analyze_gemini_error(self, mock_analyze, client, sample_frame):
        mock_analyze.side_effect = RuntimeError("Gemini API error 500: Internal error")

        resp = client.post("/analyze", json={"frames": [sample_frame]})
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data


class TestDetectContactEndpoint:
    @patch("app.detect_ball_contact")
    def test_detect_contact_success(self, mock_detect, client, sample_frame):
        mock_detect.return_value = {
            "contact_frame_index": 3,
            "ball_state": "ground",
            "description": "Ball touching the court surface near the baseline."
        }

        resp = client.post("/detect-contact", json={"frames": [sample_frame] * 5})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["contact_frame_index"] == 3
        assert data["ball_state"] == "ground"
        assert "description" in data

    def test_detect_contact_no_frames(self, client):
        resp = client.post("/detect-contact", json={"frames": []})
        assert resp.status_code == 400

    def test_detect_contact_missing_frames_key(self, client):
        resp = client.post("/detect-contact", json={})
        assert resp.status_code == 400

    def test_detect_contact_no_body(self, client):
        resp = client.post("/detect-contact", content_type="application/json")
        assert resp.status_code == 400

    def test_detect_contact_too_many_frames(self, client, sample_frame):
        frames = [sample_frame] * 11
        resp = client.post("/detect-contact", json={"frames": frames})
        assert resp.status_code == 400

    @patch("app.detect_ball_contact")
    def test_detect_contact_gemini_error(self, mock_detect, client, sample_frame):
        mock_detect.side_effect = RuntimeError("Gemini API error 500: Internal error")

        resp = client.post("/detect-contact", json={"frames": [sample_frame]})
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data

    @patch("app.detect_ball_contact")
    def test_detect_contact_with_data_url_prefix(self, mock_detect, client, sample_frame):
        mock_detect.return_value = {
            "contact_frame_index": 0,
            "ball_state": "air",
            "description": "Ball in the air."
        }
        frame_with_prefix = "data:image/jpeg;base64," + sample_frame

        resp = client.post("/detect-contact", json={"frames": [frame_with_prefix]})
        assert resp.status_code == 200


class TestHistoryEndpoint:
    def test_history_empty(self, client):
        resp = client.get("/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) == 0

    @patch("app.analyze_frames")
    def test_history_after_analyze(self, mock_analyze, client, sample_frame):
        mock_analyze.return_value = mock_gemini_response()

        client.post("/analyze", json={"frames": [sample_frame]})
        resp = client.get("/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["result_in_pct"] == 72.5


class TestDeleteEndpoint:
    @patch("app.analyze_frames")
    def test_delete_call(self, mock_analyze, client, sample_frame):
        mock_analyze.return_value = mock_gemini_response()

        # Create a call
        resp = client.post("/analyze", json={"frames": [sample_frame]})
        call_id = resp.get_json()["id"]

        # Delete it
        resp = client.delete(f"/call/{call_id}")
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] == call_id

        # Verify it's gone
        resp = client.get("/history")
        assert len(resp.get_json()) == 0


class TestIndexPage:
    def test_index_loads(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"getUserMedia" in resp.data
        assert b"MediaRecorder" in resp.data
        assert b"canvas" in resp.data
        assert b"/analyze" in resp.data

    def test_index_has_viewport_meta(self, client):
        resp = client.get("/")
        assert b'viewport' in resp.data
        assert b'width=device-width' in resp.data

    def test_index_has_camera_facing_mode(self, client):
        resp = client.get("/")
        assert b'environment' in resp.data

    def test_index_has_two_minute_recording(self, client):
        resp = client.get("/")
        assert b'MAX_RECORD_SECONDS = 120' in resp.data

    def test_index_has_detect_contact_endpoint(self, client):
        resp = client.get("/")
        assert b'/detect-contact' in resp.data

    def test_index_has_shot_type_selector(self, client):
        resp = client.get("/")
        assert b'shotTypeSelect' in resp.data
        assert b'serve' in resp.data
        assert b'rally' in resp.data

    def test_index_has_upload_option(self, client):
        resp = client.get("/")
        assert b'upload' in resp.data.lower()
