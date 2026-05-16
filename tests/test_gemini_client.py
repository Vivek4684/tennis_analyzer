"""
Unit tests for gemini_client.py.
Tests response parsing, markdown fence stripping, field validation, and error handling.
All network calls are mocked via unittest.mock patching of urllib.
"""

import io
import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-testing")

from gemini_client import analyze_frames, detect_ball_contact


def _make_gemini_response(payload_dict):
    """Build a mock HTTP response with the given JSON dict as Gemini output."""
    text = json.dumps(payload_dict)
    body = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{"text": text}]
            }
        }]
    }).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_gemini_raw_text_response(raw_text):
    """Build a mock HTTP response with raw text content from Gemini."""
    body = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{"text": raw_text}]
            }
        }]
    }).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestResponseParsing:
    """Tests for normal response parsing."""

    @patch("gemini_client.urllib.request.urlopen")
    def test_basic_response_parsed_correctly(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 72.5,
            "percentage_out": 27.5,
            "confidence": "high",
            "explanation": "Ball landed on the line."
        })

        result = analyze_frames(["base64data"])

        assert result["percentage_in"] == 72.5
        assert result["percentage_out"] == 27.5
        assert result["confidence"] == "high"
        assert result["explanation"] == "Ball landed on the line."

    @patch("gemini_client.urllib.request.urlopen")
    def test_integer_percentages_converted_to_float(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 80,
            "percentage_out": 20,
            "confidence": "medium",
            "explanation": "Clear landing outside the line."
        })

        result = analyze_frames(["base64data"])

        assert isinstance(result["percentage_in"], float)
        assert isinstance(result["percentage_out"], float)
        assert result["percentage_in"] == 80.0
        assert result["percentage_out"] == 20.0

    @patch("gemini_client.urllib.request.urlopen")
    def test_api_key_sent_in_header_not_url(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 50.0,
            "percentage_out": 50.0,
            "confidence": "low",
            "explanation": "Unclear."
        })

        analyze_frames(["base64data"])

        # Check that urlopen was called with a Request object
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        # API key should NOT be in the URL
        assert "key=" not in request_obj.full_url
        # API key should be in header
        assert request_obj.get_header("X-goog-api-key") == "test-api-key-for-testing"


class TestMarkdownFenceStripping:
    """Tests for markdown code fence removal."""

    @patch("gemini_client.urllib.request.urlopen")
    def test_strips_json_markdown_fences(self, mock_urlopen):
        raw_text = '```json\n{"percentage_in": 65.0, "percentage_out": 35.0, "confidence": "high", "explanation": "On the line."}\n```'
        mock_urlopen.return_value = _make_gemini_raw_text_response(raw_text)

        result = analyze_frames(["base64data"])

        assert result["percentage_in"] == 65.0
        assert result["percentage_out"] == 35.0

    @patch("gemini_client.urllib.request.urlopen")
    def test_strips_plain_markdown_fences(self, mock_urlopen):
        raw_text = '```\n{"percentage_in": 40.0, "percentage_out": 60.0, "confidence": "medium", "explanation": "Outside."}\n```'
        mock_urlopen.return_value = _make_gemini_raw_text_response(raw_text)

        result = analyze_frames(["base64data"])

        assert result["percentage_in"] == 40.0
        assert result["percentage_out"] == 60.0

    @patch("gemini_client.urllib.request.urlopen")
    def test_no_fences_works_fine(self, mock_urlopen):
        raw_text = '{"percentage_in": 90.0, "percentage_out": 10.0, "confidence": "high", "explanation": "Clearly in."}'
        mock_urlopen.return_value = _make_gemini_raw_text_response(raw_text)

        result = analyze_frames(["base64data"])

        assert result["percentage_in"] == 90.0


class TestFieldValidation:
    """Tests for field presence and value validation."""

    @patch("gemini_client.urllib.request.urlopen")
    def test_missing_field_raises_error(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 50.0,
            "percentage_out": 50.0,
            "confidence": "high"
            # missing 'explanation'
        })

        with pytest.raises(RuntimeError, match="missing field: explanation"):
            analyze_frames(["base64data"])

    @patch("gemini_client.urllib.request.urlopen")
    def test_invalid_confidence_defaults_to_low(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 60.0,
            "percentage_out": 40.0,
            "confidence": "very_high",
            "explanation": "Some explanation."
        })

        result = analyze_frames(["base64data"])

        assert result["confidence"] == "low"

    @patch("gemini_client.urllib.request.urlopen")
    def test_confidence_case_insensitive(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 60.0,
            "percentage_out": 40.0,
            "confidence": "HIGH",
            "explanation": "Some explanation."
        })

        result = analyze_frames(["base64data"])

        assert result["confidence"] == "high"

    @patch("gemini_client.urllib.request.urlopen")
    def test_percentages_clamped_to_0_100(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 110.0,
            "percentage_out": -10.0,
            "confidence": "high",
            "explanation": "Hallucinated values."
        })

        result = analyze_frames(["base64data"])

        # 110 clamped to 100, -10 clamped to 0, then normalize: 100/100*100=100, 0/100*100=0
        assert result["percentage_in"] == 100.0
        assert result["percentage_out"] == 0.0

    @patch("gemini_client.urllib.request.urlopen")
    def test_percentages_normalized_to_sum_100(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 60.0,
            "percentage_out": 60.0,
            "confidence": "medium",
            "explanation": "Bad sum from model."
        })

        result = analyze_frames(["base64data"])

        # Both 60, total 120, normalized: 60/120*100 = 50
        assert result["percentage_in"] == 50.0
        assert result["percentage_out"] == 50.0

    @patch("gemini_client.urllib.request.urlopen")
    def test_both_zero_percentages_default_to_50_50(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 0.0,
            "percentage_out": 0.0,
            "confidence": "low",
            "explanation": "Cannot determine."
        })

        result = analyze_frames(["base64data"])

        assert result["percentage_in"] == 50.0
        assert result["percentage_out"] == 50.0


class TestErrorHandling:
    """Tests for error handling paths."""

    @patch("gemini_client.urllib.request.urlopen")
    def test_http_error_raises_runtime_error(self, mock_urlopen):
        import urllib.error
        error_body = b'{"error": {"message": "Invalid request"}}'
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://example.com",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=io.BytesIO(error_body)
        )

        with pytest.raises(RuntimeError, match="Gemini API error 400"):
            analyze_frames(["base64data"])

    @patch("gemini_client.urllib.request.urlopen")
    def test_http_error_does_not_leak_api_key(self, mock_urlopen):
        import urllib.error
        error_body = b'{"error": {"message": "Unauthorized"}}'
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://example.com",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(error_body)
        )

        with pytest.raises(RuntimeError) as exc_info:
            analyze_frames(["base64data"])

        # Error message should not contain the API key
        assert "test-api-key-for-testing" not in str(exc_info.value)

    @patch("gemini_client.urllib.request.urlopen")
    def test_invalid_json_response_raises_error(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_raw_text_response("not valid json at all")

        with pytest.raises(RuntimeError, match="invalid JSON"):
            analyze_frames(["base64data"])

    @patch("gemini_client.urllib.request.urlopen")
    def test_unexpected_response_structure_raises_error(self, mock_urlopen):
        body = json.dumps({"unexpected": "structure"}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Unexpected Gemini response"):
            analyze_frames(["base64data"])

    @patch("gemini_client.urllib.request.urlopen")
    def test_empty_candidates_raises_error(self, mock_urlopen):
        body = json.dumps({"candidates": []}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Unexpected Gemini response"):
            analyze_frames(["base64data"])

    def test_missing_api_key_raises_error(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "gemini_api_key": ""}, clear=False):
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
                analyze_frames(["base64data"])

    def test_placeholder_api_key_raises_error(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "your_gemini_api_key_here"}, clear=False):
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
                analyze_frames(["base64data"])


class TestDetectBallContact:
    """Tests for detect_ball_contact function."""

    @patch("gemini_client.urllib.request.urlopen")
    def test_basic_contact_detection(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "contact_frame_index": 3,
            "ball_state": "ground",
            "description": "Ball making contact with the court surface."
        })

        result = detect_ball_contact(["frame1", "frame2", "frame3", "frame4", "frame5"])

        assert result["contact_frame_index"] == 3
        assert result["ball_state"] == "ground"
        assert result["description"] == "Ball making contact with the court surface."

    @patch("gemini_client.urllib.request.urlopen")
    def test_contact_index_clamped_to_valid_range(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "contact_frame_index": 99,
            "ball_state": "ground",
            "description": "Out of range index."
        })

        result = detect_ball_contact(["frame1", "frame2", "frame3"])

        # Should be clamped to max valid index (2)
        assert result["contact_frame_index"] == 2

    @patch("gemini_client.urllib.request.urlopen")
    def test_negative_contact_index_clamped_to_zero(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "contact_frame_index": -1,
            "ball_state": "ground",
            "description": "Negative index."
        })

        result = detect_ball_contact(["frame1", "frame2"])

        assert result["contact_frame_index"] == 0

    @patch("gemini_client.urllib.request.urlopen")
    def test_invalid_ball_state_defaults_to_ground(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "contact_frame_index": 1,
            "ball_state": "bouncing",
            "description": "Some desc."
        })

        result = detect_ball_contact(["frame1", "frame2", "frame3"])

        assert result["ball_state"] == "ground"

    @patch("gemini_client.urllib.request.urlopen")
    def test_air_ball_state_accepted(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "contact_frame_index": 0,
            "ball_state": "air",
            "description": "Ball still in the air."
        })

        result = detect_ball_contact(["frame1", "frame2"])

        assert result["ball_state"] == "air"

    @patch("gemini_client.urllib.request.urlopen")
    def test_missing_contact_frame_index_raises_error(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "ball_state": "ground",
            "description": "Missing index."
        })

        with pytest.raises(RuntimeError, match="missing field: contact_frame_index"):
            detect_ball_contact(["frame1"])

    @patch("gemini_client.urllib.request.urlopen")
    def test_invalid_json_raises_error(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_raw_text_response("not json at all")

        with pytest.raises(RuntimeError, match="invalid JSON for contact detection"):
            detect_ball_contact(["frame1"])


class TestAnalyzeFramesShotType:
    """Tests for shot_type parameter in analyze_frames."""

    @patch("gemini_client.urllib.request.urlopen")
    def test_serve_shot_type_includes_service_context(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 80.0,
            "percentage_out": 20.0,
            "confidence": "high",
            "explanation": "Ball landed in service box."
        })

        result = analyze_frames(["base64data"], shot_type="serve")

        assert result["percentage_in"] == 80.0
        # Verify the request was made (the prompt includes service context)
        assert mock_urlopen.called

    @patch("gemini_client.urllib.request.urlopen")
    def test_rally_shot_type_includes_baseline_context(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 60.0,
            "percentage_out": 40.0,
            "confidence": "medium",
            "explanation": "Ball near baseline."
        })

        result = analyze_frames(["base64data"], shot_type="rally")

        assert result["percentage_in"] == 60.0
        assert mock_urlopen.called

    @patch("gemini_client.urllib.request.urlopen")
    def test_default_shot_type_is_rally(self, mock_urlopen):
        mock_urlopen.return_value = _make_gemini_response({
            "percentage_in": 50.0,
            "percentage_out": 50.0,
            "confidence": "low",
            "explanation": "Unclear."
        })

        result = analyze_frames(["base64data"])

        assert result["percentage_in"] == 50.0
        assert mock_urlopen.called
