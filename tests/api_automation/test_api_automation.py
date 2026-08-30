"""
API test-automation suite for the LLM Inference API.

This is a contract/integration-style test suite that exercises the REST API as a
black box: build a request, hit an endpoint, assert on the status code and the
shape of the JSON response. It is organised the way a Test Automation engineer
would structure an API test suite, and it maps directly to the QVS role's
"Test Automation + API Testing" requirement.

Two kinds of tests live here:
  - Contract/unit (mock-based): run anywhere with no server, covering validation,
    error handling, and response schemas.
  - Live smoke (real server): run only when the API is actually up
    (set API_BASE_URL and RUN_LIVE=1).

Run all:        python tests/api_automation/test_api_automation.py
Run live:       $env:API_BASE_URL="http://localhost:8008"; $env:RUN_LIVE="1"; python tests/api_automation/test_api_automation.py
"""

import os
import sys
import json
from pathlib import Path

# Make the app importable from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ---- Configuration for live (end-to-end) tests -----------------------------
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8008")
RUN_LIVE = os.environ.get("RUN_LIVE", "0") == "1"


# ---------------------------------------------------------------------------
# A. Contract / unit tests (no server needed) — the bulk of the suite
# ---------------------------------------------------------------------------
def test_health_returns_ok():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert "status" in body and "model_loaded" in body
    print("  ✓ /health schema")


def test_models_endpoint_shape():
    with patch("app.main.httpx.get") as m:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:latest", "digest": "abc123", "size": 4096, "modified_at": "2026-01-01"}
            ]
        }
        m.return_value = mock_response
        res = client.get("/models")
        assert res.status_code == 200
        data = res.json()
        assert "models" in data and "count" in data
        assert data["count"] == 1
    print("  ✓ /models schema")


def test_generate_valid_returns_response():
    """The happy path: a valid /generate call returns a structured response."""
    with patch("app.main.httpx.post") as m:
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Hello!"}}
        m.return_value = mock_response
        res = client.post("/generate", json={"prompt": "Hello", "max_tokens": 10})
        assert res.status_code == 200
        body = res.json()
        for key in ("response", "model", "tokens_used", "inference_time_ms"):
            assert key in body, f"missing {key}"
        assert isinstance(body["tokens_used"], int)
    print("  ✓ /generate happy-path schema")


def test_generate_missing_prompt_returns_422():
    res = client.post("/generate", json={})
    assert res.status_code == 422, "missing prompt should be a validation error"
    print("  ✓ /generate validation (missing prompt -> 422)")


def test_generate_invalid_temperature_returns_422():
    res = client.post("/generate", json={"prompt": "x", "temperature": 9.0})
    assert res.status_code == 422
    print("  ✓ /generate validation (temperature out of range -> 422)")


def test_generate_negative_max_tokens_returns_422():
    res = client.post("/generate", json={"prompt": "x", "max_tokens": -1})
    assert res.status_code == 422
    print("  ✓ /generate validation (max_tokens negative -> 422)")


def test_ollama_timeout_returns_504():
    """The API must translate an upstream timeout into HTTP 504."""
    from httpx import TimeoutException
    with patch("app.main.httpx.post", side_effect=TimeoutException("timeout")):
        res = client.post("/generate", json={"prompt": "x", "max_tokens": 5})
        assert res.status_code == 504
    print("  ✓ /generate error path (timeout -> 504)")


def test_ollama_http_error_returns_502():
    from httpx import HTTPStatusError
    err = HTTPStatusError("bad", request=None, response=MagicMock(text="upstream error"))
    with patch("app.main.httpx.post", side_effect=err):
        res = client.post("/generate", json={"prompt": "x", "max_tokens": 5})
        assert res.status_code == 502
    print("  ✓ /generate error path (upstream error -> 502)")


def test_embed_valid_returns_vector():
    with patch("app.main.httpx.post") as m:
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        m.return_value = mock_response
        res = client.post("/embed", json={"input": "hello"})
        assert res.status_code == 200
        body = res.json()
        assert isinstance(body["embedding"], list)
    print("  ✓ /embed schema")


def test_embed_empty_input_returns_422():
    res = client.post("/embed", json={"input": ""})
    assert res.status_code == 422
    print("  ✓ /embed validation (empty input -> 422)")


def test_unknown_route_returns_404():
    res = client.get("/nonexistent")
    assert res.status_code == 404
    print("  ✓ unknown route -> 404")


# ---------------------------------------------------------------------------
# B. Live smoke tests (only when RUN_LIVE=1) — true end-to-end
# ---------------------------------------------------------------------------
def test_live_generate():
    if not RUN_LIVE:
        print("  · skipped live /generate (set RUN_LIVE=1)")
        return
    import httpx
    res = httpx.post(f"{API_BASE}/generate", json={"prompt": "Say hi", "max_tokens": 10}, timeout=30)
    assert res.status_code == 200
    assert "response" in res.json()
    print("  ✓ live /generate")


def test_live_models():
    if not RUN_LIVE:
        print("  · skipped live /models (set RUN_LIVE=1)")
        return
    import httpx
    res = httpx.get(f"{API_BASE}/models", timeout=10)
    assert res.status_code in (200, 502)
    print("  ✓ live /models")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
TESTS = [
    test_health_returns_ok,
    test_models_endpoint_shape,
    test_generate_valid_returns_response,
    test_generate_missing_prompt_returns_422,
    test_generate_invalid_temperature_returns_422,
    test_generate_negative_max_tokens_returns_422,
    test_ollama_timeout_returns_504,
    test_ollama_http_error_returns_502,
    test_embed_valid_returns_vector,
    test_embed_empty_input_returns_422,
    test_unknown_route_returns_404,
    test_live_generate,
    test_live_models,
]


def run_all():
    print(f"\n=== API Automation Suite — {len(TESTS)} tests ===")
    if RUN_LIVE:
        print(f"(live mode, API_BASE={API_BASE})")
    else:
        print("(contract mode, no server needed)")
    print("=" * 48)
    passed = 0
    for t in TESTS:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} FAILED: {e}")
            if os.environ.get("STRICT"):
                raise
    print(f"  -> {passed}/{len(TESTS)} passed\n")


if __name__ == "__main__":
    run_all()
