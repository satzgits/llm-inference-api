"""
Tests for the LLM Inference API — uses unittest.mock to avoid hitting real Ollama.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.config import Settings

client = TestClient(app)


def test_config_defaults_are_valid():
    """Default settings should pass validation."""
    s = Settings()
    assert s.validate() == []
    assert s.ollama_base().endswith("11434") or "localhost" in s.ollama_base()


def test_config_rejects_bad_host():
    """A malformed Ollama URL should be flagged by validation."""
    s = Settings(ollama_host="not-a-url")
    assert any("http" in e for e in s.validate())


def test_config_rejects_bad_temperature():
    s = Settings(default_temperature=5.0)
    assert any("temperature" in e for e in s.validate())


def test_config_rejects_negative_max_tokens():
    s = Settings(default_max_tokens=-1)
    assert any("max_tokens" in e for e in s.validate())


def test_health_endpoint():
    """GET /health should return status even when Ollama is down."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


@patch("app.main.httpx.post")
def test_generate_endpoint(mock_post):
    """POST /generate should return structured response when Ollama works."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "Hello! How can I help you today?"}
    }
    mock_post.return_value = mock_response

    response = client.post("/generate", json={
        "prompt": "Hello",
        "max_tokens": 10,
        "temperature": 0.1
    })
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "tokens_used" in data
    assert "inference_time_ms" in data


def test_generate_missing_prompt():
    """POST /generate without prompt should return 422."""
    response = client.post("/generate", json={})
    assert response.status_code == 422


def test_generate_invalid_temperature():
    """Temperature out of range should return 422."""
    response = client.post("/generate", json={
        "prompt": "Hello",
        "temperature": 5.0
    })
    assert response.status_code == 422


def test_generate_negative_max_tokens():
    """Negative max_tokens should return 422."""
    response = client.post("/generate", json={
        "prompt": "Hello",
        "max_tokens": -5
    })
    assert response.status_code == 422


@patch("app.main.httpx.post")
def test_embed_endpoint(mock_post):
    """POST /embed should return embedding vector when Ollama works."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    mock_post.return_value = mock_response

    response = client.post("/embed", json={
        "input": "Sample text"
    })
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)


def test_embed_empty_text():
    """Empty input text should return 422."""
    response = client.post("/embed", json={"input": ""})
    assert response.status_code == 422


def test_unrecognized_endpoint():
    """Unknown route should return 404."""
    response = client.get("/nonexistent")
    assert response.status_code == 404


@patch("app.main.httpx.post")
def test_generate_ollama_timeout(mock_post):
    """Ollama timeout should return 504."""
    from httpx import TimeoutException
    mock_post.side_effect = TimeoutException("Timed out")

    response = client.post("/generate", json={
        "prompt": "Hello",
        "max_tokens": 10
    })
    assert response.status_code == 504


@patch("app.main.httpx.get")
def test_models_endpoint(mock_get):
    """GET /models should list available Ollama models."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [
            {"name": "qwen3:latest", "digest": "abc123", "size": 4096000000, "modified_at": "2026-01-01"},
            {"name": "llama3:latest", "digest": "def456", "size": 4700000000, "modified_at": "2026-01-02"}
        ]
    }
    mock_get.return_value = mock_response

    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["models"][0]["name"] == "llama3:latest"


@patch("app.main.httpx.get")
def test_models_endpoint_ollama_down(mock_get):
    """GET /models when Ollama is unreachable should return 502."""
    from httpx import ConnectError
    mock_get.side_effect = ConnectError("Connection refused")

    response = client.get("/models")
    assert response.status_code == 502


if __name__ == "__main__":
    test_health_endpoint()
    print("[OK] test_health_endpoint")
    test_generate_endpoint()
    print("[OK] test_generate_endpoint")
    test_generate_missing_prompt()
    print("[OK] test_generate_missing_prompt")
    test_generate_invalid_temperature()
    print("[OK] test_generate_invalid_temperature")
    test_generate_negative_max_tokens()
    print("[OK] test_generate_negative_max_tokens")
    test_embed_endpoint()
    print("[OK] test_embed_endpoint")
    test_embed_empty_text()
    print("[OK] test_embed_empty_text")
    test_unrecognized_endpoint()
    print("[OK] test_unrecognized_endpoint")
    test_generate_ollama_timeout()
    print("[OK] test_generate_ollama_timeout")
    test_models_endpoint()
    print("[OK] test_models_endpoint")
    test_models_endpoint_ollama_down()
    print("[OK] test_models_endpoint_ollama_down")
    print("\nAll API tests passed.")
