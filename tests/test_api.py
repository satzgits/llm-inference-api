"""
Tests for the LLM Inference API — validates routes, schemas, and error handling.
Uses FastAPI TestClient so no server process is needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_generate_endpoint():
    response = client.post("/generate", json={
        "prompt": "Hello",
        "max_tokens": 10,
        "temperature": 0.1
    })
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "tokens_used" in data
    assert "latency_ms" in data


def test_generate_missing_prompt():
    response = client.post("/generate", json={})
    assert response.status_code == 422  # validation error


def test_generate_invalid_temperature():
    response = client.post("/generate", json={
        "prompt": "Hello",
        "temperature": 5.0
    })
    assert response.status_code == 422


def test_generate_negative_max_tokens():
    response = client.post("/generate", json={
        "prompt": "Hello",
        "max_tokens": -5
    })
    assert response.status_code == 422


def test_embed_endpoint():
    response = client.post("/embed", json={
        "text": "Sample text to embed"
    })
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)


def test_embed_empty_text():
    response = client.post("/embed", json={
        "text": ""
    })
    assert response.status_code == 422


def test_unrecognized_endpoint():
    response = client.get("/nonexistent")
    assert response.status_code == 404


if __name__ == "__main__":
    test_health_endpoint()
    print("✓ test_health_endpoint")
    test_generate_endpoint()
    print("✓ test_generate_endpoint")
    test_generate_missing_prompt()
    print("✓ test_generate_missing_prompt")
    test_generate_invalid_temperature()
    print("✓ test_generate_invalid_temperature")
    test_generate_negative_max_tokens()
    print("✓ test_generate_negative_max_tokens")
    test_embed_endpoint()
    print("✓ test_embed_endpoint")
    test_embed_empty_text()
    print("✓ test_embed_empty_text")
    test_unrecognized_endpoint()
    print("✓ test_unrecognized_endpoint")
    print("\nAll API tests passed.")
