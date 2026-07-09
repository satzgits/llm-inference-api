# LLM Inference API

I built this because I was tired of typing `ollama run qwen3` every time I wanted to test something. I needed a clean HTTP interface that any app (web, mobile, backend) could call without knowing about Ollama or caring what model runs underneath.

So I slapped FastAPI in front of Ollama, added proper input validation with Pydantic, and wrapped it all in Docker. Now I can curl, fetch, or connect from any language and get structured JSON back.

## How it looks

```
Client (curl / app / browser) ──► FastAPI (port 8000) ──► Ollama (port 11434) ──► Qwen 3 on RTX 4070
                                      │
                                      ├── GET  /health     → { status, ollama status }
                                      ├── POST /generate   → { response, tokens, latency }
                                      └── POST /embed      → { embedding vector, latency }
```

## Project layout

```
llm-inference-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # Routes, error handling, logging
│   ├── models.py            # Pydantic schemas (input validation)
│   └── config.py            # Settings from .env
├── tests/
│   ├── __init__.py
│   └── test_api.py          # 9 tests using FastAPI TestClient
├── Dockerfile
├── docker-compose.yml
├── Makefile                 # make install | run | test | docker
├── pyproject.toml
├── requirements.txt
├── test_client.py
└── README.md
```

I kept the `app/` package clean — `main.py` for routes and orchestration, `models.py` for request/response shapes, `config.py` for settings that come from environment variables.

## What I learned building this

- **Ollama's chat API**: The `/api/chat` endpoint expects a `messages` array (same format as OpenAI). I originally tried `/api/generate` but the response format was different. Switching to `/api/chat` worked with both Qwen 3 and Llama 3.1.
- **Input validation matters**: Without Pydantic constraints like `ge=0.0, le=2.0` on temperature, a client sending `temperature: 99` would silently get garbage. Now FastAPI returns a proper 422 with a message.
- **Error handling is not optional**: Ollama can timeout (GPU overload), return garbage (model not loaded), or just hang. I catch each case separately and return meaningful HTTP codes — 504 for timeout, 502 for model errors, 500 for unexpected stuff.
- **Mocking in tests**: I use `unittest.mock.patch` to simulate Ollama responses so tests run without a GPU. This was a game changer — no more staring at "CUDA out of memory" during `pytest`.
- **Docker networking**: On Windows, the container talks to Ollama on the host via `host.docker.internal:11434`. Took me an hour to figure out why localhost wasn't working inside the container.

## Running it

```bash
# Install
pip install -r requirements.txt

# Start the API
make run

# Test it (no GPU needed — uses mocking)
make test
```

### What `make test` looks like

```
========================================= test session starts =========================================
platform win32 -- Python 3.13.1
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_generate_endpoint PASSED
tests/test_api.py::test_generate_missing_prompt PASSED
tests/test_api.py::test_generate_invalid_temperature PASSED
tests/test_api.py::test_generate_negative_max_tokens PASSED
tests/test_api.py::test_embed_endpoint PASSED
tests/test_api.py::test_embed_empty_text PASSED
tests/test_api.py::test_unrecognized_endpoint PASSED
tests/test_api.py::test_generate_ollama_timeout PASSED
===================================== 9 passed in 2.90s =============================================
```

### What a real request looks like

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is quantum computing?", "max_tokens": 50}'
```

Response:

```json
{
  "response": "Quantum computing uses qubits that can exist in multiple states simultaneously, unlike classical bits that are either 0 or 1. This allows quantum computers to process certain calculations exponentially faster...",
  "model": "qwen3",
  "tokens_used": 42,
  "inference_time_ms": 3250.14
}
```

## API endpoints at a glance

| Method | Endpoint | What it does | 
|--------|----------|-------------|
| `GET` | `/health` | Check if the server and Ollama are alive |
| `POST` | `/generate` | Send a prompt, get generated text back |
| `POST` | `/embed` | Get a vector embedding for semantic search |

Each endpoint logs latency, model used, and token count so you can spot slow responses.

## Docker

```bash
make docker        # docker build -t llm-inference-api .
docker-compose up -d
```

The container connects to Ollama running on your host via `host.docker.internal:11434`.

## Why I built it this way

Running `ollama run qwen3` works fine for chatting. But when I wanted to call the model from my Streamlit dashboard, or from a cron job, or from a webhook — I needed an API. Building a thin layer on top decouples the model from whatever consumes it. The same API works whether I'm running Qwen 3, Llama 3, or Mistral underneath. I just change the model name in `.env`.

## Try it yourself

Open `http://localhost:8000/docs` in your browser — FastAPI generates interactive Swagger docs where you can test every endpoint directly. That's the screenshot below.

> **Want the screenshot?** Start the server (`make run`), open `http://localhost:8000/docs`, and take a screenshot. Drop it in `screenshots/swagger_docs.png`.
