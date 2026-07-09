# LLM Inference API

Serves LLMs (Qwen 3, Llama 3, etc.) behind a FastAPI REST API with Docker support. Instead of calling Ollama directly from the terminal, this provides clean HTTP endpoints that any application can integrate with.

## Architecture

```
Client (curl/app) ──► FastAPI (port 8000) ──► Ollama (port 11434) ──► Qwen 3 (RTX 4070)
                         │
                         ├── GET  /health     →  service status
                         ├── POST /generate   →  text generation
                         └── POST /embed      →  text embeddings
```

The FastAPI server validates inputs, forwards requests to Ollama, and returns structured responses with latency metrics.

## Project Structure

```
llm-inference-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI routes + logic
│   ├── models.py            # Pydantic request/response schemas
│   └── config.py            # Settings (model name, Ollama host)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── test_client.py
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Test it
python test_client.py
```

## API Endpoints

| Method | Endpoint | Description | Example |
|--------|----------|-------------|---------|
| GET | `/health` | Check API + Ollama status | `curl localhost:8000/health` |
| POST | `/generate` | Generate text | `{"prompt": "Hello", "max_tokens": 100}` |
| POST | `/embed` | Get embeddings | `{"input": "Hello world"}` |

## Docker

```bash
docker-compose up -d
```

The container connects to Ollama running on the host via `host.docker.internal:11434`.

## Key Decisions

- **Chat API:** Uses Ollama's `/api/chat` endpoint — compatible with instruction-tuned models (Qwen 3, Llama 3, Mistral)
- **Input validation:** Pydantic models reject malformed requests before they reach the model
- **Error handling:** Timeouts, GPU OOM, and Ollama failures return meaningful HTTP status codes
- **Logging:** Every request logs model, latency, and token count for observability

![Screenshot](screenshots/swagger_docs.png)
*FastAPI auto-generates interactive Swagger docs at /docs*

## Example Output

```json
# POST /generate
{"prompt": "What is quantum computing?", "max_tokens": 50}

# Response
{
  "response": "Quantum computing uses qubits to perform calculations...",
  "model": "qwen3",
  "tokens_used": 42,
  "inference_time_ms": 3250.14
}
```

## Why This Approach

Running `ollama run qwen3` works for interactive use, but applications need programmatic access. Building a thin API layer decouples the model from the client — any app (web, mobile, backend) can send requests without knowing about Ollama or the underlying infrastructure. Docker ensures the same setup works in development, CI, and production.
