# LLM Inference API — FastAPI + Docker + Ollama

> **Project Type:** Production ML Engineering  
> **Target Role:** AI/ML Engineer (Kisai Technologies)  
> **Time to Build:** ~30 minutes  
> **Cost:** ₹0 (all tools are free)  
> **GitHub:** Create new repo: `llm-inference-api`

---

## 1. What Is This Project?

A **production-ready LLM inference API** that wraps Ollama (running Qwen 3 locally) behind a FastAPI server with Docker deployment.

Instead of calling Ollama directly via CLI, you expose clean REST endpoints that any application can use:
- POST `/generate` → send text, get LLM response
- POST `/embed` → send text, get embeddings
- GET `/health` → check if service is running

**Why this matters for Kisai:** Kisai is building an AI-powered dev platform with an LLM gateway. This project demonstrates you understand how to productionize LLMs — the exact skill they're hiring for.

---

## 2. How It Works — Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Laptop/Server                      │
│                                                              │
│  ┌──────────────┐    HTTP calls      ┌──────────────────┐   │
│  │              │   localhost:11434  │                  │   │
│  │   Ollama     │◄──────────────────►│   FastAPI Server │   │
│  │   (Qwen 3)   │                    │   Port 8000      │   │
│  │   Port 11434 │                    │                  │   │
│  └──────────────┘                    └────────┬─────────┘   │
│                                               │             │
│                                               ▼             │
│                                   ┌──────────────────────┐ │
│                                   │   Client (curl/      │ │
│                                   │   Postman/Your App)  │ │
│                                   └──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

FLOW:
1. Client sends POST /generate with {"prompt": "Hello", "max_tokens": 100}
2. FastAPI validates input (Pydantic models)
3. FastAPI forwards request to Ollama API (http://localhost:11434/api/generate)
4. Ollama runs inference on Qwen 3 using your RTX 4070 GPU
5. Response flows back: Ollama → FastAPI → Client
6. Each request is logged with latency for monitoring
```

**Deployment Option (Docker):**
```
┌──────────────────────────────────────────────────────────┐
│                   Docker Container                        │
│  ┌──────────────────────┐   OLLAMA_HOST=host.docker.int.│
│  │   FastAPI App        │──────────────────────────────► │
│  │   Python 3.11-slim   │   (connects to host's Ollama)  │
│  │   Port 8000          │                                │
│  └──────────────────────┘                                │
└──────────────────────────────────────────────────────────┘
```

---

## 3. What You'll Learn

| Skill | Why It Matters for Kisai |
|-------|--------------------------|
| **FastAPI** | Building ML inference APIs — Kisai serves models via APIs |
| **Docker** | Containerizing ML apps — Kisai uses Docker/K8s for deployment |
| **Ollama API** | Interacting with LLMs programmatically — Kisai has an LLM gateway |
| **Pydantic Validation** | Input/output validation for ML endpoints — production best practice |
| **Error Handling** | Graceful failure for ML systems (GPU OOM, timeouts) |
| **Logging & Monitoring** | Tracking inference latency, errors — MLOps foundation |
| **REST API Design** | /health, /predict, /embed endpoints — standard ML serving pattern |

---

## 4. Tools Used

| Tool | Version | Cost | Purpose |
|------|:-------:|:----:|---------|
| **Python** | 3.11+ | Free | Programming language |
| **FastAPI** | 0.111+ | Free | Web framework for API |
| **Uvicorn** | 0.30+ | Free | ASGI server |
| **Pydantic** | 2.x | Free | Input/output validation |
| **Ollama** | Latest | Free | Local LLM runtime (Qwen 3) |
| **Qwen 3** | Latest | Free | LLM model (run via Ollama) |
| **Docker** | Latest | Free | Containerization |
| **httpx** | Latest | Free | HTTP client for Ollama calls |

**Total cost: ₹0** — Everything is open source and runs locally on your RTX 4070.

---

## 5. Step-by-Step Build Guide

### Step 1: Create Project Structure

Create folder `llm-inference-api` with:

```
llm-inference-api/
├── app/
│   ├── __init__.py          # Empty file
│   ├── main.py              # FastAPI app with endpoints
│   ├── models.py            # Pydantic request/response models
│   └── config.py            # Settings (OLLAMA_HOST, etc.)
├── Dockerfile               # Container build instructions
├── docker-compose.yml       # Orchestration (optional)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── test_client.py           # Quick test script
└── README.md                # This file
```

### Step 2: Create requirements.txt

```
# ML Inference API Dependencies
fastapi==0.111.0
uvicorn[standard]==0.30.0
pydantic==2.7.0
httpx==0.27.0
python-dotenv==1.0.1
```

### Step 3: Create config.py

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "LLM Inference API"
    ollama_host: str = "http://localhost:11434"
    default_model: str = "qwen3"
    default_temperature: float = 0.7
    default_max_tokens: int = 256

    class Config:
        env_file = ".env"

settings = Settings()
```

### Step 4: Create models.py

```python
from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text for the LLM")
    model: Optional[str] = Field(None, description="Model name (defaults to qwen3)")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    stream: bool = False

class GenerateResponse(BaseModel):
    response: str
    model: str
    tokens_used: int
    inference_time_ms: float

class EmbedRequest(BaseModel):
    input: str
    model: Optional[str] = None

class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    ollama_connected: bool
```

### Step 5: Create main.py (Core Application)

```python
import time
import logging
import httpx
from fastapi import FastAPI, HTTPException
from app.config import settings
from app.models import (
    GenerateRequest, GenerateResponse,
    EmbedRequest, EmbedResponse,
    HealthResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="1.0.0")

@app.on_event("startup")
def startup():
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Ollama host: {settings.ollama_host}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API and Ollama are running."""
    ollama_ok = False
    try:
        resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=3.0)
        ollama_ok = resp.status_code == 200
    except:
        pass

    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        model_loaded=ollama_ok,
        ollama_connected=ollama_ok
    )

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generate text using the LLM."""
    model = request.model or settings.default_model
    temperature = request.temperature or settings.default_temperature
    max_tokens = request.max_tokens or settings.default_max_tokens

    payload = {
        "model": model,
        "prompt": request.prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": False
    }

    start = time.perf_counter()
    try:
        resp = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json=payload,
            timeout=60.0
        )
        resp.raise_for_status()
        result = resp.json()
    except httpx.TimeoutException:
        logger.error("Ollama request timed out")
        raise HTTPException(504, "Model inference timed out")
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama error: {e}")
        raise HTTPException(502, f"Ollama error: {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(500, str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000

    response_text = result.get("response", "")
    tokens = len(response_text.split())

    logger.info(f"Generate: {elapsed_ms:.0f}ms, {tokens} tokens, model={model}")

    return GenerateResponse(
        response=response_text,
        model=model,
        tokens_used=tokens,
        inference_time_ms=round(elapsed_ms, 2)
    )

@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    """Generate embeddings for input text."""
    model = request.model or settings.default_model

    payload = {
        "model": model,
        "prompt": request.input
    }

    start = time.perf_counter()
    try:
        resp = httpx.post(
            f"{settings.ollama_host}/api/embeddings",
            json=payload,
            timeout=30.0
        )
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(500, str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Embed: {elapsed_ms:.0f}ms, model={model}")

    return EmbedResponse(
        embedding=result.get("embedding", []),
        model=model,
        inference_time_ms=round(elapsed_ms, 2)
    )
```

### Step 6: Create test_client.py

```python
import httpx
import json

BASE = "http://localhost:8000"

def test_health():
    r = httpx.get(f"{BASE}/health")
    print(f"Health: {r.status_code} → {r.json()}")
    return r.status_code == 200

def test_generate():
    r = httpx.post(
        f"{BASE}/generate",
        json={"prompt": "What is quantum computing? Answer in 2 sentences.", "max_tokens": 100}
    )
    print(f"Generate: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Response: {data['response'][:100]}...")
        print(f"  Latency: {data['inference_time_ms']}ms")
    return r.status_code == 200

def test_embed():
    r = httpx.post(
        f"{BASE}/embed",
        json={"input": "Hello world"}
    )
    print(f"Embed: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Embedding dims: {len(data['embedding'])}")
    return r.status_code == 200

if __name__ == "__main__":
    print("=== Testing LLM Inference API ===\n")
    test_health()
    test_generate()
    test_embed()
    print("\n=== Done ===")
```

### Step 7: Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 8: Create docker-compose.yml

```yaml
version: '3.8'
services:
  llm-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

### Step 9: Create .env.example

```
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=qwen3
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=256
```

---

## 6. How to Run

### Without Docker (Quick Test)

```bash
# 1. Ensure Ollama is running with Qwen 3
ollama pull qwen3

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Test it
python test_client.py
```

### With Docker

```bash
# 1. Make sure Ollama is running on host (not in Docker)
# 2. Build and run
docker-compose up -d

# 3. Test
python test_client.py

# Or via curl:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?"}'
```

### Access the API Docs

FastAPI auto-generates Swagger docs:
- Open browser: http://localhost:8000/docs
- Test endpoints interactively

---

## 7. Endpoints Reference

| Method | Endpoint | Description | Example Request |
|--------|----------|-------------|----------------|
| GET | `/health` | Check API + Ollama status | — |
| POST | `/generate` | Generate text from prompt | `{"prompt": "Hello", "max_tokens": 100}` |
| POST | `/embed` | Get text embeddings | `{"input": "Hello world"}` |

---

## 8. Verification Checklist

```
[ ] pip install -r requirements.txt works
[ ] uvicorn starts without errors
[ ] /health returns {"status": "healthy", "ollama_connected": true}
[ ] POST /generate returns response with latency
[ ] POST /embed returns embedding array
[ ] docker-compose up builds without errors
[ ] Docker container responds to requests
[ ] All endpoints have proper error handling
[ ] test_client.py passes all 3 tests
```

---

## 9. How to Showcase This in Your Kisai Interview

**What to say:**
> "I built an LLM inference API using FastAPI and Docker. It wraps Ollama (Qwen 3) behind clean REST endpoints — /generate for text generation, /embed for embeddings, and /health for monitoring. The API validates inputs with Pydantic, logs every request with latency metrics, handles errors gracefully (timeouts, GPU OOM), and is fully containerized with Docker. This demonstrates I understand how to productionize LLMs — not just run them in notebooks — which is directly relevant to Kisai's LLM gateway and AI platform work."

**What they'll notice:**
- Clean separation of concerns (config, models, routes)
- Docker multi-stage or single-stage deployment
- Proper error handling (not just try/except pass)
- Logging for observability
- Input validation preventing bad data from hitting the model
- They can clone your repo and have it running in 2 commands

---

## 10. The Simplest Explanation (What This Project Actually Does)

### Before This Project
You talk to Qwen 3 by opening a terminal and typing `ollama run qwen3`. That works for you, but what if a **website or an app** wants to talk to Qwen 3? They can't use your terminal.

### After This Project
This project builds a **waiter** between Qwen 3 and the outside world.

```
Before:   You ──terminal──► Qwen 3
After:    App/website ──► FastAPI (the waiter) ──► Qwen 3
```

The FastAPI waiter:
- Takes orders (your prompts) from any app/website
- Delivers them to Qwen 3
- Brings back the response
- Logs how long it took
- Handles errors if Qwen 3 is busy or fails
- Validates that the order makes sense before sending it

### FastAPI = A Waiter (In Simple Terms)
FastAPI is a tool that lets you create "waiters" in Python. You tell it:
- "When someone visits this URL, do this"
- "When someone sends data to this other URL, process it"

No HTML, no CSS, no frontend. Just a waiter that speaks **JSON** (a format apps understand).

### Ollama = The Chef (In Simple Terms)
Ollama is a program that runs AI models (Qwen 3) on your laptop. It's like a chef who can cook any dish but only speaks to people who come to the kitchen directly.

### The Full Flow (Simplest Terms)

| Step | Who | What Happens |
|:----:|:---:|-------------|
| 1 | **App** | Sends request to `http://localhost:8000/generate` with `{"prompt": "Hello"}` |
| 2 | **FastAPI (waiter)** | Checks: "Is the request valid? Does it have a prompt field?" |
| 3 | **FastAPI → Ollama** | Forwards request to Ollama at `http://localhost:11434/api/generate` |
| 4 | **Ollama (chef)** | Runs Qwen 3 on your RTX 4070 GPU and generates the response |
| 5 | **Ollama → FastAPI → App** | Response flows back through the waiter to the app |
| 6 | **FastAPI** | Logs: "Took 2.3 seconds, returned 50 tokens" |

### Why Docker?
Docker packages the waiter (FastAPI) into a box so it runs the **same on your laptop, a server, or the cloud**. One command to start: `docker-compose up`.
