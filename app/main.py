import time
import uuid
import logging
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException, Request
from app.config import settings
from app.models import (
    GenerateRequest, GenerateResponse,
    EmbedRequest, EmbedResponse,
    HealthResponse, ModelsResponse, ModelDetail
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name}")
    cfg_errors = settings.validate()
    if cfg_errors:
        logger.warning("Configuration issues: " + "; ".join(cfg_errors))
    else:
        logger.info("Configuration OK")
    logger.info(f"Ollama host: {settings.ollama_host}")
    yield

app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/health", response_model=HealthResponse)
def health_check():
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

@app.get("/models", response_model=ModelsResponse)
def list_models():
    try:
        resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(502, "Ollama not reachable")
    models = [
        ModelDetail(
            name=m.get("name", "unknown"),
            digest=m.get("digest", "")[:12],
            size_bytes=m.get("size", 0),
            modified_at=m.get("modified_at", ""),
        )
        for m in resp.json().get("models", [])
    ]
    return ModelsResponse(models=sorted(models, key=lambda m: m.name), count=len(models))

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    model = request.model or settings.default_model
    temperature = request.temperature or settings.default_temperature
    max_tokens = request.max_tokens or settings.default_max_tokens

    # Qwen 3's thinking/CoT mechanism consumes tokens from num_predict.
    # If max_tokens isn't set by the user, omit num_predict so Ollama decides.
    options = {"temperature": temperature}
    if request.max_tokens is not None:
        # Double the budget so thinking doesn't eat the entire response
        options["num_predict"] = max_tokens * 2

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "options": options,
        "stream": False
    }

    start = time.perf_counter()
    try:
        resp = httpx.post(
            f"{settings.ollama_host}/api/chat",
            json=payload,
            timeout=120.0
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
    response_text = result.get("message", {}).get("content", "")
    tokens = len(response_text.split())

    if not response_text:
        logger.warning(f"Empty response from Ollama. Result keys: {list(result.keys())}")
        if "message" in result:
            logger.warning(f"Message keys: {list(result.get('message', {}).keys())}")

    logger.info(f"Generate: {elapsed_ms:.0f}ms, {tokens} tokens, model={model}")

    return GenerateResponse(
        response=response_text,
        model=model,
        tokens_used=tokens,
        inference_time_ms=round(elapsed_ms, 2)
    )

@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
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
