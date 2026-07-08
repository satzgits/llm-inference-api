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
    model = request.model or settings.default_model
    temperature = request.temperature or settings.default_temperature
    max_tokens = request.max_tokens or settings.default_max_tokens

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
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
