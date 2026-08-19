from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text for the LLM")
    model: Optional[str] = Field(None, description="Model name (defaults to qwen3)")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)

class GenerateResponse(BaseModel):
    response: str
    model: str
    tokens_used: int
    inference_time_ms: float

class EmbedRequest(BaseModel):
    input: str = Field(..., min_length=1, description="Text to embed")
    model: Optional[str] = None

class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    ollama_connected: bool

class ModelDetail(BaseModel):
    name: str
    digest: str
    size_bytes: int
    modified_at: str

class ModelsResponse(BaseModel):
    models: list[ModelDetail]
    count: int
