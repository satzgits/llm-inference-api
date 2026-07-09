from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "LLM Inference API"
    ollama_host: str = "http://localhost:11434"
    default_model: str = "qwen3"
    default_temperature: float = 0.7
    default_max_tokens: int = 256

settings = Settings()
