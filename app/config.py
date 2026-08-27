from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "LLM Inference API"
    ollama_host: str = "http://localhost:11434"
    default_model: str = "qwen3"
    default_temperature: float = 0.7
    default_max_tokens: int = 256

    def ollama_base(self) -> str:
        """Return the host without a trailing slash (for building endpoints)."""
        return self.ollama_host.rstrip("/")

    def is_ollama_url_valid(self) -> bool:
        """Cheap sanity check that the configured Ollama host is a usable URL."""
        return self.ollama_base().startswith(("http://", "https://"))

    def validate(self) -> list[str]:
        """Return a list of configuration problems (empty means valid).

        Catches bad settings at startup so the API fails fast instead of
        silently misbehaving (e.g. an empty or malformed Ollama URL).
        """
        errors = []
        if not self.ollama_host:
            errors.append("ollama_host must not be empty")
        elif not self.is_ollama_url_valid():
            errors.append(f"ollama_host must be http(s) URL, got '{self.ollama_host}'")
        if not 0.0 <= self.default_temperature <= 2.0:
            errors.append("default_temperature must be in [0.0, 2.0]")
        if self.default_max_tokens < 1:
            errors.append("default_max_tokens must be >= 1")
        return errors

settings = Settings()
