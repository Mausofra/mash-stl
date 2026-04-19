from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Geral
    debug: bool = False
    backend_port: int = 8000

    # Ollama (visão local)
    ollama_url: str = "http://localhost:11434"
    ollama_model_vision: str = "qwen2.5vl:3b"

    # RunPod — InstantMesh
    instantmesh_runpod_url: str = ""   # https://api.runpod.ai/v2/<ID>/runsync
    instantmesh_runpod_key: str = ""

    # RunPod — chave geral (fallback)
    runpod_api_key: str = ""

    # API / frontend
    allowed_origins: str = (
        "http://localhost:5173,"
        "http://localhost:3000,"
        "http://127.0.0.1:5173"
    )
    max_upload_size_mb: int = 15

    # Timeouts e polling
    ollama_timeout: int = 60
    runpod_poll_interval: int = 5
    runpod_max_wait: int = 300

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]



@lru_cache
def get_settings() -> Settings:
    return Settings()
