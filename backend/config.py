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

    # RunPod — Hunyuan3D-2 (novo worker principal)
    hunyuan3d_runpod_url: str = ""     # https://api.runpod.ai/v2/<ID>/runsync
    hunyuan3d_runpod_key: str = ""

    # RunPod — chave geral (fallback)
    runpod_api_key: str = ""

    # API / frontend
    allowed_origins: str = (
        "http://localhost:5173,"
        "http://localhost:3000,"
        "http://127.0.0.1:5173"
    )
    max_upload_size_mb: int = 15

    # Pre-processamento de imagem
    preprocess_max_dim: int = 1024
    preprocess_autocrop: bool = True
    preprocess_pad_ratio: float = 0.08
    preprocess_autocontrast: bool = True
    preprocess_rembg: bool = True

    # Pós-processamento de mesh (trimesh)
    postprocess_remove_fragments: bool = True
    postprocess_fragment_threshold: float = 0.01
    postprocess_fill_holes: bool = True
    postprocess_fix_normals: bool = True
    postprocess_smooth: bool = False
    postprocess_smooth_iterations: int = 5
    postprocess_smooth_lambda: float = 0.5
    postprocess_decimate: bool = False
    postprocess_target_faces: int = 100_000

    # Timeouts e polling
    ollama_timeout: int = 60
    runpod_poll_interval: int = 5
    runpod_max_wait: int = 900  # 15 min — alta quality na A40 leva ~630s total

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
