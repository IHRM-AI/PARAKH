from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    artifacts_dir: Path = PACKAGE_ROOT / "artifacts"
    random_seed: int = 42
    validation_fraction: float = 0.2
    cors_origins: list[str] = ["*"]

    llm_base_url: str = Field(default="", validation_alias="VLLM_URL")
    llm_model: str = Field(default="gemma-4-31b-it", validation_alias="VLLM_MODEL_NAME")
    ocr_service_url: str = Field(default="", validation_alias="OCR_SERVICE_URL")
    llm_timeout_seconds: int = 120


settings = Settings()
