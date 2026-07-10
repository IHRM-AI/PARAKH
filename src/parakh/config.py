from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:8092",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8092",
    "http://ihrm-idbi-innovate-1525602521.us-east-1.elb.amazonaws.com",
]


def _parse_origins(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [origin.strip() for origin in value.split(",") if origin.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    artifacts_dir: Path = PACKAGE_ROOT / "artifacts"
    random_seed: int = 42
    validation_fraction: float = 0.2

    cors_origins: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CORS_ORIGINS), validation_alias="CORS_ORIGINS"
    )
    api_key: str = Field(default="", validation_alias="API_KEY")

    llm_base_url: str = Field(default="", validation_alias="VLLM_URL")
    llm_model: str = Field(default="gemma-4-31b-it", validation_alias="VLLM_MODEL_NAME")
    ocr_service_url: str = Field(default="", validation_alias="OCR_SERVICE_URL")
    llm_timeout_seconds: int = 120

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: str | list[str]) -> list[str]:
        return _parse_origins(value)


settings = Settings()
