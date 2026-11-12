"""Application settings backed by ``pydantic-settings``.

Values are read from environment variables (case-insensitive field-name
match, e.g. ``MODELS_DIR``, ``HOST``, ``PORT``) and from an optional ``.env``
file at the project root. Legacy env names that do not match the field name
are preserved through ``AliasChoices`` so existing Docker/Compose files keep
working unchanged (``SENTENCE_SERVICE_URL``, ``SERVICE_TIMEOUT_SECONDS``,
``SERVICE_CONNECT_TIMEOUT_SECONDS``).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Environment-driven application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated container env vars (PYTHONUNBUFFERED, ...)
    )

    # Project directory paths
    BASE_DIR: Path = _PROJECT_ROOT
    MODELS_DIR: Path = _PROJECT_ROOT / "models"
    FRONTEND_DIR: Path = _PROJECT_ROOT / "frontend"
    TEMPLATES_DIR: Path = _PROJECT_ROOT / "frontend" / "templates"
    STATIC_DIR: Path = _PROJECT_ROOT / "frontend" / "static"
    EXAMPLES_DIR: Path = _PROJECT_ROOT / "examples"

    # Service & Networking
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1, le=65535)
    SERVICE_URL: str = Field(
        default="http://localhost:8000/v1/extract-sentences",
        validation_alias=AliasChoices("SENTENCE_SERVICE_URL", "SERVICE_URL"),
    )
    TIMEOUT_SECONDS: float = Field(
        default=180.0,
        gt=0,
        validation_alias=AliasChoices("SERVICE_TIMEOUT_SECONDS", "TIMEOUT_SECONDS"),
    )
    CONNECT_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        gt=0,
        validation_alias=AliasChoices(
            "SERVICE_CONNECT_TIMEOUT_SECONDS", "CONNECT_TIMEOUT_SECONDS"
        ),
    )

    # Sentence segmentation language for NLTK Punkt
    # (e.g. "english", "danish"; both are shipped in the Docker image)
    NLTK_LANGUAGE: str = "english"

    # Root logging level (env LOG_LEVEL): DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL: str = "INFO"

    @field_validator("LOG_LEVEL", mode="after")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        """Accept any case (``debug`` -> ``DEBUG``) and reject unknown levels."""
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"unknown log level: {value!r}")
        return level


settings = Settings()
