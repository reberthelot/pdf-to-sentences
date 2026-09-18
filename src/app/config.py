from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Environment-driven application configuration."""

    # Project directory paths
    BASE_DIR: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    MODELS_DIR: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "MODELS_DIR",
                str(Path(__file__).resolve().parent.parent.parent / "models"),
            )
        )
    )
    FRONTEND_DIR: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "frontend"
    )
    TEMPLATES_DIR: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "frontend" / "templates"
    )
    STATIC_DIR: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "frontend" / "static"
    )
    EXAMPLES_DIR: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "examples"
    )

    # Service & Networking
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("PORT", "8000"))
    SERVICE_URL: str = os.environ.get(
        "SENTENCE_SERVICE_URL", "http://localhost:8000/v1/extract-sentences"
    )
    TIMEOUT_SECONDS: float = float(os.environ.get("SERVICE_TIMEOUT_SECONDS", "180"))
    CONNECT_TIMEOUT_SECONDS: float = float(
        os.environ.get("SERVICE_CONNECT_TIMEOUT_SECONDS", "10")
    )


settings = Settings()

