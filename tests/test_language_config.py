"""Tests for the configurable NLTK sentence-tokenization language."""

from __future__ import annotations

from pathlib import Path

import nltk

from src.app.config import settings
from src.app.services import extraction_service


def _get_pdf_path(filename: str) -> Path:
    """Resolve sample PDF path from root or examples folder."""
    root_path = settings.BASE_DIR / filename
    if root_path.exists():
        return root_path
    examples_path = settings.EXAMPLES_DIR / filename
    if examples_path.exists():
        return examples_path
    raise FileNotFoundError(
        f"Could not find {filename} in {settings.BASE_DIR} or {settings.EXAMPLES_DIR}"
    )


def test_nltk_language_setting_is_exposed():
    """The segmentation language must be a non-empty configurable string."""
    assert isinstance(settings.NLTK_LANGUAGE, str)
    assert settings.NLTK_LANGUAGE
    assert settings.NLTK_LANGUAGE == "english"  # default behaviour preserved


def test_sent_tokenize_receives_configured_language(monkeypatch):
    """Fast-path extraction must forward ``settings.NLTK_LANGUAGE`` to NLTK."""
    captured: list[str] = []
    original = nltk.sent_tokenize

    def spy(text: str, language: str = "english") -> list[str]:
        captured.append(language)
        return original(text, language=language)

    monkeypatch.setattr(extraction_service.nltk, "sent_tokenize", spy)

    sentences, page_count = extraction_service.try_extract_native_text(
        str(_get_pdf_path("studyboard.pdf"))
    )

    assert page_count > 0
    assert sentences, "expected fast-path sentences from studyboard.pdf"
    assert captured, "nltk.sent_tokenize should have been called"
    assert all(lang == settings.NLTK_LANGUAGE for lang in captured)
