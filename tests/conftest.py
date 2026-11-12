"""Shared pytest fixtures for the PDF-to-sentences test suite."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.app.main import app


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """Yield a TestClient started inside a ``with`` block so the application
    lifespan runs (NLTK data, ONNX models, RapidOCR engine initialization)."""
    with TestClient(app) as test_client:
        yield test_client
