from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from src.app.config import settings
from src.app.main import app

client = TestClient(app)


def _get_pdf_path(filename: str) -> Path:
    """Resolve sample PDF path from root or examples folder."""
    root_path = settings.BASE_DIR / filename
    if root_path.exists():
        return root_path
    examples_path = settings.EXAMPLES_DIR / filename
    if examples_path.exists():
        return examples_path
    raise FileNotFoundError(f"Could not find {filename} in {settings.BASE_DIR} or {settings.EXAMPLES_DIR}")


def test_ui_index_redirect():
    """Verify root / redirects to /ui and renders index HTML."""
    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200
    assert "PDF → Sentences" in response.text


def test_inspect_pdf():
    """Verify inspection endpoint estimating duration and detecting layer."""
    pdf_path = _get_pdf_path("studyboard.pdf")
    with open(pdf_path, "rb") as f:
        files = {"pdf_file": ("studyboard.pdf", f, "application/pdf")}
        response = client.post("/v1/inspect-pdf", files=files)

    assert response.status_code == 200, response.text
    data = response.json()
    assert "page_count" in data
    assert "has_text_layer" in data
    assert "recommended_method" in data


def test_extract_sentences_studyboard():
    """Verify sentence extraction on studyboard.pdf."""
    pdf_path = _get_pdf_path("studyboard.pdf")

    with open(pdf_path, "rb") as f:
        files = {"pdf_file": ("studyboard.pdf", f, "application/pdf")}
        response = client.post("/v1/extract-sentences", files=files)

    assert response.status_code == 200, response.text
    data = response.json()
    assert "sentences" in data
    assert isinstance(data["sentences"], list)
    assert len(data["sentences"]) > 0

    expected_sample = "Finn and Tyge were present."
    assert any(expected_sample in s for s in data["sentences"])


def test_extract_sentences_assignment_spec():
    """Verify sentence extraction on 2303.15133.pdf according to assignment specifications."""
    pdf_path = _get_pdf_path("2303.15133.pdf")

    with open(pdf_path, "rb") as f:
        files = {"pdf_file": ("2303.15133.pdf", f, "application/pdf")}
        response = client.post("/v1/extract-sentences", files=files)

    assert response.status_code == 200, response.text
    data = response.json()
    assert "sentences" in data
    assert isinstance(data["sentences"], list)
    assert len(data["sentences"]) > 0

    sentence = "How language should best be handled is not clear."
    assert any(sentence in s for s in data["sentences"])


def test_async_job_lifecycle():
    """Verify end-to-end background job submission and status polling."""
    pdf_path = _get_pdf_path("studyboard.pdf")

    with open(pdf_path, "rb") as f:
        files = {"pdf_file": ("studyboard.pdf", f, "application/pdf")}
        submit_resp = client.post("/v1/jobs/submit", files=files)

    assert submit_resp.status_code == 200, submit_resp.text
    submit_data = submit_resp.json()
    assert "job_id" in submit_data
    assert submit_data["status"] == "pending"

    job_id = submit_data["job_id"]
    status_resp = client.get(f"/v1/jobs/{job_id}")
    assert status_resp.status_code == 200, status_resp.text
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] in ("pending", "processing", "completed")


def test_metrics_updated_on_async_job():
    """Verify that operational metrics record completed asynchronous jobs."""
    metrics_before = client.get("/ui/api/metrics").json()
    total_before = metrics_before.get("total_requests", 0)

    pdf_path = _get_pdf_path("studyboard.pdf")
    with open(pdf_path, "rb") as f:
        files = {"pdf_file": ("studyboard.pdf", f, "application/pdf")}
        submit_resp = client.post("/ui/api/jobs/submit", files=files)

    assert submit_resp.status_code == 200, submit_resp.text
    job_id = submit_resp.json()["job_id"]

    for _ in range(20):
        status_resp = client.get(f"/ui/api/jobs/{job_id}")
        assert status_resp.status_code == 200
        if status_resp.json()["status"] == "completed":
            break
        time.sleep(0.05)

    metrics_after = client.get("/ui/api/metrics").json()
    total_after = metrics_after.get("total_requests", 0)
    assert total_after >= total_before + 1
    assert metrics_after.get("success_requests", 0) >= 1

