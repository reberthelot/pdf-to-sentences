from pathlib import Path
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
APP_DIR = Path(__file__).resolve().parent


def _get_pdf_path(filename: str) -> Path:
    """Resolve sample PDF path from root or examples folder."""
    root_path = APP_DIR / filename
    if root_path.exists():
        return root_path
    examples_path = APP_DIR / "examples" / filename
    if examples_path.exists():
        return examples_path
    raise FileNotFoundError(f"Could not find {filename} in {APP_DIR} or {APP_DIR / 'examples'}")


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
    """Verify sentence extraction on 2303.15133.pdf according to temp.md specifications."""
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

