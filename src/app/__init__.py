"""PDF to Sentences Microservice (DTU NLP & LLM Assignment).

Assignment Specifications:
-------------------------
- Purpose: Implement a text processing pipeline converting PDF to body text and segmenting into sentences.
- REST Endpoint: POST `/v1/extract-sentences` accepting multipart form-data with field `pdf_file`.
- Response Format: JSON dictionary with key `sentences` containing a list of strings:
  `{"sentences": ["First sentence.", "Second sentence."]}`
- Containerization: Self-contained service exposed on port 8000 via Docker / Docker Compose.
- Constraints: Standalone execution without external Web services (local offline Fast-Path & RapidOCR ONNX).
"""

__version__ = "1.3.0"
__author__ = "DTU NLP & LLM Student"

