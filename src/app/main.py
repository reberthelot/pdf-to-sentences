from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.app.config import settings
from src.app.routes.api import router as api_router
from src.app.routes.frontend import router as frontend_router


def create_app() -> FastAPI:
    """Application factory for the PDF to Sentences extraction microservice."""
    app = FastAPI(
        title="PDF to Sentences Extraction API",
        description="""
High-performance text & sentence extraction microservice with an intelligent dual-engine architecture:
- **Fast-Path Engine (`pypdfium2`)**: Instant vector text extraction (~0.02s per page) for native digital PDFs.
- **Vision OCR Engine (PP-OCRv5 Mobile + ONNX Runtime)**: Fallback vision OCR for scanned or image-based documents.

---

### Key Capabilities
- **DTU Frontend Evaluation Suite**: Interactive dashboard with real-time KPI metrics and self-testing.
- **Asynchronous Task Queue**: Long-running OCR processing with non-blocking polling and live progress.
- **Standalone Offline Execution**: Self-provisioning ONNX models without cloud API dependencies.
""",
        version="1.3.0",
    )

    # Register REST API endpoints (/v1/extract-sentences, /v1/jobs, etc.)
    app.include_router(api_router)

    # Construct and mount Frontend UI sub-application onto /ui
    ui_app = FastAPI(
        title="PDF → Sentences Frontend (DTU)",
        description="Interactive UI and evaluation harness for PDF sentence extraction.",
        version="1.3.0",
    )

    if settings.STATIC_DIR.exists():
        ui_app.mount(
            "/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static"
        )

    ui_app.include_router(frontend_router)
    app.mount("/ui", ui_app)

    @app.get("/", include_in_schema=False)
    def root_redirect() -> RedirectResponse:
        """Redirect root browser requests to the UI application."""
        return RedirectResponse(url="/ui")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

