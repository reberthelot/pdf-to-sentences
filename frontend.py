from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from frontend_service import (
    DEFAULT_SERVICE_URL,
    Metrics,
    call_sentence_service,
    inspect_pdf_service,
    pedagogic_http_error,
    run_selftest,
)

BASE_DIR = Path(__file__).resolve().parent
HTML_TEMPLATE_PATH = BASE_DIR / "template" / "index.html"
STATIC_DIR = BASE_DIR / "static"

metrics = Metrics()
app = FastAPI(
    title="PDF → Sentences Frontend (DTU)",
    description="Interactive UI and evaluation harness for PDF sentence extraction.",
    version="1.1.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Render the single-page HTML application."""
    if not HTML_TEMPLATE_PATH.exists():
        return "<h3>Error: template/index.html not found.</h3>"

    content = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    return content.replace("{SERVICE_URL}", DEFAULT_SERVICE_URL)


@app.get("/api/metrics")
async def api_metrics() -> Dict[str, Any]:
    """Return operational metrics snapshot."""
    return await metrics.snapshot()


@app.post("/api/inspect")
async def api_inspect(pdf_file: UploadFile = File(...)) -> JSONResponse:
    """Inspect uploaded PDF to determine page count and estimated processing time."""
    try:
        pdf_bytes = await pdf_file.read()
        if not pdf_bytes:
            return JSONResponse(status_code=400, content={"error": "File is empty."})

        data = await inspect_pdf_service(
            pdf_bytes, pdf_file.filename or "uploaded.pdf", DEFAULT_SERVICE_URL
        )
        return JSONResponse(content=data)
    except Exception as exc:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to inspect PDF: {exc}"}
        )


@app.post("/api/extract")
async def api_extract(pdf_file: UploadFile = File(...)) -> JSONResponse:
    """Handle PDF upload from UI and forward to sentence extraction backend."""
    t0 = time.perf_counter()
    latency_ms: Optional[float] = None

    try:
        pdf_bytes = await pdf_file.read()
        if not pdf_bytes:
            err_msg = "Uploaded file was empty. Please upload a valid PDF."
            await metrics.record_failure(latency_ms=None, error=err_msg)
            return JSONResponse(status_code=400, content={"error": err_msg})

        sentences, latency_ms, meta = await call_sentence_service(
            pdf_bytes, pdf_file.filename or "uploaded.pdf"
        )
        await metrics.record_success(latency_ms=latency_ms)
        return JSONResponse(
            content={
                "sentences": sentences,
                "latency_ms": latency_ms,
                "method": meta.get("method"),
                "page_count": meta.get("page_count"),
            }
        )
    except httpx.HTTPError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        msg = pedagogic_http_error(exc, DEFAULT_SERVICE_URL)
        await metrics.record_failure(latency_ms=latency_ms, error=msg)
        return JSONResponse(
            status_code=502, content={"error": msg, "latency_ms": latency_ms}
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        msg = (
            "An unexpected error occurred while processing the request.\n"
            f"- Type: {type(exc).__name__}\n"
            f"- Details: {exc}\n"
        )
        await metrics.record_failure(latency_ms=latency_ms, error=msg)
        return JSONResponse(
            status_code=500, content={"error": msg, "latency_ms": latency_ms}
        )


@app.post("/api/selftest")
async def api_selftest() -> Dict[str, Any]:
    """Run self-test validation on sample PDFs."""
    return await run_selftest()
