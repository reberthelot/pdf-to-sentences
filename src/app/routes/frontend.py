from __future__ import annotations

import time
from typing import Any, Dict, Optional, Set

import httpx
from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from src.app.config import settings
from src.app.services.extraction_service import jobs_db
from src.app.services.sentence_client import (
    call_sentence_service,
    get_job_status_service,
    inspect_pdf_service,
    pedagogic_http_error,
    run_selftest,
    submit_job_service,
)
from src.app.utils.metrics import Metrics

router = APIRouter(tags=["Frontend UI & Dashboard"])

metrics = Metrics()
recorded_jobs: Set[str] = set()


@router.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Render the single-page HTML application and reset operational metrics."""
    await metrics.reset()
    recorded_jobs.clear()

    template_file = settings.TEMPLATES_DIR / "index.html"
    if not template_file.exists():
        return "<h3>Error: templates/index.html not found.</h3>"

    content = template_file.read_text(encoding="utf-8")
    return content.replace("{SERVICE_URL}", settings.SERVICE_URL)


@router.post("/api/metrics/reset")
async def api_metrics_reset() -> Dict[str, Any]:
    """Explicitly reset operational metrics."""
    await metrics.reset()
    recorded_jobs.clear()
    return await metrics.snapshot(settings.SERVICE_URL)


@router.get("/api/metrics")
async def api_metrics() -> Dict[str, Any]:
    """Return operational metrics snapshot."""
    return await metrics.snapshot(settings.SERVICE_URL)


@router.post("/api/inspect")
async def api_inspect(pdf_file: UploadFile = File(...)) -> JSONResponse:
    """Inspect uploaded PDF to determine page count and estimated processing time."""
    try:
        pdf_bytes = await pdf_file.read()
        if not pdf_bytes:
            return JSONResponse(status_code=400, content={"error": "File is empty."})

        data = await inspect_pdf_service(
            pdf_bytes, pdf_file.filename or "uploaded.pdf", settings.SERVICE_URL
        )
        return JSONResponse(content=data)
    except Exception as exc:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to inspect PDF: {exc}"}
        )


@router.post("/api/extract")
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
            pdf_bytes, pdf_file.filename or "uploaded.pdf", settings.SERVICE_URL
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
        msg = pedagogic_http_error(exc, settings.SERVICE_URL)
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


@router.post("/api/selftest")
async def api_selftest() -> Dict[str, Any]:
    """Run self-test validation on sample PDFs and record in operational metrics."""
    return await run_selftest(
        service_url=settings.SERVICE_URL, metrics_tracker=metrics
    )


@router.post("/api/jobs/submit")
async def api_submit_job(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
) -> JSONResponse:
    """Submit PDF to backend asynchronous extraction job queue."""
    try:
        from src.app.routes.api import submit_job as api_submit_direct

        if (
            "localhost:8000" in settings.SERVICE_URL
            or "127.0.0.1:8000" in settings.SERVICE_URL
        ):
            job_resp = await api_submit_direct(background_tasks, pdf_file)
            return JSONResponse(content=job_resp.model_dump())

        pdf_bytes = await pdf_file.read()
        if not pdf_bytes:
            return JSONResponse(status_code=400, content={"error": "File is empty."})
        data = await submit_job_service(
            pdf_bytes, pdf_file.filename or "uploaded.pdf", settings.SERVICE_URL
        )
        return JSONResponse(content=data)
    except Exception as exc:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to submit async job: {exc}"}
        )


@router.get("/api/jobs/{job_id}")
async def api_job_status(job_id: str) -> JSONResponse:
    """Query progress and result for an asynchronous extraction job."""
    try:
        if (
            "localhost:8000" in settings.SERVICE_URL
            or "127.0.0.1:8000" in settings.SERVICE_URL
        ) and job_id in jobs_db:
            data = jobs_db[job_id].model_dump()
        else:
            data = await get_job_status_service(job_id, settings.SERVICE_URL)

        status = data.get("status")
        if status in ("completed", "failed") and job_id not in recorded_jobs:
            recorded_jobs.add(job_id)
            if status == "completed":
                result = data.get("result") or {}
                latency_ms = result.get("processing_time_ms")
                if latency_ms is None:
                    created = data.get("created_at")
                    updated = data.get("updated_at")
                    latency_ms = (
                        (updated - created) * 1000.0 if created and updated else 0.0
                    )
                await metrics.record_success(latency_ms=float(latency_ms))
            elif status == "failed":
                err = data.get("error") or "Background job extraction failed."
                await metrics.record_failure(latency_ms=None, error=err)

        return JSONResponse(content=data)
    except Exception as exc:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to query job status: {exc}"}
        )

