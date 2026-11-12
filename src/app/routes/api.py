from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from src.app.models import (
    InspectResponse,
    JobInfo,
    JobStatus,
    JobSubmitResponse,
    SentenceResponse,
)
from src.app.services.extraction_service import (
    extract_via_rapidocr,
    jobs_db,
    run_async_job,
    try_extract_native_text,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Sentence Extraction API"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for Docker HEALTHCHECK: no dependencies, no side effects."""
    return {"status": "ok"}


@router.post(
    "/v1/inspect-pdf",
    response_model=InspectResponse,
    summary="Inspect PDF to estimate processing time",
)
async def inspect_pdf(pdf_file: UploadFile = File(...)) -> InspectResponse:
    """Analyze a PDF structure to detect native vector text and estimate duration."""
    content = await pdf_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF file is empty.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(content)
        temp_path = temp_pdf.name

    try:
        # PDF parsing + text extraction are CPU-bound: run them off the event loop
        # so concurrent requests are not blocked while this one is processed.
        sentences, page_count = await asyncio.to_thread(try_extract_native_text, temp_path)
        has_text_layer = sentences is not None and len(sentences) > 0

        if has_text_layer:
            method = "fast_path"
            estimated = max(0.05, round(0.02 * page_count, 2))
        else:
            method = "rapidocr_onnx"
            estimated = max(4.0, round(12.0 * page_count, 1))

        return InspectResponse(
            page_count=page_count,
            has_text_layer=has_text_layer,
            recommended_method=method,
            estimated_seconds=estimated,
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@router.post(
    "/v1/jobs/submit",
    response_model=JobSubmitResponse,
    summary="Submit an asynchronous PDF extraction job",
)
async def submit_job(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(
        ..., description="PDF document for background extraction."
    ),
) -> JobSubmitResponse:
    """Submit a PDF file for background sentence extraction with progress tracking."""
    if not pdf_file.filename or not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a file with a .pdf extension.",
        )

    content = await pdf_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF file is empty.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(content)
        temp_path = temp_pdf.name

    # `try_extract_native_text` opens the document and reports its page count,
    # which is the exact value the former separate `pdfium.PdfDocument` probe
    # returned. Merging both avoids opening the PDF twice and keeps the
    # CPU-bound parse off the event loop.
    sentences, total_pages = await asyncio.to_thread(try_extract_native_text, temp_path)
    if sentences:
        est_seconds = 0.05
    else:
        est_seconds = round(12.0 * total_pages, 1)

    job_id = str(uuid.uuid4())
    now = time.time()
    job_info = JobInfo(
        job_id=job_id,
        filename=pdf_file.filename,
        status=JobStatus.PENDING,
        progress=0.0,
        current_page=0,
        total_pages=total_pages,
        message="Queued for processing...",
        created_at=now,
        updated_at=now,
    )
    jobs_db[job_id] = job_info

    background_tasks.add_task(run_async_job, job_id, temp_path)

    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        total_pages=total_pages,
        estimated_seconds=est_seconds,
    )


@router.get(
    "/v1/jobs/{job_id}",
    response_model=JobInfo,
    summary="Get status and progress of an asynchronous extraction job",
)
async def get_job_status(job_id: str) -> JobInfo:
    """Query progress and result for a submitted background extraction job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs_db[job_id]


@router.post(
    "/v1/extract-sentences",
    response_model=SentenceResponse,
    summary="Extract sentences from a PDF file",
)
async def extract_sentences(
    pdf_file: UploadFile = File(
        ..., description="The PDF file to extract sentences from."
    )
) -> SentenceResponse:
    """Extract natural sentences from a PDF document.

    Uses Fast-Path digital text extraction with PP-OCRv5 Mobile (ONNX Runtime)
    fallback for scanned/image pages.
    """
    if not pdf_file.filename or not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a file with a .pdf extension.",
        )

    content = await pdf_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF file is empty.")

    start_time = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(content)
        temp_path = temp_pdf.name

    try:
        # 1. Fast-Path native vector text extraction (CPU-bound → threadpool)
        native_sentences, page_count = await asyncio.to_thread(
            try_extract_native_text, temp_path
        )
        if native_sentences is not None and len(native_sentences) > 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SentenceResponse(
                sentences=native_sentences,
                method="fast_path",
                page_count=page_count,
                processing_time_ms=elapsed_ms,
            )

        # 2. Vision OCR fallback (PP-OCRv5 Mobile ONNX via RapidOCR, ~12 s/page:
        #    must never run on the event loop or every other request would stall)
        ocr_sentences = await asyncio.to_thread(extract_via_rapidocr, temp_path)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SentenceResponse(
            sentences=ocr_sentences,
            method="rapidocr_onnx",
            page_count=page_count,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        # Full details go to the server logs only: exception messages can leak
        # temp file paths, model paths and other internal implementation data.
        logger.exception("Sentence extraction failed for %s", pdf_file.filename)
        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while processing the PDF. "
                "See the server logs for details."
            ),
        ) from exc
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

