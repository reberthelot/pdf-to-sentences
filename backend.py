from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unicodedata
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Configure high-performance CPU inference flags for PaddlePaddle & PaddleX
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_PDX_PDF_RENDER_SCALE"] = "1.0"

import nltk
import numpy as np
import pypdfium2 as pdfium
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from paddleocr import PaddleOCR
from pydantic import BaseModel, Field

# Ensure required NLTK tokenizers are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)

# Instantiate PP-OCRv5 Mobile with high-performance ONNX Runtime engine
ocr_engine = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    engine="onnxruntime",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_side_len=736,
    text_det_box_thresh=0.60,
)

app = FastAPI(
    title="PDF to Sentences API",
    description="High-performance text & sentence extraction with fast-path vector text extraction and ONNX-accelerated PaddleOCR fallback.",
    version="1.2.0",
)


class SentenceResponse(BaseModel):
    """Response model containing extracted sentences and processing metadata."""

    sentences: List[str] = Field(
        ..., description="List of segmented sentences extracted from the PDF body text."
    )
    method: Optional[str] = Field(
        None, description="Extraction engine used: 'fast_path' (digital text) or 'paddleocr_onnx' (vision OCR)."
    )
    page_count: Optional[int] = Field(
        None, description="Number of pages detected in the PDF document."
    )
    processing_time_ms: Optional[float] = Field(
        None, description="Server-side processing duration in milliseconds."
    )


class InspectResponse(BaseModel):
    """Inspection result providing page count and estimated processing time."""

    page_count: int
    has_text_layer: bool
    recommended_method: str
    estimated_seconds: float


class JobStatus(str, Enum):
    """Execution status for asynchronous background extraction tasks."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    """Asynchronous background job progress and result model."""

    job_id: str
    status: JobStatus
    progress: float = 0.0
    current_page: int = 0
    total_pages: int = 0
    message: Optional[str] = None
    result: Optional[SentenceResponse] = None
    created_at: float
    updated_at: float
    error: Optional[str] = None


class JobSubmitResponse(BaseModel):
    """Response returned upon submitting an asynchronous extraction task."""

    job_id: str
    status: JobStatus
    total_pages: int
    estimated_seconds: float


# In-memory store for asynchronous background extraction jobs
jobs_db: Dict[str, JobInfo] = {}


def merge_and_sort_lines(
    dt_polys: List[Any],
    rec_texts: List[str],
    rec_scores: Optional[List[float]] = None,
    y_tol_ratio: float = 0.5,
) -> List[str]:
    """Cluster detected text boxes on the same baseline and sort left-to-right.

    Parameters
    ----------
    dt_polys : List[Any]
        Polygon bounding box corner points from DBNet.
    rec_texts : List[str]
        Recognized text strings for each box.
    rec_scores : Optional[List[float]]
        Confidence scores for each text prediction.
    y_tol_ratio : float
        Vertical tolerance fraction of line height for clustering adjacent words.

    Returns
    -------
    List[str]
        Top-to-bottom, left-to-right ordered and merged text lines.
    """
    if not dt_polys or not rec_texts:
        return [str(t).strip() for t in rec_texts if str(t).strip()]

    items: List[Dict[str, Any]] = []
    scores = rec_scores if rec_scores is not None else [1.0] * len(rec_texts)
    for poly, text, score in zip(dt_polys, rec_texts, scores):
        cleaned_text = str(text).strip()
        if not cleaned_text:
            continue
        pts = np.array(poly)
        ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
        xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
        h = max(ymax - ymin, 1.0)
        yc = (ymin + ymax) / 2.0
        items.append(
            {
                "text": cleaned_text,
                "score": float(score),
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "h": h,
                "yc": yc,
            }
        )

    if not items:
        return []

    # Sort boxes primarily by vertical position
    items.sort(key=lambda item: item["yc"])

    # Cluster words into horizontal lines
    lines: List[List[Dict[str, Any]]] = []
    for item in items:
        placed = False
        for line in lines:
            line_yc = sum(w["yc"] for w in line) / len(line)
            line_h = sum(w["h"] for w in line) / len(line)
            if abs(item["yc"] - line_yc) <= line_h * y_tol_ratio:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    # Sort words within each line from left to right
    merged: List[str] = []
    for line in lines:
        line.sort(key=lambda item: item["xmin"])
        line_str = " ".join(item["text"] for item in line).strip()
        if line_str:
            merged.append(line_str)

    return merged


def try_extract_native_text(pdf_path: str) -> Tuple[Optional[List[str]], int]:
    """Attempt fast-path extraction directly from the PDF digital vector text layer.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF document.

    Returns
    -------
    Tuple[Optional[List[str]], int]
        Extracted sentences if digital text is present (otherwise None), and total page count.
    """
    doc = pdfium.PdfDocument(pdf_path)
    page_count = len(doc)
    if page_count == 0:
        return [], 0

    pages_text: List[str] = []
    total_characters = 0

    for page in doc:
        textpage = page.get_textpage()
        raw_text = textpage.get_text_range()
        normalized = unicodedata.normalize("NFKC", raw_text)
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        page_content = " ".join(lines)
        pages_text.append(page_content)
        total_characters += len(page_content)

    # If the document contains meaningful digital text (at least 20 chars per page on average)
    min_threshold = max(25, 20 * page_count)
    if total_characters >= min_threshold:
        full_text = " ".join(pages_text)
        sentences = [s.strip() for s in nltk.sent_tokenize(full_text) if s.strip()]
        if sentences:
            return sentences, page_count

    return None, page_count


def extract_via_paddleocr(
    pdf_path: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """Fallback to ONNX-accelerated PaddleOCR vision when no digital text layer is present.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    progress_callback : Optional[Callable[[int, int], None]]
        Optional callback called with (current_page, total_pages).

    Returns
    -------
    List[str]
        Segmented sentences extracted via vision OCR.
    """
    doc = pdfium.PdfDocument(pdf_path)
    total_pages = len(doc)
    extracted_lines: List[str] = []

    for page_idx in range(total_pages):
        if progress_callback:
            progress_callback(page_idx + 1, total_pages)

        page = doc[page_idx]
        # Render at native scale 1.0 (72 DPI) to avoid 4x oversampling
        img = np.array(page.render(scale=1.0).to_pil())
        res_list = list(ocr_engine.predict(img))
        if not res_list:
            continue
        page_res = res_list[0]

        dt_polys = page_res.get("dt_polys", [])
        rec_texts = page_res.get("rec_texts", [])
        rec_scores = page_res.get("rec_scores", [])

        if dt_polys and rec_texts:
            page_lines = merge_and_sort_lines(dt_polys, rec_texts, rec_scores)
        else:
            page_lines = [str(t).strip() for t in rec_texts if str(t).strip()]

        extracted_lines.extend(page_lines)

    full_text = " ".join(extracted_lines)
    if not full_text.strip():
        return []

    sentences = nltk.sent_tokenize(full_text)
    return [s.strip() for s in sentences if s.strip()]


async def _run_async_job(job_id: str, temp_path: str) -> None:
    """Background task executor for asynchronous extraction jobs."""
    job = jobs_db[job_id]
    job.status = JobStatus.PROCESSING
    job.updated_at = time.time()
    t0 = time.perf_counter()

    def on_progress(curr: int, total: int) -> None:
        job.current_page = curr
        job.total_pages = total
        job.progress = round(curr / max(total, 1), 2)
        job.message = f"Processing page {curr} of {total} with ONNX OCR..."
        job.updated_at = time.time()

    try:
        # Check Fast-Path native text first
        native_sentences, page_count = try_extract_native_text(temp_path)
        if native_sentences is not None and len(native_sentences) > 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            job.result = SentenceResponse(
                sentences=native_sentences,
                method="fast_path",
                page_count=page_count,
                processing_time_ms=elapsed_ms,
            )
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.message = "Extraction completed instantly via Fast-Path text layer."
            return

        # Run ONNX OCR page by page
        ocr_sentences = await asyncio.to_thread(
            extract_via_paddleocr, temp_path, on_progress
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        job.result = SentenceResponse(
            sentences=ocr_sentences,
            method="paddleocr_onnx",
            page_count=job.total_pages,
            processing_time_ms=elapsed_ms,
        )
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.message = "Extraction completed successfully via PP-OCRv5 Mobile ONNX."

    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.message = f"Failed to extract text: {exc}"
    finally:
        job.updated_at = time.time()
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.post("/v1/inspect-pdf", response_model=InspectResponse, summary="Inspect PDF to estimate processing time")
async def inspect_pdf(pdf_file: UploadFile = File(...)) -> InspectResponse:
    """Analyze a PDF's structure to determine whether it has embedded text and estimate processing duration."""
    content = await pdf_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF file is empty.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(content)
        temp_path = temp_pdf.name

    try:
        sentences, page_count = try_extract_native_text(temp_path)
        has_text_layer = sentences is not None and len(sentences) > 0

        if has_text_layer:
            method = "fast_path"
            estimated = max(0.05, round(0.02 * page_count, 2))
        else:
            method = "paddleocr_onnx"
            # ONNX Runtime on CPU processes ~15s per typewriter scanned page
            estimated = max(5.0, round(15.0 * page_count, 1))

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


@app.post(
    "/v1/jobs/submit",
    response_model=JobSubmitResponse,
    summary="Submit an asynchronous PDF extraction job",
)
async def submit_job(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(..., description="PDF document for background extraction."),
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

    doc = pdfium.PdfDocument(temp_path)
    total_pages = len(doc)
    doc.close()

    # Pre-inspect to calculate estimated time
    sentences, _ = try_extract_native_text(temp_path)
    if sentences:
        est_seconds = 0.05
    else:
        est_seconds = round(15.0 * total_pages, 1)

    job_id = str(uuid.uuid4())
    now = time.time()
    job_info = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0.0,
        current_page=0,
        total_pages=total_pages,
        message="Queued for processing...",
        created_at=now,
        updated_at=now,
    )
    jobs_db[job_id] = job_info

    background_tasks.add_task(_run_async_job, job_id, temp_path)

    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        total_pages=total_pages,
        estimated_seconds=est_seconds,
    )


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobInfo,
    summary="Get status and progress of an asynchronous extraction job",
)
async def get_job_status(job_id: str) -> JobInfo:
    """Query progress and result for a submitted background extraction job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs_db[job_id]


@app.post(
    "/v1/extract-sentences",
    response_model=SentenceResponse,
    summary="Extract sentences from a PDF file",
)
async def extract_sentences(
    pdf_file: UploadFile = File(..., description="The PDF file to extract sentences from.")
) -> SentenceResponse:
    """Extract sentences from a PDF file using high-performance Fast-Path with ONNX PaddleOCR fallback.

    Parameters
    ----------
    pdf_file : UploadFile
        Uploaded PDF binary stream.

    Returns
    -------
    SentenceResponse
        JSON dictionary containing `sentences`, `method`, `page_count`, and `processing_time_ms`.
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
        # 1. Solution 1: Try Fast-Path native vector text extraction
        native_sentences, page_count = try_extract_native_text(temp_path)

        if native_sentences is not None and len(native_sentences) > 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SentenceResponse(
                sentences=native_sentences,
                method="fast_path",
                page_count=page_count,
                processing_time_ms=elapsed_ms,
            )

        # 2. Solutions 3, 4 & 5: ONNX-accelerated PP-OCRv5 with geometric line merging
        ocr_sentences = extract_via_paddleocr(temp_path)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SentenceResponse(
            sentences=ocr_sentences,
            method="paddleocr_onnx",
            page_count=page_count,
            processing_time_ms=elapsed_ms,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the PDF: {exc}",
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
