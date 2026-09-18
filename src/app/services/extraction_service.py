from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import nltk
import numpy as np
import pypdfium2 as pdfium
import yaml
from rapidocr_onnxruntime import RapidOCR

from src.app.config import settings
from src.app.models import JobInfo, JobStatus, SentenceResponse
from src.app.utils.normalization import merge_and_sort_lines, normalize_extracted_text

# Ensure required NLTK tokenizers are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)

MODEL_URLS = {
    "PP-OCRv5_mobile_det.onnx": "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx/resolve/main/inference.onnx",
    "en_PP-OCRv5_mobile_rec.onnx": "https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx",
    "inference.yml": "https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.yml",
}


def download_file(url: str, dest_path: Path) -> None:
    """Download a remote file using standard urllib with temporary buffering."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest_path.with_suffix(".download.tmp")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RapidOCR-Setup"},
    )
    with urllib.request.urlopen(req) as response, open(temp_dest, "wb") as out_file:
        chunk_size = 64 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
    temp_dest.replace(dest_path)


def ensure_models_exist(models_dir: Optional[Path | str] = None) -> Path:
    """Ensure PP-OCRv5 Mobile ONNX models and English dictionary are present locally."""
    target_dir = Path(models_dir) if models_dir else settings.MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    det_path = target_dir / "PP-OCRv5_mobile_det.onnx"
    rec_path = target_dir / "en_PP-OCRv5_mobile_rec.onnx"
    dict_path = target_dir / "en_dict.txt"

    paddlex_root = Path.home() / ".paddlex" / "official_models"
    paddlex_det = paddlex_root / "PP-OCRv5_mobile_det_onnx" / "inference.onnx"
    paddlex_rec = paddlex_root / "en_PP-OCRv5_mobile_rec_onnx" / "inference.onnx"
    paddlex_yml = paddlex_root / "en_PP-OCRv5_mobile_rec_onnx" / "inference.yml"

    # 1. Detection model
    if not det_path.exists():
        if paddlex_det.exists():
            shutil.copyfile(paddlex_det, det_path)
        else:
            download_file(MODEL_URLS["PP-OCRv5_mobile_det.onnx"], det_path)

    # 2. Recognition model
    if not rec_path.exists():
        if paddlex_rec.exists():
            shutil.copyfile(paddlex_rec, rec_path)
        else:
            download_file(MODEL_URLS["en_PP-OCRv5_mobile_rec.onnx"], rec_path)

    # 3. English dictionary
    if not dict_path.exists():
        if paddlex_yml.exists():
            with open(paddlex_yml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            yml_tmp = target_dir / "rec_inference.tmp.yml"
            download_file(MODEL_URLS["inference.yml"], yml_tmp)
            with open(yml_tmp, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if yml_tmp.exists():
                try:
                    yml_tmp.unlink()
                except OSError:
                    pass

        chars = cfg.get("PostProcess", {}).get("character_dict", [])
        with open(dict_path, "w", encoding="utf-8") as f:
            for c in chars:
                f.write(f"{c}\n")

    return target_dir


# Ensure models exist and initialize RapidOCR engine
MODELS_DIR = ensure_models_exist()
ocr_engine = RapidOCR(
    det_model_path=str(MODELS_DIR / "PP-OCRv5_mobile_det.onnx"),
    rec_model_path=str(MODELS_DIR / "en_PP-OCRv5_mobile_rec.onnx"),
    rec_keys_path=str(MODELS_DIR / "en_dict.txt"),
    det_limit_side_len=736,
    det_box_thresh=0.60,
    det_unclip_ratio=1.5,
    det_mean=[0.485, 0.456, 0.406],
    det_std=[0.229, 0.224, 0.225],
    use_cls=False,
)

# In-memory store for asynchronous background extraction jobs
jobs_db: Dict[str, JobInfo] = {}


def try_extract_native_text(pdf_path: str) -> Tuple[Optional[List[str]], int]:
    """Attempt fast-path extraction directly from the PDF digital vector text layer."""
    doc = pdfium.PdfDocument(pdf_path)
    page_count = len(doc)
    if page_count == 0:
        return [], 0

    pages_text: List[str] = []
    total_characters: int = 0

    for page in doc:
        textpage = page.get_textpage()
        raw_text = textpage.get_text_range()
        normalized_page = normalize_extracted_text(raw_text)
        pages_text.append(normalized_page)
        total_characters += len(normalized_page)

    min_threshold = max(25, 20 * page_count)
    if total_characters >= min_threshold:
        full_text = " ".join(pages_text)
        sentences = [s.strip() for s in nltk.sent_tokenize(full_text) if s.strip()]
        if sentences:
            return sentences, page_count

    return None, page_count


def extract_via_rapidocr(
    pdf_path: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """Fallback to ONNX-accelerated RapidOCR vision when no digital text layer is present."""
    doc = pdfium.PdfDocument(pdf_path)
    total_pages = len(doc)
    extracted_lines: List[str] = []

    for page_idx in range(total_pages):
        if progress_callback:
            progress_callback(page_idx + 1, total_pages)

        page = doc[page_idx]
        img = np.array(page.render(scale=1.0).to_pil())
        result, _ = ocr_engine(img)
        if not result:
            continue

        dt_polys = [item[0] for item in result]
        rec_texts = [item[1] for item in result]
        rec_scores = [item[2] for item in result]

        page_lines = merge_and_sort_lines(dt_polys, rec_texts, rec_scores)
        extracted_lines.extend(page_lines)

    full_text = " ".join(extracted_lines)
    if not full_text.strip():
        return []

    sentences = nltk.sent_tokenize(full_text)
    return [s.strip() for s in sentences if s.strip()]



async def run_async_job(job_id: str, temp_path: str) -> None:
    """Background task executor for asynchronous extraction jobs."""
    job = jobs_db[job_id]
    job.status = JobStatus.PROCESSING
    job.updated_at = time.time()
    t0 = time.perf_counter()

    def on_progress(curr: int, total: int) -> None:
        job.current_page = curr
        job.total_pages = total
        job.progress = round(curr / max(total, 1), 2)
        job.message = f"Processing page {curr} of {total} with RapidOCR ONNX..."
        job.updated_at = time.time()

    try:
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

        ocr_sentences = await asyncio.to_thread(
            extract_via_rapidocr, temp_path, on_progress
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        job.result = SentenceResponse(
            sentences=ocr_sentences,
            method="rapidocr_onnx",
            page_count=job.total_pages,
            processing_time_ms=elapsed_ms,
        )
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.message = (
            "Extraction completed successfully via PP-OCRv5 Mobile RapidOCR ONNX."
        )

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

