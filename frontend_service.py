from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from dataset import SELFTEST_DATASET

DEFAULT_SERVICE_URL = os.environ.get(
    "SENTENCE_SERVICE_URL", "http://localhost:8000/v1/extract-sentences"
)
TIMEOUT_SECONDS = float(os.environ.get("SERVICE_TIMEOUT_SECONDS", "90"))
TIMEOUT_SECONDS = float(os.environ.get("SERVICE_TIMEOUT_SECONDS", "180"))
CONNECT_TIMEOUT_SECONDS = float(os.environ.get("SERVICE_CONNECT_TIMEOUT_SECONDS", "10"))
APP_DIR = Path(__file__).resolve().parent


@dataclass
class Metrics:
    """In-memory operational metrics for request tracking and latency measurement."""

    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0
    last_latency_ms: Optional[float] = None
    latency_ms_sum: float = 0.0
    latency_ms_count: int = 0
    last_error: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record_success(self, latency_ms: float) -> None:
        async with self.lock:
            self.total_requests += 1
            self.success_requests += 1
            self.last_latency_ms = latency_ms
            self.latency_ms_sum += latency_ms
            self.latency_ms_count += 1
            self.last_error = None

    async def record_failure(self, latency_ms: Optional[float], error: str) -> None:
        async with self.lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.last_latency_ms = latency_ms
            self.last_error = error

    async def snapshot(self, service_url: str = DEFAULT_SERVICE_URL) -> Dict[str, Any]:
        async with self.lock:
            avg = (
                self.latency_ms_sum / self.latency_ms_count
                if self.latency_ms_count > 0
                else None
            )
            return {
                "service_url": service_url,
                "timeout_seconds": TIMEOUT_SECONDS,
                "connect_timeout_seconds": CONNECT_TIMEOUT_SECONDS,
                "total_requests": self.total_requests,
                "success_requests": self.success_requests,
                "failed_requests": self.failed_requests,
                "last_latency_ms": self.last_latency_ms,
                "avg_latency_ms": avg,
                "last_error": self.last_error,
            }


def pedagogic_http_error(ex: Exception, service_url: str) -> str:
    """Convert low-level exceptions into clear, educational explanations."""
    if isinstance(ex, httpx.ConnectError):
        return (
            "Could not connect to the sentence-extraction service.\n"
            f"- Configured service URL: {service_url}\n"
            "- Is the service running, and is the URL correct?\n"
            "- If running in Docker, check port mappings.\n"
        )
    if isinstance(ex, httpx.ReadTimeout):
        return (
            "The service did not respond before the timeout.\n"
            f"- Current timeout: {TIMEOUT_SECONDS} seconds\n"
            "- Large PDFs can take time on CPU.\n"
            "- Consider optimizing the service or increasing SERVICE_TIMEOUT_SECONDS.\n"
        )
    if isinstance(ex, httpx.RemoteProtocolError):
        return (
            "The connection was established, but the HTTP protocol exchange failed.\n"
            "- Check the service logs.\n"
        )
    return (
        "An unexpected error occurred while calling the service.\n"
        f"- Error type: {type(ex).__name__}\n"
        f"- Details: {ex}\n"
    )


async def call_sentence_service(
    pdf_bytes: bytes, filename: str, service_url: str = DEFAULT_SERVICE_URL
) -> Tuple[List[str], float, Dict[str, Any]]:
    """Call the sentence extraction backend service and return sentences, latency, and metadata.

    Parameters
    ----------
    pdf_bytes : bytes
        Binary content of the PDF.
    filename : str
        Original filename for multipart headers.
    service_url : str
        Target backend service endpoint.

    Returns
    -------
    Tuple[List[str], float, Dict[str, Any]]
        Extracted sentences, latency in ms, and metadata dict (method, page_count).
    """
    timeout = httpx.Timeout(
        TIMEOUT_SECONDS,
        connect=CONNECT_TIMEOUT_SECONDS,
        read=TIMEOUT_SECONDS,
        write=TIMEOUT_SECONDS,
        pool=TIMEOUT_SECONDS,
    )

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        files = {"pdf_file": (filename, pdf_bytes, "application/pdf")}
        resp = await client.post(service_url, files=files)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if resp.status_code != 200:
        raise ValueError(
            f"Service returned error HTTP status {resp.status_code}.\n"
            f"Response body (first 2k chars): {resp.text[:2000]}"
        )

    data = resp.json()
    if (
        not isinstance(data, dict)
        or "sentences" not in data
        or not isinstance(data["sentences"], list)
    ):
        raise ValueError(
            "Service response JSON did not match expected schema {'sentences': [...]}.\n"
            f"Got: {data}"
        )

    sentences = [str(s) for s in data["sentences"]]
    metadata = {
        "method": data.get("method"),
        "page_count": data.get("page_count"),
        "processing_time_ms": data.get("processing_time_ms"),
    }
    return sentences, latency_ms, metadata


async def inspect_pdf_service(
    pdf_bytes: bytes, filename: str, service_url: str = DEFAULT_SERVICE_URL
) -> Dict[str, Any]:
    """Inspect PDF to estimate page count and duration."""
    inspect_url = service_url.replace("/v1/extract-sentences", "/v1/inspect-pdf")
    timeout = httpx.Timeout(10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        files = {"pdf_file": (filename, pdf_bytes, "application/pdf")}
        resp = await client.post(inspect_url, files=files)

    if resp.status_code == 200:
        return resp.json()
    raise ValueError(f"Inspect failed with status {resp.status_code}: {resp.text[:200]}")


async def run_selftest(service_url: str = DEFAULT_SERVICE_URL) -> Dict[str, Any]:
    """Execute validation tests using reference sample PDFs."""
    results: List[Dict[str, Any]] = []
    passed = 0

    for item in SELFTEST_DATASET:
        fname = item["filename"]
        expected = item["sentences"]

        path = APP_DIR / fname
        if not path.exists():
            path = APP_DIR / "examples" / fname

        if not path.exists():
            results.append(
                {
                    "filename": fname,
                    "ok": False,
                    "error": f"Sample PDF '{fname}' not found in app or examples directory.",
                    "missing_sentences": expected,
                    "latency_ms": None,
                    "num_returned_sentences": None,
                }
            )
            continue

        try:
            pdf_bytes = path.read_bytes()
            sentences, latency_ms, meta = await call_sentence_service(
                pdf_bytes, fname, service_url=service_url
            )

            returned_set = set(sentences)
            missing = [s for s in expected if s not in returned_set]

            ok = len(missing) == 0
            if ok:
                passed += 1

            results.append(
                {
                    "filename": fname,
                    "ok": ok,
                    "method": meta.get("method"),
                    "page_count": meta.get("page_count"),
                    "error": (
                        None
                        if ok
                        else "Some expected sentences were not found in the output."
                    ),
                    "missing_sentences": missing,
                    "latency_ms": latency_ms,
                    "num_returned_sentences": len(sentences),
                }
            )
        except Exception as ex:
            results.append(
                {
                    "filename": fname,
                    "ok": False,
                    "error": (
                        pedagogic_http_error(ex, service_url)
                        if isinstance(ex, httpx.HTTPError)
                        else str(ex)
                    ),
                    "missing_sentences": expected,
                    "latency_ms": None,
                    "num_returned_sentences": None,
                }
            )

    return {
        "service_url": service_url,
        "passed": passed,
        "total": len(SELFTEST_DATASET),
        "results": results,
    }
