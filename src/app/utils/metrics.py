from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.app.config import settings


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
        """Record a successful request execution with its latency."""
        async with self.lock:
            self.total_requests += 1
            self.success_requests += 1
            self.last_latency_ms = latency_ms
            self.latency_ms_sum += latency_ms
            self.latency_ms_count += 1
            self.last_error = None

    async def record_failure(self, latency_ms: Optional[float], error: str) -> None:
        """Record a failed request execution with error details."""
        async with self.lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.last_latency_ms = latency_ms
            self.last_error = error

    async def reset(self) -> None:
        """Reset all metric counters to zero."""
        async with self.lock:
            self.total_requests = 0
            self.success_requests = 0
            self.failed_requests = 0
            self.last_latency_ms = None
            self.latency_ms_sum = 0.0
            self.latency_ms_count = 0
            self.last_error = None

    async def snapshot(self, service_url: Optional[str] = None) -> Dict[str, Any]:
        """Produce an immutable snapshot of current metrics."""
        async with self.lock:
            avg = (
                self.latency_ms_sum / self.latency_ms_count
                if self.latency_ms_count > 0
                else None
            )
            return {
                "service_url": service_url or settings.SERVICE_URL,
                "timeout_seconds": settings.TIMEOUT_SECONDS,
                "connect_timeout_seconds": settings.CONNECT_TIMEOUT_SECONDS,
                "total_requests": self.total_requests,
                "success_requests": self.success_requests,
                "failed_requests": self.failed_requests,
                "last_latency_ms": self.last_latency_ms,
                "avg_latency_ms": avg,
                "last_error": self.last_error,
            }

