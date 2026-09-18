from __future__ import annotations

import asyncio
import pytest

from src.app.utils.metrics import Metrics


@pytest.mark.anyio
async def test_metrics_initial_state():
    """Verify clean initial state of metrics tracker."""
    m = Metrics()
    snap = await m.snapshot()
    assert snap["total_requests"] == 0
    assert snap["success_requests"] == 0
    assert snap["failed_requests"] == 0
    assert snap["last_latency_ms"] is None
    assert snap["avg_latency_ms"] is None
    assert snap["last_error"] is None


@pytest.mark.anyio
async def test_metrics_record_success():
    """Verify success recordings update totals and latency averages."""
    m = Metrics()
    await m.record_success(100.0)
    await m.record_success(200.0)

    snap = await m.snapshot()
    assert snap["total_requests"] == 2
    assert snap["success_requests"] == 2
    assert snap["failed_requests"] == 0
    assert snap["last_latency_ms"] == 200.0
    assert snap["avg_latency_ms"] == 150.0


@pytest.mark.anyio
async def test_metrics_record_failure():
    """Verify failure recording updates error messages and counters."""
    m = Metrics()
    await m.record_failure(latency_ms=50.0, error="Connection refused")

    snap = await m.snapshot()
    assert snap["total_requests"] == 1
    assert snap["success_requests"] == 0
    assert snap["failed_requests"] == 1
    assert snap["last_error"] == "Connection refused"
    assert snap["last_latency_ms"] == 50.0


@pytest.mark.anyio
async def test_metrics_reset():
    """Verify reset clears all metrics back to zeros."""
    m = Metrics()
    await m.record_success(120.0)
    await m.reset()

    snap = await m.snapshot()
    assert snap["total_requests"] == 0
    assert snap["success_requests"] == 0
    assert snap["last_latency_ms"] is None


@pytest.mark.anyio
async def test_metrics_concurrent_updates():
    """Verify thread-safety and lock synchronization under concurrent updates."""
    m = Metrics()

    async def worker():
        for _ in range(25):
            await m.record_success(10.0)

    await asyncio.gather(*(worker() for _ in range(4)))
    snap = await m.snapshot()
    assert snap["total_requests"] == 100
    assert snap["success_requests"] == 100
    assert snap["avg_latency_ms"] == 10.0

