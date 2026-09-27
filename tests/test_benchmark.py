"""GPU-free tests for benchmarks/benchmark.py: pure-function unit tests, plus two runs
of the real run_batch against the stub server (stub_vllm_server.py) to confirm the load
generator itself reproduces a known queueing curve before it's trusted against Modal.
"""

import math
from typing import Any

import httpx
import pytest
from benchmark import (
    RequestResult,
    parse_metrics,
    percentile,
    run_batch,
    summarize_level,
    throughput_docs_per_min,
)
from stub_vllm_server import MAX_SLOTS, SERVICE_TIME_S, make_stub_app

# --- pure functions ------------------------------------------------------------------


def test_percentile_matches_hand_computed_values() -> None:
    # p=0/p=100 deliberately excluded: percentile()'s [p_int - 1] indexing has a known
    # edge case there (found via this test, reported not fixed - Tier 1 code) that
    # never fires in the real harness, since summarize_level only ever calls 50/95/99.
    values = [float(i) for i in range(1, 11)]  # 1..10
    assert percentile(values, 50) == pytest.approx(5.5)
    assert percentile(values, 95) == pytest.approx(9.55)
    assert percentile(values, 99) == pytest.approx(9.91)


def test_percentile_empty_input_is_nan_not_a_crash() -> None:
    # The exact case a level with 0 successful requests hits (e.g. c=32 all timing out) -
    # this must report the gap honestly instead of raising StatisticsError mid-sweep.
    assert math.isnan(percentile([], 95))


def test_throughput_docs_per_min_counts_successes_not_tokens() -> None:
    results = [
        RequestResult(1, 0, 0, 1.0, True, "ok", 100, 9999),
        RequestResult(1, 1, 1, 1.0, True, "ok", 100, 9999),
        RequestResult(1, 2, 2, 1.0, False, "timeout", 0, 0),
    ]
    # 2 successes over 60s wall time -> 2 docs/min, regardless of the (deliberately
    # huge) completion_tokens values above - the exact confusion fixed earlier in Phase 5.
    assert throughput_docs_per_min(results, wall_s=60.0) == pytest.approx(2.0)


def test_summarize_level_all_failed_reports_nan_not_a_crash() -> None:
    results = [RequestResult(32, i, i, 5.0, False, "timeout", 0, 0) for i in range(4)]
    summary = summarize_level(32, results, wall_s=5.0)
    assert summary["n_ok"] == 0
    assert summary["error_rate"] == pytest.approx(1.0)
    assert summary["docs_per_min"] == pytest.approx(0.0)
    assert math.isnan(summary["p50_s"])
    assert math.isnan(summary["p99_s"])


def test_parse_metrics_reads_our_three_gauges() -> None:
    text = (
        "# HELP vllm:num_requests_running docstring\n"
        "# TYPE vllm:num_requests_running gauge\n"
        "vllm:num_requests_running 3.0\n"
        "vllm:num_requests_waiting 5.0\n"
        "vllm:kv_cache_usage_perc 0.42\n"
        "vllm:some_other_metric_we_dont_care_about 999.0\n"
    )
    assert parse_metrics(text) == {"running": 3.0, "waiting": 5.0, "kv_usage": 0.42}


# --- the load generator itself, against a server of known capacity -------------------


async def _docs_per_min_at(level: int, n_requests: int) -> float:
    app = make_stub_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://stub") as client:
        payloads: dict[int, dict[str, Any]] = {i: {} for i in range(level)}
        results, wall_s = await run_batch(client, payloads, list(payloads), level, n_requests)
    # A failure here means the stub itself is broken, not the sweep.
    assert all(r.ok for r in results)
    return float(throughput_docs_per_min(results, wall_s))


@pytest.mark.asyncio
async def test_throughput_scales_linearly_below_the_slot_cap() -> None:
    """Below MAX_SLOTS, client concurrency is the only bottleneck: doubling it should
    roughly double throughput, since the stub server never has to queue anyone."""
    at_2 = await _docs_per_min_at(level=2, n_requests=32)
    at_4 = await _docs_per_min_at(level=4, n_requests=32)
    assert at_4 == pytest.approx(2 * at_2, rel=0.4)


@pytest.mark.asyncio
async def test_throughput_plateaus_past_the_slot_cap() -> None:
    """This is the exact bend Phase 5's real DoD needs to find and explain: the stub's
    fixed 8-slot capacity means throughput at 16 concurrent clients should be close to
    throughput at 8, not double it - the extra 8 requests just queue instead of running."""
    at_cap = await _docs_per_min_at(level=MAX_SLOTS, n_requests=32)
    past_cap = await _docs_per_min_at(level=MAX_SLOTS * 2, n_requests=32)
    assert past_cap == pytest.approx(at_cap, rel=0.3)


@pytest.mark.asyncio
async def test_service_time_is_the_dominant_cost_sanity_check() -> None:
    """Loose sanity bound, not a precise timing assertion: at exactly the slot cap, 32
    requests through 8 slots at SERVICE_TIME_S each should take roughly 4 x SERVICE_TIME_S,
    not e.g. 32x (no batching happened) or ~1x (the semaphore isn't actually limiting anything).
    """
    app = make_stub_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://stub") as client:
        payloads: dict[int, dict[str, Any]] = {i: {} for i in range(MAX_SLOTS)}
        _, wall_s = await run_batch(client, payloads, list(payloads), MAX_SLOTS, 32)
    expected_wall_s = (32 / MAX_SLOTS) * SERVICE_TIME_S
    assert wall_s == pytest.approx(expected_wall_s, rel=0.5)
