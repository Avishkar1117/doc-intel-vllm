import argparse
import asyncio
import base64
import contextlib
import csv
import json
import statistics
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from docintel.config import settings
from docintel.eval.capture import SERVER_CONFIG, VLLM_VERSION, _wait_until_healthy
from docintel.eval.cord import load_cord_test_split
from docintel.extraction.prompts import PROMPT_VERSION, build_extraction_prompt
from docintel.schemas import Receipt

CONCURRENCY_LEVELS = [1,2,4,8,16,32]
REQUESTS_PER_LEVEL = 48
WARMUP_PER_SLOT = 2
REQUEST_TIMEOUT_S = 600.0
METRICS_INTERVAL_S = 1.0
RTT_PROBES = 10
MANIFEST_PATH = Path("benchmarks/results/phase6_workload.json")
OUTPUT_DIR = Path("benchmarks/results")

METRIC_NAMES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",

}

@dataclass
class RequestResult:
    level: int
    request_idx: int
    image_id: int
    latency_s: float
    ok: bool
    status: str  # "ok" | "http_<code>" | "timeout" | "conn_error"
    prompt_tokens: int
    completion_tokens: int


@dataclass
class MetricSample:
    level: int
    t_s: float  # seconds since this level's measured batch started
    running: float
    waiting: float
    kv_usage: float


def build_payload(image_b64: str, prompt: str) -> dict[str, Any]:
    """Same body HTTPClient.complete builds; rebuilt here because that client is sync and
    the benchmark needs one raw request per doc (no pipeline repair loop multiplying load)."""
    return {
        "model": settings.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Receipt", "schema": Receipt.model_json_schema()},
        },
    }


async def send_one(
    client: httpx.AsyncClient, payload: dict[str, Any], level: int, request_idx: int, image_id: int
) -> RequestResult:
    start = time.perf_counter()
    try:
        response = await client.post("/v1/chat/completions", json=payload)
        latency = time.perf_counter() - start
        if response.status_code != 200:
            return RequestResult(level, request_idx, image_id, latency, False,
                                 f"http_{response.status_code}", 0, 0)
        usage = response.json()["usage"]
        return RequestResult(level, request_idx, image_id, latency, True, "ok",
                             usage["prompt_tokens"], usage["completion_tokens"])
    # TimeoutException is a subclass of TransportError, so it must be caught first.
    except httpx.TimeoutException:
        return RequestResult(level, request_idx, image_id, time.perf_counter() - start,
                             False, "timeout", 0, 0)
    except httpx.TransportError:
        return RequestResult(level, request_idx, image_id, time.perf_counter() - start,
                             False, "conn_error", 0, 0)


async def worker(
    client: httpx.AsyncClient,
    jobs: Iterator[tuple[int, int]],
    payloads: dict[int, dict[str, Any]],
    level: int,
    results: list[RequestResult],
) -> None:
    # All workers share one iterator: whichever finishes first takes the next job, so
    # exactly `level` requests are in flight at all times. That is what closed-loop means.
    for request_idx, image_id in jobs:
        results.append(await send_one(client, payloads[image_id], level, request_idx, image_id))


async def run_batch(
    client: httpx.AsyncClient,
    payloads: dict[int, dict[str, Any]],
    image_ids: list[int],
    level: int,
    n_requests: int,
) -> tuple[list[RequestResult], float]:
    """Runs n_requests at fixed concurrency; returns results and the batch's wall-clock seconds."""
    jobs = iter(enumerate(islice(cycle(image_ids), n_requests)))
    results: list[RequestResult] = []
    start = time.perf_counter()
    await asyncio.gather(*(worker(client, jobs, payloads, level, results) for _ in range(level)))
    return results, time.perf_counter() - start


def parse_metrics(text: str) -> dict[str, float]:
    """Pulls our three gauges out of Prometheus text format ('name{labels} value')."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key, name in METRIC_NAMES.items():
            if line.startswith(name) and line[len(name):len(name) + 1] in (" ", "{"):
                out[key] = float(line.rsplit(" ", 1)[1])
    return out


async def sample_metrics(
    client: httpx.AsyncClient, level: int, stop: asyncio.Event, samples: list[MetricSample]
) -> None:
    """Polls /metrics once a second while a level runs: this is how queueing becomes visible."""
    t0 = time.perf_counter()
    nan = float("nan")
    while not stop.is_set():
        try:
            metrics = parse_metrics((await client.get("/metrics", timeout=5.0)).text)
            samples.append(
                MetricSample(
                    level,
                    time.perf_counter() - t0,
                    metrics.get("running", nan),
                    metrics.get("waiting", nan),
                    metrics.get("kv_usage", nan),
                )
            )
        except httpx.HTTPError:
            pass  # a missed sample is fine; the load itself must never depend on this
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=METRICS_INTERVAL_S)


async def measure_rtt(client: httpx.AsyncClient) -> float:
    """Median /health round trip: the network offset baked into every latency (decision A3)."""
    times = []
    for _ in range(RTT_PROBES):
        start = time.perf_counter()
        await client.get("/health")
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def percentile(sorted_values: list[float], p: float) -> float:
    """p in [0, 100]; input is already sorted ascending and non-empty."""
    if not sorted_values:
        return float("nan")
    p_int = int(p)
    return float(statistics.quantiles(sorted_values, n=100, method="inclusive")[p_int - 1])




def throughput_docs_per_min(results: list[RequestResult], wall_s: float) -> float:
    """Docs per minute for one concurrency level."""
    return sum(r.ok for r in results) / wall_s * 60


def summarize_level(level: int, results: list[RequestResult], wall_s: float) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    latencies: list[float] = [r.latency_s for r in ok] 
    n_ok = max(len(ok), 1)  # guards the means if a whole level failed
    return {
        "concurrency": level,
        "n_requests": len(results),
        "n_ok": len(ok),
        "error_rate": 1 - len(ok) / len(results),
        "wall_s": wall_s,
        "docs_per_min": throughput_docs_per_min(results, wall_s),
        "output_tok_per_s": sum(r.completion_tokens for r in ok) / wall_s,
        "p50_s": percentile(latencies, 50),
        "p95_s": percentile(latencies, 95),
        "p99_s": percentile(latencies, 99),
        "mean_prompt_tokens": sum(r.prompt_tokens for r in ok) / n_ok,
        "mean_completion_tokens": sum(r.completion_tokens for r in ok) / n_ok,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


async def run_sweep(
    base_url: str,
    levels: list[int],
    requests_per_level: int,
    manifest_path: Path,
    tag: str,
) -> None:
    image_ids: list[int] = json.loads(manifest_path.read_text())["image_ids"]
    wanted = set(image_ids)
    prompt = build_extraction_prompt()
    # Base64-encode once up front so encoding CPU time never lands inside a latency number.
    payloads = {
        s.image_id: build_payload(base64.b64encode(s.image_bytes).decode("ascii"), prompt)
        for s in load_cord_test_split()
        if s.image_id in wanted
    }

    _wait_until_healthy(base_url)  # blocks; fine, no load is running yet
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    request_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    limits = httpx.Limits(max_connections=max(levels) + 2)  # +2: metrics sampler + slack
    async with httpx.AsyncClient(
        base_url=base_url, timeout=REQUEST_TIMEOUT_S, limits=limits
    ) as client:
        rtt_s = await measure_rtt(client)
        # Stamped on every row: a benchmark without a version is unreproducible (CLAUDE.md).
        stamp = {
            "run_id": run_id,
            "tag": tag,
            "vllm_version": VLLM_VERSION,
            "prompt_version": PROMPT_VERSION,
            **SERVER_CONFIG,
            "rtt_s": rtt_s,
            "requests_per_level": requests_per_level,
        }

        for level in levels:
            await run_batch(client, payloads, image_ids, level, WARMUP_PER_SLOT * level)

            stop = asyncio.Event()
            samples: list[MetricSample] = []
            sampler = asyncio.create_task(sample_metrics(client, level, stop, samples))
            results, wall_s = await run_batch(
                client, payloads, image_ids, level, requests_per_level
            )
            stop.set()
            await sampler

            summary = summarize_level(level, results, wall_s)
            level_rows.append({**stamp, **summary})
            request_rows += [{**stamp, **asdict(r)} for r in results]
            metric_rows += [{"run_id": run_id, **asdict(s)} for s in samples]
            # Rewritten after every level so a crash or a budget stop never loses finished levels.
            write_csv(OUTPUT_DIR / f"{tag}_{run_id}_levels.csv", level_rows)
            write_csv(OUTPUT_DIR / f"{tag}_{run_id}_requests.csv", request_rows)
            if metric_rows:
                write_csv(OUTPUT_DIR / f"{tag}_{run_id}_metrics.csv", metric_rows)
            print(f"c={level}: {summary['docs_per_min']:.1f} docs/min, "
                  f"p50={summary['p50_s']:.1f}s p95={summary['p95_s']:.1f}s "
                  f"errors={summary['error_rate']:.0%}")
    print(f"run_id={run_id}")


def analyze(run_id: str, tag: str) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: write PNGs, never open a window
    import matplotlib.pyplot as plt

    levels = pd.read_csv(OUTPUT_DIR / f"{tag}_{run_id}_levels.csv")
    metrics = pd.read_csv(OUTPUT_DIR / f"{tag}_{run_id}_metrics.csv")
    max_seqs = int(levels["max_num_seqs"].iloc[0])
    peaks = metrics.groupby("level")[["running", "waiting", "kv_usage"]].max()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(levels["concurrency"], levels["docs_per_min"], marker="o")
    axes[0].set_title("Throughput")
    axes[0].set_ylabel("docs/min")
    for col in ("p50_s", "p95_s", "p99_s"):
        axes[1].plot(levels["concurrency"], levels[col], marker="o", label=col[:3])
    axes[1].set_title("Latency")
    axes[1].set_ylabel("seconds")
    axes[1].legend()
    axes[2].plot(peaks.index, peaks["running"], marker="o", label="running (peak)")
    axes[2].plot(peaks.index, peaks["waiting"], marker="o", label="waiting (peak)")
    axes[2].set_title("Server queue")
    axes[2].set_ylabel("requests")
    axes[2].legend(loc="upper left")
    kv_ax = axes[2].twinx()  # KV usage is a 0-1 fraction; its own axis keeps it readable
    kv_ax.plot(peaks.index, peaks["kv_usage"], marker="s", color="gray", linestyle=":")
    kv_ax.set_ylabel("KV cache usage (peak)")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(levels["concurrency"]))
        ax.set_xticklabels([str(c) for c in levels["concurrency"]])
        ax.axvline(max_seqs, color="gray", linestyle="--")  # where batching should saturate
        ax.set_xlabel("concurrency")
    out = OUTPUT_DIR / f"{tag}_{run_id}_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run the concurrency sweep")
    run_p.add_argument("--base-url", default=settings.vllm_base_url)
    run_p.add_argument("--levels", type=int, nargs="+", default=CONCURRENCY_LEVELS)
    run_p.add_argument("--requests-per-level", type=int, default=REQUESTS_PER_LEVEL)
    run_p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    run_p.add_argument("--tag", default="phase5")
    an_p = sub.add_parser("analyze", help="plot curves from a finished run")
    an_p.add_argument("--run-id", required=True)
    an_p.add_argument("--tag", default="phase5")
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(
            run_sweep(
                args.base_url, args.levels, args.requests_per_level, args.manifest, args.tag
            )
        )
    else:
        analyze(args.run_id, args.tag)


if __name__ == "__main__":
    main()

