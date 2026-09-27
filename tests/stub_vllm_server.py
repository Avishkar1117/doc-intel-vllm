"""A fixed-capacity stub server standing in for vLLM: exactly MAX_SLOTS requests run at
once, in SERVICE_TIME_S each, everything past that queues. It has a known, deterministic
capacity - so if benchmark.py's own load generator can reproduce the textbook queueing
curve against it (linear throughput below MAX_SLOTS, flat above it), the harness is
trusted before it ever meets the real GPU server (Phase 4's D-006 discipline, applied to
Phase 5's instrument instead of its dataset).
"""

import asyncio

from fastapi import FastAPI, Response

MAX_SLOTS = 8  # stands in for --max-num-seqs
SERVICE_TIME_S = 0.1  # fixed "decode time" per request - deterministic, so expected
# wall-clock time is checkable by hand: wall ~= n_requests / min(level, MAX_SLOTS) * SERVICE_TIME_S


def make_stub_app() -> FastAPI:
    """A fresh app + fresh semaphore per test - server state must never leak between tests."""
    app = FastAPI()
    semaphore = asyncio.Semaphore(MAX_SLOTS)
    state = {"running": 0, "waiting": 0}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        # Real vLLM /metrics format is Prometheus text exposition - "name value" per
        # line, comments start with '#'. parse_metrics() only reads these three names.
        text = (
            "# stub metrics\n"
            f"vllm:num_requests_running {state['running']}\n"
            f"vllm:num_requests_waiting {state['waiting']}\n"
            f"vllm:kv_cache_usage_perc {state['running'] / MAX_SLOTS}\n"
        )
        return Response(text, media_type="text/plain")

    @app.post("/v1/chat/completions")
    async def chat_completions() -> dict[str, object]:
        state["waiting"] += 1
        async with semaphore:
            state["waiting"] -= 1
            state["running"] += 1
            try:
                await asyncio.sleep(SERVICE_TIME_S)
            finally:
                state["running"] -= 1
        return {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550},
        }

    return app
