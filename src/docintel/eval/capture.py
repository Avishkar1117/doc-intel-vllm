import argparse
import json
import os
import time
from pathlib import Path

import httpx
import structlog

from docintel.eval.cord import load_cord_test_split
from docintel.extraction.client import HTTPClient
from docintel.extraction.pipeline import extract
from docintel.extraction.prompts import PROMPT_VERSION
from docintel.schemas import ExtractionResponse

logger = structlog.get_logger(__name__)

# Stamped into every captured record per CLAUDE.md rule 5 ("pin the vLLM version...
# write it into every benchmark CSV") - mirrors the pin in pyproject.toml's
# modal-serving extra / modal_app.py's image build. Not re-derived at runtime: this
# script never imports vllm itself, it only ever talks HTTP to the Modal server.
VLLM_VERSION = "0.21.0"


SERVER_CONFIG = {
    "model": os.environ.get("DOCINTEL_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
    "mm_processor_kwargs": os.environ.get("DOCINTEL_MM_KWARGS", "{}"),
    "max_model_len": 16384,
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 8,
}

DEFAULT_OUTPUT_PATH = Path("benchmarks/results/phase4_cord_capture.jsonl")

# modal_app.py: scaledown_window=15 minutes. Any gap in requests longer than that (the
# time this loop spends writing/flushing between receipts is nowhere close, but a
# paused/resumed run easily is) means the very next request hits a cold container, not
# a warm one - so every run polls /health first rather than assuming last time's warm
# check still holds.
HEALTH_POLL_TIMEOUT_S = 10 * 60
HEALTH_POLL_INTERVAL_S = 5.0
HEALTHY_STREAK_REQUIRED = 3  # a single 200 right after redeploy can be a stale replica (Phase 2)

# Transient failures (cold start straddling a request, a dropped connection) are retried;
# a 400 is not - that is D-014's real, per-image encoder-cache-ceiling limit, not a blip.
MAX_TRANSIENT_RETRIES = 3
TRANSIENT_RETRY_DELAY_S = 20.0


def _wait_until_healthy(base_url: str) -> None:
    """Blocks until /health returns several consecutive 200s. Same pattern as
    modal_app.py's own `test()` entrypoint - a fresh cold start (first run, or the
    container scaled to zero since the last request) takes minutes, and a real request
    fired at a not-yet-ready server 503s or has its connection dropped mid-response,
    exactly what happened the first time this script ran without this check."""
    deadline = time.monotonic() + HEALTH_POLL_TIMEOUT_S
    streak = 0
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=10).status_code == 200:
                streak += 1
                if streak >= HEALTHY_STREAK_REQUIRED:
                    return
            else:
                streak = 0
        except httpx.RequestError:
            streak = 0
        time.sleep(HEALTH_POLL_INTERVAL_S)
    raise TimeoutError(f"{base_url} never became healthy within {HEALTH_POLL_TIMEOUT_S}s")


def _extract_with_retry(
    image_bytes: bytes, client: HTTPClient, image_id: int
) -> tuple[ExtractionResponse | None, str | None]:
    """Runs extract(), retrying transient 5xx/connection failures (a cold start
    straddling this request, a dropped connection) up to MAX_TRANSIENT_RETRIES times.
    A 400 is re-raised immediately, uncaught - the caller's own except handles it as the
    real D-014 infra-rejection, which is not a blip and must not be retried into one.

    Returns (response, None) on success, (None, last_error) if every retry was
    exhausted - a real, unresolved gap, kept distinct from both success and the D-014
    case rather than folded into either.
    """
    last_error: str | None = None
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            return extract(image_bytes, client), None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                raise
            last_error = str(exc)
        except httpx.TransportError as exc:
            last_error = str(exc)
        logger.warning(
            "capture_transient_failure", image_id=image_id, attempt=attempt, error=last_error
        )
        time.sleep(TRANSIENT_RETRY_DELAY_S)
    return None, last_error


def capture(base_url: str, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Runs every CORD test-split receipt through the live pipeline once, appending
    each result as one JSON line.

    Resumable: already-captured image_ids are skipped on a re-run instead of
    re-spending GPU time on them, so a crash on receipt 91 doesn't cost the first 90.
    """
    logger.info("capture_waiting_for_healthy", base_url=base_url)
    _wait_until_healthy(base_url)
    client = HTTPClient(base_url=base_url)

    already_captured: set[int] = set()
    if output_path.exists():
        with output_path.open() as f:
            already_captured = {json.loads(line)["image_id"] for line in f if line.strip()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as out:
        for sample in load_cord_test_split():
            if sample.image_id in already_captured:
                continue

            record: dict[str, object] = {
                "image_id": sample.image_id,
                "raw_ground_truth": sample.raw_ground_truth,
                "vllm_version": VLLM_VERSION,
                "server_config": SERVER_CONFIG,
                "prompt_version": PROMPT_VERSION,
                "outcome": "ok",
                "prediction": None,
                "error": None,
            }
            try:
                response, transient_error = _extract_with_retry(
                    sample.image_bytes, client, sample.image_id
                )
            except httpx.HTTPStatusError as exc:
                # The known D-014 failure mode: a receipt whose image needs more
                # encoder-cache tokens than the ~2048-token ceiling hard-fails with a
                # 400 before any generation happens. Tracked as a coverage gap here,
                # never as a wrong prediction - see eval/report.py's coverage accounting.
                record["outcome"] = "infra_rejected"
                record["error"] = str(exc)
                logger.warning("capture_infra_rejected", image_id=sample.image_id, error=str(exc))
            else:
                if response is None:
                    # Every transient retry was exhausted - a real gap worth
                    # investigating by hand, but not D-014's per-image capacity limit,
                    # so it must not be counted as coverage loss.
                    record["outcome"] = "transient_failure"
                    record["error"] = transient_error
                    logger.error(
                        "capture_transient_failure_exhausted",
                        image_id=sample.image_id,
                        error=transient_error,
                    )
                else:
                    record["prediction"] = response.model_dump(mode="json")
                    logger.info(
                        "capture_ok",
                        image_id=sample.image_id,
                        validation_passed=response.validation.passed,
                    )

            out.write(json.dumps(record) + "\n")
            out.flush()  # each receipt is its own unit of progress against the resume logic above

    logger.info("capture_done", output_path=str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="Deployed Modal server URL, e.g. https://....modal.direct"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    capture(args.base_url, args.output)


if __name__ == "__main__":
    main()
