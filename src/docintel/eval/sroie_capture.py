"""Single-shot SROIE capture: the self-hosted side of Phase 8's DI comparison bench.

No repair loop (unlike eval/capture.py's CORD path, which goes through pipeline.py's
bounded repair): Document Intelligence has no retry/repair mechanism of its own, so
giving the self-hosted side one here would make the latency/cost comparison uneven by
construction, not just inaccurate. Reuses capture.py's health-poll/resumable/retry
shape directly rather than reinventing it (PROJECT_BRIEF.md §7 Phase 8, D-007).
"""

import argparse
import json
import os
import time
from pathlib import Path

import httpx
import structlog

from docintel.config import settings
from docintel.eval.capture import (
    MAX_TRANSIENT_RETRIES,
    TRANSIENT_RETRY_DELAY_S,
    _wait_until_healthy,
)
from docintel.eval.sroie import load_sroie_test_split
from docintel.extraction.client import HTTPClient
from docintel.extraction.prompts import SROIE_PROMPT_VERSION, build_sroie_extraction_prompt
from docintel.schemas import SroieReceipt, Usage

logger = structlog.get_logger(__name__)

# vLLM/Modal never report their own version at runtime (this script only speaks HTTP to
# the server, same as capture.py) - stamped per CLAUDE.md rule 5, kept in sync by hand.
VLLM_VERSION = "0.21.0"

# The config decided for Phase 8's first SROIE run: 4B, reduced resolution - your Phase 6
# pick (best measured F1 on CORD) and the stronger "cheap+fast, same accuracy" argument
# against DI. A second run at 4B/default-resolution (for a CORD-comparable number) is a
# deliberately separate, later decision - not baked in here as a second code path yet.
SERVER_CONFIG = {
    "model": os.environ.get("DOCINTEL_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
    # Matches benchmarks/measure.py's CAP_1500K exactly - Phase 6's actual "4B reduced
    # res" launch config, not a re-derived guess (the shortest_edge value here was
    # wrong - 28, not 65536 - until checked against measure.py directly before this run).
    "mm_processor_kwargs": os.environ.get(
        "DOCINTEL_MM_KWARGS", '{"size": {"longest_edge": 1500000, "shortest_edge": 65536}}'
    ),
    "max_model_len": 16384,
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 8,
}

DEFAULT_OUTPUT_PATH = Path("benchmarks/results/phase8_sroie_capture.jsonl")


def _complete_with_retry(
    image_bytes: bytes, prompt: str, client: HTTPClient, image_id: int
) -> tuple[tuple[str, Usage, float] | None, str | None]:
    """Retries transient 5xx/connection failures, same policy as capture.py's
    _extract_with_retry - but wraps client.complete() directly, not pipeline.extract(),
    since there's no repair loop here to call into. A 400 is re-raised immediately,
    uncaught: SROIE receipts are far lighter than CORD's worst case (D-022's encoder
    fix applies equally here), so a 400 would be a genuine, unexpected infra rejection
    worth seeing immediately, not something to silently retry into a different outcome.

    latency_ms times only the winning attempt's own call, not any earlier failed
    attempts or their retry sleeps - matches baseline.py's analyze_receipt, which
    likewise reports just one successful request's duration. Without this, the two
    sides' latency numbers wouldn't describe the same thing.
    """
    last_error: str | None = None
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            start = time.monotonic()
            raw_output, usage = client.complete(image_bytes, prompt, schema=SroieReceipt)
            latency_ms = (time.monotonic() - start) * 1000
            return (raw_output, usage, latency_ms), None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                raise
            last_error = str(exc)
        except httpx.TransportError as exc:
            last_error = str(exc)
        logger.warning(
            "sroie_capture_transient_failure", image_id=image_id, attempt=attempt, error=last_error
        )
        time.sleep(TRANSIENT_RETRY_DELAY_S)
    return None, last_error


def capture(base_url: str, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Runs every SROIE test-split receipt through one single-shot request, appending
    each result as one JSON line. Resumable, same as capture.py: already-captured
    image_ids are skipped on a re-run.
    """
    logger.info("sroie_capture_waiting_for_healthy", base_url=base_url)
    _wait_until_healthy(base_url)
    client = HTTPClient(base_url=base_url, api_key=settings.vllm_api_key)
    prompt = build_sroie_extraction_prompt(SroieReceipt)

    already_captured: set[int] = set()
    if output_path.exists():
        with output_path.open() as f:
            already_captured = {json.loads(line)["image_id"] for line in f if line.strip()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as out:
        for sample in load_sroie_test_split():
            if sample.image_id in already_captured:
                continue

            record: dict[str, object] = {
                "image_id": sample.image_id,
                "raw_entities": sample.raw_entities,
                "vllm_version": VLLM_VERSION,
                "server_config": SERVER_CONFIG,
                "prompt_version": SROIE_PROMPT_VERSION,
                "outcome": "ok",
                "raw_output": None,
                "usage": None,
                "latency_ms": None,
                "error": None,
            }
            try:
                result, transient_error = _complete_with_retry(
                    sample.image_bytes, prompt, client, sample.image_id
                )
            except httpx.HTTPStatusError as exc:
                record["outcome"] = "infra_rejected"
                record["error"] = str(exc)
                logger.warning(
                    "sroie_capture_infra_rejected", image_id=sample.image_id, error=str(exc)
                )
            else:
                if result is None:
                    record["outcome"] = "transient_failure"
                    record["error"] = transient_error
                    logger.error(
                        "sroie_capture_transient_failure_exhausted",
                        image_id=sample.image_id,
                        error=transient_error,
                    )
                else:
                    raw_output, usage, latency_ms = result
                    record["raw_output"] = raw_output
                    record["usage"] = usage.model_dump(mode="json")
                    record["latency_ms"] = latency_ms
                    # Single-shot, no repair budget - a validation failure here just is
                    # the outcome, not a signal to retry (report.py's scoring step is
                    # where "unparseable" actually gets counted, same D-006 split as
                    # CORD: capture stays close to raw output, scoring iterates offline).
                    logger.info("sroie_capture_ok", image_id=sample.image_id)

            out.write(json.dumps(record) + "\n")
            out.flush()

    logger.info("sroie_capture_done", output_path=str(output_path))


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
