"""Single-shot SROIE capture: the Document Intelligence side of Phase 8's comparison
bench. Same shape as eval/sroie_capture.py (resumable JSONL, one line per receipt),
but DI's actual failure modes are different from Modal's, so the retry policy is its
own, not reused wholesale: no cold-start health-poll (DI is an always-on managed
service, nothing to warm up), but rate limiting (429) is real and Modal never has to
handle it - a single self-hosted server isn't rate-limiting itself.
"""

import argparse
import json
import time
from pathlib import Path

import httpx
import structlog

from docintel.baseline import API_VERSION, analyze_receipt
from docintel.eval.sroie import load_sroie_test_split
from docintel.schemas import SroieReceipt

logger = structlog.get_logger(__name__)

DEFAULT_OUTPUT_PATH = Path("benchmarks/results/phase8_sroie_di_capture.jsonl")

# 429s and connection blips are retried; a DI-side "failed" analysis (RuntimeError) or
# a missing Total field (ValueError) are per-receipt outcomes, not blips - retrying the
# same image won't change what DI extracts from it. An auth/config error (401/403) is
# deliberately NOT caught here at all: it means every subsequent request will fail the
# same way, so it should crash the whole run immediately rather than burn through 347
# retries recording the same wrong outcome.
MAX_TRANSIENT_RETRIES = 3
TRANSIENT_RETRY_DELAY_S = 15.0


def _analyze_with_retry(
    image_bytes: bytes, image_id: int
) -> tuple[tuple[SroieReceipt, float] | None, str | None, str]:
    """Returns (result, error, outcome). outcome is one of "ok", "di_failed"
    (RuntimeError - DI's own analysis failed for this document), "no_total" (DI
    returned no Total field at all - baseline.py raises ValueError for this), or
    "transient_failure" (every retry exhausted).
    """
    last_error: str | None = None
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            return analyze_receipt(image_bytes), None, "ok"
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise  # config/auth problem - every future request fails identically, stop now
            if exc.response.status_code == 429:
                # Respect Retry-After when DI sends one; otherwise fall back to the
                # fixed delay - a 429 without a hint still needs *some* backoff.
                retry_after = exc.response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else TRANSIENT_RETRY_DELAY_S
                logger.warning("di_capture_rate_limited", image_id=image_id, delay_s=delay)
                time.sleep(delay)
                last_error = str(exc)
                continue
            last_error = str(exc)
        except httpx.TransportError as exc:
            last_error = str(exc)
        except RuntimeError as exc:
            return None, str(exc), "di_failed"
        except ValueError as exc:
            return None, str(exc), "no_total"

        logger.warning(
            "di_capture_transient_failure", image_id=image_id, attempt=attempt, error=last_error
        )
        time.sleep(TRANSIENT_RETRY_DELAY_S)
    return None, last_error, "transient_failure"


def capture(output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Runs every SROIE test-split receipt through one single-shot DI call, appending
    each result as one JSON line. Resumable, same convention as every other capture
    script in this project.
    """
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
                "di_api_version": API_VERSION,
                "outcome": "ok",
                "prediction": None,
                "latency_ms": None,
                "error": None,
            }

            result, error, outcome = _analyze_with_retry(sample.image_bytes, sample.image_id)
            record["outcome"] = outcome
            if result is not None:
                receipt, latency_ms = result
                record["prediction"] = receipt.model_dump(mode="json")
                record["latency_ms"] = latency_ms
                logger.info("di_capture_ok", image_id=sample.image_id)
            else:
                record["error"] = error
                logger.warning(
                    "di_capture_non_ok", image_id=sample.image_id, outcome=outcome, error=error
                )

            out.write(json.dumps(record) + "\n")
            out.flush()

    logger.info("di_capture_done", output_path=str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    capture(args.output)


if __name__ == "__main__":
    main()
