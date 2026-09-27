"""Phase 6 probe: per-receipt visual tokens + HTTP status for all 100 CORD test receipts.

Answers two questions with one cheap pass (max_tokens=1, no real generation): (1) does the
D-014 encoder-cache ceiling still reject receipts when --mm-processor-kwargs is `{}`, and
(2) what is the visual-token distribution at this server config (needed to pick the reduced
level). Sends the production payload shape so the result reflects what the pipeline sees.
Throwaway, like the other scratch/ scripts.
"""

import argparse
import base64
import csv
import statistics
from pathlib import Path

import httpx

from docintel.config import settings
from docintel.eval.capture import _wait_until_healthy
from docintel.eval.cord import load_cord_test_split
from docintel.extraction.prompts import build_extraction_prompt
from docintel.schemas import Receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--label", required=True, help="e.g. 4b_default; names the output CSV")
    args = ap.parse_args()

    _wait_until_healthy(args.base_url)
    prompt = build_extraction_prompt()
    schema = Receipt.model_json_schema()

    with httpx.Client(base_url=args.base_url, timeout=300) as client:

        def call(content: list[dict[str, object]]) -> httpx.Response:
            return client.post(
                "/v1/chat/completions",
                json={
                    "model": settings.model_name,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 1,  # accounting only; we don't need a generation
                    "temperature": 0,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "Receipt", "schema": schema},
                    },
                },
            )

        # text-only run of the same prompt = the non-image share of prompt_tokens
        text_tokens = int(call([{"type": "text", "text": prompt}]).json()["usage"]["prompt_tokens"])

        rows: list[dict[str, object]] = []
        for sample in load_cord_test_split():
            b64 = base64.b64encode(sample.image_bytes).decode("ascii")
            resp = call(
                [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]
            )
            if resp.status_code == 200:
                visual = int(resp.json()["usage"]["prompt_tokens"]) - text_tokens
                rows.append({"image_id": sample.image_id, "status": 200, "visual_tokens": visual, "error": ""})
            else:
                rows.append({"image_id": sample.image_id, "status": resp.status_code,
                             "visual_tokens": "", "error": resp.text[:160].replace("\n", " ")})

    out = Path(f"scratch/probe_coverage_{args.label}.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "status", "visual_tokens", "error"])
        writer.writeheader()
        writer.writerows(rows)

    ok = [int(r["visual_tokens"]) for r in rows if r["status"] == 200]
    print(f"{args.label}: {len(ok)}/{len(rows)} accepted, text-only baseline {text_tokens} tokens")
    if ok:
        q = statistics.quantiles(ok, n=100, method="inclusive")
        print(f"visual tokens: mean {statistics.mean(ok):.0f}  p50 {q[49]:.0f}  p95 {q[94]:.0f}  max {max(ok)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
