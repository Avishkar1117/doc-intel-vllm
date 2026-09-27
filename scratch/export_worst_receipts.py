"""Dumps the current worst-N receipts (image + gold/prediction JSON side by side) to
disk for the Phase 4 failure-mode writeup. Throwaway utility, not part of the
shippable package - excluded from CI/lint the same way scratch/step_a_step_b.py
already is (PHASE_LOG.md Phase 3, D-016).

Usage: uv run python scratch/export_worst_receipts.py [--n 10]
"""

import argparse
import json
from pathlib import Path

from docintel.eval.cord import load_cord_test_split
from docintel.eval.report import DEFAULT_CAPTURE_PATH, load_captured, score_all, select_worst_n

OUTPUT_DIR = Path("scratch/phase4_worst_receipts")


def main(n: int) -> None:
    records = load_captured(DEFAULT_CAPTURE_PATH)
    scores, *_ = score_all(records)
    worst_ids = {s.image_id for s in select_worst_n(scores, n)}
    by_id = {r.image_id: r for r in records}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for sample in load_cord_test_split():
        if sample.image_id not in worst_ids:
            continue
        (OUTPUT_DIR / f"{sample.image_id}.png").write_bytes(sample.image_bytes)

        record = by_id[sample.image_id]
        summary = {
            "image_id": sample.image_id,
            "raw_ground_truth": record.raw_ground_truth,
            "prediction": (
                record.prediction.model_dump(mode="json") if record.prediction else None
            ),
        }
        (OUTPUT_DIR / f"{sample.image_id}.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote {len(worst_ids)} receipts (image + json) to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    main(args.n)
