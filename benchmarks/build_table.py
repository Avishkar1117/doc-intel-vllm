"""Phase 6: assemble the four-config comparison table from the committed result files.

Reads each config's summary JSON, capture JSONL and sweep levels CSV, so the table in the
README can always be regenerated from evidence instead of copied by hand.
"""

import csv
import json
from pathlib import Path
from typing import Any

from docintel.eval.report import aggregate_scores, load_captured, score_all

RESULTS = Path("benchmarks/results")
TAGS = ["4b_default", "4b_cap1500k", "2b_default", "2b_cap1500k"]
L4_USD_PER_HOUR = 0.000222 * 3600  # PROJECT_BRIEF.md §3, per-second rate x 3600
COMPARE_LEVEL = 8  # the last level before queueing (--max-num-seqs 8), a fair same-load comparison


def latest_levels(tag: str) -> list[dict[str, str]]:
    """Newest sweep for this tag; a re-run writes a new timestamped file, never overwrites."""
    path = sorted(RESULTS.glob(f"phase6_{tag}_2026*_levels.csv"))[-1]
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    scored = {}
    for tag in TAGS:
        records = load_captured(RESULTS / f"phase6_{tag}_capture.jsonl")
        scored[tag] = score_all(records)[0]
    # F1 is only comparable across configs on receipts every config scored (a 2B run
    # lost one receipt to a timeout, another to an unparseable output).
    common = set.intersection(*[{s.image_id for s in scores} for scores in scored.values()])

    rows: list[dict[str, Any]] = []
    for tag in TAGS:
        summary = json.loads((RESULTS / f"phase6_{tag}_summary.json").read_text())
        levels = latest_levels(tag)
        peak = max(levels, key=lambda r: float(r["docs_per_min"]))
        at_compare = next(r for r in levels if int(r["concurrency"]) == COMPARE_LEVEL)
        peak_dpm = float(peak["docs_per_min"])
        f1 = aggregate_scores([s for s in scored[tag] if s.image_id in common]).overall.f1
        rows.append(
            {
                "config": tag,
                "vllm_version": summary["vllm_version"],
                "mean_visual_tokens": round(summary["mean_visual_tokens"]),
                "cord_f1": round(f1, 3),
                "n_receipts_scored": len(common),
                "peak_docs_per_min": round(peak_dpm, 1),
                "peak_concurrency": int(peak["concurrency"]),
                "p95_s_at_peak": round(float(peak["p95_s"]), 1),
                f"docs_per_min_c{COMPARE_LEVEL}": round(float(at_compare["docs_per_min"]), 1),
                f"p95_s_c{COMPARE_LEVEL}": round(float(at_compare["p95_s"]), 1),
                # $/hr divided by docs/hr, x1000: assumes a fully busy GPU at peak, no idle time
                "usd_per_1k_docs": round(L4_USD_PER_HOUR / (peak_dpm * 60) * 1000, 2),
            }
        )

    csv_path = RESULTS / "phase6_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(r[h]) for h in header) + " |" for r in rows]
    md_path = RESULTS / "phase6_table.md"
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
