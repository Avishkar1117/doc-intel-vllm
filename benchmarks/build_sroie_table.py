"""Phase 8: assemble the SROIE self-hosted-vs-DI comparison table from the committed
capture files. Same "read committed evidence, don't hand-copy numbers" approach as
build_table.py (Phase 6), reusing eval/sroie.py's normalize/score functions and
eval/report.py's aggregate_scores (generic - not CORD-specific) rather than
re-deriving scoring logic here.
"""

import json
import statistics
from pathlib import Path

from pydantic import ValidationError

from docintel.eval.cord import FieldCounts, ReceiptScore
from docintel.eval.report import aggregate_scores
from docintel.eval.sroie import normalize_sroie_receipt, score_sroie_receipt
from docintel.schemas import SroieReceipt

RESULTS = Path("benchmarks/results")

L4_USD_PER_HOUR = 0.000222 * 3600  # PROJECT_BRIEF.md §3, per-second rate x 3600
# Measured wall-clock of the actual self-hosted capture run (2026-09-28, 16:56:09 ->
# 17:19:57, single-shot sequential, 4B/reduced-res) - not an idealized peak-throughput
# figure like Phase 6's table used, since this run genuinely is the workload shape
# (one request at a time), not a concurrency sweep to extrapolate from.
SELF_HOSTED_WALL_S = 1428.0

# Azure's published S0 (pay-as-you-go) rate, not what this run actually paid - the real
# run mixed a free F0 tier (first 229 receipts) with a paid S0 tier (the rest, after
# hitting F0's abuse-prevention cooldown) for reasons specific to running an eval
# script quickly, not reasons a real production deployment would have. The steady
# -state number a company would actually budget against is the plain S0 rate.
DI_S0_USD_PER_1K_PAGES = 10.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def _self_hosted_predicted(record: dict) -> SroieReceipt | None:
    if record["outcome"] != "ok" or record["raw_output"] is None:
        return None
    try:
        return SroieReceipt.model_validate_json(record["raw_output"])
    except ValueError:
        return None  # a genuinely unparseable single-shot output - no repair budget here


def _di_predicted(record: dict) -> SroieReceipt | None:
    if record["outcome"] != "ok" or record["prediction"] is None:
        return None
    return SroieReceipt.model_validate(record["prediction"])


def _is_wrong_negative_total(record: dict) -> bool:
    """DI answered, but with a negative total that SroieReceipt.total (gt=0) rejects.

    Phase 8 filed these under `no_total` alongside DI's genuine no-Total misses (D-038).
    They are wrong answers, not missing ones: gold is positive on all four, and the
    stored `prediction` is None, so only DI's total value is known (from the error text).
    """
    return "validation error for SroieReceipt" in (record.get("error") or "")


def score_system(records: list[dict], get_predicted: object) -> dict[str, object]:
    """Scores every record that produced a usable prediction. The rest are split by cause
    (D-038): no output at all (infra rejection, transient failure, DI's own no-Total
    miss, unparseable single-shot output), gold that fails SroieReceipt validation, and
    DI's wrong negative totals - scored as a `total`-only error, since the record holds
    no other DI field. Coverage gaps are never folded into accuracy (D-014).
    """
    scores = []
    latencies = []
    n_no_output = 0
    n_total_only = 0
    n_gold_invalid = 0
    for r in records:
        try:
            gold = normalize_sroie_receipt(r["raw_entities"])
        except ValidationError:
            # SROIE's own ground truth isn't perfect (same lesson as D-017's CORD gold
            # gaps): at least one row has a negative `total`, which fails
            # SroieReceipt.total's gt=0 constraint. A gold-data problem, not a
            # prediction problem - excluded from scoring on either system, not
            # crashed, not silently coerced.
            n_gold_invalid += 1
            continue
        predicted = get_predicted(r)  # type: ignore[operator]
        if predicted is None:
            if _is_wrong_negative_total(r):
                # One FP (the wrong value) and one FN (the right value missed), the same
                # bookkeeping score_field uses for a value mismatch (D-018).
                scores.append(
                    ReceiptScore(r["image_id"], {"total": FieldCounts(fp=1, fn=1)})
                )
                n_total_only += 1
            else:
                n_no_output += 1
            continue
        scores.append(score_sroie_receipt(r["image_id"], predicted, gold))
        if r.get("latency_ms") is not None:
            latencies.append(r["latency_ms"])
    return {
        "scores": scores,
        "latencies": latencies,
        "n_total": len(records),
        "n_total_only": n_total_only,
        "n_no_output": n_no_output,
        "n_gold_invalid": n_gold_invalid,
    }


def main() -> None:
    self_hosted_records = load_jsonl(RESULTS / "phase8_sroie_capture.jsonl")
    di_records = load_jsonl(RESULTS / "phase8_sroie_di_capture.jsonl")

    systems = {
        "self-hosted (Qwen3-VL-4B, reduced-res)": (
            score_system(self_hosted_records, _self_hosted_predicted),
            SELF_HOSTED_WALL_S / len(self_hosted_records) / 3600 * L4_USD_PER_HOUR * 1000,
        ),
        "Azure DI (prebuilt-receipt, S0 rate)": (
            score_system(di_records, _di_predicted),
            DI_S0_USD_PER_1K_PAGES,
        ),
    }

    rows = []
    for name, (result, usd_per_1k) in systems.items():
        report = aggregate_scores(result["scores"])  # type: ignore[arg-type]
        latencies = result["latencies"]
        row = {
            "system": name,
            "n_total": result["n_total"],
            "n_scored_all_fields": result["n_total"]
            - result["n_total_only"]
            - result["n_no_output"]
            - result["n_gold_invalid"],
            "n_total_field_only": result["n_total_only"],
            "n_no_output": result["n_no_output"],
            "n_gold_invalid": result["n_gold_invalid"],
            "overall_precision": round(report.overall.precision, 3),
            "overall_recall": round(report.overall.recall, 3),
            "overall_f1": round(report.overall.f1, 3),
            "latency_p50_ms": round(_percentile(latencies, 50)),  # type: ignore[arg-type]
            "latency_p95_ms": round(_percentile(latencies, 95)),  # type: ignore[arg-type]
            "usd_per_1k_docs": round(usd_per_1k, 2),
        }
        for field, prf1 in sorted(report.per_field.items()):
            row[f"{field}_f1"] = round(prf1.f1, 3)
        rows.append(row)

    # _v2 (D-038): the original phase8_sroie_table.* stays on record as the superseded
    # version, same convention as D-014/D-022/D-037.
    csv_path = RESULTS / "phase8_sroie_table_v2.csv"
    header = list(rows[0])
    with csv_path.open("w", newline="") as f:
        import csv as csv_module

        writer = csv_module.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(r[h]) for h in header) + " |" for r in rows]
    md_path = RESULTS / "phase8_sroie_table_v2.md"
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
