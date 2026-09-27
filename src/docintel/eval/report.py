"""CLI eval report: per-field precision/recall/F1, coverage, worst-N failure analysis.

Reads a JSONL file produced by eval/capture.py - no GPU touched here, so scoring and
normalization rules can be iterated on freely against the same one captured run
(PROJECT_BRIEF.md §7 Phase 4: "iterate on scoring offline with no GPU").
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docintel.eval.cord import FieldCounts, ReceiptScore, normalize_cord_receipt, score_receipt
from docintel.schemas import ExtractionResponse

DEFAULT_CAPTURE_PATH = Path("benchmarks/results/phase4_cord_capture.jsonl")


@dataclass
class CapturedRecord:
    """One deserialized line of eval/capture.py's output.

    outcome is one of "ok", "infra_rejected" (D-014's ~2048-token encoder-cache
    ceiling - a real per-image capacity limit) or "transient_failure" (a cold
    start/connection blip capture.py's own retries couldn't recover from - not a
    property of the receipt, and must not be counted as a coverage gap).
    """

    image_id: int
    raw_ground_truth: dict[str, Any]
    outcome: str
    prediction: ExtractionResponse | None


def load_captured(path: Path) -> list[CapturedRecord]:
    """Reads capture.py's JSONL, one CapturedRecord per line."""
    records: list[CapturedRecord] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction = (
                ExtractionResponse.model_validate(row["prediction"])
                if row["prediction"] is not None
                else None
            )
            records.append(
                CapturedRecord(
                    image_id=row["image_id"],
                    raw_ground_truth=row["raw_ground_truth"],
                    outcome=row["outcome"],
                    prediction=prediction,
                )
            )
    return records


def score_all(
    records: list[CapturedRecord],
) -> tuple[list[ReceiptScore], int, int, int, int]:
    scores: list[ReceiptScore] = []
    n_infra_rejected = 0
    n_transient_failure = 0
    n_unparseable = 0
    n_gold_incomplete = 0
    for record in records:
        if record.outcome == "infra_rejected":
            n_infra_rejected += 1
            continue
        if record.outcome == "transient_failure":
            n_transient_failure += 1
            continue
        assert record.prediction is not None
        if record.prediction.data is None:
            n_unparseable += 1
            continue
        try:
            gold = normalize_cord_receipt(record.raw_ground_truth)
        except ValueError:
            n_gold_incomplete += 1
            continue
        scores.append(score_receipt(record.image_id, record.prediction.data, gold))
    return scores, n_infra_rejected, n_transient_failure, n_unparseable, n_gold_incomplete


@dataclass
class PRF1:
    precision: float
    recall: float
    f1: float


@dataclass
class AggregateReport:
    overall: PRF1
    per_field: dict[str, PRF1]


def _prf1_from_counts(counts: FieldCounts) -> PRF1:
    precision = counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if (counts.tp + counts.fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF1(precision=precision, recall=recall, f1=f1)


def aggregate_scores(scores: list[ReceiptScore]) -> AggregateReport:
    pooled: dict[str, FieldCounts] = {}
    overall = FieldCounts()
    for receipt_score in scores:
        for field_path, counts in receipt_score.field_counts.items():
            total = pooled.setdefault(field_path, FieldCounts())
            total.tp += counts.tp
            total.fp += counts.fp
            total.fn += counts.fn
            overall.tp += counts.tp
            overall.fp += counts.fp
            overall.fn += counts.fn

    per_field = {path: _prf1_from_counts(c) for path, c in pooled.items()}
    return AggregateReport(overall=_prf1_from_counts(overall), per_field=per_field)


def _receipt_f1(score: ReceiptScore) -> float:
    total = FieldCounts()
    for counts in score.field_counts.values():
        total.tp += counts.tp
        total.fp += counts.fp
        total.fn += counts.fn
    return _prf1_from_counts(total).f1


def _receipt_field_count(score: ReceiptScore) -> int:
    return sum(c.tp + c.fp + c.fn for c in score.field_counts.values())


def select_worst_n(scores: list[ReceiptScore], n: int = 10) -> list[ReceiptScore]:
    return sorted(scores, key=lambda s: (_receipt_f1(s), -_receipt_field_count(s)))[:n]


def print_report(
    report: AggregateReport,
    worst: list[ReceiptScore],
    n_infra_rejected: int,
    n_transient_failure: int,
    n_unparseable: int,
    n_gold_incomplete: int,
    n_total: int,
) -> None:
    n_scored = (
        n_total - n_infra_rejected - n_transient_failure - n_unparseable - n_gold_incomplete
    )
    print(f"CORD test split: {n_total} receipts")
    print(f"  scored:            {n_scored}")
    print(f"  infra-rejected:    {n_infra_rejected}  (coverage gap, D-014 - not folded into F1)")
    print(f"  transient failure: {n_transient_failure}  (retry exhausted, not the model)")
    print(f"  unparseable:       {n_unparseable}  (every repair attempt failed to parse)")
    print(f"  gold incomplete:   {n_gold_incomplete}  (CORD's own ground truth missing a field)")
    print()
    o = report.overall
    print(f"Overall: P={o.precision:.3f} R={o.recall:.3f} F1={o.f1:.3f}")
    print()
    print("Per field:")
    for field_name, prf1 in sorted(report.per_field.items()):
        print(f"  {field_name:30s} P={prf1.precision:.3f} R={prf1.recall:.3f} F1={prf1.f1:.3f}")
    print()
    print(f"Worst {len(worst)} receipts (ranking rule: select_worst_n, this file):")
    for score in worst:
        print(f"  image_id={score.image_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-path", type=Path, default=DEFAULT_CAPTURE_PATH)
    parser.add_argument("--worst-n", type=int, default=10)
    args = parser.parse_args()

    records = load_captured(args.capture_path)
    scores, n_infra_rejected, n_transient_failure, n_unparseable, n_gold_incomplete = score_all(
        records
    )
    report = aggregate_scores(scores)
    worst = select_worst_n(scores, args.worst_n)
    print_report(
        report,
        worst,
        n_infra_rejected,
        n_transient_failure,
        n_unparseable,
        n_gold_incomplete,
        len(records),
    )


if __name__ == "__main__":
    main()
