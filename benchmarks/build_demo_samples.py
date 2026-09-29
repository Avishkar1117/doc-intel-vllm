"""Phase 9: build the 10-sample bundle the demo page shows (images + both systems' saved
answers). Reads the committed Phase 8 captures and the local SROIE parquet, so it needs no
GPU and no DI call. Output goes to a gitignored folder: the images are not ours to
redistribute in the repo, so they are uploaded to private Blob storage instead.
"""

import csv
import io
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import ValidationError

from docintel.eval.cord import ReceiptScore
from docintel.eval.report import aggregate_scores
from docintel.eval.sroie import load_sroie_test_split, normalize_sroie_receipt, score_sroie_receipt
from docintel.schemas import SroieReceipt

RESULTS = Path("benchmarks/results")
OUT_DIR = Path("demo_samples")
SEED = 42
KEY_FIELDS = {"company", "date", "total"}
LABEL = "Hand-picked to show different outcomes, not a representative sample."


def _load(name: str) -> dict[int, dict[str, Any]]:
    with (RESULTS / name).open(encoding="utf-8") as f:
        return {r["image_id"]: r for r in map(json.loads, f)}


def _wrong_fields(score: ReceiptScore) -> list[str]:
    return sorted(f for f, c in score.field_counts.items() if c.fp or c.fn)


def main() -> None:
    ours_rec = _load("phase8_sroie_capture.jsonl")
    di_rec = _load("phase8_sroie_di_capture.jsonl")

    # Per receipt: gold, our prediction, DI's prediction (None when DI gave no usable
    # answer), and which fields each system got wrong.
    info: dict[int, dict[str, Any]] = {}
    for image_id in sorted(ours_rec):
        try:
            gold = normalize_sroie_receipt(ours_rec[image_id]["raw_entities"])
        except ValidationError:
            continue  # negative gold total: excluded from scoring on both systems
        ours = SroieReceipt.model_validate_json(ours_rec[image_id]["raw_output"])
        d = di_rec[image_id]
        di_pred = SroieReceipt.model_validate(d["prediction"]) if d["outcome"] == "ok" else None
        if di_pred is not None:
            di_status, di_wrong = "ok", _wrong_fields(score_sroie_receipt(image_id, di_pred, gold))
        elif "validation error" in (d["error"] or ""):
            di_status, di_wrong = "wrong_negative_total", ["total"]
        else:
            di_status, di_wrong = "no_total", ["total"]
        info[image_id] = {
            "gold": gold,
            "ours": ours,
            "ours_wrong": _wrong_fields(score_sroie_receipt(image_id, ours, gold)),
            "di": di_pred,
            "di_status": di_status,
            "di_wrong": di_wrong,
        }

    # Same rule the analysis used: the two "DI wrong" samples need a clearly wrong total
    # (or no total at all) while we got every field right; "ours wrong" needs a key field.
    rng = random.Random(SEED)
    di_no_answer = sorted(
        i for i, v in info.items() if not v["ours_wrong"] and v["di_status"] != "ok"
    )
    di_wrong_total = sorted(
        i
        for i, v in info.items()
        if not v["ours_wrong"] and v["di_status"] == "ok" and "total" in v["di_wrong"]
    )
    a1, a2 = rng.choice(di_no_answer), rng.choice(di_wrong_total)
    ours_wrong = sorted(
        i
        for i, v in info.items()
        if v["di_status"] == "ok" and set(v["ours_wrong"]) & KEY_FIELDS and i not in (a1, a2)
    )
    b = rng.choice(ours_wrong)
    pool = sorted(i for i, v in info.items() if v["di_status"] == "ok" and i not in (a1, a2, b))
    picks = [
        (a1, "DI returned no usable total; the self-hosted model read every field."),
        (a2, "DI read the wrong total; the self-hosted model read every field."),
        (b, "The self-hosted model got a key field wrong."),
        *[(i, "Random pick; both systems answered.") for i in sorted(rng.sample(pool, 7))],
    ]

    # Headline numbers shown beside the samples, on the receipts both systems answered.
    both = [i for i, v in info.items() if v["di"] is not None]
    f1_ours = aggregate_scores(
        [score_sroie_receipt(i, info[i]["ours"], info[i]["gold"]) for i in both]
    ).overall.f1
    f1_di = aggregate_scores(
        [score_sroie_receipt(i, info[i]["di"], info[i]["gold"]) for i in both]
    ).overall.f1

    images = {s.image_id: s.image_bytes for s in load_sroie_test_split()}
    OUT_DIR.mkdir(exist_ok=True)
    samples = []
    for n, (image_id, role) in enumerate(picks, start=1):
        v = info[image_id]
        image_format = (Image.open(io.BytesIO(images[image_id])).format or "JPEG").lower()
        filename = f"s{n:02d}.{'jpg' if image_format == 'jpeg' else image_format}"
        (OUT_DIR / filename).write_bytes(images[image_id])
        ours_fields = v["ours"].model_dump()
        di_fields = v["di"].model_dump() if v["di"] is not None else None
        samples.append(
            {
                "id": f"s{n:02d}",
                "image": filename,
                "image_id": image_id,
                "role": role,
                "gold": ours_rec[image_id]["raw_entities"],
                "self_hosted": {
                    **ours_fields,
                    "latency_ms": round(ours_rec[image_id]["latency_ms"]),
                    "fields_wrong": v["ours_wrong"],
                },
                "di": {
                    "status": v["di_status"],
                    **(di_fields or {}),
                    # None when DI's answer was rejected before a latency was recorded
                    "latency_ms": (
                        round(di_rec[image_id]["latency_ms"])
                        if di_rec[image_id]["latency_ms"] is not None
                        else None
                    ),
                    "fields_wrong": v["di_wrong"],
                },
            }
        )

    # The page shows the committed v2 table verbatim, never hand-copied numbers.
    table_columns = [
        "system", "overall_f1", "company_f1", "address_f1", "date_f1", "total_f1",
        "latency_p50_ms", "usd_per_1k_docs",
    ]  # fmt: skip
    with (RESULTS / "phase8_sroie_table_v2.csv").open(newline="", encoding="utf-8") as f:
        table = [{c: row[c] for c in table_columns} for row in csv.DictReader(f)]

    n_total = len(ours_rec)
    di_outcomes = [v["di_status"] for v in info.values()]
    summary = {
        "label": LABEL,
        "seed": SEED,
        "table": table,
        "benchmark": {
            "n_receipts": n_total,
            "self_hosted_no_output": 0,
            "di_no_total": di_outcomes.count("no_total"),
            "di_wrong_negative_total": di_outcomes.count("wrong_negative_total"),
            "gold_invalid": n_total - len(info),
            "f1_both_answered": {
                "n": len(both),
                "self_hosted": round(f1_ours, 3),
                "di": round(f1_di, 3),
            },
            "vllm_version": ours_rec[picks[0][0]]["vllm_version"],
            "prompt_version": ours_rec[picks[0][0]]["prompt_version"],
            "di_api_version": di_rec[picks[0][0]]["di_api_version"],
        },
        "samples": samples,
    }
    (OUT_DIR / "samples.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(samples)} samples to {OUT_DIR}/")
    print(json.dumps(summary["benchmark"], indent=2))
    for s in samples:
        print(s["id"], s["image_id"], s["role"])


if __name__ == "__main__":
    main()
