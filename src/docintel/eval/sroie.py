import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from docintel.eval.cord import FieldCounts, ReceiptScore, score_field
from docintel.schemas import SroieReceipt

TEST_SPLIT_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "sroie"
    / "data"
    / "test-00000-of-00001.parquet"
)


@dataclass(frozen=True)
class SroieSample:
    """One SROIE test-split receipt, still in its raw on-disk shape.

    Only `entities` is kept - `bbox`/`text` are SROIE's own OCR-localization output,
    which this project's scoring (VLM output vs. gold entities) doesn't use, the same
    way CordSample drops `meta`/`valid_line`.
    """

    image_id: int
    image_bytes: bytes
    raw_entities: dict[str, Any]


def load_sroie_test_split(parquet_path: Path = TEST_SPLIT_PATH) -> list[SroieSample]:
    """Reads the SROIE test parquet, returns one SroieSample per row, in file order.

    Confirmed against the real file (2026-09-28): 347 rows, matching PROJECT_BRIEF.md
    §6's stated test-split size exactly - `objects.entities` present with all four
    keys (company/date/address/total) on every row sampled, all plain strings.
    """
    df = pd.read_parquet(parquet_path, columns=["image", "objects"])
    samples: list[SroieSample] = []
    for image_id, (image, objects) in enumerate(zip(df["image"], df["objects"], strict=True)):
        samples.append(
            SroieSample(
                image_id=image_id,
                image_bytes=image["bytes"],
                raw_entities=objects["entities"],
            )
        )
    return samples

_TOTAL_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def parse_sroie_total(raw: str) -> float:
    """Raw SROIE `total` string -> float.

    Verified against all 347 real gold `total` values (2026-09-28), not the 5-row
    spot-check this originally shipped with - that check missed a real, common case:
    currency prefixes. Confirmed shapes: plain decimals ("193.00", "7.30", "205.5" -
    one row has a single decimal digit), and 17+30 rows carrying a "$" or "RM"/"RM "
    prefix ("$8.20", "RM11.90", "RM 10.60"). No thousands-separator ambiguity anywhere
    in this field (unlike CORD's parse_cord_number, which needs one) - every value is
    plain-decimal once the currency prefix is stripped, so the fix is just extracting
    the numeric token, not guessing a separator convention. Zero parse failures across
    the full split with this regex.
    """
    match = _TOTAL_NUMBER_RE.search(raw)
    if match is None:
        raise ValueError(f"SROIE total has no parseable number: {raw!r}")
    return float(match.group())


def normalize_sroie_receipt(raw_entities: dict[str, Any]) -> SroieReceipt:
    """Raw SROIE `entities` dict -> SroieReceipt gold target."""
    return SroieReceipt(
        company=raw_entities.get("company") or None,
        address=raw_entities.get("address") or None,
        date=raw_entities.get("date") or None,
        total=parse_sroie_total(raw_entities["total"]),
    )


def score_sroie_receipt(image_id: int, predicted: SroieReceipt, gold: SroieReceipt) -> ReceiptScore:
    field_counts: dict[str, FieldCounts] = {}

    score_field(field_counts, "company", predicted.company, gold.company, numeric=False)

    # Word-overlap match, not score_field's character-level SequenceMatcher default -
    # confirmed necessary, not just theorized: checking all 164 DI address "mismatches"
    # under the character-level metric found 86 (52%) had >=0.7 word overlap with
    # gold, meaning DI's structured-address rejoin (fixed field order: house,
    # streetAddress, city, state, postalCode) was scoring a correct-content answer as
    # wrong purely because it reordered/dropped words relative to gold's printed
    # order. The other 78 (48%) were genuinely low-overlap - real misses, not a metric
    # artifact - so this isn't "the old metric was simply wrong," both effects are real
    # and roughly equal in size. Hand-rolled TP/FP/FN, same reason `date` is below:
    # score_field's `numeric` flag can't express a custom equality function.
    address_counts = field_counts.setdefault("address", FieldCounts())
    if predicted.address is None and gold.address is None:
        address_counts.tp += 1
    elif predicted.address is None:
        address_counts.fn += 1
    elif gold.address is None:
        address_counts.fp += 1
    elif addresses_match(predicted.address, gold.address):
        address_counts.tp += 1
    else:
        address_counts.fp += 1
        address_counts.fn += 1

    # score_field's `numeric` flag only knows exact-match vs. fuzzy-string-ratio - it
    # can't express dates_match's calendar-aware equality, so the same TP/FP/FN
    # bookkeeping (D-018's null-handling convention: both None -> TP) is replicated
    # here by hand rather than routed through score_field.
    date_counts = field_counts.setdefault("date", FieldCounts())
    if predicted.date is None and gold.date is None:
        date_counts.tp += 1
    elif predicted.date is None:
        date_counts.fn += 1
    elif gold.date is None:
        date_counts.fp += 1
    elif dates_match(predicted.date, gold.date):
        date_counts.tp += 1
    else:
        date_counts.fp += 1
        date_counts.fn += 1

    score_field(field_counts, "total", predicted.total, gold.total, numeric=True)

    return ReceiptScore(image_id=image_id, field_counts=field_counts)


_DATE_FORMATS = (
    "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
    "%d %b %Y", "%d %b %y", "%d/%b/%Y", "%d-%b-%Y",
    "%d.%m.%y", "%Y-%m-%d", "%Y/%m/%d",
    "%m/%d/%Y",
)


# The model often appends the print time ("15/01/2019 11:05:16 AM") while gold carries
# the date alone. The calendar date is still right, so scoring compares dates, not whole
# strings (D-038). Applied to gold and predicted alike, since this parser serves both.
_TRAILING_TIME_RE = re.compile(r"[\s,T]+\d{1,2}:\d{2}(:\d{2})?\s*([AaPp][Mm])?\s*$")


def _parse_sroie_date(raw: str) -> date | None:
    """Tries each known SROIE date shape in turn; None if none match."""
    cleaned = _TRAILING_TIME_RE.sub("", raw).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def dates_match(predicted: str | None, gold: str | None) -> bool:
    """Whether a predicted date string and gold's date string are the same calendar
    date, tolerant of SROIE's own format inconsistency (confirmed: "15/01/2019" and
    "20-11-17" both appear as real gold values in the same dataset).
    """
    if predicted is None and gold is None:
        return True
    if predicted is None or gold is None:
        return False

    predicted_date = _parse_sroie_date(predicted)
    gold_date = _parse_sroie_date(gold)
    if predicted_date is None or gold_date is None:
        return False
    return predicted_date == gold_date


_WORD_RE = re.compile(r"[a-z0-9]+")

ADDRESS_MATCH_THRESHOLD = 0.7  # validated against real DI mismatches, not chosen blind


def _address_words(address: str) -> set[str]:
    return set(_WORD_RE.findall(address.lower()))


def addresses_match(predicted: str, gold: str) -> bool:
    """Jaccard word-overlap ratio between two address strings, tolerant of reordering,
    line-wrap differences and minor punctuation - the failure mode a character-level
    comparison (score_field's default) penalizes even when the content is correct.
    """
    predicted_words, gold_words = _address_words(predicted), _address_words(gold)
    if not predicted_words and not gold_words:
        return True
    if not predicted_words or not gold_words:
        return False
    overlap = len(predicted_words & gold_words) / len(predicted_words | gold_words)
    return overlap >= ADDRESS_MATCH_THRESHOLD
