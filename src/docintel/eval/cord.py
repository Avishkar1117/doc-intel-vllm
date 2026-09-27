import difflib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from docintel.schemas import LineItem, Receipt, Subtotals, Totals

TEST_SPLIT_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "cord"
    / "data"
    / "test-00000-of-00001-9c204eb3f4e11791.parquet"
)

CURRENCY_RE = re.compile(r"-?\d[\d,.]*\d|-?\d")

@dataclass(frozen=True)
class CordSample:
    """One CORD test-split receipt, still in its raw on-disk shape.

    Only `gt_parse` is kept - `meta`/`valid_line` carry bounding boxes and OCR-order
    info this project's field-level scoring doesn't use.
    """

    image_id: int
    image_bytes: bytes
    raw_ground_truth: dict[str, Any]


def load_cord_test_split(parquet_path: Path = TEST_SPLIT_PATH) -> Iterator[CordSample]:
    """Reads the CORD test parquet, yields one CordSample per row, in file order."""
    df = pd.read_parquet(parquet_path, columns=["image", "ground_truth"])
    images = df["image"].tolist()
    ground_truths = df["ground_truth"].tolist()
    for image_id, (image, ground_truth) in enumerate(zip(images, ground_truths, strict=True)):
        gt_parse = json.loads(ground_truth)["gt_parse"]
        yield CordSample(image_id=image_id, image_bytes=image["bytes"], raw_ground_truth=gt_parse)

def _get_number(d: dict[str, Any], key: str) -> float | None:
    if key not in d:
        return None
    value = d[key]
    if isinstance(value, list):
        value = value[0]
    return parse_cord_number(value)


def _get_int(d: dict[str, Any], key: str) -> int | None:
    value = _get_number(d, key)
    return None if value is None else int(value)


def _line_item_from_raw(raw: dict[str, Any]) -> LineItem | None:
    if "itemsubtotal" in raw:
        price = parse_cord_number(raw["itemsubtotal"])
    elif "price" in raw:
        price = parse_cord_number(raw["price"])
    else:
        price = None

    unit_price = parse_cord_number(raw["unitprice"]) if "unitprice" in raw else None

    if price is None and unit_price is not None:
        count_for_price = parse_cord_number(raw.get("cnt", "1")) or 1.0
        price = unit_price * count_for_price

    if price is None:
        # a label/modifier with no price at  note, a size
        # descriptor like "=*LARGE*==") isn't a real separately-priced line item -
        # skip it rather than failing the whole receipt over a decorative sub-entry.
        return None

    count = parse_cord_number(raw.get("cnt", "1"))
    if count is None:
        raise ValueError(f"menu item has an unparseable cnt: {raw}")

    discount_price = (
        parse_cord_discount(raw["discountprice"], price) if "discountprice" in raw else None
    )

    return LineItem(
        name=raw["nm"], count=count, unit_price=unit_price, price=price,
        discount_price=discount_price,
    )

def normalize_cord_receipt(raw_ground_truth: dict[str, Any]) -> Receipt:
    menu_raw = raw_ground_truth.get("menu", [])
    line_items = [
        item
        for raw_item in flatten_menu_items(menu_raw)
        if (item := _line_item_from_raw(raw_item)) is not None
    ]

    sub_total_raw = raw_ground_truth.get("sub_total", {})
    subtotals = Subtotals(
        subtotal=_get_number(sub_total_raw, "subtotal_price"),
        discount=_get_number(sub_total_raw, "discount_price"),
        service_charge=_get_number(sub_total_raw, "service_price"),
        other_service_charge=None,  # no CORD key maps to this field - always absent in gold
        tax=_get_number(sub_total_raw, "tax_price"),
    )

    total_raw = raw_ground_truth.get("total", {})
    if "total_price" not in total_raw:
        raise ValueError("gold receipt has no total.total_price")
    total = parse_cord_number(total_raw["total_price"])
    if total is None:
        raise ValueError(f"gold total_price is unparseable: {total_raw['total_price']!r}")

    totals = Totals(
        total=total,
        cash_paid=_get_number(total_raw, "cashprice"),
        change_due=_get_number(total_raw, "changeprice"),
        credit_card_paid=_get_number(total_raw, "creditcardprice"),
        emoney_paid=_get_number(total_raw, "emoneyprice"),
        menu_type_count=_get_int(total_raw, "menutype_cnt"),
        menu_quantity_count=_get_int(total_raw, "menuqty_cnt"),
    )

    return Receipt(line_items=line_items, subtotals=subtotals, totals=totals)

def parse_cord_number(raw: str) -> float | None:
    match = CURRENCY_RE.search(raw)
    if match is None:
        return None
    token = match.group().replace(" ", "")

    negative = token.startswith("-")
    if negative:
        token = token[1:]

    last_sep = None
    for m in re.finditer(r"[,.]", token):
        last_sep = m
    if last_sep is not None:
        frac_len = len(token) - last_sep.end()
        if frac_len == 2:
            int_part = re.sub(r"[,.]", "", token[: last_sep.start()])
            token = f"{int_part}.{token[last_sep.start() + 1:]}"
        else:
            token = re.sub(r"[,.]", "", token)

    value = float(token)
    return -value if negative else value




def parse_cord_discount(raw: str, item_price: float) -> float | None:
    text = raw.strip()
    if text.endswith("%"):
        pct = parse_cord_number(text[:-1])
        return None if pct is None else item_price * pct / 100
    amount = parse_cord_number(text)
    return None if amount is None else abs(amount)

def flatten_menu_items(menu_raw: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = menu_raw if isinstance(menu_raw, list) else [menu_raw]
    flat: list[dict[str, Any]] = []
    for item in items:
        flat.append(item)
        if "sub" in item:
            sub = item["sub"]
            # sub can be one dict or a list of them
            sub_items = sub if isinstance(sub, list) else [sub]  
            flat.extend(sub_items)
    return flat
    
def normalize_name(name: str) -> str:
    return " ".join(name.casefold().split())


def match_line_items(
    predicted: list[LineItem], gold: list[LineItem]
) -> list[tuple[LineItem | None, LineItem | None]]:
    gold_names = [normalize_name(g.name) for g in gold]
    pred_names = [normalize_name(p.name) for p in predicted]
    matcher = difflib.SequenceMatcher(None, gold_names, pred_names, autojunk=False)

    pairs: list[tuple[LineItem | None, LineItem | None]] = []
    for tag, g_start, g_end, p_start, p_end in matcher.get_opcodes():
        if tag == "equal":
            for g_i, p_i in zip(range(g_start, g_end), range(p_start, p_end), strict=True):
                pairs.append((predicted[p_i], gold[g_i]))
        elif tag == "delete":
            pairs.extend((None, gold[g_i]) for g_i in range(g_start, g_end))
        elif tag == "insert":
            pairs.extend((predicted[p_i], None) for p_i in range(p_start, p_end))
        elif tag == "replace":
            pairs.extend((None, gold[g_i]) for g_i in range(g_start, g_end))
            pairs.extend((predicted[p_i], None) for p_i in range(p_start, p_end))
    return pairs

@dataclass
class FieldCounts:
    """One field's true-positive / false-positive / false-negative tally, one receipt's worth."""

    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class ReceiptScore:
    """One receipt's scoring result, keyed by dotted field path (e.g. 'totals.total',
    'line_items.name')."""

    image_id: int
    field_counts: dict[str, FieldCounts]

NAME_MATCH_THRESHOLD = 0.8

def value_match(predicted: object, gold: object, *, numeric: bool) -> bool:
    if numeric:
        return predicted == gold
    a, b = normalize_name(str(predicted)), normalize_name(str(gold))
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= NAME_MATCH_THRESHOLD

def score_field(
    field_counts: dict[str, FieldCounts],
    field_path: str,
    predicted: object | None,
    gold: object | None,
    *,
    numeric: bool,
    skip_unverifiable: bool = False,
) -> None:
    count = field_counts.setdefault(field_path, FieldCounts())
    if predicted is None and gold is None:
        count.tp += 1
    elif predicted is None:
        count.fn += 1
    elif gold is None:
        if not skip_unverifiable:
            count.fp += 1
    elif value_match(predicted, gold, numeric=numeric):
        count.tp += 1
    else:
        count.fp += 1
        count.fn += 1

def score_receipt(image_id: int, predicted: Receipt, gold: Receipt) -> ReceiptScore:
    field_counts: dict[str, FieldCounts] = {}
    for f in ("subtotal", "discount", "service_charge", "other_service_charge", "tax"):
        score_field(
            field_counts, f"subtotals.{f}",
            getattr(predicted.subtotals, f), getattr(gold.subtotals, f), numeric=True,
        )
    
    for f in (
         "total", "cash_paid", "change_due", "credit_card_paid",
        "emoney_paid", "menu_type_count", "menu_quantity_count",
    ):
        score_field(
            field_counts, f"totals.{f}",
            getattr(predicted.totals, f), getattr(gold.totals, f), numeric=True,
        )

    for p_item, g_item in match_line_items(predicted.line_items, gold.line_items):
        score_field(
            field_counts, "line_items.name",
            p_item.name if p_item else None, g_item.name if g_item else None, numeric=False,
        )
        for f in ("count", "unit_price", "price", "discount_price"):
            score_field(
                field_counts, f"line_items.{f}",
                getattr(p_item, f) if p_item else None,
                getattr(g_item, f) if g_item else None,
                numeric=True,
                skip_unverifiable=(f == "unit_price"),
            )

    return ReceiptScore(image_id=image_id, field_counts=field_counts)