from docintel.schemas import Receipt


# Domain-level checks a syntactically valid Receipt can still fail.
def check_business_rules(receipt: Receipt) -> list[str]:
    errors: list[str] = []
    errors.extend(_check_line_item_sums(receipt))
    errors.extend(_check_quantities(receipt))
    return errors

# Line items, adjusted by the subtotal group's tax/service/discount, should equal total.
def _check_line_item_sums(receipt: Receipt) -> list[str]:
    line_item_sum = sum((item.price for item in receipt.line_items), start=0.0)
    subtotals = receipt.subtotals
    tolerance = max(0.02, 0.01 * len(receipt.line_items))
    adjusted_total = (
        line_item_sum
        + (subtotals.tax or 0.0)
        + (subtotals.service_charge or 0.0)
        + (subtotals.other_service_charge or 0.0)
        - (subtotals.discount or 0.0)
    )
    if abs(adjusted_total - receipt.totals.total) > tolerance:
        return [
            f"Line item + subtotal adjustments ({adjusted_total}) does not match "
            f"total ({receipt.totals.total}) within tolerance {tolerance}"
        ]
    return []

# Every line item's count must be non-negative - a negative quantity has no receipt meaning.
def _check_quantities(receipt: Receipt) -> list[str]:
    return [
        f"line item {item.name!r} has negative count {item.count}"
        for item in receipt.line_items
        if item.count < 0
    ]
