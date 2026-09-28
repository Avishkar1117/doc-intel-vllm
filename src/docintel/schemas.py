from pydantic import BaseModel, Field


class LineItem(BaseModel):
    name : str
    count : float
    unit_price : float | None = Field(
        description=(
            "Price per single unit, only if a separate per-unit price is printed "
            "alongside the line total. Leave null if only one money amount is printed "
            "for this line - do not compute it by dividing the total by count."
        )
    )
    price : float = Field(
        description=(
            "The line's total price, already accounting for count. If only one money "
            "amount is printed for a line whose count is greater than 1, that printed "
            "amount IS this field - do not multiply it by count."
        )
    )
    discount_price : float | None

# The CORD `sub_total` group: adjustments applied between the line items and the final total.
class Subtotals(BaseModel):
    subtotal : float | None
    discount : float |None
    service_charge : float | None
    other_service_charge : float |None
    tax : float | None

# The CORD `total` group: the final charged amount and how it was settled.
class Totals(BaseModel):
    total : float = Field(gt=0)
    cash_paid : float | None
    change_due : float | None
    credit_card_paid : float | None
    emoney_paid : float | None
    menu_type_count : int | None
    menu_quantity_count: int | None

# The full CORD-derived extraction target: line items plus the subtotal/total breakdown.
class Receipt(BaseModel):
    line_items : list[LineItem]
    subtotals : Subtotals
    totals : Totals

# Outcome of pipeline.py's validate -> repair loop, carried alongside the (possibly partial) data.
class ValidationResult(BaseModel):
    passed : bool
    errors : list[str] | None
    repair_attempts : int

# Per-request token accounting - the input to the cost model, not just a debug field.
class Usage(BaseModel):
    prompt_tokens : int
    completion_tokens : int
    visual_tokens : int
    total_tokens : int

# Top-level response envelope: what /extract returns, and what the accuracy harness scores.
class ExtractionResponse(BaseModel):
    data : Receipt | None
    validation : ValidationResult
    usage : Usage
    latency_ms : float = Field(ge=0)

class SroieReceipt(BaseModel):
    company : str | None = Field(
        description="the store/business name as printed the brand name, not a slogan and address"
    )
    address : str |  None = Field(
        description=(
            "the full postal address as printed, merged into one string if it spans "
            "multiple lines"
        )
    )
    date : str | None = Field(
        description="the date of the transaction as printed, in whatever format it was printed"
    )
    total : float = Field( gt=0, description="final amount actually paid as printed")



