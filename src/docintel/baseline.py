"""Azure AI Document Intelligence client - the managed alternative to the extraction
pipeline, not part of it (PROJECT_BRIEF.md §4).

Phase 8 (§7/§8, D-007): the managed-baseline side of the SROIE comparison bench. Single
-shot, no retry/repair loop - DI has no equivalent to pipeline.py's bounded repair, so
giving the self-hosted side one here (rather than in eval/sroie_capture.py's matching
single-shot path) would make the latency/cost comparison uneven by construction.
"""

import base64
import time

import httpx
import structlog

from docintel.config import settings
from docintel.schemas import SroieReceipt

logger = structlog.get_logger(__name__)

# v4.0 GA, per Microsoft's own REST reference (learn.microsoft.com/.../document-models/
# analyze-document?view=rest-aiservices-v4.0+(2024-11-30)), checked 2026-09-28 rather
# than recalled from memory, per CLAUDE.md rule 5.
API_VERSION = "2024-11-30"
MODEL_ID = "prebuilt-receipt"

POLL_INTERVAL_S = 1.0
POLL_TIMEOUT_S = 60.0


def _address_string(field: dict[str, object] | None) -> str | None:
    """DI's MerchantAddress field. Confirmed live (2026-09-28, real docintel-di resource,
    SROIE image_id=0) as a structured `valueAddress` object, NOT the plain `valueString`
    v2.1's docs showed: {'house': 'NO 2 & 4', 'road'/'streetAddress': 'JALAN BAYU 4,...',
    'city': 'MASAI', 'state': 'JOHOR', 'postalCode': '81750'} - no `countryRegion` key
    inside it at all (that's a separate top-level `CountryRegion` document field, not
    part of the address). The original guess at this shape (checked in, then run for
    real) was wrong on exactly this point - fixed from the live response, not the docs.
    """
    if field is None:
        return None
    if "valueString" in field:
        return str(field["valueString"])
    value_address = field.get("valueAddress")
    if isinstance(value_address, dict):
        parts = [
            value_address.get(k) for k in ("house", "streetAddress", "city", "state", "postalCode")
        ]
        joined = ", ".join(str(p) for p in parts if p)
        return joined or None
    return None


def analyze_receipt(image_bytes: bytes) -> tuple[SroieReceipt, float]:
    """Submits one receipt image to the prebuilt-receipt model, polls until done,
    returns the 4 fields this project scores plus latency_ms - same (data, timing)
    shape as extraction/client.py's HTTPClient.complete, for the same reason: whoever
    reads the result shouldn't have to know two different conventions.
    """
    assert settings.di_endpoint and settings.di_api_key, (
        "DOCINTEL_DI_ENDPOINT / DOCINTEL_DI_API_KEY must be set to call the real DI baseline"
    )
    start = time.monotonic()
    headers = {
        "Ocp-Apim-Subscription-Key": settings.di_api_key,
        "Content-Type": "application/json",
    }
    # NOT independently verified end-to-end against a real credentialed call yet - built
    # from the REST reference's documented shape, not from memory (rule 5), but the
    # first live run against a real DI resource is what actually confirms this path and
    # the field-shape assumptions above. A wrong path here 404s loudly, not silently.
    analyze_url = f"{settings.di_endpoint}/documentintelligence/documentModels/{MODEL_ID}:analyze"
    body = {"base64Source": base64.b64encode(image_bytes).decode("ascii")}

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            analyze_url, params={"api-version": API_VERSION}, headers=headers, json=body
        )
        response.raise_for_status()
        operation_location = response.headers["Operation-Location"]

        deadline = time.monotonic() + POLL_TIMEOUT_S
        result_body: dict[str, object] = {}
        while True:
            poll = client.get(operation_location, headers=headers)
            poll.raise_for_status()
            result_body = poll.json()
            status = result_body["status"]
            if status == "succeeded":
                break
            if status == "failed":
                raise RuntimeError(f"DI analysis failed: {result_body}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"DI analysis did not finish within {POLL_TIMEOUT_S}s")
            time.sleep(POLL_INTERVAL_S)

    analyze_result = result_body["analyzeResult"]
    documents = analyze_result["documents"]  # type: ignore[index]
    fields = documents[0]["fields"] if documents else {}

    merchant_name = fields.get("MerchantName") or {}
    transaction_date = fields.get("TransactionDate") or {}
    total = fields.get("Total") or {}

    # Confirmed live (2026-09-28, docintel-di, SROIE image_id=0): the real v4.0 response
    # types Total as `currency`, not `number` - {'type': 'currency', 'valueCurrency':
    # {'amount': 193.0, 'currencyCode': 'MYR'}}, no `valueNumber` key at all. The
    # original `total.get("valueNumber")` silently found nothing and raised on a
    # receipt DI had actually extracted correctly - not a DI miss, a wrong field path.
    # `valueNumber` kept as a fallback in case a different model/version ever returns
    # a plain-number Total instead of a currency-typed one.
    value_currency = total.get("valueCurrency")
    if isinstance(value_currency, dict) and value_currency.get("amount") is not None:
        total_value = value_currency["amount"]
    else:
        total_value = total.get("valueNumber")

    if total_value is None:
        # SroieReceipt.total is required (matches Totals.total's pattern - a receipt
        # always has *some* final amount), so a genuine DI miss here isn't "null", it's
        # a real failure to extract this receipt at all - raised, not coerced, same as
        # cord.py's normalize_cord_receipt raising on a missing total_price. The caller
        # (eval/sroie_di_capture.py) is where this gets categorized as a coverage gap
        # ("no_total") rather than crashing the whole run.
        raise ValueError(f"DI returned no Total value for this receipt: {fields}")

    receipt = SroieReceipt(
        company=merchant_name.get("valueString"),
        address=_address_string(fields.get("MerchantAddress")),
        date=transaction_date.get("valueDate"),
        total=float(total_value),
    )
    latency_ms = (time.monotonic() - start) * 1000
    logger.debug("di_analyze_receipt", latency_ms=latency_ms)
    return receipt, latency_ms
