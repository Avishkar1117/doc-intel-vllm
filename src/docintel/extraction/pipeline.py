import time

from pydantic import ValidationError

from docintel.extraction.client import VLLMClient
from docintel.extraction.prompts import build_extraction_prompt, build_repair_prompt
from docintel.extraction.validate import check_business_rules
from docintel.schemas import ExtractionResponse, Receipt, Usage, ValidationResult

# Original attempt + up to this many retries, per PROJECT_BRIEF.md §5 step 5.
MAX_REPAIR_ATTEMPTS = 2

# Runs one receipt image through prompt -> decode -> validate -> bounded repair.
def extract(images_bytes: bytes, client: VLLMClient) -> ExtractionResponse:
    start = time.monotonic()
    errors: list[str] = []
    raw_output = ""
    receipt: Receipt | None = None
    usage: Usage | None = None

    attempts = 0
    while True:
        # First pass gets the plain extraction prompt; every retry gets the last
        # attempt's output and errors folded back in, not the original prompt again.
        turn_prompt = (
            build_extraction_prompt()
            if attempts == 0
            else build_repair_prompt(Receipt, raw_output, errors)
        )
        raw_output, usage = client.complete(images_bytes, turn_prompt, schema=Receipt)

        # Type-level failure: no Receipt exists to keep, so there's nothing to pass forward.
        try:
            receipt = Receipt.model_validate_json(raw_output)
        except ValidationError as exc:
            errors = [e["msg"] for e in exc.errors()]
            receipt = None
        else:
            # Parsed fine - receipt stays, even if it goes on to fail a business rule below.
            errors = check_business_rules(receipt)

        if not errors or attempts >= MAX_REPAIR_ATTEMPTS:
            break
        attempts += 1
    assert usage is not None
    latency_ms = (time.monotonic() - start) * 1000

    return ExtractionResponse(
        data=receipt,
        validation=ValidationResult(
            passed=not errors,
            errors=errors or None,
            repair_attempts=attempts,
        ),
        usage=usage,
        latency_ms=latency_ms,
    )