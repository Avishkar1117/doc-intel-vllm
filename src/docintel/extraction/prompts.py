"""Schema-to-prompt builder; versioned prompt strings.

The JSON schema itself constrains the *shape* of the model's output (vLLM's
structured_outputs, wired up in client.py) - this module only has to explain the
*semantics* behind that shape: which field means what, and how to represent absence.
"""

import json

from pydantic import BaseModel

from docintel.schemas import Receipt

# Bump whenever the instruction text changes meaning, not just wording - eval results
# and repair-loop behavior are only comparable to each other within the same version.
# v2 (Phase 4 failure-mode analysis, PHASE_LOG.md): three rules added, each targeting a
# specific, observed failure pattern in the worst-10 receipts from the v1 capture run - the
# fourth pattern found (unit-price vs. line-total ambiguity) is addressed in schemas.py's
# Field descriptions instead, since it's schema/field semantics, not reading behavior.
# Not yet re-validated against a fresh capture (that's a Phase 6 concern, not Phase 4's).
PROMPT_VERSION = "v2"

_EXTRACTION_INSTRUCTIONS = """\
You are extracting structured data from a photograph of a receipt.

Return a single JSON object matching the schema below. Follow these rules:
- Every monetary amount is a plain number (no currency symbols, no thousands separators).
- Indonesian and other Southeast Asian receipts commonly use a period (.) as a THOUSANDS
  separator, not a decimal point - "60.000" means sixty thousand, not sixty. These amounts
  are almost always whole numbers with no cents. If a printed number has a period followed
  by exactly three digits, treat it as a thousands separator unless context clearly says
  otherwise.
- Extract the item's own printed name, not a category or section heading printed above it
  (e.g. a bolded word naming a food category).
- Each physically printed row should appear exactly once in `line_items` - never repeat a row.
- If a field is not present on the receipt, and the schema allows it, use null rather than
  guessing or inventing a value - an absent field is a valid, common outcome, not an error.
- `line_items` must contain one entry per distinct row printed in the receipt's item table,
  in the order they appear.
- Do not include line items that are struck through, voided, or marked cancelled.
"""


def build_extraction_prompt(schema: type[BaseModel] = Receipt) -> str:
    """Builds the user-turn instruction text: task rules plus the target JSON schema."""
    # indent=2 costs a handful of extra tokens over compact json.dumps, but a schema an
    # interviewer can skim in the README's prompt appendix is worth more than the tokens
    schema_text = json.dumps(schema.model_json_schema(), indent=2)
    return f"{_EXTRACTION_INSTRUCTIONS}\nJSON schema:\n{schema_text}\n"


def build_repair_prompt(schema: type[BaseModel], prior_output: str, errors: list[str]) -> str:
    """Builds a re-prompt for the bounded repair loop: prior output plus what failed.

    What exactly gets carried over between retries (all errors vs. a subset, whether the
    full schema repeats) is pipeline.py's call - this just formats whatever it decides to pass in.
    """
    errors_block = "\n".join(f"- {e}" for e in errors)
    return (
        "Your previous response did not satisfy the required schema.\n\n"
        f"Previous response:\n{prior_output}\n\n"
        f"Validation errors:\n{errors_block}\n\n"
        "Correct these specific problems and return a full, corrected JSON object "
        "matching the same schema."
    )
