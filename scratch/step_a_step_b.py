import json
import os
import sys
import time
from pathlib import Path

os.environ["DOCINTEL_BACKEND"] = "http"
os.environ["DOCINTEL_VLLM_BASE_URL"] = "https://avishkarpotale--vllm-doc-intelligence-server.us-east.modal.direct"

sys.path.insert(0, "src")

import httpx
from pydantic import ValidationError

from docintel.extraction.client import get_client
from docintel.extraction.prompts import build_extraction_prompt, build_repair_prompt
from docintel.extraction.validate import check_business_rules
from docintel.schemas import Receipt

MAX_REPAIR_ATTEMPTS = 2
BASE_URL = os.environ["DOCINTEL_VLLM_BASE_URL"]
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

client = get_client()
print("client:", type(client).__name__, flush=True)

receipts_dir = Path("scratch/receipts")
files = sorted(receipts_dir.glob("*.jpg"))

# ---------- Step A ----------
# Infra-rejected (encoder cache ceiling, context-length overflow) is a distinct outcome
# from a model quality failure - the pipeline never got a chance to be right or wrong.
# Caught here, not inside pipeline.py: this is a test-harness reporting concern, not
# something the production repair loop should special-case.
results = []
for f in files:
    image_bytes = f.read_bytes()
    attempts_log: list[dict] = []
    raw_output = ""
    errors: list[str] = []
    attempts = 0
    start = time.monotonic()
    infra_rejected_reason: str | None = None
    while True:
        turn_prompt = (
            build_extraction_prompt()
            if attempts == 0
            else build_repair_prompt(Receipt, raw_output, errors)
        )
        try:
            raw_output, usage = client.complete(image_bytes, turn_prompt, schema=Receipt)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                infra_rejected_reason = exc.response.json().get("error", {}).get("message", "")
                break
            raise
        failure_kind = None
        try:
            receipt = Receipt.model_validate_json(raw_output)
        except ValidationError as exc:
            errors = [e["msg"] for e in exc.errors()]
            failure_kind = "schema_violation"
        else:
            errors = check_business_rules(receipt)
            if errors:
                failure_kind = "business_rule"
        attempts_log.append({"attempt": attempts, "errors": list(errors), "failure_kind": failure_kind})
        if not errors or attempts >= MAX_REPAIR_ATTEMPTS:
            break
        attempts += 1
    latency = time.monotonic() - start
    result = {
        "file": f.name,
        "attempts_log": attempts_log,
        "final_passed": (not errors) if infra_rejected_reason is None else None,
        "repair_attempts": attempts,
        "latency_s": round(latency, 1),
        "infra_rejected_reason": infra_rejected_reason,
    }
    results.append(result)
    if infra_rejected_reason:
        print(f"done: {f.name} INFRA_REJECTED: {infra_rejected_reason[:80]}", flush=True)
    else:
        print(f"done: {f.name} passed={not errors} attempts={attempts} latency={latency:.1f}s", flush=True)

Path("scratch/step_a_results.json").write_text(json.dumps(results, indent=2, default=str))
print("Step A complete, written to scratch/step_a_results.json", flush=True)

# ---------- Step B ----------
sample = files[0]
image_bytes = sample.read_bytes()
import base64

image_b64 = base64.b64encode(image_bytes).decode("ascii")

# Phase 1 free-form path: same prompt text as modal_app.py's local_entrypoint, no response_format.
free_form_payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "extract all the visible text and structured fields from this receipts as JSON.",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }
    ],
}
resp = httpx.post(f"{BASE_URL}/v1/chat/completions", json=free_form_payload, timeout=600)
resp.raise_for_status()
free_form_usage = resp.json()["usage"]

# Structured path: our real prompts.py + schema.
prompt = build_extraction_prompt()
_, structured_usage = client.complete(image_bytes, prompt, schema=Receipt)

step_b = {
    "sample_file": sample.name,
    "free_form_completion_tokens": free_form_usage["completion_tokens"],
    "structured_completion_tokens": structured_usage.completion_tokens,
}
Path("scratch/step_b_results.json").write_text(json.dumps(step_b, indent=2))
print("Step B:", step_b, flush=True)
