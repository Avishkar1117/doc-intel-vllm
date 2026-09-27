"""FastAPI app + routes: /extract, /health.

Thin orchestration only (PROJECT_BRIEF.md §4): image in, extraction pipeline, response out.
This file never loads a model itself - it calls out through extraction/client.py's seam,
which is what makes app-tier/model-tier separation physically true rather than a stylistic
choice (the whole point of Phase 3's "Learn" line in §7).
"""

from typing import Annotated

import httpx
import structlog
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from docintel.cost import estimate_request_cost_usd
from docintel.extraction.client import get_client
from docintel.extraction.pipeline import extract

# JSON-rendered, own configuration - the app tier's log format. extraction/client.py's
# logger inherits whatever the *caller* configures, so this only takes effect when api.py
# is the entrypoint (not when pipeline.py is imported directly by a test or a script).
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger(__name__)

app = FastAPI(title="docintel", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for `docker run` / Container Apps - no model call, no GPU dependency."""
    return {"status": "ok"}


@app.post("/extract")
async def extract_receipt(file: Annotated[UploadFile, File()]) -> JSONResponse:
    """Runs one receipt image through the pipeline; attaches an estimated cost."""
    image_bytes = await file.read()
    client = get_client()  # env-driven: MockClient locally, HTTPClient against Modal

    try:
        result = extract(image_bytes, client)
    # The one boundary where app-tier/model-tier separation becomes a real failure mode:
    # the Modal endpoint can be cold, unhealthy, or unreachable, and that's a distinct
    # error from "the model produced bad JSON" (which pipeline.py already handles via
    # validation.passed=False, not an exception). Surface it as 502, not a bare 500.
    except httpx.HTTPError as exc:
        logger.error("extract_upstream_error", filename=file.filename, error=str(exc))
        return JSONResponse(
            status_code=502,
            content={"error": "upstream vLLM endpoint unavailable", "detail": str(exc)},
        )

    cost_usd = estimate_request_cost_usd(result.latency_ms)
    logger.info(
        "extract_request",
        filename=file.filename,
        passed=result.validation.passed,
        repair_attempts=result.validation.repair_attempts,
        latency_ms=result.latency_ms,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        estimated_cost_usd=cost_usd,
    )

    payload = result.model_dump()
    payload["estimated_cost_usd"] = cost_usd
    return JSONResponse(content=payload)
