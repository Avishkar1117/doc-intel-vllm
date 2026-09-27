"""FastAPI app + routes: /extract, /health.

Thin orchestration only (PROJECT_BRIEF.md §4): image in, extraction pipeline, response out.
This file never loads a model itself - it calls out through extraction/client.py's seam,
which is what makes app-tier/model-tier separation physically true rather than a stylistic
choice (the whole point of Phase 3's "Learn" line in §7).

Phase 7 adds the public-endpoint protections §5/§7/D-003 require: an API key check and an
in-process rate limit on /extract, best-effort upload of the received image to Blob storage
(24h lifecycle policy already set on the container, PROJECT_BRIEF.md §5 point 8), and
OpenTelemetry export to Application Insights.
"""

import os
import secrets
import time
import uuid
from collections import defaultdict, deque
from typing import Annotated

import httpx
import structlog
from azure.identity import DefaultAzureCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.storage.blob import BlobServiceClient, ContainerClient
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from docintel.config import settings
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

# Auto-instruments FastAPI/httpx and exports to App Insights; a no-op locally and in CI,
# where this env var is never set, so mock/test runs never attempt a real network export.
if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()

app = FastAPI(title="docintel", version="0.1.0")

# `from fastapi import FastAPI` above binds the class before configure_azure_monitor()
# runs, which silently breaks that auto-instrumentor's FastAPI hook specifically (simpler
# libraries like httpx still auto-patch fine) - confirmed empirically: dependency spans for
# the Modal/Blob calls showed up in App Insights, but no request span ever did, until this
# explicit call was added. Instrumenting the app instance directly sidesteps the import-
# order dependency entirely, rather than relying on `import fastapi` never being written as
# `from fastapi import FastAPI` anywhere in this file.
if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    FastAPIInstrumentor.instrument_app(app)

# In-process sliding-window limiter, keyed by client IP. Deliberately not a separate
# library: one endpoint, one replica, a dict of deques is the entire mechanism, and adding
# a dependency here would be exactly the premature abstraction CLAUDE.md warns against.
_RATE_WINDOW_SECONDS = 60
_request_log: defaultdict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    # Azure Container Apps' ingress is a reverse proxy - request.client.host can be the
    # proxy's own address rather than the real caller, which would put every caller in one
    # shared bucket. X-Forwarded-For's first entry is the original client if present.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request) -> None:
    client_key = _client_ip(request)
    now = time.monotonic()
    window = _request_log[client_key]
    while window and now - window[0] > _RATE_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="rate limit exceeded, try again shortly")
    window.append(now)


def _require_api_key(x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None) -> None:
    # settings.extract_api_key unset means the check is a no-op - local/mock/CI runs never
    # set DOCINTEL_EXTRACT_API_KEY, so this never blocks a test. compare_digest avoids
    # leaking the key one character at a time through response-timing differences.
    if settings.extract_api_key and not secrets.compare_digest(
        x_api_key or "", settings.extract_api_key
    ):
        raise HTTPException(status_code=401, detail="missing or invalid API key")


_blob_container_client: ContainerClient | None = None


def _get_blob_container_client() -> ContainerClient | None:
    """Lazily builds the Blob client via Managed Identity - no storage key anywhere."""
    global _blob_container_client
    if not settings.storage_account_name:
        return None
    if _blob_container_client is None:
        account_url = f"https://{settings.storage_account_name}.blob.core.windows.net"
        service_client = BlobServiceClient(account_url, credential=DefaultAzureCredential())
        _blob_container_client = service_client.get_container_client(settings.blob_container)
    return _blob_container_client


def _upload_receipt_image(image_bytes: bytes) -> str | None:
    """Best-effort upload - a Blob hiccup must never fail the extraction itself."""
    container = _get_blob_container_client()
    if container is None:
        return None
    blob_name = str(uuid.uuid4())
    try:
        container.upload_blob(blob_name, image_bytes, overwrite=True)
    except Exception as exc:
        logger.warning("blob_upload_failed", error=str(exc))
        return None
    return blob_name


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for `docker run` / Container Apps - no model call, no GPU dependency."""
    return {"status": "ok"}


@app.post("/extract", dependencies=[Depends(_enforce_rate_limit), Depends(_require_api_key)])
async def extract_receipt(file: Annotated[UploadFile, File()]) -> JSONResponse:
    """Runs one receipt image through the pipeline; attaches an estimated cost."""
    image_bytes = await file.read()
    blob_name = _upload_receipt_image(image_bytes)
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
        blob_name=blob_name,
    )

    payload = result.model_dump()
    payload["estimated_cost_usd"] = cost_usd
    payload["retention"] = {
        "blob_name": blob_name,
        "retention_hours": 24,
        "note": "uploaded images and outputs are deleted after 24 hours",
    }
    return JSONResponse(content=payload)
