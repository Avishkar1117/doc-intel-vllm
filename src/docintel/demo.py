"""Public demo page: upload a receipt to the self-hosted model, or browse ten cached samples
side by side with Azure Document Intelligence (Phase 9).

This is the one public, keyless surface, so cost protection is layered: per-IP limits, a
global daily cap, a monthly spend ledger that fails closed (ledger.py), and no route that
wakes the GPU except an explicit, ledger-charged warm-up. /extract stays key-protected.
"""

import contextlib
import io
import json
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Protocol

import httpx
import structlog
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from opentelemetry import trace
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from docintel.config import settings
from docintel.cost import MODAL_L4_USD_PER_SEC, estimate_request_cost_usd
from docintel.extraction.client import HTTPClient, VLLMClient
from docintel.extraction.prompts import build_sroie_extraction_prompt
from docintel.ledger import BlobStore, Ledger, LedgerConfig, LedgerUnavailable
from docintel.netutil import client_ip
from docintel.schemas import SroieReceipt

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/demo")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Inline script/style are needed by the single page; the real XSS defence is Jinja2's
# autoescaping of every model-produced value. This blocks everything else by default.
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
        "script-src 'unsafe-inline'; connect-src 'self'; form-action 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


class DemoSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCINTEL_DEMO_")

    # Live extraction is off until the demo Modal app's URL and proxy token are set, so a
    # missing env var can never point the public page at the wrong GPU.
    vllm_base_url: str | None = None
    vllm_api_key: str | None = None
    request_timeout_s: float = 90.0

    budget_usd: float = 5.0
    cost_margin: float = 1.1
    # Must equal scaledown_window in the demo Modal app.
    scaledown_window_s: float = 120.0
    # Worst measured cold start (Phase 5: 6m24s = 384s).
    cold_start_s: float = 384.0
    daily_request_cap: int = 60

    extract_per_ip_per_hour: int = 5
    warm_per_ip_per_hour: int = 3
    page_per_ip_per_minute: int = 120
    max_concurrent: int = 2

    max_upload_bytes: int = 4_000_000
    max_pixels: int = 25_000_000

    samples_container: str = "demo-samples"
    ledger_container: str = "ledger"
    samples_dir: str = "demo_samples"


@lru_cache
def get_demo_settings() -> DemoSettings:
    return DemoSettings()


class SlidingWindowLimiter:
    """Per-key sliding window, in process. One replica only (max-replicas 1); the ledger's
    daily cap is the backstop that survives a scale-to-zero reset of these counters."""

    def __init__(self, limit: int, window_s: float) -> None:
        self._limit = limit
        self._window_s = window_s
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if now - t <= self._window_s]
            if len(hits) >= self._limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True


@dataclass(frozen=True)
class Limiters:
    extract: SlidingWindowLimiter
    warm: SlidingWindowLimiter
    page: SlidingWindowLimiter
    slots: threading.BoundedSemaphore


@lru_cache
def get_limiters() -> Limiters:
    demo = get_demo_settings()
    return Limiters(
        extract=SlidingWindowLimiter(demo.extract_per_ip_per_hour, 3600),
        warm=SlidingWindowLimiter(demo.warm_per_ip_per_hour, 3600),
        page=SlidingWindowLimiter(demo.page_per_ip_per_minute, 60),
        slots=threading.BoundedSemaphore(demo.max_concurrent),
    )


def _blob_container(name: str) -> ContainerClient:
    """Managed Identity, no storage key anywhere (same pattern as api.py)."""
    account_url = f"https://{settings.storage_account_name}.blob.core.windows.net"
    return BlobServiceClient(account_url, credential=DefaultAzureCredential()).get_container_client(
        name
    )


@lru_cache
def get_ledger() -> Ledger | None:
    """None (=> live extraction off) unless a persistent store is configured: an
    in-memory ledger in production would forget the month's spend on every restart."""
    if not settings.storage_account_name:
        return None
    demo = get_demo_settings()
    config = LedgerConfig(
        budget_usd=demo.budget_usd,
        gpu_usd_per_s=MODAL_L4_USD_PER_SEC,
        margin=demo.cost_margin,
        scaledown_window_s=demo.scaledown_window_s,
        cold_start_s=demo.cold_start_s,
        daily_request_cap=demo.daily_request_cap,
    )
    return Ledger(BlobStore(_blob_container(demo.ledger_container)), config)


@lru_cache
def get_llm_client() -> VLLMClient | None:
    demo = get_demo_settings()
    if not demo.vllm_base_url:
        return None
    return HTTPClient(demo.vllm_base_url, demo.vllm_api_key, timeout=demo.request_timeout_s)


class SampleSource(Protocol):
    def read(self, name: str) -> bytes: ...


class LocalSource:
    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def read(self, name: str) -> bytes:
        if Path(name).name != name:
            raise FileNotFoundError(name)
        return (self._dir / name).read_bytes()


class BlobSource:
    def __init__(self, container: ContainerClient) -> None:
        self._container = container

    def read(self, name: str) -> bytes:
        try:
            return bytes(self._container.download_blob(name).readall())
        except ResourceNotFoundError as exc:
            raise FileNotFoundError(name) from exc


class SampleStore:
    """The ten cached samples. Only names listed in samples.json can be served, so a
    request path can never reach any other file or blob."""

    def __init__(self, source: SampleSource, ttl_s: float = 300.0) -> None:
        self._source = source
        self._ttl_s = ttl_s
        self._bundle: dict[str, Any] | None = None
        self._loaded_at = 0.0

    def bundle(self) -> dict[str, Any] | None:
        if self._bundle is None or time.monotonic() - self._loaded_at > self._ttl_s:
            try:
                self._bundle = json.loads(self._source.read("samples.json"))
                self._loaded_at = time.monotonic()
            except Exception as exc:
                logger.warning("demo_samples_unavailable", error=str(exc))
                return self._bundle
        return self._bundle

    def image(self, name: str) -> bytes | None:
        bundle = self.bundle()
        if bundle is None or name not in {s["image"] for s in bundle["samples"]}:
            return None
        try:
            return self._source.read(name)
        except Exception as exc:
            logger.warning("demo_sample_image_unavailable", name=name, error=str(exc))
            return None


@lru_cache
def get_sample_store() -> SampleStore | None:
    demo = get_demo_settings()
    if settings.storage_account_name:
        return SampleStore(BlobSource(_blob_container(demo.samples_container)))
    if Path(demo.samples_dir).is_dir():
        return SampleStore(LocalSource(Path(demo.samples_dir)))
    return None


DemoDep = Annotated[DemoSettings, Depends(get_demo_settings)]
LimitersDep = Annotated[Limiters, Depends(get_limiters)]
LedgerDep = Annotated[Ledger | None, Depends(get_ledger)]
LlmDep = Annotated[VLLMClient | None, Depends(get_llm_client)]
StoreDep = Annotated[SampleStore | None, Depends(get_sample_store)]


def _live_status(ledger: Ledger | None, llm: VLLMClient | None) -> dict[str, Any]:
    """What the page and the status endpoint tell the visitor about live extraction."""
    if llm is None or ledger is None:
        return {
            "gpu": "off",
            "live": False,
            "message": "Live extraction is not set up on this deployment. "
            "The cached samples below still work.",
        }
    try:
        snap = ledger.snapshot()
    except LedgerUnavailable as exc:
        logger.error("demo_ledger_unavailable", error=str(exc))
        return {
            "gpu": "off",
            "live": False,
            "message": "Live extraction is paused because the usage counter is unavailable. "
            "The cached samples below still work.",
        }
    if snap.budget_exhausted:
        return {
            "gpu": "off",
            "live": False,
            "message": "This month's GPU budget for the demo is used up; it resets on the "
            "1st. The cached samples below still work.",
        }
    if snap.daily_cap_reached:
        return {
            "gpu": "off",
            "live": False,
            "message": "Today's limit of live requests is reached; it resets at midnight UTC. "
            "The cached samples below still work.",
        }
    messages = {
        "ready": "The GPU is awake. You can upload a receipt.",
        "warming": "Waking the GPU. This takes about 3 to 7 minutes.",
        "cold": "The GPU is asleep to save money. Wake it first; that takes about 3 to 7 minutes.",
    }
    return {"gpu": snap.gpu, "live": True, "message": messages[snap.gpu]}


def _gpu_healthy(demo: DemoSettings) -> bool:
    headers = {"Authorization": f"Bearer {demo.vllm_api_key}"} if demo.vllm_api_key else {}
    try:
        response = httpx.get(f"{demo.vllm_base_url}/health", headers=headers, timeout=5)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _poke_gpu(demo: DemoSettings) -> None:
    """One request to the scaled-to-zero server is what makes Modal start a container."""
    _gpu_healthy(demo)


def _check_image(data: bytes, max_pixels: int) -> str | None:
    """Content sniffing, not the filename or the Content-Type header - both are caller-supplied."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ("JPEG", "PNG"):
                return "Only JPEG and PNG images are accepted (no PDFs)."
            width, height = img.size
            if width * height > max_pixels:
                return "That image is too large in pixels; please use a smaller photo."
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        return "That file is not a readable JPEG or PNG image."
    return None


def _annotate(**attributes: str | int | float | bool) -> None:
    """Custom fields on the request span so App Insights can chart them."""
    trace.get_current_span().set_attributes({f"demo.{k}": v for k, v in attributes.items()})


def _run_extraction(
    data: bytes,
    demo: DemoSettings,
    ledger: Ledger | None,
    llm: VLLMClient | None,
    limiters: Limiters,
) -> tuple[int, dict[str, Any]]:
    """Everything blocking about one live request. Returns (http status, template fields)."""
    if llm is None or ledger is None:
        return 503, {"error": "Live extraction is not set up on this deployment."}
    error = _check_image(data, demo.max_pixels)
    if error:
        return 415, {"error": error}
    try:
        snap = ledger.snapshot()
    except LedgerUnavailable:
        return 503, {"error": "Live extraction is paused: the usage counter is unavailable."}
    if snap.budget_exhausted or snap.daily_cap_reached:
        return 503, {"error": "The demo's live-request budget is used up for now."}
    if snap.gpu != "ready":
        return 409, {"error": "The GPU is asleep. Press Wake the GPU first, then try again."}
    if not limiters.slots.acquire(blocking=False):
        return 503, {"error": "The demo is busy with other requests; please retry in a moment."}

    started = time.time()
    try:
        prompt = build_sroie_extraction_prompt(SroieReceipt)
        raw_output, usage = llm.complete(data, prompt, SroieReceipt)
    except httpx.HTTPError as exc:
        # A 503 or a dropped connection mid-run means the container went away.
        logger.warning("demo_upstream_error", error=str(exc))
        with contextlib.suppress(LedgerUnavailable):
            ledger.mark_cold()
        return 503, {"error": "The GPU went to sleep or is unreachable. Wake it and try again."}
    except Exception as exc:
        logger.error("demo_extract_failed", error=str(exc))
        return 500, {"error": "Something went wrong reading that receipt."}
    finally:
        limiters.slots.release()

    duration = time.time() - started
    try:
        charged = ledger.charge_request(started, duration)
    except LedgerUnavailable:
        charged = 0.0  # the next request will see the ledger is down and stop

    result: dict[str, Any] = {
        "latency_ms": round(duration * 1000),
        "visual_tokens": usage.visual_tokens,
        "cost_usd": estimate_request_cost_usd(duration * 1000),
        "read_ok": False,
    }
    try:
        receipt = SroieReceipt.model_validate_json(raw_output)
    except ValidationError:
        # Structured decoding makes this rare; a receipt with no readable positive total
        # is the usual cause, and saying so beats showing a stack trace.
        pass
    else:
        result.update(receipt.model_dump())
        result["read_ok"] = True
    _annotate(ledger_charged_usd=charged, cache_served=False, latency_ms=result["latency_ms"])
    return 200, {"result": result}


def _context(
    store: SampleStore | None,
    ledger: Ledger | None,
    llm: VLLMClient | None,
    sample_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    bundle = store.bundle() if store else None
    selected = None
    if bundle and sample_id:
        selected = next((s for s in bundle["samples"] if s["id"] == sample_id), None)
    return {
        "status": _live_status(ledger, llm),
        "bundle": bundle,
        "selected": selected,
        "error": None,
        "result": None,
        **extra,
    }


def _too_many(request: Request) -> Response:
    _annotate(rate_limited=True)
    return Response("Too many requests; please slow down.", status_code=429)


@router.get("")
def demo_page(
    request: Request,
    limiters: LimitersDep,
    ledger: LedgerDep,
    llm: LlmDep,
    store: StoreDep,
    sample: str | None = None,
) -> Response:
    if not limiters.page.allow(client_ip(request)):
        return _too_many(request)
    context = _context(store, ledger, llm, sample)
    _annotate(cache_served=sample is not None)
    return templates.TemplateResponse(request, "demo.html", context, headers=_SECURITY_HEADERS)


@router.get("/sample/{name}")
def sample_image(
    name: str,
    request: Request,
    limiters: LimitersDep,
    store: StoreDep,
) -> Response:
    if not limiters.page.allow(client_ip(request)):
        return _too_many(request)
    data = store.image(name) if store else None
    if data is None:
        return Response("Not found.", status_code=404)
    media_type = "image/png" if name.endswith(".png") else "image/jpeg"
    return Response(
        data,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/status")
def status(
    request: Request,
    demo: DemoDep,
    limiters: LimitersDep,
    ledger: LedgerDep,
    llm: LlmDep,
) -> Response:
    if not limiters.page.allow(client_ip(request)):
        return _too_many(request)
    current = _live_status(ledger, llm)
    # Only probe the GPU while a paid warm-up is in flight: probing a sleeping server would
    # start it, so an anonymous poll must never do that.
    if current["gpu"] == "warming" and ledger is not None and _gpu_healthy(demo):
        try:
            ledger.mark_ready()
            current = _live_status(ledger, llm)
        except LedgerUnavailable:
            pass
    return JSONResponse(current, headers={"Cache-Control": "no-store"})


@router.post("/warm")
def warm(
    request: Request,
    demo: DemoDep,
    limiters: LimitersDep,
    ledger: LedgerDep,
    llm: LlmDep,
) -> Response:
    if not limiters.warm.allow(client_ip(request)):
        _annotate(rate_limited=True)
        return JSONResponse({"outcome": "rate_limited"}, status_code=429)
    if llm is None or ledger is None:
        return JSONResponse({"outcome": "unavailable"}, status_code=503)
    try:
        outcome = ledger.begin_warm()
    except LedgerUnavailable:
        return JSONResponse({"outcome": "unavailable"}, status_code=503)
    if outcome == "started":
        threading.Thread(target=_poke_gpu, args=(demo,), daemon=True).start()
    if outcome == "denied":
        return JSONResponse({"outcome": "denied"}, status_code=503)
    _annotate(warm_outcome=outcome)
    return JSONResponse({"outcome": outcome}, headers={"Cache-Control": "no-store"})


@router.post("/extract")
async def extract_upload(
    request: Request,
    demo: DemoDep,
    limiters: LimitersDep,
    ledger: LedgerDep,
    llm: LlmDep,
    store: StoreDep,
) -> Response:
    # Limit first, then size, then parse: a body is never read for a caller who is
    # already over their limit, and never buffered if it is too big to be a receipt.
    if not limiters.extract.allow(client_ip(request)):
        _annotate(rate_limited=True)
        return templates.TemplateResponse(
            request,
            "demo.html",
            await run_in_threadpool(
                _context, store, ledger, llm, None, error="You've reached the hourly limit."
            ),
            status_code=429,
            headers=_SECURITY_HEADERS,
        )

    declared = request.headers.get("content-length")
    status_code, fields = 200, {}
    data = b""
    if declared is None or not declared.isdigit():
        status_code, fields = 411, {"error": "Upload size is required."}
    elif int(declared) > demo.max_upload_bytes + 65_536:
        status_code, fields = 413, {"error": "That file is too large (limit 4 MB)."}
    else:
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            status_code, fields = 400, {"error": "Choose an image file to upload."}
        else:
            data = await upload.read(demo.max_upload_bytes + 1)
            if len(data) > demo.max_upload_bytes:
                status_code, fields = 413, {"error": "That file is too large (limit 4 MB)."}
    if status_code == 200:
        status_code, fields = await run_in_threadpool(
            _run_extraction, data, demo, ledger, llm, limiters
        )

    context = await run_in_threadpool(_context, store, ledger, llm, None, **fields)
    logger.info("demo_extract", status_code=status_code, bytes=len(data))
    return templates.TemplateResponse(
        request, "demo.html", context, status_code=status_code, headers=_SECURITY_HEADERS
    )
