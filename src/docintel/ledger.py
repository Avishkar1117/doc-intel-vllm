"""GPU-spend ledger behind the public demo's monthly cost cap.

Modal's per-environment budget isn't available on this plan, so the cap lives in the app:
one small JSON state blob (spend this month, requests today, when the GPU was last busy).
It fails closed: if the state can't be read or written, callers get LedgerUnavailable and
must treat the budget as spent - a broken counter must never mean unlimited spend.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, TypeVar

import structlog
from azure.core import MatchConditions
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.storage.blob import ContainerClient

logger = structlog.get_logger(__name__)

T = TypeVar("T")
GpuState = Literal["ready", "warming", "cold"]
WarmOutcome = Literal["started", "already", "denied"]

# Optimistic-concurrency retries before giving up; one replica is the norm, so a
# conflict means a second writer raced us and one reload almost always resolves it.
_MAX_ATTEMPTS = 4


class LedgerUnavailable(Exception):
    """The ledger could not be read or written; the caller must treat the budget as spent."""


class _Conflict(Exception):
    """Someone else saved the state between our read and our write (HTTP 409/412)."""


@dataclass
class LedgerState:
    month: str
    spent_usd: float = 0.0
    # Epoch seconds the GPU last finished work; None means it is presumed asleep.
    last_activity: float | None = None
    # Epoch seconds a warm-up was paid for and requested; cleared once the GPU answers.
    warming_since: float | None = None
    day: str = ""
    day_requests: int = 0


@dataclass(frozen=True)
class LedgerConfig:
    budget_usd: float
    gpu_usd_per_s: float
    # Multiplier on every charge. The GPU rate is the only price we know; Modal also bills
    # CPU and memory, so the estimate is deliberately inflated rather than trusted.
    margin: float
    # Must equal the demo Modal app's scaledown_window, or the idle-tail estimate is wrong.
    scaledown_window_s: float
    # Worst measured cold start (Phase 5: 6m24s), not the average.
    cold_start_s: float
    daily_request_cap: int


@dataclass(frozen=True)
class Snapshot:
    gpu: GpuState
    spent_usd: float
    budget_usd: float
    day_requests: int
    budget_exhausted: bool
    daily_cap_reached: bool


def billable_seconds(
    gap_s: float | None, duration_s: float, window_s: float, cold_start_s: float
) -> float:
    """GPU seconds one request adds: its own duration, the idle time the container sat
    waiting for it (capped at the scaledown window), and a fresh boot if the container had
    already been shut down by then. gap_s is None when the GPU has never been used.
    """
    if gap_s is None or gap_s > window_s:
        # The previous burst's idle tail ran a full window, then this request booted anew.
        return duration_s + window_s + cold_start_s
    return duration_s + gap_s


class LedgerStore(Protocol):
    def load(self) -> tuple[dict[str, Any] | None, str | None]: ...

    def save(self, data: dict[str, Any], token: str | None) -> None: ...


class MemoryStore:
    """Non-persistent store for tests. Never used in the deployed app (it would forget
    the month's spend on every restart)."""

    def __init__(self) -> None:
        self._data: dict[str, Any] | None = None
        self._version = 0

    def load(self) -> tuple[dict[str, Any] | None, str | None]:
        if self._data is None:
            return None, None
        return dict(self._data), str(self._version)

    def save(self, data: dict[str, Any], token: str | None) -> None:
        current = None if self._data is None else str(self._version)
        if token != current:
            raise _Conflict
        self._data = dict(data)
        self._version += 1


class BlobStore:
    """State in one blob, written with the ETag we read - a stale writer loses, not wins."""

    def __init__(self, container: ContainerClient, blob_name: str = "state.json") -> None:
        self._blob = container.get_blob_client(blob_name)

    def load(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            download = self._blob.download_blob()
        except ResourceNotFoundError:
            return None, None
        data: dict[str, Any] = json.loads(download.readall())
        return data, str(download.properties.etag)

    def save(self, data: dict[str, Any], token: str | None) -> None:
        payload = json.dumps(data)
        try:
            if token is None:
                self._blob.upload_blob(payload, overwrite=False)
            else:
                self._blob.upload_blob(
                    payload,
                    overwrite=True,
                    etag=token,
                    match_condition=MatchConditions.IfNotModified,
                )
        except HttpResponseError as exc:
            if exc.status_code in (409, 412):
                raise _Conflict from exc
            raise


def _month(now: float) -> str:
    return datetime.fromtimestamp(now, UTC).strftime("%Y-%m")


def _day(now: float) -> str:
    return datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")


class Ledger:
    def __init__(
        self,
        store: LedgerStore,
        config: LedgerConfig,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._config = config
        self._clock = clock

    def _load(self, now: float) -> tuple[LedgerState, str | None]:
        try:
            raw, token = self._store.load()
        except Exception as exc:
            raise LedgerUnavailable(f"could not read ledger: {exc}") from exc
        state = LedgerState(**raw) if raw else LedgerState(month=_month(now))
        if state.month != _month(now):
            # Spend resets on the 1st (UTC); what the GPU is doing right now does not.
            state = LedgerState(
                month=_month(now),
                last_activity=state.last_activity,
                warming_since=state.warming_since,
            )
        if state.day != _day(now):
            state.day = _day(now)
            state.day_requests = 0
        return state, token

    def _update(self, mutate: Callable[[LedgerState, float], T]) -> T:
        for _ in range(_MAX_ATTEMPTS):
            now = self._clock()
            state, token = self._load(now)
            result = mutate(state, now)
            try:
                self._store.save(asdict(state), token)
            except _Conflict:
                continue
            except Exception as exc:
                raise LedgerUnavailable(f"could not write ledger: {exc}") from exc
            return result
        raise LedgerUnavailable("ledger kept changing under us")

    def _gpu_state(self, state: LedgerState, now: float) -> GpuState:
        cfg = self._config
        if state.warming_since is not None and now - state.warming_since < cfg.cold_start_s * 1.5:
            return "warming"
        if state.last_activity is not None and now - state.last_activity < cfg.scaledown_window_s:
            return "ready"
        return "cold"

    def snapshot(self) -> Snapshot:
        """Read-only view; raises LedgerUnavailable if the state can't be read."""
        now = self._clock()
        state, _ = self._load(now)
        return Snapshot(
            gpu=self._gpu_state(state, now),
            spent_usd=state.spent_usd,
            budget_usd=self._config.budget_usd,
            day_requests=state.day_requests,
            budget_exhausted=state.spent_usd >= self._config.budget_usd,
            daily_cap_reached=state.day_requests >= self._config.daily_request_cap,
        )

    def begin_warm(self) -> WarmOutcome:
        """Pays for a cold start up front (boot plus one idle window) and marks the GPU
        as warming. Idempotent: a GPU already ready or warming costs nothing more."""

        def mutate(state: LedgerState, now: float) -> WarmOutcome:
            cfg = self._config
            if self._gpu_state(state, now) != "cold":
                return "already"
            cost = (cfg.cold_start_s + cfg.scaledown_window_s) * cfg.gpu_usd_per_s * cfg.margin
            if state.spent_usd + cost > cfg.budget_usd:
                return "denied"
            state.spent_usd += cost
            state.warming_since = now
            return "started"

        return self._update(mutate)

    def mark_ready(self) -> None:
        def mutate(state: LedgerState, now: float) -> None:
            state.last_activity = now
            state.warming_since = None

        self._update(mutate)

    def mark_cold(self) -> None:
        def mutate(state: LedgerState, now: float) -> None:
            state.last_activity = None
            state.warming_since = None

        self._update(mutate)

    def charge_request(self, started_at: float, duration_s: float) -> float:
        """Records one finished request and returns the USD added to this month's spend."""

        def mutate(state: LedgerState, now: float) -> float:
            cfg = self._config
            if state.warming_since is not None:
                # begin_warm() already paid for the boot and one idle window.
                seconds = duration_s
            else:
                gap = None if state.last_activity is None else started_at - state.last_activity
                seconds = billable_seconds(
                    gap, duration_s, cfg.scaledown_window_s, cfg.cold_start_s
                )
            cost = seconds * cfg.gpu_usd_per_s * cfg.margin
            state.spent_usd += cost
            state.last_activity = started_at + duration_s
            state.warming_since = None
            state.day_requests += 1
            return cost

        return self._update(mutate)
