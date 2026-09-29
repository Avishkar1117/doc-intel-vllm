"""Spend ledger: the cost formula, GPU state transitions, month rollover, and the property
that matters most for a public endpoint - a broken store means 'no spend allowed'."""

from typing import Any

import pytest

from docintel.ledger import (
    Ledger,
    LedgerConfig,
    LedgerUnavailable,
    MemoryStore,
    _Conflict,
    billable_seconds,
)

RATE = 0.001  # round number so expected costs are easy to check by hand
WINDOW = 100.0
COLD = 300.0
CONFIG = LedgerConfig(
    budget_usd=5.0,
    gpu_usd_per_s=RATE,
    margin=1.0,
    scaledown_window_s=WINDOW,
    cold_start_s=COLD,
    daily_request_cap=3,
)


class Clock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:  # 2027-01-15 UTC
        self.now = now

    def __call__(self) -> float:
        return self.now


def make(config: LedgerConfig = CONFIG) -> tuple[Ledger, Clock]:
    clock = Clock()
    return Ledger(MemoryStore(), config, clock), clock


def test_billable_seconds_first_use_pays_window_and_cold_start() -> None:
    assert billable_seconds(None, 10, WINDOW, COLD) == 10 + WINDOW + COLD


def test_billable_seconds_within_window_pays_only_the_idle_gap() -> None:
    assert billable_seconds(30, 10, WINDOW, COLD) == 40


def test_billable_seconds_past_window_pays_a_new_cold_start() -> None:
    assert billable_seconds(500, 10, WINDOW, COLD) == 10 + WINDOW + COLD


def test_charge_request_accumulates_and_marks_gpu_ready() -> None:
    ledger, clock = make()
    first = ledger.charge_request(clock.now, 10)
    assert first == pytest.approx((10 + WINDOW + COLD) * RATE)
    clock.now += 40  # 30s after the first request finished
    second = ledger.charge_request(clock.now, 10)
    assert second == pytest.approx((10 + 30) * RATE)
    snap = ledger.snapshot()
    assert snap.spent_usd == pytest.approx(first + second)
    assert snap.day_requests == 2
    assert snap.gpu == "ready"


def test_gpu_goes_cold_after_the_scaledown_window() -> None:
    ledger, clock = make()
    ledger.charge_request(clock.now, 10)
    clock.now += 10 + WINDOW + 1
    assert ledger.snapshot().gpu == "cold"


def test_warm_up_is_paid_once_and_first_request_adds_duration_plus_idle_gap() -> None:
    ledger, clock = make()
    assert ledger.begin_warm() == "started"
    assert ledger.snapshot().gpu == "warming"
    assert ledger.begin_warm() == "already"  # a second click costs nothing
    warm_cost = (COLD + WINDOW) * RATE
    assert ledger.snapshot().spent_usd == pytest.approx(warm_cost)

    ledger.mark_ready()
    clock.now += 5
    # The 5s idle gap is charged again although the warm-up prepaid one idle window: a
    # deliberate over-estimate of at most one window, the safe direction for a spend cap.
    assert ledger.charge_request(clock.now, 8) == pytest.approx((8 + 5) * RATE)
    assert ledger.snapshot().gpu == "ready"

    # A request that arrives before mark_ready ever ran pays only its own duration.
    other, other_clock = make()
    other.begin_warm()
    assert other.charge_request(other_clock.now, 8) == pytest.approx(8 * RATE)


def test_warm_up_denied_when_it_would_exceed_the_budget() -> None:
    tiny = LedgerConfig(**{**CONFIG.__dict__, "budget_usd": 0.1})
    ledger, _ = make(tiny)
    assert ledger.begin_warm() == "denied"
    assert ledger.snapshot().spent_usd == 0.0


def test_a_stale_warm_up_expires_back_to_cold() -> None:
    ledger, clock = make()
    ledger.begin_warm()
    clock.now += COLD * 1.5 + 1
    assert ledger.snapshot().gpu == "cold"


def test_budget_and_daily_cap_flags() -> None:
    ledger, clock = make()
    for _ in range(3):
        ledger.charge_request(clock.now, 1)
        clock.now += 2
    assert ledger.snapshot().daily_cap_reached

    small = LedgerConfig(**{**CONFIG.__dict__, "budget_usd": 0.05})
    ledger2, clock2 = make(small)
    ledger2.charge_request(clock2.now, 1)
    assert ledger2.snapshot().budget_exhausted


def test_new_month_resets_spend_but_not_gpu_state() -> None:
    ledger, clock = make()
    ledger.charge_request(clock.now, 10)
    assert ledger.snapshot().spent_usd > 0
    clock.now += 40 * 86_400  # into a later month, and past the scaledown window
    snap = ledger.snapshot()
    assert snap.spent_usd == 0.0
    assert snap.day_requests == 0


class BrokenStore:
    def load(self) -> tuple[dict[str, Any] | None, str | None]:
        raise OSError("blob storage unreachable")

    def save(self, data: dict[str, Any], token: str | None) -> None:
        raise OSError("blob storage unreachable")


def test_unreadable_store_fails_closed() -> None:
    ledger = Ledger(BrokenStore(), CONFIG, Clock())
    with pytest.raises(LedgerUnavailable):
        ledger.snapshot()
    with pytest.raises(LedgerUnavailable):
        ledger.charge_request(0.0, 1)


class ConflictOnceStore(MemoryStore):
    """Loses the first write race, as a second writer would."""

    def __init__(self) -> None:
        super().__init__()
        self.conflicts_left = 1

    def save(self, data: dict[str, Any], token: str | None) -> None:
        if self.conflicts_left:
            self.conflicts_left -= 1
            raise _Conflict
        super().save(data, token)


def test_write_conflict_is_retried_not_lost() -> None:
    store = ConflictOnceStore()
    ledger = Ledger(store, CONFIG, Clock())
    ledger.charge_request(1_800_000_000.0, 10)
    assert ledger.snapshot().day_requests == 1


class AlwaysConflictStore(MemoryStore):
    def save(self, data: dict[str, Any], token: str | None) -> None:
        raise _Conflict


def test_endless_conflicts_fail_closed() -> None:
    ledger = Ledger(AlwaysConflictStore(), CONFIG, Clock())
    with pytest.raises(LedgerUnavailable):
        ledger.charge_request(0.0, 1)
