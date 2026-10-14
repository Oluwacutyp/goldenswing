"""Tests for init deadline wiring (goldenswing.live.startup).

- Per-call TimeoutError in Phase 2 -> the existing stale path (gap opened,
  engine ready, tick loop will health-skip).
- InitBudgetExhausted in Phase 2 -> abort outright, never the stale path.
  This pins the except-clause ordering: InitBudgetExhausted must precede the
  broad except Exception, or "out of time" gets misclassified as "probe
  failed".
- A slow Phase 0 trips the between-phase deadline check.
- Bounded lock acquisition: a wedged holder can't block boot past the budget.
"""
import json
import os
import threading
import time

import pytest

from goldenswing.live import startup as startup_mod
from goldenswing.live.data_health import GapTracker
from goldenswing.live.journal_lock import journal_transaction
from goldenswing.live.monitor import Status
from goldenswing.live.startup import (
    initialize_engine,
    InitBudgetExhausted,
)


class _Recovery:
    def __init__(self):
        self.monitor = object()

    def startup_check(self, journal_path):
        return (Status.NOMINAL, "test nominal")


def _harness(tmp_path, recovery=None):
    jp = str(tmp_path / "journal.jsonl")
    return {
        "journal_path": jp,
        "recovery": recovery or _Recovery(),
        "gap_tracker": GapTracker(str(tmp_path / "state")),
        "tracker": object(),
        "tracker_state_path": str(tmp_path / "tracker.json"),
    }


def _run(h, **kw):
    return initialize_engine(
        journal_path=h["journal_path"],
        recovery=h["recovery"],
        gap_tracker=h["gap_tracker"],
        tracker=h["tracker"],
        tracker_state_path=h["tracker_state_path"],
        check_data_health_fn=kw.get("check_data_health_fn", lambda: None),
        fetch_broker_state=lambda: ([], []),
        fetch_missing_bars=lambda s, e: ([], True),
        last_observed_price_fn=lambda: 4180.0,
        on_alert=kw.get("on_alert"),
    )


def _hang():
    threading.Event().wait(60)


def test_phase2_per_call_timeout_takes_stale_path(tmp_path, monkeypatch):
    monkeypatch.setattr(startup_mod, "HEALTH_TIMEOUT_S", 0.2)
    monkeypatch.setattr(startup_mod, "INIT_BUDGET_S", 30.0)
    h = _harness(tmp_path)
    started = time.monotonic()
    report = _run(h, check_data_health_fn=_hang)
    assert time.monotonic() - started < 10.0
    # Stale path: gap opened (fail-closed), engine ready; tick loop skips.
    assert report.ready is True
    assert len(report.gaps_open) == 1
    assert report.blocking_divergences == []


def test_phase2_budget_exhaustion_aborts_never_stale(tmp_path, monkeypatch):
    """Handler-ordering pin: InitBudgetExhausted must not be swallowed by the
    broad except Exception into the stale path."""
    monkeypatch.setattr(startup_mod, "INIT_BUDGET_S", 30.0)
    monkeypatch.setattr(startup_mod, "HEALTH_TIMEOUT_S", 30.0)
    h = _harness(tmp_path)
    alerts = []

    def _raise_budget():
        raise InitBudgetExhausted("simulated exhaustion")

    report = _run(h, check_data_health_fn=_raise_budget, on_alert=alerts.append)
    assert report.ready is False
    assert "budget exhausted" in report.ready_reason
    # The stale path never ran: no gap was opened as a side effect.
    assert report.gaps_open == []
    assert h["gap_tracker"]._state.get("open_gap") is None
    assert any("budget exhausted" in a for a in alerts)


def test_slow_phase0_trips_between_phase_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(startup_mod, "INIT_BUDGET_S", 0.15)

    class _SlowRecovery(_Recovery):
        def startup_check(self, journal_path):
            time.sleep(0.4)
            return (Status.NOMINAL, "slow nominal")

    h = _harness(tmp_path, recovery=_SlowRecovery())
    report = _run(h)
    assert report.ready is False
    assert "budget exhausted" in report.ready_reason


def test_bounded_lock_acquisition(tmp_path, monkeypatch):
    monkeypatch.setattr(startup_mod, "INIT_BUDGET_S", 0.5)
    h = _harness(tmp_path)
    jp = h["journal_path"]
    acquired_evt = threading.Event()

    def holder():
        with journal_transaction(jp, blocking=True):
            acquired_evt.set()
            time.sleep(5)

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    assert acquired_evt.wait(5)

    started = time.monotonic()
    report = _run(h)
    elapsed = time.monotonic() - started
    assert elapsed < 4.0, "boot must not wait out a wedged lock holder"
    assert report.ready is False
    assert "budget exhausted" in report.ready_reason
    assert "lock acquisition" in report.ready_reason
