"""Persisted regime state (atomic, checksummed), the checksum-mismatch
decision tree (pause vs fallback depends on OPEN RISK), and the
PendingOrderAudit on resume (stale pendings never auto-resume).

Hysteresis must survive restart, or anti-whipsaw is useless.  A
corrupt state file never trades on compromised memory, never
fabricates counts, never silently continues.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_runner import (
    MT5Runner, MT5RunnerConfig, PendingOrderState)
from goldenswing.live.hierarchy import TimeframeStack
from goldenswing.live.range import BoundaryTracker, RangeDef
from goldenswing.live.regime_state import (
    RegimeState, load_regime_state, save_regime_state)


# ------------------------------------------------------------- helpers

def _runner(tmp_path, executor=None):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("stub",), news_pause_enabled=False)
    runner = MT5Runner(cfg, feed=None, executor=executor)
    runner._send_alert = lambda msg: None  # no Telegram in tests
    return runner


class _FakeExecutor:
    def __init__(self):
        self._pending = []
        self._positions = []
        self.cancelled = []
        self.orders = []

    def pending_orders(self, symbol=None):
        return list(self._pending)

    def positions(self, symbol=None):
        return list(self._positions)

    def cancel_pending(self, ticket):
        self.cancelled.append(ticket)
        return OrderResult(True, ticket=ticket)


def _pending(ticket=777, *, kind="limit", direction=1, level=4200.0,
             zone="default", expiry_bar=100, playbook="sweep_reversal"):
    return PendingOrderState(
        ticket=ticket, kind=kind, direction=direction, level=level,
        volume=0.01, sl=4190.0, tp=4220.0, alert_id=f"alert-{ticket}",
        playbook=playbook,
        zone={"level": level, "bottom": level - 10.0,
              "top": level + 10.0} if zone == "default" else zone,
        invalidation=4190.0, placed_bar=0, expiry_bar=expiry_bar)


def _frame(rows, *, start="2024-01-01", freq="h"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    arr = np.array(rows, dtype=float)
    return pd.DataFrame(
        {"open": arr[:, 0], "high": arr[:, 1], "low": arr[:, 2],
         "close": arr[:, 3], "volume": arr[:, 4]}, index=idx)


def _flat(n, *, close=4200.0, high=4210.0, low=4190.0):
    return [[close, high, low, close, 1000.0] for _ in range(n)]


# ------------------------------------------------------------- persistence

def test_persist_restart_counts_preserved(tmp_path):
    """Persist -> restart (new instance loads the same file) ->
    lock, confirm count, cooldowns, sweep state preserved exactly."""
    path = tmp_path / "regime_state.json"
    state = RegimeState(
        regime="breakout_trend", regime_since_ts="2024-01-01T00:00:00",
        regime_since_bar="2024-01-01T00:00:00", lock_remaining_h1=2,
        breakout_confirm_count=1, breakout_direction=1,
        breakout_edge="high",
        breakout_first_close_time="2024-01-01T00:00:00",
        sweep_pending=[{"edge": "low", "wick_price": 4090.0,
                        "wick_ts": "2024-01-01T00:00:00",
                        "wick_age_m15": 3,
                        "reclaim_bars_confirmed": 1,
                        "confirmed": False}],
        cooldown_remaining_h1={"high": 0, "low": 1},
        last_range_high=4200.0, last_range_low=4100.0,
        last_range_mid=4150.0, last_range_atr=10.0,
        last_processed_bar_close="2024-01-01T05:00:00",
        last_processed_m15_close="2024-01-01T05:00:00")
    save_regime_state(path, state)
    loaded, status, _msg = load_regime_state(path)
    assert status == "ok"
    assert loaded.lock_remaining_h1 == 2
    assert loaded.breakout_confirm_count == 1
    assert loaded.breakout_direction == 1
    assert loaded.breakout_edge == "high"
    assert loaded.cooldown_remaining_h1 == {"high": 0, "low": 1}
    assert loaded.sweep_pending[0]["edge"] == "low"
    assert loaded.last_range_high == pytest.approx(4200.0)


def test_corrupt_file_rebuild_no_fabricated_counts(tmp_path):
    """Corrupt file -> status corrupt, state None (no fabricated
    counts), never raises."""
    path = tmp_path / "regime_state.json"
    save_regime_state(path, RegimeState(regime="range"))
    raw = path.read_text()
    data = json.loads(raw)
    data["lock_remaining_h1"] = 99  # tamper: checksum now wrong
    path.write_text(json.dumps(data))
    state, status, msg = load_regime_state(path)
    assert status == "corrupt"
    assert state is None
    assert "checksum" in msg
    # Unparseable JSON is corrupt too, not a crash.
    path.write_text("{not json")
    state2, status2, _ = load_regime_state(path)
    assert status2 == "corrupt" and state2 is None


def test_missing_file_is_not_corruption(tmp_path):
    state, status, _ = load_regime_state(tmp_path / "nope.json")
    assert status == "missing" and state is None


def test_atomic_write_old_intact_on_crash(tmp_path):
    """A crash during save (os.replace raises) leaves the OLD file
    intact; garbage in the tmp file never leaks into the main file."""
    path = tmp_path / "regime_state.json"
    save_regime_state(path, RegimeState(regime="range",
                                        lock_remaining_h1=1))
    real_replace = os.replace
    try:
        os.replace = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("simulated kill mid-write"))
        # save_regime_state uses os.replace via the os module in
        # its own namespace — patch there.
        import goldenswing.live.regime_state as rs
        rs.os.replace = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("simulated kill mid-write"))
        try:
            with pytest.raises(RuntimeError):
                save_regime_state(path, RegimeState(
                    regime="breakout_trend", lock_remaining_h1=4))
        finally:
            rs.os.replace = real_replace
    finally:
        os.replace = real_replace
    loaded, status, _ = load_regime_state(path)
    assert status == "ok"
    assert loaded.regime == "range"
    assert loaded.lock_remaining_h1 == 1
    # A stray garbage tmp file is ignored by the loader.
    path.with_name(path.name + ".tmp").write_text("{garbage")
    loaded2, status2, _ = load_regime_state(path)
    assert status2 == "ok" and loaded2.lock_remaining_h1 == 1


def test_tracker_restore_no_double_count(tmp_path):
    """Persist the tracker mid-count, restore into a fresh tracker,
    re-feed the same frames: no bar counted twice (lock / confirm
    unchanged, no duplicate events)."""
    rng = RangeDef(high=4200.0, low=4100.0, mid=4150.0, width=100.0,
                   atr=10.0, touches_high=2, touches_low=2,
                   compressed=False, high_index=0, low_index=0)
    tr = BoundaryTracker(rng)
    h1 = _frame(_flat(6, close=4180.0)
                 + [[4215.0, 4220.0, 4210.0, 4215.0, 1000.0]],
                 freq="h")
    tr.update(h1, _frame(_flat(4, close=4180.0), freq="15min"))
    assert tr.edge["high"]["state"] == "beyond1"
    path = tmp_path / "regime_state.json"
    save_regime_state(path, tr.to_regime_state())
    loaded, status, _ = load_regime_state(path)
    assert status == "ok"
    tr2 = BoundaryTracker(rng)
    tr2.apply_regime_state(loaded)
    assert tr2.edge["high"]["state"] in ("beyond1", "inside")
    assert tr2.last_h1_ts == tr.last_h1_ts
    again = tr2.update(h1, _frame(_flat(4, close=4180.0),
                                   freq="15min"))
    assert again == [], "re-fed bars must not be double-counted"


# ------------------------------------------------------------- corrupt decision tree

def test_corrupt_no_open_observation_two_bars(tmp_path):
    """Corrupt + no open risk -> conservative rebuild + 2-bar
    observation, no orders in the window; then auto-resume."""
    runner = _runner(tmp_path, executor=_FakeExecutor())
    runner._handle_corrupt_regime_state("corrupt", "checksum mismatch")
    assert runner._regime_pause == "observation"
    assert runner._regime_observation_remaining == 2
    runner._advance_regime_pause()
    assert runner._regime_pause == "observation"
    assert runner._regime_observation_remaining == 1
    runner._advance_regime_pause()
    assert runner._regime_pause is None  # resumed after 2 clean bars
    assert runner._last_pending_audit is not None  # audit ran first


def test_corrupt_open_positions_paused_new(tmp_path):
    """Corrupt + open risk -> PAUSE NEW execution, keep managing;
    after 2 clean bars + no contradiction -> auto-resume."""
    runner = _runner(tmp_path, executor=_FakeExecutor())
    runner._pending_orders = {123: SimpleNamespace(alert_id="x")}
    runner._handle_corrupt_regime_state("corrupt", "checksum mismatch")
    assert runner._regime_pause == "paused_new"
    runner._advance_regime_pause()
    assert runner._regime_pause == "paused_new"
    runner._advance_regime_pause()
    assert runner._regime_pause is None


def test_corrupt_journal_corrupt_full_pause_and_resume(tmp_path):
    """Corrupt + journal corrupt -> PAUSE COMPLETELY + Telegram
    notify; /resume with still-untrusted state stays paused."""
    runner = _runner(tmp_path)
    alerts = []
    runner._send_alert = alerts.append
    # Corrupt the journal: a non-JSON line makes records() raise.
    with open(tmp_path / "mt5_journal.jsonl", "a",
              encoding="utf-8") as fh:
        fh.write("\nNOT JSON\n")
    runner._handle_corrupt_regime_state("corrupt", "checksum mismatch")
    assert runner._regime_pause == "paused_complete"
    assert alerts and "PAUSED" in alerts[0]
    assert "manual check" in alerts[0]
    msg = runner.resume_engine()
    assert runner._regime_pause == "paused_complete"
    assert "still untrusted" in msg or "NOT lifted" in msg


def test_observation_escalates_when_open_risk_appears(tmp_path):
    """Open risk appearing mid-observation escalates observation ->
    paused_new (rebuild continues, contradiction check applies)."""
    runner = _runner(tmp_path, executor=_FakeExecutor())
    runner._handle_corrupt_regime_state("corrupt", "checksum mismatch")
    assert runner._regime_pause == "observation"
    runner._pending_orders = {123: SimpleNamespace(alert_id="x")}
    runner._advance_regime_pause()
    assert runner._regime_pause == "paused_new"


def test_pause_line_shows_in_status(tmp_path):
    runner = _runner(tmp_path, executor=_FakeExecutor())
    runner._handle_corrupt_regime_state("corrupt", "checksum mismatch")
    line = runner.boundary_line()
    assert "CORRUPT, rebuilt, new execution PAUSED for 2 H1 bars" in line


# ------------------------------------------------------------- PendingOrderAudit

def _audit_runner(tmp_path, executor=None):
    ex = executor or _FakeExecutor()
    runner = _runner(tmp_path, executor=ex)
    return runner, ex


def test_audit_expired_cancelled(tmp_path):
    runner, ex = _audit_runner(tmp_path)
    runner._pending_orders[777] = _pending(expiry_bar=0)
    ex._pending.append(SimpleNamespace(ticket=777))
    summary = runner._audit_pending_orders()
    assert summary["cancelled"].get("expired") == 1
    assert 777 not in runner._pending_orders
    assert 777 in ex.cancelled


def test_audit_poi_invalidated_cancelled(tmp_path):
    runner, ex = _audit_runner(tmp_path)
    runner._pending_orders[777] = _pending(zone=None, expiry_bar=100)
    ex._pending.append(SimpleNamespace(ticket=777))
    summary = runner._audit_pending_orders()
    assert summary["cancelled"].get("stale_poi") == 1
    assert 777 not in runner._pending_orders


def test_audit_filled_during_downtime_reconciled(tmp_path):
    """Filled during downtime -> reconciled, NOT double-placed,
    left tracked for the normal reconcile."""
    runner, ex = _audit_runner(tmp_path)
    state = _pending()
    runner._pending_orders[777] = state
    ex._pending.append(SimpleNamespace(ticket=777))
    ex._positions.append(Position(
        ticket=999, symbol="XAUUSD", direction=1, volume=0.01,
        entry_price=4200.0, sl=4190.0, tp=4220.0,
        magic=runner.cfg.magic, alert_id=state.alert_id))
    summary = runner._audit_pending_orders()
    assert summary["filled"] == 1
    assert 777 in runner._pending_orders  # left, not double-placed
    assert 777 not in ex.cancelled


def test_audit_broker_cancelled_dropped(tmp_path):
    runner, ex = _audit_runner(tmp_path)
    runner._pending_orders[777] = _pending()
    # Executor shows no resting pending and no fill -> gone.
    summary = runner._audit_pending_orders()
    assert summary["cancelled"].get("broker_cancelled") == 1
    assert 777 not in runner._pending_orders


def test_audit_still_valid_kept_never_repriced(tmp_path):
    runner, ex = _audit_runner(tmp_path)
    state = _pending()
    runner._pending_orders[777] = state
    ex._pending.append(SimpleNamespace(ticket=777))
    runner._live_setup = {"regime": "trend", "direction": 1}
    summary = runner._audit_pending_orders()
    assert summary["kept"] == 1
    assert runner._pending_orders[777].level == pytest.approx(4200.0)
    assert 777 not in ex.cancelled
    assert not ex.orders  # audit never places orders


def test_audit_through_price_cancelled(tmp_path):
    """Price already through the entry (would fill instantly at a
    bad price) -> distance cancel."""
    runner, ex = _audit_runner(tmp_path)
    # H4 with a swing low at 4200 -> zones_by_side support 4200.
    h4_rows = _flat(60, close=4250.0, high=4260.0, low=4240.0)
    h4_rows[10] = [4240.0, 4260.0, 4200.0, 4240.0, 1000.0]
    h4 = _frame(h4_rows, freq="4h")
    h1 = _frame(_flat(100), freq="h")
    runner._stack = TimeframeStack.from_frames({"H1": h1, "H4": h4})
    runner._last_tick = {"mid": 4198.0}
    runner._live_setup = {"regime": "trend", "direction": 1}
    state = _pending(kind="limit", direction=1, level=4200.0,
                     zone={"level": 4200.0, "bottom": 4180.0,
                           "top": 4220.0}, expiry_bar=1000)
    runner._pending_orders[777] = state
    ex._pending.append(SimpleNamespace(ticket=777))
    summary = runner._audit_pending_orders()
    assert summary["cancelled"].get("distance") == 1
    assert 777 not in runner._pending_orders


def test_audit_regime_flip_cancelled(tmp_path):
    runner, ex = _audit_runner(tmp_path)
    runner._pending_orders[777] = _pending(direction=1)
    ex._pending.append(SimpleNamespace(ticket=777))
    runner._live_setup = {"regime": "trend", "direction": -1}
    summary = runner._audit_pending_orders()
    assert summary["cancelled"].get("regime_flip") == 1


def test_audit_score_drop_cancelled(tmp_path):
    """Setup gone on current bars (flat H1, no playbook signal) ->
    score_drop cancel."""
    runner, ex = _audit_runner(tmp_path)
    h4_rows = _flat(60, close=4250.0, high=4260.0, low=4240.0)
    h4_rows[10] = [4240.0, 4260.0, 4200.0, 4240.0, 1000.0]
    h4 = _frame(h4_rows, freq="4h")
    h1 = _frame(_flat(100), freq="h")
    runner._stack = TimeframeStack.from_frames({"H1": h1, "H4": h4})
    runner._last_tick = {"mid": 4205.0}  # above a LONG limit: not through
    runner._live_setup = {"regime": "trend", "direction": 1}
    state = _pending(kind="limit", direction=1, level=4200.0,
                     zone={"level": 4200.0, "bottom": 4180.0,
                           "top": 4220.0}, expiry_bar=1000,
                     playbook="sweep_reversal")
    runner._pending_orders[777] = state
    ex._pending.append(SimpleNamespace(ticket=777))
    summary = runner._audit_pending_orders()
    assert summary["cancelled"].get("score_drop") == 1
    assert 777 not in runner._pending_orders


def test_audit_summary_line_in_status(tmp_path):
    runner, ex = _audit_runner(tmp_path)
    runner._pending_orders[777] = _pending(expiry_bar=0)
    ex._pending.append(SimpleNamespace(ticket=777))
    runner._audit_pending_orders()
    line = runner.boundary_line()
    assert "Pending audit on resume: 1 cancelled (1 expired)" in line
