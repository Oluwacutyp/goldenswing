"""Pending-order score-decay cancellation (owner's-bot
PendingOrderTracker): a pending whose setup no longer re-emits at
this level scoring at/above the floor accrues decay bars, and at
score_decay_bars (default 2) it is cancelled.  A re-evaluation crash
never cancels (fail-open).
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.executor import OrderResult
from goldenswing.execution.mt5_runner import (
    MT5Runner, MT5RunnerConfig, PendingOrderState)
from goldenswing.live.hierarchy import TimeframeStack


class _Exec:
    def __init__(self):
        self.cancelled: list[int] = []

    def cancel_pending(self, ticket):
        self.cancelled.append(ticket)
        return OrderResult(True, ticket=ticket)

    def positions(self, symbol=None):
        return []


class _StubPB:
    name = "fvg_midline"

    def __init__(self, sigs=None, exc=None):
        self._sigs = sigs
        self._exc = exc

    def live_signal(self, states, t, df=None):
        if self._exc is not None:
            raise self._exc
        return list(self._sigs or [])


def _runner(tmp_path, *, pb, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        **cfg_kw)
    runner = MT5Runner(cfg, feed=None, executor=_Exec())
    h1 = pd.DataFrame(
        {"open": [2000.0] * 60, "high": [2001.0] * 60,
         "low": [1999.0] * 60, "close": [2000.5] * 60,
         "volume": [100] * 60, "real_volume": [0] * 60,
         "spread": [30] * 60},
        index=pd.date_range("2026-01-01", periods=60, freq="h",
                            tz="UTC"))
    runner._stack = TimeframeStack.from_frames({"H1": h1})
    runner._playbooks = [pb]
    return runner


def _pending(runner, *, score=80.0, level=2000.0):
    # Placed at the live edge (bar 58 of 60): expiry is 62, so the
    # H1 close at bar 59 does NOT expire it — decay is what is
    # being tested, not expiry.
    state = PendingOrderState(
        ticket=777, kind="limit", direction=1, level=level,
        volume=0.01, sl=1990.0, tp=2030.0, alert_id="MT5-x",
        playbook="fvg_midline", zone=None, invalidation=1990.0,
        placed_bar=58, expiry_bar=58 + 4, placed_score=score)
    runner._pending_orders[777] = state
    return state


def _h1_close(runner):
    runner._on_h1_close(
        {"mid": 2000.5, "time": pd.Timestamp(
            "2026-01-04", tz="UTC").to_pydatetime()},
        {})


def _sig(score, entry_ref=2000.0, direction=1, atr=10.0):
    return SimpleNamespace(direction=direction, score=score,
                           entry_ref=entry_ref, atr=atr)


def test_decay_cancels_after_two_decayed_closes(tmp_path):
    runner = _runner(tmp_path, pb=_StubPB(sigs=[]))
    _pending(runner)
    _h1_close(runner)  # decay 1/2
    assert 777 in runner._pending_orders
    assert runner._pending_orders[777].decay_bars == 1
    _h1_close(runner)  # decay 2/2 -> cancel
    assert 777 not in runner._pending_orders
    assert runner.executor.cancelled == [777]
    notes = [r["text"] for r in runner.journal.records()
             if r.get("type") == "note"]
    assert any("score_decay" in n for n in notes)


def test_decay_resets_when_setup_recovers(tmp_path):
    pb = _StubPB(sigs=[])
    runner = _runner(tmp_path, pb=pb)
    _pending(runner)
    _h1_close(runner)
    assert runner._pending_orders[777].decay_bars == 1
    pb._sigs = [_sig(85.0)]  # setup re-emits above the floor
    _h1_close(runner)
    assert 777 in runner._pending_orders
    assert runner._pending_orders[777].decay_bars == 0


def test_below_floor_reemission_counts_as_decay(tmp_path):
    pb = _StubPB(sigs=[_sig(40.0)])  # re-emits, but below floor 60
    runner = _runner(tmp_path, pb=pb)
    _pending(runner)
    _h1_close(runner)
    assert runner._pending_orders[777].decay_bars == 1
    _h1_close(runner)
    assert 777 not in runner._pending_orders


def test_wrong_direction_or_level_counts_as_decay(tmp_path):
    pb = _StubPB(sigs=[_sig(90.0, direction=-1)])
    runner = _runner(tmp_path, pb=pb)
    _pending(runner)
    _h1_close(runner)
    assert runner._pending_orders[777].decay_bars == 1
    pb._sigs = [_sig(90.0, entry_ref=2050.0)]  # far from the level
    _h1_close(runner)
    assert 777 not in runner._pending_orders


def test_decay_disabled_leaves_pending_alone(tmp_path):
    runner = _runner(tmp_path, pb=_StubPB(sigs=[]),
                     score_decay_enabled=False)
    _pending(runner)
    _h1_close(runner)
    _h1_close(runner)
    assert 777 in runner._pending_orders
    assert runner._pending_orders[777].decay_bars == 0


def test_reevaluation_crash_never_cancels(tmp_path):
    runner = _runner(tmp_path,
                     pb=_StubPB(exc=RuntimeError("boom")))
    _pending(runner)
    _h1_close(runner)
    _h1_close(runner)
    assert 777 in runner._pending_orders
    notes = [r["text"] for r in runner.journal.records()
             if r.get("type") == "note"]
    assert any("score-decay check failed" in n for n in notes)


def test_placed_score_recorded_at_track_time(tmp_path):
    runner = _runner(tmp_path, pb=_StubPB(sigs=[]))
    sig = _sig(77.0)
    order = SimpleNamespace(ticket=888, kind="limit", direction=1,
                            price=2000.0, volume=0.01, sl=1990.0,
                            tp=2030.0, alert_id="MT5-y")
    state = runner._track_pending_order(
        order, playbook="fvg_midline", zone=None, invalidation=1990.0,
        placed_bar=50, sig=sig, verdict=None, track_request=None)
    assert state.placed_score == pytest.approx(77.0)
    assert state.decay_bars == 0
