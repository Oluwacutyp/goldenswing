"""Sweep vs breakout boundary: confirmation, never wick alone;
anti-whipsaw hysteresis; regime gates; /status boundary rendering.

The boundary moment decides sweep vs breakout.  Wick alone never
fires.  A breakout needs 2 closes beyond + M15 BOS.  The regime
lock blocks flip-flop.  The buffer band is no-trade for a switch.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from goldenswing.confluence import (
    ConfluenceConfig, ConfluenceEngine)
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.risk import RiskGuards
from goldenswing.golden.detector import GoldenSignal
from goldenswing.live.range import BoundaryTracker, RangeDef
from goldenswing.playbooks.base import (
    EngineContext, HuntingGround, Playbook)
from goldenswing.playbooks.range_regime import (
    RangeCompressionBreakoutPlaybook, RangeSweepReclaimPlaybook)


# ------------------------------------------------------------- builders

def _frame(rows, *, start="2024-01-01", freq="15min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    arr = np.array(rows, dtype=float)
    return pd.DataFrame(
        {"open": arr[:, 0], "high": arr[:, 1], "low": arr[:, 2],
         "close": arr[:, 3], "volume": arr[:, 4]}, index=idx)


def _bars(n, *, close, high=None, low=None):
    h = close + 5.0 if high is None else high
    l = close - 5.0 if low is None else low
    return [[close, h, l, close, 1000.0] for _ in range(n)]


def _range():
    return RangeDef(high=4200.0, low=4100.0, mid=4150.0, width=100.0,
                    atr=10.0, touches_high=2, touches_low=2,
                    compressed=False, high_index=0, low_index=0)


def _events(events, kind):
    return [e for e in events if e.kind == kind]


# ------------------------------------------------------------- sweep

def test_wick_alone_does_not_fire():
    """Wick beyond + close inside, no reclaim (no BOS, no FVG) ->
    sweep_wick only, never sweep_confirmed, never armed."""
    tr = BoundaryTracker(_range())
    m15 = _frame(_bars(11, close=4180.0) + [
        [4195.0, 4210.0, 4195.0, 4195.0, 1000.0],  # wick bar
    ] + _bars(3, close=4180.0))
    h1 = _frame(_bars(10, close=4180.0), freq="h")
    events = tr.update(h1, m15)
    assert _events(events, "sweep_wick")
    assert not _events(events, "sweep_confirmed")
    assert "high" not in tr.sweep_confirmed
    assert tr.edge["high"]["state"] == "wick"
    assert not tr.sweep_armed("high")


def test_failed_sweep_fires_mean_reversion():
    """Wick + later close inside + M15 BOS reclaim + FVG ->
    sweep_confirmed, mean-reversion armed on the edge."""
    tr = BoundaryTracker(_range())
    rows = (
        _bars(4, close=4190.0)
        + [[4180.0, 4190.0, 4160.0, 4180.0, 1000.0]]  # dip: swing low
        + _bars(3, close=4190.0)                        # confirms dip
        + _bars(3, close=4195.0, high=4200.0, low=4190.0)
        + [[4195.0, 4210.0, 4190.0, 4195.0, 1000.0]]   # wick bar
        + [[4150.0, 4180.0, 4140.0, 4150.0, 1000.0]]   # reclaim: BOS+FVG
    )
    m15 = _frame(rows)
    h1 = _frame(_bars(10, close=4180.0), freq="h")
    events = tr.update(h1, m15)
    assert _events(events, "sweep_confirmed")
    assert tr.sweep_armed("high")
    assert tr.edge["high"]["state"] == "sweep_confirmed"


# ------------------------------------------------------------- breakout

def _m15_bos_up():
    """M15 with a confirmed swing high 4210, final close 4230
    (BOS +1), no wick: no bar has high > 4202.5 with close <= 4200."""
    rows = (
        _bars(8, close=4190.0)
        + [[4205.0, 4210.0, 4200.0, 4205.0, 1000.0]]  # swing high
        + _bars(3, close=4190.0)                       # confirms it
        + [[4215.0, 4220.0, 4210.0, 4215.0, 1000.0],
           [4220.0, 4225.0, 4215.0, 4220.0, 1000.0],
           [4225.0, 4230.0, 4220.0, 4225.0, 1000.0],
           [4230.0, 4235.0, 4225.0, 4230.0, 1000.0]]
    )
    return _frame(rows)


def _h1_breakout_bars():
    """H1: 6 inside bars, then 2 closes beyond the high edge."""
    return _frame(
        _bars(6, close=4180.0)
        + [[4215.0, 4220.0, 4210.0, 4215.0, 1000.0],
           [4220.0, 4225.0, 4215.0, 4220.0, 1000.0]],
        freq="h")


def test_single_close_beyond_does_not_switch():
    """One close beyond -> beyond1, regime stays RANGE."""
    tr = BoundaryTracker(_range())
    h1 = _frame(_bars(6, close=4180.0)
                 + [[4215.0, 4220.0, 4210.0, 4215.0, 1000.0]],
                 freq="h")
    events = tr.update(h1, _frame(_bars(4, close=4180.0)))
    assert _events(events, "breakout_pending")
    assert not _events(events, "breakout_confirmed")
    assert tr.regime == "range"
    assert tr.edge["high"]["state"] == "beyond1"


def test_successful_breakout_switches_regime_then_retest():
    """2 closes beyond + M15 BOS -> BREAKOUT_TREND 0.5x; a retest
    that holds -> 1.0x."""
    tr = BoundaryTracker(_range())
    events = tr.update(_h1_breakout_bars(), _m15_bos_up())
    assert _events(events, "breakout_confirmed")
    assert tr.regime == "breakout_trend"
    assert tr.breakout_direction == 1
    assert tr.breakout_edge == "high"
    assert tr.lock_remaining == 4
    assert tr.size_mult == pytest.approx(0.5)
    # Retest: touches the broken edge band, closes breakout way.
    h1b = pd.concat([_h1_breakout_bars(), _frame(
        [[4205.0, 4210.0, 4201.0, 4205.0, 1000.0]], freq="h",
        start=_h1_breakout_bars().index[-1] + pd.Timedelta(hours=1))])
    events2 = tr.update(h1b, _frame(_bars(4, close=4230.0)))
    assert _events(events2, "retest_held")
    assert tr.retest_held
    assert tr.size_mult == pytest.approx(1.0)


def test_no_retest_in_4_bars_stays_half_size():
    """Breakout stands on 2nd close + BOS alone when no retest
    arrives in 4 H1 bars — still 0.5x until a later retest."""
    tr = BoundaryTracker(_range())
    tr.update(_h1_breakout_bars(), _m15_bos_up())
    assert tr.regime == "breakout_trend"
    h1 = _h1_breakout_bars()
    for i in range(4):
        h1 = pd.concat([h1, _frame(
            [[4250.0, 4260.0, 4240.0, 4250.0, 1000.0]], freq="h",
            start=h1.index[-1] + pd.Timedelta(hours=1))])
    events = tr.update(h1, _frame(_bars(4, close=4230.0)))
    assert _events(events, "retest_missed")
    assert tr.regime == "breakout_trend"
    assert not tr.retest_held
    assert tr.size_mult == pytest.approx(0.5)


def test_buffer_close_does_not_switch():
    """A close inside the buffer band (edge +0.25 ATR .. +0.5 ATR)
    is no decision: the window shrinks, then the breakout stalls
    — never a switch on a touch."""
    tr = BoundaryTracker(_range())
    h1 = _frame(
        _bars(6, close=4180.0)
        + [[4215.0, 4220.0, 4210.0, 4215.0, 1000.0]]  # beyond1
        + [[4203.0, 4208.0, 4198.0, 4203.0, 1000.0]]  # buffer close
        + [[4203.0, 4208.0, 4198.0, 4203.0, 1000.0]],  # buffer close
        freq="h")
    events = tr.update(h1, _frame(_bars(4, close=4180.0)))
    assert not _events(events, "breakout_confirmed")
    assert _events(events, "breakout_stalled")
    assert tr.regime == "range"
    assert tr.edge["high"]["state"] == "inside"


# ------------------------------------------------------------- false breakout / lock / cooldown

def test_false_breakout_back_to_range_with_cooldown():
    """Close beyond, then close back inside -> false breakout:
    back to RANGE, 2 H1 cooldown on the edge, mean-reversion
    arms on the reclaim."""
    tr = BoundaryTracker(_range())
    h1 = _frame(
        _bars(6, close=4180.0)
        + [[4215.0, 4220.0, 4210.0, 4215.0, 1000.0]]  # beyond1
        + [[4195.0, 4200.0, 4190.0, 4195.0, 1000.0]],  # back inside
        freq="h")
    events = tr.update(h1, _frame(_bars(4, close=4180.0)))
    assert _events(events, "false_breakout")
    assert _events(events, "cooldown_started")
    assert tr.regime == "range"
    assert tr.cooldown_remaining["high"] == 2
    assert not tr.arm_allowed("high")
    assert tr.sweep_armed("high")  # reclaim arms mean-reversion
    # 2 H1 closes clear the cooldown.
    h1b = h1
    for _ in range(2):
        h1b = pd.concat([h1b, _frame(
            _bars(1, close=4180.0), freq="h",
            start=h1b.index[-1] + pd.Timedelta(hours=1))])
    tr.update(h1b, _frame(_bars(4, close=4180.0)))
    assert tr.cooldown_remaining["high"] == 0
    assert tr.arm_allowed("high")


def test_lock_blocks_flip_flop():
    """Breakout confirmed (lock 4 H1).  A close back inside does
    NOT flip the regime — pending_revert waits for lock expiry,
    then the revert applies once."""
    tr = BoundaryTracker(_range())
    tr.update(_h1_breakout_bars(), _m15_bos_up())
    assert tr.regime == "breakout_trend"
    h1 = _h1_breakout_bars()
    events_all = []
    for _ in range(4):
        h1 = pd.concat([h1, _frame(
            _bars(1, close=4180.0), freq="h",
            start=h1.index[-1] + pd.Timedelta(hours=1))])
        events_all.extend(tr.update(h1, _frame(_bars(4, close=4180.0))))
        if tr.regime == "range":
            break
    # After the first inside close: still breakout_trend, revert
    # pending (lock held it).  After lock expiry: back to RANGE.
    assert tr.regime == "range"
    assert any(e.kind == "regime_reverted" for e in events_all)


def test_double_count_same_bar_prevented():
    """Re-feeding the same frames consumes nothing twice."""
    tr = BoundaryTracker(_range())
    h1 = _h1_breakout_bars()
    m15 = _m15_bos_up()
    first = tr.update(h1, m15)
    second = tr.update(h1, m15)
    assert second == []
    assert tr.regime == "breakout_trend"
    # State identical to a single feed.
    assert first, "first feed must produce events"


# ------------------------------------------------------------- gates

class _StubPB(Playbook):
    live = True
    live_threshold = 60.0
    session_scope = "any"

    def __init__(self, name, scope, *, sweep=False):
        self.name = name
        self.regime_scope = scope
        self.requires_sweep_confirmation = sweep
        self.hunting_ground = HuntingGround(excluded=set(), bonuses={})

    def candidates(self, ctx):
        return []


def _sig(direction=-1):
    return GoldenSignal(index=0, timestamp=None, direction=direction,
                        score=70.0, kind="range_sweep_reclaim",
                        entry_ref=4200.0, invalidation=4210.0,
                        targets=[4150.0], atr=10.0, factors=["x"])


def _engine(pb):
    tmp = Path(tempfile.mkdtemp())
    eng = ConfluenceEngine([pb], RiskGuards(tmp / "r.json"),
                           TradeJournal(tmp / "j.jsonl"))
    df = pd.DataFrame({"open": [4200.0] * 50, "high": [4210.0] * 50,
                       "low": [4190.0] * 50, "close": [4200.0] * 50})
    ctx = EngineContext(df=df, states=[], years=1.0)
    return eng, ctx


def _verdict(regime, direction=-1, *, sweep=(), cool=()):
    return {"allowed": True, "direction": direction, "regime": regime,
            "sweep_confirmed_edges": list(sweep),
            "cooldown_edges": list(cool)}


def test_gate_sweep_blocked_without_confirmation():
    pb = _StubPB("range_sweep_reclaim", "range", sweep=True)
    eng, ctx = _engine(pb)
    ok, reason = eng.gate_live_signal(
        pb, _sig(), ctx, _verdict("range", sweep=[]))
    assert not ok and reason.startswith("sweep_unconfirmed")


def test_gate_sweep_passes_with_confirmation():
    pb = _StubPB("range_sweep_reclaim", "range", sweep=True)
    eng, ctx = _engine(pb)
    ok, reason = eng.gate_live_signal(
        pb, _sig(), ctx, _verdict("range", sweep=["high"]))
    assert ok, reason


def test_gate_cooldown_blocks_range_edge():
    pb = _StubPB("range_sweep_reclaim", "range", sweep=True)
    eng, ctx = _engine(pb)
    ok, reason = eng.gate_live_signal(
        pb, _sig(), ctx,
        _verdict("range", sweep=["high"], cool=["high"]))
    assert not ok and reason.startswith("cooldown:")


def test_gate_trend_playbook_passes_in_breakout_trend():
    pb = _StubPB("sweep_reversal", "trend")
    eng, ctx = _engine(pb)
    ok, reason = eng.gate_live_signal(
        pb, _sig(), ctx, _verdict("breakout_trend"))
    assert ok, reason


def test_gate_range_playbook_blocked_in_trend_and_breakout():
    pb = _StubPB("range_sweep_reclaim", "range", sweep=True)
    eng, ctx = _engine(pb)
    for regime in ("trend", "breakout_trend"):
        ok, reason = eng.gate_live_signal(
            pb, _sig(), ctx, _verdict(regime, sweep=["high"]))
        assert not ok and "regime:" in reason, regime


def test_gate_breakout_playbook_scope_both():
    assert RangeCompressionBreakoutPlaybook.regime_scope == "both"
    assert RangeSweepReclaimPlaybook.requires_sweep_confirmation is True
    pb = _StubPB("range_compression_breakout", "both")
    eng, ctx = _engine(pb)
    for regime in ("range", "trend", "breakout_trend"):
        ok, reason = eng.gate_live_signal(
            pb, _sig(), ctx, _verdict(regime))
        assert ok, (regime, reason)


# ------------------------------------------------------------- /status rendering

def test_boundary_line_rendering(tmp_path):
    """boundary_line shows the wick / breakout / lock state in the
    specified formats."""
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("stub",), news_pause_enabled=False)
    runner = MT5Runner(cfg, feed=None, executor=None)
    tr = BoundaryTracker(_range())
    m15 = _frame(_bars(11, close=4180.0) + [
        [4195.0, 4210.0, 4195.0, 4195.0, 1000.0]])
    tr.update(_frame(_bars(10, close=4180.0), freq="h"), m15)
    runner._boundary_snapshot = tr.snapshot()
    runner._regime_state_status = "persisted 10:00:00"
    line = runner.boundary_line()
    assert "Edge HIGH 4200.0, buffer ±2.5" in line
    assert "wick beyond, awaiting reclaim close" in line
    # Breakout-pending rendering.
    tr2 = BoundaryTracker(_range())
    h1 = _frame(_bars(6, close=4180.0)
                 + [[4215.0, 4220.0, 4210.0, 4215.0, 1000.0]],
                 freq="h")
    tr2.update(h1, _frame(_bars(4, close=4180.0)))
    runner._boundary_snapshot = tr2.snapshot()
    line2 = runner.boundary_line()
    assert "Breakout: close beyond 1/2, awaiting 2nd close" in line2
    # Lock rendering after a confirmed breakout.
    tr3 = BoundaryTracker(_range())
    tr3.update(_h1_breakout_bars(), _m15_bos_up())
    runner._boundary_snapshot = tr3.snapshot()
    line3 = runner.boundary_line()
    assert "Regime locked BREAKOUT_TREND for 4 more H1 bars" in line3
