"""RANGE regime (D1 flat is a range, not a shutdown).

Covers: range detection, the four range playbooks firing on
synthetic range bars, regime gating (trend never in range, range
never in trend), no-defined-range -> HOLD, and the unchanged gates
(score / confluence / cost) still blocking range trades.
"""
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.confluence import (ConfluenceConfig, ConfluenceEngine,
                                    cost_gate_ok)
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.risk import RiskConfig, RiskGuards
from goldenswing.live.hierarchy import (TimeframeStack, evaluate_hierarchy,
                                        live_setup, live_setup_reason)
from goldenswing.live.range import (RangeDef, detect_range,
                                     resolve_range_touch)
from goldenswing.playbooks.fvg_midline import FVGMidlinePlaybook
from goldenswing.playbooks.range_regime import (
    RangeCompressionBreakoutPlaybook, RangeMeanReversionPlaybook,
    RangeQMPlaybook, RangeSweepReclaimPlaybook)


# ------------------------------------------------------------ builders
def _idx(n, start="2026-01-01", freq="h"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


def _frame(prices, *, start="2026-01-01", freq="h", spread=0.5):
    """Bars whose close follows ``prices``; high/low = close +/- spread.

    Open sits slightly below close (small real body): a zero-body
    bar would make every triangle peak look like a rejection wick.
    """
    idx = _idx(len(prices), start=start, freq=freq)
    closes = list(prices)
    return pd.DataFrame({
        "open": [c - 0.2 for c in closes],
        "high": [c + spread for c in closes],
        "low": [c - spread for c in closes],
        "close": closes,
        "volume": [100] * len(closes),
        "real_volume": [0] * len(closes),
        "spread": [30] * len(closes)}, index=idx)


def _triangle(n=64, lo=4100.0, hi=4200.0):
    """Triangle wave oscillating lo..hi, period 32 bars."""
    out = []
    for t in range(n):
        phase = (t % 32) / 32.0
        if phase < 0.5:
            out.append(lo + (hi - lo) * (phase / 0.5))
        else:
            out.append(hi - (hi - lo) * ((phase - 0.5) / 0.5))
    return out


def _flat_d1(n=12):
    """Monotone D1: no confirmed swings -> structure bias 0.

    Ends 2025-12-31, BEFORE the H1 triangle starts (2026-01-01):
    evaluate_hierarchy causally slices every TF to bars closed by
    the H1 end, so a D1 frame overlapping the H1 window would leave
    too few closed D1 bars for the data leg.
    """
    prices = [4000.0 + 10.0 * i for i in range(n)]
    return _frame(prices, start="2025-12-20", freq="D", spread=2.0)


def _stack(h1_extra=None, *, d1=None):
    h1_prices = _triangle(64)
    if h1_extra:
        h1_prices = h1_prices + list(h1_extra)
    h1 = _frame(h1_prices, spread=0.25)
    h4 = h1.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    if d1 is None:
        d1 = _flat_d1()
    m15 = h1.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    m15 = m15.dropna()
    m5 = h1.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    m5 = m5.dropna()
    return TimeframeStack.from_frames(
        {"D1": d1, "H4": h4, "H1": h1, "M15": m15, "M5": m5})


def _range_def():
    """The detected range on the base triangle (edges ~4200/4100)."""
    h1 = _frame(_triangle(64), spread=0.25)
    h4 = h1.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    rng = detect_range(None, h4, h1)
    assert rng is not None
    return rng, h1


def _bar_df(rows):
    """Explicit OHLC rows [(o,h,l,c), ...] as an H1 frame."""
    idx = _idx(len(rows))
    return pd.DataFrame({
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "volume": [100] * len(rows), "real_volume": [0] * len(rows),
        "spread": [30] * len(rows)}, index=idx)


def _append_bars(h1, rows):
    """Append explicit OHLC rows to an H1 frame (continuing the index)."""
    start = h1.index[-1] + pd.Timedelta(hours=1)
    extra = _bar_df(rows)
    extra.index = pd.date_range(start=start, periods=len(rows),
                                freq="h", tz="UTC")
    return pd.concat([h1, extra])


# ------------------------------------------------------------ detection
def test_detect_range_defined():
    rng, _ = _range_def()
    assert abs(rng.high - 4200.0) < 2.0
    assert abs(rng.low - 4100.0) < 2.0
    assert rng.touches_high >= 2 and rng.touches_low >= 2
    assert rng.width >= 1.5 * rng.atr
    assert abs(rng.mid - 4150.0) < 2.0


def test_detect_range_none_when_trending():
    trending = _frame([4000.0 + 5.0 * i for i in range(64)],
                      spread=0.25)
    h4 = trending.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    assert detect_range(None, h4, trending) is None


def test_live_setup_range_armed_on_flat():
    stack = _stack()
    setup = live_setup(stack)
    assert setup is not None
    assert setup["regime"] == "range"
    assert setup["direction"] == 0
    assert setup["bias_d1"] == 0
    assert len(setup["edges"]) == 2
    assert live_setup_reason(stack) == "armed_range"
    assert setup["size_mult"] == 0.5


def test_live_setup_hold_when_no_defined_range():
    trending = _frame([4000.0 + 5.0 * i for i in range(64)],
                      spread=0.25)
    h4 = trending.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    stack = TimeframeStack.from_frames(
        {"D1": _flat_d1(), "H4": h4, "H1": trending})
    assert live_setup(stack) is None
    reason = live_setup_reason(stack)
    assert reason.startswith("d1_flat")
    assert "no defined range" in reason


def test_check_zone_touch_range_aware():
    rng, _ = _range_def()
    setup = {"regime": "range", "range": rng.to_dict(),
             "edges": rng.edges(), "direction": 0}
    assert resolve_range_touch(rng, rng.high) is not None
    assert resolve_range_touch(rng, rng.mid) is None


# ------------------------------------------------------------ playbooks

def _h4_of(df):
    """Native-style H4 frame for a test H1 frame (fixture data)."""
    return df.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})


def test_range_sweep_reclaim_fires():
    rng, h1 = _range_def()
    atr = rng.atr
    # Sweep bar: wick beyond high + close back inside; confirm bar
    # closes further inside.
    rows = [(4195.0, 4215.0, 4185.0, 4190.0),
            (4195.0, 4196.0, 4175.0, 4180.0)]
    df = _append_bars(h1, rows)
    pb = RangeSweepReclaimPlaybook()
    pb.set_h4_frame(_h4_of(df))
    sigs = pb.live_signal([], len(df) - 1, df)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.direction == -1
    assert sig.score >= 60.0
    assert "range_sweep_reclaim" in sig.factors


def test_range_qm_fires():
    rng, h1 = _range_def()
    rows = [(4190.0, 4220.0, 4185.0, 4195.0),   # head beyond high
            (4145.0, 4150.0, 4130.0, 4140.0),  # pullback low 4130
            (4188.0, 4198.0, 4180.0, 4190.0),  # failed retest 4198
            (4128.0, 4130.0, 4120.0, 4125.0)]  # break < neckline
    df = _append_bars(h1, rows)
    pb = RangeQMPlaybook()
    pb.set_h4_frame(_h4_of(df))
    sigs = pb.live_signal([], len(df) - 1, df)
    assert len(sigs) == 1
    assert sigs[0].direction == -1
    assert "range_qm" in sigs[0].factors


def test_range_mean_reversion_fires():
    rng, h1 = _range_def()
    atr = rng.atr
    # Rejection bar: high inside the edge band (level not exceeded),
    # wick >= 2x body.
    hi = rng.high - 0.25 * atr
    rows = [(4180.0, hi, 4170.0, 4182.0)]
    df = _append_bars(h1, rows)
    pb = RangeMeanReversionPlaybook()
    pb.set_h4_frame(_h4_of(df))
    sigs = pb.live_signal([], len(df) - 1, df)
    assert len(sigs) == 1
    assert sigs[0].direction == -1
    assert sigs[0].score >= 60.0


def test_range_compression_breakout_fires():
    rng, h1 = _range_def()
    # Tail: a wide bar, then 3 genuine inside bars (each inside the
    # previous -> compressed), then a breakout close beyond the
    # high with displacement.
    rows = [(4105.0, 4120.0, 4090.0, 4105.0),   # wide bar
            (4105.0, 4115.0, 4095.0, 4105.0),  # inside 1
            (4105.0, 4110.0, 4100.0, 4105.0),  # inside 2
            (4105.0, 4108.0, 4102.0, 4105.0),  # inside 3
            (4105.0, 4216.0, 4100.0, 4215.0)]  # breakout close
    df = _append_bars(h1, rows)
    pb = RangeCompressionBreakoutPlaybook()
    pb.set_h4_frame(_h4_of(df))
    sigs = pb.live_signal([], len(df) - 1, df)
    assert len(sigs) == 1
    assert sigs[0].direction == +1
    assert sigs[0].targets[0] > 4200.0  # measured move above high


# --------------------------------------------------------- regime gating
def _engine(tmp_path):
    return ConfluenceEngine(
        playbooks=[], risk_guards=RiskGuards(tmp_path / "risk.json",
                                             RiskConfig()),
        journal=TradeJournal(tmp_path / "journal.jsonl"),
        risk_mode="shadow")


class _StubPB:
    def __init__(self, name, scope, sweep=False):
        self.name = name
        self.regime_scope = scope
        self.requires_sweep_confirmation = sweep
        self.session_scope = "any"
        self.hunting_ground = SimpleNamespace(excluded=[])

    def check_live_signal(self, signal, ctx):
        return True, ""


def _sig(direction=1):
    return SimpleNamespace(direction=direction, index=10,
                           entry_ref=2000.0, invalidation=1990.0,
                           targets=[2030.0], kind="")


def test_trend_playbook_blocked_in_range_regime(tmp_path):
    engine = _engine(tmp_path)
    pb = _StubPB("trend_pb", "trend")
    verdict = {"allowed": True, "blocked_leg": "", "legs": {},
               "direction": -1, "regime": "range"}
    ok, reason = engine.gate_live_signal(pb, _sig(-1),
                                         SimpleNamespace(),
                                         hierarchy_verdict=verdict)
    assert ok is False
    assert reason.startswith("regime:")


def test_range_playbook_blocked_in_trend_regime(tmp_path):
    engine = _engine(tmp_path)
    pb = _StubPB("range_pb", "range")
    verdict = {"allowed": True, "blocked_leg": "", "legs": {},
               "direction": 1, "regime": "trend"}
    ok, reason = engine.gate_live_signal(pb, _sig(1),
                                         SimpleNamespace(),
                                         hierarchy_verdict=verdict)
    assert ok is False
    assert reason.startswith("regime:")


def test_range_playbook_passes_regime_in_range(tmp_path):
    engine = _engine(tmp_path)
    pb = _StubPB("range_pb", "range")
    verdict = {"allowed": True, "blocked_leg": "", "legs": {},
               "direction": 1, "regime": "range",
               "sweep_confirmed_edges": [],
               "cooldown_edges": []}
    ok, reason = engine.gate_live_signal(pb, _sig(1),
                                         SimpleNamespace(),
                                         hierarchy_verdict=verdict)
    # Regime passes; later gates (cost) decide — never regime.
    assert not reason.startswith("regime:")


def test_cost_gate_still_blocks_range_signal(tmp_path):
    engine = _engine(tmp_path)
    pb = _StubPB("range_pb", "range")
    verdict = {"allowed": True, "blocked_leg": "", "legs": {},
               "direction": 1, "regime": "range",
               "sweep_confirmed_edges": [],
               "cooldown_edges": []}
    # Tiny risk (entry 2000, invalidation 1999) fails the cost gate.
    sig = _sig(1)
    sig.invalidation = 1999.0
    ok, reason = engine.gate_live_signal(pb, sig, SimpleNamespace(),
                                         hierarchy_verdict=verdict)
    assert ok is False
    assert reason.startswith("cost_gate:")


def test_trend_playbook_passes_regime_in_breakout_trend(tmp_path):
    engine = _engine(tmp_path)
    pb = _StubPB("trend_pb", "trend")
    verdict = {"allowed": True, "blocked_leg": "", "legs": {},
               "direction": 1, "regime": "breakout_trend"}
    ok, reason = engine.gate_live_signal(pb, _sig(1),
                                         SimpleNamespace(),
                                         hierarchy_verdict=verdict)
    assert not reason.startswith("regime:")


# ------------------------------------------------------- hierarchy verdict
def test_evaluate_hierarchy_range_verdict():
    # Last H1 bar touches the high edge (append a touch bar as
    # explicit OHLC — _stack builds plain closes, it cannot carry
    # a touch bar).
    h1 = _frame(_triangle(64), spread=0.25)
    h1 = _append_bars(h1, [(4195.0, 4200.0, 4190.0, 4196.0)])
    h4 = h1.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    m15 = h1.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    m15 = m15.dropna()
    m5 = h1.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    m5 = m5.dropna()
    verdict = evaluate_hierarchy(
        {"D1": _flat_d1(), "H4": h4, "H1": h1, "M15": m15, "M5": m5})
    assert verdict["regime"] == "range"
    assert verdict["size_mult"] == 0.5
    assert verdict["range"] is not None
    assert verdict["legs"]["h1_trigger"]["passed"] is True


def test_evaluate_hierarchy_range_no_touch_blocked():
    stack = _stack()
    verdict = evaluate_hierarchy(
        {"D1": stack.frames["D1"], "H4": stack.frames["H4"],
         "H1": stack.frames["H1"], "M15": stack.frames["M15"],
         "M5": stack.frames["M5"]})
    # Last triangle bar sits mid-range (no edge touch).
    assert verdict["regime"] == "range"
    assert verdict["legs"]["h1_trigger"]["passed"] is False
    assert "range_no_edge_touch" in verdict["legs"]["h1_trigger"]["reason"]
