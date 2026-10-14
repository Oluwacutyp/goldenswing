"""Phase 2: sweep confirmation, daily-bias veto, swing exits."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_df
from goldenswing.backtest.harness import ExitConfig, simulate_signal_v2
from goldenswing.golden.detector import DetectorConfig, GoldenSignal, GoldenSwingDetector
from goldenswing.reader import MarketReader
from goldenswing.reader.state import (
    LiquidityPool,
    MarketState,
    StructureEvent,
    SweepEvent,
    SwingPoint,
)

P2 = DetectorConfig(require_sweep_confirmation=True, daily_veto=True)


def _state(**over) -> MarketState:
    base = dict(
        index=100, timestamp="t", close=2000.0, atr=10.0,
        trend="down", bias=-1,
        pools_above=[LiquidityPool(price=2060.0, side="bsl", strength=1,
                                   created_at=50)],
        pools_below=[LiquidityPool(price=1940.0, side="ssl", strength=1,
                                   created_at=50)],
    )
    base.update(over)
    return MarketState(**base)


def _long_sweep_state(**over) -> MarketState:
    base = dict(
        last_sweep=SweepEvent(index=98, level=1990.0, extreme=1984.0,
                              direction=+1),
        sweep_age=2, bull_disp_3=1.2,
    )
    base.update(over)
    return _state(**base)


# ------------------------------------------------------------- confirmation
def test_unconfirmed_sweep_is_not_tradeable():
    s = _long_sweep_state()  # no displacement, no structure shift
    assert GoldenSwingDetector(P2).evaluate(s) is None


def test_displacement_without_structure_shift_is_not_enough():
    s = _long_sweep_state(bull_disp_age=1)
    assert GoldenSwingDetector(P2).evaluate(s) is None


def test_structure_shift_without_displacement_is_not_enough():
    ev = StructureEvent(index=99, kind="CHOCH", direction=+1, level=2005.0)
    s = _long_sweep_state(last_structure_event=ev)
    assert GoldenSwingDetector(P2).evaluate(s) is None


def test_confirmed_sweep_fires():
    ev = StructureEvent(index=99, kind="CHOCH", direction=+1, level=2005.0)
    s = _long_sweep_state(bull_disp_age=1, last_structure_event=ev)
    sig = GoldenSwingDetector(P2).evaluate(s)
    assert sig is not None
    assert sig.kind == "sweep_reversal"
    assert sig.direction == 1
    assert sig.sweep_index == 98
    assert sig.invalidation == pytest.approx(1984.0 - 0.15 * 10.0)


def test_confirmed_via_displacement_fvg_and_trend_flip():
    s = _long_sweep_state(bull_disp_fvg_age=0, trend="up", bias=1,
                          trend_since=99)
    sig = GoldenSwingDetector(P2).evaluate(s)
    assert sig is not None and sig.kind == "sweep_reversal"


def test_confirmation_after_window_expires():
    ev = StructureEvent(index=95, kind="CHOCH", direction=+1, level=2005.0)
    s = _long_sweep_state(
        last_sweep=SweepEvent(index=91, level=1990.0, extreme=1984.0,
                              direction=+1),
        sweep_age=9, bull_disp_age=3, last_structure_event=ev)
    assert GoldenSwingDetector(P2).evaluate(s) is None


def test_phase1_config_still_takes_unconfirmed_sweep():
    s = _long_sweep_state()
    sig = GoldenSwingDetector(DetectorConfig()).evaluate(s)
    assert sig is not None and sig.kind == "sweep_reversal"


# --------------------------------------------------------------- daily veto
def _long_continuation_state(**over) -> MarketState:
    base = dict(
        trend="up", bias=1, bull_disp_3=1.0,
        recent_swing_lows=[SwingPoint(price=1990.0, index=90,
                                      confirmed_at=92, kind="low")],
    )
    base.update(over)
    return _state(**base)


def test_daily_veto_blocks_counter_bias_continuation():
    s = _long_continuation_state(daily_bias=-1, daily_trend="down")
    assert GoldenSwingDetector(P2).evaluate(s) is None


def test_no_veto_when_disabled_or_aligned():
    s = _long_continuation_state(daily_bias=-1, daily_trend="down")
    cfg = DetectorConfig(require_sweep_confirmation=True, daily_veto=False)
    assert GoldenSwingDetector(cfg).evaluate(s) is not None
    s2 = _long_continuation_state(daily_bias=1, daily_trend="up")
    assert GoldenSwingDetector(P2).evaluate(s2) is not None


def _short_sweep_at_daily_high(daily_high: float) -> MarketState:
    ev = StructureEvent(index=99, kind="CHOCH", direction=-1, level=1995.0)
    return _state(
        trend="up", bias=1, bear_disp_3=1.2, bear_disp_age=1,
        last_structure_event=ev,
        last_sweep=SweepEvent(index=98, level=2010.0, extreme=2016.0,
                              direction=-1),
        sweep_age=2, daily_bias=1, daily_trend="up",
        daily_swing_high=daily_high,
    )


def test_veto_exception_confirmed_sweep_at_daily_level():
    sig = GoldenSwingDetector(P2).evaluate(_short_sweep_at_daily_high(2012.0))
    assert sig is not None and sig.kind == "sweep_reversal"


def test_veto_blocks_counter_bias_sweep_away_from_daily_level():
    s = _short_sweep_at_daily_high(2100.0)
    assert GoldenSwingDetector(P2).evaluate(s) is None


# ---------------------------------------------------------------- daily bias
def _daily_frame(day_closes, bars_per_day=24):
    rows = []
    prev = day_closes[0]
    for c in day_closes:
        for _ in range(bars_per_day):
            rows.append((prev, c + 1.0, c - 1.0, c))
            prev = c
    return make_df(rows, start="2024-01-01")


def _trend_days(n=90, direction=1):
    # 3 days up, 3 days pullback — each cycle's peak/trough exceeds the
    # last, so daily fractal swings form, confirm, and get broken.
    closes, v = [], 2000.0
    pat = [6.0, 6.0, 6.0, -2.0, -2.0, -2.0]
    for d in range(n):
        v += direction * pat[d % len(pat)]
        closes.append(v)
    return closes


def test_daily_bias_uptrend_and_downtrend():
    up = MarketReader().read(_daily_frame(_trend_days(direction=1)))
    assert up[-1].daily_bias == 1
    assert up[-1].daily_trend == "up"
    assert up[-1].daily_swing_high is not None
    down = MarketReader().read(_daily_frame(_trend_days(direction=-1)))
    assert down[-1].daily_bias == -1
    assert down[-1].daily_trend == "down"
    assert down[-1].daily_swing_low is not None


def test_daily_bias_warmup_is_neutral():
    states = MarketReader().read(_daily_frame(_trend_days(n=30)))
    assert states[30].daily_bias == 0  # inside day 1: no completed days


# --------------------------------------------------------------- swing exits
def _sig(**over) -> GoldenSignal:
    base = dict(index=0, timestamp="t", direction=1, score=80.0,
                kind="continuation", entry_ref=2000.0,
                invalidation=1990.0, targets=[2025.0], atr=10.0)
    base.update(over)
    return GoldenSignal(**base)


def test_partial_then_trail():
    df = make_df([
        (1999.0, 2001.0, 1998.0, 2000.0),
        (2000.0, 2026.0, 1995.0, 2024.0),   # partial fills @2025; trail set
        (2024.0, 2030.0, 2020.0, 2028.0),   # trail rises to 2017.5
        (2028.0, 2029.0, 2016.0, 2020.0),   # trail exit @2017.5
    ])
    o = simulate_signal_v2(df, _sig())
    assert o is not None
    assert o.partial_taken and o.exit_reason == "trail"
    # 0.5 * (25/12.5) + 0.5 * (17.5/12.5) - 0.45/12.5
    assert o.r_net == pytest.approx(1.0 + 0.7 - 0.036)
    assert o.hold_bars == 3


def test_buffer_absorbs_wick_through_invalidation():
    df = make_df([
        (1999.0, 2001.0, 1998.0, 2000.0),
        (2000.0, 2010.0, 1989.0, 2005.0),   # wick < inv 1990, > stop 1987.5
        (2005.0, 2015.0, 2000.0, 2010.0),
        (2010.0, 2016.0, 2004.0, 2012.0),
        (2012.0, 2017.0, 2006.0, 2014.0),
    ])
    o = simulate_signal_v2(df, _sig(), ExitConfig(horizon=4))
    assert o is not None
    assert o.wicked_inv and o.exit_reason == "timeout"
    assert o.r_net == pytest.approx(14.0 / 12.5 - 0.45 / 12.5)


def test_buffered_stop_still_stops():
    df = make_df([
        (1999.0, 2001.0, 1998.0, 2000.0),
        (2000.0, 2005.0, 1987.0, 1995.0),   # low through buffered stop
        (1995.0, 2000.0, 1990.0, 1998.0),
    ])
    o = simulate_signal_v2(df, _sig())
    assert o is not None
    assert o.exit_reason == "stop"
    assert o.r_net == pytest.approx(-1.0 - 0.45 / 12.5)
