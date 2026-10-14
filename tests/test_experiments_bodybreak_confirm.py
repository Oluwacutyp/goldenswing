"""Body-break confirmation experiment: trigger geometry, gates, causality."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_df

from goldenswing.experiments.bodybreak_confirm import (
    _momentum_ok,
    body_break_trigger,
    scan_variant,
)
from goldenswing.golden.detector import DetectorConfig, GoldenSwingDetector
from goldenswing.reader.state import (LiquidityPool, MarketState,
                                      StructureEvent, SweepEvent)

ATR = 10.0


def _pool(price, side="ssl"):
    return LiquidityPool(price=price, side=side, strength=1, created_at=0)


def _sweep(index=10, direction=1, level=101.0, extreme=100.0):
    return SweepEvent(index=index, level=level, extreme=extreme,
                      direction=direction)


def _state(t, sweep=None, sweep_age=None, bull_disp_age=None,
           choch_age=None, close=105.0, daily_bias=1,
           pools_below=(), pools_above=()):
    ev = None
    if choch_age is not None:
        ev = StructureEvent(index=t - choch_age, kind="CHOCH", direction=1,
                            level=110.0)
    return MarketState(
        index=t, timestamp=pd.Timestamp("2024-01-01", tz="UTC") +
        pd.Timedelta(hours=t), close=close, atr=ATR,
        trend="up", bias=1, daily_bias=daily_bias,
        last_structure_event=ev,
        structure_event_age=choch_age,
        last_sweep=sweep, sweep_age=sweep_age,
        bull_disp_age=bull_disp_age,
        pools_below=list(pools_below), pools_above=list(pools_above))


def _df_and_states(n=30, sweep_at=10, trigger_at=None):
    """30 bars; sweep at `sweep_at`; optional bullish body-break at
    `trigger_at` (sweep candle body 104-106; trigger bar closes 106.5)."""
    rows = []
    for i in range(n):
        rows.append((105.0, 107.0, 103.0, 105.5))
    rows[sweep_at] = (106.0, 108.0, 99.0, 104.0)  # sweep candle, body 104-106
    if trigger_at is not None:
        rows[trigger_at] = (105.0, 107.0, 104.5, 106.5)  # close > body top
    df = make_df(rows)
    sw = _sweep(index=sweep_at)
    states = []
    for t in range(n):
        if t < sweep_at:
            states.append(_state(t))
        else:
            states.append(_state(
                t, sweep=sw, sweep_age=t - sweep_at,
                pools_below=[_pool(104.0)],
                pools_above=[_pool(130.0)]))
    return df, states


def _det():
    return GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True))


# ------------------------------------------------------------- geometry
def test_body_break_geometry_bullish():
    o = np.array([106.0, 105.0, 104.0])
    c = np.array([104.0, 105.5, 106.5])
    # sweep candle body: 104..106
    assert body_break_trigger(o, c, 0, 1, 1) is False   # 105.5 < 106
    assert body_break_trigger(o, c, 0, 2, 1) is True    # 106.5 > 106


def test_body_break_geometry_bearish():
    o = np.array([104.0, 105.0, 106.0])
    c = np.array([106.0, 104.5, 103.5])
    # sweep candle body: 104..106
    assert body_break_trigger(o, c, 0, 1, -1) is False  # 104.5 > 104
    assert body_break_trigger(o, c, 0, 2, -1) is True    # 103.5 < 104


def test_body_break_boundary_is_strict():
    o = np.array([106.0, 105.0])
    c = np.array([104.0, 106.0])
    # close exactly AT the body top is not a break through it
    assert body_break_trigger(o, c, 0, 1, 1) is False


def test_body_break_uses_sweep_bar_only():
    # bars between sweep and t do not matter; only the sweep bar's body
    o = np.array([110.0, 120.0, 105.0, 106.5])
    c = np.array([100.0, 100.0, 104.0, 106.5])
    # sweep body 100..110; bar 2's close 104 does not break it
    assert body_break_trigger(o, c, 0, 2, 1) is False
    assert body_break_trigger(o, c, 0, 3, 1) is False   # 106.5 < 110


# ------------------------------------------------------------- momentum
def test_momentum_gate():
    o = np.array([100.0, 100.0])
    h = np.array([122.0, 122.0])
    low = np.array([98.0, 98.0])
    c = np.array([120.0, 105.0])
    assert _momentum_ok(o, h, low, c, 0) is True    # body 20 > 0.5*24
    assert _momentum_ok(o, h, low, c, 1) is False   # body 5 < 0.5*24


def test_momentum_gate_zero_range():
    o = np.array([100.0])
    h = np.array([100.0])
    low = np.array([100.0])
    c = np.array([100.0])
    assert _momentum_ok(o, h, low, c, 0) is False


# ------------------------------------------------------------- scanning
def test_b10_fires_on_body_break():
    df, states = _df_and_states(trigger_at=13)
    hits = scan_variant(df, states, _det(), "B10")
    assert len(hits) == 1
    assert hits[0].trigger_bar == 13
    assert hits[0].lag_bars == 3


def test_b5_window_expires():
    df, states = _df_and_states(trigger_at=16)  # lag 6 > 5-bar window
    assert scan_variant(df, states, _det(), "B5") == []
    assert len(scan_variant(df, states, _det(), "B10")) == 1


def test_no_trigger_without_body_break():
    df, states = _df_and_states(trigger_at=None)
    assert scan_variant(df, states, _det(), "B5") == []
    assert scan_variant(df, states, _det(), "B10") == []
    assert scan_variant(df, states, _det(), "C") == []


def test_first_trigger_per_sweep_dedupe():
    df, states = _df_and_states(trigger_at=12)
    # force a second body-break later too
    rows_df = df.copy()
    rows_df.iloc[14, rows_df.columns.get_loc("close")] = 107.0
    hits = scan_variant(rows_df, states, _det(), "B10")
    assert len(hits) == 1
    assert hits[0].trigger_bar == 12


def test_c_requires_momentum():
    df, states = _df_and_states(n=30, trigger_at=13)
    # trigger bar: body 1.5 on range 2.5 -> momentum ok (1.5 > 1.25)
    hits = scan_variant(df, states, _det(), "C")
    assert len(hits) == 1
    # now make the trigger candle a weak-drift doji: same close below body
    # top is impossible, so instead shrink the body below half the range
    rows = df.copy()
    i = rows.columns.get_loc("close")
    rows.iloc[13, i] = 106.1  # still > 106 body top, body 1.1, range 2.5
    hits2 = scan_variant(rows, states, _det(), "C")
    assert hits2 == []
    # B10 still fires on the same bar (no momentum gate)
    assert len(scan_variant(rows, states, _det(), "B10")) == 1


def test_c_min_stop_gate():
    # stop < 0.5 ATR kills C but B10 fires: sweep extreme very close to close
    df, states = _df_and_states(trigger_at=13)
    det = _det()
    hits_b = scan_variant(df, states, det, "B10")
    assert len(hits_b) == 1
    # invalidation = extreme - 0.15*ATR - ... ; stop distance here is
    # |close - inv|/atr with close=106.5, extreme=100, atr=10 -> ~0.8 ATR,
    # above the 0.5 floor, so C fires too in this fixture.
    hits_c = scan_variant(df, states, det, "C")
    assert len(hits_c) == 1


def test_variant_a_unchanged_anchor():
    # A uses the detector's _sweep_confirmed: no displacement/CHoCH in the
    # fixture -> no fire, exactly like the engulf_trigger experiment.
    df, states = _df_and_states(trigger_at=13)
    assert scan_variant(df, states, _det(), "A") == []


def test_trigger_needs_t_plus_1():
    # sweep at the last usable bar: no t+1 for entry -> no hits
    df, states = _df_and_states(n=12, sweep_at=10, trigger_at=11)
    # hi = min(10+10, 12-2) = 10 -> loop range(10, 11) -> t=10 only;
    # bar 10 is the sweep candle itself, close 104 < body top 106
    assert scan_variant(df, states, _det(), "B10") == []
