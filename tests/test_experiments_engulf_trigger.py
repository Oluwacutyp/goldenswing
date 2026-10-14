"""Engulfing-trigger experiment: trigger logic, gates, causality."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_df

from goldenswing.experiments.candles import engulfing
from goldenswing.experiments.engulf_trigger import (
    _invalidation,
    _passes_gates,
    _sweeps_in_order,
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


def _df_and_states(n=30, engulf_at=None, sweep_at=10):
    """30 bars; sweep at `sweep_at`; optional bullish engulfing at
    `engulf_at` (bar engulf_at-1 bearish, bar engulf_at bullish)."""
    rows = []
    for i in range(n):
        o, h, l, c = 105.0, 107.0, 103.0, 105.5
        rows.append((o, h, l, c))
    if engulf_at is not None:
        i = engulf_at
        rows[i - 1] = (106.0, 106.5, 103.5, 104.0)   # bearish body 104-106
        rows[i] = (103.5, 107.0, 103.0, 106.5)       # bullish engulfs it
    df = make_df(rows)
    sw = _sweep(index=sweep_at)
    states = []
    for t in range(n):
        if t < sweep_at:
            states.append(_state(t))
        else:
            states.append(_state(
                t, sweep=sw, sweep_age=t - sweep_at,
                pools_below=[_pool(104.0)],            # near close -> at_pool
                pools_above=[_pool(130.0)]))          # room ahead
    return df, states


def _det():
    return GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True))


# ---------------------------------------------------------------- geometry
def test_engulfing_geometry():
    o = np.array([106.0, 103.5])
    h = np.array([106.5, 107.0])
    l = np.array([103.5, 103.0])
    c = np.array([104.0, 106.5])
    assert engulfing(o, h, l, c, 1) == 1
    # same colour -> no signal
    c2 = np.array([104.0, 103.0])
    assert engulfing(o, h, l, c2, 1) == 0
    # bearish engulfing
    o3 = np.array([104.0, 106.5])
    c3 = np.array([106.0, 103.5])
    assert engulfing(o3, h, l, c3, 1) == -1
    # doji body -> no signal
    c4 = np.array([104.0, 104.0])
    assert engulfing(o, h, l, c4, 1) == 0


def test_invalidation_formula():
    sw = _sweep(extreme=100.0)
    assert _invalidation(sw, ATR, 1) == pytest.approx(100.0 - 0.15 * ATR)
    assert _invalidation(sw, ATR, -1) == pytest.approx(100.0 + 0.15 * ATR)


def test_sweeps_deduped_in_order():
    df, states = _df_and_states()
    sws = _sweeps_in_order(states)
    assert [s.index for s in sws] == [10]


# ------------------------------------------------------------- variant A
def test_variant_a_fires_on_confirmation():
    df, states = _df_and_states()
    # confirmation at bar 12: displacement + CHoCH in direction
    states[12] = _state(12, sweep=states[12].last_sweep, sweep_age=2,
                        bull_disp_age=0, choch_age=0,
                        pools_below=[_pool(104.0)], pools_above=[_pool(130.0)])
    hits = scan_variant(df, states, _det(), "A")
    assert len(hits) == 1
    assert hits[0].trigger_bar == 12
    assert hits[0].lag_bars == 2
    assert hits[0].signal.direction == 1


def test_variant_a_silent_without_confirmation():
    df, states = _df_and_states()  # sweep, but no displacement/CHoCH
    hits = scan_variant(df, states, _det(), "A")
    assert hits == []


# ------------------------------------------------------------- variant B
def test_variant_b_fires_on_engulfing_at_level():
    df, states = _df_and_states(engulf_at=13)
    hits = scan_variant(df, states, _det(), "B")
    assert len(hits) == 1
    assert hits[0].trigger_bar == 13
    assert hits[0].signal.direction == 1


def test_variant_b_silent_midrange():
    df, states = _df_and_states(engulf_at=13)
    # strip every level -> midrange; the video's own rule mutes it
    for t in range(10, 30):
        s = states[t]
        states[t] = _state(t, sweep=s.last_sweep, sweep_age=s.sweep_age,
                           pools_above=[_pool(500.0)])  # far away
    hits = scan_variant(df, states, _det(), "B")
    assert hits == []


def test_variant_b_silent_without_engulfing():
    df, states = _df_and_states()  # no engulfing anywhere
    # add confirmation so only the engulfing is missing
    states[12] = _state(12, sweep=states[12].last_sweep, sweep_age=2,
                        bull_disp_age=0, choch_age=0,
                        pools_below=[_pool(104.0)], pools_above=[_pool(130.0)])
    hits = scan_variant(df, states, _det(), "B")
    assert hits == []


# ------------------------------------------------------------- variant C
def test_variant_c_needs_both():
    df, states = _df_and_states(engulf_at=13)
    # engulfing at 13, confirmation only at 15 -> C fires at 15? No:
    # C needs the SAME bar to have both. Confirmation at 13 too.
    states[13] = _state(13, sweep=states[13].last_sweep, sweep_age=3,
                        bull_disp_age=0, choch_age=0,
                        pools_below=[_pool(104.0)], pools_above=[_pool(130.0)])
    hits = scan_variant(df, states, _det(), "C")
    assert len(hits) == 1
    assert hits[0].trigger_bar == 13


def test_variant_c_silent_when_engulfing_unconfirmed():
    df, states = _df_and_states(engulf_at=13)  # engulfing, no confirmation
    hits = scan_variant(df, states, _det(), "C")
    assert hits == []


# ------------------------------------------------------------------ gates
def test_daily_veto_blocks_counter_bias():
    df, states = _df_and_states(engulf_at=13)
    for t in range(10, 30):  # daily bias opposes the long
        s = states[t]
        kw = dict(sweep=s.last_sweep, sweep_age=s.sweep_age, daily_bias=-1,
                  pools_below=[_pool(104.0)], pools_above=[_pool(130.0)])
        if t == 12:  # A would fire here without the veto
            kw.update(bull_disp_age=0, choch_age=0)
        states[t] = _state(t, **kw)
    # no major-daily-level exception (daily_swing_low is None)
    assert scan_variant(df, states, _det(), "B") == []
    assert scan_variant(df, states, _det(), "A") == []


def test_first_trigger_per_sweep_wins():
    df, states = _df_and_states(engulf_at=13)
    states[12] = _state(12, sweep=states[12].last_sweep, sweep_age=2,
                        bull_disp_age=0, choch_age=0,
                        pools_below=[_pool(104.0)], pools_above=[_pool(130.0)])
    # A fires at 12 (confirmation), B fires at 13 (engulfing): each once
    assert [h.trigger_bar for h in scan_variant(df, states, _det(), "A")] == [12]
    assert [h.trigger_bar for h in scan_variant(df, states, _det(), "B")] == [13]


def test_trigger_is_causal_no_future_bars():
    # engulfing uses bars t-1, t only: moving the engulfing later moves
    # the trigger with it, never earlier.
    df, states = _df_and_states(engulf_at=16)
    hits = scan_variant(df, states, _det(), "B")
    assert [h.trigger_bar for h in hits] == [16]
