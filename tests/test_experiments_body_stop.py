"""Body-anchored stops experiment: anchor geometry, gate behavior, causality."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_df

from goldenswing.backtest.harness import ExitConfig, simulate_signal_v2
from goldenswing.experiments.body_stop import (
    body_anchor,
    body_invalidation,
    scan_variant,
    variant_exits,
    variant_invalidation,
)
from goldenswing.experiments.engulf_trigger import _invalidation
from goldenswing.golden.detector import (DetectorConfig, GoldenSignal,
                                         GoldenSwingDetector)
from goldenswing.reader.state import (LiquidityPool, MarketState,
                                      StructureEvent, SweepEvent)

ATR = 10.0


def _pool(price, side="ssl"):
    return LiquidityPool(price=price, side=side, strength=1, created_at=0)


def _sweep(index=10, direction=1, level=101.0, extreme=100.0):
    return SweepEvent(index=index, level=level, extreme=extreme,
                      direction=direction)


def _state(t, sweep=None, sweep_age=None, bull_disp_age=None,
           choch_age=None, close=106.0, daily_bias=1,
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


def _df_and_states(n=30, sweep_at=10, trigger_at=12):
    """30 bars; bullish sweep at `sweep_at` (wick low 100, body 104-106);
    confirmation at `trigger_at` (close 106)."""
    rows = []
    for i in range(n):
        rows.append((105.0, 107.0, 103.0, 106.0))
    rows[sweep_at] = (106.0, 108.0, 100.0, 104.0)  # sweep candle
    df = make_df(rows)
    sw = _sweep(index=sweep_at)
    states = []
    for t in range(n):
        if t < sweep_at:
            states.append(_state(t))
        else:
            states.append(_state(
                t, sweep=sw, sweep_age=t - sweep_at,
                bull_disp_age=0 if t == trigger_at else None,
                choch_age=0 if t == trigger_at else None,
                pools_below=[_pool(99.0)],
                pools_above=[_pool(130.0)]))
    return df, states


def _det():
    return GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True))


# ------------------------------------------------------------- geometry
def test_body_anchor_bullish_is_body_bottom():
    o = np.array([106.0])
    c = np.array([104.0])
    assert body_anchor(o, c, 0, 1) == 104.0


def test_body_anchor_bearish_is_body_top():
    o = np.array([104.0])
    c = np.array([106.0])
    assert body_anchor(o, c, 0, -1) == 106.0


def test_body_anchor_ignores_wick():
    # Long red sweep candle: wick far below, body bottom is the anchor.
    o = np.array([110.0])
    c = np.array([105.0])
    assert body_anchor(o, c, 0, 1) == 105.0


def test_body_invalidation_tighter_than_wick():
    o = np.array([106.0])
    c = np.array([104.0])
    sw = _sweep(index=0, direction=1, extreme=100.0)
    a = _invalidation(sw, ATR, 1)          # 100 - 1.5 = 98.5
    b = body_invalidation(o, c, sw, ATR, 1)  # 104 - 1.5 = 102.5
    assert a == pytest.approx(98.5)
    assert b == pytest.approx(102.5)
    assert b > a  # tighter: closer to price for a long


def test_body_invalidation_bearish():
    o = np.array([104.0])
    c = np.array([106.0])
    sw = _sweep(index=0, direction=-1, extreme=112.0)
    a = _invalidation(sw, ATR, -1)          # 112 + 1.5 = 113.5
    b = body_invalidation(o, c, sw, ATR, -1)  # 106 + 1.5 = 107.5
    assert b < a  # tighter: closer to price for a short


def test_b2_invalidation_is_pure_body_extreme():
    o = np.array([106.0])
    c = np.array([104.0])
    sw = _sweep(index=0, direction=1, extreme=100.0)
    assert variant_invalidation(o, c, sw, ATR, 1, "B2") == \
        pytest.approx(104.0)
    assert variant_invalidation(o, c, sw, ATR, 1, "A") == \
        pytest.approx(98.5)
    assert variant_invalidation(o, c, sw, ATR, 1, "B") == \
        pytest.approx(102.5)


def test_variant_exits_b2_has_zero_buffer():
    assert variant_exits("B2").stop_buffer_atr == 0.0
    assert variant_exits("A").stop_buffer_atr == 0.25
    assert variant_exits("B").stop_buffer_atr == 0.25


# ------------------------------------------------------------- scan
def test_scan_a_and_b_fire_same_trigger_bar():
    df, states = _df_and_states()
    det = _det()
    ha = scan_variant(df, states, det, "A")
    hb = scan_variant(df, states, det, "B")
    assert len(ha) == 1 and len(hb) == 1
    assert ha[0].trigger_bar == hb[0].trigger_bar == 12
    # Same trigger, different invalidation: B tighter than A.
    assert hb[0].signal.invalidation > ha[0].signal.invalidation
    assert ha[0].signal.invalidation == pytest.approx(98.5)
    assert hb[0].signal.invalidation == pytest.approx(102.5)


def test_b2_gate_rejects_too_tight_stop():
    # Body bottom (104) is only 0.2 ATR from the 106 close -> below the
    # detector's 0.25 min stop band: B2's gate honestly filters it.
    df, states = _df_and_states()
    det = _det()
    hb2 = scan_variant(df, states, det, "B2")
    assert hb2 == []


# ------------------------------------------------------------- simulation
def _signal_for_sim(inv, atr=ATR):
    return GoldenSignal(
        index=8, timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        direction=1, score=100.0, kind="sweep_reversal",
        invalidation=float(inv), targets=[130.0], atr=float(atr),
        entry_ref=105.0)


def test_b2_stopped_where_a_survives():
    """Bar dips to 103.5: below the 104 body anchor, above A's 95 stop."""
    rows = [(105.0, 107.0, 103.0, 106.0)] * 30
    rows[9] = (105.0, 106.0, 104.5, 105.5)
    rows[10] = (105.5, 106.5, 103.5, 106.0)  # tags the body, not the extreme
    rows[11] = (106.0, 131.0, 105.5, 130.0)  # rallies to the pool
    df = make_df(rows)
    oc_a = simulate_signal_v2(df, _signal_for_sim(98.5), ExitConfig())
    oc_b2 = simulate_signal_v2(
        df, _signal_for_sim(104.0), ExitConfig(stop_buffer_atr=0.0))
    assert oc_b2 is not None and oc_a is not None
    assert oc_b2.exit_reason == "stop"
    assert oc_a.exit_reason != "stop"


def test_buffered_body_b_wider_than_b2():
    """B keeps the 0.25 ATR harness buffer on top of the body anchor."""
    rows = [(105.0, 107.0, 103.0, 106.0)] * 30
    rows[9] = (105.0, 106.0, 104.5, 105.5)
    rows[10] = (105.5, 106.5, 99.5, 106.0)  # below B's 100.0 stop
    df = make_df(rows)
    # B: inv 102.5, buffer 2.5 -> stop 100.0 ; A: inv 98.5 -> stop 96.0
    oc_b = simulate_signal_v2(df, _signal_for_sim(102.5), ExitConfig())
    oc_a = simulate_signal_v2(df, _signal_for_sim(98.5), ExitConfig())
    assert oc_b.exit_reason == "stop"
    assert oc_a.exit_reason != "stop"
