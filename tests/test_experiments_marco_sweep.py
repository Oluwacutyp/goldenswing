"""Marco sweep-entry experiment: level filter, triggers, Marco exits."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_df

from goldenswing.experiments.marco_sweep import (
    _count_touches,
    _engineered,
    _nearest_qualified_opposing,
    _sweeps_in_order,
    scan_variant,
    simulate_marco,
)
from goldenswing.experiments.engulf_trigger import _sweeps_in_order as _so2
from goldenswing.golden.detector import (DetectorConfig, GoldenSignal,
                                         GoldenSwingDetector)
from goldenswing.reader.state import (LiquidityPool, MarketState,
                                      StructureEvent, SweepEvent)

ATR = 10.0
LEVEL = 101.0          # swept ssl level (below price)
OPP = 130.0            # opposing bsl level (above price)


def _pool(price, side="ssl"):
    return LiquidityPool(price=price, side=side, strength=1, created_at=0)


def _sweep(index=20, direction=1, level=LEVEL, extreme=100.0):
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


def _rows(n=30):
    # base bars touch neither level: low 104 > 101+2.5, high 107 < 130-2.5
    return [(105.0, 107.0, 104.0, 105.5) for _ in range(n)]


def _df_and_states(n=30, sweep_at=20, qualified=True,
                   opposing_qualified=True):
    """Sweep at `sweep_at` (long, ssl level 101).  Touches baked into the
    raw bars: ssl approaches at bars 5-6 (one run) and 12; bsl approaches
    at bars 3-4 (one run) and 15."""
    rows = _rows(n)
    if qualified:
        rows[5] = (102.0, 104.0, 100.0, 102.0)    # ssl touch
        rows[6] = (102.5, 104.0, 100.5, 102.5)    # same run
        rows[12] = (103.0, 105.0, 99.0, 103.0)    # second approach
    if opposing_qualified:
        rows[3] = (129.0, 131.0, 128.0, 129.0)    # bsl touch
        rows[4] = (128.5, 130.5, 127.5, 128.5)    # same run
        rows[15] = (129.5, 132.0, 128.5, 129.5)   # second approach
    df = make_df(rows)
    sw = _sweep(index=sweep_at)
    states = []
    for t in range(n):
        if t < sweep_at:
            states.append(_state(t))
        else:
            states.append(_state(
                t, sweep=sw, sweep_age=t - sweep_at,
                pools_below=[_pool(99.0, "ssl")],
                pools_above=[_pool(OPP, "bsl")]))
    return df, states


def _det():
    return GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True))


# ------------------------------------------------------- touch counting
def _arrays(rows):
    a = np.array(rows, dtype=float)
    return a[:, 1], a[:, 2], a[:, 3]   # high, low, close


def test_count_touches_ssl_runs_deduped():
    rows = _rows(10)
    rows[2] = (102.0, 104.0, 100.0, 102.0)
    rows[3] = (102.5, 104.0, 100.5, 102.5)   # same run as bar 2
    rows[6] = (103.0, 105.0, 99.0, 103.0)
    h, l, c = _arrays(rows)
    assert _count_touches(h, l, c, LEVEL, ATR, 9, "ssl") == 2


def test_count_touches_ignores_body_close_through():
    rows = _rows(10)
    rows[2] = (102.0, 104.0, 100.0, 99.0)    # closes THROUGH the level
    h, l, c = _arrays(rows)
    assert _count_touches(h, l, c, LEVEL, ATR, 9, "ssl") == 0


def test_count_touches_excludes_sweep_bar():
    rows = _rows(10)
    rows[9] = (102.0, 104.0, 100.0, 102.0)   # touch ON the sweep bar
    h, l, c = _arrays(rows)
    assert _count_touches(h, l, c, LEVEL, ATR, 9, "ssl") == 0


def test_count_touches_bsl():
    rows = _rows(10)
    rows[2] = (129.0, 131.0, 128.0, 129.0)
    rows[5] = (129.5, 132.0, 128.5, 129.5)
    h, l, c = _arrays(rows)
    assert _count_touches(h, l, c, OPP, ATR, 9, "bsl") == 2


def test_engineered_threshold():
    rows = _rows(10)
    rows[2] = (102.0, 104.0, 100.0, 102.0)
    h, l, c = _arrays(rows)
    assert _engineered(h, l, c, LEVEL, ATR, 9, "ssl") is False  # 1 touch
    rows[5] = (103.0, 105.0, 99.0, 103.0)
    h, l, c = _arrays(rows)
    assert _engineered(h, l, c, LEVEL, ATR, 9, "ssl") is True   # 2 touches


def test_nearest_qualified_opposing():
    df, states = _df_and_states()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    assert _nearest_qualified_opposing(h, low, c, states[20], 20, 1) \
        == pytest.approx(OPP)


def test_nearest_qualified_opposing_none_when_unqualified():
    df, states = _df_and_states(opposing_qualified=False)
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    assert _nearest_qualified_opposing(h, low, c, states[20], 20, 1) is None


# ---------------------------------------------------------------- variants
def test_variant_b_fires_immediate_on_qualified():
    df, states = _df_and_states()
    hits = scan_variant(df, states, _det(), "B")
    assert len(hits) == 1
    assert hits[0].trigger_bar == 20       # the sweep bar itself
    assert hits[0].lag_bars == 0
    assert hits[0].qualified is True
    assert hits[0].signal.targets == [pytest.approx(OPP)]


def test_variant_b_silent_when_level_unqualified():
    df, states = _df_and_states(qualified=False)
    assert scan_variant(df, states, _det(), "B") == []


def test_variant_b_silent_without_opposing_target():
    df, states = _df_and_states(opposing_qualified=False)
    assert scan_variant(df, states, _det(), "B") == []


def test_variant_c_fires_on_unqualified_level():
    df, states = _df_and_states(qualified=False)
    hits = scan_variant(df, states, _det(), "C")
    assert len(hits) == 1
    assert hits[0].trigger_bar == 20
    assert hits[0].qualified is False
    # C targets the nearest opposing POOL (no engineered filter)
    assert hits[0].signal.targets == [pytest.approx(OPP)]


def test_variant_a_fires_on_confirmation_of_qualified():
    df, states = _df_and_states()
    states[22] = _state(22, sweep=states[22].last_sweep, sweep_age=2,
                        bull_disp_age=0, choch_age=0,
                        pools_below=[_pool(99.0, "ssl")],
                        pools_above=[_pool(OPP, "bsl")])
    hits = scan_variant(df, states, _det(), "A")
    assert len(hits) == 1
    assert hits[0].trigger_bar == 22
    assert hits[0].qualified is True


def test_variant_a_silent_without_confirmation():
    df, states = _df_and_states()   # qualified, but no displacement/CHoCH
    assert scan_variant(df, states, _det(), "A") == []


def test_variant_a_silent_when_level_unqualified():
    df, states = _df_and_states(qualified=False)
    states[22] = _state(22, sweep=states[22].last_sweep, sweep_age=2,
                        bull_disp_age=0, choch_age=0,
                        pools_below=[_pool(99.0, "ssl")],
                        pools_above=[_pool(OPP, "bsl")])
    assert scan_variant(df, states, _det(), "A") == []


def test_daily_veto_blocks_marco_variants():
    df, states = _df_and_states()
    for t in range(20, 30):
        s = states[t]
        states[t] = _state(t, sweep=s.last_sweep, sweep_age=s.sweep_age,
                           daily_bias=-1, pools_below=[_pool(99.0, "ssl")],
                           pools_above=[_pool(OPP, "bsl")])
    assert scan_variant(df, states, _det(), "B") == []
    assert scan_variant(df, states, _det(), "C") == []


# ---------------------------------------------------------- marco exits
def _sig(t=5, target=110.0, inv=100.0, atr=ATR, direction=1):
    return GoldenSignal(index=t,
                        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
                        direction=direction, score=100.0,
                        kind="sweep_reversal", invalidation=inv,
                        targets=[target], atr=atr, entry_ref=105.0)


def _sim_df(rows):
    return make_df(rows)


def test_marco_target_hit():
    # entry = open[6] = 105; stop0 = 100 - 2.5 = 97.5; risk = 7.5
    rows = _rows(10)
    rows[6] = (105.0, 106.0, 104.0, 105.0)
    rows[7] = (105.5, 112.0, 105.0, 111.0)   # tags the 110 target
    oc = simulate_marco(_sim_df(rows), _sig())
    assert oc.exit_reason == "target"
    assert oc.r_net == pytest.approx((110.0 - 105.0) / 7.5 - 0.45 / 7.5)


def test_marco_stop_hit():
    rows = _rows(10)
    rows[6] = (105.0, 106.0, 104.0, 105.0)
    rows[7] = (104.5, 105.0, 96.0, 97.0)     # through stop0 = 97.5
    oc = simulate_marco(_sim_df(rows), _sig())
    assert oc.exit_reason == "stop"
    assert oc.r_net == pytest.approx((97.5 - 105.0) / 7.5 - 0.45 / 7.5)


def test_marco_be_roll():
    # +1R = 112.5 prints on bar 7 -> BE armed; bar 8 fades to entry.
    rows = _rows(10)
    rows[6] = (105.0, 106.0, 104.0, 105.0)
    rows[7] = (106.0, 113.0, 105.5, 112.0)
    rows[8] = (106.0, 107.0, 104.0, 106.5)   # low 104 <= entry 105
    oc = simulate_marco(_sim_df(rows), _sig(target=120.0))
    assert oc.exit_reason == "be"
    assert oc.exit_index == 8
    assert oc.r_net == pytest.approx(0.0 - 0.45 / 7.5)


def test_marco_be_armed_from_completed_bars_only():
    # bar 7 prints +1R AND dips below entry in the same bar: no exit
    # yet (BE not armed intrabar); bar 8 is clean; bar 9 fades -> "be".
    rows = _rows(12)
    rows[6] = (105.0, 106.0, 104.0, 105.0)
    rows[7] = (106.0, 113.0, 104.0, 112.0)   # +1R prints, low dips too
    rows[8] = (106.0, 107.0, 105.5, 106.0)   # clean
    rows[9] = (106.0, 107.0, 104.5, 106.0)   # fades to entry
    oc = simulate_marco(_sim_df(rows), _sig(target=120.0))
    assert oc.exit_reason == "be"
    assert oc.exit_index == 9


def test_marco_timeout():
    rows = _rows(10)
    rows[6] = (105.0, 106.0, 104.0, 105.0)
    oc = simulate_marco(_sim_df(rows), _sig(), horizon=3)
    assert oc.exit_reason == "timeout"
    assert oc.exit_index == 8


def test_marco_no_target_behind_entry():
    sig = _sig(target=100.0)   # target below entry for a long
    rows = _rows(10)
    assert simulate_marco(_sim_df(rows), sig) is None
