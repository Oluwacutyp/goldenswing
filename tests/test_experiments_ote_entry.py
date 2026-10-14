"""OTE as the entry zone: fib geometry, setup gating, fill mechanics."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_df

from goldenswing.backtest.harness import ExitConfig
from goldenswing.experiments.ote_entry import (
    _fill_price,
    _obs_active_at,
    _ote_zone,
    find_fill,
    scan_setups,
    simulate_ote_trade,
    OTESetup,
    OTEFill,
)
from goldenswing.golden.detector import GoldenSignal
from goldenswing.reader.state import MarketState, OrderBlock, SweepEvent


# ------------------------------------------------------------------ helpers
def _ob(direction=1, top=112.0, bottom=108.0, created_at=95):
    return OrderBlock(top=top, bottom=bottom, direction=direction,
                      created_at=created_at, candle_index=94)


def _sweep(direction=1, index=90, level=100.0, extreme=98.0):
    return SweepEvent(index=index, level=level, extreme=extreme,
                      direction=direction)


def _state(t, sweep=None, obs=(), atr=10.0, close=112.0):
    return MarketState(
        index=t, timestamp=None, close=close, atr=atr, trend="up", bias=1,
        last_sweep=sweep, sweep_age=(t - sweep.index) if sweep else None,
        active_obs=list(obs),
        pools_above=[], pools_below=[],
    )


def _sig(t, direction=1, score=70.0, sweep_index=90, targets=(130.0, 140.0)):
    return GoldenSignal(index=t, timestamp=None, direction=direction,
                        score=score, kind="sweep_reversal",
                        entry_ref=112.0, invalidation=97.0,
                        targets=list(targets), atr=10.0,
                        sweep_index=sweep_index)


def _df(n=140, base=100.0):
    # gentle uptrend: sweep low at bar 90, displacement high at bar 100
    rows = []
    for i in range(n):
        if i == 90:
            rows.append((102, 103, 98, 102))     # sweep bar (wick to 98)
        elif i == 100:
            rows.append((112, 122, 111, 120))   # displacement extreme 122
        else:
            c = base + i * 0.15
            rows.append((c - 0.2, c + 0.5, c - 0.5, c))
    return make_df(rows)


# ---------------------------------------------------------------- fib zone
def test_ote_zone_long():
    lo, hi = _ote_zone(1, 100.0, 120.0)
    assert lo == pytest.approx(120.0 - 0.79 * 20.0)
    assert hi == pytest.approx(120.0 - 0.62 * 20.0)
    assert lo < hi


def test_ote_zone_short():
    lo, hi = _ote_zone(-1, 100.0, 120.0)
    assert lo == pytest.approx(100.0 + 0.62 * 20.0)
    assert hi == pytest.approx(100.0 + 0.79 * 20.0)
    assert lo < hi


def test_fill_price_gap_through():
    # long limit 105, bar opens below it -> adverse open fill
    assert _fill_price(1, 105.0, 103.0) == 103.0
    assert _fill_price(1, 105.0, 106.0) == 105.0
    assert _fill_price(-1, 105.0, 107.0) == 107.0
    assert _fill_price(-1, 105.0, 104.0) == 105.0


# ------------------------------------------------------------------- scan
def test_scan_builds_setup_with_ob_confluence():
    df = _df()
    # OB overlapping the OTE zone: leg 98->122, rng 24 ->
    # zone = [122-0.79*24, 122-0.62*24] = [103.04, 107.12]
    ob = _ob(direction=1, top=110.0, bottom=106.0, created_at=95)
    states = [_state(i) for i in range(140)]
    states[110] = _state(110, sweep=_sweep(), obs=(ob,), close=115.0)
    setups = scan_setups(df, states, [_sig(110)])
    assert len(setups) == 1
    s = setups[0]
    assert s.direction == 1
    assert s.zone_lo == pytest.approx(103.04, abs=0.6)
    assert s.zone_hi == pytest.approx(107.12, abs=0.6)
    assert s.stop == pytest.approx(98.0 - 2.5)  # 0.25 * atr(10)


def test_scan_skips_without_ob_overlap():
    df = _df()
    ob = _ob(direction=1, top=130.0, bottom=128.0, created_at=95)  # far away
    states = [_state(i) for i in range(140)]
    states[110] = _state(110, sweep=_sweep(), obs=(ob,), close=115.0)
    assert scan_setups(df, states, [_sig(110)]) == []


def test_scan_skips_stale_zone():
    # price already inside the zone -> stale, no pullback left
    df = _df()
    df.iloc[110] = [106.5, 107.5, 106.0, 107.0, 1000.0]  # close inside zone
    ob = _ob(direction=1, top=110.0, bottom=106.0, created_at=95)
    states = [_state(i) for i in range(140)]
    states[110] = _state(110, sweep=_sweep(), obs=(ob,), close=107.0)
    assert scan_setups(df, states, [_sig(110)]) == []


def test_scan_skips_wrong_sweep_direction():
    df = _df()
    ob = _ob(direction=1, top=110.0, bottom=106.0, created_at=95)
    states = [_state(i) for i in range(140)]
    states[110] = _state(110, sweep=_sweep(direction=-1), obs=(ob,),
                         close=115.0)
    assert scan_setups(df, states, [_sig(110)]) == []


def test_scan_ignores_non_sweep_signals():
    df = _df()
    sig = _sig(110)
    sig.kind = "continuation"
    states = [_state(i) for i in range(140)]
    assert scan_setups(df, states, [sig]) == []


# ------------------------------------------------------------------- fill
def _setup(t=110, direction=1, limit_fib=0.70):
    rng = 24.0
    leg_hi, leg_lo = 122.0, 98.0
    limit = leg_hi - limit_fib * rng if direction == 1 \
        else leg_lo + limit_fib * rng
    zone_lo, zone_hi = _ote_zone(direction, leg_lo, leg_hi)
    stop = 98.0 - 2.5 if direction == 1 else 122.0 + 2.5
    return OTESetup(index=t, direction=direction, zone_lo=zone_lo,
                    zone_hi=zone_hi, stop=stop, atr=10.0, sweep_index=90,
                    sweep_extreme=98.0, leg_lo=leg_lo, leg_hi=leg_hi,
                    targets=[140.0], score=70.0, ob_top=110.0,
                    ob_bottom=106.0), limit


def test_fill_on_wick_touch():
    df = _df()
    setup, limit = _setup()
    # bar 111 dips to the limit
    df.iloc[111] = [114, 116, limit - 0.1, 115, 1000.0]
    fl = find_fill(df, setup, 0.70)
    assert fl is not None
    assert fl.fill_index == 111
    assert fl.entry == pytest.approx(limit)


def test_fill_expires():
    df = _df()  # uptrend never pulls back to the zone
    setup, _ = _setup()
    assert find_fill(df, setup, 0.70) is None


def test_fill_cancelled_on_zone_violation():
    df = _df()
    setup, _ = _setup()
    # bar 111 closes through the far edge (below zone_lo for longs)
    df.iloc[111] = [114, 116, 100.0, 100.0, 1000.0]
    assert find_fill(df, setup, 0.70) is None


def test_fill_same_bar_stop():
    df = _df()
    setup, limit = _setup()
    # bar 111 trades through both limit and stop
    df.iloc[111] = [114, 116, 90.0, 115, 1000.0]
    fl = find_fill(df, setup, 0.70)
    assert fl is not None and fl.same_bar_stop


def test_fill_rejects_insane_risk():
    df = _df()
    setup, _ = _setup()
    setup.stop = setup.stop - 100.0  # absurdly wide
    df.iloc[111] = [114, 116, 100.0, 115, 1000.0]
    assert find_fill(df, setup, 0.70) is None


# --------------------------------------------------------------- simulate
def test_simulate_stop_first():
    df = _df()
    setup, limit = _setup()
    df.iloc[111] = [114, 116, limit - 0.1, 115, 1000.0]
    fl = find_fill(df, setup, 0.70)
    assert fl is not None
    # force an immediate stop: drop every later bar through the stop
    for k in range(112, 140):
        df.iloc[k] = [100, 101, 90.0, 95, 1000.0]
    oc = simulate_ote_trade(df, fl, ExitConfig())
    assert oc.exit_reason == "stop"
    assert oc.r_net < 0


def test_simulate_partial_and_trail():
    df = _df()
    setup, limit = _setup()
    df.iloc[111] = [114, 116, limit - 0.1, 115, 1000.0]
    fl = find_fill(df, setup, 0.70)
    assert fl is not None
    # rally through the 140 pool, then keep running
    for k in range(112, 140):
        c = 120 + (k - 112) * 2.0
        df.iloc[k] = [c - 1, c + 1, c - 1, c, 1000.0]
    oc = simulate_ote_trade(df, fl, ExitConfig())
    assert oc.r_net > 0
    assert oc.exit_reason in ("trail", "timeout", "stop")


def test_simulate_same_bar_stop_is_minus_1r():
    df = _df()
    setup, limit = _setup()
    df.iloc[111] = [114, 116, 90.0, 115, 1000.0]
    fl = find_fill(df, setup, 0.70)
    assert fl is not None and fl.same_bar_stop
    oc = simulate_ote_trade(df, fl, ExitConfig())
    assert oc.r_net == pytest.approx(-1.0 - 0.45 / fl.risk)
    assert oc.hold_bars == 0
