"""Tests for the placement-review probes (failures as signals + inverses).

All synthetic: no market data needed except the run_placement smoke test,
which is skipped unless RUN_SLOW=1 (the full run takes ~1-2 min).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from goldenswing.experiments import placement as P


# ------------------------------------------------------------------ helpers
class _S:
    """Minimal state stub: only .atr / .killzone / .atr_percentile used."""

    def __init__(self, atr=1.0, killzone=False, ap=50.0):
        self.atr = atr
        self.killzone = killzone
        self.atr_percentile = ap
        self.index = 0
        self.recent_swing_highs = []
        self.recent_swing_lows = []
        self.last_structure_event = None
        self.structure_event_age = None


def _df(n=120, seed=7, trend=0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(trend, 1.0, n)) + 100.0
    o = walk + rng.normal(0, 0.2, n)
    c = walk + rng.normal(0, 0.2, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.5, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.5, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c},
                        index=idx)


def _states(n, atr=1.0):
    return [_S(atr=atr) for _ in range(n)]


# ----------------------------------------------------------------- simulate
def test_simulate_long_target():
    df = _df(60)
    # Rig bars so a long from bar 10 hits a +2R target quickly.
    df.loc[df.index[11], "open"] = 100.0
    for k in (11, 12):
        df.loc[df.index[k], "low"] = 99.9
        df.loc[df.index[k], "high"] = 104.0
    oc = P._simulate(df, 10, 1, 99.0, "t")
    assert oc is not None
    # risk = 1.0, target = 102.0 -> +2R gross minus cost
    assert oc.exit_reason == "target"
    assert oc.r_net == pytest.approx(2.0 - 0.45 / 1.0, abs=1e-9)


def test_simulate_short_stop():
    df = _df(60)
    df.loc[df.index[21], "open"] = 100.0
    df.loc[df.index[21], "high"] = 101.5  # stop=101 -> stopped
    df.loc[df.index[21], "low"] = 95.0
    oc = P._simulate(df, 20, -1, 101.0, "t")
    assert oc is not None and oc.exit_reason == "stop"
    assert oc.r_net == pytest.approx(-1.0 - 0.45 / 1.0, abs=1e-9)


def test_simulate_no_edge_when_risk_nonpositive():
    df = _df(60)
    df.loc[df.index[11], "open"] = 100.0
    assert P._simulate(df, 10, 1, 100.5, "t") is None  # stop above entry


def test_simulate_needs_next_bar():
    df = _df(30)
    assert P._simulate(df, 29, 1, 90.0, "t") is None
    df.loc[df.index[29], "open"] = 95.0  # entry above the stop -> valid
    assert P._simulate(df, 28, 1, 90.0, "t") is not None


# ------------------------------------------------------------------- dedupe
def test_dedupe_keeps_first_per_bar_direction():
    sigs = [(5, 1, 90.0, 110.0), (5, 1, 91.0, 111.0),
            (5, -1, 112.0, 89.0), (6, 1, 90.0, 110.0)]
    out = P._dedupe(sigs)
    assert out == [(5, 1, 90.0, 110.0), (5, -1, 112.0, 89.0),
                   (6, 1, 90.0, 110.0)]


# ------------------------------------------------------------- T2 pin setup
def _pin_df() -> pd.DataFrame:
    """Bar 20: bearish pin (upper wick). Bar 23: close above its high."""
    df = _df(60)
    i = 20
    df.loc[df.index[i], "open"] = 100.0
    df.loc[df.index[i], "close"] = 100.5
    df.loc[df.index[i], "high"] = 110.0   # upper wick 9.5
    df.loc[df.index[i], "low"] = 99.0
    # neutral bars 21..22 (not pins, no violation)
    for k in (21, 22):
        df.loc[df.index[k], "open"] = 101.0
        df.loc[df.index[k], "close"] = 101.0
        df.loc[df.index[k], "high"] = 102.0
        df.loc[df.index[k], "low"] = 100.0
    # bar 23: closes above the pin high -> violation
    df.loc[df.index[23], "open"] = 109.0
    df.loc[df.index[23], "close"] = 111.0
    df.loc[df.index[23], "high"] = 112.0
    df.loc[df.index[23], "low"] = 108.0
    return df


def test_t2_detects_pin_violation_long():
    from goldenswing.experiments.candles import pin_bar
    df = _pin_df()
    o, h, l, c = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    assert pin_bar(o, h, l, c, 20) == -1  # bearish pin confirmed
    sigs = P._t2_signals(df, _states(len(df)), fail_bars=12)
    viol = [s for s in sigs if s[0] == 23]
    assert len(viol) == 1
    bar, d, stop_bias, stop_inv = viol[0]
    assert d == 1  # bias: long the violation
    assert stop_bias == pytest.approx(99.0 - 0.25 * 1.0)   # pin low - buffer
    assert stop_inv == pytest.approx(110.0 + 0.25 * 1.0)   # pin high + buffer


def test_t2_no_violation_no_signal():
    df = _df(60)
    i = 20
    df.loc[df.index[i], "open"] = 100.0
    df.loc[df.index[i], "close"] = 100.5
    df.loc[df.index[i], "high"] = 110.0
    df.loc[df.index[i], "low"] = 99.0
    # Keep every later close below the pin high: no violation possible.
    for k in range(21, 33):
        df.loc[df.index[k], "close"] = 105.0
        df.loc[df.index[k], "high"] = 106.0
        df.loc[df.index[k], "low"] = 104.0
        df.loc[df.index[k], "open"] = 105.0
    sigs = P._t2_signals(df, _states(len(df)), fail_bars=12)
    # No long-violation signal may reference the pin at bar 20.
    assert not any(s[0] > 20 and s[0] <= 32 and s[1] == 1 for s in sigs)


def test_t2_inverse_is_mirror():
    df = _pin_df()
    sigs = P._t2_signals(df, _states(len(df)), fail_bars=12)
    bar, d, stop_bias, stop_inv = [s for s in sigs if s[0] == 23][0]
    # Inverse flips direction and uses the mirrored stop (pin high + buf).
    # Note the mirror is honest about its own weakness: the violation
    # close (111.0) is already above the inverse short's stop (110.25),
    # so most inverse fills are skipped as risk<=0 by _simulate.
    assert -d == -1
    assert stop_inv == pytest.approx(110.0 + 0.25 * 1.0)


# ------------------------------------------------------- T4 doji expansion
def _doji_df() -> pd.DataFrame:
    """Bars 30-31: dojis (tiny body, decent range). Bar 33: expansion up."""
    df = _df(80)
    for i in (30, 31):
        df.loc[df.index[i], "open"] = 100.0
        df.loc[df.index[i], "close"] = 100.05   # tiny body
        df.loc[df.index[i], "high"] = 101.0    # range 2.0 >= 0.5*atr
        df.loc[df.index[i], "low"] = 99.0
    df.loc[df.index[32], "open"] = 100.0
    df.loc[df.index[32], "close"] = 100.2
    df.loc[df.index[32], "high"] = 100.8
    df.loc[df.index[32], "low"] = 99.5
    df.loc[df.index[33], "open"] = 100.5
    df.loc[df.index[33], "close"] = 102.5   # closes above cluster high 101
    df.loc[df.index[33], "high"] = 103.0
    df.loc[df.index[33], "low"] = 100.0    # range 3.0 >= 0.75*atr
    return df


def test_t4_detects_expansion_break():
    from goldenswing.experiments.candles import is_doji
    df = _doji_df()
    o, h, l, c = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    assert is_doji(o, h, l, c, 1.0, 30) and is_doji(o, h, l, c, 1.0, 31)
    sigs = P._t4_signals(df, _states(len(df)), fail_bars=12)
    brk = [s for s in sigs if s[0] == 33]
    assert len(brk) == 1
    bar, d, stop_bias, stop_inv = brk[0]
    assert d == 1
    assert stop_bias == pytest.approx(99.0 - 0.25)   # cluster low - buffer
    assert stop_inv == pytest.approx(101.0 + 0.25)   # cluster high + buffer


def test_t4_clusters_do_not_overlap():
    df = _doji_df()
    sigs = P._t4_signals(df, _states(len(df)), fail_bars=12)
    bars = sorted(s[0] for s in sigs)
    assert bars == sorted(set(bars))  # no duplicate bars after dedupe-less scan


# ------------------------------------------------------------------ T1/T3
def test_t1_empty_on_flat_data():
    df = _df(200, trend=0.0)
    # No swings on the stub states -> the tracker never forms a pattern.
    # The scanner must not crash and every signal must respect causality.
    sigs = P._t1_signals(df, _states(len(df)), fail_bars=12)
    assert sigs == []
    assert all(s[0] < len(df) - 1 for s in sigs)


def test_t3_needs_featurized_states_note():
    # scan_setups on states with no BOS events finds nothing: the T3
    # wiring must use featurized states (read_and_featurize), which carry
    # the supply/demand zones and structure events.  This pins the
    # docstring contract.
    from goldenswing.experiments.retest_entry import scan_setups, ENTRY_EDGE
    df = _df(300, trend=0.05)
    assert scan_setups(df, _states(len(df)), entry=ENTRY_EDGE) == []


# ------------------------------------------------------------- stats/report
def test_stats_empty_and_basic():
    r = P._stats([], "x")
    assert r.n_trades == 0 and r.avg_r == 0.0
    df = _df(60)
    df.loc[df.index[11], "open"] = 100.0
    for k in (11, 12):
        df.loc[df.index[k], "low"] = 99.9
        df.loc[df.index[k], "high"] = 104.0
    oc = P._simulate(df, 10, 1, 99.0, "t")
    r = P._stats([oc], "x")
    assert r.n_trades == 1 and r.win_rate == 1.0 and r.avg_r > 1.5


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1",
                    reason="full placement run (~2 min); set RUN_SLOW=1")
def test_run_placement_smoke():
    rep = P.run_placement(fail_bars=8)
    names = [p.name for p in rep.probes]
    assert len(names) == 8  # 4 probes x (bias, inverse)
    assert any("T2_pin_violation" == n for n in names)
    assert rep.bars == 11426
