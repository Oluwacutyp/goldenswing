"""Volume character, participant footprints (hand-built states), MTF."""
import numpy as np

from conftest import closes_df, make_df
from goldenswing.features import footprints, mtf, volume
from goldenswing.features.config import FeatureConfig
from goldenswing.reader import MarketReader
from goldenswing.reader.state import MarketState, SweepEvent

CFG = FeatureConfig()


def _vol_df(vols, closes=None):
    closes = closes or [2000.0] * len(vols)
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + 0.2, min(prev, c) - 0.2, c))
        prev = c
    df = make_df(rows)
    df["volume"] = [float(v) for v in vols]
    return df


def test_volume_constant_series_rvol_one_no_climax():
    df = _vol_df([1000.0] * 60)
    out = volume.compute(df, [None] * 60, CFG)
    last = out[-1]
    assert last.available and abs(last.rvol - 1.0) < 1e-9
    assert not last.climax and not last.dry_up


def test_volume_spike_is_climax_and_zero_series_degrades():
    vols = [1000.0] * 40 + [4000.0]
    closes = [2000.0] * 40 + [1996.0]
    df = _vol_df(vols, closes)
    out = volume.compute(df, [None] * 41, CFG)
    assert out[-1].climax and out[-1].rvol > 3.0
    dead = volume.compute(_vol_df([0.0] * 40), [None] * 40, CFG)
    assert not dead[-1].available
    assert dead[-1].rvol != dead[-1].rvol      # NaN, never invented


def _fp_state(**over) -> MarketState:
    base = dict(index=50, timestamp="t", close=2005.0, atr=10.0,
                trend="up", bias=1)
    base.update(over)
    return MarketState(**base)


def test_institutional_signature_chain():
    # sweep 2 bars ago (bar 48), displacement 1 bar ago (bar 49) on
    # climax RVOL, close holding near the highs -> signature fires.
    states = [_fp_state(index=t) for t in range(51)]
    states[50] = _fp_state(
        last_sweep=SweepEvent(index=48, level=1995.0, extreme=1990.0,
                              direction=1),
        sweep_age=2, bull_disp_age=1)
    closes = np.full(51, 2000.0)
    closes[49] = 2006.0
    closes[50] = 2005.0
    df = _vol_df([1000.0] * 49 + [2500.0, 1200.0], closes.tolist())
    rvol, vwap, _ = volume.volume_arrays(df, CFG)
    out = footprints.compute(df, states, CFG, rvol, vwap)
    assert out[50].institutional_dir == 1
    assert "institutional_entry" in out[50].tags
    assert out[50].evidence            # evidence, not a bare label


def test_retail_chase_at_extreme():
    states = [_fp_state(index=t) for t in range(30)]
    states[29] = _fp_state(body_ratio=1.5, eq_position=0.95,
                           momentum=3.2)
    closes = np.linspace(1990, 2010, 30)
    df = _vol_df([1000.0] * 29 + [2200.0], closes.tolist())
    rvol, vwap, _ = volume.volume_arrays(df, CFG)
    out = footprints.compute(df, states, CFG, rvol, vwap)
    assert out[29].retail_chase == 1
    assert "retail_chase" in out[29].tags


def test_mtf_h4_alignment_in_trending_oscillation():
    # trend + oscillation: a PURE ramp forms no fractal swings on any
    # timeframe (nothing ever pulls back), so structure trend stays
    # "range" by design — the oscillation creates the swings the HTF
    # readers confirm and break.  (On this series the DAILY extremes
    # stay trend-dominated, so D1 gets its own zigzag test below.)
    n = 24 * 60
    closes = [2000.0 + 0.9 * k + 25.0 * np.sin(k / 9.0) for k in range(n)]
    df = closes_df(closes, spread=0.4)
    states = MarketReader().read(df)
    out = mtf.compute(df, states, CFG)
    tail = out[-24:]
    assert all(s.h4_bias == 1 for s in tail)
    assert any(s.alignment > 0.6 for s in tail)
    assert out[-1].h4_swing_high is not None


def test_mtf_d1_trend_from_daily_zigzag():
    # flat intraday days tracing a rising 4-day zigzag at the daily
    # level: +20/+20/-20/-30 per cycle, net +10 — daily fractal swings
    # form, confirm, and break upward.
    levels = []
    v = 2000.0
    for _cycle in range(10):
        for step in (20.0, 20.0, -20.0, -10.0):
            v += step
            levels.append(v)
    closes = []
    for lv in levels:
        closes += [lv] * 24
    df = closes_df(closes, spread=0.4)
    states = MarketReader().read(df)
    out = mtf.compute(df, states, CFG)
    tail = out[-24:]
    assert any(s.d1_bias == 1 for s in tail)
    assert out[-1].d1_swing_high is not None
    assert out[-1].d1_swing_low is not None
