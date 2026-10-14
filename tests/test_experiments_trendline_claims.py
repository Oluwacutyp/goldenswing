"""Trendline-claims experiments: builders, wedges, retests, causality."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_df

from goldenswing.experiments.trendline_claims import (
    _BodyBuilder,
    _line_key,
    _passes_gates,
    build_lines,
    detect_wedges,
    index_lines,
    scan_retests,
    scan_wedge_breakouts,
    touch_bars,
)
from goldenswing.features.config import FeatureConfig
from goldenswing.features.state import (Trendline, TrendlineEvent,
                                        TrendlinesState)
from goldenswing.features.trendlines import _Builder
from goldenswing.golden.detector import DetectorConfig, GoldenSwingDetector
from goldenswing.backtest.ablation import P2_FLAGS
from goldenswing.reader.state import (LiquidityPool, MarketState, SwingPoint)

ATR = 1.0


def _mstate(t, swing_lows=(), swing_highs=(), close=100.0, daily_bias=0,
            pools_below=(), pools_above=()):
    return MarketState(
        index=t, timestamp=pd.Timestamp("2024-01-01", tz="UTC")
        + pd.Timedelta(hours=t), close=close, atr=ATR,
        trend="range", bias=0, daily_bias=daily_bias,
        recent_swing_lows=[SwingPoint(price=p, index=i, confirmed_at=t,
                                      kind="low") for p, i in swing_lows],
        recent_swing_highs=[SwingPoint(price=p, index=i, confirmed_at=t,
                                       kind="high") for p, i in swing_highs],
        pools_below=list(pools_below), pools_above=list(pools_above))


def _pool(price):
    return LiquidityPool(price=price, side="ssl", strength=1, created_at=0)


def _df_body(n=10):
    # swing-low bar 4: low=100 (shadow), body min = 102
    rows = [(100.0, 101.0, 99.0, 100.5)] * n
    rows[4] = (103.0, 104.0, 100.0, 102.0)
    return make_df(rows)


# ------------------------------------------------------- anchor conventions
def test_body_builder_anchors_on_body_extreme():
    df = _df_body()
    b = _BodyBuilder(df, FeatureConfig())
    st = _mstate(6, swing_lows=[(100.0, 4)])
    b._ingest(st)
    assert b.stream[0][1] == pytest.approx(102.0)  # min(open, close)


def test_shadow_builder_anchors_on_shadow_extreme():
    df = _df_body()
    b = _Builder(df, FeatureConfig())
    st = _mstate(6, swing_lows=[(100.0, 4)])
    b._ingest(st)
    assert b.stream[0][1] == pytest.approx(100.0)  # the swing price itself


def test_body_builder_high_uses_body_top():
    df = _df_body()
    b = _BodyBuilder(df, FeatureConfig())
    st = _mstate(6, swing_highs=[(104.0, 4)])
    b._ingest(st)
    assert b.stream[0][1] == pytest.approx(103.0)  # max(open, close)


def test_builders_only_use_confirmed_swings():
    df = _df_body()
    b = _BodyBuilder(df, FeatureConfig())
    st = _mstate(6, swing_lows=[(100.0, 4)])
    # swing confirmed in the future must not be ingested
    st2 = _mstate(6, swing_lows=[(100.0, 4)])
    st2.recent_swing_lows[0] = SwingPoint(price=100.0, index=4,
                                          confirmed_at=9, kind="low")
    assert b._ingest(st2) == []
    assert b.stream == []


def test_lines_never_use_future_bars():
    # a real reader run: every line's anchors predate its creation bar
    df = make_df([(100 + (i % 7) - 3, 102 + (i % 7) - 3, 98 + (i % 7) - 3,
                   100 + (i % 7) - 3) for i in range(120)])
    from goldenswing.reader.reader import MarketReader
    states = MarketReader().read(df)
    for mode in ("shadow", "body"):
        for ls in build_lines(df, states, FeatureConfig(), mode):
            for tl in ls.active:
                assert tl.anchor1_index < tl.anchor2_index
                assert tl.anchor2_index < tl.created_at


# ------------------------------------------------------------------ wedges
def _wedge_states(n=80, brk=None):
    """Converging support (rising slowly) + resistance (falling slowly),
    both touched.  Boundaries cross at t=100, so the pair is a clean
    converging wedge through the whole window."""
    sup = Trendline(direction=1, anchor1_index=0, anchor1_price=100.0,
                    anchor2_index=10, anchor2_price=100.5, slope=0.05,
                    created_at=0, touches=2, state="ACTIVE")
    res = Trendline(direction=-1, anchor1_index=0, anchor1_price=110.0,
                    anchor2_index=10, anchor2_price=109.5, slope=-0.05,
                    created_at=0, touches=2, state="ACTIVE")
    rows = [(105.0, 106.0, 104.0, 105.0)] * n
    if brk is not None:
        k, o_, h_, l_, c_ = brk
        rows[k] = (o_, h_, l_, c_)
    df = make_df(rows)
    line_states = [TrendlinesState(active=(sup, res), events=())
                   for _ in range(n)]
    states = [_mstate(t) for t in range(n)]
    return df, states, line_states


def test_detect_wedge_converging():
    df, states, line_states = _wedge_states()
    wedges = detect_wedges(line_states, states, df)
    assert len(wedges) == 1
    # support rising, resistance falling -> "mixed" in our taxonomy;
    # the contraction is what the claim is about
    assert wedges[0]["type"] == "mixed"
    assert wedges[0]["d_prior"] == -1  # flat series -> not up -> -1


def test_no_wedge_when_diverging():
    df, states, line_states = _wedge_states()
    sup, _res = line_states[0].active
    res2 = Trendline(direction=-1, anchor1_index=0, anchor1_price=100.0,
                     anchor2_index=10, anchor2_price=101.5, slope=0.15,
                     created_at=0, touches=2, state="ACTIVE")
    line_states = [TrendlinesState(active=(sup, res2), events=())
                   for _ in range(80)]
    assert detect_wedges(line_states, states, df) == []


def test_no_wedge_when_untouched():
    df, states, line_states = _wedge_states()
    sup, res = line_states[0].active
    sup0 = Trendline(direction=1, anchor1_index=0, anchor1_price=100.0,
                     anchor2_index=10, anchor2_price=100.5, slope=0.05,
                     created_at=0, touches=0, state="FORMING")
    line_states = [TrendlinesState(active=(sup0, res), events=())
                   for _ in range(80)]
    assert detect_wedges(line_states, states, df) == []


def test_wedge_breakout_in_trend():
    # d_prior = -1; support at k=55 is 102.75 -> close 102 breaks it down
    df, states, line_states = _wedge_states(
        brk=(55, 105.0, 106.0, 101.0, 102.0))
    wedges = detect_wedges(line_states, states, df)
    assert len(wedges) == 1
    brks = scan_wedge_breakouts(wedges, df, states)
    assert len(brks) == 1
    assert brks[0]["kind"] == "in_trend"
    assert brks[0]["break_bar"] == 55
    assert brks[0]["bdir"] == -1


def test_wedge_breakout_counter_trend():
    # spike up through resistance at k=55 (pr = 107.25) instead
    df, states, line_states = _wedge_states(
        brk=(55, 105.0, 112.0, 104.0, 111.0))
    wedges = detect_wedges(line_states, states, df)
    assert len(wedges) == 1
    brks = scan_wedge_breakouts(wedges, df, states)
    assert len(brks) == 1
    assert brks[0]["kind"] == "counter"
    assert brks[0]["bdir"] == 1


# ------------------------------------------------------------------ retests
def _retest_fixture(n=40, touch_at=23, hover=False):
    """Break of an ACTIVE support line at bar 20 (flat line at 100)."""
    tl = Trendline(direction=1, anchor1_index=10, anchor1_price=100.0,
                   anchor2_index=15, anchor2_price=100.0, slope=0.0,
                   created_at=18, touches=1, state="BROKEN")
    ev = TrendlineEvent(kind="break", direction=1, line_price=100.0,
                        touches=1)
    line_states = [TrendlinesState(active=(), events=()) for _ in range(n)]
    line_states[20] = TrendlinesState(active=(tl,), events=(ev,))
    rows = [(99.0, 99.5, 98.5, 99.0)] * n     # below the line
    if not hover:
        rows[23] = (99.0, 100.1, 98.5, 99.5)  # wick touches, close below
    else:
        rows[21] = (99.0, 100.1, 98.5, 99.5)  # immediate hover, no away
        rows[22] = (99.0, 100.1, 98.5, 99.5)
    df = make_df(rows)
    states = [_mstate(t, close=rows[t][3], pools_below=[_pool(95.0)],
                      pools_above=[_pool(110.0)]) for t in range(n)]
    return df, states, line_states


def _det():
    return GoldenSwingDetector(DetectorConfig(**P2_FLAGS))


def test_retest_fires_after_away_then_back():
    df, states, line_states = _retest_fixture()
    sigs = scan_retests(df, states, line_states, _det())
    assert len(sigs) == 1
    assert sigs[0].index == 23
    assert sigs[0].direction == -1
    assert sigs[0].invalidation == pytest.approx(100.1)


def test_no_retest_without_away_bars():
    df, states, line_states = _retest_fixture(hover=True)
    sigs = scan_retests(df, states, line_states, _det())
    # hover touches never arm (re-arm clock keeps resetting)
    assert sigs == []


def test_no_retest_without_return():
    df, states, _ = _retest_fixture()
    df2 = make_df([(99.0, 99.5, 98.5, 99.0)] * 40)  # never comes back
    states2 = [_mstate(t, close=99.0, pools_below=[_pool(95.0)],
                       pools_above=[_pool(110.0)]) for t in range(40)]
    assert scan_retests(df2, states2, _, _det()) == []


def test_one_trade_per_break():
    df, states, line_states = _retest_fixture()
    # second return touch later must not double-fire
    rows = [r for r in
            zip(df["open"], df["high"], df["low"], df["close"])]
    rows[30] = (99.0, 100.1, 98.5, 99.5)
    df = make_df(rows)
    states = [_mstate(t, close=rows[t][3], pools_below=[_pool(95.0)],
                      pools_above=[_pool(110.0)]) for t in range(40)]
    sigs = scan_retests(df, states, line_states, _det())
    assert len(sigs) == 1


def test_break_of_forming_line_ignored():
    df, states, line_states = _retest_fixture()
    tl = Trendline(direction=1, anchor1_index=10, anchor1_price=100.0,
                   anchor2_index=15, anchor2_price=100.0, slope=0.0,
                   created_at=18, touches=0, state="BROKEN")
    ev = TrendlineEvent(kind="break", direction=1, line_price=100.0,
                        touches=0)
    line_states[20] = TrendlinesState(active=(tl,), events=(ev,))
    assert scan_retests(df, states, line_states, _det()) == []


def test_daily_veto_blocks_retest():
    df, states, _ = _retest_fixture()
    df, states, line_states = _retest_fixture()
    states = [_mstate(t, close=99.0 if t != 23 else 99.5, daily_bias=1,
                      pools_below=[_pool(95.0)], pools_above=[_pool(110.0)])
              for t in range(40)]
    # short retest (d=-1) against daily_bias=+1 -> vetoed
    assert scan_retests(df, states, line_states, _det()) == []


# ------------------------------------------------------------------- misc
def test_index_lines_matches_events():
    df, states, line_states = _retest_fixture()
    lines = index_lines(line_states)
    assert len(lines) == 1
    rec = next(iter(lines.values()))
    assert rec["breaks"] == [20]
    assert touch_bars(line_states) == set()
