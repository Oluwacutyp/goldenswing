"""Fibonacci levels/OTE on hand-built states; Ichimoku cloud states."""
import pytest

from conftest import make_df
from goldenswing.features import fibonacci, ichimoku
from goldenswing.features.config import FeatureConfig
from goldenswing.reader.state import MarketState, SwingPoint

CFG = FeatureConfig()


def _state(**over) -> MarketState:
    base = dict(index=100, timestamp="t", close=2030.0, atr=10.0,
                trend="up", bias=1, range_high=2100.0, range_low=2000.0,
                recent_swing_highs=[SwingPoint(price=2100.0, index=50,
                                               confirmed_at=52, kind="high")],
                recent_swing_lows=[SwingPoint(price=2000.0, index=40,
                                              confirmed_at=42, kind="low")])
    base.update(over)
    return MarketState(**base)


def test_fib_levels_are_range_retracements():
    s = fibonacci.compute(None, [_state()], CFG)[0]
    prices = dict(s.levels)
    assert prices[0.5] == pytest.approx(2050.0)
    assert prices[0.382] == pytest.approx(2061.8)
    assert prices[0.79] == pytest.approx(2021.0)
    assert s.leg_direction == 1     # high confirmed after the low


def test_ote_zone_membership():
    # close 2030 retraces (2100-2030)/100 = 0.70 of the up leg -> OTE
    s = fibonacci.compute(None, [_state()], CFG)[0]
    assert s.in_ote_long and not s.in_ote_short
    assert s.ote_low < 2030.0 < s.ote_high
    assert s.nearest_frac == pytest.approx(0.705)
    assert s.at_fib                       # 0.5 away, tol 0.15 x ATR(10)
    # shallow retracement: not OTE
    s2 = fibonacci.compute(None, [_state(close=2090.0)], CFG)[0]
    assert not s2.in_ote_long


def test_fib_no_range_is_empty():
    s = fibonacci.compute(
        None, [_state(range_high=float("nan"), range_low=float("nan"))],
        CFG)[0]
    assert s.levels == () and not s.at_fib


def _rows(closes, pad=0.2):
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + pad, min(prev, c) - pad, c))
        prev = c
    return rows


def test_ichimoku_flat_series_sits_inside_cloud():
    df = make_df(_rows([2000.0] * 200))
    out = ichimoku.compute(df, [None] * 200, CFG)
    s = out[-1]
    assert s.position == "inside"
    assert s.cloud_top == pytest.approx(s.cloud_bottom, abs=1.0)


def test_ichimoku_ramp_is_above_cloud_with_bullish_tk_cross():
    closes = [2000.0] * 100 + [2000.0 + 2.0 * k for k in range(1, 101)]
    df = make_df(_rows(closes))
    out = ichimoku.compute(df, [None] * len(df), CFG)
    assert out[-1].position == "above"
    assert any(s.tk_cross == 1 for s in out[100:])
    assert out[-1].tenkan > out[-1].kijun
    assert out[-1].chikou == 1
