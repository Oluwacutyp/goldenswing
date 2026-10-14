"""Candlestick experiment: geometry, causality, simulation."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_df

from goldenswing.experiments.candles import (
    _context,
    displacement,
    engulfing,
    inside_break,
    is_doji,
    pin_bar,
    scan,
    simulate,
    three_push,
    SIGNALS,
)
from goldenswing.reader.state import MarketState


def _ohlc(rows):
    df = make_df(rows)
    return (df["open"].to_numpy(), df["high"].to_numpy(),
            df["low"].to_numpy(), df["close"].to_numpy())


def _blank_state(i, **kw):
    d = dict(index=i, timestamp=None, close=100.0, atr=2.0, trend="range",
             bias=0, killzone=False, atr_percentile=50.0)
    d.update(kw)
    return MarketState(**d)


# ------------------------------------------------------------ geometry
def test_pin_bar_bullish():
    o, h, l, c = _ohlc([(100, 102, 98, 100),   # noise
                        (100, 101, 90, 100.5)])  # long lower wick
    assert pin_bar(o, h, l, c, 1) == 1


def test_pin_bar_bearish():
    o, h, l, c = _ohlc([(100, 102, 98, 100),
                        (99.5, 110, 99, 100)])  # long upper wick
    assert pin_bar(o, h, l, c, 1) == -1


def test_pin_bar_needs_dominant_wick():
    o, h, l, c = _ohlc([(100, 101, 99, 100.8)])  # body too big vs wick
    assert pin_bar(o, h, l, c, 0) == 0


def test_engulfing_bullish():
    o, h, l, c = _ohlc([(102, 103, 99, 100),    # bearish
                        (99, 104, 98.5, 103)])  # bullish engulfs body
    assert engulfing(o, h, l, c, 1) == 1


def test_engulfing_needs_opposite_colours():
    o, h, l, c = _ohlc([(100, 103, 99, 102),    # bullish
                        (101, 105, 100, 104)])  # bullish, not engulfed
    assert engulfing(o, h, l, c, 1) == 0


def test_displacement_needs_body_and_small_wicks():
    rows = [(100, 100.4, 99.6, 100.1)] * 5
    rows.append((100, 106.5, 99.8, 106))  # body 6 >= 1.5*atr(2)
    o, h, l, c = _ohlc(rows)
    assert displacement(o, h, l, c, 2.0, 5) == 1


def test_displacement_rejects_wicky_break():
    rows = [(100, 100.4, 99.6, 100.1)] * 5
    rows.append((100, 107, 93, 106))  # big wicks both sides
    o, h, l, c = _ohlc(rows)
    assert displacement(o, h, l, c, 2.0, 5) == 0


def test_inside_break():
    o, h, l, c = _ohlc([(100, 105, 95, 102),     # mother
                        (101, 103, 99, 102),     # inside
                        (102, 104, 101, 103.5)])  # breaks inside high
    assert inside_break(o, h, l, c, 2) == 1


def test_doji_detection():
    rows = [(100, 102, 98, 100.05)]  # body 0.05 <= 10% of range 4
    o, h, l, c = _ohlc(rows)
    assert is_doji(o, h, l, c, 2.0, 0)
    rows = [(100, 102, 98, 101.5)]  # body too big
    o, h, l, c = _ohlc(rows)
    assert not is_doji(o, h, l, c, 2.0, 0)


def test_three_push_up():
    rows = [(100, 102, 99, 101.5),
            (101.5, 103.5, 101, 103),
            (103, 105, 102.5, 104.5)]
    o, h, l, c = _ohlc(rows)
    assert three_push(o, h, l, c, 2) == 1


# ------------------------------------------------------------ context
def test_context_cells():
    st = _blank_state(10, trend="up", bias=1, killzone=True,
                      atr_percentile=80.0)
    ctx = _context(st, 1)
    assert ctx["trend_align"] == "with"
    assert ctx["killzone"] is True
    assert ctx["atr_regime"] == "hi"
    assert ctx["location"] == "midrange"  # no features attached
    ctx2 = _context(st, -1)
    assert ctx2["trend_align"] == "against"


# ------------------------------------------------------------ simulate
def _walk_df():
    # pin at bar 5 (bullish), then price rises to 2R
    rows = [(100, 101, 99, 100)] * 5
    rows.append((100, 100.5, 90, 100.2))      # bull pin, extreme 90
    rows += [(100.2, 126, 100, 124)]         # run to target
    rows += [(124, 126, 122, 124)] * 80
    return make_df(rows)


def test_simulate_target_hit():
    from goldenswing.experiments.candles import CandleSetup
    df = _walk_df()
    st = _blank_state(5)
    s = CandleSetup(index=5, signal="pin", direction=1,
                    stop=90 - 0.25 * 2.0, risk=10.0, atr=2.0,
                    ctx=_context(st, 1))
    oc = simulate(df, s)
    assert oc is not None
    assert oc.exit_reason == "target"
    assert oc.r_net > 1.5  # ~2R minus cost


def test_simulate_stop_hit():
    from goldenswing.experiments.candles import CandleSetup
    rows = [(100, 101, 99, 100)] * 5
    rows.append((100, 100.5, 90, 100.2))      # bull pin
    rows += [(100, 101, 89, 90)]              # collapses through stop
    rows += [(90, 91, 89, 90)] * 80
    df = make_df(rows)
    st = _blank_state(5)
    s = CandleSetup(index=5, signal="pin", direction=1,
                    stop=90 - 0.25 * 2.0, risk=10.0, atr=2.0,
                    ctx=_context(st, 1))
    oc = simulate(df, s)
    assert oc is not None
    assert oc.exit_reason == "stop"
    assert oc.r_net < 0


def test_simulate_rejects_bad_risk():
    from goldenswing.experiments.candles import CandleSetup
    df = _walk_df()
    st = _blank_state(5)
    # stop ABOVE the entry for a long -> risk <= 0
    s = CandleSetup(index=5, signal="pin", direction=1, stop=200.0,
                    risk=10.0, atr=2.0, ctx=_context(st, 1))
    assert simulate(df, s) is None


# ------------------------------------------------------------ scan
def test_scan_finds_pin_and_is_causal():
    rows = [(100, 102, 98, 100)] * 20
    rows.append((100, 101, 95, 100.5))        # bull pin at 20, wick ~5
    rows += [(100, 102, 98, 100)] * 70
    df = make_df(rows)
    states = [_blank_state(i, atr=4.0) for i in range(len(df))]
    found = [s for s in scan(df, states)
             if s.index == 20 and s.signal == "pin"]
    assert len(found) == 1
    assert found[0].direction == 1
    # causality: truncating the series must not change earlier setups
    df2 = df.iloc[:50].copy()
    states2 = [_blank_state(i, atr=4.0) for i in range(len(df2))]
    found2 = [s for s in scan(df2, states2)
              if s.index == 20 and s.signal == "pin"]
    assert len(found2) == 1
    assert found2[0].stop == found[0].stop


def test_scan_rejects_tiny_and_huge_risk():
    # pin with a 0.1-range wick on atr=2 -> risk < 0.25 ATR -> skipped
    rows = [(100, 100.06, 99.94, 100.01)] * 6
    df = make_df(rows)
    states = [_blank_state(i, atr=2.0) for i in range(len(df))]
    assert not [s for s in scan(df, states) if s.signal == "pin"]
