"""Tests for the bedrock regime engine (live/regime.py).

Synthetic bars are built from EXPLICIT swing points (deterministic),
not sine waves: linear interpolation between alternating swing highs
and swing lows, so find_swings recovers exactly the intended swings.
No FVG / OB / entry tools are used anywhere in these tests — bedrock
must be readable without them.
"""

import numpy as np
import pandas as pd
import pytest

from goldenswing.live.regime import (
    Regime,
    classify_regime,
    route_for_regime,
    session_of,
)


def bars_from_swings(highs, lows, bars_per_leg=6, start="2026-01-01 00:00",
                     freq="h", body=0.3):
    """Build OHLC bars through alternating swing points.

    highs/lows: lists of (index, price) in increasing index order.
    Close interpolates linearly between consecutive swing points.
    high = close + 0.3, low = close - 0.3, open = prev close.
    """
    points = sorted([(i, p) for i, p in highs + lows])
    n = points[-1][0] + 1
    close = np.zeros(n)
    for (i0, p0), (i1, p1) in zip(points, points[1:]):
        steps = i1 - i0
        for k in range(steps):
            close[i0 + k] = p0 + (p1 - p0) * k / steps
    close[points[-1][0]] = points[-1][1]
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    op = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": op, "high": close + body, "low": close - body, "close": close},
        index=idx,
    )


def uptrend_bars(n_legs=8):
    """Higher highs + higher lows, steady climb. seq = (bar_idx, price)."""
    seq = []
    price = 4000.0
    idx = 0
    for k in range(n_legs * 2):
        if k % 2 == 0:
            seq.append((idx, price + 10.0))
        else:
            seq.append((idx, price))
            price += 10.0
        idx += 6
    return bars_from_swings(
        [p for j, p in enumerate(seq) if j % 2 == 0],
        [p for j, p in enumerate(seq) if j % 2 == 1],
    )


def downtrend_bars(n_legs=8):
    """Lower highs + lower lows, steady fall. seq = (bar_idx, price)."""
    seq = []
    price = 4100.0
    idx = 0
    for k in range(n_legs * 2):
        if k % 2 == 0:
            seq.append((idx, price - 10.0))
        else:
            seq.append((idx, price))
            price -= 10.0
        idx += 6
    return bars_from_swings(
        [p for j, p in enumerate(seq) if j % 2 == 0],
        [p for j, p in enumerate(seq) if j % 2 == 1],
    )


def range_bars(n_legs=8, top=4050.0, bottom=4000.0):
    """Equal highs + equal lows, bounded oscillation. seq = (idx, price)."""
    seq = []
    idx = 0
    for k in range(n_legs * 2):
        seq.append((idx, top if k % 2 == 0 else bottom))
        idx += 6
    return bars_from_swings(
        [p for j, p in enumerate(seq) if j % 2 == 0],
        [p for j, p in enumerate(seq) if j % 2 == 1],
    )


def compression_bars(n=100):
    """Tiny range, tiny bodies, low volatility."""
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    close = 4000 + 0.4 * np.sin(np.arange(n) / 3.0)
    return pd.DataFrame(
        {"open": close - 0.05, "high": close + 0.2, "low": close - 0.2,
         "close": close},
        index=idx,
    )


# ---------------------------------------------------------------------------

def test_uptrend_classifies_clean_trend_up():
    reg = classify_regime(uptrend_bars())
    assert reg.structure == "clean_trend", reg
    assert reg.direction == "up", reg
    assert reg.tradeable is True, reg
    assert reg.route == "TREND_PULLBACK", reg
    assert reg.confidence >= 0.35, reg


def test_downtrend_classifies_clean_trend_down():
    reg = classify_regime(downtrend_bars())
    assert reg.structure == "clean_trend", reg
    assert reg.direction == "down", reg
    assert reg.tradeable is True, reg
    assert reg.route == "TREND_PULLBACK", reg


def test_range_classifies_range_or_compression_and_sits_out_or_edges():
    reg = classify_regime(range_bars())
    assert reg.structure in ("range", "compression", "choppy"), reg
    assert reg.direction == "flat", reg
    # Range with accumulation sits out; range edges tradeable is allowed.
    assert reg.route in ("RANGE_EDGES", "SIT_OUT",
                         "SWEEP_DISPLACEMENT_RETRACEMENT"), reg


def test_compression_sits_out():
    reg = classify_regime(compression_bars())
    assert reg.structure in ("compression", "range", "unclear"), reg
    if reg.structure == "compression":
        assert reg.tradeable is False, reg
        assert reg.route == "SIT_OUT", reg


def test_not_enough_bars_is_unclear():
    idx = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 4000.0, "high": 4001.0, "low": 3999.0, "close": 4000.5},
        index=idx)
    reg = classify_regime(df)
    assert reg.structure == "unclear", reg
    assert reg.tradeable is False, reg
    assert reg.route == "SIT_OUT", reg


def test_conflicting_windows_is_unclear():
    # Long uptrend, then short sharp downtrend WITHOUT sweep+displacement
    # confirmation of reversal: must be Unclear, not a forced label.
    up = uptrend_bars(n_legs=6)
    # Append a short down leg that does NOT sweep + displace: gentle drift.
    n = 30
    idx = pd.date_range(up.index[-1] + pd.Timedelta(hours=1), periods=n,
                        freq="h", tz="UTC")
    start = float(up["close"].iloc[-1])
    close = start - 0.15 * np.arange(n)
    down = pd.DataFrame(
        {"open": close - 0.05, "high": close + 0.3, "low": close - 0.3,
         "close": close},
        index=idx)
    df = pd.concat([up, down])
    reg = classify_regime(df)
    # Either Unclear (honest) or a low-confidence label that sits out.
    # The key assertion: it must NOT be a confident tradeable trend.
    assert (reg.structure == "unclear" or not reg.tradeable
            or reg.confidence < 0.6), reg


def test_flat_structure_is_unclear_or_sits_out():
    # Constant price: no structure at all.
    idx = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 4000.0, "high": 4000.0, "low": 4000.0, "close": 4000.0},
        index=idx)
    reg = classify_regime(df)
    assert reg.tradeable is False, reg
    assert reg.route == "SIT_OUT", reg


def test_early_detection_candle_shift_downgrades_trend():
    """Candle character shift BEFORE displacement: trend bars followed by
    20 doji/wicky bars must NOT stay a confident clean trend."""
    trend = uptrend_bars(n_legs=6)
    n = 20
    idx = pd.date_range(trend.index[-1] + pd.Timedelta(hours=1), periods=n,
                        freq="h", tz="UTC")
    base = float(trend["close"].iloc[-1])
    # Doji / wicky bars: tiny bodies, long wicks, flat closes.
    close = base + 0.1 * np.sin(np.arange(n))
    df = pd.DataFrame(
        {"open": close, "high": close + 2.0, "low": close - 2.0,
         "close": close + 0.01},
        index=idx)
    combined = pd.concat([trend, df])
    reg_full = classify_regime(trend)
    reg_shift = classify_regime(combined)
    # After the character shift, confidence must drop OR structure must
    # leave clean_trend.  It must not stay an equally confident trend.
    assert (reg_shift.structure != "clean_trend"
            or reg_shift.confidence < reg_full.confidence
            or not reg_shift.tradeable), (reg_full, reg_shift)


def test_session_classification():
    assert session_of(pd.Timestamp("2026-01-01 03:00", tz="UTC")) == "asia"
    assert session_of(pd.Timestamp("2026-01-01 08:00", tz="UTC")) == "london"
    assert session_of(pd.Timestamp("2026-01-01 13:00", tz="UTC")) == "overlap"
    assert session_of(pd.Timestamp("2026-01-01 18:00", tz="UTC")) == "ny"
    assert session_of(pd.Timestamp("2026-01-01 23:00", tz="UTC")) == "offhours"


def test_route_table_covers_all_combinations():
    # Every structure x volatility x phase gets a defined route, never a crash.
    for struct in ("clean_trend", "range", "compression", "breakout",
                   "reversal", "choppy", "unclear"):
        for vol in ("low", "normal", "high", "spike", "unclear"):
            for phase in ("accumulation", "manipulation", "distribution",
                          "transition", "unclear"):
                for direction in ("up", "down", "flat"):
                    route, tradeable, _ = route_for_regime(
                        direction, struct, vol, phase, "london")
                    assert route in ("TREND_PULLBACK",
                                     "SWEEP_DISPLACEMENT_RETRACEMENT",
                                     "BREAKOUT_RETEST", "RANGE_EDGES",
                                     "SIT_OUT"), (struct, vol, phase)
                    if route == "SIT_OUT":
                        assert tradeable is False
                    if struct == "unclear":
                        assert route == "SIT_OUT"
                    if vol == "spike":
                        assert route == "SIT_OUT"


def test_spike_always_sits_out():
    route, tradeable, _ = route_for_regime(
        "up", "clean_trend", "spike", "distribution", "ny")
    assert route == "SIT_OUT" and tradeable is False


def test_regime_is_frozen_and_has_no_entry_tools():
    reg = classify_regime(uptrend_bars())
    assert isinstance(reg, Regime)
    with pytest.raises(Exception):
        reg.direction = "down"  # frozen dataclass
    # Bedrock must not reference entry concepts in its reason.
    assert "FVG" not in reg.reason
    assert "Order Block" not in reg.reason
