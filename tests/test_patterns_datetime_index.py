import numpy as np
import pandas as pd

from goldenswing.live.patterns import detect_sweep, fvg_list, order_blocks


def _market_frame(n=50):
    idx = pd.date_range("2026-10-26T00:00:00Z", periods=n, freq="5min")
    close = 100.0 + np.arange(n) * 0.03 + np.sin(np.arange(n)) * 0.08
    open_ = close - np.where(np.arange(n) % 2, 0.04, -0.04)
    high = np.maximum(open_, close) + 0.08
    low = np.minimum(open_, close) - 0.08
    # A displacement following an opposite candle gives the OB detector a
    # deterministic candidate, while the gap at bar 35 gives fvg_list one.
    open_[24], close[24] = 101.0, 100.7
    high[24], low[24] = 101.1, 100.6
    open_[25], close[25] = 100.8, 102.5
    high[25], low[25] = 102.7, 100.7
    high[33] = 101.0
    open_[35], low[35], close[35], high[35] = 102.0, 102.0, 102.5, 102.7
    # Make the last bar a sell-side sweep followed by a close back above level.
    open_[-1], high[-1], low[-1], close[-1] = 100.2, 100.5, 98.5, 100.3
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": 100.0}, index=idx)


def test_pattern_detectors_keep_integer_positions_and_datetime_timestamps():
    frame = _market_frame()
    blocks = order_blocks(frame)
    assert blocks, "fixture should produce an order block"
    assert isinstance(blocks[0]["bar_index"], int)
    assert isinstance(blocks[0]["ts"], pd.Timestamp)

    gaps = fvg_list(frame)
    assert gaps, "fixture should produce an FVG"
    assert isinstance(gaps[0]["bar_index"], int)
    assert isinstance(gaps[0]["ts"], pd.Timestamp)

    sweep = detect_sweep(frame, 100.0, direction=1, atr=1.0)
    assert sweep["present"]
    assert sweep["bar_index"] == len(frame) - 1
    assert sweep["ts"] == frame.index[-1]
