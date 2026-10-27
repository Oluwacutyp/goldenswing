import pandas as pd

from scripts.replay_hot_scalp import _closed_asof


def test_replay_never_exposes_a_forming_candle():
    frame = pd.DataFrame(
        {"open": [1, 2, 3], "high": [2, 3, 4], "low": [0, 1, 2],
         "close": [1, 2, 3], "volume": [1, 1, 1]},
        index=pd.date_range("2026-10-27T00:00:00Z", periods=3, freq="1min"),
    )
    before_close = _closed_asof(frame, "M1", pd.Timestamp("2026-10-27T00:02:59Z"))
    at_close = _closed_asof(frame, "M1", pd.Timestamp("2026-10-27T00:03:00Z"))
    assert list(before_close["close"]) == [1, 2]
    assert list(at_close["close"]) == [1, 2, 3]
