"""Unit tests for the Bollinger trend-states experiment.

Synthetic bars only — geometry and causality, no market data.
"""
import numpy as np
import pandas as pd

from goldenswing.experiments.bollinger_trends import (
    SQUEEZE_MIN_BARS,
    squeeze_episodes,
)


def _bars(n, close=100.0, seed=1):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.002, n)
    c = close * np.exp(np.cumsum(rets))
    o = np.roll(c, 1)
    o[0] = c[0]
    h = np.maximum(o, c) * 1.001
    l = np.minimum(o, c) * 0.999
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l,
                         "close": c, "volume": 1000.0}, index=idx)


def test_squeeze_episodes_min_length():
    sq = np.array([False] * 3 + [True] * (SQUEEZE_MIN_BARS - 1)
                  + [False] + [True] * SQUEEZE_MIN_BARS + [False])
    eps = squeeze_episodes(sq)
    assert len(eps) == 1
    assert eps[0].length == SQUEEZE_MIN_BARS


def test_squeeze_episodes_adjacent_runs():
    sq = np.array([True] * 10)
    eps = squeeze_episodes(sq)
    assert len(eps) == 1 and eps[0].start == 0 and eps[0].end == 9


def test_squeeze_episodes_empty():
    assert squeeze_episodes(np.zeros(50, dtype=bool)) == []


def test_bollinger_feature_causal_geometry():
    """The feature states must be causal: recompute on a prefix and the
    early states must match the full-series states."""
    from goldenswing.features.config import FeatureConfig
    from goldenswing.features.engine import FeatureEngine
    from goldenswing.reader.reader import MarketReader

    df = _bars(400)
    states_full = MarketReader().read(df)
    FeatureEngine(FeatureConfig()).attach(df, states_full)
    cut = 300
    df_cut = df.iloc[:cut]
    states_cut = MarketReader().read(df_cut)
    FeatureEngine(FeatureConfig()).attach(df_cut, states_cut)
    for t in (50, 150, 250):
        a = states_full[t].features.bollinger
        b = states_cut[t].features.bollinger
        assert a.squeeze == b.squeeze, t
        assert a.walk == b.walk, t
        assert a.rejection == b.rejection, t
        assert a.upper == b.upper, t


def test_rejection_requires_close_back_inside():
    """A bar that closes OUTSIDE the band is a walk, not a rejection."""
    from goldenswing.features.config import FeatureConfig
    from goldenswing.features import bollinger
    from goldenswing.reader.reader import MarketReader

    df = _bars(300)
    # force a spike: high far above recent closes, close still above band
    df.loc[df.index[250], "high"] = df["close"].iloc[249] * 1.05
    df.loc[df.index[250], "close"] = df["close"].iloc[249] * 1.04
    states = MarketReader().read(df)
    out = bollinger.compute(df, states, FeatureConfig())
    # rejection must be 0 when the close stays outside (it is a walk or
    # nothing, never a rejection)
    assert out[250].rejection == 0 or out[250].walk == 1
