"""Bollinger: band math, squeeze, walk-the-band, rejection wicks."""
import numpy as np

from conftest import make_df
from goldenswing.features import bollinger
from goldenswing.features.config import FeatureConfig

CFG = FeatureConfig()


def _rows_from_closes(closes, pad=0.2):
    rows = []
    prev = closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + pad, min(prev, c) - pad, c))
        prev = c
    return rows


def test_band_math_on_flat_series():
    df = make_df(_rows_from_closes([2000.0] * 60))
    out = bollinger.compute(df, [None] * 60, CFG)
    s = out[-1]
    assert abs(s.mid - 2000.0) < 1e-9
    assert s.upper == s.lower == 2000.0 or abs(s.upper - s.lower) < 1e-9
    assert s.walk == 0 and s.rejection == 0


def test_pct_b_and_squeeze_after_volatility_collapse():
    rng = np.random.default_rng(5)
    volatile = 2000.0 + np.cumsum(rng.normal(0, 6.0, 120))
    # volatility DECAYS through the calm segment (as real collapses do),
    # so bandwidth trends down and the current value sits near the
    # trailing window's low — a mid-pack percentile would be wrong here.
    decay = 6.0 * 0.96 ** np.arange(140)
    calm = volatile[-1] + np.cumsum(rng.normal(0, 1.0, 140) * decay)
    closes = np.concatenate([volatile, calm]).tolist()
    df = make_df(_rows_from_closes(closes))
    out = bollinger.compute(df, [None] * len(df), CFG)
    last = out[-1]
    assert last.squeeze, "bandwidth percentile must be low after collapse"
    assert last.bw_percentile < 20.0
    assert 0.0 <= last.pct_b <= 1.0


def test_walk_the_band_on_explosive_jump():
    closes = [2000.0] * 60 + [2000.0 + 30.0 * k for k in range(1, 8)]
    df = make_df(_rows_from_closes(closes))
    out = bollinger.compute(df, [None] * len(df), CFG)
    assert any(s.walk == 1 for s in out[60:]), \
        "steep band-riding closes must register as walk-the-band"


def test_band_rejection_wick():
    # alternating closes give the bands real width (a lone deviating
    # close in a flat window is always >2 sigma away by construction)
    closes = [1999.0, 2001.0] * 20
    rows = _rows_from_closes(closes)
    # final bar: wick through the upper band, low inside, close back in
    rows.append((2000.0, 2006.0, 1999.5, 2000.5))
    df = make_df(rows)
    out = bollinger.compute(df, [None] * len(df), CFG)
    assert out[-1].rejection == -1
