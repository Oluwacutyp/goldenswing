"""Regression tests for pandas Copy-on-Write backed OHLC frames.

The Wilder ATR warm-up intentionally fills the first 13 values with
short-history TR means.  Its destination must therefore be writable,
even when pandas returns a read-only view from Series.to_numpy().
"""
import numpy as np
import pandas as pd
import pytest

from goldenswing.live.vol_expansion import PreparedFrame as VolPrepared
from goldenswing.live.trend_momentum import PreparedFrame as TrendPrepared
from goldenswing.live.auction_value import PreparedAuction


@pytest.fixture
def bars():
    idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    close = np.linspace(100.0, 112.0, len(idx))
    return pd.DataFrame({
        "open": close - 0.15,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close + 0.15,
        "volume": np.full(len(idx), 100.0),
    }, index=idx)


@pytest.mark.parametrize("prepared", [VolPrepared, TrendPrepared, PreparedAuction])
def test_prepared_atr_is_writable_under_pandas_copy_on_write(bars, prepared):
    with pd.option_context("mode.copy_on_write", True):
        result = prepared(bars)
    assert len(result._atr) == len(bars)
    assert np.isfinite(result._atr[:13]).all()
    assert result.atr(12) is not None
