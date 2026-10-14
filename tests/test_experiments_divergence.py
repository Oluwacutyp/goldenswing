"""Tests for RSI divergence detection (features/divergence.py)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goldenswing.features import divergence as D
from goldenswing.features import gating
from goldenswing.features.config import FeatureConfig
from goldenswing.features.engine import FeatureEngine
from goldenswing.features.state import DivergenceState


def _df(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.004, n)
    close = 2000 * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0.002, 0.001, n)) * close
    o = close * (1 + rng.normal(0, 0.0005, n))
    h = np.maximum(o, close) + spread / 2
    l = np.minimum(o, close) - spread / 2
    v = rng.integers(100, 1000, n).astype(float)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                         "volume": v})


def _divergence_df() -> tuple[pd.DataFrame, np.ndarray]:
    """40 bars with a textbook regular-bullish divergence baked in:
    swing low A at bar 10 (price 100, RSI 30), swing low B at bar 25
    (price 98 = lower low, RSI 40 = higher low).  Confirmed at bar 27.
    RSI is injected directly (not derived from closes) so the fixture
    is deterministic."""
    n = 40
    low = np.full(n, 105.0)
    low[10] = 100.0
    low[25] = 98.0
    high = low + 2.0
    close = (high + low) / 2
    open_ = close.copy()
    rsi = np.full(n, 50.0)
    rsi[10] = 30.0
    rsi[25] = 40.0
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close,
                       "volume": np.full(n, 500.0)})
    return df, rsi


# ------------------------------------------------------------- RSI unit

def test_rsi_flat_series_is_neutral():
    r = D.rsi_wilder(np.full(100, 2000.0), 14)
    assert np.all(r[14:] == 50.0)


def test_rsi_rising_series_above_50():
    c = 2000.0 + np.arange(100, dtype=float)
    r = D.rsi_wilder(c, 14)
    assert r[-1] > 70.0


def test_rsi_is_causal():
    rng = np.random.default_rng(7)
    c = 2000 * np.exp(np.cumsum(rng.normal(0, 0.004, 120)))
    full = D.rsi_wilder(c, 14)
    trunc = D.rsi_wilder(c[:60], 14)
    assert np.allclose(full[:60], trunc)


# ---------------------------------------------------------- pivot unit

def test_fractal_pivot_confirmation_lag():
    df, _ = _divergence_df()
    highs, lows = D.fractal_pivots(df["high"].to_numpy(),
                                   df["low"].to_numpy(), left=3, right=2)
    assert (10, 12) in lows     # pivot at 10, confirmed at 12
    assert (25, 27) in lows     # pivot at 25, confirmed at 27
    assert highs == []


# ------------------------------------------------------ divergence unit

def test_detect_regular_bullish_divergence():
    df, rsi = _divergence_df()
    divs = D.detect_divergences(df["high"].to_numpy(),
                               df["low"].to_numpy(), rsi, 14)
    assert len(divs) == 1
    dv = divs[0]
    assert dv.kind == "bull_reg"
    assert dv.pivot_a == 10 and dv.pivot_b == 25
    assert dv.confirmed_at == 27
    assert dv.direction == 1
    assert dv.is_regular


def test_min_drsi_filters_noise():
    df, rsi = _divergence_df()
    divs = D.detect_divergences(df["high"].to_numpy(),
                               df["low"].to_numpy(), rsi, 14,
                               min_drsi=20.0)  # |40-30| = 10 < 20
    assert divs == []


def test_hidden_bearish_classification():
    # swing highs: A at 10 (price 100, RSI 60), B at 25 (price 98 =
    # lower high, RSI 70 = higher high) -> bear_hid
    n = 40
    high = np.full(n, 95.0)
    high[10] = 100.0
    high[25] = 98.0
    low = high - 2.0
    rsi = np.full(n, 50.0)
    rsi[10] = 60.0
    rsi[25] = 70.0
    divs = D.detect_divergences(high, low, rsi, 14)
    assert len(divs) == 1
    assert divs[0].kind == "bear_hid"
    assert divs[0].direction == -1
    assert not divs[0].is_regular


# ------------------------------------------------------------ causality

def test_divergence_appears_only_at_confirmation():
    df, rsi = _divergence_df()
    cfg = FeatureConfig()
    states = [SimpleNamespace()] * len(df)
    out = D.compute(df, states, cfg, rsi_override=rsi)
    assert out[26].divergences == ()      # confirmed at 27
    assert len(out[27].divergences) == 1  # visible from 27
    assert out[27].divergences[0].kind == "bull_reg"


def test_divergence_expires_after_max_age():
    df, rsi = _divergence_df()
    cfg = FeatureConfig(div_max_age=5)
    states = [SimpleNamespace()] * len(df)
    out = D.compute(df, states, cfg, rsi_override=rsi)
    assert len(out[27].divergences) == 1
    assert out[27 + 5].divergences != ()
    assert out[27 + 6].divergences == ()


def test_compute_state_length_mismatch_raises():
    df = _df(50)
    with pytest.raises(ValueError):
        D.compute(df, [SimpleNamespace()] * 10, FeatureConfig())


# --------------------------------------------------------------- gate

def _fs_with(*divs) -> SimpleNamespace:
    return SimpleNamespace(
        divergence=DivergenceState(divergences=tuple(divs), rsi_period=14))


def _dv(kind: str) -> D.Divergence:
    return D.Divergence(kind=kind, pivot_a=1, pivot_b=10, confirmed_at=12,
                        price_a=100.0, price_b=99.0, rsi_a=30.0, rsi_b=40.0,
                        rsi_period=14)


def test_gate_aligned_divergence_scores_plus():
    deltas = gating.divergence_gate(None, _fs_with(_dv("bull_reg")),
                                    1, "continuation")
    assert len(deltas) == 1
    assert deltas[0][0] == pytest.approx(4.0)
    assert "bull_reg" in deltas[0][1]


def test_gate_opposing_divergence_scores_minus():
    deltas = gating.divergence_gate(None, _fs_with(_dv("bear_hid")),
                                    1, "continuation")
    assert len(deltas) == 1
    assert deltas[0][0] == pytest.approx(-4.0)


def test_gate_silent_without_divergence():
    assert gating.divergence_gate(None, _fs_with(), 1, "continuation") == []


# -------------------------------------------------------------- wiring

def test_engine_attaches_divergence_state():
    from goldenswing.reader.reader import MarketReader
    df = _df(300)
    states = MarketReader().read(df)
    FeatureEngine().attach(df, states, None)
    assert isinstance(states[150].features.divergence, DivergenceState)
    assert states[150].features.divergence.rsi_period == 14


def test_engine_respects_compute_switch():
    from goldenswing.reader.reader import MarketReader
    df = _df(300)
    states = MarketReader().read(df)
    FeatureEngine(FeatureConfig(divergence=False)).attach(df, states, None)
    assert states[150].features.divergence.divergences == ()


def test_rsi9_config_flows_through():
    from goldenswing.reader.reader import MarketReader
    df = _df(300)
    states = MarketReader().read(df)
    FeatureEngine(FeatureConfig(div_rsi_period=9)).attach(df, states, None)
    assert states[150].features.divergence.rsi_period == 9
