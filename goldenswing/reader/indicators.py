"""Causal indicator helpers (rolling, past-only windows)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR via causal EWM."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Causal EWM (alpha = 1/period, adjust=False) matching pandas."""
    n = len(values)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    alpha = 1.0 / period
    acc = None
    for i in range(n):
        v = values[i]
        if not np.isfinite(v):
            out[i] = acc if acc is not None else np.nan
            continue
        acc = v if acc is None else alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def rolling_percentile(values: np.ndarray, window: int) -> np.ndarray:
    """Percentile (0..100) of each value within its trailing window,
    including itself.  Warm-up bars use whatever history exists."""
    n = len(values)
    out = np.full(n, 50.0)
    for t in range(n):
        start = max(0, t - window + 1)
        w = values[start : t + 1]
        w = w[np.isfinite(w)]
        if len(w) == 0 or not np.isfinite(values[t]):
            out[t] = 50.0
        else:
            out[t] = 100.0 * float(np.mean(w <= values[t]))
    return out


def body_ratio(df: pd.DataFrame, atr_s: pd.Series) -> np.ndarray:
    """Signed candle body in ATR units: (close-open)/ATR."""
    body = (df["close"].to_numpy(dtype=float) - df["open"].to_numpy(dtype=float))
    a = atr_s.to_numpy(dtype=float)
    a = np.where(a > 0, a, np.nan)
    return body / a
