"""Shared causal helpers for the feature layer.

Everything here is trailing-only: the value at bar ``t`` is a function of
bars ``<= t`` (or, for peer alignment, of peer bars whose timestamp is
``<=`` the gold bar's timestamp).  No centred windows, no full-series
normalisation — the truncation test in tests/test_features_no_lookahead.py
proves the whole layer honours that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ohlcv_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "open": df["open"].to_numpy(dtype=float),
        "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float),
        "close": df["close"].to_numpy(dtype=float),
        "volume": (df["volume"].to_numpy(dtype=float)
                   if "volume" in df.columns
                   else np.full(len(df), np.nan)),
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean including the current bar; warm-up uses the bars
    that exist (min_periods=1).  NaNs are skipped per-window."""
    s = pd.Series(values)
    return s.rolling(window, min_periods=1).mean().to_numpy(dtype=float)


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(values)
    return s.rolling(window, min_periods=2).std(ddof=1).to_numpy(dtype=float)


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(values)
    return s.rolling(window, min_periods=1).max().to_numpy(dtype=float)


def rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(values)
    return s.rolling(window, min_periods=1).min().to_numpy(dtype=float)


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(values)
    return s.rolling(window, min_periods=1).sum().to_numpy(dtype=float)


def align_peer_close(gold_index: pd.DatetimeIndex,
                     peer: pd.DataFrame | None) -> np.ndarray:
    """Causal as-of alignment of a peer close series onto gold bars.

    For each gold bar starting at ``ts`` the value is the close of the
    last peer bar whose *end* is ``<= ts + 1h`` (i.e. fully completed by
    the gold bar's close — the moment features are evaluated).  Peer bar
    timestamps are bar starts on an hourly grid, so a peer bar starting
    at ``ts`` ends exactly at the gold bar's close and is eligible.
    Returns all-NaN when the peer frame is missing/empty.
    """
    n = len(gold_index)
    if peer is None or len(peer) == 0 or "close" not in peer.columns:
        return np.full(n, np.nan)
    p = peer.sort_index()
    p_index = p.index
    if getattr(p_index, "tz", None) is None:
        p_index = p_index.tz_localize("UTC")
    else:
        p_index = p_index.tz_convert("UTC")
    g = gold_index
    if getattr(g, "tz", None) is None:
        g = g.tz_localize("UTC")
    else:
        g = g.tz_convert("UTC")
    # Compare bar ENDS on both sides: the eligible peer bar is the last
    # one whose end (start + 1h on the hourly grid) is <= the gold
    # bar's end.  Comparing against peer *starts* would admit the peer
    # bar that merely begins at the gold bar's close — a 1-bar leak.
    left = pd.DataFrame({"key": g + pd.Timedelta(hours=1)})
    right = pd.DataFrame({"key": p_index + pd.Timedelta(hours=1),
                          "close": p["close"].to_numpy(dtype=float)})
    merged = pd.merge_asof(left, right, on="key", direction="backward")
    return merged["close"].to_numpy(dtype=float)


def utc_day_starts(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer day id (days since epoch, UTC) per bar — for session VWAP
    resets and other per-day grouping without string keys."""
    idx = index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    raw = np.asarray(idx.normalize().view("int64"))
    return raw // 86_400_000_000_000
