"""Volume profile: POC / value area / high-low volume nodes.

The owner's bot computes this from OHLCV bars (``calculate_volume_profile``):
each bar's volume is spread across the price bins its range touches,
POC is the highest-volume bin, the value area holds 70% of volume,
and the top/bottom 3 bins are the high/low volume nodes.  This module
is that computation, unchanged in structure, over this engine's
frames (MT5 ``volume`` = tick volume).

The profile is DISPATCH CONTEXT, never a gate: the order-selection
layer uses it to prefer LIMIT at thick levels (high-volume node /
near POC) and MARKET in thin air.  A missing/degenerate frame returns
``poc=None`` — never a fabricated level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Price bins across the frame's range (the old bot's 24).
PROFILE_BINS = 24
#: Value-area share of total volume (the old bot's 70%).
VALUE_AREA_PCT = 0.70
#: Nodes reported at each extreme (the old bot's top/bottom 3).
NODES = 3
#: Minimum bars for a meaningful profile.
MIN_BARS = 10


def volume_profile(frame: pd.DataFrame | None, *,
                   bins: int = PROFILE_BINS,
                   va_pct: float = VALUE_AREA_PCT) -> dict:
    """POC / VAH / VAL / high-low nodes for ``frame``.

    Returns a dict with ``poc``, ``vah``, ``val`` (None when
    unmeasurable), ``high_vol_nodes`` / ``low_vol_nodes`` (lists of
    prices), and ``status``.  Never raises on a degenerate frame.
    """
    empty = {"poc": None, "vah": None, "val": None,
             "high_vol_nodes": [], "low_vol_nodes": [],
             "status": "no_data"}
    if frame is None or len(frame) < MIN_BARS:
        return {**empty, "status": "insufficient bars"}
    try:
        if not {"high", "low", "volume"}.issubset(frame.columns):
            return {**empty, "status": "missing columns"}
        high = frame["high"].to_numpy(dtype=float)
        low = frame["low"].to_numpy(dtype=float)
        volume = frame["volume"].to_numpy(dtype=float)
        if not np.isfinite(high).all() or not np.isfinite(low).all():
            return {**empty, "status": "non-finite prices"}
        lo, hi = float(low.min()), float(high.max())
        if hi <= lo:
            return {**empty, "status": "flat range"}
        edges = np.linspace(lo, hi, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_vol = np.zeros(bins)
        for i in range(len(frame)):
            touched = np.where((centers >= low[i])
                               & (centers <= high[i]))[0]
            if len(touched):
                bin_vol[touched] += volume[i] / len(touched)
        total = float(bin_vol.sum())
        if total <= 0:
            return {**empty, "status": "no volume"}
        poc = float(centers[int(np.argmax(bin_vol))])
        order = np.argsort(bin_vol)[::-1]
        va_bins, acc = [], 0.0
        for idx in order:
            if acc >= total * va_pct:
                break
            va_bins.append(idx)
            acc += bin_vol[idx]
        va_prices = centers[va_bins]
        return {
            "poc": poc,
            "vah": float(va_prices.max()) if len(va_prices) else poc,
            "val": float(va_prices.min()) if len(va_prices) else poc,
            "high_vol_nodes": [float(centers[i])
                               for i in order[:NODES]],
            "low_vol_nodes": [float(centers[i])
                              for i in order[-NODES:]],
            "status": "ok",
        }
    except Exception as exc:
        return {**empty, "status": f"failed: {exc}"}


def near_level(price: float, levels, *, tolerance: float) -> bool:
    """True if ``price`` is within ``tolerance`` (price units) of any
    level in ``levels``.  Tolerance is the caller's ATR fraction —
    never pips (the old bot's pip tolerance is wrong for gold)."""
    if price is None or tolerance is None or tolerance <= 0:
        return False
    return any(abs(float(price) - float(lvl)) <= tolerance
               for lvl in (levels or []))
