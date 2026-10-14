"""Volume Profile for Version 2 confluence.

Plots volume horizontally against price levels to find where institutions
did the most business.

Components:
- POC (Point of Control): price with highest volume — magnet/barrier
- HVN (High Volume Node): thick profile = acceptance, S/R
- LVN (Low Volume Node): thin profile = fast move zone, rejection
- VAH/VAL (Value Area High/Low): 70% of volume range

Uses tick volume from OANDA (activity count, not exchange volume).
Good enough for identifying institutional interest zones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VolumeNode:
    """A price level with significant volume."""
    price: float
    volume: float
    kind: str  # "POC" | "HVN" | "LVN"


@dataclass
class VolumeProfile:
    """Volume Profile for a lookback window."""
    poc: float                      # Point of Control
    vah: float                      # Value Area High (70%)
    val: float                      # Value Area Low (70%)
    hvns: list[VolumeNode] = field(default_factory=list)
    lvns: list[VolumeNode] = field(default_factory=list)
    price_min: float = 0.0
    price_max: float = 0.0


def calculate_vp(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 volume: np.ndarray, num_bins: int = 24,
                 value_area_pct: float = 0.70) -> VolumeProfile | None:
    """Calculate Volume Profile from OHLCV data.

    Distributes each bar's volume across the price bins it traded through
    (simplified: all volume at the typical price).
    """
    if len(close) == 0 or volume.sum() <= 0:
        return None

    price_min = float(low.min())
    price_max = float(high.max())
    if price_max <= price_min:
        return None

    # Bin prices
    bins = np.linspace(price_min, price_max, num_bins + 1)
    bin_volume = np.zeros(num_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Distribute volume: typical price method
    typical = (high + low + close) / 3
    for i, (t, v) in enumerate(zip(typical, volume)):
        if v <= 0:
            continue
        b = np.digitize(t, bins) - 1
        b = max(0, min(num_bins - 1, b))
        bin_volume[b] += v

    total_vol = bin_volume.sum()
    if total_vol <= 0:
        return None

    # POC: bin with max volume
    poc_idx = int(np.argmax(bin_volume))
    poc = float(bin_centers[poc_idx])

    # Value Area: expand from POC until 70% of volume
    va_vol = bin_volume[poc_idx]
    lo_idx = hi_idx = poc_idx
    target = total_vol * value_area_pct

    while va_vol < target:
        # Expand to side with more volume
        lo_v = bin_volume[lo_idx - 1] if lo_idx > 0 else -1
        hi_v = bin_volume[hi_idx + 1] if hi_idx < num_bins - 1 else -1
        if lo_v < 0 and hi_v < 0:
            break
        if hi_v >= lo_v:
            hi_idx += 1
            va_vol += bin_volume[hi_idx]
        else:
            lo_idx -= 1
            va_vol += bin_volume[lo_idx]

    vah = float(bins[hi_idx + 1])
    val = float(bins[lo_idx])

    # HVN: bins with volume > 1.5x mean (excluding POC)
    mean_vol = bin_volume.mean()
    hvns = []
    lvns = []
    for i, bv in enumerate(bin_volume):
        if i == poc_idx:
            continue
        p = float(bin_centers[i])
        if bv > 1.5 * mean_vol:
            hvns.append(VolumeNode(price=p, volume=float(bv), kind="HVN"))
        elif bv < 0.5 * mean_vol and bv > 0:
            lvns.append(VolumeNode(price=p, volume=float(bv), kind="LVN"))

    return VolumeProfile(
        poc=poc, vah=vah, val=val,
        hvns=hvns, lvns=lvns,
        price_min=price_min, price_max=price_max,
    )


def vp_confluence(profile: VolumeProfile, price: float, atr: float,
                  tol_atr: float = 0.3) -> list[str]:
    """Check which VP levels are near a price. Returns list of confluences."""
    if atr <= 0 or profile is None:
        return []
    out = []
    if abs(profile.poc - price) <= tol_atr * atr:
        out.append("POC")
    if abs(profile.vah - price) <= tol_atr * atr:
        out.append("VAH")
    if abs(profile.val - price) <= tol_atr * atr:
        out.append("VAL")
    for hvn in profile.hvns:
        if abs(hvn.price - price) <= tol_atr * atr:
            out.append("HVN")
            break
    return out
