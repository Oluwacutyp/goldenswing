"""Volume character: relative volume, climax, dry-up, session VWAP.

GC=F volume is real COMEX futures volume for most bars; where the feed
has zeros/NaNs the features degrade — RVOL becomes NaN and
``available`` drops — rather than inventing numbers.  A series counts
as usable when at least half its bars carry positive finite volume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import ohlcv_arrays, rolling_max, rolling_mean, rolling_min, \
    utc_day_starts
from .config import FeatureConfig
from .state import VolumeState


def volume_arrays(df: pd.DataFrame, cfg: FeatureConfig):
    """Return (rvol, vwap, available) arrays aligned to df.

    Shared with the footprints family (institutional climax checks and
    CTA VWAP-side persistence read the same series)."""
    a = ohlcv_arrays(df)
    vol = a["volume"]
    n = len(df)
    finite_pos = np.isfinite(vol) & (vol > 0)
    available = bool(n > 0 and finite_pos.mean() >= 0.5)
    if not available:
        return (np.full(n, np.nan), np.full(n, np.nan), False)

    clean = np.where(finite_pos, vol, np.nan)
    mean_vol = pd.Series(clean).rolling(
        cfg.vol_window, min_periods=1).mean().to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rvol = np.where(finite_pos & (mean_vol > 0), vol / mean_vol, np.nan)

    # session (UTC day) VWAP on typical price, zero-volume bars carried
    # at the running VWAP (or the close before any volume prints).
    tp = (a["high"] + a["low"] + a["close"]) / 3.0
    days = utc_day_starts(df.index if isinstance(df.index, pd.DatetimeIndex)
                          else pd.to_datetime(df.index, utc=True))
    vwap = np.full(n, np.nan)
    cum_pv = cum_v = 0.0
    cur_day = None
    last = float("nan")
    for t in range(n):
        if days[t] != cur_day:
            cur_day = days[t]
            cum_pv = cum_v = 0.0
            last = float("nan")
        v = vol[t] if finite_pos[t] else 0.0
        cum_pv += tp[t] * v
        cum_v += v
        if cum_v > 0:
            last = cum_pv / cum_v
        vwap[t] = last if np.isfinite(last) else a["close"][t]
    return rvol, vwap, True


def compute(df: pd.DataFrame, states, cfg: FeatureConfig) -> list[VolumeState]:
    a = ohlcv_arrays(df)
    high, low, close, open_ = a["high"], a["low"], a["close"], a["open"]
    n = len(df)
    rvol, vwap, available = volume_arrays(df, cfg)
    hi10 = rolling_max(high, 10)
    lo10 = rolling_min(low, 10)

    out: list[VolumeState] = []
    for t in range(n):
        rv = float(rvol[t]) if np.isfinite(rvol[t]) else float("nan")
        climax = bool(available and np.isfinite(rv) and rv >= cfg.vol_climax_rvol)
        reversal = 0
        if climax:
            rng = high[t] - low[t]
            pos = (close[t] - low[t]) / rng if rng > 0 else 0.5
            if low[t] <= lo10[t] and pos >= 0.6 and close[t] > open_[t]:
                reversal = 1        # climax low, buyers took the close
            elif high[t] >= hi10[t] and pos <= 0.4 and close[t] < open_[t]:
                reversal = -1
        out.append(VolumeState(
            available=available,
            rvol=rv,
            climax=climax,
            climax_reversal=reversal,
            dry_up=bool(available and np.isfinite(rv)
                        and rv <= cfg.vol_dry_rvol),
            vwap=float(vwap[t]) if np.isfinite(vwap[t]) else float("nan"),
        ))
    return out
