"""Bollinger Bands (20, 2σ): %B, bandwidth + its trailing percentile,
and the three trend states the desk actually uses:

- **squeeze** — bandwidth percentile < 20: volatility coiled, expansion
  is statistically due (direction unknown from the bands alone);
- **walk-the-band** — trend so persistent that closes keep riding the
  outer band (>= 2 of the last 3 closes at/beyond the band, current
  close included);
- **band-rejection wick** — price poked through a band and closed back
  inside: the band acted as a wall, not a launchpad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..reader import indicators
from .common import ohlcv_arrays, rolling_mean, rolling_std
from .config import FeatureConfig
from .state import BollingerState


def compute(df: pd.DataFrame, states, cfg: FeatureConfig) -> list[BollingerState]:
    a = ohlcv_arrays(df)
    close, high, low = a["close"], a["high"], a["low"]
    n = len(df)
    mid = rolling_mean(close, cfg.bb_period)
    sd = rolling_std(close, cfg.bb_period)
    upper = mid + cfg.bb_mult * sd
    lower = mid - cfg.bb_mult * sd
    width = upper - lower
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_b = (close - lower) / width
        bandwidth = width / mid
    bw_pct = indicators.rolling_percentile(
        np.nan_to_num(bandwidth, nan=0.0), cfg.bb_pct_window)

    out: list[BollingerState] = []
    for t in range(n):
        walk = 0
        # a zero-width band (dead-flat window) rides nothing: every close
        # equals both bands, so walk/rejection require real width.
        has_width = np.isfinite(upper[t]) and np.isfinite(lower[t]) \
            and upper[t] > lower[t]
        if t >= 2 and has_width:
            up_hits = sum(1 for k in (t - 2, t - 1, t) if close[k] >= upper[k])
            dn_hits = sum(1 for k in (t - 2, t - 1, t) if close[k] <= lower[k])
            if close[t] >= upper[t] and up_hits >= 2:
                walk = 1
            elif close[t] <= lower[t] and dn_hits >= 2:
                walk = -1
        rejection = 0
        if has_width:
            if high[t] > upper[t] and close[t] < upper[t]:
                rejection = -1
            elif low[t] < lower[t] and close[t] > lower[t]:
                rejection = 1
        out.append(BollingerState(
            mid=float(mid[t]), upper=float(upper[t]), lower=float(lower[t]),
            pct_b=float(pct_b[t]) if np.isfinite(pct_b[t]) else float("nan"),
            bandwidth=float(bandwidth[t]) if np.isfinite(bandwidth[t]) else float("nan"),
            bw_percentile=float(bw_pct[t]),
            squeeze=bool(bw_pct[t] < cfg.bb_squeeze_pct),
            walk=walk, rejection=rejection,
        ))
    return out
