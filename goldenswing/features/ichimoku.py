"""Ichimoku Kinko Hyo, computed causally.

Tenkan (9) and Kijun (26) are rolling midpoints of the highest high /
lowest low.  Senkou Span A/B are *plotted* 26 bars ahead on a chart,
which means the cloud a trader reads AT bar ``t`` was computed from
data at ``t - 26`` — that is exactly what we store: ``cloud_top`` /
``cloud_bottom`` at ``t`` are the span values derived from bars
``<= t - 26``.  No future data is ever touched.

States: price above / inside / below the known cloud, TK crosses
(Tenkan crossing Kijun), cloud twists (span A crossing span B in the
known series), and Chikou position (current close vs the candle 26
bars ago, the standard Chikou reading).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import ohlcv_arrays, rolling_max, rolling_min
from .config import FeatureConfig
from .state import IchimokuState


def _midpoint(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    return 0.5 * (rolling_max(high, period) + rolling_min(low, period))


def _d1_cloud(df: pd.DataFrame, cfg: FeatureConfig):
    """The DAILY cloud mapped causally onto the H1 bars (Phase 4).

    Daily bars are resampled from the H1 frame; a daily bar's *known*
    cloud uses the raw spans ``displacement`` daily bars back (same
    convention as the H1 cloud), and it only becomes visible once
    that daily bar has COMPLETED — an H1 bar uses the last daily bar
    whose end is <= its own end (mirrors features/mtf.py).  Returns
    (d1_position per H1 bar, d1_cloud_bullish per H1 bar), where the
    position compares the H1 close to the mapped daily cloud.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    d = df.copy()
    d.index = idx
    sub = d.resample("1D", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna(subset=["open", "high", "low", "close"])
    n = len(df)
    if len(sub) == 0:
        return ["unknown"] * n, [False] * n
    hi = sub["high"].to_numpy(dtype=float)
    lo = sub["low"].to_numpy(dtype=float)
    tenkan = _midpoint(hi, lo, cfg.tenkan_period)
    kijun = _midpoint(hi, lo, cfg.kijun_period)
    span_a = 0.5 * (tenkan + kijun)
    span_b = _midpoint(hi, lo, cfg.senkou_b_period)
    disp = cfg.displacement
    tops, bots, bulls = [], [], []
    for j in range(len(sub)):
        if j - disp >= cfg.senkou_b_period - 1:
            sa, sb = float(span_a[j - disp]), float(span_b[j - disp])
            tops.append(max(sa, sb))
            bots.append(min(sa, sb))
            bulls.append(sa > sb)
        else:
            tops.append(float("nan"))
            bots.append(float("nan"))
            bulls.append(False)
    known_at = (sub.index + pd.Timedelta(days=1)).to_numpy()
    keys = (idx + pd.Timedelta(hours=1)).to_numpy()
    pos_idx = np.searchsorted(known_at, keys, side="right") - 1
    close = df["close"].to_numpy(dtype=float)
    positions, bullish = [], []
    for t in range(n):
        p = int(pos_idx[t])
        if p < 0 or tops[p] != tops[p]:
            positions.append("unknown")
            bullish.append(False)
            continue
        bullish.append(bool(bulls[p]))
        if close[t] > tops[p]:
            positions.append("above")
        elif close[t] < bots[p]:
            positions.append("below")
        else:
            positions.append("inside")
    return positions, bullish


def compute(df: pd.DataFrame, states, cfg: FeatureConfig) -> list[IchimokuState]:
    a = ohlcv_arrays(df)
    high, low, close = a["high"], a["low"], a["close"]
    n = len(df)
    d1_pos, d1_bull = _d1_cloud(df, cfg)
    tenkan = _midpoint(high, low, cfg.tenkan_period)
    kijun = _midpoint(high, low, cfg.kijun_period)
    span_b_raw = _midpoint(high, low, cfg.senkou_b_period)
    span_a_raw = 0.5 * (tenkan + kijun)
    d = cfg.displacement

    out: list[IchimokuState] = []
    for t in range(n):
        # cloud known at t comes from raw spans at t-d
        if t - d >= cfg.senkou_b_period - 1:
            sa, sb = float(span_a_raw[t - d]), float(span_b_raw[t - d])
            c_top, c_bot = max(sa, sb), min(sa, sb)
            bullish = sa > sb
            if close[t] > c_top:
                pos = "above"
            elif close[t] < c_bot:
                pos = "below"
            else:
                pos = "inside"
        else:
            sa = sb = c_top = c_bot = float("nan")
            bullish, pos = False, "unknown"

        tk = 0
        if t >= 1 and np.isfinite(tenkan[t]) and np.isfinite(tenkan[t - 1]):
            if tenkan[t - 1] <= kijun[t - 1] and tenkan[t] > kijun[t]:
                tk = 1
            elif tenkan[t - 1] >= kijun[t - 1] and tenkan[t] < kijun[t]:
                tk = -1

        twist = 0
        if t - d - 1 >= cfg.senkou_b_period - 1:
            prev_diff = float(span_a_raw[t - d - 1] - span_b_raw[t - d - 1])
            diff = sa - sb
            if prev_diff <= 0 < diff:
                twist = 1
            elif prev_diff >= 0 > diff:
                twist = -1

        chikou = 0
        if t - d >= 0:
            if close[t] > high[t - d]:
                chikou = 1
            elif close[t] < low[t - d]:
                chikou = -1

        out.append(IchimokuState(
            tenkan=float(tenkan[t]), kijun=float(kijun[t]),
            cloud_top=c_top, cloud_bottom=c_bot,
            cloud_bullish=bool(bullish), position=pos,
            tk_cross=tk, cloud_twist=twist, chikou=chikou,
            d1_position=d1_pos[t], d1_cloud_bullish=d1_bull[t],
        ))
    return out
