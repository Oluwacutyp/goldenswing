"""Relative strength vs peers: silver (SI=F) and the dollar index
(DX-Y.NYB).

The RS ratio is gold_close / peer_close, aligned causally (the peer
bar must be fully completed by the gold bar's close — see
common.align_peer_close).  Trend is the sign of the ratio's rate of
change over ``rs_roc_bars`` outside a small deadband; percentile is
the ratio's trailing percentile.  For the dollar index a RISING ratio
means gold is outperforming the dollar — the same "gold strong"
reading as for silver, so both peers share one convention.

Missing/short peer data degrades to ``available=False`` with NaN
fields — never zeros, never invented values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..reader import indicators
from .common import align_peer_close
from .config import FeatureConfig
from .state import PeerRS, RSState

MIN_RATIO_BARS = 60     # valid ratio points before a peer counts as usable


def _peer_series(gold_close: np.ndarray, peer_close: np.ndarray,
                 cfg: FeatureConfig) -> list[PeerRS]:
    n = len(gold_close)
    empty = [PeerRS()] * n
    valid = np.isfinite(peer_close) & (peer_close > 0)
    if int(valid.sum()) < MIN_RATIO_BARS:
        return empty
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(valid, gold_close / peer_close, np.nan)
    # gaps filled causally (last known ratio); the percentile window is
    # trailing-only, so warm-up/gap handling cannot leak the future.
    filled = pd.Series(ratio).ffill().to_numpy(dtype=float)
    pct = indicators.rolling_percentile(
        np.nan_to_num(filled, nan=0.0), cfg.rs_pct_window)
    out: list[PeerRS] = []
    for t in range(n):
        if not np.isfinite(ratio[t]):
            out.append(PeerRS())
            continue
        trend = 0
        if t >= cfg.rs_roc_bars and np.isfinite(filled[t - cfg.rs_roc_bars]) \
                and filled[t - cfg.rs_roc_bars] > 0:
            roc = filled[t] / filled[t - cfg.rs_roc_bars] - 1.0
            if roc > cfg.rs_deadband:
                trend = 1
            elif roc < -cfg.rs_deadband:
                trend = -1
        out.append(PeerRS(available=True, ratio=float(ratio[t]),
                          trend=trend, percentile=float(pct[t])))
    return out


def compute(df: pd.DataFrame, states, cfg: FeatureConfig,
            peers: dict | None = None) -> list[RSState]:
    peers = peers or {}
    gold_close = df["close"].to_numpy(dtype=float)
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    silver = _peer_series(gold_close,
                          align_peer_close(idx, peers.get("silver")), cfg)
    dollar = _peer_series(gold_close,
                          align_peer_close(idx, peers.get("dollar")), cfg)
    out: list[RSState] = []
    for sv, dl in zip(silver, dollar):
        trends = [p.trend for p in (sv, dl) if p.available]
        score = float(np.mean(trends)) if trends else float("nan")
        out.append(RSState(silver=sv, dollar=dl, score=score,
                           available=bool(trends)))
    return out
