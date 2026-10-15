"""Urgency score (0-100) for order execution decisions.

The owner's bot's ``UrgencyScoreCalculator``: base 50 + session
(London +20, NY +30) + ATR velocity (fast moves +10/+20/+40) + RSI
(extreme +15, neutral -10) + zone proximity (+20/+40).  This module
is that formula adapted to what this engine actually has:

- Session from the engine's DST-aware session names (london +20,
  overlap/newyork +30 — overlap is London+NY both open, so it takes
  the NY weight; asia/off 0).
- ATR velocity: current ATR vs the mean of the last 14 ATR readings
  (same +10/+20/+40 bands at >5% / >15% / >30%).
- Zone proximity: price inside the POI zone +40; within 0.5 ATR of
  an edge +20; within 1.0 ATR +10 (the old bot's SMC OB/OS bands).
- NO RSI term: this engine computes no RSI in its features, and an
  urgency score must not invent one.  Dropped deliberately and
  documented, not silently approximated.

Urgency is DISPATCH CONTEXT, recorded on the verdict and the signal
record.  It never overrides the engine's no-chase discipline: it can
make execution MORE patient (market -> limit at a thick level), it
never converts a limit into a market at a worse-than-POI price (see
``select_order_type``).  Fail-open to 50.0 on any error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE = 50.0
SESSION_BONUS = {"london": 20.0, "overlap": 30.0, "newyork": 30.0}
ATR_FAST = 0.30
ATR_MID = 0.15
ATR_SLOW = 0.05


def _atr_series(frame: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Simple-moving-average ATR series over ``frame`` (self-consistent
    for the velocity ratio — current vs its own mean).  Wilder's ATR
    lives in live/multitf; the velocity needs a SERIES, not a point.
    """
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev), np.abs(low - prev)))
    if len(tr) < period:
        return np.array([float(np.mean(tr))]) if len(tr) else np.array([])
    return pd.Series(tr).rolling(period).mean().dropna().to_numpy(float)


def _zone_distance(price: float, zone: dict | None,
                   atr: float) -> float | None:
    """Distance from ``price`` to the POI zone in ATR units: 0 when
    inside the zone, else edge distance / ATR.  None when the zone
    or ATR is unusable."""
    if zone is None or atr is None or atr <= 0:
        return None
    try:
        bottom = float(zone.get("bottom", 0.0))
        top = float(zone.get("top", 0.0))
    except (TypeError, ValueError):
        return None
    if top <= bottom:
        return None
    if bottom <= price <= top:
        return 0.0
    return min(abs(price - bottom), abs(price - top)) / atr


def urgency_score(price: float, *, session: str | None,
                  zone: dict | None, frame: pd.DataFrame | None,
                  ) -> float:
    """Urgency 0-100 for an order decision at ``price``.

    ``session`` is the engine session name (asia/london/overlap/
    newyork/off); ``zone`` the POI band; ``frame`` the H1 closed
    bars (ATR velocity source).  Any missing input degrades its
    term to 0 — never a crash, never a fabricated term.
    """
    try:
        urgency = BASE
        urgency += SESSION_BONUS.get(session or "", 0.0)
        atr_now = None
        if frame is not None and len(frame) >= 2 \
                and {"high", "low", "close"}.issubset(frame.columns):
            series = _atr_series(frame)
            if len(series) >= 2:
                atr_now = float(series[-1])
                mean_atr = float(np.mean(series[-14:]))
                if mean_atr > 0:
                    vel = atr_now / mean_atr - 1.0
                    if vel > ATR_FAST:
                        urgency += 40.0
                    elif vel > ATR_MID:
                        urgency += 20.0
                    elif vel > ATR_SLOW:
                        urgency += 10.0
        if atr_now is None and frame is not None and len(frame) >= 2:
            # Too few bars for a velocity series: still measure the
            # zone distance on a single ATR reading.
            from .multitf import atr14

            atr_now = atr14(frame)
        dist = _zone_distance(float(price), zone, atr_now or 0.0)
        if dist is not None:
            if dist <= 0.0:
                urgency += 40.0
            elif dist < 0.5:
                urgency += 20.0
            elif dist < 1.0:
                urgency += 10.0
        return max(0.0, min(100.0, urgency))
    except Exception:
        return BASE
