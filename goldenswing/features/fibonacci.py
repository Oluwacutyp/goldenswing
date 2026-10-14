"""Auto-Fibonacci from the reader's dealing range.

Levels are drawn on the current dealing range (the reader's confirmed-
swing high/low inside its lookback): for the most recently completed
leg, retracement levels sit at high - frac * (high - low).  The OTE
(optimal trade entry) zone is the 0.62-0.79 retracement band — the
zone matters for the direction of the *completed* leg: after an up leg
(low->high), a pullback into the OTE band is the long entry zone
(``in_ote_long``); after a down leg, a bounce into it is the short
entry zone (``in_ote_short``).

Leg direction: whichever of the last confirmed swing high / swing low
was confirmed with the larger *swing index* marks the end of the last
completed leg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .state import FibState

FRACS = (0.382, 0.5, 0.618, 0.705, 0.79)
OTE_LO, OTE_HI = 0.62, 0.79


def compute(df: pd.DataFrame, states, cfg: FeatureConfig) -> list[FibState]:
    out: list[FibState] = []
    for s in states:
        r_hi, r_lo = s.range_high, s.range_low
        if not (np.isfinite(r_hi) and np.isfinite(r_lo)) or r_hi <= r_lo:
            out.append(FibState())
            continue
        span = r_hi - r_lo
        levels = tuple((f, r_hi - f * span) for f in FRACS)

        leg = 0
        last_hi = s.recent_swing_highs[-1] if s.recent_swing_highs else None
        last_lo = s.recent_swing_lows[-1] if s.recent_swing_lows else None
        if last_hi is not None and last_lo is not None:
            leg = 1 if last_hi.index > last_lo.index else -1
        elif last_hi is not None:
            leg = 1
        elif last_lo is not None:
            leg = -1

        nearest_frac, nearest_dist = float("nan"), float("inf")
        for f, price in levels:
            d = abs(s.close - price)
            if d < nearest_dist:
                nearest_frac, nearest_dist = f, d
        dist_atr = nearest_dist / s.atr if s.atr > 0 else float("nan")

        ote_high = r_hi - OTE_LO * span
        ote_low = r_hi - OTE_HI * span
        in_zone = ote_low <= s.close <= ote_high
        out.append(FibState(
            range_high=float(r_hi), range_low=float(r_lo),
            leg_direction=leg, levels=levels,
            nearest_frac=float(nearest_frac),
            nearest_dist_atr=float(dist_atr),
            at_fib=bool(np.isfinite(dist_atr)
                        and dist_atr <= cfg.fib_at_tol_atr),
            ote_low=float(ote_low), ote_high=float(ote_high),
            in_ote_long=bool(in_zone and leg == 1),
            in_ote_short=bool(in_zone and leg == -1),
        ))
    return out
