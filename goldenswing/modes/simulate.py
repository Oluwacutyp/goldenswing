"""Scalp-mode trade simulation: nearest-pool target, short horizon.

Same conservative conventions as the swing harness — signal at close
of bar t, entry at the OPEN of t+1, stop wins same-bar ties, cost in R
as cost / risk-per-unit — with the scalp exit shape from the spec:

- stop = the signal's structural invalidation, buffered by
  ``stop_buffer_atr`` x ATR (same buffer the v2 swing exits use, so
  the wider stop is paid for in size via R, not hidden);
- target = the NEAREST mapped liquidity pool ahead of entry.  No 2R
  floor: a scalp takes what the map offers.  A signal with NO pool
  ahead is not a scalp setup at all (no defined target) and is
  skipped;
- time stop at ``horizon`` bars.

Every result carries ``cost_r`` — the round-trip cost expressed in R
at that trade's stop distance.  At M15/M5 stop sizes this number is
the whole story, so it is reported per-trade and aggregated, never
buried inside r_net alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..backtest.harness import Outcome
from ..golden.detector import GoldenSignal


@dataclass
class ScalpOutcome(Outcome):
    cost_r: float = 0.0     # cost / risk-per-unit for this trade
    target_r: float = 0.0   # reward-to-risk of the pool target


def simulate_scalp(df: pd.DataFrame, sig: GoldenSignal, horizon: int = 8,
                   cost: float = 0.45, stop_buffer_atr: float = 0.25
                   ) -> ScalpOutcome | None:
    t = sig.index
    n = len(df)
    if t + 1 >= n:
        return None
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    d = sig.direction
    entry = open_[t + 1]
    buffer = stop_buffer_atr * sig.atr if sig.atr > 0 else 0.0
    stop = sig.invalidation - d * buffer
    risk = (entry - stop) * d
    if risk <= 0:
        return None

    ahead = [p for p in sig.targets if (p - entry) * d > 0]
    if not ahead:
        return None  # no mapped pool ahead — no scalp target, no trade
    target = float(min(ahead, key=lambda p: abs(p - entry)))
    if (open_[t + 1] - target) * d >= 0:
        return None  # already ran to the pool before entry

    last = min(t + horizon, n - 1)
    exit_price, exit_idx, reason = float(close[last]), last, "timeout"
    for k in range(t + 1, last + 1):
        if d == 1:
            if low[k] <= stop:            # stop first — wins ties
                exit_price = stop if open_[k] > stop else open_[k]
                exit_idx, reason = k, "stop"
                break
            if high[k] >= target:
                exit_price = target if open_[k] < target else open_[k]
                exit_idx, reason = k, "target"
                break
        else:
            if high[k] >= stop:
                exit_price = stop if open_[k] < stop else open_[k]
                exit_idx, reason = k, "stop"
                break
            if low[k] <= target:
                exit_price = target if open_[k] > target else open_[k]
                exit_idx, reason = k, "target"
                break

    r_gross = d * (exit_price - entry) / risk
    cost_r = cost / risk
    return ScalpOutcome(
        signal=sig, entry_index=t + 1, exit_index=exit_idx,
        exit_reason=reason, r_net=float(r_gross - cost_r),
        hold_bars=exit_idx - t,
        cost_r=float(cost_r),
        target_r=float(d * (target - entry) / risk))
