"""RSI divergence detection — a new detector-level feature family.

Mined claim (TradingLab scalping video, user-mined 2026-10-11): RSI
divergence (price makes a new extreme, RSI doesn't) is an early sign of
a new move.  Regular divergence = reversal; hidden divergence =
continuation.  The video uses RSI(9) with auto-labelling; the reader
had RSI-based features (relstr) but NO divergence detection, so this
is a genuinely new family.

Strictly causal — every rule uses bars <= t only:

- RSI(period) is Wilder's, seeded on the first ``period`` deltas;
  rsi[t] uses closes <= t.  Warmup bars read 50.0 (neutral).
- Swing pivots are fractals with left=3, right=2: a pivot at bar i is
  CONFIRMED at bar i+2.  Nothing at bar t may use a pivot whose
  confirmation bar is > t.
- A divergence is complete when its second pivot confirms; it stays
  "active" for ``div_max_age`` bars after confirmation, then expires.

Four types (consecutive confirmed pivots of the same side):

  bull_reg  regular bullish:  price lower low,  RSI higher low   -> reversal up
  bear_reg  regular bearish:  price higher high, RSI lower high  -> reversal down
  bull_hid  hidden bullish:   price higher low,  RSI lower low   -> continuation up
  bear_hid  hidden bearish:   price lower high,  RSI higher high -> continuation down

Pre-committed rule set (not fitted): pivot separation 5..120 bars,
|dRSI| >= 2.0 points, max age 30 bars.  A sensitivity note on the dRSI
threshold lives in experiments/DIVERGENCE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FeatureConfig


# ------------------------------------------------------------------ RSI

def rsi_wilder(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI.  rsi[t] depends on closes <= t only; the first
    ``period`` bars read 50.0 (neutral warmup, never a signal)."""
    n = len(close)
    out = np.full(n, 50.0)
    if n <= period or period < 2:
        return out
    delta = np.diff(close)
    gain = np.clip(delta, 0.0, None)
    loss = np.clip(-delta, 0.0, None)
    ag = float(gain[:period].mean())
    al = float(loss[:period].mean())
    for t in range(period, n):
        if t > period:
            ag = (ag * (period - 1) + gain[t - 1]) / period
            al = (al * (period - 1) + loss[t - 1]) / period
        if al <= 0:
            rs = np.inf if ag > 0 else 1.0
        else:
            rs = ag / al
        out[t] = 100.0 - 100.0 / (1.0 + rs)
    return out


# ---------------------------------------------------------------- pivots

def fractal_pivots(high: np.ndarray, low: np.ndarray,
                   left: int = 3, right: int = 2
                   ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Fractal swing highs / lows as (pivot_index, confirmed_at) pairs.

    A swing high at i needs high[i] strictly above the ``left`` bars
    before it and at-or-above the ``right`` bars after it (ties break
    to the earlier bar, so equal highs never double-count).  The pivot
    is CONFIRMED at bar i + right — the first bar whose data completes
    the fractal.  Strictly causal by construction.
    """
    n = len(high)
    highs: list[tuple[int, int]] = []
    lows: list[tuple[int, int]] = []
    for i in range(left, n - right):
        is_high = all(high[i] > high[i - k] for k in range(1, left + 1)) \
            and all(high[i] >= high[i + k] for k in range(1, right + 1))
        is_low = all(low[i] < low[i - k] for k in range(1, left + 1)) \
            and all(low[i] <= low[i + k] for k in range(1, right + 1))
        if is_high:
            highs.append((i, i + right))
        if is_low:
            lows.append((i, i + right))
    return highs, lows


# ------------------------------------------------------------- divergence

@dataclass(frozen=True)
class Divergence:
    """One completed divergence.  Everything about it was known no
    later than ``confirmed_at``."""
    kind: str              # bull_reg | bear_reg | bull_hid | bear_hid
    pivot_a: int
    pivot_b: int
    confirmed_at: int      # confirmation bar of the second pivot
    price_a: float
    price_b: float
    rsi_a: float
    rsi_b: float
    rsi_period: int

    @property
    def direction(self) -> int:
        return 1 if self.kind.startswith("bull") else -1

    @property
    def is_regular(self) -> bool:
        return self.kind.endswith("_reg")


@dataclass(frozen=True)
class DivergenceState:
    """Divergences active at one bar (confirmed, not yet expired),
    newest first."""
    divergences: tuple = field(default_factory=tuple)
    rsi_period: int = 14

    def of_kind(self, kind: str) -> "Divergence | None":
        for dv in self.divergences:
            if dv.kind == kind:
                return dv
        return None

    def aligned(self, d: int) -> "Divergence | None":
        """Newest active divergence pointing in direction d."""
        for dv in self.divergences:
            if dv.direction == d:
                return dv
        return None

    def opposing(self, d: int) -> "Divergence | None":
        for dv in self.divergences:
            if dv.direction == -d:
                return dv
        return None


def detect_divergences(high: np.ndarray, low: np.ndarray, rsi: np.ndarray,
                       rsi_period: int, left: int = 3, right: int = 2,
                       min_sep: int = 5, max_sep: int = 120,
                       min_drsi: float = 2.0) -> list[Divergence]:
    """All completed divergences on the series, sorted by confirmation
    bar.  Causal: a divergence appears only at its second pivot's
    confirmation bar."""
    swing_highs, swing_lows = fractal_pivots(high, low, left, right)
    out: list[Divergence] = []
    for pivots, is_high in ((swing_highs, True), (swing_lows, False)):
        for (ia, _ca), (ib, cb) in zip(pivots, pivots[1:]):
            sep = ib - ia
            if sep < min_sep or sep > max_sep:
                continue
            pa = high[ia] if is_high else low[ia]
            pb = high[ib] if is_high else low[ib]
            ra, rb = float(rsi[ia]), float(rsi[ib])
            if abs(rb - ra) < min_drsi:
                continue
            if is_high:
                if pb > pa and rb < ra:
                    kind = "bear_reg"
                elif pb < pa and rb > ra:
                    kind = "bear_hid"
                else:
                    continue
            else:
                if pb < pa and rb > ra:
                    kind = "bull_reg"
                elif pb > pa and rb < ra:
                    kind = "bull_hid"
                else:
                    continue
            out.append(Divergence(kind=kind, pivot_a=ia, pivot_b=ib,
                                  confirmed_at=cb, price_a=float(pa),
                                  price_b=float(pb), rsi_a=ra, rsi_b=rb,
                                  rsi_period=rsi_period))
    out.sort(key=lambda d: (d.confirmed_at, d.pivot_b))
    return out


def compute(df: pd.DataFrame, states, cfg: FeatureConfig,
            rsi_override: np.ndarray | None = None
            ) -> list[DivergenceState]:
    """Per-bar divergence state.  Pure function of bars <= t at every t.

    ``rsi_override`` is a TEST-ONLY injection point: a precomputed RSI
    array (same length as df) used instead of deriving RSI from closes,
    so fixtures can pin exact RSI values at pivot bars.  Production
    callers never pass it.
    """
    n = len(df)
    if len(states) != n:
        raise ValueError("states must align 1:1 with df bars")
    period = cfg.div_rsi_period
    if rsi_override is not None:
        rsi = np.asarray(rsi_override, dtype=float)
        if len(rsi) != n:
            raise ValueError("rsi_override must align 1:1 with df bars")
    else:
        rsi = rsi_wilder(df["close"].to_numpy(dtype=float), period)
    divs = detect_divergences(
        df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float),
        rsi, period, left=cfg.div_pivot_left, right=cfg.div_pivot_right,
        min_sep=cfg.div_min_sep, max_sep=cfg.div_max_sep,
        min_drsi=cfg.div_min_drsi)
    active: list[list[Divergence]] = [[] for _ in range(n)]
    for dv in divs:
        end = min(n, dv.confirmed_at + cfg.div_max_age + 1)
        for t in range(dv.confirmed_at, end):
            active[t].append(dv)
    return [DivergenceState(divergences=tuple(reversed(a)),
                            rsi_period=period)
            for a in active]
