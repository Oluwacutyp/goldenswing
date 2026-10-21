"""Structural SMT divergence vs DXY (P3, spec v3 S7/F11).

The desk's transmission check is not the correlation COEFFICIENT
drifting — it is the DXY's refusal to confirm gold's new extreme
at the moment gold takes liquidity.  This module computes that
EVENT from the already-fetched DXY H1 frame (``live/dxy.py``
pulls 300 bars; nothing new is fetched):

- Gold sweeps a CONFIRMED swing high (wick or close beyond it)
  while, over the same window, DXY FAILS to break its
  corresponding prior swing low -> bearish SMT (direction -1
  for gold): the dollar refused to make the mirror extreme, so
  gold's new high is unbacked.
- Mirror case at swing lows -> bullish SMT (+1).

Validity is correlation-gated (F11): SMT presumes the inverse
link is alive.  The sentinel bands come from ``live/dxy.py``:
healthy inverse < -0.2 (strong < -0.8); drifting toward positive
is the toxic regime where SMT is meaningless.  When the link is
broken the module returns ``available=False`` with the reason —
the runner JOURNALS "SMT unavailable: correlation broken"; it
never silently reads as zero (a broken link masquerading as
"no divergence" would be a lie in the record).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Correlation health band (F11 sentinel, matches dxy.py's
# CORR_STRONG_INVERSE / toxic drift framing).
CORR_HEALTHY_MAX = -0.2


@dataclass(frozen=True)
class SmtEvent:
    direction: int        # +1 bullish for gold, -1 bearish
    kind: str             # "smt_bearish" | "smt_bullish"
    level: float          # the gold swing that was swept
    detail: str
    bars_ago: int         # bars since the sweep bar


@dataclass(frozen=True)
class SmtRead:
    available: bool
    reason: str = ""
    event: SmtEvent | None = None
    corr: float | None = None


def _confirmed_swings(frame: pd.DataFrame, n: int = 3):
    """(highs, lows) as [(pos, price)] with the swing at ``pos``
    confirmed by ``n`` bars on both sides (find_swings rules)."""
    from .multitf import find_swings
    highs, lows = find_swings(
        frame["high"].to_numpy(dtype=float),
        frame["low"].to_numpy(dtype=float), n)
    return highs, lows


def detect_smt(gold: pd.DataFrame, dxy: pd.DataFrame, *,
               corr: float | None, window: int = 12) -> SmtRead:
    """Detect the SMT event on the latest bars of the frames.

    ``window`` = how many bars back a gold sweep still counts as
    the live event (the divergence is a state of the current
    auction, not a historical curiosity).  Pure function.
    """
    if corr is None or corr > CORR_HEALTHY_MAX:
        return SmtRead(
            available=False, corr=corr,
            reason=("SMT unavailable: correlation broken "
                    f"(corr {corr if corr is None else round(corr, 2)}"
                    " — inverse link not in the healthy band "
                    f"< {CORR_HEALTHY_MAX})"))
    if gold is None or dxy is None or len(gold) < 30 \
            or len(dxy) < 30:
        return SmtRead(available=False, corr=corr,
                       reason="SMT unavailable: frames too short")
    try:
        g_highs, g_lows = _confirmed_swings(gold)
        d_highs, d_lows = _confirmed_swings(dxy)
    except Exception as exc:
        return SmtRead(available=False, corr=corr,
                       reason=f"SMT unavailable: swing read failed "
                              f"({exc})")
    if len(g_highs) < 1 or len(g_lows) < 1 \
            or len(d_highs) < 1 or len(d_lows) < 1:
        return SmtRead(available=True, corr=corr,
                       reason="no confirmed swings to diverge")
    n = len(gold)
    g_hi = gold["high"].to_numpy(dtype=float)
    g_lo = gold["low"].to_numpy(dtype=float)
    d_hi = dxy["high"].to_numpy(dtype=float)
    d_lo = dxy["low"].to_numpy(dtype=float)
    # --- bearish: gold sweeps its last confirmed swing high ---
    # The swing governing the sweep is the most recent confirmed
    # high BEFORE the sweep window.
    for pos, price in reversed(g_highs):
        if pos < n - window - 3:
            break
        swept_at = None
        for i in range(max(pos + 3, n - window), n):
            if g_hi[i] > price:
                swept_at = i
                break
        if swept_at is None:
            continue
        # DXY must, over the same window, FAIL to break its own
        # prior confirmed swing low (the mirror extreme).
        prior_d_lows = [p for p in d_lows
                        if p[0] <= swept_at - 3]
        if not prior_d_lows:
            break
        d_level = prior_d_lows[-1][1]
        d_min = float(d_lo[swept_at:].min()) if swept_at < len(d_lo) \
            else float(d_lo[-1])
        if d_min > d_level:
            return SmtRead(available=True, corr=corr, event=SmtEvent(
                direction=-1, kind="smt_bearish", level=float(price),
                detail=(f"gold swept swing high {price:.2f} while "
                        f"DXY held above its swing low "
                        f"{d_level:.2f} (min {d_min:.2f}) — dollar "
                        "refused the mirror extreme"),
                bars_ago=int(n - 1 - swept_at)))
        break
    # --- bullish: gold sweeps its last confirmed swing low ----
    for pos, price in reversed(g_lows):
        if pos < n - window - 3:
            break
        swept_at = None
        for i in range(max(pos + 3, n - window), n):
            if g_lo[i] < price:
                swept_at = i
                break
        if swept_at is None:
            continue
        prior_d_highs = [p for p in d_highs
                         if p[0] <= swept_at - 3]
        if not prior_d_highs:
            break
        d_level = prior_d_highs[-1][1]
        d_max = float(d_hi[swept_at:].max()) if swept_at < len(d_hi) \
            else float(d_hi[-1])
        if d_max < d_level:
            return SmtRead(available=True, corr=corr, event=SmtEvent(
                direction=1, kind="smt_bullish", level=float(price),
                detail=(f"gold swept swing low {price:.2f} while "
                        f"DXY held below its swing high "
                        f"{d_level:.2f} (max {d_max:.2f}) — dollar "
                        "refused the mirror extreme"),
                bars_ago=int(n - 1 - swept_at)))
        break
    return SmtRead(available=True, corr=corr,
                   reason="no live SMT divergence")
