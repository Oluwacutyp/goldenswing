"""DXY correlation as live signal context — never a gate.

The owner's bot tracks the XAUUSD vs DXYm correlation two ways: a
50-bar Pearson on returns (``calculate_correlation``) and the Ghost
Correlation Sentinel (strong inverse < -0.8 rewards trend following;
toxic positive > -0.2 penalises).  This module is that behaviour in
this engine's shape:

- :func:`dxy_correlation` — Pearson on returns over the last
  ``period`` H1 closes (the old bot's formula, min 20 bars, NaN/inf
  -> 0.0).  Gold and DXY frames are inner-joined on timestamp so a
  missing DXY bar never fabricates alignment.
- :func:`dxy_bonus` — the Ghost Sentinel bands translated to this
  engine's confluence scale: strong inverse -> +0.5 (healthy regime),
  toxic positive -> -0.5, else 0.  Bounded, direction-agnostic
  (regime health, not a signal-quality claim), and NEVER a gate —
  the confluence scorer adds it the same way it adds the volume
  bonus.  ``None`` (DXY unavailable) -> 0.0.
- :class:`DXYTracker` — fetches DXY H1 bars through the MT5 feed
  (``bars_for``), refreshes on H1 closes, degrades to
  ``available=False`` with a status string on any failure.  A dead
  DXY feed must never fail the gold path: the tracker keeps the last
  known state and the scorer treats unavailable as 0.0.

DXY bias (price vs EMA200, the MacroTrend formula) is reported for
the reasoning text and /status, but it does NOT feed the bonus —
the bonus is the correlation regime only, exactly the Sentinel's
semantics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Correlation window (the old bot's period=50).
CORR_PERIOD = 50
#: Minimum bars for a measured correlation (the old bot's floor).
CORR_MIN_BARS = 20
#: Ghost Sentinel bands (the old bot's thresholds).
SENTINEL_STRONG_INVERSE = -0.8
SENTINEL_TOXIC_POSITIVE = -0.2
#: Bounded confluence nudge (same scale as the volume bonus).
DXY_BONUS = 0.5
#: DXY bars fetched per refresh (covers the correlation window).
DXY_FETCH_BARS = 300


@dataclass
class DXYState:
    corr: float | None = None
    bias: int = 0  # +1 DXY above EMA200, -1 below, 0 flat/unavailable
    available: bool = False
    status: str = "no DXY data yet"
    n_bars: int = 0

    def describe(self) -> str:
        if not self.available:
            return f"DXY: unavailable ({self.status})"
        return (f"DXY: corr {self.corr:+.2f} "
                f"bias {self.bias:+d} ({self.status})")


def dxy_correlation(gold_h1: pd.DataFrame | None,
                    dxy_h1: pd.DataFrame | None,
                    period: int = CORR_PERIOD) -> tuple[float, str]:
    """Pearson correlation of H1 returns over the last ``period``
    closes.  Returns (corr, status); insufficient/misaligned data
    returns (0.0, reason) — never a fabricated number."""
    if gold_h1 is None or dxy_h1 is None:
        return 0.0, "missing frame"
    if len(gold_h1) < CORR_MIN_BARS or len(dxy_h1) < CORR_MIN_BARS:
        return 0.0, "insufficient bars"
    try:
        joined = pd.DataFrame({
            "gold": gold_h1["close"],
            "dxy": dxy_h1["close"],
        }).dropna()
        if len(joined) < CORR_MIN_BARS:
            return 0.0, "insufficient aligned bars"
        sub = joined.iloc[-period:]
        gold_ret = sub["gold"].pct_change().dropna().to_numpy(float)
        dxy_ret = sub["dxy"].pct_change().dropna().to_numpy(float)
        if len(gold_ret) < 5 or len(dxy_ret) < 5:
            return 0.0, "insufficient returns"
        corr = float(np.corrcoef(gold_ret, dxy_ret)[0, 1])
        if not np.isfinite(corr):
            return 0.0, "non-finite correlation"
        return corr, f"measured ({len(sub)} bars)"
    except Exception as exc:
        return 0.0, f"correlation failed: {exc}"


def dxy_bonus(corr: float | None) -> tuple[float, str]:
    """Ghost Sentinel bands as a bounded confluence nudge."""
    if corr is None:
        return 0.0, ""
    if corr < SENTINEL_STRONG_INVERSE:
        return DXY_BONUS, (
            f"+{DXY_BONUS:g} DXY strong inverse ({corr:+.2f}) — "
            "healthy regime")
    if corr > SENTINEL_TOXIC_POSITIVE:
        return -DXY_BONUS, (
            f"-{DXY_BONUS:g} DXY toxic positive ({corr:+.2f}) — "
            "gold moving with the dollar")
    return 0.0, ""


class DXYTracker:
    """DXY H1 fetch + correlation, refreshed on H1 closes."""

    def __init__(self, feed, *, symbol: str = "DXYm",
                 enabled: bool = True) -> None:
        self._feed = feed
        self.symbol = symbol
        self.enabled = enabled
        self.state = DXYState()
        #: P3: the last fetched CLOSED DXY frame, retained so the
        #: SMT event read (live/smt.py) computes from the bars the
        #: tracker already pulled — no second fetch.  None until
        #: the first successful refresh; kept (stale-marked by
        #: state.available) across failures.
        self.frame: pd.DataFrame | None = None

    def refresh(self, gold_h1: pd.DataFrame | None) -> DXYState:
        """Fetch DXY H1, recompute correlation + bias.  Any failure
        keeps the last state and marks it unavailable — the gold
        path never depends on this."""
        if not self.enabled:
            self.state = DXYState(
                available=False, status="disabled")
            return self.state
        try:
            from ..feed.mt5 import MT5Feed
            from ..live.macrotrend import ema_bias

            dxy = self._feed.bars_for(self.symbol, "H1", DXY_FETCH_BARS)
            dxy_closed = MT5Feed._closed_only(
                dxy, "H1", pd.Timestamp.now(tz="UTC"))
            # Gold comes from the stack (closed bars only) — never
            # drop its last row.  Only the DXY frame (raw bars(),
            # forming bar included) needs the closed-only cut.
            corr, status = dxy_correlation(gold_h1, dxy_closed)
            bias = ema_bias(dxy_closed)
            self.frame = dxy_closed
            self.state = DXYState(
                corr=corr, bias=int(bias), available=True,
                status=status, n_bars=len(dxy_closed))
        except Exception as exc:
            # Keep the last known correlation for the record, but
            # mark it unavailable so the scorer treats it as 0.0.
            prev = self.state.corr
            self.state = DXYState(
                corr=prev, bias=self.state.bias, available=False,
                status=f"fetch failed: {exc}",
                n_bars=self.state.n_bars)
        return self.state
