"""STATE 2 sweep vs breach: math, not vibes.

One closed M5/M15 bar at a time, HTF ATR (``ATR_H4``) throughout,
causal only.  A bar is classified by its wick, body, close, volume
and the liquidity it cleared — never by what happens later,
except the two-bar breach / ambiguous-resolution rules, which
use only bars already closed.

Valid SWEEP WICK (liquidity cleared, mean-reversion path):
- Wick beyond the HTF zone edge in [0.25, 1.5] x ATR_H4.
  Wick > 1.5 ATR + close beyond >= 0.5 ATR = MOMENTUM BREACH,
  never a sweep (volatility + momentum = breakout).
- Close back INSIDE the zone band [low - 0.25 ATR, high + 0.25 ATR]
  AND on the opposite side of the edge from the wick (wick above
  the high -> close below the high; wick below the low -> close
  above the low).
- Wick-to-body ratio >= 2.0 (long wick, small body = rejection,
  not momentum).  Body = |close - open|.  Wick = wick extreme -
  max(open, close) for a high sweep, min(open, close) - wick
  extreme for a low sweep.
- Wick length >= 0.25 x ATR_H4 (meaningful liquidity, not noise).
- Close within 0.5 ATR of the edge.  A deeper close is still a
  sweep but flagged ``deep_reclaim`` (strong reversal — needs
  displacement confirm).
- Volume (when available): sweep-bar volume >= 1.2 x the 20-bar
  mean (stops triggered, liquidity taken).  Below that = WEAK
  sweep, flagged, needs stronger displacement.
- Liquidity pool cleared: the sweep must take >= 1 prior equal
  high/low (within 0.5 ATR, 2+ equal points = pool) OR a prior
  swing high/low.  A wick into empty space is
  ``wick_without_liquidity``, not a sweep.

Invalid MOMENTUM BREACH -> BREAKOUT path (not sweep):
- Close BEYOND the edge by >= 0.5 x ATR_H4 AND the next bar
  closes beyond too (2 consecutive closes beyond, both
  >= 0.25 ATR beyond) AND BOS in the breakout direction on
  M15 AND body-dominated (body/wick >= 1.5 on a breach bar) OR
  displacement >= 1.5 ATR in <= 3 M15 bars.
- Close beyond >= 0.5 ATR but the next bar closes back inside
  within 2 bars = FALSE BREAKOUT: back to sweep logic,
  mean-reversion arms.
- Wick > 1.5 ATR + close beyond >= 0.5 ATR = MOMENTUM BREACH
  even with a long wick.

GREY ZONE: wick in [0.25, 1.5] ATR + close inside but
body/wick < 2.0 -> AMBIGUOUS: do NOT arm the sweep; wait for
the next bar.  Next bar closes inside + BOS reclaim (BOS in
the trade direction) -> sweep confirmed.  Next bar closes
beyond -> breach path.  No next bar yet -> AMBIGUOUS_PENDING.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .multitf import find_swings
from .zones import SWING_N

SWEEP_WICK_MIN_ATR = 0.25
SWEEP_WICK_MAX_ATR = 1.5
SWEEP_WICK_BODY_MIN = 2.0
SWEEP_CLOSE_BAND_ATR = 0.25
SWEEP_CLOSE_EDGE_MAX_ATR = 0.5
SWEEP_VOLUME_MIN = 1.2
BREACH_CLOSE_MIN_ATR = 0.5
BREACH_SECOND_CLOSE_MIN_ATR = 0.25
BREACH_BODY_WICK_MIN = 1.5
BREACH_DISP_MIN_ATR = 1.5
FALSE_BREAK_RETURN_BARS = 2
LIQUIDITY_TOL_ATR = 0.5


@dataclass
class SweepVerdict:
    """One bar's STATE 2 classification."""

    kind: str  # valid_sweep | weak_sweep | ambiguous |
    # ambiguous_pending | breach | breach_pending |
    # false_breakout | wick_without_liquidity | none
    direction: int = 0  # trade direction after the sweep
    edge: float = 0.0
    wick: float = 0.0
    close: float = 0.0
    wick_beyond_atr: float = 0.0
    wick_length_atr: float = 0.0
    wick_body_ratio: float = 0.0
    body_wick_ratio: float = 0.0
    volume_ratio: float | None = None
    liquidity_cleared: bool = False
    bos: bool = False
    flags: list = field(default_factory=list)
    bar_ts: str | None = None
    bar_pos: int = -1
    log: str = ""

    @property
    def is_sweep(self) -> bool:
        return self.kind in ("valid_sweep", "weak_sweep")

    @property
    def is_breach(self) -> bool:
        return self.kind == "breach"


def _zf(zone, name, default=None):
    if isinstance(zone, dict):
        return zone.get(name, default)
    return getattr(zone, name, default)


def _edge_for(zone, direction: int) -> tuple[float, float, float]:
    """(edge, zone_low, zone_high) for the trade ``direction``."""
    top = float(_zf(zone, "top", 0.0))
    bottom = float(_zf(zone, "bottom", 0.0))
    edge = top if direction < 0 else bottom
    return edge, bottom, top


def _bar(frame, i: int) -> dict:
    row = frame.iloc[i]
    o, h, l, c = (float(row["open"]), float(row["high"]),
                  float(row["low"]), float(row["close"]))
    return {"open": o, "high": h, "low": l, "close": c,
            "body": abs(c - o), "ts": frame.index[i]}


def _wick_length(bar: dict, direction: int) -> float:
    """Rejection wick: above max(open, close) for a high sweep,
    below min(open, close) for a low sweep."""
    if direction < 0:  # sweep above the high
        return bar["high"] - max(bar["open"], bar["close"])
    return min(bar["open"], bar["close"]) - bar["low"]


def _wick_extreme(bar: dict, direction: int) -> float:
    return bar["high"] if direction < 0 else bar["low"]


def _beyond(bar: dict, edge: float, direction: int) -> float:
    """Signed distance of the CLOSE beyond the edge, in price.

    Positive = beyond on the breakout side (above the high for a
    short-side zone, below the low for a long-side zone).
    """
    if direction < 0:  # edge = zone top; beyond = close above
        return bar["close"] - edge
    return edge - bar["close"]


def _liquidity_cleared(frame, pos: int, edge: float, direction: int,
                       atr: float) -> bool:
    """Did the sweep take prior liquidity?  >= 1 prior swing
    high/low within 0.5 ATR of the edge (2+ equal points = a
    pool; a single prior swing also counts — the sweep took a
    level traders were positioned at).  Bars before ``pos``
    only — causal."""
    if pos <= 0 or atr <= 0:
        return False
    prior = frame.iloc[:pos]
    if len(prior) < 2 * SWING_N + 5:
        # Too few bars for swing detection: fall back to any
        # prior extreme within tolerance of the edge.
        if direction < 0:
            return bool(np.any(np.abs(
                prior["high"].to_numpy(dtype=float) - edge)
                <= LIQUIDITY_TOL_ATR * atr))
        return bool(np.any(np.abs(
            prior["low"].to_numpy(dtype=float) - edge)
            <= LIQUIDITY_TOL_ATR * atr))
    try:
        highs, lows = find_swings(
            prior["high"].to_numpy(dtype=float),
            prior["low"].to_numpy(dtype=float), SWING_N)
    except Exception:
        return False
    swings = highs if direction < 0 else lows
    return any(abs(float(p) - edge) <= LIQUIDITY_TOL_ATR * atr
               for _, p in swings)


def _bos(frame, pos: int, direction: int) -> bool:
    """BOS in ``direction`` on bars up to ``pos`` (inclusive):
    close beyond the last opposing swing.  Insufficient swings
    -> False (a breach REQUIRES a confirmed BOS)."""
    if pos < 2 * SWING_N + 5:
        return False
    sub = frame.iloc[:pos + 1]
    try:
        highs, lows = find_swings(
            sub["high"].to_numpy(dtype=float),
            sub["low"].to_numpy(dtype=float), SWING_N)
        close = float(sub["close"].iloc[-1])
    except Exception:
        return False
    if direction > 0:
        return bool(highs) and close > highs[-1][1]
    return bool(lows) and close < lows[-1][1]


def _displacement(frame, pos: int, direction: int,
                  atr: float, *, window: int = 3) -> float:
    """Max close move in ``direction`` over ``window`` bars ending
    at ``pos``, in ATR."""
    if pos < 1 or atr <= 0:
        return 0.0
    c = frame["close"].to_numpy(dtype=float)
    start = float(c[max(0, pos - window)])
    end = float(c[pos])
    return max(0.0, (end - start) * direction / atr)


def _volume_ratio(frame, pos: int) -> float | None:
    if "volume" not in frame.columns or pos <= 0:
        return None
    try:
        vols = frame["volume"].to_numpy(dtype=float)
        mean_v = float(np.mean(vols[max(0, pos - 20):pos]))
        return float(vols[pos]) / mean_v if mean_v > 0 else None
    except Exception:
        return None


def _classify_bar(frame, pos: int, zone, direction: int,
                  atr: float) -> SweepVerdict:
    """Classify the single closed bar at ``pos``."""
    edge, zlo, zhi = _edge_for(zone, direction)
    bar = _bar(frame, pos)
    extreme = _wick_extreme(bar, direction)
    wick_len = _wick_length(bar, direction)
    body = bar["body"]
    # Beyond distances, in ATR, on the breakout side.
    wick_beyond = ((extreme - edge) if direction < 0
                   else (edge - extreme)) / atr
    close_beyond = _beyond(bar, edge, direction) / atr
    wick_body = (wick_len / body) if body > 0 else float("inf")
    body_wick = (body / wick_len) if wick_len > 0 else float("inf")
    base = SweepVerdict(kind="none", direction=direction, edge=edge,
                        wick=extreme, close=bar["close"],
                        wick_beyond_atr=wick_beyond,
                        wick_length_atr=wick_len / atr,
                        wick_body_ratio=wick_body,
                        body_wick_ratio=body_wick,
                        bar_ts=str(bar["ts"]), bar_pos=pos)
    # --- MOMENTUM BREACH, single bar: wick > 1.5 ATR + close
    # beyond >= 0.5 ATR.  Volatility + momentum = breakout,
    # even with a long wick.  Never a sweep.
    if wick_beyond > SWEEP_WICK_MAX_ATR \
            and close_beyond >= BREACH_CLOSE_MIN_ATR:
        base.kind = "breach"
        base.flags = ["momentum_breach"]
        base.log = (f"MOMENTUM BREACH: wick {wick_beyond:.2f} ATR "
                    f"beyond edge {edge:.1f} + close "
                    f"{close_beyond:.2f} ATR beyond — breakout "
                    "path, not sweep")
        return base
    # --- BREACH pair: 2 consecutive closes beyond (first >= 0.5
    # ATR, second >= 0.25 ATR) + BOS + body-dominated OR
    # displacement >= 1.5 ATR in <= 3 M15 bars.
    if pos >= 1 and close_beyond >= BREACH_SECOND_CLOSE_MIN_ATR:
        prev = _bar(frame, pos - 1)
        prev_beyond = _beyond(prev, edge, direction) / atr
        if prev_beyond >= BREACH_CLOSE_MIN_ATR \
                and close_beyond >= BREACH_SECOND_CLOSE_MIN_ATR:
            bos = _bos(frame, pos, -direction)
            prev_bw = (prev["body"] / _wick_length(prev, direction)) \
                if _wick_length(prev, direction) > 0 else float("inf")
            body_dom = body_wick >= BREACH_BODY_WICK_MIN or \
                prev_bw >= BREACH_BODY_WICK_MIN
            disp = _displacement(frame, pos, -direction, atr)
            if bos and (body_dom or disp >= BREACH_DISP_MIN_ATR):
                base.kind = "breach"
                base.bos = True
                base.flags = ["two_closes_beyond", "bos"]
                if disp >= BREACH_DISP_MIN_ATR:
                    base.flags.append("displacement")
                base.log = (f"BREAKOUT: 2 closes beyond edge "
                            f"{edge:.1f} ({prev_beyond:.2f} + "
                            f"{close_beyond:.2f} ATR) + BOS, "
                            f"breakout path")
                return base
            if bos or body_dom or disp >= BREACH_DISP_MIN_ATR:
                base.kind = "breach_pending"
                base.bos = bos
                base.log = (f"breach closes beyond edge {edge:.1f} "
                            f"but BOS/body/displacement not all "
                            f"confirmed (BOS={bos}) — awaiting "
                            "confirmation, not a sweep")
                return base
    # --- SWEEP candidate: wick in [0.25, 1.5] ATR + close back
    # inside the band + close on the opposite side of the edge.
    inside_band = (zlo - SWEEP_CLOSE_BAND_ATR * atr) <= bar["close"] \
        <= (zhi + SWEEP_CLOSE_BAND_ATR * atr)
    back_inside = (bar["close"] < edge) if direction < 0 \
        else (bar["close"] > edge)
    if SWEEP_WICK_MIN_ATR <= wick_beyond <= SWEEP_WICK_MAX_ATR \
            and inside_band and back_inside:
        if wick_len < SWEEP_WICK_MIN_ATR * atr:
            base.kind = "none"
            base.log = ("wick length "
                        f"{wick_len / atr:.2f} ATR < 0.25 ATR — "
                        "noise, not a sweep")
            return base
        # Grey zone: body too large -> AMBIGUOUS, do not arm.
        if wick_body < SWEEP_WICK_BODY_MIN:
            base.kind = "ambiguous"
            base.log = ("ambiguous sweep, body too large "
                        f"(wick/body {wick_body:.2f} < 2.0), "
                        "awaiting displacement")
            return base
        # Liquidity must have been cleared.
        if not _liquidity_cleared(frame, pos, edge, direction, atr):
            base.kind = "wick_without_liquidity"
            base.log = ("wick without liquidity, not sweep: "
                        "no prior equal high/low or swing at "
                        f"edge {edge:.1f}")
            return base
        vol = _volume_ratio(frame, pos)
        base.volume_ratio = vol
        base.liquidity_cleared = True
        close_to_edge = abs(bar["close"] - edge) / atr
        if close_to_edge > SWEEP_CLOSE_EDGE_MAX_ATR:
            base.flags.append("deep_reclaim")
        if vol is not None and vol < SWEEP_VOLUME_MIN:
            base.kind = "weak_sweep"
            base.flags.append("low_volume_weak_sweep")
            base.log = (f"WEAK sweep: wick {wick_beyond:.2f} ATR "
                        f"beyond edge {edge:.1f}, close back "
                        f"{bar['close']:.1f}, volume "
                        f"{vol:.2f}x < 1.2x — needs stronger "
                        "displacement")
            return base
        base.kind = "valid_sweep"
        base.log = (f"valid sweep: wick {wick_beyond:.2f} ATR "
                    f"beyond edge {edge:.1f}, close back "
                    f"{bar['close']:.1f}, wick/body "
                    f"{wick_body:.1f}, liquidity cleared"
                    + (", deep reclaim" if "deep_reclaim"
                       in base.flags else ""))
        return base
    # --- FALSE BREAKOUT: a close >= 0.5 ATR beyond, then a close
    # back inside within 2 bars (this bar is the return).
    if pos >= 1:
        for j in range(max(0, pos - FALSE_BREAK_RETURN_BARS), pos):
            if _beyond(_bar(frame, j), edge, direction) / atr \
                    >= BREACH_CLOSE_MIN_ATR and inside_band:
                base.kind = "false_breakout"
                base.flags = ["false_breakout"]
                base.log = (f"FALSE BREAKOUT: close beyond edge "
                            f"{edge:.1f} then back inside within "
                            f"2 bars — sweep logic, mean-reversion "
                            "arms")
                return base
    return base


def classify_sweep(frame, zone, *, direction: int, atr: float,
                   lookback: int = 6) -> SweepVerdict:
    """STATE 2 classification over the last ``lookback`` bars.

    Scans newest-first; the first decisive bar wins.  An
    AMBIGUOUS bar is resolved by its NEXT bar when that bar is
    already closed: next bar closes inside + BOS reclaim (BOS
    in the trade direction) -> sweep confirmed; next bar closes
    beyond -> breach path.  A trailing ambiguous bar with no
    next bar yet -> AMBIGUOUS_PENDING.
    """
    if frame is None or len(frame) < 2 or atr <= 0:
        return SweepVerdict(kind="none", direction=direction,
                            log="not enough bars for sweep check")
    n = len(frame)
    for pos in range(n - 1, max(-1, n - 1 - lookback), -1):
        v = _classify_bar(frame, pos, zone, direction, atr)
        if v.kind == "ambiguous":
            # Resolve with the next bar, if it is closed.
            if pos + 1 < n:
                nxt = _bar(frame, pos + 1)
                edge, _, _ = _edge_for(zone, direction)
                if _beyond(nxt, edge, direction) / atr >= \
                        BREACH_SECOND_CLOSE_MIN_ATR:
                    v.kind = "breach_pending"
                    v.log = ("ambiguous sweep resolved to breach "
                             "path: next bar closed beyond edge")
                    return v
                inside = _edge_for(zone, direction)[1] \
                    - SWEEP_CLOSE_BAND_ATR * atr <= nxt["close"] \
                    <= _edge_for(zone, direction)[2] \
                    + SWEEP_CLOSE_BAND_ATR * atr
                if inside and _bos(frame, pos + 1, direction):
                    v.kind = "valid_sweep"
                    v.flags = ["resolved_from_ambiguous"]
                    v.log = ("ambiguous sweep resolved: next bar "
                             "closed inside + BOS reclaim — sweep "
                             "confirmed")
                    return v
                v.kind = "ambiguous_pending"
                v.log = ("ambiguous sweep, next bar neither "
                         "inside + BOS reclaim nor beyond — "
                         "still awaiting displacement")
                return v
            v.kind = "ambiguous_pending"
            v.log = ("ambiguous sweep, body too large, awaiting "
                     "next bar to resolve")
            return v
        if v.kind != "none":
            return v
    return SweepVerdict(kind="none", direction=direction,
                        log="no sweep / breach / false breakout in "
                        "window")
