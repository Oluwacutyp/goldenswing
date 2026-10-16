"""Hierarchical decisive test for an AMBIGUOUS breakout
(footprint score 40-69).

HIERARCHY, NEVER DEMOCRACY.  The old-bot autopsy is the
warning: 5 strategies voted by weighted confidence and
cancelled by design, killing a +310% bot (17 trades, +0.3%).
This module contains NO voting: no majority, no counting
indicators, no weighted confidence.  The chain is sequential
— HTF Zone -> Sweep -> Displacement -> CHoCH+FVG -> Entry.
Each step is a GATE, not a vote.  If a step fails, invalidate
and go back to the previous state, never vote.

The next bar is a DECISIVE TEST, not a vote.  ONE decisive
condition decides:

- Close back INSIDE the zone band -> RETAIL
  (mean-reversion sweep).  The close is decisive; the
  reclaim evidence (wick reclaim, reclaim BOS, volume fade,
  FVG mitigation) is logged, never counted.
- Close BEYOND the edge (>= 0.25 ATR) + ALL hierarchical
  confirms hold (body-dominated, volume, BOS hold, FVG
  unmitigated) -> INSTITUTIONAL (breakout trend).
- Close beyond but a confirm fails -> MIXED: wait ONE more
  bar (max 2 bars total), then invalidate if still mixed.
- Doji / indecisive (no decisive close) -> wait ONE more
  bar, then invalidate.
- FVG mitigated during the wait (close beyond + close
  inside the FVG) -> INVALIDATE immediately: the entry
  level is gone.
- Missed move: close > 1.0 ATR from the FVG -> INVALIDATE
  "missed, price ran >1 ATR, no chase", back to STATE 1.
  Never chase, never enter late at a worse price.
- Time cap: max 2 bars total (M15 normal, M5 high-vol),
  then INVALIDATE.  Never extend indefinitely, never wait
  for a 3rd bar.

High-volatility fast path: ATR > 1.5x average ATR, or NY
open, or a news session -> M5 decisive test: the next 2
M5 bars decide, max wait 2 M5 bars (10 min), never 2 M15.
Log "high-volatility fast path, M5 decisive test, max
10 min".

Tick-level reclaim: tick close back inside the edge + tick
BOS reclaim = EARLY RETAIL WARNING.  Tick is early warning,
bar close is decisive, never trade on tick alone.

After the test: institutional -> BREAKOUT_TREND state
machine (retest hold / BOS + 2 closes, scale in); retail ->
SWEEP state machine (STATE 2 sweep confirmed, STATE 3
displacement check, etc.); never both, never trade
ambiguous.  A second-bar institutional confirm is a LATE
CONFIRM: 0.5x size, differentiated management
(``live/management.py``).

Score re-rating on the decisive test: institutional
min(100, score + 23) (55 -> 78), retail max(0, score - 23)
(55 -> 32).  The test re-rates the attempt; the 23-point
move is the documented size of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Next-bar thresholds (spec).
NEXT_CLOSE_BEYOND_ATR = 0.25
NEXT_BODY_WICK_MIN = 1.2
NEXT_VOLUME_MIN = 1.2
RETAIL_VOLUME_MAX = 0.8
DOJI_BODY_WICK_MAX = 0.5
DOJI_RANGE_ATR = 0.25
MISSED_FVG_ATR = 1.0
HIGH_VOL_ATR_MULT = 1.5
VOTE_ADJUST = 23.0


@dataclass
class NextBarDecision:
    """Result of the decisive next-bar test."""

    decision: str = "wait"  # institutional | retail | wait | invalidate
    reason: str = ""
    score_new: float | None = None
    evidence: list = field(default_factory=list)
    log: str = ""


def _body_wick(bar: dict) -> float:
    o, h, l, c = (float(bar["open"]), float(bar["high"]),
                  float(bar["low"]), float(bar["close"]))
    body = abs(c - o)
    wick = (h - max(o, c)) + (min(o, c) - l)
    return body / wick if wick > 0 else float("inf")


def _is_doji(bar: dict, atr: float) -> bool:
    bw = _body_wick(bar)
    rng = (float(bar["high"]) - float(bar["low"])) / atr \
        if atr > 0 else 0.0
    return bw < DOJI_BODY_WICK_MAX and rng < DOJI_RANGE_ATR


def is_high_vol(atr: float, atr_avg: float, *,
                session_killzone: str | None = None,
                news: bool = False) -> bool:
    """High-volatility gate for the fast path: ATR > 1.5x its
    own average, or the NY open killzone, or a news session."""
    if news:
        return True
    if session_killzone == "ny_killzone":
        return True
    return atr_avg > 0 and atr > HIGH_VOL_ATR_MULT * atr_avg


def decide_next_bar(attempt: dict, next_bar: dict, *, edge: float,
                    breakout_dir: int, atr: float,
                    zone_low: float | None = None,
                    zone_high: float | None = None,
                    swing_level: float | None = None,
                    fvg: dict | None = None,
                    waits_used: int = 0) -> NextBarDecision:
    """Decisive test from ONE next bar (hierarchical, no vote).

    ``attempt``: {"score": float} (the ambiguous footprint
    score, 40-69).  ``next_bar``: {"open","high","low","close",
    optional "volume_ratio" (vs the 20-bar mean)}.
    ``breakout_dir``: +1 up / -1 down.  ``waits_used``: how
    many extra bars have already been spent waiting (0 = this
    is the first next bar).  After 1 wait, a second
    indecisive / mixed bar INVALIDATES — max 2 bars total,
    never a 3rd.
    """
    out = NextBarDecision()
    score = float(attempt.get("score", 55.0))
    close = float(next_bar["close"])
    vol_ratio = next_bar.get("volume_ratio")
    bw = _body_wick(next_bar)

    if _is_doji(next_bar, atr):
        if waits_used >= 1:
            out.decision, out.reason = "invalidate", \
                "indecisive_twice"
            out.log = ("ambiguous, 2 bars indecisive, "
                       "invalidated, next setup — back to "
                       "STATE 1")
        else:
            out.decision, out.reason = "wait", "doji_indecisive"
            out.log = ("next bar doji / indecisive, waiting one "
                       "more bar (max 2 total)")
        return out

    # Decisive condition 1: close back INSIDE the zone band ->
    # RETAIL.  The close decides; reclaim evidence is logged,
    # never counted.
    if zone_low is not None and zone_high is not None:
        band_lo = zone_low - 0.25 * atr
        band_hi = zone_high + 0.25 * atr
        inside_band = band_lo <= close <= band_hi
        inside_opposite = inside_band and \
            (edge - close) * breakout_dir > 0
    else:
        inside_opposite = (edge - close) * breakout_dir > 0
    if inside_opposite:
        wick_beyond = (float(next_bar["high"]) - edge) \
            * breakout_dir if breakout_dir > 0 else \
            (edge - float(next_bar["low"])) * breakout_dir
        if inside_opposite and wick_beyond > 0 and bw < 1.0:
            out.evidence.append("wick_reclaim")
        if swing_level is not None and \
                (close - swing_level) * breakout_dir < 0:
            out.evidence.append("reclaim_bos")
        if vol_ratio is not None and vol_ratio < RETAIL_VOLUME_MAX:
            out.evidence.append("volume_fade")
        if fvg is not None and float(fvg["bottom"]) <= close \
                <= float(fvg["top"]):
            out.evidence.append("fvg_mitigated")
        out.decision = "retail"
        out.reason = "close_back_inside"
        out.score_new = max(0.0, score - VOTE_ADJUST)
        ev = (", " + ", ".join(out.evidence)) if out.evidence \
            else ""
        out.log = (f"ambiguous -> retail, next bar close back "
                   f"inside{ev}, score {score:.0f} -> "
                   f"{out.score_new:.0f}, sweep arms")
        return out

    # Missed move: close > 1.0 ATR from the FVG -> the entry
    # level is gone.  No chase, back to STATE 1.
    if fvg is not None:
        dist_fvg = (close - float(fvg["top"])) * breakout_dir \
            / atr if atr > 0 else 0.0
        if dist_fvg > MISSED_FVG_ATR:
            out.decision, out.reason = "invalidate", "missed"
            out.log = ("missed, price ran >1 ATR from FVG, no "
                       "chase — back to STATE 1")
            return out

    # Decisive condition 2: close BEYOND the edge.  The
    # hierarchical confirms must ALL hold for INSTITUTIONAL;
    # any confirm failing makes it MIXED (wait one more bar).
    beyond = (close - edge) * breakout_dir / atr if atr > 0 \
        else 0.0
    close_beyond = beyond >= NEXT_CLOSE_BEYOND_ATR
    if close_beyond:
        body_ok = bw >= NEXT_BODY_WICK_MIN and \
            (close - float(next_bar["open"])) * breakout_dir > 0
        bos_ok = swing_level is None or \
            (close - swing_level) * breakout_dir > 0
        vol_ok = vol_ratio is None or vol_ratio >= NEXT_VOLUME_MIN
        fvg_ok = True
        fvg_mitigated = False
        if fvg is not None and float(fvg["bottom"]) <= close \
                <= float(fvg["top"]):
            fvg_ok = False
            fvg_mitigated = True
        if fvg_mitigated:
            # The entry level is gone during the wait ->
            # invalidate immediately, never wait on a dead FVG.
            out.decision, out.reason = "invalidate", \
                "fvg_mitigated_in_wait"
            out.log = ("FVG mitigated during wait, invalidate "
                       "— back to STATE 1")
            return out
        if body_ok:
            out.evidence.append("body_dominated")
        if bos_ok:
            out.evidence.append("bos_hold")
        if vol_ok and vol_ratio is not None:
            out.evidence.append("volume")
        if fvg_ok:
            out.evidence.append("fvg_unmitigated")
        if body_ok and bos_ok and vol_ok and fvg_ok:
            out.decision = "institutional"
            out.reason = "close_beyond_plus_confirms"
            out.score_new = min(100.0, score + VOTE_ADJUST)
            out.log = (f"ambiguous -> institutional, next bar "
                       f"close beyond + body + volume + BOS, "
                       f"score {score:.0f} -> {out.score_new:.0f}")
            return out
        # Mixed: close beyond, but a confirm failed — wait one
        # more bar, then invalidate if still mixed.
        if waits_used >= 1:
            out.decision, out.reason = "invalidate", "mixed_twice"
            out.log = ("ambiguous, 2 bars mixed, invalidated, "
                       "next setup — back to STATE 1")
        else:
            out.decision, out.reason = "wait", "mixed"
            out.log = ("next bar mixed (close beyond, confirm "
                       "failed), waiting one more bar (max 2 "
                       "total)")
        return out

    # Neither decisive condition: close between the edge and
    # edge + 0.25 ATR, or back inside without crossing — wait
    # one more bar, then invalidate.
    if waits_used >= 1:
        out.decision, out.reason = "invalidate", "still_ambiguous"
        out.log = ("ambiguous, 2 bars still undecided, "
                   "invalidated, next setup — back to STATE 1")
    else:
        out.decision, out.reason = "wait", "undecided"
        out.log = "next bar undecided, waiting one more bar " \
            "(max 2 total)"
    return out


def decide_m5_pair(attempt: dict, m5_bar1: dict, m5_bar2: dict,
                   **kwargs) -> NextBarDecision:
    """High-volatility fast path: 2 M5 bars decide (max 10
    min).  Both must agree on the same decisive outcome.  If
    they disagree, the test waits — the caller maps that to
    INVALIDATE in the fast path (max 2 M5 bars, never a 3rd
    M5 bar and never an M15 wait)."""
    v1 = decide_next_bar(attempt, m5_bar1, **kwargs)
    v2 = decide_next_bar(attempt, m5_bar2, **kwargs)
    if v1.decision == v2.decision and v1.decision in (
            "institutional", "retail"):
        out = v1
        out.log = f"M5 pair agrees: {v1.decision} ({v1.log})"
        return out
    out = NextBarDecision(decision="wait", reason="m5_disagreement")
    out.log = (f"M5 bars disagree ({v1.decision} vs "
               f"{v2.decision}), no decisive outcome")
    return out


def tick_reclaim(tick_price: float, *, edge: float,
                 breakout_dir: int,
                 swing_level: float | None = None) -> dict:
    """Tick-level reclaim: EARLY RETAIL WARNING, never a trade.

    Tick close back inside the edge + tick BOS reclaim =
    early retail signal.  Tick is early warning, bar close is
    decisive, never trade on tick alone.  The runner calls
    this on ticks while a breakout is pending; the bar-close
    decisive test still decides.
    """
    back_inside = (edge - tick_price) * breakout_dir > 0
    bos_reclaim = swing_level is not None and \
        (tick_price - swing_level) * breakout_dir < 0
    early = back_inside and bos_reclaim
    return {"early_retail": early, "back_inside": back_inside,
            "bos_reclaim": bos_reclaim,
            "log": ("tick reclaim, early retail, awaiting bar "
                    "confirm but flagging" if early else
                    ("tick back inside, watching" if back_inside
                     else "tick holds beyond"))}
