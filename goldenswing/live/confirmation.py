"""STATE 4 LTF confirmation: CHoCH/BOS + FVG must overlap in
PRICE and TIME before STATE 5 Entry.  No premature entries.

M5/M15 LTF, causal closed bars only.

CHoCH vs BOS:
- CHoCH (Change of Character): FIRST break against the prior
  trend (downtrend LH/LL -> CHoCH = close above last LH).
  Reversal signal — stronger for STATE 4 after a sweep (a
  sweep IS a reversal).  Score +10 vs BOS.
- BOS (Break of Structure): continuation break (uptrend HH/HL
  -> BOS = close above last HH).  Valid for STATE 4 if the
  displacement was continuation (breakout path) or if CHoCH
  already happened and BOS confirms.
- MSB (Market Structure Break): confirmed reversal = CHoCH +
  BOS in the same direction (CHoCH breaks the LH, then BOS
  breaks the HH).  Strongest, +15, STATE 4 passes immediately
  with FVG.

CHoCH/BOS criteria:
- Break price: close beyond the last opposing swing by
  >= 0.1 x ATR_H4 (meaningful, not a wick).
- Break bar: body-dominated, body/wick >= 1.2, close in the
  break direction.  Wick-only (wick beyond + close back
  inside) is NOT CHoCH/BOS.
- Displacement confirm: the break bar must be part of the
  displacement leg or immediately after it (within the time
  overlap window below), never an isolated chop break.

FVG criteria:
- FVG from the STATE 3 displacement leg, size >= 0.25 x ATR_H4.
- Same direction as CHoCH/BOS (bullish FVG for long, bearish
  for short).  CE (50%) recorded as the ideal entry.

PRICE OVERLAP (CHoCH + FVG must overlap in price):
- CHoCH/BOS break price WITHIN the FVG or within 0.25 x ATR
  of an FVG edge.  Break price > 0.5 ATR from the FVG =
  "CHoCH far from FVG, no overlap, awaiting retracement" —
  do NOT enter; wait for retracement into the FVG (<= 3 M15
  bars normal, <= 12 M15 complex).
- Entry zone = the FVG itself (entry at FVG CE / band), NOT
  the CHoCH break price.  CHoCH proves structure; FVG is entry.
- Close THROUGH the FVG (beyond the far side) before CHoCH =
  FVG MITIGATED / INVERTED: filled + reverses back to the
  entry side = Inversion FVG (valid, +5, entry at the inverted
  FVG); filled + stays through = MITIGATED, back to STATE 3
  (await a new FVG) or STATE 1 if displacement invalidated.
- After CHoCH, a close INSIDE the FVG band = the retracement
  arriving = entry trigger (LIMIT), NOT invalidation.  A
  close THROUGH the far side after CHoCH = the level failed
  (invalid), unless it reverses back = inversion (valid).

TIME OVERLAP (CHoCH + FVG must overlap in time):
- CHoCH/BOS within <= 3 M15 bars (45 min) OR <= 6 M5 bars
  (30 min) AFTER displacement FVG formation.  CHoCH > 3 M15
  bars after the FVG = FVG STALE ("FVG stale, CHoCH too
  late, awaiting new FVG"), back to STATE 3.
- FVG must be unmitigated-or-inverted at CHoCH time: a fill
  before CHoCH without reversal = mitigated, invalid.
- Retracement window: after CHoCH/BOS, price must retrace
  into the FVG within <= 3 M15 bars for entry (STATE 5).  No
  retracement in 3 bars + price > 1.0 ATR from the FVG =
  complex-pullback check (FVG unmitigated + BOS holds ->
  extend to 12 M15 bars) else MISSED ("missed retracement,
  FVG left behind, awaiting new setup"), back to STATE 1 / 3,
  do NOT chase.
- CHoCH BEFORE the FVG: still valid IF the FVG forms within
  1 M15 bar after CHoCH and the CHoCH break price is within
  the FVG.  FVG > 1 bar after CHoCH = invalid, CHoCH stale.

Transition STATE 4 -> STATE 5: ALL of CHoCH/BOS valid + FVG
valid + price overlap + time overlap + FVG unmitigated-or-
inverted + retracement into FVG (or entry at FVG on confirm
close).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .multitf import find_swings
from .zones import SWING_N
from .patterns import trend_bias

BREAK_MIN_ATR = 0.1
BREAK_BODY_WICK_MIN = 1.2
PRICE_OVERLAP_ATR = 0.25
PRICE_FAR_ATR = 0.5
TIME_OVERLAP_M15 = 3
TIME_OVERLAP_M5 = 6
RETRACE_WINDOW_M15 = 3
COMPLEX_WINDOW_M15 = 12
CHoCH_BEFORE_FVG_MAX = 1
MISSED_RUNAWAY_ATR = 1.0
CHoCH_ADJ = 10.0
MSB_ADJ = 15.0
INVERSION_ADJ = 5.0


@dataclass
class ConfirmVerdict:
    """One STATE 4 evaluation."""

    decision: str = "wait_structure"  # confirm | wait_retracement |
    # wait_fvg | wait_structure | stale | mitigated | missed |
    # invalid | complex_pullback
    kind: str = "none"  # choch | bos | msb | none
    break_price: float | None = None
    break_pos: int = -1
    choch_extreme: float | None = None
    fvg: dict = field(default_factory=dict)
    inversion: bool = False
    overlap_atr: float = 0.0
    bars_after_fvg: int = 0
    bars_since_choch: int = 0
    retraced: bool = False
    entry_price: float | None = None
    score_adj: float = 0.0
    flags: list = field(default_factory=list)
    log: str = ""


def _last_opposing_swing(frame, pos: int, direction: int):
    """Last opposing swing before bar ``pos`` (exclusive).

    Long: last swing HIGH (the LH to break for CHoCH, the HH
    for BOS).  Short: last swing LOW.  Returns (price, kind)
    where kind is informational only.
    """
    if pos < 2 * SWING_N + 5:
        return None
    sub = frame.iloc[:pos]
    try:
        highs, lows = find_swings(
            sub["high"].to_numpy(dtype=float),
            sub["low"].to_numpy(dtype=float), SWING_N)
    except Exception:
        return None
    if direction > 0:
        return float(highs[-1][1]) if highs else None
    return float(lows[-1][1]) if lows else None


def _prior_trend(frame, pos: int) -> int:
    """Trend before bar ``pos`` (exclusive): +1 / -1 / 0."""
    if pos < 2 * SWING_N + 5:
        return 0
    try:
        return int(trend_bias(frame.iloc[:pos])["bias"])
    except Exception:
        return 0


def _body_wick(row) -> float:
    o, h, l, c = (float(row["open"]), float(row["high"]),
                  float(row["low"]), float(row["close"]))
    body = abs(c - o)
    wick = (h - max(o, c)) + (min(o, c) - l)
    return body / wick if wick > 0 else float("inf")


def _fvg_state(frame, fvg: dict, direction: int, *,
               upto_pos: int | None = None) -> str:
    """FVG state over bars after formation up to ``upto_pos``:
    'clean' (no close inside / through), 'filled' (a close
    inside or through, price stayed through), 'inverted'
    (filled, then a close back on the entry side)."""
    if not fvg:
        return "missing"
    top, bottom = float(fvg["top"]), float(fvg["bottom"])
    start = int(fvg.get("bar_pos", -1)) + 1
    end = len(frame) - 1 if upto_pos is None \
        else min(upto_pos, len(frame) - 1)
    if start > end:
        return "clean"
    c = frame["close"].to_numpy(dtype=float)
    filled_at = None
    for i in range(start, end + 1):
        if bottom <= float(c[i]) <= top:
            filled_at = i
            break
        # Close THROUGH the far side also fills the FVG.
        if direction > 0 and float(c[i]) < bottom:
            filled_at = i
            break
        if direction < 0 and float(c[i]) > top:
            filled_at = i
            break
    if filled_at is None:
        return "clean"
    # Inversion: a later close back on the entry side.
    for i in range(filled_at + 1, end + 1):
        if direction > 0 and float(c[i]) > top:
            return "inverted"
        if direction < 0 and float(c[i]) < bottom:
            return "inverted"
    return "filled"


def _find_breaks(frame, direction: int, atr: float, *,
                 start_pos: int, end_pos: int) -> list[dict]:
    """CHoCH / BOS break events in [start_pos, end_pos].

    Each event: close beyond the last opposing swing by
    >= 0.1 ATR, break bar body/wick >= 1.2, close in the break
    direction.  Wick-only breaks are excluded by the close
    requirement.  Kind: CHoCH when the prior trend opposes the
    break (or is flat — first break out of a range), BOS when
    the prior trend already runs with the break.
    """
    out = []
    for pos in range(start_pos, end_pos + 1):
        swing = _last_opposing_swing(frame, pos, direction)
        if swing is None:
            continue
        row = frame.iloc[pos]
        close = float(row["close"])
        beyond = (close - swing) * direction / atr
        if beyond < BREAK_MIN_ATR:
            continue
        if _body_wick(row) < BREAK_BODY_WICK_MIN:
            continue
        trend = _prior_trend(frame, pos)
        kind = "bos" if trend == direction else "choch"
        out.append({"pos": pos, "price": close, "swing": swing,
                    "kind": kind, "beyond_atr": beyond,
                    "ts": str(frame.index[pos])})
    return out


def evaluate_confirmation(frame, *, displacement, direction: int,
                          atr: float) -> ConfirmVerdict:
    """STATE 4: CHoCH/BOS + FVG price + time overlap."""
    base = ConfirmVerdict()
    if frame is None or len(frame) < 2 or atr <= 0:
        base.decision = "invalid"
        base.log = "no bars for STATE 4 confirmation"
        return base
    fvg = dict((displacement or {}).get("fvg") or {})
    if not fvg:
        base.decision = "wait_fvg"
        base.log = "FVG without CHoCH/BOS — awaiting FVG"
        return base
    base.fvg = fvg
    fvg_pos = int(fvg.get("bar_pos", -1))
    n = len(frame)
    top, bottom = float(fvg["top"]), float(fvg["bottom"])
    # Break events from 1 bar before FVG formation (CHoCH-before-
    # FVG case) to the last bar.
    breaks = _find_breaks(frame, direction, atr,
                          start_pos=max(0, fvg_pos - 1), end_pos=n - 1)
    if not breaks:
        # CHoCH without FVG is the mirror case; here the FVG
        # exists but no structure break yet.
        base.decision = "wait_structure"
        base.log = "FVG without CHoCH/BOS — awaiting structure"
        return base
    # MSB: a CHoCH event followed by a BOS event, same direction.
    kinds = [b["kind"] for b in breaks]
    msb = "choch" in kinds and "bos" in kinds and \
        kinds.index("bos") > kinds.index("choch")
    first = breaks[0]
    # CHoCH-before-FVG: valid only if the FVG forms within 1 bar
    # after CHoCH and the break price is inside the FVG.
    if first["pos"] < fvg_pos:
        if fvg_pos - first["pos"] > CHoCH_BEFORE_FVG_MAX:
            base.decision = "stale"
            base.log = ("CHoCH stale: FVG formed > 1 M15 bar "
                        "after CHoCH — awaiting new FVG")
            return base
        if not (bottom <= first["price"] <= top):
            base.decision = "stale"
            base.log = ("CHoCH before FVG but break price outside "
                        "the FVG — CHoCH stale")
            return base
        base.flags.append("choch_before_fvg")
    # Time overlap: break within <= 3 M15 bars after FVG.
    bars_after = first["pos"] - fvg_pos
    base.bars_after_fvg = max(0, bars_after)
    if bars_after > TIME_OVERLAP_M15:
        base.decision = "stale"
        base.log = (f"FVG stale, CHoCH too late "
                    f"({bars_after} M15 bars after FVG > 3) — "
                    "awaiting new FVG")
        return base
    # FVG state AT CHoCH time: unmitigated-or-inverted only.
    # The CHoCH bar itself is EXCLUDED: its close inside the
    # FVG is the price overlap, not a pre-CHoCH fill.
    state_at_choch = _fvg_state(frame, fvg, direction,
                                upto_pos=first["pos"] - 1)
    if state_at_choch == "filled":
        # The CHoCH bar itself closes back on the entry side
        # (above the FVG top for a long / below the bottom
        # for a short): the fill inverted AT the CHoCH bar —
        # that is an inversion, not a mitigation.
        choch_close = float(frame["close"].iloc[first["pos"]])
        reversed_at_choch = (choch_close > top) \
            if direction > 0 else (choch_close < bottom)
        if reversed_at_choch:
            state_at_choch = "inverted"
    if state_at_choch == "filled":
        base.decision = "mitigated"
        base.log = ("FVG mitigated before CHoCH, invalidated — "
                    "back to STATE 3 (await new FVG)")
        return base
    if state_at_choch == "inverted":
        base.inversion = True
        base.score_adj += INVERSION_ADJ
        base.flags.append("inversion_fvg")
    # Price overlap: break price within FVG or <= 0.25 ATR of an
    # FVG edge.  > 0.5 ATR away = wait for retracement.
    if bottom <= first["price"] <= top:
        overlap = 0.0
    else:
        overlap = min(abs(first["price"] - top),
                      abs(first["price"] - bottom)) / atr
    base.overlap_atr = overlap
    base.break_price = first["price"]
    base.break_pos = first["pos"]
    # CHoCH extreme = the broken swing level (BOS-hold reference).
    base.choch_extreme = float(first["swing"])
    if overlap > PRICE_FAR_ATR:
        base.decision = "wait_retracement"
        base.kind = "msb" if msb else first["kind"]
        base.log = (f"CHoCH far from FVG, no overlap "
                    f"({overlap:.2f} ATR > 0.5), awaiting "
                    "retracement into FVG")
        return base
    # Kind + score: CHoCH +10 vs BOS; MSB +15 (strongest).
    if msb:
        base.kind = "msb"
        base.score_adj += MSB_ADJ
    elif first["kind"] == "choch":
        base.kind = "choch"
        base.score_adj += CHoCH_ADJ
    else:
        base.kind = "bos"
    # Retracement: after CHoCH, a close inside the FVG band =
    # the retracement arriving (LIMIT entry trigger).  A close
    # THROUGH the far side after CHoCH = level failed (unless
    # it reverses = inversion, still valid).
    bars_since = (n - 1) - first["pos"]
    base.bars_since_choch = bars_since
    c = frame["close"].to_numpy(dtype=float)
    retraced = False
    through = False
    for i in range(first["pos"] + 1, n):
        if bottom <= float(c[i]) <= top:
            retraced = True
            break
        if direction > 0 and float(c[i]) < bottom:
            through = True
            break
        if direction < 0 and float(c[i]) > top:
            through = True
            break
    base.retraced = retraced
    if through and not retraced:
        # Through + reversal = inversion; through + stays = fail.
        state_now = _fvg_state(frame, fvg, direction)
        if state_now == "inverted":
            base.inversion = True
            base.score_adj += INVERSION_ADJ
            base.flags.append("inversion_fvg")
        else:
            base.decision = "mitigated"
            base.log = ("FVG mitigated: price closed through the "
                        "FVG after CHoCH — back to STATE 3")
            return base
    if retraced or base.inversion:
        base.decision = "confirm"
        base.entry_price = float(fvg.get("ce", (top + bottom) / 2.0))
        base.log = (f"STATE 4 confirmed: {base.kind.upper()} "
                    f"{first['price']:.1f} + FVG "
                    f"{bottom:.1f}-{top:.1f}, overlap "
                    f"{overlap:.1f} ATR, time "
                    f"{max(0, bars_after)} M15 bars, -> STATE 5 "
                    "Entry")
        return base
    # No retracement yet: normal window (3 M15) vs complex
    # pullback (12 M15, FVG unmitigated + BOS holds).
    last_close = float(c[-1])
    dist_to_fvg = 0.0 if bottom <= last_close <= top else \
        min(abs(last_close - top), abs(last_close - bottom)) / atr
    bos_holds = True
    # BOS holds = no close back through the CHoCH extreme
    # (the broken swing level) after CHoCH.
    for i in range(first["pos"] + 1, n):
        if direction > 0 and float(c[i]) < first["swing"]:
            bos_holds = False
            break
        if direction < 0 and float(c[i]) > first["swing"]:
            bos_holds = False
            break
    if not bos_holds:
        base.decision = "invalid"
        base.log = ("BOS broken: price closed back through the "
                    "CHoCH extreme — invalidate, back to STATE 1")
        return base
    state_now = _fvg_state(frame, fvg, direction)
    if state_now == "filled":
        base.decision = "mitigated"
        base.log = ("FVG mitigated while awaiting retracement — "
                    "back to STATE 3")
        return base
    if bars_since <= RETRACE_WINDOW_M15:
        base.decision = "wait_retracement"
        base.log = ("CHoCH without retracement yet — awaiting "
                    "retracement into FVG "
                    f"({bars_since}/3 M15 bars)")
        return base
    if bars_since <= COMPLEX_WINDOW_M15:
        base.decision = "complex_pullback"
        base.flags.append("complex_pullback")
        base.log = ("complex pullback, FVG unmitigated, BOS "
                    "holds, window extended to 12 M15")
        return base
    if dist_to_fvg > MISSED_RUNAWAY_ATR:
        base.decision = "missed"
        base.log = (f"missed retracement, FVG left behind "
                    f"(price {dist_to_fvg:.1f} ATR away, "
                    f"{bars_since} M15 bars) — no chase")
        return base
    base.decision = "missed"
    base.log = ("retracement window exhausted (12 M15 bars, "
                "complex pullback) — awaiting new setup")
    return base
