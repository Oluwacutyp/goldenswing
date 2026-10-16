"""Entry validation: candles confirm, levels locate, MTF refines.

This is the entry system the project set out to build from the start:
- HTF gives bias (D1/H4)
- H1 marks the level (S/R flip, OB, liquidity pool)
- CANDLES validate the entry (rejection, engulfing, close-back)
- Entry type is chosen by setup (limit at zone vs market on confirmation)
- LTF refines the trigger (M15/M5 shows the rejection clearly)

A level without candle validation is a hope. A candle without a level
is noise. The two together are a trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class EntryType(Enum):
    """How to enter the trade."""
    LIMIT_AT_ZONE = "limit_at_zone"      # patient: place limit at the level
    MARKET_ON_CONFIRM = "market_on_confirm"  # aggressive: market on candle close


@dataclass
class CandleValidation:
    """Result of validating a candle at a level."""
    valid: bool
    pattern: str          # "rejection_wick" | "engulfing" | "close_back" | "none"
    strength: float       # 0.0-1.0, how clean the validation is
    detail: str = ""


def validate_bearish_rejection(open_: float, high: float, low: float,
                               close: float, level: float,
                               atr: float) -> CandleValidation:
    """Check if a candle shows bearish rejection at a resistance level.

    Patterns (in order of strength):
    1. Rejection wick: upper wick > 0.5x ATR, close in lower half, level
       within the upper wick zone
    2. Bearish engulfing: body engulfs prior candle, closes below level
    3. Close-back: wicked above level but closed back below it
    """
    if atr <= 0:
        return CandleValidation(False, "none", 0.0, "no ATR")

    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    candle_range = high - low

    # level must be near the candle (within 0.5x ATR)
    if abs(high - level) > 0.5 * atr and abs(close - level) > 0.5 * atr:
        return CandleValidation(False, "none", 0.0, "level too far")

    # 1. rejection wick: long upper wick, close weak
    if upper_wick > 0.5 * atr and close < open_:
        # level should be in the wick zone (rejected from there)
        if low <= level <= high:
            strength = min(1.0, upper_wick / atr)
            return CandleValidation(True, "rejection_wick", strength,
                                    f"wick {upper_wick/atr:.2f} ATR")

    # 2. close-back: wicked above level, closed below
    if high > level and close < level and close < open_:
        strength = min(1.0, (high - level) / atr + 0.3)
        return CandleValidation(True, "close_back", strength,
                                "closed back below level")

    # 3. bearish body at resistance (weakest, needs other confluence)
    if close < open_ and body > 0.3 * atr and abs(close - level) < 0.3 * atr:
        return CandleValidation(True, "bearish_body", 0.4,
                                "bearish close at level")

    return CandleValidation(False, "none", 0.0, "no rejection")


def validate_bullish_rejection(open_: float, high: float, low: float,
                               close: float, level: float,
                               atr: float) -> CandleValidation:
    """Mirror of bearish: bullish rejection at a support level."""
    if atr <= 0:
        return CandleValidation(False, "none", 0.0, "no ATR")

    body = abs(close - open_)
    lower_wick = min(open_, close) - low

    if abs(low - level) > 0.5 * atr and abs(close - level) > 0.5 * atr:
        return CandleValidation(False, "none", 0.0, "level too far")

    # 1. rejection wick: long lower wick, close strong
    if lower_wick > 0.5 * atr and close > open_:
        if low <= level <= high:
            strength = min(1.0, lower_wick / atr)
            return CandleValidation(True, "rejection_wick", strength,
                                    f"wick {lower_wick/atr:.2f} ATR")

    # 2. close-back: wicked below level, closed above
    if low < level and close > level and close > open_:
        strength = min(1.0, (level - low) / atr + 0.3)
        return CandleValidation(True, "close_back", strength,
                                "closed back above level")

    # 3. bullish body at support
    if close > open_ and body > 0.3 * atr and abs(close - level) < 0.3 * atr:
        return CandleValidation(True, "bullish_body", 0.4,
                                "bullish close at level")

    return CandleValidation(False, "none", 0.0, "no rejection")


def choose_entry_type(setup_kind: str, validation: CandleValidation) -> EntryType:
    """Route to limit or market based on setup and validation.

    - S/R flip, OB retest: LIMIT at the zone (patient, price comes to you)
    - Breakout, MSS: MARKET on confirmation (aggressive, don't miss it)
    - Strong validation: can use limit (high confidence in the level)
    - Weak validation: market on confirm (need to see it happen)
    """
    patient_kinds = {"sr_flip", "ob_retest", "liquidity_retest"}
    aggressive_kinds = {"breakout", "turtle_soup", "v_reversal"}

    if setup_kind in patient_kinds:
        return EntryType.LIMIT_AT_ZONE
    if setup_kind in aggressive_kinds:
        return EntryType.MARKET_ON_CONFIRM
    # default: strong validation -> limit, weak -> market
    if validation.strength >= 0.7:
        return EntryType.LIMIT_AT_ZONE
    return EntryType.MARKET_ON_CONFIRM


# ------------------------------------------------------------------
# POI confirmation — full family (SHADOW annotation only)
#
# The first version of this section checked two shapes (rejection /
# engulfing).  Confirmation at a zone is a FAMILY, not one pattern,
# so this section covers all of them, each with a precise,
# testable definition, all evaluated AT the POI band [poi_bottom,
# poi_top] and direction-aware (long / short mirrors):
#
# Candle shapes (single / 2-3 bar patterns, scanned on ``frame``):
# - rejection / engulfing: as originally defined below.
# - pin_bar (hammer / shooting star): wick >= 2x body, wick >= 2/3
#   of range, body <= 1/3 of range, close in the favourable half,
#   bar overlaps the POI.  Body must be > 0 (a doji is never a pin).
# - morning_star / evening_star: 3 bars — directional bar, small
#   middle bar (body <= 1/2 of the first bar's body), third bar
#   closing past the first bar's body midpoint; first bar reached
#   the zone, third bar overlaps the POI.
# - tweezer_bottom / tweezer_top: 2 bars — first against the setup,
#   second with the setup, near-equal extremes (within 10% of the
#   smaller range), second close beyond the first close; both bars
#   overlap the POI.
# - inside_bar_breakout: mother bar at the POI, inside bar fully
#   inside it (inclusive), third bar closes beyond the mother in
#   the setup direction.
# - three_soldiers / three_crows: 3 consecutive directional bars,
#   closes strictly rising (long) / falling (short), each opening
#   inside the prior bar's body, first bar overlaps the POI.
#
# Structure (needs lookback: scanned on ``context`` when given, else
# ``frame``; the confirming bar must lie inside ``frame``):
# - bos: a close beyond the most recent confirmed fractal swing
#   (the playbooks/common.confirmed_swings definition, swing_n=3)
#   that was confirmed before the bar, in the setup direction.
# - choch: the same break, but against a trend — the last two
#   confirmed swing highs AND lows descend (long) / ascend
#   (short).  BOS and CHoCH are exclusive: a counter-trend break
#   is CHoCH, a with-trend break is BOS.
# - failed_retest: touch the band, leave it (a close outside on the
#   setup side), return into it and hold (close on/above the
#   defended edge).  A close through the far edge kills the
#   pattern; a later touch re-arms it.  Touch + leave may happen
#   on one bar; the retest itself needs a later bar.
# - hh_hl / ll_lh: after the first touch, a confirmed swing high
#   exceeds the previous one and the following confirmed swing low
#   stays above the previous one (long); mirror for short.  The
#   pattern completes on the bar the final swing confirms.
#
# Momentum (same lookback rule):
# - displacement: a bar overlapping the POI, closing beyond the far
#   edge in the setup direction, body >= 1.0x ATR(14) — the repo's
#   QM displacement rule (qm_m5 DISPLACEMENT_BODY_ATR).
# - volume_spike: a bar overlapping the POI whose volume is >= 2x
#   the mean of the prior 20 bars (needs >= 5 prior bars).
#
# Zone-based (same lookback rule):
# - sweep_reclaim: a bar sweeps past the POI edge (low < bottom /
#   high > top) and a bar within the next 5 closes back on the
#   setup side, in the setup direction.
# - fvg_reaction: a 3-bar fair-value gap (the v2_confluence/fvg
#   _detect rule: low[2] > high[0] bullish / high[2] < low[0]
#   bearish) overlapping the POI, formed within the last 100 bars,
#   then a bar taps into the gap and closes back in the setup
#   direction without closing through it, while overlapping the
#   POI.
#
# ``confirm_at_poi`` returns ALL confirmed shapes (``shapes``) plus
# the first one (``shape``: earliest confirming bar, ties broken by
# SHAPE_PRIORITY).  ``context`` is the longer entry-TF frame the
# composing bars come from; when omitted, ``frame`` is its own
# context.  Causal by construction: every detector only uses bars
# at or before its confirming bar.  Since the 2026-10-15 trigger
# redesign, the live path branches on this result: it IS the
# trigger (zone touch + a confirming shape here).  The trigger
# additionally requires the confirming bar to overlap the POI
# band — the family itself does not require that for bos / choch /
# sweep_reclaim / hh_hl, so the trigger enforces it.
# ------------------------------------------------------------------

#: Canonical shape order: listing order for ``shapes`` and the
#: tie-break when several shapes confirm on the same bar.
SHAPE_PRIORITY = (
    "rejection", "engulfing", "pin_bar",
    "morning_star", "evening_star",
    "tweezer_bottom", "tweezer_top",
    "inside_bar_breakout",
    "three_soldiers", "three_crows",
    "bos", "choch", "failed_retest",
    "hh_hl", "ll_lh",
    "displacement", "volume_spike",
    "sweep_reclaim", "fvg_reaction",
)
#: Break-bar body >= 1.0x ATR counts as displacement — the QM rule
#: (playbooks/qm_m5 DISPLACEMENT_BODY_ATR), mirrored here so this
#: module does not import the playbook package.
DISPLACEMENT_BODY_ATR = 1.0
#: Volume-spike rule: >= this multiple of the prior-20-bar mean.
VOLUME_SPIKE_MULT = 2.0
VOLUME_LOOKBACK = 20
VOLUME_MIN_PRIOR = 5
#: Sweep -> reclaim must complete within this many bars.
SWEEP_RECLAIM_WINDOW = 5
#: Fractal swing size for BOS / CHoCH / HH-HL (repo swing_n).
SWING_N = 3
#: ATR window for displacement (repo ATR_N).
ATR_N = 14
#: FVG staleness cap (FVGTracker max_age_bars).
FVG_MAX_AGE_BARS = 100


@dataclass
class POIConfirmation:
    """Result of the POI confirmation check."""

    confirmed: bool
    shape: str  # first confirmed shape, or "none"
    bar_index: object = None  # index label of the first confirming bar
    detail: str = ""
    shapes: list = field(default_factory=list)  # ALL confirmed shapes
    bar_indices: dict = field(default_factory=dict)  # shape -> bar label


def _atr14(high, low, close, i: int, n: int = ATR_N) -> float:
    """n-bar mean true range at closed bar i (needs i >= n).

    Mirrors playbooks/common.atr exactly (same formula), kept here
    so this module does not import the playbook package.
    """
    if i < n:
        return 0.0
    trs = [max(high[k] - low[k], abs(high[k] - close[k - 1]),
               abs(low[k] - close[k - 1]))
           for k in range(i - n + 1, i + 1)]
    return float(sum(trs) / n)


def _confirmed_swings(high, low, up_to: int, n: int = SWING_N,
                      lookback: int = 200):
    """Confirmed fractal swings with index <= ``up_to``.

    Mirrors playbooks/common.confirmed_swings exactly: bar ``i`` is
    a swing high iff ``high[i]`` is strictly greater than every
    other bar in ``[i-n, i+n]`` (mirror for lows).  Returns
    ``(highs, lows)`` as ``(position, price)`` pairs.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    lo = max(n, up_to - lookback)
    hi = min(up_to, len(h) - n - 1)
    for i in range(lo, hi + 1):
        if float(h[i]) > float(np.max(h[i - n:i])) \
                and float(h[i]) > float(np.max(h[i + 1:i + n + 1])):
            highs.append((i, float(h[i])))
        if float(l[i]) < float(np.min(l[i - n:i])) \
                and float(l[i]) < float(np.min(l[i + 1:i + n + 1])):
            lows.append((i, float(l[i])))
    return highs, lows


def _is_rejection(o: float, h: float, low: float, c: float,
                  direction: int, poi_bottom: float,
                  poi_top: float) -> bool:
    body = abs(c - o)
    if direction > 0:
        lower_wick = min(o, c) - low
        return (low <= poi_bottom and c > poi_bottom
                and c > o and lower_wick >= body)
    upper_wick = h - max(o, c)
    return (h >= poi_top and c < poi_top
            and c < o and upper_wick >= body)


def _is_engulfing(o: float, h: float, low: float, c: float,
                  po: float, pc: float, direction: int,
                  poi_bottom: float, poi_top: float) -> bool:
    # The engulfing bar itself must overlap the POI band.
    if not (low <= poi_top and h >= poi_bottom):
        return False
    if direction > 0:
        # Prior bearish with a real body; current body covers it.
        return po > pc and o <= pc and c >= po
    return po < pc and o >= pc and c <= po


def _is_pin_bar(o: float, h: float, low: float, c: float,
                direction: int, poi_bottom: float,
                poi_top: float) -> bool:
    body = abs(c - o)
    rng = h - low
    if body <= 0 or rng <= 0:
        return False
    if not (low <= poi_top and h >= poi_bottom):
        return False
    mid = (h + low) / 2.0
    if direction > 0:
        wick = min(o, c) - low
        return (wick >= 2.0 * body and wick >= (2.0 / 3.0) * rng
                and body <= rng / 3.0 and c >= mid)
    wick = h - max(o, c)
    return (wick >= 2.0 * body and wick >= (2.0 / 3.0) * rng
            and body <= rng / 3.0 and c <= mid)


def _star_shape(o, h, l, c, pos: int, direction: int,
                poi_bottom: float, poi_top: float) -> str | None:
    """Morning star (long) / evening star (short) completing at pos."""
    o1, c1 = o[pos - 2], c[pos - 2]
    o2, c2 = o[pos - 1], c[pos - 1]
    o3, c3 = o[pos], c[pos]
    if not (l[pos] <= poi_top and h[pos] >= poi_bottom):
        return None
    if direction > 0:
        if not (c1 < o1):  # first bar bearish, real body
            return None
        if abs(c2 - o2) > 0.5 * abs(c1 - o1):  # small middle bar
            return None
        if not (c3 > o3 and c3 > (o1 + c1) / 2.0):
            return None
        if not (l[pos - 2] <= poi_top):  # first bar reached the zone
            return None
        return "morning_star"
    if not (c1 > o1):
        return None
    if abs(c2 - o2) > 0.5 * abs(c1 - o1):
        return None
    if not (c3 < o3 and c3 < (o1 + c1) / 2.0):
        return None
    if not (h[pos - 2] >= poi_bottom):
        return None
    return "evening_star"


def _tweezer_shape(o, h, l, c, pos: int, direction: int,
                   poi_bottom: float, poi_top: float) -> str | None:
    """Tweezer bottom (long) / top (short) completing at pos."""
    if not (l[pos - 1] <= poi_top and h[pos - 1] >= poi_bottom):
        return None
    if not (l[pos] <= poi_top and h[pos] >= poi_bottom):
        return None
    tol = 0.1 * min(h[pos - 1] - l[pos - 1], h[pos] - l[pos])
    if direction > 0:
        if not (c[pos - 1] < o[pos - 1] and c[pos] > o[pos]):
            return None
        if abs(l[pos - 1] - l[pos]) > tol:
            return None
        if not (c[pos] > c[pos - 1]):
            return None
        return "tweezer_bottom"
    if not (c[pos - 1] > o[pos - 1] and c[pos] < o[pos]):
        return None
    if abs(h[pos - 1] - h[pos]) > tol:
        return None
    if not (c[pos] < c[pos - 1]):
        return None
    return "tweezer_top"


def _is_inside_bar_breakout(o, h, l, c, pos: int, direction: int,
                            poi_bottom: float, poi_top: float) -> bool:
    """Mother (pos-2) at the POI, inside bar (pos-1), breakout (pos)."""
    if not (l[pos - 2] <= poi_top and h[pos - 2] >= poi_bottom):
        return False
    if not (h[pos - 1] <= h[pos - 2] and l[pos - 1] >= l[pos - 2]):
        return False
    if direction > 0:
        return c[pos] > h[pos - 2]
    return c[pos] < l[pos - 2]


def _soldiers_shape(o, h, l, c, pos: int, direction: int,
                    poi_bottom: float, poi_top: float) -> str | None:
    """Three soldiers (long) / three crows (short) completing at pos."""
    if not (l[pos - 2] <= poi_top and h[pos - 2] >= poi_bottom):
        return None
    if direction > 0:
        for k in (pos - 2, pos - 1, pos):
            if not (c[k] > o[k]):
                return None
        if not (c[pos - 1] > c[pos - 2] and c[pos] > c[pos - 1]):
            return None
        # Each opens inside the prior bar's body: bullish body is
        # [open, close] (low to high), so open in [o_prev, c_prev].
        if not (o[pos - 2] <= o[pos - 1] <= c[pos - 2]):
            return None
        if not (o[pos - 1] <= o[pos] <= c[pos - 1]):
            return None
        return "three_soldiers"
    for k in (pos - 2, pos - 1, pos):
        if not (c[k] < o[k]):
            return None
    if not (c[pos - 1] < c[pos - 2] and c[pos] < c[pos - 1]):
        return None
    # Bearish body is [close, open] (low to high).
    if not (c[pos - 2] <= o[pos - 1] <= o[pos - 2]):
        return None
    if not (c[pos - 1] <= o[pos] <= o[pos - 1]):
        return None
    return "three_crows"


def _finite(*vals: float) -> bool:
    return all(np.isfinite(v) for v in vals)


def _candle_hits(o, h, l, c, labels, direction: int,
                 poi_bottom: float, poi_top: float) -> dict:
    """Single / 2-3 bar candle shapes on the frame arrays.

    Returns ``{shape: index label}`` for the FIRST occurrence of
    each shape.  A bar can confirm via several shapes.
    """
    hits: dict[str, object] = {}

    def _record(shape: str, pos: int) -> None:
        if shape not in hits:
            hits[shape] = labels[pos]

    m = len(o)
    for pos in range(m):
        if not _finite(o[pos], h[pos], l[pos], c[pos]):
            continue
        if _is_rejection(o[pos], h[pos], l[pos], c[pos], direction,
                         poi_bottom, poi_top):
            _record("rejection", pos)
        if _is_pin_bar(o[pos], h[pos], l[pos], c[pos], direction,
                       poi_bottom, poi_top):
            _record("pin_bar", pos)
        if pos > 0:
            if not _finite(o[pos - 1], h[pos - 1], l[pos - 1],
                            c[pos - 1]):
                continue
            if _is_engulfing(o[pos], h[pos], l[pos], c[pos],
                             o[pos - 1], c[pos - 1], direction,
                             poi_bottom, poi_top):
                _record("engulfing", pos)
            tw = _tweezer_shape(o, h, l, c, pos, direction,
                                poi_bottom, poi_top)
            if tw is not None:
                _record(tw, pos)
        if pos > 1:
            if not _finite(o[pos - 2], h[pos - 2], l[pos - 2],
                            c[pos - 2]):
                continue
            star = _star_shape(o, h, l, c, pos, direction,
                               poi_bottom, poi_top)
            if star is not None:
                _record(star, pos)
            if _is_inside_bar_breakout(o, h, l, c, pos, direction,
                                       poi_bottom, poi_top):
                _record("inside_bar_breakout", pos)
            sol = _soldiers_shape(o, h, l, c, pos, direction,
                                  poi_bottom, poi_top)
            if sol is not None:
                _record(sol, pos)
    return hits


def _lookback_hits(o, h, l, c, v, labels, frame_positions,
                   direction: int, poi_bottom: float,
                   poi_top: float) -> dict:
    """Structure / momentum / zone shapes on the context arrays.

    ``frame_positions`` are the context positions whose bars belong
    to ``frame``: only those may be the confirming bar.  Every
    detector uses only bars at or before its confirming bar.
    Returns ``{shape: index label}`` for the FIRST occurrence of
    each shape.
    """
    hits: dict[str, object] = {}

    def _record(shape: str, pos: int) -> None:
        if shape not in hits:
            hits[shape] = labels[pos]

    n = len(o)
    frame_set = set(frame_positions)
    valid = [_finite(o[p], h[p], l[p], c[p]) for p in range(n)]
    overlap = [valid[p] and l[p] <= poi_top and h[p] >= poi_bottom
               for p in range(n)]

    def _holds(p: int) -> bool:
        # Close stayed on the setup side of the defended edge.
        return c[p] >= poi_bottom if direction > 0 \
            else c[p] <= poi_top

    def _through(p: int) -> bool:
        # Close went through the far edge: the setup is dead.
        return c[p] < poi_bottom if direction > 0 \
            else c[p] > poi_top

    def _left(p: int) -> bool:
        # Close left the band on the setup side.
        return c[p] > poi_top if direction > 0 \
            else c[p] < poi_bottom

    # --- BOS / CHoCH via confirmed fractal swings ----------------
    highs_sw, lows_sw = _confirmed_swings(h, l, up_to=n - 1, n=SWING_N)
    for j in sorted(frame_set):
        if not valid[j]:
            continue
        # Reference = latest swing confirmed strictly before bar j.
        ref_h = [s for s in highs_sw if s[0] + SWING_N <= j - 1]
        ref_l = [s for s in lows_sw if s[0] + SWING_N <= j - 1]
        if direction > 0:
            if ref_h and c[j] > ref_h[-1][1]:
                downtrend = (len(ref_h) >= 2 and len(ref_l) >= 2
                             and ref_h[-1][1] < ref_h[-2][1]
                             and ref_l[-1][1] < ref_l[-2][1])
                _record("choch" if downtrend else "bos", j)
        else:
            if ref_l and c[j] < ref_l[-1][1]:
                uptrend = (len(ref_h) >= 2 and len(ref_l) >= 2
                           and ref_h[-1][1] > ref_h[-2][1]
                           and ref_l[-1][1] > ref_l[-2][1])
                _record("choch" if uptrend else "bos", j)

    # --- failed retest: touch -> leave -> return + hold ----------
    # A bar is never its own retest: the leave bar arms the
    # pattern, the retest needs a LATER bar.  A touch bar that
    # also closes outside arms on the same bar.
    last_touch: int | None = None
    left = False
    for p in range(n):
        if not valid[p]:
            continue
        if _through(p):
            last_touch, left = None, False
            continue
        if last_touch is None:
            if overlap[p]:
                last_touch = p
                if _left(p):
                    left = True
            continue
        if _left(p):
            left = True
            continue
        if left and p > last_touch and overlap[p] and _holds(p):
            if p in frame_set:
                _record("failed_retest", p)
            last_touch = p
            left = _left(p)  # retest bar may itself leave

    # --- HH+HL / LL+LH via confirmed swings after the touch ------
    first_touch = next((p for p in range(n) if overlap[p]), None)
    if first_touch is not None:
        highs_after = [s for s in highs_sw if s[0] > first_touch]
        lows_after = [s for s in lows_sw if s[0] > first_touch]
        if direction > 0:
            for a in range(len(highs_after) - 1):
                h1, h2 = highs_after[a], highs_after[a + 1]
                if h2[1] <= h1[1]:
                    continue
                lb_before = [s for s in lows_after if s[0] < h2[0]]
                lb_after = [s for s in lows_after if s[0] > h2[0]]
                if not lb_before or not lb_after:
                    continue
                if lb_after[0][1] > lb_before[-1][1]:
                    confirm_pos = lb_after[0][0] + SWING_N
                    if confirm_pos in frame_set:
                        _record("hh_hl", confirm_pos)
                    break
        else:
            for a in range(len(lows_after) - 1):
                l1, l2 = lows_after[a], lows_after[a + 1]
                if l2[1] >= l1[1]:
                    continue
                hb_before = [s for s in highs_after if s[0] < l2[0]]
                hb_after = [s for s in highs_after if s[0] > l2[0]]
                if not hb_before or not hb_after:
                    continue
                if hb_after[0][1] < hb_before[-1][1]:
                    confirm_pos = hb_after[0][0] + SWING_N
                    if confirm_pos in frame_set:
                        _record("ll_lh", confirm_pos)
                    break

    # --- displacement: impulsive body away from the zone ---------
    for j in sorted(frame_set):
        if not valid[j] or not overlap[j]:
            continue
        if direction > 0:
            if not (c[j] > o[j] and c[j] > poi_top):
                continue
        else:
            if not (c[j] < o[j] and c[j] < poi_bottom):
                continue
        atr_v = _atr14(h, l, c, j)
        if atr_v > 0 and abs(c[j] - o[j]) >= \
                DISPLACEMENT_BODY_ATR * atr_v:
            _record("displacement", j)

    # --- volume spike on a bar at the POI -------------------------
    if v is not None:
        for j in sorted(frame_set):
            if not valid[j] or not overlap[j]:
                continue
            if not np.isfinite(v[j]):
                continue
            prior = np.asarray(v[max(0, j - VOLUME_LOOKBACK):j],
                               dtype=float)
            prior = prior[np.isfinite(prior)]
            if len(prior) < VOLUME_MIN_PRIOR:
                continue
            if v[j] >= VOLUME_SPIKE_MULT * float(prior.mean()):
                _record("volume_spike", j)

    # --- sweep of the POI edge + reclaim --------------------------
    for j in sorted(frame_set):
        if not valid[j]:
            continue
        for i in range(max(0, j - SWEEP_RECLAIM_WINDOW), j):
            if not valid[i]:
                continue
            if direction > 0:
                if l[i] < poi_bottom and c[j] > poi_bottom \
                        and c[j] > o[j]:
                    _record("sweep_reclaim", j)
                    break
            else:
                if h[i] > poi_top and c[j] < poi_top \
                        and c[j] < o[j]:
                    _record("sweep_reclaim", j)
                    break

    # --- FVG tap + reaction ---------------------------------------
    for k in range(1, n - 1):
        if not (valid[k - 1] and valid[k] and valid[k + 1]):
            continue
        if direction > 0:
            if not (l[k + 1] > h[k - 1]):  # bullish gap
                continue
            z_bottom, z_top = h[k - 1], l[k + 1]
        else:
            if not (h[k + 1] < l[k - 1]):  # bearish gap
                continue
            z_bottom, z_top = h[k + 1], l[k - 1]
        if not (z_top >= poi_bottom and z_bottom <= poi_top):
            continue  # gap must overlap the POI
        for j in sorted(frame_set):
            if j <= k + 1 or j - (k + 1) > FVG_MAX_AGE_BARS:
                continue
            if not valid[j] or not overlap[j]:
                continue
            if direction > 0:
                if l[j] <= z_top and c[j] > z_bottom \
                        and c[j] > o[j]:
                    _record("fvg_reaction", j)
                    break
            else:
                if h[j] >= z_bottom and c[j] < z_top \
                        and c[j] < o[j]:
                    _record("fvg_reaction", j)
                    break
    return hits


def _extract_arrays(df):
    """(open, high, low, close, volume|None, labels) or None."""
    try:
        cols = df.columns
        if not all(col in cols
                   for col in ("open", "high", "low", "close")):
            return None
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        v = df["volume"].to_numpy(dtype=float) \
            if "volume" in cols else None
        return o, h, l, c, v, list(df.index)
    except Exception:
        return None


def confirm_at_poi(frame, direction: int, poi_bottom: float,
                   poi_top: float, context=None) -> POIConfirmation:
    """Scan for POI confirmation across the full shape family.

    ``frame`` is the composing window (entry-TF bars at the POI);
    ``context`` is the longer entry-TF frame those bars come from
    (lookback for structure / momentum / zone shapes).  When
    ``context`` is omitted or unusable, ``frame`` is its own
    context.  The confirming bar of every shape must lie inside
    ``frame``.  Never raises on data (NaN / empty) — only on a bad
    contract (bad direction, inverted band).
    """
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1/-1, got {direction!r}")
    if poi_bottom > poi_top:
        raise ValueError(
            f"inverted POI band: bottom {poi_bottom} > top {poi_top}")
    if frame is None or len(frame) == 0:
        return POIConfirmation(False, "none", None, "no bars")
    f_arrays = _extract_arrays(frame)
    if f_arrays is None:
        return POIConfirmation(False, "none", None, "no OHLC columns")
    ctx_arrays = f_arrays
    if context is not None and len(context):
        extracted = _extract_arrays(context)
        if extracted is not None:
            ctx_arrays = extracted
    o, h, l, c, v, labels = f_arrays
    candle = _candle_hits(o, h, l, c, labels, direction,
                          poi_bottom, poi_top)
    co, ch, cl, cc, cv, clabels = ctx_arrays
    frame_labels = set(labels)
    frame_positions = [p for p, lab in enumerate(clabels)
                       if lab in frame_labels]
    if not frame_positions and len(labels) and len(clabels):
        # Fall back to time-range overlap when labels do not align.
        frame_positions = [p for p, lab in enumerate(clabels)
                           if labels[0] <= lab <= labels[-1]]
    lookback = _lookback_hits(co, ch, cl, cc, cv, clabels,
                              frame_positions, direction,
                              poi_bottom, poi_top)
    hits = {**candle, **lookback}
    if not hits:
        return POIConfirmation(False, "none", None,
                               "no confirming candle at POI",
                               shapes=[], bar_indices={})
    prio = {s: i for i, s in enumerate(SHAPE_PRIORITY)}
    best_shape, best_label = min(
        hits.items(), key=lambda kv: (kv[1], prio.get(kv[0], 99)))
    shapes = sorted(hits, key=lambda s: prio.get(s, 99))
    return POIConfirmation(
        True, best_shape, best_label,
        f"confirmed via {', '.join(shapes)}",
        shapes=shapes, bar_indices=dict(hits))
