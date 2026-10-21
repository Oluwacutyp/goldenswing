"""Scenario posture composition (P3, spec v3 S1/S3/S11/S12).

ONE regime brain: the bedrock five-axis classification
(``live/regime.py``) plus the day object (``live/day_context.py``)
plus the session clock plus the loss ladder compose into a
single :class:`Posture` the decision flow consults.  The old
second machine (the boundary tracker) keeps its sweep/breakout
geometry job but no longer selects playbooks — selection is
bedrock + day + score (S1).

A posture is NOT a veto with better PR.  It decides which
families HUNT, at what size scalar, with which target
philosophy — and every shelving is journaled with its reason
(stand-aside is a reasoned posture, never a silent gate
failure).  The only postures that stop entries outright are
the ones whose scenario IS waiting: an event spike, a dead
(rollover/lunch + spread-confirmed) window, and the loss
ladder's killzone cooldown — each journaled by name.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import day_context as dc

TF_LADDER = ["M1", "M5", "M15", "H1", "H4", "D1", "W1"]


def _step(tf: str, n: int) -> str:
    try:
        i = TF_LADDER.index(tf)
    except ValueError:
        return tf
    j = min(len(TF_LADDER) - 1, max(0, i + n))
    return TF_LADDER[j]


# ------------------------------------------------------------ families

def dir_int(d) -> int:
    """Normalize a direction read: bedrock speaks "up"/"down"/
    "flat"; the desk's arithmetic speaks +1/-1/0.  Accepts both."""
    if isinstance(d, str):
        return {"up": 1, "down": -1}.get(d.strip().lower(), 0)
    try:
        return int(d)
    except (TypeError, ValueError):
        return 0


def family_of(playbook_name: str, regime_scope: str = "") -> str:
    """Decision-flow family of a playbook — the unit postures
    shelve and the style ceilings count."""
    name = str(playbook_name or "")
    if regime_scope == "range" or name.startswith("range"):
        return "range"
    if name in ("qm_m5",) or "scalp" in name:
        return "scalp"
    if name in ("narrative", "arrival_zone", "qm_d1"):
        return "zone"
    return "continuation"


# ------------------------------------------------------- session posture

@dataclass(frozen=True)
class SessionPosture:
    name: str            # asia | london | ny | ny_pm | lunch |
    #                      rollover | off
    size_scalar: float = 1.0
    dead: bool = False   # dead window: no NEW base entries
    note: str = ""


def session_posture(now, *, spread_ratio: float | None = None
                    ) -> SessionPosture:
    """Session posture from the DST-aware NY-anchored clock
    (``reader.session_timeframes`` wins over UTC buckets, S3).

    The rollover dead state is spread-MEASURED: the clock
    (17:00-18:00 ET) is only the prior; a spread at >= 2x the
    session norm is the proof.  Lunch (12:00-13:30 ET) is dead
    only with the same spread proof.
    """
    try:
        from ..reader.session_timeframes import session_of
        sess, _kz = session_of(pd.Timestamp(now))
    except Exception:
        sess = "off"
    try:
        t = pd.Timestamp(now)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        ny = t.tz_convert("America/New_York")
        et = ny.hour + ny.minute / 60.0
    except Exception:
        et = -1.0
    wide = spread_ratio is not None and spread_ratio >= 2.0
    if 17.0 <= et < 18.0:
        return SessionPosture(
            "rollover", size_scalar=0.0, dead=bool(wide),
            note=("rollover 17:00-18:00 ET"
                  + (" — spread blown (dead)" if wide
                     else " — spread normal, tradeable")))
    if sess == "asia":
        return SessionPosture(
            "asia", size_scalar=0.85,
            note="asia build: range under construction; "
                 "expansion entries de-rated")
    if sess == "london":
        return SessionPosture("london", note="london expansion")
    if sess == "overlap":
        return SessionPosture("ny", note="NY open / overlap")
    if sess == "newyork":
        if 12.0 <= et < 13.5:
            return SessionPosture(
                "lunch", size_scalar=0.6, dead=bool(wide),
                note=("NY lunch"
                      + (" — spread confirms dead" if wide
                         else " — low energy, de-rated")))
        if et >= 13.5:
            return SessionPosture(
                "ny_pm", size_scalar=0.85,
                note="NY afternoon: targets = session levels / "
                     "prior value, not open space")
        return SessionPosture("ny", note="NY session")
    return SessionPosture("off", size_scalar=0.7,
                          note="off hours: thin book, de-rated")


# ------------------------------------------------------------ loss ladder

@dataclass(frozen=True)
class LadderState:
    """S11 graduated loss response.  Posture only — the daily
    risk cap in RiskGuards remains the only HARD stop."""
    consecutive_losses: int = 0
    layer_cap_delta: int = 0          # -1 at >= 2 consecutive
    traffic_cap: str | None = None    # "YELLOW" at >= 2
    cooldown: bool = False            # >= 3: until next killzone
    style_blocked: frozenset = frozenset()   # churn ceilings hit

    def journal(self) -> dict:
        return {"consec_losses": self.consecutive_losses,
                "layer_cap_delta": self.layer_cap_delta,
                "traffic_cap": self.traffic_cap,
                "cooldown": self.cooldown,
                "style_blocked": sorted(self.style_blocked)}


def loss_ladder(consecutive_losses: int, *,
                style_counts: dict | None = None,
                style_ceiling: int = 0) -> LadderState:
    blocked = frozenset()
    if style_ceiling and style_counts:
        blocked = frozenset(
            s for s, n in style_counts.items()
            if n >= style_ceiling)
    cl = int(consecutive_losses or 0)
    return LadderState(
        consecutive_losses=cl,
        layer_cap_delta=-1 if cl >= 2 else 0,
        traffic_cap="YELLOW" if cl >= 2 else None,
        cooldown=cl >= 3,
        style_blocked=blocked)


# ---------------------------------------------------------------- posture

@dataclass
class Posture:
    name: str
    reason: str
    size_scalar: float = 1.0
    layer_ceiling_cap: int | None = None
    block_entries: bool = False       # spike / dead / cooldown
    resting_interest_live: bool = True
    target_philosophy: str = "trend"  # trend | range | value
    shelved: dict = field(default_factory=dict)  # family -> reason
    counter_trend_range_shelved: bool = False
    session: SessionPosture | None = None
    ladder: LadderState | None = None

    def shelved_reason(self, family: str, *, direction: int = 0,
                       trend_direction=0) -> str | None:
        """Why ``family`` does not hunt in this posture (or
        None).  Direction-aware for the trend-day range rule:
        on a trend day only the COUNTER-trend edge fade is
        shelved — the with-trend edge is a pullback and hunts.
        ``trend_direction`` accepts bedrock's string form too."""
        if family in self.shelved:
            return self.shelved[family]
        td = dir_int(trend_direction)
        if family == "range" and self.counter_trend_range_shelved \
                and direction and td \
                and int(direction) != td:
            return (f"posture {self.name}: counter-trend edge "
                    f"fade shelved on a trend day (edge fades "
                    f"against a trend day bleed — S2)")
        return None

    def journal(self) -> dict:
        return {"posture": self.name, "reason": self.reason,
                "size_scalar": self.size_scalar,
                "philosophy": self.target_philosophy,
                "shelved": dict(self.shelved),
                "session": self.session.name if self.session
                else None}


def ny_checkpoint(day: "dc.DayContext | None", *,
                  ny_close: float, ny_low: float, ny_high: float
                  ) -> str | None:
    """The NY-open acceptance checkpoint (S3): London's leg is
    judged at the NY open.  London swept a prior-day extreme;
    NY either ACCEPTS it (holds/trades beyond the swept level —
    continuation posture) or REJECTS it (back inside the prior
    range — reversal posture).  Returns "accepted_up" |
    "accepted_down" | "rejected_up" | "rejected_down" | None
    (London never left the prior range — nothing to judge).
    Pure; the runner evaluates it on the first NY H1 close and
    journals the event once per day.
    """
    if day is None or not day.prior or not day.session_levels:
        return None
    l_hi = day.session_levels.get("london_high")
    l_lo = day.session_levels.get("london_low")
    p_hi = float(day.prior.get("high", 0.0))
    p_lo = float(day.prior.get("low", 0.0))
    if l_hi is None or l_lo is None or p_hi <= p_lo:
        return None
    if l_hi > p_hi:      # London leg swept the prior high
        if ny_close < p_hi:
            return "rejected_up"
        if ny_low >= p_hi:
            return "accepted_up"
        return None      # straddling the level: no verdict yet
    if l_lo < p_lo:      # London leg swept the prior low
        if ny_close > p_lo:
            return "rejected_down"
        if ny_high <= p_lo:
            return "accepted_down"
    return None


def compose_posture(*, bedrock, day: "dc.DayContext | None",
                    session: SessionPosture,
                    ladder: LadderState,
                    news_block: bool = False,
                    checkpoint: str | None = None) -> Posture:
    """Compose the one posture.  Priority (highest first):
    event spike > bedrock spike > loss cooldown > dead window >
    coil > choppy > trend day > rotation day > open.  Every
    branch names itself and its reason — the journal always
    carries both.
    """
    struct = str(getattr(bedrock, "structure", "") or "")
    volatility = str(getattr(bedrock, "volatility", "") or "")
    direction = dir_int(getattr(bedrock, "direction", 0))
    dtype = day.day_type if day is not None else dc.DAY_DEVELOPING
    base = dict(session=session, ladder=ladder)

    if news_block:
        return Posture(
            "event_spike",
            "red-folder print window: the desk hides (existing "
            "news pause) — post-print reaction state follows",
            block_entries=True, resting_interest_live=True,
            target_philosophy="value", **base)
    if volatility == "spike":
        return Posture(
            "spike_stand_aside",
            "bedrock volatility reads spike (news-blowoff "
            "geometry): the trade is waiting for the first "
            "retracement structure — no entries into the spike",
            block_entries=True, target_philosophy="value", **base)
    if ladder.cooldown:
        return Posture(
            "loss_cooldown",
            f"{ladder.consecutive_losses} consecutive losses: "
            "cooldown until the next killzone — the desk resets, "
            "it does not revenge-trade (S11)",
            block_entries=True, **base)
    if session.dead:
        return Posture(
            "dead_window",
            f"{session.name}: {session.note} — new entries "
            "shelved; resting zone interest stays live (it is "
            "broker-held, spread pays once at the fill)",
            block_entries=True, resting_interest_live=True,
            **base)
    # No bedrock read at all (confidence 0.0 — classify_regime's
    # insufficient-data return: every axis uncomputed, the label
    # explicitly "do not trust"): composing scenario shelving
    # from the day type alone would manufacture a conviction the
    # desk does not hold, and the gate downstream would shelve
    # armed families on an empty read (the P3 tick-path
    # regression: 34-40-bar scenario frames shelved a confirmed
    # POI/continuation trigger as "rotation day").  The posture
    # is OPEN — the score and the hard lines decide.  The gates
    # above (news block, spike, loss cooldown, dead window) are
    # not bedrock reads and still apply.
    _conf = getattr(bedrock, "confidence", None)
    if _conf is not None and float(_conf) <= 0.0:
        return Posture(
            "open",
            "bedrock has no classification (confidence 0.0 — "
            "insufficient data): no scenario posture is "
            "composed; families hunt under the score and the "
            "hard lines", **base)
    if checkpoint and checkpoint.startswith("rejected"):
        leg = "up" if checkpoint.endswith("up") else "down"
        return Posture(
            "ny_rejection",
            f"NY rejected London's {leg} leg at the acceptance "
            "checkpoint: reversal posture — continuation shelved, "
            "value/fade families hunt the trip back",
            shelved={"continuation": "NY rejection: "
                                     "continuation shelved"},
            target_philosophy="value", **base)
    if struct == "compression" or dtype == dc.DAY_NONTREND:
        return Posture(
            "coil",
            "compression / nontrend day: no business done — "
            "mid-range entries shelved; resting limits at real "
            "zones and range edges (if a range defines) hunt",
            shelved={"continuation": "coil: expansion entries "
                                     "shelved until the spring",
                     "scalp": "coil: scalps shelved — no fuel"},
            layer_ceiling_cap=1, size_scalar=0.75,
            target_philosophy="range", **base)
    if struct == "choppy":
        return Posture(
            "choppy_range",
            "bedrock reads choppy overlap: range-edge posture "
            "at reduced size; continuation and scalp families "
            "shelved (they bleed in chop)",
            shelved={"continuation": "choppy: trend entries "
                                     "shelved",
                     "scalp": "choppy: scalps shelved"},
            size_scalar=0.5, target_philosophy="range", **base)
    if dtype == dc.DAY_TREND or (
            struct == "clean_trend" and dtype != dc.DAY_RANGE):
        return Posture(
            "trend_day",
            "trend day posture: continuation + zone families "
            "hunt; counter-trend edge fades shelved; full draws "
            "as targets",
            counter_trend_range_shelved=True,
            target_philosophy="trend", **base)
    if dtype in (dc.DAY_RANGE, dc.DAY_NEUTRAL) or struct == "range":
        return Posture(
            "rotation_day",
            "rotation/range day: value trades hunt (edges + "
            "VA traverse); continuation family shelved — "
            "mid-range is the no-trade middle",
            shelved={"continuation": "rotation day: "
                                     "continuation shelved"},
            target_philosophy="range", **base)
    return Posture(
        "open",
        "no dominant day scenario: all families hunt at their "
        "floors",
        target_philosophy="trend" if direction != 0 else "range",
        **base)


# ------------------------------------------------------------- clock float

def structure_clarity(frame) -> float:
    """How clean the structure on a frame is, 0..1 — the float
    signal of S12.  Coherence of the defended-pivot sequence
    (pivots stepping one way, or holding) plus the strong-body
    fraction of the last 20 bars.  A TF whose pivots churn
    through each other is not carrying readable structure NOW,
    whatever its name."""
    try:
        import numpy as np
        from .multitf import find_swings
        if frame is None or len(frame) < 24:
            return 0.0
        highs, lows = find_swings(
            frame["high"].to_numpy(dtype=float),
            frame["low"].to_numpy(dtype=float), 3)
        pivots = sorted([(i, p, 1) for i, p in highs]
                        + [(i, p, -1) for i, p in lows])
        coherence = 0.5
        if len(pivots) >= 4:
            steps = 0
            clean = 0
            for (i0, p0, s0), (i1, p1, s1) in zip(
                    pivots, pivots[1:]):
                if s0 == s1:
                    steps += 1
                    if (p1 - p0) * s0 >= 0:
                        clean += 1
            if steps:
                coherence = clean / steps
        body = (frame["close"] - frame["open"]).abs()
        rng = (frame["high"] - frame["low"]).replace(0, np.nan)
        strong = float((body / rng).tail(20).mean())
        if strong != strong:  # NaN
            strong = 0.4
        return max(0.0, min(1.0, 0.6 * coherence + 0.4 * strong))
    except Exception:
        return 0.0


@dataclass(frozen=True)
class ClockChoice:
    style: str
    score_tf: str        # frame the asymmetry score reads
    funding_tf: str      # frame the funding detectors read
    trigger_tf: str      # journaled trigger clock
    stand_aside: bool = False
    reason: str = ""


# Per-style default clocks (S12 + multitf presets): the desk's
# H1 evaluation path is the intraday/swing score clock; funding
# detectors read the trigger frame.
_STYLE_CLOCKS = {
    "swing": ("H1", "M15", "H1"),
    "intraday": ("H1", "M15", "M15"),
    "scalp": ("M5", "M5", "M5"),
    "scalp_m1": ("M5", "M1", "M1"),
    "qm_d1": ("D1", "D1", "D1"),
}


def clock_for(style: str, *, vol_regime: str = "NORMAL",
              clarity: dict | None = None) -> ClockChoice:
    """Float the clocks by live volatility + structure clarity
    (S12).  Invariants never move (structure / liquidity /
    value); clocks are parameters and they float:

    - HIGH / EXTREME vol: trigger + funding promote one step
      (M5 noise stops being signal; M15/H1 carries the read).
    - Crush (LOW): the scalp stands aside entirely; other
      styles' funding demotes one step (finer read of a slow
      tape).
    - Clarity: at NORMAL vol, if the score frame's structure
      is mud (< 0.30) and the frame one step up is clearly
      cleaner (+ 0.15), the score reads the cleaner frame.
    """
    score_tf, funding_tf, trigger_tf = _STYLE_CLOCKS.get(
        style, _STYLE_CLOCKS["intraday"])
    reasons = [f"{style} default {score_tf}/{funding_tf}"]
    if vol_regime in ("HIGH", "EXTREME"):
        funding_tf = _step(funding_tf, +1)
        trigger_tf = _step(trigger_tf, +1)
        if style in ("scalp", "scalp_m1"):
            score_tf = _step(score_tf, +1)
        reasons.append(f"vol {vol_regime}: trigger/funding "
                       "promoted one step")
    elif vol_regime == "LOW":
        if style in ("scalp", "scalp_m1"):
            return ClockChoice(
                style, score_tf, funding_tf, trigger_tf,
                stand_aside=True,
                reason=f"{style}: volatility crush — scalp "
                       "stands aside (S12); there is no fuel "
                       "for its stop geometry")
        funding_tf = _step(funding_tf, -1)
        reasons.append("vol crush: funding demoted one step")
    if clarity and vol_regime == "NORMAL":
        here = clarity.get(score_tf)
        up_tf = _step(score_tf, +1)
        up = clarity.get(up_tf)
        if here is not None and up is not None \
                and here < 0.30 and up >= here + 0.15:
            reasons.append(
                f"clarity: {score_tf} structure muddy "
                f"({here:.2f}) vs {up_tf} ({up:.2f}) — score "
                "reads the cleaner frame")
            score_tf = up_tf
    return ClockChoice(style, score_tf, funding_tf, trigger_tf,
                       reason="; ".join(reasons))
