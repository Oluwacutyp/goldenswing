"""Judgment (owner order 2026-11-27 — the formula core is
RETIRED, and no AI judges a trade).

"What you build is just code-set conditions you align
together for it to follow" — that shape is dead.  "You just
need to fix the code" (owner, 2026-11-27) — the brain
having to use a key is a drag, and an AI looking at the
market is not a system.  So the judgment is CODE, always.
No LLM/AI judges a trade (owner ruling 2026-10-23: LLM out
of the live decision path, explanation drafting only).

The judgment, in his words (2026-10-23): market-state
estimation → competing scenarios → evidence evaluation →
decision/abstention.

CODE (this module) — the verdict is the scenarios
competing on the market's own measured evidence:
- Market-state estimation: the measured evidence
  (``DecisionStructure``) — displacement vs absorption,
  acceptance-by-closes, follow-through, phase by structure,
  HTF bias, Asymmetry.  The measurement reads the bars at
  decision time through the organs that already measure them
  (live/displacement.py's thresholds, live/asymmetry.py's
  score, the funnel's HTF biases and regime label) — never
  a hand-set prior, weight, or bar.
- Competing scenarios: LONG and SHORT.  The evidence
  supports the scenario with more reads behind it.  A tie
  is a conflict; no reads is no evidence.  The support
  comparison is the competition — not a threshold (no
  minimum count, no ratio bar).  The 26 Nov case is the
  proof it must be support, not unanimity: the evidence
  supported LONG 3-to-2 and the verdict followed LONG.
- Evidence evaluation: the decision follows the supported
  scenario — but only when the candle sequence has answered
  at the level (a reaction read for the candidate): a level
  is a question, the candle sequence is the answer.
  Confirmations are evidence, never vetoes.  The payoff is
  the structure's own geometry, recorded, never a verdict
  condition — "no target" is not a trade-existence fact.
- Decision/abstention: take when the supported scenario is
  the candidate's and the reaction has answered; abstain
  when the evidence supports neither, conflicts, fights
  the candidate, or the reaction has not answered.
- Hard stops ONLY for trade-existence facts.  The verdict
  must change when the evidence changes — the test proves
  it (same setup, two different market-structure inputs,
  verdict changes), and the 26 Nov screenshot case is the
  acceptance case (a pullback sell against an upward
  displacement reads BUY, not SELL).

EXPLANATION DRAFTING (execution/judgment_brain.py): a
brain, if one is configured, drafts a plain-words
explanation of a verdict ALREADY DECIDED — never the
decision.  The live order path never consults it.  The
GS_JUDGMENT_BRAIN_* env vars configure the drafting
provider only; they gate no trade decision.

Hard risk invariants stay OUTSIDE this model (the ruling):
the friction gate, RiskGuards/RiskPool and the drawdown
halt keep their own authority on the order path; a judgment
verdict never steers, sizes, gates or vetoes an order.

Evidence honesty rules (mirroring the funnel record's own
doctrine):
- Correlated evidence counts ONCE per correlation group (the
  record's ``evidence_groups``): ten prints of one detector
  are one vote, not ten.
- Counter-bias and counter-thesis evidence is WEIGHED, never
  a veto (M5 counter-D1 = penalty/evidence, never a hard
  block).
- Absent inputs stay absent: no targets -> the journey room
  is the continuation payoff; no score -> no score edge.
  Nothing is invented to fill a gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .funnel import (
    CATEGORY_DIRECTION,
    CATEGORY_INVALID_GEOMETRY,
    CATEGORY_QUALITY,
    CATEGORY_REGIME,
)

# The displacement organ's own thresholds (live/displacement.py):
# a leg of >= 1.5 ATR in <= 3 entry bars is displacement; >= 1.0
# ATR is weak displacement.  Imported, never re-picked here.
try:
    from ..live.displacement import (
        DISP_MAX_M15_BARS,
        DISP_MIN_ATR,
        DISP_WEAK_MIN_ATR,
    )
except Exception:  # noqa: BLE001 - organ unavailable in a bare import
    DISP_MIN_ATR = 1.5
    DISP_WEAK_MIN_ATR = 1.0
    DISP_MAX_M15_BARS = 3


# ------------------------------------------------------------ helpers
def _f(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_regime_label(label) -> dict:
    """The bedrock label is ``direction/structure/volatility/
    phase/session`` (live/regime.py ``label()``).  Parse what is
    there; absent parts stay None — never guessed."""
    out = {"direction": None, "structure": None,
           "volatility": None, "phase": None, "session": None}
    if not label:
        return out
    parts = str(label).split("/")
    for key, part in zip(out, parts):
        out[key] = part or None
    return out


def _structure_class(regime: dict, inferred: dict) -> str:
    structure = regime.get("structure") \
        or (inferred or {}).get("regime") or ""
    structure = str(structure).lower()
    if structure in ("clean_trend", "breakout", "trend"):
        return "trend"
    if structure in ("range", "compression", "choppy",
                     "reversal"):
        return structure
    return "unknown"


def _bias_alignment(observed: dict, direction: int) -> tuple[int, bool]:
    """(alignment in [-2, +2], d1_opposed) from the recorded HTF
    biases.  Absent bias = no vote, never an assumed one."""
    align = 0
    d1_opposed = False
    for key, is_d1 in (("bias_d1", True), ("bias_h4", False)):
        bias = (observed or {}).get(key)
        if bias in (1, -1):
            if int(bias) == direction:
                align += 1
            else:
                align -= 1
                if is_d1:
                    d1_opposed = True
    return align, d1_opposed


# ------------------------------------------------------------ the measurement
#: How many recent entry-clock candles ride the chart read.
#: A presentation window, not a decision input — the verdict
#: never counts these bars; they are the record's evidence
#: of the chart at decision time.
CHART_BARS = 12


def _chart_bars(frame) -> list[dict]:
    """The entry-clock frame's recent candles as plain data
    (most recent last): time, open, high, low, close.  The
    chart the record carries — the bars themselves, not just
    the computed numbers.  [] when there is no frame."""
    if frame is None:
        return []
    try:
        tail = frame.tail(CHART_BARS)
        out = []
        for pos, (idx, row) in enumerate(tail.iterrows()):
            # A timestamp index reads as ISO; anything else
            # (a RangeIndex, a plain int) reads as its position.
            stamp = (idx.isoformat() if hasattr(idx, "isoformat")
                     else pos)
            out.append({
                "t": stamp,
                "o": round(float(row["open"]), 2),
                "h": round(float(row["high"]), 2),
                "l": round(float(row["low"]), 2),
                "c": round(float(row["close"]), 2),
            })
        return out
    except Exception:  # noqa: BLE001 - a frame that cannot be
        # read as candles is no chart, never a crash: the
        # reaction reads are absent with it, and the chart
        # is absent with them.
        return []


@dataclass
class DecisionStructure:
    """The market's own measured evidence at decision time —
    the one object the verdict reads.  Every field is a
    measurement (from the bars, the zones, the Asymmetry
    score, the HTF biases) or None (absent, never invented).

    ``reads`` is the verdict's raw material: one entry per
    measured read — (name, direction, note) — where direction
    is +1 (favours LONG), -1 (favours SHORT) or 0 (no read).
    The verdict is the scenarios competing on them: the
    scenario with more reads behind it is the one the
    evidence supports.  It never re-weighs them."""
    # Reaction reads from the entry-clock bars (when frames given).
    displacement: str = "absent"  # absent | none | weak | strong
    displacement_direction: int = 0
    displacement_atr: float | None = None
    acceptance: str = "absent"  # absent | rejected | accepted
    acceptance_direction: int = 0
    acceptance_closes: int = 0
    follow_through: str = "absent"  # absent | held | failed
    follow_through_direction: int = 0
    retracement_frac: float | None = None
    # Structure reads (from the funnel's measured evidence).
    phase: str = "unknown"  # trend | range | compression | choppy | reversal | unknown
    bias_d1: int | None = None
    bias_h4: int | None = None
    asymmetry_total: float | None = None
    asymmetry_floor: float | None = None
    asymmetry_flipped: bool = False
    room_r: float | None = None
    session: str | None = None
    adr_pct: float | None = None
    # The current MarketSituation contribution, when one was
    # read into the evidence: recorded in the market-state
    # picture, never a vote on the verdict.
    situation: dict | None = None
    entry_level: float | None = None
    stop_level: float | None = None
    target_level: float | None = None
    # The recent entry-clock candles (the chart the record
    # carries) and the clock they belong to.  [] / None when
    # there was no frame — the reaction reads are absent
    # then, and the chart is absent with them, never invented.
    bars: list = field(default_factory=list)
    chart_tf: str | None = None
    reads: list = field(default_factory=list)

    def reads_for(self, direction: int) -> list:
        return [r for r in self.reads if r[1] == direction]

    def reads_against(self, direction: int) -> list:
        return [r for r in self.reads if r[1] == -direction]

    def to_dict(self) -> dict:
        """The chart read, plain data — the record's
        evidence, verbatim (the same evidence the verdict
        reads)."""
        out = {
            "displacement": self.displacement,
            "displacement_direction": self.displacement_direction,
            "displacement_atr": self.displacement_atr,
            "acceptance": self.acceptance,
            "acceptance_direction": self.acceptance_direction,
            "acceptance_closes": self.acceptance_closes,
            "follow_through": self.follow_through,
            "follow_through_direction": self.follow_through_direction,
            "retracement_frac": self.retracement_frac,
            "phase": self.phase,
            "bias_d1": self.bias_d1,
            "bias_h4": self.bias_h4,
            "asymmetry_total": self.asymmetry_total,
            "asymmetry_floor": self.asymmetry_floor,
            "asymmetry_flipped": self.asymmetry_flipped,
            "room_r": self.room_r,
            "session": self.session,
            "adr_pct": self.adr_pct,
            "entry_level": self.entry_level,
            "stop_level": self.stop_level,
            "target_level": self.target_level,
            "chart": {"tf": self.chart_tf, "bars": self.bars},
            "reads": [{"name": r[0], "direction": r[1],
                       "note": r[2]} for r in self.reads],
        }
        if self.situation:
            # The situation rides along, never a vote.
            out["situation"] = self.situation
        return out


def _atr_last(frame) -> float | None:
    """The 14-period ATR at the frame's last bar (volatility is
    only the buffer — the scale the displacement is measured
    in, never the read itself)."""
    try:
        from ..reader.indicators import atr as _atr_series

        series = _atr_series(frame)
        value = float(series.iloc[-1])
        return value if value > 0 else None
    except Exception:
        return None


def _measure_reaction(frame, *, entry: float, stop: float,
                      direction: int) -> dict:
    """Displacement / acceptance / follow-through from the
    entry-clock bars, at the candidate's level.

    Displacement: the farthest the closes moved from the
    entry level in EITHER direction over the last
    DISP_MAX_M15_BARS entry bars, in ATR — the organ's own
    thresholds (DISP_MIN_ATR strong, DISP_WEAK_MIN_ATR
    weak).  The direction is the move's direction, never
    assumed to be the candidate's.  Acceptance: the last two
    closed bars' closes beyond the entry level in the
    displacement direction (accepted) vs back inside
    (rejected).  Follow-through: after the displacement
    leg's extreme, the retracement fraction — held at <=
    50% retraced, failed beyond it (the organ's rule)."""
    out = {"displacement": "none", "displacement_direction": 0,
           "displacement_atr": None, "acceptance": "rejected",
           "acceptance_direction": 0, "acceptance_closes": 0,
           "follow_through": "failed",
           "follow_through_direction": 0,
           "retracement_frac": None}
    if frame is None or len(frame) < 2:
        out.update({"displacement": "absent",
                    "acceptance": "absent",
                    "follow_through": "absent"})
        return out
    atr = _atr_last(frame)
    if atr is None:
        out.update({"displacement": "absent",
                    "acceptance": "absent",
                    "follow_through": "absent"})
        return out
    closes = frame["close"].to_numpy(dtype=float)
    window = closes[-DISP_MAX_M15_BARS:]
    # The leg is measured from the entry level, not from a
    # sweep — the level is the question, the closes are the
    # answer.
    up_move = (float(max(window)) - entry) / atr
    down_move = (entry - float(min(window))) / atr
    if up_move >= down_move:
        move_atr, move_dir = up_move, 1
        extreme = float(max(window))
    else:
        move_atr, move_dir = down_move, -1
        extreme = float(min(window))
    out["displacement_atr"] = round(move_atr, 3)
    out["displacement_direction"] = move_dir
    if move_atr >= DISP_MIN_ATR:
        out["displacement"] = "strong"
    elif move_atr >= DISP_WEAK_MIN_ATR:
        out["displacement"] = "weak"
    else:
        out["displacement"] = "none"
        out["displacement_direction"] = 0
    # Acceptance: the last two closed bars — only meaningful
    # when there IS a displacement to accept or reject.  With
    # no displacement the closes sitting at the level are the
    # level itself, not an acceptance of either side: the read
    # is absent and casts no vote.  (A sub-threshold move must
    # never vote through the back door of its raw direction.)
    if out["displacement"] == "none":
        out["acceptance"] = "absent"
        out["acceptance_direction"] = 0
        out["acceptance_closes"] = 0
    else:
        last_two = closes[-2:]
        beyond = [c for c in last_two
                  if (c - entry) * move_dir > 0] if move_dir else []
        out["acceptance_closes"] = len(beyond)
        out["acceptance_direction"] = move_dir
        out["acceptance"] = ("accepted" if len(beyond) == 2
                             else "rejected")
    # Follow-through: retracement of the leg from its extreme
    # to the last close, as a fraction of the leg.
    leg = abs(extreme - entry)
    if leg > 0:
        retraced = abs(extreme - float(closes[-1])) / leg
        out["retracement_frac"] = round(retraced, 3)
        out["follow_through_direction"] = move_dir
        out["follow_through"] = ("held" if retraced <= 0.5
                                 else "failed")
    return out


def measure_decision_structure(*, candidate: dict,
                               evidence: dict | None,
                               context: dict | None,
                               frames: dict | None = None,
                               entry_tf: str | None = None,
                               decision: dict | None = None
                               ) -> DecisionStructure:
    """Measure the market's own evidence at decision time.
    ``frames`` (TF -> bar frame) is the live path's bars;
    without them the reaction reads are absent and the
    verdict reads the funnel's measured structure only —
    never invented.  ``entry_tf`` is the playbook's own
    entry clock; the reaction is read on that frame.
    ``decision`` is the gate stack's floor result — its
    Asymmetry floor/total ride the measurement as evidence
    (the floor the funnel itself set), never as a verdict
    input."""
    candidate = candidate or {}
    evidence = evidence or {}
    context = context or {}
    observed = evidence.get("observed") or {}
    inferred = evidence.get("inferred") or {}
    direction = int(candidate.get("direction") or 0)
    entry = _f(candidate.get("entry"))
    stop = _f(candidate.get("stop"))

    regime = _parse_regime_label(context.get("regime"))
    phase = _structure_class(regime, inferred)
    situation = evidence.get("situation") or {}
    if not isinstance(situation, dict):
        situation = {}
    sit_phase = (situation.get("intended") or {}).get(
        "structure_phase")
    if phase == "unknown" and isinstance(sit_phase, str) \
            and sit_phase:
        phase = _structure_class({"structure": sit_phase}, {})

    # The candidate's own geometry is primary — including a
    # zero distance, which IS the invalid-geometry fact.
    if entry is not None and stop is not None:
        stop_dist = abs(entry - stop)
    else:
        stop_dist = _f(observed.get("stop_dist"))
    targets = candidate.get("targets") or []
    target_level = None
    for raw in targets:
        t = _f(raw)
        if t is not None and stop_dist:
            target_level = t
            break
    if target_level is None:
        target_level = _f(inferred.get("room_r"))

    score_total = _f((evidence.get("inferred") or {}).get(
        "score_total"))
    if score_total is None:
        score_total = _f(candidate.get("asymmetry_score"))
    floor = _f((evidence.get("inferred") or {}).get("floor"))
    if floor is None:
        # The funnel's floor lives on the gate stack's floor
        # result — the floor the funnel itself set.
        floor_result = (decision or {}).get("floor_result") or {}
        floor = _f(floor_result.get("floor"))
        if score_total is None:
            score_total = _f(floor_result.get("total"))
    bias_d1 = observed.get("bias_d1")
    bias_d1 = int(bias_d1) if bias_d1 in (1, -1) else None
    bias_h4 = observed.get("bias_h4")
    bias_h4 = int(bias_h4) if bias_h4 in (1, -1) else None

    structure = DecisionStructure(
        phase=phase, bias_d1=bias_d1, bias_h4=bias_h4,
        asymmetry_total=score_total, asymmetry_floor=floor,
        asymmetry_flipped=bool(inferred.get("score_flipped")),
        room_r=_f(inferred.get("room_r")),
        session=observed.get("session_bucket")
        or context.get("session"),
        adr_pct=_f(context.get("adr_pct")),
        situation=(situation if situation else None),
        entry_level=entry, stop_level=stop,
        target_level=target_level)

    # Reaction reads from the entry-clock bars (live path).
    entry_frame = None
    chart_tf = None
    if frames:
        tf = entry_tf or candidate.get("intended_tf") or "H1"
        entry_frame = frames.get(tf)
        if entry_frame is not None:
            chart_tf = tf
        else:
            entry_frame = frames.get("H1")
            if entry_frame is not None:
                chart_tf = "H1"
    # The chart the record carries: the recent entry-clock
    # candles around the level, most recent last — the bars
    # themselves, not just the computed numbers.  No frame =
    # no chart, never invented.
    structure.chart_tf = chart_tf
    structure.bars = _chart_bars(entry_frame)
    if entry is not None and stop is not None and stop_dist:
        reaction = _measure_reaction(
            entry_frame, entry=entry, stop=stop,
            direction=direction)
        structure.displacement = reaction["displacement"]
        structure.displacement_direction = reaction[
            "displacement_direction"]
        structure.displacement_atr = reaction["displacement_atr"]
        structure.acceptance = reaction["acceptance"]
        structure.acceptance_direction = reaction[
            "acceptance_direction"]
        structure.acceptance_closes = reaction[
            "acceptance_closes"]
        structure.follow_through = reaction["follow_through"]
        structure.follow_through_direction = reaction[
            "follow_through_direction"]
        structure.retracement_frac = reaction["retracement_frac"]
    else:
        structure.displacement = "absent"
        structure.acceptance = "absent"
        structure.follow_through = "absent"

    # The reads: one entry per measured read.  A read votes
    # only when it was measured; absent reads cast nothing.
    reads = []
    if structure.displacement in ("weak", "strong"):
        reads.append(("displacement",
                      structure.displacement_direction,
                      f"{structure.displacement} "
                      f"{structure.displacement_atr} ATR"))
    if structure.acceptance == "accepted":
        reads.append(("acceptance",
                      structure.acceptance_direction,
                      f"{structure.acceptance_closes}/2 closes "
                      "beyond the level"))
    elif structure.acceptance == "rejected":
        # A rejection votes against the displacement's
        # direction: the level held, the move failed.
        if structure.displacement_direction:
            reads.append(("acceptance",
                          -structure.displacement_direction,
                          "closes rejected back inside the level"))
    if structure.follow_through == "held":
        reads.append(("follow_through",
                      structure.follow_through_direction,
                      "leg held (retraced "
                      f"{structure.retracement_frac})"))
    elif structure.follow_through == "failed":
        if structure.follow_through_direction:
            reads.append(("follow_through",
                          -structure.follow_through_direction,
                          "leg retraced "
                          f"{structure.retracement_frac}"))
    # HTF bias: one read for the pair (D1 + H4 together —
    # correlated evidence counts once).
    if bias_d1 is not None or bias_h4 is not None:
        votes = [b for b in (bias_d1, bias_h4)
                 if b is not None]
        if votes and all(v == 1 for v in votes):
            reads.append(("htf_bias", 1, "D1+H4 with"))
        elif votes and all(v == -1 for v in votes):
            reads.append(("htf_bias", -1, "D1+H4 against"))
        # Split bias = no read (the timeframes disagree).
    # Asymmetry: the score's edge over its floor, one read.
    # The edge favours the candidate's direction when the
    # score clears its floor, and against it when it does not.
    if score_total is not None and floor is not None:
        if score_total > floor:
            reads.append(("asymmetry", direction or 0,
                          f"score {score_total} over floor {floor}"))
        elif score_total < floor:
            reads.append(("asymmetry", -(direction or 0),
                          f"score {score_total} under floor {floor}"))
    structure.reads = reads
    return structure


# ------------------------------------------------------------ the verdict
def verdict_from_structure(structure: DecisionStructure,
                           candidate: dict) -> dict:
    """The verdict, from his definition (owner, 2026-10-23):
    market-state estimation → competing scenarios → evidence
    evaluation → decision/abstention.  No AI judges a trade;
    the code is the judgment, always.

    1. Market-state estimation is already done — the
       structure's reads are the evidence.
    2. Competing scenarios: LONG and SHORT.  The evidence
       supports the scenario with more reads behind it.  A
       tie is a conflict; no reads is no evidence.  The
       support comparison IS the competition — not a
       threshold (no minimum count, no ratio bar).  The 26
       Nov case is the proof it must be support, not
       unanimity: the evidence supported LONG 3-to-2 and the
       verdict followed LONG.
    3. Evidence evaluation: the decision follows the
       supported scenario — but only when the candle sequence
       has answered at the level (a reaction read for the
       candidate): a level is a question, the candle sequence
       is the answer.  Confirmations are evidence, never
       vetoes.  The payoff is the structure's own geometry,
       recorded, never a verdict condition — "no target" is
       not a trade-existence fact, so it cannot abstain a
       trade.
    4. Decision/abstention: take when the supported scenario
       is the candidate's and the reaction has answered;
       abstain when the evidence supports neither, conflicts,
       fights the candidate, or the reaction has not
       answered.  Hard stops ONLY for trade-existence facts
       (no valid stop geometry here — the other hard stops
       live in the runner's safeguard stack, outside this
       module).

    ``confidence`` is the share of measured reads behind the
    supported scenario — a report for the audit's
    calibration, never a verdict input.  It decides nothing.
    """
    candidate = candidate or {}
    direction = int(candidate.get("direction") or 0)
    entry = _f(candidate.get("entry"))
    stop = _f(candidate.get("stop"))
    if entry is not None and stop is not None:
        stop_dist = abs(entry - stop)
    else:
        stop_dist = None

    market_state = {
        "structure": structure.phase,
        "bias_d1": structure.bias_d1,
        "bias_h4": structure.bias_h4,
        "displacement": structure.displacement,
        "displacement_direction":
            structure.displacement_direction,
        "displacement_atr": structure.displacement_atr,
        "acceptance": structure.acceptance,
        "follow_through": structure.follow_through,
        "retracement_frac": structure.retracement_frac,
        "session": structure.session,
        "adr_pct": structure.adr_pct,
        "asymmetry_total": structure.asymmetry_total,
        "asymmetry_floor": structure.asymmetry_floor,
        "asymmetry_flipped": structure.asymmetry_flipped,
        "room_r": structure.room_r,
    }
    if structure.situation:
        # Recorded in the picture, never a vote: the situation
        # rides the record exactly as the evidence carried it.
        market_state["situation"] = structure.situation

    # 1. Trade-existence: no direction, no entry, no stop,
    # or a zero stop distance — there is no trade to judge.
    if direction not in (1, -1) or entry is None \
            or stop is None or not stop_dist:
        return {
            "mode": "live", "rung": "code",
            "decision": "abstain", "direction": 0,
            "expected_r": None, "confidence": 0.0,
            "market_state": market_state,
            "structure": structure.to_dict(),
            "reason": "invalid geometry: direction/entry/stop "
                      "missing or stop distance is zero — there "
                      "is no trade to judge",
            "reason_category": CATEGORY_INVALID_GEOMETRY,
        }

    # 2. The scenarios compete on the evidence: each
    # scenario's support is the reads favouring it.
    long_support = len(structure.reads_for(1))
    short_support = len(structure.reads_for(-1))
    present = len(structure.reads)
    if long_support > short_support:
        supported = 1
    elif short_support > long_support:
        supported = -1
    else:
        supported = 0
    # The report, never an input: the share of measured
    # reads behind the supported scenario.
    agreeing = long_support if supported == 1 \
        else short_support if supported == -1 else 0
    confidence = round(agreeing / present, 4) if present else 0.0

    # The payoff is the structure's own geometry: first
    # target over the structural stop.  No target -> the
    # journey room; neither -> absent, never invented.
    # Recorded as evidence only — it never gates the verdict.
    expected_r = None
    if structure.target_level is not None and stop_dist:
        expected_r = round(
            abs(structure.target_level - entry) / stop_dist, 4)
    elif structure.room_r is not None:
        expected_r = round(structure.room_r, 4)

    reaction_for = [r for r in structure.reads
                    if r[0] in ("displacement", "acceptance",
                                "follow_through")
                    and r[1] == direction]
    side = "LONG" if direction == 1 else "SHORT"
    supported_side = ("LONG" if supported == 1 else
                      "SHORT" if supported == -1 else "none")
    reads_text = "; ".join(
        f"{name} {('LONG' if d == 1 else 'SHORT' if d == -1 else '—')} "
        f"({note})" for name, d, note in structure.reads) \
        or "no reads measured"

    def _block(decision, category, reason):
        return {
            "mode": "live", "rung": "code",
            "decision": decision, "direction": supported,
            "expected_r": expected_r,
            "confidence": confidence,
            "market_state": market_state,
            "structure": structure.to_dict(),
            "reason": reason,
            "reason_category": category,
        }

    # The evidence supports neither side — there is no
    # scenario to follow.
    if supported == 0 and present == 0:
        return _block(
            "abstain", CATEGORY_REGIME,
            f"no measured direction ({reads_text}) — no reads "
            f"measured; the evidence supports neither side and "
            f"there is no trade to take")
    # The evidence supports both sides equally — the
    # scenarios conflict, and a conflict is a wait.
    if supported == 0:
        return _block(
            "abstain", CATEGORY_REGIME,
            f"no measured direction ({reads_text}) — the reads "
            f"split evenly; the evidence conflicts and supports "
            f"both scenarios")
    # The supported scenario is the other side — the
    # candidate fights the market.
    if supported != direction:
        return _block(
            "abstain", CATEGORY_DIRECTION,
            f"structure reads {supported_side} "
            f"({reads_text}) — the candidate is {side} "
            f"and fights the measured direction")
    # The reaction has not answered at the level — the
    # candle sequence has not shown the move.  Context
    # alone (bias, score) never takes a trade.
    if not reaction_for:
        return _block(
            "abstain", CATEGORY_QUALITY,
            f"structure reads {side} but no reaction "
            f"read confirms it ({reads_text}) — "
            f"displacement / acceptance / follow-"
            f"through have not shown the move")
    # The supported scenario is the candidate's and the
    # reaction has answered: take.
    payoff_text = (f"payoff {expected_r}R to the first "
                   f"structural target") if expected_r is not None \
        else ("no first target measured — the trade rides "
              "its stop and the room")
    return _block(
        "take", CATEGORY_QUALITY,
        f"structure reads {side}: {reads_text} — the "
        f"candidate agrees with the supported scenario on "
        f"{len(reaction_for)} reaction read(s); {payoff_text}")


def evaluate_judgment(*, candidate: dict, decision: dict | None,
                      evidence: dict | None, context: dict | None,
                      frames: dict | None = None,
                      entry_tf: str | None = None) -> dict:
    """One judgment for one evaluated candidate — the code
    verdict, the judgment, always.  Pure: same measured
    structure in, same verdict out.  The returned block is
    journaled on the ``judgment`` record, with the runner
    overstamping ``mode`` with the effective authority mode
    (see the module docstring).  ``decision`` (the gate
    stack's floor result) rides the record as evidence; it
    does not weigh the verdict — the verdict is the
    scenarios competing on the structure's reads.  ``frames``
    is the live path's bars; without them the reaction reads
    are absent and the verdict reads the funnel's measured
    structure only."""
    structure = measure_decision_structure(
        candidate=candidate, evidence=evidence,
        context=context, frames=frames, entry_tf=entry_tf,
        decision=decision)
    verdict = verdict_from_structure(structure, candidate)
    # The gate stack's floor result rides the record as
    # evidence (the Asymmetry score it measured, the floor
    # it set) — never a verdict input.
    decision = decision or {}
    floor_result = decision.get("floor_result") or {}
    if floor_result:
        verdict["floor_result"] = {
            "floor": _f(floor_result.get("floor")),
            "total": _f(floor_result.get("total"))}
    return verdict
