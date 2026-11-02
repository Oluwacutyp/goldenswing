"""Judgment core (owner rulings 2026-10-23; build ordered
2026-11-02): market-state estimation -> competing scenarios ->
evidence evaluation -> decision / abstention.

SHADOW-FIRST (the ruling): this module is a pure function over
the funnel evaluation record the runner already journals.  It
decides NOTHING in v1 — the runner appends its verdict as a
``judgment_shadow`` record beside the gate stack's own record,
so the two decision models can be compared on identical inputs
against real outcomes before any promotion.  First promotion
(M5 route only) needs a separate owner approval and is NOT
implemented here.

Hard risk invariants stay OUTSIDE this model (the ruling): the
friction gate, RiskGuards/RiskPool and the drawdown halt keep
their own authority on the order path; a judgment verdict never
steers, sizes, gates or vetoes an order.  The LLM is out of the
decision path (explanation drafting only, elsewhere).

The directive's neural ladder is baseline-first: THIS is the
baseline — a deterministic, fully inspectable estimator whose
every input is a value the funnel record already carries.  No
probability here is learned or fitted; the priors and response
coefficients below are stated constants, and the shadow journal
is the dataset a later fitted rung must beat on chronological
walk-forward with a locked holdout.

Evidence honesty rules (mirroring the funnel record's own
doctrine):
- Correlated evidence counts ONCE per correlation group (the
  record's ``evidence_groups``): ten prints of one detector are
  one vote, not ten.
- Counter-bias and counter-thesis evidence is WEIGHED, never a
  veto (M5 counter-D1 = penalty/evidence, never a hard block).
- Absent inputs stay absent: no targets -> the journey room is
  the continuation payoff; no score -> no score edge.  Nothing
  is invented to fill a gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .funnel import (
    CATEGORY_DIRECTION,
    CATEGORY_INVALID_GEOMETRY,
    CATEGORY_QUALITY,
    CATEGORY_REGIME,
)

#: Scenario priors (stated, not fitted): on a fired candidate the
#: desk's base rates before this bar's evidence is weighed.
PRIOR_CONTINUATION = 0.42
PRIOR_REVERSAL = 0.38
PRIOR_CHOP = 0.20

#: Response of scenario odds to weighed support (log-odds scale).
RESPONSE = 0.45
#: How hard structural opposition (against the D1 bias, a flipped
#: score, a journey refusal) funds the reversal scenario.
OPPOSITION_RESPONSE = 1.2
#: How hard regime chop affinity funds the chop scenario.
CHOP_RESPONSE = 1.1

#: One correlation group's vote weight.  Structure and the score
#: are the desk's heaviest evidence; everything else votes 1.0.
GROUP_WEIGHTS: dict[str, float] = {
    "htf_structure": 1.5,
    "asymmetry_score": 1.5,
    "journey_geometry": 1.25,
    "zone_context": 1.0,
    "funding_events": 1.0,
}
DEFAULT_GROUP_WEIGHT = 1.0
#: The asymmetry score's own vote: its margin over the floor,
#: normalised (50 points of margin = a full vote of 1.5).
SCORE_WEIGHT = 1.5
SCORE_MARGIN_NORM = 50.0

MAX_TARGET_R = 3.0     # continuation payoff cap (first target)
MIN_TARGET_R = 0.25    # ... and floor (a target inside the noise
                       # is not a continuation payoff)


@dataclass(frozen=True)
class JudgmentConfig:
    """Abstention thresholds (the desk's stated bar, in R)."""
    min_expected_r: float = 0.05
    min_confidence: float = 0.15
    chop_cost_r: float = 0.10   # friction/time cost of a chop exit


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


def _evidence_balance(evidence: dict) -> dict:
    """One vote per correlation group (see module docstring).

    A group's stance is the sign of (its supporting texts -
    its contradictory texts) as recorded in the funnel record;
    a group split evenly abstains.  Returns the weighed support
    on each side plus the group names behind each."""
    evidence = evidence or {}
    groups = evidence.get("evidence_groups") or {}
    supporting = set(evidence.get("supporting") or [])
    contradictory = set(evidence.get("contradictory") or [])
    support = 0.0
    against = 0.0
    groups_for: list[str] = []
    groups_against: list[str] = []
    for name, texts in groups.items():
        texts = list(texts or [])
        s = sum(1 for t in texts if t in supporting)
        c = sum(1 for t in texts if t in contradictory)
        if s == c:
            continue
        weight = GROUP_WEIGHTS.get(name, DEFAULT_GROUP_WEIGHT)
        if s > c:
            support += weight
            groups_for.append(name)
        else:
            against += weight
            groups_against.append(name)
    return {"support": round(support, 3),
            "against": round(against, 3),
            "groups_for": groups_for,
            "groups_against": groups_against}


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


# ------------------------------------------------------------ the core
def evaluate_judgment(*, candidate: dict, decision: dict | None,
                      evidence: dict | None, context: dict | None,
                      config: JudgmentConfig | None = None) -> dict:
    """One shadow judgment for one evaluated candidate.

    Pure: same record in, same verdict out.  The returned block
    is journaled verbatim on the ``judgment_shadow`` record."""
    cfg = config or JudgmentConfig()
    candidate = candidate or {}
    decision = decision or {}
    evidence = evidence or {}
    context = context or {}
    observed = evidence.get("observed") or {}
    inferred = evidence.get("inferred") or {}

    direction = int(candidate.get("direction") or 0)
    entry = _f(candidate.get("entry"))
    stop = _f(candidate.get("stop"))
    # The candidate's own geometry is primary — including a
    # zero distance, which IS the invalid-geometry fact.  The
    # record's observed stop_dist only fills a gap when the
    # candidate carries no geometry at all (on the live path
    # they agree — _emit_sig computes both from one signal).
    if entry is not None and stop is not None:
        stop_dist = abs(entry - stop)
    else:
        stop_dist = _f(observed.get("stop_dist"))

    # ---- market-state estimation -------------------------------
    regime = _parse_regime_label(context.get("regime"))
    structure = _structure_class(regime, inferred)
    bias_align, d1_opposed = _bias_alignment(observed, direction)
    adr_pct = _f(context.get("adr_pct"))
    adr_state = str(context.get("adr_state") or "")
    chop_affinity = 0.0
    if structure in ("range", "compression", "choppy"):
        chop_affinity += 1.0
    elif structure == "trend":
        chop_affinity -= 0.5
    elif structure == "reversal":
        chop_affinity += 0.25
    if adr_state == "exhausted" or (adr_pct is not None
                                    and adr_pct >= 90.0):
        chop_affinity += 0.5
    if regime.get("volatility") == "spike":
        chop_affinity += 0.25
    market_state = {
        "structure": structure,
        "regime_direction": regime.get("direction"),
        "volatility": regime.get("volatility"),
        "phase": regime.get("phase"),
        "posture": context.get("posture")
                   or inferred.get("posture"),
        "day_type": context.get("day_type"),
        "adr_pct": adr_pct,
        "adr_state": context.get("adr_state"),
        "session": observed.get("session_bucket")
                   or context.get("session"),
        "smt": context.get("smt"),
        "bias_alignment": bias_align,
        "journey_state": inferred.get("journey_state"),
        "room_r": _f(inferred.get("room_r")),
    }

    # ---- geometry (a hard fact, stated before any weighing) ----
    if direction not in (1, -1) or entry is None or stop is None \
            or not stop_dist:
        return {
            "mode": "shadow", "decision": "abstain",
            "expected_r": None, "confidence": 0.0,
            "scenarios": [],
            "market_state": market_state,
            "evidence_balance": _evidence_balance(evidence),
            "reason": "invalid geometry: direction/entry/stop "
                      "missing or stop distance is zero — there "
                      "is no trade to judge",
            "reason_category": CATEGORY_INVALID_GEOMETRY,
            "thresholds": {"min_expected_r": cfg.min_expected_r,
                           "min_confidence": cfg.min_confidence},
        }

    # ---- evidence evaluation ------------------------------------
    balance = _evidence_balance(evidence)
    floor_result = decision.get("floor_result") or {}
    score_total = _f(floor_result.get("total"))
    if score_total is None:
        score_total = _f(inferred.get("score_total"))
    if score_total is None:
        score_total = _f(candidate.get("asymmetry_score"))
    floor = _f(floor_result.get("floor"))
    score_edge = None
    if score_total is not None:
        base = floor if floor is not None else 50.0
        score_edge = max(-1.0, min(
            1.0, (score_total - base) / SCORE_MARGIN_NORM))
    score_support = SCORE_WEIGHT * score_edge \
        if score_edge is not None else 0.0

    flipped = bool(inferred.get("score_flipped")
                   or decision.get("direction_flip"))
    journey_opposed = "journey_geometry" in balance["groups_against"]
    opposition = (1.0 if d1_opposed else 0.0) \
        + (1.0 if flipped else 0.0) \
        + (0.5 if journey_opposed else 0.0)

    support_score = (balance["support"] - balance["against"]) \
        + score_support + 0.5 * bias_align

    # ---- competing scenarios ------------------------------------
    targets = candidate.get("targets") or []
    target_r = None
    for raw in targets:
        t = _f(raw)
        if t is not None and stop_dist:
            target_r = abs(t - entry) / stop_dist
            break
    if target_r is None:
        target_r = _f(inferred.get("room_r"))
    if target_r is None:
        target_r = 1.0
    target_r = max(MIN_TARGET_R, min(MAX_TARGET_R, target_r))

    w_cont = PRIOR_CONTINUATION * math.exp(
        RESPONSE * support_score)
    w_rev = PRIOR_REVERSAL * math.exp(
        RESPONSE * (-support_score)
        + OPPOSITION_RESPONSE * opposition / 2.0)
    w_chop = PRIOR_CHOP * math.exp(CHOP_RESPONSE * chop_affinity)
    total_w = w_cont + w_rev + w_chop
    p_cont = w_cont / total_w
    p_rev = w_rev / total_w
    p_chop = w_chop / total_w
    scenarios = [
        {"name": "continuation", "p": round(p_cont, 4),
         "payoff_r": round(target_r, 3)},
        {"name": "reversal", "p": round(p_rev, 4),
         "payoff_r": -1.0},
        {"name": "chop", "p": round(p_chop, 4),
         "payoff_r": -cfg.chop_cost_r},
    ]
    expected_r = p_cont * target_r - p_rev * 1.0 \
        - p_chop * cfg.chop_cost_r

    mass = balance["support"] + balance["against"] \
        + abs(score_support)
    confidence = (mass / (mass + 2.0)) \
        * (0.3 + 0.7 * abs(p_cont - p_rev))

    # ---- decision / abstention ----------------------------------
    if expected_r >= cfg.min_expected_r and p_cont >= p_rev \
            and confidence >= cfg.min_confidence:
        verdict = "take"
        reason = (f"continuation leads (p={p_cont:.2f} vs "
                  f"reversal {p_rev:.2f}); expected "
                  f"{expected_r:+.2f}R at {target_r:.2f}R to the "
                  f"first target on {balance['support']:.1f} vs "
                  f"{balance['against']:.1f} weighed evidence")
        category = CATEGORY_QUALITY
    else:
        verdict = "abstain"
        if p_rev > p_cont and p_rev >= p_chop:
            category = CATEGORY_DIRECTION if opposition >= 1.0 \
                else CATEGORY_QUALITY
            reason = (f"reversal scenario leads (p={p_rev:.2f} "
                      f"vs continuation {p_cont:.2f}) — the "
                      f"weighed evidence funds the other side or "
                      f"the invalidation first")
        elif p_chop > p_cont and p_chop > p_rev:
            category = CATEGORY_REGIME
            reason = (f"chop scenario leads (p={p_chop:.2f}) in "
                      f"a {structure} state — neither target nor "
                      f"stop is the likely first outcome")
        elif confidence < cfg.min_confidence:
            category = CATEGORY_QUALITY
            reason = (f"evidence too thin to judge: confidence "
                      f"{confidence:.2f} below the "
                      f"{cfg.min_confidence:.2f} bar on "
                      f"{mass:.1f} weighed evidence mass")
        else:
            category = CATEGORY_QUALITY
            reason = (f"expected value {expected_r:+.2f}R below "
                      f"the {cfg.min_expected_r:+.2f}R bar "
                      f"(continuation {p_cont:.2f} / reversal "
                      f"{p_rev:.2f} / chop {p_chop:.2f})")

    return {
        "mode": "shadow",
        "decision": verdict,
        "expected_r": round(expected_r, 4),
        "confidence": round(confidence, 4),
        "scenarios": scenarios,
        "market_state": market_state,
        "evidence_balance": balance,
        "score": {"total": score_total, "floor": floor,
                  "edge": (round(score_edge, 4)
                           if score_edge is not None else None)},
        "support_score": round(support_score, 4),
        "opposition": opposition,
        "reason": reason,
        "reason_category": category,
        "thresholds": {"min_expected_r": cfg.min_expected_r,
                       "min_confidence": cfg.min_confidence},
    }
