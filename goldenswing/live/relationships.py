"""Cross-timeframe relationships — Pass 1 of the MarketSituation
program (SPEC §4.2(a); AMENDMENTS-1 §A1).

Doctrine D2: hierarchy is ONE relationship mode, not the whole
model.  Before this module the only cross-timeframe read in the
engine was the hierarchy's gate chain — a swing sleeve's verdict
wearing the market's name.  This module computes, for every
timeframe pair the desk actually trades across, how the two
timeframes' INDEPENDENT states (live/situation.py, Pass 0)
relate to each other: aligned, correcting, in reversal attempt,
conflicting, nested, transitioning, or independent.

Timeframes are allowed to disagree.  The engine's whole point is
that D1 bullish / H4 bearish / H1 bullish / M15 bearish is a
layered market — larger trend, intermediate correction,
continuation attempt, local retracement — NOT one flattened
directional vote.  No record this module emits carries a
pair-level "direction" or "verdict": each pair carries its kind
and the evidence fields that decided it, so any reader can
re-derive the classification and disagree with it on the record.

The definitions are hypotheses (SPEC §10 risk 4): computable as
specified, versioned in every record (``RELATIONSHIP_VERSION``),
so a definition change never silently rewrites history.

The hierarchy's own read of the D1→H1 pair is computed here too
(:func:`hierarchy_reading`) and journaled ALONGSIDE the engine's
D1→H1 row — one reading among readings, with its gate authority
exactly what it was (SPEC §2.3).  It lives in this module, not
in situation.py, because the situation builder is a measurement
layer that must never consult a verdict (Pass 0 pin).

SHADOW: nothing in the decision path reads these records.
"""
from __future__ import annotations

import copy

#: Definition version stamped on every relationship record.
RELATIONSHIP_VERSION = 1

#: The ordered pairs the desk trades across (SPEC §4.2(a)).
PAIRS: tuple[tuple[str, str], ...] = (
    ("D1", "H4"), ("D1", "H1"), ("H4", "H1"), ("H4", "M15"),
    ("H1", "M15"), ("H1", "M5"), ("M15", "M5"),
)

#: Bedrock structure phases that count as "trend-phased".
TREND_PHASES = frozenset({"clean_trend", "breakout"})
#: Phases that count as range-phased for the correcting test.
RANGE_PHASES = frozenset({"range", "compression", "choppy"})
#: Phases that are themselves transition evidence.
TRANSITION_PHASES = frozenset({"unclear", "reversal"})
#: Structure-event kinds that count as a reversal attempt when
#: printed against the slower timeframe's bias (SPEC §4.2(a):
#: msch / choch — a bos with the trend is continuation, not an
#: attempt against it).  Pass 0 states carry the multitf kinds
#: verbatim ("BOS" / "MSCH" / "CHoCH").
REVERSAL_EVENT_KINDS = frozenset({"MSCH", "CHOCH"})

KIND_ALIGNED = "aligned"
KIND_CORRECTING = "correcting"
KIND_REVERSAL_ATTEMPT = "reversal_attempt"
KIND_CONFLICTING = "conflicting"
KIND_NESTED = "nested"
KIND_TRANSITIONING = "transitioning"
KIND_INDEPENDENT = "independent"


# ------------------------------------------------------------------
# field helpers (states are plain dicts from live/situation.py)
# ------------------------------------------------------------------
def _bias(state: dict | None) -> int | None:
    if not state:
        return None
    bias = state.get("bias")
    return bias if bias in (-1, 0, 1) else None


def _phase(state: dict | None) -> str | None:
    if not state:
        return None
    return state.get("structure_phase")


def _swing_price(state: dict | None, key: str) -> float | None:
    if not state:
        return None
    point = (state.get("swings") or {}).get(key)
    if not point:
        return None
    price = point.get("price")
    try:
        return float(price) if price is not None else None
    except (TypeError, ValueError):
        return None


def _defended_swing(h_state: dict | None) -> float | None:
    """The slower TF's defended swing: the confirmed swing whose
    break would flip ``evaluate_bias(H)`` — the last confirmed
    swing LOW under an up-bias, the last confirmed swing HIGH
    under a down-bias."""
    bias = _bias(h_state)
    if bias == 1:
        return _swing_price(h_state, "confirmed_low")
    if bias == -1:
        return _swing_price(h_state, "confirmed_high")
    return None


def _event(state: dict | None) -> dict | None:
    if not state:
        return None
    return state.get("last_structure_event") or None


def _event_kind(event: dict | None) -> str | None:
    if not event:
        return None
    kind = event.get("kind")
    return str(kind).upper() if kind else None


def _structure_broken(h_state: dict | None, l_state: dict | None,
                      price: float | None) -> bool:
    """Has the counter-move broken H's defended swing (SPEC
    §4.2(a) conflicting/correcting boundary)?  Two independent
    signals, either sufficient:

    - H's own last closed bar printed a structure event AGAINST
      H's bias of MSCH/CHoCH kind — H's structure broke on H's
      own clock;
    - the current price trades beyond the defended swing level.
    """
    bias = _bias(h_state)
    if bias in (None, 0):
        return False
    h_event = _event(h_state)
    if h_event and h_event.get("against_bias") \
            and _event_kind(h_event) in REVERSAL_EVENT_KINDS:
        return True
    defended = _defended_swing(h_state)
    if defended is not None and price is not None:
        if bias == 1 and price < defended:
            return True
        if bias == -1 and price > defended:
            return True
    return False


def _transition_evidence(state: dict | None) -> bool:
    """Bedrock transition equivalents (SPEC §4.2(a)): the phase
    itself is transitional, or the classifier's direction word
    disagrees with the measured swing bias (the short/long
    window disagreement bedrock reports as flat/unclear)."""
    if not state:
        return False
    if _phase(state) in TRANSITION_PHASES:
        return True
    bias = _bias(state)
    word = state.get("direction_word")
    if bias in (-1, 1) and word in ("up", "down", "flat"):
        word_sign = {"up": 1, "down": -1, "flat": 0}[word]
        if word_sign != bias:
            return True
    return False


def _nested(h_state: dict | None, l_state: dict | None) -> bool:
    """The faster TF's confirmed range sits inside the slower
    TF's confirmed dealing range (both swings present)."""
    h_hi = _swing_price(h_state, "confirmed_high")
    h_lo = _swing_price(h_state, "confirmed_low")
    l_hi = _swing_price(l_state, "confirmed_high")
    l_lo = _swing_price(l_state, "confirmed_low")
    if None in (h_hi, h_lo, l_hi, l_lo):
        return False
    return l_lo > h_lo and l_hi < h_hi


# ------------------------------------------------------------------
# the pair classifier (SPEC §4.2(a))
# ------------------------------------------------------------------
def classify_pair(htf: str, ltf: str, h_state: dict | None,
                  l_state: dict | None, *,
                  price: float | None = None) -> dict:
    """One ordered pair's relationship, from the two TF states.

    Pure function.  The kind is decided by the FIRST matching
    definition in the documented precedence — directional
    breaks first (conflicting, reversal attempt, correcting,
    aligned), then instability (transitioning), then geometry
    (nested), then the honest fallback (independent).  When a
    bias is unmeasured the kind is None with the reason named:
    absent stays absent, and no pair is ever flattened into a
    direction this function did not measure.
    """
    bias_h = _bias(h_state)
    bias_l = _bias(l_state)
    phase_h = _phase(h_state)
    phase_l = _phase(l_state)
    l_event = _event(l_state)
    defended = _defended_swing(h_state)
    broken = _structure_broken(h_state, l_state, price)
    nested = _nested(h_state, l_state)
    evidence = {
        "bias_h": bias_h, "bias_l": bias_l,
        "phase_h": phase_h, "phase_l": phase_l,
        "defended_swing": defended, "broken": bool(broken),
        "l_event_kind": _event_kind(l_event),
        "l_event_direction": (l_event.get("direction")
                              if l_event else None),
        "price": price, "nested": bool(nested),
    }
    record = {"htf": htf, "ltf": ltf, "source": "engine",
              "version": RELATIONSHIP_VERSION,
              "kind": None, "reason": "", "evidence": evidence}
    if bias_h is None or bias_l is None:
        record["reason"] = "insufficient_state"
        return record

    h_trend = phase_h in TREND_PHASES
    l_trend = phase_l in TREND_PHASES
    l_event_against = (
        bias_h != 0 and l_event is not None
        and l_event.get("direction") == -bias_h
        and _event_kind(l_event) in REVERSAL_EVENT_KINDS)

    if bias_h != 0 and bias_l == -bias_h and h_trend and l_trend \
            and broken:
        record["kind"] = KIND_CONFLICTING
        record["reason"] = (
            "opposing trend phases and the faster move has "
            "broken the slower defended swing — the slower "
            "structure itself is contested")
    elif l_event_against:
        record["kind"] = KIND_REVERSAL_ATTEMPT
        record["reason"] = (
            "faster TF printed a structure event against the "
            "slower bias while the slower bias is unchanged")
    elif bias_h != 0 and h_trend and not broken and (
            bias_l == -bias_h or phase_l in RANGE_PHASES):
        record["kind"] = KIND_CORRECTING
        record["reason"] = (
            "counter-move inside the slower structure — the "
            "defended swing holds")
    elif bias_h != 0 and bias_l == bias_h and h_trend and l_trend:
        record["kind"] = KIND_ALIGNED
        record["reason"] = "same measured bias, both trend-phased"
    elif _transition_evidence(h_state) or _transition_evidence(
            l_state):
        record["kind"] = KIND_TRANSITIONING
        record["reason"] = (
            "a side's phase or direction read is transitional — "
            "the pair's relationship is itself changing")
    elif nested:
        record["kind"] = KIND_NESTED
        record["reason"] = (
            "faster confirmed range inside the slower dealing "
            "range")
    else:
        record["kind"] = KIND_INDEPENDENT
        record["reason"] = (
            "no shared directional or containment relationship "
            "measured")
    return record


def compute_relationships(per_tf: dict, *,
                          price: float | None = None
                          ) -> list[dict]:
    """Every SPEC §4.2(a) pair over a ``per_tf`` state mapping
    (values may be None for unbuilt TFs — those pairs classify
    as insufficient_state)."""
    per_tf = per_tf or {}
    return [classify_pair(htf, ltf, per_tf.get(htf),
                          per_tf.get(ltf), price=price)
            for htf, ltf in PAIRS]


def kinds_map(relationships: list[dict] | None) -> dict:
    """``{(htf, ltf, source): kind}`` — the change-detection
    view the situation keeper diffs between refreshes."""
    out: dict = {}
    for rel in relationships or []:
        out[(rel.get("htf"), rel.get("ltf"),
             rel.get("source"))] = rel.get("kind")
    return out


# ------------------------------------------------------------------
# the hierarchy reading (SPEC §2.3 / §4.2(a) comparison row)
# ------------------------------------------------------------------
def hierarchy_reading(stack, *, asof=None, qm_touch=None,
                      swing_n: int = 3) -> dict | None:
    """``evaluate_hierarchy``'s verdict as ONE relationship
    reading of the D1→H1 pair, journaled alongside the engine's
    own computation so the two can be compared on the demo.

    The mapping is over the verdict's measured facts, never its
    gate outcome: H4 against D1 is the hierarchy's own
    "pullback" language -> ``correcting``; H4 with D1 (or a
    fired direction with D1) -> ``aligned``; flat D1 -> no
    directional reading (kind None, the regime word carried as
    the reason).  The hierarchy has no broken-structure concept
    — its model calls every H4 counter-move a pullback — so it
    never maps to ``conflicting``: that asymmetry is exactly
    what the comparison exists to surface.

    Returns None when the verdict cannot be computed at all —
    the reading is absent, never fabricated.  Never raises.
    """
    try:
        from .hierarchy import evaluate_hierarchy
        verdict = evaluate_hierarchy(
            stack, asof=asof, qm_touch=qm_touch, swing_n=swing_n)
    except Exception:  # noqa: BLE001 — a reading, never a raise
        return None
    if not isinstance(verdict, dict):
        return None
    bias_d1 = verdict.get("bias_d1") or 0
    bias_h4 = verdict.get("bias_h4") or 0
    direction = verdict.get("direction") or 0
    regime = verdict.get("regime")
    if bias_d1 == 0:
        kind = None
        reason = (f"hierarchy regime '{regime}' — flat D1 bias "
                  f"carries no directional reading")
    elif bias_h4 == -bias_d1:
        kind = KIND_CORRECTING
        reason = "hierarchy leg 2: H4 pullback against D1"
    elif bias_h4 == bias_d1 or direction == bias_d1:
        kind = KIND_ALIGNED
        reason = "hierarchy direction stays with D1"
    elif direction == -bias_d1:
        kind = KIND_CORRECTING
        reason = "hierarchy fired against the D1 bias"
    else:
        kind = None
        reason = "hierarchy produced no directional read"
    return {
        "htf": "D1", "ltf": "H1", "source": "hierarchy",
        "version": RELATIONSHIP_VERSION, "kind": kind,
        "reason": reason,
        "evidence": {
            "bias_d1": bias_d1, "bias_h4": bias_h4,
            "direction": direction, "regime": regime,
            "allowed": bool(verdict.get("allowed")),
            "blocked_leg": verdict.get("blocked_leg"),
            "h4_context": verdict.get("h4_context"),
        },
    }


def with_hierarchy(pairs: list[dict],
                   reading: dict | None) -> list[dict]:
    """The full relationship list for a snapshot: the engine
    pairs plus the hierarchy row when a reading exists (any
    retained hierarchy row in ``pairs`` is replaced, so the
    list carries exactly one)."""
    out = [copy.deepcopy(r) for r in pairs or []
           if r.get("source") != "hierarchy"]
    if reading is not None:
        out.append(copy.deepcopy(reading))
    return out
