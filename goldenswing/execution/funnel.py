"""Decision-funnel records: every evaluation leaves a trace.

Before this module, a playbook that evaluated and found nothing —
or was never called at all, or was called on the WRONG timeframe's
bars — left no candidate-level record.  The journal could say what
happened to the five signals that reached a decision in eight days;
it could not say whether the other playbooks evaluated, starved,
or were miswired.  (The qm_m5 defect — an M5 playbook fed H1 bars
for its entire live life — was invisible precisely because of
this.)

One record per playbook per evaluation event, INCLUDING no-signal
outcomes, written to the journal as ``type == "evaluation"``:

- identity: run_id, commit, config_fingerprint, playbook;
- clock/data: intended_tf (the playbook's native clock),
  actual_input_tf (the frame actually passed — the invariant
  ``intended == actual`` is checkable from the records alone),
  last_bar_ts, data_freshness_s, evaluation_clock;
- outcome: not_eligible / no_setup / candidate_emitted /
  exception (trigger-lifecycle records use kind="trigger_event"
  with a trigger_stage instead);
- candidate (when emitted): direction, entry, stop, targets,
  score + score_components, features;
- decision (when a candidate reached one): result, reason_code,
  floor_result, execution_eligible, order_result — plus, on the
  judgment-shaped routes, the full evidence vector that was
  weighed, so a decision reads as a judgment with its evidence,
  not a bare gate name.

Reason codes are NAMESPACED so the classes can never be conflated
in a census (the Oct 22 journal census was flipped in retelling —
3 gated / 2 below-floor became 2 / 3 — because the classes were
indistinguishable in prose):

- ``gate.*``    binary trade-existence FACTS only (the
  gates-vs-scores law): location (no valid geometry), friction
  (spread/cost blowout), solvency (margin / RiskGuard / daily
  cap), instrument (symbol/tripwire fact mismatch), data
  (stale/missing bars).
- ``floor.*``   score-floor outcomes: floor.total, floor.component.
- ``regime.*``  regime/scope READINGS (recorded with their values;
  readings, not facts — the judgment-core work can re-weight them
  as evidence later without a journal migration).
- ``direction.*`` directional evidence outcomes (bias conflict,
  score flip — the weighed structure funds the other side).
- ``path.*``    funnel mechanics: trigger not fired, kill switch,
  execute-list narrowing, exceptions, parked watches.
- ``mgmt.*``    post-entry dispositions.

Instrumentation is OBSERVATION-ONLY: emitting a record never
changes a decision (pinned by test: decisions are identical with
the funnel on and off over the same drive).
"""
from __future__ import annotations

#: Evaluation outcomes (playbook_eval records).
OUTCOMES = ("not_eligible", "no_setup", "candidate_emitted",
            "exception")

#: Trigger lifecycle stages (trigger_event records).
TRIGGER_STAGES = ("armed", "fired", "expired", "invalidated")


# ------------------------------------------------------------------
# Directive §3 reason taxonomy: the TEN decision categories every
# funnel refusal/decision also carries (as ``reason_category``),
# so a census can be stated in the directive's classes without
# losing the finer namespaced code.  Never a generic "rejected".
CATEGORY_NOT_EVALUATED = 1        # not evaluated or not eligible
CATEGORY_NO_SETUP = 2             # no setup recognized
CATEGORY_INVALID_GEOMETRY = 3     # invalid trade geometry
CATEGORY_QUALITY = 4              # contextual evidence / quality
CATEGORY_DIRECTION = 5            # directional disagreement
CATEGORY_REGIME = 6               # regime assessment
CATEGORY_FLOOR = 7                # score / component-floor failure
CATEGORY_HARD_RISK = 8            # hard risk veto
CATEGORY_EXECUTION = 9            # execution / broker failure
CATEGORY_ACCEPTED = 10            # accepted / submitted / filled

_CODE_TO_CATEGORY = {
    "gate.location": CATEGORY_INVALID_GEOMETRY,
    "gate.instrument": CATEGORY_INVALID_GEOMETRY,
    "gate.data": CATEGORY_NOT_EVALUATED,
    "gate.friction": CATEGORY_HARD_RISK,
    "risk.friction": CATEGORY_HARD_RISK,
    "gate.solvency": CATEGORY_HARD_RISK,
    "floor.total": CATEGORY_FLOOR,
    "floor.component": CATEGORY_FLOOR,
    "regime.scope": CATEGORY_REGIME,
    "regime.session": CATEGORY_REGIME,
    "regime.news": CATEGORY_REGIME,
    "direction.bias_conflict": CATEGORY_DIRECTION,
    "direction.verdict_conflict": CATEGORY_DIRECTION,
    "direction.flip": CATEGORY_DIRECTION,
    "path.kill_switch": CATEGORY_NOT_EVALUATED,
    "path.paused": CATEGORY_NOT_EVALUATED,
    "path.market_closed": CATEGORY_NOT_EVALUATED,
    "path.execute_list": CATEGORY_NOT_EVALUATED,
    "path.trigger_not_fired": CATEGORY_NOT_EVALUATED,
    "path.manipulation_blocked": CATEGORY_QUALITY,
    "path.pullback_watch": CATEGORY_QUALITY,
    "path.exception": CATEGORY_EXECUTION,
    "path.invalidated": CATEGORY_NOT_EVALUATED,
    "path.invalidated_close_rule": CATEGORY_NOT_EVALUATED,
    "path.bias_flip": CATEGORY_NOT_EVALUATED,
    "order.broker_rejected": CATEGORY_EXECUTION,
    "order.invalid_geometry": CATEGORY_INVALID_GEOMETRY,
}


def category_for_code(code: str | None,
                      result: str | None = None) -> int | None:
    """The directive §3 category (1–10) for a namespaced reason
    code.  A MAPPED code wins over the result label: a decision
    step may accept a candidate whose ORDER is then refused by a
    hard-risk veto (friction, margin) or an order-path gate — the
    category must name the stage where the trade actually died
    (the §3 exit criterion), not the earlier stage it passed.
    With no mapped code, an accepted/ordered result is category
    10; an unmapped refusal returns None (the code itself still
    carries the exact reason — a category is never invented)."""
    if code:
        mapped = _CODE_TO_CATEGORY.get(str(code))
        if mapped is not None:
            return mapped
    if result in ("accepted", "ordered", "layered"):
        return CATEGORY_ACCEPTED
    return None


def classify_reason(reason: str) -> str:
    """Map a legacy free-text refusal reason to its namespaced
    reason code.  Used where the refusal text is produced by a
    shared helper (hard lines, posture, journey); the primary call
    sites pass explicit codes, this is the fallback classifier —
    namespaced conservatively so a mis-map lands in the right
    CLASS even when the exact code is generic."""
    r = str(reason or "")
    low = r.lower()
    if r.startswith("asymmetry_floor") or "asymmetry_floor" in low \
            or low.startswith("min_score"):
        return "floor.total"
    if "cost_gate" in low or "friction" in low:
        return "gate.friction"
    if "margin_shield" in low or low.startswith("margin") \
            or "risk_guard" in low or "sizing_decline" in low:
        return "gate.solvency"
    if "session_scope" in low or "session scope" in low:
        return "regime.session"
    if "tripwire" in low:
        # The required-context tripwire class: the XAUUSD vs
        # XAUUSDm symbol mismatch is its canonical member.
        return "gate.instrument"
    if "hunting" in low or "zone_used_up" in low \
            or "zone" in low and "used_up" in low:
        return "gate.location"
    if "posture" in low or "shelved" in low or "churn" in low \
            or "clock float" in low or "stand_aside" in low:
        return "regime.scope"
    if "flip" in low:
        return "direction.flip"
    if "bias" in low or "direction" in low:
        return "direction.bias_conflict"
    if "manipulation" in low:
        return "path.manipulation_blocked"
    if "paused" in low:
        return "path.paused"
    if "market_closed" in low or "market closed" in low:
        return "path.market_closed"
    if "playbook_error" in low or "exception" in low:
        return "path.exception"
    return "gate.location"


def evaluation_records(records: list[dict]) -> list[dict]:
    """All funnel evaluation records in a journal read."""
    return [r for r in records if r.get("type") == "evaluation"]


def summarize_evaluations(records: list[dict],
                          registered_names=()) -> dict:
    """Aggregate funnel records into per-playbook funnel counts.

    Returns::

        {"by_playbook": {name: {
            "evaluations": n,                # playbook_eval records
            "by_clock": {clock: n},
            "outcomes": {outcome: n},
            "candidates": n,                 # candidate_emitted
            "refusals_by_code": {code: n},   # decision reason codes
            "reached_execution": n,          # execution_eligible
            "orders": {order_result: n}}},
         "not_called": [names with ZERO evaluations — synthesized
                        here, so "never called" is a reported fact,
                        not an absence],
         "frame_mismatches": n,   # intended_tf != actual_input_tf
         "trigger_events": {stage: n}}

    Counts reconcile by construction: for every playbook,
    evaluations == sum(outcomes.values()).
    """
    by_playbook: dict[str, dict] = {}
    not_called: list[str] = []
    frame_mismatches = 0
    trigger_events: dict[str, int] = {}
    for r in evaluation_records(records):
        if r.get("kind") == "trigger_event":
            stage = str(r.get("trigger_stage") or "?")
            trigger_events[stage] = trigger_events.get(stage, 0) + 1
            continue
        name = str(r.get("playbook") or "?")
        slot = by_playbook.setdefault(name, {
            "evaluations": 0, "by_clock": {}, "outcomes": {},
            "candidates": 0, "refusals_by_code": {},
            "refusals_by_category": {},
            "reached_execution": 0, "orders": {}})
        slot["evaluations"] += 1
        clock = str(r.get("evaluation_clock") or "?")
        slot["by_clock"][clock] = slot["by_clock"].get(clock, 0) + 1
        outcome = str(r.get("outcome") or "?")
        slot["outcomes"][outcome] = slot["outcomes"].get(outcome, 0) + 1
        if r.get("intended_tf") and r.get("actual_input_tf") \
                and r["intended_tf"] != r["actual_input_tf"]:
            frame_mismatches += 1
        if outcome == "candidate_emitted":
            slot["candidates"] += 1
            dec = r.get("decision") or {}
            code = dec.get("reason_code")
            if dec.get("result") == "refused" and code:
                slot["refusals_by_code"][code] = \
                    slot["refusals_by_code"].get(code, 0) + 1
                cat = dec.get("reason_category")
                if cat is not None:
                    slot["refusals_by_category"][cat] = \
                        slot["refusals_by_category"].get(cat, 0) + 1
            if dec.get("execution_eligible"):
                slot["reached_execution"] += 1
            ores = dec.get("order_result")
            if ores:
                slot["orders"][ores] = slot["orders"].get(ores, 0) + 1
    for name in registered_names:
        if name not in by_playbook:
            not_called.append(str(name))
    return {"by_playbook": by_playbook,
            "not_called": not_called,
            "frame_mismatches": frame_mismatches,
            "trigger_events": trigger_events}
