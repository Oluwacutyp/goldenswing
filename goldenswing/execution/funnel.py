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
    # Phase 2 provenance/scheduler classes: the candidate or the
    # evaluation never legitimately reached a decision (warmup /
    # catch-up / replay provenance, duplicate instance, shadow
    # route, trigger-gated roster scan) — category 1, never a
    # fabricated quality/risk class.
    "path.warmup_ineligible": CATEGORY_NOT_EVALUATED,
    "path.duplicate_instance": CATEGORY_NOT_EVALUATED,
    "path.trigger_not_fired": CATEGORY_NOT_EVALUATED,
    "path.shadow_route": CATEGORY_NOT_EVALUATED,
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


#: Order-path results that are TERMINAL REFUSALS: the candidate
#: was analytically accepted at its decision stage but the shared
#: order path refused it (hard-risk veto, margin, sizing, broker
#: geometry).  A record carrying one of these is a REFUSAL in
#: every census, whatever the decision-stage label says.
TERMINAL_ORDER_REFUSALS = ("refused_in_order_path", "rejected")

#: Order-path results that mean an order exists at the broker.
ORDERED_RESULTS = ("filled", "placed_pending", "layer_filled",
                   "layer_pending")


def terminal_decision(decision: dict | None) -> dict | None:
    """Normalize a decision block to TERMINAL semantics.

    Phase 1 records could carry ``result="accepted"`` (the
    decision stage's analytical verdict) together with
    ``order_result="refused_in_order_path"`` — /status counted
    only ``result``, so order-path refusals rendered as
    unaccounted candidates and signal rows read "accepted" for
    trades that were never placed (EC2 run cb9b3fb522b7: 23 of
    32 candidates).  The terminal truth wins: when the order
    path refused, ``result`` becomes ``"refused"`` and the
    analytical verdict is preserved verbatim in
    ``analytical_result`` — the judgment is not erased, it is
    just no longer mistaken for the outcome.  ``reason_code``
    already carries the order-path code at those call sites
    (the stage where the trade actually died, per the §3 exit
    criterion in :func:`category_for_code`); a stale
    ``reason_category`` is dropped so the emitter recomputes it
    from the terminal code.

    Observation-only in the Phase 1 sense: this normalizes the
    RECORD, never the trading decision itself.
    """
    if not decision:
        return decision
    ores = decision.get("order_result")
    if ores in TERMINAL_ORDER_REFUSALS \
            and decision.get("result") != "refused":
        out = dict(decision)
        out["analytical_result"] = decision.get("result")
        out["result"] = "refused"
        out.pop("reason_category", None)
        return out
    return decision


def decision_bucket(decision: dict | None) -> str:
    """The ONE terminal bucket a candidate's decision falls in.

    Every candidate ever emitted lands in exactly one bucket, so
    a census reconciles by construction:
    ``candidates == ordered + refused + ineligible + no_order
    + unresolved`` — and a nonzero ``unresolved`` is itself a
    reported defect, never a silent gap.
    """
    dec = decision or {}
    ores = dec.get("order_result")
    if ores in ORDERED_RESULTS:
        return "ordered"
    if dec.get("result") == "refused" \
            or ores in TERMINAL_ORDER_REFUSALS:
        return "refused"
    if dec.get("execution_eligible") is False:
        return "ineligible"
    if ores == "layer_no_order":
        return "no_order"
    return "unresolved"


def verdict_blocks(*, analytical: str,
                     analytical_code: str | None = None,
                     eligible: bool,
                     execution_code: str | None = None,
                     outcome: str = "not_attempted") -> dict:
    """The two-layer verdict every decision record carries
    (owner ruling, 2026-10-23): ANALYTICAL VERDICT and
    EXECUTION ELIGIBILITY are separate layers and must be
    readable at a glance, distinctly.

    - ``analytical_verdict``: the STRATEGY layer's judgment —
      "qualified" (the setup qualified on its own evidence:
      score, components, floor, scenarios), "not_qualified"
      (the strategy layer itself refused it, with its code),
      or "not_evaluated" (provenance/scheduling meant the
      strategy layer never ran — warmup, expired, scan-only).
    - ``execution``: the RISK layer's answer — whether the
      candidate was eligible to reach the broker, the outcome,
      and the risk-layer reason code when refused (solvency,
      friction, margin, lifecycle ineligibility...).

    A correct refusal at a small balance is the risk layer
    doing its job: analytical "qualified" + execution refused
    gate.solvency says exactly that, and must never be readable
    as a strategy failure.  A verdict difference across
    balances would be a defect; an eligibility difference
    without a risk-layer code would be a defect.  Both are
    test-pinned (tests/test_phase2_fixes.py).
    """
    return {
        "analytical_verdict": {
            "verdict": analytical,
            "reason_code": analytical_code},
        "execution": {
            "eligible": bool(eligible),
            "outcome": outcome,
            "reason_code": execution_code},
    }


#: Capability markers stamped on every assessment (Amendment 3,
#: owner clarification 2026-10-23): capabilities the census
#: classifies as NOT live must be marked on the record itself —
#: absent / stale / not-integrated — never silently omitted.
#: Values are census facts (capability-census.md), not opinions.
ASSESSMENT_CAPABILITY_MARKERS = {
    "named_candlestick_taxonomy":
        "absent — no named-pattern candlestick detector (hammer / "
        "doji / engulfing-by-name) exists in the live decision "
        "path; candle evidence reaches decisions only as QM "
        "pattern tags and bar geometry",
    "premium_discount":
        "not_integrated — computed by live/patterns.py "
        "premium_discount(); no live consumer reads it",
    "legacy_smc_block":
        "not_integrated — live/patterns.py structure/order-block/"
        "breaker/FVG helpers have no live consumer; the live SMC "
        "evidence comes from the playbooks and live/zones.py",
}

#: The owner's assessment verdict vocabulary (Amendment 3).
ASSESSMENT_VERDICTS = ("signal", "watch", "no-setup", "rejected",
                       "expired")


def assessment_block(*, outcome: str,
                     decision: dict | None = None,
                     evidence: dict | None = None,
                     candidate: dict | None = None,
                     reason: str = "",
                     reason_code: str | None = None,
                     last_bar_ts=None,
                     data_freshness_s=None,
                     overlay: dict | None = None) -> dict:
    """Shape ONE evaluation record's MARKET ASSESSMENT block
    (Amendment 3, owner clarification 2026-10-23).

    The owner reads "signals" as a structured market assessment,
    not a bare BUY/SELL: structure read, HTF context, detected
    evidence with detector + source-bar timestamps, supporting
    vs contradictory evidence as SEPARATE lists (correlated items
    grouped — they are one event, not stacked confirmations),
    competing scenarios, a verdict in the owner's vocabulary
    (signal / watch / no-setup / rejected / expired) and a
    plain-language reason.  Where a route genuinely produces no
    structure read, no scenarios or no evidence, the field says
    so EXPLICITLY (``status: absent / not_produced``) — a missing
    field is a silent gap; a marked field is a fact.  Nothing
    here invents a score or a read: every value is derived from
    the record's own evidence/decision blocks, or marked absent.
    """
    ev = evidence or {}
    dec = decision or {}
    bucket = decision_bucket(dec) if dec else None
    code = reason_code or dec.get("reason_code")
    # --- verdict, in the owner's vocabulary -------------------
    if outcome == "candidate_emitted":
        if bucket == "ordered":
            verdict = "signal"
        elif code in ("path.trigger_not_fired",
                      "path.shadow_route"):
            # Observed-only records (roster sweep / §G shadow):
            # a setup was seen and deliberately not acted on.
            verdict = "watch"
        elif bucket == "ineligible":
            verdict = ("expired" if code in (
                "gate.data", "path.warmup_ineligible")
                else "rejected")
        elif bucket == "refused":
            verdict = ("expired" if code == "gate.data"
                       else "rejected")
        else:  # no_order / unresolved: observed, not acted on
            verdict = "watch"
    elif outcome == "not_eligible":
        # Stale/warmup provenance reads as expired; a disabled,
        # paused or market-closed evaluation produced no setup —
        # the precise cause rides in reason_text either way.
        verdict = ("expired" if code in (
            "gate.data", "path.warmup_ineligible")
            else "no-setup")
    else:  # no_setup / exception / skipped shapes
        verdict = "no-setup"
    # --- plain-language reason --------------------------------
    reason_text = (dec.get("reason") or reason or "").strip()
    if not reason_text:
        reason_text = {
            "signal": "setup qualified and the order was placed",
            "watch": "candidate observed; no order resulted",
            "no-setup": "evaluation completed; no setup present",
            "rejected": f"refused ({code})" if code
                        else "refused",
            "expired": "candidate expired or warmup-sourced — "
                       "never execution-eligible",
        }[verdict]
    # --- structure read + HTF context --------------------------
    inferred = ev.get("inferred") or {}
    observed = ev.get("observed") or {}
    structure_read: dict = {}
    for key in ("regime", "posture", "journey_state", "alignment"):
        if inferred.get(key) is not None:
            structure_read[key] = inferred[key]
    if ev.get("structure") is not None:
        structure_read["structure"] = ev["structure"]
    if not structure_read:
        structure_read = {
            "status": "absent",
            "note": "this route produces no structure read"}
    htf_context: dict = {}
    if ev.get("d1_bias_context") is not None:
        htf_context["d1_bias_context"] = ev["d1_bias_context"]
    if ev.get("hierarchy") is not None:
        htf_context["hierarchy"] = ev["hierarchy"]
    for key in ("bias_d1", "bias_h4"):
        if observed.get(key) is not None:
            htf_context[key] = observed[key]
    if observed.get("session_bucket") is not None:
        htf_context["session_bucket"] = observed["session_bucket"]
    if not htf_context:
        htf_context = {
            "status": "absent",
            "note": "this route records no HTF context snapshot"}
    # --- detected evidence, itemized ---------------------------
    # Correlation groups: evidence["evidence_groups"] maps a
    # group name to the item texts that are ONE event; invert it
    # so each item carries its group (correlated items must not
    # read as independent confirmations).
    group_of: dict[str, str] = {}
    groups = ev.get("evidence_groups")
    if isinstance(groups, dict):
        for gname, items in groups.items():
            for text in items or []:
                group_of[str(text)] = str(gname)
    prov = ev.get("provenance") or {}
    src_ts = (prov.get("evaluated_bar_ts") or prov.get("touch_ts")
              or (str(last_bar_ts) if last_bar_ts is not None
                  else None))
    detector = ev.get("route") or "route_not_recorded"

    def _items(texts, stance):
        return [{"item": str(t), "stance": stance,
                 "detector": detector,
                 "source_bar_ts": src_ts,
                 "evidence_group": group_of.get(str(t))}
                for t in (texts or [])]

    supporting = list(ev.get("supporting") or [])
    contradictory = list(ev.get("contradictory") or [])
    evidence_items = (_items(supporting, "supporting")
                      + _items(contradictory, "contradictory"))
    # --- competing scenarios ------------------------------------
    scenarios = dec.get("competing_scenarios")
    if scenarios is None:
        scenarios = {
            "status": "not_produced",
            "note": "this route produces no competing scenarios"}
    # --- capability / freshness markers --------------------------
    markers = dict(ASSESSMENT_CAPABILITY_MARKERS)
    if not evidence:
        markers["evidence"] = ("absent — this evaluation produced "
                               "no evidence block")
    if data_freshness_s is None:
        markers["data_freshness"] = ("not_recorded — no freshness "
                                     "measurement on this record")
    elif code == "gate.data":
        markers["data_freshness"] = (
            f"stale — freshness {data_freshness_s}s was refused "
            "as gate.data")
    block = {
        "verdict": verdict,
        "reason_text": reason_text,
        "structure_read": structure_read,
        "htf_context": htf_context,
        "evidence_items": evidence_items,
        "supporting_evidence": supporting,
        "contradictory_evidence": contradictory,
        "independent_evidence_groups": (
            ev.get("independent_evidence_groups")
            if isinstance(ev.get("independent_evidence_groups"),
                          list) else sorted(group_of)),
        "competing_scenarios": scenarios,
        "capability_markers": markers,
    }
    if overlay:
        for key, value in overlay.items():
            if value is not None:
                block[key] = value
    return block


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
            "reached_execution": 0, "orders": {},
            "terminal": {"ordered": 0, "refused": 0, "ineligible": 0,
                         "no_order": 0, "unresolved": 0}})
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
            dec = terminal_decision(r.get("decision")) or {}
            # Terminal bucket: candidates reconcile in ONE view —
            # candidates == sum(terminal.values()), always.
            slot["terminal"][decision_bucket(dec)] += 1
            code = dec.get("reason_code")
            if decision_bucket(dec) == "refused" and code:
                slot["refusals_by_code"][code] = \
                    slot["refusals_by_code"].get(code, 0) + 1
                cat = dec.get("reason_category")
                if cat is None:
                    cat = category_for_code(
                        code, dec.get("result"))
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
