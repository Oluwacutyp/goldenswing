"""Route PROMOTION with binding verdicts (MarketSituation
Pass 5; owner ruling 2026-11-08, AMENDMENTS-1 §A5).

The pure machinery (execution/route_promotion.py), pinned
here:

- the criteria are PRE-REGISTERED: a cycle journals the
  criteria records before any evaluation record computed
  under them, every verdict cites its criteria id +
  registration stamp, and a verdict with no preceding
  criteria record can never apply (applicable_verdict
  refuses it) — the bars may not move after results are
  seen;
- the verdicts are the range evaluator's verdicts (one
  evaluator, shared with §G): right-way discrimination over
  the route's OWN route-stamped history reads
  promotion_ready — with the BINDING voice, not the §G
  recommendation voice; inverted reads not_ready; thin data
  reads insufficient_data with the n named; the other
  route's history is never read (hot_scalp verdicts come
  from hot_scalp_promotion-stamped records only);
- the violation scan (A5b): a friction breach counts only
  on a PLACED order (a friction refusal is the gate
  working); a broker reconciliation divergence attributes
  by route/playbook; a second placed fill for one setup
  identity is a duplicate order; an explicit
  route_invariant_violation record is honored verbatim.
"""
from __future__ import annotations

import pytest

from goldenswing.execution import range_promotion as rp_range
from goldenswing.execution import route_promotion as rp


# ------------------------------------------------ fabricated journals

def _ts(i: int) -> str:
    return (f"2026-10-{1 + i // 24:02d} {i % 24:02d}:00:00+00:00")


def _shadow(ts, playbook, route, decision="take", *,
            expected_r=0.4):
    return {"type": "judgment_shadow", "ts": ts,
            "playbook": playbook, "intended_tf": "M5",
            "gate_result": "shadow",
            "gate_order_result": "not_attempted",
            "gate_reason_code": "path.shadow_route",
            "judgment": {"decision": decision,
                         "expected_r": expected_r,
                         "confidence": 0.5, "mode": "active",
                         "authority_route": route,
                         "authority": "shadow_declared"}}


def _recon(ts, playbook, *, decision="take", realized_r=0.5,
           hit=True):
    move = None if realized_r is None else realized_r * 10.0
    return {"type": "judgment_reconciliation", "judged_ts": ts,
            "playbook": playbook, "direction": 1,
            "decision": decision, "horizon_h": 4,
            "entry": 2000.0,
            "exit_price": 2000.0 + (move or 0.0),
            "realized_r": realized_r, "hit": hit,
            "move": move, "move_atr": 1.0}


def _arm(playbook, route, decision, rs, *, hits=None, start=0):
    records = []
    for i, r in enumerate(rs):
        ts = _ts(start + i)
        records.append(_shadow(ts, playbook, route, decision))
        records.append(_recon(
            ts, playbook, decision=decision, realized_r=r,
            hit=(hits[i] if hits is not None else r > 0)))
    return records


def _ready_history(playbook="hot_scalp",
                   route="hot_scalp_promotion"):
    records = _arm(playbook, route, "take",
                   [0.5] * 18 + [-0.5] * 4)
    records += _arm(playbook, route, "abstain",
                    [-0.4] * 15 + [0.3] * 6, start=100)
    return records


# ------------------------------------------------ criteria ordering

def test_cycle_journals_criteria_before_any_verdict():
    state = rp.new_state()
    out = rp.cycle(state=state, records=[],
                   now="2026-11-08 00:00:00+00:00",
                   bar="2026-11-08 00:00:00+00:00")
    types = [r["type"] for r in out]
    n_routes = len(rp.PROMOTION_ROUTES)
    assert types == (
        [rp.CRITERIA_RECORD_TYPE] * n_routes
        + [rp.EVALUATION_RECORD_TYPE])
    criteria = out[:n_routes]
    assert {c["route"] for c in criteria} == set(
        rp.PROMOTION_ROUTES)
    evaluation = out[n_routes]
    for route, row in evaluation["routes"].items():
        crit = next(c for c in criteria
                    if c["route"] == route)
        # The verdict cites the criteria it was computed
        # under — id AND registration stamp.
        assert row["criteria_id"] == crit["criteria_id"]
        assert row["criteria_registered_at"] \
            == crit["registered_at"]
    # The bars themselves are stated on the record.
    crit = criteria[0]
    assert crit["floors"] == {"min_paired_per_arm": 20}
    assert crit["horizon_h"] == 4
    assert "promotion_ready" in crit["verdict_rule"]
    # Second cycle, nothing changed: nothing journals.
    out2 = rp.cycle(state=state, records=[],
                    now="2026-11-08 01:00:00+00:00",
                    bar="2026-11-08 01:00:00+00:00")
    assert out2 == []


def test_verdict_without_criteria_can_never_apply():
    """A5a's ordering pin at the application seam: even with
    a promotion_ready snapshot sitting in the state, no
    registered criteria -> no applicable verdict."""
    state = rp.new_state()
    state["last_evaluation"] = {"routes": {
        "hot_scalp_promotion": {
            "verdict": "promotion_ready",
            "criteria_id": "route-promotion:hot_scalp_"
                           "promotion:v1"}}}
    assert rp.applicable_verdict(
        state, "hot_scalp_promotion") is None
    # A criteria id that does not match the registration is
    # likewise refused.
    state["criteria"]["hot_scalp_promotion"] = {
        "criteria_id": "route-promotion:hot_scalp_"
                       "promotion:v99",
        "criteria_version": 99}
    assert rp.applicable_verdict(
        state, "hot_scalp_promotion") is None


def test_ensure_criteria_is_idempotent_per_version():
    state = rp.new_state()
    first = rp.ensure_criteria(
        state, "resting_limit", ts="2026-11-08 00:00:00+00:00")
    assert first is not None
    assert first["criteria_id"] == rp.criteria_id_for(
        "resting_limit")
    assert rp.ensure_criteria(
        state, "resting_limit",
        ts="2026-11-08 01:00:00+00:00") is None


# ------------------------------------------------ verdicts

def test_promotion_ready_binds_on_right_way_discrimination():
    snap = rp.evaluate_records(_ready_history())
    row = snap["routes"]["hot_scalp_promotion"]
    assert row["verdict"] == "promotion_ready"
    assert row["take"]["paired"] == 22
    assert row["take"]["median_r"] == pytest.approx(0.5)
    assert row["abstain"]["paired"] == 21
    # The binding voice, not the §G recommendation voice.
    assert "BINDS" in row["verdict_reason"]
    assert "RECOMMENDATION" not in row["verdict_reason"]
    assert row["criteria_id"] == rp.criteria_id_for(
        "hot_scalp_promotion")
    # The other route read none of this history.
    other = snap["routes"]["resting_limit"]
    assert other["verdict"] == "insufficient_data"
    assert other["evaluated"] == 0
    assert "no candidates observed" in other["verdict_reason"]


def test_inverted_discrimination_is_not_ready():
    records = _arm("hot_scalp", "hot_scalp_promotion", "take",
                   [0.2] * 12 + [-0.6] * 10)
    records += _arm("hot_scalp", "hot_scalp_promotion",
                    "abstain", [0.7] * 14 + [-0.2] * 7,
                    start=100)
    row = rp.evaluate_records(records)["routes"][
        "hot_scalp_promotion"]
    assert row["verdict"] == "not_ready"
    assert "INVERTED" in row["verdict_reason"]


def test_thin_history_is_insufficient_with_n_named():
    records = _arm("arrival_zone", "resting_limit", "take",
                   [0.5] * 3)
    row = rp.evaluate_records(records)["routes"][
        "resting_limit"]
    assert row["verdict"] == "insufficient_data"
    assert "3/20" in row["verdict_reason"]


def test_unstamped_and_other_route_history_never_read():
    """Roster verdicts for the same playbook (h1_trigger
    stamp, or no stamp) are a different question — the route
    evaluation reads its own stamp only."""
    records = _arm("hot_scalp", "h1_trigger", "take",
                   [0.5] * 25)
    records += _arm("hot_scalp", "h1_trigger", "abstain",
                    [-0.4] * 25, start=100)
    row = rp.evaluate_records(records)["routes"][
        "hot_scalp_promotion"]
    assert row["evaluated"] == 0
    assert row["verdict"] == "insufficient_data"


def test_range_evaluator_defaults_unchanged():
    """The generalisation guard: the §G family's own entry
    points keep their exact prior behavior (recommendation
    voice, range routes only)."""
    records = _arm("range_qm", "range_qm", "take",
                   [0.5] * 18 + [-0.5] * 4)
    records += _arm("range_qm", "range_qm", "abstain",
                    [-0.4] * 15 + [0.3] * 6, start=100)
    snap = rp_range.evaluate_records(records)
    row = snap["routes"]["range_qm"]
    assert row["verdict"] == "promotion_ready"
    assert "RECOMMENDATION" in row["verdict_reason"]


# ------------------------------------------------ violation scan

def _filled_eval(playbook="hot_scalp", *, setup_id="hs-1",
                 friction_ratio=None, ts="2026-11-08 00:00:00",
                 order_result="filled"):
    row = {"type": "evaluation", "ts": ts, "playbook": playbook,
           "outcome": "candidate_emitted",
           "candidate": {"setup_id": setup_id},
           "decision": {"result": "accepted",
                        "order_result": order_result}}
    if friction_ratio is not None:
        row["friction_ratio"] = friction_ratio
    return row


def test_scan_explicit_violation_record():
    violations, _ = rp.scan_violations([{
        "type": rp.VIOLATION_RECORD_TYPE,
        "route": "h1_trigger", "kind": rp.KIND_DUPLICATE_ORDER,
        "evidence": {"setup_id": "x"}, "ts": "2026-11-08"}])
    assert len(violations) == 1
    assert violations[0]["route"] == "h1_trigger"
    assert violations[0]["kind"] == "duplicate_order"
    assert violations[0]["source"] == "explicit_record"


def test_scan_never_refires_the_phase_journaled_copy():
    """The runner journals a normalized copy of each
    processed violation (demotion_applied / scan_source
    marks); that copy lands in the next window and must not
    re-fire — one violation, one demotion, and a fresh
    verdict stays able to release it."""
    journaled_copy = {
        "type": rp.VIOLATION_RECORD_TYPE,
        "route": "hot_scalp_promotion",
        "kind": rp.KIND_DUPLICATE_ORDER,
        "evidence": {"setup_id": "x"},
        "demotion_applied": True, "scan_source": "order_record",
        "ts": "2026-11-08 01:00:00"}
    violations, _ = rp.scan_violations([journaled_copy])
    assert violations == []


def test_scan_friction_breach_only_on_placed_orders():
    placed = _filled_eval(friction_ratio=0.42)
    violations, _ = rp.scan_violations([placed])
    assert [v["kind"] for v in violations] == [
        rp.KIND_FRICTION_BREACH]
    assert violations[0]["route"] == "hot_scalp_promotion"
    assert violations[0]["evidence"]["friction_ratio"] == 0.42
    # At/under the limit: no violation.
    ok = _filled_eval(friction_ratio=0.10, setup_id="hs-2")
    violations, _ = rp.scan_violations([ok])
    assert violations == []
    # A friction REFUSAL (nothing placed) is the gate
    # working — never a violation, whatever the ratio.
    refused = _filled_eval(friction_ratio=0.90,
                           order_result="not_attempted",
                           setup_id="hs-3")
    refused["decision"]["result"] = "refused"
    violations, _ = rp.scan_violations([refused])
    assert violations == []


def test_scan_reconciliation_divergence_attributes_by_playbook():
    violations, _ = rp.scan_violations([{
        "type": "reconciliation",
        "event": "divergence_position_mismatch",
        "trade_id": "t-1", "playbook": "qm_m5",
        "detected_at": "2026-11-08 01:00:00"}])
    assert len(violations) == 1
    assert violations[0]["kind"] \
        == rp.KIND_RECONCILIATION_MISMATCH
    assert violations[0]["route"] == "qm_m5_native"
    # Unattributed divergence: the reconciliation layer owns
    # it (it blocks startup itself); the scan invents no
    # route for it.
    violations, _ = rp.scan_violations([{
        "type": "reconciliation",
        "event": "divergence_position_mismatch",
        "trade_id": "t-2"}])
    assert violations == []


def test_scan_duplicate_order_needs_a_second_fill():
    first = _filled_eval(ts="2026-11-08 00:00:00")
    violations, filled = rp.scan_violations([first])
    assert violations == []
    # Same window, second fill for the same setup identity.
    dupe = _filled_eval(ts="2026-11-08 00:01:00")
    violations, filled = rp.scan_violations(
        [dupe], filled_identities=filled)
    assert [v["kind"] for v in violations] == [
        rp.KIND_DUPLICATE_ORDER]
    assert violations[0]["evidence"]["setup_id"] == "hs-1"
    # A different setup is not a duplicate.
    other = _filled_eval(setup_id="hs-9",
                         ts="2026-11-08 00:02:00")
    violations, _ = rp.scan_violations(
        [other], filled_identities=filled)
    assert violations == []
