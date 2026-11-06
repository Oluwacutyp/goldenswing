"""Completion-02 (owner program 2026-11-05): connect the whole
playbook ecosystem to the shared judgment + execution
architecture — above all the two routes the Judgment promotion
declared not_authoritative (hot_scalp promotion, the A6 resting
limit / arrival_zone).

What is pinned here:

- PER-ROUTE authority: the shipped declaration (three governing
  routes, two shadow-declared), the explicit config promotion
  step (judgment_route_modes), and typo-safety toward the
  declared state in both directions.
- hot_scalp candidates now terminate in funnel evaluation
  records + journaled core evaluations (shadow-declared): an
  abstention is RECORDED and the chain's own decision is
  unchanged by it; under the explicit per-route promotion to
  governing, the same abstention refuses the attempt under
  judgment.abstain, exactly like a governed route.
- arrival_zone placements get the same treatment at their
  decision point (candidate honestly constructed from the
  placement's own score/floor/geometry).
- the manifest's per-route authority table (states, governing
  now, accruing shadow-history counts) — with the promotion
  suite's pinned boundary keys unchanged.
- the pending_trigger verdict: a registered setup's
  order_result "pending_trigger" is a WAITING state with a
  journaled binder and journaled terminal exits (the
  2026-11-05 06:00 fvg_limit shape: opposite-direction fires
  decline by record, the instance stays armed, and validity
  lapse terminates it through upkeep — never a silent strand).
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.live.zones import StructuralZone

from test_confluence_rework import _point3_frames
from test_confluence_rework import _runner as _az_runner
from test_p3_trader import _flat_h1
from test_step14_setup_lifecycle import _scan_runner, _sweep
from test_step6_hot_scalp_promotion import _candidate as _hs_cand
from test_step6_hot_scalp_promotion import _runner as _hs_runner
from test_step6_hot_scalp_promotion import _status as _hs_status
from test_step6_hot_scalp_promotion import _tick as _hs_tick


def _evals(runner, playbook):
    return [r for r in runner.journal.records()
            if r.get("type") == "evaluation"
            and r.get("playbook") == playbook]


def _shadows(runner, playbook):
    return [r for r in runner.journal.records()
            if r.get("type") == "judgment_shadow"
            and r.get("playbook") == playbook]


def _chain(runner, stage):
    return [r for r in runner.journal.records()
            if r.get("type") == "candidate_chain"
            and r.get("stage") == stage]


# ------------------------------------------------ route states

def test_route_states_shipped_declaration(tmp_path):
    runner = _hs_runner(tmp_path)
    assert runner._judgment_route_state("h1_trigger") \
        == "governing"
    assert runner._judgment_route_state("qm_m5_native") \
        == "governing"
    assert runner._judgment_route_state("qm_d1_native") \
        == "governing"
    assert runner._judgment_route_state("hot_scalp_promotion") \
        == "shadow_declared"
    assert runner._judgment_route_state("resting_limit") \
        == "shadow_declared"


def test_route_promotion_is_an_explicit_config_step(tmp_path):
    runner = _hs_runner(tmp_path)
    runner.cfg.judgment_route_modes = {
        "hot_scalp_promotion": "governing",
        "qm_m5_native": "shadow_declared"}
    assert runner._judgment_route_state("hot_scalp_promotion") \
        == "governing"
    assert runner._judgment_route_state("qm_m5_native") \
        == "shadow_declared"
    # Untouched routes keep their declared state.
    assert runner._judgment_route_state("resting_limit") \
        == "shadow_declared"


def test_route_override_typos_never_promote_or_demote(tmp_path):
    runner = _hs_runner(tmp_path)
    runner.cfg.judgment_route_modes = {
        "hot_scalp_promotion": "govering",   # typo: must NOT promote
        "h1_trigger": "shadow",             # not a state name
        "resting_limit": " SHADOW_DECLARED "}
    assert runner._judgment_route_state("hot_scalp_promotion") \
        == "shadow_declared"
    assert runner._judgment_route_state("h1_trigger") \
        == "governing"
    assert runner._judgment_route_state("resting_limit") \
        == "shadow_declared"


# ------------------------------------------------ hot_scalp

def test_hot_scalp_promotion_journals_funnel_and_core(tmp_path):
    runner = _hs_runner(tmp_path, live=True)
    cand = _hs_cand(runner)
    assert runner._promote_hot_scalp_candidate(
        cand, _hs_tick(), {}) is True
    # The chain's own outcome is exactly as before the connection.
    assert len(runner.executor.placed) == 1
    assert _hs_status(runner) == "order_reconciled"
    # ONE funnel record, funnel-shaped, terminal = filled.
    evs = _evals(runner, "hot_scalp")
    assert len(evs) == 1
    rec = evs[0]
    assert rec["outcome"] == "candidate_emitted"
    assert rec["decision"]["result"] == "accepted"
    assert rec["decision"]["order_result"] == "filled"
    cand_block = rec["candidate"]
    assert cand_block["setup_id"] == "hs-test-parent-1"
    assert cand_block["direction"] == 1
    assert cand_block["entry"] == pytest.approx(100.30)
    assert cand_block["stop"] == pytest.approx(99.0)
    assert cand_block["targets"] == [102.0]
    # Phase-2 naming honesty: sleeve base score, no asymmetry.
    assert cand_block["asymmetry_score"] is None
    assert cand_block["sleeve_base_score"] == pytest.approx(60.0)
    # ONE core evaluation, route-stamped shadow-declared.
    shadows = _shadows(runner, "hot_scalp")
    assert len(shadows) == 1
    j = shadows[0]["judgment"]
    assert j["decision"] == "take"
    assert j["mode"] == "active"
    assert j["authority"] == "shadow_declared"
    assert j["authority_route"] == "hot_scalp_promotion"
    assert shadows[0]["gate_order_result"] == "filled"
    # The chain's judgment stage carries the core summary too.
    judged = _chain(runner, "judgment")
    assert len(judged) == 1
    assert judged[0]["core"]["decision"] == "take"
    assert judged[0]["core"]["governing"] is False
    # Funnel counters reconcile the sleeve like any playbook.
    totals = runner.funnel_totals()
    assert totals["candidates"] >= 1 and totals["ordered"] >= 1


def test_hot_scalp_abstain_is_journaled_not_governing(tmp_path):
    """THE shadow-declared contract: the core abstains (forced
    by an unreachable expected-R bar), the verdict is journaled
    as shadow history — and the chain's own stack still decides:
    the promotion succeeds exactly as it did pre-connection."""
    runner = _hs_runner(tmp_path, live=True)
    runner.cfg.judgment_min_expected_r = 99.0
    cand = _hs_cand(runner)
    assert runner._promote_hot_scalp_candidate(
        cand, _hs_tick(), {}) is True
    assert len(runner.executor.placed) == 1
    assert _hs_status(runner) == "order_reconciled"
    shadows = _shadows(runner, "hot_scalp")
    assert len(shadows) == 1
    j = shadows[0]["judgment"]
    assert j["decision"] == "abstain"
    assert j["authority"] == "shadow_declared"
    assert _chain(runner, "judgment")[0]["core"]["decision"] \
        == "abstain"


def test_hot_scalp_route_gate_refusal_also_journals(tmp_path):
    """A candidate refused by the route's OWN gates (here:
    paused) still terminates in a funnel record + a tap-journaled
    core evaluation — no promotion attempt is invisible."""
    runner = _hs_runner(tmp_path, live=True, paused=True)
    cand = _hs_cand(runner)
    assert runner._promote_hot_scalp_candidate(
        cand, _hs_tick(), {}) is False
    assert runner.executor.placed == []
    evs = _evals(runner, "hot_scalp")
    assert len(evs) == 1
    assert evs[0]["decision"]["result"] == "refused"
    assert evs[0]["decision"]["reason_code"] \
        == "promotion.runner_paused"
    shadows = _shadows(runner, "hot_scalp")
    assert len(shadows) == 1
    assert shadows[0]["judgment"]["authority"] \
        == "shadow_declared"


def test_hot_scalp_explicit_promotion_makes_abstain_govern(
        tmp_path):
    """The explicit step, proven: with the route promoted to
    governing by config, the same forced abstention REFUSES the
    attempt under judgment.abstain; the claim resolves back to
    touched (a transient refusal — the lifecycle is untouched
    beyond this attempt)."""
    runner = _hs_runner(tmp_path, live=True)
    runner.cfg.judgment_min_expected_r = 99.0
    runner.cfg.judgment_route_modes = {
        "hot_scalp_promotion": "governing"}
    cand = _hs_cand(runner)
    assert runner._promote_hot_scalp_candidate(
        cand, _hs_tick(), {}) is False
    assert runner.executor.placed == []
    assert _hs_status(runner) == "touched"
    judged = _chain(runner, "judgment")
    assert judged and judged[0]["execution_status"] == "refused"
    assert judged[0]["reason_code"] == "judgment.abstain"
    evs = _evals(runner, "hot_scalp")
    assert len(evs) == 1
    assert evs[0]["decision"]["reason_code"] == "judgment.abstain"
    j = _shadows(runner, "hot_scalp")[0]["judgment"]
    assert j["decision"] == "abstain"
    assert j["authority"] == "governing"


def test_hot_scalp_global_shadow_rollback_still_journals(
        tmp_path):
    """The global kill switch set to shadow: the promotion path
    is untouched (the chain decides) and the verdict journals
    via the tap in shadow mode, route-stamped."""
    runner = _hs_runner(tmp_path, live=True)
    runner.cfg.judgment_mode = "shadow"
    cand = _hs_cand(runner)
    assert runner._promote_hot_scalp_candidate(
        cand, _hs_tick(), {}) is True
    assert len(runner.executor.placed) == 1
    shadows = _shadows(runner, "hot_scalp")
    assert len(shadows) == 1
    j = shadows[0]["judgment"]
    assert j["mode"] == "shadow"
    assert j["authority"] == "shadow_declared"


# ------------------------------------------------ arrival_zone

def _az_drive(tmp_path, **cfg):
    runner = _az_runner(tmp_path, resting_limit_enabled=True)
    for key, value in cfg.items():
        setattr(runner.cfg, key, value)
    runner._stack = SimpleNamespace(frames=_point3_frames())
    zone = StructuralZone(
        tf="H4", side="support", level=4110.0, top=4122.0,
        bottom=4098.0, score=85.0, tier="MAJOR", status="active",
        touches=0, atr=12.0, displacement_atr=2.5,
        departure_atr=2.5, swing_ts="2026-10-18T00:00:00+00:00")
    runner._structural_zones = [zone]
    runner._live_setup = {
        "direction": 1, "regime": "trend", "atr_h1": 12.0,
        "buf": 3.0, "major_zone": zone.to_dict(),
        "zone": {"top": 4122.0, "bottom": 4098.0, "level": 4110.0},
        "bias_d1": 0, "bias_h4": 1,
    }
    tick = {"mid": 4122.0 + 3.6, "spread": 0.3,
            "time": pd.Timestamp("2026-10-21 13:00", tz="UTC")}
    runner._maybe_place_resting_limit(tick, {})
    return runner


def test_arrival_zone_placement_journals_funnel_and_core(
        tmp_path):
    runner = _az_drive(tmp_path)
    # The placement itself is exactly as before the connection.
    assert len(runner._pending_orders) == 1
    state = next(iter(runner._pending_orders.values()))
    assert state.playbook == "arrival_zone"
    evs = _evals(runner, "arrival_zone")
    assert len(evs) == 1
    rec = evs[0]
    assert rec["decision"]["result"] == "accepted"
    assert rec["decision"]["order_result"] == "placed_pending"
    cand_block = rec["candidate"]
    assert cand_block["entry"] == pytest.approx(4122.0)
    # Structural invalidation: distal (4098) - buffer (3.0).
    assert cand_block["stop"] == pytest.approx(4095.0)
    assert cand_block["asymmetry_score"] is not None
    shadows = _shadows(runner, "arrival_zone")
    assert len(shadows) == 1
    j = shadows[0]["judgment"]
    assert j["authority"] == "shadow_declared"
    assert j["authority_route"] == "resting_limit"
    assert j["mode"] == "active"


def test_arrival_zone_abstain_does_not_block_placement(tmp_path):
    runner = _az_drive(tmp_path, judgment_min_expected_r=99.0)
    assert len(runner._pending_orders) == 1
    j = _shadows(runner, "arrival_zone")[0]["judgment"]
    assert j["decision"] == "abstain"
    assert j["authority"] == "shadow_declared"


def test_arrival_zone_explicit_promotion_governs(tmp_path):
    runner = _az_drive(
        tmp_path, judgment_min_expected_r=99.0,
        judgment_route_modes={"resting_limit": "governing"})
    assert runner._pending_orders == {}
    evs = _evals(runner, "arrival_zone")
    assert len(evs) == 1
    assert evs[0]["decision"]["result"] == "refused"
    assert evs[0]["decision"]["reason_code"] == "judgment.abstain"
    j = _shadows(runner, "arrival_zone")[0]["judgment"]
    assert j["authority"] == "governing"


# ------------------------------------------------ manifest

def test_manifest_route_authority_table(tmp_path):
    runner = _hs_runner(tmp_path, live=True)
    cand = _hs_cand(runner)
    assert runner._promote_hot_scalp_candidate(
        cand, _hs_tick(), {}) is True
    runner._journal_manifest()
    core = [r for r in runner.journal.records()
            if r.get("type") == "manifest"][-1][
                "doctrine"]["judgment_core"]
    # The promotion suite's pinned boundary keys are unchanged.
    assert core["authority_routes"] == [
        "h1_trigger", "qm_m5_native", "qm_d1_native"]
    assert core["not_authoritative"] == [
        "hot_scalp_promotion", "resting_limit"]
    table = {row["route"]: row
             for row in core["route_authority"]}
    # Extended by the §G promotion-evaluation wiring
    # (2026-11-05): the four range routes are declared
    # shadow_declared and join the census — the pin's intent
    # (an exact route census) is preserved.
    assert set(table) == {
        "h1_trigger", "qm_m5_native", "qm_d1_native",
        "hot_scalp_promotion", "resting_limit",
        "range_sweep_reclaim", "range_qm",
        "range_mean_reversion", "range_compression_breakout"}
    assert table["h1_trigger"]["authority"] == "governing"
    assert table["h1_trigger"]["governing_now"] is True
    hs = table["hot_scalp_promotion"]
    assert hs["authority"] == "shadow_declared"
    assert hs["governing_now"] is False
    # The run's accruing shadow history, counted.
    assert hs["judgment_records_this_run"] == 1
    assert table["resting_limit"][
        "judgment_records_this_run"] == 0
    for route in ("range_sweep_reclaim", "range_qm",
                  "range_mean_reversion",
                  "range_compression_breakout"):
        assert table[route]["authority"] == "shadow_declared"
        assert table[route]["governing_now"] is False
        assert table[route][
            "judgment_records_this_run"] == 0


# ------------------------------------------------ §G shadow route

def test_shadow_sweep_candidates_are_core_evaluated(tmp_path):
    """Census pin for the §G range route's connection state:
    a shadow-route candidate (the _sweep_eval_one mechanics the
    range family runs under shadow=True) journals its funnel
    record with the shadow disposition AND receives a tap-
    journaled core evaluation — evaluated, never registered as
    routable, never ordered.  What the §G route lacks is order
    authority, not evaluation."""
    runner, h1 = _scan_runner(tmp_path)
    _sweep(runner, h1, shadow=True, direction=1)
    scans = [r for r in _evals(runner, "fvg_midline")
             if r.get("outcome") == "candidate_emitted"]
    assert scans
    assert scans[-1]["decision"]["result"] == "shadow"
    assert scans[-1]["decision"]["order_result"] \
        == "not_attempted"
    jud = _shadows(runner, "fvg_midline")
    assert jud
    assert jud[-1]["gate_result"] == "shadow"


# ------------------------------------------------ pending_trigger

def test_pending_trigger_is_a_waiting_state_not_a_strand(
        tmp_path):
    """The 2026-11-05 06:00 fvg_limit shape, reproduced: a
    scan-registered LONG setup reads order_result
    "pending_trigger"; opposite-direction fires journal binder
    records declining it (direction_mismatch) and it stays
    armed; when its validity lapses, upkeep terminates it with
    a journaled expired transition.  Nothing is stranded: every
    state change is a record."""
    runner, h1 = _scan_runner(tmp_path)
    _sweep(runner, h1, direction=1)
    scans = [r for r in _evals(runner, "fvg_midline")
             if r.get("outcome") == "candidate_emitted"]
    assert scans and scans[-1]["decision"]["order_result"] \
        == "pending_trigger"
    store = runner._setup_store
    armed = store.active()
    assert len(armed) == 1
    setup = armed[0]
    assert setup.stage == "setup_armed"
    # Two opposite-direction fires: the binder journals its
    # selection both times and declines by record.
    for _ in range(2):
        runner._emit_trigger_binding(
            direction=-1, verdict={"entry": 2000.0}, h1=h1,
            t=len(h1) - 1, armed_all=store.active(),
            candidates={}, rules={})
    bindings = [r for r in runner.journal.records()
                if r.get("kind") == "trigger_binding"]
    assert len(bindings) == 2
    for b in bindings:
        assert b["reason_code"] \
            == "path.binding_direction_mismatch"
        inst = b["evidence"]["binding"]["instances"]
        assert inst[0]["disposition"] == "other_direction"
    # Still armed — waiting is the state, not a disappearance.
    assert store.get(setup.setup_id).stage == "setup_armed"
    # Validity lapse: a later frame's close runs upkeep, which
    # terminates the instance with a journaled transition.
    longer = _flat_h1(60, 2000.0)
    runner._upkeep_setups(longer)
    assert store.get(setup.setup_id).stage == "expired"
    events = [r for r in runner.journal.records()
              if r.get("kind") == "setup_event"
              and (r.get("evidence") or {}).get(
                  "lifecycle_stage") == "expired"]
    assert events
