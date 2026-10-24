"""Burn-down #1 (owner, 2026-10-23): trigger-path assessment
objects.

The assessment milestone shipped the renderer, but the H1
roster (trigger path + scheduled close sweeps) and the §G
range family journaled records whose structure / scenarios /
evidence sections rendered "not produced" — the objects only
existed on the native routes (qm_m5 / qm_d1).  These tests pin
the completed shape:

- every EVALUATED roster/range record carries the Amendment-3
  objects built from the same causal inputs the decider weighed
  (structure read, HTF context, itemized grouped evidence,
  competing scenarios, selected action / rejected alternatives
  on terminal trigger-path decisions);
- a no-setup record states what structure was present and why
  no scenario qualified — never a bare "not produced";
- concurrent candidates from one underlying move share
  correlation groups (evidence items carry their group);
- NOT-evaluated records (kill switch) keep the honest absent
  marking — an evaluation that never ran wears no structure
  read (the Amendment-3 pin in test_phase2_fixes);
- decisions are untouched: terminal fields on enriched records
  are exactly the pre-burn-down values (observation-only).
"""
from __future__ import annotations

from types import SimpleNamespace

from goldenswing.execution.assessment_render import (
    is_shadow_record, render_assessment)

from test_funnel_m5 import _fresh_tick, _qm_runner
from test_p3_trader import (_FakeScore, _StubPB, _drive_one,
                            _flat_h1, _frame, _runner, _sig)


def _evals(runner, playbook=None):
    out = [r for r in runner.journal.records()
           if r.get("type") == "evaluation"]
    if playbook is not None:
        out = [r for r in out if r.get("playbook") == playbook]
    return out


def _trigger_runner(tmp_path, score):
    runner = _runner(tmp_path,
                     execute_playbooks=("fvg_midline",))
    h1 = _flat_h1(30, 2000.0)
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._score_for = lambda **kw: score
    runner._last_tick = _fresh_tick(h1, mid=2000.0)
    return runner, h1


# ------------------------------------------------ trigger path

def test_trigger_candidate_carries_full_assessment_objects(
        tmp_path):
    runner, h1 = _trigger_runner(
        tmp_path, _FakeScore(70.0, 25.0, 15.0))
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
    recs = [r for r in _evals(runner, "fvg_midline")
            if r["outcome"] == "candidate_emitted"]
    assert recs
    rec = recs[-1]
    a = rec["assessment"]
    # Structure + HTF come from the weighed inputs, not absent.
    assert a["structure_read"].get("regime") == "trend"
    assert a["structure_read"].get("journey_state")
    assert a["htf_context"].get("bias_d1") == 1
    # Itemized, grouped evidence: the detector's factor, the
    # score's stance and the hierarchy agreement are separate
    # groups (one event each, never stacked confirmations).
    groups = set(a["independent_evidence_groups"])
    assert {"fvg_midline_signal", "asymmetry_score",
            "htf_structure"} <= groups
    items = {i["item"]: i for i in a["evidence_items"]}
    assert any("signal factor" in t for t in items)
    assert all(i["evidence_group"] for i in items.values())
    # Competing scenarios + the decision's own bookkeeping.
    scen = rec["decision"]["competing_scenarios"]
    assert scen[0]["status"] == "selected"
    assert "not funded" in scen[1]["status"]
    assert rec["decision"]["selected_action"]
    assert len(rec["decision"]["rejected_alternatives"]) >= 2
    # The render is the native routes' richness, verbatim.
    text = rec["assessment_text"]
    assert "Market structure:" in text
    assert "Market structure: not produced" not in text
    assert "Competing scenarios:" in text
    assert "not produced" not in text.split(
        "Competing scenarios:")[1].split("Capability")[0]
    assert "Selected action:" in text
    assert "Rejected alternative:" in text
    # Decision invariance: terminal fields are the pre-burn-down
    # values, byte for byte.
    dec = rec["decision"]
    assert dec["result"] == "accepted"
    assert dec["order_result"] == "placed_pending"
    assert dec["execution_eligible"] is True
    assert dec["analytical_verdict"]["verdict"] == "qualified"


def test_trigger_floor_refusal_scenarios_and_evidence(tmp_path):
    runner, h1 = _trigger_runner(
        tmp_path, _FakeScore(70.0, 10.0, 10.0))
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
    recs = [r for r in _evals(runner, "fvg_midline")
            if r["outcome"] == "candidate_emitted"]
    assert recs
    rec = recs[-1]
    assert rec["decision"]["result"] == "refused"
    assert rec["reason_code"] in ("floor.total", "floor.component")
    scen = rec["decision"]["competing_scenarios"]
    assert scen[0]["status"].startswith("not taken — refused")
    assert rec["assessment"]["evidence_items"]
    assert "Competing scenarios: not produced" not in \
        rec["assessment_text"]


def test_trigger_no_setup_fallback_carries_context(tmp_path):
    class _EmptyPB:
        session_scope = "any"
        name = "fvg_midline"

        def live_signal(self, states, t, df=None):
            return []

    runner, h1 = _trigger_runner(
        tmp_path, _FakeScore(70.0, 25.0, 15.0))
    _drive_one(runner, _EmptyPB(), h1)
    recs = [r for r in _evals(runner, "fvg_midline")
            if r["outcome"] == "no_setup"]
    assert recs
    rec = recs[-1]
    a = rec["assessment"]
    # The structure that WAS present (the fired verdict's read)
    # rides the record; the scenario section states why no
    # scenario qualified instead of "not produced".
    assert a["structure_read"].get("regime") == "trend"
    assert a["htf_context"].get("hierarchy")
    scen = a["competing_scenarios"]
    assert isinstance(scen, list) and scen
    assert scen[0]["status"].startswith("not formed")
    assert "Market structure: not produced" not in \
        rec["assessment_text"]


# ------------------------------------------------------ sweeps

def _sweep_runner(tmp_path):
    runner = _qm_runner(tmp_path)
    h1 = _flat_h1(60, 95.0)
    runner._stack = SimpleNamespace(
        frames={"H1": h1,
                "H4": _frame([(95.0, 95.5, 94.5, 95.0)] * 60,
                             freq="4h")})
    runner._last_tick = _fresh_tick(h1)
    runner._live_setup = {"bias_d1": 1, "bias_h4": 1,
                          "regime": "trend",
                          "h4_context": "uptrend structure",
                          "size_mult": 1.0}
    return runner, h1


def test_sweep_no_setup_carries_structure_and_scenario(tmp_path):
    runner, _h1 = _sweep_runner(tmp_path)
    runner._run_roster_close_evals({}, source="live_close")
    recs = [r for r in _evals(runner, "fvg_midline")
            if r["outcome"] == "no_setup"]
    assert recs
    a = recs[-1]["assessment"]
    assert a["structure_read"].get("regime") == "trend"
    assert a["htf_context"].get("bias_d1") == 1
    assert a["htf_context"].get("session_bucket")
    scen = a["competing_scenarios"]
    assert isinstance(scen, list)
    assert scen[0]["status"].startswith("not formed")
    assert "Competing scenarios: not produced" not in \
        recs[-1]["assessment_text"]


def test_sweep_candidate_observed_objects(tmp_path):
    runner, h1 = _sweep_runner(tmp_path)
    frame = _flat_h1(40, 2000.0)
    spec = SimpleNamespace(
        name="fvg_midline", route="h1_roster",
        playbook_class=lambda: _StubPB(
            "fvg_midline", _sig(index=len(frame) - 1)))
    runner._sweep_eval_one(
        spec, frame, clock="h1_close", intended_tf="H1",
        actual_tf="H1", source="live_close", shadow=False)
    recs = [r for r in _evals(runner, "fvg_midline")
            if r["outcome"] == "candidate_emitted"]
    assert recs
    rec = recs[-1]
    # Terminal truth unchanged: observed, never routed.
    assert rec["decision"]["result"] == "observed"
    assert rec["reason_code"] == "path.trigger_not_fired"
    assert rec["decision"]["execution_eligible"] is False
    assert rec["assessment"]["verdict"] == "watch"
    # Objects: scenarios, grouped evidence, D1 agreement.
    scen = rec["decision"]["competing_scenarios"]
    assert "observed, not routed" in scen[0]["status"]
    assert "not funded" in scen[1]["status"]
    a = rec["assessment"]
    assert a["supporting_evidence"]
    assert any("signal factor" in t
               for t in a["supporting_evidence"])
    assert any("with structural D1 bias +1" in t
               for t in a["supporting_evidence"])
    assert "fvg_midline_signal" in \
        a["independent_evidence_groups"]
    text = rec["assessment_text"]
    assert "Independent evidence groups" in text
    assert "take the +1 fvg_midline setup at 2000.00" in text


def test_sweep_candidate_range_shadow_objects(tmp_path):
    runner, _h1 = _sweep_runner(tmp_path)
    runner._live_setup = {"bias_d1": -1, "bias_h4": -1,
                          "regime": "range", "size_mult": 1.0}
    frame = _flat_h1(40, 2000.0)
    spec = SimpleNamespace(
        name="range_qm", route="range_h4_h1",
        playbook_class=lambda: _StubPB(
            "range_qm", _sig(index=len(frame) - 1)))
    runner._sweep_eval_one(
        spec, frame, clock="h4_generation", intended_tf="H4",
        actual_tf="H4", source="live_close", shadow=True,
        extra_evidence={
            "stage": "h4_generation", "d1_bias_context": -1,
            "d1_role": "context only — recorded, never "
                       "consulted as permission (owner §G: D1 "
                       "never vetoes)"})
    recs = [r for r in _evals(runner, "range_qm")
            if r["outcome"] == "candidate_emitted"]
    assert recs
    rec = recs[-1]
    assert is_shadow_record(rec)
    assert rec["decision"]["result"] == "shadow"
    scen = rec["decision"]["competing_scenarios"]
    assert "§G shadow" in scen[0]["status"]
    # D1 disagreement is weighed CONTEXT on the §G routes, and
    # the item says so — it is never dressed as a veto.
    con = rec["assessment"]["contradictory_evidence"]
    assert any("against structural D1 bias -1" in t
               and "context only" in t for t in con)
    text = rec["assessment_text"]
    assert "SHADOW EVALUATION" in text
    assert "VERDICT: SIGNAL" not in text


def test_range_no_setup_shadow_carries_context(tmp_path):
    runner, _h1 = _sweep_runner(tmp_path)
    runner._run_range_family("h4_generation", {},
                             source="live_close")
    recs = [r for r in _evals(runner, "range_qm")
            if r["outcome"] == "no_setup"]
    assert recs
    rec = recs[-1]
    assert is_shadow_record(rec)
    a = rec["assessment"]
    assert a["htf_context"].get("d1_bias_context") == 1
    assert a["structure_read"].get("regime") == "trend"
    scen = a["competing_scenarios"]
    assert isinstance(scen, list)
    assert scen[0]["status"].startswith("not formed")


def test_not_evaluated_sweep_record_keeps_absent_marking(
        tmp_path):
    """The scoping rule: context objects attach to EVALUATED
    records only.  A kill-switch sweep record — no evaluation
    ran — keeps the Amendment-3 absent marking verbatim."""
    runner = _qm_runner(tmp_path,
                        disabled_playbooks=("narrative",))
    h1 = _flat_h1(60, 95.0)
    runner._stack = SimpleNamespace(
        frames={"H1": h1,
                "H4": _frame([(95.0, 95.5, 94.5, 95.0)] * 60,
                             freq="4h")})
    runner._last_tick = _fresh_tick(h1)
    runner._run_roster_close_evals({}, source="live_close")
    recs = _evals(runner, "narrative")
    assert recs
    a = recs[0]["assessment"]
    assert a["structure_read"]["status"] == "absent"
    assert a["competing_scenarios"]["status"] == "not_produced"
    assert "evidence" in a["capability_markers"]
    assert render_assessment(recs[0], symbol="XAUUSD")
