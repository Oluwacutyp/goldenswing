"""Judgment core (owner rulings 2026-10-23; build 2026-11-02):
market-state estimation -> competing scenarios -> evidence
evaluation -> decision/abstention, SHADOW-FIRST.

Two layers pinned here:
- the pure estimator (goldenswing/execution/judgment.py) on
  hand-built funnel-record shapes — probabilities, the
  correlation-group voting rule, opposition as weighed evidence
  (never a veto inside the model), and every abstention shape;
- the runner wiring — a candidate-bearing live funnel record
  gains a judgment_shadow record, warmup/non-candidate records
  do not, the flag turns it off, and the evaluation record
  itself is byte-identical with shadow on or off (the verdict
  can never steer the path it observes).
"""
import json

import pytest

from goldenswing.execution.judgment import (
    JudgmentConfig, evaluate_judgment)
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig


# ------------------------------------------------------------ fixtures
def _candidate(**kw):
    base = {"direction": 1, "entry": 3900.0, "stop": 3890.0,
            "targets": [3920.0], "score": 72.0,
            "asymmetry_score": 72.0, "sleeve_base_score": 60.0,
            "score_components": None, "factors": [], "index": 5}
    base.update(kw)
    return base


def _evidence(*, supporting=(), contradictory=(), groups=None,
              observed=None, inferred=None):
    return {
        "supporting": list(supporting),
        "contradictory": list(contradictory),
        "evidence_groups": groups or {},
        "independent_evidence_groups": sorted(groups or {}),
        "observed": observed or {
            "entry": 3900.0, "stop": 3890.0, "stop_dist": 10.0,
            "bias_d1": 1, "bias_h4": 1,
            "session_bucket": "overlap"},
        "inferred": inferred or {
            "score_total": 72.0, "score_flipped": False,
            "regime": "clean_trend", "posture": "TREND_PULLBACK",
            "journey_state": "with_journey", "room_r": 2.0},
    }


def _decision(**kw):
    base = {"result": "accepted", "reason_code": None,
            "order_result": "filled",
            "floor_result": {"floor": 55.0, "total": 72.0,
                             "path": "INTRADAY_CONTINUATION"}}
    base.update(kw)
    return base


_TREND_CTX = {"regime": "up/clean_trend/normal/accumulation/"
                        "overlap",
              "posture": "TREND_PULLBACK", "day_type": "trend",
              "adr_pct": 40.0, "adr_state": "open", "session": "overlap",
              "smt": None}


# ------------------------------------------------------------ pure core
def test_scenarios_are_a_probability_distribution():
    out = evaluate_judgment(candidate=_candidate(),
                            decision=_decision(),
                            evidence=_evidence(), context=_TREND_CTX)
    assert [s["name"] for s in out["scenarios"]] == [
        "continuation", "reversal", "chop"]
    total = sum(s["p"] for s in out["scenarios"])
    assert total == pytest.approx(1.0, abs=0.002)
    assert out["mode"] == "shadow"


def test_invalid_geometry_abstains_as_category_3():
    out = evaluate_judgment(
        candidate=_candidate(stop=3900.0), decision=_decision(),
        evidence=_evidence(), context=_TREND_CTX)
    assert out["decision"] == "abstain"
    assert out["reason_category"] == 3
    assert out["expected_r"] is None
    assert out["scenarios"] == []


def test_aligned_evidence_takes():
    sup = ["with structural D1 bias +1", "asymmetry score 72 "
           "for the +1 side", "zone context: zone [3888.0, 3902.0]"]
    groups = {"htf_structure": [sup[0]],
              "asymmetry_score": [sup[1]],
              "zone_context": [sup[2]]}
    out = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(supporting=sup, groups=groups),
        context=_TREND_CTX)
    assert out["decision"] == "take"
    p = {s["name"]: s["p"] for s in out["scenarios"]}
    assert p["continuation"] > p["reversal"]
    assert out["expected_r"] > 0.05
    assert out["evidence_balance"]["groups_for"] == [
        "htf_structure", "asymmetry_score", "zone_context"]


def test_opposition_funds_reversal_and_abstains_category_5():
    con = ["against structural D1 bias -1 (weighed as evidence, "
           "not a veto)"]
    groups = {"htf_structure": con}
    ev = _evidence(
        contradictory=con, groups=groups,
        observed={"entry": 3900.0, "stop": 3890.0,
                  "stop_dist": 10.0, "bias_d1": -1,
                  "bias_h4": -1, "session_bucket": "overlap"},
        inferred={"score_total": 40.0, "score_flipped": True,
                  "regime": "clean_trend", "posture": None,
                  "journey_state": None, "room_r": 2.0})
    out = evaluate_judgment(
        candidate=_candidate(), decision=_decision(
            floor_result={"floor": 55.0, "total": 40.0,
                          "path": "INTRADAY_CONTINUATION"}),
        evidence=ev, context=_TREND_CTX)
    assert out["decision"] == "abstain"
    p = {s["name"]: s["p"] for s in out["scenarios"]}
    assert p["reversal"] > p["continuation"]
    # D1 opposition + flipped score: directional disagreement.
    assert out["opposition"] >= 2.0
    assert out["reason_category"] == 5


def test_correlated_group_votes_once():
    one_group = _evidence(
        supporting=["factor a", "factor b"],
        groups={"signal_family": ["factor a", "factor b"]})
    two_groups = _evidence(
        supporting=["factor a", "factor b"],
        groups={"signal_family": ["factor a"],
                "other_family": ["factor b"]})
    a = evaluate_judgment(candidate=_candidate(),
                          decision=_decision(),
                          evidence=one_group, context=_TREND_CTX)
    b = evaluate_judgment(candidate=_candidate(),
                          decision=_decision(),
                          evidence=two_groups, context=_TREND_CTX)
    assert a["evidence_balance"]["support"] == 1.0
    assert b["evidence_balance"]["support"] == 2.0
    pa = {s["name"]: s["p"] for s in a["scenarios"]}
    pb = {s["name"]: s["p"] for s in b["scenarios"]}
    assert pb["continuation"] > pa["continuation"]


def test_thin_evidence_abstains_on_confidence():
    out = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(inferred={
            "score_total": None, "score_flipped": False,
            "regime": None, "posture": None,
            "journey_state": None, "room_r": None}),
        context={})
    assert out["decision"] == "abstain"
    assert out["confidence"] < 0.15
    assert "too thin" in out["reason"]


def test_range_state_funds_chop_category_6():
    ctx = {"regime": "flat/range/normal/accumulation/asia",
           "posture": None, "day_type": "range", "adr_pct": 30.0,
           "adr_state": "open", "session": "asia", "smt": None}
    out = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(observed={
            "entry": 3900.0, "stop": 3890.0, "stop_dist": 10.0,
            "bias_d1": 0, "bias_h4": 0,
            "session_bucket": "asia"}),
        context=ctx)
    p = {s["name"]: s["p"] for s in out["scenarios"]}
    assert p["chop"] > p["continuation"]
    assert p["chop"] > p["reversal"]
    assert out["decision"] == "abstain"
    assert out["reason_category"] == 6
    assert out["market_state"]["structure"] == "range"


def test_deterministic_same_record_same_verdict():
    kw = dict(candidate=_candidate(), decision=_decision(),
              evidence=_evidence(), context=_TREND_CTX)
    assert evaluate_judgment(**kw) == evaluate_judgment(**kw)


def test_custom_thresholds_are_respected():
    sup = ["with structural D1 bias +1"]
    groups = {"htf_structure": sup}
    kw = dict(candidate=_candidate(), decision=_decision(),
              evidence=_evidence(supporting=sup, groups=groups),
              context=_TREND_CTX)
    base = evaluate_judgment(**kw)
    strict = evaluate_judgment(
        config=JudgmentConfig(min_expected_r=99.0), **kw)
    assert base["expected_r"] < 99.0
    assert strict["decision"] == "abstain"


# ------------------------------------------------------------ wiring
def _runner(tmp_path, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False, **cfg_kw)
    return MT5Runner(cfg)


def _journal_records(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


def _emit_candidate(runner):
    return runner._emit_eval(
        playbook="qm_m5", evaluation_clock="h1_trigger",
        intended_tf="M5", outcome="candidate_emitted",
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence())


def test_shadow_record_journaled_beside_evaluation(tmp_path):
    # The passive tap under SHADOW mode (the pre-promotion
    # posture; since the 2026-11-05 promotion the config default
    # is "active", so the mode is stated explicitly here).
    runner = _runner(tmp_path, judgment_mode="shadow")
    _emit_candidate(runner)
    records = _journal_records(tmp_path)
    kinds = [r["type"] for r in records]
    assert "evaluation" in kinds
    shadow = [r for r in records if r["type"] == "judgment_shadow"]
    assert len(shadow) == 1
    rec = shadow[0]
    assert rec["playbook"] == "qm_m5"
    assert rec["gate_result"] == "accepted"
    assert rec["gate_order_result"] == "filled"
    assert rec["judgment"]["mode"] == "shadow"
    assert rec["judgment"]["decision"] in ("take", "abstain")
    assert len(rec["judgment"]["scenarios"]) == 3


def test_no_candidate_no_shadow(tmp_path):
    runner = _runner(tmp_path)
    runner._emit_eval(playbook="qm_m5",
                      evaluation_clock="h1_trigger",
                      outcome="no_setup")
    assert [r for r in _journal_records(tmp_path)
            if r["type"] == "judgment_shadow"] == []


def test_warmup_records_get_no_shadow(tmp_path):
    runner = _runner(tmp_path)
    runner._emit_eval(
        playbook="qm_m5", evaluation_clock="h1_trigger",
        intended_tf="M5", outcome="candidate_emitted",
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(), source="warmup_scan")
    assert [r for r in _journal_records(tmp_path)
            if r["type"] == "judgment_shadow"] == []


def test_shadow_flag_off_and_evaluation_record_identical(tmp_path):
    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()
    _emit_candidate(_runner(on_dir))
    _emit_candidate(_runner(off_dir, judgment_shadow_enabled=False))
    on_records = _journal_records(on_dir)
    off_records = _journal_records(off_dir)
    assert [r for r in off_records
            if r["type"] == "judgment_shadow"] == []
    # Time-bearing fields legitimately differ between two runs
    # (assessment_text embeds the evaluation instant in prose);
    # everything else must be identical — the shadow verdict
    # cannot steer the record it observes.
    strip = ("ts", "run_id", "commit", "config_fingerprint",
             "assessment_text")
    on_eval = [r for r in on_records if r["type"] == "evaluation"]
    off_eval = [r for r in off_records if r["type"] == "evaluation"]
    assert len(on_eval) == len(off_eval) == 1
    for key in strip:
        on_eval[0].pop(key, None)
        off_eval[0].pop(key, None)
    assert on_eval[0] == off_eval[0]
