"""Completion-05 — the unified self-audit.

The REAL cycle (execution/self_audit.audit_cycle, the same
function the runner's S3-cadence phase calls) driven over
fabricated journals, one fixture per category:

- a missed opportunity (candidate died at the floor with its
  reason named; an armed setup expired; a material bar nobody
  evaluated);
- wrong judgments graded by the S4 pairing — a take that
  lost, an abstain that would have won, a degenerate-R pair
  excluded from grading (the +226R shape), and a pair that
  only resolves across two windows;
- a weak setup (executed on a thin evidence vector, absent
  groups named from the journaled judgment block);
- a premature exit measured from stored bars (target first
  touched after the exit) + the unmeasurable shapes (no
  target on record; no bars covering the window) + the
  not-premature shapes (target touched while held; partial
  banked by design);
- execution failures (a friction-gate refusal at the order
  path, a fill that differs materially from intent, an
  unfilled/unaccounted candidate decision);
- idempotency (watermark: a second read/cycle over the same
  window adds nothing);
- corrupt-region degradation (a damaged window reads every
  category unmeasurable — counts withheld, never cleaned up
  on paper; a torn final line is held, not damage);
- "none found" is a MEASURED zero, distinct from
  unmeasurable;
- runner wiring: the cycle journals one self_audit record
  per evidence-bearing H1 cycle and persists standing state,
  /status renders it, the manifest carries it, and the
  decision path is unchanged — the judgment block inside a
  shadow record is identical with audit feedback seeded and
  without; only the record gains an audit_feedback citation.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from goldenswing.execution import self_audit
from goldenswing.execution.self_audit import (
    audit_cycle, new_state, read_window, route_feedback)

T0 = "2026-11-03T00:00:00Z"


def _now(s: str):
    return pd.Timestamp(s)


def _evaluation(playbook="qm_m5", decision=None, outcome="candidate_emitted",
                kind="playbook_eval", **kw):
    rec = {"type": "evaluation", "kind": kind, "ts": T0,
           "playbook": playbook, "outcome": outcome,
           "candidate": {"direction": 1, "entry": 3900.0,
                         "stop": 3890.0, "targets": [3950.0]},
           "decision": decision}
    rec.update(kw)
    return rec


def _shadow(ts=T0, playbook="qm_m5", decision="take",
            order_result="filled", groups_for=("htf_structure",),
            groups_against=(), score=None, confidence=0.5,
            expected_r=0.4, gate_reason_code=None):
    return {
        "type": "judgment_shadow", "ts": ts, "playbook": playbook,
        "gate_result": "accepted", "gate_order_result": order_result,
        "gate_reason_code": gate_reason_code,
        "judgment": {
            "decision": decision, "expected_r": expected_r,
            "confidence": confidence,
            "evidence_balance": {
                "support": 1.5, "against": 0.0,
                "groups_for": list(groups_for),
                "groups_against": list(groups_against)},
            "score": {"total": score, "floor": 50.0,
                      "edge": None},
            "thresholds": {"min_expected_r": 0.05,
                           "min_confidence": 0.15}}}


def _recon(ts=T0, playbook="qm_m5", realized_r=-1.2, move=-12.0,
           entry=3900.0):
    return {"type": "judgment_reconciliation", "judged_ts": ts,
            "playbook": playbook, "horizon_h": 4,
            "realized_r": realized_r, "move": move,
            "entry": entry, "hit": realized_r > 0}


def _bars(spec):
    """spec: list of (hour_offset, high, low) from T0."""
    base = pd.Timestamp(T0)
    return [{"ts": str(base + pd.Timedelta(hours=h)),
             "high": hi, "low": lo} for h, hi, lo in spec]


# ------------------------------------------------------- (a) missed
def test_missed_opportunities_carry_the_reason_they_died():
    died = _evaluation(decision={
        "result": "refused", "reason_code": "floor.total",
        "reason_category": 7})
    expired = _evaluation(
        kind="trigger_event", outcome="not_eligible",
        trigger_stage="expired",
        reason_code="path.binding_all_expired")
    bar_miss = {
        "type": "judgment_audit_event", "kind": "missed_opportunity",
        "bar": "2026-11-03T05:00:00", "direction": -1,
        "move_atr": 1.8,
        "reason_code": "audit.no_evaluation_on_material_bar"}
    st = new_state()
    rec = audit_cycle(state=st, records=[died, expired, bar_miss],
                      bars=[], now=_now("2026-11-03T06:00:00Z"),
                      bar="2026-11-03T06:00:00")
    cat = rec["categories"]["missed_opportunities"]
    assert cat["state"] == "measured"
    assert cat["found"] == 3
    reasons = {(i["playbook"], i["reason_code"])
               for i in cat["items"]}
    assert ("qm_m5", "floor.total") in reasons
    assert ("__trigger__", "path.binding_all_expired") in reasons
    assert cat["bar_misses"][0]["move_atr"] == 1.8
    assert st["totals"]["missed_opportunities"]["found"] == 3


def test_missed_ordered_is_not_a_miss_decisionless_is_unmeasurable():
    ordered = _evaluation(decision={
        "result": "accepted", "order_result": "filled",
        "reason_category": 10})
    decisionless = _evaluation(decision=None)
    st = new_state()
    rec = audit_cycle(state=st, records=[ordered, decisionless],
                      bars=[], now=_now("2026-11-03T06:00:00Z"))
    cat = rec["categories"]["missed_opportunities"]
    assert cat["found"] == 0
    assert cat["unmeasurable"] == 1


# ------------------------------------------------------- (b) wrong judgments
def test_wrong_judgments_graded_by_s4_pairing():
    st = new_state()
    recs = [
        _shadow(decision="take"),
        _recon(realized_r=-1.2),                       # take lost
        _shadow(ts="2026-11-03T01:00:00Z", decision="abstain"),
        _recon(ts="2026-11-03T01:00:00Z", realized_r=0.8,
               move=8.0),                              # abstain won
        _shadow(ts="2026-11-03T02:00:00Z", decision="take"),
        _recon(ts="2026-11-03T02:00:00Z", realized_r=1.5,
               move=15.0),                             # take won
    ]
    rec = audit_cycle(state=st, records=recs, bars=[],
                      now=_now("2026-11-03T08:00:00Z"))
    cat = rec["categories"]["wrong_judgments"]
    assert cat["found"] == 2
    kinds = {(i["decision"], i["realized_r"]) for i in cat["items"]}
    assert ("take", -1.2) in kinds
    assert ("abstain", 0.8) in kinds
    fb = route_feedback(st)["qm_m5"]
    assert fb["takes"] == 2 and fb["wrong_take_rate"] == 0.5
    assert fb["abstains"] == 1 and fb["wrong_abstain_rate"] == 1.0


def test_degenerate_pair_is_excluded_never_graded():
    st = new_state()
    # The evidence-pack poison shape: ~$0.135 stop, +226R.
    rec = audit_cycle(
        state=st,
        records=[_shadow(), _recon(realized_r=226.0, move=30.0,
                                   entry=4100.0)],
        bars=[], now=_now("2026-11-03T08:00:00Z"))
    cat = rec["categories"]["wrong_judgments"]
    assert cat["found"] == 0
    assert cat["excluded_degenerate_r"] == 1
    assert st["wrong_judgments_excluded_degenerate_r"] == 1
    # Never entered route grading: no takes, no wrong-take rate.
    fb = route_feedback(st).get("qm_m5", {})
    assert fb.get("takes", 0) == 0
    assert fb.get("wrong_take_rate") is None


def test_pair_resolves_across_windows_and_rerun_adds_nothing():
    st = new_state()
    first = audit_cycle(state=st, records=[_shadow()], bars=[],
                        now=_now("2026-11-03T02:00:00Z"))
    assert first["categories"]["wrong_judgments"]["pending"] == 1
    assert first["categories"]["wrong_judgments"]["found"] == 0
    second = audit_cycle(state=st, records=[_recon()], bars=[],
                         now=_now("2026-11-03T06:00:00Z"))
    assert second["categories"]["wrong_judgments"]["found"] == 1
    totals = {c: dict(t) for c, t in st["totals"].items()}
    third = audit_cycle(state=st, records=[], bars=[],
                        now=_now("2026-11-03T07:00:00Z"))
    assert third is None                      # nothing new, no record
    assert st["totals"] == totals             # and nothing re-counted


# ------------------------------------------------------- (c) weak setups
def test_weak_setup_names_the_groups_that_never_voted():
    st = new_state()
    weak = _shadow(groups_for=("htf_structure",), score=None)
    strong = _shadow(
        ts="2026-11-03T01:00:00Z",
        groups_for=("htf_structure", "journey_geometry",
                    "zone_context"),
        score=80.0)
    not_an_attempt = _shadow(
        ts="2026-11-03T02:00:00Z", order_result=None,
        gate_reason_code="floor.total",
        groups_for=("htf_structure",), score=None)
    rec = audit_cycle(state=st,
                      records=[weak, strong, not_an_attempt],
                      bars=[], now=_now("2026-11-03T06:00:00Z"))
    cat = rec["categories"]["weak_setups"]
    assert cat["found"] == 1
    item = cat["items"][0]
    assert item["engaged_groups"] == ["htf_structure"]
    assert "asymmetry_score" in item["absent_groups"]
    assert "journey_geometry" in item["absent_groups"]
    fb = route_feedback(st)["qm_m5"]
    assert fb["attempts"] == 2 and fb["weak_setup_rate"] == 0.5


# ------------------------------------------------------- (d) premature exits
def _trade_records(partial=False, with_signal=True):
    recs = []
    if with_signal:
        recs.append({"type": "signal", "signal_id": "a1",
                     "ts": T0, "playbook": "qm_m5", "direction": 1,
                     "targets": [3950.0]})
    recs.append({"type": "open", "alert_id": "a1", "ts": T0,
                 "ticket": 4242, "playbook": "qm_m5",
                 "direction": 1, "entry": 3900.0, "stop": 3890.0})
    recs.append({"type": "close", "alert_id": "a1", "ts":
                 "2026-11-03T02:00:00Z", "ticket": 4242,
                 "exit": 3920.0, "reason": "trail",
                 "pnl_money": 20.0, "r_net": 0.4,
                 "partial_taken": partial})
    return recs


def test_premature_exit_measured_from_stored_bars():
    st = new_state()
    first = audit_cycle(state=st, records=_trade_records(),
                        bars=[], now=_now("2026-11-03T02:00:00Z"))
    assert first["categories"]["premature_exits"]["found"] == 0
    assert first["categories"]["premature_exits"]["pending"] == 1
    bars = _bars([(0, 3930.0, 3895.0), (1, 3935.0, 3905.0),
                  (2, 3940.0, 3910.0), (3, 3945.0, 3915.0),
                  (4, 3955.0, 3920.0), (5, 3948.0, 3918.0)])
    second = audit_cycle(state=st, records=[], bars=bars,
                         now=_now("2026-11-03T07:00:00Z"))
    cat = second["categories"]["premature_exits"]
    assert cat["found"] == 1
    item = cat["items"][0]
    assert item["ticket"] == 4242
    assert item["target"] == 3950.0
    assert item["minutes_to_target_after_exit"] == 120


def test_premature_exit_without_target_is_unmeasurable():
    st = new_state()
    rec = audit_cycle(state=st,
                      records=_trade_records(with_signal=False),
                      bars=[], now=_now("2026-11-03T02:00:00Z"))
    cat = rec["categories"]["premature_exits"]
    assert cat["found"] == 0 and cat["unmeasurable"] == 1


def test_premature_exit_without_bars_is_unmeasurable_not_estimated():
    st = new_state()
    audit_cycle(state=st, records=_trade_records(), bars=[],
                now=_now("2026-11-03T02:00:00Z"))
    rec = audit_cycle(state=st, records=[], bars=[],
                      now=_now("2026-11-03T07:00:00Z"))
    cat = rec["categories"]["premature_exits"]
    assert cat["found"] == 0 and cat["unmeasurable"] == 1
    assert cat["pending"] == 0


def test_exit_not_premature_when_target_touched_while_held():
    st = new_state()
    audit_cycle(state=st, records=_trade_records(), bars=[],
                now=_now("2026-11-03T02:00:00Z"))
    bars = _bars([(0, 3930.0, 3895.0), (1, 3955.0, 3905.0),
                  (2, 3940.0, 3910.0), (3, 3945.0, 3915.0),
                  (4, 3948.0, 3920.0), (5, 3948.0, 3918.0)])
    audit_cycle(state=st, records=[], bars=bars,
                now=_now("2026-11-03T07:00:00Z"))
    # Measured (the exit left the pending set) and NOT flagged:
    # the target was reached while the position was held.
    assert st["pending_exits"] == []
    assert st["totals"]["premature_exits"]["found"] == 0


def test_partial_taken_exit_is_never_flagged_premature():
    st = new_state()
    audit_cycle(state=st, records=_trade_records(partial=True),
                bars=[], now=_now("2026-11-03T02:00:00Z"))
    bars = _bars([(0, 3930.0, 3895.0), (1, 3935.0, 3905.0),
                  (2, 3940.0, 3910.0), (3, 3945.0, 3915.0),
                  (4, 3955.0, 3920.0), (5, 3948.0, 3918.0)])
    audit_cycle(state=st, records=[], bars=bars,
                now=_now("2026-11-03T07:00:00Z"))
    assert st["pending_exits"] == []
    assert st["totals"]["premature_exits"]["found"] == 0


# ------------------------------------------------------- (e) execution failures
def test_execution_failures_three_shapes():
    friction = _evaluation(decision={
        "result": "refused", "reason_code": "gate.friction",
        "reason_category": 8,
        "order_result": "refused_in_order_path"})
    unfilled = _evaluation(decision={"result": "accepted"})
    slipped = {"type": "open", "alert_id": "a9", "ts": T0,
               "ticket": 777, "playbook": "qm_m5",
               "intended_entry": 3900.0, "entry": 3903.0,
               "stop_distance": 10.0, "slippage": 3.0}
    st = new_state()
    rec = audit_cycle(state=st,
                      records=[friction, unfilled, slipped],
                      bars=[], now=_now("2026-11-03T06:00:00Z"))
    cat = rec["categories"]["execution_failures"]
    assert cat["found"] == 3
    kinds = {i["kind"] for i in cat["items"]}
    assert kinds == {"refused_at_execution_boundary",
                     "unfilled_unaccounted",
                     "fill_differs_from_intent"}
    fill = next(i for i in cat["items"]
                if i["kind"] == "fill_differs_from_intent")
    assert fill["slippage_frac_of_stop"] == 0.3


# ------------------------------------------------------- honesty mechanics
def test_watermark_idempotency_and_torn_tail(tmp_path):
    path = tmp_path / "j.jsonl"
    import json as _json
    line1 = _json.dumps({"type": "note", "text": "x"})
    path.write_text(line1 + "\n" + '{"type": "open", "alert_id": "a1"')
    records, offset, damaged = read_window(path, 0)
    assert damaged == 0                      # torn tail is held
    assert len(records) == 1
    assert offset == len(line1) + 1
    with path.open("a") as fh:
        fh.write('}\n')
    records2, offset2, damaged2 = read_window(path, offset)
    assert damaged2 == 0 and len(records2) == 1
    records3, _, _ = read_window(path, offset2)
    assert records3 == []                    # nothing re-read


def test_corrupt_region_degrades_every_category(tmp_path):
    path = tmp_path / "j.jsonl"
    import json as _json
    good = _json.dumps({"type": "note", "text": "x"})
    path.write_text(good + "\n" + "not json{{{\n" + good + "\n")
    records, _offset, damaged = read_window(path, 0)
    assert damaged == 1 and len(records) == 2
    st = new_state()
    rec = audit_cycle(state=st, records=records, bars=[],
                      now=_now("2026-11-03T06:00:00Z"),
                      damaged_lines=damaged)
    assert rec["journal_damage"] == {"damaged_lines": 1}
    for cat in self_audit.CATEGORIES:
        body = rec["categories"][cat]
        assert body["state"] == "unmeasurable"
        assert body["found"] is None      # withheld, not zeroed
    assert st["damaged_cycles"] == 1
    assert all(t["found"] == 0 for t in st["totals"].values())


def test_none_found_is_a_measured_zero():
    st = new_state()
    quiet = _evaluation(outcome="no_setup", decision=None)
    strong = _shadow(groups_for=("htf_structure",
                                  "journey_geometry",
                                  "zone_context"),
                     score=80.0)
    rec = audit_cycle(state=st, records=[quiet, strong], bars=[],
                      now=_now("2026-11-03T06:00:00Z"))
    for cat in self_audit.CATEGORIES:
        body = rec["categories"][cat]
        assert body["state"] == "measured"
    assert rec["categories"]["missed_opportunities"]["found"] == 0
    assert rec["categories"]["weak_setups"]["found"] == 0


# ------------------------------------------------------- runner wiring
def _h1(closes, opens=None):
    idx = pd.date_range("2026-11-03T00:00:00Z",
                        periods=len(closes), freq="h")
    opens = opens or closes
    return pd.DataFrame(
        {"open": opens,
         "high": [max(o, c) + 1 for o, c in zip(opens, closes)],
         "low": [min(o, c) - 1 for o, c in zip(opens, closes)],
         "close": closes, "volume": [100] * len(closes)},
        index=idx)


def _runner(tmp_path):
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
    from mt5_fakes import FakeMT5

    feed = MT5Feed(FakeMT5(), MT5FeedConfig(symbol="XAUUSDm"))
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False, max_daily_trades=0,
        max_hourly_trades=0)
    return MT5Runner(cfg, feed=feed)


def test_runner_cycle_journals_once_and_renders(tmp_path):
    runner = _runner(tmp_path)
    died = _evaluation(decision={
        "result": "refused", "reason_code": "floor.total",
        "reason_category": 7})
    weak = _shadow(groups_for=("htf_structure",), score=None)
    runner.journal.append(died)
    runner.journal.append(weak)
    closes = [3900.0] * 20
    runner._stack = SimpleNamespace(frames={"H1": _h1(closes)})
    runner._judgment_audit_cycle()
    records = runner.journal.records()
    audits = [r for r in records if r.get("type") == "self_audit"]
    assert len(audits) == 1
    cats = audits[0]["categories"]
    assert cats["missed_opportunities"]["found"] == 1
    assert cats["weak_setups"]["found"] == 1
    st = runner._self_audit_state()
    assert st["totals"]["missed_opportunities"]["found"] == 1
    assert st["cycles"] == 1
    line = runner.self_audit_line()
    assert "missed opportunities: 1 last cycle" in line
    assert "weak setups (thin evidence): 1 last cycle" in line
    # A fresh bar with no new journal evidence: no second record.
    runner._stack = SimpleNamespace(
        frames={"H1": _h1(closes + [3900.0])})
    runner._judgment_audit_cycle()
    audits = [r for r in runner.journal.records()
              if r.get("type") == "self_audit"]
    assert len(audits) == 1
    assert runner._self_audit_state()["cycles"] == 1


def test_manifest_and_feedback_annotation_parity(tmp_path):
    runner = _runner(tmp_path)
    # No graded history yet: no annotation is produced.
    assert runner._audit_feedback_for("qm_m5") is None
    sub = runner._self_audit_state()
    sub["cycles"] = 2
    sub["routes"]["qm_m5"] = {
        "takes": 4, "take_losses": 3, "abstains": 2,
        "abstain_wins": 1, "attempts": 5, "weak": 1}
    fb = runner._audit_feedback_for("qm_m5")
    assert fb["source"] == "self_audit"
    assert fb["wrong_take_rate"] == 0.75
    assert fb["weak_setup_rate"] == 0.2
    block = runner._self_audit_manifest_block()
    assert block["cycles"] == 2
    assert block["routes"]["qm_m5"]["wrong_take_rate"] == 0.75
    # Decision parity: the core's verdict on fixed inputs is
    # the same object with the audit state seeded or empty —
    # the feedback is attached to records, never fed in.
    from goldenswing.execution.judgment import evaluate_judgment
    candidate = {"direction": 1, "entry": 3900.0, "stop": 3890.0,
                 "targets": [3950.0]}
    decision = {"result": "accepted",
                "floor_result": {"total": 80.0, "floor": 50.0}}
    evidence = {"evidence_groups": {
                    "htf_structure": ["bullish structure"]},
                "supporting": ["bullish structure"],
                "contradictory": [],
                "observed": {"bias_d1": 1, "bias_h4": 1},
                "inferred": {}}
    before = evaluate_judgment(candidate=candidate,
                               decision=decision,
                               evidence=evidence, context={})
    sub["routes"].clear()
    after = evaluate_judgment(candidate=candidate,
                              decision=decision,
                              evidence=evidence, context={})
    assert before == after
