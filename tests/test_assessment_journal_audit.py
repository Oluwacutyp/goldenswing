"""Journal auditability (owner, 2026-10-26).

The 2026-10-26 assessment read exposed records that misled
without lying: "evaluated not recorded" timestamps, absent
freshness measurements, and observed candidates whose only
refusal explanation was an outcome code.  These tests pin
the fix at every layer:

- journal_audit pure helpers (freshness math, source
  liveness, detector facts, trigger-decision block);
- the sweep journal itself (every warmup record carries a
  parseable evaluation timestamp, its source, and a
  freshness measurement; the renderer banners non-live
  sources and prints the Trigger decision section);
- inversion determinism + no future leak (the playbook
  whose assessment WATCH motivated the trigger-decision
  work: two fresh instances agree, and bars past the
  evaluation bar never change the signal set);
- the range family's causal H4 slice (only H4 bars closed
  by the H1 bar feed the detector);
- the evidence map's coverage (keys == registry, census
  evidence-only by construction, the three UNCERTAIN flags
  stated, not silently resolved).
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from goldenswing.execution import journal_audit
from goldenswing.execution.assessment_render import (
    freshness_lines, render_assessment, trigger_decision_lines)
from goldenswing.execution.evidence_map import (
    CENSUS_EVIDENCE_ONLY, PLAYBOOK_EVIDENCE, SHARED_DECISION_INPUTS,
    SHARED_MANAGEMENT_ONLY, evidence_map)
from goldenswing.execution.playbook_registry import PLAYBOOK_NAMES
from goldenswing.playbooks.inversion import InversionPlaybook
from goldenswing.playbooks.range_regime import (
    RangeSweepReclaimPlaybook)

from test_funnel_m5 import _qm_runner
from test_p3_trader import _StubPB, _flat_h1, _frame, _sig
from test_trigger_assessment import _evals, _sweep_runner


# ---------------------------------------------------------- pure helpers

def test_freshness_seconds_h1_and_m5_math():
    assert journal_audit.freshness_seconds(
        "2026-01-07 12:30", "2026-01-07 11:00", "H1") == 1800.0
    assert journal_audit.freshness_seconds(
        "2026-01-07 11:57", "2026-01-07 11:50", "M5") == 120.0


def test_freshness_seconds_before_close_is_zero_not_negative():
    assert journal_audit.freshness_seconds(
        "2026-01-07 11:30", "2026-01-07 11:00", "H1") == 0.0


def test_freshness_seconds_unmeasurable_is_none_never_zero():
    assert journal_audit.freshness_seconds(
        "2026-01-07 12:30", "2026-01-07 11:00", "H9") is None
    assert journal_audit.freshness_seconds(
        "2026-01-07 12:30", None, "H1") is None
    assert journal_audit.freshness_seconds(
        None, "2026-01-07 11:00", "H1") is None


def test_is_live_decision_source_across_all_sources():
    assert journal_audit.is_live_decision_source("live_close") is True
    assert journal_audit.is_live_decision_source("trigger_path") is True
    for src in ("warmup_scan", "replay", "backfill", "", None):
        assert journal_audit.is_live_decision_source(src) is False


def test_source_note_only_for_non_live_sources():
    note = journal_audit.source_note("warmup_scan")
    assert note and "analytical only" in note
    assert journal_audit.source_note("live_close") is None


def test_detector_facts_scalar_twins_and_nested_dropped():
    frame = _frame([(100.0, 101.0, 99.0, 100.0)] * 30)
    sig = SimpleNamespace(live_context={
        "zone_formed_idx": 15, "inversion_idx": 17,
        "retest_idx": 18, "out_of_range_idx": 99,
        "direction": -1, "entry": 4002.0, "note": "retest",
        "nested": {"a": 1}, "lst": [1, 2]})
    facts = journal_audit.detector_facts(sig, frame)
    assert facts["zone_formed_idx"] == 15
    assert facts["zone_formed_idx_ts"] == str(frame.index[15])
    assert facts["retest_idx_ts"] == str(frame.index[18])
    assert "out_of_range_idx_ts" not in facts
    assert "nested" not in facts and "lst" not in facts
    assert facts["direction"] == -1


def test_detector_facts_empty_without_live_context():
    assert journal_audit.detector_facts(SimpleNamespace()) == {}


def test_trigger_decision_block_fields():
    block = journal_audit.trigger_decision_block(
        route="h1_roster", source="warmup_scan",
        reason_code="path.trigger_not_fired", shadow=False,
        detector_facts={"zone_formed_idx": 15})
    assert block["route"] == "h1_roster"
    assert block["order_authority"].startswith("none")
    assert block["source_is_live_decision"] is False
    assert block["trigger_path_evaluated_by_this_scan"] is False
    assert "does not run the h1_trigger" in block["trigger_path_note"]
    assert block["reason_code"] == "path.trigger_not_fired"
    assert block["detector_facts"]["zone_formed_idx"] == 15


def test_trigger_decision_block_live_close_still_no_authority():
    # The nuance the assessment demanded: a live-close scan is
    # a live decision SOURCE, but this route still holds no
    # order authority — the streaming path does.
    block = journal_audit.trigger_decision_block(
        route="h1_roster", source="live_close",
        reason_code="path.trigger_not_fired", shadow=False,
        detector_facts={})
    assert block["source_is_live_decision"] is True
    assert block["order_authority"].startswith("none")


def test_trigger_decision_block_shadow_authority():
    block = journal_audit.trigger_decision_block(
        route="range_h4_h1", source="warmup_scan",
        reason_code="path.shadow_route", shadow=True,
        detector_facts={})
    assert "shadow" in block["order_authority"]


def test_freshness_lines_absent_measurement_renders_explicitly():
    lines = freshness_lines({"last_bar_ts": "2026-01-07 11:00",
                             "data_freshness_s": None})
    assert any("data freshness: not recorded" in ln for ln in lines)


def test_trigger_decision_lines_absent_without_block():
    assert trigger_decision_lines({}) == []
    assert trigger_decision_lines({"evidence": {}}) == []


# ------------------------------------------------------------- inversion

def _inversion_rows():
    return ([(4000, 4002, 3998, 4000)] * 14
            + [(4000, 4001, 3999, 4000),
               (4000, 4015, 4009, 4014),
               (4014, 4016, 4012, 4015),
               (4000, 4002, 3998, 4000),
               (4000, 4009, 3998, 4005)])


def test_inversion_deterministic_across_fresh_instances():
    frame = _frame(_inversion_rows())
    sigs_a = InversionPlaybook().live_signal([], 18, frame)
    sigs_b = InversionPlaybook().live_signal([], 18, frame)
    assert [(s.direction, s.entry_ref) for s in sigs_a] == \
        [(s.direction, s.entry_ref) for s in sigs_b]
    assert len(sigs_a) == 2
    assert {s.entry_ref for s in sigs_a} == {4001.0, 4002.0}
    assert {s.direction for s in sigs_a} == {-1}
    assert {s.index for s in sigs_a} == {18}
    for sig in sigs_a:
        for key, value in sig.live_context.items():
            if key.endswith("_idx"):
                assert value <= 18


def test_inversion_no_future_leak_from_appended_bars():
    rows = _inversion_rows()
    base = InversionPlaybook().live_signal([], 18, _frame(rows))
    wild = rows + [(4000, 4050, 3990, 4040),
                   (4000, 3900, 3990, 3950)]
    extended = InversionPlaybook().live_signal([], 18, _frame(wild))
    assert [(s.direction, s.entry_ref) for s in base] == \
        [(s.direction, s.entry_ref) for s in extended]


def test_inversion_detector_facts_map_indices_to_timestamps():
    frame = _frame(_inversion_rows())
    sig = InversionPlaybook().live_signal([], 18, frame)[0]
    facts = journal_audit.detector_facts(sig, frame)
    assert facts["zone_formed_idx"] == 15
    assert facts["zone_formed_idx_ts"] == str(frame.index[15])
    assert facts["inversion_idx"] == 17
    assert facts["retest_idx"] == 18
    assert facts["retest_idx_ts"] == str(frame.index[18])


# ------------------------------------------------------------ sweep journal

def _warmup_runner(tmp_path):
    """_sweep_runner shape, but the tick clock is the last H1
    bar + 90 minutes, so H1 freshness is exactly 1800.0s."""
    runner = _qm_runner(tmp_path)
    h1 = _flat_h1(60, 95.0)
    runner._stack = SimpleNamespace(
        frames={"H1": h1,
                "H4": _frame([(95.0, 95.5, 94.5, 95.0)] * 60,
                             freq="4h")})
    runner._live_setup = {"bias_d1": 1, "bias_h4": 1,
                          "regime": "trend",
                          "h4_context": "uptrend structure",
                          "size_mult": 1.0}
    tick_time = h1.index[-1] + pd.Timedelta(minutes=90)
    runner._last_tick = {"mid": 95.0, "bid": 94.9, "ask": 95.1,
                         "spread": 0.2, "time": tick_time}
    return runner, h1


def test_warmup_sweep_records_carry_ts_source_freshness(tmp_path):
    runner, _h1 = _warmup_runner(tmp_path)
    runner._run_roster_close_evals({}, source="warmup_scan")
    recs = _evals(runner)
    assert recs
    for rec in recs:
        pd.Timestamp(rec["ts"])  # parseable, never missing
        assert rec["evaluation_source"] == "warmup_scan"
        assert rec["data_freshness_s"] == 1800.0


def test_warmup_sweep_render_banners_and_freshness(tmp_path):
    runner, _h1 = _warmup_runner(tmp_path)
    runner._run_roster_close_evals({}, source="warmup_scan")
    recs = [r for r in _evals(runner)
            if r["outcome"] == "no_setup" and r.get("assessment")]
    assert recs
    text = render_assessment(recs[0])
    assert "NOT A LIVE DECISION" in text
    assert "source: warmup_scan" in text
    assert "data freshness: 1800.0s" in text


def _candidate_runner(tmp_path):
    runner, _h1 = _sweep_runner(tmp_path)
    frame = _flat_h1(40, 2000.0)
    tick_time = frame.index[-1] + pd.Timedelta(minutes=90)
    runner._last_tick = {"mid": 2000.0, "bid": 1999.9,
                         "ask": 2000.1, "spread": 0.2,
                         "time": tick_time}
    return runner, frame


def _sweep_candidate(runner, frame, source):
    spec = SimpleNamespace(
        name="fvg_midline", route="h1_roster",
        playbook_class=lambda: _StubPB(
            "fvg_midline", _sig(index=len(frame) - 1)))
    runner._sweep_eval_one(
        spec, frame, clock="h1_close", intended_tf="H1",
        actual_tf="H1", source=source, shadow=False)
    recs = [r for r in _evals(runner, "fvg_midline")
            if r["outcome"] == "candidate_emitted"]
    assert recs
    return recs[-1]


def test_sweep_candidate_warmup_carries_trigger_decision(tmp_path):
    runner, frame = _candidate_runner(tmp_path)
    rec = _sweep_candidate(runner, frame, "warmup_scan")
    assert rec["reason_code"] == "path.trigger_not_fired"
    td = rec["evidence"]["trigger_decision"]
    assert td["route"] == "h1_roster"
    assert td["order_authority"].startswith("none")
    assert td["trigger_path_evaluated_by_this_scan"] is False
    # Decision competing-scenarios status text unchanged
    # (test-pinned by test_trigger_assessment).
    scen = rec["decision"]["competing_scenarios"]
    assert "order authority belongs to the trigger path, " \
        "which did not fire" in scen[0]["status"]
    # Provenance: evaluation time is the TICK time, not bar time.
    prov = rec["evidence"]["provenance"]
    assert str(prov["evaluation_ts"]) != str(frame.index[-1])
    assert str(prov["evaluation_ts"]) == \
        str(frame.index[-1] + pd.Timedelta(minutes=90))
    text = render_assessment(rec)
    assert "NOT A LIVE DECISION" in text
    assert "Trigger decision:" in text
    assert "order authority: none" in text


def test_sweep_candidate_live_close_has_no_banner(tmp_path):
    runner, frame = _candidate_runner(tmp_path)
    rec = _sweep_candidate(runner, frame, "live_close")
    text = render_assessment(rec)
    assert "NOT A LIVE DECISION" not in text
    assert "Trigger decision:" in text


# ------------------------------------------------------------ range causality

def test_range_h4_slice_is_causal_to_h1_close():
    h1 = _frame([(100.0, 101.0, 99.0, 100.0)] * 40)
    h4 = _frame([(100.0, 101.0, 99.0, 100.0)] * 20, freq="4h")
    pb = RangeSweepReclaimPlaybook()
    pb.set_h4_frame(h4)
    captured = {}

    def _capture(h4_closed, _h1_prefix):
        captured["h4"] = h4_closed

    pb._detector.update = _capture
    j = 20
    pb._range_upto(h1, j)
    closed = captured["h4"]
    assert len(closed) > 0
    # H1 bar j closes at index[j] + 60min; only H4 bars with
    # index + 240min <= that close feed the detector.
    cutoff = h1.index[j] + pd.Timedelta(minutes=60) \
        - pd.Timedelta(minutes=240)
    assert closed.index.max() <= cutoff
    # Non-vacuous: the H4 frame extends past the cutoff, and
    # those later bars were excluded.
    assert h4.index.max() > cutoff
    assert len(closed) < len(h4)


# ------------------------------------------------------------ evidence map

def test_evidence_map_covers_registry_exactly():
    assert set(PLAYBOOK_EVIDENCE) == set(PLAYBOOK_NAMES)
    assert len(PLAYBOOK_EVIDENCE) == 18
    for name, row in PLAYBOOK_EVIDENCE.items():
        assert row["route"], name
        assert row["authority"], name


def test_evidence_map_census_is_evidence_only_by_construction():
    m = evidence_map()
    assert m["census_evidence_only"] == list(CENSUS_EVIDENCE_ONLY)
    assert m["shared_decision_inputs"] == list(SHARED_DECISION_INPUTS)
    assert "funding events" in " ".join(SHARED_MANAGEMENT_ONLY)
    for family in CENSUS_EVIDENCE_ONLY:
        assert not any(family in line
                       for line in SHARED_DECISION_INPUTS)


def test_evidence_map_uncertain_flags_stated_not_resolved():
    for name in ("narrative", "sweep_reversal", "qm_d1"):
        assert "UNCERTAIN" in PLAYBOOK_EVIDENCE[name]["notes"], name
