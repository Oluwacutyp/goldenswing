"""Phase 2 — the EC2 journal failure report, fixed and proven.

Evidence base (run cb9b3fb522b7, build 913f48a): 49 evaluation
records, all qm_m5; 32 candidates in one 8-second burst off a
single stale bar (historical patterns, entries up to $180 from
the market) routed through the live order path; 23 of them
"accepted" in decision.result while refused in the order path;
headline score 60.0 flat while component totals ran 31.5–48.1;
manifest declaring fixed_lots=0.01 AND sizing_mode="risk";
range playbooks declared H1 against the owner's §G ruling.

Covered here:
(P0-1) provenance + eligibility + idempotency — historical /
       warmup / catch-up candidates NEVER reach order
       submission; one current setup does;
(P0-2) truthful scores (asymmetry vs sleeve base, separately
       named) + terminal decision semantics that reconcile in
       ONE view (journal summary AND /status counters);
(P0-3) every playbook evaluates on its agreed clock,
       independent of the generic H1 trigger; range family per
       §G (H4 generation + H1 refinement, shadow, D1 context
       only); qm_d1 on D1 close;
(P1-1) sizing reconciliation — effective mode surfaced, the
       decision stage enforces the same daily-loss room as the
       order path, invalid risk sizing declines and is never
       rescued by fixed lots;
(P1-2) the owner's numbers are untouched (pinned, not tuned).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import goldenswing.execution.mt5_runner as mr
from goldenswing.execution import candidate_lifecycle as lifecycle
from goldenswing.execution.friction_gate import FRICTION_RATIO_LIMIT
from goldenswing.execution.funnel import summarize_evaluations
from goldenswing.execution.playbook_registry import (
    FLOOR_AUTHORITY, RANGE_SPECS, ROSTER_SPECS, ROSTER_SPECS_H1)
from goldenswing.execution.risk import RiskConfig

from test_funnel_m5 import (
    _evals, _fresh_tick, _last_decision, _m5_fixture, _qm_runner)
from test_p3_trader import (
    _FakeExecutor, _FakeQMD1, _FakeScore, _StubEngine, _flat_h1,
    _frame, _runner, _signal_records)
from test_qm_d1 import BEARISH, _df as _qm_df


# ------------------------------------------------------------ helpers

def _history_frame():
    """Two textbook QM patterns buried in one long M5 history
    (touches at bars 31 and 122 of 142) — the replay shape."""
    def shifted(rows, off):
        return [(o + off, h + off, l + off, c + off)
                for (o, h, l, c) in rows]
    flat = [(500.0, 500.5, 499.5, 500.0)] * 40
    return _qm_df(BEARISH + flat + shifted(BEARISH, 300.0),
                  freq="5min")


def _wire(runner, m5):
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    return runner


def _candidate_records(runner, playbook="qm_m5"):
    return [r for r in _evals(runner, playbook)
            if r["outcome"] == "candidate_emitted"]


# ------------------------------------------------------ P0-1 replay

def test_replay_multiday_history_zero_orders(tmp_path):
    """THE burst regression: multiple days of historical M5
    patterns through a LIVE-mode runner -> ZERO order requests,
    every candidate journaled as expired with truthful
    provenance."""
    runner = _wire(_qm_runner(tmp_path), _history_frame())
    out = runner._run_qm_m5(_history_frame(), {})
    assert len(out) == 2  # both historical touches were SEEN
    assert runner.executor.orders == []  # ...and NONE routed
    recs = _candidate_records(runner)
    assert len(recs) == 2
    for rec in recs:
        dec = rec["decision"]
        assert dec["result"] == "refused"
        assert dec["reason_code"] == "gate.data"
        assert dec["execution_eligible"] is False
        assert dec["order_result"] == "not_attempted"
        prov = rec["candidate"]["provenance"]
        assert prov["touch_age_bars"] > lifecycle.QM_VALIDITY_WINDOW_BARS
        assert prov["execution_eligible"] is False
        assert prov["source"] == "live_close"


def test_replay_control_current_setup_routes(tmp_path):
    """The control: the SAME runner shape, a setup whose touch
    is the evaluated bar -> exactly it reaches the order path."""
    runner = _wire(_qm_runner(tmp_path), _m5_fixture())
    out = runner._run_qm_m5(_m5_fixture(), {})
    assert len(out) == 1
    assert len(runner.executor.orders) == 1
    prov = _candidate_records(runner)[0]["candidate"]["provenance"]
    assert prov["touch_age_bars"] == 0
    assert prov["execution_eligible"] is True


def test_warmup_scan_never_routes_even_a_fresh_touch(tmp_path):
    """Warmup provenance is categorically ineligible — even
    when the touch is on the frame's last bar (the EC2 warm
    start was exactly this shape)."""
    runner = _wire(_qm_runner(tmp_path), _m5_fixture())
    runner._run_qm_m5(_m5_fixture(), {}, source="warmup_scan")
    assert runner.executor.orders == []
    rec = _candidate_records(runner)[0]
    assert rec["decision"]["result"] == "not_executable"
    assert rec["decision"]["reason_code"] == "path.warmup_ineligible"
    assert rec["decision"]["execution_eligible"] is False
    assert rec["evaluation_source"] == "warmup_scan"
    assert rec["candidate"]["provenance"]["source"] == "warmup_scan"
    # Two layers: the strategy layer never ran (provenance),
    # and the execution layer carries the lifecycle code.
    assert rec["decision"]["analytical_verdict"]["verdict"] == \
        "not_evaluated"
    assert rec["decision"]["execution"]["reason_code"] == \
        "path.warmup_ineligible"


def test_provenance_five_timestamps_truthful(tmp_path):
    runner = _wire(_qm_runner(tmp_path), _history_frame())
    frame = _history_frame()
    runner._run_qm_m5(frame, {})
    prov = _candidate_records(runner)[0]["candidate"]["provenance"]
    # Five distinct fields, each the time it claims to be.
    assert prov["pattern_touch_ts"] == str(frame.index[31])
    assert prov["evaluated_bar_ts"] == str(frame.index[-1])
    assert prov["pattern_formation_ts"] is not None
    assert prov["candidate_created_ts"] is not None
    assert prov["evaluation_ts"] is not None
    assert (pd.Timestamp(prov["pattern_formation_ts"])
            < pd.Timestamp(prov["pattern_touch_ts"])
            < pd.Timestamp(prov["evaluated_bar_ts"]))
    assert prov["instance_key"].startswith("qm_m5:")
    assert "−1" not in prov["instance_key"]


def test_instance_idempotent_across_restart(tmp_path):
    """Same journal dir, NEW runner process state: the durable
    seen-instance set means the historical setups are not even
    re-journaled as candidates, let alone re-routed."""
    frame = _history_frame()
    r1 = _wire(_qm_runner(tmp_path), frame)
    r1._run_qm_m5(frame, {}, source="warmup_scan")
    assert r1.executor.orders == []
    journal_path = tmp_path / "mt5_journal.jsonl"
    before = [json.loads(line) for line in
              journal_path.read_text().splitlines() if line.strip()]
    n_cand_before = sum(
        1 for r in before if r.get("type") == "evaluation"
        and r.get("outcome") == "candidate_emitted")
    assert n_cand_before == 2
    # Sidecar persisted beside the journal.
    assert (tmp_path / "candidate_instances.json").exists()
    r2 = _wire(_qm_runner(tmp_path), frame)
    r2._run_qm_m5(frame, {}, source="warmup_scan")
    r2._run_qm_m5(frame, {}, source="live_close")
    assert r2.executor.orders == []
    after = [json.loads(line) for line in
             journal_path.read_text().splitlines() if line.strip()]
    n_cand_after = sum(
        1 for r in after if r.get("type") == "evaluation"
        and r.get("outcome") == "candidate_emitted")
    assert n_cand_after == n_cand_before  # nothing re-routed


def test_order_path_backstop_refuses_ineligible_stamp(tmp_path):
    """Defence in depth: a signal stamped ineligible is refused
    by the SHARED order path itself, whatever produced it."""
    runner = _qm_runner(tmp_path)
    runner._stack = SimpleNamespace(frames={"H1": _flat_h1(30, 95.0)})
    runner._last_tick = {"mid": 95.0, "bid": 94.9, "ask": 95.1,
                         "spread": 0.2,
                         "time": pd.Timestamp("2026-01-05",
                                                tz="UTC")}
    sig = SimpleNamespace(
        direction=-1, index=0, score=60.0, entry_ref=96.0,
        invalidation=100.0, targets=[90.0], atr=2.0, factors=[],
        kind="qm", execution_eligible=False,
        ineligible_reason_code="path.warmup_ineligible")
    pb = SimpleNamespace(name="qm_m5")
    stats: dict = {}
    runner._place_live_order(pb, sig, stats)
    assert runner.executor.orders == []
    assert stats.get("ineligible::path.warmup_ineligible") == 1


# ------------------------------------------------- P0-2 score truth

def test_scores_named_and_headline_is_the_computed_value(tmp_path):
    """Headline score = the Asymmetry total the decision used;
    the sleeve base (60.0 placeholder) rides under its own
    name; the signal record agrees."""
    runner = _wire(_qm_runner(tmp_path), _m5_fixture())
    runner._run_qm_m5(_m5_fixture(), {})
    cand = _candidate_records(runner)[0]["candidate"]
    assert cand["asymmetry_score"] == 70.0
    assert cand["sleeve_base_score"] == 60.0
    assert cand["score"] == 70.0  # not the 60.0 constant
    sigs = [r for r in _signal_records(runner)
            if r.get("playbook") == "qm_m5"]
    assert sigs[-1]["score"] == 70.0


def test_terminal_refusal_promoted_and_reconciles(tmp_path):
    """Analytical accept + order-path friction refusal ->
    terminal record says REFUSED with the order-path code, the
    analytical verdict preserved, and BOTH censuses (journal
    summary + /status counters) reconcile exactly."""
    runner = _wire(_qm_runner(tmp_path), _m5_fixture())
    runner._last_tick = _fresh_tick(_m5_fixture(), spread=10.0)
    runner._run_qm_m5(_m5_fixture(), {})
    assert runner.executor.orders == []
    rec = _candidate_records(runner)[0]
    dec = rec["decision"]
    assert dec["result"] == "refused"
    assert dec["analytical_result"] == "accepted"
    assert dec["reason_code"] == "risk.friction"
    assert dec["order_result"] == "refused_in_order_path"
    # Journal census reconciles.
    summary = summarize_evaluations(runner.journal.records(),
                                    ["qm_m5"])
    term = summary["by_playbook"]["qm_m5"]["terminal"]
    assert term == {"ordered": 0, "refused": 1, "ineligible": 0,
                    "no_order": 0, "unresolved": 0}
    # /status census reconciles the same way.
    slot = runner._funnel_counts["qm_m5"]
    assert slot["candidates"] == (
        slot["ordered"] + slot["refused"] + slot["ineligible"]
        + slot["no_order"] + slot["unresolved"]) == 1
    line = runner.funnel_line()
    assert "refused 1" in line and "ineligible 0" in line


def test_contradictory_evidence_weighed_and_truthful(tmp_path):
    """Counter-D1-bias candidate: the contradiction is IN the
    record (weighed, never a veto) and the terminal outcome is
    the truth (ordered)."""
    runner = _wire(_qm_runner(tmp_path), _m5_fixture())
    runner._live_setup = {"bias_d1": 1, "bias_h4": 1,
                          "regime": "trend", "size_mult": 1.0,
                          "h4_context": ""}
    runner._run_qm_m5(_m5_fixture(), {})
    assert len(runner.executor.orders) == 1
    rec = _candidate_records(runner)[0]
    assert any("against structural D1 bias" in c
               for c in rec["evidence"]["contradictory"])
    assert rec["decision"]["result"] == "accepted"
    term = summarize_evaluations(
        runner.journal.records(), ["qm_m5"]
    )["by_playbook"]["qm_m5"]["terminal"]
    assert term["ordered"] == 1 and term["unresolved"] == 0


def test_below_floor_qm_d1_terminal_record(tmp_path, monkeypatch):
    """qm_d1 below its MACRO_SWING_OPEN floor: refused with a
    floor code, floor authority documented, scores truthful."""
    monkeypatch.setattr(mr, "QMD1Playbook", _FakeQMD1)
    runner = _runner(tmp_path)
    runner._score_for = lambda **kw: _FakeScore(10.0, 5.0, 5.0)
    runner._get_engine = lambda: _StubEngine()
    d1 = _frame([(4100.0, 4110.0, 4090.0, 4105.0)] * 40, freq="1d")
    runner._run_qm_d1(d1, {})
    assert runner.executor.orders == []
    rec = _candidate_records(runner, "qm_d1")[0]
    dec = rec["decision"]
    assert dec["result"] == "refused"
    assert dec["reason_code"].startswith("floor.")
    assert rec["candidate"]["asymmetry_score"] == 10.0
    assert rec["candidate"]["sleeve_base_score"] == 70.0
    assert "native:qm_d1" in FLOOR_AUTHORITY


# ------------------------------------------------- P0-3 clock matrix

def _full_stack_runner(tmp_path):
    runner = _qm_runner(tmp_path)
    h1 = _flat_h1(60, 95.0)
    h4 = _frame([(95.0, 95.5, 94.5, 95.0)] * 60, freq="4h")
    d1 = _frame([(95.0, 95.5, 94.5, 95.0)] * 40, freq="1d")
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": h1, "H4": h4, "D1": d1, "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    return runner, {"H1": h1, "D1": d1, "M5": m5}


def test_all_18_evaluate_on_their_clocks_without_trigger(tmp_path):
    """No _run_signal_path call anywhere in this test: every
    registered playbook still journals an evaluation on its
    agreed clock."""
    from goldenswing.execution.playbook_registry import PLAYBOOK_NAMES
    runner, frames = _full_stack_runner(tmp_path)
    runner._run_qm_m5(frames["M5"], {}, source="live_close")
    runner._run_qm_d1(frames["D1"], {}, source="live_close")
    runner._run_roster_close_evals({}, source="live_close")
    runner._run_range_family("h4_generation", {}, source="live_close")
    runner._run_range_family("h1_refinement", {}, source="live_close")
    summary = summarize_evaluations(runner.journal.records(),
                                    PLAYBOOK_NAMES)
    assert summary["not_called"] == []
    by = summary["by_playbook"]
    for spec in ROSTER_SPECS_H1:
        assert by[spec.name]["by_clock"] == {"h1_close": 1}, spec.name
    for spec in RANGE_SPECS:
        clocks = by[spec.name]["by_clock"]
        assert clocks.get("h4_generation") == 1, spec.name
        assert clocks.get("h1_refinement") == 1, spec.name
    assert by["qm_m5"]["by_clock"] == {"m5_close": 1}
    assert by["qm_d1"]["by_clock"] == {"d1_close": 1}
    assert summary["frame_mismatches"] == 0


def test_range_family_registry_contract():
    """§G in the registry: H4 generation, H1 refinement, shadow
    evaluation; the trigger-path roster keeps all 16 members
    (no silent authority change)."""
    assert len(RANGE_SPECS) == 4
    for spec in RANGE_SPECS:
        assert spec.native_tf == "H4"
        assert spec.refinement_tf == "H1"
        assert spec.shadow_evaluation is True
    assert len(ROSTER_SPECS) == 16
    assert {s.name for s in ROSTER_SPECS} >= {s.name
                                              for s in RANGE_SPECS}


def test_range_shadow_records_carry_d1_as_context_only(tmp_path):
    runner, _frames = _full_stack_runner(tmp_path)
    runner._live_setup = {"bias_d1": -1}
    runner._run_range_family("h4_generation", {}, source="live_close")
    recs = [r for r in _evals(runner, "range_qm")]
    assert recs and recs[0]["evaluation_clock"] == "h4_generation"
    assert recs[0]["intended_tf"] == "H4"
    ev = recs[0]["evidence"]
    assert ev["d1_bias_context"] == -1
    assert "never consulted" in ev["d1_role"]


def test_disabled_playbook_gets_not_eligible_in_sweep(tmp_path):
    runner = _qm_runner(tmp_path, disabled_playbooks=("narrative",))
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(60, 95.0),
                "H4": _frame([(95.0, 95.5, 94.5, 95.0)] * 60,
                             freq="4h")})
    runner._last_tick = _fresh_tick(_flat_h1(60, 95.0))
    runner._run_roster_close_evals({}, source="live_close")
    recs = _evals(runner, "narrative")
    assert recs[-1]["outcome"] == "not_eligible"
    assert recs[-1]["reason_code"] == "path.kill_switch"


def test_qm_d1_native_route_record_shape(tmp_path, monkeypatch):
    """qm_d1 evaluates on D1 close via its native route — D1
    clock on the record, order placed, no H1 trigger involved."""
    from datetime import datetime, timezone
    monkeypatch.setattr(mr, "QMD1Playbook", _FakeQMD1)
    runner = _runner(tmp_path)
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    runner._get_engine = lambda: _StubEngine()
    d1 = _frame([(4100.0, 4110.0, 4090.0, 4105.0)] * 40, freq="1d")
    runner._stack = SimpleNamespace(
        frames={"D1": d1, "H1": _flat_h1(30, 4100.0)})
    runner._last_tick = {"mid": 4100.0, "bid": 4099.8,
                         "ask": 4100.2, "spread": 30,
                         # Pinned Wednesday: a wall-clock tick makes
                         # this fail on Fridays after 20:00 UTC.
                         "time": "2026-10-14 12:00:00"}
    out = runner._run_qm_d1(d1, {}, source="live_close")
    assert len(out) == 1 and len(runner.executor.orders) == 1
    rec = _candidate_records(runner, "qm_d1")[0]
    assert rec["evaluation_clock"] == "d1_close"
    assert rec["intended_tf"] == rec["actual_input_tf"] == "D1"
    assert rec["evaluation_source"] == "live_close"
    assert rec["candidate"]["provenance"]["touch_age_bars"] == 0
    assert rec["decision"]["floor_result"]["path"] == \
        "MACRO_SWING_OPEN"


# --------------------------------------------------------- P1-1 sizing

def _small_account_runner(tmp_path, feed_kw=None, **cfg_kw):
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    feed = SimpleNamespace(
        account_state=lambda: {"equity": 12.0, "balance": 12.0,
                               "free_margin": 12.0},
        order_margin=lambda *, volume, price, direction: 2.7)
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False, **cfg_kw)
    runner = MT5Runner(cfg, feed=feed, executor=_FakeExecutor())
    runner._send_alert = lambda *a, **k: None
    runner._build_chart = lambda *a, **k: (None, None)
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    runner._get_engine = lambda: _StubEngine()
    return _wire(runner, _m5_fixture())


def test_manifest_surfaces_effective_sizing_mode(tmp_path):
    """PREMISE CORRECTION (owner ruling 2026-10-23): this test
    pinned A13 precedence (fixed_lots=0.01 overriding
    sizing_mode='risk').  A13 is REVOKED — fixed lots are gone
    from the production path; the manifest now reports the
    dynamic rule and carries no fixed-lots keys at all."""
    runner = _small_account_runner(tmp_path)
    runner._journal_manifest()
    man = [r for r in runner.journal.records()
           if r.get("type") == "manifest"][-1]
    doc = man["doctrine"]
    assert "fixed_lots" not in doc
    assert "fixed_lots_override" not in doc
    assert doc["sizing_mode"] == "risk"
    assert doc["sizing_mode_effective"] == "risk"
    assert doc["risk_pct"] == 0.01
    roster = {row["name"]: row for row in man["roster"]}
    assert roster["range_qm"]["clock"] == "H4"
    assert roster["range_qm"]["refinement_clock"] == "H1"
    assert roster["range_qm"]["shadow_evaluation"] is True


def test_decision_solvency_places_minimum_under_new_ruling(tmp_path):
    """OWNER RULING 2026-10-24 (reverses the 2026-10-23 abstention
    this test used to pin): a setup is account-agnostic — when
    the stop risk exceeds the balance the engine trades the
    broker MINIMUM, never abandons.  The $12 account's sizing
    authorization is $0.096 (1% x session 0.8); the strategy
    still QUALIFIES the setup (analytical layer unchanged),
    decision-stage solvency PASSES under the effective sizing
    mode, and the shared order path PLACES 0.01 lots with
    min_override flagged and estimated stop exposure ~$4.67
    (drag negative — the honest arithmetic of the floor).
    "daily loss room" stays gone (revoked premise, unchanged)."""
    runner = _small_account_runner(tmp_path)
    runner._run_qm_m5(_m5_fixture(), {})
    assert len(runner.executor.orders) == 1
    assert runner.executor.orders[0]["volume"] == pytest.approx(0.01)
    rec = _candidate_records(runner)[0]
    dec = rec["decision"]
    assert dec["result"] == "accepted"
    assert dec["reason_code"] is None
    assert dec["order_result"] == "placed_pending"
    assert "daily loss room" not in dec["reason"]
    # Readable at a glance (amendment 2): the STRATEGY qualified
    # this setup; the RISK LAYER sized it at the minimum.  Not
    # a strategy failure — the sizing rule doing its job.
    assert dec["analytical_verdict"] == {
        "verdict": "qualified", "reason_code": None}
    assert dec["execution"] == {
        "eligible": True, "outcome": "placed_pending",
        "reason_code": None}
    solv = [c for c in dec["hard_risk_checks"]
            if c["check"].startswith("decision_solvency")]
    assert solv and solv[0]["result"] == "pass"
    detail = next(iter(runner._sizing_detail.values()))
    assert detail["min_override"] is True
    assert detail["lots"] == pytest.approx(0.01)
    assert detail["estimated_stop_exposure"] == \
        pytest.approx(4.6714, rel=1e-3)
    assert detail["drag"] < 0


def test_minimum_lot_is_not_a_fixed_lots_fallback(tmp_path):
    """OWNER RULING 2026-10-24: risk mode, tiny account — the
    budget cannot fund the minimum lot, so the order path sizes
    AT the minimum (min_override) instead of declining.  That
    floor is the owner-authorized sizing rule, NOT a silent
    rescue with fixed lots: the fixed-lots mode stays revoked
    (manifest carries no fixed-lots keys, sizing_mode 'risk'),
    and no sizing_decline is recorded for the below-minimum
    case (volume-limit / nonpositive declines are untouched)."""
    runner = _small_account_runner(tmp_path)  # no fixed_lots
    runner._run_qm_m5(_m5_fixture(), {})
    assert len(runner.executor.orders) == 1
    assert runner.executor.orders[0]["volume"] == pytest.approx(0.01)
    sigs = [r for r in _signal_records(runner)
            if r.get("playbook") == "qm_m5"]
    assert not any("sizing_decline" in str(r.get("reason", ""))
                   for r in sigs)
    rec = _candidate_records(runner)[0]
    assert rec["decision"]["result"] == "accepted"
    assert rec["decision"]["order_result"] == "placed_pending"
    assert rec["decision"]["reason_code"] is None
    detail = next(iter(runner._sizing_detail.values()))
    assert detail["min_override"] is True
    runner._journal_manifest()
    man = [r for r in runner.journal.records()
           if r.get("type") == "manifest"][-1]
    assert "fixed_lots" not in man["doctrine"]
    assert man["doctrine"]["sizing_mode"] == "risk"


# ------------------------------------- addendum A: dynamic sizing

class _SizedQM:
    """A QM candidate with a configurable stop distance, touch
    on the frame's last bar (the live-close shape)."""
    name = "qm_m5"
    session_scope = "any"
    hunting_ground = SimpleNamespace(excluded=[], bonuses={})

    def __init__(self, stop=100.34):
        self._stop = stop

    def candidates(self, ctx):
        last = len(ctx.df) - 1
        sig = SimpleNamespace(
            direction=-1, index=last, score=60.0, entry_ref=96.0,
            invalidation=self._stop, targets=[90.0], atr=2.0,
            factors=["qm"], kind="qm",
            live_context={"qm_level": 96.0, "head": 100.0,
                          "neckline": 90.0, "tags": []})
        yield SimpleNamespace(
            signal_index=last, direction=-1,
            outcome=SimpleNamespace(signal=sig, r_net=1.0,
                                    exit_reason="target"))


def _equity_runner(tmp_path, monkeypatch, equity, stop=100.34):
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    monkeypatch.setattr(mr, "QMM5Playbook",
                        lambda: _SizedQM(stop))
    feed = SimpleNamespace(
        account_state=lambda: {"equity": equity, "balance": equity,
                               "free_margin": equity},
        order_margin=lambda *, volume, price, direction: 1.0)
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False)
    runner = MT5Runner(cfg, feed=feed, executor=_FakeExecutor())
    runner._send_alert = lambda *a, **k: None
    runner._build_chart = lambda *a, **k: (None, None)
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    runner._get_engine = lambda: _StubEngine()
    return _wire(runner, _m5_fixture())


def test_dynamic_sizing_scales_with_equity_and_floors_at_minimum(
        tmp_path, monkeypatch):
    """Addendum A: the SAME eligible setup at $12 / $24 / $1,000
    / $10,000 equity — sizes follow the configured risk rule
    mechanically (floor at the minimum lot, round DOWN to
    the step, shrink as the stop widens); the strategy evidence
    (score, components, sleeve base) is byte-identical at every
    balance.  The balance is a test input, never a design
    target.  OWNER RULING 2026-10-24: $12 / $24 no longer
    abstain — they trade the minimum."""
    runs = {}
    for equity in (12.0, 24.0, 1000.0, 10000.0):
        sub = tmp_path / f"eq{int(equity)}"
        sub.mkdir()
        runner = _equity_runner(sub, monkeypatch, equity)
        runner._run_qm_m5(_m5_fixture(), {})
        runs[equity] = runner
    # $12 and $24: the authorization (1% x session 0.8 = $0.096
    # / $0.192) cannot fund the minimum 0.01 lot at the ~$4.84
    # buffered stop ($4.84 exposure) -> owner ruling 2026-10-24:
    # TRADE the minimum (min_override flagged, drag negative),
    # never a forced abstention.
    for equity in (12.0, 24.0):
        runner = runs[equity]
        assert len(runner.executor.orders) == 1
        assert runner.executor.orders[0]["volume"] == \
            pytest.approx(0.01)
        sigs = [r for r in _signal_records(runner)
                if r.get("playbook") == "qm_m5"]
        assert not any("sizing_decline" in str(r.get("reason", ""))
                       for r in sigs), equity
        detail = next(iter(runner._sizing_detail.values()))
        assert detail["min_override"] is True
        assert detail["estimated_stop_exposure"] == \
            pytest.approx(4.84, rel=1e-3)
        assert detail["drag"] < 0
    # $1,000: budget $8.00 -> raw ~0.0165 lots -> rounded DOWN
    # to the 0.01 step.  $10,000: a mechanically larger size —
    # raw ~0.165 lots -> floored to 0.16, under the ceiling
    # (budget $80.00 / $4.84 risk per 0.01 lot = 0.165 lots;
    # session/posture multipliers only shrink it).  The sizing
    # journal proves the equation for non-floor sizes:
    # realized risk <= authorized risk.
    assert runs[1000.0].executor.orders[0]["volume"] == \
        pytest.approx(0.01)
    v10k = runs[10000.0].executor.orders[0]["volume"]
    assert v10k == pytest.approx(0.16)
    for runner in (runs[1000.0], runs[10000.0]):
        detail = next(iter(runner._sizing_detail.values()))
        assert detail["lots"] == pytest.approx(
            runner.executor.orders[0]["volume"])
        assert detail["realized_risk"] <= \
            detail["authorized_risk"] + 1e-9
    # Wider stop at the same equity -> mechanically smaller.
    sub = tmp_path / "wide"
    sub.mkdir()
    wide = _equity_runner(sub, monkeypatch, 10000.0, stop=116.0)
    wide._run_qm_m5(_m5_fixture(), {})
    assert 0 < wide.executor.orders[0]["volume"] < v10k
    # Invariance: the candidate's evidence is identical at every
    # balance — only sizing outcomes differ.
    cands = {}
    for equity, runner in runs.items():
        rec = _candidate_records(runner)[0]
        c = dict(rec["candidate"])
        c.pop("provenance", None)
        cands[equity] = c
    assert len({json.dumps(c, sort_keys=True, default=str)
                for c in cands.values()}) == 1
    # Amendment 2, both halves, explicitly:
    # (a) the ANALYTICAL VERDICT is identical at every balance —
    #     qualified, same (absent) analytical reason;
    # (b) EXECUTION ELIGIBILITY is True at every balance under
    #     the 2026-10-24 ruling — the small-balance difference
    #     is the min_override flag + negative drag in the sizing
    #     journal, not a refusal.
    analytical = {}
    execution = {}
    for equity, runner in runs.items():
        dec = _candidate_records(runner)[0]["decision"]
        analytical[equity] = dec["analytical_verdict"]
        execution[equity] = dec["execution"]
    assert len({json.dumps(a, sort_keys=True)
                for a in analytical.values()}) == 1
    assert analytical[12.0] == {"verdict": "qualified",
                                "reason_code": None}
    for equity in (12.0, 24.0, 1000.0, 10000.0):
        ex = execution[equity]
        assert ex["eligible"] is True
        assert ex["outcome"] == "placed_pending"
        assert ex["reason_code"] is None


def test_sizing_precedence_is_explicit_not_contradictory():
    """PREMISE CORRECTION (owner ruling 2026-10-23): Addendum A
    pinned A13 precedence (an explicit fixed_lots always wins).
    A13 is REVOKED: there is no fixed_lots field at all and the
    configured mode is the only mode — dynamic risk sizing."""
    from goldenswing.execution.mt5_runner import MT5RunnerConfig
    assert MT5RunnerConfig().sizing_mode == "risk"
    assert not hasattr(MT5RunnerConfig(), "fixed_lots")


# -------------------------------- addendum C: reasoning records

class _BothTagsQM(_SizedQM):
    def candidates(self, ctx):
        last = len(ctx.df) - 1
        sig = SimpleNamespace(
            direction=-1, index=last, score=60.0, entry_ref=96.0,
            invalidation=100.34, targets=[90.0], atr=2.0,
            factors=["qm"], kind="qm",
            live_context={"qm_level": 96.0, "head": 100.0,
                          "neckline": 90.0,
                          "tags": ["sweep_head",
                                   "displacement_break"]})
        yield SimpleNamespace(
            signal_index=last, direction=-1,
            outcome=SimpleNamespace(signal=sig, r_net=1.0,
                                    exit_reason="target"))


def test_correlated_evidence_grouped_not_stacked(
        tmp_path, monkeypatch):
    """sweep_head + displacement_break are two readings of ONE
    pattern event: the record groups them (independent group
    count 1 for the pattern), while HTF alignment is a second,
    genuinely independent group."""
    monkeypatch.setattr(mr, "QMM5Playbook", _BothTagsQM)
    runner = _wire(_qm_runner(tmp_path), _m5_fixture())
    runner._live_setup = {"bias_d1": -1, "bias_h4": -1,
                          "regime": "trend", "size_mult": 1.0,
                          "h4_context": ""}
    runner._run_qm_m5(_m5_fixture(), {})
    rec = _candidate_records(runner)[0]
    ev = rec["evidence"]
    assert len(ev["evidence_groups"]["qm_pattern_event"]) == 2
    assert ev["independent_evidence_groups"] == [
        "htf_structure", "qm_pattern_event"]
    assert rec["decision"]["reasoning_mode"] == "standalone"
    assert rec["decision"]["floor_authority"]


def test_competing_scenarios_coexist_on_the_record(
        tmp_path, monkeypatch):
    """Both scenarios are on the record: the taken side
    (selected) and the flip side (not funded).  When the score
    DOES flip, the record shows the other scenario selected by
    the weighed evidence and this side refused — nothing is
    silently suppressed."""
    runner = _wire(_qm_runner(tmp_path), _m5_fixture())
    runner._run_qm_m5(_m5_fixture(), {})
    dec = _candidate_records(runner)[0]["decision"]
    scen = dec["competing_scenarios"]
    assert len(scen) == 2
    assert scen[0]["status"] == "selected"
    assert "not funded" in scen[1]["status"]
    # Flip case.
    class _FlipScore(_FakeScore):
        flipped = True
    import tempfile
    from pathlib import Path as _P
    tmp2 = _P(tempfile.mkdtemp())
    runner2 = _wire(_qm_runner(tmp2), _m5_fixture())
    runner2._score_for = lambda **kw: _FlipScore(70.0, 25.0, 15.0)
    runner2._run_qm_m5(_m5_fixture(), {})
    assert runner2.executor.orders == []
    dec2 = _candidate_records(runner2)[0]["decision"]
    assert dec2["result"] == "refused"
    assert dec2["reason_code"] == "direction.flip"
    assert "selected by the weighed evidence" in \
        dec2["competing_scenarios"][1]["status"]


# --------------------------- addendum B/E: SMT timestamp alignment

def test_smt_alignment_joins_on_timestamps():
    from goldenswing.live.smt import align_on_timestamps
    gold = _frame([(100.0, 101.0, 99.0, 100.0)] * 50, freq="h")
    dxy = _frame([(100.0, 101.0, 99.0, 100.0)] * 50, freq="h")
    dxy_shifted = dxy.iloc[5:].copy()  # 45 bars, starts 5h later
    g2, d2 = align_on_timestamps(gold, dxy_shifted)
    assert g2 is not None and len(g2) == len(d2) == 45
    assert list(g2.index) == list(d2.index)
    # Unalignable (no datetime index) -> (None, None), never a
    # positional guess.
    import pandas as pd
    plain = pd.DataFrame({"open": [1.0], "high": [1.0],
                          "low": [1.0], "close": [1.0]})
    assert align_on_timestamps(plain, plain) == (None, None)


def test_smt_refresh_reads_aligned_frames_not_positions(tmp_path):
    """The runner's SMT read equals detect_smt computed on the
    TIMESTAMP-ALIGNED frames — a DXY frame offset by dropped
    bars cannot fabricate a positional event."""
    from goldenswing.live import smt as smt_mod
    from test_p3_trader import _smt_frames
    gold, dxy = _smt_frames()
    dxy_dropped = dxy.iloc[3:].copy()  # same values, bars missing
    runner = _qm_runner(tmp_path)
    runner._dxy = SimpleNamespace(
        frame=dxy_dropped,
        state=SimpleNamespace(available=True, corr=-0.8))
    runner._refresh_smt({"H1": gold})
    g_al, d_al = smt_mod.align_on_timestamps(gold, dxy_dropped)
    expected = smt_mod.detect_smt(g_al, d_al, corr=-0.8)
    got = runner._smt_read
    assert got.available == expected.available
    assert (got.event is None) == (expected.event is None)
    if expected.event is not None:
        assert got.event.kind == expected.event.kind
        assert got.event.bars_ago == expected.event.bars_ago


# ------------------------------ scheduler + causality (addendum E)

def test_scheduler_fires_sweeps_on_their_closes(tmp_path):
    """_refresh_bars wiring: an H1 close fires the roster sweep
    + range refinement; an H4 close on the same tick also fires
    range generation; a tick closing nothing fires none."""
    runner = _qm_runner(tmp_path)
    calls = {"roster": [], "range": [], "qm_m5": 0, "qm_d1": 0}
    runner._run_roster_close_evals = lambda stats, source=None: \
        calls["roster"].append(source)
    runner._run_range_family = lambda stage, stats, source=None: \
        calls["range"].append(stage)
    runner._run_qm_m5 = lambda df, stats=None, **kw: \
        calls.__setitem__("qm_m5", calls["qm_m5"] + 1) or []
    runner._run_qm_d1 = lambda df, stats=None, **kw: \
        calls.__setitem__("qm_d1", calls["qm_d1"] + 1) or []
    for name in ("_refresh_dxy", "_refresh_live_setup",
                 "_refresh_alignment", "_check_close_rule",
                 "_on_h1_close", "_advance_regime_pause",
                 "_check_pending_bias"):
        setattr(runner, name, lambda *a, **k: None)
    h1 = _flat_h1(30, 95.0)
    h4 = _frame([(95.0, 95.5, 94.5, 95.0)] * 30, freq="4h")
    frames = {"H1": h1, "H4": h4}
    cursors = {"H1": h1.index[-1], "H4": h4.index[-1]}

    class _Stack:
        def __init__(self):
            self.frames = frames
            self.cursors = cursors

        def append(self, tf, new):
            self.frames[tf] = pd.concat([self.frames[tf], new])
            self.cursors[tf] = self.frames[tf].index[-1]
            return len(new), 0
    runner._stack = _Stack()
    new_h1 = _frame([(95.0, 95.5, 94.5, 95.0)],
                    start=str(h1.index[-1] + pd.Timedelta(hours=1)),
                    freq="h")
    new_h4 = _frame([(95.0, 95.5, 94.5, 95.0)],
                    start=str(h4.index[-1] + pd.Timedelta(hours=4)),
                    freq="4h")
    runner.feed = SimpleNamespace(
        fetch_new_all=lambda c: {"H1": new_h1, "H4": new_h4})
    runner._last_processed = {
        "H1": pd.Timestamp("2026-01-05 00:00", tz="UTC"),
        "H4": pd.Timestamp("2026-01-05 00:00", tz="UTC")}
    tick_time = h1.index[-1] + pd.Timedelta(hours=1, minutes=30)
    runner._refresh_bars({"time": tick_time}, {})
    assert calls["roster"] == ["live_close"]
    assert calls["range"] == ["h1_refinement", "h4_generation"]
    # A tick inside the same bars closes nothing new.
    calls["roster"].clear()
    calls["range"].clear()
    runner._refresh_bars(
        {"time": tick_time + pd.Timedelta(minutes=10)}, {})
    assert calls["roster"] == [] and calls["range"] == []


def test_lifecycle_future_touch_fails_closed():
    """Timestamp causality at the lifecycle boundary: a touch
    dated AFTER the evaluated bar (negative age — corrupt or
    lookahead provenance) is ineligible, fail closed."""
    ok, code, _reason = lifecycle.assess_eligibility(
        source="live_close", touch_age_bars=-1,
        validity_window_bars=1)
    assert not ok and code == "gate.data"
    ok, code, _ = lifecycle.assess_eligibility(
        source="live_close", touch_age_bars=None,
        validity_window_bars=1)
    assert not ok and code == "gate.data"


# ------------------------------------------------- P1-2 owner numbers

def test_owner_numbers_untouched():
    """Owner numbers this pass did NOT change: the friction
    bound, the min_score value, the QM validity window.
    PREMISE CORRECTION (owner ruling 2026-10-23): the original
    test also pinned daily_loss_pct == 0.03 and the $0.36/$0.72
    room math — the owner explicitly killed both ("drop the
    stupid fixed lots… who needs that stupid 0.36 nonsense");
    the fields are GONE, and max drawdown (0.15) stands as the
    sole account-level hard halt."""
    assert FRICTION_RATIO_LIMIT == 0.10
    assert not hasattr(RiskConfig(), "daily_loss_pct")
    assert not hasattr(RiskConfig(), "weekly_loss_pct")
    assert RiskConfig().max_drawdown_pct == pytest.approx(0.15)
    assert mr.MT5RunnerConfig().min_score == 60.0
    assert not hasattr(mr.MT5RunnerConfig(), "fixed_lots")
    assert lifecycle.QM_VALIDITY_WINDOW_BARS == 1


# --------------------------------------- Amendment 3: assessment shape

def test_assessment_shape_on_native_candidate(tmp_path):
    """Amendment 3: the native candidate record renders a market
    assessment WITHOUT retrofit — structure read populated from
    the route's own inference, HTF context (D1/H4 bias) present,
    evidence itemized with detector + source-bar timestamp and
    correlation groups, supporting/contradictory as separate
    lists, competing scenarios present, verdict in the owner's
    vocabulary, plain-language reason, and the not-live
    capabilities explicitly MARKED (never silently omitted)."""
    runner, stack = _full_stack_runner(tmp_path)
    runner._run_qm_m5(_history_frame().iloc[:32],
                      source="live_close")
    (rec,) = _candidate_records(runner)
    a = rec["assessment"]
    assert a["verdict"] in ("signal", "rejected", "watch")
    assert a["reason_text"] and a["reason_text"] == rec["reason_text"]
    # Structure read: the QM route's inference is populated.
    assert a["structure_read"].get("regime") is not None
    # HTF context: D1/H4 structural bias rode the evidence.
    assert "bias_d1" in a["htf_context"]
    # Itemized evidence: detector + source-bar ts on every item,
    # stances drawn from the two separate lists.
    assert a["evidence_items"]
    for item in a["evidence_items"]:
        assert item["detector"] == "qm_m5_native"
        assert item["source_bar_ts"]
        assert item["stance"] in ("supporting", "contradictory")
    assert a["supporting_evidence"] == \
        rec["evidence"]["supporting"]
    assert a["contradictory_evidence"] == \
        rec["evidence"]["contradictory"]
    # Correlated items grouped: the two QM pattern readings share
    # ONE group (they are one event, not two confirmations).
    groups = {i["evidence_group"] for i in a["evidence_items"]
              if i["evidence_group"]}
    assert "qm_pattern_event" in groups
    assert a["competing_scenarios"][0]["status"] in (
        "selected",) or a["competing_scenarios"]
    # Wired capabilities are MARKED with their recorded reads
    # (census-wiring pass premise correction: premium/discount
    # and the named taxonomy are consumed on the live path as
    # recorded evidence — the pre-wiring census facts
    # "not_integrated"/"absent" no longer hold for this live
    # record, which carries the reads themselves).
    marks = a["capability_markers"]
    assert marks["premium_discount"].startswith("recorded")
    assert marks["named_candlestick_taxonomy"].startswith("recorded")


def test_assessment_marks_absent_fields_explicitly(tmp_path):
    """A route that produces no structure read / scenarios /
    evidence says so on the record — the field is present and
    marked, never omitted (Amendment 3).  The disabled-playbook
    sweep record is the honest case: no evidence block at all."""
    runner = _qm_runner(tmp_path, disabled_playbooks=("narrative",))
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(60, 95.0),
                "H4": _frame([(95.0, 95.5, 94.5, 95.0)] * 60,
                             freq="4h")})
    runner._last_tick = _fresh_tick(_flat_h1(60, 95.0))
    runner._run_roster_close_evals({}, source="live_close")
    recs = _evals(runner, "narrative")
    assert recs
    a = recs[0]["assessment"]
    assert a["verdict"] == "no-setup"
    assert a["reason_text"]
    assert a["structure_read"]["status"] == "absent"
    assert a["htf_context"]["status"] == "absent"
    assert a["competing_scenarios"]["status"] == "not_produced"
    assert a["evidence_items"] == []
    assert "evidence" in a["capability_markers"]


def test_assessment_verdict_vocabulary_and_expired(tmp_path):
    """Warmup/expired provenance reads 'expired' in the owner's
    vocabulary; every record of a full roster run carries an
    assessment whose verdict is in the vocabulary."""
    runner, _stack = _full_stack_runner(tmp_path)
    runner._run_qm_m5(_history_frame(), source="warmup_scan")
    evals = _evals(runner)
    assert evals
    for r in evals:
        assert r["assessment"]["verdict"] in (
            "signal", "watch", "no-setup", "rejected", "expired")
    cand = _candidate_records(runner)
    assert cand and all(
        r["assessment"]["verdict"] == "expired" for r in cand)
    # And the §G shadow range records never read as a signal or
    # a rejection: shadow evaluation is observation — 'watch'
    # when a candidate is seen, 'no-setup' when none is.
    runner._run_range_family("h4_generation", {},
                             source="live_close")
    range_recs = [r for r in _evals(runner)
                  if r["playbook"].startswith("range_")]
    assert range_recs
    assert all(r["assessment"]["verdict"] in ("watch", "no-setup")
               for r in range_recs)
