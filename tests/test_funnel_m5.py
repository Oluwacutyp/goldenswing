"""Funnel-first patch — runner-level proof (plan §5 + rulings 1/6).

Covers:
(a) M5 positive: a known M5 QM setup driven through the RUNNER
    emits an auditable candidate on the M5 clock and traverses
    the shared, protected order path;
(b) M5 negatives: no pattern / stale data / failed safety
    condition (margin, kill switch, geometry, hard-line friction)
    -> no order, each with its own reason code;
(c) no silent generic-H1-trigger requirement for the M5 route;
(d) funnel record completeness: every evaluation incl. no-setup
    leaves a record with true input timeframe + last bar ts, and
    the per-playbook counts reconcile across the whole registry;
(e) registry consistency: every registered playbook is
    manageable/killable and every advertised control maps to a
    real playbook; roster == registry ROSTER_SPECS;
(f) observation-only: funnel on/off changes NO decision outcome;
(g) RULING 6 friction gate: unit boundaries (below / exactly at /
    above 0.10, invalid inputs fail closed, points->price unit
    normalization) + runner integration (refusal journaled as
    risk.friction category 8, setup stays alive and re-evaluates);
(h) RULING 1: counter-bias M5 scenarios are generated and
    evaluated (weighed as contradictory evidence), never
    silently suppressed; session bucket recorded;
(i) STRUCTURAL (directive §4): the H1 dataframe can never reach
    QMM5Playbook again — static AST pins + a runtime spacing
    proof that the playbook only ever sees 5-minute bars.
"""
from __future__ import annotations

import ast
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import goldenswing.execution.mt5_runner as mr
from goldenswing.execution.friction_gate import (
    FRICTION_RATIO_LIMIT, assess_entry_friction)
from goldenswing.execution.funnel import summarize_evaluations
from goldenswing.execution.mt5_config import (
    LiveConfigStore, parse_set, unknown_disabled_names)
from goldenswing.execution.playbook_registry import (
    NATIVE_SPECS, PLAYBOOK_NAMES, ROSTER_SPECS)

from test_p3_trader import (
    _FakeQMD1, _FakeScore, _StubEngine, _StubPB, _drive_one,
    _flat_h1, _frame, _runner, _sig, _signal_records)
from test_qm_d1 import BEARISH, _df as _qm_df


# ------------------------------------------------------------ fixtures

def _m5_fixture():
    """The textbook bearish QM (test_qm_d1.BEARISH) on a 5-minute
    clock: head 100 -> neckline 90 -> QM level 96 -> break ->
    touch at bar 31.

    Phase 2 revision: the frame is truncated AT the touch bar —
    a live-close evaluation presents the frame as of the touch's
    own close (the streaming machine emits the signal ON the
    touch bar).  The full-history frame (bars past the touch)
    models a late discovery and is the Phase 2 replay fixture
    (tests/test_phase2_fixes.py): under the candidate-lifecycle
    contract such a candidate is expired and never routes."""
    return _qm_df(BEARISH, freq="5min").iloc[:32]


def _m5_history_fixture():
    """The UNtruncated BEARISH frame: the touch sits 19 bars
    inside history — the warmup / catch-up / replay shape."""
    return _qm_df(BEARISH, freq="5min")


def _qm_runner(tmp_path, **cfg_kw):
    runner = _runner(tmp_path, **cfg_kw)
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    runner._get_engine = lambda: _StubEngine()
    return runner


def _fresh_tick(m5, mid=95.0, spread=0.2, minutes_after_close=2):
    last_close = m5.index[-1] + pd.Timedelta(minutes=5)
    return {"mid": mid, "bid": mid - spread / 2.0,
            "ask": mid + spread / 2.0, "spread": spread,
            "time": last_close + pd.Timedelta(
                minutes=minutes_after_close)}


def _evals(runner, playbook=None):
    out = [r for r in runner.journal.records()
           if r.get("type") == "evaluation"]
    if playbook is not None:
        out = [r for r in out if r.get("playbook") == playbook]
    return out


def _last_decision(runner, playbook="qm_m5"):
    recs = [r for r in _evals(runner, playbook)
            if r.get("outcome") == "candidate_emitted"]
    assert recs, "expected a candidate funnel record"
    return recs[-1]


# ------------------------------------------------- (g) friction units

def test_friction_below_threshold_passes():
    v = assess_entry_friction(entry=100.0, stop=90.0,
                              spread_price=0.5)
    assert not v.refused and v.ratio == pytest.approx(0.05)


def test_friction_exactly_at_threshold_passes():
    # Refusal is STRICTLY above 0.10 — exactly at the bound passes.
    v = assess_entry_friction(entry=100.0, stop=90.0,
                              spread_price=1.0)
    assert v.ratio == pytest.approx(FRICTION_RATIO_LIMIT)
    assert not v.refused


def test_friction_above_threshold_refused_with_measurements():
    v = assess_entry_friction(entry=100.0, stop=90.0,
                              spread_price=1.5)
    assert v.refused and v.reason_code == "risk.friction"
    assert v.spread_price == 1.5 and v.stop_distance == 10.0
    assert v.ratio == pytest.approx(0.15)


def test_friction_missing_spread_fails_closed():
    v = assess_entry_friction(entry=100.0, stop=90.0)
    assert v.refused and "not measurable" in v.detail


def test_friction_invalid_stop_fails_closed_no_crash():
    for stop in (100.0, None, float("nan"), "x"):
        v = assess_entry_friction(entry=100.0, stop=stop,
                                  spread_price=0.3)
        assert v.refused


def test_friction_points_normalized_through_point_size():
    # XAUUSDm spec: point = 0.01.  30 points = 0.30 price — the
    # units trap: compared raw, 30/10 = 3.0 would refuse a healthy
    # entry; normalized, 0.30/10 = 0.03 passes.  And a genuinely
    # wide quote (300 points = 3.00 on a 10 stop) still refuses.
    ok = assess_entry_friction(entry=2000.0, stop=1990.0,
                               spread_points=30, point=0.01)
    assert not ok.refused and ok.spread_price == pytest.approx(0.30)
    wide = assess_entry_friction(entry=2000.0, stop=1990.0,
                                 spread_points=300, point=0.01)
    assert wide.refused and wide.ratio == pytest.approx(0.30)
    # Points without a point size cannot be normalized: fail closed.
    blind = assess_entry_friction(entry=2000.0, stop=1990.0,
                                  spread_points=30)
    assert blind.refused


# ---------------------------------------------------- (a)+(c) positive

def test_m5_positive_through_runner_on_m5_clock(tmp_path):
    runner = _qm_runner(tmp_path)
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    out = runner._run_qm_m5(m5, {})
    assert len(out) == 1 and out[0]["direction"] == -1
    # Traversed the shared order path: a real (fake-broker) order.
    assert len(runner.executor.orders) == 1
    assert runner.executor.orders[0]["alert_id"] == "MT5-qm_m5-31"
    # The signal record: accepted, sourced from the M5 route.
    sigs = [r for r in _signal_records(runner)
            if r.get("playbook") == "qm_m5"]
    assert sigs and sigs[-1]["decision"] == "accepted"
    assert sigs[-1]["trigger_source"] == "qm_m5"
    # The funnel record: M5 clock, M5 in == M5 intended, evidence
    # vector attached (observed vs inferred, tags, session).
    rec = _last_decision(runner)
    assert rec["evaluation_clock"] == "m5_close"
    assert rec["intended_tf"] == "M5"
    assert rec["actual_input_tf"] == "M5"
    assert rec["last_bar_ts"] is not None
    dec = rec["decision"]
    assert dec["result"] == "accepted"
    assert dec["order_result"] in ("filled", "placed_pending")
    ev = rec["evidence"]
    assert ev["observed"]["tags"] == ["displacement_break"]
    assert ev["observed"]["qm_level"] == 96.0
    assert ev["observed"]["session_bucket"]
    assert ev["inferred"]["score_total"] == 70.0
    assert dec["hard_risk_checks"]
    # (c) No H1 trigger was involved: nothing armed, nothing fired.
    assert runner._pending is None
    assert runner._fired_h1_bar is None


def test_m5_touch_deduped_second_scan_is_no_setup(tmp_path):
    runner = _qm_runner(tmp_path)
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    assert len(runner._run_qm_m5(m5, {})) == 1
    assert runner._run_qm_m5(m5, {}) == []
    assert len(runner.executor.orders) == 1
    outcomes = [r["outcome"] for r in _evals(runner, "qm_m5")]
    assert outcomes[-1] == "no_setup"


# ------------------------------------------------------- (b) negatives

def test_m5_flat_frame_no_setup_no_order(tmp_path):
    runner = _qm_runner(tmp_path)
    flat = _frame([(95.0, 95.5, 94.5, 95.0)] * 60,
                  start="2026-01-05 00:00", freq="5min")
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": flat})
    runner._last_tick = _fresh_tick(flat)
    assert runner._run_qm_m5(flat, {}) == []
    assert runner.executor.orders == []
    recs = _evals(runner, "qm_m5")
    assert recs and recs[-1]["outcome"] == "no_setup"
    assert recs[-1]["actual_input_tf"] == "M5"


def test_m5_stale_data_refused_gate_data(tmp_path):
    runner = _qm_runner(tmp_path)
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5, minutes_after_close=180)
    runner._run_qm_m5(m5, {})
    assert runner.executor.orders == []
    dec = _last_decision(runner)["decision"]
    assert dec["result"] == "refused"
    assert dec["reason_code"] == "gate.data"
    assert dec["reason_category"] == 1


def test_m5_kill_switch_not_eligible(tmp_path):
    runner = _qm_runner(tmp_path, disabled_playbooks=("qm_m5",))
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    assert runner._run_qm_m5(m5, {}) == []
    assert runner.executor.orders == []
    recs = _evals(runner, "qm_m5")
    assert recs[-1]["outcome"] == "not_eligible"
    assert recs[-1]["reason_code"] == "path.kill_switch"


def test_m5_margin_refusal_is_solvency(tmp_path):
    from test_p3_trader import _FakeExecutor
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    broke_feed = SimpleNamespace(
        account_state=lambda: {"equity": 12.0, "balance": 12.0,
                               "free_margin": 0.01},
        order_margin=lambda *, volume, price, direction:
            float(volume) * 100.0 * float(price) / 500.0)
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False)
    runner = MT5Runner(cfg, feed=broke_feed,
                       executor=_FakeExecutor())
    runner._send_alert = lambda *a, **k: None
    runner._build_chart = lambda *a, **k: (None, None)
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    runner._get_engine = lambda: _StubEngine()
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    runner._run_qm_m5(m5, {})
    assert runner.executor.orders == []
    dec = _last_decision(runner)["decision"]
    assert dec["order_result"] == "refused_in_order_path"
    assert dec["reason_code"] == "gate.solvency"
    assert dec["reason_category"] == 8


class _BadGeometryQM:
    """A QM candidate whose stop sits on the WRONG side of the
    entry (short with the stop below): no valid stop geometry."""
    name = "qm_m5"
    session_scope = "any"
    hunting_ground = SimpleNamespace(excluded=[], bonuses={})

    def candidates(self, ctx):
        # The bad-geometry touch lands on the frame's LAST bar
        # (the live-close shape — Phase 2 lifecycle: an older
        # touch would be refused as expired before geometry is
        # ever consulted, which is a different test).
        last = len(ctx.df) - 1
        sig = SimpleNamespace(
            direction=-1, index=last, score=70.0, entry_ref=96.0,
            invalidation=90.0, targets=[85.0], atr=2.0,
            factors=["qm"], kind="qm",
            live_context={"qm_level": 96.0, "head": 100.0,
                          "neckline": 90.0, "tags": []})
        yield SimpleNamespace(
            signal_index=last, direction=-1,
            outcome=SimpleNamespace(signal=sig, r_net=1.0,
                                    exit_reason="target"))


def test_m5_invalid_geometry_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "QMM5Playbook", _BadGeometryQM)
    runner = _qm_runner(tmp_path)
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    runner._run_qm_m5(m5, {})
    assert runner.executor.orders == []
    dec = _last_decision(runner)["decision"]
    assert dec["result"] == "refused"
    assert dec["reason_code"] == "gate.location"
    assert dec["reason_category"] == 3
    checks = {c["check"]: c["result"]
              for c in dec["hard_risk_checks"]}
    assert checks["stop_geometry"] == "VETO"


class _CostGateEngine(_StubEngine):
    def hard_line_check(self, pb, sig, ctx, spread_px=None):
        return False, "cost_gate: net TP1 R below the friction bound"


def test_m5_hard_line_friction_refused(tmp_path):
    runner = _qm_runner(tmp_path)
    runner._get_engine = lambda: _CostGateEngine()
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    runner._run_qm_m5(m5, {})
    assert runner.executor.orders == []
    dec = _last_decision(runner)["decision"]
    assert dec["result"] == "refused"
    assert dec["reason_code"] == "gate.friction"
    assert dec["reason_category"] == 8


# ------------------------------------------- (g) friction integration

def test_friction_gate_refuses_and_setup_stays_alive(tmp_path):
    runner = _qm_runner(tmp_path)
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    # Blowout: spread 10.0 against a ~4 stop -> ratio >> 0.10.
    runner._last_tick = _fresh_tick(m5, spread=10.0)
    out = runner._run_qm_m5(m5, {})
    assert len(out) == 1  # evaluated...
    assert runner.executor.orders == []  # ...but refused at the gate
    dec = _last_decision(runner)["decision"]
    assert dec["reason_code"] == "risk.friction"
    assert dec["reason_category"] == 8
    assert dec["order_result"] == "refused_in_order_path"
    notes = [r.get("text", "") for r in runner.journal.records()
             if r.get("type") == "note"]
    assert any("risk.friction" in n and "spread=" in n
               for n in notes)
    assert runner._blocked_leg_counts.get("friction_gate") == 1
    # Setup ALIVE: the dedupe cursor held below the touch...
    assert runner._qm_m5_last_logged < 31
    # ...so when friction normalizes, the SAME touch re-evaluates
    # and reaches the order path.
    runner._last_tick = _fresh_tick(m5, minutes_after_close=7)
    out2 = runner._run_qm_m5(m5, {})
    assert len(out2) == 1
    assert len(runner.executor.orders) == 1
    assert runner._qm_m5_last_logged == 31


# ------------------------------------------------------ (h) ruling 1

def test_m5_counter_bias_evaluated_not_suppressed(tmp_path):
    runner = _qm_runner(tmp_path)
    # The fixture is a SHORT; structural D1/H4 bias is LONG.
    runner._live_setup = {"bias_d1": 1, "bias_h4": 1,
                          "regime": "trend", "size_mult": 1.0,
                          "h4_context": ""}
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    out = runner._run_qm_m5(m5, {})
    assert len(out) == 1
    assert len(runner.executor.orders) == 1  # weighed, not vetoed
    ev = _last_decision(runner)["evidence"]
    assert any("against structural D1 bias" in c
               for c in ev["contradictory"])
    assert ev["observed"]["bias_d1"] == 1
    assert ev["observed"]["session_bucket"] in (
        "asia", "london", "overlap", "newyork", "off")


# --------------------------------------------- (d) funnel completeness

def test_funnel_completeness_reconciles(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "QMD1Playbook", _FakeQMD1)
    runner = _qm_runner(tmp_path, execute_playbooks=("fvg_midline",))
    h1 = _flat_h1(30, 95.0)
    m5 = _m5_fixture()
    d1 = _frame([(4100.0, 4110.0, 4090.0, 4105.0)] * 40, freq="1d")
    runner._stack = SimpleNamespace(
        frames={"H1": h1, "M5": m5, "D1": d1})
    runner._last_tick = _fresh_tick(m5)
    # Roster: one no-signal evaluation, one ordering evaluation.
    class _NoSig:
        name = "narrative"
        session_scope = "any"
        hunting_ground = SimpleNamespace(excluded=[], bonuses={})

        def live_signal(self, states, t, df=None):
            return []

        def check_live_signal(self, signal, ctx):
            return True, ""
    _drive_one(runner, _NoSig(), h1)
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
    # Native routes.
    runner._run_qm_m5(m5, {})
    runner._run_qm_d1(d1, {})
    summary = summarize_evaluations(runner.journal.records(),
                                    PLAYBOOK_NAMES)
    by = summary["by_playbook"]
    for name, entry in by.items():
        assert entry["evaluations"] == sum(
            entry["outcomes"].values()), name
    assert summary["frame_mismatches"] == 0
    assert by["qm_m5"]["by_clock"] == {"m5_close": 1}
    assert by["qm_d1"]["by_clock"] == {"d1_close": 1}
    assert by["narrative"]["outcomes"].get("no_setup") == 1
    assert by["fvg_midline"]["candidates"] == 1
    # Registered but never driven here: reported, not absent.
    assert "sweep_reversal" in summary["not_called"]
    assert set(summary["not_called"]) | set(by) == set(PLAYBOOK_NAMES)


# ------------------------------------------------------ (e) registry

def test_registry_consistency_and_killability(tmp_path):
    assert len(PLAYBOOK_NAMES) == 18
    assert {s.name for s in ROSTER_SPECS} | {
        s.name for s in NATIVE_SPECS} == set(PLAYBOOK_NAMES)
    # Every advertised control maps to a real playbook.
    for name in PLAYBOOK_NAMES:
        change, err = parse_set(f"playbook.{name}", "off")
        assert err is None
        assert change == {"field": "playbook", "name": name,
                          "value": False}
    # Persistence round-trip: all 18 kill entries survive; an
    # unknown name is dropped by the cleaner and REPORTED by the
    # detector the runner journals at startup.
    store = LiveConfigStore(tmp_path / "live.json")
    store.save({"disabled_playbooks":
                sorted(PLAYBOOK_NAMES) + ["bogus_pb"]})
    loaded = store.load()
    assert set(loaded["disabled_playbooks"]) == set(PLAYBOOK_NAMES)
    assert unknown_disabled_names(
        {"disabled_playbooks":
         sorted(PLAYBOOK_NAMES) + ["bogus_pb"]}) == ["bogus_pb"]
    # Roster identity: the runner builds exactly the registry's
    # roster, in registry order.
    runner = _qm_runner(tmp_path)
    assert [pb.name for pb in runner._get_playbooks()] == [
        s.name for s in ROSTER_SPECS]


def test_range_playbook_kill_switch_records_not_eligible(tmp_path):
    # The verified §3 defect: range_qm was unkillable (not in the
    # config whitelist).  Now: disabled -> the signal path emits a
    # per-playbook not_eligible record and never evaluates it.
    runner = _qm_runner(tmp_path, disabled_playbooks=("range_qm",))
    h1 = _flat_h1(30, 95.0)
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._last_tick = _fresh_tick(h1, mid=95.0)
    stats = {"signals": 0, "gated": 0, "executed": 0}
    runner._run_signal_path(
        h1, len(h1) - 1, [], 2.0,
        {"bias_d1": 1, "bias_h4": 1, "regime": "trend",
         "direction": 1, "trigger": "tick-cross",
         "entry_clock": "M15", "amd_phase": "accumulation",
         "qm": {"status": "none"}, "volume": {}}, stats)
    recs = _evals(runner, "range_qm")
    assert recs and recs[-1]["outcome"] == "not_eligible"
    assert recs[-1]["reason_code"] == "path.kill_switch"


# ------------------------------------------- (f) observation-only proof

def test_funnel_observation_only_decisions_identical(tmp_path):
    def drive(enabled):
        runner = _qm_runner(
            tmp_path / ("on" if enabled else "off"),
            execute_playbooks=("fvg_midline",),
            funnel_enabled=enabled)
        h1 = _flat_h1(30, 2000.0)
        runner._stack = SimpleNamespace(frames={"H1": h1})
        runner._last_tick = {
            "mid": 2000.0, "bid": 1999.8, "ask": 2000.2,
            "spread": 0.4,
            "time": h1.index[-1] + pd.Timedelta(hours=1)}
        _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
        return runner

    on = drive(True)
    off = drive(False)
    seq = lambda r: [(x.get("decision"), x.get("reason"))
                     for x in _signal_records(r)]
    assert seq(on) == seq(off)
    assert len(on.executor.orders) == len(off.executor.orders) == 1
    assert _evals(on, "fvg_midline")
    assert _evals(off, "fvg_midline") == []


# ------------------------------------------------- (i) structural pins

def test_structural_no_h1_frame_into_qm_m5():
    src = Path(mr.__file__).read_text()
    tree = ast.parse(src)
    fn_names = {n.name for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)}
    assert "_advance_qm" not in fn_names  # annotation scanner gone
    assert "_run_qm_m5" in fn_names
    get_pb = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_get_playbooks")
    assert "QMM5Playbook" not in ast.dump(get_pb)
    assert "ROSTER_SPECS" in ast.dump(get_pb)


def test_structural_qm_m5_only_sees_five_minute_bars(
        tmp_path, monkeypatch):
    spacings = []

    class _RecordingQM(mr.QMM5Playbook):
        def candidates(self, ctx):
            idx = ctx.df.index
            if len(idx) > 1:
                deltas = pd.Series(idx).diff().dropna()
                spacings.append(deltas.median())
            return super().candidates(ctx)

    monkeypatch.setattr(mr, "QMM5Playbook", _RecordingQM)
    runner = _qm_runner(tmp_path)
    h1 = _flat_h1(30, 95.0)
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(frames={"H1": h1, "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    # Drive BOTH the H1 signal path (real roster) and the route.
    stats = {"signals": 0, "gated": 0, "executed": 0}
    runner._run_signal_path(
        h1, len(h1) - 1, [], 2.0,
        {"bias_d1": 1, "bias_h4": 1, "regime": "trend",
         "direction": 1, "trigger": "tick-cross",
         "entry_clock": "M15", "amd_phase": "accumulation",
         "qm": {"status": "none"}, "volume": {}}, stats)
    runner._run_qm_m5(m5, {})
    assert spacings, "the playbook was never evaluated"
    assert all(s == pd.Timedelta(minutes=5) for s in spacings)
