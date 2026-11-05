"""Judgment Core PROMOTION (owner decision 2026-11-05): the core
is the DEMO decision authority.

Covers:
(a) the 2026-11-04 12:30 journal shape as a fixture — the core
    abstained (reversal-led, negative expected R) while the old
    gate accepted and the broker filled.  In ACTIVE mode that
    order no longer exists; in SHADOW mode it still does (the
    pre-promotion behavior, byte-identical on the decision path).
(b) a core "take" is necessary, never sufficient: take +
    friction breach is refused by the FRICTION gate under its
    own reason (risk.friction), never relabeled as a judgment
    refusal — and the journal shows the take it rode on.
(c) take + safeguards pass -> the order proceeds identically
    to the pre-promotion path (same terminal record, same
    broker order; only the judgment record's mode differs).
(d) no-bypass pin: a scenario run across the governed routes
    (H1 roster trigger path, qm_m5, qm_d1) — every placed order
    is matched by a journaled core "take" in active mode, every
    abstention places nothing; plus a structural AST pin that
    each governed route's order dispatch sits AFTER its
    _judgment_authority call.
(e) rollback: judgment_mode="shadow" restores the
    pre-promotion decision path exactly (an abstention is
    observed, the gate stack decides, orders flow).
Plus: the kill switch itself (default active; an unrecognized
value can never silently restore the unjudged path), the
manifest's mode journaling, the owner-confirm staging branch
(governed at staging time — the placement-time decision binds
the later confirm execution), and the qm_d1 evidence standard
(the route's records carried {route, provenance} only; under
active authority that starved every candidate into a
thin-evidence abstention — the route now journals the evidence
it actually weighs).
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import goldenswing.execution.mt5_runner as mr
from goldenswing.execution.mt5_runner import MT5RunnerConfig

from test_p3_trader import (
    _FakeQMD1, _FakeScore, _StubEngine, _StubPB, _drive_one,
    _flat_h1, _frame, _runner, _sig, _signal_records)
from test_funnel_m5 import (
    _evals, _fresh_tick, _last_decision, _m5_fixture, _qm_runner)


class _FlippedScore(_FakeScore):
    """A score whose weighed structure funds the OPPOSITE side
    (the repriced flip fact) — the 2026-11-04 12:30 shape's core
    input: the asymmetry group votes against, the reversal
    scenario leads, expected R goes negative."""
    flipped = True


def _shadows(runner):
    return [r for r in runner.journal.records()
            if r.get("type") == "judgment_shadow"]


def _candidate_evals(runner, playbook=None):
    return [r for r in _evals(runner, playbook)
            if r.get("outcome") == "candidate_emitted"]


# ------------------------------------------------- (a) 12:30 shape

def _drive_qm_flipped(tmp_path, **cfg_kw):
    runner = _qm_runner(tmp_path, **cfg_kw)
    runner._score_for = lambda **kw: _FlippedScore(40.0, 25.0, 15.0)
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    out = runner._run_qm_m5(m5, {})
    return runner, out


def test_active_abstain_refuses_the_order_that_filled_2026_11_04(
        tmp_path):
    """The journal's 2026-11-04 12:30 qm_m5 record: gate
    accepted + filled, core abstained at -0.80R (reversal-led).
    Under active authority the same shape produces NO order and
    a judgment refusal carrying the core's decision + expected
    R on the record."""
    runner, out = _drive_qm_flipped(tmp_path / "a")
    assert len(out) == 1  # the touch was evaluated
    assert runner.executor.orders == []
    dec = _last_decision(runner)["decision"]
    assert dec["result"] == "refused"
    assert dec["reason_code"] == "judgment.abstain"
    assert dec["order_result"] == "not_attempted"
    # The refusal is categorized by the CORE's own category for
    # this abstention (direction — reversal-led), not flattened.
    assert dec["reason_category"] == 5
    checks = {c["check"]: c["result"]
              for c in dec["hard_risk_checks"]}
    assert checks["judgment_core (active authority)"] == "VETO"
    shadow = _shadows(runner)
    assert len(shadow) == 1
    j = shadow[0]["judgment"]
    assert j["decision"] == "abstain"
    assert j["mode"] == "active"
    assert j["expected_r"] < 0
    assert shadow[0]["gate_result"] == "refused"
    assert shadow[0]["gate_order_result"] == "not_attempted"


def test_shadow_mode_same_shape_still_fills(tmp_path):
    """The SAME fixture under shadow mode: the core abstains,
    the old gate accepts, the order fills — the 2026-11-04
    12:30 record exactly, and the rollback proof for (a)."""
    runner, out = _drive_qm_flipped(tmp_path / "s",
                                    judgment_mode="shadow")
    assert len(out) == 1
    assert len(runner.executor.orders) == 1
    dec = _last_decision(runner)["decision"]
    assert dec["result"] == "accepted"
    j = _shadows(runner)[0]["judgment"]
    assert j["decision"] == "abstain"
    assert j["mode"] == "shadow"
    assert j["expected_r"] < 0


def test_active_counter_bias_abstains(tmp_path):
    """RULING 1's weighing, decided: a counter-D1-bias candidate
    is evaluated by the route (never suppressed) — and under
    active authority the core, weighing that same evidence,
    abstains.  (The route-level weighing itself is pinned in
    test_funnel_m5 under shadow mode.)"""
    runner = _qm_runner(tmp_path)
    runner._live_setup = {"bias_d1": 1, "bias_h4": 1,
                          "regime": "trend", "size_mult": 1.0,
                          "h4_context": ""}
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    out = runner._run_qm_m5(m5, {})
    assert len(out) == 1
    assert runner.executor.orders == []
    dec = _last_decision(runner)["decision"]
    assert dec["reason_code"] == "judgment.abstain"
    j = _shadows(runner)[0]["judgment"]
    assert j["decision"] == "abstain"
    assert j["mode"] == "active"


# --------------------------------------- (b) take != sufficient

def test_take_then_friction_refuses_under_its_own_reason(tmp_path):
    runner = _runner(tmp_path, execute_playbooks=("fvg_midline",))
    h1 = _flat_h1()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    # Spread 20.0 over a 10.0 stop = 2.0 >> the 0.10 bound.
    runner._last_tick = {"mid": 2000.0, "bid": 1990.0,
                         "ask": 2010.0, "spread": 20.0,
                         "time": "2026-10-14 12:00:00"}
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
    assert runner.executor.orders == []
    dec = _candidate_evals(runner, "fvg_midline")[-1]["decision"]
    assert dec["result"] == "refused"
    assert dec["order_result"] == "refused_in_order_path"
    assert dec["reason_code"] == "risk.friction"
    assert dec["reason_category"] == 8  # hard risk, not judgment
    j = _shadows(runner)[0]["judgment"]
    assert j["decision"] == "take"
    assert j["mode"] == "active"
    assert j["expected_r"] > 0


# --------------------------------------- (c) take -> identical

def _strip(records):
    strip = ("ts", "run_id", "commit", "config_fingerprint",
             "assessment_text")
    out = []
    for r in records:
        r = dict(r)
        for key in strip:
            r.pop(key, None)
        out.append(r)
    return out


def test_take_proceeds_identically_active_vs_shadow(tmp_path):
    """Roster take in both modes: same broker order, same
    terminal evaluation record on the decision path — the ONLY
    difference in the journal is the judgment record's mode."""
    journals = {}
    for mode in ("shadow", "active"):
        runner = _runner(tmp_path / mode,
                         execute_playbooks=("fvg_midline",),
                         judgment_mode=mode)
        h1 = _flat_h1()
        runner._stack = SimpleNamespace(frames={"H1": h1})
        runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
        _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
        assert len(runner.executor.orders) == 1
        journals[mode] = runner.journal.records()
        j = _shadows(runner)[0]["judgment"]
        assert j["decision"] == "take"
        assert j["mode"] == mode
    ev_shadow = _strip([r for r in journals["shadow"]
                        if r["type"] == "evaluation"])
    ev_active = _strip([r for r in journals["active"]
                        if r["type"] == "evaluation"])
    assert ev_shadow == ev_active


# ------------------------------------------------- (d) no bypass

def test_every_order_rides_on_a_journaled_take(tmp_path):
    """Scenario run across the three governed routes in ACTIVE
    mode: takes place orders, abstentions place nothing, and
    every terminal ordered record is matched by a judgment
    record carrying decision=take + mode=active for the same
    playbook."""
    ordered_records = []

    # Roster: take -> order.
    runner = _runner(tmp_path / "roster",
                     execute_playbooks=("fvg_midline",))
    h1 = _flat_h1()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
    assert len(runner.executor.orders) == 1
    ordered_records += [(runner, r) for r in
                        _candidate_evals(runner, "fvg_midline")
                        if r["decision"].get("order_result")
                        in ("filled", "placed_pending")]

    # Roster: forced abstain (threshold) -> no order.
    runner = _runner(tmp_path / "roster_abstain",
                     execute_playbooks=("fvg_midline",),
                     judgment_min_expected_r=99.0)
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
    assert runner.executor.orders == []
    assert _candidate_evals(runner, "fvg_midline")[-1][
        "decision"]["reason_code"] == "judgment.abstain"

    # qm_m5: take -> order; flipped abstain -> none.
    runner = _qm_runner(tmp_path / "qm_take")
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    runner._run_qm_m5(m5, {})
    assert len(runner.executor.orders) == 1
    ordered_records += [(runner, r) for r in
                        _candidate_evals(runner, "qm_m5")
                        if r["decision"].get("order_result")
                        in ("filled", "placed_pending")]
    runner2, _ = _drive_qm_flipped(tmp_path / "qm_abstain")
    assert runner2.executor.orders == []

    # qm_d1: take -> order (its evidence vector is real now);
    # forced abstain -> none.
    for label, cfg, expect_orders in (
            ("d1_take", {}, 1),
            ("d1_abstain", {"judgment_min_expected_r": 99.0}, 0)):
        mp = pytest.MonkeyPatch.context()
        with mp as m:
            m.setattr(mr, "QMD1Playbook", _FakeQMD1)
            runner = _runner(tmp_path / label, **cfg)
            d1 = _frame([(4100.0, 4110.0, 4090.0, 4105.0)] * 40,
                        freq="1d")
            runner._stack = SimpleNamespace(
                frames={"H1": _flat_h1(30, 4100.0), "D1": d1})
            runner._last_tick = {
                "mid": 4100.0, "bid": 4099.8, "ask": 4100.2,
                "spread": 30, "time": "2026-10-14 12:00:00"}
            runner._score_for = lambda **kw: _FakeScore(
                70.0, 25.0, 15.0)
            runner._get_engine = lambda: _StubEngine()
            runner._run_qm_d1(d1, {})
            assert len(runner.executor.orders) == expect_orders
            if expect_orders:
                ordered_records += [
                    (runner, r) for r in
                    _candidate_evals(runner, "qm_d1")
                    if r["decision"].get("order_result")
                    in ("filled", "placed_pending")]
            else:
                assert _candidate_evals(runner, "qm_d1")[-1][
                    "decision"]["reason_code"] == "judgment.abstain"

    # The pin itself: at least one ordered record per governed
    # route, and EVERY ordered record's playbook has a take in
    # active mode journaled in the same run.
    assert {r["playbook"] for _, r in ordered_records} == {
        "fvg_midline", "qm_m5", "qm_d1"}
    for runner, rec in ordered_records:
        takes = [s for s in _shadows(runner)
                 if s["playbook"] == rec["playbook"]
                 and s["judgment"]["decision"] == "take"
                 and s["judgment"]["mode"] == "active"]
        assert takes, (f"order for {rec['playbook']} has no "
                       "journaled core take in active mode")


def test_authority_seam_structurally_precedes_dispatch():
    """AST pin: in each governed route's function body, the
    _judgment_authority call precedes every order dispatch
    (_place_live_order / _try_layer / _stage_order_confirm) —
    a future edit cannot quietly re-open a bypass."""
    src = Path(mr.__file__).read_text()
    tree = ast.parse(src)
    governed = {
        "_run_one_playbook": ("_place_live_order", "_try_layer",
                              "_stage_order_confirm"),
        "_execute_qm_m5_candidate": ("_place_live_order",
                                     "_try_layer"),
        "_execute_qm_d1_candidate": ("_place_live_order",
                                     "_try_layer"),
    }
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) \
                and node.name in governed:
            seg = ast.get_source_segment(src, node)
            seam = seg.index("_judgment_authority(")
            for dispatch in governed[node.name]:
                idx = seg.find(f".{dispatch}(")
                if idx >= 0:
                    assert seam < idx, (
                        f"{node.name}: {dispatch} precedes the "
                        "judgment authority seam")
            found[node.name] = True
    assert found == {name: True for name in governed}


# --------------------------------------------- kill switch + co.

def test_mode_default_active_and_typo_never_shadows(tmp_path):
    assert MT5RunnerConfig().judgment_mode == "active"
    assert _runner(tmp_path / "a")._judgment_mode() == "active"
    assert _runner(tmp_path / "b",
                   judgment_mode="shadow")._judgment_mode() \
        == "shadow"
    assert _runner(tmp_path / "c",
                   judgment_mode=" SHADOW ")._judgment_mode() \
        == "shadow"
    # A typo'd rollback value fails toward authority, never
    # silently back to the unjudged path.
    assert _runner(tmp_path / "d",
                   judgment_mode="shdow")._judgment_mode() \
        == "active"


def test_manifest_journals_mode_and_boundary(tmp_path):
    runner = _runner(tmp_path / "m")
    runner._journal_manifest()
    manifests = [r for r in runner.journal.records()
                 if r.get("type") == "manifest"]
    assert manifests
    core = manifests[-1]["doctrine"]["judgment_core"]
    assert core["mode"] == "active"
    assert core["mode_configured"] == "active"
    assert core["authority_routes"] == [
        "h1_trigger", "qm_m5_native", "qm_d1_native"]
    assert core["not_authoritative"] == [
        "hot_scalp_promotion", "resting_limit"]
    runner = _runner(tmp_path / "m2", judgment_mode="shadow")
    runner._journal_manifest()
    core = [r for r in runner.journal.records()
            if r.get("type") == "manifest"][-1][
                "doctrine"]["judgment_core"]
    assert core["mode"] == "shadow"


def test_confirm_staging_is_governed_at_staging_time(tmp_path):
    """The Golden Swing confirm branch stages (does not place)
    the order; the seam sits before the branch, so an abstain
    stages nothing and a take stages — the placement-time
    decision binds the later confirm execution."""
    for label, cfg, staged in (
            ("take", {}, 1),
            ("abstain", {"judgment_min_expected_r": 99.0}, 0)):
        runner = _runner(
            tmp_path / label,
            execute_playbooks=("fvg_midline",),
            golden_swing_confirm_enabled=True, **cfg)
        h1 = _flat_h1()
        runner._stack = SimpleNamespace(frames={"H1": h1})
        runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
        runner._alignment = SimpleNamespace(
            golden_swing=True, size_multiplier=1.0,
            alignment_label=None)
        _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
        assert len(runner._order_confirms) == staged
        assert runner.executor.orders == []
        if not staged:
            assert _candidate_evals(runner, "fvg_midline")[-1][
                "decision"]["reason_code"] == "judgment.abstain"
