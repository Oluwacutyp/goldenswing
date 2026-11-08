"""Route AUTHORITY (MarketSituation Pass 5; owner ruling
2026-11-08, AMENDMENTS-1 §A5 — the strict boundary).

Runner-level pins:

- COMPLETENESS: every route named anywhere in the runner
  (seam calls, book stamps, state lookups) is declared in
  JUDGMENT_ROUTE_STATES, and every declared route is
  reachable from a call site, the playbook map, or the §G
  stamp map — no route left unstamped, no stamp orphaned
  (AST audit over the runner source, both directions).
- FAIL-CLOSED: a route with NO declaration cannot reach the
  order path — the seam refuses it in every mode and the
  attempt journals an undeclared_route alarm.
- RESOLUTION: state + source resolve declared < config <
  verdict, with an invariant demotion binding over all
  three until released by a fresh verdict or a NEW config
  step.
- THE PER-CYCLE RECORD: one judgment_authority record per
  H1 close listing every route + state + source; the
  /status line renders the same table.
- THE BOUNDARY: hot_scalp and the resting limit submit NO
  independent demo orders while their route does not
  govern — candidates are preserved as shadow (claim back
  to touched / a journaled shadow eval), never placed;
  under a governing verdict the order flows THROUGH the
  seam, so a core abstention still blocks it.
- THE FULL LOOP on fabricated history through the S3 host:
  criteria journal BEFORE the first verdict; a ready
  verdict binds (route governs, source verdict); a
  violation demotes; the STALE verdict does not release the
  demotion; fresh history does; a later not_ready verdict
  demotes a verdict-promoted route.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from goldenswing.execution import range_promotion as _range_mod
from goldenswing.execution import route_promotion as rp
from goldenswing.execution.mt5_runner import (
    JUDGMENT_ROUTE_PLAYBOOK, JUDGMENT_ROUTE_STATES, MT5Runner)
from goldenswing.live.zones import StructuralZone

from test_confluence_rework import _point3_frames
from test_confluence_rework import _runner as _az_runner
from test_judgment_core import _candidate, _decision, _evidence
from test_judgment_core import _runner as _jc_runner
from test_route_promotion import _arm as _history_arm
from test_step14_setup_lifecycle import _scan_runner
from test_step6_hot_scalp_promotion import _candidate as _hs_cand
from test_step6_hot_scalp_promotion import _runner as _hs_runner
from test_step6_hot_scalp_promotion import _status as _hs_status
from test_step6_hot_scalp_promotion import _tick as _hs_tick


def _records(runner, type_=None):
    rows = runner.journal.records()
    if type_ is not None:
        rows = [r for r in rows if r.get("type") == type_]
    return rows


def _evals(runner, playbook):
    return [r for r in runner.journal.records()
            if r.get("type") == "evaluation"
            and r.get("playbook") == playbook]


# ------------------------------------------------ the audit

def _runner_source_tree():
    path = Path(MT5Runner.__module__.replace(".", "/") + ".py")
    # Resolve against the repo root (tests run from it).
    path = Path.cwd() / path
    return ast.parse(path.read_text()), path.read_text()


def _route_literals():
    tree, _ = _runner_source_tree()
    literals = set()
    seam_routes = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(
            func, "id", None)
        for kw in node.keywords:
            if kw.arg in ("route", "judgment_route") \
                    and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                literals.add(kw.value.value)
                if name == "_judgment_authority":
                    seam_routes.add(kw.value.value)
        if name in ("_judgment_route_state",
                    "_judgment_route_source",
                    "_judgment_route_resolution") \
                and node.args \
                and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            literals.add(node.args[0].value)
    return literals, seam_routes


def test_route_audit_completeness_both_directions():
    literals, seam_routes = _route_literals()
    table = set(JUDGMENT_ROUTE_STATES)
    # Every route the runner names is declared.
    assert literals <= table, (
        f"routes named in the runner but not declared: "
        f"{sorted(literals - table)}")
    # Every declared route is reachable: a call-site literal,
    # the playbook map, or the §G route map.
    reachable = literals | set(JUDGMENT_ROUTE_PLAYBOOK) \
        | set(_range_mod.RANGE_JUDGMENT_ROUTES.values())
    assert table <= reachable, (
        f"declared routes no call site can reach: "
        f"{sorted(table - reachable)}")
    # The three owner-promoted routes govern THROUGH the seam
    # with their route named at the call site.
    assert {"h1_trigger", "qm_m5_native", "qm_d1_native"} \
        <= seam_routes
    # The playbook map and the binding evaluation agree.
    for route, playbook in rp.PROMOTION_ROUTES.items():
        assert JUDGMENT_ROUTE_PLAYBOOK[route] == playbook
        assert rp.ORDER_ROUTE_BY_PLAYBOOK[playbook] == route


def test_declared_table_is_the_owners_promotion_record():
    assert JUDGMENT_ROUTE_STATES["h1_trigger"] == "governing"
    assert JUDGMENT_ROUTE_STATES["qm_m5_native"] == "governing"
    assert JUDGMENT_ROUTE_STATES["qm_d1_native"] == "governing"
    assert JUDGMENT_ROUTE_STATES["hot_scalp_promotion"] \
        == "shadow_declared"
    assert JUDGMENT_ROUTE_STATES["resting_limit"] \
        == "shadow_declared"


# ------------------------------------------------ resolution

def test_resolution_sources(tmp_path):
    runner = _jc_runner(tmp_path / "a")
    assert runner._judgment_route_state("h1_trigger") \
        == "governing"
    assert runner._judgment_route_source("h1_trigger") \
        == "declared"
    assert runner._judgment_route_state("hot_scalp_promotion") \
        == "shadow_declared"
    assert runner._route_declared("hot_scalp_promotion")
    # Config is the explicit step.
    runner = _jc_runner(
        tmp_path / "b",
        judgment_route_modes={"hot_scalp_promotion":
                              "governing"})
    assert runner._judgment_route_state("hot_scalp_promotion") \
        == "governing"
    assert runner._judgment_route_source(
        "hot_scalp_promotion") == "config"
    # A verdict entry governs with source verdict.
    runner = _jc_runner(tmp_path / "c")
    runner._route_authority_state()["hot_scalp_promotion"] = {
        "state": "governing", "source": "verdict",
        "criteria_id": rp.criteria_id_for("hot_scalp_promotion"),
        "since": "2026-11-08 00:00:00+00:00"}
    assert runner._judgment_route_state("hot_scalp_promotion") \
        == "governing"
    assert runner._judgment_route_source(
        "hot_scalp_promotion") == "verdict"
    # A demotion binds over the declared state.
    runner._route_authority_state()["h1_trigger"] = {
        "state": "shadow_declared", "source": "demoted",
        "kind": "duplicate_order",
        "config_at_demotion": None,
        "eval_cycles_at_demotion": 0,
        "since": "2026-11-08 01:00:00+00:00"}
    assert runner._judgment_route_state("h1_trigger") \
        == "shadow_declared"
    assert runner._judgment_route_source("h1_trigger") \
        == "demoted"


def test_undeclared_route_is_shadow_and_undeclared(tmp_path):
    runner = _jc_runner(tmp_path)
    assert runner._judgment_route_state("nope_route") \
        == "shadow_declared"
    assert runner._judgment_route_source("nope_route") \
        == "undeclared"
    assert not runner._route_declared("nope_route")
    table = {r["route"] for r in runner._judgment_route_table()}
    assert "nope_route" not in table
    assert table == set(JUDGMENT_ROUTE_STATES)


def test_undeclared_route_cannot_pass_the_seam(tmp_path):
    for mode in ("active", "shadow"):
        runner = _jc_runner(tmp_path / mode,
                            judgment_mode=mode)
        bundle, refusal = runner._judgment_authority(
            candidate=_candidate(), decision=_decision(),
            evidence=_evidence(), intended_tf="H1",
            eval_now=runner._evaluation_now(),
            route="nope_route")
        assert refusal is not None
        assert "undeclared" in refusal
        alarms = _records(runner, "undeclared_route")
        assert len(alarms) == 1
        assert alarms[0]["route"] == "nope_route"
        # The alarm fires once per route per run, not per
        # attempt.
        runner._judgment_authority(
            candidate=_candidate(), decision=_decision(),
            evidence=_evidence(), intended_tf="H1",
            eval_now=runner._evaluation_now(),
            route="nope_route")
        assert len(_records(runner, "undeclared_route")) == 1


def test_governing_route_seam_stamps_unchanged(tmp_path):
    runner = _jc_runner(tmp_path)
    bundle, _refusal = runner._judgment_authority(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(), intended_tf="M5",
        eval_now=runner._evaluation_now(),
        route="qm_m5_native")
    judgment = (bundle or {}).get("judgment") or {}
    assert judgment.get("authority_route") == "qm_m5_native"
    assert judgment.get("authority") == "governing"


# ------------------------------------------------ the cycle record

def test_authority_cycle_journals_once_per_h1_close(tmp_path):
    runner = _jc_runner(tmp_path)
    runner._authority_cycle(["M5"], "2026-11-06 00:00:00+00:00")
    assert _records(runner, "judgment_authority") == []
    runner._authority_cycle(
        ["H1", "M5"], "2026-11-06 01:00:00+00:00")
    rows = _records(runner, "judgment_authority")
    assert len(rows) == 1
    table = {r["route"]: r for r in rows[0]["routes"]}
    assert set(table) == set(JUDGMENT_ROUTE_STATES)
    assert table["h1_trigger"]["authority"] == "governing"
    assert table["h1_trigger"]["source"] == "declared"
    assert table["h1_trigger"]["governing_now"] is True
    assert table["hot_scalp_promotion"]["authority"] \
        == "shadow_declared"
    assert table["hot_scalp_promotion"]["governing_now"] \
        is False


def test_status_line_renders_the_same_table(tmp_path):
    runner = _jc_runner(tmp_path)
    line = runner.judgment_authority_line()
    for route in JUDGMENT_ROUTE_STATES:
        assert route in line
    assert "source: declared" in line
    assert "hot_scalp_promotion" in line
    # The commands /status surface wires the same line in.
    commands = (Path.cwd() / "goldenswing" / "execution"
                / "mt5_commands.py").read_text()
    assert "judgment_authority_line" in commands
    assert "Judgment Authority" in commands


# ------------------------------------------------ the boundary: hot scalp

def _promote_via_verdict(runner):
    runner._route_authority_state()["hot_scalp_promotion"] = {
        "state": "governing", "source": "verdict",
        "criteria_id": rp.criteria_id_for("hot_scalp_promotion"),
        "verdict": "promotion_ready",
        "since": "2026-11-08 00:00:00+00:00"}


def test_hot_scalp_boundary_preserves_shadow_at_default(
        tmp_path):
    runner = _hs_runner(tmp_path)
    placed_before = len(runner.executor.placed)
    assert runner._promote_hot_scalp_candidate(
        _hs_cand(runner), _hs_tick(), {}) is False
    # No independent demo order left the process.
    assert len(runner.executor.placed) == placed_before
    # The candidate is preserved, not consumed: the claim is
    # back to touched and the shadow history carries the
    # verdict + the seam's shadow evaluation.
    assert _hs_status(runner) == "touched"
    evals = _evals(runner, "hot_scalp")
    assert evals and evals[-1]["decision"]["reason_code"] \
        == "judgment.route_not_promoted"
    assert evals[-1]["decision"]["order_result"] \
        == "not_attempted"
    shadows = [r for r in runner.journal.records()
               if r.get("type") == "judgment_shadow"
               and r.get("playbook") == "hot_scalp"]
    assert shadows, "the boundary keeps the shadow history"
    chain = [r for r in runner.journal.records()
             if r.get("type") == "candidate_chain"
             and r.get("stage") == "authority"]
    assert chain and chain[-1]["execution_status"] \
        == "shadow_preserved"


def test_hot_scalp_promoted_submits_through_the_seam(tmp_path):
    runner = _hs_runner(tmp_path / "promoted")
    (tmp_path / "promoted").mkdir(exist_ok=True)
    _promote_via_verdict(runner)
    assert runner._promote_hot_scalp_candidate(
        _hs_cand(runner), _hs_tick(), {}) is True
    assert len(runner.executor.placed) == 1
    # And through the seam means the seam still governs: a
    # core abstention blocks the promoted route exactly like
    # every governed route.
    runner = _hs_runner(tmp_path / "abstain")
    (tmp_path / "abstain").mkdir(exist_ok=True)
    runner.cfg.judgment_min_expected_r = 99.0
    _promote_via_verdict(runner)
    assert runner._promote_hot_scalp_candidate(
        _hs_cand(runner), _hs_tick(), {}) is False
    assert runner.executor.placed == []
    evals = _evals(runner, "hot_scalp")
    assert evals[-1]["decision"]["reason_code"] \
        == "judgment.abstain"


# ------------------------------------------------ the boundary: resting limit

_AZ_TICK = {"mid": 4122.0 + 3.6, "spread": 0.3,
            "time": pd.Timestamp("2026-10-21 13:00",
                                 tz="UTC")}


def _az_drive(tmp_path, **cfg_kw):
    runner = _az_runner(tmp_path, resting_limit_enabled=True)
    for key, value in cfg_kw.items():
        setattr(runner.cfg, key, value)
    from types import SimpleNamespace

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
        "zone": {"top": 4122.0, "bottom": 4098.0,
                 "level": 4110.0},
        "bias_d1": 0, "bias_h4": 1}
    runner._maybe_place_resting_limit(dict(_AZ_TICK), {})
    return runner


def test_resting_limit_boundary_rests_nothing_at_default(
        tmp_path):
    runner = _az_drive(tmp_path)
    assert runner._pending_orders == {}
    evals = _evals(runner, "arrival_zone")
    assert len(evals) == 1
    assert evals[0]["decision"]["reason_code"] \
        == "judgment.route_not_promoted"
    assert evals[0]["decision"]["order_result"] \
        == "not_attempted"
    # The shadow journaling is deduped per zone+tick state:
    # a second call inside the same approach adds no second
    # record (the attempt is still refused, silently).
    runner._maybe_place_resting_limit(dict(_AZ_TICK), {})
    assert runner._pending_orders == {}
    assert len(_evals(runner, "arrival_zone")) == 1


def test_resting_limit_config_promotion_still_rests(tmp_path):
    runner = _az_drive(
        tmp_path,
        judgment_route_modes={"resting_limit": "governing"})
    assert runner._pending_orders != {}


# ------------------------------------------------ the full loop (S3 host)

def _ready_hot_history(start=0):
    records = _history_arm("hot_scalp", "hot_scalp_promotion",
                           "take", [0.5] * 18 + [-0.5] * 4,
                           start=start)
    records += _history_arm("hot_scalp", "hot_scalp_promotion",
                            "abstain",
                            [-0.4] * 15 + [0.3] * 6,
                            start=start + 100)
    return records


def _prime(runner, records):
    for record in records:
        runner.journal.append(record)


def test_full_loop_verdict_binds_demotes_and_releases(
        tmp_path):
    runner, h1 = _scan_runner(tmp_path)
    _prime(runner, _ready_hot_history())
    runner._self_audit_phase("2026-11-06 00:00:00+00:00", h1)
    records = runner.journal.records()
    criteria_at = [i for i, r in enumerate(records)
                   if r.get("type")
                   == rp.CRITERIA_RECORD_TYPE]
    evaluation_at = [i for i, r in enumerate(records)
                     if r.get("type")
                     == rp.EVALUATION_RECORD_TYPE]
    # A5a ordering, in the journal itself: every criteria
    # record precedes the first verdict computed under it.
    assert criteria_at and evaluation_at
    assert max(criteria_at) < min(evaluation_at)
    evaluation = records[evaluation_at[0]]
    row = evaluation["routes"]["hot_scalp_promotion"]
    assert row["verdict"] == "promotion_ready"
    assert row["criteria_id"] == rp.criteria_id_for(
        "hot_scalp_promotion")
    # The verdict BINDS: the route governs, source verdict.
    assert runner._judgment_route_state("hot_scalp_promotion") \
        == "governing"
    assert runner._judgment_route_source(
        "hot_scalp_promotion") == "verdict"
    changes = _records(runner, "route_authority_change")
    assert any(c["route"] == "hot_scalp_promotion"
               and c["to"] == "governing"
               for c in changes)

    # A5b: an invariant violation demotes it, with evidence.
    runner.journal.append({
        "type": rp.VIOLATION_RECORD_TYPE,
        "route": "hot_scalp_promotion",
        "kind": rp.KIND_DUPLICATE_ORDER,
        "evidence": {"setup_id": "hs-dupe"},
        "ts": "2026-11-06 01:00:00+00:00"})
    runner._self_audit_phase("2026-11-06 01:00:00+00:00", h1)
    assert runner._judgment_route_state("hot_scalp_promotion") \
        == "shadow_declared"
    assert runner._judgment_route_source(
        "hot_scalp_promotion") == "demoted"
    demotions = _records(runner, "route_authority_demotion")
    assert len(demotions) == 1
    assert demotions[0]["kind"] == "duplicate_order"
    assert demotions[0]["evidence"]["setup_id"] == "hs-dupe"

    # The STALE verdict must not release the demotion: a
    # quiet cycle folds no new outcomes.
    runner._self_audit_phase("2026-11-06 02:00:00+00:00", h1)
    assert runner._judgment_route_source(
        "hot_scalp_promotion") == "demoted"

    # Fresh outcomes fold a fresh evaluation: released.
    _prime(runner, _ready_hot_history(start=300))
    runner._self_audit_phase("2026-11-06 03:00:00+00:00", h1)
    assert runner._judgment_route_state("hot_scalp_promotion") \
        == "governing"
    assert runner._judgment_route_source(
        "hot_scalp_promotion") == "verdict"
    entry = runner._route_authority_entry("hot_scalp_promotion")
    assert entry.get("released_demotion", {}).get("kind") \
        == "duplicate_order"
    changes = _records(runner, "route_authority_change")
    assert sum(1 for c in changes
               if c["route"] == "hot_scalp_promotion"
               and c["to"] == "governing") == 2


def test_not_ready_verdict_demotes_a_verdict_promotion(
        tmp_path):
    runner, h1 = _scan_runner(tmp_path)
    _prime(runner, _ready_hot_history())
    runner._self_audit_phase("2026-11-06 00:00:00+00:00", h1)
    assert runner._judgment_route_source(
        "hot_scalp_promotion") == "verdict"
    # A later, larger, losing batch flips the standing
    # verdict; the route the verdict promoted falls with it.
    records = _history_arm("hot_scalp", "hot_scalp_promotion",
                           "take", [-0.6] * 30, start=500)
    records += _history_arm("hot_scalp",
                            "hot_scalp_promotion", "abstain",
                            [0.7] * 30, start=600)
    _prime(runner, records)
    runner._self_audit_phase("2026-11-06 01:00:00+00:00", h1)
    assert runner._judgment_route_state("hot_scalp_promotion") \
        == "shadow_declared"
    changes = _records(runner, "route_authority_change")
    assert any(c["route"] == "hot_scalp_promotion"
               and c["verdict"] == "not_ready"
               for c in changes)


def test_insufficient_history_changes_no_authority(tmp_path):
    runner, h1 = _scan_runner(tmp_path)
    runner._self_audit_phase("2026-11-06 00:00:00+00:00", h1)
    # The first cycle still journals criteria + the opening
    # evaluation (the deployment marker), all insufficient.
    assert _records(runner, rp.CRITERIA_RECORD_TYPE)
    evaluations = _records(runner, rp.EVALUATION_RECORD_TYPE)
    assert evaluations
    assert all(
        row["verdict"] == "insufficient_data"
        for row in evaluations[-1]["routes"].values())
    assert runner._judgment_route_state("hot_scalp_promotion") \
        == "shadow_declared"
    assert runner._judgment_route_source(
        "hot_scalp_promotion") == "declared"
    assert _records(runner, "route_authority_change") == []
