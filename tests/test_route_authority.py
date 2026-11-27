"""Route AUTHORITY (MarketSituation Pass 5; owner ruling
2026-11-08, AMENDMENTS-1 §A5 — as amended by the owner's
2026-11-25/27 orders: everything goes live directly on demo,
no shadow, no gating, no promotion, no prove-first).

Rewritten 2026-11-27: the promotion gate is DELETED.  The
authority machinery this file used to pin (JUDGMENT_ROUTE_STATES,
_judgment_route_state/source/resolution, demotion, the
undeclared-route alarm, the per-route config step) is gone.
What is pinned now:

- COMPLETENESS: every route named anywhere in the runner
  (seam calls, book stamps, state lookups) is declared in the
  registry's LIVE_ORDER_ROUTES, and every live route is
  reachable from a call site, the playbook map, or the
  connected-routes map — no route left unstamped, no stamp
  orphaned (AST audit over the runner source, both
  directions).
- THE LIVE TABLE: every route's authority is "live", source
  "live", governing now in active mode.
- THE SEAM: a live route stamps the verdict with its route
  and authority "live"; a core abstention still refuses the
  attempt under judgment.abstain.  An unknown route name is
  just a stamp — there is no gate left to fail closed.
- THE PER-CYCLE RECORD: one judgment_authority record per
  H1 close listing every route + authority + source; the
  /status line renders the same table.
- THE BOUNDARY: hot_scalp and the resting limit submit
  their independent demo orders through the seam like every
  live route — a core abstention still blocks them.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from goldenswing.execution.mt5_runner import (
    LIVE_ORDER_ROUTES,
    MT5Runner,
)
from goldenswing.execution.playbook_registry import (
    CONNECTED_ORDER_ROUTES,
    PLAYBOOK_REGISTRY,
    order_route_for,
)
from goldenswing.live.hierarchy import TimeframeStack

from test_completion02_connections import (
    _az_drive,
    _hs_cand,
    _hs_runner,
    _hs_tick,
)
from test_judgment_core import (
    _candidate,
    _decision,
    _evidence,
    _frame,
    _runner as _jc_runner,
)


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
    return literals, seam_routes


def test_route_audit_completeness_both_directions():
    literals, seam_routes = _route_literals()
    table = set(LIVE_ORDER_ROUTES)
    # Every route the runner names is declared live.
    assert literals <= table, (
        f"routes named in the runner but not declared live: "
        f"{sorted(literals - table)}")
    # Every live route is reachable: a call-site literal, the
    # playbook map, or the connected-routes map (hot_scalp and
    # arrival_zone ride routes the registry does not name).
    reachable = literals | {
        order_route_for(spec.name) or spec.name
        for spec in PLAYBOOK_REGISTRY} \
        | set(CONNECTED_ORDER_ROUTES.values())
    assert table <= reachable, (
        f"live routes no call site can reach: "
        f"{sorted(table - reachable)}")
    # The seam names its routes at the call sites: the natives
    # as literals, the families through the registry map.
    assert {"qm_m5_native", "qm_d1_native"} <= seam_routes
    assert "hot_scalp_promotion" in seam_routes
    assert "resting_limit" in seam_routes


def test_live_table_is_the_owners_record(tmp_path):
    # Owner orders 2026-11-25/27 — everything goes live
    # directly on demo; no shadow, no gating, no promotion.
    # Every route in the table is live by declaration.
    runner = _jc_runner(tmp_path)
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    assert set(table) == set(LIVE_ORDER_ROUTES)
    assert all(row["authority"] == "live"
               for row in table.values())
    assert all(row["governing_now"] is True
               for row in table.values())
    assert "hot_scalp_promotion" in table
    assert "resting_limit" in table


# ------------------------------------------------ the seam

def _seam_runner(tmp_path):
    """A runner whose M5 entry clock shows a real upward
    reaction through the _jc candidate's entry (3900): the
    long reads as a take, the short as a fight."""
    runner = _jc_runner(tmp_path)
    closes = [3880.0 + 2.0 * i for i in range(20)]
    runner._stack = TimeframeStack.from_frames(
        {"M5": _frame(closes, rng=2.0)})
    return runner


def test_live_route_seam_stamps_live(tmp_path):
    runner = _seam_runner(tmp_path)
    bundle, refusal = runner._judgment_authority(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(), intended_tf="M5",
        eval_now=runner._evaluation_now(),
        route="qm_m5_native")
    assert refusal is None
    judgment = (bundle or {}).get("judgment") or {}
    assert judgment.get("decision") == "take"
    assert judgment.get("authority_route") == "qm_m5_native"
    assert judgment.get("authority") == "live"


def test_seam_abstention_still_refuses(tmp_path):
    """The gate is gone, but the verdict still governs: a
    candidate that fights the measured structure abstains,
    and the seam refuses the attempt under judgment.abstain."""
    runner = _seam_runner(tmp_path)
    short = _candidate(direction=-1, entry=3900.0, stop=3910.0,
                       targets=[3880.0])
    bundle, refusal = runner._judgment_authority(
        candidate=short, decision=_decision(),
        evidence=_evidence(), intended_tf="M5",
        eval_now=runner._evaluation_now(),
        route="qm_m5_native")
    assert bundle is not None
    assert bundle["judgment"]["decision"] == "abstain"
    assert refusal is not None
    assert "abstained" in refusal


def test_unknown_route_is_just_a_stamp(tmp_path):
    """There is no gate left to fail closed: an unknown route
    name is stamped on the record as-is, and the verdict —
    not the route name — decides the attempt."""
    runner = _seam_runner(tmp_path)
    bundle, refusal = runner._judgment_authority(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(), intended_tf="M5",
        eval_now=runner._evaluation_now(),
        route="nope_route")
    assert refusal is None
    judgment = (bundle or {}).get("judgment") or {}
    assert judgment.get("decision") == "take"
    assert judgment.get("authority_route") == "nope_route"
    assert judgment.get("authority") == "live"
    alarms = _records(runner, "undeclared_route")
    assert alarms == []


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
    assert set(table) == set(LIVE_ORDER_ROUTES)
    assert table["qm_m5_native"]["authority"] == "live"
    assert table["qm_m5_native"]["source"] == "live"
    assert table["qm_m5_native"]["governing_now"] is True
    # Every route is live by declaration now (owner orders
    # 2026-11-25/27) — hot_scalp included.
    assert table["hot_scalp_promotion"]["authority"] == "live"
    assert table["hot_scalp_promotion"]["source"] == "live"
    assert table["hot_scalp_promotion"]["governing_now"] is True
    # The calibration verdict no longer rides this record:
    # the retired core's priors are not a verdict input.
    assert "priors" not in rows[0]


def test_status_line_renders_the_same_table(tmp_path):
    runner = _jc_runner(tmp_path)
    line = runner.judgment_authority_line()
    for route in LIVE_ORDER_ROUTES:
        assert route in line
    assert "hot_scalp_promotion" in line
    assert "live" in line
    # The commands /status surface wires the same line in.
    commands = (Path.cwd() / "goldenswing" / "execution"
                / "mt5_commands.py").read_text()
    assert "judgment_authority_line" in commands
    assert "Judgment Authority" in commands


# ------------------------------------------------ the boundary: hot scalp

def test_hot_scalp_boundary_submits_at_default(tmp_path):
    # Owner order 2026-11-25: the route is live by
    # declaration, so at default the candidate submits
    # through the seam like every live route — no verdict
    # needed, no config step.
    runner = _hs_runner(tmp_path)
    placed_before = len(runner.executor.placed)
    assert runner._place_hot_scalp_candidate(
        _hs_cand(runner), _hs_tick(), {}) is True
    assert len(runner.executor.placed) == placed_before + 1


def test_hot_scalp_submits_through_the_seam(tmp_path):
    runner = _hs_runner(tmp_path / "placed")
    (tmp_path / "placed").mkdir(exist_ok=True)
    assert runner._place_hot_scalp_candidate(
        _hs_cand(runner), _hs_tick(), {}) is True
    assert len(runner.executor.placed) == 1
    # And through the seam means the seam still governs: a
    # core abstention blocks the route exactly like every
    # live route.  The abstention is forced by structure — a
    # short candidate fights the measured upward reaction.
    runner = _hs_runner(tmp_path / "abstain")
    (tmp_path / "abstain").mkdir(exist_ok=True)
    assert runner._place_hot_scalp_candidate(
        _hs_cand(runner, direction=-1, entry=100.30,
                 stop=101.60, target=99.0),
        _hs_tick(), {}) is False
    assert runner.executor.placed == []
    evals = _evals(runner, "hot_scalp")
    assert evals[-1]["decision"]["reason_code"] \
        == "judgment.abstain"


# ------------------------------------------------ the boundary: resting limit

def test_resting_limit_boundary_rests_at_default(tmp_path):
    # Owner order 2026-11-25: the route is live by
    # declaration, so at default the limit rests — no
    # verdict, no config step needed.
    runner = _az_drive(tmp_path, structure="up")
    assert runner._pending_orders != {}
    evals = _evals(runner, "arrival_zone")
    assert len(evals) == 1
    assert evals[0]["decision"]["reason_code"] \
        == "decision.resting_limit"
    # A second call inside the same approach adds no second
    # record: the pending order short-circuits the flow.
    import pandas as pd
    runner._maybe_place_resting_limit(
        {"mid": 4122.0 + 3.6, "spread": 0.3,
         "time": pd.Timestamp("2026-10-21 13:00", tz="UTC")},
        {})
    assert runner._pending_orders != {}
    assert len(_evals(runner, "arrival_zone")) == 1


def test_resting_limit_abstention_refuses(tmp_path):
    """The abstention refuses the placement under
    judgment.abstain — the approach's one funnel record
    terminates refused, and nothing rests."""
    runner = _az_drive(tmp_path, structure="down")
    assert runner._pending_orders == {}
    evals = _evals(runner, "arrival_zone")
    assert len(evals) == 1
    assert evals[0]["decision"]["reason_code"] \
        == "judgment.abstain"


# ------------------------------------------------ the live table in shadow mode

def test_shadow_mode_governs_nothing_but_journals(tmp_path):
    """Shadow mode is the global kill switch's rollback
    posture, not a route gate: the table still lists every
    route as live, but governing_now is False for all of
    them — and orders proceed unjudged, never refused."""
    runner = _jc_runner(tmp_path, judgment_mode="shadow")
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    assert set(table) == set(LIVE_ORDER_ROUTES)
    assert all(row["authority"] == "live"
               for row in table.values())
    assert all(row["governing_now"] is False
               for row in table.values())
