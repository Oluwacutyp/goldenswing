"""§G range-family PROMOTION EVALUATION (owner program
2026-11-05 — the machinery completion-02 recorded as missing:
the four range routes sat on a declared shadow contract with
no evaluation of whether they deserve promotion).

Pinned here:

- the wiring: range scans journal ROUTE-STAMPED judgment
  verdicts (authority_route + shadow_declared), the routes are
  declared in JUDGMENT_ROUTE_STATES (closing the
  undeclared-defaults-to-governing hazard), the manifest route
  table counts their accruing history — and the stamp is
  output-only: verdicts and scan decisions are byte-identical
  with the stamp wiring off, and with the whole judgment tap
  off (decision-path parity);
- the evaluation: fabricated journals driving the REAL
  computation — right-way discrimination reads
  promotion_ready; inverted reads not_ready (the evidence
  pack's lesson, mechanized); negative-median takes read
  not_ready; thin data reads insufficient_data with the n
  named; zero detections read "no candidates observed" —
  a different fact from measured failure; degenerate-R pairs
  are excluded AND counted (the +226R shape among them);
  unstamped / other-route history is never evaluated (the
  window starts at the wiring's deployment);
- the host: the evaluation rides the self-audit cycle (same
  window, same state file), journals its record on change
  only, and surfaces on /status and the manifest.  Nothing
  here promotes a route: the verdict is a recommendation, and
  the only promotion step remains the owner's explicit config.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from goldenswing.execution import range_promotion as rp
from goldenswing.execution import mt5_runner as runner_mod
from goldenswing.execution.mt5_runner import (
    JUDGMENT_ROUTE_PLAYBOOK, JUDGMENT_ROUTE_STATES)
from goldenswing.execution.playbook_registry import RANGE_SPECS

from test_p3_trader import _runner, _sig, _StubPB
from test_step14_setup_lifecycle import _scan_runner


# ------------------------------------------------ fabricated journals

def _ts(i: int) -> str:
    return (f"2026-10-{1 + i // 24:02d} {i % 24:02d}:00:00+00:00")


def _shadow(ts, playbook="range_qm", decision="take", *,
            expected_r=0.4, stamp=True, route=None):
    judgment = {"decision": decision, "expected_r": expected_r,
                "confidence": 0.5, "mode": "active"}
    if stamp:
        judgment["authority_route"] = route or playbook
        judgment["authority"] = "shadow_declared"
    return {"type": "judgment_shadow", "ts": ts,
            "playbook": playbook, "intended_tf": "H4",
            "gate_result": "shadow",
            "gate_order_result": "not_attempted",
            "gate_reason_code": "path.shadow_route",
            "judgment": judgment}


def _recon(ts, playbook="range_qm", *, decision="take", horizon=4,
           realized_r=0.5, hit=True, r_excluded=None, entry=2000.0):
    move = None if realized_r is None else realized_r * 10.0
    rec = {"type": "judgment_reconciliation", "judged_ts": ts,
           "playbook": playbook, "direction": 1,
           "decision": decision, "horizon_h": horizon,
           "entry": entry,
           "exit_price": entry + (move or 0.0),
           "realized_r": realized_r, "hit": hit,
           "move": move, "move_atr": 1.0}
    if r_excluded:
        rec["r_excluded"] = r_excluded
    return rec


def _arm(playbook, decision, rs, *, hits=None, start=0):
    """Journal rows for one arm: one stamped shadow + one +4h
    reconciliation per outcome, on distinct hours."""
    records = []
    for i, r in enumerate(rs):
        ts = _ts(start + i)
        records.append(_shadow(ts, playbook, decision))
        records.append(_recon(
            ts, playbook, decision=decision, realized_r=r,
            hit=(hits[i] if hits is not None else r > 0)))
    return records


def _route(snap, playbook="range_qm"):
    return snap["routes"][playbook]


# ------------------------------------------------ verdicts

def test_promotion_ready_on_right_way_discrimination():
    records = _arm("range_qm", "take", [0.5] * 18 + [-0.5] * 4)
    records += _arm("range_qm", "abstain",
                    [-0.4] * 15 + [0.3] * 6, start=100)
    snap = rp.evaluate_records(records)
    row = _route(snap)
    assert row["verdict"] == "promotion_ready"
    assert row["take"]["paired"] == 22
    assert row["take"]["median_r"] == pytest.approx(0.5)
    assert row["take"]["hit_rate"] == round(18 / 22, 4)
    assert row["abstain"]["paired"] == 21
    assert row["abstain"]["median_r"] == pytest.approx(-0.4)
    assert "RECOMMENDATION" in row["verdict_reason"]
    assert snap["horizon_h"] == 4
    assert snap["floors"] == {"min_paired_per_arm": 20}


def test_inverted_discrimination_is_not_ready():
    """The evidence pack's 1h lesson: abstains outperforming
    takes (median AND hit rate) reads INVERTED — never ready."""
    records = _arm("range_qm", "take", [0.2] * 12 + [-0.6] * 10)
    records += _arm("range_qm", "abstain",
                    [0.7] * 14 + [-0.2] * 7, start=100)
    row = _route(rp.evaluate_records(records))
    assert row["verdict"] == "not_ready"
    assert "INVERTED" in row["verdict_reason"]


def test_hit_rate_inversion_alone_blocks_ready():
    """Medians right-way, hit rates inverted: still not ready
    — discrimination must hold on both reads."""
    records = _arm("range_qm", "take", [0.9] * 8 + [0.1] * 14,
                   hits=[True] * 8 + [False] * 14)
    records += _arm("range_qm", "abstain", [0.05] * 21,
                    hits=[True] * 15 + [False] * 6, start=100)
    row = _route(rp.evaluate_records(records))
    assert row["take"]["median_r"] > row["abstain"]["median_r"]
    assert row["verdict"] == "not_ready"
    assert "INVERTED" in row["verdict_reason"]


def test_negative_take_median_is_not_ready():
    records = _arm("range_qm", "take", [-0.3] * 12 + [0.4] * 10)
    records += _arm("range_qm", "abstain", [-0.9] * 21, start=100)
    row = _route(rp.evaluate_records(records))
    assert row["verdict"] == "not_ready"
    assert "negative" in row["verdict_reason"]


def test_thin_data_is_insufficient_with_n_named():
    records = _arm("range_qm", "take", [0.5] * 5)
    records += _arm("range_qm", "abstain", [-0.5] * 3, start=50)
    row = _route(rp.evaluate_records(records))
    assert row["verdict"] == "insufficient_data"
    assert "5/20" in row["verdict_reason"]
    assert "3/20" in row["verdict_reason"]
    assert row["evaluated"] == 8


def test_no_candidates_observed_is_its_own_fact():
    snap = rp.evaluate_records([])
    for playbook in rp.RANGE_PLAYBOOKS:
        row = _route(snap, playbook)
        assert row["verdict"] == "insufficient_data"
        assert "no candidates observed" in row["verdict_reason"]
        assert row["evaluated"] == 0
    assert snap["window_start"] is None


def test_unstamped_and_other_route_history_is_not_evaluated():
    """The window honesty: identical outcomes WITHOUT the §G
    route stamp (pre-wiring history), or stamped for the
    governing h1_trigger route (the same playbooks' roster
    verdicts — a different question), never enter the arms."""
    records = []
    for i in range(22):
        ts = _ts(i)
        records.append(_shadow(ts, "range_qm", "take",
                               stamp=False))
        records.append(_recon(ts, "range_qm", realized_r=0.5))
        ts2 = _ts(100 + i)
        records.append(_shadow(ts2, "range_qm", "take",
                               route="h1_trigger"))
        records.append(_recon(ts2, "range_qm", realized_r=0.5))
    row = _route(rp.evaluate_records(records))
    assert row["evaluated"] == 0
    assert row["take"]["paired"] == 0
    assert "no candidates observed" in row["verdict_reason"]


def test_degenerate_pairs_excluded_and_counted():
    """Both degenerate shapes: an S3-native exclusion (R never
    journaled, r_excluded on the record) and the pack's +226R
    (R journaled on a ~$0.135 implied stop — re-derived by
    build_pairs).  Neither grades an arm; both are counted,
    and the clean verdict survives them."""
    pb = "range_sweep_reclaim"
    records = _arm(pb, "take", [0.5] * 20)
    records += _arm(pb, "abstain", [-0.5] * 20, start=100)
    # +226R on a whisker stop (move 30.52 / 226 -> 0.135).
    ts = _ts(200)
    records.append(_shadow(ts, pb, "take"))
    records.append(_recon(ts, pb, realized_r=226.0, hit=True))
    records[-1]["move"] = 30.52
    # S3-native: R never journaled, exclusion on the record.
    ts = _ts(201)
    records.append(_shadow(ts, pb, "take"))
    records.append(_recon(ts, pb, realized_r=None, hit=True,
                          r_excluded="degenerate_stop"))
    row = _route(rp.evaluate_records(records), pb)
    assert row["excluded_degenerate_r"] == 2
    assert row["take"]["paired"] == 20
    assert row["take"]["median_r"] == pytest.approx(0.5)
    assert row["verdict"] == "promotion_ready"


def test_duplicates_never_double_grade():
    records = _arm("range_qm", "take", [0.5] * 3)
    doubled = records + [dict(r) for r in records]
    row = _route(rp.evaluate_records(doubled))
    assert row["evaluated"] == 3
    assert row["take"]["paired"] == 3
    # A DISTINCT verdict under the same (ts, playbook) key:
    # the pairing machinery keys on it, so the first counts
    # and the second is a counted duplicate, never graded.
    ts = _ts(50)
    extra = [_shadow(ts, "range_qm", "take"),
             _recon(ts, "range_qm", realized_r=0.5),
             _shadow(ts, "range_qm", "abstain"),
             _recon(ts, "range_qm", decision="abstain",
                    realized_r=-0.5, hit=False)]
    row = _route(rp.evaluate_records(records + extra))
    assert row["duplicate_key_records"] == 1
    assert row["take"]["paired"] == 4
    assert row["abstain"]["paired"] == 0


def test_only_the_4h_horizon_grades():
    records = []
    for i in range(22):
        ts = _ts(i)
        records.append(_shadow(ts, "range_qm", "take"))
        records.append(_recon(ts, "range_qm", horizon=1,
                              realized_r=0.5))
    row = _route(rp.evaluate_records(records))
    assert row["evaluated"] == 22
    assert row["take"]["paired"] == 0
    assert row["verdict"] == "insufficient_data"
    assert "0/20" in row["verdict_reason"]


def test_pending_outcomes_and_expiry():
    state = rp.new_state()
    records = [_shadow(_ts(i), "range_qm", "take")
               for i in range(3)]
    snap = rp.accumulate(state, records, now=_ts(3))
    row = _route(snap)
    assert row["evaluated"] == 3
    assert row["pending_outcomes"] == 3
    # 96h later with no reconciliations: unmeasurable, not
    # pending forever (the self-audit's 48h rule).
    snap = rp.accumulate(state, [], now="2026-10-05 00:00:00+00:00")
    row = _route(snap)
    assert row["pending_outcomes"] == 0
    assert row["unmeasurable_expired"] == 3


def test_window_start_is_the_first_stamped_verdict():
    records = _arm("range_qm", "take", [0.5], start=7)
    snap = rp.evaluate_records(records)
    assert snap["window_start"] == _ts(7)
    assert "route-stamped" in snap["window_basis"]
    assert "deployment" in snap["window_basis"]


# ------------------------------------------------ the cycle

def test_cycle_journals_on_change_only():
    state = rp.new_state()
    first = rp.cycle(state=state, records=[], now=_ts(0),
                     bar="b1", run_id="r1")
    assert first is not None
    assert first["type"] == "range_promotion_evaluation"
    assert first["routes"]["range_qm"]["verdict"] \
        == "insufficient_data"
    # Unchanged snapshot: nothing journals.
    assert rp.cycle(state=state, records=[], now=_ts(1),
                    bar="b2", run_id="r1") is None
    # New evidence changes the snapshot: journals again.
    records = _arm("range_qm", "take", [0.5] * 2)
    second = rp.cycle(state=state, records=records, now=_ts(30),
                      bar="b3", run_id="r1")
    assert second is not None
    assert second["routes"]["range_qm"]["evaluated"] == 2
    assert state["cycles"] == 2


def test_cycle_damaged_window_degrades_without_grading():
    state = rp.new_state()
    records = _arm("range_qm", "take", [0.5] * 3)
    out = rp.cycle(state=state, records=records, now=_ts(10),
                   damaged_lines=2, bar="b1", run_id="r1")
    assert out is None
    assert state["damaged_cycles"] == 1
    assert state["routes"] == {}


def test_cross_window_accumulation_reaches_a_verdict():
    """The live shape: shadows land in one window, their +4h
    reconciliations in a later one; the standing state carries
    the pending set across, and the verdict arrives when the
    outcomes do."""
    state = rp.new_state()
    shadows, recons = [], []
    for i, r in enumerate([0.5] * 18 + [-0.5] * 4):
        ts = _ts(i)
        shadows.append(_shadow(ts, "range_qm", "take"))
        recons.append(_recon(ts, "range_qm", realized_r=r,
                             hit=r > 0))
    for i, r in enumerate([-0.4] * 15 + [0.3] * 6):
        ts = _ts(22 + i)
        shadows.append(_shadow(ts, "range_qm", "abstain"))
        recons.append(_recon(ts, "range_qm", decision="abstain",
                             realized_r=r, hit=r > 0))
    rp.cycle(state=state, records=shadows, now=_ts(43),
             bar="b1", run_id="r1")
    snap = rp.snapshot(state)
    assert _route(snap)["pending_outcomes"] == 43
    record = rp.cycle(state=state, records=recons, now=_ts(47),
                      bar="b2", run_id="r1")
    assert record is not None
    assert _route(record)["verdict"] == "promotion_ready"
    assert _route(record)["pending_outcomes"] == 0


# ------------------------------------------------ route declaration

def test_range_routes_declared_shadow_and_tables_agree():
    for spec in RANGE_SPECS:
        assert JUDGMENT_ROUTE_STATES[spec.name] \
            == "shadow_declared"
        assert JUDGMENT_ROUTE_PLAYBOOK[spec.name] == spec.name
        assert rp.RANGE_JUDGMENT_ROUTES[spec.name] == spec.name
    assert rp.RANGE_PLAYBOOKS == tuple(
        s.name for s in RANGE_SPECS)


def test_range_route_state_resolution_and_typo_safety(tmp_path):
    runner = _runner(tmp_path)
    for name in rp.RANGE_PLAYBOOKS:
        assert runner._judgment_route_state(name) \
            == "shadow_declared"
    # A typo in the promotion config can neither promote nor
    # demote (the completion-02 rule, on the new routes).
    runner.cfg.judgment_route_modes = {
        "range_qm": "governingg", "range_qm2": "governing"}
    assert runner._judgment_route_state("range_qm") \
        == "shadow_declared"
    # The explicit step CAN flip the state — that config step
    # is the promotion the evaluation recommends into.
    runner.cfg.judgment_route_modes = {"range_qm": "governing"}
    assert runner._judgment_route_state("range_qm") \
        == "governing"
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    assert table["range_qm"]["governing_now"] is True
    for name in rp.RANGE_PLAYBOOKS:
        if name != "range_qm":
            assert table[name]["governing_now"] is False


# ------------------------------------------------ scan wiring

def _range_spec(frame):
    return SimpleNamespace(
        name="range_qm", route="range_h4_h1",
        playbook_class=lambda: _StubPB(
            "range_qm", _sig(direction=1, index=len(frame) - 1)))


def _drive_range_scan(tmp_path, monkeypatch, **kw):
    runner, h1 = _scan_runner(tmp_path, **kw)
    monkeypatch.setattr(runner_mod, "RANGE_SPECS",
                        (_range_spec(h1),))
    runner._run_range_family("h4_generation", {},
                             source="live_close")
    return runner


def _scan_evals(runner):
    return [r for r in runner.journal.records()
            if r.get("type") == "evaluation"
            and r.get("playbook") == "range_qm"
            and r.get("outcome") == "candidate_emitted"]


def _scan_shadows(runner):
    return [r for r in runner.journal.records()
            if r.get("type") == "judgment_shadow"
            and r.get("playbook") == "range_qm"]


def test_range_scan_journals_route_stamped_shadow(
        tmp_path, monkeypatch):
    runner = _drive_range_scan(tmp_path, monkeypatch)
    evals = _scan_evals(runner)
    assert evals
    # The §G disposition is exactly as before the wiring.
    assert evals[-1]["decision"]["result"] == "shadow"
    assert evals[-1]["decision"]["order_result"] \
        == "not_attempted"
    assert evals[-1]["reason_code"] == "path.shadow_route"
    shadows = _scan_shadows(runner)
    assert len(shadows) == 1
    judgment = shadows[0]["judgment"]
    assert judgment["authority_route"] == "range_qm"
    assert judgment["authority"] == "shadow_declared"
    assert shadows[0]["gate_result"] == "shadow"
    # The candidate's honest gap rides along: no asymmetry
    # breakdown exists on the scan path, and none is invented.
    assert evals[-1]["candidate"]["asymmetry_score"] is None
    # The manifest route table counts the accruing history.
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    assert table["range_qm"]["judgment_records_this_run"] == 1
    assert table["range_qm"]["authority"] == "shadow_declared"
    assert table["range_qm"]["governing_now"] is False


def test_stamp_is_output_only(tmp_path, monkeypatch):
    """Same scan, wiring on vs off: the core's verdict and the
    scan's own record are identical — the stamp adds two
    fields to the judgment block and changes nothing else."""
    # (two runners need two roots; build them explicitly)
    root_on = tmp_path / "on"
    root_off = tmp_path / "off"
    root_on.mkdir()
    root_off.mkdir()
    runner_on = _drive_range_scan(root_on, monkeypatch)
    monkeypatch.setattr(runner_mod,
                        "RANGE_JUDGMENT_ROUTE_NAMES", frozenset())
    runner_off = _drive_range_scan(root_off, monkeypatch)

    def core(runner):
        j = _scan_shadows(runner)[0]["judgment"]
        return {k: j.get(k) for k in (
            "decision", "expected_r", "confidence", "reason",
            "reason_category", "scenarios", "support_score")}

    assert core(runner_on) == core(runner_off)
    j_off = _scan_shadows(runner_off)[0]["judgment"]
    assert "authority_route" not in j_off
    assert "authority" not in j_off

    def scan_facts(runner):
        rec = _scan_evals(runner)[-1]
        return (rec["outcome"], rec["reason_code"],
                rec["decision"], rec["candidate"])

    assert scan_facts(runner_on) == scan_facts(runner_off)


def test_decision_path_parity_with_tap_off(tmp_path, monkeypatch):
    """The funnel-parity shape: with the whole judgment tap
    disabled, the §G scan's records are the same records —
    observation never feeds a decision."""
    root_on = tmp_path / "on"
    root_off = tmp_path / "off"
    root_on.mkdir()
    root_off.mkdir()
    runner_on = _drive_range_scan(root_on, monkeypatch)
    runner_off = _drive_range_scan(
        root_off, monkeypatch, judgment_shadow_enabled=False)
    assert _scan_shadows(runner_off) == []

    def scan_facts(runner):
        rec = _scan_evals(runner)[-1]
        return (rec["outcome"], rec["reason_code"],
                rec["decision"], rec["candidate"])

    assert scan_facts(runner_on) == scan_facts(runner_off)


# ------------------------------------------------ the host

def _prime_journal(runner, records):
    for record in records:
        runner.journal.append(record)


def test_self_audit_phase_journals_evaluation_record(tmp_path):
    runner, h1 = _scan_runner(tmp_path)
    records = _arm("range_qm", "take", [0.5] * 2)
    _prime_journal(runner, records)
    runner._self_audit_phase("2026-11-06 00:00:00+00:00", h1)
    journaled = [r for r in runner.journal.records()
                 if r.get("type") == rp.RECORD_TYPE]
    assert len(journaled) == 1
    rec = journaled[0]
    assert rec["routes"]["range_qm"]["evaluated"] == 2
    assert rec["routes"]["range_qm"]["take"]["paired"] == 2
    assert rec["window_start"] == _ts(0)
    # Second phase, no new evidence: no second record.
    runner._self_audit_phase("2026-11-06 01:00:00+00:00", h1)
    journaled = [r for r in runner.journal.records()
                 if r.get("type") == rp.RECORD_TYPE]
    assert len(journaled) == 1
    # The standing state persisted into the audit state file's
    # own block, beside the self-audit's.
    state = runner._range_promotion_state()
    assert state["last_snapshot"]["routes"]["range_qm"][
        "evaluated"] == 2


def test_manifest_block_and_status_line(tmp_path):
    runner, _h1 = _scan_runner(tmp_path)
    # Before any cycle: stated absence, never a fabricated zero
    # history presented as measurement.
    assert "no evaluation cycle" in runner.range_promotion_line()
    block = runner._range_promotion_manifest_block()
    assert block["routes"] == {}
    records = _arm("range_qm", "take", [0.5] * 18 + [-0.5] * 4)
    records += _arm("range_qm", "abstain",
                    [-0.4] * 15 + [0.3] * 6, start=100)
    _prime_journal(runner, records)
    runner._self_audit_phase("2026-11-06 00:00:00+00:00", _h1)
    block = runner._range_promotion_manifest_block()
    row = block["routes"]["range_qm"]
    assert row["verdict"] == "promotion_ready"
    assert block["window_start"] == _ts(0)
    line = runner.range_promotion_line()
    assert "range_qm" in line
    assert "PROMOTION READY" in line
    # The full manifest carries the block beside the self-audit.
    runner._journal_manifest()
    manifest = [r for r in runner.journal.records()
                if r.get("type") == "manifest"][-1]
    core = manifest["doctrine"]["judgment_core"]
    assert core["range_promotion"]["routes"]["range_qm"][
        "verdict"] == "promotion_ready"
    table = {r["route"]: r for r in core["route_authority"]}
    assert table["range_qm"]["authority"] == "shadow_declared"


# ------------------------------------------------ the report script

def test_report_script_renders_the_same_verdict():
    path = Path(__file__).resolve().parent.parent / "scripts" \
        / "range_promotion_report.py"
    spec = importlib.util.spec_from_file_location(
        "range_promotion_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = _arm("range_qm", "take", [0.5] * 18 + [-0.5] * 4)
    records += _arm("range_qm", "abstain",
                    [-0.4] * 15 + [0.3] * 6, start=100)
    snap = rp.evaluate_records(records)
    text = module.render(snap, journal="journal.jsonl",
                         records=len(records), malformed=0)
    assert "## range_qm — promotion_ready" in text
    assert "## range_sweep_reclaim — insufficient_data" in text
    assert "Evaluation window opened:" in text
    assert "promotes nothing" in text
