"""MarketSituation PASS 9 — self-evaluation, coverage-gap
consumption, book-vs-S1 paired grading.

Pinned here:

- the BOOK EVALUATION: one comparison surface grading every
  family's shadow history on the shared promotion instrument
  (the evaluator is CONSUMED from range_promotion, never
  re-implemented) — family rows are the family evaluations'
  own snapshot rows verbatim; the connected routes ride as
  their own group; the governing routes' stamped live
  history folds through the same accumulate with hand-
  computed medians / hit rates; inverted pooled
  discrimination reads not_ready; each group carries its
  own n, and the comparison basis is stated on the record;
- the COVERAGE REPORT: coverage_gap records fold into
  episodes; an episode closes when its configuration is
  covered (by judgment_book's own predicate) or leaves the
  situation; every open gap is classified from the
  situation's OWN state — data halt / timeframe not ready /
  news blackout are deliberate stand-asides, anything else
  is an unwatched hole — with the deciding field named;
  gap ages are measured, never asserted; a gap feeds back
  nowhere;
- the BOOK GRADING: closed scenarios pair by
  formed_situation_id to the journal's situation snapshot
  records; thesis survival (graded closes not invalidated)
  per family and per situation cell is exact on fixtures;
  degenerate geometry is excluded AND counted; unpaired
  scenarios and unmeasurable cells are counted, never
  dropped; below-floor cells read insufficient_data;
- the host: all three ride the self-audit cycle (same
  window, same state file, own keys), journal on change
  only, and surface on /status.  The pass is evaluation
  only — no verdict here moves authority, and nothing in
  this file touches the Pass 5 appliers.
"""
from __future__ import annotations

from test_step14_setup_lifecycle import _scan_runner

from goldenswing.execution import book_evaluation as be
from goldenswing.execution import book_grading as bg
from goldenswing.execution import coverage_report as cr
from goldenswing.execution import route_promotion as rtp
from goldenswing.execution import vol_promotion as vp

# ------------------------------------------------ fabricated journals

def _ts(i: int) -> str:
    return (f"2026-10-{1 + i // 24:02d} {i % 24:02d}:00:00+00:00")


def _shadow(ts, playbook, decision, *, route,
            expected_r=0.4):
    return {
        "type": "judgment_shadow", "ts": ts,
        "playbook": playbook, "intended_tf": "H4",
        "gate_result": "shadow",
        "gate_order_result": "not_attempted",
        "gate_reason_code": "path.shadow_route",
        "judgment": {"decision": decision,
                     "expected_r": expected_r,
                     "confidence": 0.5, "mode": "active",
                     "authority_route": route,
                     "authority": "shadow_declared"},
    }


def _recon(ts, playbook, *, decision, realized_r, hit,
           horizon=4):
    move = realized_r * 10.0
    return {
        "type": "judgment_reconciliation", "judged_ts": ts,
        "playbook": playbook, "direction": 1,
        "decision": decision, "horizon_h": horizon,
        "entry": 2000.0, "exit_price": 2000.0 + move,
        "realized_r": realized_r, "hit": hit,
        "move": move, "move_atr": 1.0,
    }


def _arm(playbook, decision, rs, *, route, hits=None,
         start=0):
    """Journal rows for one arm: one route-stamped shadow +
    one reconciliation per outcome, paired on (ts, playbook)
    exactly as judgment_validation.build_pairs joins them."""
    out = []
    for i, r in enumerate(rs):
        ts = _ts(start + i)
        hit = (r > 0) if hits is None else hits[i]
        out.append(_shadow(ts, playbook, decision,
                           route=route))
        out.append(_recon(ts, playbook, decision=decision,
                          realized_r=r, hit=hit))
    return out


# ------------------------------------------------ book evaluation

def test_governing_history_grades_on_the_shared_instrument():
    # sweep_reversal stamped to its GOVERNING route: the
    # fold is the same accumulate the families use, so the
    # row values are the evaluator's own.
    records = _arm("sweep_reversal", "take",
                   [2.0, 1.0, 0.5, -0.5], route="h1_trigger")
    snap = be.evaluate_records(records)
    group = snap["groups"][-1]
    assert group["key"] == "governing"
    assert group["authority_posture"] == "governing"
    row = group["routes"]["sweep_reversal"]
    assert row["evaluated"] == 4
    assert row["take"]["paired"] == 4
    assert row["take"]["median_r"] == 0.75
    assert row["take"]["hit_rate"] == 0.75
    assert row["authority"] == "governing"
    # The window starts at the first stamped record, and the
    # comparison basis is stated on the record.
    assert group["window_start"] == _ts(0)
    assert "one instrument for every row" in \
        snap["window_basis"]


def test_governing_pooled_ready_and_inverted_never_ready():
    records = _arm("sweep_reversal", "take",
                   [2.0] * 18 + [-0.5] * 6, route="h1_trigger")
    records += _arm("sweep_reversal", "abstain",
                    [-1.0] * 18 + [0.4] * 6,
                    route="h1_trigger", start=200)
    group = be.evaluate_records(records)["groups"][-1]
    assert group["pooled"]["verdict"] == "promotion_ready"
    assert group["route_verdicts"]["promotion_ready"] == 1
    assert group["routes"]["sweep_reversal"][
        "verdict"] == "promotion_ready"
    # Same shape, discrimination flipped: takes are mildly
    # positive but the abstains beat them on median AND hit
    # rate — inverted discrimination, never ready.
    flipped = _arm("qm_d1", "take", [0.3] * 18 + [-0.6] * 6,
                   route="qm_d1_native")
    flipped += _arm("qm_d1", "abstain",
                    [1.2] * 20 + [-0.2] * 4,
                    route="qm_d1_native", start=200)
    group = be.evaluate_records(flipped)["groups"][-1]
    assert group["pooled"]["verdict"] == "not_ready"
    assert "INVERTED" in group["pooled"]["verdict_reason"]
    # And takes that simply lose (negative median) are
    # not_ready by the failing branch — also never ready.
    losing = _arm("qm_m5", "take", [-1.5] * 18 + [0.3] * 6,
                  route="qm_m5_native")
    losing += _arm("qm_m5", "abstain",
                   [-1.0] * 18 + [0.4] * 6,
                   route="qm_m5_native", start=200)
    group = be.evaluate_records(losing)["groups"][-1]
    assert group["pooled"]["verdict"] == "not_ready"


def test_family_rows_are_the_family_evaluations_verbatim():
    # Fold vol history through the family's OWN cycle, then
    # consume its state: the comparison row must equal the
    # family snapshot's row — consumed, not recomputed.
    vol_state = vp.new_state()
    records = _arm("vol_expansion_escape", "take",
                   [0.5] * 3, route="vol_expansion_escape")
    vp.cycle(state=vol_state, records=records,
             now="2026-10-02 00:00:00+00:00")
    own = vp.snapshot(vol_state)["routes"][
        "vol_expansion_escape"]
    state = be.new_state()
    rec = be.cycle(
        state=state, records=[],
        family_states={"vol_promotion": vol_state},
        now="2026-10-02 01:00:00+00:00")
    group = next(g for g in rec["groups"] if g["key"] == "vol")
    assert group["authority_posture"] == "shadow_declared"
    assert group["routes"]["vol_expansion_escape"] == own
    assert group["pooled"]["take"]["paired"] == 3


def test_connected_routes_ride_as_their_own_group():
    records = _arm("hot_scalp", "take", [1.0] * 4,
                   route="hot_scalp_promotion")
    evaluation = rtp.evaluate_records(records)
    route_state = {"last_evaluation": evaluation}
    rec = be.cycle(
        state=be.new_state(), records=[],
        family_states={"route_promotion": route_state},
        now="2026-10-02 01:00:00+00:00")
    group = next(g for g in rec["groups"]
                 if g["key"] == "connected")
    # route_promotion's evaluation is keyed by ROUTE.
    assert group["routes"]["hot_scalp_promotion"][
        "evaluated"] == 4
    assert group["routes"]["hot_scalp_promotion"][
        "playbook"] == "hot_scalp"
    # The pooled read over binding routes is a comparison,
    # never a second binding verdict.
    assert group["pooled"]["binds"] is False


def test_family_shadow_history_never_folds_into_governing():
    # range_qm's shadow history is stamped with its OWN
    # declared route (range_qm), not its governing route —
    # it belongs to the range group, not the governing one.
    records = _arm("range_qm", "take", [1.0] * 3,
                   route="range_qm")
    snap = be.evaluate_records(records)
    governing = snap["groups"][-1]
    assert governing["routes"]["range_qm"]["evaluated"] == 0
    assert governing["pooled"]["evaluated"] == 0
    range_group = snap["groups"][0]
    assert range_group["routes"]["range_qm"][
        "evaluated"] == 3
    # Unstamped history folds nowhere at all.
    bare = [{"type": "judgment_shadow", "ts": _ts(0),
             "playbook": "sweep_reversal",
             "judgment": {"decision": "take"}}]
    snap = be.evaluate_records(bare)
    assert all(g["pooled"]["evaluated"] == 0
               for g in snap["groups"])


def test_inverted_family_pooled_reads_not_ready_in_render():
    records = _arm("session_drive_reversal", "take",
                   [-1.0] * 18 + [0.2] * 6,
                   route="session_drive_reversal")
    records += _arm("session_drive_reversal", "abstain",
                    [1.0] * 18 + [-0.3] * 6,
                    route="session_drive_reversal", start=200)
    snap = be.evaluate_records(records)
    group = next(g for g in snap["groups"]
                 if g["key"] == "session")
    assert group["pooled"]["verdict"] == "not_ready"
    state = be.new_state()
    state["last_comparison"] = snap
    lines = be.render_lines(state)
    session_line = next(ln for ln in lines
                        if "session / opening range" in ln)
    assert "not ready" in session_line
    assert "takes" in session_line
    # The route itself also carries its own row + verdict.
    route_line = next(ln for ln in lines
                      if "session_drive_reversal" in ln)
    assert "not ready" in route_line
    assert "24 paired" in route_line


def test_book_evaluation_journals_on_change_only():
    state = be.new_state()
    first = be.cycle(state=state, records=[],
                     family_states={},
                     now="2026-10-02 00:00:00+00:00")
    assert first is not None  # deployment marker
    assert first["type"] == be.RECORD_TYPE
    second = be.cycle(state=state, records=[],
                      family_states={},
                      now="2026-10-02 01:00:00+00:00")
    assert second is None
    assert state["cycles"] == 1


def test_book_evaluation_damaged_window_degrades():
    state = be.new_state()
    out = be.cycle(state=state, records=_arm(
        "sweep_reversal", "take", [1.0], route="h1_trigger"),
        family_states={}, damaged_lines=3)
    assert out is None
    assert state["damaged_cycles"] == 1
    assert state["last_comparison"] is None
    assert not state["pending"]


# ------------------------------------------------ coverage report

def _situation(*, health="healthy", blackout=False,
               h1_readiness="ready", bias=1):
    return {
        "situation_id": "sit-1",
        "per_tf": {
            "H1": {"bias": bias, "structure_phase": "expansion",
                   "readiness": h1_readiness},
            "H4": {"bias": 1, "structure_phase": "trend",
                   "readiness": "ready"}},
        "relationships": [{"htf": "H4", "ltf": "H1",
                           "kind": "aligned",
                           "source": "engine"}],
        "data_quality": {"health": health, "missing": []},
        "activity": {"news_blackout": blackout},
    }


def _gap_record(key="tf:H1:bias=+1:phase=expansion",
                config=None, first="2026-11-01 00:00:00+00:00",
                fired="2026-11-01 02:00:00+00:00"):
    config = config or {
        "key": key, "kind": "tf", "tf": "H1", "bias": 1,
        "phase": "expansion", "label": "H1 long expansion"}
    return {"type": "coverage_gap", "ts": fired,
            "first_seen_ts": first, "config_key": key,
            "configuration": config,
            "label": config["label"], "cycles": 3}


def test_gap_folds_classifies_unwatched_and_ages():
    state = cr.new_state()
    rec = cr.cycle(state=state, records=[_gap_record()],
                   now="2026-11-01 05:00:00+00:00",
                   situation=_situation(), live=[])
    gap = rec["open_gaps"][0]
    assert gap["classification"] == cr.CLASS_UNWATCHED
    assert gap["age_hours"] == 5.0
    assert "no stand-aside fact" in gap[
        "classification_basis"]
    assert rec["coverage"]["configurations"] == 3
    assert rec["coverage"]["uncovered"] == 3
    lines = cr.render_lines(state)
    assert any("UNWATCHED" in ln for ln in lines)
    assert any("0 of 3" in ln for ln in lines)


def test_gap_classification_reads_the_situation_state():
    config = _gap_record()["configuration"]
    halt = cr.classify_gap(config, _situation(health="halt"))
    assert halt[0] == cr.CLASS_DELIBERATE_DATA
    assert "halt" in halt[1]
    not_ready = cr.classify_gap(
        config, _situation(h1_readiness="forming"))
    assert not_ready[0] == cr.CLASS_DELIBERATE_DATA
    assert "H1" in not_ready[1]
    news = cr.classify_gap(config, _situation(blackout=True))
    assert news[0] == cr.CLASS_DELIBERATE_NEWS
    plain = cr.classify_gap(config, _situation())
    assert plain[0] == cr.CLASS_UNWATCHED


def test_gap_closes_when_covered_and_when_it_leaves():
    state = cr.new_state()
    cr.cycle(state=state, records=[_gap_record()],
             now="2026-11-01 05:00:00+00:00",
             situation=_situation(), live=[])
    covering = [{"family": "vol_expansion", "direction": 1,
                 "state": "triggered",
                 "clocks": {"formation_tf": "H1",
                            "trigger_tf": "H1"}}]
    rec = cr.cycle(state=state, records=[],
                   now="2026-11-01 06:00:00+00:00",
                   situation=_situation(), live=covering)
    assert rec["closed_total"] == 1
    episode = state["episodes"][
        "tf:H1:bias=+1:phase=expansion"]
    assert episode["closed_reason"] == (
        "configuration covered by a live scenario")
    # A fresh gap whose configuration then leaves the
    # situation (bias flips) closes by departure.
    state2 = cr.new_state()
    cr.cycle(state=state2, records=[_gap_record()],
             now="2026-11-01 05:00:00+00:00",
             situation=_situation(), live=[])
    cr.cycle(state=state2, records=[],
             now="2026-11-01 06:00:00+00:00",
             situation=_situation(bias=-1), live=[])
    episode2 = state2["episodes"][
        "tf:H1:bias=+1:phase=expansion"]
    assert episode2["status"] == "closed"
    assert episode2["closed_reason"] == (
        "configuration left the situation")


def test_gap_episodes_dedupe_and_refire():
    state = cr.new_state()
    cr.cycle(state=state, records=[_gap_record()],
             now="2026-11-01 05:00:00+00:00",
             situation=_situation(), live=[])
    cr.cycle(state=state, records=[_gap_record()],
             now="2026-11-01 06:00:00+00:00",
             situation=_situation(), live=[])
    assert state["episodes_total"] == 1  # still one episode
    covering = [{"family": "vol_expansion", "direction": 1,
                 "state": "triggered",
                 "clocks": {"formation_tf": "H1",
                            "trigger_tf": "H1"}}]
    cr.cycle(state=state, records=[],
             now="2026-11-01 07:00:00+00:00",
             situation=_situation(), live=covering)
    cr.cycle(state=state, records=[_gap_record(
        fired="2026-11-01 09:00:00+00:00")],
        now="2026-11-01 09:00:00+00:00",
        situation=_situation(), live=[])
    assert state["episodes_total"] == 2  # re-fired = new one
    assert state["episodes"][
        "tf:H1:bias=+1:phase=expansion"]["status"] == "open"


def test_coverage_damaged_window_degrades():
    state = cr.new_state()
    out = cr.cycle(state=state, records=[_gap_record()],
                   damaged_lines=1)
    assert out is None
    assert state["damaged_cycles"] == 1
    assert state["episodes"] == {}


# ------------------------------------------------ book grading

def _situation_record(situation, sid="sit-1"):
    return {"type": "situation", "kind": "situation_snapshot",
            "situation_id": sid, "situation": situation}


def _closed(sid, *, family="vol_expansion", reason="filled",
            direction=1, ftf="H1", ttf="H1", s1=None,
            geometry=None, formed="sit-1",
            closed_ts="2026-11-01 03:00:00+00:00"):
    return {
        "scenario_id": sid, "family": family,
        "direction": direction, "close_reason": reason,
        "formed_situation_id": formed, "closed_ts": closed_ts,
        "clocks": {"formation_tf": ftf, "trigger_tf": ttf},
        "geometry": geometry or {"entry": 2000.0,
                                  "stop": 1990.0,
                                  "targets": [2020.0]},
        "s1": s1,
    }


def test_grading_pairs_by_snapshot_id_with_exact_cells():
    records = [_situation_record(_situation())]
    closed = [
        _closed("s1", reason="filled", s1={"decision": "take"}),
        _closed("s2", reason="invalidated", direction=-1,
                ftf="H4", s1={"decision": "abstain"},
                closed_ts="2026-11-01 04:00:00+00:00"),
    ]
    snap = bg.evaluate_closed(records, closed)
    assert snap["totals"]["graded"] == 2
    assert snap["totals"]["unpaired"] == 0
    fam = snap["families"]["vol_expansion"]
    assert fam["outcomes"]["filled"] == 1
    assert fam["outcomes"]["invalidated"] == 1
    assert fam["survival_rate"] == 0.5
    assert fam["status"] == "insufficient_data"
    assert fam["s1"]["take"] == {"graded": 1,
                                 "invalidated": 0}
    assert fam["s1"]["abstain"] == {"graded": 1,
                                    "invalidated": 1}
    cell = snap["cells"]["H1:expansion:with:rel=none"]
    assert cell["graded"] == 1
    assert cell["outcomes"]["filled"] == 1
    cell2 = snap["cells"]["H4:trend:against:rel=aligned"]
    assert cell2["outcomes"]["invalidated"] == 1
    assert "formed_situation_id" in snap["pairing_basis"]


def test_grading_unpaired_cell_unknown_and_degenerate():
    records = [_situation_record(_situation())]
    closed = [
        # formed under a snapshot never folded: unpaired.
        _closed("u1", formed="sit-never-seen"),
        # snapshot lacks the formation timeframe: no cell.
        _closed("u2", ftf="D1"),
        # degenerate geometry (entry == stop): excluded.
        _closed("d1", geometry={"entry": 2000.0,
                                "stop": 2000.0,
                                "targets": [2020.0]}),
        # degenerate: no targets.
        _closed("d2", geometry={"entry": 2000.0,
                                "stop": 1990.0, "targets": []}),
        # degenerate: no stop at all.
        _closed("d3", family="auction_value",
                geometry={"entry": 2000.0, "stop": None,
                          "targets": [2020.0]}),
    ]
    snap = bg.evaluate_closed(records, closed)
    totals = snap["totals"]
    assert totals["closed_seen"] == 5
    assert totals["graded"] == 2
    assert totals["unpaired"] == 1
    assert totals["cell_unknown"] == 1
    assert totals["degenerate_excluded"] == 3
    fam = snap["families"]["vol_expansion"]
    assert fam["degenerate_excluded"] == 2
    assert fam["unpaired"] == 1
    assert fam["cell_unknown"] == 1
    # The unpaired and cell-unknown scenarios grade into the
    # family distribution but into NO cell.
    assert snap["cells"] == {}


def test_grading_measured_cell_and_survival_rate():
    records = [_situation_record(_situation())]
    closed = [
        _closed(f"m{i}", reason=("filled" if i < 20
                                 else "invalidated"),
                closed_ts=f"2026-11-01 {i:02d}:00:00+00:00")
        for i in range(25)]
    snap = bg.evaluate_closed(records, closed)
    cell = snap["cells"]["H1:expansion:with:rel=none"]
    assert cell["status"] == "measured"
    assert cell["graded"] == 25
    assert cell["survival_rate"] == 0.8
    fam = snap["families"]["vol_expansion"]
    assert fam["status"] == "measured"
    assert fam["s1"]["none"] == 25
    state = bg.new_state()
    bg.cycle(state=state, records=records,
             closed_scenarios=closed)
    lines = bg.render_lines(state)
    assert any("survival 80%" in ln for ln in lines)
    assert any("most durable cell" in ln for ln in lines)


def test_grading_dedupes_refed_scenarios_and_degrades():
    state = bg.new_state()
    records = [_situation_record(_situation())]
    closed = [_closed("s1")]
    bg.cycle(state=state, records=records,
             closed_scenarios=closed)
    again = bg.cycle(state=state, records=[],
                     closed_scenarios=closed)
    assert again is None  # nothing new, nothing re-graded
    assert state["totals"]["closed_seen"] == 1
    out = bg.cycle(state=state, records=records,
                   closed_scenarios=[_closed("s2")],
                   damaged_lines=2)
    assert out is None
    assert state["damaged_cycles"] == 1
    assert state["totals"]["closed_seen"] == 1


def test_grading_unknown_close_reason_reads_other():
    records = [_situation_record(_situation())]
    snap = bg.evaluate_closed(
        records, [_closed("x1", reason="position_closed")])
    fam = snap["families"]["vol_expansion"]
    assert fam["outcomes"]["other"] == 1
    assert fam["graded"] == 1  # counted, never renamed


# ------------------------------------------------ the host

def _prime_journal(runner, records):
    for record in records:
        runner.journal.append(record)


def test_self_audit_phase_journals_all_three_records(tmp_path):
    runner, h1 = _scan_runner(tmp_path)
    records = _arm("sweep_reversal", "take", [0.5] * 2,
                   route="h1_trigger")
    records.append(_situation_record(_situation()))
    records.append(_gap_record())
    _prime_journal(runner, records)
    runner._self_audit_phase("2026-11-06 00:00:00+00:00", h1)
    journaled = {}
    for record in runner.journal.records():
        journaled.setdefault(record.get("type"), []).append(
            record)
    assert len(journaled[be.RECORD_TYPE]) == 1
    assert len(journaled[cr.RECORD_TYPE]) == 1
    assert len(journaled[bg.RECORD_TYPE]) == 1
    book_rec = journaled[be.RECORD_TYPE][0]
    governing = book_rec["groups"][-1]
    assert governing["routes"]["sweep_reversal"][
        "evaluated"] == 2
    cov_rec = journaled[cr.RECORD_TYPE][0]
    assert cov_rec["episodes_total"] == 1
    assert cov_rec["open_gaps"][0]["classification"] == \
        cr.CLASS_UNWATCHED
    # States persisted under their own keys.
    assert runner._book_evaluation_state()[
        "last_comparison"] is not None
    assert runner._coverage_report_state()[
        "episodes_total"] == 1
    assert runner._book_grading_state()[
        "last_snapshot"] is not None
    # Second phase, no new evidence: the comparison and the
    # grading do not re-journal.  The coverage account MAY
    # journal once more — the first cycle ran before the
    # situation keeper had folded the primed snapshot, so
    # the standing account genuinely gains its coverage
    # block when the read arrives; the episode itself is
    # unchanged either way.
    runner._self_audit_phase("2026-11-06 01:00:00+00:00", h1)
    types = [r.get("type") for r in runner.journal.records()]
    assert types.count(be.RECORD_TYPE) == 1
    assert types.count(bg.RECORD_TYPE) == 1
    assert runner._coverage_report_state()[
        "episodes_total"] == 1


def test_status_lines_render_the_three_blocks(tmp_path):
    runner, h1 = _scan_runner(tmp_path)
    # Before any cycle: stated absence, never fabricated.
    assert "no comparison cycle" in \
        runner.book_evaluation_line()
    assert "no coverage cycle" in \
        runner.coverage_report_line()
    assert "no grading cycle" in runner.book_grading_line()
    _prime_journal(runner, _arm(
        "sweep_reversal", "take", [0.5] * 2,
        route="h1_trigger"))
    runner._self_audit_phase("2026-11-06 00:00:00+00:00", h1)
    line = runner.book_evaluation_line()
    assert "book evaluation (Pass 9)" in line
    assert "governing routes" in line
    assert "sweep_reversal" in line
    assert "coverage (Pass 9)" in \
        runner.coverage_report_line()
    assert "book grading (Pass 9)" in \
        runner.book_grading_line()
