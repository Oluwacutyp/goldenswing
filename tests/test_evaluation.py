"""The shared S4-pairing evaluator (goldenswing/execution/
evaluation.py) — the measurement the seven deleted
promotion modules shared, extracted whole when the
promotion gate was deleted (owner order 2026-11-27).

Pinned here (ported from the deleted range-promotion
evaluation tests, re-pointed at the new module and the
live record shape — type "judgment", authority "live",
route-stamped):

- the verdicts: right-way discrimination reads ready;
  inverted reads not_ready (the evidence pack's lesson,
  mechanized); negative-median takes read not_ready; thin
  data reads insufficient_data with the n named; zero
  detections read "no candidates observed" — a different
  fact from measured failure; degenerate-R pairs are
  excluded AND counted (the +226R shape among them);
  unstamped / other-route history is never evaluated (the
  window starts at the wiring's deployment);
- the accumulation: pending outcomes carried across
  windows, expired as unmeasurable (never pending forever),
  duplicates never double-grade, only the 4h horizon
  grades, window_start is the first stamped verdict.

Nothing here promotes, demotes, grants or changes
anything: the verdict is a measurement only — the record
says so.  The host cycle that journals it lives in
book_evaluation.py (tested in test_book_evaluation.py).
"""
from __future__ import annotations

import pytest

from goldenswing.execution import evaluation as ev

# ------------------------------------------------ fabricated journals

def _ts(i: int) -> str:
    return (f"2026-10-{1 + i // 24:02d} {i % 24:02d}:00:00+00:00")


def _judgment(ts, playbook="range_qm", decision="take", *,
              expected_r=0.4, stamp=True, route=None):
    judgment = {"decision": decision, "expected_r": expected_r,
                "confidence": 0.5, "mode": "active"}
    if stamp:
        judgment["authority_route"] = route or playbook
        judgment["authority"] = "live"
    return {"type": "judgment", "ts": ts,
            "playbook": playbook, "intended_tf": "H4",
            "gate_result": "accepted",
            "gate_order_result": "not_attempted",
            "gate_reason_code": None,
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
    """Journal rows for one arm: one stamped judgment + one
    +4h reconciliation per outcome, on distinct hours."""
    records = []
    for i, r in enumerate(rs):
        ts = _ts(start + i)
        records.append(_judgment(ts, playbook, decision))
        records.append(_recon(
            ts, playbook, decision=decision, realized_r=r,
            hit=(hits[i] if hits is not None else r > 0)))
    return records


def _arm_records(route, take_rs, abstain_rs, *, start=0):
    """One judgment + one +4h reconciliation per outcome, per
    arm — the family files' fabricator, shared here so every
    family grades the same shapes."""
    return (_arm(route, "take", take_rs, start=start)
            + _arm(route, "abstain", abstain_rs,
                   start=start + 1000))


def _route(snap, playbook="range_qm"):
    return snap["routes"][playbook]


# ------------------------------------------------ verdicts

def test_ready_on_right_way_discrimination():
    records = _arm("range_qm", "take", [0.5] * 18 + [-0.5] * 4)
    records += _arm("range_qm", "abstain",
                    [-0.4] * 15 + [0.3] * 6, start=100)
    snap = ev.evaluate_records(records)
    row = _route(snap)
    assert row["verdict"] == "ready"
    assert row["take"]["paired"] == 22
    assert row["take"]["median_r"] == pytest.approx(0.5)
    assert row["take"]["hit_rate"] == round(18 / 22, 4)
    assert row["abstain"]["paired"] == 21
    assert row["abstain"]["median_r"] == pytest.approx(-0.4)
    assert "a measurement, never an action" in row["verdict_reason"]
    assert snap["horizon_h"] == 4
    assert snap["floors"] == {"min_paired_per_arm": 20}


def test_inverted_discrimination_is_not_ready():
    """The evidence pack's 1h lesson: abstains outperforming
    takes (median AND hit rate) reads INVERTED — never ready."""
    records = _arm("range_qm", "take", [0.2] * 12 + [-0.6] * 10)
    records += _arm("range_qm", "abstain",
                    [0.7] * 14 + [-0.2] * 7, start=100)
    row = _route(ev.evaluate_records(records))
    assert row["verdict"] == "not_ready"
    assert "INVERTED" in row["verdict_reason"]


def test_hit_rate_inversion_alone_blocks_ready():
    """Medians right-way, hit rates inverted: still not ready
    — discrimination must hold on both reads."""
    records = _arm("range_qm", "take", [0.9] * 8 + [0.1] * 14,
                   hits=[True] * 8 + [False] * 14)
    records += _arm("range_qm", "abstain", [0.05] * 21,
                    hits=[True] * 15 + [False] * 6, start=100)
    row = _route(ev.evaluate_records(records))
    assert row["take"]["median_r"] > row["abstain"]["median_r"]
    assert row["verdict"] == "not_ready"
    assert "INVERTED" in row["verdict_reason"]


def test_negative_take_median_is_not_ready():
    records = _arm("range_qm", "take", [-0.3] * 12 + [0.4] * 10)
    records += _arm("range_qm", "abstain", [-0.9] * 21, start=100)
    row = _route(ev.evaluate_records(records))
    assert row["verdict"] == "not_ready"
    assert "negative" in row["verdict_reason"]


def test_thin_data_is_insufficient_with_n_named():
    records = _arm("range_qm", "take", [0.5] * 5)
    records += _arm("range_qm", "abstain", [-0.5] * 3, start=50)
    row = _route(ev.evaluate_records(records))
    assert row["verdict"] == "insufficient_data"
    assert "5/20" in row["verdict_reason"]
    assert "3/20" in row["verdict_reason"]
    assert row["evaluated"] == 8


def test_no_candidates_observed_is_its_own_fact():
    snap = ev.evaluate_records([])
    for playbook in ev.RANGE_PLAYBOOKS:
        row = _route(snap, playbook)
        assert row["verdict"] == "insufficient_data"
        assert "no candidates observed" in row["verdict_reason"]
        assert row["evaluated"] == 0
    assert snap["window_start"] is None


def test_unstamped_and_other_route_history_is_not_evaluated():
    """The window honesty: identical outcomes WITHOUT the
    route stamp (pre-wiring history), or stamped for the
    governing h1_trigger route (the same playbooks' roster
    verdicts — a different question), never enter the arms."""
    records = []
    for i in range(22):
        ts = _ts(i)
        records.append(_judgment(ts, "range_qm", "take",
                                 stamp=False))
        records.append(_recon(ts, "range_qm", realized_r=0.5))
        ts2 = _ts(100 + i)
        records.append(_judgment(ts2, "range_qm", "take",
                                 route="h1_trigger"))
        records.append(_recon(ts2, "range_qm", realized_r=0.5))
    row = _route(ev.evaluate_records(records))
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
    records.append(_judgment(ts, pb, "take"))
    records.append(_recon(ts, pb, realized_r=226.0, hit=True))
    records[-1]["move"] = 30.52
    # S3-native: R never journaled, exclusion on the record.
    ts = _ts(201)
    records.append(_judgment(ts, pb, "take"))
    records.append(_recon(ts, pb, realized_r=None, hit=True,
                          r_excluded="degenerate_stop"))
    row = _route(ev.evaluate_records(records), pb)
    assert row["excluded_degenerate_r"] == 2
    assert row["take"]["paired"] == 20
    assert row["take"]["median_r"] == pytest.approx(0.5)
    assert row["verdict"] == "ready"


def test_duplicates_never_double_grade():
    records = _arm("range_qm", "take", [0.5] * 3)
    doubled = records + [dict(r) for r in records]
    row = _route(ev.evaluate_records(doubled))
    assert row["evaluated"] == 3
    assert row["take"]["paired"] == 3
    # A DISTINCT verdict under the same (ts, playbook) key:
    # the pairing machinery keys on it, so the first counts
    # and the second is a counted duplicate, never graded.
    ts = _ts(50)
    extra = [_judgment(ts, "range_qm", "take"),
             _recon(ts, "range_qm", realized_r=0.5),
             _judgment(ts, "range_qm", "abstain"),
             _recon(ts, "range_qm", decision="abstain",
                    realized_r=-0.5, hit=False)]
    row = _route(ev.evaluate_records(records + extra))
    assert row["duplicate_key_records"] == 1
    assert row["take"]["paired"] == 4
    assert row["abstain"]["paired"] == 0


def test_only_the_4h_horizon_grades():
    records = []
    for i in range(22):
        ts = _ts(i)
        records.append(_judgment(ts, "range_qm", "take"))
        records.append(_recon(ts, "range_qm", horizon=1,
                              realized_r=0.5))
    row = _route(ev.evaluate_records(records))
    assert row["evaluated"] == 22
    assert row["take"]["paired"] == 0
    assert row["verdict"] == "insufficient_data"
    assert "0/20" in row["verdict_reason"]


def test_pending_outcomes_and_expiry():
    state = ev.new_state()
    records = [_judgment(_ts(i), "range_qm", "take")
               for i in range(3)]
    snap = ev.accumulate(state, records, now=_ts(3))
    row = _route(snap)
    assert row["evaluated"] == 3
    assert row["pending_outcomes"] == 3
    # 96h later with no reconciliations: unmeasurable, not
    # pending forever (the self-audit's 48h rule).
    snap = ev.accumulate(state, [], now="2026-10-05 00:00:00+00:00")
    row = _route(snap)
    assert row["pending_outcomes"] == 0
    assert row["unmeasurable_expired"] == 3


def test_window_start_is_the_first_stamped_verdict():
    records = _arm("range_qm", "take", [0.5], start=7)
    snap = ev.evaluate_records(records)
    assert snap["window_start"] == _ts(7)
    assert "route-stamped" in snap["window_basis"]
    assert "deployment" in snap["window_basis"]


def test_cross_window_accumulation_reaches_a_verdict():
    """The live shape: judgments land in one window, their
    +4h reconciliations in a later one; the standing state
    carries the pending set across, and the verdict arrives
    when the outcomes do."""
    state = ev.new_state()
    judgments, recons = [], []
    for i, r in enumerate([0.5] * 18 + [-0.5] * 4):
        ts = _ts(i)
        judgments.append(_judgment(ts, "range_qm", "take"))
        recons.append(_recon(ts, "range_qm", realized_r=r,
                             hit=r > 0))
    for i, r in enumerate([-0.4] * 15 + [0.3] * 6):
        ts = _ts(22 + i)
        judgments.append(_judgment(ts, "range_qm", "abstain"))
        recons.append(_recon(ts, "range_qm", decision="abstain",
                             realized_r=r, hit=r > 0))
    ev.accumulate(state, judgments, now=_ts(43))
    snap = ev.snapshot(state)
    assert _route(snap)["pending_outcomes"] == 43
    snap = ev.accumulate(state, recons, now=_ts(47))
    assert _route(snap)["verdict"] == "ready"
    assert _route(snap)["pending_outcomes"] == 0
