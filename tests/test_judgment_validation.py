"""Judgment Core S4 — chronological validation.

Pins: the shadow⋈reconciliation join on (ts, playbook, horizon);
gate classification (executed / weighed refusal / HARD refusal /
execution failure / not-a-decision) from the funnel's own
categories; hard refusals reported but NEVER scored; the
locked-holdout split by judged time; decision-quality bars and
their verdicts; torn-line tolerance in the journal reader.
"""
from __future__ import annotations

import json

from goldenswing.execution.judgment_validation import (
    GATE_EXECUTED,
    GATE_EXECUTION_FAILED,
    GATE_NOT_A_DECISION,
    GATE_REFUSED_HARD,
    GATE_REFUSED_WEIGHED,
    build_pairs,
    gate_class,
    read_journal,
    render_markdown,
    validate,
)


def _shadow(ts: str, *, decision="take", expected=0.4, conf=0.6,
            gate_order=None, gate_code=None, gate_result=None,
            playbook="qm_m5"):
    return {
        "type": "judgment_shadow", "ts": ts, "playbook": playbook,
        "intended_tf": "M5",
        "gate_result": gate_result,
        "gate_order_result": gate_order,
        "gate_reason_code": gate_code,
        "judgment": {"decision": decision, "expected_r": expected,
                     "confidence": conf,
                     "reason_category": 4},
    }


def _recon(ts: str, *, r: float, horizon=4, playbook="qm_m5"):
    return {
        "type": "judgment_reconciliation", "judged_ts": ts,
        "playbook": playbook, "direction": 1, "decision": "take",
        "horizon_h": horizon, "entry": 4000.0,
        "exit_price": 4000.0 + r, "realized_r": r,
        "move": r, "move_atr": 0.5, "hit": r > 0,
    }


def _pair(ts, r, **kw):
    return [_shadow(ts, **kw), _recon(ts, r=r)]


# ── gate classification ─────────────────────────────────────────


def test_gate_class_reads_the_funnels_own_categories():
    assert gate_class(_shadow("t", gate_order="filled")) \
        == GATE_EXECUTED
    assert gate_class(_shadow("t", gate_code="gate.friction")) \
        == GATE_REFUSED_HARD
    assert gate_class(_shadow(
        "t", gate_order="refused_in_order_path",
        gate_code="floor.total")) == GATE_REFUSED_HARD
    assert gate_class(_shadow("t", gate_code="floor.total")) \
        == GATE_REFUSED_WEIGHED
    assert gate_class(_shadow(
        "t", gate_code="direction.flip")) == GATE_REFUSED_WEIGHED
    assert gate_class(_shadow(
        "t", gate_code="order.broker_rejected")) \
        == GATE_EXECUTION_FAILED
    assert gate_class(_shadow(
        "t", gate_code="path.warmup_ineligible")) \
        == GATE_NOT_A_DECISION
    assert gate_class(_shadow("t", gate_code=8)) \
        == GATE_REFUSED_HARD  # numeric category passthrough


# ── the join ────────────────────────────────────────────────────


def test_pairs_join_on_ts_playbook_and_horizon():
    records = (
        _pair("2026-11-01T00:00:00", 1.5)
        + _pair("2026-11-01T01:00:00", -1.0, playbook="v_rev")
        + [_shadow("2026-11-01T02:00:00")]  # no outcome yet
        + [_recon("2026-11-01T00:00:00", r=9.9, horizon=1)]
    )
    pairs = build_pairs(records, horizon_h=4)
    assert len(pairs) == 3
    assert pairs[0]["realized_r"] == 1.5     # not the 1h 9.9
    assert pairs[1]["playbook"] == "v_rev"
    assert pairs[2]["realized_r"] is None    # pending stays honest
    pairs_1h = build_pairs(records, horizon_h=1)
    assert pairs_1h[0]["realized_r"] == 9.9


def test_torn_journal_lines_are_counted_not_fatal(tmp_path):
    path = tmp_path / "j.jsonl"
    good = json.dumps(_shadow("2026-11-01T00:00:00"))
    path.write_text(good + "\n{torn line\n" + good + "\n",
                    encoding="utf-8")
    records, malformed = read_journal(path)
    assert len(records) == 2 and malformed == 1


# ── the comparison ──────────────────────────────────────────────


def _dataset(*, take_r=1.0, abstain_r=-0.5, n=100):
    """n judged pairs, chronologically ordered: judgment takes
    do take_r, abstains do abstain_r; the gates execute the
    takes and weigh-refuse the abstains."""
    records = []
    for i in range(n):
        ts = f"2026-11-{(i % 27) + 1:02d}T{i % 24:02d}:00:00"
        if i % 2 == 0:
            records += _pair(ts, take_r, decision="take",
                             gate_order="filled")
        else:
            records += _pair(ts, abstain_r, decision="abstain",
                             gate_code="floor.total")
    return records


def test_ready_verdict_when_judgment_outdecides():
    result = validate(_dataset())
    assert result["verdict"] == "ready_for_owner_decision", result
    assert result["bars"]["discrimination_eval"] == "met"
    assert result["bars"]["discrimination_holdout"] == "met"
    report = render_markdown(result)
    assert "promotion decision" in report
    assert "Holdout" in report


def test_not_ready_when_judgment_is_worse():
    result = validate(_dataset(take_r=-0.8, abstain_r=0.6))
    assert result["verdict"] == "not_ready"
    assert result["bars"]["discrimination_eval"] == "unmet"


def test_insufficient_data_is_a_verdict_not_a_crash():
    result = validate(_dataset(n=10))
    assert result["verdict"] == "insufficient_data"
    assert result["bars"]["coverage"] == "unmet"


def test_hard_refusals_are_reported_never_scored():
    records = _dataset(n=80)
    # 20 friction-refused candidates judgment loved, all mooned —
    # if these leaked into the weighed cells, left_on_table
    # would light up and libel the friction gate.
    for i in range(20):
        ts = f"2026-11-{(i % 27) + 1:02d}T{i % 24:02d}:30:00"
        records += _pair(ts, 5.0, decision="take",
                         gate_code="gate.friction")
    result = validate(records)
    hard = result["evaluation"]["hard_refusals"]
    assert hard["n"] >= 10 and hard["judgment_take_n"] >= 10
    assert hard["mean_r"] == 5.0
    cells = result["evaluation"]["cells"]
    assert cells["take_refused_weighed"]["mean_r"] is None \
        or cells["take_refused_weighed"]["mean_r"] < 5.0


def test_holdout_is_the_chronological_tail():
    records = _dataset(take_r=1.0, abstain_r=-0.5, n=100)
    result = validate(records)
    ev, ho = result["evaluation"], result["holdout"]
    assert ho["with_outcome"] >= 20
    assert ev["with_outcome"] > ho["with_outcome"]
    assert abs(ev["with_outcome"] + ho["with_outcome"]
               - 100) <= 30  # pending stragglers only shrink it
