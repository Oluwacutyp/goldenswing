"""Judgment Core S4 — chronological validation (owner master
directive 2026-10-23; S4 of the judgment build).

The question S4 answers is NOT "did the account make money".
It is: **on identical inputs, whose DECISIONS were better —
the gate stack's or the judgment core's?**  Judged by decision
quality, never by balance (the directive).  The dataset is the
live journal itself:

- every ``judgment_shadow`` record already carries BOTH
  verdicts on the same candidate (the gate stack's result +
  reason, and the judgment core's take/abstain with its
  expected R and confidence);
- every ``judgment_reconciliation`` record (S3) carries what
  actually happened next, in the candidate's own geometry
  (realized R at +1h / +4h), for judged candidates whether or
  not the gates traded them.

Chronology is the honesty mechanism: pairs are walked in
judged-time order; the final segment is a LOCKED HOLDOUT —
reported with the same stated bars, never tuned on.  The
thresholds in the judgment config are stated constants, not
fitted values, so the holdout is honest by construction; this
module states that rather than implying a search happened.

Hard invariants stay outside the comparison (the ruling):
candidates the gate stack refused on HARD RISK grounds
(friction, solvency — category 8) are reported separately and
never scored as gate "mistakes": the judgment core never had
authority there, and pretending a friction refusal was a
decision error would corrupt the verdict.  Execution failures
(category 9) are execution quality, not decision quality —
also separate.  Invalid geometry / not-evaluated rows are not
decisions at all.

The output is an ACCEPTANCE REPORT for the owner's promotion
decision (first promotion, if any, is M5-route-only and needs
its own approval — this module promotes nothing).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .funnel import (
    CATEGORY_ACCEPTED,
    CATEGORY_EXECUTION,
    CATEGORY_HARD_RISK,
    CATEGORY_INVALID_GEOMETRY,
    CATEGORY_NOT_EVALUATED,
    ORDERED_RESULTS,
    TERMINAL_ORDER_REFUSALS,
)
from .judgment_audit import (
    calibration_add, calibration_view, degenerate_stop)

try:  # the code->category table is the funnel's own truth
    from .funnel import _CODE_TO_CATEGORY
except ImportError:  # pragma: no cover - table renamed upstream
    _CODE_TO_CATEGORY = {}

#: Gate classes (the comparison's rows).
GATE_EXECUTED = "executed"
GATE_REFUSED_WEIGHED = "refused_weighed"
GATE_REFUSED_HARD = "refused_hard"
GATE_EXECUTION_FAILED = "execution_failed"
GATE_NOT_A_DECISION = "not_a_decision"
GATE_UNCLASSIFIED = "unclassified"

#: Acceptance bars (stated, in R).  Decision quality only.
BAR_MIN_PAIRS = 60          # judged pairs with outcomes (eval segment)
BAR_MIN_HOLDOUT = 20        # pairs with outcomes in the holdout
BAR_CALIBRATION_GAP_R = 0.35
BAR_CALIBRATION_MIN_N = 10

HOLDOUT_FRAC = 0.30


def read_journal(path: str | Path) -> tuple[list[dict], int]:
    """Tolerant JSONL read: returns (records, malformed_count).
    A torn final line (the runner was mid-append at a crash) is
    counted, never fatal — the journal is evidence, and evidence
    with a torn edge still testifies."""
    records: list[dict] = []
    malformed = 0
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if isinstance(row, dict):
            records.append(row)
        else:
            malformed += 1
    return records, malformed


def _category_of(shadow: dict) -> int | None:
    code = shadow.get("gate_reason_code")
    if isinstance(code, int):
        return code
    if isinstance(code, str) and code:
        return _CODE_TO_CATEGORY.get(code)
    return None


def gate_class(shadow: dict) -> str:
    """What the gate stack actually did with this candidate."""
    order = str(shadow.get("gate_order_result") or "")
    if order in ORDERED_RESULTS:
        return GATE_EXECUTED
    category = _category_of(shadow)
    if order in TERMINAL_ORDER_REFUSALS \
            or category == CATEGORY_HARD_RISK:
        return GATE_REFUSED_HARD
    if category == CATEGORY_EXECUTION:
        return GATE_EXECUTION_FAILED
    if category in (CATEGORY_INVALID_GEOMETRY,
                    CATEGORY_NOT_EVALUATED):
        return GATE_NOT_A_DECISION
    if category == CATEGORY_ACCEPTED:
        return GATE_EXECUTED
    if category is not None:
        return GATE_REFUSED_WEIGHED
    result = str(shadow.get("gate_result") or "").lower()
    if result in ("accepted", "promoted", "executed", "filled"):
        return GATE_EXECUTED
    if result:
        return GATE_REFUSED_WEIGHED
    return GATE_UNCLASSIFIED


def build_pairs(records: list[dict],
                horizon_h: int = 4) -> list[dict]:
    """Join shadows to their reconciliations on (ts, playbook).

    One pair per judged candidate.  ``realized_r`` is None while
    the horizon outcome is still pending (the candidate was
    judged too recently) — pending pairs count toward coverage
    honesty, never toward outcome means.

    Degenerate-R guard (completion-04): a reconciled pair whose
    implied stop distance (|move| / |realized_r|, both journaled
    on the reconciliation) breaches the shared price floor
    (judgment_audit.degenerate_stop) is EXCLUDED from every
    R statistic — its realized_r is nulled and the pair carries
    ``r_excluded`` + the implied distance, so segment stats
    compute hit rates and means on the clean set and report the
    excluded count beside them.  The +226R record in the
    2026-11-05 evidence pack (a ~$0.135 stop) is the shape this
    exists for."""
    outcomes: dict[tuple[str, str], dict] = {}
    for row in records:
        if row.get("type") != "judgment_reconciliation":
            continue
        if int(row.get("horizon_h") or 0) != int(horizon_h):
            continue
        key = (str(row.get("judged_ts") or ""),
               str(row.get("playbook") or ""))
        outcomes[key] = row
    pairs: list[dict] = []
    for row in records:
        if row.get("type") != "judgment_shadow":
            continue
        judgment = row.get("judgment") or {}
        key = (str(row.get("ts") or ""),
               str(row.get("playbook") or ""))
        outcome = outcomes.get(key)
        realized_r = outcome.get("realized_r") if outcome \
            else None
        r_excluded = None
        implied_stop = None
        if outcome is not None and realized_r not in (None, 0):
            try:
                implied_stop = abs(float(outcome.get("move"))) \
                    / abs(float(realized_r))
                entry_f = float(outcome.get("entry"))
            except (TypeError, ValueError, ZeroDivisionError):
                implied_stop = None
                entry_f = None
            if implied_stop is not None and entry_f is not None \
                    and degenerate_stop(
                        entry_f,
                        entry_f - implied_stop) is not None:
                r_excluded = "degenerate_stop"
                realized_r = None
        pairs.append({
            "ts": str(row.get("ts") or ""),
            "playbook": str(row.get("playbook") or ""),
            "intended_tf": row.get("intended_tf"),
            "judgment_decision": judgment.get("decision"),
            "expected_r": judgment.get("expected_r"),
            "confidence": judgment.get("confidence"),
            "reason_category": judgment.get("reason_category"),
            "gate": gate_class(row),
            "gate_reason_code": row.get("gate_reason_code"),
            "realized_r": realized_r,
            "hit": (outcome.get("hit") if outcome else None),
            "r_excluded": r_excluded,
            "implied_stop_distance": (
                round(implied_stop, 4)
                if implied_stop is not None else None),
        })
    pairs.sort(key=lambda p: p["ts"])
    return pairs


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _segment_stats(pairs: list[dict]) -> dict[str, Any]:
    done = [p for p in pairs if p["realized_r"] is not None]
    excluded = [p for p in pairs if p.get("r_excluded")]
    takes = [p for p in done if p["judgment_decision"] == "take"]
    abstains = [p for p in done
                if p["judgment_decision"] == "abstain"]
    executed = [p for p in done if p["gate"] == GATE_EXECUTED]
    refused_w = [p for p in done
                 if p["gate"] == GATE_REFUSED_WEIGHED]

    def cell(j_decision: str, gate: str) -> dict[str, Any]:
        rows = [p for p in done
                if p["judgment_decision"] == j_decision
                and p["gate"] == gate]
        return {"n": len(rows),
                "mean_r": _mean([float(p["realized_r"])
                                 for p in rows]),
                "hit_rate": (round(
                    sum(1 for p in rows if p["hit"]) / len(rows), 3)
                    if rows else None)}

    buckets: dict[str, Any] = {}
    for p in done:
        calibration_add(
            buckets, confidence=float(p.get("confidence") or 0.0),
            expected_r=p.get("expected_r"),
            realized_r=float(p["realized_r"]))
    hard = [p for p in done if p["gate"] == GATE_REFUSED_HARD]
    return {
        "pairs": len(pairs),
        "with_outcome": len(done),
        "pending": len(pairs) - len(done) - len(excluded),
        "excluded_degenerate_r": len(excluded),
        "judgment": {
            "take_n": len(takes),
            "take_mean_r": _mean([float(p["realized_r"])
                                  for p in takes]),
            "take_hit_rate": (round(
                sum(1 for p in takes if p["hit"]) / len(takes), 3)
                if takes else None),
            "abstain_n": len(abstains),
            "abstain_mean_r": _mean([float(p["realized_r"])
                                     for p in abstains]),
        },
        "gates": {
            "executed_n": len(executed),
            "executed_mean_r": _mean([float(p["realized_r"])
                                      for p in executed]),
            "refused_weighed_n": len(refused_w),
            "refused_weighed_mean_r": _mean(
                [float(p["realized_r"]) for p in refused_w]),
        },
        "cells": {
            "take_executed": cell("take", GATE_EXECUTED),
            "take_refused_weighed": cell(
                "take", GATE_REFUSED_WEIGHED),
            "abstain_executed": cell("abstain", GATE_EXECUTED),
            "abstain_refused_weighed": cell(
                "abstain", GATE_REFUSED_WEIGHED),
        },
        "hard_refusals": {
            "n": len(hard),
            "judgment_take_n": sum(
                1 for p in hard
                if p["judgment_decision"] == "take"),
            "mean_r": _mean([float(p["realized_r"]) for p in hard]),
        },
        "execution_failed_n": sum(
            1 for p in done if p["gate"] == GATE_EXECUTION_FAILED),
        "not_a_decision_n": sum(
            1 for p in pairs if p["gate"] == GATE_NOT_A_DECISION),
        "calibration": calibration_view(buckets),
    }


def _bar(ok: bool | None) -> str:
    return {True: "met", False: "unmet", None: "insufficient_data"}[ok]


def validate(records: list[dict], *, horizon_h: int = 4,
             malformed: int = 0) -> dict[str, Any]:
    """The full chronological comparison + acceptance verdict."""
    pairs = build_pairs(records, horizon_h=horizon_h)
    done = [p for p in pairs if p["realized_r"] is not None]
    split = int(len(done) * (1.0 - HOLDOUT_FRAC))
    # Segment membership follows judged-time order of the
    # OUTCOME-BEARING pairs; pending pairs ride with the segment
    # their judged time falls in, for honest coverage counts.
    if split:
        boundary = done[split - 1]["ts"]
        eval_pairs = [p for p in pairs if p["ts"] <= boundary]
        holdout_pairs = [p for p in pairs if p["ts"] > boundary]
    else:
        eval_pairs, holdout_pairs = [], list(pairs)
    evaluation = _segment_stats(eval_pairs)
    holdout = _segment_stats(holdout_pairs)

    def _gt(a: Any, b: Any) -> bool | None:
        if a is None or b is None:
            return None
        return a > b

    def _lt(a: Any, b: Any) -> bool | None:
        if a is None or b is None:
            return None
        return a < b

    ej = evaluation["judgment"]
    hj = holdout["judgment"]
    cells = evaluation["cells"]
    cal_gaps = [
        abs(float(b["mean_expected_r"]) - float(b["mean_realized_r"]))
        for b in (evaluation["calibration"] or {}).values()
        if b.get("n", 0) >= BAR_CALIBRATION_MIN_N
        and b.get("mean_expected_r") is not None
        and b.get("mean_realized_r") is not None]
    bars = {
        "coverage": _bar(
            evaluation["with_outcome"] >= BAR_MIN_PAIRS
            if evaluation["with_outcome"] else False),
        "discrimination_eval": _bar(_gt(
            ej["take_mean_r"], ej["abstain_mean_r"])),
        "discrimination_holdout": _bar(_gt(
            hj["take_mean_r"], hj["abstain_mean_r"])
            if holdout["with_outcome"] >= BAR_MIN_HOLDOUT
            else None),
        "left_on_table": _bar(_gt(
            cells["take_refused_weighed"]["mean_r"], 0.0)
            if cells["take_refused_weighed"]["n"] else None),
        "protection": _bar(_lt(
            cells["abstain_executed"]["mean_r"], ej["take_mean_r"])
            if cells["abstain_executed"]["n"] else None),
        "calibration": _bar(
            max(cal_gaps) <= BAR_CALIBRATION_GAP_R
            if cal_gaps else None),
    }
    ready = bars["coverage"] == "met" \
        and bars["discrimination_eval"] == "met" \
        and bars["discrimination_holdout"] == "met" \
        and bars["calibration"] == "met"
    if evaluation["with_outcome"] < BAR_MIN_PAIRS:
        verdict = "insufficient_data"
    elif ready:
        verdict = "ready_for_owner_decision"
    else:
        verdict = "not_ready"
    return {
        "horizon_h": int(horizon_h),
        "journal_records": len(records),
        "malformed_lines": malformed,
        "pairs_total": len(pairs),
        "excluded_degenerate_r": (
            evaluation["excluded_degenerate_r"]
            + holdout["excluded_degenerate_r"]),
        "evaluation": evaluation,
        "holdout": holdout,
        "bars": bars,
        "verdict": verdict,
        "note": (
            "Decision quality only — realized R is the forward "
            "outcome in each candidate's own geometry, identical "
            "for both decision models. Hard-risk refusals are "
            "reported, never scored. The holdout was evaluated "
            "with the same stated bars and never tuned on."),
    }


def _r(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.2f}R"


def render_markdown(result: dict[str, Any]) -> str:
    """Plain language first (the owner's rule), tables after."""
    ev = result["evaluation"]
    ho = result["holdout"]
    ej, hj = ev["judgment"], ho["judgment"]
    cells = ev["cells"]
    verdict_line = {
        "ready_for_owner_decision": (
            "The judgment core's decisions beat the gate stack's "
            "on the same candidates, in both the evaluation window "
            "and the untouched holdout. This is the evidence for "
            "the owner's promotion decision (M5 route only, "
            "separate approval) — nothing is promoted by this "
            "report."),
        "not_ready": (
            "On the same candidates, the judgment core has not "
            "yet out-decided the gate stack by the stated bars. "
            "It stays shadow. The bars below say exactly where "
            "it fell short."),
        "insufficient_data": (
            "There is not enough judged history yet to call this "
            "— the shadow journal needs more reconciled pairs "
            "before any verdict is honest."),
    }[result["verdict"]]
    lines = [
        "# Judgment Core — S4 acceptance report",
        "",
        verdict_line,
        "",
        f"Pairs judged: {result['pairs_total']} "
        f"(+{result['horizon_h']}h outcomes; "
        f"{result['malformed_lines']} torn journal lines "
        "skipped).",
        "",
        "## The comparison (evaluation window)",
        "",
        f"- Judgment said TAKE {ej['take_n']} times: those "
        f"candidates went {_r(ej['take_mean_r'])} on average. "
        f"It ABSTAINED {ej['abstain_n']} times: those went "
        f"{_r(ej['abstain_mean_r'])} — the gap between the two "
        "is whether its judgment discriminates at all.",
        f"- The gates EXECUTED {ev['gates']['executed_n']} "
        f"({_r(ev['gates']['executed_mean_r'])}) and refused on "
        f"weighed grounds {ev['gates']['refused_weighed_n']} "
        f"({_r(ev['gates']['refused_weighed_mean_r'])}).",
        f"- Judgment wanted {cells['take_refused_weighed']['n']} "
        "the gates weighed out: they went "
        f"{_r(cells['take_refused_weighed']['mean_r'])} — "
        "positive means the gates are leaving money on the "
        "table; negative means their refusals were right.",
        f"- The gates took {cells['abstain_executed']['n']} that "
        "judgment would have skipped: they went "
        f"{_r(cells['abstain_executed']['mean_r'])}.",
        f"- Hard-risk refusals (friction/solvency): "
        f"{ev['hard_refusals']['n']} — reported, never scored; "
        f"judgment said take on {ev['hard_refusals']['judgment_take_n']} "
        "of them, which changes nothing: that boundary is not "
        "judgment's to cross.",
        "",
        "## Holdout (locked, never tuned on)",
        "",
        f"- {ho['with_outcome']} pairs with outcomes. Judgment "
        f"TAKE {_r(hj['take_mean_r'])} (n={hj['take_n']}) vs "
        f"ABSTAIN {_r(hj['abstain_mean_r'])} "
        f"(n={hj['abstain_n']}).",
        "",
        "## Acceptance bars",
        "",
    ]
    for name, state in result["bars"].items():
        lines.append(f"- {name}: **{state}**")
    lines += ["", f"*{result['note']}*"]
    return "\n".join(lines)
