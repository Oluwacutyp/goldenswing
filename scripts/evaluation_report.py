"""Build the §G range-family evaluation from an exported
journal — the rerunnable companion to the live cycle (the
owner-facing numbers can be regenerated per journal pull,
from the journal itself).

Usage:
    python3 scripts/evaluation_report.py <journal.jsonl>
        [--json out.json]

Method (identical to the live evaluation —
goldenswing/execution/evaluation.py, which this script
drives over the full record list):

- Only §G route-stamped judgment records count
  (judgment.authority_route naming one of the four range
  routes).  Unstamped history predates the route wiring (or
  belongs to the h1_trigger route, a different question)
  and is not evaluated: the window starts at the wiring's
  deployment.
- Outcomes pair via the S4 machinery
  (judgment_validation.build_pairs) at +4h, in each
  candidate's own geometry.  Degenerate-R pairs are excluded
  from every statistic and counted beside them.
- Per route and arm: paired n, hit rate, MEDIAN realized R —
  never a mean alone.  A route whose would-abstains outperform
  its would-takes (median or hit rate) reads INVERTED and can
  never read ready; below 20 paired outcomes per arm the
  verdict is insufficient_data with the n named.

A verdict is a MEASUREMENT only — it grants, promotes,
demotes and changes nothing (the promotion gate is deleted).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goldenswing.execution.judgment_validation import read_journal
from goldenswing.execution.evaluation import (
    HORIZON_H, MIN_PAIRED_PER_ARM, RANGE_PLAYBOOKS,
    evaluate_records)


def _r(value) -> str:
    return "—" if value is None else f"{float(value):+.2f}R"


def _pct(value) -> str:
    return "—" if value is None else f"{float(value):.0%}"


def render(snapshot: dict, *, journal: str, records: int,
           malformed: int) -> str:
    lines = [
        "# §G Range Family — Evaluation",
        "",
        f"Journal: {journal} ({records} records, {malformed} "
        f"torn lines skipped). Graded at +{HORIZON_H}h; floor "
        f"{MIN_PAIRED_PER_ARM} paired outcomes per arm; "
        "degenerate-R pairs excluded and counted.",
        f"Evaluation window opened: "
        f"{snapshot.get('window_start') or '—'} "
        "(route-stamped judgment history only — earlier history "
        "carries no stamp and is not evaluated).",
        "",
    ]
    for playbook in RANGE_PLAYBOOKS:
        row = snapshot["routes"][playbook]
        take, abstain = row["take"], row["abstain"]
        lines += [
            f"## {playbook} — {row['verdict']}",
            "",
            f"- evaluated: {row['evaluated']} §G candidates "
            f"({row['pending_outcomes']} awaiting outcomes, "
            f"{row['excluded_degenerate_r']} degenerate-R "
            f"excluded, {row['duplicate_key_records']} "
            "duplicate-key records not double-graded)",
            f"- would-take: {take['paired']} paired · hit "
            f"{_pct(take['hit_rate'])} · median "
            f"{_r(take['median_r'])}",
            f"- would-abstain: {abstain['paired']} paired · "
            f"hit {_pct(abstain['hit_rate'])} · median "
            f"{_r(abstain['median_r'])}",
            f"- {row['verdict_reason']}",
            "",
        ]
    lines.append(
        "*A verdict is a measurement only — it grants, promotes, "
        "demotes and changes nothing.*")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="§G range-family evaluation "
                    "over an exported journal")
    parser.add_argument("journal", help="path to mt5_journal.jsonl")
    parser.add_argument("--json", default=None,
                        help="also write the snapshot JSON here")
    args = parser.parse_args()
    records, malformed = read_journal(args.journal)
    snapshot = evaluate_records(records)
    print(render(snapshot, journal=args.journal,
                 records=len(records), malformed=malformed))
    if args.json:
        Path(args.json).write_text(
            json.dumps(snapshot, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
