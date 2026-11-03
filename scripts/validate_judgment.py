#!/usr/bin/env python3
"""S4 acceptance report runner.

Reads the live journal (mt5_journal.jsonl), joins every
judgment_shadow verdict to its S3 reconciliation, and prints
the chronological decision-quality comparison between the
judgment core and the gate stack — evaluation window + locked
holdout, acceptance bars, one verdict.

    python3 scripts/validate_judgment.py --journal data/mt5_journal.jsonl
    python3 scripts/validate_judgment.py --journal ... --horizon 1 --out report.md

Exit codes: 0 = report produced (its verdict may still be
insufficient_data / not_ready — those are findings, not tool
failures); 2 = the journal could not be read at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goldenswing.execution.judgment_validation import (  # noqa: E402
    read_journal,
    render_markdown,
    validate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True,
                        help="path to mt5_journal.jsonl")
    parser.add_argument("--horizon", type=int, default=4,
                        choices=(1, 4),
                        help="reconciliation horizon in hours")
    parser.add_argument("--out", default="",
                        help="also write the markdown report here")
    parser.add_argument("--json", action="store_true",
                        help="print the full result as JSON "
                             "instead of markdown")
    args = parser.parse_args()
    try:
        records, malformed = read_journal(args.journal)
    except OSError as exc:
        print(f"cannot read journal: {exc}", file=sys.stderr)
        return 2
    result = validate(records, horizon_h=args.horizon,
                      malformed=malformed)
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        report = render_markdown(result)
        print(report)
        if args.out:
            Path(args.out).write_text(report + "\n",
                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
