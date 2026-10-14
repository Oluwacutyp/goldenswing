"""Reset a latched risk halt — the human action SYSTEM.md §7.4 describes.

Usage:
    python scripts/risk_reset.py --reason "tail event, regime intact"
    python scripts/risk_reset.py --reason "..." --kill-switch  # also clear KILL

The halt is latched by design: a halted system never resumes itself.
This script prints the current halt state, requires an explicit
--reason (journaled), and refuses to run without it.  It never touches
the trade journal.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.risk import RiskConfig, RiskGuards


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reason", required=True,
                    help="why the halt is being cleared (journaled)")
    ap.add_argument("--state", default=None,
                    help="risk_state.json path (default: data/risk_state.json)")
    ap.add_argument("--journal", default=None,
                    help="trade journal path (default: data/trade_journal.jsonl)")
    ap.add_argument("--kill-switch", action="store_true",
                    help="also remove the KILL file if present")
    ap.add_argument("--kill-path", default=None,
                    help="KILL file path (default: data/KILL)")
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    state_path = Path(args.state) if args.state else repo / "data" / "risk_state.json"
    guards = RiskGuards(state_path, RiskConfig())
    st = guards.state
    print(f"halted: {st.get('halted')}")
    print(f"halt_reason: {st.get('halt_reason') or '(none)'}")
    if not st.get("halted"):
        print("nothing to reset — the halt is not latched.")
        return 0

    guards.reset_halt()
    print("halt cleared.")

    if args.kill_switch:
        kp = Path(args.kill_path) if args.kill_path else repo / "data" / "KILL"
        if kp.exists():
            kp.unlink()
            print(f"removed {kp}")
        else:
            print("no KILL file present")

    journal_path = Path(args.journal) if args.journal else repo / "data" / "trade_journal.jsonl"
    TradeJournal(journal_path).log_note(
        f"RISK HALT RESET by operator: {args.reason}")
    print(f"journaled reset rationale to {journal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
