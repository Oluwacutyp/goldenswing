"""Paper OOS report: per-playbook stats from the paper trade journal.

The paper loop journals every fill (open) and every managed close with
playbook, direction, score, r_net and timestamps.  This script joins
them into the same shape the walk-forward/promotion analysis consumes:
per-playbook trade counts, total R, profit factor, win rate — the OOS
growth the paper loop exists to produce.

Stats come from goldenswing.execution.reporting (shared with the
Telegram /report command) — one implementation, two surfaces.

Usage: python3 scripts/paper_report.py [journal_path]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from goldenswing.execution.reporting import (open_positions,  # noqa: E402
                                             paper_book_stats)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        REPO_ROOT / "data" / "trade_journal.jsonl"
    from goldenswing.execution.learning import TradeJournal
    journal = TradeJournal(path)
    stats = paper_book_stats(journal)
    per_pb = stats["per_playbook"]
    if not any(s["trades"] for s in per_pb.values()):
        print(f"no paper opens in {path}")
        return 1
    print(f"paper journal: {path}")
    print(f"{'playbook':<18}{'trades':>8}{'win%':>7}{'+R':>9}"
          f"{'avgR':>8}{'PF':>7}{'pnl':>12}")
    total = 0
    for pb, s in sorted(per_pb.items()):
        if s["trades"] == 0:
            continue
        total += s["trades"]
        pf = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
        print(f"{pb:<18}{s['trades']:>8}{s['win_rate']:>6.0f}%"
              f"{s['total_r']:>+8.2f}{s['avg_r']:>+7.2f}"
              f"{pf:>7}{s['total_pnl']:>+11.2f}")
    unclosed = len(open_positions(journal))
    print(f"total closed paper trades: {total} "
          f"({unclosed} opens without a close record)")
    print("promotion bar: >=50 OOS trades per sleeve + stability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
