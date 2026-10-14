"""Start the GoldenSwing paper loop (live feed, paper money).

Paper only: OANDA multi-TF feed (D1/H4/H1/M15/M30/M5) -> hierarchy
gate (D1 bias -> H4 zones -> H1 trigger -> session entry clock) ->
playbook signals -> confluence -> Telegram alerts -> paper fills ->
journal.  No MT5, no live orders.

A signal that fails the hierarchy NEVER fires — that is the correct
new behaviour, not a regression (a trigger without its
higher-timeframe zone/bias is not a trade).

Telegram/Email credentials come from the environment (GS_TELEGRAM_BOT_TOKEN,
GS_TELEGRAM_CHAT_ID, ...); when absent, alerts land in the outbox dir.

Usage:
    python3 scripts/paper_live.py [--max-bars N] [--balance X]
    python3 scripts/paper_live.py --once      # one cron tick, then exit

--once processes any new closed bars since the last tick and exits.
State (bar buffer, playbook streaming state, paper positions, MTF
stack) persists in data/paper_state/ between ticks.  See SYSTEM.md
§9.5 for the cron pattern.  Never run --once concurrently with the
daemon loop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from goldenswing.execution.runner import LiveRunner, RunnerConfig
from goldenswing.feed import MultiOandaFeed, MultiOandaFeedConfig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing paper loop (live)")
    ap.add_argument("--max-bars", type=int, default=None,
                    help="stop after N new bars (default: run forever)")
    ap.add_argument("--balance", type=float, default=None,
                    help="paper starting balance")
    ap.add_argument("--poll-seconds", type=float, default=300.0,
                    help="seconds between feed polls")
    ap.add_argument("--once", action="store_true",
                    help="process new closed bars since the last tick, "
                         "then exit (cron mode; state in data/paper_state/)")
    args = ap.parse_args(argv)

    overrides = {}
    if args.balance:
        overrides["paper_balance"] = args.balance
    cfg = RunnerConfig.from_env(**overrides)
    # Production paper loop: hierarchy REQUIRED — a missing stack
    # blocks every signal instead of falling back to H1-only trading.
    cfg.hierarchy_required = True
    runner = LiveRunner(cfg)
    feed = MultiOandaFeed(
        MultiOandaFeedConfig(poll_seconds=args.poll_seconds))
    if args.once:
        print("paper --once: one cron tick, "
              f"journal {cfg.journal_path}")
        summary = runner.run_paper_once(feed)
        for key, value in summary.items():
            print(f"{key}: {value}")
        return
    print("paper loop starting: live OANDA multi-TF XAU/USD feed, "
          "hierarchy-gated, paper executor, "
          f"journal {cfg.journal_path}")
    summary = runner.run_paper_loop(feed, max_bars=args.max_bars)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
