"""Start the $12 Exness-model paper loop, SIDE-BY-SIDE with the live loop.

Paper only: same signal pipeline as ``scripts/paper_live.py`` (OANDA
spot XAU feed -> live_signal -> confluence -> Telegram alerts ->
paper fills -> journal), but every fill is priced and margined like an
Exness Standard MT5 $12 account:

- $12.00 start, fixed 0.01 lot = 1 oz (no 1%-risk sizing, no
  forecaster sizing dial — the Exness timing model sizes by timing,
  not R).
- 1:2000 leverage (assumed constant — see the caveat below),
  $0.20/oz spread, $0.00 commission.
- Margin + equity stop-out (equity <= 0) via ExnessPaperExecutor.
- R-based risk guards DISABLED (daily/weekly/drawdown/consecutive/
  exposure caps all 0): margin + stop-out are the real constraint on
  a $12 timing account. Kill switch, require_stops, and
  max_positions=1 still enforced.

ISOLATION (never confuse the two loops):

- Journal: ``data/exness12/paper_journal.jsonl`` (live: ``data/trade_journal.jsonl``)
- State:  ``data/exness12/paper_state/`` (live: ``data/paper_state/``)
- Risk:   ``data/exness12/risk_state.json`` (live: ``data/risk_state.json``)
- Kill:   ``data/exness12/KILL`` (live: ``data/KILL``)
- Outbox (dry mode): ``data/exness12/outbox/``
- Telegram: same bot/chat, but every alert carries ``Mode: exness12``
  (live loop alerts carry ``Mode: swing``).
- Telegram COMMANDS are NOT polled here (``poll_commands=False``):
  the live loop owns the getUpdates poller; a second consumer would
  steal its commands. This loop only SENDS.

CPI TIMING: this loop ticks hourly at :38 and processes only CLOSED
H1 bars, so the first tick that can evaluate the CPI bar (8:00-9:00am
ET) is 9:38am ET (7:38am MDT), NOT 8:38am. Silence at 8:38am ET is
normal, not failure.

NEWS-LEVERAGE CAVEAT: Exness can restrict / reduce leverage around
news / weekends / high-margin-requirement periods. This loop assumes
constant 1:2000. A live $12 trade on the CPI spike could face higher
margin + stop-out beyond what is modeled. The assumed leverage is
recorded in the Exness journal on every start so the CPI result is
read with its assumption attached.

Usage:
    python3 scripts/paper_live_exness.py --once      # one cron tick
    python3 scripts/paper_live_exness.py --max-bars N  # daemon, N bars

Cron (hourly at :38, matching the live loop's cadence):
    38 * * * * cd ~/workspace/goldenswing && python3 scripts/paper_live_exness.py --once >> data/exness12/paper_cron.log 2>&1

Never run --once concurrently with the daemon form of THIS loop.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from goldenswing.execution.exness_paper import (
    COMMISSION_PER_TRADE,
    LEVERAGE,
    LOT_OZ,
    SPREAD_PER_OZ,
    START_BALANCE_12,
    ExnessPaperExecutor,
)
from goldenswing.execution.risk import RiskConfig
from goldenswing.execution.runner import LiveRunner, RunnerConfig
from goldenswing.feed import OandaFeed, OandaFeedConfig

EXNESS_DIR = REPO_ROOT / "data" / "exness12"
JOURNAL_PATH = EXNESS_DIR / "paper_journal.jsonl"
RISK_STATE_PATH = EXNESS_DIR / "risk_state.json"
KILL_SWITCH_PATH = EXNESS_DIR / "KILL"
STATE_STORE_PATH = EXNESS_DIR / "paper_state"
STATE_DIR = EXNESS_DIR / "live_state"
FIXED_LOTS = LOT_OZ / 100.0  # 1 oz = 0.01 lot

ASSUMED_MODEL_NOTE = (
    "exness12 paper loop started: $12.00 start, fixed 0.01 lot = 1 oz, "
    f"leverage 1:{LEVERAGE:.0f} ASSUMED CONSTANT, spread "
    f"${SPREAD_PER_OZ:.2f}/oz, commission ${COMMISSION_PER_TRADE:.2f}, "
    "margin + equity stop-out (equity <= 0), R-guards disabled. "
    "CAVEAT: Exness can restrict/reduce leverage around news/weekends — "
    "1:2000 is NOT guaranteed at CPI; read results with this assumption."
)


def build_runner() -> LiveRunner:
    """The Exness loop: separate everything from the live loop."""
    # Dry-mode outbox must not collide with the live loop's outbox:
    # alert_ids are (instrument, bar, direction), identical across
    # loops, so shared outbox files would overwrite each other.
    os.environ.setdefault("GS_ALERT_OUTBOX", str(EXNESS_DIR / "outbox"))
    cfg = RunnerConfig(
        instrument="XAUUSD",
        mode="exness12",  # surfaces as "Mode: exness12" on every alert
        min_score=60.0,
        auto_execute=False,
        risk_pct=0.01,  # unused in fixed-lots mode; kept for the record
        paper_balance=START_BALANCE_12,
        journal_path=JOURNAL_PATH,
        risk_state_path=RISK_STATE_PATH,
        kill_switch_path=KILL_SWITCH_PATH,
        state_dir=STATE_DIR,
        fixed_lots=FIXED_LOTS,
        # Exness timing model: margin + stop-out are the constraint.
        # R-based guards disabled (all 0); kill switch, require_stops
        # and max_positions=1 still enforced.  (Daily/weekly loss
        # caps no longer exist in RiskConfig — owner ruling
        # 2026-10-23: max drawdown is the sole account-level halt,
        # and it is 0 = disabled here.)
        risk=RiskConfig(
            max_drawdown_pct=0.0,
            max_consecutive_losses=0,
            max_positions=1,
            max_total_risk_pct=0.0,
            require_stops=True,
        ),
    )
    paper = ExnessPaperExecutor(
        balance=START_BALANCE_12,
        spread_cost=SPREAD_PER_OZ,
        leverage=LEVERAGE,
    )
    runner = LiveRunner(cfg, paper_executor=paper)
    runner.journal.log_note(ASSUMED_MODEL_NOTE)
    return runner


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="GoldenSwing $12 Exness-model paper loop (side-by-side)")
    ap.add_argument("--max-bars", type=int, default=None,
                    help="stop after N new bars (default: run forever)")
    ap.add_argument("--poll-seconds", type=float, default=300.0,
                    help="seconds between feed polls")
    ap.add_argument("--once", action="store_true",
                    help="process new closed bars since the last tick, "
                         "then exit (cron mode; state in "
                         "data/exness12/paper_state/)")
    args = ap.parse_args(argv)

    runner = build_runner()
    feed = OandaFeed(OandaFeedConfig(poll_seconds=args.poll_seconds))
    if args.once:
        print("exness12 paper --once: one cron tick, "
              f"journal {JOURNAL_PATH}")
        summary = runner.run_paper_once(
            feed, state_store_path=STATE_STORE_PATH, poll_commands=False)
        for key, value in summary.items():
            print(f"{key}: {value}")
        return
    print("exness12 paper loop starting: live OANDA XAU/USD feed, "
          "Exness paper executor, "
          f"journal {JOURNAL_PATH}")
    summary = runner.run_paper_loop(feed, max_bars=args.max_bars,
                                    poll_commands=False)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
