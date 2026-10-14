"""Paper-loop rehearsal: the full loop against recent history as-if live.

Runs ``LiveRunner.run_paper_loop`` over a ``ReplayFeed`` (warm-started
on earlier history, then N bars replayed one-by-one like live closes),
and proves the whole chain:

  signal fired -> alert rendered -> paper fill -> position managed ->
  journal entry (open + close) -> performance aggregates.

Paper money only.  Exits non-zero when any link of the chain is
missing.  Prints the rehearsal report.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from goldenswing.alerts import EmailSender, TelegramSender
from goldenswing.backtest.data import load_gold_h1
from goldenswing.execution.runner import LiveRunner, RunnerConfig
from goldenswing.feed import ReplayFeed

WARM_BARS = 600
REPLAY_BARS = 500


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="paper_rehearsal_"))
    outbox = tmp / "outbox"
    journal = tmp / "journal.jsonl"

    df = load_gold_h1()
    feed_df = df.iloc[-(WARM_BARS + REPLAY_BARS):].copy()
    feed = ReplayFeed(feed_df, warm_bars=WARM_BARS, chunk_bars=1)

    cfg = RunnerConfig(
        journal_path=journal,
        risk_state_path=tmp / "risk.json",
        kill_switch_path=tmp / "KILL",
        forecast_bundle_path=None,
        auto_execute=True,          # paper loop always paper-trades sleeves
        use_confluence=True,
    )
    senders = [TelegramSender(outbox_dir=outbox),
               EmailSender(outbox_dir=outbox)]
    runner = LiveRunner(cfg, senders=senders)

    print(f"rehearsal: warm {WARM_BARS} bars + {REPLAY_BARS} live bars, "
          f"journal {journal}")
    summary = runner.run_paper_loop(feed, max_bars=REPLAY_BARS)
    print()

    records = runner.journal.records()
    opens = [r for r in records if r.get("type") == "open"]
    closes = [r for r in records if r.get("type") == "close"]
    notes = [r for r in records if r.get("type") == "note"]
    paper_notes = [r for r in notes if "paper loop [PAPER]" in r.get("text", "")]
    shadow_notes = [r for r in notes if "paper loop [SHADOW]" in r.get("text", "")]
    outbox_files = list(outbox.iterdir()) if outbox.exists() else []
    perf = summary["performance"]["overall"]

    by_playbook: dict[str, int] = {}
    for r in opens:
        by_playbook[r.get("playbook", "?")] = \
            by_playbook.get(r.get("playbook", "?"), 0) + 1

    print("=== PAPER LOOP REHEARSAL REPORT ===")
    print(f"bars processed : {summary['bars']}")
    print(f"signals        : {summary['signals']} "
          f"(paper {summary['executed']}, shadow {summary['shadow']}, "
          f"gated {summary['gated']})")
    print(f"alerts emitted : {summary['alerts']}")
    print(f"outbox files   : {len(outbox_files)} (alert renders)")
    print(f"journal opens  : {len(opens)} {by_playbook}")
    print(f"journal closes : {len(closes)}")
    print(f"paper balance  : {summary['paper_balance']:.2f} "
          f"(open positions: {summary['paper_open']})")
    print(f"performance    : {perf['trades']} trades, "
          f"{perf['total_r']:+.2f}R total, PF {perf['pf']:.2f}, "
          f"win {perf['win_rate']:.0f}%")
    if paper_notes:
        print("--- sample paper signal ---")
        print(" ", paper_notes[0]["text"][:220])
    if shadow_notes:
        print(f"--- anchor shadow signals journaled: {len(shadow_notes)} ---")

    failures = []
    if summary["bars"] != REPLAY_BARS:
        failures.append(f"only {summary['bars']}/{REPLAY_BARS} bars processed")
    if summary["signals"] == 0:
        failures.append("no signals fired — loop is deaf")
    if summary["alerts"] == 0:
        failures.append("no alerts emitted")
    if not outbox_files:
        failures.append("no alert renders in the outbox")
    if not opens:
        failures.append("no paper fills reached the journal")
    if not closes:
        failures.append("no managed closes reached the journal "
                         "(fill->management->close unproven)")
    if any(r.get("playbook") not in ("fvg_midline", "trendline_retest")
           for r in opens):
        failures.append("a fill was journaled for an unexpected playbook "
                        f"{sorted({r.get('playbook') for r in opens})}")
    if perf["trades"] == 0:
        failures.append("performance book is empty — OOS growth unproven")

    print()
    if failures:
        print("REHEARSAL FAILED:")
        for f in failures:
            print(f"  - {f}")
        print(f"(workdir kept at {tmp})")
        return 1
    print(f"REHEARSAL PASSED — workdir {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
