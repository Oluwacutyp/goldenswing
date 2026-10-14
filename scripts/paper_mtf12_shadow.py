"""Paper-shadow runner for the multi-TF structural engine ($12).

Replays closed bars through the multitf engine and journals each
completed shadow trade to a SEPARATE journal file. It never touches
the live paper loop's journal (data/trade_journal.jsonl) or its
state (data/paper_state/) unless the caller explicitly passes those
paths.

This is a replay/paper-shadow tool, not a live feed loop. Live
side-by-side paper (OANDA feed + separate state dir + cron) is the
next wiring step; it is NOT faked here.

Usage:
    python3 scripts/paper_mtf12_shadow.py --csv data/OANDA_XAU_USD_H1_20241001_20261012.csv --preset swing --balance 12 --journal data/mtf12_shadow_journal.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from goldenswing.backtest.mtf12 import (  # noqa: E402
    build_tf_frames,
    load_ohlc_csv,
    run_mtf12_backtest,
)
from goldenswing.live.journal_lock import append_journal_record  # noqa: E402
from goldenswing.live.multitf import PAPER_START_BALANCE_12, PRESETS  # noqa: E402


def run_shadow(*, csv_path: str | Path, preset: str = "swing",
               balance: float = PAPER_START_BALANCE_12,
               journal_path: str | Path) -> dict:
    """Replay CSV bars, journal each shadow trade, return summary.

    Journal records are type ``mtf12_shadow_paper``. Each carries the
    fractional-unit R, the OANDA 1-unit feasibility flag, and the
    $12 risk budget, so a reader can never mistake fractional
    measurement for OANDA-executable money.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}, choose from {sorted(PRESETS)}")
    if balance <= 0:
        raise ValueError(f"balance must be positive (got {balance})")
    df = load_ohlc_csv(csv_path)
    from goldenswing.backtest.mtf12 import infer_tf
    frames = build_tf_frames({infer_tf(df): df}, PRESETS[preset])
    result = run_mtf12_backtest(frames, PRESETS[preset], start_balance=balance)
    journal_path = str(journal_path)
    seqs = []
    for t in result["trades"]:
        record = {
            "v": 1,
            "type": "mtf12_shadow_paper",
            "preset": preset,
            "direction": t["direction"],
            "entry": t["entry"],
            "stop": t["stop"],
            "exit": t["exit_price"],
            "exit_reason": t["exit_reason"],
            "r_net": t["r_net"],
            "fractional_units": t["fractional_units"],
            "oanda_1unit_feasible": t["oanda_1unit_feasible"],
            "session": t["session"],
            "regime": t["regime"],
            "start_balance": balance,
            "risk_money": balance * 0.01,
        }
        seqs.append(append_journal_record(journal_path, record))
    summary = {
        "preset": preset,
        "balance": balance,
        "trades": result["summary"]["total_trades"],
        "total_R": result["summary"]["total_R"],
        "dollar_pnl_fractional": result["summary"]["dollar_pnl"],
        "feasible_trades": result["feasible_1unit"]["total_trades"],
        "journal_path": journal_path,
        "journal_seqs": seqs,
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="MTF12 $12 paper-shadow replay")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--preset", default="swing", choices=sorted(PRESETS))
    ap.add_argument("--balance", type=float, default=PAPER_START_BALANCE_12)
    ap.add_argument("--journal", default="data/mtf12_shadow_journal.jsonl")
    args = ap.parse_args(argv)
    summary = run_shadow(csv_path=args.csv, preset=args.preset,
                         balance=args.balance, journal_path=args.journal)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
