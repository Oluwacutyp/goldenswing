"""Validate the POI candle-shape confirmation filter (rejection /
engulfing at the POI) against the existing hierarchy backtest.

WITH vs WITHOUT on the SAME trade book: the harness takes every
trade as before (WITHOUT) and annotates each with
``candle_confirmed`` / ``candle_shape`` from the entry-TF bars
composing the signal H1 bar — the bars live would have seen before
the next-open entry, so the comparison is causally clean.  WITH =
the confirmed-only subset.  No params are tuned; the numbers are
reported as they are, even if the filter hurts.

Window is limited by the M5 file (2026-04 -> 2026-10): the harness
needs the full stack incl. M5, and M5 history starts 2026-04-01.

Usage:  python3 scripts/validate_candle_confirm.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from goldenswing.backtest.mtf_hierarchy import (
    PRODUCTION_STAMPS, build_hierarchy_frames, run_hierarchy_backtest)

ROOT = Path(__file__).resolve().parents[1]
H1 = ROOT / "data" / "OANDA_XAU_USD_H1_20241001_20261012.csv"
M15 = ROOT / "data" / "OANDA_XAU_USD_M15_20241001_20261012.csv"
M5 = ROOT / "data" / "OANDA_XAU_USD_M5_20260401_20261012.csv"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    df = df.set_index("ts").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close", "volume"]]


def _row(label: str, s: dict) -> str:
    pf = "inf" if s["profit_factor"] == float("inf") \
        else f"{s['profit_factor']:.2f}"
    return (f"  {label}: trades={s['total_trades']} "
            f"win={s['win_rate']:.1f}% total_R={s['total_R']:+.1f} "
            f"PF={pf} avg_R={s['avg_R']:+.2f} "
            f"maxDD={s['max_drawdown_R']:.1f}R")


def main() -> None:
    h1 = load(H1)
    m15 = load(M15)
    m5 = load(M5)
    frames = build_hierarchy_frames(h1, m15, m5)
    h1f = frames["H1"]
    print(f"window: {h1f.index[0].date()} -> {h1f.index[-1].date()} "
          f"({len(h1f)} H1 bars, full-stack intersection)")
    result = run_hierarchy_backtest(frames, stamps=PRODUCTION_STAMPS)
    trades = result["trades"]
    print(f"  gate_counts: {result['gate_counts']}")
    print(_row("WITHOUT (all trades)", result["summary"]))
    print(_row("WITH (candle-confirmed only)",
               result["candle_confirmed"]))
    n = len(trades)
    n_conf = result["candle_confirmed_trades"]
    print(f"  confirm rate: {n_conf}/{n} "
          f"({100.0 * n_conf / n:.0f}% of trades)" if n else
          "  confirm rate: no trades")
    shapes = Counter(t["candle_shape"] for t in trades)
    print(f"  shapes (first): {dict(shapes)}")
    fam = Counter(s for t in trades
                  for s in t.get("candle_shapes", []))
    print(f"  shapes (family, all confirmed): {dict(fam)}")
    if n < 30:
        print(f"  INCONCLUSIVE: {n} trades is below the 30-trade "
              f"minimum for a WITH-vs-WITHOUT verdict — the filter "
              f"is built and annotated, but NOT validated.")
    for direction, label in ((1, "WITH long"), (-1, "WITH short")):
        sub = [t for t in trades
               if t["candle_confirmed"] and t["direction"] == direction]
        if not sub:
            print(f"  {label}: no trades")
            continue
        rs = [t["r_net"] for t in sub]
        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r <= 0]
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) \
            else (float("inf") if wins else 0.0)
        print(f"  {label}: trades={len(sub)} "
              f"win={100.0 * len(wins) / len(sub):.1f}% "
              f"total_R={sum(rs):+.1f} PF={pf:.2f} "
              f"avg_R={sum(rs) / len(rs):+.2f}")


if __name__ == "__main__":
    main()
