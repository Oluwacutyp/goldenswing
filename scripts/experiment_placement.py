#!/usr/bin/env python3
"""Placement review runner: the same market objects in practitioner roles,
plus the INVERSE check (opposite direction, same trigger, mirrored stop).

T1 failed-pattern continuation (Turtle Soup) | T2 pin-wick violation |
T3 failed zone-retest (ICT breaker).  2y GC=F H1, repo conventions.
Sweeps fail_bars {8,12,16}: a placement (or inverse) is only real if it
survives both window halves AND the parameter variants.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.experiments.placement import (
    run_placement, wedge_breakout_drift)
from goldenswing.backtest.data import load_gold_h1
from goldenswing.features.engine import read_and_featurize


def main() -> None:
    for fb in (8, 12, 16):
        rep = run_placement(fail_bars=fb)
        print(f"=== fail_bars={fb} "
              f"(bars={rep.bars} {rep.start} -> {rep.end}) ===")
        print(f"{'probe':<20}{'n':>6}{'win%':>7}{'avgR':>8}{'PF':>7}"
              f"{'H1':>8}{'H2':>8}")
        for p in rep.probes:
            pf = f"{p.pf:.2f}" if p.pf != float("inf") else "inf"
            print(f"{p.name:<20}{p.n_trades:>6}{p.win_rate * 100:>6.1f}%"
                  f"{p.avg_r:>8.3f}{pf:>7}{p.h1_avg_r:>8.3f}"
                  f"{p.h2_avg_r:>8.3f}")
        print()
    # Wedge breakout-direction drift (#30 placement): diagnostic, not a book.
    df = load_gold_h1()
    states = read_and_featurize(df)
    w = wedge_breakout_drift(df, states)
    print("=== wedge breakout-direction drift (placement for #30) ===")
    print(f"wedges detected: {w['n_wedges']}")
    for key in ("all", "in_trend", "counter"):
        d = w[key]
        print(f"  {key:<9} n={d['n']:>3} mean_drift={d['mean_drift']:+.3f} ATR "
              f"win={d['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
