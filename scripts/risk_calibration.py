#!/usr/bin/env python3
"""Risk calibration: derive guard parameters from the anchor book's
actual loss-cluster statistics (Phase 10).

The old defaults (3 consecutive losses, 10% max drawdown, 2% daily)
were calibrated for a different system: on the anchor book
(199 trades, 45.7% win, GC=F H1 Oct 2024–Oct 2026) the 3-loss halt
would have latched 14 times, the 10% drawdown halt would have
tripped (in-sample MaxDD 10.38%), and the 2% daily halt would have
tripped (worst day -2.2%).  Guards must protect the TAIL, not the
normal — a halt that fires in normal operation is a bug, not safety.

Prints the measured statistics and the recommended RiskConfig values.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.data import load_gold_h1, load_peers
from goldenswing.backtest.harness import book_metrics
from goldenswing.confluence import ConfluenceEngine
from goldenswing.features.engine import read_and_featurize
from goldenswing.playbooks import SweepReversalPlaybook
from goldenswing.playbooks.base import EngineContext


def main() -> None:
    df = load_gold_h1()
    states = read_and_featurize(df, load_peers())
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    ctx = EngineContext(df=df, states=states, years=years,
                        instrument="GC=F")
    pb = SweepReversalPlaybook()
    cands = [c for c in pb.candidates(ctx)
             if c.base_score >= pb.live_threshold]
    book = ConfluenceEngine._greedy([c.outcome for c in cands],
                                   pb.live_threshold)
    print(f"anchor book: {len(book)} trades")

    runs, cur = [], 0
    for o in book:
        if o.r_net < 0:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    hist = collections.Counter(runs)
    print("consecutive-loss-run histogram:", dict(sorted(hist.items())))
    for k in (3, 4, 5, 6, 7):
        n = sum(1 for r in runs if r >= k)
        print(f"  runs >= {k}: {n} in 2y")

    cum, peak, max_dd_r = 0.0, 0.0, 0.0
    for o in book:
        cum += o.r_net
        peak = max(peak, cum)
        max_dd_r = max(max_dd_r, peak - cum)
    print(f"max R-drawdown: {max_dd_r:.1f}R (total {cum:+.1f}R)")

    day_pnl, week_pnl = collections.defaultdict(float), \
        collections.defaultdict(float)
    for o in book:
        ts = df.index[o.exit_index]
        pnl = o.r_net * 100.0  # 1% risk on $10k = $100/trade
        day_pnl[ts.date().isoformat()] += pnl
        iso = ts.isocalendar()
        week_pnl[f"{iso.year}-W{iso.week:02d}"] += pnl
    print(f"worst day: ${min(day_pnl.values()):.0f} "
          f"({min(day_pnl.values())/100:.1%} of $10k)")
    print(f"worst week: ${min(week_pnl.values()):.0f} "
          f"({min(week_pnl.values())/100:.1%} of $10k)")
    m = book_metrics(book, years)
    print(f"in-sample MaxDD: {m['max_dd_pct']:.2f}%")

    print("\nRecommended RiskConfig (tail guards, not normal-operation "
          "trips):")
    print("  max_consecutive_losses = 6   "
          "(3-runs hit 14x/2y; 6+ hit once — genuine tail)")
    print("  max_drawdown_pct       = 0.15  "
          "(1.5x in-sample max 10.38%; live regime breaking the "
          "historical envelope)")
    print("  daily_loss_pct         = 0.03  "
          "(worst in-sample day -2.2% at 1% risk)")
    print("  weekly_loss_pct        = 0.05  (unchanged; worst -3.1%)")


if __name__ == "__main__":
    main()
