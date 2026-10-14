"""Run the imbalance deep-mining measurements and print the report tables."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from goldenswing.backtest.data import load_gold_h1
from goldenswing.experiments import imbalance as I


def main() -> None:
    df = load_gold_h1()
    print(f"bars: {len(df)}  {df.index[0]} -> {df.index[-1]}")
    gaps = I.detect_gaps(df)
    bull = sum(1 for g in gaps if g.direction == 1)
    print(f"gaps: {len(gaps)}  (bullish/BISI {bull}, bearish/SIBI {len(gaps) - bull})")
    lives = I.replay_lives(df, gaps)

    print("\n== 1. MAGNET: revisit (fill) rate ==")
    mag = I.magnet_stats(lives)
    for k, (hit, n, r) in mag.items():
        print(f"  {k:22s} {hit:5d}/{n:<5d}  {r:.3f}")
    print("  -- placebo (random zones, same sizes) --")
    for k, (hit, n, r) in I.placebo_stats(df, gaps).items():
        print(f"  {k:22s} {hit:5d}/{n:<5d}  {r:.3f}")
    kinds = {}
    for lf in lives:
        if lf.end_kind:
            kinds[lf.end_kind] = kinds.get(lf.end_kind, 0) + 1
    print(f"  first-touch outcomes: {kinds}  (survived 240 bars: "
          f"{len(lives) - sum(kinds.values())})")

    print("\n== 2. CE (midline) vs EDGE entry at first wick touch ==")
    atr = I._atr(df["high"].to_numpy(), df["low"].to_numpy(),
                 df["close"].to_numpy(), 14)
    entries = I.ce_vs_edge(df, lives, atr)
    for kind, s in I.summarize(entries).items():
        print(f"  {kind:5s} n={s['n']:4d}  avgR={s['avg_r']:+.3f}  "
              f"win={s['win_rate']:.3f}  bounce1R={s['bounce_1r']:.3f}")

    print("\n== 3. IFVG: drift after body-close inversion ==")
    drift = I.ifvg_drift(df, lives)
    print(f"  inversions: {drift['n_inversions']}")
    for h in (12, 24, 48):
        d = drift[f"h{h}"]
        print(f"  h={h:2d}  n={d['n']:4d}  mean_drift={d['mean_drift_atr']:+.3f} ATR  "
              f"hit={d['hit_rate']:.3f}  base_abs={d['base_abs_drift_atr']:.3f} ATR")

    print("\n== 4. STACKING: hold rate stacked vs single ==")
    for k, b in I.stacking_stats(df, gaps, lives).items():
        print(f"  {k:8s} n={b['n']:4d}  hold_rate={b['hold_rate']:.3f}")


if __name__ == "__main__":
    main()
