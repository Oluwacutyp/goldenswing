"""Run the QQQ opening-range breakout + retest experiment on both grids."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from goldenswing.experiments.opening_range import run_experiment


def _row(name: str, v: dict) -> str:
    return (f"{name:34s} n={v['trades']:3d} "
            f"win={v['win_rate']:5.1f}% pf={v['pf']:5.2f} "
            f"avgR={v['avg_r']:+.3f} ret={v['return_pct']:+7.2f}% "
            f"dd={v['max_dd_pct']:5.2f}% shr={v['sharpe']:+.2f} "
            f"hold={v['avg_hold']:4.1f}b")


def main() -> None:
    for grid in ("5m", "1m"):
        try:
            r = run_experiment(grid=grid)
        except RuntimeError as e:
            print(f"[{grid}] {e}")
            continue
        print(f"\n=== QQQ opening-range, grid={grid} ===")
        print(f"bars={r.bars}  {r.start} -> {r.end}")
        print(f"sessions={r.n_sessions} setups={r.n_setups} "
              f"fills={r.n_fills} cancelled={r.n_cancelled} "
              f"gated={r.n_gated} bias_blocked={r.n_bias_blocked}")
        for name in sorted(r.variants):
            print(_row(name, r.variants[name]))


if __name__ == "__main__":
    main()
