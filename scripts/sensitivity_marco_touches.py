"""Sensitivity: Marco variant B book vs engineered-touch threshold."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from goldenswing.backtest.data import load_gold_h1  # noqa: E402
from goldenswing.backtest.harness import (book_metrics, build_book,  # noqa: E402
                                          slice_stats)
from goldenswing.experiments import marco_sweep as M  # noqa: E402
from goldenswing.experiments.marco_sweep import _years  # noqa: E402
from goldenswing.features.engine import read_and_featurize  # noqa: E402
from goldenswing.golden.detector import (DetectorConfig,  # noqa: E402
                                         GoldenSwingDetector)


def main() -> None:
    df = load_gold_h1()
    states = read_and_featurize(df)
    det = GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True))
    yrs = _years(df)
    n = len(df)
    print(f"bars={n}")
    for mt in (2, 3, 4, 5, 6):
        hits = M.scan_variant(df, states, det, "B", min_touches=mt)
        outcomes = M.simulate_hits(df, hits, "B")
        book = build_book(outcomes, 0.0)
        bm = book_metrics(book, yrs)
        sh = slice_stats(outcomes)
        print(f"min_touches={mt}: hits={len(hits)} "
              f"shadow_n={sh['n']} avgR={sh['avg_r']:+.3f} "
              f"book_trades={bm['trades']} ret={bm['return_pct']:+.2f}% "
              f"pf={bm['pf']:.2f} wr={bm['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
