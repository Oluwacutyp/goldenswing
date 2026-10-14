#!/usr/bin/env python3
"""Bollinger trend states, measured.

Q1 squeeze -> expansion (magnitude vs baseline, breakout drift, HTF-bias
   direction, tradable breakout book) |
Q2 walk-the-band as continuation entry context (anchor shadow split) |
Q3 band-rejection wick drift.  2y GC=F H1, repo conventions, all causal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.experiments.bollinger_trends import run_bollinger


def main() -> None:
    rep = run_bollinger()
    print(f"=== bars={rep.bars} {rep.start} -> {rep.end} "
          f"({rep.years:.2f}y) ===")

    s = rep.squeeze
    print("\n--- Q1 squeeze -> expansion ---")
    print(f"episodes: {s.n_episodes}")
    print(f"24-bar |move|/ATR: {s.mag_mean_atr:.3f} "
          f"(random-bar baseline {s.mag_base_mean_atr:.3f})")
    print(f"breakouts: {s.n_breakouts}  drift24/ATR: "
          f"{s.brk_drift_mean_atr:.3f}  win: {s.brk_win_rate * 100:.1f}%")
    print(f"HTF-bias direction hit: {s.ctx_hit_rate * 100:.1f}% "
          f"(n={s.ctx_n})")
    b = s.book
    print(f"breakout book: trades={b.get('trades')} "
          f"win={b.get('win_rate', 0):.1f}% PF={b.get('pf', 0):.2f} "
          f"ret={b.get('return_pct', 0):+.2f}% "
          f"MaxDD={b.get('max_dd_pct', 0):.2f}% "
          f"Sharpe={b.get('sharpe', 0):.2f} avgR={b.get('avg_r', 0):+.3f}")
    print(f"  H1: n={s.breakout_n_h1} "
          f"ret={s.book_h1.get('return_pct', 0):+.2f}%  "
          f"H2: n={s.breakout_n_h2} "
          f"ret={s.book_h2.get('return_pct', 0):+.2f}%")

    w = rep.walk
    print("\n--- Q2 walk-the-band entry context (anchor shadow) ---")
    print(f"walk==dir: n={w.n_with} avgR={w.avg_r_with:+.3f} "
          f"PF={w.pf_with:.2f} win={w.win_with * 100:.1f}% "
          f"(H1 {w.avg_r_with_h1:+.3f} / H2 {w.avg_r_with_h2:+.3f})")
    print(f"no walk  : n={w.n_without} avgR={w.avg_r_without:+.3f} "
          f"PF={w.pf_without:.2f} win={w.win_without * 100:.1f}%")

    r = rep.rejection
    print("\n--- Q3 band-rejection wick drift (ATR, rejection dir) ---")
    print(f"events: {r.n_events}  d8={r.drift_8:+.3f} "
          f"d12={r.drift_12:+.3f} d24={r.drift_24:+.3f} "
          f"(base |d24|={r.base_24:.3f}) win24={r.win_24 * 100:.1f}%")
    print(f"  H1 d24={r.drift_24_h1:+.3f}  H2 d24={r.drift_24_h2:+.3f}")


if __name__ == "__main__":
    main()
