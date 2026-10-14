#!/usr/bin/env python3
"""Failure-mode diagnosis runner: direction failure vs timing failure.

Reuses each killed strategy's own scanner/simulator, diagnoses every
clean initial-stop loser, and prints the per-strategy splits with
halves plus the verdict (flip-candidate / refine-entry / stays-buried).
Ends with the random-entry control that calibrates the classifier.

    python3 scripts/experiment_failure_modes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.data import load_gold_h1
from goldenswing.experiments.failure_modes import (
    StrategyDiagnosis, run_all, run_random_control, verdict,
)


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def _print_diags(diags: list[StrategyDiagnosis]) -> None:
    for dg in diags:
        sp = dg.split()
        classified = sp["direction"] + sp["timing"] + sp["neither"]
        print(f"\n=== {dg.strategy} ===")
        print(f"signals={dg.n_signals} trades={dg.n_trades} "
              f"losers={dg.n_losers} other_exits={dg.n_other_exits} "
              f"diagnosed={len(dg.records)}")
        print(f"direction={sp['direction']} ({_pct(sp['direction'], classified)})  "
              f"timing={sp['timing']} ({_pct(sp['timing'], classified)})  "
              f"neither={sp['neither']} ({_pct(sp['neither'], classified)})  "
              f"insufficient={sp['insufficient']}")
        halves = dg.split_by_half()
        for h in ("H1", "H2"):
            s = halves[h]
            c = s["direction"] + s["timing"] + s["neither"]
            print(f"  {h}: direction={s['direction']} timing={s['timing']} "
                  f"neither={s['neither']} insufficient={s['insufficient']} "
                  f"(classified={c})")
        print(f"verdict: {verdict(sp)}")
        timing = [r for r in dg.records if r.kind.value == "timing"]
        if timing:
            sess: dict[str, int] = {}
            for r in timing:
                sess[r.session] = sess.get(r.session, 0) + 1
            dr = sorted(r.drift_24 for r in timing
                        if r.drift_24 is not None)
            med = dr[len(dr) // 2] if dr else float("nan")
            mr = sorted(r.max_recovery_24 for r in timing
                        if r.max_recovery_24 is not None)
            med_mr = mr[len(mr) // 2] if mr else float("nan")
            print(f"  timing-failure sessions: {sess}  "
                  f"median net drift_24={med:.2f}R  "
                  f"median max recovery={med_mr:.2f}R")


def main() -> None:
    _print_diags(run_all())
    print("\n\n########## CLASSIFIER CALIBRATION: random-entry control ##########")
    df = load_gold_h1()
    _print_diags(run_random_control(df))


if __name__ == "__main__":
    main()
