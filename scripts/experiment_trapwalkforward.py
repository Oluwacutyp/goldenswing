"""Trap-salvage walk-forward: does the trap filter survive OOS?

Same expanding-window grid as scripts/walkforward.py (train 12mo, step
3mo, four OOS windows).  For each playbook and each trap variant, the
variant is applied as a candidate pre-filter (trap state read at the
ENTRY bar, from trap_df computed once on the full history — the state
machines are causal, no lookahead), then the playbook's validated
booking rule runs (anchor: live_threshold=60; sleeves: 0).

Variants under test (the ones that improved the full-sample book):
  - trendline_retest: boost_only
  - sweep_reversal:  boost_only, nobait, nooppose  (rescue check)

Also prints the IS (0-12mo) vs OOS (12-24mo) split for baseline vs
variant — both halves must agree in sign for a claim of stability.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_salvage import (  # noqa: E402
    engine_gated_candidates,
    load_ctx_and_candidates,
)
from experiment_trapsalvage import _dir_flags  # noqa: E402
from goldenswing.backtest.harness import book_metrics, build_book  # noqa: E402
from goldenswing.features.traps import compute_trap_features  # noqa: E402

THRESHOLDS = {"sweep_reversal": 60.0, "fvg_midline": 0.0, "trendline_retest": 0.0}
TESTS = {
    "trendline_retest": ("boost_only",),
    "sweep_reversal": ("boost_only", "nobait", "nooppose"),
    "fvg_midline": (),  # nothing improved the book; nothing to walk forward
}


def trap_filter(cands, variant: str, trap_df: pd.DataFrame):
    pred = {
        "boost_only": lambda f: f["boost"],
        "nobait": lambda f: not f["bait"],
        "nooppose": lambda f: not f["oppose"],
    }[variant]

    def ok(c):
        i = c.entry_index
        if i < 0 or i >= len(trap_df):
            return True
        return bool(pred(_dir_flags(trap_df.iloc[i], c.direction)))

    return [c for c in cands if ok(c)]


def main() -> None:
    ctx, playbooks, _ = load_ctx_and_candidates()
    gated = engine_gated_candidates(ctx, playbooks)
    print("computing trap features ...", flush=True)
    trap_df = compute_trap_features(ctx.df)
    assert len(trap_df) == len(ctx.df)

    df = ctx.df
    t0 = df.index[0]
    months = (df.index - t0).days / 30.44
    total_months = float(months[-1])
    train_months, step_months = 12, 3

    for name, variants in TESTS.items():
        if not variants:
            print(f"\n{name}: no surviving variant — skipped")
            continue
        cands = gated[name]
        thr = THRESHOLDS[name]
        print(f"\n{'=' * 78}\n{name} (threshold {thr})")
        print(f"{'window':>12} {'variant':>12} {'n':>5} {'ret%':>9} "
              f"{'PF':>6} {'win%':>6} {'avgR':>7}")
        is_all, oos_all = [], []
        for v in ("baseline",) + tuple(variants):
            vc = cands if v == "baseline" else trap_filter(cands, v, trap_df)
            start = train_months
            rows = []
            while start < total_months - 1e-9:
                end = min(start + step_months, total_months)
                wc = [c for c in vc
                      if start <= months[c.signal_index] < end]
                book = build_book([c.outcome for c in wc], thr)
                m = book_metrics(book, (end - start) / 12.0)
                rows.append((f"{start:.0f}-{end:.0f}mo", m))
                if v == "baseline":
                    oos_all.extend(book)
                start += step_months
            # IS (0-12) vs OOS (12-24) aggregates for this variant
            is_book = build_book(
                [c.outcome for c in vc if months[c.signal_index] < 12], thr)
            oos_book = build_book(
                [c.outcome for c in vc if months[c.signal_index] >= 12], thr)
            m_is = book_metrics(is_book, 1.0)
            m_oos = book_metrics(oos_book, total_months / 12.0 - 1.0)
            for win, m in rows:
                print(f"{win:>12} {v:>12} {m['trades']:>5} "
                      f"{m['return_pct']:>+9.2f} {m['pf']:>6.2f} "
                      f"{m['win_rate']:>6.1f} {m['avg_r']:>+7.3f}")
            print(f"{'IS 0-12mo':>12} {v:>12} {m_is['trades']:>5} "
                  f"{m_is['return_pct']:>+9.2f} {m_is['pf']:>6.2f} "
                  f"{m_is['win_rate']:>6.1f} {m_is['avg_r']:>+7.3f}")
            print(f"{'OOS 12-24mo':>12} {v:>12} {m_oos['trades']:>5} "
                  f"{m_oos['return_pct']:>+9.2f} {m_oos['pf']:>6.2f} "
                  f"{m_oos['win_rate']:>6.1f} {m_oos['avg_r']:>+7.3f}")


if __name__ == "__main__":
    main()
