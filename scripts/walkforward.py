#!/usr/bin/env python3
"""Walk-forward / out-of-sample harness (Phase 10).

Expanding-window walk-forward over the 2y data, for any playbook:

    train [0, 12mo] -> test [12mo, 15mo] -> roll forward by 3mo ...

On each in-sample window the score threshold is fit (grid over
{40,50,60,70,80}, maximizing IS Sharpe with >= 15 trades); the fitted
threshold is applied to the next out-of-sample window.  The playbook's
VALIDATED threshold is also run as a reference column, so the fit
can't take credit for what the fixed config already does.

Session-scoped playbooks are judged on their session only (test
candidates filtered to the scope).  The playbook's hunting ground
applies — the walk-forward answers "does the validated system hold
up out of sample", not "would we have discovered it".

Caveat (stated, not hidden): 2y of H1 gives four 3-month OOS
windows — thin for the sleeves (61-97 trades total).  This is a
smoke test for OOS stability, not a promotion certificate.  Real
promotion needs the 2022–2024 window or a second instrument
(CONFLUENCE.md checklist).

Usage:
    python scripts/walkforward.py --playbook sweep_reversal
    python scripts/walkforward.py --playbook fvg_midline
    python scripts/walkforward.py --playbook trendline_retest
    python scripts/walkforward.py --playbook all
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.data import REPO_ROOT, load_gold_h1, load_peers
from goldenswing.backtest.harness import book_metrics
from goldenswing.confluence import ConfluenceEngine
from goldenswing.contextmap import context_levels
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.risk import RiskGuards
from goldenswing.features.engine import read_and_featurize
from goldenswing.playbooks import (FVGMidlinePlaybook, SweepReversalPlaybook,
                                   TrendlineRetestPlaybook)
from goldenswing.playbooks.base import EngineContext, Playbook

PLAYBOOKS: dict[str, type[Playbook]] = {
    "sweep_reversal": SweepReversalPlaybook,
    "fvg_midline": FVGMidlinePlaybook,
    "trendline_retest": TrendlineRetestPlaybook,
}

THRESHOLD_GRID = (40.0, 50.0, 60.0, 70.0, 80.0)
MIN_IS_TRADES = 15


def _window_book(cands: list, threshold: float, years: float) -> dict:
    # _greedy filters by signal.score >= threshold itself; no pre-filter.
    outs = [c.outcome for c in cands]
    book = ConfluenceEngine._greedy(outs, threshold)
    m = book_metrics(book, years)
    m["trades_list"] = book
    return m


def walkforward(pb: Playbook, ctx: EngineContext,
                train_months: int = 12, step_months: int = 3,
                ) -> list[dict]:
    """Expanding-window walk-forward.  Returns per-window dicts."""
    states, df = ctx.states, ctx.df
    cands = pb.candidates(ctx)
    # Apply the SAME three gates the engine applies (in the same order):
    # session scope -> required-context tripwire -> hunting ground.
    # These are firing rules, not fit knobs.
    hg = pb.hunting_ground
    scoped = []
    for c in cands:
        if pb.session_scope != "any":
            t = c.signal_index
            if t >= len(states) or states[t].session != pb.session_scope:
                continue
        ok, _ = pb.check_context(c, ctx)
        if not ok:
            continue
        if hg.excluded:
            lv = context_levels(states, c.signal_index, c.direction)
            if any(lv.get(d) == l for d, l in hg.excluded):
                continue
        scoped.append(c)
    # time grid in months from the first bar
    t0 = df.index[0]
    months = (df.index - t0).days / 30.44
    total_months = float(months[-1])
    windows = []
    start = train_months
    rows = []
    while start < total_months - 1e-9:
        end = min(start + step_months, total_months)
        is_c = [c for c in scoped if months[c.signal_index] < start]
        oos_c = [c for c in scoped
                 if start <= months[c.signal_index] < end]
        years_oos = (end - start) / 12.0
        # IS fit: threshold maximizing IS Sharpe (>= MIN_IS_TRADES)
        best_t, best_sharpe = pb.live_threshold, None
        if pb.live_threshold > 0:  # threshold playbooks only
            for th in THRESHOLD_GRID:
                m = _window_book(is_c, th, start / 12.0)
                if m["trades"] >= MIN_IS_TRADES:
                    sh = m["sharpe"] if np.isfinite(m["sharpe"]) else -9e9
                    if best_sharpe is None or sh > best_sharpe:
                        best_sharpe, best_t = sh, th
        fit_m = _window_book(oos_c, best_t, years_oos)
        fix_m = _window_book(oos_c, pb.live_threshold, years_oos)
        rows.append({
            "is_end_mo": start, "oos": f"{start:.0f}-{end:.0f}mo",
            "is_trades": len(is_c), "fit_th": best_t,
            "oos_trades_fit": fit_m["trades"],
            "oos_ret_fit": fit_m["return_pct"],
            "oos_pf_fit": fit_m["pf"],
            "oos_trades_fix": fix_m["trades"],
            "oos_ret_fix": fix_m["return_pct"],
            "oos_pf_fix": fix_m["pf"],
            "oos_book_fit": fit_m["trades_list"],
        })
        start += step_months
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="walk-forward OOS harness")
    ap.add_argument("--playbook", type=str, default="all",
                    choices=list(PLAYBOOKS) + ["all"])
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--step-months", type=int, default=3)
    args = ap.parse_args(argv)

    df = load_gold_h1()
    peers = load_peers()
    states = read_and_featurize(df, peers)
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    ctx = EngineContext(df=df, states=states, years=years,
                        instrument="GC=F")

    names = list(PLAYBOOKS) if args.playbook == "all" else [args.playbook]
    for name in names:
        pb = PLAYBOOKS[name]()
        rows = walkforward(pb, ctx, args.train_months, args.step_months)
        print(f"\n{'=' * 78}\n{name} "
              f"(scope={pb.session_scope}, hunting-ground applied)")
        print(f"{'OOS window':>12} {'IS n':>6} {'fit_th':>7} | "
              f"{'FIT: n':>7} {'ret%':>8} {'PF':>6} | "
              f"{'FIX: n':>7} {'ret%':>8} {'PF':>6}")
        agg = []
        for r in rows:
            agg.extend(r["oos_book_fit"])
            print(f"{r['oos']:>12} {r['is_trades']:>6} {r['fit_th']:>7.0f} | "
                  f"{r['oos_trades_fit']:>7} {r['oos_ret_fit']:>+8.2f} "
                  f"{r['oos_pf_fit']:>6.2f} | "
                  f"{r['oos_trades_fix']:>7} {r['oos_ret_fix']:>+8.2f} "
                  f"{r['oos_pf_fix']:>6.2f}")
        if agg:
            m = book_metrics(sorted(agg, key=lambda o: o.signal.index),
                             args.step_months * len(rows) / 12.0)
            print(f"aggregate OOS (IS-fit thresholds): {m['trades']} trades | "
                  f"win {m['win_rate']:.1f}% | PF {m['pf']:.2f} | "
                  f"return {m['return_pct']:+.2f}% | Sharpe {m['sharpe']:.2f}")
        else:
            print("no OOS trades")


if __name__ == "__main__":
    main()
