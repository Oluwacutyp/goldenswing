#!/usr/bin/env python3
"""Salvage experiment harness (Workstream B) — reusable filter experiments.

One harness for every candidate-filter experiment (cost gate, trap
intelligence, ...):

- :func:`load_ctx_and_candidates` builds the EngineContext EXACTLY the
  way ``scripts/run_confluence_book.py`` does (same data, same peers,
  same featurization) and scans the three Phase-9 playbooks.
- :func:`engine_gated_candidates` applies the confluence engine's own
  context gates (session scope, setup tripwire, hunting ground) so an
  experiment starts from the playbook's EXISTING book construction.
- :func:`run_filter_experiment` takes ``(playbook_name, ctx,
  candidates, keep_fn, trap_df=None)`` and returns baseline vs
  filtered books plus the profile of the cut trades.

``keep_fn(cand, trap_row_or_None) -> bool`` is the filter under test.
``trap_df`` is an optional per-bar boolean DataFrame aligned to
``ctx.df`` BY POSITION (row i describes bar i).  Columns contract
(second workstream builds the real producer — code against these
names)::

    trap_boost_long, trap_boost_short,   # genuine trap: fade is safe
    bait_long, bait_short,               # bait: do NOT take the signal
    lure_long, lure_short                # suspicious break: blocked unless
                                         # a confirmed trap overrides

The harness only passes the row; it never interprets the columns.
``trap_row_or_None`` is a pandas Series (the ``trap_df`` row at the
candidate's entry bar) or None when no trap data was supplied.

Phase-routing lesson (hard-won): removing trades re-times everything
downstream, so the filtered book is RE-BUILT greedily from the kept
candidates — never derived by subtracting cut trades from the
baseline book.

Importable: ``from experiment_salvage import load_ctx_and_candidates,
run_filter_experiment`` (scripts dir on sys.path).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from goldenswing.backtest.data import load_gold_h1, load_peers
from goldenswing.backtest.harness import book_metrics, build_book
from goldenswing.confluence import ConfluenceEngine
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.risk import RiskGuards
from goldenswing.features.engine import read_and_featurize
from goldenswing.playbooks import (FVGMidlinePlaybook, SweepReversalPlaybook,
                                   TrendlineRetestPlaybook)
from goldenswing.playbooks.base import Candidate, EngineContext, Playbook

# Reuse the single source of truth for the machinery canary.
from run_confluence_book import MACHINERY_COUNTS

#: Trap-feature column contract.  The second workstream produces this
#: frame; the harness validates presence (fail fast, loud) but never
#: interprets the values.
TRAP_COLUMNS = ("trap_boost_long", "trap_boost_short",
                "bait_long", "bait_short",
                "lure_long", "lure_short")

PLAYBOOKS: dict[str, type[Playbook]] = {
    "sweep_reversal": SweepReversalPlaybook,
    "fvg_midline": FVGMidlinePlaybook,
    "trendline_retest": TrendlineRetestPlaybook,
}


# ------------------------------------------------------------ loading
def load_ctx_and_candidates(
) -> tuple[EngineContext, dict[str, Playbook], dict[str, list[Candidate]]]:
    """Build EngineContext exactly as ``scripts/run_confluence_book.py``
    does, then scan the three Phase-9 playbooks.

    Returns ``(ctx, playbooks, candidates)`` — ``candidates[name]`` is
    the RAW scanned list, before any context gate.  Raises
    ``AssertionError`` on machinery drift (the run_confluence_book
    canary): numbers measured on drifted machinery are meaningless.
    """
    df = load_gold_h1()
    peers = load_peers()  # the canary lesson: relstr needs its peers
    states = read_and_featurize(df, peers)
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    ctx = EngineContext(df=df, states=states, years=years)

    playbooks = {name: cls() for name, cls in PLAYBOOKS.items()}
    candidates = {name: pb.candidates(ctx) for name, pb in playbooks.items()}
    for name, cands in candidates.items():
        want = MACHINERY_COUNTS[name]
        assert len(cands) == want, \
            f"MACHINERY DRIFT {name}: {len(cands)} != {want}"
    return ctx, playbooks, candidates


def engine_gated_candidates(
        ctx: EngineContext, playbooks: dict[str, Playbook] | None = None
) -> dict[str, list[Candidate]]:
    """Apply the confluence engine's own context gates to each playbook's
    candidates — the same three gates ``ConfluenceEngine.run`` applies
    (session scope, setup-context tripwire, measured hunting ground).

    Returns ``{name: [Candidate]}`` — the exact input the playbook's
    existing book is built from.  Risk runs in SHADOW mode against a
    throwaway journal/state so production state is untouched.
    """
    playbooks = playbooks or {n: c() for n, c in PLAYBOOKS.items()}
    journal = TradeJournal(
        str(Path(tempfile.mkdtemp()) / "salvage_journal.jsonl"))
    risk = RiskGuards(Path(tempfile.mkdtemp()) / "risk_state.json")
    engine = ConfluenceEngine(list(playbooks.values()), risk, journal,
                              risk_mode="shadow")
    rep = engine.run(ctx)
    return {pb.name: [sc.candidate for sc in rep.scored[pb.name]]
            for pb in playbooks.values()}


# ------------------------------------------------------------ experiment
def _check_trap_df(trap_df: pd.DataFrame | None, n_bars: int) -> None:
    if trap_df is None:
        return
    if len(trap_df) != n_bars:
        raise ValueError(
            f"trap_df has {len(trap_df)} rows but ctx.df has {n_bars} bars "
            "— trap_df must be aligned to ctx.df by POSITION (row i = bar i)")
    missing = [c for c in TRAP_COLUMNS if c not in trap_df.columns]
    if missing:
        raise ValueError(
            f"trap_df missing contract columns {missing} — required: "
            f"{list(TRAP_COLUMNS)}")


def _trap_row_for(trap_df: pd.DataFrame | None,
                  entry_index: int) -> pd.Series | None:
    if trap_df is None:
        return None
    if entry_index < 0 or entry_index >= len(trap_df):
        return None
    return trap_df.iloc[entry_index]


def cut_profile(cut: list[Candidate]) -> dict:
    """What did the filter remove?  r_net distribution of the cut
    candidates' pre-simulated standalone outcomes — did the gate cut
    losers or winners?

    Caveat: these are standalone simulations, not the re-timed book.
    Descriptive of the cut set, not of the book delta.
    """
    if not cut:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "pf": 0.0,
                "total_r": 0.0, "buckets": {}, "exit_reasons": {}}
    rs = np.array([c.outcome.r_net for c in cut], dtype=float)
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    pf = float(wins.sum() / abs(losses.sum())) \
        if len(losses) and losses.sum() != 0 else float("inf")
    buckets = {
        "<=-1R": int(np.sum(rs <= -1.0)),
        "-1R..0R": int(np.sum((rs > -1.0) & (rs <= 0.0))),
        "0R..+1R": int(np.sum((rs > 0.0) & (rs <= 1.0))),
        ">+1R": int(np.sum(rs > 1.0)),
    }
    reasons: dict[str, int] = {}
    for c in cut:
        r = c.outcome.exit_reason
        reasons[r] = reasons.get(r, 0) + 1
    return {
        "n": int(len(rs)),
        "win_rate": float(np.mean(rs > 0) * 100.0),
        "avg_r": float(rs.mean()),
        "pf": pf,
        "total_r": float(rs.sum()),
        "buckets": buckets,
        "exit_reasons": reasons,
    }


def run_filter_experiment(playbook_name: str, ctx: EngineContext,
                          candidates: list[Candidate], keep_fn,
                          trap_df: pd.DataFrame | None = None) -> dict:
    """Run one filter experiment on one playbook.

    ``keep_fn(cand, trap_row_or_None) -> bool`` is the filter under
    test; ``trap_row_or_None`` is the ``trap_df`` row at the
    candidate's entry bar (positional), or None when ``trap_df`` is
    not supplied.  ``trap_df`` is validated against the column
    contract and the bar alignment.

    Both books are built with ``harness.build_book`` (greedy
    chronological, threshold 0.0 on the GIVEN candidates — callers
    pre-apply any score threshold, e.g. the anchor's live_threshold)
    and graded with ``harness.book_metrics``.  The filtered book is
    re-built from the kept candidates, never subtracted.

    Returns a dict with ``baseline`` / ``filtered`` metric dicts,
    ``cut_profile`` (r_net distribution of removed trades), and the
    ``kept`` / ``cut`` candidate lists.
    """
    _check_trap_df(trap_df, len(ctx.df))
    kept: list[Candidate] = []
    cut: list[Candidate] = []
    for cand in candidates:
        row = _trap_row_for(trap_df, cand.entry_index)
        if keep_fn(cand, row):
            kept.append(cand)
        else:
            cut.append(cand)
    baseline_book = build_book([c.outcome for c in candidates], 0.0)
    filtered_book = build_book([c.outcome for c in kept], 0.0)
    return {
        "playbook": playbook_name,
        "n_candidates": len(candidates),
        "n_kept": len(kept),
        "n_cut": len(cut),
        "baseline": book_metrics(baseline_book, ctx.years),
        "filtered": book_metrics(filtered_book, ctx.years),
        "cut_profile": cut_profile(cut),
        "kept": kept,
        "cut": cut,
    }


# ------------------------------------------------------------ demo main
def costgate_keep_fn(ctx: EngineContext, cfg=None):
    """Cost-gate ``keep_fn`` for :func:`run_filter_experiment`: entry at
    the harness fill price (open of the entry bar), gate from
    ``goldenswing.confluence.cost_gate_ok``.  Pine defaults, untuned.
    Shared by the demo :func:`main` and ``scripts/experiment_costgate.py``
    so the two paths agree by construction."""
    from goldenswing.confluence import ConfluenceConfig, cost_gate_ok
    cfg = cfg or ConfluenceConfig()
    opens = ctx.df["open"].to_numpy(dtype=float)

    def keep(cand: Candidate, trap_row) -> bool:
        i = cand.entry_index
        if i < 0 or i >= len(opens):
            return False  # fail closed: no entry bar, no trade
        ok, _reason = cost_gate_ok(cand.outcome.signal, float(opens[i]),
                                   cfg, playbook=cand.playbook)
        return ok

    return keep


def main(argv: list[str] | None = None) -> dict:
    """Demonstrate the cost gate THROUGH the harness: baseline vs
    cost-gated book per playbook, printed.  ``scripts/
    experiment_costgate.py`` is the full measurement (writes
    experiments/COSTGATE.md); this demo must print the same numbers —
    the two paths agree by construction (same keep_fn, same harness).
    """
    from goldenswing.confluence import ConfluenceConfig
    ctx, playbooks, _raw = load_ctx_and_candidates()
    gated = engine_gated_candidates(ctx, playbooks)
    cfg = ConfluenceConfig()  # pine defaults, untuned
    keep = costgate_keep_fn(ctx, cfg)
    results = {}
    for name, pb in playbooks.items():
        eligible = [c for c in gated[name]
                    if c.outcome.signal.score >= pb.live_threshold]
        res = run_filter_experiment(name, ctx, eligible, keep)
        results[name] = res
        b, f = res["baseline"], res["filtered"]
        cp = res["cut_profile"]
        print(f"== {name}: {res['n_candidates']} eligible, "
              f"kept {res['n_kept']}, cut {res['n_cut']} ==")
        print(f"  baseline: n={b['trades']} R={b['avg_r'] * b['trades']:+.1f} "
              f"PF={b['pf']:.2f} win={b['win_rate']:.1f}% "
              f"avgR={b['avg_r']:+.3f} MaxDD={b['max_dd_pct']:.2f}%")
        print(f"  gated:    n={f['trades']} R={f['avg_r'] * f['trades']:+.1f} "
              f"PF={f['pf']:.2f} win={f['win_rate']:.1f}% "
              f"avgR={f['avg_r']:+.3f} MaxDD={f['max_dd_pct']:.2f}%")
        print(f"  cut: n={cp['n']} avgR={cp['avg_r']:+.3f} "
              f"win={cp['win_rate']:.1f}% totalR={cp['total_r']:+.1f} "
              f"buckets={cp['buckets']}")
    return results


if __name__ == "__main__":
    main()
