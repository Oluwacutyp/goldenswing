"""Phase-3 ablation: Phase-2 baseline vs +feature families.

Method: the reader and the FeatureEngine run ONCE.  Each config then
re-scans the same states with the Phase-2 detector settings
(``require_sweep_confirmation=True``, ``daily_veto=True``, v2 swing
exits, same costs/horizon) plus ONE feature family switched on as
score backing (and finally ALL families).  A family is judged on:

- the traded book @ >=60 (return / PF vs baseline), and
- the shadow book (every signal standalone): avg-R delta vs baseline
  — dilution shows up here first, because a family that inflates weak
  signals' scores poisons the shadow grade even when the sequential
  book happens to dodge them.

Verdicts are mechanical: ADDS / DILUTES / NEUTRAL by the thresholds in
``verdict_for`` — and FEATURES.md reports the numbers either way.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import FAMILIES, FeatureEngine
from ..golden.detector import DetectorConfig, GoldenSwingDetector
from ..reader.reader import MarketReader
from .data import DEFAULT_CACHE, REPO_ROOT, load_gold_h1, load_peers
from .harness import book_metrics, build_book, simulate_signal_v2, slice_stats

P2_FLAGS = dict(require_sweep_confirmation=True, daily_veto=True)


@dataclass
class ConfigResult:
    name: str
    n_signals: int
    book60: dict
    book50: dict
    shadow: dict          # slice_stats over ALL simulated outcomes
    shadow60: dict        # slice_stats over outcomes with score >= 60:
                          # the per-signal quality of what a family
                          # PROMOTES into the traded region (no lock)


def _run_config(df: pd.DataFrame, states, years: float,
                families: frozenset | None, name: str) -> ConfigResult:
    detector = GoldenSwingDetector(DetectorConfig(
        emit_floor=35.0, golden_threshold=60.0,
        feature_families=families, **P2_FLAGS))
    signals = detector.scan(states)
    outcomes = []
    for sig in signals:
        o = simulate_signal_v2(df, sig)
        if o is not None:
            outcomes.append(o)
    return ConfigResult(
        name=name, n_signals=len(outcomes),
        book60=book_metrics(build_book(outcomes, 60.0), years),
        book50=book_metrics(build_book(outcomes, 50.0), years),
        shadow=slice_stats(outcomes),
        shadow60=slice_stats([o for o in outcomes
                              if o.signal.score >= 60.0]),
    )


@dataclass
class AblationResult:
    bars: int
    start: str
    end: str
    years: float
    peers_available: dict = field(default_factory=dict)
    base: ConfigResult | None = None
    families: dict = field(default_factory=dict)   # name -> ConfigResult
    all_features: ConfigResult | None = None


def run_ablation(df: pd.DataFrame, peers: dict | None = None,
                 families: tuple = FAMILIES) -> AblationResult:
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    years = span_days / 365.25

    plain_states = MarketReader().read(df)
    base = _run_config(df, plain_states, years, None, "Phase 2 baseline")

    featured_states = MarketReader().read(df)
    FeatureEngine().attach(df, featured_states, peers)
    fam_results = {}
    for fam in families:
        fam_results[fam] = _run_config(
            df, featured_states, years, frozenset([fam]), fam)
    all_res = _run_config(df, featured_states, years,
                          frozenset(families), "ALL families")

    peers_avail = {}
    for name, pdf in (peers or {}).items():
        peers_avail[name] = bool(pdf is not None and len(pdf) > 100)
    return AblationResult(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        years=years, peers_available=peers_avail, base=base,
        families=fam_results, all_features=all_res)


def verdict_for(base: ConfigResult, res: ConfigResult) -> str:
    """Mechanical verdict from the two deltas (book return in pp,
    shadow-@>=60 avg-R).  FEATURES.md prints the numbers beside it."""
    d_ret = res.book60["return_pct"] - base.book60["return_pct"]
    d_shadow = res.shadow60["avg_r"] - base.shadow60["avg_r"]
    if d_ret > 1.0 and d_shadow >= -0.005:
        return "ADDS"
    if d_ret < -1.0 or d_shadow < -0.010:
        return "DILUTES"
    return "NEUTRAL"


def _row(label: str, base: ConfigResult, res: ConfigResult) -> str:
    b, r = res.book60, res.book50
    return (
        f"| {label} | {res.n_signals} | {res.shadow60['n']} | "
        f"{b['trades']} | {b['pf']:.2f} | "
        f"{b['return_pct']:+.2f}% | {b['max_dd_pct']:.2f}% | "
        f"{b['avg_r']:+.3f} | {res.shadow60['avg_r']:+.3f} | "
        f"{res.shadow60['avg_r'] - base.shadow60['avg_r']:+.3f} | "
        f"{r['return_pct']:+.2f}% | {verdict_for(base, res)} |")


def build_features_report(r: AblationResult) -> str:
    base = r.base
    lines = [
        "# GoldenSwing — Feature Layer (Phase 3)",
        "",
        f"**Data:** GC=F hourly, {r.bars:,} bars, {r.start} → {r.end} "
        f"({r.years:.2f} years). Same Phase-2 entries/exits/costs; the "
        "ONLY change per row is which feature families add score "
        "backing in the detector.",
        "",
        "**RS peers:** " + ", ".join(
            f"{k} {'available' if v else 'UNAVAILABLE (degraded)'}"
            for k, v in sorted(r.peers_available.items()))
        + ". RS rows below are meaningful only for available peers.",
        "",
        f"## Baseline (Phase 2, features off) — book @ ≥60",
        "",
        f"{base.book60['trades']} trades · PF {base.book60['pf']:.2f} · "
        f"return {base.book60['return_pct']:+.2f}% · maxDD "
        f"{base.book60['max_dd_pct']:.2f}% · shadow-@≥60 avg R "
        f"{base.shadow60['avg_r']:+.3f} over {base.shadow60['n']} "
        "promoted signals.",
        "This MUST reproduce BACKTEST2.md (+18.17%, PF 1.16, 199 trades) "
        "— the feature layer is flag-gated off here.",
        "",
        "## Ablation — one family at a time, then all",
        "",
        "\"Promoted\" = signals the config scores at >= 60 (the traded "
        "region); \"Shadow @≥60\" grades those signals standalone, "
        "without the sequential book lock — it measures the QUALITY of "
        "what a family promotes, which is where dilution shows up.",
        "",
        "| Config | Signals | Promoted | Book trades | PF | Return | "
        "MaxDD | Book avg R | Shadow @≥60 avg R | Δ shadow R | "
        "Book@50 return | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        _row("Phase 2 baseline", base, base).replace(
            verdict_for(base, base), "—"),
    ]
    for fam in FAMILIES:
        if fam in r.families:
            lines.append(_row(fam, base, r.families[fam]))
    lines.append(_row("ALL families", base, r.all_features))
    lines += [
        "",
        "Verdict rule: ADDS = book return up > 1pp without poisoning "
        "the promoted-signal grade (Δ shadow @≥60 >= -0.005R); "
        "DILUTES = book down > 1pp OR shadow @≥60 avg R down > 0.010R; "
        "else NEUTRAL.  A DILUTES family is flagged here, not silently "
        "kept: it should stay OFF in the detector config until "
        "re-designed.",
        "",
        "## Caveats",
        "",
        "- One instrument, one timeframe, one window (same in-sample "
        "window the Phase-2 design decisions were made on).  Ablation "
        "deltas are diagnostics, not out-of-sample proof.",
        "- Score deltas per family are hand-set priors (see "
        "goldenswing/golden/detector.py `_feat_*`), not fitted — a "
        "family can fail because its WEIGHT is wrong, not because the "
        "feature carries no information.  The shadow grade helps tell "
        "those apart only partly.",
        "- Feature families are correlated with the core families "
        "(e.g. MTF re-uses trend information); solo deltas do not sum "
        "to the ALL delta.",
        "",
    ]
    return "\n".join(lines)


def main3(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing Phase-3 ablation")
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--cache", type=str, default=str(DEFAULT_CACHE))
    args = ap.parse_args(argv)

    df = load_gold_h1(Path(args.cache), refresh=args.refresh_data)
    peers = load_peers(refresh=args.refresh_data)
    result = run_ablation(df, peers)
    report = build_features_report(result)
    out = REPO_ROOT / "FEATURES.md"
    out.write_text(report)
    print(report)
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main3()
