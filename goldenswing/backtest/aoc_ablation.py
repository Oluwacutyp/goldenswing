"""Phase-4.5 ablation: the AOC family (area of concentration).

Two questions, one harness (identical to Phases 3/3.5/4 — same
states, same Phase-2 flags, v2 exits, $0.45/oz, 1% risk):

1. GATE REPAIR — the Phase-4 patterns gate still diluted (+19.04%)
   because its "major level" test (pool proximity) was too
   permissive on H1.  The gate now requires an AOC pivot zone
   (>= 3 independent families).  Does ``A + patterns_ctx`` stop
   diluting?
2. NEW FAMILY — ``A + aoc``: the aoc_gate scores +4 when the entry
   sits in a direction-aligned concentration and -4 when a
   concentration sits between entry and the first pool target.
   Measured with the same mechanical verdict rule as Phases 3/4.

Anchors must reproduce: Phase-2 baseline +18.17% / PF 1.16 and
Combo A {relstr, volume, phases} +25.62% / PF 1.22 (199 trades).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..features import FeatureEngine
from ..golden.detector import DetectorConfig, GoldenSwingDetector
from ..reader.reader import MarketReader
from .ablation import P2_FLAGS
from .data import DEFAULT_CACHE, REPO_ROOT, load_gold_h1, load_peers
from .validation import COMBO_A, run_config, ValResult

CONFIGS: list[tuple[str, frozenset | None]] = [
    ("phase2", None),
    ("combo_a", COMBO_A),
    ("aoc", COMBO_A | {"aoc"}),
    ("patterns_ctx", COMBO_A | {"patterns_ctx"}),
    ("patterns_ctx+aoc", COMBO_A | {"patterns_ctx", "aoc"}),
]


def _gate_hits(df: pd.DataFrame, states, families,
               needle: str) -> tuple[int, int]:
    """How many emitted signals carried a factor containing
    ``needle`` (diagnostic firing count for the gate)."""
    detector = GoldenSwingDetector(DetectorConfig(
        emit_floor=35.0, golden_threshold=60.0,
        feature_families=families, **P2_FLAGS))
    signals = detector.scan(states)
    hits = sum(1 for s in signals
               if any(needle in f for f in s.factors))
    return hits, len(signals)


def run_aoc(df: pd.DataFrame, peers: dict | None) -> dict:
    plain_states = MarketReader().read(df)
    featured_states = MarketReader().read(df)
    FeatureEngine().attach(df, featured_states, peers)

    results: dict[str, ValResult] = {}
    for key, fams in CONFIGS:
        states = plain_states if fams is None else featured_states
        results[key] = run_config(df, states, fams, key)

    # diagnostic firing counts over the emitted signals
    diag = {}
    diag["aoc_gate_hits"], diag["n_signals"] = _gate_hits(
        df, featured_states, COMBO_A | {"aoc"}, "concentration")
    diag["patterns_gate_hits"], _ = _gate_hits(
        df, featured_states, COMBO_A | {"patterns_ctx"}, "pivot zone")
    return {"results": results, "diag": diag}


def _row(label: str, base: ValResult, res: ValResult,
         verdict: str) -> str:
    b = res.book60
    return (
        f"| {label} | {res.n_signals} | {res.shadow60['n']} | "
        f"{b['trades']} | {b['pf']:.2f} | {b['return_pct']:+.2f}% | "
        f"{b['max_dd_pct']:.2f}% | {b['sharpe']:.2f} | "
        f"{res.shadow60['avg_r']:+.3f} | "
        f"{res.shadow60['avg_r'] - base.shadow60['avg_r']:+.3f} | "
        f"{res.book50['return_pct']:+.2f}% | {verdict} |")


def build_aoc_report(bundle: dict, df: pd.DataFrame) -> str:
    from .ablation import verdict_for
    results = bundle["results"]
    diag = bundle["diag"]
    base = results["combo_a"]
    anchor = results["phase2"]
    lines = [
        "# GoldenSwing — AOC: Area of Concentration (Phase 4.5)",
        "",
        "**Mined from a trading comment section (2026-10-11):** "
        "*\"AOC — AREA OF CONCENTRATION, or your pivot zone: support "
        "& resistance levels.\"* Where several independent level "
        "families agree within a tight band, price treats the area "
        "as a pivot.  The system had all the levels; it never "
        "measured their concentration.",
        "",
        "## Concentration score",
        "",
        "Per bar, the family harvests every live level from confirmed "
        "state — liquidity pools, supply/demand zones, order blocks, "
        "FVGs, trendline values at the bar, Fib levels, daily swings, "
        "the dealing-range EQ — clusters levels within 0.35 ATR "
        "(cluster span capped at 1.0 ATR), and scores each cluster as "
        "**sum of weight x recency** over members:",
        "",
        "- pool 0.8 + 0.2 x strength · sd_zone 1.0 fresh / 0.6 tested · "
        "ob 1.0 · fvg 0.7 · trendline 1.0 ACTIVE / 0.6 FORMING · "
        "fib 0.7 · daily swing 1.2 · EQ 0.5",
        "- recency = 0.5 + 0.5 x 2^(-age / 96): fresh counts double, "
        "ancient levels never drop below half weight",
        "- an AOC needs **2 distinct families**; a **MAJOR** pivot "
        "zone needs **3** (families dedupe inside a cluster)",
        "",
        "Causality: clusters are pure functions of the bar's confirmed "
        "MarketState + FeatureSet.  One subtlety found while building: "
        "the reader mutates level objects in place while per-bar lists "
        "are snapshots — list *membership* is the causal truth, so the "
        "harvester trusts the lists and never re-checks the mutable "
        "flags (re-checking read future mutations and failed the "
        "truncation test; documented in `features/aoc.py`).",
        "",
        "## The two tests",
        "",
        "1. **Gate repair** — Phase-4's patterns gate still diluted "
        "(+19.04%) because its \"major level\" test (pool proximity) "
        "was too permissive on H1.  The gate now requires the "
        "pattern's neckline/extreme to sit inside an AOC pivot zone "
        "(>= 3 families).",
        "2. **New family** — `aoc_gate`: +4 when the entry sits inside "
        "a direction-aligned concentration (>= 2 families), -4 when "
        "a concentration sits strictly between the entry and the "
        "first pool target.  Silent otherwise.",
        "",
        f"**Data:** GC=F hourly, {len(df):,} bars, {df.index[0]} → "
        f"{df.index[-1]}. Same Phase-2 entries/exits/costs; the ONLY "
        "change per row is the detector's feature-family set.",
        "",
        "**Anchors (must reproduce):** Phase-2 baseline "
        f"{anchor.book60['return_pct']:+.2f}% / PF "
        f"{anchor.book60['pf']:.2f} / {anchor.book60['trades']} trades "
        "(BACKTEST2: +18.17% / 1.16 / 199) · Combo A "
        f"{base.book60['return_pct']:+.2f}% / PF "
        f"{base.book60['pf']:.2f} / {base.book60['trades']} trades "
        "(VALIDATION: +25.62% / 1.22 / 199).",
        "",
        "## Ablation — Combo A is the baseline to beat",
        "",
        "| Config | Signals | Promoted | Book trades | PF | Return | "
        "MaxDD | Sharpe | Shadow @≥60 avg R | Δ shadow R | "
        "Book@50 return | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        _row("Phase 2 (anchor)", base, anchor, "—"),
        _row("Combo A {relstr, volume, phases}", base, base, "—"),
    ]
    for key, label in (("aoc", "A + aoc"),
                       ("patterns_ctx", "A + patterns_ctx (repaired)"),
                       ("patterns_ctx+aoc", "A + patterns_ctx + aoc")):
        res = results[key]
        lines.append(_row(label, base, res, verdict_for(base, res)))
    lines += [
        "",
        "Gate firing diagnostics (over the emitted signals): "
        f"aoc_gate contributed on **{diag['aoc_gate_hits']}** of "
        f"{diag['n_signals']} signals; the repaired patterns gate "
        f"fired on **{diag['patterns_gate_hits']}** signals.",
        "",
        "Verdict rule (same as Phases 3/4, judged against Combo A): "
        "ADDS = book return up > 1pp without poisoning the promoted "
        "grade (Δ shadow @≥60 >= -0.005R); DILUTES = book down > 1pp "
        "OR shadow @≥60 avg R down > 0.010R; else NEUTRAL.",
        "",
        "## Honest verdict",
        "",
        "_Filled in after the run — see the numbers above._",
        "",
        "## Caveats",
        "",
        "- Same single in-sample window as Phases 3–4 (one bullish "
        "gold regime); verdicts are diagnostics, not out-of-sample "
        "proof.",
        "- Gate weights (+4/-4) and the 3-family major threshold are "
        "hand-set trader priors, not fitted parameters.",
        "- The repaired patterns gate changes Phase-4's documented "
        "patterns_ctx row (+19.04% was the OLD pool-proximity test); "
        "CONTEXT.md keeps the Phase-4 numbers as history, this file "
        "is the repair's record.",
        "",
    ]
    return "\n".join(lines)


def main_aoc(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing Phase-4.5 "
                                             "AOC ablation")
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--cache", type=str, default=str(DEFAULT_CACHE))
    args = ap.parse_args(argv)

    df = load_gold_h1(Path(args.cache), refresh=args.refresh_data)
    peers = load_peers(refresh=args.refresh_data)
    bundle = run_aoc(df, peers)
    report = build_aoc_report(bundle, df)
    out = REPO_ROOT / "AOC.md"
    out.write_text(report)
    print(report)
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main_aoc()
