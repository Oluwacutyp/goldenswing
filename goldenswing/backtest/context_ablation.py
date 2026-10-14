"""Phase-4 ablation: context-gated families + trendlines vs Combo A.

Phase 3 scored ten families FLAT and six diluted.  The owner's rule —
"the problem might not be the fib and the rest but how they are used,
where they are used and when they are used" — is tested here: each
diluter was re-designed as a CONTEXT GATE (goldenswing/features/
gating.py) that is silent outside its trader context, plus a new
trendline family whose events are contextual by construction.

Method (identical harness to Phases 3/3.5 — same states, same Phase-2
flags, v2 exits, $0.45/oz, 1% risk):

- anchors: Phase-2 baseline (features off, must reproduce +18.17%)
  and Combo A {relstr, volume, phases} (must reproduce +25.62%,
  PF 1.22) — Combo A is the baseline every gated row is judged on;
- one at a time: Combo A + each gated family, and A + trendlines;
- the full gated stack: Combo A + all six gates + trendlines.

Verdicts are the Phase-3 mechanical rule (ablation.verdict_for)
applied against Combo A: a gated family that still dilutes stays OFF.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..features import FeatureEngine
from ..reader.reader import MarketReader
from .ablation import verdict_for
from .data import DEFAULT_CACHE, REPO_ROOT, load_gold_h1, load_peers
from .validation import COMBO_A, run_config

GATED = ("fib_ctx", "zones_ctx", "ichimoku_ctx", "patterns_ctx",
         "footprints_ctx", "mtf_ctx")
ADDITIONS = GATED + ("trendlines",)
FULL_STACK = COMBO_A | frozenset(ADDITIONS)


def run_phase4(df: pd.DataFrame, peers: dict | None) -> dict:
    plain_states = MarketReader().read(df)
    featured_states = MarketReader().read(df)
    FeatureEngine().attach(df, featured_states, peers)

    results = {}
    results["phase2"] = run_config(df, plain_states, None,
                                   "Phase 2 baseline (anchor)")
    results["combo_a"] = run_config(df, featured_states, COMBO_A,
                                    "Combo A (baseline)")
    for name in ADDITIONS:
        results[name] = run_config(
            df, featured_states, COMBO_A | {name}, f"A + {name}")
    results["full_stack"] = run_config(df, featured_states, FULL_STACK,
                                        "Full gated stack")
    return results


def _row(label: str, base, res, verdict: str) -> str:
    b = res.book60
    return (
        f"| {label} | {res.n_signals} | {res.shadow60['n']} | "
        f"{b['trades']} | {b['pf']:.2f} | {b['return_pct']:+.2f}% | "
        f"{b['max_dd_pct']:.2f}% | {b['sharpe']:.2f} | "
        f"{res.shadow60['avg_r']:+.3f} | "
        f"{res.shadow60['avg_r'] - base.shadow60['avg_r']:+.3f} | "
        f"{res.book50['return_pct']:+.2f}% | {verdict} |")


_GATE_DOCS = """## The gates (what "used properly" means, per family)

Each gate returns NOTHING outside its context — no bonus, no penalty.
Full rationale lives in `goldenswing/features/gating.py`.

| Family | Speaks ONLY when… | Rationale |
|---|---|---|
| Fib / OTE (`fib_ctx`) | price is retracing INTO the OTE zone (0.62–0.79) of a confirmed displacement leg, the leg broke structure in the trade direction, momentum is no longer extending (the pullback is in progress), structure has not flipped against, and the HTF (daily or MTF D1) bias agrees | Fib is a pullback-entry tool, not a level oracle — the zone means something only while a real leg is being retraced with the HTF behind it |
| Zones (`zones_ctx`) | the zone is FRESH (untested), the daily bias agrees with trading it, price is at/near it, and the approach is slowing (|momentum| ≤ 1 ATR, no knife-through displacement against) | a tested zone is spent; a fresh zone against the daily bias is a coin flip; how price ARRIVES at a level is half the signal |
| Ichimoku (`ichimoku_ctx`) | regime vote comes from the DAILY cloud only (price above/below it); the H1 cloud confirms entries ONLY in an expansion phase — never in consolidation/bases; inside the D1 cloud the family is silent | the cloud is a regime instrument first; on H1 inside a base it is chopped noise, which is why the flat version read as a trend proxy and diluted |
| Patterns (`patterns_ctx`) | the pattern stands at a MAJOR level (liquidity pool / daily or HTF swing), after an extended move (dealing range ≥ 5 ATR, height ≥ 1.5 ATR), AND the neckline has BROKEN with displacement (or a confirming structure event) | pattern alone never scores — location + trigger is the trade; mid-range, unbroken patterns are trivia |
| Footprints (`footprints_ctx`) | the institutional signature counts only POST-SWEEP AT A LEVEL (OB/FVG at hand, or the sweep level itself is major); the retail-chase caution applies only at climax extremes (climax volume + range extreme); CTA is NOT scored (its information is the trend the core already owns) | size entering through a flushed level is the story — the same signature mid-range is not; and chasing is only fatal at extremes, not in every strong continuation |
| MTF (`mtf_ctx`) | CONFIRM: +3 only when D1 AND H4 both align. VETO: when both oppose, the score is CAPPED at 59 — visible, but never a Golden Swing | MTF is a veto/confirm instrument, not a points fountain; Phase 3's additive version re-bought trend points and diluted by double-counting |
| Trendlines (`trendlines`) | BOUNCE at a validated line (≥ 1 retest — the 3rd contact) scores only inside trend context; a line BREAK scores only WITH displacement (a structure event), for/against accordingly; riding a validated line in trend earns a small proximity credit | two points are a guess, the retest validates; a break without displacement is a wick story |

### Trendline lifecycle (features/trendlines.py)

Support = through two HIGHER confirmed swing lows; resistance =
through two LOWER confirmed swing highs.  A line is known at the
second anchor's confirmation bar (causal).  States: FORMING (2
anchors) → ACTIVE (first retest touch; touches re-arm only after
price spends bars away) → BROKEN (close through by > 0.2 ATR —
terminal, an event on that bar) / EXPIRED (age > 300 bars or drift
> 6 ATR — silent).  A touch whose close rejects the line by
≥ 0.25 ATR is a BOUNCE.  Near-parallel duplicates are suppressed at
creation (price within 0.25 ATR and slope within 25% — the older line
owns the geometry).  Creation (builder) and lifecycle (manager) are
separate passes, mirroring the MQL5 builder/manager split.
"""


def build_context_report(results: dict, df: pd.DataFrame,
                         peers_available: dict) -> str:
    base = results["combo_a"]
    anchor = results["phase2"]
    lines = [
        "# GoldenSwing — Context Gates + Trendlines (Phase 4)",
        "",
        "**Owner's rule under test:** *the problem might not be the "
        "fib and the rest but how they are used, where they are used "
        "and when they are used.*  Phase 3 applied the families flat "
        "and six diluted.  Phase 4 re-tests them as context gates "
        "(silent outside their context) plus a new trendline family.",
        "",
        f"**Data:** GC=F hourly, {len(df):,} bars, {df.index[0]} → "
        f"{df.index[-1]}. Same Phase-2 entries/exits/costs as "
        "Phases 3/3.5; the ONLY change per row is the detector's "
        "feature-family set.",
        "",
        "**Anchors (must reproduce):** Phase-2 baseline "
        f"{anchor.book60['return_pct']:+.2f}% / PF "
        f"{anchor.book60['pf']:.2f} / {anchor.book60['trades']} trades "
        "(BACKTEST2: +18.17% / 1.16 / 199) · Combo A "
        f"{base.book60['return_pct']:+.2f}% / PF "
        f"{base.book60['pf']:.2f} / {base.book60['trades']} trades "
        "(VALIDATION: +25.62% / 1.22 / 199).",
        "",
        _GATE_DOCS,
        "## Ablation — Combo A is the baseline to beat",
        "",
        "\"Promoted\" / \"Shadow @≥60\" as in FEATURES.md: the shadow "
        "grade measures the quality of what a config promotes into "
        "the traded region, where dilution shows first.",
        "",
        "| Config | Signals | Promoted | Book trades | PF | Return | "
        "MaxDD | Sharpe | Shadow @≥60 avg R | Δ shadow R | "
        "Book@50 return | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        _row("Combo A {relstr, volume, phases}", base, base, "—"),
    ]
    for name in ADDITIONS:
        res = results[name]
        lines.append(_row(f"A + {name}", base, res,
                          verdict_for(base, res)))
    stack = results["full_stack"]
    lines.append(_row("FULL gated stack (A + all gates + trendlines)",
                      base, stack, verdict_for(base, stack)))
    lines += [
        "",
        "Verdict rule (same as Phase 3, judged against Combo A): "
        "ADDS = book return up > 1pp without poisoning the promoted "
        "grade (Δ shadow @≥60 >= -0.005R); DILUTES = book down > 1pp "
        "OR shadow @≥60 avg R down > 0.010R; else NEUTRAL.  A gated "
        "family that still dilutes stays OFF — recorded here, not "
        "silently kept.",
        "",
        "## Caveats",
        "",
        "- Same single in-sample window as Phases 3/3.5 (one bullish "
        "gold regime); gated verdicts are diagnostics, not "
        "out-of-sample proof.  The OOS instruments inherit whatever "
        "config this phase recommends and must be re-run before any "
        "deployment claim.",
        "- Gate conditions are hand-set trader rules, not fitted "
        "parameters — a gate can fail because a condition is wrong "
        "(too strict: never fires; too loose: flat behaviour returns), "
        "not because the family's information is worthless.  The "
        "promoted-count column shows how often each gate actually "
        "spoke.",
        "- The MTF veto cap (59) acts on the score AFTER family "
        "deltas; a capped signal still appears in the shadow book at "
        "its capped score.",
        "",
    ]
    return "\n".join(lines)


def main4(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing Phase-4 ablation")
    ap.add_argument("--refresh-data", action="store_true")
    ap.add_argument("--cache", type=str, default=str(DEFAULT_CACHE))
    args = ap.parse_args(argv)

    df = load_gold_h1(Path(args.cache), refresh=args.refresh_data)
    peers = load_peers(refresh=args.refresh_data)
    results = run_phase4(df, peers)
    peers_avail = {k: bool(v is not None and len(v) > 100)
                   for k, v in (peers or {}).items()}
    report = build_context_report(results, df, peers_avail)
    out = REPO_ROOT / "CONTEXT.md"
    out.write_text(report)
    print(report)
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main4()
