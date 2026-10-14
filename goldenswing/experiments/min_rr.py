"""Minimum R:R qualification filter experiment.

Mined claim (Pro Trading School video, 2026-10-11, user-mined):
"Before entering, confirm the trade offers at least 2-to-1 reward-to-risk."

The exit stepper currently takes partials at the first pool >= 1R and
trails — there is NO minimum-R:R qualification on entry.  This experiment
adds one: at signal time, compute

    risk   = |assumed entry - buffered structural stop|
    reward = |first opposing pool - assumed entry|

and skip the signal when reward / risk < threshold.  Thresholds swept:
1.5, 2.0, 2.5, 3.0.  The anchor is the current recommended config
Combo A + aoc (+28.92% / PF 1.25 / 199 trades, AOC.md) — it must
reproduce exactly before any filter verdict is trusted.

Causality contract (documented, tested):
- assumed entry = sig.entry_ref = close of the SIGNAL bar (the detector's
  own causal entry reference).  The outcome simulation still fills at
  the open of t+1 per repo convention; the one-bar gap between the
  qualification price and the fill price is disclosed, not hidden.
- stop = the Phase-2 buffered structural stop (invalidation -/+ 0.25 ATR),
  the same stop the book's R is measured on.
- "first pool" mirrors the exit stepper's partial-target definition:
  nearest opposing pool at least partial_min_r (1.0R) beyond the assumed
  entry.  No such pool -> no measurable target -> the signal FAILS the
  qualification (conservative: no target, no trade).
- The video's "structural stop" is our buffered stop; R is measured on
  the buffered stop, consistent with the book.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import FeatureEngine
from ..golden.detector import (DetectorConfig, GoldenSignal,
                               GoldenSwingDetector)
from ..reader.reader import MarketReader
from ..backtest.ablation import P2_FLAGS
from ..backtest.data import DEFAULT_CACHE, REPO_ROOT, load_gold_h1, load_peers
from ..backtest.harness import (ExitConfig, Outcome, book_metrics, build_book,
                                simulate_signal_v2, slice_stats)
from ..backtest.validation import COMBO_A

FAMILIES = COMBO_A | {"aoc"}
THRESHOLDS: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
BOOK_THRESHOLD = 60.0

# Anchor numbers from AOC.md ("A + aoc" row).  The unfiltered run must
# reproduce these before any filter verdict is trusted.
ANCHOR_RETURN = 28.92
ANCHOR_PF = 1.25
ANCHOR_TRADES = 199


def qualification(sig: GoldenSignal, min_rr: float,
                  exits: ExitConfig | None = None) -> tuple[bool, float | None]:
    """Signal-time R:R qualification.  Pure function of the signal —
    causal by construction (entry_ref, invalidation, atr and targets are
    all known at the signal bar's close).

    Returns ``(passes, rr)``; ``rr`` is None when the trade has no
    measurable reward (no opposing pool >= 1R away) or no measurable
    risk (risk <= 0) — both FAIL the qualification.
    """
    exits = exits or ExitConfig()
    d = sig.direction
    entry = sig.entry_ref
    buffer = exits.stop_buffer_atr * sig.atr if sig.atr > 0 else 0.0
    stop = sig.invalidation - d * buffer
    risk = (entry - stop) * d
    if risk <= 0:
        return False, None
    ahead = sorted(
        (p for p in sig.targets
         if (p - entry) * d >= exits.partial_min_r * risk),
        key=lambda p: abs(p - entry))
    if not ahead:
        return False, None
    rr = float((ahead[0] - entry) * d / risk)
    return rr >= min_rr, rr


@dataclass
class FilterResult:
    threshold: float
    kept_signals: int
    killed_signals: int
    book: dict
    shadow_kept: dict
    shadow_killed: dict
    halves: dict = field(default_factory=dict)  # "h1"/"h2" -> book dict


@dataclass
class MinRRBundle:
    bars: int
    start: str
    end: str
    years: float
    n_outcomes: int
    anchor: dict            # unfiltered book60
    anchor_shadow60: dict
    results: list[FilterResult] = field(default_factory=list)


def run_min_rr(df: pd.DataFrame, peers: dict | None,
               thresholds: tuple[float, ...] = THRESHOLDS,
               exits: ExitConfig | None = None) -> MinRRBundle:
    exits = exits or ExitConfig()
    states = MarketReader().read(df)
    FeatureEngine().attach(df, states, peers)
    detector = GoldenSwingDetector(DetectorConfig(
        emit_floor=35.0, golden_threshold=BOOK_THRESHOLD,
        feature_families=FAMILIES, **P2_FLAGS))
    signals = detector.scan(states)

    outcomes: list[Outcome] = []
    for sig in signals:
        o = simulate_signal_v2(df, sig, exits)
        if o is not None:
            outcomes.append(o)

    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    years = span_days / 365.25
    mid = len(df) // 2

    anchor_book = book_metrics(build_book(outcomes, BOOK_THRESHOLD), years)
    anchor_shadow = slice_stats([o for o in outcomes
                                 if o.signal.score >= BOOK_THRESHOLD])

    # --- anchor must reproduce AOC.md before any verdict is trusted ---
    assert anchor_book["trades"] == ANCHOR_TRADES, (
        f"anchor trades {anchor_book['trades']} != {ANCHOR_TRADES}")
    assert abs(anchor_book["return_pct"] - ANCHOR_RETURN) <= 0.10, (
        f"anchor return {anchor_book['return_pct']:.2f} != {ANCHOR_RETURN}")
    assert abs(anchor_book["pf"] - ANCHOR_PF) <= 0.02, (
        f"anchor PF {anchor_book['pf']:.2f} != {ANCHOR_PF}")

    results: list[FilterResult] = []
    for th in thresholds:
        flags = [qualification(o.signal, th, exits)[0] for o in outcomes]
        kept = [o for o, k in zip(outcomes, flags) if k]
        killed = [o for o, k in zip(outcomes, flags) if not k]
        book = book_metrics(build_book(kept, BOOK_THRESHOLD), years)
        g60 = [o for o in kept if o.signal.score >= BOOK_THRESHOLD]
        x60 = [o for o in killed if o.signal.score >= BOOK_THRESHOLD]
        halves = {}
        for name, pred in (("h1", lambda o: o.signal.index < mid),
                           ("h2", lambda o: o.signal.index >= mid)):
            halves[name] = book_metrics(
                build_book([o for o in kept if pred(o)],
                           BOOK_THRESHOLD), years / 2.0)
        results.append(FilterResult(
            threshold=th, kept_signals=len(kept),
            killed_signals=len(killed), book=book,
            shadow_kept=slice_stats(g60), shadow_killed=slice_stats(x60),
            halves=halves))
    return MinRRBundle(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        years=years, n_outcomes=len(outcomes), anchor=anchor_book,
        anchor_shadow60=anchor_shadow, results=results)


def verdict_for(anchor: dict, res: FilterResult) -> str:
    """Mechanical verdict vs the anchor book@60.

    ADDS = book return up > 1pp AND risk-adjusted no worse (Sharpe up,
    or MaxDD down >= 1pp, or PF up >= 0.05).
    DILUTES = book return down > 1pp.
    NEUTRAL = otherwise.  A filter that raises win rate while lowering
    total return is DILUTES — stated explicitly, not hidden.
    """
    b, a = res.book, anchor
    d_ret = b["return_pct"] - a["return_pct"]
    risk_ok = (b["sharpe"] >= a["sharpe"]
               or b["max_dd_pct"] <= a["max_dd_pct"] - 1.0
               or b["pf"] >= a["pf"] + 0.05)
    if d_ret > 1.0 and risk_ok:
        return "ADDS"
    if d_ret < -1.0:
        return "DILUTES"
    return "NEUTRAL"


def build_min_rr_report(bundle: MinRRBundle) -> str:
    a = bundle.anchor
    lines = [
        "# GoldenSwing — Minimum R:R Qualification Filter",
        "",
        "**Mined claim (Pro Trading School, user-mined 2026-10-11):** "
        "*before entering, require the trade to offer at least 2-to-1 "
        "reward-to-risk.* The exit stepper banks partials at the first "
        "pool >= 1R and trails — but no entry-time R:R qualification "
        "exists.  This experiment adds one: at signal time, "
        "risk = |signal-bar close - buffered structural stop|, "
        "reward = |first opposing pool (>= 1R away) - signal-bar close|; "
        "skip the signal when reward/risk < threshold.",
        "",
        f"**Data:** GC=F hourly, {bundle.bars:,} bars, {bundle.start} → "
        f"{bundle.end} ({bundle.years:.2f} years). Same Phase-2 "
        "entries/exits/costs as the anchor; the ONLY change per row is "
        "the entry-time filter.",
        "",
        f"**Anchor (verified, must reproduce):** Combo A + aoc, "
        f"{a['trades']} trades, PF {a['pf']:.2f}, "
        f"{a['return_pct']:+.2f}%, MaxDD {a['max_dd_pct']:.2f}%, "
        f"Sharpe {a['sharpe']:.2f} (AOC.md: +28.92% / 1.25 / 199). "
        f"Shadow @≥60: {bundle.anchor_shadow60['n']} signals, "
        f"avg R {bundle.anchor_shadow60['avg_r']:+.3f}.",
        "",
        "## Threshold sweep (traded book @ ≥60)",
        "",
        "| Min R:R | Kept / killed signals | Book trades | Win% | PF | "
        "Return | MaxDD | Sharpe | Avg R | H1 ret | H2 ret | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for res in bundle.results:
        b = res.book
        lines.append(
            f"| ≥{res.threshold:.1f} | {res.kept_signals} / "
            f"{res.killed_signals} | {b['trades']} | "
            f"{b['win_rate']:.1f} | {b['pf']:.2f} | "
            f"{b['return_pct']:+.2f}% | {b['max_dd_pct']:.2f}% | "
            f"{b['sharpe']:.2f} | {b['avg_r']:+.3f} | "
            f"{res.halves['h1']['return_pct']:+.2f}% | "
            f"{res.halves['h2']['return_pct']:+.2f}% | "
            f"{verdict_for(a, res)} |")
    lines += [
        "",
        "## Does the filter remove losers? (shadow @≥60, kept vs killed)",
        "",
        "| Min R:R | Kept n / avg R / PF | Killed n / avg R / PF |",
        "|---|---|---|",
    ]
    for res in bundle.results:
        k, x = res.shadow_kept, res.shadow_killed
        lines.append(
            f"| ≥{res.threshold:.1f} | {k['n']} / {k['avg_r']:+.3f} / "
            f"{k['pf']:.2f} | {x['n']} / {x['avg_r']:+.3f} / {x['pf']:.2f} |")
    lines += [
        "",
        "## Verdict rule",
        "",
        "ADDS = book return up > 1pp AND risk-adjusted no worse "
        "(Sharpe up, or MaxDD down ≥ 1pp, or PF up ≥ 0.05).  DILUTES = "
        "book return down > 1pp.  A filter that raises win rate while "
        "lowering total return is a dilution, stated as such.",
        "",
        "## Honest verdict",
        "",
        "_Filled in after the run — see the numbers above._",
        "",
        "## Caveats",
        "",
        "- The qualification prices the trade at the signal bar's close "
        "(sig.entry_ref); the outcome simulation fills at the next bar's "
        "open per repo convention.  The one-bar gap is disclosed; it "
        "affects the filter and the fill identically for every signal, "
        "so it cannot favour any threshold.",
        "- Signals with no opposing pool >= 1R away FAIL the "
        "qualification (no measurable target → no trade).  This is "
        "conservative and matches the video's logic.",
        "- Same single in-sample window as all Phase 3–4.5 work; "
        "verdicts are diagnostics, not out-of-sample proof.",
        "- The R:R thresholds (1.5/2.0/2.5/3.0) are the video's claim "
        "plus neighbours — not fitted.",
        "",
    ]
    return "\n".join(lines)


def main_min_rr(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing min-R:R filter "
                                             "experiment")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args(argv)

    df = load_gold_h1(DEFAULT_CACHE, refresh=args.refresh_data)
    peers = load_peers(refresh=args.refresh_data)
    bundle = run_min_rr(df, peers)
    report = build_min_rr_report(bundle)
    out = REPO_ROOT / "experiments" / "MIN_RR.md"
    out.write_text(report)
    print(report)
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main_min_rr()
