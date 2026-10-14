"""Runner: engulfing-close as the sweep-reversal trigger.

Runs goldenswing.experiments.engulf_trigger.run_experiment(), prints the
A/B/C comparison, and writes experiments/ENGULF_TRIGGER.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from goldenswing.experiments import engulf_trigger as E  # noqa: E402


def _pf(pf: float) -> str:
    return f"{pf:.2f}" if pf != float("inf") else "inf"


def _book_row(name: str, b: dict) -> str:
    return (f"| {name} | {b['trades']} | {b['win_rate']:.1f}% | "
            f"{_pf(b['pf'])} | {b['avg_r']:+.3f} | {b['return_pct']:+.2f}% | "
            f"{b['max_dd_pct']:.2f}% | {b['sharpe']:.2f} | {b['avg_hold']:.1f} |")


def _shadow_row(name: str, s: dict) -> str:
    return (f"| {name} | {s['n']} | {s['win_rate']:.1f}% | "
            f"{_pf(s['pf'])} | {s['avg_r']:+.3f} |")


def main() -> None:
    res = E.run_experiment()
    print(f"bars={res['bars']} years={res['years']:.2f} "
          f"{res['start']} -> {res['end']}")
    for mode in E.VARIANTS:
        r = res[mode]
        print(f"variant {mode}: hits={len(r['hits'])} "
              f"shadow_n={r['shadow']['n']} book={r['book']['trades']} "
              f"ret={r['book']['return_pct']:+.2f}%")
    print("anchor:", res["anchor"])
    print("timing:", res["timing"])

    L = []
    L.append("# Experiment: engulfing-close as the sweep-reversal trigger")
    L.append("")
    L.append("**Source:** Pro Trading School \"Tsutsumi\" video (user-mined "
             "2026-10-11). The claim: for sweep reversals, enter at the CLOSE "
             "of the engulfing candle instead of on displacement + CHoCH/BOS "
             "confirmation. Never trade engulfings mid-range -- only at "
             "edges/zones.")
    L.append("")
    L.append(f"**Data:** GC=F H1, {res['bars']} bars, "
             f"{res['start']} -> {res['end']} "
             f"({res['years']:.2f}y). $0.45/oz cost, 72-bar horizon, Phase-2 "
             "exit machinery (stop = sweep extreme +/- 0.15 ATR, buffered "
             "0.25 ATR; 50% partial at first pool >= 1R; trail after +1.5R).")
    L.append("")
    L.append("**Variants** (identical gates: daily veto, room-to-pool, "
             "stop-distance band; NO score gate -- scores are a separate "
             "system, this isolates the trigger):")
    L.append("- **A (ours):** sweep -> displacement + CHoCH/BOS confirmation "
             "within 8 bars (the detector's exact `_sweep_confirmed`), first "
             "trigger bar per sweep.")
    L.append("- **B (theirs):** sweep -> first engulfing candle (body engulfs "
             "prior body) closing in the reversal direction within 8 bars, "
             "at a pool/zone/AOC (never mid-range), same gates.")
    L.append("- **C (confluence):** first bar where BOTH the engulfing-at-level "
             "and the displacement + CHoCH confirmation hold.")
    L.append("Entry is next-bar-open for all three (harness convention; the "
             "video says enter at the engulfing close -- on H1 the close->open "
             "gap is noise and the shared harness keeps the comparison about "
             "the trigger).")
    L.append("")
    L.append("## Anchor validation")
    L.append("")
    a = res["anchor"]
    L.append(f"The real detector (Phase-2 config, emit floor 35) emits "
             f"**{a['n_detector_signals']}** sweep-reversal signals on this "
             f"window (BACKTEST2 shadow slice: 49). Variant A (per-sweep "
             f"first-trigger, no score gate) fires on "
             f"**{a['n_variant_a']}** sweeps; **{a['overlap']}** of its "
             f"trigger bars coincide with a detector signal bar. The "
             "difference is dedupe (one opportunity per sweep here vs the "
             "detector's min-gap re-emission), not the trigger definition.")
    L.append("")
    L.append("## Shadow (every trigger standalone)")
    L.append("")
    L.append("| Variant | n | Win% | PF | Avg R |")
    L.append("|---|---|---|---|---|")
    for mode in E.VARIANTS:
        L.append(_shadow_row(f"{mode}", res[mode]["shadow"]))
    L.append("")
    L.append("## Books (one position at a time, sequential)")
    L.append("")
    L.append("| Variant | Trades | Win% | PF | Avg R | Return | MaxDD | "
             "Sharpe | Avg hold |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for mode in E.VARIANTS:
        L.append(_book_row(f"{mode}", res[mode]["book"]))
    L.append("")
    L.append("## Window halves (book)")
    L.append("")
    L.append("| Variant | Half | Trades | Win% | PF | Avg R | Return |")
    L.append("|---|---|---|---|---|---|---|")
    for mode in E.VARIANTS:
        for half, key in (("H1", "book_h1"), ("H2", "book_h2")):
            b = res[mode][key]
            L.append(f"| {mode} | {half} | {b['trades']} | "
                     f"{b['win_rate']:.1f}% | {_pf(b['pf'])} | "
                     f"{b['avg_r']:+.3f} | {b['return_pct']:+.2f}% |")
    L.append("")
    L.append("## Timing (sweeps that fired in both variants)")
    L.append("")
    L.append("| Pair | n | Mean lag A (bars) | Mean lag X (bars) |")
    L.append("|---|---|---|---|")
    for t in res["timing"]:
        L.append(f"| {t['pair']} | {t['n']} | {t['mean_lag_a']:.2f} | "
                 f"{t['mean_lag_x']:.2f} |")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append("_To be written from the numbers above._")
    L.append("")
    L.append("### Honest limits")
    L.append("")
    L.append("- One instrument (gold H1), one 2-year bullish window, "
             "in-sample design.")
    L.append("- The engulfing trigger is evaluated on H1 bars; the video's "
             "examples are lower timeframes -- the verdict is for "
             "GoldenSwing's market and timeframe.")
    L.append("- Entry at next-bar-open (harness convention), not literally "
             "at the engulfing close as the video states.")
    out = REPO / "experiments" / "ENGULF_TRIGGER.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
