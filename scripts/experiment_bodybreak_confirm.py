"""Runner: sweep-candle body-break confirmation vs our pillar.

Runs goldenswing.experiments.bodybreak_confirm.run_experiment(), prints the
A/B5/B10/C comparison, and writes experiments/BODYBREAK_CONFIRM.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from goldenswing.experiments import bodybreak_confirm as E  # noqa: E402


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
    L.append("# Experiment: sweep-candle body-break confirmation vs our pillar")
    L.append("")
    L.append("**Source:** SMC Liquidity Sweep Scalper (RanaAli878, MQL5 "
             "CodeBase, user-mined 2026-10-11, claim #49). The claim: after a "
             "liquidity sweep, wait N bars for a confirmation break back "
             "through THE SWEEP CANDLE'S BODY and enter at market. The third "
             "confirmation definition for the identical sweep setup -- ours "
             "is displacement+CHoCH/BOS (the Phase-2 pillar), the Tsutsumi "
             "video's was the engulfing close (killed, #27).")
    L.append("")
    L.append(f"**Data:** GC=F H1, {res['bars']} bars, "
             f"{res['start']} -> {res['end']} "
             f"({res['years']:.2f}y). $0.45/oz cost, 72-bar horizon, Phase-2 "
             "exit machinery (stop = sweep extreme +/- 0.15 ATR, buffered "
             "0.25 ATR; 50% partial at first pool >= 1R; trail after +1.5R).")
    L.append("")
    L.append("**Variants** (identical sweep universe, identical gates: daily "
             "veto, room-to-pool, stop-distance band; NO score gate -- this "
             "isolates the confirmation definition):")
    L.append("- **A (ours):** sweep -> displacement + CHoCH/BOS within 8 "
             "bars (the detector's exact `_sweep_confirmed`), first trigger "
             "per sweep.")
    L.append("- **B5 / B10 (theirs):** sweep -> first bar within 5 / 10 bars "
             "whose CLOSE crosses back through the sweep candle's body in "
             "the reversal direction, same gates.")
    L.append("- **C (theirs + quality gates):** B10 plus the author's stated "
             "gates -- confirmation candle body > 0.5x range (momentum, not "
             "drift) and stop >= 0.5 ATR (meaningful vs ATR).")
    L.append("Entry is next-bar-open for all variants (harness convention; "
             "the author enters at market on the trigger bar -- on H1 the "
             "close->open gap is noise and the shared harness keeps the "
             "comparison about the confirmation, not the fill).")
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
    L.append("- The author's mechanics come from his detailed writeup, not "
             "a direct source read (CodeBase download needs login); the "
             "threshold-level details of his quality gates are "
             "unverified -- B5/B10 test the core confirmation definition, "
             "C is our best-effort reconstruction of his gates.")
    L.append("- The author's HTF EMA filter and session window are NOT "
             "tested here -- this isolates the confirmation definition "
             "only.")
    L.append("- Entry at next-bar-open (harness convention), not literally "
             "at market on the trigger bar as the author states.")
    out = REPO / "experiments" / "BODYBREAK_CONFIRM.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
