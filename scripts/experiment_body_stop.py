"""Runner: body-anchored stops vs wick-extreme stops for sweep reversals.

Runs goldenswing.experiments.body_stop.run_experiment(), prints the
A/B/B2 comparison, and writes experiments/BODY_STOP.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from goldenswing.experiments import body_stop as E  # noqa: E402


def _pf(pf: float) -> str:
    return f"{pf:.2f}" if pf != float("inf") else "inf"


def _book_row(name: str, b: dict) -> str:
    return (f"| {name} | {b['trades']} | {b['win_rate']:.1f}% | "
            f"{_pf(b['pf'])} | {b['avg_r']:+.3f} | {b['return_pct']:+.2f}% | "
            f"{b['max_dd_pct']:.2f}% | {b['sharpe']:.2f} | {b['avg_hold']:.1f} |")


def _shadow_row(name: str, s: dict) -> str:
    return (f"| {name} | {s['n']} | {s['win_rate']:.1f}% | "
            f"{_pf(s['pf'])} | {s['avg_r']:+.3f} |")


def _mix_row(name: str, m: dict) -> str:
    return (f"| {name} | {m['n']} | {m['stop']} | {m['trail']} | "
            f"{m['timeout']} | {m['stop_rate'] * 100:.1f}% | "
            f"{m['wicked_inv_rate'] * 100:.1f}% |")


def main() -> None:
    res = E.run_experiment()
    print(f"bars={res['bars']} years={res['years']:.2f} "
          f"{res['start']} -> {res['end']}")
    for mode in E.VARIANTS:
        r = res[mode]
        print(f"variant {mode}: hits={len(r['hits'])} "
              f"shadow_n={r['shadow']['n']} book={r['book']['trades']} "
              f"ret={r['book']['return_pct']:+.2f}% "
              f"stop_rate={r['exit_mix']['stop_rate'] * 100:.1f}%")
    print("anchor:", res["anchor"])

    L = []
    L.append("# Experiment: body-anchored stops for sweep reversals")
    L.append("")
    L.append("**Source:** Faiz SMC's ICT turtle soup video (user-mined "
             "2026-10-11, claim #63). The claim: anchor the stop at the "
             "sweep candle's BODY (not the wick extreme) -- deliberately "
             "tighter, accepting more stop-outs for better R:R. Our "
             "convention is wick extreme + 0.15 ATR invalidation + 0.25 ATR "
             "harness buffer.")
    L.append("")
    L.append(f"**Data:** GC=F H1, {res['bars']} bars, "
             f"{res['start']} -> {res['end']} "
             f"({res['years']:.2f}y). $0.45/oz cost, 72-bar horizon, "
             "Phase-2 exit machinery (50% partial at first pool >= 1R; "
             "trail after +1.5R). Entries and exits held constant -- this "
             "is a pure stop-placement A/B.")
    L.append("")
    L.append("**Variants** (identical sweep universe, identical "
             "displacement+CHoCH trigger within 8 bars, identical gates "
             "evaluated against each variant's own invalidation; NO score "
             "gate):")
    L.append("- **A (ours):** invalidation at the sweep wick extreme -/+ "
             "0.15 ATR; stop = invalidation -/+ 0.25 ATR buffer (total "
             "0.40 ATR beyond the extreme).")
    L.append("- **B (theirs):** invalidation at the sweep candle's body "
             "extreme (min(open,close) for longs, max(open,close) for "
             "shorts) -/+ 0.15 ATR; same 0.25 ATR buffer. The anchor point "
             "is the ONLY change.")
    L.append("- **B2 (pure body):** invalidation AT the body extreme "
             "exactly, zero buffer anywhere (stop_buffer_atr=0.0).")
    L.append("")
    L.append("## Anchor validation")
    L.append("")
    a = res["anchor"]
    L.append(f"The real detector (Phase-2 config, emit floor 35) emits "
             f"**{a['n_detector_signals']}** sweep-reversal signals on this "
             f"window. Variant A fires on **{a['n_variant_a']}** sweeps; "
             f"**{a['overlap']}** of its trigger bars coincide with a "
             f"detector signal bar.")
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
    L.append("## Exit mix (the stop-out question)")
    L.append("")
    L.append("`stop_rate` = share of trades exiting at the initial stop. "
             "`wicked_inv_rate` = share of trades where price crossed the "
             "RAW invalidation at some point (for A: how often the body "
             "anchor was violated but the extreme stop survived).")
    L.append("")
    L.append("| Variant | n | Stop | Trail | Timeout | Stop-out rate | "
             "Wicked-inv rate |")
    L.append("|---|---|---|---|---|---|---|")
    for mode in E.VARIANTS:
        L.append(_mix_row(f"{mode}", res[mode]["exit_mix"]))
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append("_To be written from the numbers above._")
    L.append("")
    L.append("### Honest limits")
    L.append("")
    L.append("- One instrument (gold H1), one 2-year bullish window, "
             "in-sample design.")
    L.append("- The tighter stop changes R itself (R is measured on the "
             "buffered stop), so avg-R comparisons across variants are "
             "indicative; the book return is the bottom line.")
    L.append("- Entry at next-bar-open (harness convention).")
    out = REPO / "experiments" / "BODY_STOP.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
