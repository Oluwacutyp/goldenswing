"""Run the OTE-as-entry-zone experiment (claim #60, ICT MMXM).

Compares the sweep-reversal playbook's confirmation trigger against a
resting limit at the OTE zone (62/70/79% fib of the manipulation leg),
same setups, Phase-2 exits held constant.  Writes
experiments/OTE_ENTRY.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.experiments.ote_entry import FIB_LEVELS, run_ote


def _fmt_book(name, book):
    return (f"| {name} | {book['trades']} | {book['win_rate']:.1f} | "
            f"{book['pf']:.2f} | {book['avg_r']:+.3f} | "
            f"{book['return_pct']:+.2f}% | {book['max_dd_pct']:.2f}% | "
            f"{book['sharpe']:.2f} |")


def _fmt_shadow(name, s):
    return (f"| {name} | {s['n']} | {s['win_rate']:.1f} | {s['pf']:.2f} | "
            f"{s['avg_r']:+.3f} |")


def _halves(name, outcomes, mid_bar, years_half):
    from goldenswing.backtest.harness import book_metrics, build_book
    h1 = [o for o in outcomes if o.signal.index < mid_bar]
    h2 = [o for o in outcomes if o.signal.index >= mid_bar]
    b1 = book_metrics(build_book(h1, 60.0), years_half)
    b2 = book_metrics(build_book(h2, 60.0), years_half)
    return (f"| {name} H1 | {b1['trades']} | {b1['return_pct']:+.2f}% | "
            f"{b1['pf']:.2f} |\n"
            f"| {name} H2 | {b2['trades']} | {b2['return_pct']:+.2f}% | "
            f"{b2['pf']:.2f} |")


def main() -> None:
    res = run_ote()
    mid_bar = res.bars // 2
    years_half = res.years / 2.0

    # Strict same-setup anchor: only the signals that produced OTE setups.
    from goldenswing.backtest.harness import book_metrics, build_book
    setup_idx = set(res.setup_indices)
    anchor_same = [o for o in res.anchor_outcomes
                   if o.signal.index in setup_idx]
    anchor_same_book = book_metrics(build_book(anchor_same, 60.0), res.years)

    lines = []
    A = lines.append
    A("# Experiment: OTE as the entry zone (claim #60)")
    A("")
    A("**Claim (ICT MMXM, mined 2026-10-11):** after the manipulation leg "
      "(sweep + displacement), place a resting limit at the Optimal Trade "
      "Entry zone (62-79% fib retracement), confluent with the displacement "
      "order block — instead of chasing a confirmation trigger.")
    A("")
    A(f"**Data:** GC=F H1, {res.bars:,} bars, {res.start} -> {res.end} "
      f"({res.years:.2f}y). $0.45/oz cost, Phase-2 exits, 72-bar horizon.")
    A("")
    A(f"**Universe:** {res.n_signals} detector sweep_reversal signals @ "
      f">=60 (the confirmed sweep universe); {res.n_setups} produced an "
      "OTE setup (fib zone + OB confluence + pullback side).")
    A("")
    A("## Books (one position at a time, entry mechanic is the variable)")
    A("")
    A("| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe |")
    A("|---|---|---|---|---|---|---|---|")
    A(_fmt_book("Anchor (confirmation trigger)", res.anchor_book))
    A(_fmt_book("Anchor on OTE setups only", anchor_same_book))
    for fib in FIB_LEVELS:
        fr = res.fib[fib]
        A(_fmt_book(f"OTE {int(fib*100)}% limit", fr["book"]))
    A("")
    A("## Shadow (every fill standalone)")
    A("")
    A("| Slice | n | Win% | PF | Avg R |")
    A("|---|---|---|---|---|")
    A(_fmt_shadow("Anchor sweep slice", res.anchor_shadow))
    for fib in FIB_LEVELS:
        fr = res.fib[fib]
        A(_fmt_shadow(f"OTE {int(fib*100)}% fills", fr["shadow"]))
    A("")
    A("## Fill mechanics per fib level")
    A("")
    A("| Fib | Setups | Fills | Fill rate | Expired/violated | Risk-band skip |")
    A("|---|---|---|---|---|---|")
    for fib in FIB_LEVELS:
        fr = res.fib[fib]
        A(f"| {int(fib*100)}% | {res.n_setups} | {fr['n_fills']} | "
          f"{fr['fill_rate']:.1%} | {fr['n_expired']} | {fr['n_risk_skip']} |")
    A("")
    A("## The two questions the experiment was built to answer")
    A("")
    A("**1. Fill-price improvement** (OTE fill vs the anchor's next-bar-open "
      "entry, in OTE-risk units; positive = OTE got the better price):")
    A("")
    for fib in FIB_LEVELS:
        fr = res.fib[fib]
        A(f"- {int(fib*100)}%: {fr['fill_improve_avg_r']:+.3f}R per filled trade")
    A("")
    A("**2. Missed-trade cost** (what the anchor did with the setups the "
      "OTE limit never filled — the runners that never pulled back):")
    A("")
    for fib in FIB_LEVELS:
        fr = res.fib[fib]
        A(f"- {int(fib*100)}%: {fr['missed_n']} missed setups, "
          f"avg {fr['missed_avg_r']:+.3f}R, total {fr['missed_total_r']:+.1f}R")
    A("")
    A("## Window halves (regime robustness)")
    A("")
    A("| Book | Trades | Return | PF |")
    A("|---|---|---|---|")
    A(_halves("Anchor", res.anchor_outcomes, mid_bar, years_half))
    for fib in FIB_LEVELS:
        fr = res.fib[fib]
        A(_halves(f"OTE {int(fib*100)}%", fr["outcomes"], mid_bar,
                  years_half))
    A("")
    A("## Verdict")
    A("")
    A("**KILLED as an entry-mechanic replacement — with a clean mechanism.**")
    A("")
    A("The MMXM promise is real: the resting OTE limit gets dramatically "
      "better prices on fills (+0.84R at 62%, +1.21R at 70%, +2.09R at "
      "79% vs the anchor's next-bar-open entry). But the runners that "
      "never pull back are where the money was: the setups OTE never "
      "filled averaged +0.53R for the anchor entry (+8.0R total at 62/70%, "
      "+12.9R at 79%). The missed winners cost more than the price "
      "improvement earns, at every fib level, in both window halves "
      "combined. Entry mechanic vs entry mechanic, same setups, same "
      "exits: resting at OTE loses to chasing the confirmation.")
    A("")
    A("Secondary read: the 70% level is the best of the three (book "
      "-0.45%, shadow +0.053R) and the 79% level is the worst (book "
      "-3.20%, fill rate 34%) — deep pullbacks after displacement tend "
      "to keep going. The zone placement rhymes with the model, but "
      "best-of-three-losers is not an adoption.")
    A("")
    A("What survives: nothing as an entry mechanic. The OTE zone keeps "
      "its existing role as a context gate (Fib/OTE family); this kill "
      "is about the resting-limit entry only.")
    A("")
    A("## Honest limits")
    A("")
    A("- One instrument (gold H1), one 2-year bullish window, in-sample.")
    A("- The OTE stop sits 0.25 ATR beyond the sweep extreme; the anchor's "
      "stop sits 0.15 + 0.25 = 0.40 ATR beyond it (detector invalidation + "
      "harness buffer). The 0.15 ATR delta is disclosed, not hidden.")
    A("- Pools ahead are the signal bar's pools for both variants.")
    A("- The MMXM model also uses SMT divergence and session filters we "
      "do not model; this tests the OTE entry mechanic alone.")

    out = Path(__file__).resolve().parents[1] / "experiments" / "OTE_ENTRY.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
