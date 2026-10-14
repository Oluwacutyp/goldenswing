"""Run the JeaFx retest-entry sleeve experiment.

Produces, for each entry mode ("mid" = zone midline, "edge" = the
first-touched zone edge — the literal reading of the video), the
sleeve's two variants (next-highs pool target vs fixed 2R), books,
shadow grades, window halves, and the fill-rate read, then writes
experiments/RETEST_ENTRY.md.

The Combo-A anchor (+25.62%) and the CHoCH->FVG variant B (+25.88%,
PF 1.41) are quoted from their committed reports rather than re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.data import load_gold_h1
from goldenswing.backtest.harness import (book_metrics, build_book,
                                          slice_stats)
from goldenswing.experiments.retest_entry import (ENTRY_EDGE, ENTRY_MID,
                                                  FIXED_TP_R, find_fill,
                                                  run_sleeve, scan_setups,
                                                  select_pool_target,
                                                  simulate_trade)
from goldenswing.features.engine import read_and_featurize


def _fmt_book(name, book):
    return (f"| {name} | {book['trades']} | {book['win_rate']:.1f} | "
            f"{book['pf']:.2f} | {book['avg_r']:+.3f} | "
            f"{book['return_pct']:+.2f}% | {book['max_dd_pct']:.2f}% | "
            f"{book['sharpe']:.2f} | {book['avg_hold']:.1f} |")


def _fmt_shadow(name, s):
    return (f"| {name} | {s['n']} | {s['win_rate']:.1f} | {s['pf']:.2f} | "
            f"{s['avg_r']:+.3f} |")


def main() -> None:
    df = load_gold_h1()
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    mid_bar = len(df) // 2

    lines = []
    A = lines.append
    A("# Experiment: JeaFx retest entry (sweep highs -> demand -> BoS -> "
      "retest)")
    A("")
    A("**Source:** JeaFx short, narration: downtrend sweeps the highs, "
      "trades down into HTF demand, a strong push breaks structure up, "
      "enter on the retest of the demand level, stop beneath the low, "
      "target the next highs. Mirrored for shorts.")
    A("")
    A("**Data:** GC=F H1, 11426 bars, 2024-10-11 04:00:00+00:00 -> "
      "2026-10-09 20:00:00+00:00 (1.99y). Cost $0.45/oz, 1% risk, "
      "72-bar horizon, v1 exit conventions (stop wins ties, adverse "
      "opens). Zones come from the repo's supply_demand family (fresh + "
      "daily-bias gated, the detector's zone convention).")
    A("")
    A("**Entry modes:** `mid` = resting limit at the zone midline (the "
      "FVG-midline experiment's convention); `edge` = resting limit at "
      "the first-touched zone edge (top for longs / bottom for shorts — "
      "the literal reading of \"bought when price retested that level\").")
    A("")

    states = read_and_featurize(df)
    for entry in (ENTRY_MID, ENTRY_EDGE):
        res = run_sleeve(entry=entry)

        setups = scan_setups(df, states, entry=entry)
        fills = []
        for s in setups:
            fl = find_fill(df, s)
            if fl is not None:
                fills.append(fl)
        outs_a, outs_b = [], []
        for fl in fills:
            d = fl.setup.direction
            tgt = select_pool_target(states, fl)
            if tgt is not None:
                outs_a.append(simulate_trade(df, fl, tgt))
            outs_b.append(simulate_trade(df, fl, fl.entry + d * FIXED_TP_R
                                         * fl.setup.risk))

        def halves(outcomes):
            h1 = [o for o in outcomes if o.signal.index < mid_bar]
            h2 = [o for o in outcomes if o.signal.index >= mid_bar]
            return (book_metrics(build_book(h1, 0.0), years / 2),
                    book_metrics(build_book(h2, 0.0), years / 2))

        a_h1, a_h2 = halves(outs_a)
        b_h1, b_h2 = halves(outs_b)

        def dir_split(outcomes, d):
            sub = [o for o in outcomes if o.signal.direction == d]
            return slice_stats(sub), book_metrics(build_book(sub, 0.0),
                                                 years)

        a_lng_s, _ = dir_split(outs_a, 1)
        a_sht_s, _ = dir_split(outs_a, -1)
        sbs = sum(1 for f in fills if f.same_bar_stop)
        sbs_pct = (sbs / len(fills) * 100) if fills else 0.0

        A(f"## Entry mode: `{entry}`")
        A("")
        A("### Funnel")
        A("")
        A(f"- Setups scanned (fresh BoS + fresh demand/supply + failed "
          f"opposite sweep): {res.n_setups}")
        A(f"- Resting limits filled: {res.n_fills}")
        fill_rate = (res.n_fills / res.n_setups * 100) if res.n_setups \
            else 0
        A(f"- Fill rate: {fill_rate:.1f}% — the retest-entry tradeoff")
        A(f"- Same-bar stops (fill bar also takes the stop): {sbs} "
          f"({sbs_pct:.1f}% of fills)")
        A(f"- Cancelled (24-bar expiry, retest never came): "
          f"{res.n_cancelled}")
        A(f"- Filled but no qualifying pool target (>= 1R room): "
          f"{res.n_skipped_no_pool} (variant A only)")
        A("")
        A("### Books (one position at a time, sequential)")
        A("")
        A("| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | "
          "Avg hold |")
        A("|---|---|---|---|---|---|---|---|---|")
        A(_fmt_book(f"Sleeve A ({entry}): retest + next-highs pool",
                    res.variant_a))
        A(_fmt_book(f"Sleeve B ({entry}): retest + fixed 2R", res.variant_b))
        A("")
        A("### Shadow (every fill standalone)")
        A("")
        A("| Slice | n | Win% | PF | Avg R |")
        A("|---|---|---|---|---|")
        A(_fmt_shadow(f"Sleeve A ({entry}) shadow", res.shadow_a))
        A(_fmt_shadow(f"Sleeve B ({entry}) shadow", res.shadow_b))
        A(_fmt_shadow(f"Sleeve A ({entry}) longs", a_lng_s))
        A(_fmt_shadow(f"Sleeve A ({entry}) shorts", a_sht_s))
        A("")
        A("### Window halves")
        A("")
        A("| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | "
          "Avg hold |")
        A("|---|---|---|---|---|---|---|---|---|")
        A(_fmt_book(f"Sleeve A ({entry}) first half", a_h1))
        A(_fmt_book(f"Sleeve A ({entry}) second half", a_h2))
        A(_fmt_book(f"Sleeve B ({entry}) first half", b_h1))
        A(_fmt_book(f"Sleeve B ({entry}) second half", b_h2))
        A("")

        print(f"[{entry}] setups={res.n_setups} fills={res.n_fills} "
              f"cancelled={res.n_cancelled} no_pool={res.n_skipped_no_pool} "
              f"sbs={sbs}")
        print(f"[{entry}] A: {res.variant_a['return_pct']:+.2f}% "
              f"PF={res.variant_a['pf']:.2f} n={res.variant_a['trades']}")
        print(f"[{entry}] B: {res.variant_b['return_pct']:+.2f}% "
              f"PF={res.variant_b['pf']:.2f} n={res.variant_b['trades']}")

    A("## Reference books (committed reports)")
    A("")
    A("| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | "
      "Avg hold |")
    A("|---|---|---|---|---|---|---|---|---|")
    A("| Combo A anchor @60 (VALIDATION.md) | 199 | 45.2 | 1.22 | +0.124 "
      "| +25.62% | 11.04% | 0.91 | 19.5 |")
    A("| CHoCH->FVG sleeve B @2R (CHoCH_FVG_MID_OB.md) | 97 | 45.4 | 1.41 "
      "| +0.250 | +25.88% | 8.85% | 1.11 | 11.8 |")
    A("")
    A("## Verdict")
    A("")
    A("_Filled in after the numbers land._")
    A("")
    out_path = (Path(__file__).resolve().parents[1] / "experiments"
                / "RETEST_ENTRY.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
