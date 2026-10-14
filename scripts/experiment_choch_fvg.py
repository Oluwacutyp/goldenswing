"""Run the CHoCH -> FVG-midline -> opposing-OB sleeve experiment.

Produces the sleeve's two variants (OB target vs fixed 2R), the Combo-A
detector anchor book, and the detector's own CHoCH-adjacent slice, then
writes experiments/CHoCH_FVG_MID_OB.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.data import load_gold_h1, load_peers
from goldenswing.backtest.harness import (book_metrics, build_book,
                                          simulate_signal_v2, slice_stats)
from goldenswing.experiments.choch_fvg import (CHOCH_FRESH_BARS,
                                               find_fill, run_sleeve,
                                               scan_setups, select_ob_target,
                                               simulate_trade)
from goldenswing.features.engine import read_and_featurize
from goldenswing.golden.detector import GoldenSwingDetector
from goldenswing.modes.config import SWING
from goldenswing.reader.reader import MarketReader


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

    # ---- the sleeve ----------------------------------------------------
    res = run_sleeve()

    # halves for the sleeve books (regime-robustness read)
    states = MarketReader().read(df)
    setups = scan_setups(df, states)
    fills_a, fills_b = [], []
    for s in setups:
        fl = find_fill(df, s)
        if fl is None:
            continue
        fills_a.append(fl)
    outs_a, outs_b = [], []
    for fl in fills_a:
        from goldenswing.experiments.choch_fvg import FIXED_TP_R
        d = fl.setup.direction
        tgt = select_ob_target(df, states, fl)
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

    # ---- detector anchor: Combo-A swing book (must reproduce +25.62%) ---
    peers = load_peers()
    fstates = read_and_featurize(df, peers)
    det = GoldenSwingDetector(SWING.detector_config())
    sigs = det.scan(fstates)
    det_outs = [o for o in (simulate_signal_v2(df, s) for s in sigs)
                if o is not None]
    anchor = book_metrics(build_book(det_outs, 60.0), years)

    # ---- detector's own CHoCH-adjacent slice ----------------------------
    choch_outs = []
    for o in det_outs:
        st = fstates[o.signal.index]
        ev = st.last_structure_event
        if (ev is not None and ev.kind == "CHOCH"
                and ev.direction == o.signal.direction
                and st.structure_event_age is not None
                and st.structure_event_age <= CHOCH_FRESH_BARS):
            choch_outs.append(o)
    choch_shadow = slice_stats(choch_outs)
    choch_book = book_metrics(build_book(choch_outs, 60.0), years)

    # ---- direction split for the sleeve --------------------------------
    def dir_split(outcomes, d):
        sub = [o for o in outcomes if o.signal.direction == d]
        return slice_stats(sub), book_metrics(build_book(sub, 0.0), years)

    a_lng_s, a_lng_b = dir_split(outs_a, 1)
    a_sht_s, a_sht_b = dir_split(outs_a, -1)

    lines = []
    A = lines.append
    A("# Experiment: CHoCH -> FVG-midline -> opposing-OB (user sleeve)")
    A("")
    A("**Source:** user's own words from the LuxAlgo video: wait for a "
      "bullish CHoCH, enter at the midline of a bullish FVG, exit at the "
      "next opposing ('red') order block scaled by risk:reward.")
    A("")
    A(f"**Data:** GC=F H1, {res.bars} bars, {res.start} -> {res.end} "
      f"({res.years:.2f}y). Cost $0.45/oz, 1% risk, 72-bar horizon, "
      "v1 exit conventions (stop wins ties, adverse opens).")
    A("")
    A("## Funnel")
    A("")
    A(f"- Setups scanned (fresh CHoCH + paired untraded FVG midline): "
      f"{res.n_setups}")
    A(f"- Resting limits filled: {res.n_fills}")
    A(f"- Cancelled (FVG died or 24-bar expiry, not trades): "
      f"{res.n_cancelled}")
    A(f"- Filled but no qualifying opposing OB (>= 1R room): "
      f"{res.n_skipped_no_ob} (variant A only)")
    A("")
    A("## Books (one position at a time, sequential)")
    A("")
    A("| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | "
      "Avg hold |")
    A("|---|---|---|---|---|---|---|---|---|")
    A(_fmt_book("Sleeve A: FVG-mid + OB target", res.variant_a))
    A(_fmt_book("Sleeve B: FVG-mid + fixed 2R", res.variant_b))
    A(_fmt_book("Detector anchor (Combo A @60)", anchor))
    A(_fmt_book("Detector CHoCH-adjacent slice (@60)", choch_book))
    A("")
    A("## Shadow (every fill standalone)")
    A("")
    A("| Slice | n | Win% | PF | Avg R |")
    A("|---|---|---|---|---|")
    A(_fmt_shadow("Sleeve A shadow", res.shadow_a))
    A(_fmt_shadow("Sleeve B shadow", res.shadow_b))
    A(_fmt_shadow("Detector CHoCH-adjacent shadow", choch_shadow))
    A(_fmt_shadow("Sleeve A longs", a_lng_s))
    A(_fmt_shadow("Sleeve A shorts", a_sht_s))
    A("")
    A("## Window halves (sleeve books)")
    A("")
    A("| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | "
      "Avg hold |")
    A("|---|---|---|---|---|---|---|---|---|")
    A(_fmt_book("Sleeve A first half", a_h1))
    A(_fmt_book("Sleeve A second half", a_h2))
    A(_fmt_book("Sleeve B first half", b_h1))
    A(_fmt_book("Sleeve B second half", b_h2))
    A("")
    A("## Verdict")
    A("")
    A("_Filled in after the numbers land._")
    A("")
    out_path = (Path(__file__).resolve().parents[1] / "experiments"
                / "CHoCH_FVG_MID_OB.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"setups={res.n_setups} fills={res.n_fills} "
          f"cancelled={res.n_cancelled} no_ob={res.n_skipped_no_ob}")
    print(f"A: {res.variant_a['return_pct']:+.2f}% "
          f"PF={res.variant_a['pf']:.2f} n={res.variant_a['trades']}")
    print(f"B: {res.variant_b['return_pct']:+.2f}% "
          f"PF={res.variant_b['pf']:.2f} n={res.variant_b['trades']}")
    print(f"anchor: {anchor['return_pct']:+.2f}% n={anchor['trades']}")
    print(f"detector CHoCH slice book: {choch_book['return_pct']:+.2f}% "
          f"n={choch_book['trades']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
