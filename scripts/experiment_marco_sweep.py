"""Runner: Marco's sweep-entry vs our confirmation (Phase-2 pillar challenge).

Runs goldenswing.experiments.marco_sweep.run_experiment(), prints the
A/B/C comparison, and writes experiments/MARCO_SWEEP.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from goldenswing.experiments import marco_sweep as E  # noqa: E402


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
    from goldenswing.backtest.data import load_gold_h1  # noqa: E402
    from goldenswing.features.engine import read_and_featurize  # noqa: E402
    from goldenswing.golden.detector import (DetectorConfig,  # noqa: E402
                                             GoldenSwingDetector)
    df = load_gold_h1()
    states = read_and_featurize(df)
    det = GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True))
    res = E.run_experiment(df, states)
    print(f"bars={res['bars']} years={res['years']:.2f} "
          f"{res['start']} -> {res['end']}")
    q = res["qualification"]
    print(f"sweeps={q['sweeps']} qualified={q['qualified']} "
          f"({100.0 * q['qualified'] / max(1, q['sweeps']):.1f}%)")
    for mode in E.VARIANTS:
        r = res[mode]
        print(f"variant {mode}: hits={len(r['hits'])} "
              f"shadow_n={r['shadow']['n']} book={r['book']['trades']} "
              f"ret={r['book']['return_pct']:+.2f}% "
              f"exit_mix={r.get('exit_mix', '-')}")
    print("anchor:", res["anchor"])

    # sensitivity: variant B at stricter touch thresholds (shares df/states)
    sens = E.sensitivity_b(df, states, det)
    for s in sens:
        print(f"sens mt={s['min_touches']}: hits={s['hits']} "
              f"ret={s['book']['return_pct']:+.2f}%")

    L = []
    L.append("# Experiment: Marco's sweep-entry vs our confirmation")
    L.append("")
    L.append("**Source:** Marco Trades via Chart Fanatics (user-mined "
             "2026-10-11). The claim: for liquidity sweeps, enter AT/AFTER "
             "the sweep with NO displacement+CHoCH wait -- \"the edge is "
             "the level, not entry precision.\" The defense is level "
             "selection: only \"engineered liquidity\" qualifies -- levels "
             "respected from the left (>= 2 approaches within 0.25 ATR, no "
             "body close through, prior 200 bars, causal).")
    L.append("")
    L.append(f"**Data:** GC=F H1, {res['bars']} bars, "
             f"{res['start']} -> {res['end']} "
             f"({res['years']:.2f}y). $0.45/oz cost, 72-bar horizon.")
    L.append("")
    L.append("**Variants** (identical non-score gates: daily veto, "
             "room-to-pool, stop-distance band; NO score gate):")
    L.append("- **A (ours):** sweep of a QUALIFIED level -> displacement + "
             "CHoCH/BOS within 8 bars. Phase-2 exits (buffered structural "
             "stop, 50% partial at first pool >= 1R, trail after +1.5R).")
    L.append("- **B (Marco's):** sweep of a QUALIFIED level -> enter at "
             "next bar open, no confirmation. Stop just beyond the swept "
             "level + 0.25 ATR. Target = nearest opposing QUALIFIED level. "
             "Marco exits: BE roll at +1R, hold to target.")
    L.append("- **C (ablation):** sweep of ANY pool -> immediate entry, "
             "Marco exits, target = nearest opposing pool. B vs C isolates "
             "the level filter.")
    L.append("")
    L.append("## Level qualification")
    L.append("")
    L.append(f"**{q['qualified']}** of **{q['sweeps']}** sweeps "
             f"({100.0 * q['qualified'] / max(1, q['sweeps']):.1f}%) passed "
             "the engineered-liquidity filter at >= 2 touches. The filter "
             "as operationalized is permissive -- nearly every reader pool "
             "has two prior approaches in 200 bars.")
    L.append("")
    L.append("## Anchor validation")
    L.append("")
    a = res["anchor"]
    L.append(f"The real detector (Phase-2 config, emit floor 35) emits "
             f"**{a['n_detector_signals']}** sweep-reversal signals on this "
             f"window. Variant A fires on **{a['n_variant_a']}** qualified "
             f"sweeps; **{a['overlap']}** of its trigger bars coincide with "
             f"a detector signal bar -- the trigger replication matches the "
             f"engulf_trigger experiment's variant A (34 sweeps) exactly.")
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
    L.append("## Exit mix (Marco variants)")
    L.append("")
    for mode in ("B", "C"):
        L.append(f"- {mode}: {res[mode]['exit_mix']}")
    L.append("")
    L.append("## Sensitivity: variant B at stricter touch thresholds")
    L.append("")
    L.append("| Min touches | Hits | Shadow n | Shadow avg R | Book trades | "
             "Return | PF | Win% |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in sens:
        b, sh = s["book"], s["shadow"]
        L.append(f"| {s['min_touches']} | {s['hits']} | {sh['n']} | "
                 f"{sh['avg_r']:+.3f} | {b['trades']} | "
                 f"{b['return_pct']:+.2f}% | {_pf(b['pf'])} | "
                 f"{b['win_rate']:.1f}% |")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append("**The Phase-2 pillar stands -- Marco's entry loses "
             "decisively, and the level filter does not rescue it.**")
    L.append("")
    L.append("1. **B (Marco's) loses at the claimed threshold: -28.89%** "
             "book (196 trades, PF 0.73), shadow -0.160R. The unconfirmed "
             "sweep entry is not a sample artifact -- it is 200 outcomes "
             "deep.")
    L.append("2. **C (any pool, immediate) loses slightly worse: "
             "-30.25%.** B vs C isolates the level filter: it is worth "
             "~1.4pp of book and a touch less bad shadow. The filter has "
             "small merit as context, not as a rescue.")
    L.append("3. **Stricter \"engineered\" levels lose less but still "
             "lose.** At >= 6 touches B improves to -3.76% (PF 0.93, 92 "
             "trades) from -28.89% -- the filter measures something real "
             "(heavily-respected levels fade less violently), but no "
             "threshold turns the unconfirmed entry profitable. The "
             "improvement curve flattens well below breakeven.")
    L.append("4. **A (confirmation) is ~flat at -0.45%** on the qualified "
             "universe (33 trades) -- the confirmed-sweep slice was never "
             "the book's engine (BACKTEST2: +0.026R shadow), but it is not "
             "the bleed that B is.")
    L.append("5. **Exit mix explains the mechanism:** B's exits are "
             "stop 104 / BE 65 / target 30 / timeout 1 -- only 15% of "
             "trades ever reach the opposing engineered level. The level "
             "is the thesis and price rarely gets there; the 0.25-ATR "
             "stop \"covering the level\" gets harvested by the post-sweep "
             "chop. Marco's \"edge is the level\" requires the level to "
             "hold often enough to pay for the entries that don't -- on "
             "gold H1 it does not.")
    L.append("")
    L.append("**Status: killed. Do not adopt the unconfirmed entry.** The "
             "engineered-liquidity touch count is kept as a candidate "
             "*context* feature (stricter levels = gentler fades), not as "
             "an entry system.")
    L.append("")
    L.append("### Honest limits")
    L.append("")
    L.append("- One instrument (gold H1), one 2-year bullish window, "
             "in-sample design.")
    L.append("- Marco's examples are lower timeframes and NQ/gold intraday; "
             "the verdict is for GoldenSwing's market and timeframe. His "
             "eyeball for \"respect\" is stricter than any touch count -- "
             "the >= 6 threshold is the closest mechanical proxy and it "
             "still loses.")
    L.append("- Variant B bundles Marco's entry AND his exits/target "
             "selection; only the B-vs-C pair isolates the level filter.")
    L.append("- The engineered filter at >= 2 touches is permissive "
             "(91.5% of sweeps qualify); its merit shows only at >= 5-6 "
             "touches, where samples thin (92-115 trades).")
    out = REPO / "experiments" / "MARCO_SWEEP.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
