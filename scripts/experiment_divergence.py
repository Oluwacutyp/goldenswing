"""Run the RSI-divergence experiment and print the report tables."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.data import load_gold_h1, load_peers
from goldenswing.backtest.validation import COMBO_A
from goldenswing.experiments import divergence as D


def _book_row(label: str, res) -> str:
    b = res.book60
    return (f"| {label} | {res.n_signals} | {b['trades']} | "
            f"{b['win_rate']:.1f} | {b['pf']:.2f} | {b['return_pct']:+.2f}% | "
            f"{b['max_dd_pct']:.2f}% | {b['sharpe']:.2f} | "
            f"{b['avg_r']:+.3f} | {res.shadow60['avg_r']:+.3f} |")


def _ctx_row(label: str, s: dict) -> str:
    return (f"| {label:16s} | {s['n']:4d} | {s['avg_r']:+.3f} | "
            f"{s['win']:.3f} | {s['pf']:.2f} |")


def main() -> None:
    df = load_gold_h1()
    print(f"bars: {len(df)}  {df.index[0]} -> {df.index[-1]}")
    peers = load_peers()
    print(f"peers: {sorted(peers) if peers else None}")

    variants = {
        "rsi14_drsi2": D.featurize(df, peers, rsi_period=14, min_drsi=2.0),
        "rsi9_drsi2": D.featurize(df, peers, rsi_period=9, min_drsi=2.0),
        "rsi14_drsi1": D.featurize(df, peers, rsi_period=14, min_drsi=1.0),
        "rsi14_drsi4": D.featurize(df, peers, rsi_period=14, min_drsi=4.0),
    }
    for name, states in variants.items():
        n_div = sum(len(s.features.divergence.divergences) for s in states)
        print(f"census {name}: active-divergence bar-slots={n_div} "
              f"{D.divergence_census(states)}")

    print("\n== ROLE 1: scored family (anchor = COMBO_A + aoc) ==")
    print("| config | signals | trades | win% | PF | return | MaxDD | "
          "Sharpe | avgR | shadow60 |")
    anchor, test14 = D.run_family_test(df, variants["rsi14_drsi2"])
    print(_book_row("anchor", anchor))
    print(_book_row("divergence rsi14", test14))
    _, test9 = D.run_family_test(df, variants["rsi9_drsi2"])
    print(_book_row("divergence rsi9", test9))
    for name, states in (("rsi14", variants["rsi14_drsi2"]),
                         ("rsi9", variants["rsi9_drsi2"])):
        hits, n = D.gate_firing_rate(df, states)
        print(f"gate firing {name}: {hits}/{n} signals carried a "
              f"divergence factor")

    print("\n== ROLE 2: entry context (anchor-signal shadow R) ==")
    print("| bucket | n | avgR | win | PF |")
    outcomes = D.anchor_outcomes(df, variants["rsi14_drsi2"])
    print(f"anchor signals simulated: {len(outcomes)}")
    for name, states in variants.items():
        print(f"-- {name} --")
        for bucket, s in D.context_table(states, outcomes).items():
            print(_ctx_row(bucket, s))


if __name__ == "__main__":
    main()
