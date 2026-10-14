"""Run the three trendline-claim experiments and print report tables."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.experiments import trendline_claims as T


def _q_row(label: str, q: dict) -> str:
    return (f"| {label:8s} | {q['n_lines']:5d} | {q['n_touches']:6d} | "
            f"{q['bounce_rate']:.3f} | {q['mean_adverse_atr']:+.3f} | "
            f"{q['median_adverse_atr']:+.3f} | {q['failed_touch_rate']:.3f} |")


def _ctx_row(label: str, s: dict) -> str:
    return (f"| {label:10s} | {s['n']:4d} | {s['avg_r']:+.3f} | "
            f"{s['win_rate']:5.1f} | {s['pf']:.2f} |")


def _book_row(label: str, b: dict) -> str:
    return (f"| {label:12s} | {b['trades']:4d} | {b['win_rate']:5.1f} | "
            f"{b['pf']:.2f} | {b['return_pct']:+7.2f}% | "
            f"{b['max_dd_pct']:5.2f}% | {b['sharpe']:+.2f} | "
            f"{b['avg_r']:+.3f} |")


def _shadow_row(label: str, s: dict) -> str:
    return (f"| {label:12s} | {s['n']:4d} | {s['win_rate']:5.1f} | "
            f"{s['pf']:.2f} | {'':9s} | {'':7s} | {'':6s} | "
            f"{s['avg_r']:+.3f} |")


def main() -> None:
    res = T.run_experiment()
    print(f"bars: {res['bars']}  {res['start']} -> {res['end']}")

    a = res["anchor"].book60
    print("\n== ANCHOR (COMBO_A + aoc @ >=60) ==")
    print(_book_row("anchor", a))
    ok = (abs(a["return_pct"] - 28.92) < 0.05 and a["trades"] == 199
          and abs(a["pf"] - 1.25) < 0.01)
    print("anchor reproduces:", "YES" if ok else "NO — STOP, CHECK FEED")

    c29 = res["c29"]
    print("\n== CLAIM #29a: touch quality (shadow vs body anchors) ==")
    print("| conv.    | lines | touches | bounce | meanAdv | medAdv | "
          "failed |")
    print(_q_row("shadow", c29["quality_shadow"]))
    print(_q_row("body", c29["quality_body"]))
    print("\n== CLAIM #29b: entry context "
          f"(n_outcomes={c29['n_outcomes']}) ==")
    print("| bucket     | n    | avgR   | win%  | PF   |")
    for label, ctx in (("sh:touch", c29["context_shadow"]["touch"]),
                       ("sh:no_touch", c29["context_shadow"]["no_touch"]),
                       ("bd:touch", c29["context_body"]["touch"]),
                       ("bd:no_touch", c29["context_body"]["no_touch"])):
        print(_ctx_row(label, ctx))

    c30 = res["c30"]
    print("\n== CLAIM #30: wedges as continuation ==")
    print(f"wedges: {c30['n_wedges']} {c30['wedge_types']}  "
          f"breakouts: {c30['n_breakouts']}")
    br = c30["breakouts"]
    kinds: dict[str, list[float]] = {}
    for b in br:
        kinds.setdefault(b["kind"], []).append(b["drift_r"])
    import numpy as np
    print("| group              | n    | meanDrift | win%  |")
    for k, v in kinds.items():
        arr = np.array(v)
        print(f"| {k:18s} | {len(v):4d} | {arr.mean():+.3f}     | "
              f"{np.mean(arr > 0) * 100:5.1f} |")
    base = c30["base_rate"]
    print(f"| {'base_rate':18s} | {base['n']:4d} | "
          f"{base['mean_drift']:+.3f}     | {base['win_rate']:5.1f} |")
    bk = c30["book"]
    print("\n| book         | trds | win%  | PF   | return  | MaxDD | "
          "Sharpe | avgR   |")
    print(_shadow_row("shadow", bk["shadow"]))
    print(_book_row("book", bk["book"]))
    print(_book_row("book_h1", bk["book_h1"]))
    print(_book_row("book_h2", bk["book_h2"]))

    c31 = res["c31"]
    print(f"\n== CLAIM #31: trendline breakout-retest "
          f"(n_signals={c31['n_signals']}) ==")
    print("| book         | trds | win%  | PF   | return  | MaxDD | "
          "Sharpe | avgR   |")
    print(_shadow_row("shadow", c31["shadow"]))
    print(_book_row("book", c31["book"]))
    print(_book_row("book_h1", c31["book_h1"]))
    print(_book_row("book_h2", c31["book_h2"]))
    print("exit mix:", c31["exit_mix"])


if __name__ == "__main__":
    main()
