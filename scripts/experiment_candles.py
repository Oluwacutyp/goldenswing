"""Phase-B runner: candle context map on 2y GC=F H1.

Runs goldenswing.experiments.candles.run_experiment(), prints the
signal x context table, and writes experiments/CANDLES.md with the
mined claims, the map, and the verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from goldenswing.experiments import candles as C  # noqa: E402


def _fmt_row(name: str, st: dict) -> str:
    pf = st["pf"]
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
    return (f"| {name} | {st['n']} | {st['win_rate']:.1f}% | "
            f"{st['avg_r']:+.3f} | {pf_s} |")


def main() -> None:
    res = C.run_experiment()
    outcomes = res["outcomes"]
    table = res["map"]
    print(f"setups={res['setups']} outcomes={len(outcomes)}")

    lines = []
    lines.append("# CANDLES — the how/where/when map")
    lines.append("")
    lines.append("Owner's question: \"we need more knowledge on this candle "
                 "stick stuff.\"  The standing rule applies: candles are not "
                 "good or bad — they are good or bad WHERE they fire.  This "
                 "document is Phase A (mined claims) + Phase B (measured map "
                 "on 2y GC=F H1, repo conventions: next-bar-open entry, "
                 "structural stop + 0.25 ATR, fixed 2R target, 72-bar horizon, "
                 "$0.45/oz).")
    lines.append("")
    lines.append("## Phase A — mined claims")
    lines.append("")
    lines.append("### Bulkowski (Encyclopedia of Candlestick Charts, via "
                 "thepatternsite.com)")
    lines.append("")
    lines.append("- Top reversal rates: three stars in the south 86%, bearish "
                 "three-line strike 84% (BUT only 85 samples in 4.7M candle "
                 "lines — Bulkowski himself warns the best-looking candles "
                 "are too rare to trade), three white soldiers 82%, identical "
                 "three crows 79%, **bearish engulfing 79%**, morning star "
                 "78%, three black crows 78%, morning doji star 76%, three "
                 "outside up 75%, evening star 72%.")
    lines.append("- **Bullish engulfing: 63% reversal but ranks 84/103 "
                 "overall** — it reverses, then the move fizzles. Best used "
                 "as a retrace inside a rising trend; do NOT trade it "
                 "against a primary downtrend.")
    lines.append("- Bearish harami: 53% — a coin flip. Doji: ~55% — a coin "
                 "flip.")
    lines.append("- **Reversal rate is not profitability.** A pattern can "
                 "\"work\" 63% of the time and still be a poor trade.")
    lines.append("- Pattern potency decays 3–5 bars after completion.")
    lines.append("")
    lines.append("### ICT / SMC practitioner claims")
    lines.append("")
    lines.append("- **Displacement candle**: body >= ~1.5x ATR, minimal "
                 "wicks, and it must break structure with the BODY — a "
                 "wick-only break is not a valid MSS.")
    lines.append("- Valid MSS = displacement + structure break + FVG "
                 "creation (all three).")
    lines.append("- **Rejection block** = a zone defined by WICKS (not "
                 "bodies) at a PD array — the long-wick candle IS the level.")
    lines.append("- **Indecision candle** (doji) marks zones; its significance "
                 "comes from location, not the candle alone.")
    lines.append("- **Mitigation blocks are one-touch**: after the first "
                 "retest fills the imbalance, the zone is consumed — do not "
                 "trade the 2nd/3rd approach.")
    lines.append("- **Breaker block body-fixation**: the breaking candle must "
                 "close its body past the extreme; on retest, a wick may "
                 "pierce to the 50% mean threshold but a body close past it "
                 "invalidates.")
    lines.append("- **Inducement gate**: the first pullback after BOS/CHoCH "
                 "must be swept + reversed + MSS-confirmed before any entry.")
    lines.append("- OB entry tiers: aggressive = touch, standard = reaction "
                 "at the 50% midline, conservative = micro-BoS inside the "
                 "zone.")
    lines.append("")
    lines.append("### Our own history (this repo)")
    lines.append("")
    lines.append("- Only **bearish pin bars at supply** carried edge "
                 "(+$1,755 / 31 trades); every bullish candle trigger lost.")
    lines.append("- JeaFx retest-entry experiment: rejection candles AFTER "
                 "confirmation buy a spent move — the BoS bar holds the "
                 "momentum.")
    lines.append("")
    lines.append("## Phase B — measured context map")
    lines.append("")
    lines.append(f"Signals scanned: {res['setups']}, trades taken: "
                 f"{len(outcomes)}.  Entry at next-bar open; stop = signal "
                 "extreme + 0.25 ATR; target fixed 2R (held constant so "
                 "candles are comparable).")
    lines.append("")
    for sig in C.SIGNALS:
        rows = table.get(sig)
        if not rows:
            lines.append(f"### {sig}: no qualifying trades")
            lines.append("")
            continue
        lines.append(f"### {sig}")
        lines.append("")
        lines.append("| cell | n | win% | avg R | PF |")
        lines.append("|---|---|---|---|---|")
        for name, st in rows.items():
            lines.append(_fmt_row(name, st))
        lines.append("")
    lines.append("## Verdict — which candles earn a place")
    lines.append("")
    earned = []
    for sig in C.SIGNALS:
        rows = table.get(sig) or {}
        ov = rows.get("overall", {})
        best_cell, best_r, best_n = "overall", ov.get("avg_r", -9), \
            ov.get("n", 0)
        for name, st in rows.items():
            if name == "overall":
                continue
            if st["n"] >= 30 and st["avg_r"] > best_r:
                best_cell, best_r, best_n = name, st["avg_r"], st["n"]
        verdict = " earns a place" if (
            (best_n >= 30 and best_r >= 0.10)
            or (ov.get("n", 0) >= 50 and ov.get("avg_r", -9) >= 0.05)
        ) else " does NOT earn a place"
        earned.append((sig, best_cell, best_r, best_n, verdict))
        lines.append(f"- **{sig}**: best cell `{best_cell}` "
                     f"(n={best_n}, avg R={best_r:+.3f}){verdict}.")
    lines.append("")
    out = REPO / "experiments" / "CANDLES.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    for sig, cell, r, n, v in earned:
        print(f"{sig}: {cell} n={n} avgR={r:+.3f}{v}")


if __name__ == "__main__":
    main()
