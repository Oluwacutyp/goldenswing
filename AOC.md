# GoldenSwing — AOC: Area of Concentration (Phase 4.5)

**Mined from a trading comment section (2026-10-11):** *"AOC — AREA OF CONCENTRATION, or your pivot zone: support & resistance levels."* Where several independent level families agree within a tight band, price treats the area as a pivot.  The system had all the levels; it never measured their concentration.

## Concentration score

Per bar, the family harvests every live level from confirmed state — liquidity pools, supply/demand zones, order blocks, FVGs, trendline values at the bar, Fib levels, daily swings, the dealing-range EQ — clusters levels within 0.35 ATR (cluster span capped at 1.0 ATR), and scores each cluster as **sum of weight x recency** over members:

- pool 0.8 + 0.2 x strength · sd_zone 1.0 fresh / 0.6 tested · ob 1.0 · fvg 0.7 · trendline 1.0 ACTIVE / 0.6 FORMING · fib 0.7 · daily swing 1.2 · EQ 0.5
- recency = 0.5 + 0.5 x 2^(-age / 96): fresh counts double, ancient levels never drop below half weight
- an AOC needs **2 distinct families**; a **MAJOR** pivot zone needs **3** (families dedupe inside a cluster)

Causality: clusters are pure functions of the bar's confirmed MarketState + FeatureSet.  One subtlety found while building: the reader mutates level objects in place while per-bar lists are snapshots — list *membership* is the causal truth, so the harvester trusts the lists and never re-checks the mutable flags (re-checking read future mutations and failed the truncation test; documented in `features/aoc.py`).

## The two tests

1. **Gate repair** — Phase-4's patterns gate still diluted (+19.04%) because its "major level" test (pool proximity) was too permissive on H1.  The gate now requires the pattern's neckline/extreme to sit inside an AOC pivot zone (>= 3 families).
2. **New family** — `aoc_gate`: +4 when the entry sits inside a direction-aligned concentration (>= 2 families), -4 when a concentration sits strictly between the entry and the first pool target.  Silent otherwise.

**Data:** GC=F hourly, 11,426 bars, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00. Same Phase-2 entries/exits/costs; the ONLY change per row is the detector's feature-family set.

**Anchors (must reproduce):** Phase-2 baseline +18.17% / PF 1.16 / 199 trades (BACKTEST2: +18.17% / 1.16 / 199) · Combo A +25.62% / PF 1.22 / 199 trades (VALIDATION: +25.62% / 1.22 / 199).

## Ablation — Combo A is the baseline to beat

| Config | Signals | Promoted | Book trades | PF | Return | MaxDD | Sharpe | Shadow @≥60 avg R | Δ shadow R | Book@50 return | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase 2 (anchor) | 467 | 337 | 199 | 1.16 | +18.17% | 12.63% | 0.68 | +0.122 | -0.002 | +16.71% | — |
| Combo A {relstr, volume, phases} | 467 | 352 | 199 | 1.22 | +25.62% | 11.04% | 0.91 | +0.124 | +0.000 | +14.00% | — |
| A + aoc | 467 | 351 | 199 | 1.25 | +28.92% | 10.38% | 1.00 | +0.134 | +0.011 | +14.00% | ADDS |
| A + patterns_ctx (repaired) | 467 | 383 | 210 | 1.16 | +19.08% | 14.19% | 0.69 | +0.114 | -0.009 | +11.04% | DILUTES |
| A + patterns_ctx + aoc | 467 | 381 | 208 | 1.18 | +21.53% | 14.19% | 0.76 | +0.120 | -0.003 | +11.04% | DILUTES |

Gate firing diagnostics (over the emitted signals): aoc_gate contributed on **139** of 467 signals; the repaired patterns gate fired on **265** signals.

Verdict rule (same as Phases 3/4, judged against Combo A): ADDS = book return up > 1pp without poisoning the promoted grade (Δ shadow @≥60 >= -0.005R); DILUTES = book down > 1pp OR shadow @≥60 avg R down > 0.010R; else NEUTRAL.

## Honest verdict

**AOC as a scored family EARNS its place.** `A + aoc` = **+28.92%**
(PF 1.25, Sharpe 1.00, MaxDD 10.38%) vs Combo A +25.62% — the book
is up 3.3pp, the promoted grade is up (+0.011R shadow @≥60), and
drawdown fell.  The gate fired on 139 of 467 signals: the entry
credit (+4 inside a direction-aligned concentration) and the
overhead-headwind penalty (-4 for a concentration between entry and
the first pool target) re-rank the traded region for the better.
By the project's mechanical rule this is ADDS, and the rule is not
overridden by hand.

**The patterns-gate repair FAILED.** `A + patterns_ctx` with the
AOC pivot-zone major test = **+19.08%**, still DILUTES — essentially
identical to the old pool-proximity test (+19.04%).  The repair was
faithful (91.5% of the major zones it requires contain a hard
level — pool, zone, OB or daily swing), so the verdict is about
patterns, not about the location test: broken-neckline classical
patterns do not add on H1 gold under either definition of "major
level".  Patterns stay OFF.

**Recommended config after Phase 4.5: Combo A + aoc =
{relstr, volume, phases, aoc}** (+28.92% / PF 1.25 / Sharpe 1.00).
Standard caveat, unchanged from every prior phase: this is
in-sample on one bullish gold window.  The config earns the
recommendation by the mechanical rule; out-of-sample confirmation
is still owed before any deployment claim.  The gated ichimoku_ctx
and mtf_ctx candidates from Phase 4 remain queued behind it.

## Caveats

- Same single in-sample window as Phases 3–4 (one bullish gold regime); verdicts are diagnostics, not out-of-sample proof.
- Gate weights (+4/-4) and the 3-family major threshold are hand-set trader priors, not fitted parameters.
- The repaired patterns gate changes Phase-4's documented patterns_ctx row (+19.04% was the OLD pool-proximity test); CONTEXT.md keeps the Phase-4 numbers as history, this file is the repair's record.
