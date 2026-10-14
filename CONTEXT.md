# GoldenSwing — Context Gates + Trendlines (Phase 4)

**Owner's rule under test:** *the problem might not be the fib and the rest but how they are used, where they are used and when they are used.*  Phase 3 applied the families flat and six diluted.  Phase 4 re-tests them as context gates (silent outside their context) plus a new trendline family.

**Data:** GC=F hourly, 11,426 bars, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00. Same Phase-2 entries/exits/costs as Phases 3/3.5; the ONLY change per row is the detector's feature-family set.

**Anchors (must reproduce):** Phase-2 baseline +18.17% / PF 1.16 / 199 trades (BACKTEST2: +18.17% / 1.16 / 199) · Combo A +25.62% / PF 1.22 / 199 trades (VALIDATION: +25.62% / 1.22 / 199).

## The gates (what "used properly" means, per family)

Each gate returns NOTHING outside its context — no bonus, no penalty.
Full rationale lives in `goldenswing/features/gating.py`.

| Family | Speaks ONLY when… | Rationale |
|---|---|---|
| Fib / OTE (`fib_ctx`) | price is retracing INTO the OTE zone (0.62–0.79) of a confirmed displacement leg, the leg broke structure in the trade direction, momentum is no longer extending (the pullback is in progress), structure has not flipped against, and the HTF (daily or MTF D1) bias agrees | Fib is a pullback-entry tool, not a level oracle — the zone means something only while a real leg is being retraced with the HTF behind it |
| Zones (`zones_ctx`) | the zone is FRESH (untested), the daily bias agrees with trading it, price is at/near it, and the approach is slowing (|momentum| ≤ 1 ATR, no knife-through displacement against) | a tested zone is spent; a fresh zone against the daily bias is a coin flip; how price ARRIVES at a level is half the signal |
| Ichimoku (`ichimoku_ctx`) | regime vote comes from the DAILY cloud only (price above/below it); the H1 cloud confirms entries ONLY in an expansion phase — never in consolidation/bases; inside the D1 cloud the family is silent | the cloud is a regime instrument first; on H1 inside a base it is chopped noise, which is why the flat version read as a trend proxy and diluted |
| Patterns (`patterns_ctx`) | the pattern stands at a MAJOR level (liquidity pool / daily or HTF swing), after an extended move (dealing range ≥ 5 ATR, height ≥ 1.5 ATR), AND the neckline has BROKEN with displacement (or a confirming structure event) | pattern alone never scores — location + trigger is the trade; mid-range, unbroken patterns are trivia |
| Footprints (`footprints_ctx`) | the institutional signature counts only POST-SWEEP AT A LEVEL (OB/FVG at hand, or the sweep level itself is major); the retail-chase caution applies only at climax extremes (climax volume + range extreme); CTA is NOT scored (its information is the trend the core already owns) | size entering through a flushed level is the story — the same signature mid-range is not; and chasing is only fatal at extremes, not in every strong continuation |
| MTF (`mtf_ctx`) | CONFIRM: +3 only when D1 AND H4 both align. VETO: when both oppose, the score is CAPPED at 59 — visible, but never a Golden Swing | MTF is a veto/confirm instrument, not a points fountain; Phase 3's additive version re-bought trend points and diluted by double-counting |
| Trendlines (`trendlines`) | BOUNCE at a validated line (≥ 1 retest — the 3rd contact) scores only inside trend context; a line BREAK scores only WITH displacement (a structure event), for/against accordingly; riding a validated line in trend earns a small proximity credit | two points are a guess, the retest validates; a break without displacement is a wick story |

### Trendline lifecycle (features/trendlines.py)

Support = through two HIGHER confirmed swing lows; resistance =
through two LOWER confirmed swing highs.  A line is known at the
second anchor's confirmation bar (causal).  States: FORMING (2
anchors) → ACTIVE (first retest touch; touches re-arm only after
price spends bars away) → BROKEN (close through by > 0.2 ATR —
terminal, an event on that bar) / EXPIRED (age > 300 bars or drift
> 6 ATR — silent).  A touch whose close rejects the line by
≥ 0.25 ATR is a BOUNCE.  Near-parallel duplicates are suppressed at
creation (price within 0.25 ATR and slope within 25% — the older line
owns the geometry).  Creation (builder) and lifecycle (manager) are
separate passes, mirroring the MQL5 builder/manager split.

## Ablation — Combo A is the baseline to beat

"Promoted" / "Shadow @≥60" as in FEATURES.md: the shadow grade measures the quality of what a config promotes into the traded region, where dilution shows first.

| Config | Signals | Promoted | Book trades | PF | Return | MaxDD | Sharpe | Shadow @≥60 avg R | Δ shadow R | Book@50 return | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Combo A {relstr, volume, phases} | 467 | 352 | 199 | 1.22 | +25.62% | 11.04% | 0.91 | +0.124 | +0.000 | +14.00% | — |
| A + fib_ctx | 467 | 352 | 199 | 1.22 | +25.62% | 11.04% | 0.91 | +0.124 | +0.000 | +14.00% | NEUTRAL |
| A + zones_ctx | 467 | 352 | 199 | 1.22 | +25.62% | 11.04% | 0.91 | +0.124 | +0.000 | +14.00% | NEUTRAL |
| A + ichimoku_ctx | 467 | 381 | 209 | 1.26 | +32.60% | 11.76% | 1.07 | +0.118 | -0.005 | +13.11% | NEUTRAL |
| A + patterns_ctx | 465 | 395 | 208 | 1.16 | +19.04% | 14.43% | 0.68 | +0.107 | -0.016 | +6.67% | DILUTES |
| A + footprints_ctx | 467 | 351 | 199 | 1.22 | +25.62% | 11.04% | 0.91 | +0.120 | -0.003 | +14.00% | NEUTRAL |
| A + mtf_ctx | 467 | 376 | 207 | 1.26 | +31.54% | 11.78% | 1.04 | +0.117 | -0.006 | +13.16% | NEUTRAL |
| A + trendlines | 467 | 358 | 203 | 1.17 | +18.69% | 12.81% | 0.70 | +0.101 | -0.022 | +14.00% | DILUTES |
| FULL gated stack (A + all gates + trendlines) | 467 | 413 | 214 | 1.11 | +12.04% | 14.43% | 0.47 | +0.108 | -0.015 | +9.32% | DILUTES |

Verdict rule (same as Phase 3, judged against Combo A): ADDS = book return up > 1pp without poisoning the promoted grade (Δ shadow @≥60 >= -0.005R); DILUTES = book down > 1pp OR shadow @≥60 avg R down > 0.010R; else NEUTRAL.  A gated family that still dilutes stays OFF — recorded here, not silently kept.

## Reading of the results (hand-written, post-run)

Gate firing counts over the 467 emitted signals (diagnostic run):
fib_ctx spoke on 5 signals, zones_ctx on **0**, ichimoku_ctx on 376,
patterns_ctx appended 911 factor hits, footprints_ctx on 8, mtf_ctx
confirmed on 329 and veto-capped 3, trendline events on 116.

1. **The owner's rule is PARTIALLY vindicated.** Context changed
   what families do: Ichimoku went from flat DILUTES (+16.89%, worse
   than the Phase-2 book it decorated) to the best book in this table
   when gated (+32.60%, PF 1.26, Sharpe 1.07 on top of Combo A);
   patterns improved by ~17pp over their flat form (+1.72% →
   +19.04%); footprints' diagnosed harm (the chase penalty fighting
   valid continuations) is gone (−2.4pp → 0.0pp).  How/where/when a
   tool is used is demonstrably not a detail.
2. **Ichimoku_ctx is the phase's real candidate — and the rule
   still says NEUTRAL.** Its Δ shadow is −0.005R, a hair past the
   ADDS bound (≥ −0.005): the gate re-ranks the sequential book
   favourably (regime context removes bad trades) while the
   standalone grade of what it promotes is fractionally worse.  The
   mechanical verdict is NOT overridden by hand.  Status: candidate
   for an out-of-sample run, NOT enabled.
3. **MTF_ctx is the second candidate, same status.** Book +31.54%
   (PF 1.26) with Δ shadow −0.006R → NEUTRAL by one thousandth of R.
   The +3 confirm fires often (329 signals); the veto capped only 3.
   Attribution between "veto removed a few disasters" and
   "re-ranking luck" is not clean at these counts.
4. **Fib_ctx and zones_ctx were not rescued — they were muted.**
   Fib's joint conditions (OTE + structure-broken leg + stalled
   momentum + HTF agreement) coincide on 5 of 467 signals and
   changed nothing; the zones gate (fresh + daily-aligned + at-zone
   + slowing arrival, all on the signal bar) fired ZERO times.  A
   gate that never speaks is neither proven nor disproven — it is
   unevaluated.  Both stay OFF; a redesign would judge zone/fib
   context on the approach bars, not only the signal bar.
5. **Patterns_ctx still DILUTES.** Location + trigger helped
   enormously vs flat, but the gate promotes 43 extra signals at
   Δ shadow −0.016R.  The "major level" test is too permissive on
   H1 — liquidity pools are dense enough that almost any pattern is
   within 0.6 ATR of one.  Stays OFF.
6. **Trendlines DILUTE as score backing.** The events are real
   (116 firings, lifecycle verified by unit tests) but at these
   weights they promote worse trades (+18.69%, Δ shadow −0.022R).
   The family keeps its value as *state* — chart/EA objects in the
   owner's MQL5 sense — but it is OFF in the detector config.
   Stated plainly: gating did not rescue trendline scoring.
7. **The full gated stack DILUTES (+12.04%).** Even gated,
   stacking everything compounds the patterns/trendline promotions
   faster than the ichimoku/mtf gains.  There is no configuration in
   this phase that beats enabling candidates selectively — and none
   is enabled on in-sample evidence alone.

**Recommended config after Phase 4: unchanged — Combo A
{relstr, volume, phases}.**  Candidates for the next out-of-sample
run, in order: A + ichimoku_ctx, A + mtf_ctx.  Everything else in
this phase stays OFF.

## Caveats

- Same single in-sample window as Phases 3/3.5 (one bullish gold regime); gated verdicts are diagnostics, not out-of-sample proof.  The OOS instruments inherit whatever config this phase recommends and must be re-run before any deployment claim.
- Gate conditions are hand-set trader rules, not fitted parameters — a gate can fail because a condition is wrong (too strict: never fires; too loose: flat behaviour returns), not because the family's information is worthless.  The promoted-count column shows how often each gate actually spoke.
- The MTF veto cap (59) acts on the score AFTER family deltas; a capped signal still appears in the shadow book at its capped score.

---

## Phase 4.5 (hand-written, 2026-10-11) — AOC: area of concentration

*Not produced by `scripts/backtest4.py`; regenerating this file will
drop this section.  The canonical record is `AOC.md`.*

**Trader rationale (the owner's mined concept):** an *area of
concentration* — where several independent level families (pool,
zone, order block, trendline, Fib, daily swing) agree inside ~0.35
ATR — is a pivot zone.  The system had all the levels; Phase 4.5
(`goldenswing/features/aoc.py`) measures their concentration:
cluster score = Σ weight × recency, ≥2 families for an AOC, ≥3 for
a MAJOR pivot zone.

Two results:

1. **Gate repair — FAILED.** The patterns gate's "major level" test
   was rebuilt on AOC pivot zones (the old pool-proximity test was
   too permissive).  `A + patterns_ctx` = +19.08%, still DILUTES
   (was +19.04%).  The repair was faithful — 91.5% of the required
   major zones contain a hard level — so the verdict is about
   patterns, not the location test.  Patterns stay OFF.  The
   Phase-4 table above keeps the old numbers as history.
2. **New family — ADDS.** `A + aoc` (+4 entry inside a
   direction-aligned concentration, −4 for a concentration between
   entry and the first pool target, silent otherwise) = **+28.92%**,
   PF 1.25, Sharpe 1.00, MaxDD 10.38%, Δ shadow +0.011R — ADDS by
   the mechanical rule.

**Recommended config after Phase 4.5: Combo A + aoc =
{relstr, volume, phases, aoc}.** In-sample; OOS confirmation still
owed.
