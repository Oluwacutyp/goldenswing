# GoldenSwing — Feature Layer (Phase 3)

**Data:** GC=F hourly, 11,426 bars, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00 (1.99 years). Same Phase-2 entries/exits/costs; the ONLY change per row is which feature families add score backing in the detector.

**RS peers:** dollar available, silver available. RS rows below are meaningful only for available peers.

## Baseline (Phase 2, features off) — book @ ≥60

199 trades · PF 1.16 · return +18.17% · maxDD 12.63% · shadow-@≥60 avg R +0.122 over 337 promoted signals.
This MUST reproduce BACKTEST2.md (+18.17%, PF 1.16, 199 trades) — the feature layer is flag-gated off here.

## Ablation — one family at a time, then all

"Promoted" = signals the config scores at >= 60 (the traded region); "Shadow @≥60" grades those signals standalone, without the sequential book lock — it measures the QUALITY of what a family promotes, which is where dilution shows up.

| Config | Signals | Promoted | Book trades | PF | Return | MaxDD | Book avg R | Shadow @≥60 avg R | Δ shadow R | Book@50 return | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase 2 baseline | 467 | 337 | 199 | 1.16 | +18.17% | 12.63% | +0.093 | +0.122 | +0.000 | +16.71% | — |
| bollinger | 467 | 341 | 199 | 1.16 | +17.23% | 11.38% | +0.089 | +0.111 | -0.011 | +16.71% | DILUTES |
| fib | 467 | 337 | 198 | 1.16 | +17.12% | 12.63% | +0.089 | +0.116 | -0.006 | +15.52% | DILUTES |
| ichimoku | 467 | 382 | 208 | 1.15 | +16.89% | 12.66% | +0.084 | +0.114 | -0.008 | +14.92% | DILUTES |
| relstr | 467 | 352 | 202 | 1.20 | +23.37% | 11.31% | +0.113 | +0.124 | +0.002 | +15.49% | ADDS |
| patterns | 467 | 383 | 212 | 1.03 | +1.72% | 16.15% | +0.017 | +0.086 | -0.036 | +15.19% | DILUTES |
| zones | 467 | 336 | 199 | 1.14 | +15.68% | 12.63% | +0.082 | +0.119 | -0.003 | +16.71% | DILUTES |
| phases | 467 | 342 | 199 | 1.18 | +19.61% | 12.67% | +0.099 | +0.131 | +0.009 | +15.53% | ADDS |
| footprints | 467 | 363 | 204 | 1.14 | +15.77% | 13.10% | +0.081 | +0.112 | -0.010 | +15.53% | DILUTES |
| volume | 467 | 336 | 193 | 1.21 | +23.00% | 11.96% | +0.117 | +0.128 | +0.006 | +16.41% | ADDS |
| mtf | 467 | 393 | 216 | 1.14 | +17.06% | 13.53% | +0.082 | +0.093 | -0.029 | +13.16% | DILUTES |
| ALL families | 467 | 433 | 218 | 1.12 | +13.65% | 15.27% | +0.068 | +0.083 | -0.038 | +10.57% | DILUTES |

Verdict rule: ADDS = book return up > 1pp without poisoning the promoted-signal grade (Δ shadow @≥60 >= -0.005R); DILUTES = book down > 1pp OR shadow @≥60 avg R down > 0.010R; else NEUTRAL.  A DILUTES family is flagged here, not silently kept: it should stay OFF in the detector config until re-designed.

## Caveats

- One instrument, one timeframe, one window (same in-sample window the Phase-2 design decisions were made on).  Ablation deltas are diagnostics, not out-of-sample proof.
- Score deltas per family are hand-set priors (see goldenswing/golden/detector.py `_feat_*`), not fitted — a family can fail because its WEIGHT is wrong, not because the feature carries no information.  The shadow grade helps tell those apart only partly.
- Feature families are correlated with the core families (e.g. MTF re-uses trend information); solo deltas do not sum to the ALL delta.

## Diagnosis (from the numbers above)

1. **Three families genuinely add: relstr, volume, phases.** Relative
   strength is the standout (+23.37% vs +18.17%, PF 1.20, MaxDD down)
   — and it is the only family bringing EXOGENOUS information (silver
   and dollar leadership) that the price-only reader cannot see.
   Volume is nearly free edge: fewer trades (193 vs 199), better ones
   (book avg R +0.117, PF 1.21) — confirming-flow and dead-volume
   penalties mostly *remove* bad trades. Phases add modestly
   (+19.61%, best shadow grade +0.131): the AMD base/manipulation
   context is worth a little, not a lot.
2. **Classical patterns are the worst diluter, by far** (+1.72%, PF
   1.03, shadow Δ -0.036R). As scored (+4 formed / +6 broken), H&S and
   double tops/bottoms promote 46 extra signals into the traded
   region and those signals are junk on H1 gold. The detector already
   reads swings/BOS/CHoCH directly; the pattern layer decorates
   entries instead of adding information. FLAGGED OFF — needs a
   redesign (e.g. score only neckline breaks WITH displacement), not
   a weight tweak.
3. **MTF dilutes by double-counting.** It promotes the most extra
   signals (393 vs 337) at the second-worst quality (Δ -0.029R):
   D1/H4 alignment is nearly the same information as the reader's
   trend state + the Phase-2 daily veto, so its +6 mostly re-promotes
   what trend already promoted. FLAGGED OFF as scored.
4. **Ichimoku is a trend proxy here** (cloud position ≈ trend state):
   45 extra promotions, slightly worse quality, book -1.3pp. No
   independent signal on H1 gold in this form. FLAGGED OFF.
5. **Footprints net negative despite the strongest single idea.** The
   institutional signature (+7) is conceptually the best backing in
   the layer, but the family's retail-chase penalty (-5) appears to
   punish valid late continuations — and continuations ARE the
   Phase-2 book. Book -2.4pp, shadow Δ -0.010R. FLAGGED OFF pending a
   redesign that scores the institutional leg alone.
6. **Zones and Fib are redundant, not wrong.** Both sit near-neutral
   on the shadow grade (Δ -0.003 / -0.006) but cost the book ~1-2.5pp
   through lock interactions: S/D zones are cousins of the reader's
   own order blocks, and the dealing-range Fib re-states the
   premium/discount discipline the regime score already applies.
   FLAGGED OFF at these weights.
7. **Bollinger is the borderline case.** Book -0.94pp, PF unchanged,
   but MaxDD IMPROVES (12.63% -> 11.38%) — its verdict trips on the
   shadow threshold by 0.001R. Honest read: NEUTRAL with a
   risk-reduction flavour. Kept OFF for the headline config; worth
   re-judging on risk-adjusted terms, not return alone.
8. **ALL families together are WORSE than baseline** (+13.65%):
   dilution compounds (433 promotions vs 337) because every family's
   bonus stacks on the same correlated trend information. The
   defensible Phase-3 detector config is the three ADDS families
   only — `frozenset({"relstr", "volume", "phases"})` — and that
   exact combo is NOT tested in this run; it is the next experiment,
   together with any out-of-sample window.
9. **Structural note:** the emitted signal count is 467 in EVERY row.
   At these weights (±2..8) family deltas almost never cross the
   35-point emit floor, so families act purely through the >=60 gate
   — they re-rank the traded region, they do not find new trades.
10. **AMD labelling caveat found in testing:** with the default
    48-bar range window and 48-bar prior-move lookback, bases only
    register after the prior move has left BOTH windows, so
    accumulation/distribution labels are rarer than the Wyckoff
    story implies (consolidation dominates). A shorter range window
    relative to the prior lookback (e.g. 20/40) labels more bases;
    left as config, not silently changed.
