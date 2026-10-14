# Experiment: JeaFx retest entry (sweep highs -> demand -> BoS -> retest)

**Source:** JeaFx short, narration: downtrend sweeps the highs, trades down into HTF demand, a strong push breaks structure up, enter on the retest of the demand level, stop beneath the low, target the next highs. Mirrored for shorts.

**Data:** GC=F H1, 11426 bars, 2024-10-11 04:00:00+00:00 -> 2026-10-09 20:00:00+00:00 (1.99y). Cost $0.45/oz, 1% risk, 72-bar horizon, v1 exit conventions (stop wins ties, adverse opens). Zones come from the repo's supply_demand family (fresh + daily-bias gated, the detector's zone convention).

**Entry modes:** `mid` = resting limit at the zone midline (the FVG-midline experiment's convention); `edge` = resting limit at the first-touched zone edge (top for longs / bottom for shorts — the literal reading of "bought when price retested that level").

## Entry mode: `mid`

### Funnel

- Setups scanned (fresh BoS + fresh demand/supply + failed opposite sweep): 143
- Resting limits filled: 20
- Fill rate: 14.0% — the retest-entry tradeoff
- Same-bar stops (fill bar also takes the stop): 15 (75.0% of fills)
- Cancelled (24-bar expiry, retest never came): 123
- Filled but no qualifying pool target (>= 1R room): 0 (variant A only)

### Books (one position at a time, sequential)

| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | Avg hold |
|---|---|---|---|---|---|---|---|---|
| Sleeve A (mid): retest + next-highs pool | 16 | 6.2 | 0.31 | -0.675 | -10.40% | 14.48% | -1.32 | 1.8 |
| Sleeve B (mid): retest + fixed 2R | 16 | 6.2 | 0.13 | -0.849 | -12.79% | 14.48% | -3.20 | 0.2 |

### Shadow (every fill standalone)

| Slice | n | Win% | PF | Avg R |
|---|---|---|---|---|
| Sleeve A (mid) shadow | 20 | 25.0 | 1.54 | +0.422 |
| Sleeve B (mid) shadow | 20 | 25.0 | 0.63 | -0.285 |
| Sleeve A (mid) longs | 9 | 55.6 | 5.87 | +2.213 |
| Sleeve A (mid) shorts | 11 | 0.0 | 0.00 | -1.042 |

### Window halves

| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | Avg hold |
|---|---|---|---|---|---|---|---|---|
| Sleeve A (mid) first half | 1 | 100.0 | inf | +4.765 | +4.76% | -0.00% | 0.00 | 29.0 |
| Sleeve A (mid) second half | 15 | 0.0 | 0.00 | -1.037 | -14.48% | 14.48% | -455.18 | 0.0 |
| Sleeve B (mid) first half | 1 | 100.0 | inf | +1.972 | +1.97% | -0.00% | 0.00 | 4.0 |
| Sleeve B (mid) second half | 15 | 0.0 | 0.00 | -1.037 | -14.48% | 14.48% | -455.18 | 0.0 |

## Entry mode: `edge`

### Funnel

- Setups scanned (fresh BoS + fresh demand/supply + failed opposite sweep): 445
- Resting limits filled: 73
- Fill rate: 16.4% — the retest-entry tradeoff
- Same-bar stops (fill bar also takes the stop): 15 (20.5% of fills)
- Cancelled (24-bar expiry, retest never came): 372
- Filled but no qualifying pool target (>= 1R room): 10 (variant A only)

### Books (one position at a time, sequential)

| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | Avg hold |
|---|---|---|---|---|---|---|---|---|
| Sleeve A (edge): retest + next-highs pool | 20 | 0.0 | 0.00 | -1.030 | -18.70% | 18.70% | -193.19 | 0.8 |
| Sleeve B (edge): retest + fixed 2R | 22 | 13.6 | 0.30 | -0.620 | -12.89% | 16.07% | -1.96 | 2.2 |

### Shadow (every fill standalone)

| Slice | n | Win% | PF | Avg R |
|---|---|---|---|---|
| Sleeve A (edge) shadow | 63 | 0.0 | 0.00 | -1.037 |
| Sleeve B (edge) shadow | 73 | 20.5 | 0.49 | -0.419 |
| Sleeve A (edge) longs | 34 | 0.0 | 0.00 | -1.047 |
| Sleeve A (edge) shorts | 29 | 0.0 | 0.00 | -1.026 |

### Window halves

| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | Avg hold |
|---|---|---|---|---|---|---|---|---|
| Sleeve A (edge) first half | 3 | 0.0 | 0.00 | -1.054 | -3.13% | 3.13% | -50.95 | 4.0 |
| Sleeve A (edge) second half | 17 | 0.0 | 0.00 | -1.025 | -16.07% | 16.07% | -647.50 | 0.3 |
| Sleeve B (edge) first half | 5 | 60.0 | 2.79 | +0.756 | +3.78% | 1.09% | 1.02 | 8.8 |
| Sleeve B (edge) second half | 17 | 0.0 | 0.00 | -1.025 | -16.07% | 16.07% | -647.50 | 0.3 |

## Reference books (committed reports)

| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | Avg hold |
|---|---|---|---|---|---|---|---|---|
| Combo A anchor @60 (VALIDATION.md) | 199 | 45.2 | 1.22 | +0.124 | +25.62% | 11.04% | 0.91 | 19.5 |
| CHoCH->FVG sleeve B @2R (CHoCH_FVG_MID_OB.md) | 97 | 45.4 | 1.41 | +0.250 | +25.88% | 8.85% | 1.11 | 11.8 |

## Verdict

**KILLED — the sleeve does not work on 2y of gold H1, in either entry
mode or either exit variant.** This is a clean, structural rejection,
not a sample-size artifact (edge mode: 73 fills, 63 pool-target trades
with a 0.0% win rate).

### What the numbers say

1. **Midline entry is geometrically broken.** 75% of fills are
   same-bar stops: average risk is only 0.78 ATR, so the fill bar's own
   range routinely covers both the midline limit and the stop. The
   midline was the FVG experiment's convention, not the video's — it
   does not transfer to zone geometry.
2. **Edge entry (the faithful reading) fixes the geometry but still
   loses.** Same-bar stops drop to 20%, fill rate rises to 16%, yet
   variant A books -18.7% with PF 0.00 — not one of 63 pool-target
   trades wins. Diagnosis: the nearest pool above sits **4.5R away on
   average** (median 4.5R, min 2.5R) — "the next highs" is unreachable
   within the 72-bar horizon. The fixed-2R control also loses
   (PF 0.30), so the exit is not the whole story.
3. **The setup itself grades negative.** A control that enters at the
   BoS confirmation close (no retest wait, same stop/target logic)
   shadows -0.105R at 38% win over 115 signal bars. Waiting for the
   retest is not the problem — the sweep→zone→BoS sequence as defined
   here carries no edge on this data. The BoS bar holds the momentum;
   the retest entry buys bars that are already reversing against the
   position.
4. **Fill rate confirms the retest-entry tradeoff is all cost, no
   benefit:** 14% (mid) / 16% (edge) of setups ever fill, and the
   fills lose.

### Why the FVG sleeve worked and this doesn't

The CHoCH→FVG-midline sleeve (+0.205R shadow) enters at a level
created *by the displacement leg itself* — inside the impulsive move,
before confirmation. The JeaFx sleeve enters *after* confirmation at a
separate structure (the supply/demand zone). Same "limit at a level"
mechanics, opposite geometry: one rides the impulse's own footprint,
the other chases a retest the market has already shown it will
reject.

### Honest limits

- Tested on gold H1; the video's market was EURUSD — the verdict is
  for GoldenSwing's market, not a universal claim about the video.
- One bullish 2-year window, in-sample design, 73 edge-mode fills.
- Nothing from this sleeve is adopted. The zone-retest entry stays
  OUT of the detector. No further variants are warranted: the
  setup-level control (entry at confirmation) already grades negative.

**Status: killed. Do not paper-trade this sleeve.**
