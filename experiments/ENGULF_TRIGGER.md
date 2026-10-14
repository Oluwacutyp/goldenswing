# Experiment: engulfing-close as the sweep-reversal trigger

**Source:** Pro Trading School "Tsutsumi" video (user-mined 2026-10-11). The claim: for sweep reversals, enter at the CLOSE of the engulfing candle instead of on displacement + CHoCH/BOS confirmation. Never trade engulfings mid-range -- only at edges/zones.

**Data:** GC=F H1, 11426 bars, 2024-10-11 04:00:00+00:00 -> 2026-10-09 20:00:00+00:00 (1.99y). $0.45/oz cost, 72-bar horizon, Phase-2 exit machinery (stop = sweep extreme +/- 0.15 ATR, buffered 0.25 ATR; 50% partial at first pool >= 1R; trail after +1.5R).

**Variants** (identical gates: daily veto, room-to-pool, stop-distance band; NO score gate -- scores are a separate system, this isolates the trigger):
- **A (ours):** sweep -> displacement + CHoCH/BOS confirmation within 8 bars (the detector's exact `_sweep_confirmed`), first trigger bar per sweep.
- **B (theirs):** sweep -> first engulfing candle (body engulfs prior body) closing in the reversal direction within 8 bars, at a pool/zone/AOC (never mid-range), same gates.
- **C (confluence):** first bar where BOTH the engulfing-at-level and the displacement + CHoCH confirmation hold.
Entry is next-bar-open for all three (harness convention; the video says enter at the engulfing close -- on H1 the close->open gap is noise and the shared harness keeps the comparison about the trigger).

## Anchor validation

The real detector (Phase-2 config, emit floor 35) emits **49** sweep-reversal signals on this window (BACKTEST2 shadow slice: 49). Variant A (per-sweep first-trigger, no score gate) fires on **34** sweeps; **34** of its trigger bars coincide with a detector signal bar. The difference is dedupe (one opportunity per sweep here vs the detector's min-gap re-emission), not the trigger definition.

## Shadow (every trigger standalone)

| Variant | n | Win% | PF | Avg R |
|---|---|---|---|---|
| A | 34 | 44.1% | 0.93 | -0.041 |
| B | 101 | 37.6% | 0.67 | -0.259 |
| C | 11 | 27.3% | 0.44 | -0.411 |

## Books (one position at a time, sequential)

| Variant | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | Avg hold |
|---|---|---|---|---|---|---|---|---|
| A | 33 | 45.5% | 0.98 | -0.011 | -0.60% | 6.80% | -0.04 | 30.9 |
| B | 95 | 37.9% | 0.69 | -0.243 | -21.94% | 32.12% | -0.90 | 14.3 |
| C | 11 | 27.3% | 0.44 | -0.411 | -4.48% | 7.81% | -0.91 | 24.3 |

## Window halves (book)

| Variant | Half | Trades | Win% | PF | Avg R | Return |
|---|---|---|---|---|---|---|
| A | H1 | 19 | 47.4% | 1.05 | +0.027 | +0.39% |
| A | H2 | 14 | 42.9% | 0.89 | -0.064 | -0.99% |
| B | H1 | 46 | 34.8% | 0.57 | -0.399 | -17.83% |
| B | H2 | 49 | 40.8% | 0.86 | -0.096 | -5.00% |
| C | H1 | 6 | 50.0% | 1.17 | +0.087 | +0.48% |
| C | H2 | 5 | 0.0% | 0.00 | -1.008 | -4.94% |

## Timing (sweeps that fired in both variants)

| Pair | n | Mean lag A (bars) | Mean lag X (bars) |
|---|---|---|---|
| A vs B | 14 | 3.71 | 3.93 |
| A vs C | 11 | 3.27 | 4.82 |

## Verdict

**KILLED — the engulfing close does not beat displacement + CHoCH/BOS
confirmation as the sweep-reversal trigger, on 2y of gold H1.** This is a
clean rejection, not a sample artifact (B: 101 outcomes, negative in
both halves, MaxDD 32%).

### What the numbers say

1. **B loses decisively: -0.259R shadow, -21.94% book** (95 trades, PF
   0.69), negative in both halves (-17.83% / -5.00%). The video's own
   "never mid-range" rule was enforced and did not save it.
2. **B is not earlier, just looser.** On the 14 sweeps where both fired,
   mean trigger lag is 3.93 bars (B) vs 3.71 (A) -- the engulfing buys
   no timing advantage. It fires 3.3x as often (114 vs 34 hits): most
   post-sweep engulfings at levels are noise, not reversals.
3. **C (confluence) adds nothing:** requiring the engulfing on top of
   confirmation shrinks the sample to 11 and stays negative (-0.411R).
   The engulfing contributes no information beyond what confirmation
   already selected.
4. **A validates the anchor exactly:** all 34 of variant A's trigger
   bars coincide with a real detector sweep-reversal signal bar
   (detector emits 49 with min-gap re-emission; the 15-signal gap is
   dedupe, not definition). Without the score gate, the confirmed-sweep
   slice is ~breakeven (-0.011R book) -- the detector's positive slice
   (+0.024R on 23 @>=60 trades) comes from the score gate on top of
   the trigger, a separate already-validated system.

### Why the video's logic fails here

An engulfing candle is a single-bar sentiment print; displacement +
CHoCH is a two-leg structural event (impulse + break of structure).
After a liquidity sweep, plenty of bars print engulfing shapes at
levels as price chops -- the trigger has no structure filter beyond
location, and location alone does not discriminate. This rhymes with
the candle map's finding (engulf-reversal-at-pool +0.107R as a
*standalone next-bar signal*): the engulfing carries weak signal as a
raw pattern, but as a *replacement* for confirmation inside the full
sweep-reversal machinery it dilutes decisively.

**Status: killed as a trigger. Do not adopt.** The engulfing keeps its
existing status from CANDLES.md (gated standalone sleeve, needs OOS).

### Honest limits

- One instrument (gold H1), one 2-year bullish window, in-sample design.
- The engulfing trigger is evaluated on H1 bars; the video's examples are lower timeframes -- the verdict is for GoldenSwing's market and timeframe.
- Entry at next-bar-open (harness convention), not literally at the engulfing close as the video states.
