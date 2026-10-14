# IMBALANCE — FVG deep mining (Phase A mined, Phase B measured)

Owner: "imbalance too."  2,167 FVGs on 11,426 GC=F H1 bars (1,212 bullish/BISI,
955 bearish/SIBI).  Repo conventions: $0.45/oz, structural stops, causal geometry.

## Phase A — mined claims (practitioner sources, paraphrased)

1. **What an imbalance is**: a 3-candle displacement gap — candle 1's wick does
   not overlap candle 3's wick.  BISI = bullish gap (buyside imbalance, sellside
   inefficiency); SIBI = bearish.  The midpoint is Consequent Encroachment (CE);
   the far edge is IOFED.  Entries: at CE (deeper, tighter risk) or IOFED.
2. **Rebalance claim**: smart money displaces, leaves unfilled limit orders,
   then guides price back to "rebalance" the inefficiency; after rebalancing,
   price continues the original direction.
3. **IFVG**: a body close THROUGH the gap (not a wick poke) inverts it —
   support becomes resistance and vice versa.  Wick-through = liquidity sweep,
   gap intact.  Trade the retest in the inversion direction, stop beyond the
   far edge, target the next pool / 1:2.
4. **IFVG as signal**: sources split — some read it as continuation (in the
   displacement's direction after the flip), some as a shift/reversal idea.
5. **Entry stack** (ICT trade flow): HTF bias → premium/discount → bias-aligned
   FVG → retrace → LTF MSS → enter post-MSS retest at CE or IOFED, stop beyond
   the far edge or the MSS extreme.

## Phase B — measured

### Test 1: the magnet — do unfilled gaps get revisited?

| window | FVG fill rate | distance-matched placebo |
|---|---|---|
| 12 bars | 63.3% | — |
| 24 bars | 73.0% | 68.2% |
| 48 bars | 80.3% | — |
| 72 bars | 83.8% | 79.6% |
| 168 bars | 89.2% | 85.2% |

By size (gap/ATR): small 95.0% / mid 92.9% / large 85.0%.
By leg context: with-bias 89.5% / against-bias 92.8% / no-bias 98.9%.
First-touch outcomes (240-bar window): 1,091 wick holds, 881 inversions,
195 never touched.

**Verdict: WEAK CONFIRMATION.**  Gaps are revisited ~4–5pp more often than
distance-matched random zones — the magnet is real, but it is a gentle
pull, not the "price must return" of ICT lore.  (The naive placebo —
zones centered on the close — fills at 93% in 24 bars and would have
falsely "rejected" the claim; distance-matching is what makes the
control honest.)  Note the base rate is high because gold H1 mean-reverts
across days; the marginal magnetism is small.

### Test 2: CE (midline) vs edge entry at first wick touch

Naive entry at first touch, stop beyond far edge + 0.25 ATR, fixed 2R:

| entry | n | avg R | win rate | reached +1R |
|---|---|---|---|---|
| midline (CE) | 1,091 | **-0.379** | 0.238 | 0.657 |
| edge (IOFED) | 1,090 | **-0.191** | 0.294 | 0.685 |

**Verdict: BOTH LOSE unconditionally — the midline is not the edge.**
The adopted LuxAlgo entry (+0.205R shadow) fired midline limits only after
a FRESH CHoCH at an UNTRADED gap with structural stops.  Strip that context
away and the same midline loses -0.379R.  The candle's location is not the
trade; the surrounding confirmation was.  (Owner's rule, measured again.)
The edge beats the midline here — deeper entries sit closer to the stop
without better fills.

### Test 3: IFVG — drift after a body-close inversion

881 inversions.  Drift in the inversion direction:

| horizon | n | mean drift | hit rate | base abs drift |
|---|---|---|---|---|
| 12 bars | 880 | +0.056 ATR | 0.487 | 1.828 ATR |
| 24 bars | 878 | +0.291 ATR | 0.528 | 2.752 ATR |
| 48 bars | 876 | +0.290 ATR | 0.525 | 4.086 ATR |

**Verdict: NOISE.**  A +0.29 ATR drift over 24 bars at a 52.8% hit rate is
not a tradable signal — costs alone consume it.  The inversion marks a
momentum shift, but the shift has no exploitable follow-through on gold H1
at these horizons.  IFVG stays as chart context (it names the flip), not
as a trigger.

### Test 4: stacking — do stacked gaps hold better?

| zone | n | hold rate (wick, not inversion) |
|---|---|---|
| single gap | 1,645 | 0.550 |
| 2+ stacked gaps | 327 | 0.572 |

**Verdict: NO.**  +2.2pp on 327 samples is not a finding.  Stacking adds
nothing measurable.

## What earns a place

- **The magnet**: real but weak (+4–5pp).  Worth a small weight in level
  scoring (it is already implicit in AOC: zones that hold get revisited),
  not a standalone family.
- **The CE entry**: CONFIRMED as context-dependent only.  The adopted
  LuxAlgo sleeve stands exactly as specified (fresh CHoCH + untraded gap);
  do NOT generalize "enter at midlines" — the unconditional version loses.
- **IFVG**: chart context only.  Names the flip; does not trade.
- **Stacking**: nothing.  Killed.

## Honest limits

Single instrument, single bullish 2y window, H1 only.  The 881 inversions
are the largest sample in this study and still only support a "noise"
verdict.  Volume-imbalance (tick-volume confirmation of the displacement
leg) was mined but not separately tested — displacement volume is already
a Combo-A input via RVOL.
