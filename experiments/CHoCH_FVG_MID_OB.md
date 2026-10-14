# Experiment: CHoCH -> FVG-midline -> opposing-OB (user sleeve)

**Source:** user's own words from the LuxAlgo video: wait for a bullish CHoCH, enter at the midline of a bullish FVG, exit at the next opposing ('red') order block scaled by risk:reward.

**Data:** GC=F H1, 11426 bars, 2024-10-11 04:00:00+00:00 -> 2026-10-09 20:00:00+00:00 (1.99y). Cost $0.45/oz, 1% risk, 72-bar horizon, v1 exit conventions (stop wins ties, adverse opens).

## Funnel

- Setups scanned (fresh CHoCH + paired untraded FVG midline): 1211
- Resting limits filled: 393
- Cancelled (FVG died or 24-bar expiry, not trades): 818
- Filled but no qualifying opposing OB (>= 1R room): 16 (variant A only)

## Books (one position at a time, sequential)

| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | Avg hold |
|---|---|---|---|---|---|---|---|---|
| Sleeve A: FVG-mid + OB target | 89 | 37.1 | 1.32 | +0.225 | +19.85% | 12.43% | 0.71 | 12.3 |
| Sleeve B: FVG-mid + fixed 2R | 97 | 45.4 | 1.41 | +0.250 | +25.88% | 8.85% | 1.11 | 11.8 |
| Detector anchor (Combo A @60) | 199 | 45.2 | 1.22 | +0.124 | +25.62% | 11.04% | 0.91 | 19.5 |
| Detector CHoCH-adjacent slice (@60) | 111 | 43.2 | 1.21 | +0.120 | +12.96% | 9.54% | 0.62 | 20.9 |

## Shadow (every fill standalone)

| Slice | n | Win% | PF | Avg R |
|---|---|---|---|---|
| Sleeve A shadow | 377 | 35.3 | 1.58 | +0.403 |
| Sleeve B shadow | 393 | 43.0 | 1.34 | +0.205 |
| Detector CHoCH-adjacent shadow | 187 | 36.9 | 0.91 | -0.059 |
| Sleeve A longs | 149 | 46.3 | 2.39 | +0.781 |
| Sleeve A shorts | 228 | 28.1 | 1.20 | +0.156 |

## Window halves (sleeve books)

| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | Avg hold |
|---|---|---|---|---|---|---|---|---|
| Sleeve A first half | 39 | 38.5 | 1.43 | +0.284 | +10.81% | 7.21% | 0.85 | 11.9 |
| Sleeve A second half | 50 | 36.0 | 1.24 | +0.179 | +8.16% | 11.63% | 0.59 | 12.6 |
| Sleeve B first half | 42 | 47.6 | 1.60 | +0.334 | +14.49% | 5.30% | 1.44 | 12.7 |
| Sleeve B second half | 55 | 43.6 | 1.29 | +0.186 | +9.95% | 8.58% | 0.85 | 11.1 |

## Verdict

**ADOPT variant B as a candidate sleeve; variant A (the OB exit) does not
earn its place.**

What the numbers say:

1. **The user's entry is the real contribution.** Resting limits at the
   FVG midline after a fresh CHoCH grade far better than how the
   detector currently enters CHoCH-adjacent moves: sleeve shadow
   +0.205R (n=393) vs the detector's own CHoCH-adjacent shadow **-0.059R**
   (n=187). Same market events, better entry mechanics — the limit lets
   price come to the level instead of chasing the break.
2. **Variant B (fixed 2R) matches the anchor book** (+25.88% vs +25.62%)
   with **half the trades** (97 vs 199), better PF (1.41 vs 1.22),
   better Sharpe (1.11 vs 0.91) and lower drawdown (8.85% vs 11.04%).
   Positive in both halves (+14.49% / +9.95%). The sleeve's trades are
   largely a different set from the detector book (~1/3 coincide with a
   same-direction detector trade) — it diversifies rather than
   duplicates.
3. **The "red OB" exit dilutes.** Variant A books +19.85% vs +25.88%
   for the plain 2R exit, with a lower win rate (37.1% vs 45.4%). The
   farther OB targets win bigger when they hit (shadow +0.403R) but hit
   too rarely to carry the book. OB levels stay as chart context; they
   are not the exit rule.
4. **Longs carry it** (shadow A longs +0.781R, shorts +0.156R — both
   positive, unlike the detector's short problem, but the asymmetry is
   real and matches the bullish window).

Honest limits: one bullish 2-year window, in-sample for the design, no
score gate on the sleeve (it fishes a larger universe than the @60
book), 97 book trades. **Status: candidate sleeve for paper trading and
out-of-sample validation — not a live-money book.** The natural next
test is grafting the FVG-midline resting entry onto the detector as an
entry variant, scored and gated like everything else.
