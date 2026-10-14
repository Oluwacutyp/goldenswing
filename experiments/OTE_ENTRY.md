# Experiment: OTE as the entry zone (claim #60)

**Claim (ICT MMXM, mined 2026-10-11):** after the manipulation leg (sweep + displacement), place a resting limit at the Optimal Trade Entry zone (62-79% fib retracement), confluent with the displacement order block — instead of chasing a confirmation trigger.

**Data:** GC=F H1, 11,426 bars, 2024-10-11 04:00:00+00:00 -> 2026-10-09 20:00:00+00:00 (1.99y). $0.45/oz cost, Phase-2 exits, 72-bar horizon.

**Universe:** 35 detector sweep_reversal signals @ >=60 (the confirmed sweep universe); 32 produced an OTE setup (fib zone + OB confluence + pullback side).

## Books (one position at a time, entry mechanic is the variable)

| Book | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe |
|---|---|---|---|---|---|---|---|
| Anchor (confirmation trigger) | 30 | 46.7 | 1.02 | +0.009 | +0.06% | 5.85% | 0.03 |
| Anchor on OTE setups only | 28 | 50.0 | 1.16 | +0.082 | +2.11% | 5.85% | 0.25 |
| OTE 62% limit | 20 | 40.0 | 0.84 | -0.100 | -2.12% | 6.41% | -0.25 |
| OTE 70% limit | 19 | 42.1 | 0.97 | -0.015 | -0.45% | 4.25% | -0.03 |
| OTE 79% limit | 11 | 27.3 | 0.62 | -0.285 | -3.20% | 4.16% | -0.44 |

## Shadow (every fill standalone)

| Slice | n | Win% | PF | Avg R |
|---|---|---|---|---|
| Anchor sweep slice | 35 | 51.4 | 1.21 | +0.105 |
| OTE 62% fills | 20 | 40.0 | 0.84 | -0.100 |
| OTE 70% fills | 20 | 45.0 | 1.09 | +0.053 |
| OTE 79% fills | 11 | 27.3 | 0.62 | -0.285 |

## Fill mechanics per fib level

| Fib | Setups | Fills | Fill rate | Expired/violated | Risk-band skip |
|---|---|---|---|---|---|
| 62% | 32 | 20 | 62.5% | 12 | 0 |
| 70% | 32 | 20 | 62.5% | 12 | 0 |
| 79% | 32 | 11 | 34.4% | 21 | 0 |

## The two questions the experiment was built to answer

**1. Fill-price improvement** (OTE fill vs the anchor's next-bar-open entry, in OTE-risk units; positive = OTE got the better price):

- 62%: +0.843R per filled trade
- 70%: +1.211R per filled trade
- 79%: +2.090R per filled trade

**2. Missed-trade cost** (what the anchor did with the setups the OTE limit never filled — the runners that never pulled back):

- 62%: 15 missed setups, avg +0.534R, total +8.0R
- 70%: 15 missed setups, avg +0.534R, total +8.0R
- 79%: 24 missed setups, avg +0.539R, total +12.9R

## Window halves (regime robustness)

| Book | Trades | Return | PF |
|---|---|---|---|
| Anchor H1 | 18 | +1.42% | 1.17 |
| Anchor H2 | 12 | -1.34% | 0.82 |
| OTE 62% H1 | 11 | -1.55% | 0.80 |
| OTE 62% H2 | 9 | -0.58% | 0.90 |
| OTE 70% H1 | 11 | -1.32% | 0.80 |
| OTE 70% H2 | 8 | +0.88% | 1.19 |
| OTE 79% H1 | 7 | -0.73% | 0.88 |
| OTE 79% H2 | 4 | -2.49% | 0.18 |

## Verdict

**KILLED as an entry-mechanic replacement — with a clean mechanism.**

The MMXM promise is real: the resting OTE limit gets dramatically better prices on fills (+0.84R at 62%, +1.21R at 70%, +2.09R at 79% vs the anchor's next-bar-open entry). But the runners that never pull back are where the money was: the setups OTE never filled averaged +0.53R for the anchor entry (+8.0R total at 62/70%, +12.9R at 79%). The missed winners cost more than the price improvement earns, at every fib level, in both window halves combined. Entry mechanic vs entry mechanic, same setups, same exits: resting at OTE loses to chasing the confirmation.

Secondary read: the 70% level is the best of the three (book -0.45%, shadow +0.053R) and the 79% level is the worst (book -3.20%, fill rate 34%) — deep pullbacks after displacement tend to keep going. The zone placement rhymes with the model, but best-of-three-losers is not an adoption.

What survives: nothing as an entry mechanic. The OTE zone keeps its existing role as a context gate (Fib/OTE family); this kill is about the resting-limit entry only.

## Honest limits

- One instrument (gold H1), one 2-year bullish window, in-sample.
- The OTE stop sits 0.25 ATR beyond the sweep extreme; the anchor's stop sits 0.15 + 0.25 = 0.40 ATR beyond it (detector invalidation + harness buffer). The 0.15 ATR delta is disclosed, not hidden.
- Pools ahead are the signal bar's pools for both variants.
- The MMXM model also uses SMT divergence and session filters we do not model; this tests the OTE entry mechanic alone.
