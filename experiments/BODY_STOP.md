# Experiment: body-anchored stops for sweep reversals

**Source:** Faiz SMC's ICT turtle soup video (user-mined 2026-10-11, claim #63). The claim: anchor the stop at the sweep candle's BODY (not the wick extreme) -- deliberately tighter, accepting more stop-outs for better R:R. Our convention is wick extreme + 0.15 ATR invalidation + 0.25 ATR harness buffer.

**Data:** GC=F H1, 11426 bars, 2024-10-11 04:00:00+00:00 -> 2026-10-09 20:00:00+00:00 (1.99y). $0.45/oz cost, 72-bar horizon, Phase-2 exit machinery (50% partial at first pool >= 1R; trail after +1.5R). Entries and exits held constant -- this is a pure stop-placement A/B.

**Variants** (identical sweep universe, identical displacement+CHoCH trigger within 8 bars, identical gates evaluated against each variant's own invalidation; NO score gate):
- **A (ours):** invalidation at the sweep wick extreme -/+ 0.15 ATR; stop = invalidation -/+ 0.25 ATR buffer (total 0.40 ATR beyond the extreme).
- **B (theirs):** invalidation at the sweep candle's body extreme (min(open,close) for longs, max(open,close) for shorts) -/+ 0.15 ATR; same 0.25 ATR buffer. The anchor point is the ONLY change.
- **B2 (pure body):** invalidation AT the body extreme exactly, zero buffer anywhere (stop_buffer_atr=0.0).

## Anchor validation

The real detector (Phase-2 config, emit floor 35) emits **49** sweep-reversal signals on this window. Variant A fires on **34** sweeps; **34** of its trigger bars coincide with a detector signal bar.

## Shadow (every trigger standalone)

| Variant | n | Win% | PF | Avg R |
|---|---|---|---|---|
| A | 34 | 44.1% | 0.93 | -0.041 |
| B | 38 | 39.5% | 0.81 | -0.119 |
| B2 | 37 | 32.4% | 0.69 | -0.217 |

## Books (one position at a time, sequential)

| Variant | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | Avg hold |
|---|---|---|---|---|---|---|---|---|
| A | 33 | 45.5% | 0.98 | -0.011 | -0.60% | 6.80% | -0.04 | 30.9 |
| B | 37 | 40.5% | 0.84 | -0.095 | -3.69% | 7.24% | -0.34 | 21.0 |
| B2 | 37 | 32.4% | 0.69 | -0.217 | -7.99% | 12.75% | -0.73 | 18.1 |

## Window halves (book)

| Variant | Half | Trades | Win% | PF | Avg R | Return |
|---|---|---|---|---|---|---|
| A | H1 | 19 | 47.4% | 1.05 | +0.027 | +0.39% |
| A | H2 | 14 | 42.9% | 0.89 | -0.064 | -0.99% |
| B | H1 | 21 | 42.9% | 0.94 | -0.034 | -0.86% |
| B | H2 | 16 | 37.5% | 0.72 | -0.174 | -2.85% |
| B2 | H1 | 20 | 35.0% | 0.82 | -0.125 | -2.64% |
| B2 | H2 | 17 | 29.4% | 0.55 | -0.325 | -5.49% |

## Exit mix (the stop-out question)

`stop_rate` = share of trades exiting at the initial stop. `wicked_inv_rate` = share of trades where price crossed the RAW invalidation at some point (for A: how often the body anchor was violated but the extreme stop survived).

| Variant | n | Stop | Trail | Timeout | Stop-out rate | Wicked-inv rate |
|---|---|---|---|---|---|---|
| A | 34 | 19 | 11 | 4 | 55.9% | 55.9% |
| B | 38 | 24 | 13 | 1 | 63.2% | 68.4% |
| B2 | 37 | 26 | 10 | 1 | 70.3% | 67.6% |

## Verdict

**KILLED -- the tighter stop's R:R improvement does not pay for its
higher stop-out rate.** The degradation is monotonic with tightness:
A -0.60% -> B -3.69% -> B2 -7.99%, and both body variants are negative
in both window halves. Stop-outs rise 55.9% -> 63.2% -> 70.3% while
average hold collapses 30.9 -> 21.0 -> 18.1 bars -- the tighter stop
does not just lose more trades, it strangles the survivors before the
trail can work.

**Mechanism:** the sweep wick extreme is where the liquidity was taken;
price routinely retests into the wick zone (the second-touch / stop-hunt
retest) without invalidating the reversal. The body-anchored stop sits
inside that noise zone: B's wicked-inv rate is 68.4% (vs 55.9% for A),
meaning two-thirds of its trades see price cross the body anchor. The
author's "deliberately tighter" is deliberately donating on gold H1.
B also takes 4 extra marginal trades (38 vs 34 hits) because the
tighter stop slides under the stop band's max gate -- both effects point
the wrong way.

**Status: killed as a stop convention.** The wick-extreme + 0.15 ATR
invalidation + 0.25 ATR buffer stands. (Scope note per the placement
doctrine: this is gold H1, all sessions; a session-scoped variant was
not tested.)

### Honest limits

- One instrument (gold H1), one 2-year bullish window, in-sample design.
- The tighter stop changes R itself (R is measured on the buffered stop), so avg-R comparisons across variants are indicative; the book return is the bottom line.
- Entry at next-bar-open (harness convention).
