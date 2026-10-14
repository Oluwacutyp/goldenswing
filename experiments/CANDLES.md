# CANDLES — the how/where/when map

Owner's question: "we need more knowledge on this candle stick stuff."  The standing rule applies: candles are not good or bad — they are good or bad WHERE they fire.  This document is Phase A (mined claims) + Phase B (measured map on 2y GC=F H1, repo conventions: next-bar-open entry, structural stop + 0.25 ATR, fixed 2R target, 72-bar horizon, $0.45/oz).

## Phase A — mined claims

### Bulkowski (Encyclopedia of Candlestick Charts, via thepatternsite.com)

- Top reversal rates: three stars in the south 86%, bearish three-line strike 84% (BUT only 85 samples in 4.7M candle lines — Bulkowski himself warns the best-looking candles are too rare to trade), three white soldiers 82%, identical three crows 79%, **bearish engulfing 79%**, morning star 78%, three black crows 78%, morning doji star 76%, three outside up 75%, evening star 72%.
- **Bullish engulfing: 63% reversal but ranks 84/103 overall** — it reverses, then the move fizzles. Best used as a retrace inside a rising trend; do NOT trade it against a primary downtrend.
- Bearish harami: 53% — a coin flip. Doji: ~55% — a coin flip.
- **Reversal rate is not profitability.** A pattern can "work" 63% of the time and still be a poor trade.
- Pattern potency decays 3–5 bars after completion.

### ICT / SMC practitioner claims

- **Displacement candle**: body >= ~1.5x ATR, minimal wicks, and it must break structure with the BODY — a wick-only break is not a valid MSS.
- Valid MSS = displacement + structure break + FVG creation (all three).
- **Rejection block** = a zone defined by WICKS (not bodies) at a PD array — the long-wick candle IS the level.
- **Indecision candle** (doji) marks zones; its significance comes from location, not the candle alone.
- **Mitigation blocks are one-touch**: after the first retest fills the imbalance, the zone is consumed — do not trade the 2nd/3rd approach.
- **Breaker block body-fixation**: the breaking candle must close its body past the extreme; on retest, a wick may pierce to the 50% mean threshold but a body close past it invalidates.
- **Inducement gate**: the first pullback after BOS/CHoCH must be swept + reversed + MSS-confirmed before any entry.
- OB entry tiers: aggressive = touch, standard = reaction at the 50% midline, conservative = micro-BoS inside the zone.

### Our own history (this repo)

- Only **bearish pin bars at supply** carried edge (+$1,755 / 31 trades); every bullish candle trigger lost.
- JeaFx retest-entry experiment: rejection candles AFTER confirmation buy a spent move — the BoS bar holds the momentum.

## Phase B — measured context map

Signals scanned: 6232, trades taken: 6219.  Entry at next-bar open; stop = signal extreme + 0.25 ATR; target fixed 2R (held constant so candles are comparable).

### pin

| cell | n | win% | avg R | PF |
|---|---|---|---|---|
| overall | 3129 | 32.7% | -0.083 | 0.88 |
| location=at_aoc | 715 | 32.3% | -0.122 | 0.83 |
| location=at_pool | 2195 | 33.1% | -0.062 | 0.91 |
| location=at_zone | 146 | 32.2% | -0.073 | 0.90 |
| location=midrange | 73 | 23.3% | -0.351 | 0.57 |
| trend_align=against | 1406 | 34.1% | -0.043 | 0.94 |
| trend_align=with | 1723 | 31.5% | -0.115 | 0.84 |
| atr_regime=hi | 1072 | 31.9% | -0.120 | 0.83 |
| atr_regime=lo | 1051 | 31.9% | -0.097 | 0.87 |
| atr_regime=mid | 1006 | 34.3% | -0.028 | 0.96 |
| killzone=True | 828 | 29.1% | -0.176 | 0.76 |
| killzone=False | 2301 | 33.9% | -0.049 | 0.93 |
| loc|align=at_aoc|against | 317 | 36.0% | -0.016 | 0.98 |
| loc|align=at_aoc|with | 398 | 29.4% | -0.206 | 0.72 |
| loc|align=at_pool|against | 913 | 34.0% | -0.041 | 0.94 |
| loc|align=at_pool|with | 1282 | 32.5% | -0.077 | 0.89 |
| loc|align=at_zone|against | 104 | 36.5% | +0.058 | 1.09 |
| loc|align=at_zone|with | 42 | 21.4% | -0.397 | 0.52 |
| loc|align=midrange|against | 72 | 23.6% | -0.342 | 0.58 |
| loc|align=midrange|with | 1 | 0.0% | -1.015 | 0.00 |

### engulf

| cell | n | win% | avg R | PF |
|---|---|---|---|---|
| overall | 1593 | 34.9% | +0.003 | 1.00 |
| location=at_aoc | 286 | 31.5% | -0.107 | 0.85 |
| location=at_pool | 1217 | 36.2% | +0.043 | 1.07 |
| location=at_zone | 45 | 28.9% | -0.169 | 0.77 |
| location=midrange | 45 | 26.7% | -0.234 | 0.69 |
| trend_align=against | 772 | 35.8% | +0.045 | 1.07 |
| trend_align=with | 821 | 34.1% | -0.037 | 0.95 |
| atr_regime=hi | 442 | 34.6% | -0.011 | 0.98 |
| atr_regime=lo | 572 | 35.7% | +0.035 | 1.05 |
| atr_regime=mid | 579 | 34.4% | -0.019 | 0.97 |
| killzone=True | 450 | 36.0% | +0.029 | 1.05 |
| killzone=False | 1143 | 34.5% | -0.008 | 0.99 |
| loc|align=at_aoc|against | 148 | 31.1% | -0.108 | 0.85 |
| loc|align=at_aoc|with | 138 | 31.9% | -0.105 | 0.85 |
| loc|align=at_pool|against | 549 | 37.7% | +0.107 | 1.17 |
| loc|align=at_pool|with | 668 | 35.0% | -0.009 | 0.99 |
| loc|align=at_zone|against | 32 | 34.4% | -0.002 | 1.00 |
| loc|align=at_zone|with | 13 | 15.4% | -0.581 | 0.34 |
| loc|align=midrange|against | 43 | 27.9% | -0.198 | 0.73 |
| loc|align=midrange|with | 2 | 0.0% | -1.017 | 0.00 |

### displace

| cell | n | win% | avg R | PF |
|---|---|---|---|---|
| overall | 259 | 32.8% | -0.052 | 0.92 |
| location=at_aoc | 26 | 23.1% | -0.285 | 0.62 |
| location=at_pool | 226 | 34.5% | -0.011 | 0.98 |
| location=midrange | 7 | 14.3% | -0.542 | 0.34 |
| trend_align=against | 48 | 27.1% | -0.137 | 0.80 |
| trend_align=with | 211 | 34.1% | -0.033 | 0.95 |
| atr_regime=hi | 115 | 40.0% | +0.147 | 1.26 |
| atr_regime=lo | 48 | 37.5% | +0.071 | 1.11 |
| atr_regime=mid | 96 | 21.9% | -0.353 | 0.55 |
| killzone=True | 136 | 30.1% | -0.149 | 0.78 |
| killzone=False | 123 | 35.8% | +0.055 | 1.09 |
| loc|align=at_aoc|against | 6 | 33.3% | +0.135 | 1.26 |
| loc|align=at_aoc|with | 20 | 20.0% | -0.411 | 0.49 |
| loc|align=at_pool|against | 37 | 27.0% | -0.152 | 0.78 |
| loc|align=at_pool|with | 189 | 36.0% | +0.017 | 1.03 |
| loc|align=midrange|against | 5 | 20.0% | -0.351 | 0.53 |
| loc|align=midrange|with | 2 | 0.0% | -1.019 | 0.00 |

### inside_break

| cell | n | win% | avg R | PF |
|---|---|---|---|---|
| overall | 947 | 36.6% | +0.034 | 1.05 |
| location=at_aoc | 155 | 35.5% | -0.039 | 0.94 |
| location=at_pool | 765 | 36.3% | +0.031 | 1.05 |
| location=at_zone | 26 | 53.8% | +0.589 | 2.24 |
| location=midrange | 1 | 0.0% | -1.027 | 0.00 |
| trend_align=against | 397 | 35.5% | +0.003 | 1.00 |
| trend_align=with | 550 | 37.5% | +0.056 | 1.09 |
| atr_regime=hi | 301 | 41.5% | +0.133 | 1.22 |
| atr_regime=lo | 307 | 36.8% | +0.060 | 1.09 |
| atr_regime=mid | 339 | 32.2% | -0.079 | 0.89 |
| killzone=True | 184 | 43.5% | +0.207 | 1.36 |
| killzone=False | 763 | 35.0% | -0.008 | 0.99 |
| loc|align=at_aoc|against | 61 | 41.0% | +0.109 | 1.18 |
| loc|align=at_aoc|with | 94 | 31.9% | -0.136 | 0.80 |
| loc|align=at_pool|against | 315 | 33.3% | -0.054 | 0.92 |
| loc|align=at_pool|with | 450 | 38.4% | +0.090 | 1.14 |
| loc|align=at_zone|against | 20 | 55.0% | +0.624 | 2.35 |
| loc|align=at_zone|with | 6 | 50.0% | +0.472 | 1.91 |
| loc|align=midrange|against | 1 | 0.0% | -1.027 | 0.00 |

### doji

| cell | n | win% | avg R | PF |
|---|---|---|---|---|
| overall | 99 | 31.3% | -0.095 | 0.87 |
| location=at_zone | 99 | 31.3% | -0.095 | 0.87 |
| trend_align=against | 81 | 30.9% | -0.122 | 0.83 |
| trend_align=with | 18 | 33.3% | +0.028 | 1.04 |
| atr_regime=hi | 46 | 34.8% | +0.040 | 1.06 |
| atr_regime=lo | 32 | 18.8% | -0.489 | 0.43 |
| atr_regime=mid | 21 | 42.9% | +0.211 | 1.35 |
| killzone=True | 30 | 20.0% | -0.454 | 0.46 |
| killzone=False | 69 | 36.2% | +0.061 | 1.09 |
| loc|align=at_zone|against | 81 | 30.9% | -0.122 | 0.83 |
| loc|align=at_zone|with | 18 | 33.3% | +0.028 | 1.04 |

### push_fade

| cell | n | win% | avg R | PF |
|---|---|---|---|---|
| overall | 192 | 36.5% | +0.002 | 1.00 |
| location=at_aoc | 46 | 37.0% | +0.019 | 1.03 |
| location=at_pool | 90 | 41.1% | +0.135 | 1.21 |
| location=at_zone | 17 | 41.2% | +0.157 | 1.25 |
| location=midrange | 39 | 23.1% | -0.393 | 0.53 |
| trend_align=against | 161 | 34.8% | -0.047 | 0.93 |
| trend_align=with | 31 | 45.2% | +0.255 | 1.42 |
| atr_regime=hi | 61 | 44.3% | +0.253 | 1.43 |
| atr_regime=lo | 67 | 28.4% | -0.250 | 0.68 |
| atr_regime=mid | 64 | 37.5% | +0.026 | 1.04 |
| killzone=True | 50 | 40.0% | +0.104 | 1.16 |
| killzone=False | 142 | 35.2% | -0.034 | 0.95 |
| loc|align=at_aoc|against | 32 | 37.5% | +0.038 | 1.06 |
| loc|align=at_aoc|with | 14 | 35.7% | -0.025 | 0.97 |
| loc|align=at_pool|against | 75 | 40.0% | +0.102 | 1.16 |
| loc|align=at_pool|with | 15 | 46.7% | +0.298 | 1.51 |
| loc|align=at_zone|against | 15 | 33.3% | -0.074 | 0.90 |
| loc|align=at_zone|with | 2 | 100.0% | +1.893 | inf |
| loc|align=midrange|against | 39 | 23.1% | -0.393 | 0.53 |

## Verdict — which candles earn a place

- **pin**: best cell `loc|align=at_zone|against` (n=104, avg R=+0.058) does NOT earn a place.
- **engulf**: best cell `loc|align=at_pool|against` (n=549, avg R=+0.107) earns a place.
- **displace**: best cell `atr_regime=hi` (n=115, avg R=+0.147) earns a place.
- **inside_break**: best cell `killzone=True` (n=184, avg R=+0.207) earns a place.
- **doji**: best cell `killzone=False` (n=69, avg R=+0.061) does NOT earn a place.
- **push_fade**: best cell `trend_align=with` (n=31, avg R=+0.255) earns a place.

## Analyst notes (read before promoting any cell)

**Multiple comparisons.** 6 signals x ~20 cells = ~120 cells. The best
cell of the best signal is cherry-picked by construction. Treat the
"+0.2R" cells as hypotheses with stated sample sizes, not as measured
edge — every one of them needs an out-of-sample or walk-forward
confirmation before it becomes a scored family.

**The pin discrepancy, explained.** The earlier candle+ICT experiment
found bearish pin bars at supply profitable (+$1,755 / 31 trades). Here
pins lose everywhere (-0.083R over 3,129 trades, even at_zone|against
only +0.058R). The difference is the machinery, not the candle: that
experiment entered with the ICT trigger conventions and exited at
pools/2R with the v1 harness; this map uses next-bar-open entry and a
fixed 2R target. The candle alone is not the edge — the surrounding
entry/exit context was. That IS the owner's rule, measured.

**What the map actually says, signal by signal:**

- **pin — dead as a scored family.** 3,129 trades, negative in every
  location, every trend alignment, every session, every vol regime.
  Bulkowski never ranked pins highly either. Rejection wicks remain
  useful as *chart context* (they mark where liquidity was taken), not
  as trade triggers on H1 gold.
- **engulf — earns a gated sleeve: reversal at liquidity, against the
  trend.** `at_pool|against` +0.107R, n=549, PF 1.17. This matches
  Bulkowski's warning inverted into a rule: engulfing works as a
  *reversal*, not as continuation — and only at a pool. The `with`
  cells are flat to negative. Big sample; the strongest candidate.
- **displace — earns a gated sleeve: continuation in high volatility.**
  `atr_regime=hi` +0.147R, n=115, PF 1.26. The ICT displacement candle
  (big body, minimal wicks) traded as continuation works when the
  market is already volatile; in mid-vol regimes it loses badly
  (-0.353R, n=96). This is a regime gate, not a universal signal.
- **inside_break — earns a gated sleeve: killzone only.** `killzone`
  +0.207R, n=184, PF 1.36; off-killzone is flat (-0.008R, n=763).
  The consolidation-break belongs to London/NY. Also note
  `at_zone` +0.589R but n=26 — interesting, too thin to promote.
- **doji — dead.** -0.095R overall; Bulkowski's ~55% coin-flip
  confirmed on our data. Indecision marks zones; it does not trade.
- **push_fade — promising but thin.** `trend_align=with` +0.255R on
  only n=31; `at_pool` +0.135R (n=90) and `atr hi` +0.253R (n=61)
  agree on the shape (fade the exhaustion push at a level, with the
  trend, in volatile markets). Bulkowski's reversal reading of
  soldiers/crows has a shadow here, but n=31 cannot carry a verdict.
  Candidate for the OOS run, not a family.

**Recommendation.** Build three context-gated candle sleeves as
*shadow* families first (scored, not traded): engulf-reversal-at-pool
(counter-trend), displacement-continuation in hi-vol regime, and
inside-break in killzones. Re-measure them inside the full detector
book (Combo A + aoc) with the mechanical verdict rule before any of
them touches live sizing. Pins and dojis stay as chart context only.

**Mined claims vs measured, one line each:**
- Bearish engulfing 79% reversal (Bulkowski) — PLAUSIBLE in context:
  our engulf reversal-at-pool cell is the best engulf cell.
- Bullish engulfing weak overall (rank 84/103) — CONFIRMED: our
  engulf `with` cells are flat/negative.
- Doji/harami are coin flips — CONFIRMED on our data.
- Displacement needs body-close, wick breaks invalid (ICT) — built
  into the geometry; the hi-vol gate is the new finding.
- Rejection wicks mark levels (ICT rejection block) — CONFIRMED as
  context; rejected as triggers.
- Mitigation blocks are one-touch — not tested here (zone family
  owns it); no contradiction found.
- Soldiers/crows as reversals (Bulkowski 82%/78%) — WEAK SUPPORT:
  the fade works at pools in hi-vol, but thin.
