# Experiment: sweep-candle body-break confirmation vs our pillar

**Source:** SMC Liquidity Sweep Scalper (RanaAli878, MQL5 CodeBase, user-mined 2026-10-11, claim #49). The claim: after a liquidity sweep, wait N bars for a confirmation break back through THE SWEEP CANDLE'S BODY and enter at market. The third confirmation definition for the identical sweep setup -- ours is displacement+CHoCH/BOS (the Phase-2 pillar), the Tsutsumi video's was the engulfing close (killed, #27).

**Data:** GC=F H1, 11426 bars, 2024-10-11 04:00:00+00:00 -> 2026-10-09 20:00:00+00:00 (1.99y). $0.45/oz cost, 72-bar horizon, Phase-2 exit machinery (stop = sweep extreme +/- 0.15 ATR, buffered 0.25 ATR; 50% partial at first pool >= 1R; trail after +1.5R).

**Variants** (identical sweep universe, identical gates: daily veto, room-to-pool, stop-distance band; NO score gate -- this isolates the confirmation definition):
- **A (ours):** sweep -> displacement + CHoCH/BOS within 8 bars (the detector's exact `_sweep_confirmed`), first trigger per sweep.
- **B5 / B10 (theirs):** sweep -> first bar within 5 / 10 bars whose CLOSE crosses back through the sweep candle's body in the reversal direction, same gates.
- **C (theirs + quality gates):** B10 plus the author's stated gates -- confirmation candle body > 0.5x range (momentum, not drift) and stop >= 0.5 ATR (meaningful vs ATR).
Entry is next-bar-open for all variants (harness convention; the author enters at market on the trigger bar -- on H1 the close->open gap is noise and the shared harness keeps the comparison about the confirmation, not the fill).

## Anchor validation

The real detector (Phase-2 config, emit floor 35) emits **49** sweep-reversal signals on this window (BACKTEST2 shadow slice: 49). Variant A (per-sweep first-trigger, no score gate) fires on **34** sweeps; **34** of its trigger bars coincide with a detector signal bar. The difference is dedupe (one opportunity per sweep here vs the detector's min-gap re-emission), not the trigger definition.

## Shadow (every trigger standalone)

| Variant | n | Win% | PF | Avg R |
|---|---|---|---|---|
| A | 34 | 44.1% | 0.93 | -0.041 |
| B5 | 174 | 37.4% | 0.81 | -0.121 |
| B10 | 197 | 38.6% | 0.85 | -0.093 |
| C | 150 | 38.0% | 0.84 | -0.105 |

## Books (one position at a time, sequential)

| Variant | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | Avg hold |
|---|---|---|---|---|---|---|---|---|
| A | 33 | 45.5% | 0.98 | -0.011 | -0.60% | 6.80% | -0.04 | 30.9 |
| B5 | 151 | 37.1% | 0.81 | -0.123 | -17.94% | 21.00% | -0.84 | 18.5 |
| B10 | 168 | 39.3% | 0.87 | -0.084 | -14.24% | 18.17% | -0.61 | 20.5 |
| C | 130 | 40.0% | 0.90 | -0.064 | -8.90% | 16.89% | -0.41 | 21.8 |

## Window halves (book)

| Variant | Half | Trades | Win% | PF | Avg R | Return |
|---|---|---|---|---|---|---|
| A | H1 | 19 | 47.4% | 1.05 | +0.027 | +0.39% |
| A | H2 | 14 | 42.9% | 0.89 | -0.064 | -0.99% |
| B5 | H1 | 83 | 36.1% | 0.84 | -0.108 | -9.24% |
| B5 | H2 | 68 | 38.2% | 0.78 | -0.140 | -9.59% |
| B10 | H1 | 89 | 39.3% | 0.87 | -0.081 | -7.64% |
| B10 | H2 | 79 | 39.2% | 0.86 | -0.086 | -7.15% |
| C | H1 | 67 | 41.8% | 1.04 | +0.022 | +0.90% |
| C | H2 | 63 | 38.1% | 0.76 | -0.155 | -9.71% |

## Timing (sweeps that fired in both variants)

| Pair | n | Mean lag A (bars) | Mean lag X (bars) |
|---|---|---|---|
| A vs B5 | 25 | 3.64 | 2.36 |
| A vs B10 | 30 | 4.23 | 3.17 |
| A vs C | 26 | 4.15 | 3.31 |

## Verdict

**KILLED — the Phase-2 pillar stands, with stronger evidence.**

1. **B5/B10 lose decisively at both windows** (−17.94% / −14.24% book, PF
   0.81/0.87), negative in both halves, ~150–170 trades deep — not a
   sample artifact. The body-break confirmation is not a cheaper version
   of our confirmation; it is a worse one.
2. **C's quality gates halve the damage** (−8.90%) but do not rescue it:
   H1 prints +0.90% (PF 1.04) while H2 prints −9.71% (PF 0.76). The
   momentum/stop gates measure something real (momentum quality matters),
   but no gate turns the definition profitable.
3. **B is genuinely earlier** — mean lag 2.4–3.3 bars vs A's 3.6–4.2 on
   shared sweeps (unlike the engulfing trigger, which had no timing
   advantage). The timing edge is real and still insufficient: B fires
   5–6x as often, and the extra trades are chop.
4. **The confirmation definition matters.** This answers the experiment's
   core question: the pillar is not "any post-sweep confirmation." A
   single-bar close back through the sweep candle's body lets in the
   post-sweep chop; displacement + CHoCH/BOS is a two-leg structural
   event that filters it. −0.60% vs −14% to −18% on the identical sweep
   universe, identical gates, identical exits.

**Status: killed. Do not adopt the body-break confirmation.** The author's
momentum/min-stop quality gates are noted as a ~5pp improvement on a
losing base — worth remembering as a trigger-quality dimension, not as
a confirmation definition.

### Honest limits

- One instrument (gold H1), one 2-year bullish window, in-sample design.
- The author's mechanics come from his detailed writeup, not a direct source read (CodeBase download needs login); the threshold-level details of his quality gates are unverified -- B5/B10 test the core confirmation definition, C is our best-effort reconstruction of his gates.
- The author's HTF EMA filter and session window are NOT tested here -- this isolates the confirmation definition only.
- Entry at next-bar-open (harness convention), not literally at market on the trigger bar as the author states.
