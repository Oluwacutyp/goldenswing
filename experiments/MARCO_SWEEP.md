# Experiment: Marco's sweep-entry vs our confirmation

**Source:** Marco Trades via Chart Fanatics (user-mined 2026-10-11). The claim: for liquidity sweeps, enter AT/AFTER the sweep with NO displacement+CHoCH wait -- "the edge is the level, not entry precision." The defense is level selection: only "engineered liquidity" qualifies -- levels respected from the left (>= 2 approaches within 0.25 ATR, no body close through, prior 200 bars, causal).

**Data:** GC=F H1, 11426 bars, 2024-10-11 04:00:00+00:00 -> 2026-10-09 20:00:00+00:00 (1.99y). $0.45/oz cost, 72-bar horizon.

**Variants** (identical non-score gates: daily veto, room-to-pool, stop-distance band; NO score gate):
- **A (ours):** sweep of a QUALIFIED level -> displacement + CHoCH/BOS within 8 bars. Phase-2 exits (buffered structural stop, 50% partial at first pool >= 1R, trail after +1.5R).
- **B (Marco's):** sweep of a QUALIFIED level -> enter at next bar open, no confirmation. Stop just beyond the swept level + 0.25 ATR. Target = nearest opposing QUALIFIED level. Marco exits: BE roll at +1R, hold to target.
- **C (ablation):** sweep of ANY pool -> immediate entry, Marco exits, target = nearest opposing pool. B vs C isolates the level filter.

## Level qualification

**1037** of **1133** sweeps (91.5%) passed the engineered-liquidity filter at >= 2 touches. The filter as operationalized is permissive -- nearly every reader pool has two prior approaches in 200 bars.

## Anchor validation

The real detector (Phase-2 config, emit floor 35) emits **49** sweep-reversal signals on this window. Variant A fires on **34** qualified sweeps; **33** of its trigger bars coincide with a detector signal bar -- the trigger replication matches the engulf_trigger experiment's variant A (34 sweeps) exactly.

## Shadow (every trigger standalone)

| Variant | n | Win% | PF | Avg R |
|---|---|---|---|---|
| A | 34 | 44.1% | 0.94 | -0.036 |
| B | 200 | 15.5% | 0.73 | -0.160 |
| C | 254 | 16.1% | 0.77 | -0.135 |

## Books (one position at a time, sequential)

| Variant | Trades | Win% | PF | Avg R | Return | MaxDD | Sharpe | Avg hold |
|---|---|---|---|---|---|---|---|---|
| A | 33 | 45.5% | 0.99 | -0.007 | -0.45% | 6.79% | -0.02 | 31.0 |
| B | 196 | 15.8% | 0.73 | -0.162 | -28.89% | 32.60% | -1.05 | 4.8 |
| C | 247 | 16.6% | 0.77 | -0.134 | -30.25% | 36.46% | -0.97 | 5.0 |

## Window halves (book)

| Variant | Half | Trades | Win% | PF | Avg R | Return |
|---|---|---|---|---|---|---|
| A | H1 | 19 | 47.4% | 1.06 | +0.031 | +0.47% |
| A | H2 | 14 | 42.9% | 0.90 | -0.058 | -0.91% |
| B | H1 | 93 | 9.7% | 0.56 | -0.295 | -24.96% |
| B | H2 | 103 | 21.4% | 0.92 | -0.042 | -5.24% |
| C | H1 | 125 | 12.0% | 0.62 | -0.254 | -28.38% |
| C | H2 | 122 | 21.3% | 0.98 | -0.011 | -2.62% |

## Exit mix (Marco variants)

- B: {'stop': 104, 'be': 65, 'target': 30, 'timeout': 1}
- C: {'stop': 132, 'target': 40, 'be': 81, 'timeout': 1}

## Sensitivity: variant B at stricter touch thresholds

| Min touches | Hits | Shadow n | Shadow avg R | Book trades | Return | PF | Win% |
|---|---|---|---|---|---|---|---|
| 2 | 201 | 200 | -0.160 | 196 | -28.89% | 0.73 | 15.8% |
| 3 | 171 | 170 | -0.159 | 168 | -25.13% | 0.73 | 16.1% |
| 4 | 149 | 148 | -0.181 | 146 | -24.58% | 0.69 | 17.1% |
| 5 | 115 | 114 | -0.026 | 112 | -3.99% | 0.95 | 20.5% |
| 6 | 92 | 92 | -0.033 | 92 | -3.76% | 0.93 | 20.7% |

## Verdict

**The Phase-2 pillar stands -- Marco's entry loses decisively, and the level filter does not rescue it.**

1. **B (Marco's) loses at the claimed threshold: -28.89%** book (196 trades, PF 0.73), shadow -0.160R. The unconfirmed sweep entry is not a sample artifact -- it is 200 outcomes deep.
2. **C (any pool, immediate) loses slightly worse: -30.25%.** B vs C isolates the level filter: it is worth ~1.4pp of book and a touch less bad shadow. The filter has small merit as context, not as a rescue.
3. **Stricter "engineered" levels lose less but still lose.** At >= 6 touches B improves to -3.76% (PF 0.93, 92 trades) from -28.89% -- the filter measures something real (heavily-respected levels fade less violently), but no threshold turns the unconfirmed entry profitable. The improvement curve flattens well below breakeven.
4. **A (confirmation) is ~flat at -0.45%** on the qualified universe (33 trades) -- the confirmed-sweep slice was never the book's engine (BACKTEST2: +0.026R shadow), but it is not the bleed that B is.
5. **Exit mix explains the mechanism:** B's exits are stop 104 / BE 65 / target 30 / timeout 1 -- only 15% of trades ever reach the opposing engineered level. The level is the thesis and price rarely gets there; the 0.25-ATR stop "covering the level" gets harvested by the post-sweep chop. Marco's "edge is the level" requires the level to hold often enough to pay for the entries that don't -- on gold H1 it does not.

**Status: killed. Do not adopt the unconfirmed entry.** The engineered-liquidity touch count is kept as a candidate *context* feature (stricter levels = gentler fades), not as an entry system.

### Honest limits

- One instrument (gold H1), one 2-year bullish window, in-sample design.
- Marco's examples are lower timeframes and NQ/gold intraday; the verdict is for GoldenSwing's market and timeframe. His eyeball for "respect" is stricter than any touch count -- the >= 6 threshold is the closest mechanical proxy and it still loses.
- Variant B bundles Marco's entry AND his exits/target selection; only the B-vs-C pair isolates the level filter.
- The engineered filter at >= 2 touches is permissive (91.5% of sweeps qualify); its merit shows only at >= 5-6 touches, where samples thin (92-115 trades).
