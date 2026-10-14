# Failure-Mode Diagnosis: Direction Failure vs Timing Failure

**Owner's distinction:** when a strategy's buy loses, two different failures demand opposite responses:
1. **Direction failure** — price goes the wrong way and *stays* there → the detection found something real but the label is wrong → flip candidate.
2. **Timing failure** — price dips, hits the stop, *then rallies the predicted way* → direction was right, entry was early → fix the entry, do NOT flip.

## Method (and the calibration catch)

For every losing trade that exited via its initial structural stop, measure **net** post-stop drift over the next 24 bars in units of the trade's own R:

- `drift_24 = direction × (close[exit+24] − stop_price) / risk`
- TIMING if drift_24 ≥ +1.0R · DIRECTION if drift_24 ≤ −1.0R · NEITHER otherwise

Net drift (not max excursion) — the owner's scenarios are about what *prevails* ("stays down" vs "rallies"), and a wick that touches back and fades is not a rally.

**The calibration catch that saved this experiment:** the first classifier used max-recovery (≥1R wick-back = timing). A random-entry control exposed it as biased: random trades with 1-ATR stops scored **75% timing**. The net-drift classifier is symmetric under the null by construction — the control confirms it (0.5 ATR: 43% direction / 44% timing; 1.0 ATR: 36% / 38%; 2.0 ATR: 32% / 25%). Every verdict below is measured against that baseline.

Clean failures only: partial-banked, trailed, breakeven and timeout exits are counted separately, never diagnosed. Halves split by signal bar. Verdict bar: ≥60% of classified losers.

## Results

| Strategy | n diag | Direction | Timing | Neither | Verdict |
|---|---|---|---|---|---|
| T1_failed_pattern (bias) | 134 | 34.3% | 27.6% | 38.1% | stays-buried |
| T2_pin_violation (bias) | 856 | 35.8% | 30.8% | 33.5% | stays-buried |
| T3_zone_breaker (bias) | 3 | 33/33/33 | — | — | stays-buried (thin) |
| T4_doji_expansion (bias) | 53 | 17.0% | 32.1% | 50.9% | stays-buried |
| T2_pin_violation_inv | 536 | 49.9% | 46.7% | 3.4% | stays-buried |
| engulf_trigger_B | 63 | 39.7% | 23.8% | 36.5% | stays-buried |
| marco_sweep_B | 104 | 40.4% | 40.4% | 19.2% | stays-buried |
| control_random_1.0atr | 1197 | 36.4% | 38.2% | 25.4% | (baseline) |

Halves are consistent for every strategy (no half flips a verdict).

## Reading

**No flip candidates. No entry-refinement candidates. The honest answer to "flip or fix the entry?" is: neither dominates — these strategies lose the way random trades lose.** T1, T2-bias and T4 sit at the control baseline; engulf-B is 3pp above it on direction (noise); marco-B splits 40/40 with slightly less chop than random.

Two non-random signatures, neither actionable:

1. **T2-inverse's "no chop" signature** (3.4% neither vs 25% control): after stopping out the fade-the-violation short, price *always* goes somewhere big — 49.9% keeps rallying (the long was right), 46.7% collapses back (the short was eventually right, entry was toxic). Median max excursion 30R on a tight stop. This is the stop-hunt bar's two-way explosiveness, measured. It confirms the placement review's asymmetry finding (bias and inverse behave nothing alike) without promoting either.
2. **Marco-B's 65 breakeven scratches** (of 169 losers): the tight "cover the level" stop gets harvested, BE saves the rest. The losers that do die split evenly — the entry is bad in both ways at once.

**What this rules out:** the fear that a killed strategy is "secretly good, just inverted" — no strategy's losers show a clean dominant failure mode, and the verdicts are robust to the threshold choice (absolute 60% and control-relative bars agree).

## Files

- `goldenswing/experiments/failure_modes.py` — classifier, drift measurement, strategy runners (reuse placement/engulf/marco scanners), random-entry control
- `scripts/experiment_failure_modes.py` — runner
- `tests/test_experiments_failure_modes.py` — 15 tests (synthetic bars: sustained rally, wick-without-close, mirrored shorts, exclusions, halves, session tagging)
