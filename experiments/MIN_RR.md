# GoldenSwing — Minimum R:R Qualification Filter

**Mined claim (Pro Trading School, user-mined 2026-10-11):** *before entering, require the trade to offer at least 2-to-1 reward-to-risk.* The exit stepper banks partials at the first pool >= 1R and trails — but no entry-time R:R qualification exists.  This experiment adds one: at signal time, risk = |signal-bar close - buffered structural stop|, reward = |first opposing pool (>= 1R away) - signal-bar close|; skip the signal when reward/risk < threshold.

**Data:** GC=F hourly, 11,426 bars, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00 (1.99 years). Same Phase-2 entries/exits/costs as the anchor; the ONLY change per row is the entry-time filter.

**Anchor (verified, must reproduce):** Combo A + aoc, 199 trades, PF 1.25, +28.92%, MaxDD 10.38%, Sharpe 1.00 (AOC.md: +28.92% / 1.25 / 199). Shadow @≥60: 351 signals, avg R +0.134.

## Threshold sweep (traded book @ ≥60)

| Min R:R | Kept / killed signals | Book trades | Win% | PF | Return | MaxDD | Sharpe | Avg R | H1 ret | H2 ret | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ≥1.5 | 150 / 317 | 96 | 38.5 | 0.88 | -7.41% | 12.97% | -0.38 | -0.072 | -10.40% | +3.34% | DILUTES |
| ≥2.0 | 80 / 387 | 57 | 40.4 | 0.92 | -3.06% | 5.95% | -0.18 | -0.046 | +0.05% | -3.12% | DILUTES |
| ≥2.5 | 53 / 414 | 41 | 39.0 | 1.05 | +0.92% | 5.00% | 0.10 | +0.033 | +3.02% | -2.04% | DILUTES |
| ≥3.0 | 40 / 427 | 32 | 43.8 | 1.03 | +0.30% | 5.00% | 0.05 | +0.018 | -1.72% | +2.06% | DILUTES |

## Does the filter remove losers? (shadow @≥60, kept vs killed)

| Min R:R | Kept n / avg R / PF | Killed n / avg R / PF |
|---|---|---|
| ≥1.5 | 123 / -0.016 / 0.97 | 228 / +0.215 / 1.41 |
| ≥2.0 | 68 / -0.029 / 0.95 | 283 / +0.173 / 1.32 |
| ≥2.5 | 47 / +0.029 / 1.05 | 304 / +0.150 / 1.28 |
| ≥3.0 | 37 / -0.014 / 0.98 | 314 / +0.152 / 1.28 |

## Verdict rule

ADDS = book return up > 1pp AND risk-adjusted no worse (Sharpe up, or MaxDD down ≥ 1pp, or PF up ≥ 0.05).  DILUTES = book return down > 1pp.  A filter that raises win rate while lowering total return is a dilution, stated as such.

## Honest verdict

**KILLED — the filter is anti-predictive at every threshold.** Not only
does every threshold dilute the book (≥2.0, the video's claim: +28.92% →
−3.06%), the mechanism check shows the filter keeps the losers and kills
the winners: killed @≥60 signals grade **+0.15R to +0.22R** while kept
signals grade **−0.03R to +0.03R**. Win rate barely moves (38–44% vs
anchor ~45%); the damage is done by deleting profitable trades.

**Diagnosis — why the video's logic inverts here.** The claim assumes a
fixed-R:R world (enter, stop, target placed at 2R). Our system exits at
*liquidity pools* and trails. In a pool-targeting system, "distance to
the first pool" is not the reward — it is often the *reachability*:
far first pools (high measured R:R) are the unreachable ones (the JeaFx
retest experiment found the same: nearest pool ~4.5R away, 0/63 wins),
while near pools (low measured R:R) are where the partial-bank +
trail machinery actually harvests. The filter selects for unreachable
targets and deletes the harvestable ones. The R:R qualification is
structurally backwards under pool-based exits — it would need fixed-R:R
targets to make sense, and fixed-R:R targets are not our system.

Per the verdict rule, a filter that raises nothing and lowers total
return is a dilution, stated as such: **DILUTES at 1.5 / 2.0 / 2.5 /
3.0.** The claim is killed as an entry filter. (Its spirit — don't take
trades with no room to run — is already encoded: signals with no
opposing pool ≥ 1R away fail the simulation's own target guard.)

## Caveats

- The qualification prices the trade at the signal bar's close (sig.entry_ref); the outcome simulation fills at the next bar's open per repo convention.  The one-bar gap is disclosed; it affects the filter and the fill identically for every signal, so it cannot favour any threshold.
- Signals with no opposing pool >= 1R away FAIL the qualification (no measurable target → no trade).  This is conservative and matches the video's logic.
- Same single in-sample window as all Phase 3–4.5 work; verdicts are diagnostics, not out-of-sample proof.
- The R:R thresholds (1.5/2.0/2.5/3.0) are the video's claim plus neighbours — not fitted.
