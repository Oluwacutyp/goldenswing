# GoldenSwing — Backtest Report (Phase 1)

**Data:** GC=F hourly (yfinance), 11,426 bars, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00 (1.99 years). GC=F is a futures proxy for spot XAUUSD.

**Method:** signal at close of bar t, entry at open of t+1, stop = structural invalidation, target = nearest opposing liquidity pool (2R floor), stop wins same-bar ties, 72-bar time stop, $0.45/oz round-trip cost, 1% risk, one position at a time (traded book).

**Signals simulated (shadow, score ≥ 35):** 1836

## Traded book @ score ≥ 60

| Metric | Value |
|---|---|
| Trades | 584 |
| Win rate | 25.2% |
| Profit factor | 0.87 |
| Total return | -49.82% |
| Max drawdown | 61.09% |
| Sharpe (per-trade, annualised) | -0.91 |
| Avg R | -0.101 |
| Avg hold | 11 bars |

## Threshold comparison (traded book)

| Threshold | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |
|---|---|---|---|---|---|---|---|
| ≥50 | 704 | 25.7 | 0.87 | -56.48% | 63.21% | -1.05 | -0.102 |
| ≥60 | 584 | 25.2 | 0.87 | -49.82% | 61.09% | -0.91 | -0.101 |
| ≥70 | 234 | 20.9 | 0.72 | -44.01% | 54.65% | -1.34 | -0.231 |

## Score discrimination (shadow book — every signal, standalone)

This is the test of the core concept: if 'backing' means anything, higher scores must produce better outcomes than lower scores.

| Score band | Signals | Win% | Avg R | PF |
|---|---|---|---|---|
| 35–50 | 208 | 23.6 | -0.260 | 0.68 |
| 50–60 | 629 | 27.5 | -0.084 | 0.89 |
| 60–75 | 883 | 27.0 | -0.042 | 0.95 |
| 75–101 | 116 | 22.4 | -0.120 | 0.85 |

## Caveats

- One instrument (gold), one timeframe (H1), one 2-year window that was strongly bullish for gold — long-heavy systems get a tailwind.
- GC=F futures ≠ spot XAUUSD: spreads, roll and session gaps differ.
- No slippage modelling beyond the flat $0.45/oz cost; news spikes would be worse in live trading.
- Score weights are hand-set priors, NOT fitted to this data — which keeps the test honest but leaves performance on the table.
- The detector fires on H1 bars; the 3-day horizon is the 72-bar time stop, most trades exit earlier on target/stop.

## Diagnostics (post-run analysis, shadow book splits)

| Slice | n | Win% | Avg R |
|---|---|---|---|
| All signals | 1,836 | 26.5 | −0.086 |
| Longs | 949 | 28.9 | −0.026 |
| Shorts | 887 | 23.9 | −0.151 |
| Sweep reversals | 1,551 | 25.0 | −0.113 |
| Continuations | 285 | 34.7 | **+0.062** |
| Long continuation | 167 | 38.3 | **+0.185** |
| Long continuation @ ≥60 | 155 | 38.7 | **+0.185** |
| Sweep reversals @ ≥60 | 844 | 24.2 | −0.095 |

Exits: 73% stop, 25% target, 2% timeout. Average hold 11 bars — far
short of the 72-bar "3-day swing" design horizon.

**What this says, honestly:**

1. **The detector does not work as a whole.** The traded book loses at
   every threshold. Do not trade this.
2. **Sweep reversals are the failure.** They are 84% of all signals and
   negative at every score level. As implemented — any wick through a
   liquidity pool counts, entry allowed for 8 bars after — H1 sweeps are
   mostly noise, not stop hunts. Real ICT sweep entries demand
   displacement + a structure shift *before* entry; this detector makes
   confirmation optional, and pays for it.
3. **The score half-works.** Avg R improves monotonically from the
   35–50 band (−0.260) through 50–60 (−0.084) to 60–75 (−0.042) — the
   backing factors do rank setups — then the 75+ band inverts (−0.120,
   n=116), because high scores are dominated by "fresh sweep + big room"
   combinations that don't follow through. No band is profitable overall.
4. **One slice has a pulse:** trend-aligned continuation, especially
   long (+0.185R over ~160 trades, stable under the ≥60 filter). Small
   sample, and a 2-year gold bull market is a tailwind for anything
   long — treat as a hypothesis for Phase 2, not an edge.
5. **Shorts into a historic gold bull run** (−0.151R) were fighting the
   tape; the system has no higher-timeframe regime veto.

**Next fixes to try (in order):** require displacement + CHoCH
confirmation before sweep entries count; add a daily-bias regime filter
(no shorts against a rising daily structure); re-test whether the
continuation slice survives on its own.
