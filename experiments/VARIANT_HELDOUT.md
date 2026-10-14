# Variant Held-Out Validation — sweep_reversal long/with-bias/expanded

**Data:** XAU/USD H1, Dukascopy, 2022-01-01 → 2024-10-01 (16258 bars, 2.74y). yfinance cannot serve H1 before ~Oct 2024 (730-day intraday wall), so the task's 2022–2024 GC=F request was fulfilled with the same underlying market (spot XAU/USD vs GC=F futures — spreads/rolls differ slightly, price action tracks). Silver peer XAG/USD from Dukascopy; dollar peer unavailable (graceful degradation, documented). No overlap with the 2024-10→2026-10 development window.

**Variant definition** (from ANCHOR_AUTOPSY.md hunting ground): LONG ONLY + HTF bias alignment == 'with-bias' + vol_regime == 'expanded'. Scanner, gates, greedy book at validated threshold 60.0, and exits are byte-identical to the validated anchor. The ONLY changes are the three context restrictions.

## Headline

| Book | Trades | Net % | PF | Win% | Avg R | MaxDD% |
|---|---|---:|---:|---:|---:|---:|
| control | 319 | -36.14 | 0.79 | 37.9 | -0.133 | 37.4 |
| variant | 83 | -1.20 | 0.99 | 39.8 | -0.005 | 9.4 |

## control — by year

| Year | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| 2022 | 111 | -17.9 | -0.162 | 39 |
| 2023 | 124 | -6.7 | -0.054 | 41 |
| 2024 | 84 | -17.8 | -0.212 | 32 |

## control — by regime

| Regime | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| compressed | 89 | -3.5 | -0.040 | 42 |
| expanded | 116 | -22.7 | -0.196 | 35 |
| normal | 114 | -16.1 | -0.142 | 38 |

## control — by session

| Session | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| asia | 55 | -0.3 | -0.005 | 44 |
| london | 96 | -26.0 | -0.271 | 32 |
| newyork | 44 | -14.7 | -0.335 | 30 |
| off | 7 | -1.6 | -0.232 | 29 |
| overlap | 117 | +0.3 | +0.002 | 44 |

## variant — by year

| Year | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| 2022 | 24 | -4.2 | -0.176 | 38 |
| 2023 | 31 | +5.2 | +0.168 | 45 |
| 2024 | 28 | -1.4 | -0.050 | 36 |

## variant — by regime

| Regime | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| expanded | 83 | -0.4 | -0.005 | 40 |

## variant — by session

| Session | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| asia | 7 | -5.5 | -0.780 | 14 |
| london | 10 | -1.9 | -0.191 | 40 |
| newyork | 22 | -5.7 | -0.258 | 32 |
| off | 5 | -2.4 | -0.487 | 20 |
| overlap | 39 | +15.1 | +0.387 | 51 |

## Verdict

**VARIANT KILLED AS TESTED.** 83 trades clear the count bar, but the edge did not hold: avg -0.005R, PF 0.99. The hunting ground does not replicate on 2022–2024 data. No variant, no promotion.

Control (full anchor, unchanged) on the same window: 319 trades, -36.14%, PF 0.79, avg -0.133R — for the 'does the whole thing work here' question.

**Method notes:** no tuning was performed against this window; all parameters are the validated anchor's. Spot-vs-futures and missing-dollar-peer are the documented deviations from the development setup.
