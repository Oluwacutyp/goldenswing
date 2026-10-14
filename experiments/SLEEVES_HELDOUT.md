# Sleeves Held-Out Validation — fvg_midline + trendline_retest

**Data:** XAU/USD H1, Dukascopy, 2022-01-01 → 2024-10-01 (16258 bars, 2.74y). yfinance cannot serve H1 before ~Oct 2024 (730-day intraday wall), so the 2022–2024 window uses the same underlying market as spot (GC=F futures vs spot XAU/USD — spreads/rolls differ slightly, price action tracks). Silver peer XAG/USD from Dukascopy; dollar peer unavailable (graceful degradation, documented). Zero overlap with the 2024-10→2026-10 development window. **No tuning was performed against this window** — all parameters are the validated ones, byte-identical.

## Headline

| Sleeve | Trades | Net % | PF | Win% | Avg R | MaxDD% |
|---|---|---:|---:|---:|---:|---:|
| fvg_midline | 294 | -13.71 | 0.94 | 36.4 | -0.040 | 42.3 |
| trendline_retest | 103 | -11.08 | 0.85 | 37.9 | -0.104 | 15.9 |

Development-window baselines for comparison: fvg_midline 48 / +34.61% / PF 2.47 · trendline_retest 61 / +15.40% / PF 1.43.

## fvg_midline — by year

| Year | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| 2022 | 84 | -36.4 | -0.433 | 23 |
| 2023 | 100 | +57.1 | +0.571 | 58 |
| 2024 | 110 | -32.3 | -0.294 | 27 |

## fvg_midline — by regime

| Regime | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| compressed | 100 | +1.1 | +0.011 | 38 |
| expanded | 94 | +7.4 | +0.078 | 41 |
| normal | 100 | -20.2 | -0.202 | 30 |

## fvg_midline — by session

| Session | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| asia | 103 | -23.7 | -0.230 | 32 |
| london | 50 | -14.4 | -0.287 | 28 |
| newyork | 68 | -22.3 | -0.328 | 25 |
| off | 11 | +6.5 | +0.593 | 55 |
| overlap | 62 | +42.2 | +0.680 | 60 |

## fvg_midline — by side

| Side | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| long | 167 | +46.4 | +0.278 | 46 |
| short | 127 | -58.0 | -0.457 | 24 |

## trendline_retest — by year

| Year | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| 2022 | 32 | -11.9 | -0.372 | 31 |
| 2023 | 40 | +11.6 | +0.289 | 50 |
| 2024 | 31 | -10.4 | -0.335 | 29 |

## trendline_retest — by regime

| Regime | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| compressed | 35 | -2.1 | -0.059 | 46 |
| expanded | 38 | -1.6 | -0.043 | 34 |
| normal | 30 | -7.0 | -0.234 | 33 |

## trendline_retest — by session

| Session | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| asia | 14 | -0.6 | -0.044 | 50 |
| london | 19 | -2.5 | -0.133 | 37 |
| newyork | 17 | -8.1 | -0.475 | 24 |
| off | 2 | -0.7 | -0.330 | 50 |
| overlap | 51 | +1.2 | +0.023 | 39 |

## trendline_retest — by side

| Side | Trades | Net R | Avg R | Win% |
|---|---:|---:|---:|---:|
| long | 59 | -6.4 | -0.109 | 36 |
| short | 44 | -4.3 | -0.097 | 41 |

## trendline_retest — trap treatment (held-out test of the +4 bonus premise)

The promoted +4 bonus is a RANKING input (same-bar conflicts), not a filter — on a standalone book it cannot change membership. Its held-out test is the trap subset map (salvage §3a convention, trap state at the entry bar):

| Trap state | n | Avg R | Win% | Total R |
|---|---:|---:|---:|---:|
| boost | 66 | +0.124 | 45 | +8.2 |
| oppose | 29 | -0.541 | 21 | -15.7 |
| neutral | 8 | -0.404 | 38 | -3.2 |

Secondary (labeled, NOT the promoted convention): boost_only filter book → 66 trades, +7.78%, PF 1.21, avg +0.124R.

## Verdicts

**fvg_midline: KILLED AS TESTED.** 294 trades clear the count bar, but the edge did not replicate: avg -0.040R, PF 0.94 (dev baseline 48 / +34.61% / PF 2.47).
**trendline_retest: KILLED AS TESTED.** 103 trades clear the count bar, but the edge did not replicate: avg -0.104R, PF 0.85 (dev baseline 61 / +15.40% / PF 1.43).

**Method notes:** scanner, gates, exits, and booking are the validated ones, unchanged. Spot-vs-futures and missing-dollar-peer are the documented deviations from the development setup. The paper loop is unaffected by these verdicts — it keeps earning live OOS evidence either way.
