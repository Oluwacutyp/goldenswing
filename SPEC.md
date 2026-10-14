# GoldenSwing — Owner Spec (2026-10-10)

The named spec is the FLOOR, not the ceiling. "And many more, not limited to this."

## Core identity
One AWARE system that reads the chart and the market itself. Not a pile of
strategies. A Golden Swing = a big move BACKED by confluence. ML forecasts
big moves across MULTIPLE horizons (owner correction 2026-10-11: "not
limited to 3 days" — the 3-day figure was an example; horizons span
intraday → multi-week, each horizon with its own label/model/evaluation).
The system auto-trades (MetaTrader5 Python library — see Execution) AND
alerts.

## Market reading (all required, not limited to)
- Structure: HH / HL / LL / LH sequences, BOS, CHoCH
- Liquidity: SSL sweep, BSL sweep, equal highs/lows, pools
- PD zones (premium / discount), dealing ranges
- FVG, order blocks, breaker/mitigation blocks, IFVG
- Zone detection (supply / demand zones)
- Fibonacci: retracement levels, OTE zone
- Bollinger Bands + Bollinger trend states
- Ichimoku cloud (trend, cloud support/resistance, TK alignment)
- Relative Strength (RS) and comparative strength vs peers/index
- Head & shoulders and classical chart patterns ("head")
- Accumulation / Consolidation / Manipulation phase detection (AMD, Power of 3)
- Institutional footprint detection: bank entry signatures, institution
  footprints, CTA flow character, retail positioning/chase character, CTI
- Volume: profile, delta character, climax/ dry-up behaviour
- Multi-timeframe: HTF bias → LTF execution, timeframe alignment scoring

## Modes
- Swing (multi-day, the 3-day forecast horizon)
- Scalp
- Fast scalp
Each mode shares the same market read; entries/exits/horizons differ.

## Alerts
- Telegram bot alerts with inline buttons (Execute / Skip / Paper)
- Email alerts
- Chart images at TradingView level: 5 chart classes/styles (incl. phone-first `signal` alert chart), volume drawn,
  all studies/levels drawn on the chart (zones, FVGs, Fib, cloud, swings)
  — study-able: a human can read the whole thesis off the image.

## Learning
Ensemble ML (supervised swing predictor + DRL sizing/timing + proven
deterministic cores as voters). Retrains on its own trade history. Gets
better over time — for real, measured by backtest + live paper record.

## Execution (owner correction 2026-10-10)
NOT an EA bridge. Execution uses the official **MetaTrader5 Python library**
(`MetaTrader5` package), which requires an MT5 terminal installed, logged
in, and ACTIVE on the same machine. The library is Windows-only, so live
deployment = the full stack on a Windows host (Windows EC2 on the owner's
existing credits; AWS Linux box cannot run it). Brain and executor are both
Python, direct calls: copy_rates for data, order_send for execution — demo
or real by which account the terminal is logged into.
Risk system: kill switch, daily/weekly loss limits, adaptive sizing,
never-round-up lot sizing, guards persisted across restarts (design
patterns from the MQL5 ATS Risk Library mining, implemented in Python).

## Honesty rule
Every component is backtested before it is trusted. Losing components are
reported as losing, with diagnosis. No codegen parameter-stamp "strategies".
