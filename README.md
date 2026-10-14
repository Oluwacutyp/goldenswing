# GoldenSwing

One engine that trades like a real trader. **MT5 is the source of truth** —
prices, bars, ticks, volumes, account, and positions all come from the
MT5 terminal. The engine reads **all timeframes together, with hierarchy**:
daily sets the direction, 4-hour marks the zones, 1-hour gives the trigger,
15/30-minute times the entry, 5-minute supplies the fine entry leg. A higher
timeframe can block a lower one; a lower one can never overrule a higher
one. Trades are managed on **tick**, not on poll — breakeven, trailing,
and partials fire the moment price crosses the level.

## How it trades

- **Hierarchy (never democracy):** D1 bias → H4 agreement + H4 zone →
  H1 trigger (BOS / MSCH / CHoCH) → session entry clock (M15 in killzone,
  M30 elsewhere) → M5 entry leg. No zone, no trade. No trigger, no trade.
- **Executed sleeves:** `fvg_midline` + `trendline_retest`, gated by the
  whole hierarchy. The sweep-reversal anchor stays shadow-only.
- **AMD phase** rides as a tag on the stack, never a veto. Hard veto is OFF.
- **Volume** comes from MT5 (tick volume, real volume where the symbol
  provides it) and nudges the score — it never gates a trade.
- **Tick management:** breakeven at +1R (stop to entry), trailing,
  partials, stops — all applied on the crossing tick, with failed
  modifies retried on the next tick, never booked as done.

## Run it (Windows EC2, MT5 terminal logged into demo)

```powershell
git clone https://github.com/Oluwacutyp/goldenswing.git
cd goldenswing
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install MetaTrader5 pandas numpy
# + repo requirements
.\.venv\Scripts\python.exe scripts\run_mt5.py
```

Set `GS_TELEGRAM_BOT_TOKEN` and `GS_TELEGRAM_CHAT_ID` in the environment
for Telegram. See `docs/EC2_MT5_RUNBOOK.md` for the full steps, cutover,
and verification.

## Telegram

The engine listens and answers: `/status /report /db /trades /harvest
/ticks`. Every signal and every fill sends a **multi-timeframe chart**
(H4 structure, H1 trigger, entry timeframe) with the zone, entry, stop,
targets, and verdict. `/harvest` bundles the journal, state, tick CSVs,
and results into a ZIP. `/ticks` exports tick history for backtesting.

## Safety

- Demo only by default — the engine **refuses real accounts** unless
  explicitly allowed.
- One position at a time.
- Stop with `Ctrl+C` or by creating `data\mt5_KILL`.

## Layout

- `goldenswing/feed/mt5.py` — MT5 feed: bars, ticks, volumes, account.
- `goldenswing/live/hierarchy.py` — the one hierarchy behind live +
  backtest.
- `goldenswing/execution/mt5_runner.py` — the engine loop.
- `goldenswing/execution/tick_manager.py` — tick-driven management.
- `goldenswing/execution/mt5_commands.py` — Telegram commands.
- `goldenswing/charts/engine.py` — multi-timeframe signal/fill charts.

## Status

The engine has been tested on a fake terminal (900+ tests green). It has
not yet run against a real terminal or real Telegram — the first live run
needs watching. The old OANDA paper loop remains in the repo for local
paper trading; it is not the engine.
