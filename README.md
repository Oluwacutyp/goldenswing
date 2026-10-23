# GoldenSwing

One engine that trades like a real trader. **MT5 is the source of truth** —
prices, bars, ticks, volumes, account, and positions all come from the
MT5 terminal. The engine reads **all timeframes together, with hierarchy**:
daily sets the direction, 4-hour marks the zones, 1-hour gives the trigger,
15/30-minute times the entry, 5-minute supplies the fine entry leg. The
hierarchy is the desk's structural read — it feeds the score, the Macro
block and the traffic scalar; it is weighed as evidence, not run as a
veto chain. The hard stops are trade-existence facts only (location,
friction, solvency, valid stop geometry). Trades are managed on **tick**,
not on poll — management actions fire the moment price crosses the level.

## How it trades

- **Hierarchy (never democracy):** D1 bias → H4 agreement + H4 zone →
  H1 trigger (BOS / MSCH / CHoCH) → session entry clock (M15 in killzone,
  M30 elsewhere) → M5 entry leg. Entries are taken at zones **of the
  entry timeframe**, nested inside the higher-timeframe context —
  playbooks with their own native clock (`qm_m5` on M5, `qm_d1` on D1)
  evaluate on that clock and need no H1 trigger.
- **Executed sleeves:** the full registry executes on the demo — all 18
  playbooks (the 16-playbook H1 roster plus `qm_m5` and `qm_d1` on their
  native routes). There is no shadow tier; the per-playbook kill switch
  (`/set playbook.<name> off`) is the only opt-out, and the demo itself
  is the measurement (live-demo doctrine: backtests map, demo decides).
- **AMD phase** rides as a tag on the stack, never a veto. Hard veto is OFF.
- **Volume** comes from MT5 (tick volume, real volume where the symbol
  provides it) and nudges the score — it never gates a trade.
- **Tick management:** breakeven is **earned** — the stop moves to entry
  when a funding event (a partial fill / structure event) pays for the
  risk; a +1R touch alone is only the backstop, not the trigger.
  Trailing, partials, stops — all applied on the crossing tick, with
  failed modifies retried on the next tick, never booked as done.
- **Decision funnel:** every playbook evaluation — including no-setup
  outcomes — is journaled with its clock, the timeframe it actually
  read, a reason code (gate / floor / regime / direction / friction
  kept distinct) and, for candidates, the evidence that was weighed.
  `/status` shows the funnel: who was evaluated, and where candidates
  disappeared.

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

The engine runs live on a Windows EC2 host against a real MT5 terminal
(demo account) and its own Telegram bot — deployed 2026-10-21 and on the
current build since (the demo IS the test: forward live-demo performance
is the verdict on every sleeve). The test suite (900+ tests) runs against
a fake terminal in the build environment. The old OANDA paper loop
remains in the repo for local paper trading; it is not the engine.
