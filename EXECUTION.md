# Phase 8 — MT5 Python Execution + Risk + Learning Loop

## What was built

Execution per the owner's correction (SPEC): the official
**MetaTrader5 Python library** — NOT an EA. Brain and executor are
both Python; the library talks to a terminal that must be installed,
logged in, and ACTIVE on the same machine.

```
goldenswing/execution/
  risk.py       SymbolSpec, size_position (never-round-up sizer),
                RiskGuards (persisted, latching)
  executor.py   the Executor interface + OrderRequest/Result/Position
  manager.py    ExitStepper — Phase-2 exit semantics as a live stepper
  paper.py      PaperExecutor (bar-driven simulated broker)
  mt5.py        MT5Executor (lazy import, injectable module)
  learning.py   TradeJournal (JSONL) + PerformanceBook + export
  callbacks.py  ExecutionCallbackHandler (alerts CallbackHandler)
  runner.py     LiveRunner: run_bars (anywhere) / run_live (Windows)
scripts/live_runner.py   CLI entry point
```

### Risk layer (`risk.py`) — ported ATS patterns

The ATS Risk Library design (docs/mql5_mining.md §4), in Python.
The layer is a pure guard: it can refuse, it can never trade.

- **Sizer**: risk-% → lots from stop distance and the symbol's tick
  value/size. Lots are **floored to the volume step, never rounded
  up**; if the risk budget cannot buy the broker minimum the answer
  is **0 lots + a reason string**, never a quiet round-up.
- **Guards**: daily loss limit (% of day-start balance and/or money
  — smaller wins), weekly loss limit, max-drawdown halt (from peak),
  consecutive-loss halt, max open positions, and an **exposure cap:
  the actual money lost if every open stop is hit** (positions with
  no stop block new risk — their exposure is unknowable).
- **Kill switch**: a file on disk; its existence blocks every new
  order (checked on every pipeline pass).
- **Persistence**: guard state is JSON on disk
  (`data/risk_state.json` by default). Day/week anchors roll over by
  calendar date; a tripped halt **latches** — restarts cannot reset
  it; a human clears it with `RiskGuards.reset_halt()`.

### Executors (one interface)

- **PaperExecutor**: orders fill at the next bar's open ± half the
  $0.45/oz round-trip cost; positions are managed by the shared
  ExitStepper — the backtest's Phase-2 semantics verbatim (stop
  buffered 0.25×ATR beyond invalidation, 50% banked at the first
  pool ≥1R away, trail 1.0R behind best excursion after +1.5R,
  72-bar time stop, stops checked before partials). Every close
  fires a `CloseRecord` (money P/L, R, reason, hold bars).
- **MT5Executor**: builds real `order_send` requests — price from
  the live tick (ask to buy, bid to sell), SL from the signal's
  invalidation, deviation from config, filling mode read from the
  symbol's `filling_mode` bitmask and tried **IOC → FOK → RETURN**
  until the broker accepts. `modify_sl` uses TRADE_ACTION_SLTP;
  closes are opposite deals (partial volumes supported); the
  alert id rides in the order comment (`GS:<id>`) so terminal
  positions map back to journal records. The library is imported
  lazily — the module loads on any OS, and live calls raise
  `MT5Unavailable` with the precise blocker.

### Exit management

One stepper, both hosts. On paper it prices its own events; live,
`_LiveManager` (runner.py) tracks MT5 positions opened from the
runner's alerts and translates stepper events into broker calls —
partial → partial close, trail advance → SL move, stop/timeout →
full close — while the SL placed with the original order protects
the position between bars. A tracked position that vanishes
(broker-side SL) is journalled as a note, not silently dropped.

### Learning loop (`learning.py`)

Append-only JSONL journal: `open` (score, factors, forecaster
probability/rank, mode, playbook, **phase at entry**), `close`
(exit, reason, P/L, R), `skip`, `note` (every refusal, with its
reason). `PerformanceBook` joins opens to closes and aggregates per
**(phase × playbook × mode)** — the measured effectiveness data the
routing layer and future retrains consume.
`export_training_data()` writes (features → label, R) rows for
closed trades; model retraining itself stays in `forecast/`.

### Callback wiring (`callbacks.py`)

Implements the Phase-7 `CallbackHandler` protocol. A tap flows:
alert → handler → **size → guard check → place → journal**, and the
reply text says what actually happened (lots, stop, risk, ticket —
or the refusal and its reason). Execute uses the live executor,
Paper the paper book (labelled `source="paper"`, no guard gate —
it is simulation), Skip journals the declined setup.

## Running it

### Anywhere (dry proof — what was run here)

```bash
python3 scripts/live_runner.py [--max-bars N] [--auto]
```

Replays cached bars through the whole loop with alerts forced to
the offline outbox. Verified in this sandbox on 3,000 real bars:
127 signals → 50 alerts → auto-executed paper trades → journal +
performance book + persisted guard state produced.

### Windows host (live)

1. Install MT5, log the terminal in (demo or real — that choice IS
   the account choice), leave it running.
2. `pip install MetaTrader5` (and the project's deps).
3. Train the forecaster bundle if wanted:
   `python3 scripts/train_forecast.py`.
4. Set env and run:

```powershell
$env:GS_TELEGRAM_BOT_TOKEN="..."; $env:GS_TELEGRAM_CHAT_ID="..."
$env:GS_SYMBOL="XAUUSD"; $env:GS_RISK_PCT="0.01"
# optional: $env:GS_MT5_LOGIN / GS_MT5_PASSWORD / GS_MT5_SERVER / GS_MT5_PATH
# optional: $env:GS_AUTO_EXECUTE="1"   (else trades wait for button taps)
python scripts/live_runner.py --live
```

Config env: `GS_SYMBOL, GS_MODE, GS_AUTO_EXECUTE, GS_RISK_PCT,
GS_PAPER_BALANCE, GS_JOURNAL, GS_RISK_STATE, GS_KILL_SWITCH,
GS_FORECAST_BUNDLE` (paths default under `data/`). Kill switch:
create the file at `GS_KILL_SWITCH` (default `data/KILL`) to block
all new orders instantly.

## What is NOT verified — read before trusting it live

- **The live MT5 path has never run against a real terminal.**
  There is no Windows host or MT5 terminal in the build
  environment; `MetaTrader5` cannot even be imported here.
  MT5Executor's request-building, filling-mode fallback and sizing
  are unit-tested against an injected fake module that records the
  requests — fills, broker rejections beyond filling mode, margin
  behaviour and real retcodes are **untested**. First live runs
  belong on a demo account with the journal watched trade by trade.
- **The forecaster bundle could not be loaded in this sandbox**
  (joblib/scikit-learn unavailable — PEP 668 externally-managed
  environment), so the runner's live scoring path is exercised only
  with injected fake models in tests. On the Windows host,
  `ForecasterScorer.try_load` reports its own status at startup
  and the runner degrades to detector-only rather than pretending.
- **Broker-side closes** (the SL placed with the order firing
  between bars) are detected and journalled as notes; their exact
  exit price/PnL lives in the terminal history, not the journal.
- GC=F (the backtest/paper feed) ≠ the broker's spot XAUUSD:
  spreads and session gaps differ; paper results are a rehearsal,
  not a promise.

## Tests

`tests/test_execution_{risk,paper,mt5,learning,runner}.py` — 36
tests: sizer flooring/refusals, guard trips + persistence across
instances + rollover + kill switch, paper fill/partial/trail/stop/
timeout/manual-close semantics, MT5 request construction against
the fake module (no fills faked), journal aggregation + export,
the full tap → risk → order callback flow, and the dry runner loop
end to end. Full suite: **175 passed, 1 failed** — the failure is
the pre-existing `test_forecast.py::test_walk_forward_smoke`,
which needs scikit-learn (not installed in this sandbox).
