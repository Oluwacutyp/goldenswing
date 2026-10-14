# GoldenSwing SYSTEM.md — Production Architecture

**Phase 10: "a system worthy of the strategies."** The strategies problem
is solved (Phase 9: confluence engine, three playbooks, anchor +28.92%
exact). This document describes the SYSTEM that carries them to production.

> The bar is god-tier. This isn't just wiring — it's a production trading
> desk. Every failure mode is named, every recovery is procedural, every
> deployment step is verifiable.

---

## 1. Architecture Overview

GoldenSwing is a multi-playbook confluence trading system for XAUUSD (GC=F
data, XAUUSD execution). One chart reader, multiple entry playbooks, ranked
by confluence, acting like a real trader.

```
┌─────────────────────────────────────────────────────────────────┐
│                         FEED LAYER                              │
│  MT5 terminal (Windows) → copy_rates_h1 → bar buffer            │
│  CSV (sandbox) → load_gold_h1                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │ bars (OHLCV, H1)
┌────────────────────────▼────────────────────────────────────────┐
│                      READER LAYER                               │
│  MarketReader.read → MarketState per bar                        │
│  FeatureEngine.compute → FeatureSet (regime, structure, phases)  │
│  read_and_featurize (Phase-9 unified path)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │ states (features attached)
┌────────────────────────▼────────────────────────────────────────┐
│                     PLAYBOOK LAYER                              │
│  SweepReversalPlaybook (LIVE, anchor, session_scope=any)         │
│  FVGMidlinePlaybook (SHADOW, session_scope=any)                 │
│  TrendlineRetestPlaybook (SHADOW, session_scope=any)            │
│  Each: candidates() [batch] / live_signal() [incremental]       │
│        check_context() [tripwire] / hunting_ground [exclusions]  │
└────────────────────────┬────────────────────────────────────────┘
                         │ candidates with base scores
┌────────────────────────▼────────────────────────────────────────┐
│                    CONFLUENCE LAYER                             │
│  ConfluenceEngine: 3-gate enforcement per candidate              │
│    1. session scope (first-class; never fires outside)           │
│    2. required-context tripwire (firing outside = system bug)    │
│    3. hunting-ground exclusions (measured, not asserted)         │
│  Scoring: base + signal bonuses (AOC/div/magnet) + map bonuses   │
│  Live book: greedy one-position-at-a-time, chronological         │
└────────────────────────┬────────────────────────────────────────┘
                         │ scored signals (live playbooks only)
┌────────────────────────▼────────────────────────────────────────┐
│                      ALERT LAYER                                │
│  AlertService.process → Alert (5 chart classes — `signal` is the phone-first alert default — + Telegram +      │
│  email, outbox fallback). CallbackRegistry for EXECUTE taps.     │
└────────────────────────┬────────────────────────────────────────┘
                         │ alerts (human or auto)
┌────────────────────────▼────────────────────────────────────────┐
│                       RISK LAYER                                │
│  RiskGuards: pre-trade checks (daily/weekly/drawdown/consec/    │
│  positions/total-risk/stops) → allow/deny. Kill-switch file.    │
│  Shadow mode (backtest) vs enforce mode (live).                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ approved orders
┌────────────────────────▼────────────────────────────────────────┐
│                     EXECUTION LAYER                             │
│  PaperExecutor (sandbox/dry-run) | MT5Executor (live, Windows)   │
│  ExitStepper: partials, trails, time stops (Phase-2)             │
│  _LiveManager: bar-by-bar MT5 position management                │
└────────────────────────┬────────────────────────────────────────┘
                         │ fills, closes
┌────────────────────────▼────────────────────────────────────────┐
│                      JOURNAL LAYER                              │
│  TradeJournal (JSONL, append-only): opens, closes, notes,        │
│  refusals. Performance.aggregate() for the book.                 │
└─────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────┐
                         │   WATCHDOG (live)    │
                         │ - bar staleness      │
                         │ - MT5 connectivity  │
                         │ - dead-man's alert   │
                         │ - heartbeat file     │
                         └──────────────────────┘

                         ┌──────────────────────┐
                         │  STATE STORE (live)  │
                         │ - bar buffer (2000)  │
                         │ - meta (last bar)    │
                         │ - playbook state     │
                         └──────────────────────┘
```

---

## 2. Data Flow: Feed to Fill

### 2.1 The live path (Windows host, `run_live`)

1. **Feed**: `MT5Executor.copy_rates_h1(symbol, 1500)` pulls H1 bars from
   the terminal. The forming bar is dropped (`df.iloc[:-1]`).
2. **Resume**: `StateStore` loads the 2000-bar buffer; only bars newer
   than the buffer are appended. Cold start (no buffer) does a full fetch.
3. **Watchdog**: `note_bar(bar_time)` → `check_staleness()` (3× timeframe
   = 3h limit). `ping()` verifies terminal connectivity. `beat()` touches
   the heartbeat file.
4. **Read**: `read_and_featurize(closed, None)` → `states` (MarketState +
   FeatureSet per bar). No peers in live (relstr uses GC=F peers in
   backtest; live uses the symbol's own history — documented limitation).
5. **Playbooks**: for each `pb.live` playbook, `pb.live_signal(states, t)`
   emits a `GoldenSignal` or None. (Currently only `sweep_reversal`
   implements `live_signal`; sleeves are shadow.)
6. **Confluence**: `engine.gate_live_signal(pb, sig, ctx)` applies the
   three gates (session → tripwire → hunting ground). If blocked, the
   reason is journaled (a firing outside declared context is a SYSTEM BUG).
7. **Score**: `engine.score_live_signal(pb, sig, ctx)` → confluence score
   + bonus notes.
8. **Alert**: `_emit_alert()` → chart + Telegram/email + outbox. The alert
   carries the signal, confluence, and context.
9. **Risk**: if `auto_execute`, `registry.dispatch(alert_id, "EXECUTE")`
   → `ExecutionCallbackHandler` → `RiskGuards.check()` → allow/deny.
   Denials are journaled ("refused").
10. **Execute**: `MT5Executor.place_order()` (live) or
    `PaperExecutor` (dry-run). The broker-side SL is placed with the order.
11. **Manage**: `_LiveManager.on_bar()` runs the `ExitStepper` (partials at
    1R, trail via SL moves, time stop). Between bars, the broker SL is the
    protection of last resort.
12. **Journal**: opens, closes, PnL, R-multiples, notes — append-only JSONL.
13. **Persist**: `store.save_bars()`, `save_meta()`, `save_playbook()` for
    warm resume.

### 2.2 The dry-run path (sandbox, `run_bars`)

Identical pipeline, but:
- Feed is `load_gold_h1()` (CSV, 11,426 bars).
- The engine runs in **batch** (`engine.run(ctx)`) to generate the canary
  + signal map.
- The rehearsal loop replays signals bar-by-bar through alert → risk →
  `PaperExecutor` → journal (simulating live decisions, not replaying the
  greedy book).
- **CANARY**: on full data, asserts 199 trades / +28.92% / PF 1.25. If the
  canary fails, the strategies are broken — halt, don't trade.

### 2.3 The legacy path (`use_confluence=False`)

The pre-Phase-10 Combo-A detector (`GoldenSwingDetector`). Available as a
fallback flag, not the default. No confluence, no hunting grounds, no
watchdog (in live). Kept for regression testing and emergency fallback.

---

## 3. Run Modes

| Mode | Method | Host | Feed | Executor | Risk | Use |
|------|--------|------|------|----------|------|-----|
| Dry-run | `run_bars()` | anywhere | CSV | Paper | shadow* | end-to-end proof, canary |
| **Paper loop** | `run_paper_loop()` | anywhere | **yfinance GC=F (live)** | Paper | **enforce** | **live paper trading: feed → live_signal → confluence → alerts → paper fills → journal** |
| Live | `run_live()` | Windows + MT5 | MT5 | MT5 | enforce | production |
| Legacy dry | `run_bars(use_confluence=False)` | anywhere | CSV | Paper | shadow | fallback test |
| Legacy live | `run_live(use_confluence=False)` | Windows + MT5 | MT5 | MT5 | enforce | emergency fallback |

*Shadow mode: risk decisions are computed and logged but don't alter the
historical sequence (the anchor canary stays exact).

**Env config**: `GS_USE_CONFLUENCE=1` (default), `GS_AUTO_EXECUTE=0`
(default — alerts, don't trade), `GS_RISK_PCT=0.01`, `GS_SYMBOL=XAUUSD`.

---

## 4. Risk Calibration (Phase 10)

The old defaults (3 consecutive losses, 10% max drawdown, 2% daily) were
calibrated for a different system. Measured against the anchor book
(199 trades, 45.7% win, 2y):

| Guard | Old | New | Justification |
|-------|-----|-----|---------------|
| `max_consecutive_losses` | 3 | **6** | 3-loss runs hit 14× in 2y (normal); 6+ hit once (tail). A halt that fires in normal operation is a bug. |
| `max_drawdown_pct` | 0.10 | **0.15** | In-sample MaxDD 10.38% — the 10% halt WOULD have tripped. 15% = 1.5× historical max; live breaking the envelope means regime break. |
| `daily_loss_pct` | 0.02 | **0.03** | Worst in-sample day -2.2% at 1% risk — the 2% halt WOULD have tripped. 3% guards the tail. |
| `weekly_loss_pct` | 0.05 | 0.05 | Unchanged; worst -3.1%, never breached. |
| `max_positions` | 3 | 3 | Unchanged (one-at-a-time in practice). |
| `max_total_risk_pct` | 0.03 | 0.03 | Unchanged. |
| `require_stops` | True | True | Unchanged; kill-switch semantics kept. |

**Philosophy**: guards protect the TAIL, not the normal. A latching halt
should fire ~once per 2y on a genuine tail event (the 7-loss run), not
14× on normal variance. All halts latch (require human reset) — a halted
system never resumes itself.

**Reproduce**: `python scripts/risk_calibration.py`

---

## 5. Walk-Forward / OOS Results (Phase 10)

Expanding 12mo train / 3mo test over the 2y data. IS-fit threshold
(grid 40-80, max IS Sharpe, ≥15 trades); OOS scored on next window.

### sweep_reversal (anchor, LIVE) — **FAILS**

| OOS window | Trades | Return | PF | fit_th |
|------------|--------|--------|----|--------|
| 12-15mo | 20 | +3.71% | 1.37 | 60 |
| 15-18mo | 20 | +1.05% | 1.10 | 70 (fixed 60: +7.61%, 1.63) |
| 18-21mo | 28 | -4.32% | 0.77 | 60 |
| 21-24mo | 26 | -0.55% | 0.98 | 60 |
| **Aggregate** | **94** | **-0.28%** | **1.01** | — |

Sharpe 0.04, win 39.4%. **The +28.92% in-sample does not hold up
out-of-sample.** The edge is concentrated in months 0-12; months 12-24
are flat to negative. This is a **blocker for live trading** — see §10.

### fvg_midline (SHADOW) — **SURVIVES** (thin)

Aggregate OOS: 27 trades | +18.82% | PF 2.54 | Sharpe 2.30 | win 59.3%.
Strong but thin. **Caveat**: the hunting ground (trend=down excluded) was
derived in-sample; the OOS test is optimistic for the filter.

### trendline_retest (SHADOW) — **SURVIVES**

Aggregate OOS: 39 trades | +14.03% | PF 1.62 | Sharpe 1.23 | win 46.2%.
Solid across 4 windows (one negative window: 15-18mo, -2.49%).

**Reproduce**: `python scripts/walkforward.py --playbook {name|all}`

**Interpretation**: the "live" anchor is less robust OOS than the "shadow"
sleeves. The promotion checklist (CONFLUENCE.md) now requires: (1) OOS
confirmation on 2022–2024 or second instrument, (2) resolution of the
anchor's in-sample concentration, before any live promotion.

---

## 6. Failure Modes & Mitigations

Every failure mode, what catches it, what happens.

### 6.1 Feed failures

| Failure | Catch | Behavior |
|---------|-------|----------|
| No new bar for 3× timeframe (3h on H1) | `FeedWatchdog.check_staleness()` | Alert + **halt new entries** (management continues). Auto-recovers on fresh bar. |
| MT5 terminal disconnects | `mt5.ping()` each loop | Alert "MT5 TERMINAL LOST" + halt entries until `ping()` succeeds. |
| MT5 `copy_rates` returns empty/garbage | length check (`len(closed) > 100`) | Skip iteration, retry next loop. |
| Network blip (Telegram) | `TelegramSender._post_with_retry` (3 tries, exp backoff, 429 honours `retry_after`) | Retried; photo falls back to text; fatal codes (400/401/403/404) not retried. |
| Corrupt bar buffer on disk | `StateStore.load_bars()` try/except | `load_error` set, journaled, **cold start** (full fetch). |

### 6.2 Strategy failures

| Failure | Catch | Behavior |
|---------|-------|----------|
| Playbook fires outside declared context | `gate_live_signal` 3-gate check | Blocked + journaled as "SYSTEM BUG CANDIDATE". Never executes. |
| Playbook fires outside session scope | `_session_allowed` | Blocked. Session scope is first-class. |
| Signal score below threshold | `sig.score < min_score` | Skipped silently (normal). |
| Two playbooks signal same bar | chronological greedy (batch) / first-wins (live) | One position at a time. |
| Feature computation fails | try/except in reader | Degraded mode; detector falls back (legacy) or skips (confluence). |

### 6.3 Risk failures

| Failure | Catch | Behavior |
|---------|-------|----------|
| Daily loss exceeds 3% | `RiskGuards.check()` | **Latched halt** — no new risk until human resets. |
| Weekly loss exceeds 5% | `RiskGuards.check()` | Latched halt. |
| Drawdown exceeds 15% from peak | `RiskGuards.check()` | Latched halt. |
| 6 consecutive losses | `RiskGuards.check()` | Latched halt (tail guard). |
| Kill-switch file appears | `RiskGuards` file check | Immediate halt, all new risk denied. |
| Position without stop | `require_stops` | Blocks new risk until stopped. |

### 6.4 Execution failures

| Failure | Catch | Behavior |
|---------|-------|----------|
| MT5 order rejected | `place_order` returns error | Journaled; alert sent; no phantom position. |
| Partial close fails | `ExitStepper` error handling | Retried next bar; broker SL remains. |
| Loop process dies | dead-man's armed marker | Next startup alerts "previous loop died uncleanly". |
| Heartbeat goes stale | external watcher / operator cron | "The loop is dead" alert (see §8.3). |
| Power loss on Windows host | — | On reboot: operator starts terminal + runner; StateStore resumes. |

### 6.5 Silent-stop failures (the worst kind)

A trading system that silently stops is worse than no system. Three
independent guards:

1. **Heartbeat file** (`state/live_state/heartbeat.txt`): touched every
   loop. External watcher alarms if older than 5 min.
2. **Dead-man's marker** (`outbox/deadman.log.armed`): armed on startup,
   disarmed on clean shutdown. If armed at startup, the previous loop
   died uncleanly → alert.
3. **Clean-shutdown notice**: `disarm_deadman()` sends "RUNNER STOPPED
   (clean shutdown)". Silence always means something unexpected.

---

## 7. Recovery Procedures

### 7.1 Restart (process died)

1. Check the dead-man's alert: `outbox/deadman.log` (last lines).
2. Check the journal tail: `journal.jsonl` (last records).
3. Check the heartbeat age: `heartbeat.txt` mtime.
4. Start the runner: `python -m goldenswing.run_live` (Windows).
5. The runner: `check_previous_death()` → alerts if unclean → `arm_deadman()`
   → loads bar buffer from `StateStore` → fetches only new bars → resumes.
6. Verify: heartbeat updating, no "died uncleanly" alert, positions match
   MT5 terminal.

**Do NOT**: delete the state dir (loses the buffer — forces cold start).
**Do NOT**: manually edit the journal (append-only).

### 7.2 Reconnect (MT5 terminal lost)

1. The watchdog alerts "MT5 TERMINAL LOST" and halts entries (management
   of existing positions continues via broker SL).
2. Operator: check the terminal (logged in? connected? EA/autotrading
   enabled?).
3. The loop's `ping()` succeeds → entries auto-resume. No restart needed.
4. If the terminal needs a restart: close/reopen, log in, verify connection,
   the loop picks it up on the next `ping()`.

### 7.3 Resume (warm vs cold)

- **Warm resume** (default): `StateStore` loads the 2000-bar buffer, fetches
  only new bars, re-reads. Tail states match full-history to within
  indicator warmup. Playbook working state restored via
  `export_state`/`import_state`.
- **Cold start** (fallback): buffer missing/corrupt → full `copy_rates_h1`
  fetch (1500 bars) → full read. Slower, but the reference implementation.
- **Force cold start**: `GS_COLD_START=1` (wipes the buffer on startup) or delete `state/live_state/bars/`.

### 7.4 Halt reset (risk latched)

1. **Investigate**: why did it halt? (journal, risk_state.json, market
   regime).
2. **Decide**: is the regime broken (stay halted) or was it a tail event
   (reset)?
3. **Reset**: `python scripts/risk_reset.py --reason "..." [--kill-switch]` — requires a journaled reason. The kill-switch file (`KILL`) must also be removed if present.
4. **Document**: journal a note explaining the reset rationale.

**Never** auto-reset a latched halt. A halted system never resumes itself.

---

## 8. State Backup Strategy

### 8.1 What's persisted (and where)

| State | Path | Format | Durability |
|-------|------|--------|------------|
| Bar buffer (2000 H1) | `state/live_state/bars/bars.pkl` | pickle | atomic replace (tmp + rename) |
| Meta (last bar, signal bar) | `state/live_state/bars/meta.json` | JSON | atomic replace |
| Playbook working state | `state/live_state/bars/playbooks/*.json` | JSON | atomic replace |
| Risk state (PnL, halts, peak) | `data/risk_state.json` | JSON | `RiskGuards` owns it |
| Journal (trades, notes) | `data/trade_journal.jsonl` | JSONL | append-only, never rewritten |
| Kill-switch | `data/KILL` | empty file | presence = halt |
| Heartbeat | `state/live_state/heartbeat.txt` | ISO timestamp | touched every loop |
| Dead-man's outbox | `state/live_state/outbox/deadman.log` | text | append-only |
| Alert outbox | `state/outbox/` | PNG + JSON | `AlertService` fallback |

### 8.2 Backup procedure

**Daily** (operator cron or manual):
1. Copy `data/trade_journal.jsonl` → backup (dated). The journal is the
   audit trail — never lose it.
2. Copy `data/risk_state.json` → backup.
3. Copy `state/live_state/` → backup (buffer + meta + playbooks).

**The journal is append-only**: back it up by copying, never by moving.
Old backups can be pruned (keep 30d).

### 8.3 External monitoring (the operator's job)

The system cannot watch itself. The operator (or a second machine) must:

1. **Heartbeat watcher** (cron every 5 min):
   ```bash
   age=$(($(date +%s) - $(stat -c %Y state/live_state/heartbeat.txt)))
   [ $age -gt 300 ] && echo "ALARM: runner heartbeat stale ${age}s" | \
     telegram-notify  # or email, SMS, etc.
   ```
2. **Dead-man's check** (on every runner start — automatic via
   `check_previous_death()`).
3. **Disk space** (the journal grows; alert at 80%).

---

## 9. Deployment Runbook (Windows Host)

### 9.1 Prerequisites

- Windows 10/11 (the $100 EC2 credits host, or any Windows machine).
- MetaTrader 5 terminal installed, logged into a **DEMO** account.
- Python 3.11+ (from python.org, "Add to PATH" checked).
- Git (for the repo).

### 9.2 Install (in order)

1. **Python**: install 3.11+, verify `python --version`.
2. **Repo**: `git clone <repo>` (LOCAL ONLY — no remote yet; copy via USB
   or SCP from the build machine). `cd goldenswing`.
3. **Deps**: `pip install -r requirements.txt` (pandas, numpy, MetaTrader5,
   requests, etc.). Verify: `python -c "import MetaTrader5; print('MT5 OK')"`.
4. **MT5 terminal**: install, log in (demo), enable algorithmic trading
   (the green "Algo Trading" button). Verify: terminal shows "Connected".
5. **Env**: copy `.env.example` → `.env` (or set system env):
   ```
   GS_SYMBOL=XAUUSD
   GS_MODE=swing
   GS_AUTO_EXECUTE=0          # alerts only until validated
   GS_RISK_PCT=0.01
   GS_PAPER_BALANCE=10000
   GS_MT5_LOGIN=<demo login>
   GS_MT5_PASSWORD=<demo password>
   GS_MT5_SERVER=<demo server>
   GS_TELEGRAM_BOT_TOKEN=<token>
   GS_TELEGRAM_CHAT_ID=<chat id>
   ```
6. **Verify MT5**: `python -c "from goldenswing.execution.mt5 import
   MT5Executor; m=MT5Executor(); print(m.connect())"` → `(True, ...)`.
7. **Verify data**: `python -c "from goldenswing.execution.mt5 import
   MT5Executor; m=MT5Executor(); m.connect(); df=m.copy_rates_h1('XAUUSD');
   print(len(df))"` → `1500` (or `bars_count`).

### 9.3 First run (paper mode)

1. `GS_AUTO_EXECUTE=0 python -m goldenswing.run_live --max-iterations 5`
   (or the equivalent entry point).
2. Verify: heartbeat file appears, no errors, journal has "live runner
   connected".
3. Let it run for a full session (24h). Verify: alerts arrive (Telegram),
   no watchdog trips, heartbeat fresh.
4. **Do NOT enable auto-execute** until: (a) 2 weeks of clean paper alerts,
   (b) the anchor OOS issue is resolved (§10), (c) the owner explicitly
   authorizes.

### 9.4 Go-live checklist (all must be YES)

- [ ] MT5 demo validated for 2+ weeks (alerts correct, no missed bars)
- [ ] Anchor OOS issue resolved or anchor demoted (see §10)
- [ ] Sleeves promoted per CONFLUENCE.md checklist (OOS + second window)
- [ ] Risk calibration reviewed by owner (the 6/15%/3% numbers)
- [ ] Kill-switch tested (touch `KILL` → verify halt → remove → verify resume)
- [ ] Watchdog tested (kill feed → verify staleness alert; kill process →
      verify dead-man's on restart)
- [ ] State backup cron running (journal + risk_state + live_state daily)
- [ ] Heartbeat watcher cron running (external)
- [ ] Owner explicitly authorizes real money (never demo-to-real silently)

### 9.5 Paper loop runbook (any host — the live paper-trading system)

The paper loop is the system running live on fake money: real feed →
live signals → Telegram alerts → paper fills → journal. Paper only —
no MT5, no live-order code paths.

**Start it** (needs `yfinance`; no MT5, no Windows):

```
python3 scripts/paper_live.py            # runs until stopped (Ctrl-C)
```

or from Python:

```python
from goldenswing.execution.runner import LiveRunner, RunnerConfig
from goldenswing.feed import PollingFeed
runner = LiveRunner(RunnerConfig.from_env())
runner.run_paper_loop(PollingFeed())      # None = run forever
```

**What it does per closed H1 bar**: `PollingFeed` (warm-started from the
cached 2y history + recent fetch) → each playbook's `live_signal`
(incremental, parity-proven against the backtest scanners) → confluence
gate + score (incl. the +4 trap boost on trendline_retest) → Telegram
alert with narrator + phone-first chart → paper fill at next bar open ±
half spread ($0.45/oz) → `ExitStepper` manages stops/targets → journal
open + close records.

**Who trades**: `fvg_midline` + `trendline_retest` are auto-paper-traded.
`sweep_reversal` (failed walk-forward) runs **shadow** — its signals are
journaled, never filled. One position at a time.

**Feed health**: `PollingFeed.poll_status()` reports ok/stale with reason;
fetch failures back off exponentially and never raise into the loop;
stale polls are journaled. Weekend/market-closed grace is built in
(`market_closed_now()`); the daily 21:00–22:00 UTC break is expected.

**Restart**: the loop fast-forwards each playbook's streaming state over
history on start (signals for past bars discarded) — no state files to
corrupt. The journal is append-only; `scripts/paper_report.py` prints
per-playbook paper OOS stats from it.

**Cron mode** (`--once`): for hosts where a daemon is awkward, run one
tick per hour from cron instead of the daemon loop:

```
# crontab — H1 bars close at minute 0; give the feed a minute to settle
1 * * * * cd ~/workspace/goldenswing && python3 scripts/paper_live.py --once >> data/paper_cron.log 2>&1
```

Each tick processes any new closed bars since the last tick, answers
Telegram commands (`/report`, `/db`, `/status`, `/trades`, `/help` —
owner-only, sharing the button-tap getUpdates poller), then exits.
State persists in `data/paper_state/` (bar buffer, playbook streaming
state, paper executor state incl. open positions, last bar time);
`GS_COLD_START=1` forces a full cold start.  Never run `--once` and
the daemon loop at the same time (or overlapping cron ticks) — both
would process the same bars and double-trade.  Pick one.

**Telegram commands** (owner-only; the chat id in
`GS_TELEGRAM_CHAT_ID`; everyone else is ignored silently):

| Command | What it does |
|---|---|
| `/report` | Builds a fresh results document from the journal + `data/measured_oos.json` and sends it as a file (per-playbook stats, walk-forward summary, last trades) |
| `/db` | Sends the paper journal database file (forward it to chat) |
| `/status` | Feed health (ok/stale + reason), open paper positions with current R, today's P&L, last signal time |
| `/trades [n]` | Last n closed paper trades with R multiples (default 5, max 20) |
| `/help` | Command list |

Files go out via `sendDocument` (bot limit 50MB — larger journals are
refused up front, honestly).  Adding a command is one decorated
function in `goldenswing/alerts/commands.py` — see its module docstring;
`/help` picks it up automatically.  Commands and the Execute/Paper/Skip
button taps share one `getUpdates` consumer (`LiveRunner._poll_telegram`);
never add a second poller.

**Prove it**: `python3 scripts/paper_rehearsal.py` replays 500 recent bars
as-if live and asserts signal → alert → fill → managed close → journal.
It must pass before any claim about the loop.

---

## 10. What Blocks Live Trading (honest list)

1. **The anchor FAILS walk-forward** (§5). +28.92% in-sample, -0.28% OOS
   (94 trades, PF 1.01). The edge is concentrated in months 0-12. **Do not
   trade the anchor live until this is resolved.** Options: (a) investigate
   the regime break, (b) demote anchor to shadow, promote a sleeve, (c)
   re-derive on 2022–2024 data.

2. **Sleeves need OOS promotion**. fvg_midline (+18.82% OOS, 27 trades) and
   trendline_retest (+14.03% OOS, 39 trades) survive walk-forward but need:
   (a) 2022–2024 or second-instrument confirmation, (b) CONFLUENCE.md
   checklist sign-off. `live_signal` is now implemented for both with
   parity tests (`tests/test_paper_loop.py`), and the **paper loop**
   (§9.5) grows their OOS counts with live-sampled data toward the
   50-trade bar.

3. **Windows host + MT5 demo validation**. The live path (`run_live`) has
   never run against a real terminal. First runs are DEMO ONLY. The
   `MT5Executor` is code-complete but unproven.

4. **Credentials**. The repo has exposed credentials/token in history —
   rotate before any live deployment.

5. **Owner authorization**. Explicit, per the standing rule: demo before
   real, always. No silent promotion.

---

## 11. Module Reference

| Module | Purpose |
|--------|---------|
| `goldenswing/execution/runner.py` | `LiveRunner`: `run_bars` (dry-run), `run_live` (Windows), `run_paper_loop` (live paper), legacy fallbacks |
| `goldenswing/feed/` | `PollingFeed` (yfinance GC=F, warm-started, stale-aware) + `ReplayFeed` (rehearsal) |
| `goldenswing/execution/watchdog.py` | `FeedWatchdog`: staleness, heartbeat, dead-man's |
| `goldenswing/execution/state_store.py` | `StateStore`: bar buffer, meta, playbook state (warm resume) |
| `goldenswing/execution/risk.py` | `RiskGuards`, `RiskConfig` (recalibrated §4) |
| `goldenswing/execution/mt5.py` | `MT5Executor` (+ `ping()` for watchdog) |
| `goldenswing/execution/paper.py` | `PaperExecutor` (dry-run fills) |
| `goldenswing/execution/manager.py` | `ExitStepper`, `_LiveManager` (position management) |
| `goldenswing/execution/learning.py` | `TradeJournal` (append-only) |
| `goldenswing/confluence.py` | `ConfluenceEngine`: 3-gate enforcement, scoring |
| `goldenswing/playbooks/` | Three playbooks + `base.py` (contract) |
| `goldenswing/contextmap.py` | Measured context maps, hunting grounds |
| `scripts/walkforward.py` | OOS harness (§5) |
| `scripts/risk_calibration.py` | Risk parameter derivation (§4) |
| `scripts/run_confluence_book.py` | Canary + shadow books (CI) |
| `scripts/paper_rehearsal.py` | Paper-loop end-to-end proof (replay as-if live) |
| `scripts/paper_report.py` | Per-playbook paper OOS stats from the journal |
| `scripts/paper_live.py` | Entry point: start the live paper loop (`--once` = one cron tick) |
| `goldenswing/alerts/commands.py` | Owner-only Telegram commands (`/report` `/db` `/status` `/trades` `/help`); shares the getUpdates poller |
| `goldenswing/execution/reporting.py` | Paper-book stats + report-file generation (journal → document; powers `/report` and `paper_report.py`) |
| `data/measured_oos.json` | Versioned measured-OOS records with provenance (read by `/report`, never hardcoded in logic) |
| `tests/test_paper_loop.py` | live_signal parity vs backtest scanners |

---

## 12. What's Still Weak (the footer)

- The anchor's OOS failure is the elephant. The system is worthy; the
  live strategy isn't (yet). The sleeves are more robust — that's backwards
  from the "anchor = live, sleeves = shadow" narrative.
- The paper loop is proven on replay, not on the live yfinance feed yet —
  first live run should be watched for feed quirks (forming-bar timing,
  DST/index oddities, weekend gaps).
- Paper fills assume next-bar-open ± half spread; the FVG sleeve's edge
  was measured on resting-limit fills *at the midline* — adverse selection
  between signal bar and fill bar is not modeled.
- The heartbeat watcher is operator-owned (no built-in cron).
- Peers not available in live (`read_and_featurize(closed, None)`) —
  relstr features degrade. Documented, not fixed.
- 2y of H1 is thin for walk-forward (4 windows). 2022–2024 data would
  strengthen the OOS case.
- yfinance is the single live feed source — no fallback provider; a
  prolonged yfinance outage halts new bars (positions still managed).
