# EC2 MT5 Engine — Runbook

> AUTHORITATIVE RULINGS: `OWNER_RULINGS.md` (repo root) is the tiebreaker.
> Where this runbook disagrees with it, this runbook is stale. In
> particular: there is NO fixed-lots mode, NO `fixed_lots` field, and NO
> `--fixed-lots` flag (revoked 2026-10-23); dynamic risk sizing is the
> sole sizing path, and a below-minimum size floors at the 0.01 broker
> minimum instead of abandoning (ruling 2026-10-24).

One Windows EC2 host. One engine. MT5 is the source of truth for
EVERYTHING: prices, bars, ticks, volumes, spread, account state,
open positions. No OANDA cross-check, no second source. (OANDA stays
only for the local paper loops until this engine takes over.)

## Already done on the host (2026-10-14)

1. Python 3.11 installed, "Add python.exe to PATH" ticked.
2. Git installed.
3. Exness MT5 installed, logged into the $12 demo account, XAUUSD
   chart live.

## 1. Get the engine code

```powershell
cd $HOME
git clone https://github.com/Oluwacutyp/goldenswing.git
cd goldenswing
```

## 2. Dependencies

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # if present; else: pip install -e .
pip install MetaTrader5 pandas numpy
```

`MetaTrader5` is Windows-only — that is why the engine runs here,
not on a Linux box.

## 3. Start the engine (demo)

The MT5 terminal must be running, logged in (demo), and ACTIVE.
The engine attaches to the terminal as-is — no credentials needed.

```powershell
.\.venv\Scripts\python.exe scripts\run_mt5.py
```

For the $12 demo sizing (dynamic risk sizing is the sole path — no
fixed-lots flag exists):

```powershell
.\.venv\Scripts\python.exe scripts\run_mt5.py
```

Rehearsal (1000 ticks, then stops):

```powershell
.\.venv\Scripts\python.exe scripts\run_mt5.py --max-ticks 1000
```

## 4. Verify it is alive

- First lines: `MT5 runner: started (...)` with per-TF bar counts,
  `account login=... equity=... trade_mode=0` (0 = demo).
- Journal: `data\mt5_journal.jsonl` gains `mt5 tick:` /
  `mt5: hierarchy...` notes; hierarchy-blocked signals are EXPECTED
  (the hierarchy blocks most H1-only signals — that is the design).
- Telegram: signal / fill alerts arrive with the `MT5` tag.
- A breakeven move shows as `mt5 tick: ticket ... SL -> ... (breakeven)`
  on the tick that crossed +1R — not minutes later.

## 5. Keep it running

Windows Task Scheduler: create a task that runs
`...\.venv\Scripts\python.exe scripts\run_mt5.py` on system startup,
"run whether user is logged on or not", restart on failure.
Do NOT run a second copy — one engine, one journal.

Stop it: create the kill file (`data\mt5_KILL`) or Ctrl+C in the
console. The engine stops at the next tick.

## 6. Telegram commands (this engine owns getUpdates)

Command polling is ON by default — the MT5 engine IS the engine.
`/status` / `/report` / `/db` / `/trades` / `/harvest` / `/ticks`
are answered from the MT5 journal/state (`data/mt5_*`), never the
OANDA loop's files.  Only the owner chat may use any of this;
everyone else is ignored (commands) or refused (buttons).

### Persistent bottom menu

`/start` (or `/menu`) shows the always-visible reply-keyboard menu
under the message box — it stays there across messages (Telegram
`is_persistent` + `resize_keyboard`):

```
[Status]    [Positions]  [Pendings]
[Playbooks] [Config]     [Trades]
[Harvest]   [Pause]      [Resume]
```

Each button runs the same handler as its slash command.  The menu
is NOT re-sent with every reply (a message carries exactly one
`reply_markup`, and the inline buttons below need theirs) — sending
a message with an inline keyboard leaves the reply keyboard in
place, so both surfaces coexist.

### Inline buttons on status / position / pending messages

`⏸ Pause` `▶ Resume` `✂️ Trim` `❌ Cancel pending` `🔄 Refresh`
`⚙️ Config`.  Refresh / Config re-render the message in place.

### Confirm before anything mutates

Pause, Resume, Trim, Cancel pending, and every `/set` change
require a Confirm tap: the button / command sends a "confirm?"
message with Confirm / Cancel buttons, and NOTHING acts until
Confirm is tapped.  Cancel aborts.  Read-only commands
(`/status`, `/config`, `/report`, `/db`, `/trades`, `/ticks`,
`/harvest`, `/position`, `/pending`, `/playbooks`) need no confirm.

`/trim` closes the top (most recent) added layer early while the base
layer rides — the manual version of "close some".  With only the base
layer it replies "nothing to trim".

### /status answers "why no trades"

`/status` shows engine state (RUNNING / PAUSED), positions WITH
layers, pending orders, today's P&L, BLOCKED-LEG STATS (per-leg
block counts + last blocked reason), manipulation-filter blocks,
and per-playbook signal counts.  `/position` and `/pending` show
the same positions / pendings slices with the action buttons.
`/playbooks` shows every playbook: executing vs shadow, on/off,
signal counts.

### /config / /set (safe keys only, persisted)

`/config` shows the live config.  `/set <key> <value>` changes one
safe key at runtime (confirm required) and persists it to
`data/mt5_config.json` — it survives a restart (the runner loads
the file at startup):

```
sizing               dynamic risk sizing only (no fixed-lots mode exists)
max_layers           0–3
manipulation_filter  on/off (the liquidity-test entry filter)
pending_expiry_bars  1–24 (applies to pendings placed after the change)
playbook.<name>      on/off (only playbooks that exist)
```

Pause is NOT a `/set` key — use Pause / Resume.  There is NO key,
command, or button for live-trading permission, tokens, or paths:
`/set` rejects them, and no command prints them.  Pause blocks new
signals only — tick management (breakeven / trail / partial) of the
held position keeps running.

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_mt5.py   # polling ON (default)
```

### Cutover: switching the poller from the OANDA loop to MT5

ONE getUpdates consumer only — two pollers steal each other's
updates. The cutover is:

1. **Overlap window** (OANDA loop still running): start MT5 with
   polling OFF so the OANDA loop keeps owning updates:
   ```powershell
   .\\.venv\\Scripts\\python.exe scripts\\run_mt5.py --no-poll-commands
   ```
2. **Cutover**: stop the OANDA loop's polling by setting
   `GS_TELEGRAM_POLL=0` in the OANDA loop's environment (or stop
   the OANDA loop entirely), then restart the MT5 engine WITHOUT
   the flag — polling ON by default, MT5 owns getUpdates.
3. Verify: send `/status` from the owner chat. The reply header
   must read `MT5 ENGINE STATUS`. If it reads `PAPER LOOP STATUS`,
   the OANDA loop is still polling — set `GS_TELEGRAM_POLL=0` and
   retry.

### /harvest (MT5 bundle)

`/harvest` sends one ZIP: MT5 journal + MT5 risk state + tick CSVs
+ RESULTS.md (trades, win%, total R, per-playbook, open positions)
+ manifest.json (sha256 per file). Same caps as the OANDA harvest:
20MB/file, 45MB ZIP, slim fallback (journal + RESULTS.md +
manifest). Chart PNGs are excluded by design (they already ride
the Telegram alerts).

```
/harvest          all (journal + state + tick CSVs + results)
/harvest journal  journal only
/harvest state    risk state only
/harvest ticks    tick CSVs only (the backtest feed)
/harvest results  RESULTS.md + manifest only
```

### /ticks (tick-history export for backtesting)

```
/ticks     last 1 day of terminal tick history as CSV
/ticks 3   last 3 days (max 7)
```

Rows are exactly as seen — no resampling. The live engine also
logs EVERY processed tick to per-day CSVs automatically:
`data\mt5_ticks_YYYYMMDD.csv` (time, bid, ask, tick_volume,
real_volume, spread), next to the journal. Disable with
`--no-tick-log`.

### Charts on signals and fills

Every MT5 signal alert and every fill generates a chart PNG
(`data\mt5_charts\<alert_id>[ -fill].png`) — three stacked panels: H4
structure (candles + H4 POI zone band + D1/H4 bias), H1 trigger (trigger
bar marked), entry timeframe (M15/M30, entry bar marked), all sharing
entry / stop / targets / 2R lines, plus the hierarchy verdict (D1/H4
bias, trigger, entry clock, AMD, QM, volume) and the actual fill price
on fill charts. The PNG is attached to the Telegram
message as a photo; if the photo fails, the text still arrives.

### /terminal (interactive terminal, step 8)

Send `/terminal` and the engine starts its own read-only chart page
and replies with the URL (default `http://127.0.0.1:8899/`). Open it
in a browser ON this box (RDP + localhost) — the server binds
localhost only, so nothing is exposed to the network. The page shows
live stack candles (M5–D1, TF switch fetches that timeframe), study
toggles (sessions, nearby levels, Fib+OTE, premium/discount, signal
zone — the studies come from the latest signal's chart and are
labeled with it), zoom/pan/crosshair, and a freshness banner (LIVE /
STALE / MARKET CLOSED, 30s polling). It never trades: there is no
order or control endpoint, and write methods answer 405. Host knobs:
`GS_TERMINAL_HOST` / `GS_TERMINAL_PORT` (a port opening or tunnel is
a host network change, never a default).

## Safety — read before touching

- DEMO-ONLY by default. The engine REFUSES a real MT5 account
  (`trade_mode=2`) unless started with `--allow-live`. Never pass
  `--allow-live` without thinking.
- One position at a time. Max one open trade with the engine's magic.
- One getUpdates consumer: MT5 polling ON by default; OANDA
  polling OFF at cutover (`GS_TELEGRAM_POLL=0`). Never run both
  pollers.
- Server time: the feed calibrates the MT5 server -> UTC offset
  from a fresh tick on connect, so bars are stamped UTC and the
  session killzones (New York wall time) read the right clock.
  If your broker's offset is known, set `GS_MT5_*` env vars are NOT
  for the offset — the offset is auto-detected; only set
  `GS_MT5_LOGIN` / `GS_MT5_PASSWORD` / `GS_MT5_SERVER` when the
  terminal itself is NOT logged in.

## What this engine does NOT do (yet)

- MT5 tick history backtest: the hierarchy backtest still runs on
  CSV frames; feeding it MT5 `copy_ticks` history is a follow-up.
  (Tick CSVs from `/harvest` / `/ticks` are the feed for it.)
- Peer bars (silver/dollar RS features) from MT5 — RS is OFF on
  this path (peers=None), never faked.
- Real-money execution: `--allow-live` exists but has never been
  exercised against a real account. Demo first, always.
