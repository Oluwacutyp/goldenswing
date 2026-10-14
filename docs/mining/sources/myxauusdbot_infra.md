# Mining: myxauusdbot infrastructure (v18)

**Source:** `~/workspace/user/files/myxauusdbot__4__1_gn20.py` (21,900 lines,
the newer of the two bot files; `__2__` is the older sibling with the same
architecture). Mined 2026-10-11 for INFRASTRUCTURE PATTERNS ONLY.

> **Security note:** both files contain plaintext credentials (API keys,
> tokens). This document records patterns only — zero secret values, zero
> chat ids, zero keys. Config references use `GS_*` placeholder names.

## Worth porting (adopted or queued)

### 1. Photo → text fallback on send failure
`_send_photo_safe`: if `sendPhoto` fails (non-200 or exception), it falls
back to `_send_message_safe(caption)` — the thesis text still arrives even
when the chart doesn't. Our `TelegramSender.send` has outbox fallback but
no photo→text fallback. **Adopted.**

### 2. Error dedup with a time window (STAGE-10 anti-flood)
`notify_error`: dedupes repeated errors keyed on the message's first 80
chars, 600s window, so a stuck condition can't flood Telegram every cycle.
Our runner has no error dedup — a wedged feed could buzz the phone
indefinitely. **Adopted** as `alerts.ErrorDeduper`.

### 3. Notification tiering (silent vs loud)
Non-urgent messages (`notify_status`, `notify_balance`, rejected signals)
go out with `disable_notification=True`; errors go loud. The phone buzzes
only for things that need a human. **Adopted** in the position notifier and
daily summary (quiet) vs risk halts / errors (loud).

### 4. Trade-update notifications with per-action semantics
`notify_trade_update`: distinct emoji + text for partial close, profit-lock
SL move, breakeven SL move, and full close with PnL + R. This is the
position-lifecycle chatter a real desk expects. Our `_LiveManager` did all
of this silently. **Adopted** as `alerts.PositionNotifier`, wired into
`_LiveManager.on_bar`.

### 5. Rejected-signal visibility (optional flag)
`notify_signal` can report REJECTED signals (direction, score, reason) when
`notify_rejected_signals` is on — the learning loop made visible. We journal
refusals already; the alert is the new part. **Queued** (flag-gated, default
off — a rejection-heavy day would be noise).

### 6. Confirm/reject inline flow with a pending-signal registry
`notify_signal_with_confirmation` + `handle_swing_confirmation_callback`:
signals sit in `pending_swing_confirmations`, buttons confirm/reject, and
callbacks for expired signals return `status: "expired"` instead of acting.
Our callback registry has EXECUTE/PAPER/SKIP but no pending-state machine —
an EXECUTE tap on a stale alert executes a stale thesis. **Queued**: add
alert-expiry to `CallbackRegistry.dispatch` (alert older than N bars →
refuse with "expired").

### 7. Health monitor with latency → slippage protection
`HealthMonitor.monitor_health`: pings MT5 every 60s, measures tick latency,
and flips a global slippage-protection flag when latency breaches the
threshold. Our watchdog pings connectivity but doesn't measure latency.
**Queued**: latency measurement in `FeedWatchdog`; the protection flag is a
larger execution change, parked until the MT5 path is proven on demo.

### 8. State persistence hygiene
`StatePersistence.save`: atomic tmp+rename, 30s throttle between writes,
version stamp in the payload, `setdefault` backfill on load so old state
files keep working. Our `StateStore` already does atomic replace; the
**throttle + version stamp + backfill** are worth adopting.

### 9. Media-group fallback chain
`notify_signal_with_confirmation`: study package (3 charts) → text with
keyboard → plain `notify`. Every layer has a fallback. Our pipeline:
chart+buttons → outbox. **Partially adopted** via photo→text fallback;
media groups (multiple charts per alert) are queued — one chart is the
house style until a second chart earns its place.

## Deliberately NOT ported (and why)

- **The ML stack** (TransformerEncoder, LSTMSequenceModel, RLTradingAgent,
  DeepLearningSignalGenerator, ~15 strategy classes): unmeasured deep
  models are exactly the overfitting factory the doctrine warns about. Our
  forecaster is the measured model (1d/3d AUC ~0.52, rank scores only).
  Nothing here has a measured edge on our data — porting it would be
  importing someone else's unverified beliefs.
- **News machinery** (NewsProtector, NewsAdaptiveExecution,
  CPIEconomicShockHandler, DynamicNewsStraddle): event-trading is a
  different game with different data feeds. Unmeasured on gold H1.
- **Execution optimizers** (TWAP/VWAP/Iceberg/ExecutionOptimizer,
  TierExecutionManager): built for size we don't trade. Our fills are
  single market orders at 1% risk — the optimizer would be theater.
- **SQLite trade store** (`Store`, 38 methods): our append-only JSONL
  journal is the audit trail. A second store is a second source of truth —
  the exact thing the journal design avoids.
- **Plotly charting** (DynamicPlotlyCharter, VisualSuite): our matplotlib
  charts are the house style, dependency-light, and already carry the
  thesis. Plotly adds a heavy dep for no measured gain.
- **ZScoreMeanReversion / LiquiditySentinel / MacroTrendSentinel strategy
  classes**: strategy pile, not infrastructure. Strategies come from the
  claims ledger, not from another bot's corpse.
- **LLMGovernor**: an LLM in the execution path with no measured role.
  The doctrine is explicit: no model in the loop without a measured job.
