# Phase 7 — Chart Drawing + Alerts

## What was built

**Chart engine** (`goldenswing/charts/`) — `render(df, state, signal,
style=...) -> bytes` (PNG, 1600×1000, deterministic: same input renders
byte-identical output — pinned size/DPI/metadata, no wall-clock text).

Four chart classes, per the owner spec ("TradingView level, 4 classes,
volume and all — study-able"):

| style | look |
|---|---|
| `classic` | TradingView-light candlesticks |
| `heikin_ashi` | Heikin-Ashi candles, same light theme |
| `line` | close line/area with full structure overlay |
| `blueprint` | dark study view — candles subdued, levels/zones full strength |

Drawn on every chart (when the state carries it): volume panel; swing
markers labelled HH/HL/LH/LL; liquidity pools (dashed, BSL/SSL +
strength); FVG zones and order blocks (shaded from creation bar);
premium/discount shading of the dealing range + EQ line; killzone
session shading; trendlines (Phase-4 family); Bollinger bands and the
Ichimoku cloud (when the aligned FeatureSet history is passed);
Fib levels when price is at/near them (always on blueprint); entry
zone, ENTRY / STOP / T1–T3 / 2R lines; and a monospace annotation box
with the whole thesis: direction, score (★ GOLDEN), setup kind, mode,
entry/stop/targets, risk distance in ATR, up to 8 backing factors, and
the forecaster line (ensemble prob + rank) when available.

**Alert service** (`goldenswing/alerts/`):

- `Alert` model + `build_alert_text` — the full thesis as plain text
  (Telegram caption / email body / outbox file).
- `TelegramSender` — Bot API `sendPhoto` via `requests`, inline
  keyboard [✅ Execute] [📄 Paper] [⏭ Skip]. Env: `GS_TELEGRAM_BOT_TOKEN`,
  `GS_TELEGRAM_CHAT_ID`.
- `EmailSender` — `smtplib` (STARTTLS, or implicit SSL on 465), chart
  PNG attached. Env: `GS_SMTP_HOST/PORT/USER/PASSWORD`,
  `GS_ALERT_EMAIL_TO`.
- **Offline outbox fallback (both senders):** with credentials unset,
  sends write `<id>.<channel>.png/.txt` + `<id>.json` to the outbox
  dir (`GS_ALERT_OUTBOX` or `data/alert_outbox/`) and report
  `delivered=False, mode="outbox"`. Nothing is ever faked as sent;
  live-mode API/network failures also report `delivered=False` with
  the error.
- **Callbacks** (`alerts/callbacks.py`) — `gs:<action>:<alert_id>`
  callback_data, `CallbackRegistry.dispatch`, and the
  `CallbackHandler` protocol the EXECUTION phase implements
  (Execute → MT5-python executor, Paper → paper book, Skip → learning
  log). No default handler ships: a tap with no handler registered
  returns `handled=False` with the reason — a button can never
  silently pretend a trade happened.
- **Dedup/cooldown** (`AlertService`) — same (instrument, direction)
  does not re-alert within 24 bars (configurable); an alert_id never
  fires twice. Cooldown is checked before rendering.

## Demo evidence (real data, offline by design)

`scripts/alert_demo.py` strips alert credentials from its own process,
runs the production config (Combo A + Phase-2 flags) over the cached
2y GC=F H1 series (11,426 bars → 467 signals, 352 golden), and:

- renders the best signal (bar 8462, LONG continuation, score 100) in
  all four styles → `samples/chart_{classic,heikin_ashi,line,blueprint}.png`
  (237–254 KB each; verified 1600×1000, std 0.11–0.38, ~1,500 unique
  sampled colours — rich, non-blank);
- pushes 3 golden signals through `AlertService` → `samples/outbox/`
  (chart + text + JSON per alert, per channel):
  `XAU2090S` (short, 78), `XAU6132S` (short sweep-reversal, 83 —
  thesis includes sweep, BOS confirm, compressed vol, distribution
  base, RVOL 4.9, forecaster rank top 31%), `XAU10352L` (long, 67).

## Tests

`tests/test_charts.py` + `tests/test_alerts.py` — 17 tests: every
style renders a real non-blank PNG of exact size; render determinism
(byte-identical); Heikin-Ashi math; alert-text thesis contents;
callback round-trip + malformed data; keyboard wiring; registry
dispatch incl. the no-handler and unknown-alert paths; outbox
fallback file outputs for both senders; live/outbox mode gating on
credentials; service dedup/cooldown/seen-id behaviour.

Full suite: **139 passed, 1 failed** — the failure is pre-existing and
unrelated to Phase 7: `tests/test_forecast.py::test_walk_forward_smoke`
needs `scikit-learn` (the optional `forecast` extra), which is not
installed in this sandbox and cannot be pip-installed (PEP 668
externally-managed environment).

## Known gaps

- Charts draw the reader's swing snapshot (most recent swings), not a
  full-history swing redraw — the relevant swings near the signal are
  all present; very old window bars may lack markers.
- Telegram caption is capped at 1024 chars by the API; long factor
  lists are truncated with "…" in the caption only (the full thesis
  still goes out via email and the outbox JSON).
- `getUpdates` polling for button taps belongs to the live runner
  (Phase 8+); this phase defines and tests the dispatch interface it
  will feed.
