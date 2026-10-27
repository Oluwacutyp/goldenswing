# GoldenSwing Research Build — Hot Scalp v1

Build ID: `GS-HOT-SCALP-RESEARCH-2026-10-27`

## What is implemented in this source bundle

- A new correlation-aware evidence ledger (`goldenswing/live/evidence.py`) with explicit source/time/timeframe/correlation group/stance, neutral handling of unavailable evidence, deduplication and category caps. One common set of provisional research category weights is shared with Hot Scalp.
- A SQLite-backed setup lifecycle store (`goldenswing/live/setup_store.py`) with stable setup identity, atomic trigger claims, fencing tokens, event idempotency and explicit broker-reconciliation handling for ambiguous claims.
- A separate Hot Scalp shadow sleeve (`goldenswing/live/hot_scalp.py`) with three real setup families: pullback continuation into an unfilled imbalance, confirmed liquidity sweep/reclaim, and breakout/retest. It supports persistent structures from closed history and live tick reactions, and emits only shadow candidates.
- A risk-aware stacking policy (`goldenswing/live/stacking.py`) for pyramiding and multiple distinct entries under one parent, with a maximum of three layers, shared risk budget, fresh independent triggers, a verified broker stop, and no additions while the parent is losing. This is a reusable policy component, not a broker order route yet.
- A separate, opt-in M1 feed stream, kept outside `TimeframeStack`, refreshed on its own minute clock and used by Hot Scalp. The hierarchy roster itself is unchanged.
- M1 EMA(9)/EMA(21) alignment as small, non-gating evidence; directional interpretation of volume expansion; session state logged even outside a killzone; a fix for the unreachable New York silver-bullet label.
- Common evidence attached to all emitted, non-management MT5 signal records via `signal_log.py`: per-signal score, evidence items, correlation groups, data timestamps/ages, session and explicit non-gating status. Sensor links include M15/M5 swing structure, BOS/CHoCH/MSB, FVG, nearby order blocks, liquidity pools + confirmed sweeps, recent named candle patterns, premium/discount, directional volume, M1 EMA(9)/EMA(21), session and live spread vs ATR. This is wired across existing signal logger call sites, but does not rewrite their decision scores or execution rules.
- Fixed datetime-index assumptions in order-block/FVG detectors (integer position vs timestamp fields).
- A no-lookahead tick replay CLI (`scripts/replay_hot_scalp.py`) that consumes historical M1/M5/M15 candles and bid/ask ticks without connecting to MT5 or dispatching orders.
- Regression tests for all of the above.

## Validation performed

- `python -m compileall -q goldenswing scripts/replay_hot_scalp.py`: passed.
- Targeted scoring / persistence / scalp / replay / pattern / session / feed / confluence suites: **122 passed** before shared signal migration.
- Shared signal evidence + signal logger regression suites: **10 passed** (includes 4 new shared-evidence tests and existing signal-log tests).
- Existing execution / risk / layering / native-data / safety / runner / owner-ruling suites: **115 passed**.
- MT5 runner construction smoke test: passed, Hot Scalp research engine initializes on a temporary shared SQLite path.
- `scripts/replay_hot_scalp.py --help`: passed.

The entire repository test suite was not clean in this container. Failures encountered are environmental or pre-existing fixture assumptions: `tests/test_alerts_infra.py` hardcodes `/home/hatch/workspace/goldenswing`; `tests/test_backtest_notify_telegram.py` expects local historical CSVs that are absent; `tests/test_journal_upcasting.py` references a missing `tests/fixtures/trade_records_v1.json`; and `tests/test_runner_confluence.py` requires `yfinance`, which is not installed. These failures are not claimed as passes, and the targeted results do not substitute for a full clean run in the project environment.

## Not yet implemented / not to be claimed

- No Hot Scalp order placement. It has no order-dispatch path and remains `shadow_only`.
- The stacking evaluator is implemented and tested, but the live Hot Scalp runner does not yet feed it authoritative parent P&L, normalized broker-stop exposure and the live shared risk budget; it therefore does not execute layer additions.
- The common evidence ledger is journaled for emitted signals across existing MT5 playbooks, but is intentionally non-gating and has not replaced their separate strategy logic, scores or execution decisions. The per-playbook logic still needs outcome-based review and a controlled migration. It is not yet an empirically validated confluence score.
- DST-aware sessions are used, but a unified broker trading calendar / symbol-holiday oracle / explicit broker-open agreement is not complete.
- The source archive has no Git metadata; this build has not been proven identical to the running EC2 `ed317ab` source and has not been deployed to EC2.

Keep demo order authority disabled until order lifecycle, risk consistency, shared setup ownership, restart/concurrency safety, replay and regression acceptance all pass against the exact runtime source.
