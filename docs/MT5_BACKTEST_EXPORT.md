# MT5 native backtest data — auto-fetch, no resampling

The full-engine backtest (`scripts/backtest_mt5_full.py`) runs the REAL
`MT5Runner` on NATIVE per-timeframe data. Every TF's bars come from
that TF's own file, or are aggregated from the MT5 tick stream
(flagged `built_from_ticks`). There is no resampling anywhere on
this path, and there is **no manual CSV step**.

## Auto-fetch (the only path)

On start, the backtest fetches the data itself from the running MT5
terminal (`goldenswing/backtest/mt5_fetch.py`):
`copy_rates_range` per TF + `copy_ticks_range` (one tick stream),
cached to `data/mt5_native/` so a second run is instant. One command:

```powershell
python scripts\backtest_mt5_full.py --out data\mt5_bt_out
```

If MT5 is not running / not logged in, it FAILS with diagnostics
(terminal path / account / probe) — never a manual-export script.

If several MT5 terminals are installed, bare `initialize()` can
attach to a DIFFERENT terminal (different data path, no history,
wrong broker) and `copy_rates` then returns 0 even though the chart
on the running terminal shows years of data. Pin the running
terminal with `--terminal-path "C:\...\terminal64.exe"` (or the
`MT5_TERMINAL_PATH` env var) and verify via
`terminal_info().path / .data_path` and `account_info().login`.

## Tick window (broker tick history is limited)

Broker tick history is LIMITED: bars go back to 2023, ticks only
from about Jan 2026. Requiring every TF to share one identical
3-year window blocks forever — that was the empty-intersection
block bug.

The run window is the **tick window**: [first tick, last tick].
Every TF must cover the tick window — native bars when they cover
it, tick-built bars (flagged `built_from_ticks`) when the native
file is missing or does not cover it (e.g. a stale file ending
Dec 2025 while ticks run to Oct 2026). The report states the tick
window explicitly ("ticks available 2026-01-01..2026-10-16, bars
2023-06-28..2026-10-16, full-engine tick run on tick window").
A cache that covers the tick window is reused; re-fetching would
return the same limited ticks.

## Rules

- Closed bars only (`copy_rates_range` returns closed bars).
- Tick-build fallback (any TF): if the terminal has no cached
  bars for a TF's window but HAS ticks, that TF's bars are
  aggregated from the tick stream itself (Bid OHLC per bucket)
  and flagged `built_from_ticks` in the provenance. This is NOT
  resampling from H1 — it aggregates the tick stream, the same
  source MT5 builds bars from. No ticks at all still BLOCKS —
  with diagnostics (terminal path / account / probe).
- Do NOT stitch across history holes. If the terminal returns gapped
  history, the loader reports segments and the largest gap per TF —
  see `scripts/backtest_mt5_full.py --coverage`. NOTE: in MT5 there
  is NO History Center (that is MT4 — in MT5, F2 is Task Manager).
  History comes via chart scroll / API request.
- Ticks: `COPY_TICKS_ALL`. Without ticks the harness can smoke-run on
  the labelled M1 proxy (`--allow-m1-proxy`), but that is NOT a tick
  backtest and is always flagged `ticks: m1_proxy` in the report.

## What "no resampling" means here

MT5 ticks are one stream. TF bars are MT5's own per-TF `copy_rates`
or tick-stream aggregations. "MT5 ticks for every timeframe" = real
tick stream + bars for every TF. Resampling H1 → H4/D1 fabricates TF
structure (own opens/closes, session boundaries, Sunday candles,
gaps) and is refused on this path.
