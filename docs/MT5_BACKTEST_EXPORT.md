> AUTO-FETCH (2026-10-16): you do NOT need this manual export anymore.
> `scripts/backtest_mt5_full.py` fetches the data itself from the running
> MT5 terminal on start (`copy_rates_range` per TF + `copy_ticks_range`),
> cached to `data/mt5_native/`. One command: `python scripts/backtest_mt5_full.py --out data/mt5_bt_out`.
> This document is kept as the manual fallback / reference for the exact
> calls the auto-fetch makes.

# MT5 native backtest export — no resampling

The full-engine backtest (`scripts/backtest_mt5_full.py`) runs the REAL
`MT5Runner` on NATIVE per-timeframe data. Every TF's bars come from
that TF's own file. A missing TF fails loudly — there is no
resampling fallback anywhere on this path.

## What to export

On the EC2 MT5 terminal (or any terminal with full history):

```python
import MetaTrader5 as mt5
import pandas as pd
mt5.initialize()
SYMBOL = "XAUUSDm"  # broker symbol on that terminal (Exness)
FROM = pd.Timestamp("2024-10-01", tz="UTC")
TO = pd.Timestamp("2026-10-12", tz="UTC")
TFS = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
       "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
       "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
       "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
       "MN1": mt5.TIMEFRAME_MN1}
for name, tf in TFS.items():
    rates = mt5.copy_rates_range(SYMBOL, tf, FROM.to_pydatetime(),
                                 TO.to_pydatetime())
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.to_csv(f"MT5_XAUUSDm_{name}_native.csv", index=False)
# Ticks — a SINGLE stream, not one per TF:
ticks = mt5.copy_ticks_range(SYMBOL, FROM.to_pydatetime(),
                             TO.to_pydatetime(), mt5.COPY_TICKS_ALL)
tdf = pd.DataFrame(ticks)
tdf["time"] = pd.to_datetime(tdf["time"], unit="s", utc=True)
tdf.to_csv("MT5_XAUUSDm_ticks.csv", index=False)
```

Copy the 10 CSVs to `goldenswing/data/` and re-run.

## Rules

- Same window for every TF. Common intersection must be non-empty.
- Closed bars only (`copy_rates_range` returns closed bars).
- Sub-hourly fallback: if the terminal has no cached M1/M5/M15/M30
  bars for the window but HAS ticks, those TFs' bars are aggregated
  from the tick stream itself (Bid OHLC per minute bucket) and
  flagged `built_from_ticks` in the provenance. This is NOT
  resampling from H1 — it aggregates the tick stream, the same
  source MT5 builds bars from. HTF bars (H1/H4/D1/W1/MN1) with no
  terminal history still BLOCK. No ticks at all still BLOCKS.
- Do NOT stitch across history holes. If the terminal returns gapped
  history, re-download first (F2 → download) or export a window
  inside one contiguous segment. The loader reports segments and the
  largest gap per TF — see `scripts/backtest_mt5_full.py --coverage`.
- Ticks: `COPY_TICKS_ALL`. Without ticks the harness can smoke-run on
  the labelled M1 proxy (`--allow-m1-proxy`), but that is NOT a tick
  backtest and is always flagged `ticks: m1_proxy` in the report.

## Current repo state (2026-10-16)

| TF | File | Coverage |
|---|---|---|
| D1 | MT5 native | 727 bars, gapped (longest segment 2024-03-31→2026-07-17) |
| H4 | OANDA per-TF | 2022-01→2024-10 only — no overlap with H1 |
| H1 | OANDA per-TF | 2024-10→2026-10 |
| M15 | OANDA per-TF | 2024-10→2026-10 |
| M5 | OANDA per-TF | 2026-04→2026-10 only |
| M1 | OANDA per-TF | 2026-09→2026-10 only |
| M30 / W1 / MN1 | — | NO FILE — export required |
| Ticks | — | NO FILE — export required |

OANDA per-TF files are that vendor's own bars for that TF — not H1
bars aggregated by us. They are usable as native bars, but their
windows do not line up, so the full native run is BLOCKED until the
MT5 export above lands with one common window.

## What "no resampling" means here

MT5 ticks are one stream. TF bars are MT5's own per-TF `copy_rates`.
"MT5 ticks for every timeframe" = real tick stream + native bars
for every TF. Resampling H1 → H4/D1 fabricates TF structure (own
opens/closes, session boundaries, Sunday candles, gaps) and is
refused on this path.
