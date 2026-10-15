"""Multi-timeframe OANDA data for hierarchical backtests.

The hierarchical architecture needs more than one clock:

- HTF (H1/H4): full history, paginated fetch (5000/request), CSV cached.
- LTF (M15/M5): fetched on demand in windows around HTF setup
  timestamps — full multi-year M5 history would be hundreds of
  requests; we only need the LTF tape where a setup actually fired.

All bars are spot XAU/USD mid prices from OANDA's practice API
(same real candles as live). Conventions match the rest of the
repo: lowercase OHLCV, UTC DatetimeIndex, sorted, deduplicated.

Auth reuses the feed module's token file (``.oanda_token``).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..feed.oanda import _get, normalize_oanda_candles, to_oanda_granularity

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

INSTRUMENT = "XAU_USD"
MAX_PER_REQUEST = 5000


def fetch_oanda_range(instrument: str = INSTRUMENT,
                      granularity: str = "H1",
                      start: str | None = None,
                      end: str | None = None) -> pd.DataFrame:
    """All complete candles in [start, end) via paginated requests.

    ``start``/``end`` are ISO-8601 UTC strings. Walks forward in
    5000-candle pages until a short page or ``end`` is reached.
    """
    frames: list[pd.DataFrame] = []
    cursor = start
    end_ts = pd.Timestamp(end, tz="UTC") if end else None
    while True:
        # OANDA pagination: from + count (max 5000). `to` cannot
        # combine with `count`, and a bare from/to range errors when
        # it spans more than 5000 candles — so walk with count and
        # stop at `end` ourselves.
        params: dict = {"granularity": to_oanda_granularity(granularity),
                        "price": "M",
                        "count": MAX_PER_REQUEST}
        if cursor:
            params["from"] = cursor
        payload = _get(f"/v3/instruments/{instrument}/candles", params)
        df = normalize_oanda_candles(payload)
        if len(df) == 0:
            break
        if end_ts is not None:
            df = df[df.index < end_ts]
        frames.append(df)
        if len(df) < MAX_PER_REQUEST:
            break
        if end_ts is not None and df.index[-1] >= end_ts:
            break
        # advance past the last candle (OANDA `from` is inclusive)
        cursor = (df.index[-1] + pd.Timedelta(seconds=1)).isoformat()
    if not frames:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"])
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if end:
        out = out[out.index < pd.Timestamp(end, tz="UTC")]
    return out


def _cache_path(granularity: str, start: str, end: str) -> Path:
    s = pd.Timestamp(start, tz="UTC").strftime("%Y%m%d")
    e = pd.Timestamp(end, tz="UTC").strftime("%Y%m%d")
    return DATA_DIR / f"OANDA_XAU_USD_{granularity}_{s}_{e}.csv"


def load_oanda(granularity: str, start: str, end: str,
               refresh: bool = False) -> pd.DataFrame:
    """Cached HTF history for a backtest window."""
    cp = _cache_path(granularity, start, end)
    if cp.exists() and not refresh:
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        if len(df) > 100:
            return df
    df = fetch_oanda_range(INSTRUMENT, granularity, start, end)
    if len(df) == 0:
        raise RuntimeError(
            f"OANDA returned no {granularity} candles for {start}..{end}")
    cp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cp)
    return df


def fetch_ltf_window(center: pd.Timestamp, granularity: str = "M5",
                     bars_before: int = 96, bars_after: int = 96
                     ) -> pd.DataFrame:
    """LTF tape around an HTF setup timestamp.

    ``bars_before``/``bars_after`` M5 bars around ``center`` — enough
    for the LTF trigger logic (return to zone + displacement/MSS)
    without pulling years of M5 history. Single request.
    """
    # generous time bounds; OANDA returns what exists
    start = (center - pd.Timedelta(minutes=5 * bars_before * 2)).isoformat()
    end = (center + pd.Timedelta(minutes=5 * bars_after * 2)).isoformat()
    # granularity-aware bounds
    mult = {"M1": 1, "M5": 5, "M15": 15, "M30": 30}.get(granularity, 5)
    start = (center - pd.Timedelta(minutes=mult * bars_before * 2)
             ).isoformat()
    end = (center + pd.Timedelta(minutes=mult * bars_after * 2)
           ).isoformat()
    df = fetch_oanda_range(INSTRUMENT, granularity, start, end)
    # trim to the requested window around center
    lo = center - pd.Timedelta(minutes=mult * bars_before)
    hi = center + pd.Timedelta(minutes=mult * bars_after)
    return df[(df.index >= lo) & (df.index <= hi)]
