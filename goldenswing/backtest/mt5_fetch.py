"""Auto-fetch native MT5 bars + ticks for the full-engine backtest.

OWNER RULE (2026-10-16): the backtest runs on the machine WITH
MT5, so it fetches the data itself on start. No manual CSV export,
no OANDA fallback, no resampling, no silent approximation.

On start the backtest calls :func:`ensure_mt5_native`, which:

1. checks the cache dir (``data/mt5_native/``) for native CSVs
   already covering the requested window — a second run is instant;
2. otherwise connects to the running MT5 terminal (the terminal's
   logged-in account — no hardcoded login, no credentials in code)
   and fetches ``copy_rates_range`` for EVERY timeframe (MN1, W1,
   D1, H4, H1, M30, M15, M5, M1) plus ``copy_ticks_range`` (one
   single tick stream, NOT per-TF) for the same window, and writes
   them to the cache in the ``MT5_{SYMBOL}_{TF}_native.csv`` /
   ``MT5_{SYMBOL}_ticks.csv`` format the native loader reads, where
   SYMBOL is the broker symbol on the terminal (Exness: ``XAUUSDm``).

If MT5 is not available — package missing, ``initialize()`` fails,
terminal not running — this FAILS with a clear message telling the
owner to start MT5 and log in, then rerun. It never falls back to
OANDA CSVs, never resamples, never silently uses the M1 proxy.
``--allow-m1-proxy`` stays opt-in only, in the harness.

Ticks are fetched in chunks (default 90 days) so a multi-year
``copy_ticks_range`` does not blow up in one call. Rates are one
call per TF (small).
"""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "mt5_native"

#: MT5 timeframe constants, by our TF name. Resolved against the
#: real MetaTrader5 module at fetch time (so tests can inject a
#: fake module with the same attribute names).
TF_ATTRS = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5",
            "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30",
            "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
            "D1": "TIMEFRAME_D1", "W1": "TIMEFRAME_W1",
            "MN1": "TIMEFRAME_MN1"}
REQUIRED_TFS = tuple(TF_ATTRS)

#: Default broker symbol for the backtest fetch. The owner's
#: terminal is Exness, where gold is listed as ``XAUUSDm`` — plain
#: ``XAUUSD`` does not exist there, so ``symbol_select("XAUUSD")``
#: fails and the fetch returns nothing. ``--symbol`` overrides.
DEFAULT_SYMBOL = "XAUUSDm"

MT5_NOT_RUNNING_MSG = (
    "MT5 terminal not running on this machine - start MT5 and log "
    "in, then rerun. The backtest fetches its own data from the "
    "running terminal; it never falls back to OANDA CSVs, "
    "resampling, or the M1 proxy.")


class MT5Unavailable(RuntimeError):
    """MT5 is not reachable — terminal not running / not logged in."""


def _rates_filename(tf: str, symbol: str) -> str:
    return f"MT5_{symbol}_{tf}_native.csv"


def _ticks_filename(symbol: str) -> str:
    return f"MT5_{symbol}_ticks.csv"


def _cache_covers(cache_dir: Path, symbol: str, start: pd.Timestamp,
                  end: pd.Timestamp) -> bool:
    """True iff every TF CSV + ticks CSV exists and covers [start, end].

    Reads only the ``time`` column. A missing or short file means
    the cache does NOT cover the window — fetch (or re-fetch) it.
    Filenames carry the symbol (``MT5_{symbol}_...``) so caches
    for different broker symbols never get mixed up.
    """
    for tf in REQUIRED_TFS:
        path = cache_dir / _rates_filename(tf, symbol)
        if not path.exists():
            return False
        try:
            times = pd.read_csv(path, usecols=["time"])
        except Exception:
            return False
        if len(times) == 0:
            return False
        ts = pd.to_datetime(times["time"], utc=True)
        if ts.min() > start or ts.max() < end:
            return False
    ticks = cache_dir / _ticks_filename(symbol)
    if not ticks.exists():
        return False
    try:
        times = pd.read_csv(ticks, usecols=["time"])
    except Exception:
        return False
    if len(times) == 0:
        return False
    ts = pd.to_datetime(times["time"], utc=True)
    return bool(ts.min() <= start and ts.max() >= end)


def _fetch_with_module(mt5, symbol: str, start: pd.Timestamp,
                       end: pd.Timestamp, cache_dir: Path,
                       tick_chunk_days: int = 90) -> dict:
    """Fetch via an already-imported MetaTrader5 module. Raises
    MT5Unavailable when initialize / symbol_select fails."""
    if not mt5.initialize():
        raise MT5Unavailable(MT5_NOT_RUNNING_MSG)
    try:
        if not mt5.symbol_select(symbol, True):
            raise MT5Unavailable(
                f"MT5 symbol_select({symbol}) failed — check the "
                f"symbol on this terminal, then rerun.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict = {"symbol": symbol,
                          "start": str(start), "end": str(end),
                          "source": "mt5_terminal_live_fetch",
                          "resampling": "NONE",
                          "tfs": {}}
        start_dt = start.to_pydatetime()
        end_dt = end.to_pydatetime()
        for tf in REQUIRED_TFS:
            rates = mt5.copy_rates_range(
                symbol, getattr(mt5, TF_ATTRS[tf]), start_dt, end_dt)
            if rates is None or len(rates) == 0:
                raise MT5Unavailable(
                    f"MT5 returned no {tf} bars for {symbol} "
                    f"{start.date()}..{end.date()} — download history "
                    f"in the terminal (F2) and rerun.")
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.to_csv(cache_dir / _rates_filename(tf, symbol), index=False)
            manifest["tfs"][tf] = {
                "rows": int(len(df)),
                "start": str(df["time"].min()),
                "end": str(df["time"].max())}
        # Ticks: single stream, chunked so a multi-year range does
        # not blow up in one call.
        chunks: list[pd.DataFrame] = []
        chunk_start = start_dt
        step = timedelta(days=tick_chunk_days)
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + step, end_dt)
            ticks = mt5.copy_ticks_range(
                symbol, chunk_start, chunk_end, mt5.COPY_TICKS_ALL)
            if ticks is not None and len(ticks) > 0:
                tdf = pd.DataFrame(ticks)
                tdf["time"] = pd.to_datetime(tdf["time"], unit="s",
                                             utc=True)
                chunks.append(tdf)
            chunk_start = chunk_end
        if not chunks:
            raise MT5Unavailable(
                f"MT5 returned no ticks for {symbol} "
                f"{start.date()}..{end.date()} — download history "
                f"in the terminal (F2) and rerun.")
        ticks_df = pd.concat(chunks, ignore_index=True)
        ticks_df = ticks_df.sort_values("time").drop_duplicates("time")
        ticks_df.to_csv(cache_dir / _ticks_filename(symbol), index=False)
        manifest["ticks"] = {
            "rows": int(len(ticks_df)),
            "start": str(ticks_df["time"].min()),
            "end": str(ticks_df["time"].max())}
        return manifest
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def fetch_mt5_native(symbol: str, start: pd.Timestamp,
                     end: pd.Timestamp, cache_dir: Path | None = None,
                     *, mt5_module=None,
                     tick_chunk_days: int = 90) -> dict:
    """Fetch native MT5 bars + ticks into ``cache_dir``. Always
    fetches (no cache check) — use :func:`ensure_mt5_native` for
    the cached path. ``mt5_module`` injects a fake MetaTrader5 for
    tests; when None the real ``MetaTrader5`` package is imported.
    """
    cache = Path(cache_dir) if cache_dir is not None \
        else DEFAULT_CACHE_DIR
    if mt5_module is None:
        try:
            mt5_module = importlib.import_module("MetaTrader5")
        except ImportError:
            raise MT5Unavailable(
                "MetaTrader5 Python package not installed on this "
                "machine — pip install MetaTrader5, start MT5 and "
                "log in, then rerun. " + MT5_NOT_RUNNING_MSG)
    return _fetch_with_module(mt5_module, symbol, start, end, cache,
                              tick_chunk_days=tick_chunk_days)


def ensure_mt5_native(symbol: str, start: pd.Timestamp,
                      end: pd.Timestamp, cache_dir: Path | None = None,
                      *, mt5_module=None, force: bool = False,
                      tick_chunk_days: int = 90) -> dict:
    """Fetch unless the cache already covers [start, end].

    Returns the manifest. ``source`` is ``mt5_cache`` on a cache hit
    (no terminal contact), ``mt5_terminal_live_fetch`` on a fetch.
    """
    cache = Path(cache_dir) if cache_dir is not None \
        else DEFAULT_CACHE_DIR
    if not force and _cache_covers(cache, symbol, start, end):
        return {"symbol": symbol, "start": str(start), "end": str(end),
                "source": "mt5_cache",
                "resampling": "NONE",
                "cache_dir": str(cache)}
    manifest = fetch_mt5_native(symbol, start, end, cache,
                                mt5_module=mt5_module,
                                tick_chunk_days=tick_chunk_days)
    manifest["cache_dir"] = str(cache)
    return manifest


def default_window(years: int = 3,
                   end: datetime | None = None
                   ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Default fetch window: the last ``years`` years, ending now."""
    end_ts = pd.Timestamp(end or datetime.now(timezone.utc))
    start_ts = end_ts - pd.Timedelta(days=365 * years)
    return start_ts, end_ts
