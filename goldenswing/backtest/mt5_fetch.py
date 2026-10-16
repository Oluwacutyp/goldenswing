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

Tick-aggregation fallback (2026-10-16): the terminal often has
no cached bars for a window even though it HAS the tick stream
(the owner tried the F2 history download and H1 STILL came back
empty). When ANY TF's ``copy_rates_range`` comes back empty but
ticks were fetched, the bars for that TF are aggregated from the
tick stream itself (:func:`build_bars_from_ticks`) and cached as
``MT5_{symbol}_{TF}_from_ticks.csv``, flagged ``built_from_ticks``
in the manifest. This aggregates ticks — the same stream MT5
builds bars from — never a coarser bar series. Ticks are the
source truth and they contain the session boundaries, so HTF
bars are built this way too (buckets aligned to the broker
server timezone when the terminal reports one, UTC otherwise).
Only a window with no ticks at all still BLOCKS — and then it
blocks WITH diagnostics (terminal/server/symbol state, last
error, a 30-day H1 probe, requested vs returned range), not
with a bare "F2 download" message.
"""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .mt5_native_data import build_bars_from_ticks

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

#: EVERY TF's bars may be aggregated from the fetched tick
#: stream when the terminal has no cached bars for them. Ticks
#: are the source truth and contain the session boundaries, so
#: HTF bars are built this way too — blocking on missing native
#: HTF history (owner tried F2, H1 still empty) was the bug.
#: Tick-built TFs are always flagged ``built_from_ticks``.
BUILDABLE_FROM_TICKS = REQUIRED_TFS

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


def _server_utc_offset_hours(mt5) -> float:
    """Broker server UTC offset in hours, best-effort.

    Reads ``terminal_info().timezone`` when the terminal reports
    it (MQL5 TERMINAL_TIMEZONE is seconds; some builds report
    minutes or hours — normalize by magnitude). Returns 0.0
    (UTC buckets) when the terminal reports nothing usable.
    Never raises — a missing offset must not block the fetch.
    """
    try:
        raw = getattr(mt5.terminal_info(), "timezone", None)
    except Exception:
        return 0.0
    if raw is None:
        return 0.0
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if abs(raw) > 24 * 60:  # seconds
        return raw / 3600.0
    if abs(raw) > 24:  # minutes
        return raw / 60.0
    return raw  # hours


def _copy_rates(mt5, symbol: str, tf_attr: str,
                start_dt, end_dt) -> tuple[object, str]:
    """``copy_rates_range`` with a naive-datetime retry.

    MT5 ``copy_rates`` interprets datetimes in broker server
    time. Passing tz-aware UTC datetimes usually works (epoch is
    absolute), but some terminal builds treat them differently
    and return nothing. If the first call comes back empty, retry
    once with tz-naive datetimes (server-time interpretation) and
    report which representation was used. Returns (rates, mode)
    where mode is ``"tz_aware"``, ``"naive_server_time"`` or
    ``"empty"``.
    """
    rates = mt5.copy_rates_range(symbol, tf_attr, start_dt, end_dt)
    if rates is not None and len(rates) > 0:
        return rates, "tz_aware"
    naive_start = start_dt.replace(tzinfo=None) \
        if getattr(start_dt, "tzinfo", None) else start_dt
    naive_end = end_dt.replace(tzinfo=None) \
        if getattr(end_dt, "tzinfo", None) else end_dt
    rates = mt5.copy_rates_range(symbol, tf_attr,
                                 naive_start, naive_end)
    if rates is not None and len(rates) > 0:
        return rates, "naive_server_time"
    return rates, "empty"


def diagnose_mt5(mt5, symbol: str, start, end) -> dict:
    """Why is MT5 returning nothing? Collect the evidence.

    Returns terminal state (connected / server / trade_allowed /
    build), symbol state (visible / select / trade mode),
    ``last_error()``, and a 30-day H1 probe (rows returned for
    the most recent 30 days — if the probe returns rows while
    the 3y window returns none, the broker's history is depth-
    limited, not missing). Never raises — every field is guarded,
    because a diagnostics call must never be the thing that
    breaks the fetch. Collected lazily on failure paths only, so
    a healthy fetch makes no extra terminal calls.
    """
    diag: dict = {"symbol": symbol,
                  "requested": {"start": str(start), "end": str(end)}}
    try:
        ti = mt5.terminal_info()
        diag["terminal"] = {
            k: str(getattr(ti, k, None))
            for k in ("connected", "trade_allowed", "server",
                      "build", "name", "company", "path")}
    except Exception as exc:  # noqa: BLE001 - diagnostics
        diag["terminal"] = {"error": str(exc)}
    try:
        si = mt5.symbol_info(symbol)
        if si is None:
            diag["symbol_info"] = {"exists": False}
        else:
            diag["symbol_info"] = {
                k: str(getattr(si, k, None))
                for k in ("visible", "select", "trade_mode",
                          "digits", "point", "description")}
    except Exception as exc:  # noqa: BLE001 - diagnostics
        diag["symbol_info"] = {"error": str(exc)}
    try:
        diag["last_error"] = str(mt5.last_error())
    except Exception as exc:  # noqa: BLE001 - diagnostics
        diag["last_error"] = f"last_error() failed: {exc}"
    try:
        probe_start = end - timedelta(days=30)
        rates, mode = _copy_rates(
            mt5, symbol, getattr(mt5, TF_ATTRS["H1"]),
            probe_start.to_pydatetime()
            if hasattr(probe_start, "to_pydatetime") else probe_start,
            end.to_pydatetime()
            if hasattr(end, "to_pydatetime") else end)
        diag["probe_h1_last_30d"] = {
            "rows": int(len(rates)) if rates is not None else 0,
            "mode": mode,
            "window": f"{probe_start.date()}..{end.date()}"}
    except Exception as exc:  # noqa: BLE001 - diagnostics
        diag["probe_h1_last_30d"] = {"error": str(exc)}
    return diag


def _diag_summary(diag: dict) -> str:
    """One-line diagnostics for error messages."""
    term = diag.get("terminal", {}) or {}
    sym = diag.get("symbol_info", {}) or {}
    probe = diag.get("probe_h1_last_30d", {}) or {}
    req = diag.get("requested", {}) or {}
    return (
        f"diagnostics: terminal_connected={term.get('connected')}, "
        f"server={term.get('server')}, "
        f"trade_allowed={term.get('trade_allowed')}, "
        f"symbol_visible={sym.get('visible')}, "
        f"symbol_select={sym.get('select')}, "
        f"last_error={diag.get('last_error')}, "
        f"probe_H1_last_30d_rows={probe.get('rows')}, "
        f"requested={req.get('start')}..{req.get('end')}")


def _rates_filename(tf: str, symbol: str) -> str:
    return f"MT5_{symbol}_{tf}_native.csv"


def _ticks_filename(symbol: str) -> str:
    return f"MT5_{symbol}_ticks.csv"


def _from_ticks_filename(tf: str, symbol: str) -> str:
    return f"MT5_{symbol}_{tf}_from_ticks.csv"


def _rates_file(cache_dir: Path, symbol: str, tf: str) -> Path | None:
    """Native rates file for ``tf`` if present, else the
    tick-aggregated file, else None."""
    native = cache_dir / _rates_filename(tf, symbol)
    if native.exists():
        return native
    derived = cache_dir / _from_ticks_filename(tf, symbol)
    if derived.exists():
        return derived
    return None


def _cache_covers(cache_dir: Path, symbol: str, start: pd.Timestamp,
                  end: pd.Timestamp) -> bool:
    """True iff every TF CSV + ticks CSV exists and covers [start, end].

    Reads only the ``time`` column. A missing or short file means
    the cache does NOT cover the window — fetch (or re-fetch) it.
    Filenames carry the symbol (``MT5_{symbol}_...``) so caches
    for different broker symbols never get mixed up.
    """
    for tf in REQUIRED_TFS:
        path = _rates_file(cache_dir, symbol, tf)
        if path is None:
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
        # Ticks FIRST: the tick stream is the source of truth. If
        # the terminal has no ticks, nothing else can be built and
        # the run blocks. Sub-hourly bars fall back to tick
        # aggregation below, so they need the ticks in hand.
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
            diag = diagnose_mt5(mt5, symbol, start, end)
            manifest["diagnostics"] = diag
            raise MT5Unavailable(
                f"MT5 returned no ticks for {symbol} "
                f"{start.date()}..{end.date()}. "
                f"{_diag_summary(diag)}")
        ticks_df = pd.concat(chunks, ignore_index=True)
        ticks_df = ticks_df.sort_values("time").drop_duplicates("time")
        ticks_df.to_csv(cache_dir / _ticks_filename(symbol), index=False)
        manifest["ticks"] = {
            "rows": int(len(ticks_df)),
            "start": str(ticks_df["time"].min()),
            "end": str(ticks_df["time"].max())}
        offset = _server_utc_offset_hours(mt5)
        manifest["server_utc_offset_hours"] = offset
        built_from_ticks: list[str] = []
        datetime_modes: dict[str, str] = {}
        for tf in REQUIRED_TFS:
            rates, mode = _copy_rates(
                mt5, symbol, getattr(mt5, TF_ATTRS[tf]),
                start_dt, end_dt)
            datetime_modes[tf] = mode
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s",
                                            utc=True)
                df.to_csv(cache_dir / _rates_filename(tf, symbol),
                          index=False)
                manifest["tfs"][tf] = {
                    "rows": int(len(df)),
                    "start": str(df["time"].min()),
                    "end": str(df["time"].max()),
                    "source": "mt5_copy_rates_native",
                    "datetime_mode": mode,
                    "requested": f"{start.date()}..{end.date()}",
                    "returned": f"{df['time'].min()}..{df['time'].max()}"}
                continue
            if tf in BUILDABLE_FROM_TICKS:
                bars = build_bars_from_ticks(
                    ticks_df, tf,
                    server_utc_offset_hours=offset)
                if len(bars) == 0:
                    diag = diagnose_mt5(mt5, symbol, start, end)
                    manifest["diagnostics"] = diag
                    raise MT5Unavailable(
                        f"MT5 returned no {tf} bars for {symbol} "
                        f"{start.date()}..{end.date()} and tick "
                        f"aggregation produced none. "
                        f"{_diag_summary(diag)}")
                bars.reset_index().rename(
                    columns={"timestamp": "time"}).to_csv(
                        cache_dir / _from_ticks_filename(tf, symbol),
                        index=False)
                manifest["tfs"][tf] = {
                    "rows": int(len(bars)),
                    "start": str(bars.index[0]),
                    "end": str(bars.index[-1]),
                    "source": "built_from_ticks",
                    "note": "aggregated from the MT5 tick stream; "
                            f"terminal had no cached {tf} bars",
                    "requested": f"{start.date()}..{end.date()}",
                    "returned": f"{bars.index[0]}..{bars.index[-1]}"}
                built_from_ticks.append(tf)
                continue
            diag = diagnose_mt5(mt5, symbol, start, end)
            manifest["diagnostics"] = diag
            raise MT5Unavailable(
                f"MT5 returned no {tf} bars for {symbol} "
                f"{start.date()}..{end.date()}. "
                f"{_diag_summary(diag)}")
        manifest["built_from_ticks"] = built_from_ticks
        manifest["datetime_modes"] = datetime_modes
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
