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
(manual history download did not fix it and H1 STILL came back
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
with a bare "download history" message.
"""
from __future__ import annotations

import importlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .mt5_native_data import TF_MINUTES, build_bars_from_ticks

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
#: HTF history (manual download did not fix it, H1 still empty) was the bug.
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


def _initialize(mt5, terminal_path: str | None = None) -> bool:
    """Connect to MT5, pinned to the RUNNING terminal when known.

    ``initialize()`` with no path can attach to a DIFFERENT MT5
    terminal than the one on screen (several installs / brokers)
    — different data path, no history, wrong broker — and then
    ``copy_rates`` returns 0 even though the chart shows years of
    data. That is exactly the 2026-10-16 failure: the terminal's
    H1 chart scrolled back to Jun 2023, yet the fetch reported
    "no H1 bars". Pass the running terminal's exe via
    ``terminal_path`` (or ``MT5_TERMINAL_PATH`` env) to pin it.
    Falls back to bare ``initialize()`` when the module does not
    accept ``path`` (test fakes).
    """
    if terminal_path:
        try:
            return bool(mt5.initialize(path=terminal_path))
        except TypeError:
            pass
    return bool(mt5.initialize())


def _copy_rates_from_pos(mt5, symbol: str, tf_attr: str,
                         count: int = 100) -> tuple[object, str]:
    """``copy_rates_from_pos`` probe: last ``count`` bars, no dates.

    Proves whether the fetch works AT ALL. If this returns rows
    while ``copy_rates_range`` returns 0 for the same TF, the bug
    is in the date params / terminal history window — NOT a
    history limit and NOT missing data. Returns (rates, mode).
    """
    fn = getattr(mt5, "copy_rates_from_pos", None)
    if fn is None:
        return None, "unavailable"
    try:
        rates = fn(symbol, tf_attr, 0, count)
    except Exception:
        return None, "error"
    if rates is not None and len(rates) > 0:
        return rates, "from_pos"
    return rates, "empty"


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


def _copy_ticks_range(mt5, symbol: str, start_dt,
                      end_dt) -> tuple[object, str]:
    """``copy_ticks_range`` with a naive-datetime retry.

    Mirrors :func:`_copy_rates`: MT5 interprets datetimes in broker
    server time, and some terminal builds return nothing for
    tz-aware UTC datetimes. If the tz-aware call comes back empty,
    retry once with tz-naive datetimes and report which
    representation was used. Returns (ticks, mode) where mode is
    ``"tz_aware"``, ``"naive_server_time"`` or ``"empty"``.
    """
    ticks = mt5.copy_ticks_range(symbol, start_dt, end_dt,
                                 mt5.COPY_TICKS_ALL)
    if ticks is not None and len(ticks) > 0:
        return ticks, "tz_aware"
    naive_start = start_dt.replace(tzinfo=None) \
        if getattr(start_dt, "tzinfo", None) else start_dt
    naive_end = end_dt.replace(tzinfo=None) \
        if getattr(end_dt, "tzinfo", None) else end_dt
    ticks = mt5.copy_ticks_range(symbol, naive_start, naive_end,
                                 mt5.COPY_TICKS_ALL)
    if ticks is not None and len(ticks) > 0:
        return ticks, "naive_server_time"
    return ticks, "empty"


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
                      "build", "name", "company", "path", "data_path")}
    except Exception as exc:  # noqa: BLE001 - diagnostics
        diag["terminal"] = {"error": str(exc)}
    try:
        ai = mt5.account_info()
        if ai is None:
            diag["account"] = {"logged_in": False}
        else:
            diag["account"] = {
                k: str(getattr(ai, k, None))
                for k in ("login", "server", "name", "currency",
                          "company")}
    except Exception as exc:  # noqa: BLE001 - diagnostics
        diag["account"] = {"error": str(exc)}
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
    # from_pos probe: last 100 H1 bars with NO dates. If this
    # returns rows while copy_rates_range returns 0, the fetch
    # works and the bug is in the date params / window — the
    # terminal DOES have the history (owner's chart proves it:
    # H1 back to Jun 2023, inside the requested window).
    try:
        rates, mode = _copy_rates_from_pos(
            mt5, symbol, getattr(mt5, TF_ATTRS["H1"]), 100)
        if rates is not None and len(rates) > 0:
            import pandas as _pd
            times = _pd.to_datetime(
                _pd.DataFrame(rates)["time"], unit="s", utc=True)
            diag["probe_from_pos_h1"] = {
                "rows": int(len(rates)), "mode": mode,
                "first": str(times.min()), "last": str(times.max())}
        else:
            diag["probe_from_pos_h1"] = {"rows": 0, "mode": mode}
    except Exception as exc:  # noqa: BLE001 - diagnostics
        diag["probe_from_pos_h1"] = {"error": str(exc)}
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
    acct = diag.get("account", {}) or {}
    sym = diag.get("symbol_info", {}) or {}
    probe = diag.get("probe_h1_last_30d", {}) or {}
    from_pos = diag.get("probe_from_pos_h1", {}) or {}
    req = diag.get("requested", {}) or {}
    return (
        f"diagnostics: terminal_connected={term.get('connected')}, "
        f"server={term.get('server')}, "
        f"terminal_path={term.get('path')}, "
        f"account_login={acct.get('login')}, "
        f"account_server={acct.get('server')}, "
        f"trade_allowed={term.get('trade_allowed')}, "
        f"symbol_visible={sym.get('visible')}, "
        f"symbol_select={sym.get('select')}, "
        f"last_error={diag.get('last_error')}, "
        f"probe_H1_last_30d_rows={probe.get('rows')}, "
        f"probe_from_pos_H1_rows={from_pos.get('rows')} "
        f"({from_pos.get('first')}..{from_pos.get('last')}), "
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


def _bar_covers(first: pd.Timestamp, last: pd.Timestamp, tf: str,
                start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True iff bars stamped [``first``..``last``] cover [``start``, ``end``].

    Bars are stamped by OPEN time, so the last bar covers up to
    ``last_open + TF length``. A naive ``last >= end`` check fails
    real caches (the last closed H1 bar always opens before
    ``end``) and forces a pointless re-fetch every run.
    """
    last_covers_to = last + pd.Timedelta(minutes=TF_MINUTES[tf])
    return bool(first <= start
                and last_covers_to >= end - pd.Timedelta(seconds=1))


def _cache_covers(cache_dir: Path, symbol: str, start: pd.Timestamp,
                  end: pd.Timestamp) -> bool:
    """True iff every TF CSV + ticks CSV covers [start, end].

    Reads only the ``time`` column. A missing or short file means
    the cache does NOT cover the window — fetch (or re-fetch) it.
    Filenames carry the symbol (``MT5_{symbol}_...``) so caches
    for different broker symbols never get mixed up. Bars are
    checked bar-aware (last open + TF length covers ``end``).
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
        if not _bar_covers(ts.min(), ts.max(), tf, start, end):
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
    return bool(ts.min() <= start
                and ts.max() >= end - pd.Timedelta(hours=1))


def _cache_covers_tick_window(cache_dir: Path, symbol: str
                              ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(tick_start, tick_end) iff the cache covers the TICK window.

    Broker tick history is LIMITED (bars go back to 2023, ticks
    only from Jan 2026), so a cache that covers the tick window
    is sufficient for a tick-window run even though it does not
    cover the originally requested 3-year window. Returns None
    when ticks are missing or any TF's bars do not cover
    [first tick, last tick].
    """
    ticks = cache_dir / _ticks_filename(symbol)
    if not ticks.exists():
        return None
    try:
        times = pd.read_csv(ticks, usecols=["time"])
    except Exception:
        return None
    if len(times) == 0:
        return None
    ts = pd.to_datetime(times["time"], utc=True)
    tick_start, tick_end = ts.min(), ts.max()
    if tick_end <= tick_start:
        return None
    for tf in REQUIRED_TFS:
        path = _rates_file(cache_dir, symbol, tf)
        if path is None:
            return None
        try:
            ftimes = pd.read_csv(path, usecols=["time"])
        except Exception:
            return None
        if len(ftimes) == 0:
            return None
        fts = pd.to_datetime(ftimes["time"], utc=True)
        if not _bar_covers(fts.min(), fts.max(), tf,
                           tick_start, tick_end):
            return None
    return (tick_start, tick_end)


def _fetch_with_module(mt5, symbol: str, start: pd.Timestamp,
                       end: pd.Timestamp, cache_dir: Path,
                       tick_chunk_days: int = 90,
                       terminal_path: str | None = None) -> dict:
    """Fetch via an already-imported MetaTrader5 module. Raises
    MT5Unavailable when initialize / symbol_select fails."""
    if not _initialize(mt5, terminal_path):
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
                          "terminal_path": terminal_path
                          or "(bare initialize() — no path pinned)",
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
        # Per-chunk log: WHICH chunks returned ticks. The owner's
        # 2026-10-16 run got ~74k ticks (~1 day) from a 3y request
        # with no visibility into why. If only the last chunk has
        # rows, the broker/terminal tick history is depth-limited;
        # if all chunks are empty, the fetch itself failed. The
        # log (in the manifest) proves which it is.
        tick_chunks: list[dict] = []
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + step, end_dt)
            ticks, tick_mode = _copy_ticks_range(
                mt5, symbol, chunk_start, chunk_end)
            n_ticks = int(len(ticks)) if ticks is not None else 0
            tick_chunks.append({
                "chunk_start": str(chunk_start),
                "chunk_end": str(chunk_end),
                "rows": n_ticks, "mode": tick_mode})
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
        manifest["tick_chunks"] = tick_chunks
        manifest["tick_fetch"] = {
            "chunks_requested": len(tick_chunks),
            "chunks_with_data": sum(1 for c in tick_chunks
                                    if c["rows"] > 0),
            "total_rows": int(len(ticks_df))}
        manifest["ticks"] = {
            "rows": int(len(ticks_df)),
            "start": str(ticks_df["time"].min()),
            "end": str(ticks_df["time"].max())}
        # Broker tick history is LIMITED: the terminal returns
        # ticks only for the recent window (e.g. from Jan 2026)
        # even when bars go back to 2023. Record the actual tick
        # window — the loader runs the full engine on it instead
        # of blocking on the requested 3-year window.
        manifest["tick_window"] = {
            "start": str(ticks_df["time"].min()),
            "end": str(ticks_df["time"].max())}
        manifest["tick_history_limited"] = bool(
            ticks_df["time"].min() > start)
        manifest["requested_window"] = {"start": str(start),
                                        "end": str(end)}
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
        try:
            (cache_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, default=str))
        except OSError:
            pass  # manifest is a debugging aid; cache files matter
        return manifest
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def fetch_mt5_native(symbol: str, start: pd.Timestamp,
                     end: pd.Timestamp, cache_dir: Path | None = None,
                     *, mt5_module=None,
                     tick_chunk_days: int = 90,
                     terminal_path: str | None = None) -> dict:
    """Fetch native MT5 bars + ticks into ``cache_dir``. Always
    fetches (no cache check) — use :func:`ensure_mt5_native` for
    the cached path. ``mt5_module`` injects a fake MetaTrader5 for
    tests; when None the real ``MetaTrader5`` package is imported.
    ``terminal_path`` pins ``initialize(path=...)`` to the running
    terminal's exe (else ``MT5_TERMINAL_PATH`` env, else bare
    ``initialize()`` — which can attach to the wrong terminal).
    """
    cache = Path(cache_dir) if cache_dir is not None \
        else DEFAULT_CACHE_DIR
    if terminal_path is None:
        terminal_path = os.environ.get("MT5_TERMINAL_PATH") or None
    if mt5_module is None:
        try:
            mt5_module = importlib.import_module("MetaTrader5")
        except ImportError:
            raise MT5Unavailable(
                "MetaTrader5 Python package not installed on this "
                "machine — pip install MetaTrader5, start MT5 and "
                "log in, then rerun. " + MT5_NOT_RUNNING_MSG)
    return _fetch_with_module(mt5_module, symbol, start, end, cache,
                              tick_chunk_days=tick_chunk_days,
                              terminal_path=terminal_path)


def ensure_mt5_native(symbol: str, start: pd.Timestamp,
                      end: pd.Timestamp, cache_dir: Path | None = None,
                      *, mt5_module=None, force: bool = False,
                      tick_chunk_days: int = 90,
                      terminal_path: str | None = None) -> dict:
    """Fetch unless the cache already covers [start, end].

    Returns the manifest. ``source`` is ``mt5_cache`` on a cache hit
    (no terminal contact), ``mt5_cache_tick_window`` when the cache
    covers the broker-limited tick window (ticks start after the
    requested start — re-fetching would return the same limited
    ticks, so the run proceeds on the tick window instead of
    re-fetching forever), ``mt5_terminal_live_fetch`` on a fetch.
    """
    cache = Path(cache_dir) if cache_dir is not None \
        else DEFAULT_CACHE_DIR
    if not force and _cache_covers(cache, symbol, start, end):
        return {"symbol": symbol, "start": str(start), "end": str(end),
                "source": "mt5_cache",
                "resampling": "NONE",
                "cache_dir": str(cache)}
    if not force:
        tick_win = _cache_covers_tick_window(cache, symbol)
        if tick_win is not None:
            tick_start, tick_end = tick_win
            overlap_start = max(tick_start, start)
            overlap_end = min(tick_end, end)
            # Ticks must be current (reach the requested end) and
            # the overlap must be a real run window (>= 30 days).
            # A historical request that does not overlap the tick
            # window falls through to a fresh fetch.
            if tick_end >= end - pd.Timedelta(days=1) and \
                    (overlap_end - overlap_start) >= pd.Timedelta(
                        days=30):
                return {"symbol": symbol,
                        "start": str(start), "end": str(end),
                        "source": "mt5_cache_tick_window",
                        "resampling": "NONE",
                        "cache_dir": str(cache),
                        "tick_window": {"start": str(tick_start),
                                        "end": str(tick_end)},
                        "run_window": {"start": str(overlap_start),
                                       "end": str(overlap_end)}}
    manifest = fetch_mt5_native(symbol, start, end, cache,
                                mt5_module=mt5_module,
                                tick_chunk_days=tick_chunk_days,
                                terminal_path=terminal_path)
    manifest["cache_dir"] = str(cache)
    return manifest


def default_window(years: int = 3,
                   end: datetime | None = None
                   ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Default fetch window: the last ``years`` years, ending now."""
    end_ts = pd.Timestamp(end or datetime.now(timezone.utc))
    start_ts = end_ts - pd.Timedelta(days=365 * years)
    return start_ts, end_ts
