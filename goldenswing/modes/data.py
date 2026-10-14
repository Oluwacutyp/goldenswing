"""Intraday data for the sub-hourly modes (yfinance, cached).

yfinance caps intraday history hard: 15m and 5m bars are limited to
the last ~60 days.  The scalp modes therefore run on a much smaller
sample than the swing mode's 2 years — MODES.md states the actual bar
counts and date ranges, and the verdicts are labelled accordingly.

Peers (silver SI=F, dollar DX-Y.NYB) are fetched at the SAME interval
as gold, so the feature layer's causal peer alignment (bar-end as-of,
see features/common.py) holds exactly: a peer bar is eligible only
once it has fully closed, and on a matched grid it closes at the same
instant as the gold bar it aligns to.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..backtest.data import PEER_SYMBOLS, REPO_ROOT

GOLD_SYMBOL = "GC=F"


def fetch_intraday(symbol: str, interval: str, period: str,
                   cache_path: Path) -> pd.DataFrame | None:
    """Fetch one symbol at one interval; None on any failure (modes
    degrade the same way the RS peers do — missing data is data)."""
    try:
        import yfinance as yf

        raw = yf.download(symbol, period=period, interval=interval,
                          auto_adjust=False, progress=False, threads=False)
        if raw is None or len(raw) == 0:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[
            ["open", "high", "low", "close", "volume"]]
        df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
        df.index = pd.to_datetime(df.index, utc=True)
        if len(df) == 0:
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        return df
    except Exception:
        return None


def _load_cached(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    try:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        if len(df) > 100:
            return df
    except Exception:
        pass
    return None


def load_intraday(symbol: str, interval: str, period: str, tag: str,
                  refresh: bool = False) -> pd.DataFrame | None:
    """Load (cache-first) ``symbol`` at ``interval``; ``tag`` names
    the cache file (e.g. GCF_M15_60d)."""
    cache_path = REPO_ROOT / "data" / f"{tag}.csv"
    if not refresh:
        df = _load_cached(cache_path)
        if df is not None:
            return df
    return fetch_intraday(symbol, interval, period, cache_path)


def _tag(symbol: str, interval: str, period: str) -> str:
    sym = {"GC=F": "GCF", "SI=F": "SILVER", "DX-Y.NYB": "DOLLAR"}.get(
        symbol, symbol.replace("=", "").replace("-", ""))
    return f"{sym}_{interval.upper()}_{period}"


def load_mode_universe(interval: str, period: str,
                       refresh: bool = False):
    """(gold_df, peers) for one mode timeframe.

    Gold is mandatory (RuntimeError if unavailable — a mode without
    its instrument is not a mode).  Peers degrade to None, exactly as
    in the swing pipeline.
    """
    gold = load_intraday(GOLD_SYMBOL, interval, period,
                         _tag(GOLD_SYMBOL, interval, period),
                         refresh=refresh)
    if gold is None or len(gold) == 0:
        raise RuntimeError(
            f"gold {interval} data unavailable — mode cannot run")
    peers = {}
    for name, symbol in PEER_SYMBOLS.items():
        peers[name] = load_intraday(symbol, interval, period,
                                    _tag(symbol, interval, period),
                                    refresh=refresh)
    return gold, peers
