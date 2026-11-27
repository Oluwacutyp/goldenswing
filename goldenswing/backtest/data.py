"""XAUUSD H1 data via GC=F (COMEX front-month gold futures) on yfinance.

Fetches 2 years of hourly bars once and caches them as CSV under the
repo's ``data/`` directory (git-ignored).  GC=F is a futures proxy for
spot XAUUSD — spreads/rolls differ slightly from a spot feed.

When the cache is absent and yfinance is unavailable (or the fetch
fails), the loader falls back to the vendored OANDA spot feed
(``data/OANDA_XAU_USD_H1_20241001_20261012.csv``, shipped with the
repo) — same shape, recorded in ``df.attrs["feed"]``.  The vendored
feed is NEVER written into the GCF cache path: spot bars must not
be mislabelled as futures.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "data" / "GCF_H1_2y.csv"
#: Vendored OANDA spot H1 feed, shipped with the repo (2024-10-01 ->
#: 2026-10-11).  The fallback when the GC=F cache is absent and
#: yfinance cannot fetch — the GC=F cache is a git-ignored fetch
#: artifact that never ships, so a fresh checkout has no cache.
VENDORED_OANDA_H1 = REPO_ROOT / "data" / "OANDA_XAU_USD_H1_20241001_20261012.csv"


def _load_oanda_h1() -> pd.DataFrame:
    """The vendored OANDA spot H1 feed, in the loader's shape:
    lowercase OHLC + volume, UTC DatetimeIndex, sorted."""
    df = pd.read_csv(VENDORED_OANDA_H1)
    df.index = pd.to_datetime(df["ts"], utc=True)
    df = df.drop(columns=["ts"])
    df.columns = [str(c).lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
    df.attrs["feed"] = "oanda_vendored"
    return df


def fetch_gold_h1(cache_path: Path = DEFAULT_CACHE) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        "GC=F", period="2y", interval="1h",
        auto_adjust=False, progress=False, threads=False,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no GC=F data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    return df


def load_gold_h1(cache_path: Path = DEFAULT_CACHE,
                 refresh: bool = False) -> pd.DataFrame:
    """Gold H1, preferring the GC=F cache, then a yfinance fetch,
    then the vendored OANDA spot feed.

    The GC=F cache is a git-ignored fetch artifact that never
    ships, and yfinance is not installed in every environment —
    so a fresh checkout has neither, and the loader used to die
    on ModuleNotFoundError.  The vendored OANDA feed is the
    fallback: same shape, and the feed used is recorded in
    ``df.attrs["feed"]`` ("gcf_cache" | "gcf_yfinance" |
    "oanda_vendored") so a caller can tell spot bars from
    futures bars.  The OANDA data is never written into the GCF
    cache path."""
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # CSV round-trip can yield an object index of mixed-offset aware
        # datetimes (DST transitions); normalise to a UTC DatetimeIndex.
        df.index = pd.to_datetime(df.index, utc=True)
        if len(df) > 1000:
            df.attrs["feed"] = "gcf_cache"
            return df
    try:
        df = fetch_gold_h1(cache_path)
    except Exception as exc:  # yfinance missing, or the fetch failed
        logger.warning(
            "GC=F fetch failed (%s); falling back to the vendored "
            "OANDA H1 feed", exc)
        if not VENDORED_OANDA_H1.exists():
            raise RuntimeError(
                "no GC=F cache, yfinance fetch failed, and the "
                "vendored OANDA H1 feed is missing") from exc
        return _load_oanda_h1()
    df.attrs["feed"] = "gcf_yfinance"
    return df


# ------------------------------------------------------------- RS peers
#: Relative-strength peers: silver futures and the dollar index.
PEER_SYMBOLS = {"silver": "SI=F", "dollar": "DX-Y.NYB"}


def fetch_series_h1(symbol: str, cache_path: Path) -> pd.DataFrame | None:
    """Fetch one symbol's 2y hourly series; None on any failure (the RS
    feature degrades gracefully — a missing peer is data, not an
    error)."""
    try:
        import yfinance as yf

        raw = yf.download(symbol, period="2y", interval="1h",
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


def load_peer_h1(name: str, refresh: bool = False) -> pd.DataFrame | None:
    """Load a named RS peer ('silver' | 'dollar') from cache or yfinance.

    Returns None when unavailable — callers must treat that as "RS
    unavailable", never as zeros."""
    symbol = PEER_SYMBOLS[name]
    cache_path = REPO_ROOT / "data" / f"{name.upper()}_H1_2y.csv"
    if cache_path.exists() and not refresh:
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            if len(df) > 100:
                return df
        except Exception:
            pass
    return fetch_series_h1(symbol, cache_path)


def load_peers(refresh: bool = False) -> dict:
    """{'silver': df|None, 'dollar': df|None} for the FeatureEngine."""
    return {name: load_peer_h1(name, refresh=refresh)
            for name in PEER_SYMBOLS}
