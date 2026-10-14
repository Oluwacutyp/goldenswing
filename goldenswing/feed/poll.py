"""Polling gold feed (yfinance GC=F) plus a replay feed for rehearsal.

The paper loop owns its bar frame; the feed only supplies frames:

- ``warm_start() -> pd.DataFrame`` — full history ending at the latest
  closed bar.
- ``fetch_new(since: pd.Timestamp) -> pd.DataFrame`` — closed bars with
  ``index > since`` (empty when there is nothing new or the fetch
  failed).
- ``poll_status() -> dict`` — ``{"ok": bool, "reason": str | None}``.
- ``poll_interval`` — seconds the loop should sleep between polls.

Bar-frame conventions match ``backtest.data.load_gold_h1``: lowercase
OHLCV columns, UTC DatetimeIndex, sorted, deduplicated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd


# ---------------------------------------------------------------- config
@dataclass
class FeedConfig:
    symbol: str = "GC=F"
    interval: str = "1h"
    #: seconds between polls for fresh bars
    poll_seconds: float = 300.0
    #: how far back the "recent" fetch reaches (covers outages)
    recent_days: int = 5
    #: a poll is stale if no fetch succeeded within this window
    stale_poll_minutes: float = 30.0
    #: bars are stale if the newest closed bar is older than this
    stale_bar_hours: float = 8.0
    #: skip the bar-staleness check while the market is closed
    weekend_grace: bool = True
    #: max backoff between retries after consecutive fetch failures
    max_backoff_seconds: float = 900.0


# ------------------------------------------------------------- fetching
def normalize_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance -> load_gold_h1 conventions (shared by warm start and
    every poll so the two can never disagree on format)."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)
    cols = [c for c in ("open", "high", "low", "close", "volume")
            if c in df.columns]
    df = df[cols].dropna(subset=["open", "high", "low", "close"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def fetch_recent(symbol: str, days: int = 5,
                interval: str = "1h") -> pd.DataFrame:
    """Latest closed hourly bars from yfinance.

    The last returned bar is the currently-forming one — callers drop
    it and treat the rest as closed.
    """
    import yfinance as yf

    raw = yf.download(symbol, period=f"{days}d", interval=interval,
                      auto_adjust=False, progress=False, threads=False)
    return normalize_bars(raw)


def drop_forming_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the last row: the bar still forming at fetch time.

    At the top of the hour the "forming" bar may have just closed; it
    is picked up as closed on the next poll.  Slightly late beats
    trading an incomplete bar.
    """
    return df.iloc[:-1] if len(df) else df


# ---------------------------------------------------------- market hours
def market_closed_now(now: datetime | None = None) -> bool:
    """Rough COMEX gold-futures hours: Sun 22:00 UTC – Fri 21:00 UTC,
    daily break 21:00–22:00 UTC.  Used only for the staleness grace —
    never for trading decisions."""
    now = now or datetime.now(timezone.utc)
    wd, hr = now.weekday(), now.hour
    if wd == 5:                       # Saturday
        return True
    if wd == 6 and hr < 22:            # Sunday before the open
        return True
    if wd == 4 and hr >= 21:           # Friday after the close
        return True
    if hr == 21:                       # daily break
        return True
    return False


# ---------------------------------------------------------- polling feed
class PollingFeed:
    """Live H1 gold feed over yfinance.

    ``warm_start`` loads the cached 2-year history and merges the
    latest fetched bars so the loop starts with valid indicators;
    ``fetch_new`` returns only closed bars newer than ``since``.
    Fetch failures never raise to the loop — they back off and report
    through ``poll_status`` (the watchdog pattern: degrade loudly, keep
    running).
    """

    def __init__(self, cfg: FeedConfig | None = None) -> None:
        self.cfg = cfg or FeedConfig()
        self._failures = 0
        self._last_poll_ok: float | None = None
        self._last_bar_time: pd.Timestamp | None = None
        self._last_error: str | None = None

    # ------------------------------------------------------------ iface
    @property
    def poll_interval(self) -> float:
        if self._failures:
            backoff = self.cfg.poll_seconds * (2 ** self._failures)
            return min(backoff, self.cfg.max_backoff_seconds)
        return self.cfg.poll_seconds

    def warm_start(self) -> pd.DataFrame:
        from ..backtest.data import load_gold_h1

        history = load_gold_h1()
        try:
            recent = drop_forming_bar(
                fetch_recent(self.cfg.symbol, self.cfg.recent_days,
                             self.cfg.interval))
            df = pd.concat([history, recent])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            self._note_success(df)
        except Exception as exc:      # network down: history alone
            self._note_failure(str(exc))
            df = history
        return df

    def fetch_new(self, since: pd.Timestamp) -> pd.DataFrame:
        try:
            recent = drop_forming_bar(
                fetch_recent(self.cfg.symbol, self.cfg.recent_days,
                             self.cfg.interval))
            self._note_success(recent)
        except Exception as exc:
            self._note_failure(str(exc))
            return pd.DataFrame(columns=["open", "high", "low", "close",
                                         "volume"])
        return recent[recent.index > since]

    def poll_status(self) -> dict:
        now = time.time()
        if self._last_poll_ok is None:
            return {"ok": False, "reason": "no successful poll yet"}
        idle = (now - self._last_poll_ok) / 60.0
        if idle > self.cfg.stale_poll_minutes:
            return {"ok": False,
                    "reason": f"no successful poll for {idle:.0f} min "
                              f"(last error: {self._last_error})"}
        if self._last_bar_time is not None and not (
                self.cfg.weekend_grace and market_closed_now()):
            age_h = (datetime.now(timezone.utc)
                     - self._last_bar_time.to_pydatetime()) \
                .total_seconds() / 3600.0
            if age_h > self.cfg.stale_bar_hours:
                return {"ok": False,
                        "reason": f"newest closed bar is {age_h:.1f}h old"}
        return {"ok": True, "reason": None,
                "consecutive_failures": self._failures}

    # ---------------------------------------------------------- helpers
    def _note_success(self, df: pd.DataFrame) -> None:
        self._failures = 0
        self._last_error = None
        self._last_poll_ok = time.time()
        if len(df):
            self._last_bar_time = df.index[-1]

    def _note_failure(self, error: str) -> None:
        self._failures += 1
        self._last_error = error


# ---------------------------------------------------------- replay feed
class ReplayFeed:
    """Replays a historical frame through the feed interface.

    ``warm_bars`` rows form the warm-start frame; each ``fetch_new``
    releases up to ``chunk_bars`` further rows as "newly closed" bars.
    The rehearsal harness uses it to run the paper loop as-if live;
    unit tests use it to prove the loop without the network.
    """

    def __init__(self, df: pd.DataFrame, warm_bars: int = 1500,
                 chunk_bars: int = 1) -> None:
        self._df = normalize_bars(df)
        self._warm_bars = min(warm_bars, len(self._df))
        self._chunk = max(1, chunk_bars)
        self._cursor = self._warm_bars

    @property
    def poll_interval(self) -> float:
        return 0.0

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._df)

    def warm_start(self) -> pd.DataFrame:
        self._cursor = self._warm_bars
        return self._df.iloc[:self._warm_bars].copy()

    def fetch_new(self, since: pd.Timestamp) -> pd.DataFrame:
        # advance past anything the loop already has (defensive: the
        # loop owns the frame, we only track our release cursor)
        while self._cursor < len(self._df) \
                and self._df.index[self._cursor] <= since:
            self._cursor += 1
        out = self._df.iloc[self._cursor:self._cursor + self._chunk]
        self._cursor += len(out)
        return out.copy()

    def poll_status(self) -> dict:
        return {"ok": True, "reason": None,
                "exhausted": self.exhausted}
