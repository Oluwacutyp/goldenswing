"""News calendar + news-adaptive execution for the live engine.

The live engine previously had NO news awareness at all (the only
news-adjacent code was a spike detector in ``live/regime.py`` and
NFP-day flags in the GOD-CE backtest indicators — neither pauses
entries).  This module is the engine's single news source:

- :class:`NewsCalendar` fetches the ForexFactory this-week calendar
  (high-impact events only), caches it for 24h, and falls back to a
  synthetic NFP schedule (first Friday of the month, 13:30 UTC) when
  the fetch fails.  Only successful FETCHES are cached — a fallback
  result is recomputed fresh each call, so an hour of network
  trouble never poisons the cache for a day (the owner's bot cached
  the fallback, which hides real events).
- :meth:`NewsCalendar.execution_state` is what the order path uses:
  inside a high-impact window ``[event-15m, event+45m]`` new entries
  are PAUSED (the owner's window); in the 90 minutes after a window
  ends, size blends back from the news multiplier to 1.0 (the
  owner's cooldown blend).  Management of already-open positions is
  never paused — the runner only consults this on the entry path.

Deliberate non-applications (structural, not oversight):

- The owner's bot also scales SL/TP by ATR multiples in news
  regimes.  This engine's SL/TP are STRUCTURAL (POI invalidation /
  liquidity pools) and R is measured on them — rescaling them by a
  news multiple would corrupt the measured R.  Only the size
  multiplier is applied.
- The owner's bot also has an ``order_mode`` per news regime
  (market / limit_snipe / smart).  This engine already picks the
  order type by setup (``select_order_type``) — a second
  order-mode system is not bolted on.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

#: Pause window around each high-impact event (owner's window).
NEWS_BEFORE_MIN = 15
NEWS_AFTER_MIN = 45
#: Cooldown after a window ends: size blends back to 1.0 over this.
NEWS_COOLDOWN_MIN = 90
#: Cache TTL for a successful calendar fetch.
NEWS_CACHE_TTL_SEC = 86400
#: Titles that make an event EXTREME (owner's extreme_news_titles).
EXTREME_TITLES = ("NFP", "CPI", "FOMC", "Rate Decision", "Core CPI")
#: Size multiplier per news regime (owner's news_regime_params).
NEWS_SIZE_MULT = {"NORMAL": 1.0, "HIGH_VOL_NEWS": 0.5,
                  "EXTREME_NEWS": 0.4}


@dataclass(frozen=True)
class NewsEvent:
    title: str
    time: datetime  # aware, UTC
    impact: str = "high"


@dataclass(frozen=True)
class NewsExecution:
    paused: bool
    size_mult: float
    regime: str  # NORMAL | HIGH_VOL_NEWS | EXTREME_NEWS
    event_title: str | None = None


def _is_extreme(title: str) -> bool:
    upper = (title or "").upper()
    return any(key.upper() in upper for key in EXTREME_TITLES)


def _parse_time(raw: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


class NewsCalendar:
    """High-impact economic calendar with a news-regime view.

    ``fetcher`` is ``(url, timeout) -> parsed JSON`` (default: urllib)
    and ``clock`` is ``() -> aware datetime`` — both injectable so
    tests never touch the network or the wall clock.  Nothing here
    ever raises: a failed fetch / corrupt cache / bad row degrades
    to the NFP fallback.
    """

    def __init__(self, cache_path: str | Path | None = None, *,
                 cache_ttl_sec: float = NEWS_CACHE_TTL_SEC,
                 fetcher=None, clock=None):
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache_ttl_sec = float(cache_ttl_sec)
        self._fetcher = fetcher or self._default_fetch
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._cached: list[NewsEvent] | None = None
        self._cached_at: float = 0.0

    # ------------------------------------------------------------ events
    def events(self, now: datetime | None = None) -> list[NewsEvent]:
        """High-impact events (cached 24h; NFP fallback on failure)."""
        now = now or self._clock()
        if self._cached is not None and \
                time.time() - self._cached_at < self.cache_ttl_sec:
            return list(self._cached)
        events = self._load_cache()
        if events is None:
            events = self._fetch()
            if events:
                self._save_cache(events)
        if events:
            # Only real calendar data is cached in memory — a
            # fallback result is recomputed fresh each call.
            self._cached = list(events)
            self._cached_at = time.time()
            return list(events)
        return self._fallback_events(now)

    def execution_state(
            self, now: datetime | None = None) -> NewsExecution:
        """Pause + size multiplier for a new entry at ``now``."""
        now = now or self._clock()
        in_window: NewsEvent | None = None
        last_end: datetime | None = None
        last_regime = "NORMAL"
        for ev in self.events(now):
            start = ev.time - timedelta(minutes=NEWS_BEFORE_MIN)
            end = ev.time + timedelta(minutes=NEWS_AFTER_MIN)
            if start <= now <= end:
                in_window = ev
                break
            if end <= now and (last_end is None or end > last_end):
                last_end = end
                last_regime = ("EXTREME_NEWS" if _is_extreme(ev.title)
                               else "HIGH_VOL_NEWS")
        if in_window is not None:
            regime = ("EXTREME_NEWS" if _is_extreme(in_window.title)
                      else "HIGH_VOL_NEWS")
            return NewsExecution(paused=True,
                                 size_mult=NEWS_SIZE_MULT[regime],
                                 regime=regime,
                                 event_title=in_window.title)
        if last_end is not None:
            mins = (now - last_end).total_seconds() / 60.0
            if 0 <= mins <= NEWS_COOLDOWN_MIN:
                alpha = max(0.0, min(1.0, 1.0 - mins / NEWS_COOLDOWN_MIN))
                mult = (1.0 * (1.0 - alpha)
                        + NEWS_SIZE_MULT[last_regime] * alpha)
                return NewsExecution(paused=False, size_mult=mult,
                                     regime="NORMAL")
        return NewsExecution(paused=False, size_mult=1.0,
                             regime="NORMAL")

    # ------------------------------------------------------------- fetch
    @staticmethod
    def _default_fetch(url: str, timeout: float):
        req = urllib.request.Request(url, headers={
            "User-Agent": "goldenswing-news/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _fetch(self) -> list[NewsEvent]:
        try:
            data = self._fetcher(FF_CALENDAR_URL, 10)
        except Exception:
            return []
        events: list[NewsEvent] = []
        try:
            for item in data or []:
                if str(item.get("impact", "")).lower() != "high":
                    continue
                when = _parse_time(item.get("date", ""))
                if when is None:
                    continue
                events.append(NewsEvent(
                    title=str(item.get("title", "")), time=when,
                    impact="high"))
        except Exception:
            return []
        return sorted(events, key=lambda e: e.time)

    def _fallback_events(self, now: datetime) -> list[NewsEvent]:
        """Synthetic NFP: first Friday of the month, 13:30 UTC, next
        7 days (the owner's _generate_major fallback — CPI dates are
        not computable, so only NFP is synthesized)."""
        events: list[NewsEvent] = []
        for i in range(7):
            day = now + timedelta(days=i)
            if day.weekday() == 4 and day.day <= 7:
                events.append(NewsEvent(
                    title="NFP",
                    time=day.replace(hour=13, minute=30, second=0,
                                     microsecond=0),
                    impact="high"))
        return events

    # ------------------------------------------------------------- cache
    def _load_cache(self) -> list[NewsEvent] | None:
        if self.cache_path is None or not self.cache_path.exists():
            return None
        try:
            data = json.loads(self.cache_path.read_text())
            if time.time() - float(data["ts"]) >= self.cache_ttl_sec:
                return None
            events: list[NewsEvent] = []
            for row in data.get("events", []):
                when = _parse_time(row.get("time", ""))
                if when is None:
                    continue
                events.append(NewsEvent(title=str(row.get("title", "")),
                                        time=when,
                                        impact=str(row.get(
                                            "impact", "high"))))
            return sorted(events, key=lambda e: e.time) or None
        except Exception:
            return None

    def _save_cache(self, events: list[NewsEvent]) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps({
                "ts": time.time(),
                "events": [{"title": e.title,
                            "time": e.time.isoformat(),
                            "impact": e.impact} for e in events],
            }))
        except Exception:
            pass
