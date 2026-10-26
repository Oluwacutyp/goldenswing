"""Execution health: tick freshness, order latency, slippage.

Bar freshness already lives in ``live/data_health.py`` (a pipeline
precondition: stale bars skip the cycle).  This module is the OTHER
half — execution health on the live path:

- tick freshness: seconds since the last tick the pump delivered.
  (Checked on demand by /status — if ticks stop, the tick handler
  stops too, so no in-loop check could ever fire.)
- order latency: place_order round-trip, per placement.
- slippage: fill vs the price the order was aimed at, in ATR units
  (dimensionless, comparable across volatility regimes).  Signed
  adverse: long filled above the aim, short filled below it.

Thresholds are observability, not trading parameters, so they live
here as constructor defaults (overridable), not as runner config.
Breaches are reported as a set of keys — the runner journals only
NEW breaches (a stuck breach gets one note, not one per order).
Nothing here gates or blocks: health degrades loudly, never
silently, and never stops the gold path.

Runtime Truth (2026-10-23): a stale tick is only a BREACH when
the market is supposed to be open.  The runner feeds the broker's
market state in via :meth:`note_market_open`; while the market is
KNOWN closed, an old tick is the expected calendar pause — it is
described as exactly that, never flagged ``tick_stale``, so the
/status Feed and Health sections cannot contradict each other
again (EC2: ``feed: OK`` beside ``BREACH: tick_stale`` for a
2501s-old tick on a closed market).  An UNKNOWN market state
keeps the fail-loud reading: stale is a breach until the feed
proves the closure was expected.

Calendar wiring (2026-10-25): the engine's own weekly calendar
(``live/market_hours.py``) now counts as known-closed alongside the
broker flag, so a Saturday pause reads ``closed_expected`` even when
the broker flag is unknown or stale. When the broker flag and the
calendar disagree, :meth:`describe` says so explicitly — the
difference is surfaced, never silently resolved.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import pandas as pd

from .market_hours import xauusd_market_open

#: Ticks older than this are stale (market hours; the pump delivers
#: continuously while the market is open).
DEFAULT_TICK_STALE_SEC = 120.0
#: place_order round-trip above this is a latency breach.
DEFAULT_LATENCY_WARN_SEC = 2.0
#: Adverse slippage above this (in ATR units) is a slippage breach.
DEFAULT_SLIPPAGE_WARN_ATR = 0.5
#: Samples kept for mean/max (bounded memory).
MAX_SAMPLES = 100


class HealthMonitor:
    def __init__(self, *, tick_stale_sec: float = DEFAULT_TICK_STALE_SEC,
                 latency_warn_sec: float = DEFAULT_LATENCY_WARN_SEC,
                 slippage_warn_atr: float = DEFAULT_SLIPPAGE_WARN_ATR
                 ) -> None:
        self.tick_stale_sec = float(tick_stale_sec)
        self.latency_warn_sec = float(latency_warn_sec)
        self.slippage_warn_atr = float(slippage_warn_atr)
        self._last_tick: datetime | None = None
        # Broker market state as last reported by the runner:
        # True = open, False = KNOWN closed, None = unknown.
        self._market_open: bool | None = None
        self._latencies: deque[float] = deque(maxlen=MAX_SAMPLES)
        self._slippages: deque[float] = deque(maxlen=MAX_SAMPLES)
        self.orders_placed = 0
        self.fills_recorded = 0

    # ------------------------------------------------------------ inputs
    def note_tick(self, tick_time) -> None:
        """Record a tick arrival.  Accepts datetime / Timestamp / ISO
        string (the pump's tick['time'] is a datetime)."""
        try:
            self._last_tick = pd.Timestamp(tick_time).to_pydatetime()
        except Exception:
            pass

    def note_market_open(self, is_open) -> None:
        """Record the broker's market state (True = open, False =
        known closed, None = unknown).  The runner refreshes this
        from the feed's own market check — the same answer the
        order path gates on, so health and execution never read
        two different markets."""
        self._market_open = None if is_open is None else bool(is_open)

    @property
    def market_open_state(self) -> bool | None:
        return self._market_open

    def note_order(self, latency_sec: float) -> None:
        self.orders_placed += 1
        self._latencies.append(float(latency_sec))

    def note_fill(self, adverse_slip: float, atr: float) -> None:
        """Adverse slippage in price units + the ATR it is measured
        against.  ATR <= 0 -> the sample is dropped (no ratio)."""
        if atr is None or atr <= 0:
            return
        self.fills_recorded += 1
        self._slippages.append(float(adverse_slip) / float(atr))

    # ----------------------------------------------------------- readings
    def tick_age_sec(self, now: datetime | None = None) -> float | None:
        if self._last_tick is None:
            return None
        now = now or datetime.now(timezone.utc)
        last = self._last_tick
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return max(0.0, (now - last).total_seconds())

    def calendar_disagreement(self, now: datetime | None = None) -> str | None:
        """Broker flag vs the engine's weekly calendar, when they differ.

        Returns None when they agree or the broker state is unknown.
        Broker-open-while-calendar-closed means a stale broker flag
        inside the weekly close; broker-closed-while-calendar-open
        usually means a real holiday the weekly calendar does not
        know. Neither changes the state class — both are surfaced
        in :meth:`describe` so the difference is visible, not silent.
        """
        if self._market_open is None:
            return None
        if self._market_open and not xauusd_market_open(now):
            return "broker says open, calendar says weekly close"
        if not self._market_open and xauusd_market_open(now):
            return "broker says closed, calendar says open (possible holiday)"
        return None

    def tick_state(self, now: datetime | None = None) -> str:
        """The tick freshness class — ONE vocabulary for the Feed
        line, the Health line and the breach set, so they cannot
        disagree:

        - ``live``: a tick inside the stale threshold;
        - ``stale``: older than the threshold while the market is
          open or its state is unknown — a genuine feed failure
          class (fail loud);
        - ``closed_expected``: older than the threshold (or never
          arrived) while the market is KNOWN closed — by the broker
          flag OR by the engine's own weekly calendar — the
          calendar pause, not a failure;
        - ``no_ticks``: nothing has ever arrived and the market
          is not known closed.
        """
        age = self.tick_age_sec(now)
        known_closed = (self._market_open is False
                        or not xauusd_market_open(now))
        if age is None:
            return "closed_expected" if known_closed else "no_ticks"
        if age <= self.tick_stale_sec:
            return "live"
        return "closed_expected" if known_closed else "stale"

    def latency_stats(self) -> dict:
        vals = list(self._latencies)
        return {"n": len(vals),
                "mean": sum(vals) / len(vals) if vals else None,
                "max": max(vals) if vals else None}

    def slippage_stats(self) -> dict:
        vals = list(self._slippages)
        return {"n": len(vals),
                "mean_atr": sum(vals) / len(vals) if vals else None,
                "max_atr": max(vals) if vals else None}

    def breaches(self, now: datetime | None = None) -> set[str]:
        """Current breach keys: tick_stale / latency / slippage.

        ``tick_stale`` fires only for the ``stale`` / ``no_ticks``
        classes (see :meth:`tick_state`): a tick pause on a KNOWN
        closed market is expected and is never a breach."""
        out: set[str] = set()
        if self.tick_state(now) in ("stale", "no_ticks"):
            out.add("tick_stale")
        lat = self.latency_stats()["max"]
        if lat is not None and lat > self.latency_warn_sec:
            out.add("latency")
        slip = self.slippage_stats()["max_atr"]
        if slip is not None and slip > self.slippage_warn_atr:
            out.add("slippage")
        return out

    def describe(self, now: datetime | None = None) -> str:
        """One line for /status."""
        age = self.tick_age_sec(now)
        state = self.tick_state(now)
        if state == "closed_expected":
            tick = (f"tick {age:.0f}s ago — market closed, quote "
                    f"pause expected (not a feed failure)"
                    if age is not None else
                    "no ticks yet — market closed (expected)")
        elif age is None:
            tick = "no ticks yet"
        else:
            tick = f"tick {age:.0f}s ago"
        lat = self.latency_stats()
        lat_txt = ("latency n/a" if lat["mean"] is None
                   else f"latency mean {lat['mean']:.2f}s "
                        f"max {lat['max']:.2f}s")
        slip = self.slippage_stats()
        slip_txt = ("slippage n/a" if slip["mean_atr"] is None
                    else f"slippage mean {slip['mean_atr']:+.2f} ATR "
                         f"max {slip['max_atr']:+.2f} ATR")
        breaches = self.breaches(now)
        flag = "" if not breaches else (
            f" ⚠ BREACH: {', '.join(sorted(breaches))}")
        note = self.calendar_disagreement(now)
        dis = "" if note is None else f" ⚠ calendar disagreement: {note}"
        return f"health: {tick} | {lat_txt} | {slip_txt}{flag}{dis}"
