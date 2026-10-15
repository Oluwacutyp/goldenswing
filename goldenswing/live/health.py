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
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import pandas as pd

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
        """Current breach keys: tick_stale / latency / slippage."""
        out: set[str] = set()
        age = self.tick_age_sec(now)
        if age is None:
            out.add("tick_stale")
        elif age > self.tick_stale_sec:
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
        tick = "no ticks yet" if age is None else f"tick {age:.0f}s ago"
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
        return f"health: {tick} | {lat_txt} | {slip_txt}{flag}"
