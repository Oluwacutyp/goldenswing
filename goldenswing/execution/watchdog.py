"""Feed watchdog — a trading system that silently stops is worse than
no system (Phase 10).

Three independent guards for the live loop:

1. **Bar staleness**: no new bar in N x the timeframe -> alert + halt
   new entries.  The market doesn't pause; if our feed does, we're
   blind and must not trade.

2. **Terminal connectivity**: MT5 terminal check each loop.  A dead
   terminal means no fills and no position visibility.

3. **Dead-man's alert**: the loop itself dying.  A heartbeat file is
   touched every loop; an external watcher (or the operator's cron)
   raises the alarm if it goes stale.  The loop ALSO sends a
   "dying" alert on clean shutdown, so silence always means
   something unexpected happened.

All three report through the same alert path (Telegram/outbox) so a
failure of the primary sender still leaves the outbox file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class WatchdogConfig:
    # bar staleness: halt entries if no new bar within this many
    # timeframe-multiples (H1 -> 3h default)
    staleness_bars: int = 3
    timeframe_seconds: int = 3600
    # heartbeat: touched every loop; external watcher alarms if older
    # than this (seconds)
    heartbeat_path: Path | None = None
    heartbeat_stale_s: int = 300
    # dead-man's: alert text sent on unclean death (via outbox)
    deadman_outbox: Path | None = None


class FeedWatchdog:
    """Tracks feed health across loop iterations."""

    def __init__(self, cfg: WatchdogConfig | None = None,
                 on_alert=None) -> None:
        self.cfg = cfg or WatchdogConfig()
        self.on_alert = on_alert  # callable(str) -> None
        self.last_bar_time: datetime | None = None
        self.stale_halted = False
        self.checks = 0
        self.stale_trips = 0

    # ------------------------------------------------------------ bars
    def note_bar(self, bar_time: datetime) -> None:
        """Record a fresh bar.  Returns to healthy if we were halted."""
        if self.last_bar_time is not None and bar_time <= self.last_bar_time:
            return  # not a new bar
        self.last_bar_time = bar_time
        if self.stale_halted:
            self.stale_halted = False
            self._alert(f"FEED RECOVERED: new bar {bar_time.isoformat()} — "
                        f"entries re-enabled")

    def check_staleness(self, now: datetime | None = None) -> bool:
        """True if entries are allowed.  Halts + alerts on staleness."""
        now = now or datetime.now(timezone.utc)
        self.checks += 1
        if self.last_bar_time is None:
            return True  # no baseline yet
        # normalize to UTC for comparison
        last = self.last_bar_time
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_s = (now - last).total_seconds()
        limit_s = self.cfg.staleness_bars * self.cfg.timeframe_seconds
        if age_s > limit_s and not self.stale_halted:
            self.stale_halted = True
            self.stale_trips += 1
            self._alert(
                f"FEED STALE: no new bar for {age_s/3600:.1f}h "
                f"(limit {self.cfg.staleness_bars}x timeframe) — "
                f"new entries HALTED")
            return False
        return not self.stale_halted

    @property
    def entries_allowed(self) -> bool:
        return not self.stale_halted

    # ------------------------------------------------------------ heartbeat
    def beat(self) -> None:
        """Touch the heartbeat file (call every loop)."""
        if self.cfg.heartbeat_path is None:
            return
        p = Path(self.cfg.heartbeat_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(datetime.now(timezone.utc).isoformat())

    @staticmethod
    def heartbeat_age_s(path: Path) -> float | None:
        """How stale the heartbeat is (for the external watcher)."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            ts = datetime.fromisoformat(p.read_text().strip())
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - ts).total_seconds()
        except Exception:
            return None

    # ------------------------------------------------------------ dead-man
    def arm_deadman(self) -> None:
        """Write the armed marker.  On clean shutdown call
        :meth:`disarm_deadman`; if the marker is still armed on the
        next start, the previous loop died uncleanly."""
        if self.cfg.deadman_outbox is None:
            return
        marker = Path(str(self.cfg.deadman_outbox) + ".armed")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now(timezone.utc).isoformat())

    def disarm_deadman(self, reason: str = "clean shutdown") -> None:
        if self.cfg.deadman_outbox is None:
            return
        marker = Path(str(self.cfg.deadman_outbox) + ".armed")
        if marker.exists():
            marker.unlink()
        # the "dying" alert: silence must always mean something
        self._alert(f"RUNNER STOPPED ({reason}) — this is the clean-"
                    f"shutdown notice.  If you didn't stop it, investigate.")

    def check_previous_death(self) -> bool:
        """True if the previous loop died without disarming.  Call on
        startup; alerts if so."""
        if self.cfg.deadman_outbox is None:
            return False
        marker = Path(str(self.cfg.deadman_outbox) + ".armed")
        if marker.exists():
            try:
                ts = marker.read_text().strip()
            except Exception:
                ts = "unknown"
            self._alert(f"DEAD-MAN: previous loop died uncleanly "
                        f"(armed {ts}) — check logs before trusting state")
            return True
        return False

    # ------------------------------------------------------------ internals
    def _alert(self, text: str) -> None:
        if self.on_alert is not None:
            try:
                self.on_alert(text)
            except Exception:
                pass
        # outbox fallback: the alert must survive sender failure
        if self.cfg.deadman_outbox is not None:
            try:
                p = Path(self.cfg.deadman_outbox)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a") as f:
                    f.write(f"{datetime.now(timezone.utc).isoformat()} "
                            f"{text}\n")
            except Exception:
                pass
