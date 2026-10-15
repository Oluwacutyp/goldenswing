"""Entry gates for the live order path (base orders only).

- Trade-count caps (per UTC day / hour),
- cooldowns after a close (per outcome),
- fingerprint dedup (the same setup cannot re-fire within TTL).

State is in-memory per process run: a restart resets the counts,
the cooldown, and the fingerprint set.  That is the documented
trade-off for a gate that must never block startup on a corrupt
state file — the journal remains the durable record; these gates
pace entries, they do not reconstruct history.

Also in this module (same order path, same pure-function shape):
:func:`session_multiplier` (Feature 5) and :func:`is_friday_close`
(Feature 5) — pure time helpers, wired in their own commit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..reader.session_timeframes import session_of

#: Session size multipliers (owner's session_multipliers mapped onto
#: the engine's DST-aware session names — ASIAN_LONDON has no engine
#: equivalent, it is a boundary point, not a bucket, so it is dropped).
DEFAULT_SESSION_MULTIPLIERS: dict[str, float] = {
    "asia": 0.8,
    "london": 1.0,
    "overlap": 1.2,   # LONDON_NY
    "newyork": 1.1,   # NY
    "off": 0.7,       # NY_CLOSE
    "unknown": 1.0,
}
#: Friday late-close hour (UTC) — no new base entries at/after this.
DEFAULT_FRIDAY_CLOSE_HOUR = 20


@dataclass(frozen=True)
class EntryGateConfig:
    max_daily_trades: int = 0   # 0 = unlimited (owner's default)
    max_hourly_trades: int = 0  # 0 = unlimited (owner's default)
    loss_cooldown_sec: float = 300.0  # owner's 300s
    win_cooldown_sec: float = 0.0
    fingerprint_ttl_sec: float = 3600.0


class EntryGate:
    """Pacing gate for new base entries.  Clock-injectable."""

    def __init__(self, cfg: EntryGateConfig | None = None, *,
                 clock=None):
        self.cfg = cfg or EntryGateConfig()
        self._clock = clock or time.time
        self._day_counts: dict[str, int] = {}
        self._hour_counts: dict[str, int] = {}
        self._last_close_at: float | None = None
        self._last_close_won: bool = False
        self._fingerprints: dict[str, float] = {}

    # ------------------------------------------------------------ check
    def check_new_entry(self, now: float, fingerprint: str,
                        ) -> tuple[bool, str]:
        """(ok, reason) — reason is "" when ok, else the gate name."""
        cfg = self.cfg
        day = datetime.fromtimestamp(now, tz=timezone.utc).date().isoformat()
        hour = datetime.fromtimestamp(
            now, tz=timezone.utc).strftime("%Y-%m-%dT%H")
        self._prune(now, day, hour)
        if cfg.max_daily_trades > 0 and \
                self._day_counts.get(day, 0) >= cfg.max_daily_trades:
            return False, "cap_daily"
        if cfg.max_hourly_trades > 0 and \
                self._hour_counts.get(hour, 0) >= cfg.max_hourly_trades:
            return False, "cap_hourly"
        if self._last_close_at is not None:
            cd = cfg.win_cooldown_sec if self._last_close_won \
                else cfg.loss_cooldown_sec
            if cd > 0 and now - self._last_close_at < cd:
                return False, "cooldown"
        seen = self._fingerprints.get(fingerprint)
        if seen is not None and now - seen < cfg.fingerprint_ttl_sec:
            return False, "duplicate"
        return True, ""

    # ----------------------------------------------------------- record
    def record_open(self, now: float, fingerprint: str) -> None:
        day = datetime.fromtimestamp(now, tz=timezone.utc).date().isoformat()
        hour = datetime.fromtimestamp(
            now, tz=timezone.utc).strftime("%Y-%m-%dT%H")
        self._prune(now, day, hour)
        self._day_counts[day] = self._day_counts.get(day, 0) + 1
        self._hour_counts[hour] = self._hour_counts.get(hour, 0) + 1
        self._fingerprints[fingerprint] = now

    def record_close(self, now: float, won: bool) -> None:
        self._last_close_at = now
        self._last_close_won = bool(won)

    # ------------------------------------------------------------ prune
    def _prune(self, now: float, day: str, hour: str) -> None:
        for key in list(self._day_counts):
            if key != day:
                del self._day_counts[key]
        for key in list(self._hour_counts):
            if key != hour:
                del self._hour_counts[key]
        ttl = self.cfg.fingerprint_ttl_sec
        for fp, seen in list(self._fingerprints.items()):
            if now - seen >= ttl:
                del self._fingerprints[fp]


# ------------------------------------------------------------- fingerprint
def signal_fingerprint(playbook: str, direction: int, entry_ref: float,
                       invalidation: float) -> str:
    """Stable identity of a setup: same playbook + direction + levels
    (cent precision for XAUUSD) is the same signal re-firing."""
    return (f"{playbook}:{direction:+d}:{entry_ref:.2f}:"
            f"{invalidation:.2f}")


# ------------------------------------------------------------ session / friday
def session_multiplier(
        now: datetime,
        multipliers: dict[str, float] | None = None) -> float:
    """Size multiplier for the session ``now`` falls in (DST-aware
    session names from ``reader.session_timeframes``).  Unknown
    sessions and missing map entries default to 1.0 (no-op)."""
    table = multipliers if multipliers is not None \
        else DEFAULT_SESSION_MULTIPLIERS
    try:
        session, _kz = session_of(now)
    except Exception:
        return 1.0
    return float(table.get(session, 1.0))


def is_friday_close(now: datetime, close_hour: int = 20) -> bool:
    """Friday at/after ``close_hour`` UTC — no new base entries."""
    try:
        ts = now if now.tzinfo is not None \
            else now.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)
    except Exception:
        return False
    return ts.weekday() == 4 and ts.hour >= close_hour
