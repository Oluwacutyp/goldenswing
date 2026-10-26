"""XAUUSD market calendar — the engine's OWN calendar.

The engine previously trusted the broker's market-state flag for
"is the market open" (health.tick_state: unknown state fails loud
as stale). This module states the known weekly session directly:
XAUUSD trades Sunday 22:00 UTC through Friday 21:00 UTC. The
Friday 21:00 UTC -> Sunday 22:00 UTC window is the weekly close —
an old tick inside it is the expected calendar pause, never a
feed failure. This is the calendar the Saturday fix needs;
feed-freshness reporting uses it, and health.tick_state treats
calendar-closed as known-closed (with broker-vs-calendar
disagreement surfaced in the health line).
"""

from __future__ import annotations

from datetime import datetime, time, timezone

# Weekly close: Friday 21:00 UTC -> Sunday 22:00 UTC.
_CLOSE_START = (4, time(21, 0))  # Friday 21:00 UTC
_OPEN_START = (6, time(22, 0))  # Sunday 22:00 UTC


def xauusd_market_open(now: datetime | None = None) -> bool:
    """True when XAUUSD is in its weekly session at ``now`` (UTC).

    Closed exactly: Friday from 21:00 UTC, all of Saturday, and
    Sunday before 22:00 UTC. Open at exactly Sunday 22:00 UTC and
    Friday up to 21:00 UTC. Naive datetimes are read as UTC.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    weekday, t = now.weekday(), now.time()
    if weekday == 4:  # Friday
        return t < _CLOSE_START[1]
    if weekday == 5:  # Saturday
        return False
    if weekday == 6:  # Sunday
        return t >= _OPEN_START[1]
    return True
