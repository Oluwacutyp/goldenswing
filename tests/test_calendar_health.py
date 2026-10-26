"""Calendar wiring for execution health: the engine's own weekly
calendar counts as known-closed in tick_state, and broker-vs-calendar
disagreement is surfaced, never silently resolved.

Dates (verified against the 2026 calendar): 2026-10-28 is a Wednesday,
2026-10-30 a Friday, 2026-10-31 a Saturday, 2026-11-01 a Sunday.
"""
from datetime import datetime, timedelta, timezone

from goldenswing.live.health import HealthMonitor

SAT = datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc)
WED = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
FRI_CLOSE = datetime(2026, 10, 30, 21, 0, 0, tzinfo=timezone.utc)
SUN_OPEN = datetime(2026, 11, 1, 22, 0, 0, tzinfo=timezone.utc)


def _old_tick(now):
    h = HealthMonitor(tick_stale_sec=120.0)
    h.note_tick(now - timedelta(seconds=3600))
    return h


def test_calendar_closed_is_known_closed_without_broker_flag():
    """The Saturday fix itself: unknown broker flag + weekly close
    reads closed_expected, NOT a feed failure. Before this wiring
    the same state read stale and breached tick_stale."""
    h = _old_tick(SAT)
    assert h.tick_state(SAT) == "closed_expected"
    assert "tick_stale" not in h.breaches(SAT)


def test_calendar_closed_no_ticks_ever():
    h = HealthMonitor()
    assert h.tick_state(SAT) == "closed_expected"
    assert "tick_stale" not in h.breaches(SAT)


def test_stale_broker_flag_surfaced_not_silent():
    """Broker flag stuck on open inside the weekly close: the state
    is still the calendar pause, but the disagreement is printed."""
    h = _old_tick(SAT)
    h.note_market_open(True)
    assert h.tick_state(SAT) == "closed_expected"
    assert "broker says open, calendar says weekly close" in h.describe(SAT)


def test_open_market_fail_loud_unchanged():
    """Weekday behavior is exactly as before: unknown broker state
    keeps the fail-loud reading."""
    h = _old_tick(WED)
    assert h.tick_state(WED) == "stale"
    assert "tick_stale" in h.breaches(WED)
    h2 = HealthMonitor()
    assert h2.tick_state(WED) == "no_ticks"
    assert "tick_stale" in h2.breaches(WED)


def test_broker_holiday_note():
    """Broker closed on a weekday (a real holiday the weekly calendar
    does not know): still the expected pause, with the difference
    named as a possible holiday."""
    h = _old_tick(WED)
    h.note_market_open(False)
    assert h.tick_state(WED) == "closed_expected"
    assert "possible holiday" in h.describe(WED)


def test_fresh_tick_wins_over_calendar():
    """Freshness beats the calendar: a live tick on Saturday is live."""
    h = HealthMonitor(tick_stale_sec=120.0)
    h.note_tick(SAT - timedelta(seconds=30))
    assert h.tick_state(SAT) == "live"


def test_weekly_close_boundaries():
    """Friday 21:00 UTC exactly is closed; Sunday 22:00 UTC exactly
    is open again."""
    assert _old_tick(FRI_CLOSE).tick_state(FRI_CLOSE) == "closed_expected"
    assert _old_tick(SUN_OPEN).tick_state(SUN_OPEN) == "stale"


def test_no_disagreement_when_agreeing():
    h = _old_tick(WED)
    h.note_market_open(True)
    assert h.calendar_disagreement(WED) is None
    h2 = _old_tick(SAT)
    assert h2.calendar_disagreement(SAT) is None
