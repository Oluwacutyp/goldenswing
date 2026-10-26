"""XAUUSD market calendar tests (live/market_hours.py).

The weekly session: opens Sunday 22:00 UTC, closes Friday 21:00 UTC.
The Friday 21:00 -> Sunday 22:00 window is the weekly close — an old
tick inside it is the expected calendar pause, never a feed failure.
"""

from datetime import datetime, timezone

import pytest

from goldenswing.live.market_hours import xauusd_market_open


def _dt(weekday_iso: str) -> datetime:
    # 2026-10-26 is a Monday; pick dates by weekday explicitly.
    days = {"Mon": 26, "Tue": 27, "Wed": 28, "Thu": 29, "Fri": 30,
            "Sat": 31, "Sun": 25}
    return datetime(2026, 10, days[weekday_iso], tzinfo=timezone.utc)


@pytest.mark.parametrize("day,hour,minute,expected", [
    ("Mon", 0, 0, True),
    ("Fri", 20, 59, True),
    ("Fri", 21, 0, False),
    ("Fri", 23, 30, False),
    ("Sat", 12, 0, False),
    ("Sun", 21, 59, False),
    ("Sun", 22, 0, True),
    ("Sun", 23, 0, True),
])
def test_xauusd_weekly_session(day, hour, minute, expected):
    dt = _dt(day).replace(hour=hour, minute=minute)
    assert xauusd_market_open(dt) is expected


def test_naive_datetime_read_as_utc():
    assert xauusd_market_open(
        datetime(2026, 10, 31, 12, 0)) is False  # Saturday, naive
