"""Feed freshness tests (runner.freshness_line).

"Feed OK" (terminal connected) must never stand in for fresh data.
freshness_line reports per-TF age of the last closed bar against
the engine clock, judged by the XAUUSD calendar: market open + age
beyond 2x the TF interval + 5 min grace -> STALE; market closed ->
CLOSED-EXPECTED; no frame -> UNKNOWN.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from goldenswing.execution.mt5_runner import MT5Runner


def _runner_with_frames(frames):
    runner = object.__new__(MT5Runner)
    runner._stack = SimpleNamespace(frames=frames)
    return runner


def _frame_ending(end: datetime, n: int = 10, freq: str = "5min"):
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": 4000.0, "high": 4001.0, "low": 3999.0,
         "close": 4000.5},
        index=idx)


def test_freshness_states_when_market_open():
    now = datetime.now(timezone.utc)
    frames = {
        "M5": _frame_ending(now - timedelta(minutes=2)),
        "H1": _frame_ending(now - timedelta(days=10), freq="h"),
        "M15": pd.DataFrame(),  # empty -> UNKNOWN
    }
    runner = _runner_with_frames(frames)
    with mock.patch(
            "goldenswing.live.market_hours.xauusd_market_open",
            return_value=True):
        line = runner.freshness_line()
    assert "M5:" in line and "fresh" in line
    assert "H1:" in line and "STALE" in line
    assert "M15: UNKNOWN" in line
    assert "market OPEN" in line


def test_freshness_closed_expected_when_market_closed():
    now = datetime.now(timezone.utc)
    frames = {"H1": _frame_ending(now - timedelta(days=10), freq="h")}
    runner = _runner_with_frames(frames)
    with mock.patch(
            "goldenswing.live.market_hours.xauusd_market_open",
            return_value=False):
        line = runner.freshness_line()
    assert "H1:" in line and "CLOSED-EXPECTED" in line
    assert "STALE" not in line


def test_freshness_no_stack():
    runner = object.__new__(MT5Runner)
    runner._stack = None
    assert runner.freshness_line() == "freshness: no stack"
