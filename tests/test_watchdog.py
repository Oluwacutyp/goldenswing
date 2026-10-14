"""Tests for the feed watchdog (Phase 10)."""
from datetime import datetime, timedelta, timezone

import pytest

from goldenswing.execution.watchdog import FeedWatchdog, WatchdogConfig


def _cfg(tmp_path, **kw):
    return WatchdogConfig(
        staleness_bars=3, timeframe_seconds=3600,
        heartbeat_path=tmp_path / "heartbeat.txt",
        deadman_outbox=tmp_path / "outbox" / "deadman.log",
        **kw)


def test_staleness_halts_entries(tmp_path):
    alerts = []
    wd = FeedWatchdog(_cfg(tmp_path), on_alert=alerts.append)
    now = datetime.now(timezone.utc)
    wd.note_bar(now - timedelta(hours=4))  # 4h ago, limit is 3h
    assert wd.check_staleness(now) is False
    assert wd.entries_allowed is False
    assert wd.stale_trips == 1
    assert any("STALE" in a for a in alerts)


def test_fresh_bar_allows_entries(tmp_path):
    wd = FeedWatchdog(_cfg(tmp_path))
    now = datetime.now(timezone.utc)
    wd.note_bar(now - timedelta(minutes=30))
    assert wd.check_staleness(now) is True
    assert wd.entries_allowed is True


def test_recovery_re_enables(tmp_path):
    alerts = []
    wd = FeedWatchdog(_cfg(tmp_path), on_alert=alerts.append)
    now = datetime.now(timezone.utc)
    wd.note_bar(now - timedelta(hours=5))
    assert wd.check_staleness(now) is False
    wd.note_bar(now)  # fresh bar arrives
    assert wd.entries_allowed is True
    assert any("RECOVERED" in a for a in alerts)


def test_heartbeat_file(tmp_path):
    wd = FeedWatchdog(_cfg(tmp_path))
    wd.beat()
    age = FeedWatchdog.heartbeat_age_s(tmp_path / "heartbeat.txt")
    assert age is not None and age < 5.0


def test_deadman_clean_shutdown(tmp_path):
    alerts = []
    wd = FeedWatchdog(_cfg(tmp_path), on_alert=alerts.append)
    wd.arm_deadman()
    assert (tmp_path / "outbox" / "deadman.log.armed").exists()
    assert wd.check_previous_death() is True  # armed from "previous" run
    wd.disarm_deadman("test")
    assert not (tmp_path / "outbox" / "deadman.log.armed").exists()
    assert any("STOPPED" in a for a in alerts)


def test_deadman_outbox_written(tmp_path):
    wd = FeedWatchdog(_cfg(tmp_path))
    wd._alert("test alert")
    log = tmp_path / "outbox" / "deadman.log"
    assert log.exists()
    assert "test alert" in log.read_text()
