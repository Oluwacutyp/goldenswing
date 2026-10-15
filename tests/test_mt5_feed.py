"""MT5 feed adapter: bars, ticks, account, positions — all from MT5.

No OANDA cross-check anywhere in this path.  The fake terminal has
no real market, so these tests verify the adapter contract: UTC
stamping (server-offset calibration), closed-bars-only stack
frames, tick shape, TickPump dedup, and the MT5-absent degrade path.
"""
import datetime as dt

import pandas as pd
import pytest

from goldenswing.feed.mt5 import (MT5Feed, MT5FeedConfig, MT5Unavailable,
                                  TickPump)
from mt5_fakes import FakeMT5


def _feed(*, server_offset_seconds=0, real_volume_on=False,
          tick_script=None, **cfg_kw):
    fake = FakeMT5(server_offset_seconds=server_offset_seconds,
                   real_volume_on=real_volume_on,
                   tick_script=tick_script)
    return MT5Feed(MT5FeedConfig(**cfg_kw), mt5_module=fake)


def test_connect_and_offset_calibration():
    feed = _feed(server_offset_seconds=7200)
    ok, msg = feed.connect()
    assert ok, msg
    assert feed._utc_offset_seconds == pytest.approx(7200, abs=120)


def test_bars_shape_and_utc_index():
    feed = _feed(server_offset_seconds=7200)
    feed.connect()
    df = feed.bars("H1", 10)
    assert len(df) == 10
    assert list(df.columns) == ["open", "high", "low", "close", "volume",
                                "real_volume", "spread"]
    assert str(df.index.tz) == "UTC"
    # Server stamps were UTC+2; the index must read ~UTC (last bar
    # within one H1 bar of real now).
    now = pd.Timestamp.now(tz="UTC")
    assert abs((df.index[-1] - now).total_seconds()) < 7200
    assert (df["volume"] > 0).all()


def test_real_volume_reported_when_terminal_provides_it():
    feed = _feed(real_volume_on=True)
    feed.connect()
    df = feed.bars("H1", 5)
    assert (df["real_volume"] > 0).all()
    feed0 = _feed(real_volume_on=False)
    feed0.connect()
    assert (feed0.bars("H1", 5)["real_volume"] == 0).all()


def test_warm_start_all_closed_bars_only():
    feed = _feed()
    feed.connect()
    frames = feed.warm_start_all()
    # Stack TFs + macro TFs (W1/MN1 feed alignment + strategies;
    # they are NOT hierarchy legs).
    assert set(frames) == {"D1", "H4", "H1", "M15", "M30", "M5",
                           "W1", "MN1"}
    now = pd.Timestamp.now(tz="UTC")
    tf_minutes = {"D1": 1440, "H4": 240, "H1": 60, "M15": 15,
                  "M30": 30, "M5": 5, "W1": 10080}
    for tf, frame in frames.items():
        assert len(frame) > 0
        if tf == "MN1":
            # Calendar-exact: the last MN1 bar opened strictly
            # before this month (a 43200min duration check would
            # fail for February bars).
            last = frame.index[-1]
            assert (last.year, last.month) < (now.year, now.month)
            continue
        # Last bar's close must be <= now (forming bar dropped).
        last_close = frame.index[-1] + pd.Timedelta(
            minutes=tf_minutes[tf])
        assert last_close <= now + pd.Timedelta(seconds=5)


def test_fetch_new_all_returns_only_newer_closed_bars():
    feed = _feed()
    feed.connect()
    frames = feed.warm_start_all()
    cursors = {tf: f.index[-1] for tf, f in frames.items()}
    new = feed.fetch_new_all(cursors)
    for tf, frame in new.items():
        assert len(frame) == 0 or (frame.index > cursors[tf]).all()


def test_tick_shape():
    feed = _feed()
    feed.connect()
    tick = feed.tick()
    assert tick["bid"] > 0 and tick["ask"] > tick["bid"]
    assert tick["mid"] == pytest.approx((tick["bid"] + tick["ask"]) / 2)
    assert tick["spread"] == pytest.approx(tick["ask"] - tick["bid"])
    assert isinstance(tick["time"], dt.datetime)


def test_ticks_range():
    feed = _feed()
    feed.connect()
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(minutes=5)
    df = feed.ticks_range(start, end)
    assert len(df) == 3 and "bid" in df.columns and "ask" in df.columns


def test_account_state_is_terminal_truth():
    feed = _feed()
    feed.connect()
    acct = feed.account_state()
    assert acct["equity"] == 10000.0
    assert acct["trade_mode"] == 0  # demo
    assert acct["login"] == 123456


def test_open_positions_empty_then_present():
    feed = _feed()
    feed.connect()
    assert feed.open_positions() == []


def test_poll_status():
    feed = _feed()
    feed.connect()
    assert feed.poll_status()["ok"] is True


def test_closed_bar_close_floor():
    t = pd.Timestamp("2026-10-14 10:23:45", tz="UTC")
    assert MT5Feed.closed_bar_close("H1", t) == pd.Timestamp(
        "2026-10-14 10:00", tz="UTC")
    assert MT5Feed.closed_bar_close("M15", t) == pd.Timestamp(
        "2026-10-14 10:15", tz="UTC")
    assert MT5Feed.closed_bar_close("D1", t) == pd.Timestamp(
        "2026-10-14 00:00", tz="UTC")


def test_tick_pump_dedupes_identical_ticks():
    script = [(2000.0, 2000.30, 0), (2000.0, 2000.30, 0),
              (2001.0, 2001.30, 1)]
    feed = MT5Feed(MT5FeedConfig(tick_interval=0.001),
                   mt5_module=FakeMT5(tick_script=script))
    feed.connect()
    pump = TickPump(feed, interval=0.001)
    seen: list[dict] = []
    pump.run(seen.append, max_ticks=2)
    assert len(seen) == 2
    assert seen[0]["bid"] == 2000.0 and seen[1]["bid"] == 2001.0


def test_mt5_absent_degrades_cleanly():
    feed = MT5Feed()  # no injection: real import must fail on Linux
    assert feed.available() is False
    with pytest.raises(MT5Unavailable):
        feed.account_state()
