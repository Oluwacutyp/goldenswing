"""MT5 tick CSV logging + tick-history export.

Ticks are logged exactly as seen: every row matches the tick dict
that was fed in (bid/ask/volume/spread), one file per UTC day.
"""
from datetime import datetime, timezone

import pytest

from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from goldenswing.feed.tick_log import (TickLogger, export_tick_history,
                                       normalize_tick_frame)
from mt5_fakes import FakeMT5


def test_tick_logger_exact_rows(tmp_path):
    logger = TickLogger(tmp_path)
    ticks = [
        {"time": datetime(2026, 10, 14, 10, 0, 0, tzinfo=timezone.utc),
         "bid": 4160.10, "ask": 4160.40, "tick_volume": 3,
         "real_volume": 0, "spread": 0.30},
        {"time": datetime(2026, 10, 14, 10, 0, 1, tzinfo=timezone.utc),
         "bid": 4160.20, "ask": 4160.50, "tick_volume": 1,
         "real_volume": 0, "spread": 0.30},
    ]
    assert logger.log_many(ticks) == 2
    logger.close()
    day_file = tmp_path / "mt5_ticks_20261014.csv"
    assert day_file.exists()
    rows = day_file.read_text().strip().splitlines()
    assert rows[0] == "time,bid,ask,tick_volume,real_volume,spread"
    assert len(rows) == 3  # header + 2 ticks, exactly as seen
    assert rows[1].split(",")[1:4] == ["4160.10000", "4160.40000", "3"]
    assert rows[2].split(",")[1:4] == ["4160.20000", "4160.50000", "1"]


def test_tick_logger_rotates_per_day(tmp_path):
    logger = TickLogger(tmp_path)
    logger.log({"time": datetime(2026, 10, 14, 23, 59, 59,
                                 tzinfo=timezone.utc),
                "bid": 1.0, "ask": 1.1})
    logger.log({"time": datetime(2026, 10, 15, 0, 0, 1,
                                 tzinfo=timezone.utc),
                "bid": 2.0, "ask": 2.1})
    logger.close()
    assert (tmp_path / "mt5_ticks_20261014.csv").exists()
    assert (tmp_path / "mt5_ticks_20261015.csv").exists()
    assert "2.00000" in (tmp_path / "mt5_ticks_20261015.csv").read_text()


def test_tick_logger_never_raises_on_bad_input(tmp_path):
    logger = TickLogger(tmp_path)
    assert logger.log({}) is False
    assert logger.log({"time": None}) is False
    logger.close()


def test_export_tick_history_uses_terminal_rows(tmp_path):
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    end = datetime.now(timezone.utc)
    start = end.replace(hour=0, minute=0, second=0, microsecond=0)
    path, rows = export_tick_history(feed, start, end,
                                     tmp_path / "ticks.csv")
    assert rows == 3  # FakeMT5 copy_ticks_range returns exactly 3 rows
    text = path.read_text().strip().splitlines()
    assert text[0] == "time,bid,ask,tick_volume,real_volume,spread"
    assert len(text) == 4
    # spread is recomputed ask - bid, never assumed
    first = text[1].split(",")
    assert float(first[5]) == pytest.approx(float(first[2])
                                             - float(first[1]))


def test_normalize_tick_frame_empty():
    import pandas as pd
    out = normalize_tick_frame(pd.DataFrame())
    assert list(out.columns) == ["time", "bid", "ask", "tick_volume",
                                 "real_volume", "spread"]
    assert len(out) == 0
