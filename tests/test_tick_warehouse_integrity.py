"""§3.11 tick-warehouse integrity: one labeled, monotonic series.

Finding (2026-10-27, from the owner's harvest): the Oct 27 tick
file was not replay-grade — it mixed two price series with no
symbol column (four rows stamped 19:40:00 carrying bids
4002.79–4003.02 AND 4062.72, the live-trade price) and
concatenated an evening segment with a full day starting at
midnight (one inversion).  Queued verbatim: "tick-warehouse
integrity — single labeled series, monotonic, no divergent
duplicate timestamps."

The warehouse now enforces that contract at the write path:
rows are labeled with the logger's symbol, foreign-symbol ticks
are refused, out-of-order appends are refused (with the tail
seeded from the existing file, so restarts cannot invert
either), and a same-timestamp print divergent from its
neighbours is refused.  Every refusal is counted on the logger.
"""
from datetime import datetime, timezone

from goldenswing.feed.tick_log import TickLogger


def _tick(hour, minute, second, bid, **kw):
    t = {"time": datetime(2026, 10, 27, hour, minute, second,
                          tzinfo=timezone.utc),
         "bid": bid, "ask": bid + 0.30}
    t.update(kw)
    return t


def _rows(path):
    return path.read_text().strip().splitlines()


def test_rows_labeled_and_foreign_symbol_refused(tmp_path):
    logger = TickLogger(tmp_path, symbol="XAUUSD")
    assert logger.log(_tick(19, 40, 0, 4002.79)) is True
    # A tick from a different series never enters this file.
    assert logger.log(_tick(19, 40, 1, 3000.0,
                            symbol="XAGUSD")) is False
    assert logger.dropped_foreign == 1
    logger.close()
    rows = _rows(tmp_path / "mt5_ticks_20261027.csv")
    assert rows[0] == ("symbol,time,bid,ask,tick_volume,"
                       "real_volume,spread")
    assert len(rows) == 2
    assert rows[1].split(",")[0] == "XAUUSD"
    assert rows[1].split(",")[2] == "4002.79000"  # the kept bid


def test_out_of_order_refused_and_counted(tmp_path):
    logger = TickLogger(tmp_path, symbol="XAUUSD")
    assert logger.log(_tick(19, 40, 5, 4002.90)) is True
    # The Oct-27 inversion shape: an earlier segment appended
    # after a later one.
    assert logger.log(_tick(19, 40, 2, 4002.85)) is False
    assert logger.dropped_out_of_order == 1
    # The series still accepts the next in-order tick, and the
    # refused row is nowhere in the file.
    assert logger.log(_tick(19, 40, 6, 4002.95)) is True
    logger.close()
    rows = _rows(tmp_path / "mt5_ticks_20261027.csv")
    assert len(rows) == 3
    assert "4002.85000" not in "".join(rows)


def test_restart_seeds_tail_from_existing_file(tmp_path):
    first = TickLogger(tmp_path, symbol="XAUUSD")
    assert first.log(_tick(19, 40, 5, 4002.90)) is True
    first.close()
    # A restarted process must not append an inversion into the
    # file the previous process left behind.
    second = TickLogger(tmp_path, symbol="XAUUSD")
    assert second.log(_tick(19, 40, 1, 4002.80)) is False
    assert second.dropped_out_of_order == 1
    assert second.log(_tick(19, 40, 7, 4003.00)) is True
    second.close()
    rows = _rows(tmp_path / "mt5_ticks_20261027.csv")
    assert len(rows) == 3  # header + the two accepted ticks


def test_same_timestamp_burst_kept_divergent_print_refused(tmp_path):
    logger = TickLogger(tmp_path, symbol="XAUUSD")
    # The real 19:40:00 burst: several genuine prints inside one
    # timestamp, a $0.23-wide series.
    for bid in (4002.79, 4002.95, 4003.02):
        assert logger.log(_tick(19, 40, 0, bid)) is True
    # The corrupt print from the finding: $60 away at the same
    # timestamp (the live-trade price of another series).
    assert logger.log(_tick(19, 40, 0, 4062.72)) is False
    assert logger.dropped_divergent == 1
    # The refusal did not poison the tail: the genuine series
    # continues at the same timestamp.
    assert logger.log(_tick(19, 40, 0, 4003.05)) is True
    logger.close()
    text = (tmp_path / "mt5_ticks_20261027.csv").read_text()
    assert "4062" not in text


def test_legacy_file_keeps_its_shape_and_guards_still_apply(tmp_path):
    # A pre-integrity day file (no symbol column) keeps its own
    # shape for its remaining life; the guards read its tail and
    # enforce order on it identically.
    path = tmp_path / "mt5_ticks_20261027.csv"
    path.write_text(
        "time,bid,ask,tick_volume,real_volume,spread\n"
        "2026-10-27T19:40:05+00:00,4002.90000,4003.20000,1,0,0.30000\n")
    logger = TickLogger(tmp_path, symbol="XAUUSD")
    assert logger.log(_tick(19, 40, 2, 4002.85)) is False
    assert logger.dropped_out_of_order == 1
    assert logger.log(_tick(19, 40, 6, 4002.95)) is True
    logger.close()
    rows = _rows(path)
    assert rows[0] == "time,bid,ask,tick_volume,real_volume,spread"
    assert len(rows) == 3
    # The appended row follows the FILE's layout (no symbol cell).
    assert rows[2].split(",")[1] == "4002.95000"
    assert len(rows[2].split(",")) == 6
