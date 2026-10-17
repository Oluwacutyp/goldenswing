"""Tick dedupe: same-second ticks are real ticks, not duplicates.

Owner screenshot (2026-10-16): chunk log said one chunk returned
495,711 ticks, but the tick window showed only 85,118 ticks.
Root cause: MT5 stamps ``time`` in whole seconds and gold prints
several real ticks per second, but the fetch + loader did
``drop_duplicates("time")`` — keeping only the FIRST tick per
second and silently destroying ~83% of the stream (plus most of
the summed bar volume).

Fix: dedupe only EXACT duplicate rows (e.g. a tick returned twice
at a chunk boundary). The chunk log also records the RETURNED
span per chunk, not just the requested window — the 495,711 ticks
actually spanned only 2026-01-01..2026-01-05, and the old log
made that look like a 90-day span.
"""
import numpy as np
import pandas as pd

from goldenswing.backtest import mt5_fetch as mf
from goldenswing.backtest.mt5_native_data import (
    build_bars_from_ticks, dedupe_ticks, load_native_frames,
    load_native_ticks)


def _ticks_arr(times, bids):
    n = len(times)
    arr = np.zeros(n, dtype=[("time", "i8"), ("bid", "f8"),
                             ("ask", "f8"), ("last", "f8"),
                             ("volume", "f8"), ("flags", "i8")])
    arr["time"] = np.array(
        [(pd.Timestamp(t).value // 10 ** 9) for t in times])
    arr["bid"] = np.array(bids, dtype=float)
    arr["ask"] = np.array(bids, dtype=float) + 0.2
    arr["last"] = np.array(bids, dtype=float) + 0.1
    arr["volume"] = 1.0
    arr["flags"] = 0
    return arr


class _SameSecondTicksMT5:
    """Fake MT5: one chunk returns 3 same-second distinct ticks
    + 1 exact duplicate + 1 later tick; bars are continuous."""

    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408
    TIMEFRAME_W1 = 32769
    TIMEFRAME_MN1 = 49153
    COPY_TICKS_ALL = 3

    def initialize(self):
        return True

    def shutdown(self):
        pass

    def symbol_select(self, symbol, enable):
        return True

    def copy_rates_range(self, symbol, tf, start, end):
        times = pd.date_range(start, end, freq="h", tz="UTC")
        n = len(times)
        arr = np.zeros(n, dtype=[("time", "i8"), ("open", "f8"),
                                 ("high", "f8"), ("low", "f8"),
                                 ("close", "f8"), ("tick_volume", "i8"),
                                 ("spread", "i8"), ("real_volume", "i8")])
        arr["time"] = (times.astype("int64") // 10 ** 9).to_numpy()
        arr["open"] = 4000.0
        arr["high"] = 4001.0
        arr["low"] = 3999.0
        arr["close"] = 4000.5
        arr["tick_volume"] = 10
        arr["spread"] = 20
        arr["real_volume"] = 0
        return arr

    def copy_ticks_range(self, symbol, start, end, flags):
        # All ticks cluster on 2026-01-01..2026-01-02 even though
        # the chunk requests a 90-day window.
        t0 = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
        t1 = pd.Timestamp("2026-01-02 00:00:00", tz="UTC")
        # 3 distinct ticks in the SAME second + 1 exact dup of #1.
        return _ticks_arr(
            [t0, t0, t0, t0, t1],
            [4000.0, 4000.1, 4000.2, 4000.0, 4001.0])


def test_dedupe_keeps_same_second_ticks():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2026-01-01 00:00:00"] * 3
                               + ["2026-01-01 00:00:01"], utc=True),
        "bid": [100.0, 100.1, 100.2, 101.0],
        "ask": [100.2, 100.3, 100.4, 101.2],
        "volume": [1.0, 1.0, 1.0, 1.0]})
    assert len(dedupe_ticks(df)) == 4


def test_dedupe_drops_exact_duplicates_only():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2026-01-01 00:00:00"] * 3, utc=True),
        "bid": [100.0, 100.0, 100.1],
        "ask": [100.2, 100.2, 100.3],
        "volume": [1.0, 1.0, 1.0]})
    out = dedupe_ticks(df)
    assert len(out) == 2  # exact dup collapsed, distinct kept


def test_fetch_keeps_same_second_ticks_and_logs_returned_span(tmp_path):
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-04-01", tz="UTC")
    manifest = mf.fetch_mt5_native(
        "XAUUSDm", start, end, tmp_path,
        mt5_module=_SameSecondTicksMT5(), tick_chunk_days=90)
    # Raw 5 rows (3 distinct same-second + 1 exact dup + 1 later);
    # only the exact dup is dropped.
    assert manifest["tick_fetch"]["rows_raw"] == 5
    assert manifest["tick_fetch"]["total_rows"] == 4
    assert manifest["tick_fetch"]["exact_dupes_dropped"] == 1
    assert manifest["ticks"]["rows"] == 4
    # Chunk log shows the RETURNED span (Jan 1-2), not the 90-day
    # requested window.
    chunk = manifest["tick_chunks"][0]
    assert chunk["rows"] == 5
    assert chunk["returned_start"].startswith("2026-01-01")
    assert chunk["returned_end"].startswith("2026-01-02")
    assert "2026-04-01" not in chunk["returned_end"]
    # Ticks CSV on disk holds all 4 distinct ticks.
    csv = pd.read_csv(tmp_path / "MT5_XAUUSDm_ticks.csv")
    assert len(csv) == 4


def test_loader_tick_window_counts_all_same_second_ticks(tmp_path):
    # Ticks: 3 same-second distinct on Jan 1 + 1 exact dup,
    # 1 on Jan 2, 1 on Jan 3. No native files -> all TFs built
    # from ticks; tick window rows must be 5 (not 3 seconds).
    t0 = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    rows = []
    for t, bid in [(t0, 100.0), (t0, 100.1), (t0, 100.2),
                   (t0, 100.0),
                   (pd.Timestamp("2026-01-02 00:00:00", tz="UTC"), 101.0),
                   (pd.Timestamp("2026-01-03 00:00:00", tz="UTC"), 102.0)]:
        rows.append({"time": t, "bid": bid, "ask": bid + 0.2,
                     "last": bid + 0.1, "volume": 1.0, "flags": 0})
    pd.DataFrame(rows).to_csv(tmp_path / "MT5_XAUUSDm_ticks.csv",
                              index=False)
    ticks = load_native_ticks(tmp_path)
    assert len(ticks) == 5  # exact dup dropped, same-second kept
    _frames, prov = load_native_frames(tmp_path)
    assert prov["tick_window"]["rows"] == 5
    assert prov["tick_window"]["start"].startswith("2026-01-01")
    assert prov["tick_window"]["end"].startswith("2026-01-03")


def test_bar_volume_sums_all_same_second_ticks():
    # Same-second ticks must ALL contribute volume to the bar.
    df = pd.DataFrame({
        "time": pd.to_datetime(["2026-01-01 00:00:00"] * 3, utc=True),
        "bid": [100.0, 100.1, 100.2],
        "ask": [100.2, 100.3, 100.4],
        "volume": [2.0, 3.0, 5.0]})
    bars = build_bars_from_ticks(df, "M1")
    assert len(bars) == 1
    assert bars["volume"].iloc[0] == 10.0
