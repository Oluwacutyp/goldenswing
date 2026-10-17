"""History holes: never stitch across them, report them.

Owner evidence (2026-10-16): the broker's own MT5 D1 export CSV
runs 2023-06-26..2023-06-30, then JUMPS to 2024-03-31 — 9 months
missing (Jul 2023 – Mar 2024). Bars history is gapped, not just
ticks. The old native-only loader used max(first)..min(last) as
the run window, which SPANNED the hole: indicators / swings
computed across it would mix pre-hole and post-hole regimes as if
continuous.

Fix: a gap > max(14 days, 3x TF length) is a HOLE (weekends are
<= ~4 days even with a holiday Monday, Christmas/New Year <= ~10
days, so calendar never flags). The run uses the longest
hole-free piece; holes are reported in provenance + fetch
manifest (which flow into the report JSON).
"""
import numpy as np
import pandas as pd

from goldenswing.backtest import mt5_fetch as mf
from goldenswing.backtest.mt5_native_data import (
    _holes, load_native_frames)


def _native_file(tmp_path, tf, idx):
    df = pd.DataFrame({"ts": idx, "open": 100.0, "high": 110.0,
                       "low": 90.0, "close": 100.0, "volume": 10.0})
    df.to_csv(tmp_path / f"MT5_XAUUSDm_{tf}_native.csv", index=False)


def test_holes_detect_nine_month_d1_hole_not_weekends():
    # Owner's exact shape: Jun 26-30 2023, then Mar 31 2024.
    idx = pd.DatetimeIndex(list(
        pd.date_range("2023-06-26", "2023-06-30", freq="D", tz="UTC"))
        + list(pd.date_range("2024-03-31", "2024-04-05", freq="D",
                             tz="UTC")))
    holes = _holes(idx, "D1")
    assert len(holes) == 1
    assert holes[0]["start"].startswith("2023-06-30")
    assert holes[0]["end"].startswith("2024-03-31")
    assert holes[0]["gap_days"] > 270


def test_holes_ignore_calendar():
    # Fri -> Mon weekend (3d) is not a hole for D1.
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-01-02", tz="UTC"),  # Friday
        pd.Timestamp("2026-01-05", tz="UTC")])  # Monday
    assert _holes(idx, "D1") == []
    # Christmas / New Year stretch (9d) is not a hole.
    idx = pd.DatetimeIndex([
        pd.Timestamp("2025-12-24", tz="UTC"),
        pd.Timestamp("2026-01-02", tz="UTC")])
    assert _holes(idx, "D1") == []
    # H1 weekend gap (49h) is not a hole.
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-01-02 21:00", tz="UTC"),
        pd.Timestamp("2026-01-04 22:00", tz="UTC")])
    assert _holes(idx, "H1") == []
    # W1: one missing week (14d) is not a hole (floor 21d).
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-01-05", tz="UTC"),
        pd.Timestamp("2026-01-19", tz="UTC")])
    assert _holes(idx, "W1") == []
    # W1: three missing weeks (28d) IS a hole.
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-01-05", tz="UTC"),
        pd.Timestamp("2026-02-02", tz="UTC")])
    assert len(_holes(idx, "W1")) == 1
    # MN1: normal ~30d spacing is not a hole (floor 90d).
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-02-01", tz="UTC")])
    assert _holes(idx, "MN1") == []


def test_native_only_run_does_not_span_d1_hole(tmp_path):
    # D1 holed (owner's shape); every other TF continuous.
    # Run must use the Mar 2024+ piece, never span the hole.
    d1_idx = pd.DatetimeIndex(list(
        pd.date_range("2023-06-26", "2023-06-30", freq="D", tz="UTC"))
        + list(pd.date_range("2024-03-31", "2024-06-30", freq="D",
                             tz="UTC")))
    _native_file(tmp_path, "D1", d1_idx)
    cont = pd.date_range("2023-06-26", "2024-06-30", freq="D", tz="UTC")
    for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "W1", "MN1"):
        _native_file(tmp_path, tf, cont)
    frames, prov = load_native_frames(tmp_path)
    assert prov["window"]["start"].startswith("2024-03-31")
    assert prov["window"]["end"].startswith("2024-06-30")
    # The HOLED TF (D1) drops its pre-hole bars: no indicator
    # may be computed across the hole. Continuous TFs keep
    # pre-hole bars as warm-up — their own series has no hole,
    # so nothing is stitched within them.
    assert frames["D1"].index.min() >= pd.Timestamp("2024-03-31",
                                                    tz="UTC")
    # Hole reported, excluded segment reported.
    assert len(prov["holes"]["D1"]) == 1
    assert prov["holes"]["D1"][0]["gap_days"] > 270
    assert prov["excluded_segments"][0]["start"].startswith("2023-06-26")


def test_tick_path_run_does_not_span_tick_hole(tmp_path):
    # Ticks Jan 1-2 (short) + Feb 1-10 (long) with a 30d hole.
    # Run must use the Feb piece, never span the hole.
    idx = pd.DatetimeIndex(list(
        pd.date_range("2026-01-01", "2026-01-02", freq="min", tz="UTC"))
        + list(pd.date_range("2026-02-01", "2026-02-10", freq="min",
                             tz="UTC")))
    df = pd.DataFrame({"time": idx,
                       "bid": 100.0, "ask": 100.2, "last": 100.1,
                       "volume": 1.0, "flags": 0})
    df.to_csv(tmp_path / "MT5_XAUUSDm_ticks.csv", index=False)
    frames, prov = load_native_frames(tmp_path)
    assert prov["window"]["start"].startswith("2026-02-01")
    assert prov["window"]["end"].startswith("2026-02-10")
    assert len(prov["tick_holes"]) == 1
    assert prov["tick_holes"][0]["gap_days"] >= 29
    assert prov["excluded_segments"][0]["start"].startswith("2026-01-01")
    # TFs with fine bars start at Feb 1. Coarse TFs may keep one
    # CONTAINING bar opening before Feb 1: W1's Jan 26 bar holds
    # only Feb 1 (Sunday) ticks — pure post-hole data, not
    # stitching — and MN1's Jan bar is normal monthly spacing.
    for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "D1"):
        assert frames[tf].index.min() >= pd.Timestamp(
            "2026-02-01", tz="UTC"), tf
    assert frames["W1"].index.min() >= pd.Timestamp(
        "2026-01-26", tz="UTC")
    assert frames["MN1"].index.min() >= pd.Timestamp(
        "2026-01-01", tz="UTC")


class _HoledBarsMT5:
    """Fake MT5: D1 bars with the owner's 9-month hole, other TFs
    continuous, ticks Jan-Mar 2024 (no tick hole)."""

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
        if tf == self.TIMEFRAME_D1:
            times = pd.DatetimeIndex(list(
                pd.date_range("2023-06-26", "2023-06-30", freq="D",
                              tz="UTC")) + list(
                pd.date_range("2024-03-31", "2024-06-30", freq="D",
                              tz="UTC")))
        else:
            times = pd.date_range("2023-06-26", "2024-06-30", freq="D",
                                  tz="UTC")
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
        start = pd.Timestamp(start)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        start = max(start, pd.Timestamp("2024-01-01", tz="UTC"))
        end_ts = pd.Timestamp(end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        if end_ts <= pd.Timestamp("2024-01-01", tz="UTC"):
            return None
        times = pd.date_range(start, min(end_ts,
                                         pd.Timestamp("2024-03-01",
                                                      tz="UTC")),
                              freq="h", tz="UTC")
        n = len(times)
        arr = np.zeros(n, dtype=[("time", "i8"), ("bid", "f8"),
                                 ("ask", "f8"), ("last", "f8"),
                                 ("volume", "f8"), ("flags", "i8")])
        arr["time"] = (times.astype("int64") // 10 ** 9).to_numpy()
        arr["bid"] = 4000.0
        arr["ask"] = 4000.2
        arr["last"] = 4000.1
        arr["volume"] = 1.0
        arr["flags"] = 0
        return arr


def test_fetch_manifest_reports_holes(tmp_path):
    start = pd.Timestamp("2023-06-01", tz="UTC")
    end = pd.Timestamp("2024-06-01", tz="UTC")
    manifest = mf.fetch_mt5_native(
        "XAUUSDm", start, end, tmp_path,
        mt5_module=_HoledBarsMT5(), tick_chunk_days=90)
    assert len(manifest["tfs"]["D1"]["holes"]) == 1
    assert manifest["tfs"]["D1"]["holes"][0]["gap_days"] > 270
    assert manifest["tfs"]["H1"]["holes"] == []
    assert manifest["tick_holes"] == []
