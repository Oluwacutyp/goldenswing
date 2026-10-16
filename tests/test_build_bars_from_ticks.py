"""Tick-aggregation fallback: sub-hourly bars built from MT5 ticks.

Owner screenshot (2026-10-16): backtest BLOCKED because the terminal
had no cached M1 bars for 3y. M1 bars are redundant when ticks are
present — ticks are the source truth, M1 is just their aggregation.
Requiring cached M1 bars was the bug, not missing data.
"""
import numpy as np
import pandas as pd
import pytest

from goldenswing.backtest import mt5_fetch as mf
from goldenswing.backtest.mt5_native_data import (
    MissingNativeData, build_bars_from_ticks, load_native_frames)


def _tick_df():
    # 3 minutes of ticks: min 0 -> 3 ticks, min 1 -> 2 ticks, min 2 -> 1.
    base = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    rows = [
        (0, 100.0, 100.2, 2.0),
        (20, 101.0, 101.2, 2.0),
        (50, 99.0, 99.2, 2.0),
        (70, 102.0, 102.2, 2.0),
        (110, 103.0, 103.2, 2.0),
        (130, 105.0, 105.2, 2.0),
    ]
    return pd.DataFrame({
        "time": [base + pd.Timedelta(seconds=s) for s, _, _, _ in rows],
        "bid": [b for _, b, _, _ in rows],
        "ask": [a for _, _, a, _ in rows],
        "volume": [v for _, _, _, v in rows],
    })


def test_build_m1_matches_hand_computed_ohlc():
    bars = build_bars_from_ticks(_tick_df(), "M1")
    assert len(bars) == 3
    # Minute 0: open 100 (first bid), high 101, low 99, close 99 (last).
    assert bars["open"].iloc[0] == 100.0
    assert bars["high"].iloc[0] == 101.0
    assert bars["low"].iloc[0] == 99.0
    assert bars["close"].iloc[0] == 99.0
    assert bars["volume"].iloc[0] == 6.0  # 3 ticks x 2.0
    assert bars["spread"].iloc[0] == pytest.approx(0.2)
    # Minute 1: open 102, high 103, low 102, close 103.
    assert bars["open"].iloc[1] == 102.0
    assert bars["close"].iloc[1] == 103.0
    # Minute 2: single tick -> OHLC all equal.
    assert bars["open"].iloc[2] == bars["close"].iloc[2] == 105.0


def test_build_m5_aggregates_across_minutes():
    bars = build_bars_from_ticks(_tick_df(), "M5")
    assert len(bars) == 1  # all ticks inside one 5-min bucket
    assert bars["open"].iloc[0] == 100.0
    assert bars["high"].iloc[0] == 105.0
    assert bars["low"].iloc[0] == 99.0
    assert bars["close"].iloc[0] == 105.0
    assert bars["volume"].iloc[0] == 12.0


def test_build_rejects_htf():
    with pytest.raises(ValueError):
        build_bars_from_ticks(_tick_df(), "H1")
    with pytest.raises(ValueError):
        build_bars_from_ticks(_tick_df(), "D1")


def _native_tf_file(tmp_path, tf, idx):
    df = pd.DataFrame(
        {"ts": idx, "open": 100.0, "high": 110.0, "low": 90.0,
         "close": 100.0, "volume": 10.0})
    df.to_csv(tmp_path / f"MT5_XAUUSDm_{tf}_native.csv", index=False)


def _ticks_file(tmp_path, start, end):
    idx = pd.date_range(start, end, freq="min", tz="UTC")
    df = pd.DataFrame({
        "time": idx,
        "bid": 100.0 + np.arange(len(idx)) * 0.01,
        "ask": 100.2 + np.arange(len(idx)) * 0.01,
        "last": 100.1 + np.arange(len(idx)) * 0.01,
        "volume": 1.0, "flags": 0})
    df.to_csv(tmp_path / "MT5_XAUUSDm_ticks.csv", index=False)
    return idx


def test_loader_builds_missing_m1_from_ticks(tmp_path):
    # Every TF native EXCEPT M1, plus a ticks file -> loads fine,
    # M1 flagged built_from_ticks, M1 bars match tick aggregation.
    idx = pd.date_range("2024-10-01", "2024-11-10", periods=40, tz="UTC")
    for tf in ("M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"):
        _native_tf_file(tmp_path, tf, idx)
    _ticks_file(tmp_path, "2024-10-01", "2024-11-10")
    frames, prov = load_native_frames(tmp_path)
    assert "M1" in frames
    assert prov["M1"]["built_from_ticks"] is True
    assert "built_from_ticks" in prov["M1"]["source"]
    assert "M1" in prov["built_from_ticks"]
    assert prov["resampling"].startswith("NONE")
    # M1 bars came from ticks (per-minute), not from H1 bars.
    assert len(frames["M1"]) > len(frames["H1"])


def test_loader_missing_m1_and_missing_ticks_still_blocks(tmp_path):
    # No M1 file AND no ticks file -> still MissingNativeData naming M1.
    idx = pd.date_range("2024-10-01", "2024-11-10", periods=40, tz="UTC")
    for tf in ("M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"):
        _native_tf_file(tmp_path, tf, idx)
    with pytest.raises(MissingNativeData) as exc:
        load_native_frames(tmp_path)
    assert "M1" in exc.value.missing


def test_loader_missing_htf_still_blocks_even_with_ticks(tmp_path):
    # D1 missing + ticks present -> D1 is HTF, never built from ticks.
    idx = pd.date_range("2024-10-01", "2024-11-10", periods=40, tz="UTC")
    for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "W1", "MN1"):
        _native_tf_file(tmp_path, tf, idx)
    _ticks_file(tmp_path, "2024-10-01", "2024-11-10")
    with pytest.raises(MissingNativeData) as exc:
        load_native_frames(tmp_path)
    assert "D1" in exc.value.missing


class _FakeMT5MissingM1:
    """Fake MT5: native bars for every TF except M1; ticks present."""

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

    def __init__(self, missing_tfs=()):
        self.missing_tfs = set(missing_tfs)

    def initialize(self):
        return True

    def shutdown(self):
        pass

    def symbol_select(self, symbol, enable):
        return True

    def copy_rates_range(self, symbol, tf, start, end):
        tf_name = {v: k for k, v in {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 16385,
            "H4": 16388, "D1": 16408, "W1": 32769,
            "MN1": 49153}.items()}.get(tf)
        if tf_name in self.missing_tfs:
            return None
        times = pd.date_range(start, end, freq="h", tz="UTC")
        n = len(times)
        base = 4000.0 + np.arange(n) * 0.1
        arr = np.zeros(n, dtype=[("time", "i8"), ("open", "f8"),
                                 ("high", "f8"), ("low", "f8"),
                                 ("close", "f8"), ("tick_volume", "i8"),
                                 ("spread", "i8"), ("real_volume", "i8")])
        arr["time"] = (times.astype("int64") // 10 ** 9).to_numpy()
        arr["open"] = base
        arr["high"] = base + 1.0
        arr["low"] = base - 1.0
        arr["close"] = base + 0.5
        arr["tick_volume"] = 10
        arr["spread"] = 20
        arr["real_volume"] = 0
        return arr

    def copy_ticks_range(self, symbol, start, end, flags):
        times = pd.date_range(start, end, freq="min", tz="UTC")
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


def test_fetch_builds_missing_m1_from_ticks(tmp_path):
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-03", tz="UTC")
    fake = _FakeMT5MissingM1(missing_tfs=("M1",))
    manifest = mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                                   mt5_module=fake)
    # No block: M1 bars aggregated from ticks, cached as from_ticks.
    derived = tmp_path / "MT5_XAUUSDm_M1_from_ticks.csv"
    assert derived.exists()
    assert manifest["tfs"]["M1"]["source"] == "built_from_ticks"
    assert manifest["built_from_ticks"] == ["M1"]
    # Cache covers the window (derived file counts as M1 coverage).
    assert mf._cache_covers(tmp_path, "XAUUSDm", start, end) is True
    # Loader reads the derived file and flags it.
    frames, prov = load_native_frames(tmp_path, symbol="XAUUSDm")
    assert prov["M1"]["built_from_ticks"] is True


def test_fetch_missing_htf_still_blocks(tmp_path):
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-03", tz="UTC")
    fake = _FakeMT5MissingM1(missing_tfs=("H1",))
    with pytest.raises(mf.MT5Unavailable) as exc:
        mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                            mt5_module=fake)
    assert "H1" in str(exc.value)


def test_fetch_no_ticks_still_blocks(tmp_path):
    class _NoTicks(_FakeMT5MissingM1):
        def copy_ticks_range(self, symbol, start, end, flags):
            return None

    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-03", tz="UTC")
    with pytest.raises(mf.MT5Unavailable) as exc:
        mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                            mt5_module=_NoTicks())
    assert "no ticks" in str(exc.value).lower()


def test_no_resample_in_tick_aggregation_code():
    import inspect

    import goldenswing.backtest.mt5_fetch as fetch_mod
    import goldenswing.backtest.mt5_native_data as native_mod

    for mod in (fetch_mod, native_mod):
        assert ".resample(" not in inspect.getsource(mod)
    assert ".resample(" not in inspect.getsource(
        build_bars_from_ticks)
