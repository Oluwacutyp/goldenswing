"""Tick-window run: broker tick history is limited, bars are not.

Owner screenshot (2026-10-16): backtest BLOCKED with
"common window empty (latest start 2026-01-01 >= earliest end
2025-12-29) — export one common window". The broker keeps bars
back to 2023 but ticks only from ~Jan 2026, so requiring every TF
to share one identical 3-year window blocks forever — and the
error then told the owner to manually export 10 CSVs, which the
owner already rejected (backtest fetches itself).

Fix: the run window is the TICK window. Every TF must cover it —
native bars when they cover it, tick-built bars when the native
file is missing or stale (ends before the tick window ends).
No manual-export instruction appears in any error path.
"""
import numpy as np
import pandas as pd
import pytest

from goldenswing.backtest import mt5_fetch as mf
from goldenswing.backtest.mt5_native_data import (
    MissingNativeData, load_native_frames)


FORBIDDEN_ERROR_STRINGS = (
    "MT5 EXPORT NEEDED",
    "export 10 CSVs",
    "Copy the 10 CSVs",
    "export one common window",
    "common window empty",
)


def _native_file(tmp_path, tf, start, end, freq="h"):
    idx = pd.date_range(start, end, freq=freq, tz="UTC")
    df = pd.DataFrame({"ts": idx, "open": 100.0, "high": 110.0,
                       "low": 90.0, "close": 100.0, "volume": 10.0})
    df.to_csv(tmp_path / f"MT5_XAUUSDm_{tf}_native.csv", index=False)
    return idx


def _ticks_file(tmp_path, start, end):
    idx = pd.date_range(start, end, freq="min", tz="UTC")
    df = pd.DataFrame({
        "time": idx,
        "bid": 100.0 + np.arange(len(idx)) * 0.001,
        "ask": 100.2 + np.arange(len(idx)) * 0.001,
        "last": 100.1 + np.arange(len(idx)) * 0.001,
        "volume": 1.0, "flags": 0})
    df.to_csv(tmp_path / "MT5_XAUUSDm_ticks.csv", index=False)
    return idx


def test_disjoint_tf_windows_do_not_block(tmp_path):
    # The owner's exact shape: ticks Jan 2026.., one TF's native
    # file stale (ends Dec 2025), the rest cover the tick window.
    # Must NOT block: D1 is rebuilt from ticks, run = tick window.
    tick_start, tick_end = "2026-01-01", "2026-01-03"
    for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "W1", "MN1"):
        _native_file(tmp_path, tf, "2025-01-01", "2026-06-01")
    _native_file(tmp_path, "D1", "2025-01-01", "2025-12-29", freq="D")
    _ticks_file(tmp_path, tick_start, tick_end)
    frames, prov = load_native_frames(tmp_path)
    assert set(frames) == {"MN1", "W1", "D1", "H4", "H1", "M30",
                           "M15", "M5", "M1"}
    # D1 was stale -> rebuilt from ticks, flagged.
    assert prov["D1"]["built_from_ticks"] is True
    assert "D1" in prov["built_from_ticks"]
    # H1 native covered the tick window -> kept native.
    assert prov["H1"]["built_from_ticks"] is False
    # Run window IS the tick window.
    assert prov["window"]["start"] == "2026-01-01 00:00:00+00:00"
    assert prov["window"]["end"] == "2026-01-03 00:00:00+00:00"


def test_ticks_limited_window_reported(tmp_path):
    for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"):
        _native_file(tmp_path, tf, "2023-06-28", "2026-10-16")
    _ticks_file(tmp_path, "2026-01-01", "2026-01-03")
    _frames, prov = load_native_frames(tmp_path)
    assert prov["tick_window"]["start"] == "2026-01-01 00:00:00+00:00"
    assert prov["tick_window"]["end"] == "2026-01-03 00:00:00+00:00"
    assert "2026-01-01" in prov["tick_coverage"]
    assert prov["window"]["start"] == prov["tick_window"]["start"]
    assert prov["window"]["end"] == prov["tick_window"]["end"]


def test_no_manual_export_strings_in_error_paths(tmp_path):
    import inspect
    import pathlib
    import sys

    import goldenswing.backtest.mt5_fetch as fetch_mod
    import goldenswing.backtest.mt5_native_data as native_mod

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                           / "scripts"))
    import backtest_mt5_full as harness_mod

    for mod in (fetch_mod, native_mod, harness_mod):
        src = inspect.getsource(mod)
        for forbidden in FORBIDDEN_ERROR_STRINGS:
            assert forbidden not in src, f"{mod.__name__}: {forbidden!r}"
    # And the raised error itself carries no manual-export text.
    with pytest.raises(MissingNativeData) as exc:
        load_native_frames(tmp_path)  # empty dir, no ticks
    for forbidden in FORBIDDEN_ERROR_STRINGS:
        assert forbidden not in str(exc.value)
    assert "copy_rates_range" in str(exc.value)


def test_no_ticks_disjoint_native_files_still_block_cleanly(tmp_path):
    # No ticks + native files with no shared window -> still blocks,
    # but with auto-fetch diagnostics, never a manual-export script.
    for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "W1", "MN1"):
        _native_file(tmp_path, tf, "2024-01-01", "2024-02-01", freq="D")
    _native_file(tmp_path, "D1", "2024-03-01", "2024-04-01", freq="D")
    with pytest.raises(MissingNativeData) as exc:
        load_native_frames(tmp_path)
    for forbidden in FORBIDDEN_ERROR_STRINGS:
        assert forbidden not in str(exc.value)


class _LimitedTicksMT5:
    """Fake MT5: bars for the full requested window, ticks only
    from 2026-01-01 (broker tick-history limit)."""

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
    TICK_LIMIT = pd.Timestamp("2026-01-01", tz="UTC")

    def __init__(self):
        self.calls = {"rates": 0, "ticks": 0}

    def initialize(self):
        return True

    def shutdown(self):
        pass

    def symbol_select(self, symbol, enable):
        return True

    def copy_rates_range(self, symbol, tf, start, end):
        self.calls["rates"] += 1
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
        self.calls["ticks"] += 1
        start = pd.Timestamp(start)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        start = max(start, self.TICK_LIMIT)
        end_ts = pd.Timestamp(end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        if end_ts <= self.TICK_LIMIT:
            return None
        times = pd.date_range(start, end, freq="h", tz="UTC")
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


def test_fetch_cache_reused_on_tick_window(tmp_path):
    # Requested 1y, ticks limited to Jan 2026.. -> first fetch
    # records tick_history_limited; second ensure() must NOT
    # re-fetch (it would return the same limited ticks).
    start = pd.Timestamp("2025-10-17", tz="UTC")
    end = pd.Timestamp("2026-10-16", tz="UTC")
    fake = _LimitedTicksMT5()
    first = mf.ensure_mt5_native("XAUUSDm", start, end, tmp_path,
                                 mt5_module=fake)
    assert first["source"] == "mt5_terminal_live_fetch"
    assert first["tick_history_limited"] is True
    assert (tmp_path / "manifest.json").exists()
    calls_after_first = dict(fake.calls)
    second = mf.ensure_mt5_native("XAUUSDm", start, end, tmp_path,
                                  mt5_module=fake)
    assert second["source"] == "mt5_cache_tick_window"
    assert fake.calls == calls_after_first  # no terminal contact
    assert second["run_window"]["start"].startswith("2026-01-01")
