"""Native per-TF loader: no resampling, missing TF fails loudly."""
import pandas as pd
import pytest

from goldenswing.backtest.mt5_native_data import (
    MissingNativeData, coverage_report, load_native_frames,
    load_native_ticks, ticks_from_m1)


def _bars(start, n, freq="h"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.5, "volume": 10.0}, index=idx)


def test_no_resample_in_loader():
    # Direct check: the loader module must not call resample().
    import inspect

    import goldenswing.backtest.mt5_native_data as mod
    assert ".resample(" not in inspect.getsource(mod)
    assert "_resample" not in inspect.getsource(mod)


def test_missing_tf_raises_with_export_spec(tmp_path):
    # Empty dir: every TF missing -> MissingNativeData naming W1/MN1.
    with pytest.raises(MissingNativeData) as exc:
        load_native_frames(tmp_path)
    assert "W1" in str(exc.value.missing) or "W1" in str(exc.value)
    assert "copy_rates_range" in str(exc.value)


def test_per_tf_files_loaded_not_resampled(tmp_path):
    # One native file per TF, each with its OWN close values that
    # resampling could never reproduce (D1 close != last H1 close).
    import numpy as np
    # Same span for every TF (loader-mechanics test, not TF spacing).
    idx = pd.date_range("2024-10-01", "2024-11-10", periods=40, tz="UTC")
    for i, tf in enumerate(("M1", "M5", "M15", "M30", "H1", "H4",
                                 "D1", "W1", "MN1")):
        df = pd.DataFrame(
            {"ts": idx, "open": 100.0, "high": 110.0, "low": 90.0,
             "close": 100.0 + np.arange(40) * 0.01 + i * 0.5,
             "volume": 10.0})
        df.to_csv(tmp_path / f"MT5_XAUUSD_{tf}_native.csv", index=False)
    frames, prov = load_native_frames(tmp_path)
    assert set(frames) == {"MN1", "W1", "D1", "H4", "H1", "M30",
                           "M15", "M5", "M1"}
    assert prov["resampling"].startswith("NONE")
    # D1 closes are its own series, not H1 aggregation.
    assert frames["D1"]["close"].iloc[-1] != frames["H1"]["close"].iloc[-1]


def test_d1_gap_segments_reported(tmp_path):
    idx1 = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    idx2 = pd.date_range("2024-06-01", periods=5, freq="D", tz="UTC")
    idx = idx1.append(idx2)
    df = pd.DataFrame({"ts": idx, "open": 100.0, "high": 101.0,
                       "low": 99.0, "close": 100.0, "volume": 10.0})
    df.to_csv(tmp_path / "MT5_XAUUSD_D1_native.csv", index=False)
    cov = coverage_report(tmp_path)
    assert len(cov["D1"].segments) == 2
    assert cov["D1"].gap_days > 100


def test_ticks_mt5_format(tmp_path):
    idx = pd.date_range("2024-10-01", periods=3, freq="s", tz="UTC")
    df = pd.DataFrame({"time": idx, "bid": [100.0, 100.1, 100.2],
                       "ask": [100.2, 100.3, 100.4],
                       "last": [100.1, 100.2, 100.3],
                       "volume": [1, 1, 1], "flags": [0, 0, 0]})
    df.to_csv(tmp_path / "MT5_XAUUSD_ticks.csv", index=False)
    ticks = load_native_ticks(tmp_path)
    assert len(ticks) == 3
    assert ticks[0]["mid"] == pytest.approx(100.1)
    assert ticks[0]["spread"] == pytest.approx(0.2)


def test_ticks_missing_raises(tmp_path):
    with pytest.raises(MissingNativeData):
        load_native_ticks(tmp_path)


def test_m1_proxy_labelled_approximation():
    m1 = _bars("2024-10-01", 2, "min")
    ticks = ticks_from_m1(m1)
    assert len(ticks) == 8  # 4 ticks per M1 bar
    assert ticks[0]["time"] == m1.index[0]
