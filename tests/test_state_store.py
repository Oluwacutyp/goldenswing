"""Tests for the warm-resume state store (Phase 10)."""
import pandas as pd

from goldenswing.execution.state_store import StateStore


def _df(n=100):
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": 2000.0, "high": 2005.0, "low": 1995.0,
        "close": 2002.0, "volume": 100.0}, index=idx)


def test_save_load_bars_roundtrip(tmp_path):
    store = StateStore(tmp_path / "store")
    df = _df(100)
    store.save_bars(df)
    loaded = store.load_bars()
    assert loaded is not None
    assert len(loaded) == 100
    assert loaded.index[0] == df.index[0]
    assert loaded["close"].iloc[-1] == 2002.0


def test_buffer_truncates(tmp_path):
    store = StateStore(tmp_path / "store", buffer_bars=50)
    store.save_bars(_df(100))
    loaded = store.load_bars()
    assert len(loaded) == 50
    # keeps the TAIL (recent bars for indicator warmup)
    assert loaded.index[-1] == _df(100).index[-1]


def test_load_missing_returns_none(tmp_path):
    store = StateStore(tmp_path / "empty")
    assert store.load_bars() is None
    assert store.load_meta() == {}


def test_meta_roundtrip(tmp_path):
    store = StateStore(tmp_path / "store")
    store.save_meta({"last_bar_time": "2026-10-11T12:00:00+00:00",
                     "last_signal_bar": 42})
    meta = store.load_meta()
    assert meta["last_signal_bar"] == 42
    assert "2026-10-11" in meta["last_bar_time"]


def test_playbook_state_roundtrip(tmp_path):
    store = StateStore(tmp_path / "store")
    store.save_playbook("fvg_midline", {"pending": [1, 2, 3]})
    assert store.load_playbook("fvg_midline") == {"pending": [1, 2, 3]}
    assert store.load_playbook("nonexistent") == {}


def test_corrupt_buffer_reports_not_crashes(tmp_path):
    store = StateStore(tmp_path / "store")
    store.save_bars(_df(10))
    # corrupt the pickle file
    (tmp_path / "store" / "bars.pkl").write_bytes(b"not a pickle")
    assert store.load_bars() is None
    assert store.load_error is not None
    assert "cold start" in store.load_error
