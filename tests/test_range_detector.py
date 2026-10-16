"""RangeDetector: incremental == stateless detect_range, no resample."""
import numpy as np
import pandas as pd
import pytest

from goldenswing.live.range import RangeDetector, detect_range
from goldenswing.playbooks.range_regime import RangeSweepReclaimPlaybook


def _frames(n_h1=240):
    """Oscillating H1 (range 4100-4200) + its 4h aggregation (fixture)."""
    idx = pd.date_range("2026-01-01", periods=n_h1, freq="h", tz="UTC")
    t = np.arange(n_h1)
    mid = 4150.0 + 50.0 * np.sin(t / 9.0)
    close = mid + 2.0 * np.sin(t / 3.0)
    h1 = pd.DataFrame({
        "open": close - 1.0, "high": close + 5.0,
        "low": close - 5.0, "close": close,
        "volume": 100.0, "real_volume": 0.0, "spread": 0.25},
        index=idx)
    h4 = h1.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return h1, h4


def test_detect_range_no_h4_returns_none():
    h1, _ = _frames()
    assert detect_range(None, None, h1) is None  # no resample fallback


def test_detector_matches_stateless():
    h1, h4 = _frames()
    stateless = detect_range(None, h4, h1)
    det = RangeDetector()
    det.update(h4, h1)
    inc = det.detect()
    assert stateless is not None and inc is not None
    assert inc.high == pytest.approx(stateless.high, rel=1e-9)
    assert inc.low == pytest.approx(stateless.low, rel=1e-9)
    assert inc.atr == pytest.approx(stateless.atr, rel=1e-6)
    assert inc.touches_high == stateless.touches_high
    assert inc.touches_low == stateless.touches_low
    assert inc.compressed == stateless.compressed


def test_detector_incremental_chunks_match_full():
    h1, h4 = _frames()
    det = RangeDetector()
    # Feed in 3 chunks, as the live stack would append bars.
    for a, b in ((0, 80), (80, 160), (160, 240)):
        det.update(h4.iloc[a // 4:b // 4], h1.iloc[a:b])
    det.update(h4, h1)  # re-feed full frames: idempotent, no double-count
    inc = det.detect()
    stateless = detect_range(None, h4, h1)
    assert inc is not None and stateless is not None
    assert inc.high == pytest.approx(stateless.high, rel=1e-9)
    assert inc.low == pytest.approx(stateless.low, rel=1e-9)
    assert inc.atr == pytest.approx(stateless.atr, rel=1e-6)


def test_detector_none_when_trending():
    idx = pd.date_range("2026-01-01", periods=240, freq="h", tz="UTC")
    close = 4000.0 + 5.0 * np.arange(240)
    h1 = pd.DataFrame({
        "open": close - 1.0, "high": close + 5.0,
        "low": close - 5.0, "close": close,
        "volume": 100.0, "real_volume": 0.0, "spread": 0.25},
        index=idx)
    h4 = h1.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    det = RangeDetector()
    det.update(h4, h1)
    assert det.detect() is None
    assert detect_range(None, h4, h1) is None


def test_range_playbook_without_h4_emits_nothing():
    h1, _ = _frames()
    pb = RangeSweepReclaimPlaybook()
    assert pb.live_signal([], len(h1) - 1, h1) == []
