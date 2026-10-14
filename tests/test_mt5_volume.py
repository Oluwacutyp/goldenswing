"""Volume features — the old bots' usage: 20-bar mean ratio, z-score,
VWAP, OBV, and the bounded confirmation bonus."""
import numpy as np
import pandas as pd
import pytest

from goldenswing.live.volume import (volume_bonus, volume_features)


def _frame(volumes, closes=None, real=None):
    n = len(volumes)
    closes = closes if closes is not None else [2000.0 + i for i in range(n)]
    data = {"open": closes, "high": closes, "low": closes,
            "close": closes, "volume": volumes}
    if real is not None:
        data["real_volume"] = real
    idx = pd.date_range("2026-10-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(data, index=idx)


def test_volume_ratio_uses_20_bar_mean_including_last():
    # 19 bars at 100, last at 200 -> mean = 105, ratio = 200/105.
    vols = [100.0] * 19 + [200.0]
    f = volume_features(_frame(vols))
    assert f["volume"] == 200.0
    assert f["volume_mean20"] == pytest.approx(105.0)
    assert f["volume_ratio"] == pytest.approx(200.0 / 105.0)
    assert f["volume_confirms"] is True
    assert f["volume_surge"] is False
    assert f["status"] == "ok"


def test_volume_surge_at_twice_average():
    vols = [100.0] * 19 + [300.0]
    f = volume_features(_frame(vols))
    assert f["volume_ratio"] == pytest.approx(300.0 / 110.0)
    assert f["volume_surge"] is True


def test_volume_below_average_does_not_confirm():
    vols = [100.0] * 19 + [50.0]
    f = volume_features(_frame(vols))
    assert f["volume_confirms"] is False


def test_vol_z_vwap_obv():
    vols = [100.0] * 20
    closes = [2000.0 + i for i in range(20)]
    f = volume_features(_frame(vols, closes))
    assert f["vol_z"] == pytest.approx(0.0)
    assert f["vwap"] == pytest.approx(np.mean(closes))
    # OBV: every close up -> sum of all volumes except the first bar
    # (no prior close -> sign 0), i.e. 19 * 100.
    assert f["obv"] == pytest.approx(1900.0)


def test_real_volume_flagged_honestly():
    f0 = volume_features(_frame([100.0] * 20, real=[0] * 20))
    assert f0["real_volume_available"] is False
    assert f0["real_volume"] is None
    f1 = volume_features(_frame([100.0] * 20, real=[0] * 19 + [42.0]))
    assert f1["real_volume_available"] is True
    assert f1["real_volume"] == 42.0


def test_missing_or_zero_volume_degrades():
    f = volume_features(pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.date_range("2026-10-01", periods=2, freq="h", tz="UTC")))
    assert f["volume_ratio"] is None and f["status"] == "no_volume"
    f2 = volume_features(_frame([0.0] * 20))
    assert f2["volume_ratio"] is None


def test_volume_bonus_old_bot_formula():
    assert volume_bonus({"volume_ratio": 1.0}) == (0.0, "volume_ratio=1.00")
    assert volume_bonus({"volume_ratio": 2.0})[0] == pytest.approx(0.5)
    assert volume_bonus({"volume_ratio": 0.0})[0] == pytest.approx(-0.5)
    assert volume_bonus({"volume_ratio": 3.0})[0] == pytest.approx(0.5)
    assert volume_bonus(None) == (0.0, "")
    assert volume_bonus({"volume_ratio": None}) == (0.0, "")
