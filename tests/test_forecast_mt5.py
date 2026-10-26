"""MT5 forecast service pins (owner rulings §3).

The service forecasts and alerts; it never trades. Tests use a
synthetic OHLC frame purely as a wiring fixture — it proves the
loader, scorer, alert gate, and trainer run end to end. It is NOT
training data for deployment: deployment trains on the engine's own
MT5 history.
"""

import numpy as np
import pandas as pd
import pytest

from goldenswing.forecast.mt5 import (
    DEFAULT_THRESHOLD,
    MT5Forecaster,
    format_forecast_alert,
    notable,
    score_last_bar,
    train_from_frame,
    ForecastRanks,
)


def _synthetic_frame(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    close = 2000.0 + np.cumsum(rng.normal(0, 2.0, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 1.5, size=n)
    low = np.minimum(open_, close) - rng.uniform(0, 1.5, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(50.0, 200.0, size=n)},
        index=idx)


def test_missing_bundle_is_honest(tmp_path):
    ranks, note = score_last_bar(_synthetic_frame(), tmp_path / "nope.joblib")
    assert ranks is None
    assert "no forecaster bundle" in note


def test_train_and_score_roundtrip(tmp_path):
    frame = _synthetic_frame()
    out = tmp_path / "forecast_models.joblib"
    summary = train_from_frame(frame, out)
    assert summary["horizons"] and all(v > 0 for v in summary["horizons"].values())
    ranks, note = score_last_bar(frame, out)
    assert ranks is not None, note
    assert ranks.bar_time == frame.index[-1]
    assert ranks.close == pytest.approx(float(frame["close"].iloc[-1]))
    for h in ranks.horizons:
        for direction in ("up", "dn"):
            r = ranks.rank(h, direction)
            assert np.isfinite(r) and 0.0 <= r <= 100.0


def test_notable_picks_strongest_gating_rank():
    ranks = ForecastRanks(
        bar_time="t", close=2000.0,
        ranks={24: {"up": 85.0, "dn": 30.0},
               72: {"up": 60.0, "dn": 90.0},
               168: {"up": 95.0, "dn": 10.0}},
        horizons=(24, 72, 168))
    # 7d's 95 does NOT gate; the strongest gating rank is dn 90 @3d.
    assert notable(ranks) == ("dn", 72, 90.0)
    assert notable(ranks, threshold=91.0) is None


def test_format_labels_forecast_not_signal():
    ranks = ForecastRanks(
        bar_time="2025-01-01", close=2000.0,
        ranks={24: {"up": 87.0, "dn": 20.0}, 72: {"up": 82.0, "dn": 15.0}},
        horizons=(24, 72))
    text = format_forecast_alert(ranks)
    assert "FORECAST" in text
    assert "not a trade signal" in text
    assert "not probabilities" in text
    assert "87" in text and "82" in text


def test_forecaster_alerts_on_crossing_only():
    # Crossing state machine, driven directly: a rank sitting above
    # threshold must not re-alert; dropping below re-arms it.
    forecaster = MT5Forecaster(bundle_path="none", threshold=80.0)
    above = ForecastRanks(
        bar_time="t1", close=2000.0,
        ranks={24: {"up": 87.0, "dn": 20.0}, 72: {"up": 60.0, "dn": 15.0}},
        horizons=(24, 72))
    below = ForecastRanks(
        bar_time="t2", close=2000.0,
        ranks={24: {"up": 40.0, "dn": 20.0}, 72: {"up": 60.0, "dn": 15.0}},
        horizons=(24, 72))
    text, _ = forecaster._decide(above)
    assert text is not None and "87" in text  # first crossing fires
    text, note = forecaster._decide(above)
    assert text is None and "no new threshold crossing" in note
    text, _ = forecaster._decide(below)
    assert text is None  # dropping below re-arms, alerts nothing
    text, _ = forecaster._decide(above)
    assert text is not None  # crossing again fires


def test_forecaster_real_path_quiet_while_persisting(tmp_path):
    frame = _synthetic_frame()
    out = tmp_path / "forecast_models.joblib"
    train_from_frame(frame, out)
    forecaster = MT5Forecaster(bundle_path=out, threshold=0.0)
    text, note = forecaster.maybe_alert(frame)
    assert text is not None, note
    assert "FORECAST" in text
    frame2 = pd.concat([frame, frame.iloc[[-1]].copy()])
    frame2.index = pd.DatetimeIndex(list(frame.index) + [frame.index[-1] + pd.Timedelta(hours=1)])
    text2, note2 = forecaster.maybe_alert(frame2)
    assert text2 is None  # still above threshold: persistence is quiet
    assert "no new threshold crossing" in note2


def test_forecaster_caches_bundle_across_bars(tmp_path):
    frame = _synthetic_frame()
    out = tmp_path / "forecast_models.joblib"
    train_from_frame(frame, out)
    forecaster = MT5Forecaster(bundle_path=out, threshold=0.0)
    forecaster.maybe_alert(frame)
    first = forecaster._bundle
    assert first is not None
    # A new bar reuses the cached bundle dict object, no re-read.
    frame2 = pd.concat([frame, frame.iloc[[-1]].copy()])
    frame2.index = pd.DatetimeIndex(list(frame.index) + [frame.index[-1] + pd.Timedelta(hours=1)])
    forecaster.maybe_alert(frame2)
    assert forecaster._bundle is first
