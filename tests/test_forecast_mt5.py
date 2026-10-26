"""MTF forecast tests (bundle v2, feature set mtf-v1).

The forecast's features come from the engine's ACTUAL M5/M15/H1/H4/
D1 closed bars — not one H1 frame with higher timeframes resampled
out of it. These tests cover: the MTF frame build (including the
no-lookahead causality proof), training + scoring roundtrip,
crossing-only alerts, full bundle validation, the horizon-aware
validation purge, and the automatic-training lifecycle primitives.
Training-bound: the file trains twice (one shared module bundle +
one background train) and takes minutes; run it standalone, not
inside the fast suite.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from goldenswing.forecast.mtf import (
    BUNDLE_VERSION,
    MIN_USABLE_ROWS_TRAIN,
    _purge_mask,
    _validate_bundle,
    build_mtf_frame,
    needs_training,
    score_last_row,
    train_from_frames,
)
from goldenswing.forecast.mt5 import (
    MT5Forecaster,
    format_forecast_alert,
    notable,
)


def _walk(n: int, seed: int, start: float = 4000.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start + np.cumsum(rng.normal(0, 1.5, size=n))


def _frame(n: int, freq: str, seed: int) -> pd.DataFrame:
    idx = pd.date_range("2026-06-01", periods=n, freq=freq, tz="UTC")
    close = _walk(n, seed)
    rng = np.random.default_rng(seed + 1)
    open_ = close + rng.normal(0, 0.4, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, size=n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx)


def _frames(days: int = 100) -> dict:
    """Synthetic stack frames over ``days`` days (all TFs present).

    100 days: 63 days of stacked D1 warmup (14 ATR + 50 percentile
    - 1) + 37 days (888 rows) usable — above the 500-row training
    gate, with 2400 raw H1 bars clearing the 2012-bar gate.
    """
    return {
        "M5": _frame(days * 288, "5min", 11),
        "M15": _frame(days * 96, "15min", 12),
        "H1": _frame(days * 24, "h", 13),
        "H4": _frame(days * 6, "4h", 14),
        "D1": _frame(days, "D", 15),
    }


@pytest.fixture(scope="module")
def bundle_and_summary(tmp_path_factory):
    path = tmp_path_factory.mktemp("forecast") / "forecast_models.joblib"
    summary = train_from_frames(_frames(), path)
    return path, summary


def test_mtf_frame_uses_all_timeframes():
    X = build_mtf_frame(_frames())
    prefixes = {c.split("_")[0] for c in X.columns if "_" in c}
    assert {"m5", "m15", "h1", "h4", "d1"} <= prefixes
    assert "x_trend_agree" in X.columns
    # Rows survive only after every TF's 50-bar warmup (D1: 50 days).
    assert len(X) >= 500


def test_mtf_frame_is_causal_future_bars_change_nothing():
    """No-lookahead proof: features for rows already computed do not
    change when future bars are appended. Trailing-only by test,
    not by assertion. Short = first 75 of 100 days (D1 still clears
    its 63-bar stacked warmup); full = 100 days."""
    full = _frames()
    short = {tf: df.iloc[: int(len(df) * 0.75)]
             for tf, df in full.items()}
    X_short = build_mtf_frame(short)
    X_full = build_mtf_frame(full)
    common = X_short.index.intersection(X_full.index)
    assert len(common) >= 100
    pd.testing.assert_frame_equal(
        X_short.loc[common], X_full.loc[common],
        check_exact=False, rtol=1e-9, atol=1e-9)


def test_purge_mask_keeps_label_windows_out_of_holdout():
    mask = _purge_mask(1000, 800, 360)
    assert mask.sum() == 440  # i < 800 and i + 360 < 800 -> i < 440
    assert bool(mask[439]) and not bool(mask[440]) and not bool(mask[799])
    assert _purge_mask(1000, 800, 12).sum() == 788


def test_missing_bundle_is_honest(tmp_path):
    fc = MT5Forecaster(bundle_path=tmp_path / "nope.joblib")
    text, note = fc.maybe_alert(_frames())
    assert text is None
    assert "no forecaster bundle" in note
    assert fc.status_dict(h1_bars=2400)["ready"] is False


def test_train_and_score_roundtrip(bundle_and_summary):
    bundle_path, summary = bundle_and_summary
    assert summary["version"] == BUNDLE_VERSION
    assert summary["feature_set"] == "mtf-v1"
    assert summary["train_bars"] >= 500
    bundle = joblib.load(bundle_path)
    valid, reason = _validate_bundle(bundle)
    assert valid, reason
    # Purge visible in validation counts: the 15d horizon loses more
    # train rows near the split than the 12h horizon does.
    val = summary["validation"]
    assert val[360]["n_train"] < val[12]["n_train"]
    ranks, note = score_last_row(_frames(), bundle)
    assert ranks is not None, note
    assert set(ranks.horizons) == {12, 24, 72, 168, 360}
    for h in ranks.horizons:
        for d in ("up", "dn"):
            r = ranks.rank(h, d)
            assert np.isfinite(r) and 0.0 <= r <= 100.0
    assert "not a trade signal" in format_forecast_alert(ranks)


def test_notable_picks_strongest_gating_rank():
    from goldenswing.forecast.mt5 import ForecastRanks

    ranks = ForecastRanks(
        bar_time=pd.Timestamp("2026-10-25", tz="UTC"), close=4013.7,
        ranks={24: {"up": 85.0, "dn": 10.0},
               72: {"up": 20.0, "dn": 91.0},
               168: {"up": 99.0, "dn": 5.0}},
        horizons=(24, 72, 168))
    assert notable(ranks) == ("dn", 72, 91.0)  # 7d never gates
    assert notable(ranks, threshold=95.0) is None


def test_forecaster_alerts_on_crossing_only(bundle_and_summary):
    bundle_path, _ = bundle_and_summary
    fc = MT5Forecaster(bundle_path=bundle_path, threshold=0.0)
    frames = _frames()
    text, _ = fc.maybe_alert(frames)
    assert text is not None  # threshold 0: first scoring crosses
    text2, _ = fc.maybe_alert(frames)
    assert text2 is None  # persisting above threshold stays quiet
    fc._above = {}  # dropping below re-arms: simulate via high threshold
    fc.threshold = 101.0
    text3, _ = fc.maybe_alert(frames)
    assert text3 is None
    fc.threshold = 0.0
    text4, _ = fc.maybe_alert(frames)
    assert text4 is not None  # re-armed crossing fires again


def test_forecaster_caches_bundle_across_bars(bundle_and_summary):
    bundle_path, _ = bundle_and_summary
    fc = MT5Forecaster(bundle_path=bundle_path)
    fc.maybe_alert(_frames())
    first_mtime = fc._bundle_mtime
    fc.maybe_alert(_frames())
    assert fc._bundle_mtime == first_mtime
    assert fc._bundle is not None


def test_needs_training_states(tmp_path, bundle_and_summary):
    bundle_path, _ = bundle_and_summary
    frames = _frames()
    h1 = frames["H1"]
    missing = tmp_path / "missing.joblib"
    need, reason = needs_training(missing, h1)
    assert need and "no forecaster bundle" in reason
    need, reason = needs_training(bundle_path, h1)
    assert not need and "bundle current" in reason
    # Stale: 500+ new H1 bars after trained_through.
    longer_h1 = _frame(len(h1) + 600, "h", 13)
    need, _ = needs_training(bundle_path, longer_h1)
    assert need
    # Corrupt file -> retrain, never crash.
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a bundle")
    need, reason = needs_training(corrupt, h1)
    assert need and "unloadable" in reason
    # Old-format (v1-shaped) bundle -> invalid -> retrain.
    bundle = joblib.load(bundle_path)
    bundle.pop("version")
    old_path = tmp_path / "old.joblib"
    joblib.dump(bundle, old_path)
    need, reason = needs_training(old_path, h1)
    assert need and "invalid" in reason


def test_bundle_validation_rejects_partial_bundles(bundle_and_summary):
    bundle_path, _ = bundle_and_summary
    bundle = joblib.load(bundle_path)
    assert _validate_bundle(bundle)[0]
    for mutate in (
        lambda b: b.update(features=[]),
        lambda b: b.update(horizons={}),
        lambda b: b.update(ref_quantiles=None),
        lambda b: b.update(symbol="EURUSD"),
    ):
        tampered = dict(bundle)
        mutate(tampered)
        assert not _validate_bundle(tampered)[0]


def test_train_writes_atomically_and_maybe_train_background(tmp_path):
    bundle_path = tmp_path / "forecast_models.joblib"
    fc = MT5Forecaster(bundle_path=bundle_path)
    thread, note = fc.maybe_train(_frames())
    assert thread is not None, note
    thread.join(timeout=600)
    assert bundle_path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    ok, note = fc._train_result
    assert ok, note
    assert fc.status_dict(h1_bars=2400)["ready"] is True
    # Already running guard.
    fc._training = True
    thread2, note2 = fc.maybe_train(_frames())
    assert thread2 is None and "already running" in note2
    fc._training = False
    # Too few bars: no thread, truthful note.
    short = {"H1": _frame(100, "h", 13)}
    thread3, note3 = fc.maybe_train(short)
    assert thread3 is None and "too few usable MTF rows" in note3
    assert MIN_USABLE_ROWS_TRAIN == 500


def test_train_frame_validation_errors(tmp_path):
    with pytest.raises(ValueError, match="no H1 frame"):
        train_from_frames({}, tmp_path / "x.joblib")
    with pytest.raises(ValueError, match="DatetimeIndex"):
        train_from_frames(
            {"H1": pd.DataFrame({"close": [1.0, 2.0]})},
            tmp_path / "x.joblib")
    bad = _frames()
    bad["H1"] = bad["H1"].drop(columns=["close"])
    with pytest.raises(ValueError, match="lacks 'close'"):
        train_from_frames(bad, tmp_path / "x.joblib")
