"""Forecast F2 — trained estimators + calibration.

Pins: trained artifact shape + stamps; the walk-forward cutoff
(a future bar cannot influence the artifact — trained_through
never passes the cutoff and predictions at the cutoff are
identical whether or not wild future bars were appended);
isotonic-calibrated probabilities summing to one; the live
probability book's graduation gates (n floor, ECE bound, source
separation — an empirical record never graduates the trained
estimator); the outlook integration (shadow until graduated,
whole-field substitution after, disagreement preserved);
contradiction narration built from the computed fields only; a
failed retrain leaving the previous artifact live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goldenswing.forecast.outlook import (
    OUTLOOK_METHOD, TRAINED_METHOD, build_outlook,
    contradiction_narration)
from goldenswing.forecast.trained import (
    GRAD_MAX_ECE, GRAD_MIN_N, ProbabilityBook, TrainedOutlook,
    direction_class, needs_retrain, stated_probabilities,
    train_trained)


def _frames(bars: int = 2600, seed: int = 7) -> dict:
    """H1-only stack frames: a momentum random walk (features
    carry a little real signal), hourly, tz-naive like the
    engine's test frames."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1.5, bars)
    for i in range(2, bars):
        steps[i] += 0.25 * steps[i - 1]
    close = 2000.0 + np.cumsum(steps)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 1.2, bars)) + 0.5
    idx = pd.date_range("2024-01-01", periods=bars, freq="h")
    h1 = pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close}, index=idx)
    return {"H1": h1}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    frames = _frames()
    path = tmp_path_factory.mktemp("f2") / "trained.joblib"
    summary = train_trained(frames, path)
    service = TrainedOutlook(path)
    return frames, path, summary, service


# ── training + prediction ───────────────────────────────────────


def test_train_summary_and_artifact_stamps(trained):
    frames, _path, summary, service = trained
    assert summary["version"] == TRAINED_METHOD
    assert summary["trained_through"] == str(frames["H1"].index[-1])
    assert 24 in summary["horizons"]
    art, note = service.artifact()
    assert art is not None, note
    assert art["train_rows"] >= 500
    assert art["trained_at"] and art["train_start"]


def test_predict_fields_mirror_the_f1_shape(trained):
    frames, _path, _summary, service = trained
    close = float(frames["H1"]["close"].iloc[-1])
    preds = service.predict(frames, close=close, atr=8.0)
    assert 24 in preds
    fields = preds[24]["fields"]
    assert fields["method"] == TRAINED_METHOD
    assert fields["direction"] in ("UP", "DOWN", "NEUTRAL")
    iv = fields["interval"]
    assert iv["low"] <= fields["point_price"] <= iv["high"]
    assert iv["coverage"] == 0.80
    assert fields["trained_at"]
    window = fields["train_window"]
    assert window["through"] == str(frames["H1"].index[-1])
    assert window["rows"] >= 500
    assert fields["barriers"] is not None
    assert fields["scenario"] and fields["invalidation"]


def test_predict_probabilities_calibrated_and_sum_to_one(trained):
    frames, _path, _summary, service = trained
    close = float(frames["H1"]["close"].iloc[-1])
    preds = service.predict(frames, close=close, atr=8.0)
    probs = preds[24]["fields"]["probabilities"]
    assert probs["state"] == "calibrated"
    total = probs["up"] + probs["dn"] + probs["neutral"]
    assert abs(total - 1.0) < 0.01
    assert "isotonic" in probs["basis"]
    shadow = preds[24]["shadow"]
    assert shadow is not None and shadow["state"] == "shadow"
    assert shadow["up"] == probs["up"]


def test_no_artifact_answers_nothing(tmp_path):
    service = TrainedOutlook(tmp_path / "absent.joblib")
    frames = _frames(300)
    assert service.predict(frames, close=2000.0, atr=8.0) == {}
    art, note = service.artifact()
    assert art is None and "no trained artifact" in note


# ── the walk-forward cutoff (leakage guard) ─────────────────────


def test_cutoff_future_bars_cannot_influence_the_artifact(tmp_path):
    frames = _frames()
    h1 = frames["H1"]
    cut = h1.index[1800]
    truncated = {"H1": h1.loc[:cut].copy()}

    path_a = tmp_path / "a.joblib"
    train_trained(truncated, path_a)
    svc_a = TrainedOutlook(path_a)
    close_t = float(truncated["H1"]["close"].iloc[-1])
    pred_a = svc_a.predict(truncated, close=close_t, atr=8.0)

    # The same history, but with wild future bars appended AFTER
    # the cutoff, trained with through=cut: if the cutoff leaked,
    # the training window (and so the predictions at the cutoff)
    # would move.
    wild = h1.copy()
    post = wild.index > cut
    wild.loc[post, "close"] = wild.loc[post, "close"] * 3.0
    wild.loc[post, "high"] = wild.loc[post, "high"] * 3.0
    wild.loc[post, "low"] = wild.loc[post, "low"] * 3.0
    path_b = tmp_path / "b.joblib"
    summary_b = train_trained({"H1": wild}, path_b, through=cut)
    assert pd.Timestamp(summary_b["trained_through"]) <= cut
    svc_b = TrainedOutlook(path_b)
    pred_b = svc_b.predict(truncated, close=close_t, atr=8.0)

    assert pred_a.keys() == pred_b.keys()
    for h in pred_a:
        fa, fb = pred_a[h]["fields"], pred_b[h]["fields"]
        assert fa["point_price"] == fb["point_price"]
        assert fa["direction"] == fb["direction"]
        pa, pb = fa["probabilities"], fb["probabilities"]
        assert pa.get("up") == pb.get("up")
        assert pa.get("dn") == pb.get("dn")


def test_failed_retrain_leaves_previous_artifact_live(trained,
                                                       tmp_path):
    _frames_, src, _summary, _svc = trained
    path = tmp_path / "keep.joblib"
    path.write_bytes(src.read_bytes())
    before = path.read_bytes()
    service = TrainedOutlook(path)
    art_before, _ = service.artifact()

    empty = pd.DataFrame(columns=["open", "high", "low", "close"],
                         index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError):
        train_trained({"H1": empty}, path)
    with pytest.raises(ValueError):
        train_trained(_frames(120), path)  # below the row floor

    assert path.read_bytes() == before
    art_after, _ = service.artifact()
    assert art_after["trained_at"] == art_before["trained_at"]


# ── the live probability book + graduation ──────────────────────


def _fill(book, horizon, source, probs, realized_seq):
    for realized in realized_seq:
        assert book.add(horizon, source, probs, realized)


def test_book_graduation_needs_n_and_calibration():
    book = ProbabilityBook()
    ok, info = book.graduated(24)
    assert not ok and info["n"] == 0

    probs = {"up": 0.7, "dn": 0.2, "neutral": 0.1}
    realized = (["up"] * 7 + ["dn"] * 2 + ["neutral"]) * 8  # n=80
    _fill(book, 24, "trained", probs, realized)
    ok, info = book.graduated(24)
    assert ok, info
    assert info["n"] == 80 and info["ece"] <= GRAD_MAX_ECE

    # The same record under the empirical source must NOT
    # graduate the trained estimator (source separation).
    book2 = ProbabilityBook()
    _fill(book2, 24, "empirical", probs, realized)
    ok2, _ = book2.graduated(24)
    assert not ok2
    view = book2.view()["horizons"]["24"]
    assert view["empirical"]["n"] == 80
    assert "trained" not in view


def test_book_miscalibration_never_graduates():
    book = ProbabilityBook()
    probs = {"up": 0.9, "dn": 0.05, "neutral": 0.05}
    _fill(book, 24, "trained", probs, ["dn"] * (GRAD_MIN_N + 20))
    ok, info = book.graduated(24)
    assert not ok
    assert info["n"] >= GRAD_MIN_N and info["ece"] > GRAD_MAX_ECE


def test_book_ignores_unmeasurable_resolutions():
    book = ProbabilityBook()
    assert not book.add(24, "trained", {"up": 0.5}, None)
    assert not book.add(24, "bogus", {"up": 0.5}, "up")
    assert book.view()["horizons"] == {}


def test_direction_class_uses_the_stated_deadband():
    assert direction_class(0.5, 0.1) == "up"
    assert direction_class(-0.5, 0.1) == "dn"
    assert direction_class(0.05, 0.1) == "neutral"
    assert direction_class(None, 0.1) is None


def test_stated_probabilities_prefers_trained_then_empirical():
    outlook = {"horizons": {
        "24": {"method": TRAINED_METHOD, "deadband_pct": 0.2,
               "probabilities": {"state": "calibrated", "up": 0.5,
                                 "dn": 0.3, "neutral": 0.2}},
        "72": {"method": "unconditional_h1", "deadband_pct": 0.4,
               "probabilities": {"state": "uncalibrated"},
               "trained_shadow": {"up": 0.4, "dn": 0.4,
                                  "neutral": 0.2,
                                  "deadband_pct": 0.35}},
        "168": {"method": "band_conditional", "deadband_pct": 0.5,
                "probabilities": {"state": "calibrated", "up": 0.6,
                                  "dn": 0.25, "neutral": 0.15}},
        "360": {"method": "unconditional_h1",
                "probabilities": {"state": "uncalibrated"}},
    }}
    stated = stated_probabilities(outlook)
    assert stated["24"]["source"] == "trained"
    assert stated["72"]["source"] == "trained"  # the shadow
    assert stated["72"]["deadband_pct"] == 0.35
    assert stated["168"]["source"] == "empirical"
    assert "360" not in stated


# ── outlook integration: shadow first, substitution on trust ────


def _f1_inputs(bars: int = 240):
    idx = pd.date_range("2026-01-01", periods=bars, freq="h")
    close = 2000.0 + np.cumsum(np.sin(np.arange(bars) / 9.0) * 3)
    h1 = pd.DataFrame({
        "open": close, "high": close + 4.0, "low": close - 4.0,
        "close": close}, index=idx)
    ranks = {"24": {"up": 85.0, "dn": 20.0}}
    return h1, ranks


def _trained_param(preds, *, graduated: bool):
    horizons = {}
    for h, pred in preds.items():
        horizons[str(h)] = {**pred, "graduated": graduated}
    return {"version": TRAINED_METHOD, "trained_at": "t",
            "trained_through": "x", "horizons": horizons}


def test_outlook_shadow_until_graduated_then_substitutes(trained):
    frames, _p, _s, service = trained
    close = float(frames["H1"]["close"].iloc[-1])
    preds = service.predict(frames, close=close, atr=8.0)
    assert 24 in preds
    h1, ranks = _f1_inputs()
    model_probs = {24: {"up": [0.7, 0.9], "dn": [0.2, 0.4]}}

    shadow_block = build_outlook(
        ranks=ranks, close=2000.0, atr=8.0, samples=[], h1=h1,
        model_probs=model_probs,
        trained=_trained_param(preds, graduated=False))
    ho = shadow_block["horizons"]["24"]
    assert ho["method"] != TRAINED_METHOD          # F1 answered
    assert ho["trained_shadow"]["state"] == "shadow"
    assert ho["trained"]["graduated"] is False
    assert shadow_block["method"] == OUTLOOK_METHOD
    assert shadow_block["trained"]["graduated_horizons"] == []

    grad_block = build_outlook(
        ranks=ranks, close=2000.0, atr=8.0, samples=[], h1=h1,
        model_probs=model_probs,
        trained=_trained_param(preds, graduated=True))
    ho = grad_block["horizons"]["24"]
    assert ho["method"] == TRAINED_METHOD
    assert ho["point_price"] == preds[24]["fields"]["point_price"]
    assert ho["probabilities"]["state"] == "calibrated"
    assert "trained_shadow" not in ho
    assert ho["trained"]["graduated"] is True
    # the ensemble disagreement survives the substitution
    assert ho["disagreement"] is not None
    assert grad_block["method"] == TRAINED_METHOD
    assert grad_block["trained"]["graduated_horizons"] == ["24"]
    assert "trained median" in grad_block["point_estimator"]


def test_outlook_without_trained_is_the_pure_f1_block():
    h1, ranks = _f1_inputs()
    block = build_outlook(ranks=ranks, close=2000.0, atr=8.0,
                          samples=[], h1=h1)
    ho = block["horizons"]["24"]
    assert block["method"] == OUTLOOK_METHOD
    assert "trained" not in ho and "trained" not in block


# ── narration + retrain triggers ────────────────────────────────


def test_narration_names_the_split_in_numbers():
    horizons = {
        "12": {"direction": "UP", "expected_move_pct": 0.42,
               "disagreement": 0.11},
        "24": {"direction": "UP", "expected_move_pct": 0.30,
               "disagreement": 0.09},
        "168": {"direction": "DOWN", "expected_move_pct": -0.55,
                "disagreement": 0.21},
        "360": {"direction": "DOWN", "expected_move_pct": -0.61,
                "disagreement": 0.18},
    }
    text = contradiction_narration(horizons)
    assert text is not None
    assert "12h" in text and "15d" in text
    assert "+0.42%" in text and "-0.61%" in text
    assert "clock split" in text
    assert "7d" in text and "0.21" in text  # disagreement peak

    aligned = {"24": {"direction": "UP", "expected_move_pct": 0.3,
                      "disagreement": None},
               "72": {"direction": "UP", "expected_move_pct": 0.5,
                      "disagreement": None}}
    assert contradiction_narration(aligned) is None


def test_needs_retrain_triggers(trained):
    import datetime
    frames, _p, _s, service = trained
    art, _ = service.artifact()
    idx = frames["H1"].index
    now = pd.Timestamp(art["trained_at"]).to_pydatetime()
    need, _ = needs_retrain(None, idx, now)
    assert need
    need, reason = needs_retrain(art, idx, now)
    assert not need, reason
    # 600 fresh bars past the training cutoff -> retrain
    extra = pd.date_range(idx[-1] + pd.Timedelta(hours=1),
                          periods=600, freq="h")
    need, _ = needs_retrain(art, idx.append(extra), now)
    assert need
    # an old artifact -> retrain even with no new bars
    need, _ = needs_retrain(
        art, idx, now + datetime.timedelta(days=30))
    assert need
