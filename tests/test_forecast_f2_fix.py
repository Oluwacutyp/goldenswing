"""Forecast F2 fix — horizon-aware OOF folds, explicit
uncalibrated state, calibration-aware retrain trigger.

Pins the live failure, replayed at live size (the Nov 4
artifact trained on 733 usable MTF rows):

- the old OOF fold design (initial max(200, n//2)) left h=360
  with 7 OOF rows from a 6-row model, so its calibrator was
  None by construction and the horizon never stated, never
  resolved, never graduated — silently;
- the new design tiles the purge-usable span from the first
  fold that can actually train, roughly doubling OOF coverage
  on the short horizons and naming h=360 uncalibrated with
  its pair count and the reason (longer training window);
- a live book ECE outside the graduation gate now marks the
  artifact stale (the Nov 13-18 drift: h72 graduated, then
  fell to ECE 0.127 while the 500-bar / 21-day triggers
  slept), with an age floor so the cumulative book cannot
  refire the trigger on the cycle right after a retrain.

The gates themselves are pinned unchanged: GRAD_MIN_N 60,
GRAD_MAX_ECE 0.10, MIN_OOF_PAIRS 100 — no bar is lowered.
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from goldenswing.forecast.trained import (
    GRAD_MAX_ECE, GRAD_MIN_N, MIN_OOF_PAIRS, ProbabilityBook,
    TrainedOutlook, _oof_fold_params, needs_retrain,
    retrain_accrual, train_trained)
from goldenswing.forecast.walkforward import make_folds


def _frames(bars: int = 800, seed: int = 7) -> dict:
    """H1-only stack frames at live size: 800 bars -> ~733
    usable MTF rows, the Nov 4 artifact's window.  Momentum
    random walk (features carry a little real signal), hourly,
    tz-naive like the engine's test frames."""
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
def live_sized(tmp_path_factory):
    frames = _frames()
    path = tmp_path_factory.mktemp("f2fix") / "trained.joblib"
    summary = train_trained(frames, path)
    service = TrainedOutlook(path)
    art, note = service.artifact()
    assert art is not None, note
    return frames, path, summary, service, art


# ── the gates are not lowered ───────────────────────────────────


def test_graduation_gates_unchanged():
    assert GRAD_MIN_N == 60
    assert GRAD_MAX_ECE == 0.10
    assert MIN_OOF_PAIRS == 100


# ── horizon-aware OOF fold placement (pure, no training) ───────


def test_oof_fold_params_leave_no_usable_span_for_h360_live():
    # Live window, 733 rows: a 360-bar purge needs a fold start
    # in [460, 373) — empty.  No honest OOF coverage exists.
    # (known is realistic here: a row's label exists only when
    # its full forward window lies inside the data.)
    first, _step = _oof_fold_params(733, 360)
    assert first >= 733 - 360
    known = np.arange(733) < (733 - 360)
    folds = make_folds(733, known, 360,
                       initial_train=first, step=100)
    assert folds == []


def test_oof_fold_params_open_the_span_where_it_exists():
    # h=168 on the live window: first fold can train at 268 and
    # test rows run to 565 — ~297 out-of-fold rows, not 199.
    first, step = _oof_fold_params(733, 168)
    assert first == 268
    known = np.arange(733) < (733 - 168)
    folds = make_folds(733, known, 168,
                       initial_train=first, step=step)
    assert len(folds) >= 2
    oof_rows = sum(len(f.test_idx) for f in folds)
    assert oof_rows >= 250
    # h=360 on a 2600-row window: the span exists again.
    first360, _ = _oof_fold_params(2600, 360)
    assert first360 == 460
    assert 2600 - 360 > first360


# ── live-sized window: coverage up, h360 named uncalibrated ────


def test_live_sized_window_oof_coverage(live_sized):
    _frames_, _path, _summary, _svc, art = live_sized
    assert art["train_rows"] >= 500
    oof = {h: art["horizons"][h]["oof_n"]
           for h in art["horizons"]}
    # Before the fix (same window, old folds): 355 / 343 / 295 /
    # 199.  The floors below sit above the old values with room
    # for folds a thin early model legitimately skips.
    assert oof[12] >= 400, oof
    assert oof[24] >= 400, oof
    assert oof[72] >= 400, oof
    # h168: the fold UNION is 302 rows (pinned by the pure fold
    # test above), but this fixture's first fold trains on 100
    # rows that are all one direction class (seed-7 walk trends
    # down hard early), so that fold is legitimately skipped —
    # a model trained on one class cannot produce out-of-fold
    # probabilities — and its 100 test rows carry none.  The
    # realized 202 still clears the calibration floor with 2x
    # margin and the horizon calibrates (next test).  On real
    # data (reproduce_train.json) h168 realizes 302.
    assert oof[168] >= 2 * MIN_OOF_PAIRS, oof
    assert oof[360] < MIN_OOF_PAIRS, oof


def test_live_sized_window_short_horizons_calibrated(live_sized):
    frames, _path, _summary, service, art = live_sized
    for h in (12, 24, 72, 168):
        cal = art["horizons"][h]["calibration"]
        assert cal["status"] == "calibrated", (h, cal)
        assert cal["dir_calibrated"] is True
    close = float(frames["H1"]["close"].iloc[-1])
    preds = service.predict(frames, close=close, atr=8.0)
    probs = preds[24]["fields"]["probabilities"]
    assert probs["state"] == "calibrated"
    total = probs["up"] + probs["dn"] + probs["neutral"]
    assert abs(total - 1.0) < 0.01
    assert preds[24]["shadow"] is not None


def test_h360_uncalibrated_with_reason_on_live_window(live_sized):
    frames, _path, summary, service, art = live_sized
    cal = art["horizons"][360]["calibration"]
    assert cal["status"] == "uncalibrated"
    assert cal["oof_n"] < MIN_OOF_PAIRS
    assert "out-of-fold" in cal["reason"]
    assert "longer training window" in cal["reason"]
    assert 360 in summary["uncalibrated_horizons"]
    assert summary["calibration"][360] == "uncalibrated"
    close = float(frames["H1"]["close"].iloc[-1])
    preds = service.predict(frames, close=close, atr=8.0)
    probs = preds[360]["fields"]["probabilities"]
    assert probs["state"] == "uncalibrated"
    assert "out-of-fold" in probs["reason"]
    assert probs["oof_n"] == cal["oof_n"]
    # No shadow: nothing stated, nothing to resolve — and the
    # record says why instead of staying silent.
    assert preds[360]["shadow"] is None


# ── calibration-aware retrain trigger ───────────────────────────


def _drifted_book(horizon: int, n: int = 80) -> ProbabilityBook:
    """A book that has spoken and says miscalibrated: stated
    up 0.9, realized dn, n resolutions."""
    book = ProbabilityBook()
    probs = {"up": 0.9, "dn": 0.05, "neutral": 0.05}
    for _ in range(n):
        assert book.add(horizon, "trained", probs, "dn")
    ok, info = book.graduated(horizon)
    assert not ok and info["n"] == n
    assert info["ece"] > GRAD_MAX_ECE
    return book


def _good_book(horizon: int, n: int = 80) -> ProbabilityBook:
    book = ProbabilityBook()
    probs = {"up": 0.7, "dn": 0.2, "neutral": 0.1}
    realized = (["up"] * 7 + ["dn"] * 2 + ["neutral"]) * 8
    assert len(realized) == n
    for r in realized:
        assert book.add(horizon, "trained", probs, r)
    ok, info = book.graduated(horizon)
    assert ok, info
    return book


def test_calibration_drift_triggers_retrain_once_aged(live_sized):
    frames, _path, _summary, _svc, art = live_sized
    idx = frames["H1"].index
    now = pd.Timestamp(art["trained_at"]).to_pydatetime()
    book = _drifted_book(24)

    # Fresh artifact: the drift is REPORTED but does not fire —
    # the book is cumulative, so firing now would refire on the
    # cycle right after every retrain.
    acc = retrain_accrual(art, idx, now, book=book)
    assert acc["calibration"]["due"] is True
    assert acc["calibration"]["horizons"][24]["outside_gate"] \
        is True
    assert acc["due"] is False
    assert acc["cause"] is None
    assert "calibration drift noted" in acc["reason"]
    assert needs_retrain(art, idx, now, book=book)[0] is False

    # Same book, artifact 5 days old: due, cause named.
    aged = dict(art)
    aged["trained_at"] = str(
        pd.Timestamp(art["trained_at"])
        - pd.Timedelta(days=5))
    acc = retrain_accrual(aged, idx, now, book=book)
    assert acc["due"] is True and acc["cause"] == "calibration"
    assert "ECE" in acc["reason"]
    assert needs_retrain(aged, idx, now, book=book)[0] is True


def test_calibration_trigger_ignores_good_books_and_no_book(
        live_sized):
    frames, _path, _summary, _svc, art = live_sized
    idx = frames["H1"].index
    now = pd.Timestamp(art["trained_at"]).to_pydatetime()
    aged = dict(art)
    aged["trained_at"] = str(
        pd.Timestamp(art["trained_at"])
        - pd.Timedelta(days=5))

    # A well-calibrated horizon never triggers, however old.
    acc = retrain_accrual(aged, idx, now, book=_good_book(24))
    assert acc["calibration"]["due"] is False
    assert acc["due"] is False
    # No book passed: the old behaviour, unchanged.
    assert needs_retrain(aged, idx, now)[0] is False
    assert needs_retrain(art, idx, now)[0] is False


def test_calibration_trigger_ignores_horizons_it_cannot_state(
        live_sized):
    # h=360 is uncalibrated on this window: the artifact states
    # nothing for it, so a drifted h360 book entry (left over
    # from an older artifact that could state) must not force
    # retrains of a window that cannot fix it.
    frames, _path, _summary, _svc, art = live_sized
    idx = frames["H1"].index
    now = pd.Timestamp(art["trained_at"]).to_pydatetime()
    aged = dict(art)
    aged["trained_at"] = str(
        pd.Timestamp(art["trained_at"])
        - pd.Timedelta(days=5))
    book = _drifted_book(360)
    acc = retrain_accrual(aged, idx, now, book=book)
    assert 360 not in acc["calibration"]["horizons"]
    assert acc["calibration"]["due"] is False
    assert needs_retrain(aged, idx, now, book=book)[0] is False


def test_bar_trigger_still_wins_over_calibration(live_sized):
    frames, _path, _summary, _svc, art = live_sized
    idx = frames["H1"].index
    now = pd.Timestamp(art["trained_at"]).to_pydatetime()
    aged = dict(art)
    aged["trained_at"] = str(
        pd.Timestamp(art["trained_at"])
        - pd.Timedelta(days=5))
    ext = pd.date_range(idx[-1] + pd.Timedelta(hours=1),
                        periods=600, freq="h")
    acc = retrain_accrual(aged, idx.append(ext), now,
                          book=_drifted_book(24))
    assert acc["due"] is True and acc["cause"] == "new_bars"
    need, reason = needs_retrain(aged, idx.append(ext), now,
                                 book=_drifted_book(24))
    assert need is True
    assert reason == acc["reason"]


def test_accrual_without_book_has_no_calibration_section(
        live_sized):
    # Backward compatibility: old callers get the same shape
    # plus an empty calibration block, and the verdict agrees.
    frames, _path, _summary, _svc, art = live_sized
    idx = frames["H1"].index
    now = datetime.datetime.now(datetime.timezone.utc)
    acc = retrain_accrual(art, idx, now)
    assert acc["calibration"] == {"horizons": {}, "due": False,
                                  "cause": None, "reason": ""}
    assert needs_retrain(art, idx, now) == (False, acc["reason"])
