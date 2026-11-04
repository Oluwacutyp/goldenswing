"""Forecast scalar-conversion regression (2026-11-04 live failure).

The owner's journal: the last ``forecast`` records land at
2026-11-04T00:00:42Z; from 01:43:20Z — the first scoring after
that morning's runner restart — every forecast cycle dies with
"forecast alert failed (loop continues): only 0-dimensional
arrays can be converted to Python scalars", and no forecast
record ever lands again.

Root: ``score_last_row`` carried the F1 per-model probabilities
as ``[float(x) for x in ps]`` where each ``x`` is a 1-element
ndarray (a single-row ``predict_proba`` column).  ``float()`` on
an ndim >= 1 array is a deprecation warning on NumPy 1.x and a
hard TypeError on NumPy >= 2 — the restarted runner was the
first process on the NumPy 2 stack.  These tests pin:

- ``_scalar`` accepts numbers / 0-d / 1-element arrays and
  raises the typed ``ScalarConversionError`` (naming the field)
  on genuinely multi-element values;
- scoring with live-shaped (1, 2) ``predict_proba`` outputs
  completes under the NumPy-2 scalar regime (the NumPy 1.x
  conversion deprecation promoted to an error — the local
  stand-in for the live TypeError);
- a broken model answering multi-element raises typed, and the
  forecaster's ``_last_error`` (the /status surface) records it;
- the runner's failure notes name the cadence + bar and count
  consecutive failures, and a full real cycle journals its
  forecast records on both cadences again.

Second root, same journal: forecast records only EVER landed
in restart/midnight pairs (16 total) — the D1 block's bare
``return``s sat in ``_maybe_forecast_alert`` itself, so on every
tick where the D1 day had not changed they skipped the
"independent" intraday H1 block too.  That is why the scalar
failure produced one pair of notes per process and then
silence: no retry ever ran.  The D1 half now lives in its own
method (``_forecast_alert_d1``) whose exits are local, and the
cadence-independence test below pins it.

The 2026-11-04T02:00:16Z ``forecast_trained_training`` event
(F2 artifact, 733 rows) is the same stoppage's second face, not
a third root: records were already dead from 01:43, and the
training neither caused nor could cure that.  But the trained
path had never run live, so the last two tests pin inference
continuity across BOTH retraining events — an ensemble bundle
swap and a landed F2 artifact — end to end.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from goldenswing.forecast.mtf import (
    ScalarConversionError,
    _scalar,
    build_mtf_frame,
    score_last_row,
)
from goldenswing.forecast.mt5 import MT5Forecaster
from mt5_fakes import FakeMT5


@pytest.fixture(autouse=True)
def _strict_scalar_regime():
    """Error on exactly the NumPy ndim>0 -> scalar conversion
    deprecation — the call class NumPy >= 2 turns into the live
    TypeError.  Scoped by message so unrelated deprecations in
    pandas/sklearn do not leak into these tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=".*Conversion of an array with ndim > 0.*",
            category=DeprecationWarning)
        yield


# ── _scalar ──────────────────────────────────────────────────────

def test_scalar_accepts_numbers_and_single_element_arrays():
    assert _scalar(2.5, field="f") == 2.5
    assert _scalar(3, field="f") == 3.0
    assert _scalar(np.float64(1.25), field="f") == 1.25
    assert _scalar(np.array(4.5), field="f") == 4.5          # 0-d
    assert _scalar(np.array([5.5]), field="f") == 5.5        # (1,)
    assert _scalar(np.array([[6.5]]), field="f") == 6.5      # (1, 1)


def test_scalar_rejects_multi_element_loudly():
    with pytest.raises(ScalarConversionError) as ei:
        _scalar(np.array([1.0, 2.0]), field="point price")
    assert "point price" in str(ei.value)
    assert "2 values" in str(ei.value)
    with pytest.raises(ScalarConversionError):
        _scalar(np.zeros((2, 2)), field="interval")


# ── frames + fake bundle (live shapes, no training) ─────────────

def _frame(n, freq, base=3900.0):
    idx = pd.date_range("2026-08-01", periods=n, freq=freq, tz="UTC")
    close = [base + (i % 9) * 0.5 for i in range(n)]
    return pd.DataFrame(
        {"open": close, "high": [c + 0.4 for c in close],
         "low": [c - 0.4 for c in close], "close": close,
         "volume": [100] * n, "real_volume": [0] * n,
         "spread": [30] * n},
        index=idx)


def _frames(h1_bars=1680):
    # 70 days: clears the 63-bar D1 warmup so the MTF frame has
    # usable rows at its tail (the rows scoring reads).
    return {"H1": _frame(h1_bars, "h"), "H4": _frame(420, "4h"),
            "D1": _frame(70, "D")}


class _Clf:
    """predict_proba with the live shape: (n_rows, 2)."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.tile([1.0 - self.p, self.p], (len(X), 1))


class _BrokenClf:
    """Answers two rows for a one-row input — the genuinely
    multi-element case that must fail loudly, never be picked
    apart silently."""

    def predict_proba(self, X):
        return np.array([[0.2, 0.8], [0.3, 0.7]])


def _bundle(frames, clf_up=None, clf_dn=None):
    X = build_mtf_frame(frames)
    quant = np.linspace(0.02, 0.98, 101)
    models = {
        "up": {"a": clf_up or _Clf(0.70), "b": _Clf(0.60)},
        "dn": {"a": clf_dn or _Clf(0.30), "b": _Clf(0.40)},
    }
    return {
        "version": "forecast-bundle-v2", "feature_set": "mtf-v1",
        "trained_through": "2026-09-01", "trained_at": "2026-09-01",
        "features": list(X.columns),
        "horizons": {24: {"models": models}},
        "ref_quantiles": {24: {"up": quant, "dn": quant}},
    }


# ── score_last_row under the NumPy-2 scalar regime ──────────────

def test_score_last_row_live_shaped_probas_complete():
    frames = _frames()
    ranks, note = score_last_row(frames, _bundle(frames))
    assert ranks is not None, note
    assert ranks.horizons == (24,)
    assert np.isfinite(ranks.rank(24, "up"))
    # the F1 carry: per-model probabilities are plain floats now
    probs = ranks.model_probs[24]
    assert probs["up"] == [0.70, 0.60]
    assert all(type(v) is float for v in probs["up"] + probs["dn"])


def test_score_last_row_multi_element_fails_loudly():
    frames = _frames()
    bundle = _bundle(frames, clf_up=_BrokenClf())
    with pytest.raises(ScalarConversionError) as ei:
        score_last_row(frames, bundle)
    msg = str(ei.value)
    assert "horizon 24" in msg and "up" in msg


def test_forecaster_records_last_error_on_scoring_raise(tmp_path):
    frames = _frames()
    bundle = _bundle(frames, clf_up=_BrokenClf())
    fc = MT5Forecaster(bundle_path=tmp_path / "nope.joblib")
    fc._load_bundle = lambda: (bundle, "fake bundle")
    with pytest.raises(ScalarConversionError):
        fc.score_and_record(frames)
    # the /status surface tells the truth after a raise
    assert fc._last_error is not None
    assert "ScalarConversionError" in fc._last_error


# ── runner: visibility + the full real cycle ────────────────────

class _Journal:
    def __init__(self):
        self.notes = []
        self.records = []

    def log_note(self, text):
        self.notes.append(str(text))

    def append(self, record):
        self.records.append(record)


class _RaisingForecaster:
    _train_result = None

    def maybe_train(self, frames):
        return None, "bundle current"

    def score_and_record(self, frames):
        raise RuntimeError("only 0-dimensional arrays can be "
                           "converted to Python scalars")


def _runner(tmp_path):
    feed = MT5Feed(FakeMT5(), MT5FeedConfig(symbol="XAUUSDm"))
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False, max_daily_trades=0,
        max_hourly_trades=0)
    runner = MT5Runner(cfg, feed=feed)
    runner.journal = _Journal()
    return runner


def test_runner_failure_notes_name_cadence_bar_and_streak(tmp_path):
    runner = _runner(tmp_path)
    runner._forecaster = _RaisingForecaster()
    runner._stack = SimpleNamespace(frames={
        "D1": _frame(30, "D"), "H1": _frame(120, "h")})
    runner._maybe_forecast_alert()
    fails = [n for n in runner.journal.notes
             if "forecast alert failed" in n]
    assert len(fails) == 2
    assert "[D1 @" in fails[0] and "consecutive failures: 1" in fails[0]
    assert "[H1 @" in fails[1] and "consecutive failures: 2" in fails[1]
    # a fresh H1 bar retries the H1 cadence and the streak grows
    runner._stack = SimpleNamespace(frames={
        "D1": _frame(30, "D"), "H1": _frame(121, "h")})
    runner._maybe_forecast_alert()
    fails = [n for n in runner.journal.notes
             if "forecast alert failed" in n]
    assert len(fails) == 3
    assert "consecutive failures: 3" in fails[2]
    assert runner._forecast_consecutive_failures == 3


def _full_bundle(frames, up_ps=(0.70, 0.60), dn_ps=(0.30, 0.40)):
    """A validation-complete fake ensemble bundle (all five
    horizons, top-level models, list quantiles — the real
    bundle's shape), so the forecaster's real _load_bundle /
    _validate_bundle path accepts it.  trained_through is the
    frames' last H1 bar, so maybe_train reads the bundle as
    current and never spawns a background train in tests."""
    import joblib  # noqa: F401  (kept local: dump call sites read better)
    X = build_mtf_frame(frames)
    quant = np.linspace(0.02, 0.98, 101).tolist()

    def models(ps):
        return {f"m{i}": _Clf(p) for i, p in enumerate(ps)}

    horizons = {h: {"models": {"up": models(up_ps),
                                "dn": models(dn_ps)}}
                for h in (12, 24, 72, 168, 360)}
    return {
        "version": "forecast-bundle-v2", "symbol": "XAUUSD",
        "feature_set": "mtf-v1", "features": list(X.columns),
        "trained_through": str(frames["H1"].index[-1]),
        "trained_at": "2026-09-01T00:00:00+00:00",
        "models": {"up": models(up_ps), "dn": models(dn_ps)},
        "horizons": horizons,
        "ref_quantiles": {h: {"up": quant, "dn": quant}
                          for h in horizons},
    }


def test_ensemble_retrain_swap_keeps_the_cycle_running(tmp_path):
    """Inference continuity across an ENSEMBLE retraining event:
    a retrain swaps the bundle file atomically; the next bars
    must score against the NEW bundle (mtime reload) and keep
    journaling — never park, never keep scoring the old one."""
    import joblib
    import os

    frames = _frames()
    bundle_path = tmp_path / "forecast_models.joblib"
    joblib.dump(_full_bundle(frames), bundle_path)
    runner = _runner(tmp_path)
    runner._forecaster = MT5Forecaster(bundle_path=bundle_path)
    runner._stack = SimpleNamespace(frames=frames)
    runner._maybe_forecast_alert()
    forecasts = [r for r in runner.journal.records
                 if r.get("type") == "forecast"]
    assert forecasts, "first cycle must journal"
    assert forecasts[0]["horizons"]["24"]["up"] == pytest.approx(
        66.0, abs=2.0)

    # the retraining event: a new bundle lands on the same path
    joblib.dump(_full_bundle(frames, up_ps=(0.75, 0.72)), bundle_path)
    bumped = bundle_path.stat().st_mtime + 5
    os.utime(bundle_path, (bumped, bumped))
    runner._stack = SimpleNamespace(frames=_frames(h1_bars=1681))
    runner._maybe_forecast_alert()
    assert not any("forecast alert failed" in n
                   for n in runner.journal.notes)
    forecasts = [r for r in runner.journal.records
                 if r.get("type") == "forecast"]
    assert len(forecasts) > 1, "cycle must keep journaling post-swap"
    # the new record scores the NEW bundle's probabilities
    assert forecasts[-1]["horizons"]["24"]["up"] == pytest.approx(
        75.0, abs=2.0)
    assert runner._forecaster._last_error is None


def _walk_frames(h1_bars=2112, seed=7):
    """Coherent stack frames from one momentum walk (the F2
    trainer's test idiom): H4/D1 resampled from the H1 walk.
    2112 H1 bars = 88 days -> ~600 usable MTF rows, clearing
    the F2 floors (500 train rows; 150 labeled for h=360)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1.5, h1_bars)
    for i in range(2, h1_bars):
        steps[i] += 0.25 * steps[i - 1]
    close = 3900.0 + np.cumsum(steps)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 1.2, h1_bars)) + 0.5
    idx = pd.date_range("2026-08-01", periods=h1_bars,
                        freq="h", tz="UTC")
    h1 = pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + spread,
         "low": np.minimum(open_, close) - spread, "close": close},
        index=idx)
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last"}
    return {"H1": h1, "H4": h1.resample("4h").agg(agg).dropna(),
            "D1": h1.resample("D").agg(agg).dropna()}


def test_f2_retraining_event_keeps_the_cycle_running(tmp_path):
    """The audit's second face, pinned: on 2026-11-04T02:00:16Z a
    forecast_trained_training event completed (F2 artifact,
    trained through Nov 4 00:00, 733 rows) and no forecast
    records appear after it.  Records were in fact already dead
    from the 01:43 scalar failure — but the trained path had
    never been exercised live either, so this proves a landed
    F2 artifact neither parks nor corrupts the cycle: the next
    bar journals, its outlook carries the trained block, and
    the ensemble ranks still lead the record."""
    import joblib

    from goldenswing.forecast.trained import train_trained

    frames = _walk_frames()
    bundle_path = tmp_path / "forecast_models.joblib"
    joblib.dump(_full_bundle(frames), bundle_path)
    runner = _runner(tmp_path)
    runner._forecaster = MT5Forecaster(bundle_path=bundle_path)
    runner._stack = SimpleNamespace(frames=frames)
    runner._maybe_forecast_alert()
    forecasts = [r for r in runner.journal.records
                 if r.get("type") == "forecast"]
    assert forecasts, "pre-training cycle must journal"
    assert all("trained" not in (r.get("outlook") or {})
               for r in forecasts)

    # the retraining event: the F2 artifact lands beside the bundle
    trained_path = tmp_path / "forecast_trained_v1.joblib"
    summary = train_trained(frames, trained_path)
    assert summary["train_rows"] >= 500

    runner._stack = SimpleNamespace(frames=_walk_frames(2113))
    runner._maybe_forecast_alert()
    assert not any("forecast alert failed" in n
                   for n in runner.journal.notes)
    assert not any("outlook unavailable" in n
                   for n in runner.journal.notes)
    forecasts = [r for r in runner.journal.records
                 if r.get("type") == "forecast"]
    assert len(forecasts) > 1, "cycle must keep journaling post-train"
    latest = forecasts[-1]
    assert latest["cadence"] == "H1"
    assert "24" in latest["horizons"]  # ensemble ranks still lead
    outlook = latest.get("outlook") or {}
    assert "trained" in outlook, "trained block must engage post-train"
    assert outlook["trained"]["trained_through"] == \
        summary["trained_through"]
    assert runner._forecaster._last_error is None


def test_runner_real_cycle_journals_both_cadences(tmp_path):
    """The work-item proof: with the conversion fixed, a forecast
    cycle produces its journal record AND its alert evaluation —
    D1 cadence and the intraday H1 cadence — under the strict
    scalar regime, with no failure notes."""
    runner = _runner(tmp_path)
    frames = _frames()
    bundle = _bundle(frames)
    fc = MT5Forecaster(bundle_path=tmp_path / "nope.joblib")
    fc._load_bundle = lambda: (bundle, "fake bundle")
    runner._forecaster = fc
    runner._stack = SimpleNamespace(frames=frames)
    runner._maybe_forecast_alert()
    assert not any("forecast alert failed" in n
                   for n in runner.journal.notes)
    forecasts = [r for r in runner.journal.records
                 if r.get("type") == "forecast"]
    assert {r["cadence"] for r in forecasts} == {"D1", "H1"}
    for r in forecasts:
        assert r["horizons"]["24"]["up"] == pytest.approx(66.0, abs=2.0)
        assert r["alert_fired"] is False
    assert runner._forecast_consecutive_failures == 0
    # Cadence independence: a fresh H1 bar on the SAME D1 day
    # journals a new H1 record — no new D1 record, no silence.
    # (Before the _forecast_alert_d1 split, the D1 block's
    # day-unchanged return skipped the H1 block entirely.)
    runner._stack = SimpleNamespace(frames=_frames(h1_bars=1681))
    runner._maybe_forecast_alert()
    assert not any("forecast alert failed" in n
                   for n in runner.journal.notes)
    forecasts = [r for r in runner.journal.records
                 if r.get("type") == "forecast"]
    assert [r["cadence"] for r in forecasts].count("H1") == 2
    assert [r["cadence"] for r in forecasts].count("D1") == 1
