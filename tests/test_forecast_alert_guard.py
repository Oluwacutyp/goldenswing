"""Forecast alert guard regression (Nov 3 missed-short audit, E).

From 2026-10-26 every forecast alert died on the same line —
``if not frames.get("H1")`` — because a DataFrame's truth value is
ambiguous in pandas.  The guard raised before scoring ever ran;
the journal filled with "forecast alert failed (loop continues)"
while /status kept reporting the ensemble as wired.  This pins
the guard: with real DataFrames in the stack, the D1-close path
must reach the forecaster.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from mt5_fakes import FakeMT5


def _frame(n, freq, base=3900.0):
    idx = pd.date_range("2026-10-14", periods=n, freq=freq, tz="UTC")
    close = [base + (i % 9) * 0.5 for i in range(n)]
    return pd.DataFrame(
        {"open": close, "high": [c + 0.4 for c in close],
         "low": [c - 0.4 for c in close], "close": close,
         "volume": [100] * n, "real_volume": [0] * n,
         "spread": [30] * n},
        index=idx)


class _Forecaster:
    def __init__(self):
        self._train_result = None
        self.alert_calls = 0
        self.train_calls = 0

    def maybe_train(self, frames):
        self.train_calls += 1
        return None, "bundle current"

    def maybe_alert(self, frames):
        self.alert_calls += 1
        return None, "no new threshold crossing"

    def score_and_record(self, frames):
        self.alert_calls += 1
        return None, "no new threshold crossing", {
            "bar_time": "2026-11-03 00:00:00+00:00", "close": 3906.6,
            "horizons": {"24": {"up": 0.4, "dn": 0.6}},
            "model_version": "test-v1", "feature_set": "mtf",
            "trained_through": "2026-11-01", "alert_fired": False,
            "note": "no new threshold crossing"}


class _Journal:
    def __init__(self):
        self.notes = []
        self.records = []

    def log_note(self, text):
        self.notes.append(str(text))

    def append(self, record):
        self.records.append(record)


def _runner(tmp_path):
    feed = MT5Feed(FakeMT5(), MT5FeedConfig(symbol="XAUUSDm"))
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False, max_daily_trades=0,
        max_hourly_trades=0)
    runner = MT5Runner(cfg, feed=feed)
    runner._forecaster = _Forecaster()
    runner.journal = _Journal()
    runner._stack = SimpleNamespace(frames={
        "D1": _frame(30, "D"), "H1": _frame(120, "h")})
    return runner


def test_forecast_alert_reaches_the_forecaster(tmp_path):
    runner = _runner(tmp_path)
    runner._maybe_forecast_alert()
    assert runner._forecaster.alert_calls >= 1
    assert not any("forecast alert failed" in n
                   for n in runner.journal.notes)


def test_forecast_records_are_journaled_both_cadences(tmp_path):
    """E1 runtime proof: a scoring lands in the journal as a
    structured forecast record — D1 cadence and the intraday H1
    cadence — carrying horizons and model provenance."""
    runner = _runner(tmp_path)
    runner._maybe_forecast_alert()
    forecasts = [r for r in runner.journal.records
                 if r.get("type") == "forecast"]
    assert {r["cadence"] for r in forecasts} == {"D1", "H1"}
    for r in forecasts:
        assert r["model_version"] == "test-v1"
        assert r["horizons"]["24"]["dn"] == 0.6
        assert r["alert_fired"] is False
    # same bars again: cursors dedupe, no duplicate records
    runner._maybe_forecast_alert()
    assert len([r for r in runner.journal.records
                if r.get("type") == "forecast"]) == 2


def test_forecast_alert_short_h1_stays_quiet(tmp_path):
    runner = _runner(tmp_path)
    runner._stack = SimpleNamespace(frames={
        "D1": _frame(30, "D"), "H1": _frame(10, "h")})
    runner._maybe_forecast_alert()
    assert runner._forecaster.alert_calls == 0
    assert not any("forecast alert failed" in n
                   for n in runner.journal.notes)
