"""Forecast lifecycle acceptance tests (owner directive 2026-10-25).

One module-scoped bundle: the acceptance suite trains ONCE and
shares it, because ensemble fitting is the slow step. Each test
then checks one acceptance property against that shared artifact
or a fresh forecaster pointed at it:

1. clean-install bootstrap without manual file creation
2. artifact persists across restart + valid forecast
3. /forecast_train retrains + truthful success/failure
4. /status reflects actual state
5. insufficient history / failed training / corrupt artifacts /
   missing TF handled
6. failed retraining doesn't destroy the previous valid model
7. no duplicate alerts for unchanged notable state
8. forecast code cannot affect trading/risk decisions
9. fresh install with insufficient history never claims operational
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from goldenswing.forecast.mtf import (
    _validate_bundle,
    train_from_frames,
)
from goldenswing.forecast.mt5 import MT5Forecaster
from test_forecast_mt5 import _frames


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("forecast") / "forecast_models.joblib"
    train_from_frames(_frames(), path)
    return path


def test_clean_install_bootstraps_without_manual_files(tmp_path):
    """Acceptance 1: no bundle -> maybe_train starts a real train."""
    fc = MT5Forecaster(bundle_path=tmp_path / "forecast_models.joblib")
    assert fc.status_dict()["ready"] is False
    thread, note = fc.maybe_train(_frames())
    assert thread is not None, note
    thread.join(timeout=600)
    assert (tmp_path / "forecast_models.joblib").exists()
    assert fc.status_dict()["ready"] is True


def test_artifact_persists_across_restart(bundle_path):
    """Acceptance 2: a fresh forecaster scores the saved bundle."""
    fresh = MT5Forecaster(bundle_path=bundle_path)
    text, note = fresh.maybe_alert(_frames())
    assert fresh.status_dict()["ready"] is True
    assert fresh.status_dict()["model_loaded"] is True
    # Either a crossing alert or a quiet note — never a crash, and
    # the evaluation is recorded either way.
    assert fresh.status_dict()["last_evaluation"] is not None


def test_forecast_train_command_retrains_and_reports(tmp_path, bundle_path):
    """Acceptance 3: the operator lever delegates to the runner
    method and reports truthfully (here: bundle current)."""
    from goldenswing.execution.mt5_commands import _mt5_forecast_train
    from goldenswing.alerts.commands import CommandContext

    class Runner:
        def __init__(self, path):
            self._fc = MT5Forecaster(bundle_path=path)

        def forecast_train_now(self):
            thread, note = self._fc.maybe_train(_frames())
            return f"forecast_train: {note}"

    ctx = CommandContext(mt5_control=Runner(bundle_path))
    result = _mt5_forecast_train(ctx, [])
    assert "bundle current" in result.text


def test_status_reflects_actual_state(bundle_path):
    """Acceptance 4: /status values come from the live bundle."""
    fc = MT5Forecaster(bundle_path=bundle_path)
    fc.maybe_alert(_frames())
    st = fc.status_dict(h1_bars=2400)
    assert st["ready"] is True
    assert st["version"] == "forecast-bundle-v2"
    assert st["feature_set"] == "mtf-v1"
    assert st["symbol"] == "XAUUSD"
    assert st["trained_at"] and st["trained_through"]
    assert st["train_bars"] >= 500
    assert st["h1_bars"] == 2400
    assert st["alerts_enabled"] is True
    assert st["validation"] is not None


def test_insufficient_history_never_claims_operational(tmp_path):
    """Acceptance 5 + 9: short history -> no train, NOT READY."""
    fc = MT5Forecaster(bundle_path=tmp_path / "forecast_models.joblib")
    short = {"H1": _frames()["H1"].iloc[:100]}
    thread, note = fc.maybe_train(short)
    assert thread is None
    assert "too few H1 bars" in note
    st = fc.status_dict(h1_bars=100)
    assert st["ready"] is False
    assert st["alerts_enabled"] is False


def test_failed_retrain_keeps_previous_valid_model(tmp_path, bundle_path):
    """Acceptance 6: a failed retrain leaves the old bundle valid."""
    import shutil

    import goldenswing.forecast.mt5 as mt5_module

    target = tmp_path / "forecast_models.joblib"
    shutil.copy(bundle_path, target)
    before = joblib.load(target)
    assert _validate_bundle(before)[0]
    real_train = mt5_module.train_from_frames

    def boom(frames, out_path):
        raise RuntimeError("simulated training failure")

    # Patch the namespace the forecaster actually calls (mt5 imports
    # train_from_frames from mtf at module load).
    mt5_module.train_from_frames = boom
    try:
        fc = MT5Forecaster(bundle_path=target)
        # Force staleness: point the bundle's stamp far back.
        bundle = joblib.load(target)
        bundle["trained_through"] = "2020-01-01"
        joblib.dump(bundle, target)
        thread, note = fc.maybe_train(_frames())
        assert thread is not None, note
        thread.join(timeout=600)
    finally:
        mt5_module.train_from_frames = real_train
    ok, note = fc._train_result
    assert not ok and "simulated training failure" in note
    after = joblib.load(target)
    assert _validate_bundle(after)[0]
    assert after["trained_through"] == "2020-01-01"  # old bundle intact


def test_no_duplicate_alerts_for_unchanged_state(bundle_path):
    """Acceptance 7: a persistently notable rank alerts once."""
    fc = MT5Forecaster(bundle_path=bundle_path, threshold=0.0)
    frames = _frames()
    first, _ = fc.maybe_alert(frames)
    assert first is not None
    second, _ = fc.maybe_alert(frames)
    assert second is None


def test_forecast_cannot_touch_orders():
    """Acceptance 8: the forecaster object is never read by any
    order/sizing/risk function in the runner. Each _forecaster /
    maybe_alert / maybe_train reference is mapped to its enclosing
    function; only the forecast service functions may contain them.
    """
    import re
    from pathlib import Path

    # Warm start lives in run() (startup training) — a forecast
    # service call site, not an order path.
    allowed = {"__init__", "run", "forecast_line", "forecast_train_now",
               "_maybe_forecast_alert"}
    runner = Path("goldenswing/execution/mt5_runner.py").read_text()
    lines = runner.splitlines()
    for i, line in enumerate(lines):
        if "_forecaster" in line or "maybe_alert" in line \
                or "maybe_train" in line:
            enclosing = None
            for prev in reversed(lines[:i]):
                m = re.match(r"    def (\w+)\(", prev)
                if m:
                    enclosing = m.group(1)
                    break
            assert enclosing in allowed, (i + 1, enclosing, line.strip())


def test_missing_tf_degrades_but_trains(tmp_path):
    """Acceptance 5 (missing peers analogue): a missing TF frame is
    skipped, not fatal — the bundle trains on the remaining TFs."""
    frames = _frames()
    frames.pop("M5")
    path = tmp_path / "forecast_models.joblib"
    summary = train_from_frames(frames, path)
    assert summary["train_bars"] >= 500
    bundle = joblib.load(path)
    assert _validate_bundle(bundle)[0]
    assert not any(c.startswith("m5_") for c in bundle["features"])
    # Scoring with the full stack still works: extra M5 columns are
    # ignored, bundle columns select exactly.
    from goldenswing.forecast.mtf import score_last_row

    ranks, note = score_last_row(_frames(), bundle)
    assert ranks is not None, note


def test_corrupt_bundle_reports_not_ready_and_retrains(tmp_path):
    """Acceptance 5: corrupt artifact -> NOT READY + retrain starts."""
    target = tmp_path / "forecast_models.joblib"
    target.write_bytes(b"garbage")
    fc = MT5Forecaster(bundle_path=target)
    assert fc.status_dict()["ready"] is False
    thread, note = fc.maybe_train(_frames())
    assert thread is not None, note
    thread.join(timeout=600)
    assert fc.status_dict()["ready"] is True
