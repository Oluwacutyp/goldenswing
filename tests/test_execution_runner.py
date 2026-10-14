"""Phase 8 — the dry runner loop end to end, and the forecaster
scorer against injected fake models (no sklearn on this host)."""
import numpy as np
import pandas as pd
import pytest

from goldenswing.alerts import EmailSender, TelegramSender
from goldenswing.execution.runner import (ForecasterScorer, LiveRunner,
                                          RunnerConfig)
from goldenswing.forecast.features import build_feature_frame
from goldenswing.golden.detector import DetectorConfig, GoldenSignal
from goldenswing.reader.reader import MarketReader


class _StubDetector:
    """Emits one fabricated golden signal at a fixed bar."""

    def __init__(self, signal):
        self.cfg = DetectorConfig()
        self._signal = signal

    def scan(self, states):
        return [self._signal]

    def evaluate(self, state):
        return None


def _df(n=200):
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 2000.0 + 0.5 * np.arange(n)
    return pd.DataFrame(
        {"open": close - 0.1, "high": close + 1.0, "low": close - 1.0,
         "close": close, "volume": 1000.0}, index=idx)


def test_dry_runner_loop_alert_execute_journal(tmp_path):
    df = _df()
    sig = GoldenSignal(
        index=100, timestamp=df.index[100], direction=1, score=80.0,
        kind="continuation", factors=["trend aligned"],
        entry_zone=(2049.0, 2051.0), entry_ref=2050.0,
        invalidation=2040.0, targets=[2075.0], atr=10.0)
    cfg = RunnerConfig(
        journal_path=tmp_path / "journal.jsonl",
        risk_state_path=tmp_path / "risk.json",
        kill_switch_path=tmp_path / "KILL",
        forecast_bundle_path=None, auto_execute=True,
        use_confluence=False)  # stub detector tests the legacy path
    senders = [TelegramSender(outbox_dir=tmp_path / "outbox"),
               EmailSender(outbox_dir=tmp_path / "outbox")]
    runner = LiveRunner(cfg, detector=_StubDetector(sig), senders=senders)
    summary = runner.run_bars(df)
    assert summary["bars"] == 200
    assert summary["alerts"] == 1
    assert summary["forecaster"].startswith("no forecaster")
    # the tap-path really ran: an open record exists in the journal,
    # and the paper executor filled it on the next bar
    records = runner.journal.records()
    opens = [r for r in records if r["type"] == "open"]
    assert len(opens) == 1 and opens[0]["source"] == "execute"
    assert opens[0]["playbook"] == "trend_continuation"
    files = {p.name for p in (tmp_path / "outbox").iterdir()}
    assert any(name.endswith(".telegram.png") or "telegram" in name
               for name in files) or files     # outbox received the alert


class _FakeModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.tile([1.0 - self.p, self.p], (len(X), 1))


def test_forecaster_scorer_with_fake_models():
    df = _df(60)
    states = MarketReader().read(df)
    cols = list(build_feature_frame(states).columns)
    bundle = {"models": {"up": {"m": _FakeModel(0.7)},
                         "dn": {"m": _FakeModel(0.2)}},
              "features": cols, "base_rates": {"up": 0.3, "dn": 0.3}}
    scorer = ForecasterScorer(bundle, states)
    sig = GoldenSignal(index=10, timestamp=df.index[10], direction=1,
                       score=70.0, kind="continuation")
    note = scorer.note_for(sig)
    assert note is not None
    assert note.prob_direction == pytest.approx(0.7)
    assert note.rank_pct == pytest.approx(100.0)   # all probs equal
    assert "big-up" in note.text


def test_forecaster_try_load_missing_bundle_is_honest(tmp_path):
    scorer, why = ForecasterScorer.try_load(tmp_path / "nope.joblib", [])
    assert scorer is None and "no forecaster bundle" in why
