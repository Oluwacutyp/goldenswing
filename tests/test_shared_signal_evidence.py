from __future__ import annotations

import pandas as pd

from goldenswing.execution.signal_log import log_signal
from goldenswing.execution.mt5_runner import MT5Runner
from goldenswing.live.hierarchy import TimeframeStack


class _CaptureJournal:
    def __init__(self, provider=None):
        self.records = []
        if provider is not None:
            self.shared_evidence_provider = provider

    def append(self, record):
        self.records.append(record)
        return record


def test_signal_logger_attaches_shared_evidence_without_gating():
    called = []

    def provider(**kwargs):
        called.append(kwargs)
        return {"status": "measured", "score": 67.0,
                "score_is_calibrated_probability": False,
                "execution_gate": False}

    journal = _CaptureJournal(provider)
    out = log_signal(journal, signal_id="s1", playbook="qm_m5",
                     direction=1, score=54, decision="accepted", entry=2000.0)
    assert out["shared_evidence"]["score"] == 67.0
    assert out["decision"] == "accepted"  # observational only
    assert len(called) == 1


def test_signal_logger_does_not_run_evidence_for_management_records():
    called = []
    journal = _CaptureJournal(lambda **kwargs: called.append(kwargs) or {"status": "measured"})
    record = log_signal(journal, signal_id="s2", playbook="qm_m5",
                        direction=1, score=54, decision="management")
    assert "shared_evidence" not in record
    assert called == []


def test_shared_evidence_provider_error_does_not_break_signal_record():
    def broken(**kwargs):
        raise RuntimeError("measurement failed")

    journal = _CaptureJournal(broken)
    record = log_signal(journal, signal_id="s3", playbook="qm_m5",
                        direction=-1, score=44, decision="gated")
    assert record["decision"] == "gated"
    assert record["shared_evidence"]["status"] == "unavailable"
    assert record["shared_evidence"]["execution_gate"] is False


def _bars(tf: str, periods: int, freq: str, base: float) -> pd.DataFrame:
    idx = pd.date_range("2026-10-26T00:00:00Z", periods=periods, freq=freq)
    # Smooth repeating swings produce enough data to exercise the detectors.
    import numpy as np
    x = np.arange(periods, dtype=float)
    close = base + 0.3 * x + 2.0 * np.sin(x / 2.3)
    open_ = close - 0.15 * np.cos(x / 1.7)
    high = np.maximum(open_, close) + 0.35
    low = np.minimum(open_, close) - 0.35
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": 100 + (x % 7) * 15},
                        index=idx)


def test_runner_shared_evidence_is_auditable_and_never_an_execution_gate():
    runner = MT5Runner.__new__(MT5Runner)
    runner._stack = TimeframeStack.from_frames({
        "M5": _bars("M5", 120, "5min", 2000),
        "M15": _bars("M15", 80, "15min", 1998),
    })
    runner._hot_scalp_aux_frames = {"M1": _bars("M1", 300, "min", 2000)}
    runner._last_tick = {"time": "2026-10-26T05:01:00Z", "bid": 2015.0, "ask": 2015.15}
    result = runner._build_shared_signal_evidence(
        direction=1, playbook="qm_m5", entry=2015.1,
        bar_ts="2026-10-26T05:00:00Z", context={"trigger_tf": "M5"})
    assert result["status"] == "measured"
    assert result["score_is_calibrated_probability"] is False
    assert result["execution_gate"] is False
    assert result["evidence"]
    assert {"M5", "M15", "M1"}.issubset(result["data_timestamps"])
    assert all("source" in item and "timeframe" in item and
               "correlation_group" in item and "stance" in item
               for item in result["evidence"])
