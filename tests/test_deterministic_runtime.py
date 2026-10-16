"""Deterministic runtime: REPLAY -> RECONCILING -> LIVE + pipeline + runner bridge."""
import json
import os
import tempfile

from goldenswing.engine.pipeline import aggregate_bars, merge_events, transition
from goldenswing.engine.runtime import DeterministicRuntime


def _runtime():
    tmp = tempfile.mkdtemp()
    return DeterministicRuntime(tmp, secret=b"test-secret-32-bytes-long-xxxx")


def test_runtime_lifecycle_replay_reconciling_live():
    rt = _runtime()
    assert rt.phase == "REPLAY"
    out = rt.begin_live()
    assert out["phase"] == "LIVE"
    assert out["reconciliation_complete"] is True
    assert rt.status()["phase"] == "LIVE"


def test_runtime_submit_intent_idempotent_and_conflict():
    rt = _runtime()
    rt.begin_live()
    first = rt.submit_intent("k1", {"side": "buy", "qty": 1})
    assert first["action"] == "new"
    again = rt.submit_intent("k1", {"side": "buy", "qty": 1})
    assert again["action"] == "replay"
    assert again["committed_seq"] == first["seq"]
    conflict = rt.submit_intent("k1", {"side": "sell", "qty": 1})
    assert conflict["action"] == "conflict"


def test_runtime_journal_event_and_snapshot_and_reopen():
    rt = _runtime()
    rt.begin_live()
    meta = rt.journal_event("trade_opened", {"ticket": 7})
    assert meta is not None and meta["seq"] >= 1
    snap = rt.snapshot()
    assert snap["watermark"] == rt.journal.head
    # Reopen: journal reloads, chain verifies, head preserved.
    rt2 = DeterministicRuntime(rt.data_dir, secret=b"test-secret-32-bytes-long-xxxx")
    assert rt2.journal.head == rt.journal.head
    assert rt2.journal.verify() is True


def test_runtime_reconcile_unknown_blocks_live():
    rt = _runtime()
    rt.begin_live()
    rt.submit_intent("k-un", {"side": "buy"})
    rt.mark_unconfirmed("k-un")
    out = rt.reconcile(lambda key: ("unknown", {"error": "timeout"}))
    assert out["phase"] == "RECONCILING"
    assert out["reconciliation_complete"] is False


def test_pipeline_merge_dedup_and_watermark():
    events = [
        {"timestamp": 10.0, "source_id": "a", "source_seq": 1, "kind": "tick"},
        {"timestamp": 10.0, "source_id": "a", "source_seq": 1, "kind": "tick"},
        {"timestamp": 11.0, "source_id": "b", "source_seq": 1, "kind": "tick"},
    ]
    merged = merge_events(events, skew_s=0.0)
    assert len(merged["ordered"]) == 2
    assert merged["ordered"][0].source_id == "a"


def test_pipeline_bars_and_transition_pure():
    ticks = [{"timestamp": 0.0, "price": 100.0},
             {"timestamp": 10.0, "price": 101.0},
             {"timestamp": 3700.0, "price": 102.0}]
    bars = aggregate_bars(ticks, timeframe_s=3600.0)
    assert len(bars) == 2 and bars[0]["close"] == 101.0
    state, effects = transition({}, bars[0])
    assert state["bar_count"] == 1 and effects[0]["type"] == "bar_closed"


def test_runner_bridge_dual_write(tmp_path, monkeypatch):
    from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        deterministic_dir=tmp_path / "deterministic",
        tick_log=False)
    runner = MT5Runner(cfg)
    assert runner.deterministic is not None
    meta = runner._det_journal("runner_started", {"symbol": "XAUUSD"})
    assert meta is not None
    assert runner.deterministic.journal.head >= 1
    assert runner.deterministic_status()["enabled"] is True
    # Idempotent dual-write: same key replays, does not allocate seq.
    first = runner._det_journal("trade_closed", {"ticket": 1},
                                idempotency_key="close-a-1")
    second = runner._det_journal("trade_closed", {"ticket": 1},
                                 idempotency_key="close-a-1")
    assert second is not None and second["seq"] == first["seq"]
    assert runner.deterministic.journal.head == first["seq"]
