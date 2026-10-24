"""MT5 startup progress via Telegram: the owner watches TG, not console.

Regression: 2026-10-17 — startup went silent for minutes after the
history-fetch warnings (zone seed + 17-playbook replay over H1 history
takes minutes on EC2) with no Telegram ping until ALL of it finished,
so the owner could not tell a slow-but-healthy start from a hang.

Contract: ping FIRST ("starting…"), then a TG progress line per stage
(history fetched / zones built / playbooks warmed), then the final
"engine started" ping.  A genuine startup failure sends a TG
"FAILED to start: <reason>" and still raises.  Zone building has NO
timeout — a slow build is reported, never cut short.
"""
from types import SimpleNamespace

import pytest

from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
# Roster truth for the warmup-count assertions below: the H1 warmup
# warms exactly the H1-clocked registry (qm_m5/qm_d1 run on their
# native clocks, not here).  The count was hardcoded as 17 when the
# test was written (2026-10-17); Phase 2's clock split left 16, and
# the hardcoded count went stale.  Deriving it from the registry
# asserts the real invariant — warmup covers the whole H1 roster —
# and cannot drift again.
from goldenswing.execution.playbook_registry import ROSTER_SPECS
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from mt5_fakes import FakeMT5


class RecordingSender:
    def __init__(self, fail=False):
        self.texts: list[str] = []
        self.fail = fail

    def send_text(self, text, **kw):
        if self.fail:
            raise RuntimeError("telegram down")
        self.texts.append(text)
        return SimpleNamespace(delivered=True, detail="ok")


class FakeExecutor:
    def __init__(self, connect_ok=True):
        self._connect_ok = connect_ok

    def connect(self, **kw):
        if self._connect_ok:
            return True, "fake connected"
        return False, "boom"

    def positions(self, symbol=None):
        return []

    def pending_orders(self, symbol=None):
        return []


def _runner(tmp_path, *, sender=None, executor=None, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        **cfg_kw)
    senders = [sender] if sender is not None else []
    script = [(2000.0, 2000.30, 0), (2000.10, 2000.40, 1)]
    feed = MT5Feed(MT5FeedConfig(tick_interval=0.001),
                   mt5_module=FakeMT5(tick_script=script))
    return MT5Runner(cfg, feed=feed,
                      executor=executor or FakeExecutor(),
                      senders=senders)


def _idx(texts, needle):
    for i, t in enumerate(texts):
        if needle in t:
            return i
    return -1


def test_progress_order_starting_history_zones_playbooks_started(tmp_path):
    sender = RecordingSender()
    runner = _runner(tmp_path, sender=sender)
    runner.run(max_ticks=1)
    texts = sender.texts
    i_start = _idx(texts, "MT5 engine starting")
    i_hist = _idx(texts, "MT5 engine: history fetched")
    i_zones = _idx(texts, "MT5 engine: zones built")
    i_pb = _idx(texts, "MT5 engine: playbooks warmed")
    i_started = _idx(texts, "MT5 engine started")
    assert 0 <= i_start < i_hist < i_zones < i_pb < i_started


def test_progress_content(tmp_path):
    sender = RecordingSender()
    runner = _runner(tmp_path, sender=sender)
    runner.run(max_ticks=1)
    texts = sender.texts
    starting = texts[_idx(texts, "MT5 engine starting")]
    assert "symbol: XAUUSD" in starting
    hist = texts[_idx(texts, "MT5 engine: history fetched")]
    for tf in ("D1:", "H4:", "H1:", "M15:", "M5:", "W1:", "MN1:"):
        assert tf in hist
    zones = texts[_idx(texts, "MT5 engine: zones built")]
    assert "zones)" in zones
    pb = texts[_idx(texts, "MT5 engine: playbooks warmed")]
    assert f"({len(ROSTER_SPECS)} playbooks)" in pb
    # Final "started" ping still exactly once.
    assert sum(1 for t in texts if t.startswith("MT5 engine started")) == 1


def test_startup_failure_sends_tg_error_and_still_raises(tmp_path):
    sender = RecordingSender()
    runner = _runner(tmp_path, sender=sender,
                     executor=FakeExecutor(connect_ok=False))
    with pytest.raises(RuntimeError, match="executor connect failed"):
        runner.run(max_ticks=1)
    failed = [t for t in sender.texts if "FAILED to start" in t]
    assert len(failed) == 1
    assert "executor connect failed" in failed[0]
    # The starting ping still went out first.
    assert _idx(sender.texts, "MT5 engine starting") < \
        _idx(sender.texts, "FAILED to start")


def test_no_senders_no_crash_console_progress(tmp_path, capsys):
    runner = _runner(tmp_path)
    stats = runner.run(max_ticks=1)
    assert stats["ticks"] == 1
    out = capsys.readouterr().out
    assert "MT5 startup: MT5 engine starting" in out
    assert "MT5 startup: building zones" in out
    assert f"MT5 startup: warming {len(ROSTER_SPECS)} playbooks" in out


def test_replay_logs_each_playbook_on_console(tmp_path, capsys):
    sender = RecordingSender()
    runner = _runner(tmp_path, sender=sender)
    runner.run(max_ticks=1)
    out = capsys.readouterr().out
    assert "MT5 startup: warmed sweep_reversal in " in out
    assert "MT5 startup: warmed narrative in " in out
    # Per-playbook lines stay on console only — never 17 TG messages.
    assert sum(1 for t in sender.texts if "warmed sweep_reversal" in t) == 0
