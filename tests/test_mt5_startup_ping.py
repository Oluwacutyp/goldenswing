"""MT5 startup ping: one Telegram 'engine started' message per start.

The engine used to start silently — no console output, no Telegram
message — so the owner could not tell it was alive.  The ping fires
after feed connect + runner start, independent of command polling
(--no-poll-commands still pings), and never crashes the start.
"""
import json
from types import SimpleNamespace

import pytest

from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from mt5_fakes import FakeMT5


class RecordingSender:
    """Fake TelegramSender: records send_text calls, reports delivered."""

    def __init__(self, fail=False):
        self.texts: list[str] = []
        self.fail = fail

    def send_text(self, text, **kw):
        if self.fail:
            raise RuntimeError("telegram down")
        self.texts.append(text)
        return SimpleNamespace(delivered=True, detail="ok")


class FakeExecutor:
    def connect(self, **kw):
        return True, "fake connected"

    def positions(self, symbol=None):
        return []

    def pending_orders(self, symbol=None):
        return []


def _runner(tmp_path, *, sender=None, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        **cfg_kw)
    senders = [sender] if sender is not None else []
    script = [(2000.0, 2000.30, 0), (2000.10, 2000.40, 1)]
    feed = MT5Feed(MT5FeedConfig(tick_interval=0.001),
                   mt5_module=FakeMT5(tick_script=script))
    return MT5Runner(cfg, feed=feed, executor=FakeExecutor(),
                      senders=senders)


def _journal_texts(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line).get("text", "")
            for line in path.read_text().splitlines() if line.strip()]


def _pings(sender):
    return [t for t in sender.texts if t.startswith("MT5 engine started")]


def test_startup_ping_sent_on_run_with_content(tmp_path):
    sender = RecordingSender()
    runner = _runner(tmp_path, sender=sender)
    runner.run(max_ticks=1)
    pings = _pings(sender)
    assert len(pings) == 1
    ping = pings[0]
    assert "symbol: XAUUSD" in ping
    assert "trade_mode: 0 (demo)" in ping
    assert "equity: 10000.0" in ping
    assert "playbooks: " in ping
    assert "sweep_reversal" in ping
    assert "judas_swing" in ping
    assert "state: RUNNING" in ping
    assert "command polling: ON" in ping


def test_startup_ping_fires_with_polling_off(tmp_path):
    sender = RecordingSender()
    runner = _runner(tmp_path, sender=sender, poll_commands=False)
    runner.run(max_ticks=1)
    pings = _pings(sender)
    assert len(pings) == 1
    assert "command polling: OFF" in pings[0]


def test_startup_ping_shows_paused_state(tmp_path):
    sender = RecordingSender()
    runner = _runner(tmp_path, sender=sender, paused=True)
    runner.run(max_ticks=1)
    assert "state: PAUSED" in _pings(sender)[0]


def test_startup_ping_respects_disabled_playbooks(tmp_path):
    sender = RecordingSender()
    runner = _runner(tmp_path, sender=sender,
                     disabled_playbooks=("inversion", "judas_swing"))
    runner.run(max_ticks=1)
    ping = _pings(sender)[0]
    assert "inversion" not in ping
    assert "judas_swing" not in ping
    assert "sweep_reversal" in ping


def test_startup_ping_send_failure_does_not_crash_start(tmp_path):
    sender = RecordingSender(fail=True)
    runner = _runner(tmp_path, sender=sender)
    stats = runner.run(max_ticks=1)
    assert stats["ticks"] == 1
    notes = _journal_texts(tmp_path)
    assert any("alert send failed" in n for n in notes)


def test_startup_ping_skipped_without_senders(tmp_path):
    runner = _runner(tmp_path)
    stats = runner.run(max_ticks=1)
    assert stats["ticks"] == 1
    notes = _journal_texts(tmp_path)
    assert any("startup ping skipped" in n for n in notes)


def test_startup_ping_exactly_once_per_start(tmp_path):
    sender = RecordingSender()
    runner = _runner(tmp_path, sender=sender)
    runner.run(max_ticks=1)
    # A second start attempt on the same runner must not re-ping.
    runner._send_startup_ping({"trade_mode": 0, "equity": 10000.0})
    assert len(_pings(sender)) == 1
