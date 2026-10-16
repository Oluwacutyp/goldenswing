"""Tests for /backtest /backtest_send /journal /snapshot /reconcile /health.

The backtest commands send a FILE + concise summary caption (the file
is the contract the owner drops back to Muse), never silent: no
journal means an explicit text reply.
"""
from pathlib import Path

from goldenswing.alerts.commands import (
    COMMANDS, CommandContext, CommandRouter)
from goldenswing.execution.mt5_commands import MT5_COMMANDS


class _FakeJournal:
    def __init__(self, records):
        self._records = records
        self.path = Path("/tmp/fake-journal.jsonl")

    def records(self):
        return list(self._records)


def _open_close_records():
    return [
        {"type": "open", "alert_id": "a1", "playbook": "fvg_limit",
         "direction": 1, "score": 70, "entry": 4000.0, "stop": 3990.0,
         "volume": 0.01, "risk_money": 10.0, "ticket": 1, "ts": "2026-01-01T00:00:00Z"},
        {"type": "close", "alert_id": "a1", "ticket": 1, "exit": 4020.0,
         "reason": "tp", "pnl_money": 20.0, "r_net": 2.0,
         "hold_bars": 3, "ts": "2026-01-02T00:00:00Z"},
        {"type": "open", "alert_id": "a2", "playbook": "fvg_limit",
         "direction": -1, "score": 65, "entry": 4020.0, "stop": 4030.0,
         "volume": 0.01, "risk_money": 10.0, "ticket": 2, "ts": "2026-01-03T00:00:00Z"},
        {"type": "close", "alert_id": "a2", "ticket": 2, "exit": 4030.0,
         "reason": "sl", "pnl_money": -10.0, "r_net": -1.0,
         "hold_bars": 2, "ts": "2026-01-04T00:00:00Z"},
    ]


def test_backtest_command_sends_file_plus_summary(tmp_path):
    ctx = CommandContext(journal=_FakeJournal(_open_close_records()),
                         journal_path=tmp_path / "journal.jsonl",
                         reports_dir=tmp_path / "reports",
                         feed_status={"ok": True})
    _, handler = COMMANDS["/backtest"]
    result = handler(ctx, [])
    assert result.document is not None
    assert Path(result.document).exists()
    text = Path(result.document).read_text()
    assert "trades: 2" in text
    assert "total R: +1.00R" in text
    assert "trades: 2" in (result.caption or "")


def test_backtest_send_caption_tells_owner_to_drop_back(tmp_path):
    ctx = CommandContext(journal=_FakeJournal(_open_close_records()),
                         journal_path=tmp_path / "journal.jsonl",
                         reports_dir=tmp_path / "reports",
                         feed_status={"ok": True})
    _, handler = COMMANDS["/backtest_send"]
    result = handler(ctx, [])
    assert result.document is not None
    assert "Drop this file back to Muse" in (result.caption or "")


def test_backtest_no_journal_is_explicit_not_silent():
    ctx = CommandContext(journal=None)
    _, handler = COMMANDS["/backtest"]
    result = handler(ctx, [])
    assert result.document is None
    assert "no journal" in (result.text or "")


def test_backtest_d1_stale_warns(tmp_path):
    ctx = CommandContext(journal=_FakeJournal(_open_close_records()),
                         journal_path=tmp_path / "journal.jsonl",
                         reports_dir=tmp_path / "reports",
                         feed_status={"ok": False,
                                      "reason": "D1 stale"})
    _, handler = COMMANDS["/backtest"]
    result = handler(ctx, [])
    text = Path(result.document).read_text()
    assert "D1 coverage insufficient" in text


def test_journal_snapshot_reconcile_health_commands():
    ctx = CommandContext(journal=_FakeJournal(_open_close_records()),
                         journal_path=Path("/tmp/j.jsonl"),
                         feed_status={"ok": True})
    for name in ("/journal", "/snapshot", "/reconcile", "/health"):
        _, handler = COMMANDS[name]
        result = handler(ctx, [])
        assert result.text  # never silent, never empty


def test_mt5_registry_has_backtest_family():
    for name in ("/backtest", "/backtest_send", "/journal",
                 "/snapshot", "/reconcile", "/health"):
        assert name in MT5_COMMANDS, f"{name} missing from MT5 registry"


def test_router_dispatches_backtest_owner_only(tmp_path):
    router = CommandRouter(owner_chat_id="123", registry=COMMANDS)
    ctx = CommandContext(journal=_FakeJournal(_open_close_records()),
                         journal_path=tmp_path / "journal.jsonl",
                         reports_dir=tmp_path / "reports",
                         feed_status={"ok": True})
    update = {"message": {"text": "/backtest",
                          "chat": {"id": 123}}}
    result, _ = router.dispatch(update, ctx)
    assert result is not None and result.document is not None
    stranger = {"message": {"text": "/backtest",
                            "chat": {"id": 999}}}
    result2, _ = router.dispatch(stranger, ctx)
    assert result2 is None  # silent for non-owner
