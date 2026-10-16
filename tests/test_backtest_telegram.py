"""Tests for /backtest /backtest_send /backtest_export /journal /snapshot
/reconcile /health.

/backtest RUNS a real whole-engine backtest async: the handler returns
a "running…" ack immediately and the report is delivered through
ctx.deliver when the run finishes.  It must NEVER return the old
"no closed trades — run the paper loop" exporter text, even on an
empty journal.  The exporter lives at /backtest_export now.
"""
import time
from pathlib import Path

from goldenswing.alerts.commands import (
    COMMANDS, CommandContext, CommandRouter,
    parse_backtest_args, reset_backtest_jobs_for_tests,
    set_backtest_runner_for_tests)
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


def _fake_report():
    return {
        "window": {"start": "2026-09-12", "end_open": "2026-10-12",
                   "ticks": 100000, "days_requested": 30,
                   "only_playbook": None},
        "trades": {"closed": 4, "open_at_end": 0,
                   "open_at_end_unrealized_r": 0.0, "win_rate": 0.5,
                   "total_r": 1.5, "total_dollar": 150.0, "avg_r": 0.375,
                   "profit_factor": 2.0, "max_drawdown_r": 1.0},
        "per_playbook": {"fvg_limit": {"trades": 4, "total_r": 1.5,
                                       "total_dollar": 150.0}},
    }


def _ctx_with_deliver(tmp_path, journal):
    delivered = []

    def deliver(result):
        delivered.append(result)

    ctx = CommandContext(journal=journal,
                         journal_path=tmp_path / "journal.jsonl",
                         reports_dir=tmp_path / "reports",
                         feed_status={"ok": True},
                         deliver=deliver)
    return ctx, delivered


def _wait_delivered(delivered, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not delivered:
        time.sleep(0.05)
    assert delivered, "backtest report was never delivered"


def _install_fake_runner():
    set_backtest_runner_for_tests(lambda job: _fake_report())


def _teardown():
    set_backtest_runner_for_tests(None)
    reset_backtest_jobs_for_tests()


def test_backtest_on_empty_journal_starts_a_run_not_exporter(tmp_path):
    # The regression: /backtest used to answer "no closed trades in
    # the journal yet — nothing to export. Run the paper loop first."
    _install_fake_runner()
    try:
        ctx, delivered = _ctx_with_deliver(tmp_path, None)
        _, handler = COMMANDS["/backtest"]
        result = handler(ctx, [])
        assert "running" in (result.text or "").lower()
        assert "no closed trades" not in (result.text or "")
        assert "paper loop" not in (result.text or "")
        _wait_delivered(delivered)
        assert delivered[0].document is not None
        assert Path(delivered[0].document).exists()
        assert "Backtest" in (delivered[0].caption or "")
    finally:
        _teardown()


def test_backtest_delivers_septorch_report(tmp_path):
    _install_fake_runner()
    try:
        ctx, delivered = _ctx_with_deliver(
            tmp_path, _FakeJournal(_open_close_records()))
        _, handler = COMMANDS["/backtest"]
        handler(ctx, ["30"])
        _wait_delivered(delivered)
        result = delivered[0]
        caption = result.caption or ""
        assert "📊" in caption
        assert "win rate" in caption
        assert "Limits" in caption  # honest limits block
        text = Path(result.document).read_text()
        assert "fvg_limit" in text
        assert "total" in text
    finally:
        _teardown()


def test_backtest_send_caption_tells_owner_to_drop_back(tmp_path):
    _install_fake_runner()
    try:
        ctx, delivered = _ctx_with_deliver(
            tmp_path, _FakeJournal(_open_close_records()))
        _, handler = COMMANDS["/backtest_send"]
        handler(ctx, [])
        _wait_delivered(delivered)
        assert "Drop this file back to Muse" in (delivered[0].caption or "")
    finally:
        _teardown()


def test_backtest_second_run_while_running_is_refused(tmp_path):
    def slow_runner(job):
        time.sleep(2)
        return _fake_report()

    set_backtest_runner_for_tests(slow_runner)
    try:
        ctx, _ = _ctx_with_deliver(tmp_path, None)
        _, handler = COMMANDS["/backtest"]
        first = handler(ctx, [])
        assert "running" in (first.text or "").lower()
        second = handler(ctx, [])
        assert "already running" in (second.text or "").lower()
    finally:
        _teardown()


def test_parse_backtest_args():
    assert parse_backtest_args([]) == (30, None, None)
    assert parse_backtest_args(["30"]) == (30, None, None)
    assert parse_backtest_args(["90", "sweep"]) == (90, None, None)
    assert parse_backtest_args(["sweep"]) == (90, None, None)
    assert parse_backtest_args(["200"]) == (90, None, None)  # capped
    assert parse_backtest_args(["fvg_limit"]) == (30, "fvg_limit", None)
    days, _pb, err = parse_backtest_args(["bogus"])
    assert err and "unknown" in err


def test_backtest_export_keeps_old_exporter(tmp_path):
    ctx = CommandContext(journal=_FakeJournal(_open_close_records()),
                         journal_path=tmp_path / "journal.jsonl",
                         reports_dir=tmp_path / "reports",
                         feed_status={"ok": True})
    _, handler = COMMANDS["/backtest_export"]
    result = handler(ctx, [])
    assert result.document is not None
    assert Path(result.document).exists()
    text = Path(result.document).read_text()
    assert "trades: 2" in text
    assert "total R: +1.00R" in text
    assert "trades: 2" in (result.caption or "")


def test_backtest_export_empty_journal_still_says_no_trades(tmp_path):
    ctx = CommandContext(journal=None,
                         journal_path=tmp_path / "journal.jsonl",
                         reports_dir=tmp_path / "reports",
                         feed_status={"ok": True})
    _, handler = COMMANDS["/backtest_export"]
    result = handler(ctx, [])
    assert result.document is None
    assert "no journal" in (result.text or "")


def test_backtest_export_d1_stale_warns(tmp_path):
    ctx = CommandContext(journal=_FakeJournal(_open_close_records()),
                         journal_path=tmp_path / "journal.jsonl",
                         reports_dir=tmp_path / "reports",
                         feed_status={"ok": False,
                                      "reason": "D1 stale"})
    _, handler = COMMANDS["/backtest_export"]
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
    for name in ("/backtest", "/backtest_send", "/backtest_export",
                 "/journal", "/snapshot", "/reconcile", "/health"):
        assert name in MT5_COMMANDS, f"{name} missing from MT5 registry"


def test_router_dispatches_backtest_owner_only(tmp_path):
    _install_fake_runner()
    try:
        router = CommandRouter(owner_chat_id="123", registry=COMMANDS)
        ctx, delivered = _ctx_with_deliver(
            tmp_path, _FakeJournal(_open_close_records()))
        update = {"message": {"text": "/backtest",
                              "chat": {"id": 123}}}
        result, _ = router.dispatch(update, ctx)
        assert result is not None and "running" in (result.text or "").lower()
        _wait_delivered(delivered)
        stranger = {"message": {"text": "/backtest",
                                "chat": {"id": 999}}}
        result2, _ = router.dispatch(stranger, ctx)
        assert result2 is None  # silent for non-owner
    finally:
        _teardown()
