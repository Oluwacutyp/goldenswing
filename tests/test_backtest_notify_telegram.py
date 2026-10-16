"""Tests for --notify-telegram in scripts/backtest_full_engine.py.

The standalone script runs in a second terminal (e.g. a 365-day run);
on completion it must send the Septorch-style summary + results.json to
Telegram.  Notify failure must never mark the backtest failed, missing
env must skip honestly, and the token must never appear in an error
string.  All HTTP is faked via an injected opener — no real network.
"""
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backtest_full_engine.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "backtest_full_engine_notify", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeResp:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeOpener:
    """Captures requests; optionally fails every call."""

    def __init__(self, fail=False):
        self.requests = []
        self.fail = fail

    def open(self, req, timeout=None):
        self.requests.append(req)
        if self.fail:
            raise urllib.error.URLError("boom")
        return _FakeResp({"ok": True, "result": {"message_id": 7}})


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


def test_format_notify_success_message():
    mod = _load_script()
    text = mod.format_notify_message(_fake_report(), "data/reports/x")
    assert "📊 Backtest done" in text
    assert "win rate" in text
    assert "fvg_limit" in text
    assert "+1.50R" in text
    assert "Limits" in text
    assert "data/reports/x" in text


def test_format_notify_failure_message():
    mod = _load_script()
    text = mod.format_notify_message(
        None, "data/reports/x", error=RuntimeError("kaput"),
        error_tail="line1\nline2")
    assert "❌ Backtest failed" in text
    assert "kaput" in text
    assert "line2" in text


def test_notify_missing_env_skips_without_http(monkeypatch):
    mod = _load_script()
    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("GS_TELEGRAM_CHAT_ID", raising=False)

    def _must_not_call(req, timeout=None):
        raise AssertionError("must not touch the network without env")

    status = mod.notify_telegram(_fake_report(), "data/reports/x",
                                 opener=_must_not_call)
    assert "skipped" in status
    assert "GS_TELEGRAM_BOT_TOKEN" in status


def test_notify_success_sends_text_then_document(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setenv("GS_TELEGRAM_BOT_TOKEN", "FAKE")
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", "123")
    out = tmp_path / "bt"
    out.mkdir()
    (out / "results.json").write_text('{"trades": {}}')
    opener = _FakeOpener()
    status = mod.notify_telegram(_fake_report(), out, opener=opener)
    assert "summary + results.json" in status
    assert len(opener.requests) == 2
    first = json.loads(opener.requests[0].data.decode())
    assert "📊 Backtest done" in first["text"]
    assert first["chat_id"] == "123"
    assert b'filename="results.json"' in opener.requests[1].data


def test_notify_failure_path_sends_failure_text_only(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setenv("GS_TELEGRAM_BOT_TOKEN", "FAKE")
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", "123")
    opener = _FakeOpener()
    status = mod.notify_telegram(None, tmp_path, error=RuntimeError("kaput"),
                                 error_tail="tail", opener=opener)
    assert "failure report" in status
    assert len(opener.requests) == 1  # no document on failure
    assert "❌ Backtest failed" in json.loads(
        opener.requests[0].data.decode())["text"]


def test_notify_network_failure_never_raises(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setenv("GS_TELEGRAM_BOT_TOKEN", "FAKE")
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", "123")
    status = mod.notify_telegram(_fake_report(), tmp_path,
                                 opener=_FakeOpener(fail=True))
    assert "failed" in status  # honest, not an exception


def test_send_document_too_large_refused(tmp_path):
    mod = _load_script()
    p = tmp_path / "results.json"
    p.write_bytes(b"x" * 100)
    ok, detail = mod.send_telegram_document(
        "FAKE", "123", p, "caption", opener=_FakeOpener(), limit_bytes=10)
    assert not ok
    assert "50MB" in detail


def test_notify_too_large_results_sends_summary_only(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setenv("GS_TELEGRAM_BOT_TOKEN", "FAKE")
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(mod, "_TELEGRAM_DOCUMENT_LIMIT_BYTES", 1)
    out = tmp_path / "bt"
    out.mkdir()
    (out / "results.json").write_text('{"trades": {}}')
    opener = _FakeOpener()
    status = mod.notify_telegram(_fake_report(), out, opener=opener)
    assert "too large" in status
    assert len(opener.requests) == 1  # summary only, no doomed upload
    assert "over Telegram's 50MB" in json.loads(
        opener.requests[0].data.decode())["text"]


def test_token_never_in_error_string():
    mod = _load_script()

    class _HttpOpener:
        def open(self, req, timeout=None):
            raise urllib.error.HTTPError(
                req.full_url, 403, "Forbidden", {},
                io.BytesIO(b'{"ok": false, "description": "Forbidden"}'))

    ok, detail = mod.send_telegram_text("TESTTOKEN123", "123", "hi",
                                        opener=_HttpOpener())
    assert not ok
    assert "TESTTOKEN123" not in detail
    assert "403" in detail


def test_main_accepts_notify_flag_without_running():
    mod = _load_script()
    # --days 0 fails validation (return 2) before any run or notify.
    assert mod.main(["--notify-telegram", "--days", "0"]) == 2
    assert mod.main(["--notify-telegram", "--only-playbook", "bogus"]) == 2
