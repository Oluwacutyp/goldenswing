"""Step 8 — interactive terminal: engine-served live chart page.

The 2026-10-25 design: a page served by the engine itself, live
stack frames as data (not screenshots), studies as toggles, a
freshness banner that never shows old candles as live, and a hard
read-only boundary — it views, orders stay on the engine/Telegram
path. These tests pin the server contract with a fake provider, so
no runner or broker is needed to prove it.
"""
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goldenswing.alerts.commands import CommandContext
from goldenswing.charts.terminal import build_terminal_page
from goldenswing.charts.terminal_server import (
    TerminalServer,
    bars_to_rows,
    compute_status,
    session_spans,
)
from goldenswing.execution.mt5_commands import MT5_COMMANDS
from goldenswing.execution.mt5_runner import MT5RunnerConfig

rng = np.random.default_rng(31)


def _frame(n, start, freq):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    walk = 4000.0 + np.cumsum(rng.normal(0, 3, n))
    return pd.DataFrame({
        "open": walk, "high": walk + 4.0, "low": walk - 4.0,
        "close": walk + rng.normal(0, 1, n),
    }, index=idx)


STUDIES = {
    "signal": "MT5-test-1", "signal_time": "2026-10-27T12:00:00+00:00",
    "zone": {"top": 4020.0, "bottom": 4000.0, "kind": "support"},
    "nearby": [{"level": 4050.0, "tier": "MAJOR", "side": "resistance"}],
    "fib": {"levels": [[0.5, 4020.0]], "ote": [4005.0, 4018.0],
            "range": [3990.0, 4060.0]},
}


class FakeProvider:
    def __init__(self, frames, studies=STUDIES, status=None):
        self._frames = frames
        self._studies = studies
        self._status = status

    def frames(self, tf):
        return self._frames.get(tf)

    def studies(self):
        return self._studies

    def status(self):
        if self._status is not None:
            return self._status
        now = datetime.now(timezone.utc)
        return {"server_time": now.isoformat(), "market_open": True,
                "tfs": {tf: (f.index[-1].isoformat()
                             if f is not None and len(f) > 0 else None)
                        for tf, f in self._frames.items()}}


@pytest.fixture()
def server():
    frames = {"H1": _frame(200, "2026-10-26", "h"),
              "M5": _frame(100, "2026-10-28", "5min")}
    srv = TerminalServer(FakeProvider(frames), host="127.0.0.1", port=0)
    srv.start()
    yield srv
    srv.stop()


def _get(srv, path, method="GET"):
    req = urllib.request.Request(srv.url.rstrip("/") + path,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_page_served_self_contained(server):
    code, body = _get(server, "/")
    html = body.decode()
    assert code == 200
    assert "GoldenSwing terminal" in html
    assert "<script src" not in html
    assert "https://" not in html
    assert "http://" not in html
    for marker in ("api/frames", "api/status", "setInterval(poll,30000)",
                   'id="ov-sess"', 'id="ov-nearby"', 'id="ov-fib"',
                   'id="ov-prem"', 'id="ov-zone"', 'id="theme-btn"',
                   'id="fresh"'):
        assert marker in html


def test_frames_endpoint_shape_and_cap(server):
    code, body = _get(server, "/api/frames?tf=H1")
    assert code == 200
    payload = json.loads(body)
    assert payload["tf"] == "H1"
    assert len(payload["bars"]) == 200
    row = payload["bars"][0]
    assert len(row) == 5 and isinstance(row[0], str)
    assert all(isinstance(v, float) for v in row[1:])
    assert isinstance(payload["sessions"], list)
    assert payload["studies"]["signal"] == "MT5-test-1"


def test_frames_cap_500():
    frame = _frame(600, "2026-10-26", "h")
    assert len(bars_to_rows(frame)) == 500


def test_frames_errors(server):
    assert _get(server, "/api/frames")[0] == 400
    assert _get(server, "/api/frames?tf=ZZZ")[0] == 400
    assert _get(server, "/api/frames?tf=H4")[0] == 404
    assert _get(server, "/nope")[0] == 404


def test_write_methods_are_405(server):
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        code, body = _get(server, "/", method=method)
        assert code == 405
        assert json.loads(body)["error"] == "terminal is read-only"
        assert _get(server, "/api/frames?tf=H1",
                    method=method)[0] == 405


def test_status_endpoint_carries_freshness(server):
    code, body = _get(server, "/api/status")
    assert code == 200
    payload = json.loads(body)
    assert payload["market_open"] is True
    assert payload["instrument"] == "XAUUSD"
    assert set(payload["tfs"]) == {"M5", "M15", "M30", "H1", "H4", "D1"}
    assert payload["tfs"]["H1"]["last_bar"] is not None
    assert payload["tfs"]["H4"]["last_bar"] is None
    assert payload["tfs"]["H4"]["stale"] is False


def test_compute_status_stale_matrix():
    now = datetime(2026, 10, 27, 15, 0, tzinfo=timezone.utc)
    old_h1 = (now - timedelta(seconds=8000)).isoformat()
    young_h1 = (now - timedelta(seconds=60)).isoformat()
    open_old = compute_status(server_time=now.isoformat(),
                              market_open=True,
                              last_bars={"H1": old_h1}, now=now)
    assert open_old["tfs"]["H1"]["stale"] is True
    open_young = compute_status(server_time=now.isoformat(),
                                market_open=True,
                                last_bars={"H1": young_h1}, now=now)
    assert open_young["tfs"]["H1"]["stale"] is False
    closed_old = compute_status(server_time=now.isoformat(),
                                market_open=False,
                                last_bars={"H1": old_h1}, now=now)
    assert closed_old["tfs"]["H1"]["stale"] is False
    missing = compute_status(server_time=now.isoformat(),
                             market_open=True, last_bars={}, now=now)
    assert missing["tfs"]["H1"]["last_bar"] is None
    assert missing["tfs"]["H1"]["stale"] is False


def test_session_spans_shape():
    frame = _frame(200, "2026-10-26", "h")
    spans = session_spans(frame)
    assert isinstance(spans, list)
    for start, end in spans:
        assert 0 <= start < end <= 200


def test_terminal_page_builder_instrument():
    html = build_terminal_page(instrument="XAUUSD")
    assert "GoldenSwing terminal · XAUUSD" in html


def test_runner_config_terminal_defaults_and_env(monkeypatch):
    cfg = MT5RunnerConfig()
    assert cfg.terminal_host == "127.0.0.1"
    assert cfg.terminal_port == 8899
    monkeypatch.setenv("GS_TERMINAL_HOST", "0.0.0.0")
    monkeypatch.setenv("GS_TERMINAL_PORT", "9999")
    cfg2 = MT5RunnerConfig.from_env()
    assert cfg2.terminal_host == "0.0.0.0"
    assert cfg2.terminal_port == 9999


def test_terminal_command_starts_and_reports():
    desc, fn = MT5_COMMANDS["/terminal"]
    assert "terminal" in desc.lower()
    fake = SimpleNamespace(
        start_terminal_server=lambda: "http://127.0.0.1:8899/",
        terminal_status=lambda: {"running": True, "host": "127.0.0.1",
                                 "port": 8899})
    res = fn(CommandContext(mt5_control=fake), [])
    assert "http://127.0.0.1:8899/" in res.text
    assert "localhost" in res.text
    assert "Read-only" in res.text
    assert res.reply_markup is not None


def test_terminal_command_unavailable_without_runner():
    _, fn = MT5_COMMANDS["/terminal"]
    res = fn(CommandContext(mt5_control=None), [])
    assert "unavailable" in res.text


def test_terminal_command_unavailable_on_old_runner():
    _, fn = MT5_COMMANDS["/terminal"]
    res = fn(CommandContext(mt5_control=SimpleNamespace()), [])
    assert "predates" in res.text
