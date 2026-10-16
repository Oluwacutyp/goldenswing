"""Live-engine audit: backtest fetch bugs must not exist on live.

Backtest failures 2026-10-16: bare initialize() attaching to the
wrong MT5 terminal, XAUUSD vs XAUUSDm, tz-aware date params,
missing-history blocks, F2/History Center (MT4, not MT5).

Live verdict (audit): bars use copy_rates_from_pos (no dates —
immune), ticks are live (not history — immune), symbol auto-detect
already exists, per-TF fetch (no common-window block). The REAL live
gaps were terminal pinning (never wired: run_mt5 had no
--terminal-path / MT5_TERMINAL_PATH, executor.connect() took no
path) and thin connect diagnostics. These tests pin those.
"""
import os
from types import SimpleNamespace

import pytest

from goldenswing.execution.mt5 import MT5Executor
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from tests.mt5_fakes import FakeMT5


class RecordingMT5(FakeMT5):
    """Fake that records initialize() kwargs and reports a rich terminal."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.init_kwargs: list[dict] = []

    def initialize(self, **kw):
        self.init_kwargs.append(dict(kw))
        return True

    def terminal_info(self):
        return SimpleNamespace(connected=True, build=3815,
                               server="Exness-MT5Trial10",
                               path="C:/Program Files/Exness MT5/terminal64.exe",
                               data_path="C:/Users/Admin/AppData/Roaming/MetaQuotes/Terminal/ABC",
                               name="MetaTrader 5", company="Exness")

    def account_info(self):
        return SimpleNamespace(balance=10000.0, equity=10000.0,
                               margin=0.0, margin_free=10000.0,
                               currency="USD", login=81800936,
                               leverage=2000, trade_mode=0)


class FailingMT5(RecordingMT5):
    def initialize(self, **kw):
        self.init_kwargs.append(dict(kw))
        return False

    def last_error(self):
        return (2, "terminal not found")


def test_feed_connect_uses_cfg_terminal_path():
    fake = RecordingMT5()
    feed = MT5Feed(MT5FeedConfig(terminal_path="C:/MT5/terminal64.exe"),
                   mt5_module=fake)
    ok, msg = feed.connect()
    assert ok
    assert fake.init_kwargs and fake.init_kwargs[0].get("path") == "C:/MT5/terminal64.exe"


def test_feed_connect_falls_back_to_env_terminal_path(monkeypatch):
    monkeypatch.setenv("MT5_TERMINAL_PATH", "C:/Env/terminal64.exe")
    fake = RecordingMT5()
    feed = MT5Feed(MT5FeedConfig(), mt5_module=fake)
    ok, _ = feed.connect()
    assert ok
    assert fake.init_kwargs[0].get("path") == "C:/Env/terminal64.exe"


def test_feed_connect_failure_carries_diagnostics(monkeypatch):
    monkeypatch.setenv("MT5_TERMINAL_PATH", "C:/Env/terminal64.exe")
    fake = FailingMT5()
    feed = MT5Feed(MT5FeedConfig(max_connect_attempts=1), mt5_module=fake)
    ok, msg = feed.connect()
    assert not ok
    assert "diagnostics:" in msg
    assert "Exness-MT5Trial10" in msg  # terminal server
    assert "81800936" in msg  # account login
    assert "terminal64.exe" in msg  # terminal path


def test_executor_connect_uses_env_terminal_path(monkeypatch):
    monkeypatch.setenv("MT5_TERMINAL_PATH", "C:/Env/terminal64.exe")
    fake = RecordingMT5()
    ex = MT5Executor(mt5_module=fake)
    ok, _ = ex.connect()
    assert ok
    assert fake.init_kwargs[0].get("path") == "C:/Env/terminal64.exe"


def test_executor_connect_explicit_path_wins_over_env(monkeypatch):
    monkeypatch.setenv("MT5_TERMINAL_PATH", "C:/Env/terminal64.exe")
    fake = RecordingMT5()
    ex = MT5Executor(mt5_module=fake)
    ok, _ = ex.connect(path="C:/Explicit/terminal64.exe")
    assert ok
    assert fake.init_kwargs[0].get("path") == "C:/Explicit/terminal64.exe"


def test_live_bars_use_from_pos_not_date_range():
    """Live bars never pass dates to MT5 — immune to the backtest
    tz-aware vs server-time copy_rates_range bug by construction."""
    import inspect
    from goldenswing.feed import mt5 as feed_mod
    src = inspect.getsource(feed_mod.MT5Feed.bars_for)
    assert "copy_rates_from_pos" in src
    assert "copy_rates_range" not in src
