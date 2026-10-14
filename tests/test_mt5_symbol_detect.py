"""MT5 gold-symbol auto-detect: brokers name gold differently
(Exness: XAUUSDm, not XAUUSD).  The engine must resolve the broker's
gold symbol instead of dying on symbol_select(XAUUSD).

Rules under test: configured symbol tried first, always.  Auto-detect
is ONLY the fallback for a non-explicit (default) symbol — an
explicit --symbol / GS_SYMBOL is never second-guessed.  The runner
adopts the feed's resolved symbol so orders go to the same symbol
the feed reads.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from mt5_fakes import FakeMT5
from run_mt5 import _resolve_symbol


class _FakeExecutor:
    def connect(self, **kw):
        return True, "fake connected"

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return []

    def place_order(self, request):
        return OrderResult(True, ticket=1, price=request.entry_ref,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        return OrderResult(True, ticket=ticket)


def _feed(symbols, **cfg_kw):
    cfg_kw.setdefault("max_connect_attempts", 1)
    return MT5Feed(MT5FeedConfig(**cfg_kw),
                   mt5_module=FakeMT5(symbols=symbols))


def test_exact_symbol_present_uses_it_unchanged():
    feed = _feed(["XAUUSD", "XAUUSDm", "EURUSD"])
    ok, msg = feed.connect()
    assert ok, msg
    assert feed.cfg.symbol == "XAUUSD"
    assert msg == "MT5 connected"


def test_exact_missing_xauusdm_present_uses_xauusdm():
    feed = _feed(["XAUUSDm", "EURUSD"])
    ok, msg = feed.connect()
    assert ok, msg
    assert feed.cfg.symbol == "XAUUSDm"
    assert "XAUUSD not found" in msg and "using XAUUSDm" in msg


def test_other_xauusd_variant_used_when_no_plain_or_m():
    feed = _feed(["XAUUSD.pro", "EURUSD"])
    ok, msg = feed.connect()
    assert ok, msg
    assert feed.cfg.symbol == "XAUUSD.pro"
    assert "using XAUUSD.pro" in msg


def test_no_xauusd_symbol_fails_with_candidates_and_hint():
    feed = _feed(["GOLD", "EURUSD"])
    ok, msg = feed.connect()
    assert not ok
    assert "no XAUUSD* symbol" in msg
    assert "GOLD" in msg  # gold-like candidate listed
    assert "--symbol" in msg


def test_no_xauusd_and_no_candidates_still_clear():
    feed = _feed(["EURUSD", "GBPUSD"])
    ok, msg = feed.connect()
    assert not ok
    assert "no XAUUSD* symbol" in msg
    assert "--symbol" in msg


def test_explicit_symbol_failure_never_auto_detects():
    feed = _feed(["XAUUSDm", "EURUSD"],
                 symbol="XAUUSD", symbol_explicit=True)
    ok, msg = feed.connect()
    assert not ok
    assert "explicit" in msg and "auto-detect disabled" in msg
    assert feed.cfg.symbol == "XAUUSD"  # unchanged, not switched


def test_explicit_symbol_present_is_used_as_is():
    feed = _feed(["XAUUSD", "XAUUSDm"],
                 symbol="XAUUSDm", symbol_explicit=True)
    ok, msg = feed.connect()
    assert ok, msg
    assert feed.cfg.symbol == "XAUUSDm"


def test_runner_adopts_detected_symbol(tmp_path):
    feed = _feed(["XAUUSDm", "EURUSD"])
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, feed=feed, executor=_FakeExecutor())
    runner.run(max_ticks=1)
    assert runner.cfg.symbol == "XAUUSDm"
    note = (tmp_path / "mt5_journal.jsonl").read_text()
    assert "symbol=XAUUSDm" in note
    assert "using XAUUSDm" in note


def test_resolve_symbol_cli_and_env():
    assert _resolve_symbol(None, {}) == ("XAUUSD", False)
    assert _resolve_symbol("XAUUSDm", {}) == ("XAUUSDm", True)
    assert _resolve_symbol(None, {"GS_SYMBOL": "XAUUSD.r"}) == (
        "XAUUSD.r", True)
    # CLI beats env.
    assert _resolve_symbol("XAUUSDm", {"GS_SYMBOL": "XAUUSD.r"}) == (
        "XAUUSDm", True)
