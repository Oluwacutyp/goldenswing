"""Safety wiring on the MT5 live path: RiskGuards enforcement,
cost gate in the live signal gate, margin + swing margin shield,
market-closed guard — all must actually BLOCK, never just log.

Against the real MT5Feed + FakeMT5 terminal (no real MT5, no money).
"""
from types import SimpleNamespace

import pytest

from goldenswing.confluence import (ConfluenceConfig, ConfluenceEngine,
                                    cost_gate_ok)
from goldenswing.execution.executor import OrderRequest, OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import RiskConfig, RiskGuards, gold_spec
from goldenswing.execution.learning import TradeJournal
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from goldenswing.live.hierarchy import TimeframeStack
from goldenswing.live.macrotrend import MacroTrendResult
from mt5_fakes import FakeMT5

import pandas as pd


class FakeExecutor:
    def __init__(self):
        self.orders: list[dict] = []
        self._ticket = 9000

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return []

    def place_order(self, request):
        self._ticket += 1
        self.orders.append({"ticket": self._ticket,
                            "volume": request.volume})
        return OrderResult(True, ticket=self._ticket,
                           price=request.entry_ref,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        return OrderResult(True, ticket=ticket)


class LowMarginFake(FakeMT5):
    """Fake terminal with a nearly-exhausted free margin."""

    def __init__(self, *, free_margin, equity=10000.0, **kw):
        super().__init__(**kw)
        self._free_margin = free_margin
        self._equity = equity

    def account_info(self):
        return SimpleNamespace(balance=10000.0, equity=self._equity,
                               margin=0.0, margin_free=self._free_margin,
                               currency="USD", login=123456,
                               leverage=self.leverage,
                               trade_mode=self.trade_mode)


def _runner(tmp_path, *, feed=None, executor=None, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        # News pause OFF: the order path's news check reads the
        # REAL wall-clock calendar — these tests must not fail on
        # a real news day (e.g. PPI).  The pause itself is covered
        # by test_news_calendar.py.
        news_pause_enabled=False,
        **cfg_kw)
    return MT5Runner(cfg, feed=feed, executor=executor)


def _open_feed(**fake_kw):
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5(**fake_kw))
    feed.connect()
    return feed


def _sig(**over):
    sig = SimpleNamespace(direction=1, score=80.0, entry_ref=2000.0,
                          invalidation=1990.0, targets=[2030.0],
                          atr=10.0, index=100)
    for k, v in over.items():
        setattr(sig, k, v)
    return sig


def _place(runner, sig=None, **kw):
    stats: dict = {"executed": 0, "reconciled_closed": 0}
    runner._place_live_order(SimpleNamespace(name="fvg_midline"),
                             sig or _sig(), stats, **kw)
    return stats


def _stack(runner):
    """Minimal non-empty H1 stack: from_frames drops empty frames,
    and the fill path reads frames["H1"]."""
    runner._stack = TimeframeStack.from_frames(
        {"H1": pd.DataFrame({"open": [2000.0], "high": [2001.0],
                             "low": [1999.0], "close": [2000.5],
                             "volume": [100], "real_volume": [0],
                             "spread": [30]},
                            index=pd.DatetimeIndex(
                                [pd.Timestamp.now(tz="UTC")], tz="UTC"))})


def _journal_text(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    return path.read_text() if path.exists() else ""


# ------------------------------------------------------------- market open
def test_feed_market_open_full_vs_disabled_vs_closeonly(tmp_path):
    assert _open_feed().market_open()[0] is True
    closed = _open_feed(symbol_trade_mode=0)
    ok, reason = closed.market_open()
    assert ok is False and "trade_mode=0" in reason
    closeonly = _open_feed(symbol_trade_mode=3)
    ok, reason = closeonly.market_open()
    assert ok is False and "trade_mode=3" in reason


def test_place_live_order_refused_when_market_closed(tmp_path):
    runner = _runner(tmp_path, feed=_open_feed(symbol_trade_mode=0),
                     executor=FakeExecutor())
    runner._stack = TimeframeStack.from_frames(
        {"H1": pd.DataFrame({"open": [2000.0], "high": [2001.0],
                             "low": [1999.0], "close": [2000.5],
                             "volume": [100], "real_volume": [0],
                             "spread": [30]},
                            index=pd.DatetimeIndex(
                                [pd.Timestamp.now(tz="UTC")], tz="UTC"))})
    _place(runner)
    assert runner.executor.orders == []
    assert "market closed" in _journal_text(tmp_path)


def test_signal_path_refused_when_market_closed(tmp_path):
    runner = _runner(tmp_path, feed=_open_feed(symbol_trade_mode=0),
                     executor=FakeExecutor())
    runner._run_signal_path(None, 0, None, 0.0, {}, {})
    assert runner.blocked_stats()["legs"].get("market_closed", 0) == 1


# ------------------------------------------------------------- RiskGuards
def test_risk_guard_latched_halt_blocks_live_order(tmp_path):
    runner = _runner(tmp_path, feed=_open_feed(),
                     executor=FakeExecutor())
    _stack(runner)
    # Trip the halt honestly: anchor the peak, then book a loss
    # past the max-drawdown halt (15%).  PREMISE CORRECTION
    # (owner ruling 2026-10-23): this test originally tripped
    # the 3% DAILY loss cap — that cap is revoked; max drawdown
    # is the sole account-level hard halt.
    runner.guards.check_new_trade(balance=10000.0, open_risk_money=0.0,
                                  new_risk_money=0.0, open_count=0)
    runner.guards.record_outcome(-1600.0, 8400.0)
    assert runner.guards.halted
    _place(runner)
    assert runner.executor.orders == []
    assert runner.blocked_stats()["legs"].get("risk_guard", 0) == 1
    assert "risk guard" in _journal_text(tmp_path)


def test_risk_guard_kill_switch_blocks_live_order(tmp_path):
    runner = _runner(tmp_path, feed=_open_feed(),
                     executor=FakeExecutor())
    _stack(runner)
    (tmp_path / "mt5_KILL").write_text("stop")
    _place(runner)
    assert runner.executor.orders == []
    assert "kill switch" in _journal_text(tmp_path)


def test_risk_guard_max_positions_blocks_live_order(tmp_path):
    runner = _runner(tmp_path, feed=_open_feed(),
                     executor=FakeExecutor())
    _stack(runner)
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    for ticket in (301, 302, 303):
        runner.tick_manager.open(ticket=ticket, request=req,
                                 fill_price=2000.0, entry_bar=6)
    _place(runner)
    assert runner.executor.orders == []
    assert "max positions" in _journal_text(tmp_path)


def test_risk_guard_unstopped_position_blocks_live_order(tmp_path):
    runner = _runner(tmp_path, feed=_open_feed(),
                     executor=FakeExecutor())
    _stack(runner)
    # Adopted without an SL: risk_price == 0, exposure unknowable.
    runner.tick_manager.adopt(ticket=401, symbol="XAUUSD", direction=1,
                              volume=0.01, entry_price=2000.0, sl=None)
    _place(runner)
    assert runner.executor.orders == []
    assert "NO stop" in _journal_text(tmp_path)


def test_record_close_books_outcome_and_trips_consecutive_halt(tmp_path):
    runner = _runner(tmp_path, feed=_open_feed(),
                     executor=FakeExecutor())
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    # One booked loss: -$0.01 (price 1999.99, 0.01 lot) — far under
    # the daily/weekly/drawdown limits, so only the counters move.
    runner.tick_manager.open(ticket=501, request=req, fill_price=2000.0,
                             entry_bar=6)
    stats: dict = {"reconciled_closed": 0}
    runner._record_close(501, 1999.99, "stop", stats)
    assert runner.guards.state["consecutive_losses"] == 1
    assert runner.guards.state["day_realized"] == pytest.approx(-0.01)
    assert not runner.guards.halted
    # Five more tiny losses: 6 consecutive -> latched halt.
    for ticket in range(502, 507):
        runner.tick_manager.open(ticket=ticket, request=req,
                                 fill_price=2000.0, entry_bar=6)
        runner._record_close(ticket, 1999.99, "stop", stats)
    assert runner.guards.halted
    assert "consecutive losses" in runner.guards.state["halt_reason"]


# ------------------------------------------------------------- cost gate
def _cost_sig(**over):
    sig = SimpleNamespace(direction=1, index=10, entry_ref=2000.0,
                          invalidation=1990.0, targets=[2030.0],
                          kind="")
    for k, v in over.items():
        setattr(sig, k, v)
    return sig


class _StubPlaybook:
    name = "stub_pb"
    session_scope = "any"
    hunting_ground = SimpleNamespace(excluded=[])

    def check_live_signal(self, signal, ctx):
        return True, ""


def _engine(tmp_path):
    return ConfluenceEngine(
        playbooks=[], risk_guards=RiskGuards(tmp_path / "risk.json",
                                              RiskConfig()),
        journal=TradeJournal(tmp_path / "journal.jsonl"),
        risk_mode="shadow")


def test_cost_gate_spread_override_blocks_and_passes():
    sig = _cost_sig()
    ok, reason = cost_gate_ok(sig, 2000.0, ConfluenceConfig(),
                              spread_px=6.0)
    assert ok is False and "live-spread" in reason
    ok, _ = cost_gate_ok(sig, 2000.0, ConfluenceConfig(), spread_px=0.30)
    assert ok is True
    # No spread -> the cost_bps estimate (4 bps = $0.80) decides.
    ok, _ = cost_gate_ok(sig, 2000.0, ConfluenceConfig())
    assert ok is True
    ok, reason = cost_gate_ok(_cost_sig(invalidation=1999.0), 2000.0,
                              ConfluenceConfig())
    assert ok is False and "risk-too-small" in reason


def test_gate_live_signal_cost_gate_leg(tmp_path):
    engine = _engine(tmp_path)
    pb = _StubPlaybook()
    ctx = SimpleNamespace()
    ok, reason = engine.gate_live_signal(
        pb, _cost_sig(invalidation=1999.0), ctx)
    assert ok is False and reason.startswith("cost_gate:")
    ok, reason = engine.gate_live_signal(pb, _cost_sig(), ctx)
    assert ok is True
    ok, reason = engine.gate_live_signal(pb, _cost_sig(), ctx,
                                         spread_px=6.0)
    assert ok is False and reason.startswith("cost_gate:")


# ------------------------------------------------------------- margin
def test_margin_refused_when_free_margin_insufficient(tmp_path):
    feed = MT5Feed(MT5FeedConfig(),
                   mt5_module=LowMarginFake(free_margin=1.0))
    feed.connect()
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    _stack(runner)
    _place(runner)
    assert runner.executor.orders == []
    assert runner.blocked_stats()["legs"].get("margin", 0) == 1
    assert "margin" in _journal_text(tmp_path)


def test_margin_passes_and_order_places_when_affordable(tmp_path):
    runner = _runner(tmp_path, feed=_open_feed(),
                     executor=FakeExecutor())
    _stack(runner)
    _place(runner)
    assert len(runner.executor.orders) == 1
    assert runner.blocked_stats()["legs"].get("margin", 0) == 0


def test_margin_shield_blocks_golden_swing_below_40pct(tmp_path):
    feed = MT5Feed(MT5FeedConfig(),
                   mt5_module=LowMarginFake(free_margin=4000.0,
                                            equity=10000.0))
    feed.connect()
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    _stack(runner)
    runner._alignment = MacroTrendResult(
        d1_ema=1, w1_ema=1, mn1_ema=1, alignment_level=1.5,
        alignment_label="Golden_Swing", golden_swing=True,
        size_multiplier=1.5)
    _place(runner)
    assert runner.executor.orders == []
    assert runner.blocked_stats()["legs"].get("margin_shield", 0) == 1


def test_margin_shield_scoped_out_for_plain_base_order(tmp_path):
    # Same tight free margin, but NOT a Golden Swing base order and
    # NOT a layer: the shield does not apply, the plain margin
    # check (buffered 33.60 < free 100) passes, the order places.
    feed = MT5Feed(MT5FeedConfig(),
                   mt5_module=LowMarginFake(free_margin=100.0,
                                            equity=10000.0))
    feed.connect()
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    _stack(runner)
    _place(runner)
    assert len(runner.executor.orders) == 1
    assert runner.blocked_stats()["legs"].get("margin_shield", 0) == 0


def test_margin_shield_blocks_layer_below_40pct(tmp_path):
    feed = MT5Feed(MT5FeedConfig(),
                   mt5_module=LowMarginFake(free_margin=4000.0,
                                            equity=10000.0))
    feed.connect()
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    _stack(runner)
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    runner.tick_manager.open(ticket=601, request=req, fill_price=2000.0,
                             entry_bar=6)
    _place(runner, is_layer=True, layer_lots=0.08)
    assert runner.executor.orders == []
    assert runner.blocked_stats()["legs"].get("margin_shield", 0) == 1


def test_margin_shield_allows_layer_with_headroom(tmp_path):
    runner = _runner(tmp_path, feed=_open_feed(),
                     executor=FakeExecutor())
    _stack(runner)
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    runner.tick_manager.open(ticket=601, request=req, fill_price=2000.0,
                             entry_bar=6)
    _place(runner, is_layer=True, layer_lots=0.08)
    assert len(runner.executor.orders) == 1
    assert runner.blocked_stats()["legs"].get("margin_shield", 0) == 0
