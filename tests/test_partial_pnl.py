"""Partial-P&L booking on the live MT5 path (PNL-PARTIAL).

Before this fix, a partial leg's realized P&L never entered ANY
book: the tick/management partial paths reduced the tracked volume
and logged a note, and the full close booked the remainder only —
so day/week totals and the journal close record understated the
trade by exactly the banked legs.  Now:

- a partial leg books exactly its fraction's money into day/week
  at partial time (RiskGuards.record_partial — NOT a closed trade,
  so trades_today / the loss streak are untouched);
- the full close's RiskGuards increment is the remainder ONLY
  (no double-count);
- the journal close record carries the trade TOTAL (banked +
  remainder, as the paper CloseRecord does) at the volume-weighted
  total R, so day/week and the journal reconcile;
- a failed partial books nothing.

Gold spec: $0.01 tick worth $1.00/lot, so money = move / 0.01 *
1.00 * volume = move * 100 * volume.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from goldenswing.execution.executor import OrderRequest
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import RiskGuards, gold_spec


class FakeExecutor:
    def __init__(self, fail_close=False):
        self.fail_close = fail_close

    def symbol_spec(self, symbol):
        return gold_spec()

    def modify_sl(self, ticket, sl, tp=None):
        return SimpleNamespace(ok=True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        if self.fail_close:
            return SimpleNamespace(ok=False, ticket=ticket,
                                   message="broker refused")
        return SimpleNamespace(ok=True, ticket=ticket)

    def positions(self, symbol=None):
        return []


def _runner(tmp_path, fail_close=False):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, executor=FakeExecutor(fail_close=fail_close))
    runner.feed = SimpleNamespace(
        account_state=lambda: {"balance": 10000.0})
    return runner


def _open_long(runner, ticket=7, entry=4000.0, volume=0.02,
               stop=3990.0, targets=(4020.0,), alert_id="MT5-test-1"):
    """Long @ entry, ATR 0 -> buffered stop == raw stop, risk $10."""
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=volume,
                       stop_loss=stop, entry_ref=entry,
                       targets=list(targets), atr=0.0,
                       alert_id=alert_id, bar_index=10, magic=260101)
    return runner.tick_manager.open(ticket=ticket, request=req,
                                    fill_price=entry, entry_bar=10,
                                    broker_sl=stop)


def _open_short(runner, ticket=7, entry=4000.0, volume=0.02,
                stop=4010.0, targets=(3980.0,), alert_id="MT5-test-1"):
    req = OrderRequest(symbol="XAUUSD", direction=-1, volume=volume,
                       stop_loss=stop, entry_ref=entry,
                       targets=list(targets), atr=0.0,
                       alert_id=alert_id, bar_index=10, magic=260101)
    return runner.tick_manager.open(ticket=ticket, request=req,
                                    fill_price=entry, entry_bar=10,
                                    broker_sl=stop)


def _closes(runner):
    return [r for r in runner.journal.records()
            if r.get("type") == "close"]


def test_tick_partial_books_exact_fraction(tmp_path):
    runner = _runner(tmp_path)
    pos = _open_long(runner)  # 0.02 @ 4000, risk $10, TP 4020
    runner._manage_tick(4020.0, {"tick_actions": 0})
    # 0.01 lots banked at +$20: $20 move * 100 * 0.01 = $20.
    assert pos.banked_pnl == pytest.approx(20.0)
    assert len(pos.banked_legs) == 1
    assert pos.volume == pytest.approx(0.01)
    st = runner.guards.state
    assert st["day_realized"] == pytest.approx(20.0)
    assert st["week_realized"] == pytest.approx(20.0)
    # NOT a closed trade: no trade count, no streak movement.
    assert st["trades_today"] == 0
    assert st["consecutive_losses"] == 0


def test_full_close_books_remainder_only_and_reconciles(tmp_path):
    runner = _runner(tmp_path)
    pos = _open_long(runner)
    runner._manage_tick(4020.0, {"tick_actions": 0})
    assert runner.guards.state["day_realized"] == pytest.approx(20.0)
    stats = {"tick_actions": 0, "reconciled_closed": 0}
    runner._record_close(7, 3990.0, "stop", stats)
    # Close-time ledger increment is the remainder ONLY: -$10
    # ($10 move * 100 * 0.01).  Cumulative day == trade total.
    st = runner.guards.state
    assert st["day_realized"] == pytest.approx(10.0)
    assert st["week_realized"] == pytest.approx(10.0)
    assert st["trades_today"] == 1
    closes = _closes(runner)
    assert len(closes) == 1
    assert closes[0]["pnl_money"] == pytest.approx(10.0)
    # Volume-weighted total R: 0.5 * +2R + 0.5 * -1R = +0.5R.
    assert closes[0]["r_net"] == pytest.approx(0.5)
    assert closes[0]["partial_taken"] is True
    # Journal total == ledger total: both surfaces reconcile.
    assert closes[0]["pnl_money"] == pytest.approx(st["day_realized"])


def test_failed_partial_books_nothing(tmp_path):
    runner = _runner(tmp_path, fail_close=True)
    pos = _open_long(runner)
    runner._manage_tick(4020.0, {"tick_actions": 0})
    assert pos.banked_pnl == 0.0
    assert pos.banked_legs == []
    assert pos.volume == pytest.approx(0.02)
    st = runner.guards.state
    assert st["day_realized"] == 0.0
    assert st["week_realized"] == 0.0
    assert _closes(runner) == []


def test_layer_stop_leg_uses_layer_entry(tmp_path):
    runner = _runner(tmp_path)
    pos = _open_long(runner, volume=0.01, targets=(4020.0,))
    runner.tick_manager.add_layer(7, entry=4010.0, volume=0.01,
                                  sl=4005.0, alert_id="MT5-test-1")
    assert pos.volume == pytest.approx(0.02)
    runner._manage_tick(4005.0, {"tick_actions": 0})
    # Layer leg vs ITS entry 4010: -$5 ($5 move * 100 * 0.01).
    # Against the base entry 4000 it would (wrongly) read +$5.
    assert pos.banked_pnl == pytest.approx(-5.0)
    assert runner.guards.state["day_realized"] == pytest.approx(-5.0)
    assert pos.volume == pytest.approx(0.01)
    assert len(runner.tick_manager.layers(7)) == 1


def test_management_partial_books_leg(tmp_path):
    runner = _runner(tmp_path)
    pos = _open_long(runner)
    ok = runner._runner_partial(7, pos, 0.5, "zone_bank", {},
                                4020.0)
    assert ok is True
    assert pos.banked_pnl == pytest.approx(20.0)
    assert pos.volume == pytest.approx(0.01)
    assert runner.guards.state["day_realized"] == pytest.approx(20.0)


def test_no_partial_close_unchanged(tmp_path):
    runner = _runner(tmp_path)
    _open_long(runner)
    stats = {"tick_actions": 0, "reconciled_closed": 0}
    runner._record_close(7, 4010.0, "stop", stats)
    closes = _closes(runner)
    assert len(closes) == 1
    assert closes[0]["pnl_money"] == pytest.approx(20.0)
    assert closes[0]["r_net"] == pytest.approx(1.0)
    assert closes[0]["partial_taken"] is False
    assert runner.guards.state["day_realized"] == pytest.approx(20.0)
    assert runner.guards.state["trades_today"] == 1


def test_short_partial_sign(tmp_path):
    runner = _runner(tmp_path)
    pos = _open_short(runner)  # short 0.02 @ 4000, stop 4010
    runner._manage_tick(3980.0, {"tick_actions": 0})
    # Short wins when price falls: 0.01 lots * $20 move = +$20.
    assert pos.banked_pnl == pytest.approx(20.0)
    assert runner.guards.state["day_realized"] == pytest.approx(20.0)


def test_record_partial_touches_no_trade_counters(tmp_path):
    guards = RiskGuards(state_path=tmp_path / "risk.json")
    guards.record_partial(-5.0, 10000.0)
    st = guards.state
    assert st["day_realized"] == pytest.approx(-5.0)
    assert st["week_realized"] == pytest.approx(-5.0)
    assert st["trades_today"] == 0
    assert st["consecutive_losses"] == 0
    guards.record_outcome(-5.0, 9995.0)
    st = guards.state
    assert st["trades_today"] == 1
    assert st["consecutive_losses"] == 1


def test_bank_leg_refuses_unknown_and_closed(tmp_path):
    runner = _runner(tmp_path)
    tm = runner.tick_manager
    assert tm.bank_leg(999, volume=0.01, price=4020.0, entry=4000.0,
                       stop=3990.0, pnl_money=20.0) is False
    _open_long(runner)
    stats = {"tick_actions": 0, "reconciled_closed": 0}
    runner._record_close(7, 4010.0, "stop", stats)
    assert tm.bank_leg(7, volume=0.01, price=4020.0, entry=4000.0,
                       stop=3990.0, pnl_money=20.0) is False
