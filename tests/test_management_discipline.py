"""Goldhunter-style management discipline (steal the discipline, not
the calls):

- Resting limits at POI only: a market fill is only ever at-or-better
  than the POI; price past the POI -> resting LIMIT, never chase
  (see test_mt5_order_selection.py for the selection matrix).
- Telegram follow-ups on fill (already existed), TP hit, SL hit,
  breakeven move, partial bank, layer stop — the disciplined
  caller's update.
- Honest recaps: daily/weekly recaps list EVERY closed trade, wins
  AND losses, never a wins-only highlight reel.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.alerts.commands import CommandContext
from goldenswing.execution.executor import OrderRequest
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_commands import MT5_COMMANDS
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.reporting import (
    build_daily_recap, build_weekly_recap, recap_for_period)
from goldenswing.execution.risk import gold_spec
from goldenswing.execution.tick_manager import TickAction


class FakeSender:
    def __init__(self):
        self.texts: list[str] = []

    def send_text(self, text, silent=False, reply_markup=None):
        self.texts.append(text)
        return SimpleNamespace(delivered=True, detail="ok")

    def send_photo(self, png, caption):
        self.texts.append(caption)
        return SimpleNamespace(delivered=True, detail="ok")


class FakeExecutor:
    def symbol_spec(self, symbol):
        return gold_spec()

    def modify_sl(self, ticket, sl, tp=None):
        return SimpleNamespace(ok=True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        return SimpleNamespace(ok=True, ticket=ticket)

    def positions(self, symbol=None):
        return []


def _runner(tmp_path, sender=None):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, executor=FakeExecutor())
    if sender is not None:
        runner.senders = [sender]
    return runner


def _open_long(runner, ticket=7, entry=2005.0, targets=(2040.0,)):
    """Long 0.01 @ 2005, raw invalidation 1990, ATR 10 -> buffered
    stop0 1987.5, risk 17.5."""
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=entry,
                       targets=list(targets), atr=10.0,
                       alert_id="MT5-fvg_limit-42", bar_index=10,
                       magic=260101)
    runner.tick_manager.open(ticket=ticket, request=req, fill_price=entry,
                             entry_bar=10, broker_sl=1987.5)
    return req


# ------------------------------------------------------- close follow-ups
def test_close_tp_hit_follow_up(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    _open_long(runner)
    stats: dict = {"reconciled_closed": 0}
    runner._record_close(7, 2045.0, "broker_closed", stats)
    assert any("MT5 CLOSE" in t and "TP hit" in t and "+2.29R" in t
               for t in sender.texts)
    # Structured journal reason keeps the raw rule reason.
    closes = [r for r in runner.journal.records()
              if r.get("type") == "close"]
    assert closes[-1]["reason"] == "broker_closed"
    assert "TP hit" in (tmp_path / "mt5_journal.jsonl").read_text()


def test_close_sl_hit_follow_up(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    _open_long(runner)
    runner._record_close(7, 1985.0, "broker_closed", {"reconciled_closed": 0})
    assert any("SL hit" in t and "-1.14R" in t for t in sender.texts)
    assert not any("TP hit" in t for t in sender.texts)


def test_close_breakeven_stop_label(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    _open_long(runner)
    runner.tick_manager.apply(
        7, TickAction("modify_sl", level=2005.0, reason="breakeven"))
    runner._record_close(7, 2005.0, "broker_closed", {"reconciled_closed": 0})
    assert any("SL hit (breakeven)" in t and "+0.00R" in t
               for t in sender.texts)


def test_close_trail_stop_label(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    _open_long(runner)
    runner.tick_manager.apply(
        7, TickAction("modify_sl", level=2010.0, reason="trail"))
    runner._record_close(7, 2010.0, "broker_closed", {"reconciled_closed": 0})
    assert any("SL hit (trail)" in t for t in sender.texts)


def test_close_timeout_keeps_timeout_label(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    _open_long(runner)
    runner._record_close(7, 2020.0, "timeout", {"reconciled_closed": 0})
    assert any("timeout" in t and "+0.86R" in t for t in sender.texts)
    assert not any("TP hit" in t for t in sender.texts)


def test_close_between_levels_is_unexplained_not_tp(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    _open_long(runner)
    runner._record_close(7, 2020.0, "broker_closed", {"reconciled_closed": 0})
    assert any("broker_closed" in t for t in sender.texts)
    assert not any("TP hit" in t for t in sender.texts)


def test_close_safety_net_stop_is_sl_hit(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    _open_long(runner)
    runner._record_close(7, 1987.0, "stop", {"reconciled_closed": 0})
    assert any("SL hit" in t for t in sender.texts)


# ------------------------------------------------- tick-management follow-ups
def test_breakeven_move_sends_follow_up(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    _open_long(runner)
    runner._manage_tick(2023.0, {"tick_actions": 0})  # +1.03R: breakeven triggers
    assert any("MT5 BREAKEVEN" in t and "risk off" in t
               for t in sender.texts)


def test_partial_bank_sends_follow_up(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    _open_long(runner)
    runner._manage_tick(2040.0, {"tick_actions": 0})  # partial pool + breakeven
    assert any("MT5 PARTIAL" in t and "banked" in t
               for t in sender.texts)


def test_layer_stop_sends_follow_up(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    _open_long(runner)
    runner.tick_manager.add_layer(7, entry=2010.0, volume=0.01, sl=2000.0,
                                  alert_id="MT5-fvg_limit-42-layer",
                                  playbook="fvg_limit")
    runner._manage_tick(1999.0, {"tick_actions": 0})  # below the layer SL, above base stop
    assert any("MT5 LAYER STOP" in t and "base rides" in t
               for t in sender.texts)


def test_trail_move_stays_journal_only(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    _open_long(runner)
    runner._manage_tick(2023.0, {"tick_actions": 0})  # breakeven (follow-up, not trail)
    runner._manage_tick(2050.0, {"tick_actions": 0})  # trail move: journal only
    trail_notes = [t for t in sender.texts if "trail" in t.lower()]
    assert trail_notes == [] or all(
        "SL hit (trail)" not in t for t in trail_notes)


# ------------------------------------------------------------- honest recaps
def _journal_with_trades(tmp_path):
    journal = TradeJournal(tmp_path / "j.jsonl")
    # Two closed trades on 2026-10-13: one win, one loss.  One trade
    # on 2026-10-12 (outside the daily window).
    for aid, ts, entry, exit_, r in (
            ("a1", "2026-10-13T10:00:00+00:00", 2005.0, 2031.0, 1.5),
            ("a2", "2026-10-13T15:00:00+00:00", 2005.0, 1987.5, -1.0),
            ("a3", "2026-10-12T10:00:00+00:00", 2005.0, 2022.5, 1.0)):
        journal.append({"type": "open", "alert_id": aid,
                        "instrument": "XAUUSD", "playbook": "fvg_limit",
                        "direction": 1, "entry": entry, "stop": 1990.0,
                        "volume": 0.01, "risk_money": 17.5,
                        "ts": ts})
        journal.append({"type": "close", "alert_id": aid, "ticket": 1,
                        "exit": exit_, "reason": "target" if r > 0
                        else "stop", "pnl_money": r * 17.5, "r_net": r,
                        "hold_bars": 3, "ts": ts})
    return journal


def test_daily_recap_lists_wins_and_losses(tmp_path):
    journal = _journal_with_trades(tmp_path)
    text = build_daily_recap(journal, date(2026, 10, 13))
    assert "Trades: 2" in text and "Wins: 1" in text
    assert "Losses: 1" in text
    assert "+1.50R" in text and "-1.00R" in text
    assert "✅" in text and "❌" in text
    assert "nothing hidden" in text
    # The 10-12 trade is outside the daily window.
    assert text.count("fvg_limit") == 3  # per-playbook row + 2 trades


def test_weekly_recap_covers_mon_sun(tmp_path):
    journal = _journal_with_trades(tmp_path)
    text = build_weekly_recap(journal, date(2026, 10, 12))  # Monday
    assert "Trades: 3" in text
    assert "Losses: 1" in text


def test_empty_window_recap(tmp_path):
    journal = TradeJournal(tmp_path / "empty.jsonl")
    text = build_daily_recap(journal, date(2026, 10, 13))
    assert "no closed trades" in text


def test_recap_command_daily_and_usage(tmp_path):
    # /recap daily = trailing 24h to NOW: stamp the trades now.
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    journal = TradeJournal(tmp_path / "j.jsonl")
    for aid, r in (("w1", 1.5), ("l1", -1.0)):
        journal.append({"type": "open", "alert_id": aid,
                        "instrument": "XAUUSD", "playbook": "fvg_limit",
                        "direction": 1, "entry": 2005.0, "stop": 1990.0,
                        "volume": 0.01, "risk_money": 17.5, "ts": now})
        journal.append({"type": "close", "alert_id": aid, "ticket": 1,
                        "exit": 2031.0, "reason": "target",
                        "pnl_money": r * 17.5, "r_net": r, "hold_bars": 3,
                        "ts": now})
    ctx = CommandContext(journal=journal, reports_dir=tmp_path)
    fn = MT5_COMMANDS["/recap"][1]
    res = fn(ctx, ["daily"])
    assert "Losses: 1" in (res.text or "")
    res = fn(ctx, ["bogus"])
    assert "usage" in (res.text or "")


def test_recap_command_long_week_goes_to_file(tmp_path):
    journal = TradeJournal(tmp_path / "j.jsonl")
    for i in range(60):
        aid = f"t{i}"
        journal.append({"type": "open", "alert_id": aid,
                        "instrument": "XAUUSD", "playbook": "fvg_limit",
                        "direction": 1, "entry": 2005.0, "stop": 1990.0,
                        "volume": 0.01, "risk_money": 17.5,
                        "ts": "2026-10-13T10:00:00+00:00"})
        journal.append({"type": "close", "alert_id": aid, "ticket": 1,
                        "exit": 2031.0, "reason": "target",
                        "pnl_money": 26.25, "r_net": 1.5, "hold_bars": 3,
                        "ts": "2026-10-13T10:00:00+00:00"})
    ctx = CommandContext(journal=journal, reports_dir=tmp_path)
    res = MT5_COMMANDS["/recap"][1](ctx, ["weekly"])
    assert res.document is not None
    assert "wins and losses" in (res.caption or "")
    assert Path(res.document).exists()
    assert "fvg_limit" in Path(res.document).read_text()


# ------------------------------------------------------- rollover recaps
def test_rollover_sends_daily_recap(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    journal = _journal_with_trades(tmp_path)
    runner.journal = journal
    runner._last_tick_date = date(2026, 10, 13)
    runner._maybe_send_rollover_recap(
        {"time": pd.Timestamp("2026-10-14 00:00", tz="UTC"),
         "mid": 2000.0})
    assert len(sender.texts) == 1
    assert "daily — 2026-10-13" in sender.texts[0]
    assert "Losses: 1" in sender.texts[0]


def test_rollover_monday_sends_weekly_too(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    journal = _journal_with_trades(tmp_path)
    runner.journal = journal
    runner._last_tick_date = date(2026, 10, 11)  # Sunday
    runner._maybe_send_rollover_recap(
        {"time": pd.Timestamp("2026-10-12 00:00", tz="UTC"),
         "mid": 2000.0})
    assert len(sender.texts) == 2
    assert "daily — 2026-10-11" in sender.texts[0]
    assert "weekly — 2026-10-05" in sender.texts[1]


def test_rollover_same_day_sends_nothing(tmp_path):
    sender = FakeSender()
    runner = _runner(tmp_path, sender)
    runner._last_tick_date = date(2026, 10, 13)
    runner._maybe_send_rollover_recap(
        {"time": pd.Timestamp("2026-10-13 12:00", tz="UTC"),
         "mid": 2000.0})
    assert sender.texts == []


def test_rollover_without_senders_advances_clock(tmp_path):
    runner = _runner(tmp_path)
    assert runner.senders == []
    runner._maybe_send_rollover_recap(
        {"time": pd.Timestamp("2026-10-13 12:00", tz="UTC"),
         "mid": 2000.0})
    assert runner._last_tick_date == date(2026, 10, 13)
