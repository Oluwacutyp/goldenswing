"""Phase 8 — trade journal, performance book, retraining export,
and the alert-callback flow end to end (tap -> risk -> order)."""
import pytest

from goldenswing.alerts import CallbackRegistry, build_alert
from goldenswing.alerts.callbacks import AlertAction, make_callback_data
from goldenswing.execution.callbacks import ExecutionCallbackHandler
from goldenswing.execution.learning import (PerformanceBook, TradeJournal,
                                            export_training_data)
from goldenswing.execution.paper import PaperExecutor
from goldenswing.execution.risk import RiskConfig, RiskGuards, gold_spec
from goldenswing.golden.detector import GoldenSignal


# ----------------------------------------------------------------- journal
def test_journal_round_trip_and_aggregation(tmp_path):
    j = TradeJournal(tmp_path / "journal.jsonl")
    for i, (phase, r) in enumerate((("accumulation", 1.5),
                                    ("accumulation", -1.0),
                                    ("expansion_up", 2.0))):
        j.log_open(alert_id=f"A{i}", instrument="XAUUSD", mode="swing",
                   playbook="trend_continuation", phase=phase, direction=1,
                   score=70.0, source="execute", ticket=100 + i,
                   entry=2000.0, stop=1990.0, volume=0.1, risk_money=100.0,
                   factors=["f"], forecaster_rank=88.0, risk_atr=1.0)
        j.log_close(alert_id=f"A{i}", exit_price=2010.0, reason="trail",
                    pnl_money=r * 100.0, r_net=r, hold_bars=20)
    j.log_skip(alert_id="A9", instrument="XAUUSD", mode="swing",
               direction=-1, score=65.0, phase="consolidation")
    book = PerformanceBook(j.records())
    agg = book.aggregate()
    assert agg["overall"]["trades"] == 3
    assert agg["overall"]["total_r"] == pytest.approx(2.5)
    acc = agg["by_phase_playbook_mode"]["accumulation × trend_continuation × swing"]
    assert acc["trades"] == 2 and acc["win_rate"] == pytest.approx(50.0)
    assert agg["skips"] == 1


def test_training_export_carries_features_and_labels(tmp_path):
    j = TradeJournal(tmp_path / "journal.jsonl")
    j.log_open(alert_id="A0", instrument="XAUUSD", mode="swing",
               playbook="sweep_reversal", phase="manipulation", direction=-1,
               score=83.0, source="paper", ticket=1, entry=2000.0,
               stop=2010.0, volume=0.1, risk_money=100.0,
               forecaster_prob=0.62, forecaster_rank=91.0, risk_atr=1.2)
    j.log_close(alert_id="A0", exit_price=1980.0, reason="target",
                pnl_money=180.0, r_net=1.8, hold_bars=30)
    out = tmp_path / "train.jsonl"
    assert export_training_data(j.records(), out) == 1
    import json
    row = json.loads(out.read_text().strip())
    assert row["label"] == 1 and row["r_net"] == pytest.approx(1.8)
    assert row["features"]["phase"] == "manipulation"
    assert row["features"]["forecaster_prob"] == pytest.approx(0.62)


# ------------------------------------------------------------- callback flow
def _signal(**kw):
    base = dict(index=5, timestamp="2026-10-10T05:00:00Z", direction=1,
                score=80.0, kind="continuation", factors=["trend aligned"],
                entry_zone=(1995.0, 2005.0), entry_ref=2000.0,
                invalidation=1990.0, targets=[2030.0], atr=10.0)
    base.update(kw)
    return GoldenSignal(**base)


def _handler(tmp_path, paper_live=None, paper_book=None):
    live = paper_live or PaperExecutor(balance=10_000.0)
    book = paper_book or PaperExecutor(balance=10_000.0)
    guards = RiskGuards(tmp_path / "risk.json", RiskConfig(),
                        kill_switch_path=tmp_path / "KILL")
    journal = TradeJournal(tmp_path / "journal.jsonl")
    handler = ExecutionCallbackHandler(
        executor=live, paper_executor=book, guards=guards,
        spec=gold_spec(), journal=journal, risk_pct=0.01)
    return handler, live, book, journal, guards


def test_execute_tap_flows_through_risk_to_paper_fill(tmp_path):
    handler, live, _book, journal, _g = _handler(tmp_path)
    alert = build_alert(_signal(), instrument="XAUUSD", mode="swing")
    handler.register_context(alert.alert_id, phase="expansion_up",
                             playbook="trend_continuation",
                             forecaster_rank=91.0)
    registry = CallbackRegistry()
    registry.register(handler)
    result = registry.dispatch(
        make_callback_data(AlertAction.EXECUTE, alert.alert_id),
        {alert.alert_id: alert})
    assert result.handled
    assert "0.10 lots" in result.message
    live.advance({"index": 6, "time": None, "open": 2000.0, "high": 2004.0,
                  "low": 1996.0, "close": 2001.0})
    assert len(live.positions()) == 1
    opens = [r for r in journal.records() if r["type"] == "open"]
    assert len(opens) == 1
    assert opens[0]["source"] == "execute"
    assert opens[0]["phase"] == "expansion_up"
    assert opens[0]["forecaster_rank"] == pytest.approx(91.0)


def test_paper_tap_uses_paper_book_and_skip_logs(tmp_path):
    handler, live, book, journal, _g = _handler(tmp_path)
    alert = build_alert(_signal(), instrument="XAUUSD", mode="swing")
    msg = handler.handle(AlertAction.PAPER, alert)
    assert "PAPER" in msg
    book.advance({"index": 6, "time": None, "open": 2000.0, "high": 2004.0,
                  "low": 1996.0, "close": 2001.0})
    assert len(book.positions()) == 1 and len(live.positions()) == 0
    handler.handle(AlertAction.SKIP, alert)
    types = [r["type"] for r in journal.records()]
    assert types == ["open", "skip"]


def test_guard_halt_refuses_execute_and_journals_the_refusal(tmp_path):
    handler, live, _book, journal, guards = _handler(tmp_path)
    guards.record_outcome(-350.0, 9_650.0)       # trips the 3% daily halt
    assert guards.halted
    alert = build_alert(_signal(), instrument="XAUUSD", mode="swing")
    msg = handler.handle(AlertAction.EXECUTE, alert)
    assert msg.startswith("⛔")
    assert live.positions() == []
    notes = [r for r in journal.records() if r["type"] == "note"]
    assert notes and "refused" in notes[0]["text"]


def test_unsizable_alert_refused_with_reason(tmp_path):
    handler, live, _book, _j, _g = _handler(tmp_path)
    alert = build_alert(_signal(invalidation=2000.0),  # zero stop distance
                        instrument="XAUUSD", mode="swing")
    msg = handler.handle(AlertAction.EXECUTE, alert)
    assert msg.startswith("⛔") and "stop distance" in msg
