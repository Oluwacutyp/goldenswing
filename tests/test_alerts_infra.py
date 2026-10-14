"""Phase-11 alert infrastructure: position notifier, daily summary,
forecaster quality dial, risk-reset script."""
import json

import pandas as pd
import pytest

from goldenswing.alerts import (PositionNotifier, TelegramSender,
                                build_daily_summary)
from goldenswing.alerts.model import SendResult
from goldenswing.confluence import (ConfluenceConfig, forecast_agreement,
                                    forecast_bonuses)


# ------------------------------------------------------- position notifier
class _CapSender:
    def __init__(self):
        self.sent = []

    def send_text(self, text, *, silent=False, reply_markup=None):
        self.sent.append((text, silent))
        return SendResult("telegram", True, "live", "ok")


def test_position_notifier_formats_without_sender():
    n = PositionNotifier(None)
    txt = n.on_close(ticket=7, direction=-1, exit_price=2900.0,
                     reason="stop", pnl_money=-55.0, r_net=-1.0,
                     hold_bars=30)
    assert "CLOSED" in txt and "7" in txt and "-55.00" in txt
    assert "-1.00R" in txt and "stopped" in txt


def test_position_notifier_sends_quiet_except_closes():
    cap = _CapSender()
    n = PositionNotifier(cap)
    n.on_partial(ticket=1, direction=1, fill_price=2910.0,
                 volume_closed=0.05, pnl_money=120.0)
    n.on_breakeven(ticket=1, direction=1)
    n.on_trail(ticket=1, direction=1, new_stop=2912.0, locked_r=0.6)
    n.on_close(ticket=1, direction=1, exit_price=2930.0, reason="trail",
               pnl_money=200.0, r_net=1.8, hold_bars=40)
    kinds = [t for t, _ in cap.sent]
    assert any("PARTIAL" in t for t in kinds)
    assert any("BREAKEVEN" in t for t in kinds)
    assert any("TRAIL" in t for t in kinds)
    # routine chatter is silent; closes buzz
    silents = [s for _, s in cap.sent]
    assert silents == [True, True, True, False]
    assert any("trailed stop" in t for t in kinds)


def test_position_notifier_fill_and_broker_close():
    n = PositionNotifier(None)
    txt = n.on_fill(ticket=3, alert_id="XAU9L", direction=1,
                    fill_price=2912.0, volume=0.1, stop=2901.0)
    assert "FILLED" in txt and "2,912.0" in txt
    txt = n.on_broker_close(ticket=3, alert_id="XAU9L")
    assert "POSITION GONE" in txt


# ---------------------------------------------------------- daily summary
def _rec(rtype, day, **kw):
    rec = {"type": rtype, "ts": f"{day}T12:00:00+00:00"}
    rec.update(kw)
    return rec


def test_daily_summary_counts_the_day():
    recs = [
        _rec("open", "2026-10-11", alert_id="a"),
        _rec("close", "2026-10-11", pnl_money=100.0, r_net=1.0),
        _rec("close", "2026-10-11", pnl_money=-40.0, r_net=-0.5),
        _rec("close", "2026-10-10", pnl_money=999.0, r_net=9.0),  # yesterday
        _rec("skip", "2026-10-11"),
    ]
    txt = build_daily_summary(day="2026-10-11", records=recs,
                              balance=10600.0, drawdown_pct=0.02,
                              open_positions=1, alerts_sent=5)
    assert "DAILY SUMMARY — 2026-10-11" in txt
    assert "+60.00" in txt and "+0.50R" in txt
    assert "Trades closed: 2 (1W / 1L)" in txt
    assert "Opened: 1" in txt and "alerts sent: 5" in txt
    assert "Open positions: 1" in txt
    assert "Balance: 10,600.00" in txt


def test_daily_summary_halt_banner():
    txt = build_daily_summary(day="2026-10-11", records=[],
                              halted=True, halt_reason="6 consecutive losses")
    assert "RISK HALT ACTIVE" in txt
    assert "6 consecutive losses" in txt


# ------------------------------------------------------- forecaster dial
class _FakeHorizon:
    def __init__(self, tag, ranks):
        self.tag = tag
        self._ranks = ranks  # {(direction, t): rank}

    def rank(self, direction, t):
        return self._ranks.get((direction, t), float("nan"))


class _FakeTable:
    def __init__(self, horizons):
        self.scores = horizons


def _table(up24=95.0, up72=93.0):
    return _FakeTable({
        24: _FakeHorizon("1d", {(1, 100): up24, (-1, 100): 5.0}),
        72: _FakeHorizon("3d", {(1, 100): up72, (-1, 100): 5.0}),
    })


def test_forecast_agreement_counts_top_decile_horizons():
    n, notes = forecast_agreement(100, 1, _table(), top_pct=10.0)
    assert n == 2 and len(notes) == 2
    n, _ = forecast_agreement(100, 1, _table(up72=50.0), top_pct=10.0)
    assert n == 1
    n, _ = forecast_agreement(100, -1, _table(), top_pct=10.0)
    assert n == 0  # wrong direction: no credit
    n, _ = forecast_agreement(100, 1, None)
    assert n == 0  # no table: silent, never a veto


def test_forecast_bonuses_scale_with_agreement():
    cfg = ConfluenceConfig()
    b, notes = forecast_bonuses(100, 1, _table(), cfg)
    assert b == cfg.bonus_forecast_agree == 2.0
    assert any("1d+3d agreement" in x for x in notes)
    b, notes = forecast_bonuses(100, 1, _table(up72=50.0), cfg)
    assert b == cfg.bonus_forecast_single == 1.0
    b, notes = forecast_bonuses(100, 1, _table(up24=10.0, up72=10.0), cfg)
    assert b == 0.0 and notes == []
    b, notes = forecast_bonuses(100, 1, None, cfg)
    assert b == 0.0 and notes == []


def test_forecast_dial_flows_into_live_score():
    from goldenswing.confluence import ConfluenceEngine
    from goldenswing.playbooks.base import Candidate, EngineContext
    from goldenswing.golden.detector import GoldenSignal
    sig = GoldenSignal(
        index=100, timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        direction=1, score=70.0, kind="sweep_reversal", factors=[],
        entry_zone=(1.0, 2.0), entry_ref=1.5, invalidation=1.0,
        targets=[2.5], atr=1.0)
    df = pd.DataFrame({"close": [1.0] * 200})
    ctx = EngineContext(df=df, states=[], years=0.0,
                        forecast_table=_table())
    eng = ConfluenceEngine(playbooks=[], risk_guards=None, journal=None)
    bonus, notes = eng.score_live_signal(_StubPB(), sig, ctx)
    assert bonus == 72.0  # 70 base + 2 agreement
    assert any("1d+3d agreement" in x for x in notes)


class _StubPB:
    name = "stub"
    hunting_ground = type("HG", (), {"bonuses": {}})()


def test_handler_applies_forecast_size_mult():
    from goldenswing.alerts.model import build_alert
    from goldenswing.execution.callbacks import ExecutionCallbackHandler
    from goldenswing.execution.paper import PaperExecutor
    from goldenswing.execution.risk import RiskConfig, RiskGuards
    from goldenswing.execution.learning import TradeJournal
    from goldenswing.golden.detector import GoldenSignal
    import tempfile
    tmp = tempfile.mkdtemp()
    sig = GoldenSignal(
        index=1, timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        direction=1, score=70.0, kind="sweep_reversal", factors=[],
        entry_zone=(2909.0, 2913.0), entry_ref=2912.0,
        invalidation=2901.0, targets=[2925.0], atr=10.0)
    alert = build_alert(sig, chart_png=b"png")
    paper = PaperExecutor.__new__(PaperExecutor)  # sizing only; no orders
    h = ExecutionCallbackHandler(
        executor=paper, paper_executor=paper,
        guards=RiskGuards(f"{tmp}/risk.json", RiskConfig()),
        journal=TradeJournal(f"{tmp}/j.jsonl"),
        spec=__import__("goldenswing.execution.risk",
                        fromlist=["gold_spec"]).gold_spec(),
        risk_pct=0.01)
    h.register_context(alert.alert_id, forecast_size_mult=1.25)
    assert h._context_for(alert)["forecast_size_mult"] == 1.25
    h.register_context(alert.alert_id)  # default
    assert h._context_for(alert)["forecast_size_mult"] == 1.0


def test_risk_reset_requires_reason_and_journals(tmp_path):
    import subprocess
    state = tmp_path / "risk.json"
    journal = tmp_path / "j.jsonl"
    state.write_text(json.dumps({"halted": True, "halt_reason": "test halt"}))
    r = subprocess.run(
        ["python3", "scripts/risk_reset.py", "--reason", "tail event",
         "--state", str(state), "--journal", str(journal)],
        cwd="/home/hatch/workspace/goldenswing", capture_output=True,
        text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(state.read_text())["halted"] is False
    assert "tail event" in journal.read_text()
    # second run: nothing latched -> no-op
    r = subprocess.run(
        ["python3", "scripts/risk_reset.py", "--reason", "x",
         "--state", str(state), "--journal", str(journal)],
        cwd="/home/hatch/workspace/goldenswing", capture_output=True,
        text=True)
    assert r.returncode == 0 and "nothing to reset" in r.stdout
