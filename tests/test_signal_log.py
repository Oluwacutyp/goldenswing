"""Structured signal persistence: every emission gets a journal record.

The MT5 runner already wrote prose notes for gated/accepted signals,
but the blocked-leg counters lived only in memory (a restart wiped
them) and nothing machine-readable answered "what was seen, what
blocked it, which playbook produced it".  ``signal_log`` writes one
``type == "signal"`` record per emission through the same journal,
with the same alert_id the open record will carry.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.execution.signal_log import (
    log_signal, signal_records, summarize_signals)
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from goldenswing.live.hierarchy import TimeframeStack
from mt5_fakes import FakeMT5


class _FakeExecutor:
    def __init__(self):
        self.orders: list[dict] = []
        self._ticket = 9000
        self._positions: list[Position] = []

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return list(self._positions)

    def place_order(self, request):
        self._ticket += 1
        self.orders.append({"ticket": self._ticket,
                            "direction": request.direction})
        self._positions.append(Position(
            ticket=self._ticket, symbol=request.symbol,
            direction=request.direction, volume=request.volume,
            entry_price=request.entry_ref, sl=request.stop_loss,
            tp=request.take_profit, magic=request.magic,
            alert_id=request.alert_id))
        return OrderResult(True, ticket=self._ticket,
                           price=request.entry_ref,
                           volume=request.volume)

    def cancel_pending(self, ticket):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        self._positions = [p for p in self._positions
                           if p.ticket != ticket]
        return OrderResult(True, ticket=ticket)

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)


class _StubPlaybook:
    def __init__(self, name, score, direction=1):
        self.name = name
        self._score = score
        self._direction = direction

    def live_signal(self, states, t, df=None):
        return [SimpleNamespace(
            direction=self._direction, score=self._score,
            entry_ref=2000.0, invalidation=1990.0, targets=[2030.0],
            atr=10.0, index=t, factors=["stub"])]


class _StubEngine:
    def __init__(self, gate=(True, "")):
        self._gate = gate

    def gate_live_signal(self, pb, sig, ctx, hierarchy_verdict=None,
                         spread_px=None):
        return self._gate

    def score_live_signal(self, pb, sig, ctx, hierarchy_verdict=None):
        return float(sig.score) + 5.0, ["stub-bonus"]


def _runner(tmp_path, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("fvg_midline",),
        # News pause OFF: the order path's news check reads the
        # REAL wall-clock calendar — these tests must not fail on
        # a real news day (e.g. PPI).  The pause itself is covered
        # by test_news_calendar.py.
        news_pause_enabled=False,
        **cfg_kw)
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    runner = MT5Runner(cfg, feed=feed, executor=_FakeExecutor())
    h1 = pd.DataFrame(
        {"open": [2000.0] * 30, "high": [2001.0] * 30,
         "low": [1999.0] * 30, "close": [2000.5] * 30,
         "volume": [100] * 30, "real_volume": [0] * 30,
         "spread": [30] * 30},
        index=pd.date_range("2026-01-01", periods=30, freq="h",
                            tz="UTC"))
    runner._stack = TimeframeStack.from_frames({"H1": h1})
    return runner, h1


def _verdict():
    return {"bias_d1": 1, "bias_h4": 1, "trigger": "tick-cross",
            "entry_clock": "M15", "amd_phase": "accumulation",
            "qm": {"status": "none"}, "volume": {"volume_ratio": 1.0},
            "entry": 2000.0, "zone": {"bottom": 1995.0, "top": 2005.0}}


def _drive(runner, h1, pb, engine, t=29, zone_d=None):
    from goldenswing.playbooks.base import EngineContext
    ctx = EngineContext(df=h1, states=[], years=0.0,
                        instrument="XAUUSD")
    runner._run_one_playbook(pb, sig_ctx=(ctx, engine), h1=h1, t=t,
                             verdict=_verdict(),
                             stats={"signals": 0, "gated": 0},
                             zone_d=zone_d)
    return [r for r in runner.journal.records()
            if r.get("type") == "signal"]


def test_below_floor_emission_gets_signal_record(tmp_path):
    runner, h1 = _runner(tmp_path)
    # Production reaches _run_one_playbook via _run_signal_path,
    # which supplies the verdict's armed band as zone context.
    # _drive passes the same armed-band dict explicitly so this
    # test exercises the ZONE_PATH floor.  (P3 scope G: with
    # zone_d=None the signal is floored on its open-space style
    # row instead — no unfloored path remains.)
    zone_d = {"bottom": 1995.0, "top": 2005.0, "tier": "MINOR",
              "lifecycle": "tested", "touches": 1, "status": "active",
              "score": 40.0, "armed_band": True}
    recs = _drive(runner, h1, _StubPlaybook("fvg_midline", 40.0),
                  _StubEngine(), zone_d=zone_d)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["decision"] == "below_floor"
    assert rec["playbook"] == "fvg_midline"
    assert rec["score"] == pytest.approx(40.0)
    # New reason contract: the flat min_score gate is gone; the
    # record's reason IS the asymmetry-floor verdict (the legacy
    # "min_score" prefix survives only in blocked-leg bookkeeping,
    # never in the signal record's reason).
    assert rec["reason"].startswith("asymmetry_floor")
    assert "min_score" not in rec["reason"]
    assert rec["score_breakdown"]["total"] < 50.0  # ZONE_PATH floor
    assert rec["signal_id"] == "MT5-fvg_midline-29"
    assert runner.executor.orders == []  # still dropped, just recorded


def test_gated_emission_gets_signal_record_with_reason(tmp_path):
    runner, h1 = _runner(tmp_path)
    recs = _drive(runner, h1, _StubPlaybook("fvg_midline", 80.0),
                  _StubEngine(gate=(False, "hierarchy: h4_zone")))
    assert len(recs) == 1
    assert recs[0]["decision"] == "gated"
    assert recs[0]["reason"] == "hierarchy: h4_zone"
    assert recs[0]["hierarchy"]["bias_h4"] == 1
    assert runner.executor.orders == []


class _FakeScore:
    """A passing Asymmetry Score stand-in.  P3 (scope G) floors
    open-space signals too — the dead-flat fixture frame cannot
    clear any floor with the REAL scorer, and these tests are
    about journal plumbing, not scoring (scoring has its own
    suite).  Structure+Efficiency = 50 >= the Path B 35 rule."""

    total = 80.0
    flipped = False

    def component(self, name):
        return {"structure": 30.0, "efficiency": 20.0,
                "location": 15.0, "pullback": 10.0,
                "session": 5.0, "macro": 0.0}.get(name, 0.0)

    def breakdown(self):
        return {"total": 80.0}


def test_accepted_emission_joins_to_open_via_alert_id(tmp_path):
    runner, h1 = _runner(tmp_path)
    runner._score_for = lambda **kw: _FakeScore()
    recs = _drive(runner, h1, _StubPlaybook("fvg_midline", 80.0),
                  _StubEngine())
    assert len(recs) == 1
    assert recs[0]["decision"] == "accepted"
    assert recs[0]["confluence"] == pytest.approx(85.0)
    assert len(runner.executor.orders) == 1
    opens = [r for r in runner.journal.records()
             if r.get("type") == "open"]
    assert len(opens) == 1
    assert opens[0]["alert_id"] == recs[0]["signal_id"]


def test_shadow_emission_recorded_not_ordered(tmp_path):
    runner, h1 = _runner(tmp_path)
    # Same P3 note as the accepted test: pin a passing score so
    # the flow reaches the accepted/shadow emission point.
    runner._score_for = lambda **kw: _FakeScore()
    recs = _drive(runner, h1, _StubPlaybook("qm_m5", 80.0),
                  _StubEngine())
    assert len(recs) == 1
    assert recs[0]["decision"] == "shadow"
    assert runner.executor.orders == []


def test_summarize_signals_counts_by_playbook_and_reason(tmp_path):
    journal = TradeJournal(tmp_path / "j.jsonl")
    log_signal(journal, signal_id="a", playbook="pb1", direction=1,
               score=40.0, decision="below_floor",
               reason="min_score: score 40 < 60")
    log_signal(journal, signal_id="b", playbook="pb1", direction=-1,
               score=80.0, decision="gated",
               reason="hierarchy: h4_zone")
    log_signal(journal, signal_id="c", playbook="pb2", direction=1,
               score=80.0, decision="accepted")
    summary = summarize_signals(journal.records())
    assert summary["by_playbook"]["pb1"] == {
        "below_floor": 1, "gated": 1, "total": 2}
    assert summary["by_playbook"]["pb2"]["accepted"] == 1
    assert summary["by_reason"]["hierarchy: h4_zone"] == 1
    assert summary["by_reason"]["min_score: score 40 < 60"] == 1


def test_log_signal_never_raises_on_broken_journal():
    class _Broken:
        def append(self, record):
            raise RuntimeError("disk gone")

    assert log_signal(_Broken(), signal_id="x", playbook="pb",
                      direction=1, score=1.0,
                      decision="gated") == {}
