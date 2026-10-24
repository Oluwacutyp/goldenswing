"""Runtime Truth pass pins (owner directive 2026-10-23).

Item 1 — feed/stale reconciliation: an EXPECTED market closure
is not a feed failure and never journals a tick_stale breach;
an open-market stale tick is one, and the /status feed line
agrees with the Health line by construction.

Item 2 — every candidate outcome explained: the funnel carries
reason-code breakdowns for EVERY terminal bucket (ineligible
included), counts reconcile, and the runner's live counters +
/status cross-reference tell the same story as the blocked legs.

Item 3 — D1 freshness proof: an EXPIRED historical candidate
(the qm_d1 bar 2025-07-30 case) renders its provenance — source
bar, touch, age vs validity window, eligibility reason — and is
labelled stale history, never a current setup.

Item 4 — sizing correction (owner ruling 2026-10-23): fixed
lots are gone from the production path, dynamic risk sizing is
the sole path (abstention when the smallest permitted order
exceeds the budget), daily/weekly loss caps no longer halt, and
max drawdown is the sole account-level hard halt.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from goldenswing.alerts.commands import fmt_feed_line
from goldenswing.execution.assessment_render import (
    freshness_lines, render_assessment)
from goldenswing.execution.funnel import summarize_evaluations
from goldenswing.execution.mt5_config import (
    LiveConfigStore, _clean, parse_set)
from goldenswing.execution.mt5_runner import MT5RunnerConfig
from goldenswing.execution.risk import RiskConfig, RiskGuards, gold_spec
from goldenswing.live.health import HealthMonitor
from goldenswing.live.sizing import size_for_risk

from test_p3_trader import _runner

FRI_CLOSE = datetime(2026, 10, 23, 21, 30, tzinfo=timezone.utc)


# ============================================================ item 1

def test_closed_market_old_tick_is_not_a_breach():
    """The owner's EC2 case: market closed, last tick 2501s old.
    Expected closure is NOT a feed failure."""
    h = HealthMonitor()
    h.note_tick(FRI_CLOSE - timedelta(seconds=2501))
    h.note_market_open(False)
    assert h.tick_age_sec(FRI_CLOSE) == pytest.approx(2501.0)
    assert h.tick_state(FRI_CLOSE) == "closed_expected"
    assert "tick_stale" not in h.breaches(FRI_CLOSE)
    text = h.describe(FRI_CLOSE)
    assert "market closed" in text
    assert "BREACH" not in text


def test_open_market_old_tick_is_a_breach():
    h = HealthMonitor()
    h.note_tick(FRI_CLOSE - timedelta(seconds=2501))
    h.note_market_open(True)
    assert h.tick_state(FRI_CLOSE) == "stale"
    assert "tick_stale" in h.breaches(FRI_CLOSE)
    assert "BREACH: tick_stale" in h.describe(FRI_CLOSE)


def test_unknown_market_state_keeps_the_fail_loud_breach():
    """No market reading yet (the state every pre-pass test and
    pre-first-refresh moment lives in): stale still breaches —
    the reconciliation never silences a possibly-real failure."""
    h = HealthMonitor()
    h.note_tick(FRI_CLOSE - timedelta(seconds=2501))
    assert h.tick_state(FRI_CLOSE) == "stale"
    assert "tick_stale" in h.breaches(FRI_CLOSE)


def test_no_ticks_closed_vs_open():
    h = HealthMonitor()
    h.note_market_open(False)
    assert h.tick_state(FRI_CLOSE) == "closed_expected"
    assert "tick_stale" not in h.breaches(FRI_CLOSE)
    h.note_market_open(True)
    assert h.tick_state(FRI_CLOSE) == "no_ticks"
    assert "tick_stale" in h.breaches(FRI_CLOSE)


def test_feed_line_agrees_with_health_on_closure():
    line = fmt_feed_line({"ok": True, "reason": None},
                         "2026-10-23 20:48", 4054.3,
                         market_open=False,
                         tick_state="closed_expected",
                         tick_age_s=2501.0)
    assert "OK ✅" in line
    assert "market CLOSED" in line
    assert "STALE" not in line


def test_feed_line_open_market_stale_tick_is_stale():
    line = fmt_feed_line({"ok": True, "reason": None},
                         "2026-10-23 20:48", 4054.3,
                         market_open=True, tick_state="stale",
                         tick_age_s=2501.0)
    assert "STALE ⚠️" in line
    assert "2501s ago" in line


def test_feed_line_legacy_callers_unchanged():
    """OANDA-loop callers pass no truth kwargs: the pre-pass
    rendering is byte-identical."""
    assert fmt_feed_line({"ok": True, "reason": None},
                         "2026-10-23 20:48", 4054.3) == \
        "feed: OK ✅ (MT5 terminal connected) · last bar " \
        "`2026-10-23 20:48` · last price `4,054.3`"
    assert "STALE ⚠️" in fmt_feed_line(
        {"ok": False, "reason": "terminal disconnected"},
        None, None)


def test_runner_journals_no_breach_on_closed_market(tmp_path):
    """Runner-level before/after: the same stale tick journals
    ZERO tick_stale breaches while the market is closed and ONE
    when it is open (new-breach-only journaling)."""
    runner = _runner(tmp_path)
    runner._health.note_tick(
        datetime.now(timezone.utc) - timedelta(seconds=2501))
    runner.feed.market_open = lambda: (False, "weekend close")
    runner._journal_new_breaches()
    journal_path = tmp_path / "mt5_journal.jsonl"
    journal = (journal_path.read_text() if journal_path.exists()
               else "")
    assert "tick_stale" not in journal
    assert runner.feed_truth()["tick_state"] == "closed_expected"
    assert "market closed" in runner.health_line()

    runner.feed.market_open = lambda: (True, "session open")
    runner._health_market_ts = 0.0      # expire the 30s cache
    runner._journal_new_breaches()
    journal = (tmp_path / "mt5_journal.jsonl").read_text()
    assert journal.count("health breach: tick_stale") == 1
    assert runner.feed_truth()["tick_state"] == "stale"


# ============================================================ item 2

def _ineligible_record(playbook: str, code: str,
                       analytical: str = "not_qualified") -> dict:
    return {
        "type": "evaluation", "kind": "playbook_eval",
        "playbook": playbook, "outcome": "candidate_emitted",
        "evaluation_clock": "close", "reason_code": code,
        "decision": {
            "result": "candidate_ineligible",
            "execution_eligible": False,
            "order_result": "not_attempted",
            "reason_code": code,
            "analytical_verdict": {"verdict": analytical,
                                   "reason_code": "strategy.x"},
            "execution": {"eligible": False, "reason_code": code}}}


def _ec2_fixture_records() -> list[dict]:
    """The owner's EC2 shape: 31 qm_m5 + 10 qm_d1 candidates,
    ALL ineligible, blocked legs empty."""
    recs = [_ineligible_record("qm_m5", "path.warmup_ineligible")
            for _ in range(31)]
    recs += [_ineligible_record("qm_d1", "gate.data")
             for _ in range(10)]
    return recs


def test_funnel_reconciles_the_ec2_31_10_shape():
    s = summarize_evaluations(_ec2_fixture_records())
    m5 = s["by_playbook"]["qm_m5"]
    d1 = s["by_playbook"]["qm_d1"]
    assert m5["candidates"] == 31
    assert m5["terminal"]["ineligible"] == 31
    assert m5["terminal_by_code"]["ineligible"] == \
        {"path.warmup_ineligible": 31}
    assert d1["candidates"] == 10
    assert d1["terminal_by_code"]["ineligible"] == {"gate.data": 10}
    totals = s["totals"]
    assert totals["candidates"] == 41
    assert totals["terminal"]["ineligible"] == 41
    # Reconciliation invariant, every playbook + the run:
    for slot in s["by_playbook"].values():
        assert slot["candidates"] == sum(slot["terminal"].values())
        for bucket, codes in slot["terminal_by_code"].items():
            assert sum(codes.values()) == slot["terminal"][bucket]


def test_funnel_keeps_analytical_qualified_separate():
    recs = _ec2_fixture_records()
    recs.append(_ineligible_record("qm_m5", "risk.friction",
                                   analytical="qualified"))
    recs[-1]["decision"]["result"] = "refused"
    recs[-1]["decision"]["execution_eligible"] = True
    recs[-1]["decision"]["order_result"] = "refused_in_order_path"
    s = summarize_evaluations(recs)
    m5 = s["by_playbook"]["qm_m5"]
    assert m5["qualified"] == 1
    assert m5["terminal"]["refused"] == 1
    assert m5["terminal_by_code"]["refused"] == {"risk.friction": 1}
    assert s["totals"]["qualified"] == 1


def test_runner_funnel_line_names_ineligible_reasons(tmp_path):
    runner = _runner(tmp_path)
    for rec in _ec2_fixture_records():
        runner._emit_eval(playbook=rec["playbook"],
                          outcome="candidate_emitted",
                          evaluation_clock="close",
                          decision=rec["decision"],
                          reason_code=rec["reason_code"])
    line = runner.funnel_line()
    assert "ineligible reasons: path.warmup_ineligible ×31" in line
    assert "ineligible reasons: gate.data ×10" in line
    assert ("reconcile: candidates 41 = submitted 0 + "
            "risk-refused 0 + ineligible 41 + no-order 0 + "
            "unresolved 0") in line
    totals = runner.funnel_totals()
    assert totals["candidates"] == 41
    assert totals["ineligible"] == 41
    assert totals["ineligible_codes"] == {
        "path.warmup_ineligible": 31, "gate.data": 10}


# ============================================================ item 3

def _expired_d1_record() -> dict:
    """The EC2 /assess defect record: qm_d1, touch bar
    2025-07-30, verdict EXPIRED, provenance-bearing."""
    return {
        "type": "evaluation", "kind": "playbook_eval",
        "playbook": "qm_d1", "outcome": "candidate_emitted",
        "last_bar_ts": "2025-07-30",
        "data_freshness_s": 86400,
        "assessment": {"verdict": "expired",
                       "reason_code": "path.warmup_ineligible",
                       "order_result": "not_attempted",
                       "score": None},
        "decision": {
            "result": "candidate_ineligible",
            "reason": "candidate source is not a live close "
                      "(warmup_scan)",
            "execution_eligible": False},
        "evidence": {"provenance": {
            "pattern_formation_ts": "2025-06-11",
            "pattern_touch_ts": "2025-07-30",
            "candidate_created_ts": "2025-07-30",
            "evaluated_bar_ts": "2026-10-23",
            "evaluation_ts": "2026-10-23T20:00:00+00:00",
            "source": "warmup_scan",
            "touch_age_bars": 296,
            "validity_window_bars": 30,
            "execution_eligible": False,
            "eligibility_code": "path.warmup_ineligible"}}}


def test_expired_d1_render_shows_freshness_and_stale_label():
    text = render_assessment(_expired_d1_record())
    assert "EXPIRED — historical record" in text
    assert "source bar (evaluated): 2026-10-23" in text
    assert "pattern touch: 2025-07-30" in text
    assert "source warmup_scan" in text
    assert "touch 296 bar(s)" in text
    assert "validity window 30 bar(s)" in text
    assert "execution eligibility: REFUSED " \
           "(path.warmup_ineligible)" in text


def test_freshness_lines_empty_without_provenance():
    assert freshness_lines({"playbook": "x"}) == []
    assert freshness_lines(
        {"data_freshness_s": 12}) == [
            "Freshness & provenance:", "  data freshness: 12s"]


# ============================================================ item 4

def test_fixed_lots_gone_from_the_production_path(tmp_path):
    assert not hasattr(MT5RunnerConfig(), "fixed_lots")
    change, error = parse_set("fixed_lots", "0.01")
    assert change is None and error is not None
    store = LiveConfigStore(tmp_path / "mt5_config.json")
    cleaned = _clean({"fixed_lots": 0.01, "max_layers": 2})
    assert "fixed_lots" not in cleaned
    assert cleaned["max_layers"] == 2
    snap = store.snapshot_from(SimpleNamespace(
        max_layers=2, manipulation_filter=True,
        pending_expiry_bars=6, risk_pct=0.01))
    assert snap["sizing_mode"] == "risk"
    assert snap["risk_pct"] == 0.01
    assert "fixed_lots" not in snap


def test_dynamic_sizing_scales_with_equity_and_trades_minimum_small():
    """The production rule (k = 0.01 sizing coefficient of equity)
    at a $5.00 stop on the gold spec ($500 risk per 1.0 lot),
    owner ruling 2026-10-24: small accounts TRADE the broker
    minimum instead of abstaining — $24 -> 0.01 lots (min floor,
    estimated stop exposure $5), $1,000 -> 0.02 lots, $10,000 ->
    0.20 lots, floored down for raw >= minimum."""
    spec = gold_spec()
    small = size_for_risk(Decimal("24") * Decimal("0.01"), 5.0, spec)
    assert small.ok and small.verdict == "ok"
    assert small.lots == Decimal("0.01")
    assert small.min_override is True
    assert small.raw_lots == Decimal("0.00048")
    assert small.estimated_stop_exposure == Decimal("5")
    mid = size_for_risk(Decimal("1000") * Decimal("0.01"), 5.0, spec)
    assert mid.ok and mid.lots == Decimal("0.02")
    assert mid.min_override is False
    big = size_for_risk(Decimal("10000") * Decimal("0.01"), 5.0, spec)
    assert big.ok and big.lots == Decimal("0.20")
    # Floored DOWN, never up, for raw >= minimum: budget for
    # 0.029 lots -> 0.02.
    frac = size_for_risk(Decimal("1450") * Decimal("0.01"), 5.0,
                          spec)
    assert frac.lots == Decimal("0.02")


def test_daily_and_weekly_caps_no_longer_exist_or_halt(tmp_path):
    cfg = RiskConfig()
    assert not hasattr(cfg, "daily_loss_pct")
    assert not hasattr(cfg, "daily_loss_money")
    assert not hasattr(cfg, "weekly_loss_pct")
    assert not hasattr(cfg, "weekly_loss_money")
    # A 6% single-day realized loss (past the old 5% daily cap)
    # does NOT halt new trades while drawdown stays under 15%.
    g = RiskGuards(tmp_path / "risk.json", RiskConfig())
    g.note_balance(1000.0)
    g.record_outcome(pnl_money=-60.0, balance=940.0)
    assert not g.halted
    dec = g.check_new_trade(balance=940.0, open_risk_money=0.0,
                            new_risk_money=10.0, open_count=0)
    assert dec.allowed, dec.reason


def test_max_drawdown_is_the_account_level_halt(tmp_path):
    g = RiskGuards(tmp_path / "risk.json",
                   RiskConfig())   # max_drawdown_pct = 0.15
    g.note_balance(1000.0)         # the peak the DD is measured from
    g.record_outcome(pnl_money=-200.0, balance=800.0)
    assert g.halted
    assert "max drawdown" in g.state["halt_reason"]
    dec = g.check_new_trade(balance=800.0, open_risk_money=0.0,
                            new_risk_money=10.0, open_count=0)
    assert not dec.allowed
    assert "risk halt latched" in dec.reason
