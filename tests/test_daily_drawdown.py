"""Daily drawdown halt — owner correction 2026-11-09.

The max-drawdown halt is measured from the CURRENT DAY's starting
balance, never from the all-time peak.  A drawdown halt belongs to
its day and clears when the day rolls; the consecutive-loss halt
keeps its latch-until-human-reset semantics.  The all-time peak is
still tracked, as display information only.
"""
import json
from datetime import date, timedelta

import pytest

from goldenswing.execution.risk import (RiskConfig, RiskGuards,
                                        drawdown_snapshot)

DAY1 = date(2026, 11, 9)          # a Monday
DAY2 = DAY1 + timedelta(days=1)


def _guards(tmp_path, today=DAY1, **cfg):
    return RiskGuards(tmp_path / "risk.json", RiskConfig(**cfg),
                      kill_switch_path=tmp_path / "KILL",
                      today_fn=lambda: today)


def _check(g, **kw):
    base = dict(balance=10_000.0, open_risk_money=0.0,
                new_risk_money=100.0, open_count=0,
                unstopped_positions=0)
    base.update(kw)
    return g.check_new_trade(**base)


def test_breach_latches_within_the_day(tmp_path):
    g = _guards(tmp_path)
    g.note_balance(10_000.0)                    # day start anchored
    g.record_outcome(-1_600.0, 8_400.0)         # -16% today
    assert g.halted
    assert "daily drawdown hit" in g.state["halt_reason"]
    assert "day start" in g.state["halt_reason"]
    assert g.state["halt_causes"] == ["drawdown"]
    d = _check(g, balance=8_400.0)
    assert not d.allowed and "latched" in d.reason


def test_partial_leg_breach_also_latches(tmp_path):
    g = _guards(tmp_path)
    g.note_balance(10_000.0)
    g.record_partial(-1_600.0, 8_400.0)
    assert g.halted and g.state["halt_causes"] == ["drawdown"]


def test_below_peak_but_not_below_day_start_does_not_halt(tmp_path):
    # Day 1 the account sees 20,000 (the all-time peak is set).
    g1 = _guards(tmp_path, today=DAY1)
    g1.note_balance(20_000.0)
    assert g1.state["peak_balance"] == 20_000.0
    # Day 2 starts at 17,500 — already 12.5% under the peak — and
    # slips to 17,000.  Peak-basis math: (20000-17000)/20000 = 15%
    # -> the OLD code halted.  Daily math: (17500-17000)/17500 =
    # 2.9% -> no halt, and the peak is still tracked for display.
    g2 = _guards(tmp_path, today=DAY2)
    g2.note_balance(17_500.0)
    g2.record_outcome(-500.0, 17_000.0)
    assert not g2.halted
    assert g2.state["day_start_balance"] == pytest.approx(17_500.0)
    assert g2.state["peak_balance"] == 20_000.0
    assert _check(g2, balance=17_000.0).allowed


def test_day_roll_clears_drawdown_halt_and_reanchors(tmp_path):
    g1 = _guards(tmp_path, today=DAY1)
    g1.note_balance(10_000.0)
    g1.record_outcome(-1_600.0, 8_400.0)
    assert g1.halted
    # Next day: the halt is gone before anything trades, the day
    # re-anchors on first sight, and trading is allowed.
    g2 = _guards(tmp_path, today=DAY2)
    assert not g2.halted
    assert g2.state["halt_reason"] == ""
    assert g2.state["halt_causes"] == []
    assert g2.state["day_start_balance"] is None
    g2.note_balance(8_400.0)
    assert g2.state["day_start_balance"] == pytest.approx(8_400.0)
    assert _check(g2, balance=8_400.0).allowed


def test_legacy_peak_era_halt_clears_at_first_roll(tmp_path):
    # The exact EC2 state shape that prompted the correction: a
    # halt latched by the OLD peak basis, in a state file written
    # before halt_causes existed.
    state = {
        "day": "2026-11-08", "day_start_balance": 85.35,
        "day_realized": -18.11,
        "week": "2026-W45", "week_start_balance": 85.35,
        "week_realized": -18.11,
        "peak_balance": 85.35, "consecutive_losses": 2,
        "halted": True,
        "halt_reason": "max drawdown hit (21.2% from peak 85.35 "
                       ">= 15%)",
        "trades_today": 3,
    }
    (tmp_path / "risk.json").write_text(json.dumps(state))
    g = _guards(tmp_path, today=DAY1)
    assert not g.halted
    assert g.state["halt_reason"] == ""


def test_consecutive_loss_halt_is_not_cleared_by_day_roll(tmp_path):
    g1 = _guards(tmp_path, today=DAY1, max_consecutive_losses=3)
    for _ in range(3):
        g1.record_outcome(-10.0, 10_000.0)
    assert g1.halted
    assert "consecutive_losses" in g1.state["halt_causes"]
    g2 = _guards(tmp_path, today=DAY2, max_consecutive_losses=3)
    assert g2.halted
    assert "consecutive" in g2.state["halt_reason"]
    # ...until a human clears it.
    g2.reset_halt()
    assert not g2.halted and g2.state["halt_causes"] == []


def test_mixed_halt_is_not_cleared_by_day_roll(tmp_path):
    # One outcome trips BOTH causes: a 20% daily drawdown that is
    # also the (first) consecutive loss.  The halt carries the
    # streak cause, so the roll must not clear it.
    g1 = _guards(tmp_path, today=DAY1, max_consecutive_losses=1)
    g1.note_balance(10_000.0)
    g1.record_outcome(-2_000.0, 8_000.0)
    assert g1.halted
    assert set(g1.state["halt_causes"]) == {"drawdown",
                                            "consecutive_losses"}
    g2 = _guards(tmp_path, today=DAY2, max_consecutive_losses=1)
    assert g2.halted


def test_restart_midday_keeps_the_day_anchor(tmp_path):
    g1 = _guards(tmp_path, today=DAY1)
    g1.note_balance(10_000.0)                   # anchor: 10,000
    g1.record_outcome(-300.0, 9_700.0)
    assert not g1.halted
    # Restart at the lower balance: the anchor must NOT re-base
    # to 9,700 — the day's start is a persisted fact.
    g2 = _guards(tmp_path, today=DAY1)
    g2.note_balance(9_700.0)
    assert g2.state["day_start_balance"] == pytest.approx(10_000.0)
    # A further slide to 8,400 is -16% from the DAY's start: halt.
    g2.record_outcome(-1_300.0, 8_400.0)
    assert g2.halted


def test_snapshot_arithmetic_matches_the_guard(tmp_path):
    # drawdown_snapshot (the /health display) and _drawdown_reason
    # (the enforcement) must agree on every pair, boundary included.
    cases = [
        (10_000.0, 10_000.0, False),
        (10_000.0, 9_250.0, False),
        (10_000.0, 8_501.0, False),
        (10_000.0, 8_500.0, True),      # exactly at the 15% limit
        (10_000.0, 8_499.0, True),
        (80.0, 67.24, True),            # 15.95% down today
        (85.35, 84.0, False),           # the owner's figures, daily
    ]
    for i, (start, balance, breached) in enumerate(cases):
        sub = tmp_path / f"case{i}"
        sub.mkdir()
        g = RiskGuards(sub / "risk.json", RiskConfig(),
                       today_fn=lambda: DAY1)
        g.note_balance(start)
        snap = drawdown_snapshot(day_start_balance=start,
                                 balance=balance, limit=0.15)
        assert snap["breached"] is breached
        assert (g._drawdown_reason(balance) != "") is breached
        assert snap["drawdown_pct"] == pytest.approx(
            (start - balance) / start)


def test_snapshot_no_day_start_matches_guard_silence(tmp_path):
    g = _guards(tmp_path)
    assert g._drawdown_reason(9_000.0) == ""
    snap = drawdown_snapshot(day_start_balance=None, balance=9_000.0,
                             limit=0.15)
    assert snap["drawdown_pct"] is None and snap["breached"] is False
