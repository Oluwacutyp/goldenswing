"""Phase 8 — lot sizer (never rounds up) + persisted risk guards."""
from datetime import date, timedelta

import pytest

from goldenswing.execution.risk import (RiskConfig, RiskGuards, gold_spec,
                                        size_position)

SPEC = gold_spec()   # tick 0.01 @ $1.00 -> $100 per $1.00 move per lot


# ------------------------------------------------------------------ sizer
def test_sizer_exact():
    r = size_position(balance=10_000, risk_pct=0.01, entry=2000.0,
                      stop=1990.0, spec=SPEC)
    assert r.ok and r.lots == pytest.approx(0.10)
    assert r.risk_money == pytest.approx(100.0)


def test_sizer_never_rounds_up():
    # budget $100, per-lot risk $950 -> raw 0.10526 lots; rounding UP
    # would give 0.11 and risk $104.50.  The sizer must floor to 0.10.
    r = size_position(balance=10_000, risk_pct=0.01, entry=2000.0,
                      stop=1990.5, spec=SPEC)
    assert r.lots == pytest.approx(0.10)
    assert r.risk_money <= 100.0


def test_sizer_below_broker_minimum_is_zero_with_reason():
    r = size_position(balance=100.0, risk_pct=0.01, entry=2000.0,
                      stop=1990.0, spec=SPEC)
    assert r.lots == 0.0 and not r.ok
    assert "minimum" in r.reason


def test_sizer_zero_stop_distance_refused():
    r = size_position(balance=10_000, risk_pct=0.01, entry=2000.0,
                      stop=2000.0, spec=SPEC)
    assert r.lots == 0.0 and "zero" in r.reason


def test_sizer_caps_at_volume_max():
    r = size_position(balance=10_000_000, risk_pct=0.01, entry=2000.0,
                      stop=1999.0, spec=SPEC)
    assert r.lots == pytest.approx(SPEC.volume_max)


# ----------------------------------------------------------------- guards
def _guards(tmp_path, today=date(2026, 10, 10), **cfg):
    return RiskGuards(tmp_path / "risk.json", RiskConfig(**cfg),
                      kill_switch_path=tmp_path / "KILL",
                      today_fn=lambda: today)


def _check(g, **kw):
    base = dict(balance=10_000.0, open_risk_money=0.0, new_risk_money=100.0,
                open_count=0, unstopped_positions=0)
    base.update(kw)
    return g.check_new_trade(**base)


def test_guard_allows_clean_trade(tmp_path):
    assert _check(_guards(tmp_path)).allowed


def test_guard_exposure_cap(tmp_path):
    g = _guards(tmp_path)   # cap 3% of 10k = 300
    assert not _check(g, open_risk_money=250.0, new_risk_money=100.0).allowed
    assert _check(g, open_risk_money=150.0, new_risk_money=100.0).allowed


def test_guard_max_positions_and_unstopped(tmp_path):
    g = _guards(tmp_path)
    assert not _check(g, open_count=3).allowed
    assert not _check(g, unstopped_positions=1).allowed


def test_daily_loss_limit_revoked_by_owner_ruling(tmp_path):
    # PREMISE CORRECTION (owner ruling 2026-10-23): the daily
    # loss cap this test pinned was REMOVED — daily/weekly caps
    # are gone as halt constraints; max drawdown is the sole
    # account-level hard halt.  The field itself no longer
    # exists, and a day loss past the old cap books as realised
    # P/L without halting or refusing anything.
    with pytest.raises(TypeError):
        RiskConfig(daily_loss_pct=0.02)
    g = _guards(tmp_path)
    g.record_outcome(-250.0, 9_750.0)   # past the old $200 cap
    assert not g.halted
    assert g.state["day_realized"] == pytest.approx(-250.0)
    assert _check(g, balance=9_750.0).allowed
    # a fresh instance on the same state file: still no halt.
    g2 = _guards(tmp_path)
    assert not g2.halted
    assert _check(g2, balance=9_750.0).allowed


def test_consecutive_losses_halt(tmp_path):
    g = _guards(tmp_path, max_consecutive_losses=3)
    for _ in range(3):
        g.record_outcome(-10.0, 10_000.0)
    assert g.halted and "consecutive" in g.state["halt_reason"]


def test_kill_switch_blocks(tmp_path):
    g = _guards(tmp_path)
    assert _check(g).allowed
    (tmp_path / "KILL").write_text("stop")
    d = _check(g)
    assert not d.allowed and "kill switch" in d.reason
    (tmp_path / "KILL").unlink()
    assert _check(g).allowed


def test_day_rollover_resets_daily_totals(tmp_path):
    g = _guards(tmp_path, today=date(2026, 10, 10))
    g.record_outcome(-50.0, 9_950.0)
    assert g.state["day_realized"] == pytest.approx(-50.0)
    nxt = _guards(tmp_path, today=date(2026, 10, 10) + timedelta(days=1))
    nxt.note_balance(9_950.0)
    assert nxt.state["day"] == "2026-10-11"
    assert nxt.state["day_realized"] == 0.0
    assert nxt.state["day_start_balance"] == pytest.approx(9_950.0)


def test_weekly_losses_book_but_never_refuse(tmp_path):
    # PREMISE CORRECTION (owner ruling 2026-10-23): the weekly
    # loss room this test pinned was REMOVED with the daily cap.
    # Week realised P/L still books (reporting), but no weekly
    # limit refuses a trade anymore — only max drawdown halts,
    # and 4.5% is far under its 15%.
    with pytest.raises(TypeError):
        RiskConfig(weekly_loss_pct=0.05)
    balance = 10_000.0
    kw = dict(max_consecutive_losses=0)
    for day in range(5, 10):                          # Oct 5..9, one ISO week
        g = _guards(tmp_path, today=date(2026, 10, day), **kw)
        g.record_outcome(-90.0, balance - 90.0)
        assert not g.halted
        balance -= 90.0
    sat = _guards(tmp_path, today=date(2026, 10, 10), **kw)
    assert sat.state["week_realized"] == pytest.approx(-450.0)
    assert _check(sat, balance=balance, new_risk_money=100.0).allowed
    assert _check(sat, balance=balance, new_risk_money=40.0).allowed
