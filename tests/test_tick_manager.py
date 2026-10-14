"""Tick-driven management: breakeven / trailing / partial / stop /
timeout fire on the crossing tick — never on the next poll or bar.

Setup used throughout: long, entry 2000, invalidation 1990,
ATR 10 -> buffered stop0 = 1990 - 0.25*10 = 1987.50, risk = 12.50.
Breakeven at +1.0R, trail triggers at +1.5R, trail distance 1.0R,
partial 50% at the first pool >= 1.0R away.
"""
import pytest

from goldenswing.backtest.harness import ExitConfig
from goldenswing.execution.executor import OrderRequest
from goldenswing.execution.tick_manager import (TickConfig, TickManager)


def _request(direction=1, targets=(2030.0,)):
    return OrderRequest(symbol="XAUUSD", direction=direction, volume=1.0,
                        stop_loss=1990.0 if direction == 1 else 2010.0,
                        entry_ref=2000.0, targets=list(targets),
                        atr=10.0, bar_index=5, alert_id="T1")


def _open(mgr, direction=1, **kw):
    req = _request(direction, **kw)
    return mgr.open(ticket=101, request=req, fill_price=2000.0,
                    entry_bar=6)


def test_planned_stop_is_buffered_stop():
    assert TickManager.planned_stop(_request()) == pytest.approx(1987.5)
    assert TickManager.planned_stop(_request(-1)) == pytest.approx(2012.5)


def test_breakeven_fires_on_crossing_tick_not_next_poll():
    mgr = TickManager()
    _open(mgr)
    # +0.8R: nothing.
    assert mgr.on_tick(101, 2010.0) == []
    # +1.0R exactly: breakeven fires on THIS tick.
    actions = mgr.on_tick(101, 2012.5)
    assert len(actions) == 1
    assert actions[0].kind == "modify_sl"
    assert actions[0].reason == "breakeven"
    assert actions[0].level == pytest.approx(2000.0)
    mgr.apply(101, actions[0])
    # Same price again: no repeat (breakeven is done).
    assert mgr.on_tick(101, 2012.5) == []


def test_trailing_follows_breakeven_on_later_ticks():
    mgr = TickManager()
    _open(mgr)
    be = mgr.on_tick(101, 2012.5)[0]
    mgr.apply(101, be)
    # +2.0R: trail level = 2000 + (2.0 - 1.0) * 12.5 = 2012.50.
    actions = mgr.on_tick(101, 2025.0)
    assert len(actions) == 1
    assert actions[0].reason == "trail"
    assert actions[0].level == pytest.approx(2012.5)
    mgr.apply(101, actions[0])
    # Price falls back: no SL loosening, no new action.
    assert mgr.on_tick(101, 2015.0) == []


def test_partial_fires_on_crossing_tick_and_halves_volume():
    mgr = TickManager()
    _open(mgr)
    # Breakeven fires first on the way up (+2.32R at 2029).
    be = mgr.on_tick(101, 2029.0)
    assert [a.reason for a in be] == ["breakeven"]
    mgr.apply(101, be[0])
    actions = mgr.on_tick(101, 2030.0)
    kinds = [a.kind for a in actions]
    assert "partial" in kinds
    partial = [a for a in actions if a.kind == "partial"][0]
    assert partial.volume == pytest.approx(0.5)
    mgr.apply(101, partial)
    assert mgr.position(101).volume == pytest.approx(0.5)
    assert mgr.position(101).partial_done is True
    # Partial fires once.
    assert not [a for a in mgr.on_tick(101, 2031.0)
                if a.kind == "partial"]


def test_stop_hit_is_alone_and_closes():
    mgr = TickManager()
    _open(mgr)
    actions = mgr.on_tick(101, 1987.5)
    assert [a.kind for a in actions] == ["stop_hit"]
    mgr.apply(101, actions[0])
    assert mgr.position(101).closed is True
    assert mgr.on_tick(101, 1990.0) == []


def test_failed_modify_is_retried_next_tick():
    mgr = TickManager()
    _open(mgr)
    actions = mgr.on_tick(101, 2012.5)
    assert len(actions) == 1  # decided, NOT applied (executor failed)
    actions2 = mgr.on_tick(101, 2013.0)
    assert len(actions2) == 1 and actions2[0].reason == "breakeven"


def test_short_direction_mirror():
    mgr = TickManager()
    req = _request(-1, targets=(1970.0,))
    mgr.open(ticket=202, request=req, fill_price=2000.0, entry_bar=6)
    assert mgr.on_tick(202, 1990.0) == []  # -0.8R
    actions = mgr.on_tick(202, 1987.5)  # -1.0R
    assert len(actions) == 1 and actions[0].reason == "breakeven"
    assert actions[0].level == pytest.approx(2000.0)
    mgr.apply(202, actions[0])
    actions = mgr.on_tick(202, 1970.0)  # -2.4R: partial + trail
    assert "partial" in [a.kind for a in actions]
    assert any(a.reason == "trail" for a in actions)


def test_timeout_on_bar_close_only():
    mgr = TickManager()
    _open(mgr)  # signal_bar = 5 (request.bar_index)
    assert mgr.on_bar_close(101, 5 + 10) == []
    horizon = ExitConfig().horizon
    actions = mgr.on_bar_close(101, 5 + horizon)
    assert [a.kind for a in actions] == ["timeout"]
    mgr.apply(101, actions[0])
    assert mgr.position(101).closed is True


def test_adopted_position_gets_breakeven_and_trail_but_no_partial():
    mgr = TickManager()
    mgr.adopt(ticket=303, symbol="XAUUSD", direction=1, volume=1.0,
              entry_price=2000.0, sl=1987.5)
    assert mgr.on_tick(303, 2012.5)[0].reason == "breakeven"
    # No partial even far past any pool — no target book.
    assert not [a for a in mgr.on_tick(303, 2100.0)
                if a.kind == "partial"]


def test_breakeven_does_not_loosen_an_already_tighter_stop():
    mgr = TickManager()
    mgr.adopt(ticket=404, symbol="XAUUSD", direction=1, volume=1.0,
              entry_price=2000.0, sl=2005.0)  # SL already above entry
    # Breakeven level (2000) does not improve 2005; trail at +2R
    # would be 2012.5 and DOES improve — only the trail fires.
    actions = mgr.on_tick(404, 2025.0)
    assert [a.reason for a in actions] == ["trail"]
