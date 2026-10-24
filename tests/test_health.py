"""Execution health: tick freshness, order latency, slippage.

Bar freshness is data_health.py's job (a pipeline precondition).
This is the other half — execution health on the live path, observed
never gating.  Breaches journal once (new-breach-only), never once
per order.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from goldenswing.live.health import HealthMonitor

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_no_ticks_is_a_stale_breach():
    h = HealthMonitor()
    assert h.tick_age_sec(NOW) is None
    assert "tick_stale" in h.breaches(NOW)
    assert "no ticks yet" in h.describe(NOW)


def test_fresh_tick_is_not_stale_old_tick_is():
    h = HealthMonitor(tick_stale_sec=120.0)
    h.note_tick(NOW - timedelta(seconds=30))
    assert h.tick_age_sec(NOW) == pytest.approx(30.0)
    assert "tick_stale" not in h.breaches(NOW)
    h.note_tick(NOW - timedelta(seconds=300))
    assert "tick_stale" in h.breaches(NOW)


def test_latency_stats_and_breach():
    h = HealthMonitor(latency_warn_sec=2.0)
    h.note_order(0.5)
    h.note_order(1.5)
    assert h.latency_stats() == {"n": 2, "mean": pytest.approx(1.0),
                                 "max": pytest.approx(1.5)}
    assert "latency" not in h.breaches(NOW)
    h.note_order(3.0)
    assert "latency" in h.breaches(NOW)


def test_slippage_adverse_sign_and_breach():
    h = HealthMonitor(slippage_warn_atr=0.5)
    # Adverse +0.2 ATR (long filled above the aim), favourable -0.4.
    h.note_fill(2.0, 10.0)
    h.note_fill(-4.0, 10.0)
    stats = h.slippage_stats()
    assert stats["n"] == 2
    assert stats["mean_atr"] == pytest.approx(-0.1)
    assert stats["max_atr"] == pytest.approx(0.2)
    assert "slippage" not in h.breaches(NOW)
    h.note_fill(8.0, 10.0)  # +0.8 ATR adverse
    assert "slippage" in h.breaches(NOW)


def test_slippage_without_atr_is_dropped():
    h = HealthMonitor()
    h.note_fill(5.0, 0.0)
    h.note_fill(5.0, None)
    assert h.slippage_stats()["n"] == 0
    assert h.fills_recorded == 0


def test_runner_records_latency_and_slippage_on_fill(tmp_path):
    """Drive _place_live_order with the signal-log harness: one
    order -> one latency sample; fill above the aim (long) -> one
    adverse slippage sample in ATR units."""
    from types import SimpleNamespace

    from goldenswing.execution.executor import OrderResult, Position
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    from goldenswing.execution.risk import gold_spec
    from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
    from goldenswing.live.hierarchy import TimeframeStack
    from mt5_fakes import FakeMT5

    class _Exec:
        def __init__(self):
            self.orders = []
            self._ticket = 9000
            self._positions = []

        def symbol_spec(self, symbol):
            return gold_spec()

        def positions(self, symbol=None):
            return list(self._positions)

        def place_order(self, request):
            self._ticket += 1
            self.orders.append(request)
            self._positions.append(Position(
                ticket=self._ticket, symbol=request.symbol,
                direction=request.direction, volume=request.volume,
                entry_price=request.entry_ref, sl=request.stop_loss,
                tp=request.take_profit, magic=request.magic,
                alert_id=request.alert_id))
            # Fill 1.0 above the aimed price: adverse for a long.
            return OrderResult(True, ticket=self._ticket,
                               price=request.entry_ref + 1.0,
                               volume=request.volume)

        def cancel_pending(self, ticket):
            return OrderResult(True, ticket=ticket)

        def close_position(self, ticket, volume=None):
            return OrderResult(True, ticket=ticket)

        def modify_sl(self, ticket, sl, tp=None):
            return OrderResult(True, ticket=ticket)

    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("fvg_midline",),
        # News pause OFF: the order path's news check reads the
        # REAL wall-clock calendar — this test must not fail on
        # a real news day (e.g. PPI).  The pause itself is covered
        # by test_news_calendar.py.
        news_pause_enabled=False)
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    runner = MT5Runner(cfg, feed=feed, executor=_Exec())
    # Pinned Wednesday tick: a timeless/absent tick makes the order
    # path fall back to the wall clock, so this test fails on
    # Fridays after 20:00 UTC (Friday-close gate).
    runner._last_tick = {"mid": 2000.0, "bid": 1999.8, "ask": 2000.2,
                         "time": "2026-10-14 12:00:00"}
    h1 = pd.DataFrame(
        {"open": [2000.0] * 30, "high": [2001.0] * 30,
         "low": [1999.0] * 30, "close": [2000.5] * 30,
         "volume": [100] * 30, "real_volume": [0] * 30,
         "spread": [30] * 30},
        index=pd.date_range("2026-01-01", periods=30, freq="h",
                            tz="UTC"))
    runner._stack = TimeframeStack.from_frames({"H1": h1})
    sig = SimpleNamespace(direction=1, score=80.0, entry_ref=2000.0,
                          invalidation=1990.0, targets=[2030.0],
                          atr=10.0, index=100, factors=[])
    verdict = {"zone": {"bottom": 1995.0, "top": 2005.0},
               "entry": 2000.0}
    runner._place_live_order(SimpleNamespace(name="fvg_midline"), sig,
                             {"executed": 0}, verdict=verdict,
                             entry_bar=29)
    assert runner._health.orders_placed == 1
    assert runner._health.latency_stats()["n"] == 1
    # Fill 2001.0 vs aim 2000.0, long, ATR 10 -> +0.1 ATR adverse.
    assert runner._health.slippage_stats() == {
        "n": 1, "mean_atr": pytest.approx(0.1),
        "max_atr": pytest.approx(0.1)}
    assert "health:" in runner.health_line()
