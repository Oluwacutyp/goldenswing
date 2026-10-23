"""Urgency + volume-profile dispatch + multi-zone limit ladder.

Urgency: base 50 + session (london +20, overlap/newyork +30) +
ATR velocity (+10/+20/+40) + zone proximity (+40 inside, +20 <0.5
ATR, +10 <1 ATR).  NO RSI term — this engine computes no RSI, and
urgency must not invent one.  Volume profile: POC / VAH / VAL /
nodes from OHLCV bars (the old bot's calculation).  Ladder:
sequential rungs from the playbook level to the adverse zone edge,
one live pending at a time, next rung on expiry, rest discarded on
fill.  Default OFF.

Urgency NEVER upgrades a limit to a market at a worse-than-POI
price (no-chase discipline — the old bot's urgency>80 -> MARKET
has no no-chase rule; this engine does).
"""
import pandas as pd
import pytest

from goldenswing.execution.mt5 import (
    build_ladder, select_order_type)
from goldenswing.live.urgency import urgency_score
from goldenswing.live.volume_profile import volume_profile


def _flat_frame(n=60, spread=10.0, base=2000.0):
    return pd.DataFrame(
        {"open": [base] * n, "high": [base + spread / 2] * n,
         "low": [base - spread / 2] * n, "close": [base] * n,
         "volume": [100] * n},
        index=pd.date_range("2026-01-01", periods=n, freq="h",
                            tz="UTC"))


def test_urgency_base_and_session():
    assert urgency_score(2000.0, session="off", zone=None,
                         frame=None) == pytest.approx(50.0)
    assert urgency_score(2000.0, session="london", zone=None,
                         frame=None) == pytest.approx(70.0)
    assert urgency_score(2000.0, session="newyork", zone=None,
                         frame=None) == pytest.approx(80.0)
    assert urgency_score(2000.0, session="overlap", zone=None,
                         frame=None) == pytest.approx(80.0)
    assert urgency_score(2000.0, session="asia", zone=None,
                         frame=None) == pytest.approx(50.0)


def test_urgency_zone_proximity_bands():
    zone = {"bottom": 1990.0, "top": 2010.0}
    frame = _flat_frame()  # constant ATR 10: no velocity term
    # Inside the zone: +40.
    assert urgency_score(2000.0, session="off", zone=zone,
                         frame=frame) == pytest.approx(90.0)
    # 0.4 ATR from the edge: +20.
    assert urgency_score(2014.0, session="off", zone=zone,
                         frame=frame) == pytest.approx(70.0)
    # 0.9 ATR from the edge: +10.
    assert urgency_score(2019.0, session="off", zone=zone,
                         frame=frame) == pytest.approx(60.0)
    # 1.5 ATR away: nothing.
    assert urgency_score(2025.0, session="off", zone=zone,
                         frame=frame) == pytest.approx(50.0)


def test_urgency_atr_velocity_band():
    # Flat TR 2 for 57 bars, then 3 wide bars (TR 20): current ATR
    # jumps >30% above its 14-bar mean -> +40.
    frame = _flat_frame(60, spread=2.0)
    frame.loc[frame.index[57:], "high"] = 2010.0
    frame.loc[frame.index[57:], "low"] = 1990.0
    assert urgency_score(2000.0, session="off", zone=None,
                         frame=frame) == pytest.approx(90.0)


def test_urgency_fail_open():
    assert urgency_score(2000.0, session=None, zone=None,
                         frame=None) == pytest.approx(50.0)
    bad = pd.DataFrame({"high": [float("nan")] * 60,
                        "low": [float("nan")] * 60,
                        "close": [float("nan")] * 60})
    assert urgency_score(2000.0, session="off", zone=None,
                         frame=bad) == pytest.approx(50.0)


def test_volume_profile_poc_and_value_area():
    frame = _flat_frame(20, spread=20.0, base=2000.0)
    # Concentrate volume in 3 bars around 2005: POC must sit there.
    frame.loc[frame.index[5:8], "high"] = 2006.0
    frame.loc[frame.index[5:8], "low"] = 2004.0
    frame.loc[frame.index[5:8], "volume"] = 10000
    prof = volume_profile(frame)
    assert prof["status"] == "ok"
    assert 2004.0 <= prof["poc"] <= 2006.0
    assert prof["val"] <= prof["poc"] <= prof["vah"]
    assert len(prof["high_vol_nodes"]) == 3
    assert len(prof["low_vol_nodes"]) == 3


def test_volume_profile_degenerate_frames():
    assert volume_profile(None)["poc"] is None
    assert volume_profile(_flat_frame(5))["poc"] is None
    flat = _flat_frame(20)
    flat["volume"] = 0
    assert volume_profile(flat)["poc"] is None


def test_dispatch_thick_level_downgrades_market_to_limit():
    prof = {"poc": 2000.0, "high_vol_nodes": []}
    kind, level, reason = select_order_type(
        direction=1, price=2000.0, entry_ref=2000.0,
        zone={"bottom": 1990.0, "top": 2010.0}, atr=10.0,
        urgency=40.0, profile=prof)
    assert (kind, level) == ("limit", 2000.0)
    assert "thick level" in reason
    # High urgency -> no downgrade (market at a thick level is fine
    # when price is running).
    kind, _, _ = select_order_type(
        direction=1, price=2000.0, entry_ref=2000.0,
        zone={"bottom": 1990.0, "top": 2010.0}, atr=10.0,
        urgency=80.0, profile=prof)
    assert kind == "market"
    # Price away from the POC -> no downgrade.
    kind, _, _ = select_order_type(
        direction=1, price=2000.0, entry_ref=2000.0,
        zone={"bottom": 1990.0, "top": 2010.0}, atr=10.0,
        urgency=40.0, profile={"poc": 2100.0,
                               "high_vol_nodes": []})
    assert kind == "market"


def test_dispatch_urgency_never_upgrades_limit_to_market():
    # Pullback limit (long, price above the POI): even urgency 95
    # must NOT chase — market would pay worse than the POI.
    kind, level, _ = select_order_type(
        direction=1, price=2010.0, entry_ref=2000.0,
        zone={"bottom": 1990.0, "top": 2010.0}, atr=10.0,
        urgency=95.0, profile={"poc": 2000.0,
                                "high_vol_nodes": []})
    assert (kind, level) == ("limit", 2000.0)


def test_dispatch_unchanged_without_urgency_profile():
    kind, level, _ = select_order_type(
        direction=1, price=2000.0, entry_ref=2000.0,
        zone={"bottom": 1990.0, "top": 2010.0}, atr=10.0)
    assert (kind, level) == ("market", 2000.0)


def test_build_ladder_long_and_short():
    zone = {"bottom": 1990.0, "top": 2010.0}
    rungs = build_ladder(direction=1, entry_ref=2000.0, zone=zone,
                         lots=0.03, rungs=3,
                         split=(0.5, 0.3, 0.2))
    assert [r[0] for r in rungs] == [2000.0, 1995.0, 1990.0]
    assert [r[1] for r in rungs] == pytest.approx([0.015, 0.009,
                                                  0.006])
    assert sum(r[1] for r in rungs) == pytest.approx(0.03)
    rungs = build_ladder(direction=-1, entry_ref=2000.0, zone=zone,
                         lots=0.03, rungs=3,
                         split=(0.5, 0.3, 0.2))
    assert [r[0] for r in rungs] == [2000.0, 2005.0, 2010.0]


def test_build_ladder_degenerate_is_no_ladder():
    assert build_ladder(direction=1, entry_ref=2000.0, zone=None,
                        lots=0.03) == []
    assert build_ladder(direction=1, entry_ref=2000.0,
                        zone={"bottom": 2010.0, "top": 1990.0},
                        lots=0.03) == []
    # Entry already at the adverse edge: rungs collapse to one.
    assert build_ladder(direction=1, entry_ref=1990.0,
                        zone={"bottom": 1990.0, "top": 2010.0},
                        lots=0.03) == []
    assert build_ladder(direction=1, entry_ref=2000.0,
                        zone={"bottom": 1990.0, "top": 2010.0},
                        lots=0.03, rungs=1) == []


class _Exec:
    def __init__(self):
        self.placed = []
        self.cancelled = []
        self._ticket = 9000

    def symbol_spec(self, symbol):
        from goldenswing.execution.risk import gold_spec

        return gold_spec()

    def account_state(self):
        return {"balance": 10000.0, "equity": 10000.0,
                "margin": 0.0, "free_margin": 10000.0}

    def positions(self, symbol=None):
        return []

    def place_order(self, request):
        from goldenswing.execution.executor import OrderResult

        self._ticket += 1
        self.placed.append(request)
        return OrderResult(True, ticket=self._ticket,
                           price=request.order_price or 0.0,
                           volume=request.volume)

    def cancel_pending(self, ticket):
        from goldenswing.execution.executor import OrderResult

        self.cancelled.append(ticket)
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        from goldenswing.execution.executor import OrderResult

        return OrderResult(True, ticket=ticket)

    def modify_sl(self, ticket, sl, tp=None):
        from goldenswing.execution.executor import OrderResult

        return OrderResult(True, ticket=ticket)


def _runner(tmp_path, *, ladder_enabled=True):
    from types import SimpleNamespace

    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
    from goldenswing.live.hierarchy import TimeframeStack
    from mt5_fakes import FakeMT5

    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("fvg_midline",),
        fixed_lots=0.03,
        ladder_enabled=ladder_enabled,
        # News pause OFF: the order path's news check reads the
        # REAL wall-clock calendar — these tests must not fail on
        # a real news day (e.g. PPI).  The pause itself is covered
        # by test_news_calendar.py.
        news_pause_enabled=False,
        score_decay_enabled=False)
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    runner = MT5Runner(cfg, feed=feed, executor=_Exec())
    h1 = _flat_frame(60, spread=2.0)
    runner._stack = TimeframeStack.from_frames({"H1": h1})
    runner._last_tick = {"mid": 2010.0, "bid": 2009.8, "ask": 2010.2,  # above the POI: pullback
                         "time": pd.Timestamp(
                             "2026-01-05 10:00", tz="UTC"
                         ).to_pydatetime()}
    return runner


def _order(runner):
    from types import SimpleNamespace

    pb = SimpleNamespace(name="fvg_midline")
    sig = SimpleNamespace(direction=1, score=80.0, entry_ref=2000.0,
                          invalidation=1990.0, targets=[2030.0],
                          atr=10.0, index=100, factors=[])
    verdict = {"zone": {"bottom": 1990.0, "top": 2010.0},
               "entry": 2000.0}
    runner._place_live_order(pb, sig, {"executed": 0},
                             verdict=verdict, entry_bar=59)


def test_runner_ladder_places_rung_zero_and_advances_on_expiry(tmp_path):
    runner = _runner(tmp_path)
    _order(runner)
    assert len(runner._pending_orders) == 1
    state = next(iter(runner._pending_orders.values()))
    assert state.level == pytest.approx(2000.0)
    assert state.volume == pytest.approx(0.015)
    assert state.ladder_levels == [1995.0, 1990.0]
    assert state.ladder_lots == pytest.approx([0.009, 0.006])
    # One live pending at a time.
    assert len(runner.executor.placed) == 1
    # Expiry: the rung cancels, the next rung is placed.
    state.expiry_bar = 59
    runner._on_h1_close(
        {"mid": 2000.5, "bid": 2000.3, "ask": 2000.7, "time": pd.Timestamp(
            "2026-01-05 11:00", tz="UTC").to_pydatetime()}, {})
    assert len(runner._pending_orders) == 1
    state = next(iter(runner._pending_orders.values()))
    assert state.level == pytest.approx(1995.0)
    assert state.volume == pytest.approx(0.009)
    assert state.ladder_levels == [1990.0]
    assert len(runner.executor.placed) == 2


def test_runner_ladder_off_places_single_pending(tmp_path):
    runner = _runner(tmp_path, ladder_enabled=False)
    _order(runner)
    state = next(iter(runner._pending_orders.values()))
    assert state.volume == pytest.approx(0.03)
    assert state.ladder_levels == []


def test_runner_ladder_fill_discards_remaining_rungs(tmp_path):
    from goldenswing.execution.executor import Position

    runner = _runner(tmp_path)
    _order(runner)
    state = next(iter(runner._pending_orders.values()))
    pos = Position(ticket=state.ticket, symbol="XAUUSDm",
                   direction=1, volume=state.volume,
                   entry_price=state.level, sl=state.sl,
                   tp=state.tp, magic=0, alert_id=state.alert_id)
    runner._record_pending_fill(state, pos, {})
    # No new pending for the remaining rungs: fill discards them.
    assert runner._pending_orders == {}
    assert len(runner.executor.placed) == 1
