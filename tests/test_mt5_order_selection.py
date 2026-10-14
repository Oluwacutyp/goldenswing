"""MT5 engine: setup-dependent order selection.

A real trader picks the order type by where price is vs the entry
level when the signal fires — never always-market:

- price AT the entry level, trigger confirmed -> MARKET at the tick
- entry level BEHIND price (pullback) -> PENDING LIMIT at the POI
- entry level AHEAD of price (breakout) -> PENDING STOP beyond it
- price beyond the POI on the adverse side -> INVALID (zone failed)

Pending orders are tracked until they fill, cancel, or expire:
invalidation (zone break / invalidation cross) cancels on the tick,
expiry (N H1 bars without fill) cancels on the H1 close, a bias flip
cancels, engine stop cancels.  A fill becomes a TickManager
position exactly like a market fill; reconciliation picks up a
fill the engine missed (never double-counted); a pending COUNTS
toward the one-position rule.
"""
import pandas as pd
import pytest
from types import SimpleNamespace

from goldenswing.execution.executor import (OrderRequest, OrderResult,
                                              PendingOrder, Position)
from goldenswing.execution.mt5 import (MT5Executor, select_order_type)
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.golden.detector import GoldenSignal
from goldenswing.live.hierarchy import TimeframeStack
from mt5_fakes import FakeMT5

ZONE = {"bottom": 2000.0, "top": 2010.0, "kind": "support",
        "level": 2005.0}
ATR = 10.0  # tolerance = 0.25 * ATR = 2.5


# ------------------------------------------------------------- selection
def test_at_level_with_trigger_confirmed_is_market():
    kind, level, _ = select_order_type(
        direction=1, price=2005.0, entry_ref=2005.0, zone=ZONE, atr=ATR)
    assert kind == "market"
    assert level == pytest.approx(2005.0)


def test_within_tolerance_of_level_is_market():
    kind, _, _ = select_order_type(
        direction=1, price=2006.0, entry_ref=2005.0, zone=ZONE, atr=ATR)
    assert kind == "market"


def test_long_pullback_is_limit_at_entry_ref_not_tick():
    # Price ran up to 2030; the POI entry is 2005 — the market must
    # come back to the order.  The limit price is the entry level,
    # NEVER the current tick price.
    kind, level, reason = select_order_type(
        direction=1, price=2030.0, entry_ref=2005.0, zone=ZONE, atr=ATR)
    assert kind == "limit"
    assert level == pytest.approx(2005.0)
    assert "pullback" in reason


def test_short_pullback_is_limit_at_entry_ref():
    kind, level, _ = select_order_type(
        direction=-1, price=1970.0, entry_ref=2005.0, zone=ZONE, atr=ATR)
    assert kind == "limit"
    assert level == pytest.approx(2005.0)


def test_long_breakout_is_stop_beyond_level():
    # Entry level sits ABOVE current price: price must break up
    # through it to enter.
    kind, level, reason = select_order_type(
        direction=1, price=2000.0, entry_ref=2020.0, zone=ZONE, atr=ATR)
    assert kind == "stop"
    assert level == pytest.approx(2020.0)
    assert "breakout" in reason


def test_short_breakout_is_stop_beyond_level():
    kind, level, _ = select_order_type(
        direction=-1, price=2010.0, entry_ref=1990.0, zone=ZONE, atr=ATR)
    assert kind == "stop"
    assert level == pytest.approx(1990.0)


def test_long_zone_failed_below_is_invalid():
    kind, level, reason = select_order_type(
        direction=1, price=1990.0, entry_ref=2005.0, zone=ZONE, atr=ATR)
    assert kind == "invalid"
    assert level == 0.0
    assert "zone failed" in reason


def test_short_zone_failed_above_is_invalid():
    kind, _, _ = select_order_type(
        direction=-1, price=2020.0, entry_ref=2005.0, zone=ZONE, atr=ATR)
    assert kind == "invalid"


def test_no_zone_guard_without_zone():
    # No zone to fail: price above the entry level is still a
    # pullback limit, decided by entry_ref vs price alone.
    kind, level, _ = select_order_type(
        direction=1, price=2020.0, entry_ref=2005.0, zone=None, atr=ATR)
    assert kind == "limit"
    assert level == pytest.approx(2005.0)


def test_missing_entry_ref_falls_back_to_market():
    kind, _, _ = select_order_type(
        direction=1, price=2030.0, entry_ref=0.0, zone=ZONE, atr=ATR)
    assert kind == "market"


# -------------------------------------------------------------- executor
def _mt5_executor(fake=None):
    fake = fake or FakeMT5()
    return MT5Executor(mt5_module=fake), fake


def _request(**kw):
    base = dict(symbol="XAUUSD", direction=1, volume=0.01,
                stop_loss=1990.0, entry_ref=2005.0, take_profit=2040.0,
                alert_id="MT5-test-1", magic=260101)
    base.update(kw)
    return OrderRequest(**base)


def test_executor_limit_sends_pending_buy_limit_at_zone():
    ex, fake = _mt5_executor()  # tick: bid 2000.0 / ask 2000.30
    res = ex.place_order(_request(order_kind="limit",
                                  order_price=1995.0))
    assert res.ok
    assert res.price == pytest.approx(1995.0)
    sent = fake.sent[-1]
    assert sent["action"] == fake.TRADE_ACTION_PENDING
    assert sent["type"] == fake.ORDER_TYPE_BUY_LIMIT
    assert sent["price"] == pytest.approx(1995.0)
    assert sent["sl"] == pytest.approx(1990.0)
    assert fake.orders_store  # the broker holds the pending


def test_executor_stop_sends_pending_buy_stop():
    ex, fake = _mt5_executor()
    res = ex.place_order(_request(order_kind="stop",
                                  order_price=2010.0))
    assert res.ok
    sent = fake.sent[-1]
    assert sent["action"] == fake.TRADE_ACTION_PENDING
    assert sent["type"] == fake.ORDER_TYPE_BUY_STOP
    assert sent["price"] == pytest.approx(2010.0)


def test_executor_sell_limit_and_sell_stop():
    ex, fake = _mt5_executor()
    res = ex.place_order(_request(direction=-1, order_kind="limit",
                                  order_price=2010.0))
    assert res.ok
    assert fake.sent[-1]["type"] == fake.ORDER_TYPE_SELL_LIMIT
    res = ex.place_order(_request(direction=-1, order_kind="stop",
                                  order_price=1995.0))
    assert res.ok
    assert fake.sent[-1]["type"] == fake.ORDER_TYPE_SELL_STOP


def test_executor_refuses_wrong_sided_pending():
    ex, fake = _mt5_executor()
    # BUY_LIMIT must sit BELOW the ask (2000.30) — above it is a
    # broker rejection, refused here instead.
    res = ex.place_order(_request(order_kind="limit",
                                  order_price=2005.0))
    assert not res.ok
    assert "not below ask" in res.message
    assert not fake.orders_store
    # BUY_STOP must sit ABOVE the ask.
    res = ex.place_order(_request(order_kind="stop",
                                  order_price=1995.0))
    assert not res.ok
    assert "not above ask" in res.message


def test_executor_default_is_still_market():
    ex, fake = _mt5_executor()
    res = ex.place_order(_request())
    assert res.ok
    assert fake.sent[-1]["action"] == fake.TRADE_ACTION_DEAL
    assert fake.sent[-1]["type"] == fake.ORDER_TYPE_BUY


def test_executor_pending_orders_and_cancel():
    ex, fake = _mt5_executor()
    res = ex.place_order(_request(order_kind="limit",
                                  order_price=1995.0))
    pendings = ex.pending_orders("XAUUSD")
    assert len(pendings) == 1
    assert isinstance(pendings[0], PendingOrder)
    assert pendings[0].kind == "limit"
    assert pendings[0].direction == 1
    assert pendings[0].price == pytest.approx(1995.0)
    assert pendings[0].alert_id == "MT5-test-1"
    assert ex.cancel_pending(res.ticket).ok
    assert ex.pending_orders("XAUUSD") == []
    # Cancelling a gone ticket fails cleanly (never a fake success).
    assert not ex.cancel_pending(res.ticket).ok


# ---------------------------------------------------------------- runner
class FakeFeed:
    def account_state(self):
        return {"equity": 10000.0, "balance": 10000.0}


class FakeExecutor:
    """Executor stub honouring order_kind: market fills at once,
    limit/stop rest as pendings until cancelled or filled."""

    def __init__(self):
        self._ticket = 9000
        self._positions: list[Position] = []
        self._pendings: list[PendingOrder] = []
        self.sent_kinds: list[str] = []

    def connect(self, **kw):
        return True, "fake connected"

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return [p for p in self._positions
                if symbol is None or p.symbol == symbol]

    def pending_orders(self, symbol=None):
        return [p for p in self._pendings
                if symbol is None or p.symbol == symbol]

    def place_order(self, request):
        self._ticket += 1
        self.sent_kinds.append(request.order_kind)
        if request.order_kind in ("limit", "stop"):
            pend = PendingOrder(
                ticket=self._ticket, symbol=request.symbol,
                kind=request.order_kind, direction=request.direction,
                price=float(request.order_price or request.entry_ref),
                volume=request.volume, sl=request.stop_loss,
                tp=request.take_profit, magic=request.magic,
                alert_id=request.alert_id)
            self._pendings.append(pend)
            return OrderResult(True, ticket=self._ticket,
                               price=pend.price, volume=request.volume)
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
        for p in list(self._pendings):
            if p.ticket == ticket:
                self._pendings.remove(p)
                return OrderResult(True, ticket=ticket)
        return OrderResult(False, message="order not found")

    def fill_pending(self, ticket, price):
        """Broker fill: pending leaves, a position with the same
        alert_id appears."""
        for p in list(self._pendings):
            if p.ticket == ticket:
                self._pendings.remove(p)
                self._positions.append(Position(
                    ticket=p.ticket, symbol=p.symbol,
                    direction=p.direction, volume=p.volume,
                    entry_price=price, sl=p.sl, tp=p.tp,
                    magic=p.magic, alert_id=p.alert_id))
                return True
        return False

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        self._positions = [p for p in self._positions
                           if p.ticket != ticket]
        return OrderResult(True, ticket=ticket)


def _frames():
    idx = pd.date_range("2026-10-14 00:00", periods=30, freq="h",
                        tz="UTC")
    base = pd.DataFrame(
        {"open": [2000.0] * 30, "high": [2001.0] * 30,
         "low": [1999.0] * 30, "close": [2000.0] * 30,
         "volume": [100] * 30, "real_volume": [0] * 30,
         "spread": [30] * 30}, index=idx)
    return {"D1": base.copy(), "H4": base.copy(), "H1": base.copy(),
            "M15": base.copy(), "M30": base.copy(), "M5": base.copy()}


def _runner(tmp_path, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        fixed_lots=0.01, **cfg_kw)
    runner = MT5Runner(cfg, feed=FakeFeed(), executor=FakeExecutor())
    runner._stack = TimeframeStack.from_frames(_frames())
    return runner


def _sig(**kw):
    base = dict(index=29, timestamp=None, direction=1, score=80.0,
                kind="fvg_midline", factors=[], entry_zone=(2000.0, 2010.0),
                entry_ref=2005.0, invalidation=1990.0,
                targets=[2040.0], atr=10.0)
    base.update(kw)
    return GoldenSignal(**base)


def _verdict(**kw):
    base = {"allowed": True, "blocked_leg": "", "legs": {},
            "bias_d1": 1, "bias_h4": 1, "direction": 1, "zone": ZONE,
            "trigger": "BOS_bull", "entry": 2030.0, "stop": 1990.0,
            "target": 2040.0, "risk": 40.0, "session": {},
            "entry_clock": "M15", "amd_phase": "unknown",
            "qm": {"status": "qm_absent"}, "veto_enabled": False,
            "asof": "2026-10-14T00:00:00+00:00", "volume": {}}
    base.update(kw)
    return base


def _journal(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    return path.read_text() if path.exists() else ""


def _place(runner, sig, verdict, stats, price):
    runner._last_tick = {"mid": price, "bid": price, "ask": price + 0.3,
                         "tick_volume": 1, "real_volume": 0, "spread": 30}
    runner._place_live_order(SimpleNamespace(name="fvg_midline"), sig,
                             stats, verdict=verdict, entry_bar=29)


def test_runner_away_setup_places_limit_not_market(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    # Price 2030 is away from the 2005 POI entry: pullback setup.
    _place(runner, _sig(), _verdict(), stats, price=2030.0)
    assert runner.executor.sent_kinds == ["limit"]
    assert runner.executor.positions() == []  # NO market fill
    assert len(runner._pending_orders) == 1
    state = next(iter(runner._pending_orders.values()))
    assert state.kind == "limit"
    assert state.level == pytest.approx(2005.0)  # POI, not the tick
    assert state.direction == 1
    assert stats["pending"] == 1
    assert "mt5 pending: LIMIT" in _journal(tmp_path)


def test_runner_at_zone_places_market(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    _place(runner, _sig(), _verdict(), stats, price=2005.0)
    assert runner.executor.sent_kinds == ["market"]
    assert len(runner.executor.positions()) == 1
    assert runner._pending_orders == {}
    assert stats["executed"] == 1


def test_runner_breakout_places_stop(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    # Entry level 2020 sits ABOVE current price 2000: breakout setup.
    _place(runner, _sig(entry_ref=2020.0), _verdict(), stats, price=2000.0)
    assert runner.executor.sent_kinds == ["stop"]
    assert len(runner._pending_orders) == 1
    state = next(iter(runner._pending_orders.values()))
    assert state.kind == "stop"
    assert state.level == pytest.approx(2020.0)


def test_runner_zone_failed_places_nothing(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    # Price 1990 is below the POI bottom (long): the zone failed.
    _place(runner, _sig(), _verdict(), stats, price=1990.0)
    assert runner.executor.sent_kinds == []
    assert runner._pending_orders == {}
    assert stats["invalidated"] == 1
    assert "order invalidated" in _journal(tmp_path)


def test_no_second_pending_while_one_active(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    _place(runner, _sig(), _verdict(), stats, price=2030.0)
    assert len(runner._pending_orders) == 1
    _place(runner, _sig(index=30), _verdict(), stats, price=2030.0)
    assert len(runner._pending_orders) == 1  # one-position rule
    assert "already holding/pending" in _journal(tmp_path)


def test_pending_cancelled_on_zone_break(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    _place(runner, _sig(), _verdict(), stats, price=2030.0)
    ticket = next(iter(runner._pending_orders))
    # Price drops through the POI bottom: the zone failed.
    runner._check_pending_orders(1995.0, stats)
    assert runner._pending_orders == {}
    assert runner.executor.pending_orders() == []
    assert stats["pending_cancelled"] == 1
    assert "invalidated" in _journal(tmp_path)


def test_pending_cancelled_on_expiry(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    _place(runner, _sig(), _verdict(), stats, price=2030.0)
    ticket = next(iter(runner._pending_orders))
    # Fast-forward: the pending is past its expiry bar, then an H1
    # close fires the expiry check.
    runner._pending_orders[ticket].expiry_bar = 29
    runner._on_h1_close(
        {"time": pd.Timestamp("2026-10-14 06:00", tz="UTC"),
         "mid": 2030.0, "spread": 30}, stats)
    assert runner._pending_orders == {}
    assert "expired" in _journal(tmp_path)


def test_pending_cancelled_on_bias_flip(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    _place(runner, _sig(), _verdict(), stats, price=2030.0)
    assert len(runner._pending_orders) == 1
    runner._live_setup = {"direction": -1}
    runner._check_pending_bias(stats)
    assert runner._pending_orders == {}
    assert "bias_flipped" in _journal(tmp_path)


def test_pending_fill_becomes_tracked_position(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    _place(runner, _sig(), _verdict(), stats, price=2030.0)
    ticket = next(iter(runner._pending_orders))
    # Broker fills the pending at the POI level.
    assert runner.executor.fill_pending(ticket, 2005.0)
    runner._reconcile(2005.0, stats)
    assert runner._pending_orders == {}
    assert stats["pending_filled"] == 1
    assert stats["executed"] == 1
    assert runner.tick_manager.tickets() == [ticket]
    assert "PENDING LIMIT filled" in _journal(tmp_path)


def test_reconcile_drops_externally_cancelled_pending(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    _place(runner, _sig(), _verdict(), stats, price=2030.0)
    ticket = next(iter(runner._pending_orders))
    # Broker-side cancel (manual / broker expiry): the order leaves
    # orders_get with no position behind it.
    assert runner.executor.cancel_pending(ticket).ok
    runner._reconcile(2030.0, stats)
    assert runner._pending_orders == {}
    assert stats["pending_cancelled"] == 1
    assert "cancelled externally" in _journal(tmp_path)


def test_cancel_failure_then_fill_records_fill_not_retry_loop(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    _place(runner, _sig(), _verdict(), stats, price=2030.0)
    ticket = next(iter(runner._pending_orders))
    # The order filled between the invalidation check and the cancel:
    # cancel fails (ticket gone) but the position proves the fill.
    assert runner.executor.fill_pending(ticket, 2005.0)
    assert runner._cancel_pending(ticket, "invalidated", stats) is True
    assert runner._pending_orders == {}
    assert stats["pending_filled"] == 1
    assert runner.tick_manager.tickets() == [ticket]


def test_stop_cancels_all_pendings(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    _place(runner, _sig(), _verdict(), stats, price=2030.0)
    assert len(runner._pending_orders) == 1
    # Engine stop: no stale pending is ever left on the broker.
    for ticket in list(runner._pending_orders):
        runner._cancel_pending(ticket, "engine_stop", stats)
    assert runner._pending_orders == {}
    assert runner.executor.pending_orders() == []
    assert stats["pending_cancelled"] == 1
