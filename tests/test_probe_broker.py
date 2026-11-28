"""The binding probe's broker simulator (fill-funnel lane).

The old fake filled EVERY order the instant it was placed and
never resolved anything after, so the first order froze the
one-position slot for the rest of the replay and the probe
measured "1 order in 30 days".  These tests pin the broker
model: market fills at once, pendings rest until touched,
SL/TP closes on touch (stop first when both touch), and the
SL moves the tick manager makes land on the broker's book.
"""

from goldenswing.execution.executor import OrderRequest
from scripts.family_binding_probe import _FakeExecutor


def _req(kind, direction, entry_ref, price, sl, tp):
    return OrderRequest(
        symbol="XAUUSD", direction=direction, volume=0.01,
        stop_loss=sl, entry_ref=entry_ref,
        take_profit=tp, targets=[tp] if tp else [],
        atr=5.0, risk_money=0.0, alert_id="MT5-test-1",
        bar_index=0, magic=12345, order_kind=kind,
        order_price=price)


def test_market_fills_at_once():
    ex = _FakeExecutor()
    res = ex.place_order(
        _req("market", 1, 3500.0, 0.0, 3400.0, 3600.0))
    assert res.ok
    assert len(ex.positions()) == 1
    assert ex.positions()[0].entry_price == 3500.0
    assert ex.orders[0]["kind"] == "market"


def test_limit_rests_until_touched():
    ex = _FakeExecutor()
    ex.place_order(_req("market", 1, 3500.0, 0.0, 3400.0, 3600.0))
    ex.place_order(_req("limit", 1, 3480.0, 3480.0, 3470.0, 3500.0))
    assert len(ex.positions()) == 1  # the limit is not a position
    assert len(ex.pending_orders()) == 1
    assert ex.pending_orders()[0].price == 3480.0
    filled, closed = ex.broker_step(high=3495.0, low=3485.0)
    assert filled == [] and closed == []
    assert len(ex.pending_orders()) == 1
    filled, closed = ex.broker_step(high=3490.0, low=3478.0)
    assert len(filled) == 1 and filled[0][1] == "limit"
    assert filled[0][2] == 3480.0  # filled at the pending level
    assert len(ex.pending_orders()) == 0
    assert len(ex.positions()) == 2
    assert ex.positions()[1].entry_price == 3480.0


def test_stop_close_at_the_level_and_deal_recorded():
    ex = _FakeExecutor()
    ex.place_order(_req("limit", 1, 3480.0, 3480.0, 3470.0, 3500.0))
    ex.broker_step(high=3490.0, low=3478.0)
    ticket = ex.positions()[0].ticket
    ex.modify_sl(ticket, 3480.0)  # the tick manager's breakeven
    assert ex.positions()[0].sl == 3480.0
    _filled, closed = ex.broker_step(high=3490.0, low=3479.0)
    assert len(closed) == 1 and closed[0][2] == "stop"
    assert closed[0][1] == 3480.0  # exit priced at the stop level
    deal = ex.closing_deal(closed[0][0])
    assert deal and deal["price"] == 3480.0
    assert deal["reason"] == "stop"
    assert len(ex.positions()) == 0


def test_target_close_and_short_side():
    ex = _FakeExecutor()
    ex.place_order(_req("market", 1, 3500.0, 0.0, 3400.0, 3520.0))
    filled, closed = ex.broker_step(high=3521.0, low=3505.0)
    assert len(closed) == 1 and closed[0][2] == "target"
    assert closed[0][1] == 3520.0
    ex.place_order(_req("stop", -1, 3510.0, 3510.0, 3520.0, 3490.0))
    filled, closed = ex.broker_step(high=3512.0, low=3500.0)
    assert len(filled) == 1 and filled[0][1] == "stop"
    assert ex.positions()[0].entry_price == 3510.0
    filled, closed = ex.broker_step(high=3521.0, low=3505.0)
    assert len(closed) == 1 and closed[0][2] == "stop"
    assert closed[0][1] == 3520.0


def test_cancel_pending_and_stop_first():
    ex = _FakeExecutor()
    res = ex.place_order(
        _req("limit", 1, 3400.0, 3400.0, 3390.0, 3420.0))
    assert len(ex.pending_orders()) == 1
    ex.cancel_pending(res.ticket)
    assert len(ex.pending_orders()) == 0
    ex.place_order(_req("market", 1, 3500.0, 0.0, 3495.0, 3505.0))
    _filled, closed = ex.broker_step(high=3510.0, low=3490.0)
    assert len(closed) == 1 and closed[0][2] == "stop"
    assert closed[0][1] == 3495.0  # both touched -> stop wins
