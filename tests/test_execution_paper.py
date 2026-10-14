"""Phase 8 — PaperExecutor fills and Phase-2 exit semantics."""
import pytest

from goldenswing.execution.executor import OrderRequest
from goldenswing.execution.paper import PaperExecutor

HALF = 0.225          # half of the $0.45 round-trip cost


def bar(idx, o, h, l, c):
    return {"index": idx, "time": None, "open": o, "high": h, "low": l,
            "close": c}


def _long_request(**kw):
    base = dict(symbol="XAUUSD", direction=1, volume=0.10,
                stop_loss=1990.0, entry_ref=2000.0, targets=[2030.0],
                atr=10.0, risk_money=100.0, alert_id="T1", bar_index=0)
    base.update(kw)
    return OrderRequest(**base)


def test_fill_at_next_bar_open_with_spread():
    ex = PaperExecutor(balance=10_000.0)
    res = ex.place_order(_long_request())
    assert res.ok and res.ticket is not None
    assert ex.positions() == []                      # nothing filled yet
    events = ex.advance(bar(1, 2000.0, 2005.0, 1995.0, 2002.0))
    assert events[0]["kind"] == "fill"
    pos = ex.positions()[0]
    assert pos.entry_price == pytest.approx(2000.0 + HALF)
    # buffered stop: 1990 - 0.25*ATR(10) = 1987.5
    assert pos.sl == pytest.approx(1987.5)


def test_wrong_side_stop_refused():
    ex = PaperExecutor()
    res = ex.place_order(_long_request(stop_loss=2010.0))
    assert not res.ok


def test_partial_then_trail_exit_full_cycle():
    closes = []
    ex = PaperExecutor(balance=10_000.0, on_close=closes.append)
    ex.place_order(_long_request())
    ex.advance(bar(1, 2000.0, 2005.0, 1995.0, 2002.0))   # fill 2000.225
    ev = ex.advance(bar(2, 2002.0, 2031.0, 2000.0, 2028.0))
    partials = [e for e in ev if e["kind"] == "partial"]
    assert len(partials) == 1
    assert partials[0]["volume"] == pytest.approx(0.05)  # 50% banked
    # banked: (2030 - 0.225 - 2000.225) * 100 * 0.05 = 147.75
    assert partials[0]["pnl"] == pytest.approx(147.75)
    assert ex.account_state().balance == pytest.approx(10_147.75)
    assert ex.positions()[0].volume == pytest.approx(0.05)
    # trail level after bar 2: best 2.42R -> ~2018.3; bar 3 dips through
    ex.advance(bar(3, 2025.0, 2026.0, 2010.0, 2015.0))
    assert ex.positions() == []
    assert len(closes) == 1
    rec = closes[0]
    assert rec.reason == "trail" and rec.partial_taken
    assert rec.hold_bars == 3
    # total: 147.75 + (2018.07 - 2000.225)*100*0.05 ~= 236.98; risk 127.25
    assert rec.pnl_money == pytest.approx(236.98, abs=1.0)
    assert rec.r_net == pytest.approx(236.98 / 127.25, abs=0.02)


def test_wick_through_raw_invalidation_survives_buffer_then_timeout():
    closes = []
    ex = PaperExecutor(balance=10_000.0, on_close=closes.append)
    ex.place_order(_long_request(targets=[]))            # no partial pool
    ex.advance(bar(1, 2000.0, 2003.0, 1996.0, 2001.0))   # fill
    # wick to 1989: through raw invalidation 1990, above buffered 1987.5
    ex.advance(bar(2, 2001.0, 2002.0, 1989.0, 1998.0))
    assert len(ex.positions()) == 1                      # survived
    for i in range(3, 73):                               # flat to horizon
        ex.advance(bar(i, 1998.0, 1999.0, 1997.0, 1998.0))
    assert ex.positions() == []
    rec = closes[0]
    assert rec.reason == "timeout" and rec.wicked_inv


def test_buffered_stop_exit():
    closes = []
    ex = PaperExecutor(balance=10_000.0, on_close=closes.append)
    ex.place_order(_long_request(targets=[]))
    ex.advance(bar(1, 2000.0, 2003.0, 1996.0, 2001.0))
    ex.advance(bar(2, 2001.0, 2002.0, 1987.0, 1990.0))    # low < 1987.5
    assert closes and closes[0].reason == "stop"
    assert closes[0].pnl_money < 0


def test_manual_close_at_next_open():
    closes = []
    ex = PaperExecutor(balance=10_000.0, on_close=closes.append)
    ticket = ex.place_order(_long_request(targets=[])).ticket
    ex.advance(bar(1, 2000.0, 2003.0, 1996.0, 2001.0))
    assert ex.close_position(ticket).ok
    ex.advance(bar(2, 2010.0, 2012.0, 2008.0, 2011.0))
    assert closes and closes[0].reason == "manual"
    assert closes[0].exit_price == pytest.approx(2010.0 - HALF)


def test_short_side_fill_and_stop():
    closes = []
    ex = PaperExecutor(balance=10_000.0, on_close=closes.append)
    ex.place_order(_long_request(direction=-1, stop_loss=2010.0,
                                 entry_ref=2000.0, targets=[]))
    ex.advance(bar(1, 2000.0, 2004.0, 1997.0, 1999.0))   # fill 1999.775
    pos = ex.positions()[0]
    assert pos.entry_price == pytest.approx(2000.0 - HALF)
    assert pos.sl == pytest.approx(2012.5)               # 2010 + 0.25*10
    ex.advance(bar(2, 1999.0, 2013.0, 1995.0, 2005.0))   # high > stop
    assert closes and closes[0].reason == "stop"
    assert closes[0].pnl_money < 0
