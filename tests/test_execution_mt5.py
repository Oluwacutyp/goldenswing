"""Phase 8 — MT5Executor against a stubbed fake MetaTrader5 module.

No real terminal exists here and none is pretended: the fake records
the requests the executor builds, so request construction, filling
-mode fallback and sizing inputs are verified — not live fills.
"""
from types import SimpleNamespace

import pytest

from goldenswing.execution.executor import OrderRequest
from goldenswing.execution.mt5 import MT5Executor, MT5Unavailable
from goldenswing.execution.risk import size_position


class FakeMT5:
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TIMEFRAME_H1 = 16385

    def __init__(self):
        self.sent: list[dict] = []
        self.reject_fillings: set[int] = set()
        self.positions_store: list[SimpleNamespace] = []
        self._ticket = 12340

    # --- terminal surface -------------------------------------------------
    def initialize(self, **kw):
        return True

    def last_error(self):
        return (0, "ok")

    def terminal_info(self):
        return SimpleNamespace(connected=True, build=3815)

    def account_info(self):
        return SimpleNamespace(balance=10_000.0, equity=10_050.0,
                               margin_free=9_900.0, currency="USD",
                               login=555001)

    def symbol_info(self, symbol):
        return SimpleNamespace(
            trade_tick_value=1.0, trade_tick_size=0.01, volume_min=0.01,
            volume_max=50.0, volume_step=0.01, digits=2, point=0.01,
            filling_mode=3)                     # FOK|IOC both offered

    def symbol_select(self, symbol, flag):
        return True

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(ask=2000.30, bid=2000.00)

    def positions_get(self, symbol=None, ticket=None):
        out = [p for p in self.positions_store
               if (symbol is None or p.symbol == symbol)
               and (ticket is None or p.ticket == ticket)]
        return out

    def order_send(self, request):
        self.sent.append(dict(request))
        if request.get("type_filling") in self.reject_fillings:
            return SimpleNamespace(retcode=10030, comment="unsupported filling",
                                   order=0, deal=0, price=0.0, volume=0.0)
        self._ticket += 1
        if request["action"] == self.TRADE_ACTION_DEAL \
                and "position" not in request:
            self.positions_store.append(SimpleNamespace(
                ticket=self._ticket, symbol=request["symbol"],
                type=request["type"], volume=request["volume"],
                price_open=request["price"], sl=request["sl"],
                tp=request["tp"], magic=request["magic"],
                comment=request["comment"], time=0))
        if request["action"] == self.TRADE_ACTION_SLTP:
            for p in self.positions_store:
                if p.ticket == request["position"]:
                    p.sl = request["sl"]
        return SimpleNamespace(retcode=10009, comment="done",
                               order=self._ticket, deal=self._ticket,
                               price=request.get("price", 0.0),
                               volume=request.get("volume", 0.0))


def _executor(fake):
    return MT5Executor(mt5_module=fake, magic=260101)


def _request(**kw):
    base = dict(symbol="XAUUSD", direction=1, volume=0.10,
                stop_loss=1990.0, entry_ref=2000.15, targets=[2030.0],
                atr=10.0, risk_money=103.0, alert_id="T1", bar_index=7,
                magic=260101)
    base.update(kw)
    return OrderRequest(**base)


def test_module_loads_and_reports_unavailable_without_library():
    ex = MT5Executor()          # no injection: real import must fail here
    with pytest.raises(MT5Unavailable):
        ex.account_state()


def test_symbol_spec_from_terminal_facts():
    spec = _executor(FakeMT5()).symbol_spec("XAUUSD")
    assert spec.tick_value == 1.0 and spec.tick_size == 0.01
    sizing = size_position(balance=10_000, risk_pct=0.01, entry=2000.0,
                            stop=1990.0, spec=spec)
    assert sizing.lots == pytest.approx(0.10)


def test_place_order_request_shape_and_filling_fallback():
    fake = FakeMT5()
    fake.reject_fillings = {FakeMT5.ORDER_FILLING_IOC}   # IOC refused first
    res = _executor(fake).place_order(_request())
    assert res.ok and res.ticket == 12341
    assert len(fake.sent) == 2                            # IOC, then FOK
    req = fake.sent[-1]
    assert req["type_filling"] == FakeMT5.ORDER_FILLING_FOK
    assert req["action"] == FakeMT5.TRADE_ACTION_DEAL
    assert req["type"] == FakeMT5.ORDER_TYPE_BUY
    assert req["price"] == pytest.approx(2000.30)         # the ASK
    assert req["sl"] == pytest.approx(1990.0)
    assert req["comment"] == "GS:T1"
    assert req["magic"] == 260101


def test_positions_round_trip_and_close():
    fake = FakeMT5()
    ex = _executor(fake)
    ex.place_order(_request())
    positions = ex.positions()
    assert len(positions) == 1
    p = positions[0]
    assert p.direction == 1 and p.alert_id == "T1"
    assert p.risk_money == pytest.approx((2000.30 - 1990.0) * 100 * 0.10)
    res = ex.close_position(p.ticket, volume=0.04)
    assert res.ok
    close_req = fake.sent[-1]
    assert close_req["type"] == FakeMT5.ORDER_TYPE_SELL  # opposite deal
    assert close_req["position"] == p.ticket
    assert close_req["volume"] == pytest.approx(0.04)
    assert close_req["price"] == pytest.approx(2000.00)  # long exits at BID


def test_modify_sl_uses_sltp_action():
    fake = FakeMT5()
    ex = _executor(fake)
    ticket = ex.place_order(_request()).ticket
    res = ex.modify_sl(ticket, 1995.0)
    assert res.ok
    req = fake.sent[-1]
    assert req["action"] == FakeMT5.TRADE_ACTION_SLTP
    assert req["sl"] == pytest.approx(1995.0)
    assert fake.positions_store[0].sl == pytest.approx(1995.0)


def test_account_state_from_terminal():
    acct = _executor(FakeMT5()).account_state()
    assert acct.balance == 10_000.0 and acct.login == 555001
    assert acct.connected


def test_copy_rates_frame():
    import pandas as pd

    fake = FakeMT5()
    fake.copy_rates_from_pos = lambda symbol, tf, start, count: [
        {"time": 1_700_000_000 + 3600 * i, "open": 2000.0 + i,
         "high": 2001.0 + i, "low": 1999.0 + i, "close": 2000.5 + i,
         "tick_volume": 100 + i} for i in range(3)]
    df = _executor(fake).copy_rates_h1("XAUUSD", 3)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex) and len(df) == 3
