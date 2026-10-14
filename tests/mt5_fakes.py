"""Fake MetaTrader5 module for MT5 tests (no real terminal here).

Mimics the real API surface the feed / executor / runner use:
initialize, terminal_info, symbol_select, symbol_info,
symbol_info_tick, copy_rates_from_pos, copy_ticks_range,
account_info, positions_get, order_send — plus the TIMEFRAME_*,
COPY_TICKS_ALL, TRADE_ACTION_* and ORDER_* constants.

Bars are deterministic (gentle uptrend) with server-time stamps
(``time.time() + server_offset_seconds``), so the feed's UTC-offset
calibration is exercised.  Ticks come from a scriptable queue:
each ``symbol_info_tick`` call consumes the next (bid, ask,
time_offset) entry; when the script is exhausted the last price
repeats with time advancing 1s per call.
"""
import time
from types import SimpleNamespace

import numpy as np

_RATES_DTYPE = [
    ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
    ("close", "f8"), ("tick_volume", "i8"), ("real_volume", "i8"),
    ("spread", "i8"),
]

_TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
               "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800,
               "MN1": 2592000}


class FakeMT5:
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16386
    TIMEFRAME_D1 = 16408
    TIMEFRAME_W1 = 32769
    TIMEFRAME_MN1 = 49153
    COPY_TICKS_ALL = 0
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_REMOVE = 7
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008

    _TF_BY_CONST = {v: k for k, v in [
        ("M1", 1), ("M5", 5), ("M15", 15), ("M30", 30),
        ("H1", 16385), ("H4", 16386), ("D1", 16408),
        ("W1", 32769), ("MN1", 49153)]}

    def __init__(self, *, server_offset_seconds: int = 0,
                 base_price: float = 2000.0,
                 trade_mode: int = 0,
                 real_volume_on: bool = False,
                 tick_script: list | None = None,
                 symbols: list | None = None):
        self.server_offset_seconds = server_offset_seconds
        self.base_price = base_price
        self.trade_mode = trade_mode
        self.real_volume_on = real_volume_on
        self.tick_script = list(tick_script) if tick_script else []
        # Broker symbol universe: symbol_select only succeeds for a
        # listed symbol (default: plain XAUUSD, like most brokers).
        self.symbols = list(symbols) if symbols is not None else ["XAUUSD"]
        self._tick_calls = 0
        # Fixed base for scripted ticks: identical script entries must
        # carry identical timestamps (else pump dedup is untestable).
        self._script_base = int(time.time()) + server_offset_seconds
        self.sent: list[dict] = []
        self.positions_store: list[SimpleNamespace] = []
        # Pending (unfilled) broker orders, keyed by ticket — the
        # orders_get surface the pending-order lifecycle tests use.
        self.orders_store: list[SimpleNamespace] = []
        self._ticket = 5000
        self.shutdown_called = False

    # ------------------------------------------------------------- terminal
    def initialize(self, **kw):
        return True

    def shutdown(self):
        self.shutdown_called = True

    def last_error(self):
        return (0, "ok")

    def terminal_info(self):
        return SimpleNamespace(connected=True, build=3815)

    def symbol_select(self, symbol, flag):
        return symbol in self.symbols

    def symbols_get(self):
        return [SimpleNamespace(name=n) for n in self.symbols]

    def symbol_info(self, symbol):
        return SimpleNamespace(
            trade_tick_value=1.0, trade_tick_size=0.01, volume_min=0.01,
            volume_max=50.0, volume_step=0.01, digits=2, point=0.01,
            filling_mode=3)

    # ---------------------------------------------------------------- ticks
    def _tick_now(self):
        if self._tick_calls < len(self.tick_script):
            bid, ask, off = self.tick_script[self._tick_calls]
            t = self._script_base + off
        else:
            bid = self.base_price
            ask = self.base_price + 0.30
            if self.tick_script:
                bid, ask, _ = self.tick_script[-1]
            t = self._script_base + self._tick_calls
        self._tick_calls += 1
        return bid, ask, t

    def symbol_info_tick(self, symbol):
        bid, ask, t = self._tick_now()
        return SimpleNamespace(bid=bid, ask=ask, time=t,
                               time_msc=t * 1000)

    def copy_ticks_range(self, symbol, start, end, flags):
        import datetime as _dt
        start_ts = int(start.timestamp()) if hasattr(start, "timestamp") \
            else int(start)
        rows = []
        for i in range(3):
            t = start_ts + i
            rows.append((t, self.base_price + i * 0.01,
                         self.base_price + 0.30 + i * 0.01,
                         1, 0))
        return np.array(rows, dtype=[("time", "i8"), ("bid", "f8"),
                                     ("ask", "f8"), ("volume", "i8"),
                                     ("flags", "i8")])

    # ----------------------------------------------------------------- bars
    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        tf = self._TF_BY_CONST.get(int(timeframe), "H1")
        tf_sec = _TF_SECONDS[tf]
        total = int(count) + int(start_pos)
        end = int(time.time()) + self.server_offset_seconds
        rows = []
        for i in range(total):
            # Ascending: oldest first, newest last (position 0).
            t = end - (total - 1 - i) * tf_sec
            p = self.base_price + i * 0.10 + 0.05 * ((i % 7) - 3) * 0.1
            vol = 100 + (i % 37)
            rows.append((t, p, p + 0.15, p - 0.10, p + 0.05, vol,
                         vol // 2 if self.real_volume_on else 0, 30))
        arr = np.array(rows, dtype=_RATES_DTYPE)
        if start_pos:
            return arr[len(arr) - int(start_pos) - int(count):
                       len(arr) - int(start_pos)]
        return arr[len(arr) - int(count):]

    def copy_rates_range(self, symbol, timeframe, start, end):
        return self.copy_rates_from_pos(symbol, timeframe, 0, 10)

    # ------------------------------------------------------- account / pos
    def account_info(self):
        return SimpleNamespace(balance=10000.0, equity=10000.0,
                               margin=0.0, margin_free=10000.0,
                               currency="USD", login=123456,
                               leverage=500, trade_mode=self.trade_mode)

    def positions_get(self, symbol=None, ticket=None):
        out = [p for p in self.positions_store
               if (symbol is None or p.symbol == symbol)
               and (ticket is None or p.ticket == ticket)]
        return out

    def orders_get(self, symbol=None, ticket=None):
        return [o for o in self.orders_store
                if (symbol is None or o.symbol == symbol)
                and (ticket is None or o.ticket == ticket)]

    def fill_pending(self, ticket, price):
        """Simulate the broker filling a pending order: the order
        leaves orders_get and a position with the same comment
        (GS:<alert_id>) appears — exactly what the runner's
        reconcile uses to recognise a fill.  Returns True on fill."""
        for o in list(self.orders_store):
            if o.ticket == ticket:
                self.orders_store.remove(o)
                # A filled pending becomes a real position: the
                # position type is BUY/SELL, never the order type.
                pos_type = self.POSITION_TYPE_BUY \
                    if o.type in (self.ORDER_TYPE_BUY_LIMIT,
                                  self.ORDER_TYPE_BUY_STOP) \
                    else self.POSITION_TYPE_SELL
                self.positions_store.append(SimpleNamespace(
                    ticket=o.ticket, symbol=o.symbol, type=pos_type,
                    volume=o.volume_current, price_open=price,
                    sl=o.sl, tp=o.tp, magic=o.magic,
                    comment=o.comment, time=0))
                return True
        return False

    def order_send(self, request):
        self.sent.append(dict(request))
        action = request.get("action")
        if action == self.TRADE_ACTION_SLTP:
            for p in self.positions_store:
                if p.ticket == request["position"]:
                    p.sl = request.get("sl", p.sl)
                    p.tp = request.get("tp", p.tp)
            return SimpleNamespace(retcode=10009, comment="done",
                                   order=request["position"],
                                   deal=request["position"],
                                   price=0.0, volume=0.0)
        if action == self.TRADE_ACTION_DEAL and "position" in request:
            # Close (full or partial) of an open position.
            for p in list(self.positions_store):
                if p.ticket == request["position"]:
                    p.volume = round(p.volume - request["volume"], 4)
                    if p.volume <= 0:
                        self.positions_store.remove(p)
            return SimpleNamespace(retcode=10009, comment="done",
                                   order=request["position"],
                                   deal=request["position"],
                                   price=request.get("price", 0.0),
                                   volume=request.get("volume", 0.0))
        if action == self.TRADE_ACTION_DEAL:
            self._ticket += 1
            self.positions_store.append(SimpleNamespace(
                ticket=self._ticket, symbol=request["symbol"],
                type=request["type"], volume=request["volume"],
                price_open=request["price"], sl=request.get("sl", 0.0),
                tp=request.get("tp", 0.0), magic=request.get("magic", 0),
                comment=request.get("comment", ""), time=0))
            return SimpleNamespace(retcode=10009, comment="done",
                                   order=self._ticket, deal=self._ticket,
                                   price=request.get("price", 0.0),
                                   volume=request.get("volume", 0.0))
        if action == self.TRADE_ACTION_PENDING:
            self._ticket += 1
            self.orders_store.append(SimpleNamespace(
                ticket=self._ticket, symbol=request["symbol"],
                type=request["type"], volume_current=request["volume"],
                price_open=request["price"], sl=request.get("sl", 0.0),
                tp=request.get("tp", 0.0), magic=request.get("magic", 0),
                comment=request.get("comment", ""), time=0))
            return SimpleNamespace(retcode=10008, comment="placed",
                                   order=self._ticket, deal=0,
                                   price=request.get("price", 0.0),
                                   volume=request.get("volume", 0.0))
        if action == self.TRADE_ACTION_REMOVE:
            for o in list(self.orders_store):
                if o.ticket == request["order"]:
                    self.orders_store.remove(o)
                    return SimpleNamespace(retcode=10009, comment="removed",
                                           order=o.ticket, deal=0,
                                           price=0.0, volume=0.0)
            return SimpleNamespace(retcode=10006, comment="order not found",
                                   order=0, deal=0, price=0.0,
                                   volume=0.0)
        return SimpleNamespace(retcode=10009, comment="done",
                               order=0, deal=0, price=0.0, volume=0.0)
