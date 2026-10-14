"""MT5Executor — execution via the official MetaTrader5 Python
library (SPEC, owner correction 2026-10-10: NOT an EA).

Requirements of the live path, all of them on the HOST machine:

- Windows (the ``MetaTrader5`` package is Windows-only),
- an MT5 terminal installed, logged in (demo or real — whichever
  account the terminal is logged into is the account that trades),
  and ACTIVE.

The library is imported LAZILY: this module loads fine anywhere, and
every live call raises a precise error when the library or terminal
is unavailable instead of failing at import time.  Tests inject a
fake ``MetaTrader5`` module and exercise request-building, filling
-mode selection and sizing against it — the live order path itself
is code-complete but UNVERIFIED: it has never run against a real
terminal (there is none in the build environment).  See EXECUTION.md.

Filling mode: read from the symbol's ``filling_mode`` bitmask
(SYMBOL_FILLING_FOK=1, SYMBOL_FILLING_IOC=2) and tried in the order
IOC -> FOK -> RETURN, moving to the next mode when the broker
rejects the request — the standard deviation/filling handling.
"""
from __future__ import annotations

from .executor import (AccountState, Executor, OrderRequest, OrderResult,
                        PendingOrder, Position)
from .risk import SymbolSpec

#: retcodes that mean "the order went through"
_DONE = (10009, 10008)          # TRADE_RETCODE_DONE / DONE_PARTIAL->PLACED
_FILLING_FOK_BIT = 1            # SYMBOL_FILLING_FOK
_FILLING_IOC_BIT = 2            # SYMBOL_FILLING_IOC

#: "At the level" tolerance for order selection, as a fraction of ATR:
#: price within this of the entry level counts as AT the level.
_AT_LEVEL_ATR = 0.25


def select_order_type(*, direction: int, price: float,
                      entry_ref: float,
                      zone: dict | None = None,
                      atr: float = 0.0) -> tuple[str, float, str]:
    """Setup-dependent order selection — the real-trader entry rule.

    A real trader picks the order type by where price is vs the
    level when the signal fires, never always-market:

    - price AT the entry level (within ``_AT_LEVEL_ATR`` x ATR) with
      the trigger confirmed -> ``"market"`` at the current price
      (waiting would mean missing it);
    - entry level BEHIND price (long: below; short: above) ->
      ``"limit"`` at ``entry_ref`` (pullback setup: price must come
      back to the POI — the market comes to the order);
    - entry level AHEAD of price (long: above; short: below) ->
      ``"stop"`` at ``entry_ref`` (breakout setup: price must break
      through the level to enter);
    - price beyond the POI band on the ADVERSE side (long: below the
      zone bottom; short: above the zone top) -> ``"invalid"`` (the
      zone failed — the setup's premise is gone, no order).

    ``entry_ref`` is the playbook's POI entry level (FVG edge, QM
    level, trendline), NOT the current tick.  Returns
    ``(kind, level, reason)``; ``level`` is the pending price for
    limit/stop, the current price for market, ``0.0`` for invalid.
    A missing/zero ``entry_ref`` falls back to market (legacy).
    """
    if entry_ref <= 0:
        return "market", float(price), "no entry_ref: market at price"
    tol = _AT_LEVEL_ATR * float(atr) if atr and atr > 0 else 0.0
    if zone is not None:
        bottom = float(zone.get("bottom", 0.0))
        top = float(zone.get("top", 0.0))
        if direction > 0 and price < bottom - tol:
            return ("invalid", 0.0,
                    f"zone failed: price {price:.2f} below POI "
                    f"bottom {bottom:.2f} (long)")
        if direction < 0 and price > top + tol:
            return ("invalid", 0.0,
                    f"zone failed: price {price:.2f} above POI "
                    f"top {top:.2f} (short)")
    if abs(float(price) - float(entry_ref)) <= tol:
        return ("market", float(price),
                f"at level: price {price:.2f} within {tol:.2f} of "
                f"entry {entry_ref:.2f}, trigger confirmed")
    if direction > 0:
        if entry_ref < price - tol:
            return ("limit", float(entry_ref),
                    f"pullback: entry {entry_ref:.2f} below price "
                    f"{price:.2f} — limit at POI")
        return ("stop", float(entry_ref),
                f"breakout: entry {entry_ref:.2f} above price "
                f"{price:.2f} — stop beyond level")
    if entry_ref > price + tol:
        return ("limit", float(entry_ref),
                f"pullback: entry {entry_ref:.2f} above price "
                f"{price:.2f} — limit at POI")
    return ("stop", float(entry_ref),
            f"breakout: entry {entry_ref:.2f} below price "
            f"{price:.2f} — stop beyond level")


class MT5Unavailable(RuntimeError):
    """The MetaTrader5 library or terminal cannot be reached."""


class MT5Executor(Executor):
    def __init__(self, mt5_module=None, *, deviation: int = 20,
                 magic: int = 260101):
        self._injected = mt5_module
        self._mt5 = None
        self.deviation = int(deviation)
        self.magic = int(magic)

    # ------------------------------------------------------------- library
    def _lib(self):
        if self._mt5 is not None:
            return self._mt5
        if self._injected is not None:
            self._mt5 = self._injected
            return self._mt5
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise MT5Unavailable(
                "the MetaTrader5 Python package is not importable here — "
                "live execution requires a Windows host with the package "
                "installed and an MT5 terminal logged in and active "
                f"(import error: {exc})") from exc
        self._mt5 = mt5
        return mt5

    def connect(self, *, path: str | None = None, login: int | None = None,
                password: str | None = None,
                server: str | None = None) -> tuple[bool, str]:
        """Initialise the library against the running terminal.  With
        no credentials it attaches to the terminal as-is (the normal
        case: the owner logs the terminal in by hand)."""
        mt5 = self._lib()
        kwargs = {}
        if path:
            kwargs["path"] = path
        if login is not None:
            kwargs.update(login=int(login), password=password, server=server)
        if not mt5.initialize(**kwargs):
            return False, f"mt5.initialize failed: {mt5.last_error()}"
        info = mt5.terminal_info()
        if info is None or not getattr(info, "connected", False):
            return False, "terminal is not connected to a broker server"
        return True, f"connected (terminal build {getattr(info, 'build', '?')})"

    def shutdown(self) -> None:
        if self._mt5 is not None and self._injected is None:
            self._mt5.shutdown()

    def ping(self) -> tuple[bool, str]:
        """Lightweight terminal connectivity check for the watchdog.
        Returns (ok, message); never raises."""
        try:
            mt5 = self._lib()
            info = mt5.terminal_info()
            if info is None:
                return False, "terminal_info returned None"
            if not getattr(info, "connected", False):
                return False, "terminal not connected to broker server"
            return True, "terminal connected"
        except Exception as exc:
            return False, f"ping failed: {exc}"

    def copy_rates_h1(self, symbol: str, count: int = 1500):
        """The most recent ``count`` H1 bars as a DataFrame (UTC
        index, open/high/low/close/volume).  The LAST row is the
        still-forming bar — callers trade on closed bars only."""
        import pandas as pd

        mt5 = self._lib()
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, count)
        if rates is None or len(rates) == 0:
            raise MT5Unavailable(
                f"copy_rates_from_pos({symbol}) failed: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").rename(
            columns={"tick_volume": "volume"})
        return df[["open", "high", "low", "close", "volume"]]

    # -------------------------------------------------------- symbol facts
    def symbol_spec(self, symbol: str) -> SymbolSpec:
        mt5 = self._lib()
        info = mt5.symbol_info(symbol)
        if info is None:
            raise MT5Unavailable(f"symbol_info({symbol}) returned nothing — "
                                 "symbol unknown to the terminal")
        return SymbolSpec(
            name=symbol,
            tick_value=float(info.trade_tick_value),
            tick_size=float(info.trade_tick_size),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            digits=int(info.digits), point=float(info.point))

    def _filling_modes(self, symbol: str) -> list[int]:
        mt5 = self._lib()
        info = mt5.symbol_info(symbol)
        mask = int(getattr(info, "filling_mode", 0)) if info else 0
        modes: list[int] = []
        if mask & _FILLING_IOC_BIT:
            modes.append(mt5.ORDER_FILLING_IOC)
        if mask & _FILLING_FOK_BIT:
            modes.append(mt5.ORDER_FILLING_FOK)
        modes.append(mt5.ORDER_FILLING_RETURN)
        seen: list[int] = []
        for m in modes:
            if m not in seen:
                seen.append(m)
        return seen

    # --------------------------------------------------------- Executor API
    def account_state(self) -> AccountState:
        mt5 = self._lib()
        info = mt5.account_info()
        if info is None:
            raise MT5Unavailable(f"account_info failed: {mt5.last_error()}")
        term = mt5.terminal_info()
        return AccountState(
            balance=float(info.balance), equity=float(info.equity),
            free_margin=float(info.margin_free), currency=str(info.currency),
            connected=bool(getattr(term, "connected", False)) if term else False,
            login=int(info.login))

    def positions(self, symbol: str | None = None) -> list[Position]:
        mt5 = self._lib()
        raw = mt5.positions_get(symbol=symbol) if symbol \
            else mt5.positions_get()
        if raw is None:
            raise MT5Unavailable(f"positions_get failed: {mt5.last_error()}")
        out = []
        for p in raw:
            # ALL account positions are returned: exposure accounting in
            # the guards must see foreign risk too.  close_all() is the
            # operation that filters to our own magic.
            direction = 1 if int(p.type) == int(mt5.POSITION_TYPE_BUY) else -1
            comment = str(getattr(p, "comment", "") or "")
            alert_id = comment[3:] if comment.startswith("GS:") else ""
            spec = None
            risk = 0.0
            if p.sl:
                spec = self.symbol_spec(p.symbol)
                risk = spec.money_for_move(abs(p.price_open - p.sl), p.volume)
            out.append(Position(
                ticket=int(p.ticket), symbol=str(p.symbol),
                direction=direction, volume=float(p.volume),
                entry_price=float(p.price_open),
                sl=float(p.sl) if p.sl else None,
                tp=float(p.tp) if p.tp else None,
                magic=int(getattr(p, "magic", 0)), alert_id=alert_id,
                open_time=getattr(p, "time", None), risk_money=risk))
        return out

    def _send_with_fillings(self, base: dict) -> OrderResult:
        mt5 = self._lib()
        last: OrderResult | None = None
        for filling in self._filling_modes(base["symbol"]):
            request = dict(base, type_filling=filling)
            res = mt5.order_send(request)
            if res is None:
                last = OrderResult(False, message=f"order_send returned "
                                                  f"None: {mt5.last_error()}")
                continue
            if int(res.retcode) in _DONE:
                return OrderResult(
                    True, ticket=int(getattr(res, "order", 0) or
                                     getattr(res, "deal", 0)),
                    price=float(getattr(res, "price", 0.0)),
                    volume=float(getattr(res, "volume", base.get("volume", 0.0))),
                    message=str(getattr(res, "comment", "") or "done"),
                    retcode=int(res.retcode))
            last = OrderResult(
                False, price=float(getattr(res, "price", 0.0)),
                message=f"retcode {res.retcode}: "
                        f"{getattr(res, 'comment', '')}",
                retcode=int(res.retcode))
        return last or OrderResult(False, message="no filling mode attempted")

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Open one position — dispatching on ``request.order_kind``.

        ``"market"``: immediate DEAL at the current tick (the legacy
        path).  ``"limit"`` / ``"stop"``: pending BUY_LIMIT /
        SELL_LIMIT / BUY_STOP / SELL_STOP at ``order_price``
        (``entry_ref`` when ``order_price`` is 0), GTC, SL/TP
        attached.  A wrong-sided pending price (a BUY_LIMIT above the
        ask, a BUY_STOP below it, ...) is refused before it reaches
        the broker — the broker would reject it anyway, and a
        refusal here is a clean error, not a broker rejection.
        """
        kind = getattr(request, "order_kind", "market") or "market"
        if kind in ("limit", "stop"):
            return self._place_pending(request, kind)
        if kind != "market":
            return OrderResult(False, message=f"unknown order_kind {kind!r}")
        mt5 = self._lib()
        symbol = request.symbol
        if not mt5.symbol_select(symbol, True):
            return OrderResult(False, message=f"symbol_select({symbol}) failed")
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(False, message=f"no tick for {symbol}")
        price = float(tick.ask) if request.direction == 1 else float(tick.bid)
        comment = (f"GS:{request.alert_id}" if request.alert_id
                   else "GS")[:31]
        base = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(request.volume),
            "type": mt5.ORDER_TYPE_BUY if request.direction == 1
                    else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": float(request.stop_loss),
            "tp": float(request.take_profit) if request.take_profit else 0.0,
            "deviation": self.deviation,
            "magic": int(request.magic or self.magic),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        return self._send_with_fillings(base)

    def _place_pending(self, request: OrderRequest,
                       kind: str) -> OrderResult:
        """Pending LIMIT / STOP at the POI entry level.

        Single ``order_send`` (no filling-mode retries — filling modes
        do not apply to pending orders).  Success is retcode PLACED
        (10008): the broker holds the order; the position appears
        only when the market reaches the level.
        """
        mt5 = self._lib()
        symbol = request.symbol
        if not mt5.symbol_select(symbol, True):
            return OrderResult(False, message=f"symbol_select({symbol}) failed")
        price = float(request.order_price) if request.order_price \
            else float(request.entry_ref)
        if price <= 0:
            return OrderResult(False, message="pending order needs a price")
        # Wrong-sided pending prices are refused, not sent: a BUY_LIMIT
        # must sit BELOW the ask, a BUY_STOP ABOVE it (mirrored for
        # sells).  The selection logic guarantees the side; this is
        # the backstop against a broker rejection.
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            ask, bid = float(tick.ask), float(tick.bid)
            if request.direction == 1 and kind == "limit" \
                    and price >= ask:
                return OrderResult(
                    False, message=f"BUY_LIMIT price {price:.2f} not below "
                                   f"ask {ask:.2f}")
            if request.direction == 1 and kind == "stop" \
                    and price <= ask:
                return OrderResult(
                    False, message=f"BUY_STOP price {price:.2f} not above "
                                   f"ask {ask:.2f}")
            if request.direction == -1 and kind == "limit" \
                    and price <= bid:
                return OrderResult(
                    False, message=f"SELL_LIMIT price {price:.2f} not above "
                                   f"bid {bid:.2f}")
            if request.direction == -1 and kind == "stop" \
                    and price >= bid:
                return OrderResult(
                    False, message=f"SELL_STOP price {price:.2f} not below "
                                   f"bid {bid:.2f}")
        order_type = {
            (1, "limit"): mt5.ORDER_TYPE_BUY_LIMIT,
            (-1, "limit"): mt5.ORDER_TYPE_SELL_LIMIT,
            (1, "stop"): mt5.ORDER_TYPE_BUY_STOP,
            (-1, "stop"): mt5.ORDER_TYPE_SELL_STOP,
        }[(1 if request.direction > 0 else -1, kind)]
        comment = (f"GS:{request.alert_id}" if request.alert_id
                   else "GS")[:31]
        base = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(request.volume),
            "type": order_type,
            "price": price,
            "sl": float(request.stop_loss),
            "tp": float(request.take_profit) if request.take_profit else 0.0,
            "magic": int(request.magic or self.magic),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        res = mt5.order_send(base)
        if res is None:
            return OrderResult(False, message=f"order_send returned None: "
                                              f"{mt5.last_error()}")
        if int(res.retcode) in _DONE:
            return OrderResult(
                True, ticket=int(getattr(res, "order", 0) or 0),
                price=price, volume=float(request.volume),
                message=str(getattr(res, "comment", "") or "placed"),
                retcode=int(res.retcode))
        return OrderResult(
            False, price=price,
            message=f"retcode {res.retcode}: "
                    f"{getattr(res, 'comment', '')}",
            retcode=int(res.retcode))

    def pending_orders(self, symbol: str | None = None) -> list[PendingOrder]:
        """Pending (unfilled) broker orders — ``orders_get`` mapped to
        :class:`PendingOrder`.  Market DEALs never appear here (they
        become positions on fill)."""
        mt5 = self._lib()
        raw = mt5.orders_get(symbol=symbol) if symbol \
            else mt5.orders_get()
        if raw is None:
            raise MT5Unavailable(f"orders_get failed: {mt5.last_error()}")
        out: list[PendingOrder] = []
        for o in raw:
            t = int(o.type)
            if t == int(mt5.ORDER_TYPE_BUY_LIMIT):
                kind, direction = "limit", 1
            elif t == int(mt5.ORDER_TYPE_SELL_LIMIT):
                kind, direction = "limit", -1
            elif t == int(mt5.ORDER_TYPE_BUY_STOP):
                kind, direction = "stop", 1
            elif t == int(mt5.ORDER_TYPE_SELL_STOP):
                kind, direction = "stop", -1
            elif t == int(mt5.ORDER_TYPE_BUY):
                kind, direction = "market", 1
            elif t == int(mt5.ORDER_TYPE_SELL):
                kind, direction = "market", -1
            else:
                continue  # unknown order type: never invent a kind
            comment = str(getattr(o, "comment", "") or "")
            alert_id = comment[3:] if comment.startswith("GS:") else ""
            out.append(PendingOrder(
                ticket=int(o.ticket), symbol=str(o.symbol), kind=kind,
                direction=direction, price=float(o.price_open),
                volume=float(o.volume_current),
                sl=float(o.sl) if o.sl else None,
                tp=float(o.tp) if o.tp else None,
                magic=int(getattr(o, "magic", 0)), alert_id=alert_id))
        return out

    def cancel_pending(self, ticket: int) -> OrderResult:
        """Cancel one pending order (TRADE_ACTION_REMOVE).  A ticket
        that is already gone (filled / cancelled) reports failure —
        the caller reconciles against ``orders_get`` + positions to
        tell a fill from an external cancel."""
        mt5 = self._lib()
        res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE,
                              "order": int(ticket)})
        if res is None:
            return OrderResult(False, message=f"order_send returned None: "
                                              f"{mt5.last_error()}")
        if int(res.retcode) in _DONE:
            return OrderResult(True, ticket=int(ticket),
                               message=str(getattr(res, "comment", "")
                                           or "removed"),
                               retcode=int(res.retcode))
        return OrderResult(
            False, ticket=int(ticket),
            message=f"remove failed: retcode {res.retcode}: "
                    f"{getattr(res, 'comment', '')}",
            retcode=int(res.retcode))

    def modify_sl(self, ticket: int, sl: float,
                  tp: float | None = None) -> OrderResult:
        mt5 = self._lib()
        raw = mt5.positions_get(ticket=ticket)
        if not raw:
            return OrderResult(False, message=f"no position {ticket}")
        p = raw[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": p.symbol,
            "sl": float(sl),
            "tp": float(tp) if tp is not None else float(p.tp or 0.0),
        }
        res = mt5.order_send(request)
        if res is not None and int(res.retcode) in _DONE:
            return OrderResult(True, ticket=ticket,
                               message=str(getattr(res, "comment", "") or "done"),
                               retcode=int(res.retcode))
        return OrderResult(
            False, ticket=ticket,
            message=f"modify failed: {mt5.last_error() if res is None else res.comment}",
            retcode=int(res.retcode) if res is not None else None)

    def close_position(self, ticket: int,
                       volume: float | None = None) -> OrderResult:
        mt5 = self._lib()
        raw = mt5.positions_get(ticket=ticket)
        if not raw:
            return OrderResult(False, message=f"no position {ticket}")
        p = raw[0]
        vol = float(volume) if volume is not None else float(p.volume)
        vol = min(vol, float(p.volume))
        tick = mt5.symbol_info_tick(p.symbol)
        if tick is None:
            return OrderResult(False, message=f"no tick for {p.symbol}")
        is_long = int(p.type) == int(mt5.POSITION_TYPE_BUY)
        base = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": vol,
            "type": mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
            "position": int(ticket),
            "price": float(tick.bid) if is_long else float(tick.ask),
            "deviation": self.deviation,
            "magic": int(getattr(p, "magic", 0) or self.magic),
            "comment": "GS close",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        return self._send_with_fillings(base)

    def close_all(self, symbol: str | None = None) -> list[OrderResult]:
        return [self.close_position(p.ticket) for p in self.positions(symbol)
                if p.magic == self.magic]
