"""PaperExecutor — a simulated broker driven by bars.

Orders placed now fill at the NEXT bar's open, adjusted by half the
round-trip spread cost (the other half is charged on exit) — the same
$0.45/oz cost model as the backtest harness.  Open positions are
managed by the shared Phase-2 :class:`ExitStepper` (buffered
structural stop, partial bank at the first pool, trail after +1.5R,
72-bar time stop), so a paper trade lives and dies by exactly the
rules the backtest validated.

Every close is reported through the ``on_close`` callback as a
:class:`CloseRecord` (and returned from :meth:`advance` as events) —
that is the feed the trade journal and the risk guards consume.
Nothing here pretends to be a real fill: it is a simulation, labelled
as one everywhere it surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..backtest.harness import COST_PER_OZ, ExitConfig
from .executor import (AccountState, Executor, OrderRequest, OrderResult,
                        Position)
from .manager import ExitEvent, ExitStepper, TrackedPosition
from .risk import SymbolSpec, gold_spec


@dataclass
class CloseRecord:
    ticket: int
    alert_id: str
    symbol: str
    direction: int
    entry_price: float
    exit_price: float           # final tranche's exit price
    reason: str                 # stop | trail | timeout | manual
    pnl_money: float            # total, incl. banked partials
    r_net: float                # pnl / initial money at risk
    hold_bars: int              # exit bar - signal bar (backtest parity)
    partial_taken: bool = False
    wicked_inv: bool = False


@dataclass
class _Meta:
    alert_id: str = ""
    magic: int = 0
    open_time: object = None
    banked_pnl: float = 0.0
    initial_risk_money: float = 0.0
    final_reason: str = ""


class PaperExecutor(Executor):
    def __init__(self, spec: SymbolSpec | None = None, *,
                 balance: float = 10_000.0,
                 spread_cost: float = COST_PER_OZ,
                 exits: ExitConfig | None = None,
                 on_close=None, on_fill=None):
        self._spec = spec or gold_spec()
        self._balance = float(balance)
        self._half_spread = float(spread_cost) / 2.0
        self._stepper = ExitStepper(exits)
        self.on_close = on_close      # callable(CloseRecord) | None
        self.on_fill = on_fill        # callable(Position) | None
        self._next_ticket = 1
        self._pending_orders: list[tuple[int, OrderRequest]] = []
        self._pending_closes: list[tuple[int, float | None]] = []
        self._tracked: dict[int, TrackedPosition] = {}
        self._meta: dict[int, _Meta] = {}
        self._last_close: dict[str, float] = {}
        self.closed: list[CloseRecord] = []

    # ------------------------------------------------------------ helpers
    def _money(self, distance: float, volume: float) -> float:
        return self._spec.money_for_move(distance, volume)

    def _exit_price(self, raw: float, direction: int) -> float:
        return raw - direction * self._half_spread

    # -------------------------------------------------------- Executor API
    def account_state(self) -> AccountState:
        floating = 0.0
        for ticket, pos in self._tracked.items():
            last = self._last_close.get(pos.symbol, pos.entry_price)
            floating += pos.direction * (last - pos.entry_price) \
                / self._spec.tick_size * self._spec.tick_value * pos.volume
        return AccountState(balance=self._balance,
                            equity=self._balance + floating,
                            free_margin=self._balance + floating)

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        if symbol != self._spec.name:
            raise KeyError(f"paper executor trades {self._spec.name}, "
                           f"not {symbol}")
        return self._spec

    def positions(self, symbol: str | None = None) -> list[Position]:
        out = []
        for ticket, pos in self._tracked.items():
            if symbol is not None and pos.symbol != symbol:
                continue
            meta = self._meta[ticket]
            out.append(Position(
                ticket=ticket, symbol=pos.symbol, direction=pos.direction,
                volume=pos.volume, entry_price=pos.entry_price,
                sl=pos.active_stop, magic=meta.magic,
                alert_id=meta.alert_id, open_time=meta.open_time,
                risk_money=self._money(abs(pos.entry_price - pos.active_stop),
                                       pos.volume)))
        return out

    def place_order(self, request: OrderRequest) -> OrderResult:
        if request.direction not in (1, -1):
            return OrderResult(False, message=f"bad direction {request.direction}")
        if request.volume <= 0:
            return OrderResult(False, message="volume must be positive")
        if request.entry_ref and \
                (request.entry_ref - request.stop_loss) * request.direction <= 0:
            return OrderResult(
                False, message="stop is on the wrong side of the entry "
                               "reference — no honest size exists")
        ticket = self._next_ticket
        self._next_ticket += 1
        request = OrderRequest(**{**request.__dict__})  # copy; caller keeps theirs
        self._pending_orders.append((ticket, request))
        return OrderResult(True, ticket=ticket, volume=request.volume,
                           message="queued — fills at next bar open (paper)")

    def modify_sl(self, ticket: int, sl: float,
                  tp: float | None = None) -> OrderResult:
        pos = self._tracked.get(ticket)
        if pos is None:
            return OrderResult(False, message=f"no open position {ticket}")
        pos.stop0 = float(sl)   # manual stop replaces the structural one
        return OrderResult(True, ticket=ticket, message="stop updated (paper)")

    def close_position(self, ticket: int,
                       volume: float | None = None) -> OrderResult:
        if ticket not in self._tracked:
            return OrderResult(False, message=f"no open position {ticket}")
        self._pending_closes.append((ticket, volume))
        return OrderResult(True, ticket=ticket,
                           message="queued — closes at next bar open (paper)")

    def close_all(self, symbol: str | None = None) -> list[OrderResult]:
        return [self.close_position(p.ticket)
                for p in self.positions(symbol)]

    # ------------------------------------------------------------ bar pump
    def advance(self, bar: dict) -> list[dict]:
        """Feed one bar: fill pending orders at its open, execute
        pending closes at its open, then manage every open position
        through the bar.  Returns event dicts (fills/partials/closes).
        """
        idx = int(bar["index"])
        o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
        self._last_close[self._spec.name] = c
        events: list[dict] = []

        # 1. manual closes at the open
        for ticket, volume in self._pending_closes:
            pos = self._tracked.get(ticket)
            if pos is None:
                continue
            raw = o
            self._close_volume(pos, volume, self._exit_price(raw, pos.direction),
                               "manual", idx, events)
        self._pending_closes.clear()

        # 2. pending orders fill at the open
        for ticket, req in self._pending_orders:
            fill = o + req.direction * self._half_spread
            pos = self._stepper.open(ticket=ticket, request=req,
                                     fill_price=fill, entry_bar=idx)
            self._tracked[ticket] = pos
            self._meta[ticket] = _Meta(
                alert_id=req.alert_id, magic=req.magic, open_time=bar.get("time"),
                initial_risk_money=self._money(pos.risk_price, pos.volume))
            events.append({"kind": "fill", "ticket": ticket,
                           "price": fill, "volume": pos.volume})
            if self.on_fill is not None:
                self.on_fill([p for p in self.positions()
                              if p.ticket == ticket][0])
        self._pending_orders.clear()

        # 3. manage open positions through this bar
        for ticket in list(self._tracked):
            pos = self._tracked[ticket]
            for ev in self._stepper.step(pos, bar, idx):
                self._price_event(pos, ev, bar, events)
        return events

    # ------------------------------------------------------------- pricing
    def _price_event(self, pos: TrackedPosition, ev: ExitEvent,
                     bar: dict, events: list[dict]) -> None:
        d = pos.direction
        o = float(bar["open"])
        if ev.kind == "partial":
            raw = ev.level
            if (d == 1 and o > ev.level) or (d == -1 and o < ev.level):
                raw = o                      # gapped through: take the open
            price = self._exit_price(raw, d)
            pnl = d * (price - pos.entry_price) \
                / self._spec.tick_size * self._spec.tick_value * ev.volume
            self._meta[pos.ticket].banked_pnl += pnl
            self._balance += pnl
            self._stepper.apply(pos, ev)
            events.append({"kind": "partial", "ticket": pos.ticket,
                           "price": price, "volume": ev.volume, "pnl": pnl})
            return
        # close
        if ev.level is None:
            raw = float(bar["close"])        # timeout: the bar's close
        else:
            raw = ev.level
            if (d == 1 and o < ev.level) or (d == -1 and o > ev.level):
                raw = o                      # gapped through the stop
        price = self._exit_price(raw, d)
        self._close_volume(pos, None, price, ev.reason, int(bar["index"]),
                           events, wicked=ev.wicked_inv)

    def _close_volume(self, pos: TrackedPosition, volume: float | None,
                      price: float, reason: str, bar_index: int,
                      events: list[dict], wicked: bool = False) -> None:
        meta = self._meta[pos.ticket]
        vol = pos.volume if volume is None else min(volume, pos.volume)
        pnl = pos.direction * (price - pos.entry_price) \
            / self._spec.tick_size * self._spec.tick_value * vol
        self._balance += pnl
        pos.volume -= vol
        events.append({"kind": "close", "ticket": pos.ticket, "price": price,
                       "volume": vol, "pnl": pnl, "reason": reason})
        if pos.volume > 1e-9:
            return                          # partial manual close: still open
        total_pnl = meta.banked_pnl + pnl
        record = CloseRecord(
            ticket=pos.ticket, alert_id=meta.alert_id, symbol=pos.symbol,
            direction=pos.direction, entry_price=pos.entry_price,
            exit_price=price, reason=reason, pnl_money=total_pnl,
            r_net=(total_pnl / meta.initial_risk_money
                   if meta.initial_risk_money > 0 else 0.0),
            hold_bars=bar_index - pos.signal_bar,
            partial_taken=pos.partial_done, wicked_inv=pos.wicked_inv or wicked)
        self.closed.append(record)
        del self._tracked[pos.ticket]
        del self._meta[pos.ticket]
        if self.on_close is not None:
            self.on_close(record)

    # ------------------------------------------------------- persistence
    def export_state(self) -> dict:
        """JSON-serializable snapshot for cron --once ticks.

        Carries everything ``advance`` needs to keep managing open
        positions across processes: the exit-stepper state per tracked
        position, pending orders/closes, balance and the ticket counter.
        The journal stays the record of closed trades — ``closed`` is
        deliberately NOT persisted (it is rebuilt from the journal).
        """
        from dataclasses import asdict

        def _clean(value):
            # Timestamps -> ISO strings; everything else must already
            # be JSON-safe (floats/ints/strs/bools/None/lists/dicts).
            if hasattr(value, "isoformat"):
                try:
                    return value.isoformat()
                except Exception:
                    return str(value)
            return value

        return {
            "balance": self._balance,
            "next_ticket": self._next_ticket,
            "tracked": {str(t): {k: _clean(v)
                                 for k, v in asdict(pos).items()}
                        for t, pos in self._tracked.items()},
            "meta": {str(t): {k: _clean(v)
                              for k, v in asdict(m).items()}
                     for t, m in self._meta.items()},
            "pending_orders": [
                (t, asdict(req)) for t, req in self._pending_orders],
            "pending_closes": [[t, v] for t, v in self._pending_closes],
            "last_close": dict(self._last_close),
        }

    def import_state(self, state: dict | None) -> None:
        """Restore a snapshot from :meth:`export_state` (cron resume)."""
        if not state:
            return
        tracked, meta = {}, {}
        for t, pos in (state.get("tracked") or {}).items():
            tracked[int(t)] = TrackedPosition(**pos)
        for t, m in (state.get("meta") or {}).items():
            meta[int(t)] = _Meta(**m)
        self._tracked = tracked
        self._meta = meta
        self._balance = float(state.get("balance", self._balance))
        self._next_ticket = int(state.get("next_ticket",
                                          self._next_ticket))
        self._pending_orders = [
            (int(t), OrderRequest(**req))
            for t, req in (state.get("pending_orders") or [])]
        self._pending_closes = [
            (int(t), v) for t, v in (state.get("pending_closes") or [])]
        self._last_close = dict(state.get("last_close") or {})

    def shift_bar_indices(self, delta: int) -> None:
        """Shift positional bar indices by ``delta`` (buffer rebase).

        The exit stepper's time-stop counts bars positionally; when the
        bar buffer is rebased (old bars dropped), every stored index
        moves by the same delta so hold-time arithmetic is preserved.
        """
        if not delta:
            return
        for pos in self._tracked.values():
            pos.signal_bar += delta
            pos.entry_bar += delta
        for _t, req in self._pending_orders:
            req.bar_index += delta
