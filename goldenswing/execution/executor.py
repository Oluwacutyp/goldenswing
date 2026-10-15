"""The executor interface — one contract, two implementations.

``PaperExecutor`` simulates fills against bar data (works anywhere).
``MT5Executor`` drives the official MetaTrader5 Python library on a
Windows host with an active, logged-in terminal (SPEC: execution is
the MT5 python library, NOT an EA).  Both answer the same calls so
the callback handler and the runner never care which is live.

Nothing in this module imports MetaTrader5.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .risk import SymbolSpec


@dataclass
class OrderRequest:
    """Everything needed to open one position.  Built by the risk
    pipeline (handler) from an Alert: direction/stop/targets come from
    the signal, ``volume`` from the lot sizer, ``risk_money`` is the
    sizer's actual money-at-risk at that volume.

    ``order_kind`` selects HOW the broker is asked to enter (the
    setup-dependent order selection — a real trader picks the order
    type by the setup, never always-market):

    - ``"market"`` (default): immediate DEAL at the current tick.
    - ``"limit"``: pending BUY_LIMIT / SELL_LIMIT at ``order_price``
      (price must come back to the level — the pullback entry).
    - ``"stop"``: pending BUY_STOP / SELL_STOP at ``order_price``
      (price must break through the level — the breakout entry).

    For ``"limit"`` / ``"stop"``, ``order_price`` is the pending
    price (the POI entry level from the verdict/playbook
    ``entry_ref``, NOT the current tick); ``0.0`` falls back to
    ``entry_ref``.  Paper executors ignore these fields (they
    simulate an immediate fill) — only the MT5 executor acts on them.
    """

    symbol: str
    direction: int                  # +1 buy | -1 sell
    volume: float
    stop_loss: float                # structural invalidation
    entry_ref: float = 0.0          # price the sizing was computed from
    take_profit: float | None = None
    targets: list[float] = field(default_factory=list)  # pools ahead
    atr: float = 0.0
    risk_money: float = 0.0
    alert_id: str = ""
    bar_index: int = 0              # signal bar (hold-time accounting)
    magic: int = 0
    comment: str = ""
    order_kind: str = "market"      # "market" | "limit" | "stop"
    order_price: float = 0.0        # pending price (limit/stop only)
    is_scalp: bool = False          # scalp holding cap applies


@dataclass
class OrderResult:
    ok: bool
    ticket: int | None = None
    price: float = 0.0              # fill price (0 while a paper order pends)
    volume: float = 0.0
    message: str = ""
    retcode: int | None = None     # MT5 trade retcode, when applicable


@dataclass
class Position:
    ticket: int
    symbol: str
    direction: int                  # +1 long | -1 short
    volume: float
    entry_price: float
    sl: float | None = None
    tp: float | None = None
    magic: int = 0
    alert_id: str = ""
    open_time: object = None
    risk_money: float = 0.0         # money lost if the stop is hit


@dataclass
class PendingOrder:
    """A pending (not yet filled) broker order — a LIMIT or STOP
    placed by setup-dependent order selection, waiting for the
    market to come to it (limit) or break through it (stop).

    ``kind`` is ``"limit"`` | ``"stop"``; ``price`` is the pending
    price (the POI entry level).  A pending order is NOT a position:
    it has no entry price and no P&L until it fills, at which point
    the broker turns it into a position carrying the same
    ``alert_id`` (via the ``GS:<alert_id>`` comment)."""

    ticket: int
    symbol: str
    kind: str                       # "limit" | "stop"
    direction: int                  # +1 buy | -1 sell
    price: float                    # pending price (the POI entry level)
    volume: float
    sl: float | None = None
    tp: float | None = None
    magic: int = 0
    alert_id: str = ""


@dataclass
class AccountState:
    balance: float
    equity: float
    free_margin: float = 0.0
    currency: str = "USD"
    connected: bool = True
    login: int | None = None


class Executor(ABC):
    """The contract.  Implementations must be honest: a method that
    cannot do what it says returns ``OrderResult(ok=False, ...)``
    with the reason — never a fabricated success."""

    @abstractmethod
    def account_state(self) -> AccountState:
        ...

    @abstractmethod
    def symbol_spec(self, symbol: str) -> SymbolSpec:
        ...

    @abstractmethod
    def positions(self, symbol: str | None = None) -> list[Position]:
        ...

    @abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResult:
        ...

    @abstractmethod
    def modify_sl(self, ticket: int, sl: float,
                  tp: float | None = None) -> OrderResult:
        ...

    @abstractmethod
    def close_position(self, ticket: int,
                       volume: float | None = None) -> OrderResult:
        """Close ``volume`` (default: all) of one position."""
        ...

    @abstractmethod
    def close_all(self, symbol: str | None = None) -> list[OrderResult]:
        ...

    # --------------------------------------------------- pending orders
    # Default: unsupported (paper executors simulate immediate fills and
    # never hold pending orders).  The MT5 executor overrides both.
    # Deliberately NOT abstract: forcing every paper executor to
    # implement pending orders would touch the OANDA loops for no gain.
    def pending_orders(self, symbol: str | None = None) -> list["PendingOrder"]:
        """Pending (unfilled) broker orders with our filtering left to
        the caller.  Empty for executors that never hold pendings."""
        return []

    def cancel_pending(self, ticket: int) -> OrderResult:
        """Cancel one pending order by its broker ticket."""
        return OrderResult(False, message="pending orders unsupported")

    # ------------------------------------------------------------- shared
    def open_risk(self, spec: SymbolSpec | None = None) -> tuple[float, int]:
        """(money lost if every open stop is hit, #positions with no
        stop).  Exposure accounting the ATS way: it is the actual
        money at the stops, not margin, not notional."""
        total = 0.0
        unstopped = 0
        for p in self.positions():
            if p.sl is None:
                unstopped += 1
                continue
            sp = spec or self.symbol_spec(p.symbol)
            total += sp.money_for_move(abs(p.entry_price - p.sl), p.volume)
        return total, unstopped
