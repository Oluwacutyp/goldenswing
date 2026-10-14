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
    sizer's actual money-at-risk at that volume."""

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
