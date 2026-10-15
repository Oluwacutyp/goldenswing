"""Position exit management — the Phase-2 exit machinery as a live
stepper, shared by the paper simulator and the MT5 live manager.

The semantics are exactly ``backtest.harness.simulate_signal_v2``:

- the stop sits ``stop_buffer_atr`` x ATR beyond the raw structural
  invalidation (R is measured on that buffered stop);
- ``partial_frac`` of the position banks at the first liquidity pool
  at least ``partial_min_r`` away;
- once the best excursion reaches ``trail_trigger_r``, the remainder
  trails ``trail_dist_r`` behind the best excursion — levels are set
  from COMPLETED bars only, so a trail set by bar k is first active
  on bar k+1 (causal, identical to the backtest);
- the ``horizon``-bar time stop closes what is left;
- per bar, stops are checked BEFORE the partial (the backtest's
  conservative ordering), and the stop wins ties.

The stepper only DECIDES.  It emits :class:`ExitEvent`s carrying the
triggering level; pricing a fill (spread, gaps) is the executor's
business, and on MT5 the broker's own stop order is the last line of
defence between steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..backtest.harness import ExitConfig
from .executor import OrderRequest


@dataclass
class ExitEvent:
    kind: str               # "partial" | "close"
    reason: str             # "partial" | "stop" | "trail" | "timeout"
    volume: float           # lots to close with this event
    level: float | None     # triggering price level (None = market/close)
    wicked_inv: bool = False  # price crossed the RAW invalidation this trade


@dataclass
class TrackedPosition:
    ticket: int
    symbol: str
    direction: int
    volume: float               # remaining lots
    orig_volume: float
    entry_price: float
    signal_bar: int             # bar index of the signal
    entry_bar: int              # bar index of the fill
    invalidation: float         # raw structural level
    stop0: float                # buffered structural stop
    risk_price: float           # |entry - stop0|
    alert_id: str = ""
    #: The signal's target levels (first = the TP the close
    #: classifier measures against).  Adopted positions have none.
    targets: tuple[float, ...] = ()
    partial_price: float | None = None
    partial_done: bool = False
    best_r: float = 0.0
    trail_level: float | None = None
    wicked_inv: bool = False
    closed: bool = False
    #: ATR at entry (0.0 = unknown, e.g. adopted positions) — the
    #: Chandelier / profit-lock ATR legs need it.
    atr: float = 0.0
    #: Never partial this position (adopted: no target book, the
    #: setup is unknown).  Distinct from ``partial_done`` ("a partial
    #: was taken"), which tiered partials also set.
    no_partial: bool = False
    #: Tiered partials fired so far (TickManager only).
    tiers_done: int = 0
    #: Worst adverse excursion in R (panic-candle accounting).
    worst_r: float = 0.0
    #: Scalp position (holding cap applies).
    is_scalp: bool = False
    #: Wall-clock open time (epoch seconds) for the scalp cap.
    opened_at: float | None = None
    #: Min-lot profit-lock applied (one-shot).
    profit_lock_done: bool = False

    @property
    def active_stop(self) -> float:
        stop = self.stop0
        if self.trail_level is not None:
            if (self.direction == 1 and self.trail_level > stop) or \
                    (self.direction == -1 and self.trail_level < stop):
                stop = self.trail_level
        return stop

    @property
    def stop_is_trail(self) -> bool:
        return self.active_stop == self.trail_level \
            and self.trail_level is not None


class ExitStepper:
    """Owns the exit state of every tracked position."""

    def __init__(self, exits: ExitConfig | None = None):
        self.exits = exits or ExitConfig()

    def open(self, *, ticket: int, request: OrderRequest,
             fill_price: float, entry_bar: int) -> TrackedPosition:
        ex = self.exits
        d = request.direction
        buffer = ex.stop_buffer_atr * request.atr if request.atr > 0 else 0.0
        stop0 = request.stop_loss - d * buffer
        risk = (fill_price - stop0) * d
        partial_price: float | None = None
        if risk > 0 and ex.partial_frac > 0:
            ahead = sorted(
                (p for p in request.targets
                 if (p - fill_price) * d >= ex.partial_min_r * risk),
                key=lambda p: abs(p - fill_price))
            if ahead:
                partial_price = float(ahead[0])
        return TrackedPosition(
            ticket=ticket, symbol=request.symbol, direction=d,
            alert_id=request.alert_id,
            volume=request.volume, orig_volume=request.volume,
            entry_price=fill_price, signal_bar=request.bar_index,
            entry_bar=entry_bar, invalidation=request.stop_loss,
            stop0=stop0, risk_price=risk,
            targets=tuple(request.targets or ()),
            partial_price=partial_price, atr=float(request.atr))

    def step(self, pos: TrackedPosition, bar: dict,
             bar_index: int) -> list[ExitEvent]:
        """Advance one position over one bar.  ``bar`` needs
        open/high/low/close.  Mutates ``pos`` (trail, partial flag,
        remaining volume on emitted events is the caller's to apply
        via :meth:`apply`)."""
        if pos.closed or pos.risk_price <= 0:
            return []
        ex = self.exits
        d = pos.direction
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]

        if (d == 1 and l < pos.invalidation) or (d == -1 and h > pos.invalidation):
            pos.wicked_inv = True

        # 1. stops first — buffered structural stop or the trail level
        #    set from bars BEFORE this one.
        stop = pos.active_stop
        stop_hit = l <= stop if d == 1 else h >= stop
        if stop_hit:
            pos.closed = True
            return [ExitEvent(
                "close", "trail" if pos.stop_is_trail else "stop",
                pos.volume, stop, pos.wicked_inv)]

        events: list[ExitEvent] = []
        # 2. partial bank at the first liquidity pool
        if not pos.partial_done and pos.partial_price is not None:
            crossed = h >= pos.partial_price if d == 1 else l <= pos.partial_price
            if crossed:
                vol = ex.partial_frac * pos.orig_volume
                vol = min(vol, pos.volume)
                pos.partial_done = True
                events.append(ExitEvent("partial", "partial", vol,
                                        pos.partial_price, pos.wicked_inv))
        # 3. trailing update from THIS bar's excursion (active next bar)
        fav = h if d == 1 else l
        pos.best_r = max(pos.best_r, d * (fav - pos.entry_price) / pos.risk_price)
        if pos.best_r >= ex.trail_trigger_r:
            lvl = pos.entry_price + d * (pos.best_r - ex.trail_dist_r) * pos.risk_price
            if pos.trail_level is None \
                    or (d == 1 and lvl > pos.trail_level) \
                    or (d == -1 and lvl < pos.trail_level):
                pos.trail_level = lvl
        # 4. time stop: horizon bars after the SIGNAL bar (backtest parity)
        if bar_index - pos.signal_bar >= ex.horizon:
            pos.closed = True
            events.append(ExitEvent("close", "timeout", pos.volume, None,
                                    pos.wicked_inv))
        return events

    @staticmethod
    def apply(pos: TrackedPosition, event: ExitEvent) -> None:
        """Reduce the tracked volume after an event was executed."""
        if event.kind == "partial":
            pos.volume = max(0.0, pos.volume - event.volume)
        elif event.kind == "close":
            pos.volume = 0.0
