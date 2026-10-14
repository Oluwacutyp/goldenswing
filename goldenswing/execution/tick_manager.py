"""Tick-driven trade management — breakeven, trailing, partials and
stops react on the TICK that crosses the level, never on the next
poll or bar.

Why this module exists: the bar-driven :class:`ExitStepper
<goldenswing.execution.manager.ExitStepper>` sets trail levels from
COMPLETED bars only ("a trail set by bar k is first active on bar
k+1") and has NO breakeven move at all.  On a 5-minute poll that
means up to 5 minutes to notice a breakeven or an entry — the exact
failure the owner corrected (2026-10-14).  This manager runs on the
MT5 tick pump instead: every tick re-evaluates the position, and the
caller executes the returned actions immediately (``modify_sl`` /
partial close via the executor) on the same tick.

Layers — scale-in onto ONE position (the owner's "I start
layering"): a layer is an added entry in the same direction with its
own entry / volume / SL / provenance.  The base layer is the first
entry; added layers ride on the same netted MT5 position (same
symbol / direction / magic — the broker nets the deal into the one
position).  The broker SL is the BASE layer's stop (the catastrophic
backstop); added-layer SLs are enforced engine-side by
``layer_stop`` partial closes on the tick that crosses them.  The
base layer's stop is the broker SL, so it never gets an engine-side
layer-stop — when the broker SL is hit, the whole position dies
(``stop_hit``).  ``close_top_layer`` is the manual trim: close the
highest (most recent) layer early while the base rides.

Design (two-phase, execution-safe):

- :meth:`TickManager.on_tick` DECIDES: it updates the observed best
  excursion / wicked-invalidation flag and returns
  :class:`TickAction`\\ s, but does NOT commit execution-dependent
  state (partial fills, SL moves, closes).
- :meth:`TickManager.apply` COMMITS an action after the executor
  reports success.  A failed modify/close is retried on the next
  tick instead of being booked as done.
- The broker's own stop order (placed at the buffered stop0) is the
  primary stop.  A ``stop_hit`` action is the bookkeeping /
  safety-net path: the caller reconciles against
  ``positions_get`` and, only if the position is still open,
  closes at market (the broker SL should already have fired).
- Exit arithmetic reuses :class:`ExitStepper` (same
  ``ExitConfig``: buffered stop, partial at the first pool >=
  ``partial_min_r``, trail ``trail_dist_r`` behind the best
  excursion once ``trail_trigger_r`` is reached) — the tick manager
  changes the RESOLUTION (tick, not bar), not the exit rules.
  Breakeven is the one rule the stepper lacks, so it lives here:
  at ``breakeven_trigger_r`` the SL moves to entry (+
  ``breakeven_offset_r``), on that tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..backtest.harness import ExitConfig
from .executor import OrderRequest
from .manager import ExitStepper, TrackedPosition


@dataclass(frozen=True)
class TickConfig:
    #: Favourable excursion (R) at which the stop moves to breakeven.
    breakeven_trigger_r: float = 1.0
    #: Breakeven SL = entry + direction * breakeven_offset_r * risk.
    #: 0.0 = exactly entry (pure breakeven).
    breakeven_offset_r: float = 0.0


@dataclass
class Layer:
    """One entry leg of a position.  The base layer is layers[0]."""

    entry: float
    volume: float
    sl: float                   # the layer's own stop (engine-side)
    alert_id: str = ""
    playbook: str = ""


@dataclass(frozen=True)
class TickAction:
    kind: str          # "modify_sl" | "partial" | "stop_hit" | "timeout" |
                       # "layer_stop" | "trim"
    level: float | None = None   # SL level (modify_sl) / trigger (others)
    volume: float = 0.0          # lots to close (partial / stop / timeout /
                                 # layer_stop / trim)
    reason: str = ""             # "breakeven" | "trail" | "partial" |
                                 # "stop" | "timeout" | "layer_stop" | "trim"


@dataclass
class _TickState:
    pos: TrackedPosition
    breakeven_done: bool = False
    #: SL actually on the broker (breakeven/trailing moves only tighten).
    broker_sl: float | None = None
    #: Entry legs: base first, added layers after.
    layers: list = field(default_factory=list)

    @property
    def effective_stop(self) -> float:
        if self.broker_sl is not None:
            return self.broker_sl
        return self.pos.active_stop


class TickManager:
    """Per-tick exit management for live MT5 positions."""

    def __init__(self, exits: ExitConfig | None = None,
                 tick: TickConfig | None = None) -> None:
        self.exits = exits or ExitConfig()
        self.tick_cfg = tick or TickConfig()
        self._stepper = ExitStepper(self.exits)
        self._states: dict[int, _TickState] = {}

    # ------------------------------------------------------------ lifecycle
    @staticmethod
    def planned_stop(request: OrderRequest,
                     exits: ExitConfig | None = None) -> float:
        """Broker SL for a new order: the BUFFERED structural stop.

        The backtest/paper stepper measures R on the buffered stop
        (invalidation + ``stop_buffer_atr`` x ATR), so the live
        broker SL must sit there too — placing it at the raw
        invalidation would trade a tighter stop than was measured.
        """
        ex = exits or ExitConfig()
        d = request.direction
        buffer = ex.stop_buffer_atr * request.atr if request.atr > 0 else 0.0
        return float(request.stop_loss - d * buffer)

    def open(self, *, ticket: int, request: OrderRequest,
             fill_price: float, entry_bar: int,
             broker_sl: float | None = None) -> TrackedPosition:
        """Start tracking a live position (after the order filled).

        ``request.stop_loss`` is the raw structural invalidation;
        the stepper derives the buffered stop0 from it.
        ``broker_sl`` is the SL actually placed on the broker
        (``planned_stop``); when omitted it defaults to stop0.
        """
        pos = self._stepper.open(ticket=ticket, request=request,
                                 fill_price=fill_price,
                                 entry_bar=entry_bar)
        self._states[ticket] = _TickState(
            pos=pos, broker_sl=(float(broker_sl) if broker_sl is not None
                                else pos.stop0),
            layers=[Layer(entry=float(fill_price),
                          volume=float(request.volume),
                          sl=float(broker_sl) if broker_sl is not None
                          else float(pos.stop0),
                          alert_id=request.alert_id,
                          playbook=getattr(request, "playbook", ""))])
        return pos

    def adopt(self, *, ticket: int, symbol: str, direction: int,
              volume: float, entry_price: float, sl: float | None,
              entry_bar: int = 0, alert_id: str = "") -> TrackedPosition:
        """Track a position the broker already holds (restart sync).

        No targets are known for an adopted position, so it gets
        breakeven + trailing + stop — but NO partial (there is no
        target book to bank against).  Its live SL is taken as both
        invalidation and stop0 (buffer 0).
        """
        pos = TrackedPosition(
            ticket=ticket, symbol=symbol, direction=direction,
            volume=volume, orig_volume=volume, entry_price=entry_price,
            signal_bar=entry_bar, entry_bar=entry_bar,
            invalidation=float(sl) if sl is not None else entry_price,
            stop0=float(sl) if sl is not None else entry_price,
            risk_price=abs(entry_price - sl) if sl is not None else 0.0,
            alert_id=alert_id, partial_price=None, partial_done=True)
        self._states[ticket] = _TickState(
            pos=pos, broker_sl=float(sl) if sl is not None else None,
            layers=[Layer(entry=float(entry_price), volume=float(volume),
                          sl=float(sl) if sl is not None else entry_price,
                          alert_id=alert_id)])
        return pos

    def remove(self, ticket: int) -> None:
        self._states.pop(ticket, None)

    def tickets(self) -> list[int]:
        return list(self._states)

    def position(self, ticket: int) -> TrackedPosition | None:
        st = self._states.get(ticket)
        return st.pos if st is not None else None

    def effective_stop(self, ticket: int) -> float | None:
        """SL actually governing the position (broker SL if moved,
        else the stepper's active stop).  None when untracked."""
        st = self._states.get(ticket)
        return st.effective_stop if st is not None else None

    # ---------------------------------------------------------------- layers
    def layers(self, ticket: int) -> list[Layer]:
        """Entry legs of a position (base first).  Copy — mutate via
        add_layer / apply only."""
        st = self._states.get(ticket)
        return list(st.layers) if st is not None else []

    def add_layer(self, ticket: int, *, entry: float, volume: float,
                  sl: float, alert_id: str = "",
                  playbook: str = "") -> Layer | None:
        """Append an added layer after its fill.  The position's
        tracked volume grows by the layer; management (breakeven /
        trail / partial) keeps keying off the BASE entry + risk."""
        st = self._states.get(ticket)
        if st is None or st.pos.closed:
            return None
        layer = Layer(entry=float(entry), volume=float(volume),
                      sl=float(sl), alert_id=alert_id, playbook=playbook)
        st.layers.append(layer)
        st.pos.volume += float(volume)
        st.pos.orig_volume += float(volume)
        return layer

    def close_top_layer(self, ticket: int) -> TickAction | None:
        """Manual trim: close the most recent added layer early while
        the base rides.  Returns the action for the caller to execute
        (``apply`` commits it).  None when there is no added layer."""
        st = self._states.get(ticket)
        if st is None or st.pos.closed or len(st.layers) <= 1:
            return None
        top = st.layers[-1]
        return TickAction("trim", level=top.sl, volume=top.volume,
                          reason="trim")

    # ------------------------------------------------------------ per tick
    def on_tick(self, ticket: int, price: float) -> list[TickAction]:
        """Actions this tick demands (decide; caller executes + applies).

        Order mirrors the stepper's conservative ordering: stops
        first, then the partial, then SL-tightening moves.  At most
        one SL move is emitted per tick (breakeven wins over trail
        when both trigger on the same tick — same level family,
        tightest wins via ``apply`` ordering below).
        """
        st = self._states.get(ticket)
        if st is None:
            return []
        pos = st.pos
        if pos.closed or pos.risk_price <= 0:
            return []
        d = pos.direction
        ex = self.exits

        # Wicked invalidation (observation only, never a close).
        if (d == 1 and price < pos.invalidation) or \
                (d == -1 and price > pos.invalidation):
            pos.wicked_inv = True

        # 1. Stop: the broker's own SL is primary; this action is the
        #    bookkeeping + safety-net path (close if still open).
        stop = st.effective_stop
        if (d == 1 and price <= stop) or (d == -1 and price >= stop):
            return [TickAction("stop_hit", level=stop, volume=pos.volume,
                                reason="stop")]

        actions: list[TickAction] = []
        # 1b. Added-layer stops (engine-side): an added layer whose
        # own SL is crossed closes alone — the base layer rides.
        # The base layer's stop is the broker SL (handled above).
        for layer in st.layers[1:]:
            hit = (d == 1 and price <= layer.sl) or \
                (d == -1 and price >= layer.sl)
            if hit:
                actions.append(TickAction(
                    "layer_stop", level=layer.sl, volume=layer.volume,
                    reason="layer_stop"))
        if actions:
            return actions
        # 2. Partial bank at the first liquidity pool.
        if not pos.partial_done and pos.partial_price is not None:
            crossed = price >= pos.partial_price if d == 1 \
                else price <= pos.partial_price
            if crossed:
                vol = min(ex.partial_frac * pos.orig_volume, pos.volume)
                actions.append(TickAction("partial", level=pos.partial_price,
                                          volume=vol, reason="partial"))

        # 3. Best excursion from THIS tick (not the next bar).
        pos.best_r = max(pos.best_r,
                         d * (price - pos.entry_price) / pos.risk_price)

        # 4. Breakeven: SL -> entry (+ offset) on the crossing tick.
        sl_action: TickAction | None = None
        if not st.breakeven_done \
                and pos.best_r >= self.tick_cfg.breakeven_trigger_r:
            level = pos.entry_price + d * self.tick_cfg.breakeven_offset_r \
                * pos.risk_price
            if (d == 1 and level > stop) or (d == -1 and level < stop):
                sl_action = TickAction("modify_sl", level=level,
                                       reason="breakeven")
        # 5. Trailing: evaluated independently when breakeven did not
        #    already move the SL this tick (one SL move per tick).
        if sl_action is None and pos.best_r >= ex.trail_trigger_r:
            level = pos.entry_price + d * (
                pos.best_r - ex.trail_dist_r) * pos.risk_price
            if (d == 1 and level > stop) or (d == -1 and level < stop):
                sl_action = TickAction("modify_sl", level=level,
                                       reason="trail")
        if sl_action is not None:
            actions.append(sl_action)
        return actions

    def apply(self, ticket: int, action: TickAction) -> None:
        """Commit an action after the executor reported success."""
        st = self._states.get(ticket)
        if st is None:
            return
        pos = st.pos
        if action.kind == "modify_sl" and action.level is not None:
            st.broker_sl = float(action.level)
            if action.reason == "breakeven":
                st.breakeven_done = True
            # Keep the stepper's trail level in sync so active_stop
            # never disagrees with the broker SL on the next tick.
            if pos.trail_level is None or \
                    (pos.direction == 1 and action.level > pos.trail_level) or \
                    (pos.direction == -1 and action.level < pos.trail_level):
                pos.trail_level = float(action.level)
        elif action.kind == "partial":
            pos.partial_done = True
            pos.volume = max(0.0, pos.volume - action.volume)
            if pos.volume <= 0:
                pos.closed = True
        elif action.kind == "layer_stop":
            # Drop the layer whose SL was crossed (match on volume —
            # a layer is uniquely identified by (volume, sl) within
            # a position; the base layer is never emitted here).
            for i, layer in enumerate(st.layers[1:], start=1):
                if layer.volume == action.volume \
                        and (action.level is None
                             or layer.sl == action.level):
                    st.layers.pop(i)
                    break
            pos.volume = max(0.0, pos.volume - action.volume)
            if pos.volume <= 0:
                pos.closed = True
        elif action.kind == "trim":
            # Manual trim: drop the top (most recent) added layer.
            if len(st.layers) > 1:
                st.layers.pop()
            pos.volume = max(0.0, pos.volume - action.volume)
            if pos.volume <= 0:
                pos.closed = True
        elif action.kind in ("stop_hit", "timeout"):
            pos.closed = True

    # ----------------------------------------------------------- bar close
    def on_bar_close(self, ticket: int, bar_index: int) -> list[TickAction]:
        """Time stop, evaluated on H1 bar closes (inherently bar-based).

        Everything price-based is tick-based (:meth:`on_tick`); the
        horizon time stop is the one rule that is defined in bars, so
        it fires here, on the bar close — never from a poll timer.
        """
        st = self._states.get(ticket)
        if st is None:
            return []
        pos = st.pos
        if pos.closed:
            return []
        if bar_index - pos.signal_bar >= self.exits.horizon:
            return [TickAction("timeout", level=None, volume=pos.volume,
                               reason="timeout")]
        return []
