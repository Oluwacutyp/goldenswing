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

import time
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
    #: Tiered partials — (trigger_R, fraction_of_orig_volume) per tier,
    #: fired in order as the excursion crosses each trigger.  Default
    #: is the owner's 50/25/25 split (50% at 1R, 25% at 2R, 25% at
    #: 3R, remainder rides).  When non-empty AND the position is at
    #: least ``tier_min_volume`` lots, tiers OWN partials and the
    #: single pool partial is skipped (two partial mechanisms would
    #: double-bank).  Empty tuple = legacy single pool partial.
    partial_tiers: tuple[tuple[float, float], ...] = (
        (1.0, 0.50), (2.0, 0.25), (3.0, 0.25))
    #: Tier logic applies only at/above this size (owner's gate:
    #: 0.03 lots — below it, one plain partial).
    tier_min_volume: float = 0.03
    #: Min-lot profit-lock (owner's rule): once the excursion hits
    #: ``profit_lock_trigger_r`` OR ``profit_lock_progress`` of the
    #: way to the first target, move the SL to entry + the lock
    #: distance in favour — ``max(fraction_of_risk * risk,
    #: atr_mult * ATR)``.  One-shot, only ever tightens.
    profit_lock_enabled: bool = True
    profit_lock_trigger_r: float = 1.0
    profit_lock_progress: float = 0.50
    profit_lock_fraction_of_risk: float = 0.50
    profit_lock_atr_mult: float = 0.75
    #: Chandelier trailing (owner's TrailingStopEngine): once the
    #: excursion reaches ``chandelier_activation_r``, the stop is the
    #: best price since entry minus ``chandelier_atr_mult`` x ATR
    #: (long; mirrored for short), ratcheting only.  Needs ATR —
    #: adopted positions (ATR unknown) never get it.
    chandelier_enabled: bool = True
    chandelier_atr_mult: float = 3.0
    chandelier_activation_r: float = 1.5
    #: Panic-candle emergency close (owner's PanicCandleProtector):
    #: when the adverse excursion since entry exceeds
    #: ``panic_mult`` x ATR with severity
    #: adverse / (panic_mult * ATR) > ``panic_severity``, close the
    #: whole position NOW.  Tick-resolution version of the owner's
    #: current-candle rule: same trigger level, trips on the
    #: crossing tick instead of at the candle close.  Needs ATR.
    panic_enabled: bool = True
    panic_mult: float = 3.0
    panic_severity: float = 1.5
    #: Scalp holding cap (owner's scalping_max_holding_minutes): a
    #: position flagged ``is_scalp`` closes after this many minutes
    #: of wall-clock hold, on the tick that crosses the cap.
    scalp_cap_minutes: float = 15.0
    #: Regime-dependent time exit: max holding H1 bars per bedrock
    #: structure.  None = the fixed ExitConfig.horizon for every
    #: structure.  The runner default is DEFAULT_TIME_EXIT_BARS
    #: (engine-derived values, see below).
    time_exit_bars: dict[str, int] | None = None
#: Default regime-dependent time-exit map (H1 bars from the signal
#: bar), set deliberately from THIS engine's own evidence — NOT the
#: owner's old-bot 8–20 values, whose bar unit is unverified (copying
#: them onto H1 bars would silently cut swing holds from 72h to
#: 8–20h).  Anchors:
#:
#: - The fixed horizon is 72 H1 bars (~3 days): Phase-2's own swing
#:   horizon ("gives the 3-day swing room to breathe").
#: - ``clean_trend`` gets the full 72: a trend needs room; a trend
#:   that hasn't hit stop/target in 3 days is stale.
#: - ``range`` 48 (2 days): a range trade should reach the opposite
#:   edge, or the range read is stale.
#: - ``compression`` 36 (1.5 days): compression resolves by breakout
#:   or fails fast; longer is dead money.
#: - ``breakout`` / ``reversal`` 24 (1 day): breakouts either follow
#:   through immediately or fail; reversals are counter-trend and
#:   the lowest-confidence label (0.45) — both get a short leash.
#: - ``choppy`` 12 (~half a day): choppy has no core fit (wicky +
#:   doji hunting); ~Phase-1's ~11-bar scalp-out scale.
#: - ``unclear`` 72: classification failed — same as the fixed
#:   horizon, explicit rather than a silent fallback.
#: Keys are exactly the structures ``classify_regime`` produces.
DEFAULT_TIME_EXIT_BARS: dict[str, int] = {
    "clean_trend": 72,
    "range": 48,
    "compression": 36,
    "breakout": 24,
    "reversal": 24,
    "choppy": 12,
    "unclear": 72,
}


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
                       # "layer_stop" | "trim" | "panic" | "scalp_cap"
    level: float | None = None   # SL level (modify_sl) / trigger (others)
    volume: float = 0.0          # lots to close (partial / stop / timeout /
                                 # layer_stop / trim / panic / scalp_cap)
    reason: str = ""             # "breakeven" | "trail" | "partial" |
                                 # "stop" | "timeout" | "layer_stop" |
                                 # "trim" | "panic" | "scalp_cap" |
                                 # "partial_tier_N" | "profit_lock" |
                                 # "chandelier"


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
             broker_sl: float | None = None,
             now: float | None = None) -> TrackedPosition:
        """Start tracking a live position (after the order filled).

        ``request.stop_loss`` is the raw structural invalidation;
        the stepper derives the buffered stop0 from it.
        ``broker_sl`` is the SL actually placed on the broker
        (``planned_stop``); when omitted it defaults to stop0.
        ``now`` is the wall-clock open time (epoch seconds) for the
        scalp cap — defaults to ``time.time()``.
        """
        pos = self._stepper.open(ticket=ticket, request=request,
                                 fill_price=fill_price,
                                 entry_bar=entry_bar)
        pos.is_scalp = bool(getattr(request, "is_scalp", False))
        pos.opened_at = float(now) if now is not None else time.time()
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
            alert_id=alert_id, partial_price=None, partial_done=True,
            no_partial=True)
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
    def on_tick(self, ticket: int, price: float,
                now: float | None = None) -> list[TickAction]:
        """Actions this tick demands (decide; caller executes + applies).

        Order mirrors the stepper's conservative ordering: stops
        first, then the partial, then SL-tightening moves.  At most
        one SL move is emitted per tick — it is the TIGHTEST
        candidate (breakeven / profit-lock / R-trail / Chandelier)
        that improves on the current stop.
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

        # 1a. Panic: adverse excursion since entry past the panic
        #     threshold — close the whole position NOW, before any
        #     partial or layer stop.  A crossed stop above already
        #     returned, so panic never mislabels a stop hit.
        pos.worst_r = max(pos.worst_r,
                          -d * (price - pos.entry_price) / pos.risk_price)
        tcfg = self.tick_cfg
        if tcfg.panic_enabled and pos.atr > 0:
            adverse = pos.worst_r * pos.risk_price
            severity = adverse / (tcfg.panic_mult * pos.atr)
            if severity > tcfg.panic_severity:
                return [TickAction("panic", level=price,
                                   volume=pos.volume, reason="panic")]

        # 1b. Scalp holding cap: wall-clock hold past the cap closes
        #     the whole position.  Panic (above) wins on a tie —
        #     an emergency close outranks a time exit.
        if pos.is_scalp and pos.opened_at is not None:
            now_ts = float(now) if now is not None else time.time()
            if now_ts - pos.opened_at >= tcfg.scalp_cap_minutes * 60.0:
                return [TickAction("scalp_cap", level=price,
                                   volume=pos.volume,
                                   reason="scalp_cap")]

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

        # 3. Best excursion from THIS tick (not the next bar) —
        #    evaluated BEFORE partials/SL moves so a tier, the
        #    profit-lock, and the Chandelier all see this tick's
        #    excursion (a gap tick must not leave them a tick late).
        pos.best_r = max(pos.best_r,
                         d * (price - pos.entry_price) / pos.risk_price)

        # 2. Partial bank: tiered partials own partials when
        #    configured and the position is big enough; otherwise
        #    the legacy single pool partial.
        tiers = self.tick_cfg.partial_tiers
        if tiers and not pos.no_partial \
                and pos.orig_volume >= self.tick_cfg.tier_min_volume:
            if pos.tiers_done < len(tiers):
                trig_r, frac = tiers[pos.tiers_done]
                if pos.best_r >= trig_r:
                    vol = min(frac * pos.orig_volume, pos.volume)
                    if vol > 0:
                        actions.append(TickAction(
                            "partial",
                            level=pos.entry_price + d * trig_r
                            * pos.risk_price,
                            volume=vol,
                            reason=f"partial_tier_{pos.tiers_done + 1}"))
        elif not pos.no_partial and not pos.partial_done \
                and pos.partial_price is not None:
            crossed = price >= pos.partial_price if d == 1 \
                else price <= pos.partial_price
            if crossed:
                vol = min(ex.partial_frac * pos.orig_volume, pos.volume)
                actions.append(TickAction("partial", level=pos.partial_price,
                                          volume=vol, reason="partial"))

        # 4/5. SL-tightening candidates — at most ONE SL move per
        #    tick, and it is the TIGHTEST candidate that improves on
        #    the current stop.  Breakeven, profit-lock, the R-trail,
        #    and the Chandelier are all "move the SL tighter" — the
        #    tightest wins; a one-shot whose action is not applied
        #    stays pending (its level stays a candidate next tick).
        #    A candidate that would LOOSEN the stop never fires.
        candidates: list[tuple[float, str]] = []
        if not pos.profit_lock_done and tcfg.profit_lock_enabled:
            tp = pos.targets[0] if pos.targets else None
            target_dist = abs(tp - pos.entry_price) if tp is not None \
                else 2.0 * pos.risk_price
            progress = abs(price - pos.entry_price) / target_dist \
                if target_dist > 0 else 0.0
            if pos.best_r >= tcfg.profit_lock_trigger_r \
                    or progress >= tcfg.profit_lock_progress:
                lock_dist = max(
                    tcfg.profit_lock_fraction_of_risk * pos.risk_price,
                    tcfg.profit_lock_atr_mult * pos.atr
                    if pos.atr > 0 else 0.0)
                candidates.append((
                    pos.entry_price + d * lock_dist, "profit_lock"))
        if not st.breakeven_done \
                and pos.best_r >= tcfg.breakeven_trigger_r:
            candidates.append((
                pos.entry_price + d * tcfg.breakeven_offset_r
                * pos.risk_price, "breakeven"))
        if pos.best_r >= ex.trail_trigger_r:
            candidates.append((
                pos.entry_price + d * (
                    pos.best_r - ex.trail_dist_r) * pos.risk_price,
                "trail"))
        if tcfg.chandelier_enabled and pos.atr > 0 \
                and pos.best_r >= tcfg.chandelier_activation_r:
            best_price = pos.entry_price + d * pos.best_r * pos.risk_price
            candidates.append((
                best_price - d * tcfg.chandelier_atr_mult * pos.atr,
                "chandelier"))
        tighter = [(lvl, rsn) for lvl, rsn in candidates
                   if (d == 1 and lvl > stop)
                   or (d == -1 and lvl < stop)]
        if tighter:
            level, reason = max(tighter, key=lambda c: c[0]) \
                if d == 1 else min(tighter, key=lambda c: c[0])
            actions.append(TickAction("modify_sl", level=level,
                                      reason=reason))
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
            if action.reason == "profit_lock":
                pos.profit_lock_done = True
            # Keep the stepper's trail level in sync so active_stop
            # never disagrees with the broker SL on the next tick.
            if pos.trail_level is None or \
                    (pos.direction == 1 and action.level > pos.trail_level) or \
                    (pos.direction == -1 and action.level < pos.trail_level):
                pos.trail_level = float(action.level)
        elif action.kind == "partial":
            if action.reason.startswith("partial_tier_"):
                pos.tiers_done += 1
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
        elif action.kind in ("stop_hit", "timeout", "panic", "scalp_cap"):
            pos.closed = True

    # ----------------------------------------------------------- bar close
    def on_bar_close(self, ticket: int, bar_index: int,
                     regime_structure: str | None = None,
                     ) -> list[TickAction]:
        """Time stop, evaluated on H1 bar closes (inherently bar-based).

        Everything price-based is tick-based (:meth:`on_tick`); the
        horizon time stop is the one rule that is defined in bars, so
        it fires here, on the bar close — never from a poll timer.
        ``regime_structure`` is the bedrock structure at the close;
        when the configured ``time_exit_bars`` map has an entry for
        it, that horizon replaces the fixed one.
        """
        st = self._states.get(ticket)
        if st is None:
            return []
        pos = st.pos
        if pos.closed:
            return []
        horizon = self.exits.horizon
        time_map = self.tick_cfg.time_exit_bars
        if time_map and regime_structure is not None:
            horizon = int(time_map.get(regime_structure, horizon))
        if bar_index - pos.signal_bar >= horizon:
            return [TickAction("timeout", level=None, volume=pos.volume,
                               reason="timeout")]
        return []
