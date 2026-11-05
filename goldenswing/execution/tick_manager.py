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
    #: P3 (S9/F13): HOW breakeven is earned on the TickConfig path.
    #: "touch" = the legacy rule (the trigger above).  "earned" =
    #: breakeven arms only when a funding/structure event or a
    #: banked partial has EARNED it (see mark_be_earned), with the
    #: price touch demoted to a backstop at breakeven_backstop_r.
    #: F13's arithmetic is why touch is no longer the desk
    #: default: the runner constructs its manager with "earned".
    breakeven_mode: str = "touch"
    breakeven_backstop_r: float = 2.0
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
    #: Playbooks whose generic early-lock progress arm is DISABLED:
    #: for these, travelling 50% of the way to TP1 does NOT arm
    #: profit_lock — only a full 1R best excursion does.  Ordered
    #: 2026-10-27: qm_m5's close targets made the progress arm
    #: fire on small moves ("premature profit locking").  Extended
    #: 2026-11-05: hot_scalp joins on the same evidence — ticket
    #: 540274935's progress-armed lock (entry + 0.50R) sat 0.04
    #: above its own entry zone's top edge and 0.26 below TP1;
    #: the first routine retest of the zone ate it 11 seconds
    #: later for +0.46R against a +1.43R objective.  Re-enable
    #: per playbook only on replay-grade evidence.  The best-R
    #: arm, breakeven, trail, Chandelier, and the initial
    #: protective stop are unaffected, and other playbooks keep
    #: the progress arm.
    profit_lock_progress_excluded_playbooks: tuple[str, ...] = (
        "qm_m5", "hot_scalp")
    #: Structure floor for the profit-lock on ZONE-BORN positions
    #: (owner's doctrine: structure determines the stop — applied
    #: to management, 2026-11-05, ticket 540274935): when the
    #: position's entry zone is known, a long's lock level may not
    #: sit within ``profit_lock_zone_buffer_r`` x risk of the
    #: zone's far edge — it is capped at zone_high - buffer
    #: (shorts: zone_low + buffer), so a mere TOUCH of the edge
    #: the trade was built on cannot eat the stop; price must
    #: penetrate the buffer into the zone.  If the cap makes the
    #: lock no tighter than the current stop, the lock does not
    #: fire that tick (the decision is recorded on the tick
    #: state; see profit_lock_state).  Positions without zone
    #: data are untouched — today's exact behaviour.
    profit_lock_zone_buffer_r: float = 0.05
    #: Earned profit-lock (P3 mirror, 2026-11-05): for these
    #: playbooks the +1R touch alone does NOT arm profit_lock —
    #: the lock must be EARNED, either by ACCEPTANCE (an M5 close
    #: beyond the position's first target in its favour, stamped
    #: by the runner's bar-close pass via mark_pl_earned — the
    #: owner's reaction doctrine: acceptance-by-closes, not a
    #: wick touch) or by the BACKSTOP excursion below.  Evidence:
    #: two same-morning live trades — 540274935 (progress arm,
    #: +0.46R; the exclusion above) and 540791314, where the
    #: best-R arm locked at entry + 0.50R exactly (4124.49 ->
    #: 4131.52 on a 14.06 risk) and a normal pullback four
    #: minutes later converted a runner-shaped trade into a
    #: +0.49R scratch.  The +0.5R level had become the de-facto
    #: exit.  The lock LEVEL formula is unchanged — only WHEN
    #: it may fire changes.  Every other playbook keeps the
    #: touch arms exactly as before.
    profit_lock_earned_playbooks: tuple[str, ...] = ("hot_scalp",)
    #: Backstop for the earned profit-lock: an excursion this
    #: deep earns the lock even without an acceptance close
    #: (same value and posture as breakeven_backstop_r).
    profit_lock_backstop_r: float = 2.0
    #: Step 13 (2026-10-29, owner's pick after the Oct 29 journal
    #: evidence): FIXED-BRACKET playbooks — no post-entry stop
    #: tightening of any kind.  Profit-lock, breakeven, the R-trail
    #: and the Chandelier never fire; the position rides its
    #: placement bracket (buffered structural SL + broker TP) to
    #: the exit.  The Oct 29 trades showed the lock and the trail
    #: closing qm_m5 winners well short of their targets (+0.53R
    #: and +0.80R against targets at +1.10R / +1.52R, and the
    #: trail banking +1.05R after price missed TP by ~1pt); the
    #: owner ordered a controlled demo test of the pure bracket.
    #: Closes are NOT management moves and stay live: the initial
    #: stop, panic, holding caps, layer stops, timeouts.  Partials
    #: close volume without moving the stop and also stay.
    #: Supersedes step 10's narrower progress-arm exclusion for
    #: the listed playbooks (the exclusion still governs any
    #: playbook removed from this list later).
    fixed_bracket_playbooks: tuple[str, ...] = ("qm_m5",)
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
    #: P3 earned-breakeven state: set by mark_be_earned (funding/
    #: structure event) or by a banked partial (TP1) in apply().
    be_earned: bool = False
    be_earned_by: str = ""
    #: Earned profit-lock state (mirror of be_earned): set by
    #: mark_pl_earned (an acceptance close beyond TP1, fed by
    #: the runner's bar-close pass) or in on_tick when the
    #: backstop excursion first arms the lock.  Only consulted
    #: for TickConfig.profit_lock_earned_playbooks.
    pl_earned: bool = False
    pl_earned_by: str = ""
    #: SL actually on the broker (breakeven/trailing moves only tighten).
    broker_sl: float | None = None
    #: Broker says this position is gone (a modify reported "no
    #: position"): management stands down for this ticket and waits
    #: for _reconcile to book the broker_closed close.  Never
    #: retried — the Oct 27 journal shows "no position" retried on
    #: the very next tick after the broker already closed the trade.
    sl_gone: bool = False
    #: Consecutive modify_sl failures per (reason, level) inside the
    #: current side-valid window.  A side-invalid tick closes the
    #: window (see reset_sl_window): the level may be tried again
    #: when price recovers, but never more than
    #: SL_MAX_FAILURES_PER_WINDOW times per window — the Oct 27
    #: journal shows 52 identical "Invalid stops" for one ticket.
    sl_failures: dict = field(default_factory=dict)
    #: Playbook that opened this position ("" when unknown, e.g.
    #: restart adoption): the progress-arm exclusion above keys
    #: on it.  Set by the runner at fill time; never parsed out
    #: of the alert id for gating.
    playbook: str = ""
    #: Entry legs: base first, added layers after.
    layers: list = field(default_factory=list)
    #: Narrative management plan (``live/management.py`` dict from
    #: ``mark_entered``) attached by the runner.  When present, the
    #: plan OWNS breakeven / partials / trailing / time-stop for
    #: this position — the TickConfig defaults below are NOT used
    #: for those (profit-lock, the R-trail and the Chandelier are
    #: skipped, so two trailing mechanisms never fight).  Panic,
    #: scalp-cap and the broker stop are safety, not management,
    #: and stay active.  None = legacy TickConfig behaviour.
    narrative_plan: dict | None = None
    #: ATR at fill for the narrative trailing step (ATR-step
    #: trailing = best price since entry minus trail_step_atr x
    #: this ATR).  The SL locked at fill is never recalculated
    #: from a LATER ATR — this snapshot only sets the trailing
    #: step.
    narrative_atr: float = 0.0
    #: Narrative invalidation context: {"choch_extreme": float|None,
    #: "fvg": {"top": float, "bottom": float}|None}.  See
    #: :meth:`TickManager.check_narrative_invalidation`.
    narrative_context: dict = field(default_factory=dict)
    #: The entry zone the position was born from, (low, high),
    #: carried from the signal at fill time (``open(entry_zone=)``).
    #: None = unknown (adopted positions, zone-less signals):
    #: the profit-lock zone cap does not apply and management
    #: behaves exactly as before the cap existed.
    entry_zone: tuple | None = None
    #: Reason of the last modify_sl actually applied to the
    #: broker ("breakeven" | "trail" | "profit_lock" |
    #: "chandelier" | ...).  "" = the stop never moved: the
    #: placement stop is still the stop.  The close classifier
    #: names the exit from THIS record, never from price
    #: geometry — geometry mislabelled ticket 540274935's
    #: profit_lock exit as "(trail)" (2026-11-05).
    last_sl_reason: str = ""
    #: The last profit-lock zone-cap decision, when the cap was
    #: evaluated: {"raw_level", "level", "zone_low", "zone_high",
    #: "buffer", "suppressed"}.  Refreshed on every tick that
    #: evaluates a profit-lock candidate; None when no candidate
    #: was evaluated or the cap did not bind.  The runner reads
    #: it via profit_lock_state() to journal capped / held locks.
    profit_lock_zone_cap: dict | None = None
    #: Lifecycle event log (chart truth): every EXECUTED event on
    #: this position, in order — {"kind": "open"|"sl"|"partial",
    #: "ts": epoch, ...}.  Appended only at the booking seams
    #: (open / apply / bank_leg), so it records what the broker
    #: confirmed, never what a tick merely proposed.  The close
    #: chart (charts.signal_model.LifecycleStudy) reads it via
    #: lifecycle_events(); the journal notes remain the audit
    #: trail, this is the same facts in structured form.
    events: list = field(default_factory=list)

    @property
    def effective_stop(self) -> float:
        if self.broker_sl is not None:
            return self.broker_sl
        return self.pos.active_stop


#: Max identical modify_sl submissions per side-valid window.
#: A window ends when the quote goes side-invalid for the level
#: (price fell back); a fresh window opens when price recovers.
#: 3 bounds the broker spam without forfeiting a level that
#: becomes valid again — the Oct 27 journal's 52 identical
#: rejections for one ticket came from an unbounded retry.
SL_MAX_FAILURES_PER_WINDOW = 3


def sl_level_valid(direction: int, level: float, bid: float, ask: float,
                   margin: float) -> bool:
    """True when a proposed SL can be sent to the broker.

    Long: the stop must sit at least ``margin`` below the bid.
    Short: at least ``margin`` above the ask.  A stop on the wrong
    side of the quote, or inside the broker's stops/freeze
    distance, is rejected as "Invalid stops" — and resubmitting
    it unchanged every tick is spam, not management.
    """
    if direction == 1:
        return level <= bid - margin
    if direction == -1:
        return level >= ask + margin
    return False


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
             now: float | None = None,
             playbook: str = "",
             management_horizon: str = "",
             management_policy_id: str = "",
             narrative_plan: dict | None = None,
             narrative_atr: float = 0.0,
             narrative_choch_extreme: float | None = None,
             narrative_fvg: dict | None = None,
             entry_zone=None) -> TrackedPosition:
        """Start tracking a live position (after the order filled).

        ``request.stop_loss`` is the raw structural invalidation;
        the stepper derives the buffered stop0 from it.
        ``broker_sl`` is the SL actually placed on the broker
        (``planned_stop``); when omitted it defaults to stop0.
        ``now`` is the wall-clock open time (epoch seconds) for the
        scalp cap — defaults to ``time.time()``.
        ``narrative_plan`` is the management dict from
        ``mark_entered`` (attached via
        :meth:`attach_narrative_plan`); when given, the plan owns
        breakeven / partials / trailing / time-stop for this
        position instead of the TickConfig defaults.
        ``entry_zone`` is the signal's entry zone (low, high)
        the position was born from; it rides onto the tick state
        and floors the profit-lock outside the zone (see
        ``TickConfig.profit_lock_zone_buffer_r``).  A (0.0, 0.0)
        / empty / degenerate zone means "unknown" — no cap.
        """
        pos = self._stepper.open(ticket=ticket, request=request,
                                 fill_price=fill_price,
                                 entry_bar=entry_bar)
        pos.is_scalp = bool(getattr(request, "is_scalp", False))
        pos.opened_at = float(now) if now is not None else time.time()
        # Step 18: the declared management horizon + policy ride
        # onto the position — the close record names them, and
        # the policy id must agree with the playbook-derived
        # resolution (asserted in the step-18 acceptance tests).
        pos.management_horizon = str(management_horizon or "")
        pos.management_policy_id = str(management_policy_id or "")
        zone = None
        if entry_zone:
            try:
                _zlo, _zhi = float(entry_zone[0]), float(entry_zone[1])
            except (TypeError, ValueError, IndexError):
                _zlo = _zhi = 0.0
            if _zhi > _zlo > 0.0:
                zone = (_zlo, _zhi)
        _st = _TickState(
            pos=pos, broker_sl=(float(broker_sl) if broker_sl is not None
                                else pos.stop0),
            playbook=str(playbook), entry_zone=zone,
            layers=[Layer(entry=float(fill_price),
                          volume=float(request.volume),
                          sl=float(broker_sl) if broker_sl is not None
                          else float(pos.stop0),
                          alert_id=request.alert_id,
                          playbook=getattr(request, "playbook", ""))])
        _st.events.append({
            "kind": "open", "ts": float(pos.opened_at),
            "price": float(fill_price),
            "broker_sl": (float(broker_sl) if broker_sl is not None
                          else float(pos.stop0))})
        self._states[ticket] = _st
        if narrative_plan is not None:
            self.attach_narrative_plan(
                ticket, narrative_plan, atr=narrative_atr,
                choch_extreme=narrative_choch_extreme,
                fvg=narrative_fvg)
        return pos

    # --------------------------------------------- narrative plan wiring
    def attach_narrative_plan(self, ticket: int, plan: dict, *,
                              atr: float = 0.0,
                              choch_extreme: float | None = None,
                              fvg: dict | None = None) -> bool:
        """Attach a narrative management plan to a tracked position.

        Returns False when the ticket is unknown (never creates a
        phantom position).  The plan is stored as-is; derived
        values (breakeven trigger, partial tiers, trailing step,
        time-stop) are read from it on every tick, so a plan the
        runner updates (e.g. an upgrade) takes effect on the next
        tick without re-attaching.  ``atr`` is the ATR snapshot at
        fill for the trailing step; ``choch_extreme`` / ``fvg``
        are the invalidation context (see
        :meth:`check_narrative_invalidation`).
        """
        st = self._states.get(ticket)
        if st is None:
            return False
        st.narrative_plan = dict(plan)
        st.narrative_atr = float(atr)
        st.narrative_context = {"choch_extreme": choch_extreme,
                                "fvg": dict(fvg) if fvg else None}
        return True

    def narrative_plan(self, ticket: int) -> dict | None:
        """The narrative management plan attached to a position
        (copy), or None when the position runs on TickConfig."""
        st = self._states.get(ticket)
        return dict(st.narrative_plan) \
            if st is not None and st.narrative_plan is not None else None

    # --------------------------------------- P3 earned breakeven
    def mark_be_earned(self, ticket: int, via: str) -> bool:
        """Arm breakeven for a position: a funding/structure event
        printed in its favour (or another earner the runner
        names).  The stop itself moves on the next tick via the
        ordinary candidate path — this only records the earning,
        so /status and the journal can say WHY.  False when the
        ticket is unknown or breakeven already happened."""
        st = self._states.get(ticket)
        if st is None or st.breakeven_done:
            return False
        if not st.be_earned:
            st.be_earned = True
            st.be_earned_by = str(via)
        return True

    def be_state(self, ticket: int) -> dict:
        st = self._states.get(ticket)
        if st is None:
            return {}
        return {"earned": bool(st.be_earned),
                "by": st.be_earned_by,
                "done": bool(st.breakeven_done)}

    # --------------------------------------- earned profit-lock
    def mark_pl_earned(self, ticket: int, via: str) -> bool:
        """Arm the earned profit-lock for a position: an
        acceptance close beyond TP1 printed in its favour (the
        runner's bar-close pass names the event).  The lock
        itself fires on the next tick via the ordinary candidate
        path — this only records the earning, so /status and
        the journal can say WHY.  False when the ticket is
        unknown or the lock already happened."""
        st = self._states.get(ticket)
        if st is None or st.pos.profit_lock_done:
            return False
        if not st.pl_earned:
            st.pl_earned = True
            st.pl_earned_by = str(via)
        return True

    def pl_state(self, ticket: int) -> dict:
        st = self._states.get(ticket)
        if st is None:
            return {}
        return {"earned": bool(st.pl_earned),
                "by": st.pl_earned_by,
                "done": bool(st.pos.profit_lock_done)}

    def last_sl_reason(self, ticket: int) -> str:
        """Reason of the last modify_sl applied for this ticket
        ("profit_lock" / "trail" / "breakeven" / "chandelier");
        "" when the stop never moved (the placement stop is
        still the stop) or the ticket is unknown."""
        st = self._states.get(ticket)
        return st.last_sl_reason if st is not None else ""

    def profit_lock_state(self, ticket: int) -> dict:
        """Profit-lock bookkeeping for the journal path:
        {"done", "zone_cap"} — zone_cap is the last zone-cap
        decision (raw vs capped level, the zone, whether the
        cap suppressed the lock this evaluation) or None."""
        st = self._states.get(ticket)
        if st is None:
            return {}
        return {"done": bool(st.pos.profit_lock_done),
                "zone_cap": (dict(st.profit_lock_zone_cap)
                             if st.profit_lock_zone_cap is not None
                             else None)}

    def book_external_partial(self, ticket: int,
                              volume: float) -> bool:
        """Book a partial the RUNNER executed outside the tick
        action flow (bar-close management: progress trim, drift
        trim, zone bank, ADR bank, pre-print trim).  Keeps the
        tracked volume honest so tiers/stops stay consistent.
        False when the ticket is unknown.  Volume only — the
        leg's P&L is recorded separately via :meth:`bank_leg`
        (the runner owns pricing/specs, this class owns state).
        """
        st = self._states.get(ticket)
        if st is None or st.pos.closed:
            return False
        st.pos.volume = max(0.0, st.pos.volume - float(volume))
        if st.pos.volume <= 0:
            st.pos.closed = True
        return True

    def bank_leg(self, ticket: int, *, volume: float, price: float,
                 entry: float, stop: float, pnl_money: float) -> bool:
        """Record one banked partial leg's realized P&L on the
        position (live MT5 path).  The runner computes the leg
        money (it owns the symbol spec); this only stores it, so
        the full close can report the trade total (banked +
        remainder) and the volume-weighted total R.  False when
        the ticket is unknown or the position is closed — a leg
        on a dead position is never recorded."""
        st = self._states.get(ticket)
        if st is None or st.pos.closed:
            return False
        st.pos.banked_legs.append({
            "volume": float(volume), "price": float(price),
            "entry": float(entry), "stop": float(stop),
            "pnl": float(pnl_money)})
        st.pos.banked_pnl += float(pnl_money)
        st.events.append({
            "kind": "partial", "ts": time.time(),
            "price": float(price), "volume": float(volume)})
        return True

    def check_narrative_invalidation(self, ticket: int, price: float,
                                     *, fvg_mitigated: bool = False) -> bool:
        """Tick-level narrative invalidation query for the runner.

        True when price is back through the CHoCH extreme AND the
        confirm FVG is mitigated — both required, one alone is
        noise (mirrors ``live/management.py``).  With no FVG
        recorded (e.g. a breakout-stop entry) the CHoCH break
        alone invalidates.  False when no narrative plan is
        attached or no CHoCH extreme was recorded.  Query only:
        the runner decides the exit — this never closes anything.
        """
        st = self._states.get(ticket)
        if st is None or st.narrative_plan is None:
            return False
        choch = st.narrative_context.get("choch_extreme")
        if choch is None:
            return False
        d = st.pos.direction
        through = (d == 1 and price < choch) or \
            (d == -1 and price > choch)
        if not through:
            return False
        if st.narrative_context.get("fvg") is None:
            return True
        return bool(fvg_mitigated)

    @staticmethod
    def _narrative_tiers(plan: dict) -> tuple[tuple[float, float], ...]:
        """Partial tiers from a narrative plan: (TP_R, fraction)
        per TP with a price and a positive fraction.  TP3 None
        (opposing GREEN, exit all at TP2) is skipped — the TP2
        fraction already includes the runner there."""
        tiers = []
        for key in ("tp1", "tp2", "tp3"):
            r = plan.get(f"{key}_r")
            pct = plan.get(f"{key}_pct")
            if r is None or pct is None:
                continue
            if float(pct) > 0:
                tiers.append((float(r), float(pct)))
        return tuple(tiers)

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

    # ------------------------------------------- broker-aware SL moves
    def mark_broker_gone(self, ticket: int) -> bool:
        """Stand a ticket down: the broker reports no such position.

        True when the ticket was tracked and not already stood
        down.  The state itself stays so _reconcile can book the
        broker_closed close on its normal cadence — only further
        management actions stop.
        """
        st = self._states.get(ticket)
        if st is None or st.sl_gone:
            return False
        st.sl_gone = True
        return True

    def note_sl_failure(self, ticket: int, reason: str,
                        level: float) -> bool:
        """Record one failed modify_sl.  Returns True when the
        (reason, level) window is exhausted and the level is
        blocked until the runner re-opens the window (quote went
        side-invalid) or a success clears it."""
        st = self._states.get(ticket)
        if st is None:
            return False
        key = (str(reason), float(level))
        st.sl_failures[key] = int(st.sl_failures.get(key, 0)) + 1
        return st.sl_failures[key] >= SL_MAX_FAILURES_PER_WINDOW

    def _sl_blocked(self, ticket: int, reason: str,
                    level: float) -> bool:
        st = self._states.get(ticket)
        if st is None:
            return False
        return int(st.sl_failures.get((str(reason), float(level)), 0)) \
            >= SL_MAX_FAILURES_PER_WINDOW

    def sl_blocked(self, ticket: int, reason: str,
                   level: float) -> bool:
        """Public read of the failure window (the runner checks it
        before submitting, belt-and-braces over on_tick's skip)."""
        return self._sl_blocked(ticket, reason, level)

    def reset_sl_window(self, ticket: int, reason: str,
                        level: float) -> None:
        """Close the failure window for (reason, level): the quote
        went side-invalid, so the next side-valid window starts
        with a fresh failure budget."""
        st = self._states.get(ticket)
        if st is not None:
            st.sl_failures.pop((str(reason), float(level)), None)

    def tickets(self) -> list[int]:
        return list(self._states)

    def position(self, ticket: int) -> TrackedPosition | None:
        st = self._states.get(ticket)
        return st.pos if st is not None else None

    def lifecycle_events(self, ticket: int) -> list[dict]:
        """The position's executed lifecycle, in order (open /
        stop modifications / partial legs), as recorded at the
        booking seams. Empty for an unknown ticket — the chart
        layer treats that as "no lifecycle recorded", never as
        license to reconstruct one from prices."""
        st = self._states.get(ticket)
        if st is None:
            return []
        return [dict(ev) for ev in st.events]

    def effective_stop(self, ticket: int) -> float | None:
        """SL actually governing the position (broker SL if moved,
        else the stepper's active stop).  None when untracked."""
        st = self._states.get(ticket)
        return st.effective_stop if st is not None else None

    def is_fixed_bracket(self, ticket: int) -> bool:
        """Step 13: True when this position's playbook rides a
        fixed bracket (no post-entry stop tightening).  Runner
        paths that move stops outside ``on_tick`` (news pre-print
        breakeven, layer-fill rolls) consult this BEFORE touching
        the broker stop."""
        st = self._states.get(ticket)
        return st is not None and \
            st.playbook in self.tick_cfg.fixed_bracket_playbooks

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
        if pos.closed or st.sl_gone or pos.risk_price <= 0:
            return []
        d = pos.direction
        ex = self.exits

        # Failure windows are runs of side-valid ticks: a tick
        # whose mid price puts a failed level back on the wrong
        # side closes its window, so the level gets a fresh
        # budget when price recovers.  Without this, a blocked
        # level could never be retried — on_tick skips blocked
        # levels, so the runner would never see the tick that
        # should re-open the window.  (Mid is the conservative
        # proxy here: level above mid is certainly above bid for
        # a long; the runner's margin-precise check still gates
        # every actual submission.)
        for _key in list(st.sl_failures):
            _rsn, _lvl = _key
            if (d == 1 and _lvl > price) or (d == -1 and _lvl < price):
                st.sl_failures.pop(_key, None)

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

        # 2. Partial bank: a narrative plan owns partials (its TP
        #    ladder); otherwise tiered partials own them when
        #    configured and the position is big enough, else the
        #    legacy single pool partial.  A plan with no usable
        #    TPs falls back to the TickConfig tiers.
        plan = st.narrative_plan
        if plan is not None:
            tiers = self._narrative_tiers(plan) \
                or self.tick_cfg.partial_tiers
        else:
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

        # Step 13: fixed-bracket playbooks get NO SL-tightening
        #    candidates at all — the placement bracket is the whole
        #    exit policy.  Partial actions computed above still
        #    stand (they close volume; they never move the stop).
        if st.playbook in tcfg.fixed_bracket_playbooks:
            return actions

        # 4/5. SL-tightening candidates — at most ONE SL move per
        #    tick, and it is the TIGHTEST candidate that improves on
        #    the current stop.  Breakeven, profit-lock, the R-trail,
        #    and the Chandelier are all "move the SL tighter" — the
        #    tightest wins; a one-shot whose action is not applied
        #    stays pending (its level stays a candidate next tick).
        #    A candidate that would LOOSEN the stop never fires.
        candidates: list[tuple[float, str]] = []
        if plan is not None:
            # Narrative plan owns SL moves: breakeven at the
            # plan's threshold (late confirm 0.0 = immediate)
            # and ATR-step trailing behind the best excursion
            # (best price since entry minus trail_step_atr x
            # the ATR snapshot at fill, capped at the opposing
            # barrier).  Profit-lock, the R-trail and the
            # Chandelier are skipped — the plan is the single
            # owner of SL moves.  Structure-level (FVG/swing)
            # trailing stays in live/management.py on bar
            # closes; this tick trailing is its best-excursion
            # equivalent.
            be_trig = float(plan.get("breakeven_at_r", 1.0))
            # P3: the plan's be_mode governs.  "earned" (clean
            # profile): the plan's touch threshold no longer arms
            # breakeven by itself — an earning event (funding /
            # TP1) or the backstop excursion does.  "touch" (late
            # profile): the threshold arms it, as before.
            if str(plan.get("be_mode", "touch")) == "earned":
                backstop = float(plan.get(
                    "be_backstop_r", tcfg.breakeven_backstop_r))
                be_armed = st.be_earned or pos.best_r >= backstop
            else:
                be_armed = pos.best_r >= be_trig
            if not st.breakeven_done and be_armed:
                candidates.append((pos.entry_price, "breakeven"))
            if st.narrative_atr > 0:
                best_price = pos.entry_price + d * pos.best_r \
                    * pos.risk_price
                cand = best_price - d * float(
                    plan.get("trail_step_atr", 0.5)) \
                    * st.narrative_atr
                cap = plan.get("trail_cap_price")
                if cap is not None:
                    if d == 1 and float(cap) >= stop:
                        cand = min(cand, float(cap))
                    elif d == -1 and float(cap) <= stop:
                        cand = max(cand, float(cap))
                candidates.append((cand, "trail"))
        else:
            if not pos.profit_lock_done and tcfg.profit_lock_enabled:
                tp = pos.targets[0] if pos.targets else None
                target_dist = abs(tp - pos.entry_price) \
                    if tp is not None else 2.0 * pos.risk_price
                # Favourable progress ONLY: the old abs() armed
                # profit_lock on adverse excursion too (a long
                # falling away from its target counted as "progress"
                # toward it) — the move then fired while the trade
                # was underwater.
                progress = d * (price - pos.entry_price) / target_dist \
                    if target_dist > 0 else 0.0
                if st.playbook in tcfg.profit_lock_earned_playbooks:
                    # EARNED lock (P3 mirror): the +1R touch alone
                    # does NOT arm it — an acceptance close beyond
                    # TP1 (mark_pl_earned, fed by the runner's
                    # bar-close pass) or the backstop excursion
                    # earns it.  The backstop earning is stamped
                    # the tick it first holds, so the record names
                    # it even when a tighter candidate (trail /
                    # chandelier) wins that tick's modify.
                    if not st.pl_earned \
                            and pos.best_r \
                            >= tcfg.profit_lock_backstop_r:
                        st.pl_earned = True
                        st.pl_earned_by = (
                            f"backstop (best_r {pos.best_r:.2f}R)")
                    lock_armed = st.pl_earned
                else:
                    lock_armed = \
                        pos.best_r >= tcfg.profit_lock_trigger_r \
                        or (progress >= tcfg.profit_lock_progress
                            and st.playbook not in
                            tcfg.profit_lock_progress_excluded_playbooks)
                if lock_armed:
                    lock_dist = max(
                        tcfg.profit_lock_fraction_of_risk
                        * pos.risk_price,
                        tcfg.profit_lock_atr_mult * pos.atr
                        if pos.atr > 0 else 0.0)
                    lock_level = pos.entry_price + d * lock_dist
                    # Structure floor (2026-11-05): a zone-born
                    # position's lock may not sit within the buffer
                    # of its entry zone's far edge — ticket
                    # 540274935's lock sat 0.04 ABOVE the zone top
                    # and a touch of the edge ate it.  Cap the
                    # level a buffer INSIDE the far edge; if the
                    # capped level no longer tightens the current
                    # stop, the tighter-filter below drops it and
                    # the lock stays pending (never marked done).
                    # The decision is recorded on the state either
                    # way — the runner journals capped/held locks
                    # from profit_lock_state().
                    st.profit_lock_zone_cap = None
                    if st.entry_zone is not None:
                        zlo, zhi = st.entry_zone
                        buffer = tcfg.profit_lock_zone_buffer_r \
                            * pos.risk_price
                        cap = zhi - buffer if d == 1 else zlo + buffer
                        if (d == 1 and lock_level > cap) or \
                                (d == -1 and lock_level < cap):
                            st.profit_lock_zone_cap = {
                                "raw_level": lock_level,
                                "level": cap,
                                "zone_low": zlo, "zone_high": zhi,
                                "buffer": buffer,
                                "suppressed": not (
                                    (d == 1 and cap > stop)
                                    or (d == -1 and cap < stop))}
                            lock_level = cap
                    candidates.append((lock_level, "profit_lock"))
            # P3: earned mode — the +1R touch alone does NOT move
            # the stop (F13); an earning event or the backstop
            # excursion does.  Touch mode keeps the legacy rule
            # (default TickConfig — the runner opts into earned).
            if tcfg.breakeven_mode == "earned":
                be_armed = st.be_earned or \
                    pos.best_r >= tcfg.breakeven_backstop_r
            else:
                be_armed = pos.best_r >= tcfg.breakeven_trigger_r
            if not st.breakeven_done and be_armed:
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
                best_price = pos.entry_price + d * pos.best_r \
                    * pos.risk_price
                candidates.append((
                    best_price - d * tcfg.chandelier_atr_mult
                    * pos.atr, "chandelier"))
        tighter = [(lvl, rsn) for lvl, rsn in candidates
                   if (d == 1 and lvl > stop)
                   or (d == -1 and lvl < stop)]
        if tighter:
            level, reason = max(tighter, key=lambda c: c[0]) \
                if d == 1 else min(tighter, key=lambda c: c[0])
            # A level that exhausted its failure window is NOT
            # re-emitted: the runner re-opens the window when the
            # quote goes side-invalid for it (price fell back),
            # and a fresh window tries again when price recovers.
            if not self._sl_blocked(ticket, reason, level):
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
            if st.playbook in self.tick_cfg.fixed_bracket_playbooks:
                # Step 13 defense in depth: a fixed-bracket position's
                # stop is never booked as moved, whatever path built
                # the action.  (Decision points gate before the
                # broker is touched; this keeps the book honest if
                # one is ever missed.)
                return
            st.broker_sl = float(action.level)
            st.events.append({
                "kind": "sl", "ts": time.time(),
                "level": float(action.level),
                "reason": str(action.reason)})
            # The close classifier names the exit from the LAST
            # mechanism that actually moved the stop — recorded
            # here, at the one seam every modify path books
            # through, never inferred from price geometry later.
            st.last_sl_reason = str(action.reason)
            # A success closes the failure window for this level:
            # the next candidate (if any) starts with a fresh budget.
            st.sl_failures.pop((str(action.reason), float(action.level)),
                               None)
            if action.reason == "breakeven":
                st.breakeven_done = True
                if not st.be_earned:
                    # Fired on the backstop (or legacy touch): the
                    # record says so — earned stays distinguishable
                    # from touched in the journal/status.
                    st.be_earned_by = st.be_earned_by or "backstop"
            if action.reason == "profit_lock":
                pos.profit_lock_done = True
                if not st.pl_earned and st.playbook in \
                        self.tick_cfg.profit_lock_earned_playbooks:
                    # Fired on the backstop excursion (the in-tick
                    # stamp normally precedes it): the record says
                    # so — earned-by-backstop stays distinguishable
                    # from earned-by-acceptance in the journal.
                    st.pl_earned_by = st.pl_earned_by or "backstop"
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
            # P3/S9: banking TP1 EARNS breakeven (the desk's rule:
            # once the first bank is in, the rest rides free).
            if not st.be_earned:
                st.be_earned = True
                st.be_earned_by = "tp1_banked"
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
        # A narrative plan owns the time-stop: its time_stop_h1
        # (tier- and ATR-scaled at fill) replaces both the fixed
        # horizon and the regime-structure map for this position.
        if st.narrative_plan is not None and \
                st.narrative_plan.get("time_stop_h1") is not None:
            horizon = int(st.narrative_plan["time_stop_h1"])
        else:
            time_map = self.tick_cfg.time_exit_bars
            if time_map and regime_structure is not None:
                horizon = int(time_map.get(regime_structure, horizon))
        if bar_index - pos.signal_bar >= horizon:
            return [TickAction("timeout", level=None, volume=pos.volume,
                               reason="timeout")]
        return []
