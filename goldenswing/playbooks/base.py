"""Playbook interface (Phase 9).

A playbook is a self-contained entry system: it scans chart state for
its setup, fires its trigger, and simulates its outcome with the EXACT
machinery it was validated with.  The confluence engine never
re-simulates and never edits a playbook's outcome — it only ranks
pre-simulated candidates and decides which (if any) enters the book.
"""
from __future__ import annotations

import abc
import copy
from dataclasses import dataclass, field

import pandas as pd

from ..backtest.harness import Outcome
from ..contextmap import HuntingGround
from ..reader.state import MarketState


@dataclass
class EngineContext:
    """Everything a playbook or the engine may read.  Built once per
    run; states are the featurized MarketState list (reader fields are
    identical to plain reads — FeatureEngine.attach only ADDS)."""

    df: pd.DataFrame
    states: list[MarketState]
    years: float
    instrument: str = "GC=F"
    start_balance: float = 10_000.0
    forecast_table: object = None  # ForecastTable | None — the quality
    # dial: horizon agreement earns a confluence bonus / sizing input,
    # NEVER a veto (see confluence.forecast_bonuses)
    trap_features: pd.DataFrame | None = None  # GOD-CE salvage trap
    # intelligence (goldenswing.features.traps): 18 boolean columns
    # aligned to df by position.  Lazily computed + cached by
    # confluence.ensure_trap_features on first use; the trap-boost
    # bonus reads it (signal-bar convention).


@dataclass
class Candidate:
    """One tradable setup from one playbook, pre-simulated.

    ``outcome`` carries the playbook's own entry/exit machinery —
    opaque to the engine.  ``base_score`` is the playbook's own quality
    score (detector score for the anchor; documented placeholder for
    sleeves until OOS calibrates them).  ``thesis`` is the full
    trader-style reasoning, logged verbatim to the journal.
    """

    playbook: str
    outcome: Outcome
    base_score: float
    thesis: str
    context: dict = field(default_factory=dict)

    @property
    def signal_index(self) -> int:
        return self.outcome.signal.index

    @property
    def direction(self) -> int:
        return self.outcome.signal.direction

    @property
    def entry_index(self) -> int:
        return self.outcome.entry_index

    @property
    def exit_index(self) -> int:
        return self.outcome.exit_index

    @property
    def r_net(self) -> float:
        return self.outcome.r_net


class Playbook(abc.ABC):
    """One entry system.  Subclasses implement :meth:`candidates`.

    REQUIRED CONTEXT (Phase 10 refinement): a real trader only trades
    a setup where it is good and sits out everywhere else.  Each
    playbook therefore declares the context its experiment report
    validated — the session/regime/HTF state and setup gates where it
    is genuinely good ("where I trade").  The confluence engine
    enforces it: a candidate firing outside its playbook's declared
    context is skipped as a ``context-violation`` and journalled as a
    SYSTEM BUG, never traded.  Backtests grade mechanical rules, not
    the trader using them — this contract is the trader's discipline.
    """

    #: machine name, used in the journal and the confluence log
    name: str = "base"
    #: human description of the setup this playbook fishes for
    description: str = ""
    #: True -> candidates may enter the live P&L book.
    #: False -> shadow/paper only: scored, logged, never in live P&L.
    live: bool = False
    #: Regime this playbook may fire in: "trend" (D1 bias != 0),
    #: "range" (D1 flat + defined range), or "both".  Enforced by
    #: ``confluence.ConfluenceEngine.gate_live_signal`` — a range
    #: playbook can never fire in a TREND verdict and a trend
    #: playbook can never fire in a RANGE verdict.  Default
    #: "trend": every pre-existing playbook is a trend playbook.
    regime_scope: str = "trend"
    #: True -> this playbook fires only when the boundary machine
    #: (``live/range.py`` BoundaryTracker) has CONFIRMED a sweep on
    #: the signal's edge (wick beyond + close inside + M15 BOS
    #: reclaim + FVG).  A wick alone never arms it.  Enforced by
    #: ``confluence.ConfluenceEngine.gate_live_signal`` against the
    #: verdict's ``sweep_confirmed_edges``.  Default False; only the
    #: range edge sweep playbook opts in.
    requires_sweep_confirmation: bool = False
    #: True -> this playbook fires only with a COMPLETE narrative
    #: chain (``live/narrative.py`` ENTRY state at the signal's
    #: direction: HTF zone -> sweep -> displacement -> CHoCH/BOS +
    #: FVG).  A single pattern alone never trades.  Enforced by
    #: ``confluence.ConfluenceEngine.gate_live_signal`` (gate 0e)
    #: against the verdict's ``narrative`` state.  Default False;
    #: only the narrative playbook opts in.
    requires_narrative_chain: bool = False
    #: minimum base_score for live-book membership (anchor: 60.0).
    live_threshold: float = 60.0
    #: Where this playbook is genuinely good, in the words of its
    #: experiment report.  :meth:`check_context` enforces the checkable
    #: part; the rest is the documented scope (market, timeframe,
    #: sessions) for anyone porting the playbook elsewhere.
    context_description: str = ""
    #: Session scope — a FIRST-CLASS playbook property.  "any" means
    #: session-agnostic (the works-always core); otherwise the single
    #: session name the playbook may fire in ("asia" | "london" |
    #: "overlap" | "newyork" | "off").  The engine never lets a
    #: session-scoped playbook fire outside its session, and its
    #: backtest is judged on its session only.
    session_scope: str = "any"
    #: The playbook's hunting ground, DERIVED FROM ITS MEASURED CONTEXT
    #: MAP (goldenswing/contextmap.py) — data, not prose.  ``excluded``
    #: cells never fire; ``bonuses`` reward proven cells.  Derived by
    #: scripts/measure_context_maps.py; the derivation rules live in
    #: contextmap.derive_hunting_ground.
    hunting_ground: HuntingGround = HuntingGround()

    @abc.abstractmethod
    def candidates(self, ctx: EngineContext) -> list[Candidate]:
        """Scan and simulate.  Every returned candidate is a complete,
        causal, pre-simulated trade — the engine only ranks and books."""
        raise NotImplementedError

    def describe_context(self) -> str:
        """Human-readable required context, from the experiment report."""
        return self.context_description

    def check_context(self, candidate: Candidate,
                      ctx: EngineContext) -> tuple[bool, str]:
        """Is this candidate inside the playbook's declared required
        context?  Returns ``(True, "")`` or ``(False, reason)``.

        Delegates to :meth:`check_live_signal` — one gate
        implementation for the batch and live paths.  The checks must
        be INDEPENDENT re-verifications of the validated context: loose
        enough that no legitimately scanned candidate can fail them
        (the confluence canary proves this), strict enough to catch a
        scanner regression as a system bug instead of a trade.

        The live loop calls :meth:`check_live_signal` directly, which
        shares the same gates without needing a pre-simulated outcome.
        """
        return self.check_live_signal(candidate.outcome.signal, ctx,
                                      candidate.context)

    def check_live_signal(self, signal, ctx: EngineContext,
                          context: dict | None = None) -> tuple[bool, str]:
        """Required-context check for a live (un-simulated) signal.

        ``signal`` is the GoldenSignal-like; ``context`` carries the
        playbook-specific setup facts the batch path keeps on
        ``Candidate.context`` (may be None for playbooks whose gates
        need only the signal and chart state).  Playbooks whose batch
        :meth:`check_context` needs candidate fields override this to
        share the gate logic.
        """
        return True, ""

    def live_signal(self, states: list, t: int, df=None) -> list:
        """Incremental signal check at closed bar ``t`` for the live loop.

        Unlike :meth:`candidates`, this does NOT pre-simulate — the
        live executor manages the position with the playbook's exit
        policy.  ``df`` is the loop's bar frame (needed by playbooks
        whose trigger needs OHLC, e.g. resting-limit fills); playbooks
        that only need chart state ignore it.  Returns a (possibly
        empty) list of signals — most playbooks emit at most one per
        bar, but a resting-limit sleeve can fill several setups on the
        same bar.  Must be causal: only bars ``<= t`` may inform the
        result.  Default []: the playbook doesn't support live
        incremental mode yet.
        """
        return []

    def export_state(self) -> dict:
        """Working state for warm resume (pending setups, cursors).
        Default: stateless.  Playbooks with cross-bar working state
        (e.g. resting-limit orders) override both this and
        :meth:`import_state`."""
        return {}

    def import_state(self, state: dict) -> None:
        """Restore working state after a restart.  Default: no-op."""
        return

    # --------------------------------------------- trigger-aware boundary
    # Trigger direction is IMMUTABLE authority.  A playbook never
    # re-decides direction: it inherits the trigger direction and
    # validates only its own tactical pattern, location, score,
    # invalidation and costs at that direction.  A tactical pattern
    # that fires the other way is ``pattern_absent`` for this trigger
    # (the trigger did not fire that way), NEVER a
    # ``direction_mismatch`` rejection — that string must not appear
    # on this path.  TTL / invalidation / zone-exit stay with the
    # trigger; the playbook never extends them.
    def _tactical_pattern(self, bars, direction: int,
                          trigger_ctx: dict) -> dict | None:
        """Playbook-specific tactical pattern at ``direction``.

        Return a dict with at least ``score``, ``invalidation`` and
        ``costs`` when the playbook's own pattern is present at this
        direction, else None.  Default None: the base playbook has no
        tactical pattern of its own, so ``evaluate`` reports
        ``pattern_absent``.  Subclasses override this hook; they must
        NOT return a direction field that contradicts the trigger —
        ``evaluate`` treats any such result as absent.
        """
        return None

    def evaluate(self, bars, ctx) -> "PlaybookVerdict":
        """Evaluate this playbook for ONE trigger fire.

        ``bars`` is the bar frame (DataFrame or dict of frames);
        ``ctx`` is the trigger context (dict or object carrying
        ``direction`` / ``trigger_direction``, ``zone``,
        ``invalidation``, ``ttl`` / ``ttl_bars``, ``zone_exit``).
        The trigger context is deep-copied per playbook, so one
        playbook can never mutate another's snapshot.
        """
        trigger_ctx = snapshot_trigger_context(ctx)
        direction = int(trigger_ctx.get("direction", 0) or 0)
        if direction not in (1, -1):
            return PlaybookVerdict(
                playbook=self.name, status="pattern_absent",
                direction=direction, score=0.0,
                reason="pattern_absent: no trigger direction",
                trigger_context=trigger_ctx)
        try:
            found = self._tactical_pattern(bars, direction, trigger_ctx)
        except Exception as exc:  # a playbook never breaks the trigger
            return PlaybookVerdict(
                playbook=self.name, status="invalid",
                direction=direction, score=0.0,
                reason=f"invalid: tactical pattern error: {exc}",
                trigger_context=trigger_ctx)
        if not found:
            return PlaybookVerdict(
                playbook=self.name, status="pattern_absent",
                direction=direction, score=0.0,
                reason="pattern_absent",
                trigger_context=trigger_ctx)
        # A tactical result pointing the other way is absent for
        # this trigger, never a direction_mismatch rejection.
        if int(found.get("direction", direction) or direction) != direction:
            return PlaybookVerdict(
                playbook=self.name, status="pattern_absent",
                direction=direction, score=0.0,
                reason="pattern_absent: tactical pattern not at trigger direction",
                trigger_context=trigger_ctx)
        score = float(found.get("score", 0.0) or 0.0)
        invalidation = found.get("invalidation")
        costs = found.get("costs")
        if invalidation is None or costs is None:
            return PlaybookVerdict(
                playbook=self.name, status="invalid",
                direction=direction, score=score,
                reason="invalid: missing invalidation or costs",
                trigger_context=trigger_ctx)
        return PlaybookVerdict(
            playbook=self.name, status="valid",
            direction=direction, score=score,
            invalidation=invalidation, costs=costs,
            entry=found.get("entry"), stop=found.get("stop"),
            target=found.get("target"),
            pattern=found.get("pattern", self.name),
            location=found.get("location"),
            reason=found.get("reason", "valid"),
            trigger_context=trigger_ctx)


@dataclass(frozen=True)
class PlaybookVerdict:
    """Result of :meth:`Playbook.evaluate` for one trigger fire.

    ``status`` is ``pattern_absent`` (normal — most triggers have no
    pattern for this playbook), ``valid`` or ``invalid``.  There is
    no ``direction_mismatch`` status on this path by design.
    ``direction`` is always the trigger direction, never re-decided.
    """

    playbook: str
    status: str = "pattern_absent"
    direction: int = 0
    score: float = 0.0
    invalidation: object = None
    costs: object = None
    entry: object = None
    stop: object = None
    target: object = None
    pattern: str = ""
    location: object = None
    reason: str = "pattern_absent"
    trigger_context: dict = field(default_factory=dict)


def snapshot_trigger_context(ctx) -> dict:
    """Deep-copy a trigger context so one playbook cannot mutate
    another's snapshot.  Accepts a dict or an object with attributes.
    Normalised keys: ``direction``, ``zone``, ``invalidation``,
    ``ttl_bars``, ``zone_exit``, plus any other trigger fields,
    preserved verbatim.
    """
    if ctx is None:
        return {"direction": 0}
    if isinstance(ctx, dict):
        raw = copy.deepcopy(ctx)
    else:
        raw = {}
        for key in ("direction", "trigger_direction", "zone",
                    "invalidation", "ttl", "ttl_bars", "zone_exit",
                    "timestamp", "bar_index", "session", "killzone",
                    "amd_phase", "trigger_tf", "zone_tf", "bias_tf"):
            if hasattr(ctx, key):
                raw[key] = copy.deepcopy(getattr(ctx, key))
        # EngineContext-style objects carry no trigger fields; the
        # caller passes the trigger dict separately in that case.
    out = copy.deepcopy(raw)
    if "direction" not in out and "trigger_direction" in out:
        out["direction"] = out.pop("trigger_direction")
    out["direction"] = int(out.get("direction", 0) or 0)
    if "ttl_bars" not in out and "ttl" in out:
        out["ttl_bars"] = out["ttl"]
    return out


def evaluate_all_playbooks(playbooks, bars, ctx) -> list[PlaybookVerdict]:
    """Evaluate EVERY playbook exactly once per trigger fire.

    Each playbook gets its own deep-copied trigger snapshot, so
    evaluation order cannot leak state between playbooks.  A
    playbook that raises is reported ``invalid`` by ``evaluate``,
    never allowed to abort the remaining playbooks.
    """
    return [pb.evaluate(bars, ctx) for pb in playbooks]
