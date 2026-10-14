"""Playbook interface (Phase 9).

A playbook is a self-contained entry system: it scans chart state for
its setup, fires its trigger, and simulates its outcome with the EXACT
machinery it was validated with.  The confluence engine never
re-simulates and never edits a playbook's outcome — it only ranks
pre-simulated candidates and decides which (if any) enters the book.
"""
from __future__ import annotations

import abc
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
