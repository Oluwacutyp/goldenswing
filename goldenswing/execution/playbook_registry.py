"""The ONE playbook registry for the MT5 engine (funnel patch §3).

Before this module, the playbook roster existed in two places that
had drifted apart: the runner's ``_get_playbooks()`` (17 names) and
``mt5_config.ALL_PLAYBOOK_NAMES`` (14).  The four range playbooks
were missing from the config list, which meant ``/set
playbook.range_qm off`` was rejected as "unknown playbook" and a
persisted kill-switch entry for one of them was silently dropped by
``mt5_config._clean`` — four live, executing playbooks could not be
killed from Telegram at all.

This registry is the single source of truth.  Everything else is
derived from it:

- the runner's H1-clocked roster (``_get_playbooks``),
- the native-clock routes (qm_m5 on M5 closes, qm_d1 on D1 closes),
- ``mt5_config``'s manageable name lists (``/set`` validation,
  ``_clean`` sanitisation, ``/config`` / ``/playbooks`` display).

Each entry declares the playbook's NATIVE clock and its route.  A
playbook is evaluated on its own clock or not at all: running one
timeframe's logic on another timeframe's bars mislabels the
patterns (the rule ``_get_playbooks``' docstring always stated for
qm_d1 — the registry makes it structural for every playbook).

Invariant (CI-pinned in tests/test_funnel_m5.py): roster routes
∪ native routes == the config-manageable set, exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..playbooks import (FVGLimitPlaybook, FVGMidlinePlaybook,
                         InversionPlaybook, JudasSwingPlaybook,
                         KillzoneOpenPlaybook, NarrativePlaybook,
                         OrderBlockPlaybook, QMD1Playbook, QMM5Playbook,
                         RangeCompressionBreakoutPlaybook,
                         RangeMeanReversionPlaybook, RangeQMPlaybook,
                         RangeSweepReclaimPlaybook,
                         RetracementTrapPlaybook,
                         SweepReversalPlaybook, TrendlineRetestPlaybook,
                         TurtleSoupPlaybook, VReversalPlaybook)

#: Route: evaluated inside the H1-clocked signal path (the roster).
ROUTE_H1_ROSTER = "h1_roster"
#: Route: evaluated on its own native clock by a dedicated runner
#: path (``_run_qm_m5`` / ``_run_qm_d1``), never through the roster.
ROUTE_NATIVE = "native"
#: Route (Phase 2, owner §G ruling): the range family's SCHEDULED
#: evaluation contract — H4-close scenario generation + H1
#: refinement.  The §G evaluation routes are SHADOW (they journal
#: evaluations and analytical candidates; they hold no order
#: authority).  Range playbooks ALSO remain members of the
#: trigger-path roster below — that legacy order authority is a
#: standing owner state (2026-10-22 promotion) and is NOT
#: silently removed by declaring the §G contract here (rulings
#: §G.2: no silent live-decision changes).  Promotion of the §G
#: route itself is a separate owner decision.
ROUTE_RANGE_H4_H1 = "range_h4_h1"


@dataclass(frozen=True)
class PlaybookSpec:
    """One registered playbook: identity, class, native clock,
    route.  ``manageable`` is every registered playbook today (the
    kill switch is universal); the field exists so a future
    internal-only playbook must say so explicitly, in one place.

    ``native_tf`` is the playbook's GENERATION clock (the frame
    its patterns form on); ``refinement_tf`` is set only where a
    second clock refines entries (the range family: H4 generates,
    H1 refines).  ``shadow_evaluation`` marks a scheduled
    evaluation route that journals but holds no order authority
    (the §G range routes in v1)."""
    name: str
    playbook_class: type
    native_tf: str
    route: str
    manageable: bool = True
    refinement_tf: str | None = None
    shadow_evaluation: bool = False


#: The registry, in canonical display order: the H1-clocked roster
#: first (in signal-path evaluation order), then the native-clock
#: routes.
PLAYBOOK_REGISTRY: tuple[PlaybookSpec, ...] = (
    PlaybookSpec("sweep_reversal", SweepReversalPlaybook, "H1",
                 ROUTE_H1_ROSTER),
    PlaybookSpec("fvg_midline", FVGMidlinePlaybook, "H1",
                 ROUTE_H1_ROSTER),
    PlaybookSpec("trendline_retest", TrendlineRetestPlaybook, "H1",
                 ROUTE_H1_ROSTER),
    PlaybookSpec("fvg_limit", FVGLimitPlaybook, "H1",
                 ROUTE_H1_ROSTER),
    PlaybookSpec("inversion", InversionPlaybook, "H1",
                 ROUTE_H1_ROSTER),
    PlaybookSpec("killzone_open", KillzoneOpenPlaybook, "H1",
                 ROUTE_H1_ROSTER),
    PlaybookSpec("judas_swing", JudasSwingPlaybook, "H1",
                 ROUTE_H1_ROSTER),
    PlaybookSpec("turtle_soup", TurtleSoupPlaybook, "H1",
                 ROUTE_H1_ROSTER),
    PlaybookSpec("v_reversal", VReversalPlaybook, "H1",
                 ROUTE_H1_ROSTER),
    PlaybookSpec("retracement_trap", RetracementTrapPlaybook, "H1",
                 ROUTE_H1_ROSTER),
    PlaybookSpec("order_block", OrderBlockPlaybook, "H1",
                 ROUTE_H1_ROSTER),
    # Range family (§G contract): scenarios GENERATE on H4
    # closes, entries REFINE on H1 closes, D1 is context only.
    # The scheduled §G evaluations are shadow in v1 (see
    # ROUTE_RANGE_H4_H1); trigger-path membership is retained.
    PlaybookSpec("range_sweep_reclaim", RangeSweepReclaimPlaybook,
                 "H4", ROUTE_RANGE_H4_H1, refinement_tf="H1",
                 shadow_evaluation=True),
    PlaybookSpec("range_qm", RangeQMPlaybook,
                 "H4", ROUTE_RANGE_H4_H1, refinement_tf="H1",
                 shadow_evaluation=True),
    PlaybookSpec("range_mean_reversion", RangeMeanReversionPlaybook,
                 "H4", ROUTE_RANGE_H4_H1, refinement_tf="H1",
                 shadow_evaluation=True),
    PlaybookSpec("range_compression_breakout",
                 RangeCompressionBreakoutPlaybook,
                 "H4", ROUTE_RANGE_H4_H1, refinement_tf="H1",
                 shadow_evaluation=True),
    PlaybookSpec("narrative", NarrativePlaybook, "H1",
                 ROUTE_H1_ROSTER),
    # Native-clock routes: evaluated by dedicated runner paths on
    # their own frames (M5 closes / D1 closes).  They are NOT in
    # the H1 roster — feeding them H1 bars was the qm_m5 defect.
    PlaybookSpec("qm_m5", QMM5Playbook, "M5", ROUTE_NATIVE),
    PlaybookSpec("qm_d1", QMD1Playbook, "D1", ROUTE_NATIVE),
)

_BY_NAME: dict[str, PlaybookSpec] = {s.name: s
                                    for s in PLAYBOOK_REGISTRY}

#: Every registered playbook name, registry order.
PLAYBOOK_NAMES: tuple[str, ...] = tuple(
    s.name for s in PLAYBOOK_REGISTRY)

#: The pure H1-clocked roster specs (no second clock).
ROSTER_SPECS_H1: tuple[PlaybookSpec, ...] = tuple(
    s for s in PLAYBOOK_REGISTRY if s.route == ROUTE_H1_ROSTER)

#: The range-family specs (§G H4-generation / H1-refinement).
RANGE_SPECS: tuple[PlaybookSpec, ...] = tuple(
    s for s in PLAYBOOK_REGISTRY if s.route == ROUTE_RANGE_H4_H1)

#: The trigger-path roster: every playbook the H1-clocked signal
#: path evaluates when the trigger lifecycle fires — the pure H1
#: roster PLUS the range family (whose scheduled §G evaluation
#: contract above is additional, shadow, and separate).  This is
#: the set ``_get_playbooks`` builds, in registry order.  (Named
#: ROSTER_SPECS for continuity with the Phase 1 invariant:
#: ROSTER_SPECS ∪ NATIVE_SPECS == the full registry.)
ROSTER_SPECS: tuple[PlaybookSpec, ...] = tuple(
    s for s in PLAYBOOK_REGISTRY
    if s.route in (ROUTE_H1_ROSTER, ROUTE_RANGE_H4_H1))

#: The native-clock route specs.
NATIVE_SPECS: tuple[PlaybookSpec, ...] = tuple(
    s for s in PLAYBOOK_REGISTRY if s.route == ROUTE_NATIVE)

#: SCORE-FLOOR AUTHORITY, per route (Phase 2, P0-2).  Two scores
#: exist and were conflated in Phase 1 records: the SLEEVE BASE
#: score (a playbook-declared placeholder that ranks sleeves —
#: e.g. QM's documented 60.0, NOT a measured quality score) and
#: the ASYMMETRY score (the computed decision score; its
#: breakdown total is the number decisions actually use).  The
#: config ``min_score = 60.0`` is a legacy default that NO live
#: route applies as a flat floor any more; the floors that DO
#: apply, per route, are:
FLOOR_AUTHORITY: dict[str, str] = {
    ROUTE_H1_ROSTER:
        "per-style Asymmetry floors: floor_for(style_key) rows "
        "(total + component minimums) applied to the Asymmetry "
        "total in _run_one_playbook; cfg.min_score is NOT the "
        "roster floor (the flat floor was removed in the "
        "confluence rework)",
    ROUTE_RANGE_H4_H1:
        "shadow evaluation route: no floor, no order authority. "
        "When the legacy trigger path fires for a range "
        "playbook, the ROUTE_H1_ROSTER floor authority applies",
    ROUTE_NATIVE + ":qm_m5":
        "NO score floor (judgment route): the weighed Asymmetry "
        "evidence decides; the sleeve base score (60.0) is a "
        "ranking placeholder and is never a floor input",
    ROUTE_NATIVE + ":qm_d1":
        "MACRO_SWING_OPEN Asymmetry floor (Path B): floor_for "
        "row applied to the Asymmetry total in "
        "_execute_qm_d1_candidate",
}


def decay_exempt(playbook_name: str) -> bool:
    """True when a playbook's route carries NO score floor, so the
    legacy score-decay cancellation must not touch its pendings.

    Step 2 (2026-10-27): decay re-scores on the H1 roster frame
    against ``min_score`` (legacy 60.0) — but the judgment route
    (native qm_m5) has NO floor by FLOOR_AUTHORITY, and native
    playbooks are not even in the H1 roster, so a re-score returns
    "gone" for every native pending and decay cancels it within
    ``score_decay_bars`` H1 closes (the 2026-10-26 4019.16 cancel).
    Exempt routes skip floor-decay until step 4 restores a
    native-frame decay; expiry, bias-flip and invalidation still
    guard the pending.  Roster and range routes keep their floors.
    """
    text = FLOOR_AUTHORITY.get("native:" + playbook_name, "")
    return text.startswith("NO score floor")

#: Names the Telegram control surface may manage (kill switch,
#: /config, /playbooks).  Today: every registered playbook.
MANAGEABLE_NAMES: tuple[str, ...] = tuple(
    s.name for s in PLAYBOOK_REGISTRY if s.manageable)


def spec_for(name: str) -> PlaybookSpec | None:
    """The registry entry for ``name`` (None when unregistered)."""
    return _BY_NAME.get(str(name))


def native_tf_of(name: str) -> str | None:
    """A playbook's native clock (None when unregistered)."""
    spec = _BY_NAME.get(str(name))
    return spec.native_tf if spec is not None else None
