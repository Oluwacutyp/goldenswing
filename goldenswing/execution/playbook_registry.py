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

from ..playbooks import (AuctionLVNTraversalPlaybook,
                         AuctionValueAcceptancePlaybook,
                         AuctionValueRejectionPlaybook,
                         FVGLimitPlaybook, FVGMidlinePlaybook,
                         InversionPlaybook, JudasSwingPlaybook,
                         KillzoneOpenPlaybook, NarrativePlaybook,
                         OrderBlockPlaybook, QMD1Playbook, QMM5Playbook,
                         RangeCompressionBreakoutPlaybook,
                         RangeMeanReversionPlaybook, RangeQMPlaybook,
                         RangeSweepReclaimPlaybook,
                         RetracementTrapPlaybook,
                         SweepReversalPlaybook,
                         TrendImpulsePullbackPlaybook,
                         TrendMomentumAccelerationPlaybook,
                         TrendlineRetestPlaybook,
                         TurtleSoupPlaybook, VReversalPlaybook,
                         VolExpansionEscapePlaybook,
                         VolFailedExpansionPlaybook)

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
#: Route (MarketSituation Pass 7a, SPEC §6.2): the volatility-
#: expansion family's SCHEDULED evaluation contract — scans on
#: each formation clock's close (H4 / H1 / M15), reading the
#: MarketSituation's multi-TF volatility state.  The route is
#: SHADOW (it journals evaluations, arm records and analytical
#: candidates, proposes scenarios to the book, and holds NO
#: order authority in any mode).  Unlike the range family,
#: these playbooks are NOT members of the trigger-path roster:
#: they have no legacy order authority to preserve, so they
#: stay out of ROSTER_SPECS entirely and the trigger path
#: never evaluates them.  Promotion is the family's own
#: evaluation (execution/vol_promotion.py) plus the owner's
#: separate decision — the SPEC §6 common contract.
ROUTE_VOLATILITY_EXPANSION = "volatility_expansion"
#: Route (MarketSituation Pass 7b, SPEC §6.1): the trend /
#: momentum family's SCHEDULED evaluation contract — scans
#: on each TRIGGER clock's close (H1 for H4-formed trends,
#: H4 for D1-formed), reading the MarketSituation's per-TF
#: states and relationship kinds.  The route is SHADOW (it
#: journals evaluations, arm records and analytical
#: candidates, proposes scenarios to the book, and holds
#: NO order authority in any mode).  Like the vol family,
#: these playbooks are NOT members of the trigger-path
#: roster: they have no legacy order authority to preserve,
#: so they stay out of ROSTER_SPECS entirely.  Promotion
#: is the family's own evaluation
#: (execution/trend_promotion.py) plus the owner's separate
#: decision — the SPEC §6 common contract.
ROUTE_TREND_MOMENTUM = "trend_momentum"
#: Route (MarketSituation Pass 7c, SPEC §6.3): the auction /
#: value family's SCHEDULED evaluation contract — scans on
#: each TRIGGER clock's close (M15, for the H1-formed
#: developing value area), reading the developing value
#: state (live/developing_value.py — the §6.3 extension
#: the situation's location.value carries) on the H1
#: formation frame.  The route is SHADOW (it journals
#: evaluations, arm records and analytical candidates,
#: proposes scenarios to the book, and holds NO order
#: authority in any mode).  Like the vol and trend
#: families, these playbooks are NOT members of the
#: trigger-path roster: they have no legacy order
#: authority to preserve, so they stay out of ROSTER_SPECS
#: entirely.  Promotion is the family's own evaluation
#: (execution/auction_promotion.py) plus the owner's
#: separate decision — the SPEC §6 common contract.
ROUTE_AUCTION_VALUE = "auction_value"


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
    #: Setup-lifecycle contract (owner rulings 2026-10-30 §1):
    #: how many native-clock bars a scan-detected candidate stays
    #: a live, routable setup, and which trigger routes it.  Only
    #: the H1 roster's scheduled scans register setups in v1;
    #: native routes keep their candidate_lifecycle windows and
    #: the §G range scans stay journal-only shadow.
    validity_bars: int = 6
    trigger_kind: str = "h1_trigger_fire"
    #: Step 15 (owner rulings 2026-10-30 §2): the playbook's declared
    #: ENTRY THESIS — "limit" (the setup enters when price returns to
    #: its level: pullback / mitigation / retracement / return-to-zone)
    #: or "market" (the setup's premise is the trigger firing itself:
    #: confirmed breakout / displacement continuation / immediate
    #: reaction — a resting limit contradicts the premise).  The order
    #: path journals the intent per candidate and enforces the family
    #: in select_order_type.  Default "limit": every roster playbook's
    #: own definition is a level-return premise except v_reversal,
    #: which declares "market" below (its definition enters on the
    #: reclaim bar's close).
    entry_thesis: str = "limit"
    #: Step 16 (owner rulings 2026-10-30 §3): the playbook's declared
    #: FLIP PREMISE — the premise, in the playbook's own words, that
    #: an asymmetry structure flip (a close through the last
    #: defended pivot on the score frame) invalidates.  Non-empty =
    #: a flip hard-refuses for this playbook, with the premise NAMED
    #: and the flip evidence journaled on the refusal.  Empty (the
    #: default, and every playbook today) = the flip is weighed
    #: evidence only: it zeroes the candidate's structure share and
    #: is recorded as the strongest opposing scenario, never a veto.
    #: Declaration discipline: declare ONLY where the playbook's own
    #: definition states the defended-structure close-through kills
    #: the setup.  Every registered playbook's premise invalidation
    #: is its own declared level (the signal's invalidation / the
    #: zone's far edge), already enforced by the setup lifecycle
    #: (Step 14) and the hard lines — the score flip is a
    #: floated-clock context read, not the playbook's own level, so
    #: no playbook declares one.
    flip_premise: str = ""
    #: Step 18 (owner rulings 2026-10-30 §4): the playbook's
    #: declared MANAGEMENT HORIZON — "scalp" | "intraday" |
    #: "swing" — determined by the setup's thesis, structural
    #: objective, intended holding period and target timeframe,
    #: NOT by its entry timeframe.  Grounding (the multi-TF
    #: hierarchy doctrine: Swing D1 / Intraday H4 / Scalp H1):
    #: the H1 roster and the H4 range family run H4-context
    #: theses with session/day objectives — intraday, hence the
    #: default; the native routes declare their own (qm_m5 is
    #: an M5-clock scalp system; qm_d1's QM objective is a
    #: multi-day D1 structure — swing).  The horizon selects
    #: the management POLICY at the resolution point
    #: (management_policy_for); policies are built only from
    #: established, tested rule sets — new horizon-specific
    #: parameters require chronological, cost-aware demo
    #: evaluation before they may resolve here (rulings §4).
    management_horizon: str = "intraday"
    #: MarketSituation Pass 2 (SPEC §4.4 / §8.4): the chart
    #: studies this family declares for its chart clocks —
    #: METADATA ONLY in this pass; the consumer is the chart
    #: program (Pass 10).  Empty = the family declares no
    #: studies of its own (the renderer's base set applies).
    chart_studies: tuple[str, ...] = ()
    #: MarketSituation Pass 2 (SPEC §4.4 / §6): optional CLOCK
    #: ROLE overrides for families whose roles differ from the
    #: route default — (role, timeframe) pairs over the roles
    #: formation_tf / refinement_tf / trigger_tf / entry_tf /
    #: management_tf.  The scenario book reads them when it
    #: builds a scenario's clocks; empty = the route default
    #: (native / refinement / trigger / trigger / management
    #: context) applies.  Metadata only: no detector reads this.
    clocks: tuple[tuple[str, str], ...] = ()


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
                 ROUTE_H1_ROSTER, entry_thesis="market"),
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
                 shadow_evaluation=True, entry_thesis="market"),
    PlaybookSpec("narrative", NarrativePlaybook, "H1",
                 ROUTE_H1_ROSTER),
    # Volatility-expansion family (MarketSituation Pass 7a,
    # SPEC §6.2): vol-state-defined on any of its formation
    # clocks (H4/H1/M15 — the scan threads the clock per
    # evaluation; native_tf is the family's home clock for
    # bare/research use).  Scheduled shadow route only;
    # entry thesis is market (the premise is the accepted
    # escape / the rejection itself — a resting limit
    # contradicts it).  Horizon intraday (SPEC §6.2).
    PlaybookSpec("vol_expansion_escape", VolExpansionEscapePlaybook,
                 "H1", ROUTE_VOLATILITY_EXPANSION,
                 shadow_evaluation=True, entry_thesis="market",
                 chart_studies=("bollinger",)),
    PlaybookSpec("vol_failed_expansion", VolFailedExpansionPlaybook,
                 "H1", ROUTE_VOLATILITY_EXPANSION,
                 shadow_evaluation=True, entry_thesis="market",
                 chart_studies=("bollinger",)),
    # Trend / momentum family (MarketSituation Pass 7b,
    # SPEC §6.1): the trend STATE defined on its formation
    # clocks (H4 intraday / D1 swing — the scan threads the
    # clock per evaluation; native_tf is the family's
    # intraday home clock), entered on the trigger clock
    # one step down.  Scheduled shadow route only; entry
    # thesis is market (the premise is the trigger turn /
    # the acceleration break itself — a resting limit
    # contradicts it).  The registry horizon is the §6.1
    # default (intraday); a D1-formed instance declares
    # "swing" on its own signal (the additive declared-
    # horizon read in management_block_for / the book), so
    # its management resolves to the swing policy.
    # chart_studies is empty by declaration: the family
    # reads structure and velocity, never an indicator
    # overlay — phase by structure, never an EMA read.
    PlaybookSpec("trend_impulse_pullback",
                 TrendImpulsePullbackPlaybook,
                 "H4", ROUTE_TREND_MOMENTUM,
                 shadow_evaluation=True, entry_thesis="market"),
    PlaybookSpec("trend_momentum_acceleration",
                 TrendMomentumAccelerationPlaybook,
                 "H4", ROUTE_TREND_MOMENTUM,
                 shadow_evaluation=True, entry_thesis="market"),
    # Auction / value family (MarketSituation Pass 7c, SPEC
    # §6.3): the developing value area defined on its
    # formation clock (H1 — the day's bars; native_tf is
    # the family's formation clock), entered on the
    # trigger clock one step down (M15).  Scheduled shadow
    # route only; entry thesis is market (the premise is
    # the auction's answer itself — acceptance's retest
    # hold, rejection's first read, the traversal's
    # crossing — a resting limit contradicts it).  Horizon
    # intraday (SPEC §6.3): the registry default, declared
    # here by the absence of an override — every instance
    # is intraday because every sequence is day-bound.
    PlaybookSpec("auction_value_acceptance",
                 AuctionValueAcceptancePlaybook,
                 "H1", ROUTE_AUCTION_VALUE,
                 shadow_evaluation=True, entry_thesis="market",
                 chart_studies=("volume_profile",)),
    PlaybookSpec("auction_value_rejection",
                 AuctionValueRejectionPlaybook,
                 "H1", ROUTE_AUCTION_VALUE,
                 shadow_evaluation=True, entry_thesis="market",
                 chart_studies=("volume_profile",)),
    PlaybookSpec("auction_lvn_traversal",
                 AuctionLVNTraversalPlaybook,
                 "H1", ROUTE_AUCTION_VALUE,
                 shadow_evaluation=True, entry_thesis="market",
                 chart_studies=("volume_profile",)),
    # Native-clock routes: evaluated by dedicated runner paths on
    # their own frames (M5 closes / D1 closes).  They are NOT in
    # the H1 roster — feeding them H1 bars was the qm_m5 defect.
    PlaybookSpec("qm_m5", QMM5Playbook, "M5", ROUTE_NATIVE,
                 management_horizon="scalp"),
    PlaybookSpec("qm_d1", QMD1Playbook, "D1", ROUTE_NATIVE,
                 management_horizon="swing"),
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
#: ROSTER_SPECS for continuity with the Phase 1 invariant —
#: extended at Passes 7a, 7b and 7c: ROSTER_SPECS ∪
#: NATIVE_SPECS ∪ VOL_SPECS ∪ TREND_SPECS ∪ AUCTION_SPECS ==
#: the full registry, the VOL/TREND/AUCTION partitions being
#: the ones that are scheduled-shadow only and therefore
#: deliberately NOT in this trigger-path set.)
ROSTER_SPECS: tuple[PlaybookSpec, ...] = tuple(
    s for s in PLAYBOOK_REGISTRY
    if s.route in (ROUTE_H1_ROSTER, ROUTE_RANGE_H4_H1))

#: The native-clock route specs.
NATIVE_SPECS: tuple[PlaybookSpec, ...] = tuple(
    s for s in PLAYBOOK_REGISTRY if s.route == ROUTE_NATIVE)

#: The volatility-expansion family specs (Pass 7a): the
#: scheduled-shadow partition.  The registry partition is
#: ROSTER_SPECS ∪ NATIVE_SPECS ∪ VOL_SPECS == the full
#: registry (the Phase 1 invariant, extended by the one
#: route class that is neither trigger-path nor native).
VOL_SPECS: tuple[PlaybookSpec, ...] = tuple(
    s for s in PLAYBOOK_REGISTRY
    if s.route == ROUTE_VOLATILITY_EXPANSION)

#: The trend / momentum family specs (Pass 7b): the second
#: scheduled-shadow partition.  The registry partition is
#: ROSTER_SPECS ∪ NATIVE_SPECS ∪ VOL_SPECS ∪ TREND_SPECS ==
#: the full registry (extended at Pass 7c by AUCTION_SPECS
#: below).
TREND_SPECS: tuple[PlaybookSpec, ...] = tuple(
    s for s in PLAYBOOK_REGISTRY
    if s.route == ROUTE_TREND_MOMENTUM)

#: The auction / value family specs (Pass 7c): the third
#: scheduled-shadow partition.  The registry partition is
#: ROSTER_SPECS ∪ NATIVE_SPECS ∪ VOL_SPECS ∪ TREND_SPECS ∪
#: AUCTION_SPECS == the full registry.
AUCTION_SPECS: tuple[PlaybookSpec, ...] = tuple(
    s for s in PLAYBOOK_REGISTRY
    if s.route == ROUTE_AUCTION_VALUE)

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
    ROUTE_VOLATILITY_EXPANSION:
        "shadow evaluation route (Pass 7a): no floor, no order "
        "authority in any mode — the family is not on the "
        "trigger path; its candidates are journaled, judged "
        "in shadow and proposed to the scenario book only",
    ROUTE_TREND_MOMENTUM:
        "shadow evaluation route (Pass 7b): no floor, no order "
        "authority in any mode — the family is not on the "
        "trigger path; its candidates are journaled, judged "
        "in shadow and proposed to the scenario book only",
    ROUTE_AUCTION_VALUE:
        "shadow evaluation route (Pass 7c): no floor, no order "
        "authority in any mode — the family is not on the "
        "trigger path; its candidates are journaled, judged "
        "in shadow and proposed to the scenario book only",
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


#: Step 18: the management horizons a setup may declare.
MANAGEMENT_HORIZONS: tuple[str, ...] = ("scalp", "intraday", "swing")

#: Step 18: management policies, by id -> version.  A policy is a
#: named, versioned bundle of ESTABLISHED, tested management
#: rules (rulings 2026-10-30 §4: no invented R thresholds, ATR
#: multipliers or point distances):
#: - "fixed_bracket_v1": Step 13's rule set — the placement
#:   bracket only (initial structural stop + broker target; no
#:   profit-lock / breakeven / trail / Chandelier movement).
#: - "standard_v1": the desk's standing TickConfig exactly as
#:   the runner constructs it (earned breakeven, profit lock,
#:   R-trail, Chandelier, tiered partials).
#: - "standard_scalp_v1" / "standard_swing_v1" (MarketSituation
#:   Pass 6, SPEC §7.1): the horizon resolutions.  Their
#:   parameters come ONLY from established rule sets — the
#:   scalp policy is Hot Scalp's own established treatment
#:   (its profile, its earned-by-acceptance profit lock, its
#:   sleeve target hierarchy); the swing policy is the standard
#:   profile read on the D1 management clock with structure
#:   trail and value partials.  Where no established rule set
#:   exists for a horizon parameter, the parameter inherits
#:   standard_v1's value (POLICY_PARAMS says so per policy).
MANAGEMENT_POLICIES: dict[str, int] = {
    "fixed_bracket_v1": 1,
    "standard_v1": 1,
    "standard_scalp_v1": 1,
    "standard_swing_v1": 1,
}

#: MarketSituation Pass 6 (SPEC §7.1): each policy's parameter
#: bundle — the canonical profile style its declared plan is
#: built from, how its profit lock arms, where its partial
#: schedule's targets come from, and whether its plan arms the
#: thesis exit (a parent S2 thesis invalidation exits the
#: position on that evidence).  "inherits_standard" names the
#: fields a horizon policy deliberately takes from standard_v1
#: unchanged (the spec's inheritance rule, stated not hidden).
POLICY_PARAMS: dict[str, dict] = {
    "fixed_bracket_v1": {
        "profile_style": None,
        "profit_lock_mode": "none",
        "partial_source": "placement_bracket",
        "thesis_exit": None,
        "inherits_standard": (),
    },
    "standard_v1": {
        "profile_style": "trend",
        "profit_lock_mode": "standard",
        "partial_source": "tick_tiers",
        "thesis_exit": "parent_thesis_invalidation",
        "inherits_standard": (),
    },
    "standard_scalp_v1": {
        # Hot Scalp's established rule set, lifted: the scalp
        # profile (earned BE, 1.5R backstop, ATR trail), the
        # earned-by-acceptance profit lock (TickConfig's
        # profit_lock_earned_playbooks treatment), and partial
        # targets from the sleeve's own target hierarchy.
        "profile_style": "scalp",
        "profit_lock_mode": "earned_acceptance",
        "partial_source": "sleeve_target_hierarchy",
        "thesis_exit": "parent_thesis_invalidation",
        "inherits_standard": ("partial_tiers", "tier_min_volume",
                              "breakeven_backstop_r"),
    },
    "standard_swing_v1": {
        # The standard profile read on the swing management
        # clock (D1): structure trail, progress windows in D1
        # bars (the XD-4 conversion), partial style "value"
        # (bank at value/zone, per the profile vocabulary).
        "profile_style": "trend",
        "partial_style_override": "value",
        "profit_lock_mode": "standard",
        "partial_source": "tick_tiers",
        "thesis_exit": "parent_thesis_invalidation",
        "inherits_standard": ("partial_tiers", "tier_min_volume",
                              "breakeven_mode",
                              "breakeven_backstop_r",
                              "profit_lock"),
    },
}


def policy_params_for(policy_id: str) -> dict:
    """The Pass 6 parameter bundle for a resolved policy id
    (a copy; {} for an unknown id)."""
    params = POLICY_PARAMS.get(str(policy_id))
    return dict(params) if params is not None else {}


def management_policy_for_horizon(horizon: str) -> tuple[str, int]:
    """The Pass 6 horizon resolution itself (policy_id,
    version) for a DECLARED horizon — the rule
    management_policy_for applies to a playbook's registry
    horizon, factored out so an instance that declares its
    own horizon (Pass 7b: a D1-formed trend scenario
    declares "swing" on its signal) resolves through the
    SAME rule, never a second table.  qm_m5's fixed bracket
    is a NAME resolution, not a horizon one — it does not
    live here and no horizon input can reach it."""
    if horizon == "scalp":
        return ("standard_scalp_v1",
                MANAGEMENT_POLICIES["standard_scalp_v1"])
    if horizon == "swing":
        return ("standard_swing_v1",
                MANAGEMENT_POLICIES["standard_swing_v1"])
    return "standard_v1", MANAGEMENT_POLICIES["standard_v1"]


def management_policy_for(name: str) -> tuple[str, int]:
    """Resolve a playbook's management policy (policy_id,
    version).  Step 18 resolved only qm_m5's fixed bracket;
    MarketSituation Pass 6 (SPEC §7.1) completes the horizon
    resolution the Step-18 declaration was plumbed for: qm_m5
    keeps fixed_bracket_v1 (D7 — untouched), the scalp horizon
    resolves to standard_scalp_v1, the swing horizon to
    standard_swing_v1, and intraday keeps standard_v1 with
    byte-identical parameters (the flip renames nothing about
    how an intraday position is managed)."""
    if str(name) == "qm_m5":
        return "fixed_bracket_v1", MANAGEMENT_POLICIES["fixed_bracket_v1"]
    return management_policy_for_horizon(
        management_horizon_for(name))


def management_horizon_for(name: str) -> str:
    """Step 18: a playbook's declared management horizon.
    Unregistered names (hot_scalp) declare through their route:
    hot scalp is a scalp system by construction; anything else
    unregistered falls back to the registry default."""
    spec = _BY_NAME.get(str(name))
    if spec is not None:
        return spec.management_horizon
    return "scalp" if str(name) == "hot_scalp" else "intraday"


def native_tf_of(name: str) -> str | None:
    """A playbook's native clock (None when unregistered)."""
    spec = _BY_NAME.get(str(name))
    return spec.native_tf if spec is not None else None
