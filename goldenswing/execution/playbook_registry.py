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


@dataclass(frozen=True)
class PlaybookSpec:
    """One registered playbook: identity, class, native clock,
    route.  ``manageable`` is every registered playbook today (the
    kill switch is universal); the field exists so a future
    internal-only playbook must say so explicitly, in one place."""
    name: str
    playbook_class: type
    native_tf: str
    route: str
    manageable: bool = True


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
    PlaybookSpec("range_sweep_reclaim", RangeSweepReclaimPlaybook,
                 "H1", ROUTE_H1_ROSTER),
    PlaybookSpec("range_qm", RangeQMPlaybook, "H1", ROUTE_H1_ROSTER),
    PlaybookSpec("range_mean_reversion", RangeMeanReversionPlaybook,
                 "H1", ROUTE_H1_ROSTER),
    PlaybookSpec("range_compression_breakout",
                 RangeCompressionBreakoutPlaybook, "H1",
                 ROUTE_H1_ROSTER),
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

#: The H1-clocked roster specs (runner signal-path order).
ROSTER_SPECS: tuple[PlaybookSpec, ...] = tuple(
    s for s in PLAYBOOK_REGISTRY if s.route == ROUTE_H1_ROSTER)

#: The native-clock route specs.
NATIVE_SPECS: tuple[PlaybookSpec, ...] = tuple(
    s for s in PLAYBOOK_REGISTRY if s.route == ROUTE_NATIVE)

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
