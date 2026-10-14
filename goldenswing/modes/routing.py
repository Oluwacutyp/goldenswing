"""Phase→playbook routing: measured, not believed.

The owner's architecture principle: *the bot must know what works best
in situations and phases — and the phase decides the playbook.*  This
module makes that a measurement, not an opinion:

1. ``build_matrix`` slices an EXISTING shadow book (every signal
   simulated standalone) by (phase at the signal bar × playbook) and
   reports trades / win rate / avg R per cell.
2. ``derive_route_table`` turns the matrix into muting rules: a cell
   whose measured avg R is negative on at least ``MUTE_MIN_N`` trades
   is muted — signals of that playbook in that phase are capped just
   below the golden threshold, so they are visible but never traded.
   Cells with less data are left alone: absence of evidence is not
   evidence, in either direction.

Honesty note: the matrix is computed from the same window it then
routes on (that is the Phase-5 method — derive the table from the
shadow book, re-backtest with it on).  The routing on/off comparison
is therefore IN-SAMPLE evidence about the routing mechanism, not
out-of-sample proof.  MODES.md says so, in those words.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..golden.detector import PLAYBOOKS, playbook_of

#: Minimum sample before a losing cell may be muted.  Below this the
#: cell is reported but never routed on.
MUTE_MIN_N = 30

#: Phase groups for the matrix: the AMD machine's eight raw labels
#: collapse to the owner's five situations + bases + unknown.
_PHASE_GROUPS = {
    "accumulation": "accumulation",
    "distribution": "distribution",
    "consolidation": "consolidation",
    "manipulation": "manipulation",
    "expansion_up": "expansion",
    "expansion_down": "expansion",
    "trend_up": "trend",
    "trend_down": "trend",
    "unknown": "unknown",
}

PHASE_GROUPS = ("accumulation", "distribution", "consolidation",
                "manipulation", "expansion", "trend", "unknown")


def phase_group(raw: str | None) -> str:
    """Group a raw AMD phase label for the routing matrix."""
    if raw is None:
        return "unknown"
    return _PHASE_GROUPS.get(raw, "unknown")


@dataclass(frozen=True)
class CellStats:
    n: int = 0
    win_rate: float = 0.0     # %
    avg_r: float = 0.0
    total_r: float = 0.0


def build_matrix(outcomes, states) -> dict[tuple[str, str], CellStats]:
    """Slice shadow-book outcomes by (phase group × playbook).

    ``outcomes`` are harness Outcomes (each signal simulated
    standalone); ``states`` are the featured MarketStates the signals
    were scanned from — the phase is the AMD label AT the signal bar,
    and the playbook is the detector's own ``playbook_of`` definition,
    so the matrix and the router can never drift apart.
    """
    agg: dict[tuple[str, str], list[float]] = {}
    for o in outcomes:
        st = states[o.signal.index]
        raw = st.features.phase.phase if st.features is not None else None
        key = (phase_group(raw), playbook_of(st, o.signal.kind))
        agg.setdefault(key, []).append(float(o.r_net))
    cells: dict[tuple[str, str], CellStats] = {}
    for key, rs in agg.items():
        arr = np.array(rs)
        cells[key] = CellStats(
            n=int(len(arr)),
            win_rate=float(np.mean(arr > 0) * 100.0),
            avg_r=float(arr.mean()),
            total_r=float(arr.sum()))
    return cells


@dataclass
class RouteTable:
    """Muting rules derived from a measured matrix.

    ``cap_for`` is the duck-typed hook the detector calls: it returns
    the score cap for a (phase, playbook) pair, or None when the pair
    is not muted.  The cap sits one point below the golden threshold
    (59 vs 60) — the same "visible but never a Golden Swing" pattern
    as the Phase-4 MTF veto.
    """

    cells: dict = field(default_factory=dict)
    muted: frozenset = frozenset()
    min_n: int = MUTE_MIN_N
    cap: float = 59.0

    def cap_for(self, phase_raw: str | None, playbook: str) -> float | None:
        if (phase_group(phase_raw), playbook) in self.muted:
            return self.cap
        return None


def derive_route_table(cells: dict[tuple[str, str], CellStats],
                       min_n: int = MUTE_MIN_N,
                       golden_threshold: float = 60.0) -> RouteTable:
    """Mute exactly the cells that are measured losers with enough
    sample: avg R < 0 AND n >= min_n.  Nothing else is touched."""
    muted = frozenset(
        key for key, c in cells.items()
        if c.n >= min_n and c.avg_r < 0.0)
    return RouteTable(cells=dict(cells), muted=muted, min_n=min_n,
                      cap=golden_threshold - 1.0)


def matrix_rows(cells: dict[tuple[str, str], CellStats]):
    """Full (phase × playbook) grid for reporting — observed cells
    plus explicit empty cells, in stable order."""
    for ph in PHASE_GROUPS:
        for pb in PLAYBOOKS:
            yield ph, pb, cells.get((ph, pb), CellStats())
