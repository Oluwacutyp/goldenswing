"""Context-performance maps (Phase 10 methodology).

The goal of a backtest is not a binary killed/adopted — it is a MAP of
where the strategy genuinely performed well: by session, by regime, by
HTF state, by AMD phase, by day-of-week.  Every playbook carries its
map, measured not asserted; the map becomes the playbook's declared
hunting ground (data, not prose).

A hunting ground has two parts, both derived from the measured map by
conservative rules (see :func:`derive_hunting_ground`):

- ``excluded``: (dimension, level) cells where the playbook is
  decisively bad.  The engine does not let the playbook fire there at
  all.  Exclusions are verified against the SEQUENTIAL book before
  adoption — removing trades re-times everything downstream
  (position-lock), and a cell that looks bad in isolation can still be
  load-bearing in sequence (the Phase-5 routing lesson).
- ``bonuses``: (dimension, level) -> bonus size for cells where the
  playbook is decisively good.  The confluence engine's context
  bonuses come from these maps, not from hand-set priors.

Conservative derivation rules (documented, not fitted):

- ``MIN_CELL_N = 25``: no verdict on thinner cells — the map shows
  them, the hunting ground ignores them.
- Exclusion: avg R < -0.05 with n >= 25 AND negative in both time
  halves (each half n >= 10).  A one-regime artifact is not a hunting
  ground.
- Proven (bonus): avg R > overall avg R + 0.15 with n >= 25.
  Bonus size scales with the edge: +2 below +0.30, +4 at/above.
"""
from __future__ import annotations

from dataclasses import dataclass, field


#: Minimum cell sample for any hunting-ground verdict.  Thinner cells
#: are measured and shown but never gated or bonused.
MIN_CELL_N = 25
#: Exclusion needs materially negative average R, not just noise.
EXCLUDE_MAX_AVG_R = -0.05
#: A proven cell must beat the playbook's overall average R by this.
PROVEN_EDGE_R = 0.15
#: Bonus tiers for proven cells.
BONUS_TIER_1 = 2.0
BONUS_TIER_2 = 4.0
PROVEN_TIER2_EDGE_R = 0.30


def _phase_of(state) -> str:
    feats = getattr(state, "features", None)
    ph = getattr(feats, "phase", None) if feats is not None else None
    p = getattr(ph, "phase", None) if ph is not None else None
    return str(p) if p else "unknown"


def _dow_of(state) -> str:
    ts = getattr(state, "timestamp", None)
    try:
        return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][
            int(ts.dayofweek)]
    except Exception:
        return "unknown"


def _htf_of(state, direction: int) -> str:
    b = state.daily_bias
    if b == direction:
        return "with-bias"
    if b == -direction:
        return "against-bias"
    return "no-bias"


#: Context dimensions: name -> (state, direction) -> level label.
DIMENSIONS: dict[str, object] = {
    "session": lambda s, d: s.session,
    "killzone": lambda s, d: "killzone" if s.killzone else "off-killzone",
    "vol_regime": lambda s, d: s.vol_regime,
    "trend": lambda s, d: s.trend,
    "htf": _htf_of,
    "phase": lambda s, d: _phase_of(s),
    "dow": lambda s, d: _dow_of(s),
}


@dataclass
class CellStats:
    n: int = 0
    wins: int = 0
    total_r: float = 0.0
    half1_r: float = 0.0
    half1_n: int = 0
    half2_r: float = 0.0
    half2_n: int = 0

    @property
    def avg_r(self) -> float:
        return self.total_r / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {"n": self.n, "wins": self.wins,
                "win_rate": self.wins / self.n if self.n else 0.0,
                "avg_r": self.avg_r, "total_r": self.total_r,
                "half1_n": self.half1_n,
                "half1_avg_r": self.half1_r / self.half1_n
                if self.half1_n else None,
                "half2_n": self.half2_n,
                "half2_avg_r": self.half2_r / self.half2_n
                if self.half2_n else None}


def measure_map(outcomes: list, states: list) -> dict[str, dict[str, dict]]:
    """Measure where the R concentrates, per dimension per level.

    ``outcomes`` are pre-simulated trades (Outcome, with
    ``signal.index`` the signal bar and ``r_net`` the result).
    Returns ``{dim: {level: cell_dict}}``.  Halves split by signal-bar
    order (first half vs second half of the window).
    """
    n = len(outcomes)
    half_cut = n // 2
    # order outcomes by signal bar for the halves split
    ordered = sorted(outcomes, key=lambda o: o.signal.index)
    half_of = {id(o): (0 if i < half_cut else 1)
               for i, o in enumerate(ordered)}
    result: dict[str, dict[str, CellStats]] = {
        dim: {} for dim in DIMENSIONS}
    for o in outcomes:
        t = o.signal.index
        if t >= len(states):
            continue
        s = states[t]
        d = o.signal.direction
        half = half_of[id(o)]
        for dim, fn in DIMENSIONS.items():
            try:
                level = str(fn(s, d))
            except Exception:
                level = "unknown"
            cell = result[dim].setdefault(level, CellStats())
            cell.n += 1
            cell.total_r += o.r_net
            if o.r_net > 0:
                cell.wins += 1
            if half == 0:
                cell.half1_n += 1
                cell.half1_r += o.r_net
            else:
                cell.half2_n += 1
                cell.half2_r += o.r_net
    return {dim: {lvl: c.to_dict() for lvl, c in levels.items()}
            for dim, levels in result.items()}


@dataclass
class HuntingGround:
    """A playbook's declared hunting ground, derived from its measured
    context map.  ``excluded`` cells never fire; ``bonuses`` map
    (dimension, level) -> confluence bonus for proven cells."""

    excluded: set[tuple[str, str]] = field(default_factory=set)
    bonuses: dict[tuple[str, str], float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"excluded": sorted(f"{d}:{l}" for d, l in self.excluded),
                "bonuses": {f"{d}:{l}": b
                            for (d, l), b in self.bonuses.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "HuntingGround":
        hg = cls()
        for item in d.get("excluded", []):
            dim, _, lvl = item.partition(":")
            hg.excluded.add((dim, lvl))
        for key, bonus in d.get("bonuses", {}).items():
            dim, _, lvl = key.partition(":")
            hg.bonuses[(dim, lvl)] = float(bonus)
        return hg


def derive_hunting_ground(map_data: dict[str, dict[str, dict]],
                          overall_avg_r: float) -> HuntingGround:
    """Derive the hunting ground from a measured map, by the
    conservative rules documented in the module docstring.  Pure
    function of the map — no book access, no fitting knobs."""
    hg = HuntingGround()
    for dim, levels in map_data.items():
        for level, c in levels.items():
            n = c["n"]
            if n < MIN_CELL_N:
                continue
            avg = c["avg_r"]
            if avg < EXCLUDE_MAX_AVG_R:
                h1, h2 = c["half1_avg_r"], c["half2_avg_r"]
                if h1 is not None and h2 is not None and h1 < 0 and h2 < 0 \
                        and c["half1_n"] >= 10 and c["half2_n"] >= 10:
                    hg.excluded.add((dim, level))
            elif avg > overall_avg_r + PROVEN_EDGE_R:
                bonus = BONUS_TIER_2 \
                    if avg > overall_avg_r + PROVEN_TIER2_EDGE_R \
                    else BONUS_TIER_1
                hg.bonuses[(dim, level)] = bonus
    return hg


def context_levels(states: list, signal_index: int,
                   direction: int) -> dict[str, str]:
    """The context levels of one signal bar, for hunting-ground
    checks and bonus lookup.  Missing state -> empty (no bonuses,
    no exclusions)."""
    if signal_index >= len(states):
        return {}
    s = states[signal_index]
    out: dict[str, str] = {}
    for dim, fn in DIMENSIONS.items():
        try:
            out[dim] = str(fn(s, direction))
        except Exception:
            out[dim] = "unknown"
    return out
