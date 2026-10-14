"""Area of Concentration (AOC) — the pivot zone (Phase 4.5).

Mined from a trading comment section the owner brought in
(2026-10-11): *"AOC — AREA OF CONCENTRATION, or your pivot zone:
support & resistance levels."*  The concept: where several
INDEPENDENT level families agree within a tight band — a liquidity
pool, a trendline, a demand zone, a Fib level and the daily swing
high all sitting inside half an ATR of each other — a real trader
treats the area as a pivot, not any one of the levels alone.

The system already *has* all the levels; it never measured their
concentration.  This family does:

1. **Harvest** every live level at the bar from confirmed state only
   (pools, supply/demand zones, order blocks, FVGs, trendlines, Fib
   levels, daily swings, the dealing-range EQ).  Mitigated/swept
   structures are excluded by construction — every source below is
   an *active* list.
2. **Cluster** levels within ``aoc_tol_atr`` of each other (greedy
   price sweep, cluster span capped at 1.0 ATR so a dense ladder of
   pools cannot fuse into one meaningless mega-zone).
3. **Score** each cluster = sum of ``weight x recency`` over members:
   concentration, not size.  A fresh pool + a fresh zone + an active
   trendline agreeing outranks a lone strong pool.

Family priors (hand-set, documented — the same honesty standard as
the detector's other priors):

- ``pool`` 0.8 + 0.2 x strength  — strength counts the swing points
  clustered at the level, so equal highs/lows count more;
- ``sd_zone`` 1.0 fresh / 0.6 tested — a tested zone is partly spent;
- ``ob`` 1.0 — institutional footprints are first-class levels;
- ``fvg`` 0.7 — gaps are reaction levels, softer than blocks;
- ``trendline`` 1.0 ACTIVE / 0.6 FORMING — the retest validates;
- ``fib`` 0.7 — a Fib level means something only as confluence;
- ``daily`` 1.2 — daily swings are the market's agreed pivots;
- ``eq`` 0.5 — the dealing-range equilibrium, a soft magnet.

Recency: ``0.5 + 0.5 * 2^(-age / half_life)`` — a fresh level counts
twice an ancient one, but an old unmitigated level never drops to
zero (a six-month high is still a high).  Timeless levels (fib
levels of the current leg, today's daily swings, the current EQ)
carry age 0.

An AOC needs ``aoc_min_families`` (2) DISTINCT families; a MAJOR
pivot zone needs ``aoc_major_families`` (3).  Families dedupe inside
a cluster — two pools and a zone is 2 families, not 3.

Causality: ``compute`` is a pure function of the bar's confirmed
``MarketState`` and its partially-built ``FeatureSet`` (zones,
trendlines, fib) — all themselves causal.  No df access, no
forward reference.  The truncation test proves it.
"""
from __future__ import annotations

import math

from .config import FeatureConfig
from .state import AOCMember, AOCState, AOCZone

#: family -> base weight (see module docstring for the rationale)
_POOL_W = 0.8
_POOL_STRENGTH_W = 0.2
_ZONE_FRESH_W = 1.0
_ZONE_TESTED_W = 0.6
_OB_W = 1.0
_FVG_W = 0.7
_TL_ACTIVE_W = 1.0
_TL_FORMING_W = 0.6
_FIB_W = 0.7
_DAILY_W = 1.2
_EQ_W = 0.5

#: a cluster may never span more than this (ATR) — the anti-fusion cap
_MAX_SPAN_ATR = 1.0


def _recency(age: int, half_life: float) -> float:
    age = max(0, age)
    return 0.5 + 0.5 * (2.0 ** (-age / half_life))


def harvest(s, fs, cfg: FeatureConfig, t: int) -> list[AOCMember]:
    """Every live level at bar ``t`` as (family, price, weight,
    recency).

    CAUSALITY CONTRACT: the reader snapshots ``pools_above/below``,
    ``active_fvgs`` and ``active_obs`` per bar from objects it then
    keeps mutating (``mitigated_at``/``swept_at`` are set in place by
    later bars).  List MEMBERSHIP is the causal truth — a structure
    in the list was live at this bar.  Re-checking ``.active`` here
    would read *future* mutations, so this function trusts the
    lists and never re-checks the flags."""
    out: list[AOCMember] = []
    hl = cfg.aoc_recency_half_life

    def add(family: str, price: float, weight: float, created_at: int,
            timeless: bool = False) -> None:
        if price != price:                    # NaN guard
            return
        rec = 1.0 if timeless else _recency(t - created_at, hl)
        out.append(AOCMember(family=family, price=float(price),
                             weight=weight, recency=rec))

    # liquidity pools (the list holds only pools live at this bar)
    for pool in list(s.pools_above) + list(s.pools_below):
        add("pool", pool.price,
            _POOL_W + _POOL_STRENGTH_W * max(1, pool.strength),
            pool.created_at)

    # supply / demand zones (the full live list, not just the nearest)
    for z in fs.zones.active:
        w = _ZONE_FRESH_W if z.freshness == "fresh" else _ZONE_TESTED_W
        add("sd_zone", 0.5 * (z.top + z.bottom), w, z.created_at)

    # order blocks and fair value gaps (the lists hold only the
    # structures live at this bar — see the contract above)
    for ob in s.active_obs:
        add("ob", 0.5 * (ob.top + ob.bottom), _OB_W, ob.created_at)
    for g in s.active_fvgs:
        add("fvg", g.midpoint, _FVG_W, g.created_at)

    # trendlines — the line's value AT this bar, live lines only
    for ln in fs.trendlines.active:
        if ln.state not in ("FORMING", "ACTIVE"):
            continue
        w = _TL_ACTIVE_W if ln.state == "ACTIVE" else _TL_FORMING_W
        add("trendline", ln.price_at(t), w, ln.created_at)

    # Fib levels of the current leg (timeless while the leg is current)
    for _frac, price in fs.fib.levels:
        add("fib", price, _FIB_W, 0, timeless=True)

    # daily swings (today's agreed pivots) and the dealing-range EQ
    if s.daily_swing_high is not None:
        add("daily", s.daily_swing_high, _DAILY_W, 0, timeless=True)
    if s.daily_swing_low is not None:
        add("daily", s.daily_swing_low, _DAILY_W, 0, timeless=True)
    if s.range_high == s.range_high and s.range_low == s.range_low:
        add("eq", 0.5 * (s.range_high + s.range_low), _EQ_W, 0,
            timeless=True)

    return out


def cluster(levels: list[AOCMember], tol: float,
            max_span: float) -> list[list[AOCMember]]:
    """Greedy price-ordered clustering: a level joins the open cluster
    when it sits within ``tol`` of the last member, unless that would
    push the cluster wider than ``max_span``."""
    clusters: list[list[AOCMember]] = []
    for m in sorted(levels, key=lambda m: m.price):
        if clusters and m.price - clusters[-1][-1].price <= tol \
                and m.price - clusters[-1][0].price <= max_span:
            clusters[-1].append(m)
        else:
            clusters.append([m])
    return clusters


def _zone_of(members: list[AOCMember]) -> AOCZone:
    sw = sum(m.weight * m.recency for m in members)
    cw = sum(m.weight * m.recency * m.price for m in members)
    center = cw / sw if sw > 0 else members[0].price
    top = max(m.price for m in members)
    bottom = min(m.price for m in members)
    fams = tuple(sorted({m.family for m in members}))
    return AOCZone(center=center, top=top, bottom=bottom,
                   width=top - bottom, families=fams, score=sw,
                   members=tuple(members))


def compute(df, states, cfg: FeatureConfig,
            feature_sets) -> list[AOCState]:
    """One :class:`AOCState` per bar.  ``feature_sets`` are the
    partially-built FeatureSets (zones/trendlines/fib already
    filled); the engine attaches the result afterwards."""
    n = len(states)
    out: list[AOCState] = []
    for t in range(n):
        s = states[t]
        atr_t = s.atr if s.atr and s.atr > 0 else 1e-9
        tol = cfg.aoc_tol_atr * atr_t
        max_span = _MAX_SPAN_ATR * atr_t
        zones: list[AOCZone] = []
        for members in cluster(harvest(s, feature_sets[t], cfg, t),
                               tol, max_span):
            if len({m.family for m in members}) >= cfg.aoc_min_families:
                zones.append(_zone_of(members))
        zones.sort(key=lambda z: z.score, reverse=True)
        zones = zones[:cfg.aoc_max_zones]
        close = s.close
        above = min((z for z in zones if z.center > close),
                    key=lambda z: z.center, default=None)
        below = max((z for z in zones if z.center < close),
                    key=lambda z: z.center, default=None)
        out.append(AOCState(
            zones=tuple(zones),
            above=above, below=below,
            above_dist_atr=((above.center - close) / atr_t
                            if above is not None else math.nan),
            below_dist_atr=((close - below.center) / atr_t
                            if below is not None else math.nan),
        ))
    return out
