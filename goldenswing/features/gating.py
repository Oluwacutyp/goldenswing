"""Context gates (Phase 4) — *how, where and when* a family may speak.

    "The problem might not be the fib and the rest but how they are
    used, where they are used and when they are used — that's what
    real traders know and do."  — the owner

Phase 3 scored every family flat: whenever the family's pattern was
present, it added (or subtracted) its points, in every context.  Six
of ten families diluted.  The Phase-3 diagnosis already hinted why:
patterns "decorate entries", MTF "double-counts" trend, the Fib
"re-states" premium/discount discipline.  A real trader does not use
a tool everywhere — each tool has a *job*, a place, and a moment:

- **Fib/OTE** is a *pullback-entry* tool.  It means something only
  while price is retracing into the OTE zone of a confirmed
  displacement leg, in the leg's direction, with the higher timeframe
  agreeing.  Anywhere else it is decoration.
- **Supply/demand zones** are *fresh-level* tools.  A tested zone is
  spent; a fresh zone traded against the daily bias is a coin flip;
  and price must ARRIVE properly — a slowing approach, not a knife
  through the level.
- **Ichimoku** is a *regime* tool first: the DAILY cloud sets the
  bias context.  The H1 cloud is an entry aid only while the market
  is expanding — inside a base (consolidation) the cloud is noise
  chopped through ten times a day.
- **Classical patterns** are *location + trigger* tools.  A head &
  shoulders in the middle of nowhere is trivia; the same shape AT a
  major level (daily swing / liquidity pool) after an extended move,
  CONFIRMED by a neckline break with displacement, is a trade.
- **Footprints**: the institutional signature is a *post-sweep at a
  level* story — size entering where stops just flushed.  The retail
  chase tag is a caution ONLY at climax extremes (that is where
  chasing actually kills); penalising every strong continuation, as
  Phase 3 did, fights the book's best trades.
- **MTF** is a *veto/confirm* tool, not a points fountain: it may
  CONFIRM modestly when both HTFs align, and it may CAP a score when
  both HTFs stand against the trade.  It never re-buys the trend
  points the core score already paid for.

Contract: each gate returns a list of ``(delta, factor)`` contributions
— EMPTY when its context is absent.  Outside its context a gated
family is silent: no bonus, no penalty.  All inputs are the causal
per-bar states (MarketState + FeatureSet); gates are pure functions
of the current bar's state, so causality is inherited from the layer
below.
"""
from __future__ import annotations


def _near(price: float | None, target: float, tol: float) -> bool:
    return price is not None and abs(price - target) <= tol


def _zone_dist(s, bottom: float, top: float) -> float:
    """Distance from the close to a [bottom, top] zone (0 inside)."""
    if s.close > top:
        return s.close - top
    if s.close < bottom:
        return bottom - s.close
    return 0.0


def _at_major_level(s, fs, price: float, tol_atr: float = 0.6) -> bool:
    """Is ``price`` at a level that matters — a mapped liquidity pool,
    a confirmed daily swing, or an HTF (H4/D1) swing?"""
    tol = tol_atr * s.atr
    for pool in list(s.pools_above) + list(s.pools_below):
        if abs(pool.price - price) <= tol:
            return True
    if _near(s.daily_swing_high, price, tol) \
            or _near(s.daily_swing_low, price, tol):
        return True
    m = fs.mtf if fs is not None else None
    if m is not None:
        for lvl in (m.h4_swing_high, m.h4_swing_low,
                    m.d1_swing_high, m.d1_swing_low):
            if _near(lvl, price, tol):
                return True
    return False


def _aoc_major_level(fs, price: float, tol: float) -> bool:
    """Phase 4.5 repair: a level is MAJOR only inside a pivot zone —
    an AOC cluster of >= 3 independent families.  The pool-proximity
    test above was too permissive on H1: pools are dense enough that
    almost any pattern qualified, which is why the gated patterns
    family still diluted (+19.04%).  Concentration, not proximity,
    is the major-level test."""
    ao = fs.aoc if fs is not None else None
    if ao is None:
        return False
    return ao.major_at(price, tol) is not None


# --------------------------------------------------------------------- Fib
def fib_gate(s, fs, d: int, kind: str) -> list:
    """OTE speaks only inside a trend-aligned pullback into the zone
    of the confirmed leg, with HTF agreement.

    (a) the last completed leg ran in direction ``d`` and price sits
        in that leg's OTE band (``in_ote_long`` / ``in_ote_short``
        already encode exactly this);
    (b) the leg was a *displacement* leg — structure broke in its
        direction (the reader's last structure event agrees), so the
        retracement is against real movement, not drift;
    (c) the pullback is actually in progress: momentum is no longer
        extending in ``d`` (it has stalled or turned against), while
        H1 structure has NOT flipped against the trade;
    (d) HTF agrees: daily bias or the MTF D1 bias points with ``d``.
    """
    f = fs.fib
    in_ote = (d == 1 and f.in_ote_long) or (d == -1 and f.in_ote_short)
    if not in_ote:
        return []
    ev = s.last_structure_event
    if ev is None or ev.direction != d:
        return []                       # leg never broke structure
    if s.bias == -d:
        return []                       # structure flipped against us
    if s.momentum * d > 0.35:
        return []                       # still extending, not pulling back
    htf_agrees = s.daily_bias == d or fs.mtf.d1_bias == d
    if not htf_agrees:
        return []
    out = [(6.0, "OTE pullback entry: retracement into the 0.62-0.79 "
                 "zone of a confirmed displacement leg, HTF aligned")]
    if f.at_fib:
        out.append((2.0, f"pullback sitting on Fib {f.nearest_frac:.3f}"))
    # an unbroken Fib shelf just ahead still caps the first leg of the
    # move — inside the context, that warning is part of the tool's job
    for frac, price in f.levels:
        if 0 < (price - s.close) * d <= 0.30 * s.atr:
            out.append((-3.0, f"Fib {frac:.3f} shelf just ahead of entry"))
            break
    return out


# ------------------------------------------------------------------- zones
def zones_gate(s, fs, d: int, kind: str) -> list:
    """Zones speak only when FRESH, daily-bias aligned, and arrived
    at properly (slowing approach, no knife-through)."""
    z = fs.zones
    if s.daily_bias != d:
        return []                       # (b) daily bias must agree
    slowing = abs(s.momentum) <= 1.0 and abs(s.body_ratio) <= 0.8
    knife = (s.bear_disp_3 if d == 1 else s.bull_disp_3) >= 1.2
    if d == 1:
        zn = z.nearest_demand
        at = z.in_demand or (zn is not None
                             and z.demand_dist_atr == z.demand_dist_atr
                             and z.demand_dist_atr <= 0.35)
        if at and zn is not None and zn.freshness == "fresh" \
                and slowing and not knife:
            return [(6.0, "fresh demand zone, daily-bias aligned, "
                          "slowing approach — first test")]
        opp = z.nearest_supply
        if z.in_supply and opp is not None \
                and opp.freshness == "fresh":
            return [(-4.0, "buying into a FRESH supply zone against "
                           "the daily bias's own level")]
    else:
        zn = z.nearest_supply
        at = z.in_supply or (zn is not None
                             and z.supply_dist_atr == z.supply_dist_atr
                             and z.supply_dist_atr <= 0.35)
        if at and zn is not None and zn.freshness == "fresh" \
                and slowing and not knife:
            return [(6.0, "fresh supply zone, daily-bias aligned, "
                          "slowing approach — first test")]
        opp = z.nearest_demand
        if z.in_demand and opp is not None \
                and opp.freshness == "fresh":
            return [(-4.0, "shorting into a FRESH demand zone against "
                           "the daily bias's own level")]
    return []


# --------------------------------------------------------------- ichimoku
def ichimoku_gate(s, fs, d: int, kind: str) -> list:
    """Ichimoku speaks as CONTEXT: the D1 cloud is the bias vote; the
    H1 cloud may confirm entries only in an expansion phase — never
    inside a base/consolidation."""
    i = fs.ichimoku
    out = []
    # (a) regime vote from the DAILY cloud
    if i.d1_position == "above" and d == 1:
        out.append((3.0, "above the DAILY Ichimoku cloud — HTF regime "
                         "supports longs"))
    elif i.d1_position == "below" and d == -1:
        out.append((3.0, "below the DAILY Ichimoku cloud — HTF regime "
                         "supports shorts"))
    elif i.d1_position in ("above", "below"):
        out.append((-4.0, "trade fights the DAILY cloud regime"))
    else:
        return []                       # inside/unknown D1 cloud: the
                                        # family has no context — silent
    # (b) H1 entry confirmation, expansion phases only
    phase = fs.phase.phase
    if phase in ("expansion_up", "expansion_down"):
        aligned_phase = (phase == "expansion_up") == (d == 1)
        if aligned_phase:
            if i.position == ("above" if d == 1 else "below"):
                out.append((3.0, "H1 cloud confirms the expansion "
                                 "direction"))
            if i.tk_cross == d:
                out.append((2.0, "fresh TK cross inside the expansion"))
    return out


# --------------------------------------------------------------- patterns
def patterns_gate(s, fs, d: int, kind: str) -> list:
    """Patterns speak only at major levels, after an extended move,
    and only once the neckline has BROKEN with displacement.  A
    formed-but-unbroken pattern never scores.

    Phase 4.5 repair: "major level" is an AOC pivot zone (>= 3
    independent families agreeing) — the old pool-proximity test
    let nearly every pattern through on H1."""
    p = fs.patterns
    disp = s.bull_disp_3 if d == 1 else s.bear_disp_3
    ev = s.last_structure_event
    triggered = disp >= 1.0 or (
        ev is not None and ev.direction == d
        and s.structure_event_age is not None
        and s.structure_event_age <= 5)
    if not triggered:
        return []
    # extended move: the dealing range spans >= 5 ATR (a real move
    # preceded the pattern — the >=3-legs requirement, proxied by
    # range span since pattern instances carry no leg count)
    span = s.range_high - s.range_low
    if s.atr <= 0 or span != span or span < 5.0 * s.atr:
        return []
    tol = 0.35 * s.atr
    out = []
    for x in p.active:
        if x.status != "broken":
            continue                    # pattern alone never scores
        height = abs(x.extreme - x.neckline)
        if height < 1.5 * s.atr:
            continue
        if not (_aoc_major_level(fs, x.extreme, tol)
                or _aoc_major_level(fs, x.neckline, tol)):
            continue                    # in the middle of nowhere
        if x.direction == d:
            out.append((6.0, f"{x.kind} at a MAJOR pivot zone (AOC), "
                             f"neckline broken WITH displacement"))
        else:
            out.append((-6.0, f"opposing {x.kind} broke its neckline "
                              f"with displacement at a MAJOR pivot zone"))
    return out


# ------------------------------------------------------------- footprints
def footprints_gate(s, fs, d: int, kind: str) -> list:
    """Institutional signature counts only POST-SWEEP AT A LEVEL;
    the retail-chase caution applies only at climax extremes.  CTA
    is not scored here at all: its information is the trend itself,
    which the core score already owns (Phase-3 dilution diagnosis)."""
    f = fs.footprints
    out = []
    if f.institutional_dir == d:
        lvl = False
        ob = s.nearest_ob(d)
        if ob is not None and _zone_dist(s, ob.bottom, ob.top) <= 0.5 * s.atr:
            lvl = True
        g = s.nearest_fvg(d)
        if g is not None and _zone_dist(s, g.bottom, g.top) <= 0.5 * s.atr:
            lvl = True
        sw = s.last_sweep
        if sw is not None and _at_major_level(s, fs, sw.level, 0.75):
            lvl = True
        if lvl:
            out.append((7.0, "institutional signature post-sweep AT a "
                             "level — size entering through the flush"))
    elif f.institutional_dir == -d:
        out.append((-4.0, "institutional signature AGAINST the move"))
    if f.retail_chase == d and fs.volume.climax:
        extreme = (d == 1 and s.eq_position >= 0.88) or \
            (d == -1 and s.eq_position <= 0.12)
        if extreme:
            out.append((-5.0, "retail chase at a climax extreme — the "
                               "crowd is late; do not join the chase"))
    if f.retail_trap == d:
        out.append((3.0, "retail trap sprung — chase crowd underwater, "
                         "fade fuel"))
    return out


# --------------------------------------------------------------------- MTF
def mtf_gate(s, fs, d: int, kind: str) -> list:
    """MTF confirms modestly — it never re-buys trend points."""
    m = fs.mtf
    if m.aligned_count(d) == 2:
        return [(3.0, "D1 + H4 both aligned (MTF confirmation)")]
    return []


def mtf_cap(s, fs, d: int) -> float | None:
    """The veto half of MTF: both HTFs strongly opposed caps the
    score below the golden threshold — the trade may still be seen,
    it may not be called a Golden Swing."""
    m = fs.mtf
    if m.opposed_count(d) == 2:
        return 59.0
    return None


# -------------------------------------------------------------- trendlines
def trendlines_score(s, fs, d: int, kind: str) -> list:
    """Trendline events, context-gated by construction:

    - a BOUNCE at a VALIDATED (ACTIVE, >= 1 retest) line scores only
      inside trend context (H1 bias with the trade — the line is
      ridden, not fought);
    - a line BREAK is a structure event in its own right: WITH
      displacement (>= 1.0 ATR body in the break direction, the
      reader's 3-bar displacement) it scores for the break direction
      and against the other; a break without displacement is a wick
      story and stays silent;
    - merely sitting near a validated line in trend direction earns a
      small proximity credit.
    """
    tl = fs.trendlines
    disp = s.bull_disp_3 if d == 1 else s.bear_disp_3
    out = []
    for ev in tl.events:
        if ev.kind == "bounce":
            line_is_support = ev.direction == 1
            bounce_dir = 1 if line_is_support else -1
            if bounce_dir == d and s.bias == d:
                out.append((5.0, "bounce off a validated trendline "
                                 "inside trend context"))
            elif bounce_dir == -d:
                out.append((-3.0, "price bouncing off the opposing "
                                 "trendline against the trade"))
        elif ev.kind == "break":
            break_dir = -ev.direction   # breaking support is bearish
            if break_dir == d and disp >= 1.0:
                out.append((5.0, "trendline break WITH displacement — "
                                 "structure event"))
            elif break_dir == -d:
                opp_disp = s.bull_disp_3 if d == -1 else s.bear_disp_3
                if opp_disp >= 1.0:
                    out.append((-5.0, "opposing trendline broke WITH "
                                       "displacement against the trade"))
    if not out and s.bias == d:
        for ln in tl.active:
            if ln.state != "ACTIVE" or ln.direction != d:
                continue
            dist = abs(s.close - ln.price_at(s.index))
            if dist <= 0.35 * s.atr:
                out.append((2.0, "price riding a validated trendline "
                                 "in trend direction"))
                break
    return out


# --------------------------------------------------------------------- AOC
def _aoc_entry_support(ao, d: int, close: float,
                       tol: float) -> "AOCZone | None":
    """The AOC cluster (>= 2 families) the entry sits inside, aligned
    with the trade direction: at/below the entry for longs (support),
    at/above for shorts (resistance)."""
    best = None
    for z in ao.zones:
        inside = (z.bottom - tol <= close <= z.top + tol) \
            or abs(z.center - close) <= tol
        if not inside:
            continue
        aligned = (d == 1 and z.center <= close + tol) or \
                  (d == -1 and z.center >= close - tol)
        if aligned and (best is None or z.score > best.score):
            best = z
    return best


def aoc_gate(s, fs, d: int, kind: str) -> list:
    """The area of concentration speaks in two trader situations:

    - ENTRY BACKING: the entry sits inside a concentration (>= 2
      independent families) positioned as support/resistance for the
      trade direction — the pivot zone backs the level (+4).
    - OVERHEAD HEADWIND: a concentration sits strictly between the
      entry and the first liquidity target (the nearest pool ahead,
      which is what the signal's own targets are built from) — the
      trade must chew through agreed supply/demand first (-4).

    Outside both situations the family is silent: no bonus, no
    penalty.
    """
    ao = fs.aoc if fs is not None else None
    if ao is None or not ao.zones:
        return []
    tol = 0.35 * s.atr if s.atr and s.atr > 0 else 0.0
    close = s.close
    out = []
    entry = _aoc_entry_support(ao, d, close, tol)
    if entry is not None:
        side = "support" if d == 1 else "resistance"
        out.append((4.0, "entry inside an area of concentration "
                         f"({len(entry.families)} families: "
                         f"{', '.join(entry.families)}) acting as "
                         f"{side} — the pivot zone backs the level"))
    # overhead headwind: a concentration strictly between entry and
    # the first pool target (targets are pools; pools_* are nearest
    # first, so [0] is the trade's first objective)
    pools = s.pools_above if d == 1 else s.pools_below
    target = pools[0].price if pools else None
    if target is not None:
        lo, hi = (close, target) if d == 1 else (target, close)
        for z in ao.zones:
            if z is entry:
                continue
            if lo + tol < z.center < hi - tol:
                out.append((-4.0, "a concentration sits between entry "
                                  f"and the first target ({z.center:.1f}, "
                                  f"{len(z.families)} families) — "
                                  "overhead supply/demand to chew through"))
                break
    return out


def divergence_gate(s, fs, d: int, kind: str) -> list:
    """RSI divergence as entry backing (mined: TradingLab video,
    2026-10-11).

    - A divergence ALIGNED with the trade direction backs the entry
      (+4): regular divergence is reversal fuel, hidden divergence is
      continuation fuel — the video's own taxonomy, named in the factor.
    - A divergence OPPOSING the trade direction warns against it (-4):
      momentum disagrees with the trade.
    Outside both situations the family is silent.
    """
    dv = fs.divergence if fs is not None else None
    if dv is None or not dv.divergences:
        return []
    out = []
    al = dv.aligned(d)
    if al is not None:
        role = "reversal" if al.is_regular else "continuation"
        out.append((4.0,
                    f"RSI({dv.rsi_period}) {al.kind} divergence "
                    f"({role} fuel) backs the move — price made the "
                    f"extreme, momentum refused to confirm"))
    op = dv.opposing(d)
    if op is not None:
        out.append((-4.0,
                    f"RSI({dv.rsi_period}) {op.kind} divergence opposes "
                    f"the move — momentum disagrees with the trade"))
    return out
