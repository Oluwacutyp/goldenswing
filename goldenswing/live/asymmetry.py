"""Asymmetry Score — the rework's single decision number (spec §2).

Replaces the STATE 3-5 AND-chain's permission logic on the arrival
path: six evidence blocks, each scored on its own terms, summed to
0-100.  Confirmations (sweep / displacement / CHoCH / FVG) do NOT
live here — they are funding events (live/funding.py consumers in
the runner) that release size against a score that already
authorized layer 1.  The only hard lines are elsewhere (risk,
invalidation, manipulation, caps — spec §3.1/§4.4); this module
never gates, it measures.

Design rules (program directive, owner 2026-10-21):
- Components are a REGISTRY: name -> (evaluator, max points).
  P3 adds components / styles by registration, not surgery.
- Style floors are config data (STYLE_FLOORS), not constants
  buried in call sites.
- NO timeframe role is a premise: the caller passes the frames it
  chose (style presets pick defaults); every component reads only
  the frame it is handed.
- Every input comes from frames / ATR / zones / DXY the engine
  already loads.  Where the ideal input does not exist in the
  engine today, the component degrades to a stated neutral and
  flags it — nothing is invented:
  * per-pivot ATR *at formation* is not stored by find_swings;
    spacing uses the causal ATR of the frame up to each pivot
    (atr14 of the prefix), which is the same series the engine
    itself would have seen at that bar.
  * narrative STATE 6 acceptance evidence has no numeric feed;
    it is not a component (spec §2 scope note).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .dxy import dxy_bonus
from .multitf import atr14, find_swings
from .patterns import session_label
from .zones import SWING_N

# ------------------------------------------------------------- config

#: Block maxima (owner-reviewed weights, spec §2).  Sum = 100.
DEFAULT_WEIGHTS: dict[str, float] = {
    "structure": 30.0,
    "efficiency": 25.0,
    "pullback": 15.0,
    "location": 15.0,
    "session": 10.0,
    "macro": 5.0,
}

#: Per-style score floors (spec §3.3 v2).  Registry-shaped: P3
#: registers new styles here without touching the scorer.  The
#: ZONE_PATH row is the P1 arrival path: floor 50 + Location >= 12
#: as a trade-TYPE condition (without the zone, this is not that
#: trade), NO structure minimum.  Open-space rows are config for
#: build 2 — declared so the registry is complete, unused by P1.
STYLE_FLOORS: dict[str, dict] = {
    "ZONE_PATH": {"floor": 50.0, "min_location": 12.0,
                  "min_structure": 0.0},
    "MACRO_SWING_OPEN": {"floor": 60.0, "min_location": 0.0,
                         "min_structure": 15.0, "min_efficiency": 10.0},
    "INTRADAY_CONTINUATION": {"floor": 65.0, "min_location": 0.0,
                              "min_structure": 18.0,
                              "min_efficiency": 12.0, "min_pullback": 6.0},
    "MICRO_SCALP_OPEN": {"floor": 55.0, "min_location": 0.0,
                         "min_structure": 15.0, "min_efficiency": 10.0},
    "RANGE": {"floor": 45.0, "min_location": 8.0,
              "min_structure": 0.0},
}


def floor_for(path: str) -> dict:
    """Floor row for a decision path (copy — callers may not
    mutate the registry)."""
    return dict(STYLE_FLOORS.get(path, STYLE_FLOORS["ZONE_PATH"]))


# ------------------------------------------------------------- types

@dataclass
class ScoreContext:
    """Everything a component may read.  Built by the caller from
    state the engine already holds; components never fetch."""

    direction: int                      # +1 long / -1 short
    frame: pd.DataFrame | None = None   # trigger-TF closed bars
    atr: float = 0.0                    # ATR on the trigger frame
    price: float = 0.0
    style: str = ""
    zone: dict | None = None            # armed zone (zones.to_dict)
    opposing_zone: dict | None = None   # nearest opposing zone dict
    range_frame: pd.DataFrame | None = None  # dealing-range frame
    verdict: dict | None = None         # hierarchy verdict (macro)
    dxy_corr: float | None = None
    session_info: dict | None = None    # patterns.session_label()


@dataclass
class ComponentResult:
    name: str
    points: float
    max_points: float
    subs: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)


@dataclass
class AsymmetryScore:
    total: float
    direction: int
    flipped: bool
    components: dict = field(default_factory=dict)  # name -> result
    weights: dict = field(default_factory=dict)

    def component(self, name: str) -> float:
        r = self.components.get(name)
        return float(r.points) if r is not None else 0.0

    def breakdown(self) -> dict:
        """Journal/status shape (spec §6): every component's
        points, sub-scores and flags, plus the total."""
        out = {"total": round(float(self.total), 2),
               "direction": int(self.direction),
               "flipped": bool(self.flipped)}
        for name, r in self.components.items():
            out[name] = {
                "points": round(float(r.points), 2),
                "max": float(r.max_points),
                "subs": {k: (round(float(v), 3)
                             if isinstance(v, (int, float)) else v)
                         for k, v in r.subs.items()},
                "flags": list(r.flags),
            }
        return out


# ------------------------------------------------------------- helpers

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _atr_prefix(frame: pd.DataFrame, upto: int, fallback: float) -> float:
    """Causal ATR at bar ``upto`` (what the engine saw then)."""
    try:
        a = atr14(frame.iloc[:upto + 1])
        return a if a > 0 else fallback
    except Exception:
        return fallback


def _pivots(frame: pd.DataFrame, direction: int):
    """Defended-side pivots (HLs for long, LHs for short) from the
    engine's own swing detector, oldest -> newest."""
    highs, lows = find_swings(frame["high"].to_numpy(dtype=float),
                              frame["low"].to_numpy(dtype=float),
                              SWING_N)
    return lows if direction > 0 else highs


def _spacing_map(dist_atr: float) -> float:
    """<= ~1 ATR full marks; ~3 ATR zero (owner-session curve)."""
    return _clamp01(1.0 - (dist_atr - 1.0) / 2.0)


# ------------------------------------------------------------- components

def _structure(ctx: ScoreContext, max_points: float) -> ComponentResult:
    """Structure integrity: geometric mean of spacing / frequency /
    defense over close-confirmed defended-side pivots.  Geometric
    on purpose: no sub-score compensates for a dead one (a wide
    stop is not excused by frequent pivots)."""
    res = ComponentResult("structure", 0.0, max_points)
    frame = ctx.frame
    if frame is None or len(frame) < 2 * SWING_N + 5 or ctx.atr <= 0:
        res.flags.append("structure_no_data")
        res.subs = {"spacing": 0.0, "frequency": 0.0, "defense": 0.0}
        return res
    close = frame["close"].to_numpy(dtype=float)
    pivots = _pivots(frame, ctx.direction)
    if not pivots:
        res.flags.append("structure_no_pivots")
        res.subs = {"spacing": 0.0, "frequency": 0.0, "defense": 0.0}
        return res
    # --- flip rule: a close through the last defended pivot by
    # >= 0.1 ATR kills the block and flips the evaluated side
    # (score_asymmetry re-scores the other direction).
    last_pos, last_price = pivots[-1]
    breach = 0.1 * ctx.atr
    after = close[last_pos + 1:]
    if len(after):
        through = (after < last_price - breach) if ctx.direction > 0 \
            else (after > last_price + breach)
        if bool(through.any()):
            res.flags.append("structure_flip")
            res.subs = {"spacing": 0.0, "frequency": 0.0,
                        "defense": 0.0, "flip": 1.0}
            return res
    # --- spacing: median inter-pivot distance (formation ATR)
    # blended with the live distance to the last defended pivot
    # (the stop distance actually being underwritten).
    if len(pivots) >= 2:
        dists = []
        for (p0, v0), (p1, v1) in zip(pivots[:-1], pivots[1:]):
            a = _atr_prefix(frame, p1, ctx.atr)
            if a > 0:
                dists.append(abs(v1 - v0) / a)
        spacing_form = _spacing_map(float(np.median(dists))) \
            if dists else 0.3
    else:
        spacing_form = 0.3
    live_dist = abs(ctx.price - last_price) / ctx.atr
    spacing = 0.5 * spacing_form + 0.5 * _spacing_map(live_dist)
    # --- frequency: recency-decayed confirmed-pivot count over
    # the trailing ~80 bars (half-life ~20 bars).
    n = len(frame)
    freq_raw = 0.0
    for pos, _ in pivots:
        age = (n - 1) - pos
        if age <= 80:
            freq_raw += 0.5 ** (age / 20.0)
    frequency = _clamp01(freq_raw / 3.0)
    # --- defense: pivots tested (a later wick within 0.25 ATR)
    # and held (no close through).  Latest pivot double-weighted —
    # it is the one the stop hides behind.  Nothing tested yet:
    # the held ratio over all pivots is the evidence available.
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)

    def _held(pos: int, price: float) -> bool:
        seg = close[pos + 1:]
        if not len(seg):
            return True
        return bool((seg >= price - breach).all()) \
            if ctx.direction > 0 else bool((seg <= price + breach).all())

    def _tested(pos: int, price: float) -> bool:
        if ctx.direction > 0:
            seg = lows[pos + 2:]
            return bool(len(seg) and (seg <= price + 0.25 * ctx.atr).any())
        seg = highs[pos + 2:]
        return bool(len(seg) and (seg >= price - 0.25 * ctx.atr).any())

    tested = [(p, v) for p, v in pivots if _tested(p, v)]
    pool = tested if tested else pivots
    wsum = 0.0
    hsum = 0.0
    for i, (p, v) in enumerate(pool):
        w = 2.0 if (p, v) == pivots[-1] else 1.0
        wsum += w
        if _held(p, v):
            hsum += w
    defense = (hsum / wsum) if wsum > 0 else 0.0
    if not tested:
        res.flags.append("defense_untested")
    geo = (max(spacing, 0.0) * max(frequency, 0.0)
           * max(defense, 0.0)) ** (1.0 / 3.0)
    res.points = max_points * geo
    res.subs = {"spacing": round(spacing, 3),
                "spacing_formation": round(spacing_form, 3),
                "live_dist_atr": round(live_dist, 2),
                "frequency": round(frequency, 3),
                "defense": round(defense, 3),
                "pivots": float(len(pivots))}
    return res


def _leg_window(frame: pd.DataFrame, direction: int, bars: int = 40):
    """(start_pos, extreme_price) of the current directional leg:
    from the most extreme counter-price of the last ``bars`` bars
    to the last close."""
    sub = frame.tail(bars)
    if direction > 0:
        pos = int(np.argmin(sub["low"].to_numpy(dtype=float)))
        extreme = float(sub["low"].iloc[pos])
    else:
        pos = int(np.argmax(sub["high"].to_numpy(dtype=float)))
        extreme = float(sub["high"].iloc[pos])
    return len(frame) - len(sub) + pos, extreme


def _efficiency(ctx: ScoreContext, max_points: float) -> ComponentResult:
    """Leg efficiency: directional progress per unit of travel on
    the current leg + body dominance of the leg's bars."""
    res = ComponentResult("efficiency", 0.0, max_points)
    frame = ctx.frame
    if frame is None or len(frame) < 10 or ctx.atr <= 0:
        res.flags.append("efficiency_no_data")
        res.subs = {"leg_efficiency": 0.0, "body_dominance": 0.0}
        return res
    start, extreme = _leg_window(frame, ctx.direction)
    seg = frame.iloc[start:]
    highs = seg["high"].to_numpy(dtype=float)
    lows = seg["low"].to_numpy(dtype=float)
    closes = seg["close"].to_numpy(dtype=float)
    opens = seg["open"].to_numpy(dtype=float)
    path = float(np.sum(np.abs(np.diff(
        np.concatenate([[extreme], closes])))))
    net = abs(float(closes[-1]) - extreme) if len(closes) else 0.0
    leg_eff = _clamp01(net / path) if path > 0 else 0.0
    rng = np.maximum(highs - lows, 1e-9)
    body = np.abs(closes - opens) / rng
    with_dir = ((closes - opens) * ctx.direction) > 0
    dom = float(np.mean(body[with_dir])) if bool(with_dir.any()) else 0.0
    against = float(np.mean(body[~with_dir])) \
        if bool((~with_dir).any()) else 0.0
    body_dom = _clamp01(dom - 0.5 * against)
    score01 = 0.6 * leg_eff + 0.4 * body_dom
    res.points = max_points * score01
    res.subs = {"leg_efficiency": round(leg_eff, 3),
                "body_dominance": round(body_dom, 3),
                "leg_bars": float(len(seg))}
    return res


def _pullback(ctx: ScoreContext, max_points: float) -> ComponentResult:
    """Pullback discipline / arrival character (spec §2.3 + A5).

    At a zone (arrival context): the approach leg INTO the zone is
    graded — a sharp, impulsive arrival scores high (initiative
    exhausting into the level); a corrective stair-step grind
    scores low (every step built overhead structure).  Away from a
    zone: the current counter-move is graded as a pullback —
    shallow + corrective scores, deep or impulsive scores 0.
    """
    res = ComponentResult("pullback", 0.0, max_points)
    frame = ctx.frame
    if frame is None or len(frame) < 10 or ctx.atr <= 0:
        res.flags.append("pullback_no_data")
        res.subs = {"depth_frac": 0.0, "character": 0.0}
        return res
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    zone = ctx.zone
    if zone is not None:
        # Approach leg: from the last bar price sat >= 0.5 ATR
        # outside the band (on the approach side) to now.
        top, bottom = float(zone["top"]), float(zone["bottom"])
        start = max(0, len(frame) - 30)
        for i in range(len(frame) - 1, max(0, len(frame) - 31), -1):
            outside = (close[i] < bottom - 0.5 * ctx.atr) \
                if ctx.direction > 0 else \
                (close[i] > top + 0.5 * ctx.atr)
            if outside:
                start = i
                break
        seg_c = close[start:]
        seg_h, seg_l = high[start:], low[start:]
        if len(seg_c) >= 2:
            path = float(np.sum(np.abs(np.diff(seg_c))))
            net = abs(float(seg_c[-1]) - float(seg_c[0]))
            eff = _clamp01(net / path) if path > 0 else 0.0
            rng = np.maximum(seg_h - seg_l, 1e-9)
            opens = frame["open"].to_numpy(dtype=float)[start:]
            body = np.abs(seg_c - opens) / rng
            overlaps = 0
            for i in range(1, len(seg_c)):
                lo = max(seg_l[i], seg_l[i - 1])
                hi = min(seg_h[i], seg_h[i - 1])
                if hi >= lo:
                    overlaps += 1
            overlap_ratio = overlaps / max(1, len(seg_c) - 1)
            char = _clamp01(0.5 * eff + 0.5 * float(np.mean(body)))
            if overlap_ratio > 0.7:
                char *= 0.6
                res.flags.append("arrival_stairstep")
            elif eff >= 0.7:
                res.flags.append("arrival_impulsive")
            res.points = max_points * char
            res.subs = {"arrival_efficiency": round(eff, 3),
                        "arrival_body": round(float(np.mean(body)), 3),
                        "overlap_ratio": round(overlap_ratio, 3),
                        "approach_bars": float(len(seg_c))}
            return res
    # Pullback context: depth of the current counter-move against
    # the last impulse leg + its character.
    start, extreme = _leg_window(frame, -ctx.direction, bars=40)
    # The impulse leg runs the other way: from the extreme BEFORE
    # the counter-move's origin.  Depth = counter-move size / the
    # impulse that preceded it, both read off the same window.
    seg = frame.iloc[max(0, start - 40):]
    if len(seg) < 5:
        res.subs = {"depth_frac": 0.0, "character": 0.0}
        return res
    closes = seg["close"].to_numpy(dtype=float)
    cur = float(close[-1])
    if ctx.direction > 0:
        peak = float(np.max(seg["high"].to_numpy(dtype=float)))
        trough_before = float(np.min(
            seg["low"].to_numpy(dtype=float)[:int(
                np.argmax(seg["high"].to_numpy(dtype=float))) + 1]))
        impulse = peak - trough_before
        depth = (peak - cur) / impulse if impulse > 0 else 0.0
    else:
        trough = float(np.min(seg["low"].to_numpy(dtype=float)))
        peak_before = float(np.max(
            seg["high"].to_numpy(dtype=float)[:int(
                np.argmin(seg["low"].to_numpy(dtype=float))) + 1]))
        impulse = peak_before - trough
        depth = (cur - trough) / impulse if impulse > 0 else 0.0
    depth = _clamp01(depth)
    opens = seg["open"].to_numpy(dtype=float)
    rng = np.maximum(seg["high"].to_numpy(dtype=float)
                     - seg["low"].to_numpy(dtype=float), 1e-9)
    body = np.abs(closes - opens) / rng
    counter_bodies = body[-max(3, len(seg) // 3):]
    impulsive_counter = float(np.mean(counter_bodies)) \
        if len(counter_bodies) else 0.0
    if depth > 0.62:
        pts01 = 0.0
        res.flags.append("pullback_deep")
    elif impulsive_counter >= 0.65 and depth > 0.25:
        pts01 = 0.0
        res.flags.append("pullback_impulsive")
    else:
        pts01 = _clamp01(1.0 - max(0.0, depth - 0.382) / (0.618 - 0.382))
        pts01 *= _clamp01(1.2 - impulsive_counter)
    res.points = max_points * pts01
    res.subs = {"depth_frac": round(depth, 3),
                "character": round(impulsive_counter, 3)}
    return res


def _location(ctx: ScoreContext, max_points: float) -> ComponentResult:
    """Location & draw (spec §2.4): curve position (30%) + zone
    context (40%, graded by the CORRECTED zone grade + lifecycle)
    + draw to the next opposing magnet (30%)."""
    res = ComponentResult("location", 0.0, max_points)
    subs = {"curve": 0.0, "zone": 0.0, "draw": 0.0}
    frame = ctx.range_frame if ctx.range_frame is not None \
        else ctx.frame
    # --- curve position within the dealing range
    if frame is not None and len(frame) >= 20:
        win = frame.tail(120)
        hi = float(win["high"].max())
        lo = float(win["low"].min())
        if hi > lo:
            pos = (ctx.price - lo) / (hi - lo)
            if ctx.direction > 0:
                curve01 = _clamp01(1.5 - 1.5 * max(0.0, pos - 0.25) / 0.75) \
                    if pos > 0.25 else 1.0
                if pos >= 0.75:
                    curve01 = 0.0
            else:
                curve01 = _clamp01(1.5 - 1.5 * max(0.0, 0.75 - pos) / 0.75) \
                    if pos < 0.75 else 1.0
                if pos <= 0.25:
                    curve01 = 0.0
            subs["curve"] = curve01 * 0.30
            subs["curve_pos"] = round(pos, 3)
    # --- zone context: the corrected grade (score after A1) +
    # lifecycle (A2).  A used-up zone contributes nothing and is
    # flagged — the decision flow abandons it before floors.
    zone = ctx.zone
    if zone is not None and ctx.atr > 0:
        top, bottom = float(zone["top"]), float(zone["bottom"])
        lifecycle = str(zone.get("lifecycle") or "")
        if not lifecycle:
            touches = int(zone.get("touches", 0) or 0)
            status = str(zone.get("status", "active"))
            lifecycle = "used_up" if status != "active" or touches >= 3 \
                else ("fresh" if touches == 0 else "tested")
        tier = str(zone.get("tier", "MINOR"))
        if lifecycle == "used_up":
            res.flags.append("zone_used_up")
        else:
            inside = bottom <= ctx.price <= top
            dist = 0.0 if inside else min(
                abs(ctx.price - top), abs(ctx.price - bottom)) / ctx.atr
            # An armed_band zone is the setup's own entry level
            # (marked by the hierarchy, no ZoneStore grade): the
            # at-the-level share is credited in full — inventing a
            # MINOR/tested discount for it would grade a level the
            # system itself armed below a level it merely stored.
            armed_band = bool(zone.get("armed_band"))
            base = 6.0 if tier == "MAJOR" or armed_band else 4.0
            if inside:
                z = base
            elif dist <= 0.5:
                z = base * 0.5
            else:
                z = 0.0
            if lifecycle == "tested" and not armed_band:
                z *= 0.75
            subs["zone"] = (z / 6.0) * 0.40
            subs["zone_lifecycle"] = lifecycle
    # --- draw: room to the next opposing magnet
    opp = ctx.opposing_zone
    if opp is not None and ctx.atr > 0:
        d = abs(float(opp.get("level", ctx.price)) - ctx.price) / ctx.atr
        subs["draw"] = _clamp01((d - 0.5) / 1.5) * 0.30
        subs["draw_atr"] = round(d, 2)
    else:
        subs["draw"] = 0.8 * 0.30  # no known magnet: open road,
        # small uncertainty discount (input absent, not invented)
        res.flags.append("draw_unmeasured")
    res.points = max_points * _clamp01(
        subs["curve"] + subs["zone"] + subs["draw"])
    res.subs = {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in subs.items()}
    return res


def _session(ctx: ScoreContext, max_points: float) -> ComponentResult:
    """Session & vol character (spec §2.5): killzone/session quality
    (7) + volume character (3).  A scalar on aggression, never a
    gate — thin hours score low, they do not forbid."""
    res = ComponentResult("session", 0.0, max_points)
    info = ctx.session_info
    if info is None and ctx.frame is not None and len(ctx.frame):
        try:
            info = session_label(ctx.frame.index[-1])
        except Exception:
            info = None
    sess_pts = 0.5  # neutral when unknown
    if info:
        if info.get("in_killzone"):
            sess_pts = 1.0
        elif str(info.get("session", "")) in ("london", "new_york",
                                              "london_ny_overlap"):
            sess_pts = 0.75
        elif str(info.get("session", "")) == "asia":
            sess_pts = 0.35
    vol_pts = 0.5
    frame = ctx.frame
    if frame is not None and "volume" in frame.columns \
            and len(frame) >= 21:
        try:
            v = frame["volume"].to_numpy(dtype=float)[-21:]
            mean = float(np.mean(v[:-1]))
            ratio = float(v[-1]) / mean if mean > 0 else 0.0
            vol_pts = 1.0 if ratio >= 1.0 else (
                0.6 if ratio >= 0.7 else 0.3)
            res.subs["volume_ratio"] = round(ratio, 2)
        except Exception:
            pass
    else:
        res.flags.append("volume_unmeasured")
    res.points = max_points * (0.7 * sess_pts + 0.3 * vol_pts)
    res.subs.update({"session_pts": round(sess_pts, 2),
                     "vol_pts": round(vol_pts, 2)})
    return res


def _macro(ctx: ScoreContext, max_points: float) -> ComponentResult:
    """Macro alignment (spec §2.6): hierarchy verdict legs as
    CONTEXT (gate 0 demoted — the verdict informs, never permits)
    + DXY colour.  Opposing HTF biases bleed the block toward 0;
    they never veto."""
    res = ComponentResult("macro", 0.0, max_points)
    aligned = 0.0
    legs = 0.0
    verdict = ctx.verdict or {}
    for key in ("bias_d1", "bias_h4"):
        b = verdict.get(key)
        if b is None:
            continue
        try:
            b = int(b)
        except Exception:
            continue
        legs += 1.0
        if b == ctx.direction:
            aligned += 1.0
        elif b == 0:
            aligned += 0.5
    base01 = (aligned / legs) if legs else 0.5
    if not legs:
        res.flags.append("macro_no_verdict")
    bonus, _ = dxy_bonus(ctx.dxy_corr)
    pts = max_points * base01 + bonus
    res.points = max(0.0, min(max_points, pts))
    res.subs = {"aligned": round(base01, 2), "dxy_bonus": bonus}
    return res


# ------------------------------------------------------------- registry

#: Component registry (program directive: plug-in, not surgery).
#: name -> evaluator(ctx, max_points) -> ComponentResult.
COMPONENTS: dict[str, Callable] = {
    "structure": _structure,
    "efficiency": _efficiency,
    "pullback": _pullback,
    "location": _location,
    "session": _session,
    "macro": _macro,
}


def register_component(name: str, evaluator: Callable,
                       weight: float) -> None:
    """Register (or replace) a score component + its weight.
    P3's extension point."""
    COMPONENTS[name] = evaluator
    DEFAULT_WEIGHTS[name] = float(weight)


def score_asymmetry(ctx: ScoreContext, *,
                    weights: dict | None = None,
                    allow_flip: bool = True) -> AsymmetryScore:
    """Score one candidate direction.  The flip rule lives here:
    if the structure component reports a close through the last
    defended pivot, the WHOLE score is recomputed for the opposite
    direction and returned with ``flipped=True`` (a flipped score
    can never fund the original direction).

    ``allow_flip=False`` is the AT-ZONE evaluation (the caller —
    the arrival decision flow — decides when it applies): the
    approach into a zone characteristically closes through minor
    trigger-frame pivots ON THE WAY IN, which is arrival, not a
    thesis flip; the zone's far edge is the invalidation that
    governs there (§5 probe-death rule).  The score then prices
    the trade's OWN direction with the structure block zeroed
    against it (flag ``structure_flip_against``) — the flip
    evidence still costs the trade its entire structure share."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    struct = COMPONENTS["structure"](ctx, w.get("structure", 30.0))
    if "structure_flip" in struct.flags and not allow_flip:
        struct.points = 0.0
        struct.flags.append("structure_flip_against")
        components = {"structure": struct}
        total = 0.0
        for name, fn in COMPONENTS.items():
            if name == "structure":
                continue
            r = fn(ctx, w.get(name, 0.0))
            components[name] = r
            total += r.points
        return AsymmetryScore(total=float(total),
                              direction=ctx.direction,
                              flipped=False, components=components,
                              weights=w)
    if "structure_flip" in struct.flags and not getattr(
            ctx, "_flip_guard", False):
        flipped_ctx = ScoreContext(
            direction=-ctx.direction, frame=ctx.frame, atr=ctx.atr,
            price=ctx.price, style=ctx.style, zone=ctx.zone,
            opposing_zone=ctx.opposing_zone,
            range_frame=ctx.range_frame, verdict=ctx.verdict,
            dxy_corr=ctx.dxy_corr, session_info=ctx.session_info)
        flipped_ctx._flip_guard = True  # type: ignore[attr-defined]
        out = score_asymmetry(flipped_ctx, weights=weights)
        out.flipped = True
        return out
    components = {"structure": struct}
    total = struct.points
    for name, fn in COMPONENTS.items():
        if name == "structure":
            continue
        r = fn(ctx, w.get(name, 0.0))
        components[name] = r
        total += r.points
    return AsymmetryScore(total=float(total), direction=ctx.direction,
                          flipped=False, components=components,
                          weights=w)
