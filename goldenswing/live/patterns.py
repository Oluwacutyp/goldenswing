"""Institutional price-action + order-flow pattern library.

Every detector here is a CHAINED component, never a standalone trade
signal.  A pattern alone fires nothing: patterns only matter inside
the narrative state machine (``live/narrative.py``):

    HTF Zone -> Liquidity Sweep -> Displacement -> CHoCH/BOS + FVG
    -> Entry (FVG / OB / breaker / mitigation block)

All detectors are pure functions of CLOSED bars only — no lookahead,
no persistent state, nothing to desync.  Each returns a dict with
at least ``present`` (bool), ``score`` (0-100 contribution),
``invalidation`` (price, or None) and ``detail`` (human-readable
evidence).  ``present=False`` always carries ``reason`` so a caller
can tell "pattern absent" from "not enough bars".

The library covers the full institutional toolkit, as chained
components:

- Market structure: HH/HL, LH/LL, CHoCH, BOS, MSB
- Order blocks: bullish/bearish OB, breaker, mitigation block
- FVG: bullish/bearish, inversion, BPR, CE
- Liquidity: equal highs/lows pools, sweep, run, stop hunt,
  buy-side / sell-side, session highs/lows
- QM (Quasimodo), premium/discount + OTE, AMD phases,
  inducement (IDM), turtle soup, SMT divergence,
  sessions/killzones, volume confirm, news pause.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .multitf import atr14, detect_bos_msch_choch, find_swings

SWING_N = 3
DISPLACEMENT_MIN_ATR = 1.5
SWEEP_MIN_ATR = 0.25
FVG_MIN_ATR = 0.05


# ---------------------------------------------------------------- helpers

def _arr(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy(dtype=float)


def _atr(df: pd.DataFrame) -> float:
    try:
        v = float(atr14(df))
    except Exception:
        return 0.0
    return v if v > 0 else 0.0


def _absent(reason: str) -> dict:
    return {"present": False, "score": 0.0, "invalidation": None,
            "detail": reason, "reason": reason}


def _swings(df: pd.DataFrame, n: int = SWING_N):
    return find_swings(_arr(df, "high"), _arr(df, "low"), n)


# ---------------------------------------------------------------- structure

def structure_event(df: pd.DataFrame, *, trend: int = 0) -> dict:
    """Last-bar structure event: CHoCH / BOS / MSCH / MSB / none.

    Wraps ``detect_bos_msch_choch`` (which already enforces the
    displacement gate) and adds MSB: a CHoCH that also clears the
    OLDER minor level — a confirmed reversal, not just a first
    break against the trend.
    """
    if df is None or len(df) < 2 * SWING_N + 5:
        return _absent("not enough bars for structure")
    highs, lows = _swings(df)
    if len(highs) < 2 or len(lows) < 2:
        return _absent("fewer than 2 confirmed swings per side")
    ref_high, ref_low = highs[-1][1], lows[-1][1]
    min_high = highs[-2][1] if len(highs) >= 2 else None
    min_low = lows[-2][1] if len(lows) >= 2 else None
    atr = _atr(df)
    ev = detect_bos_msch_choch(df, ref_high, ref_low, atr, trend=trend,
                               minor_high=min_high, minor_low=min_low)
    if ev is None:
        return _absent("no BOS/CHoCH/MSCH on last bar")
    close = float(df["close"].iloc[-1])
    is_msb = False
    if ev.startswith("CHoCH"):
        # MSB: CHoCH that also cleared the older minor level.
        if ev.endswith("bull") and min_high is not None \
                and close > min_high:
            is_msb = True
        if ev.endswith("bear") and min_low is not None \
                and close < min_low:
            is_msb = True
    direction = 1 if ev.endswith("bull") else -1
    kind = "MSB" if is_msb else ev.split("_")[0]
    inv = ref_low if direction > 0 else ref_high
    return {"present": True, "score": 75.0 if kind in ("CHoCH", "MSB")
            else 60.0, "invalidation": float(inv),
            "detail": f"{kind} {ev} close {close:.1f}",
            "kind": kind, "event": ev, "direction": direction,
            "ref_high": float(ref_high), "ref_low": float(ref_low)}


def trend_bias(df: pd.DataFrame, *, swing_n: int = SWING_N) -> dict:
    """HH/HL vs LH/LL on the last two confirmed swing pairs."""
    if df is None or len(df) < 2 * swing_n + 5:
        return {"bias": 0, "detail": "not enough bars"}
    highs, lows = _swings(df, swing_n)
    if len(highs) < 2 or len(lows) < 2:
        return {"bias": 0, "detail": "fewer than 2 swings per side"}
    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1] > lows[-2][1]
    lh = highs[-1][1] < highs[-2][1]
    ll = lows[-1][1] < lows[-2][1]
    if hh and hl:
        return {"bias": 1, "detail": "HH + HL"}
    if lh and ll:
        return {"bias": -1, "detail": "LH + LL"}
    return {"bias": 0, "detail": "mixed swings (range)"}


# ---------------------------------------------------------------- order blocks

def order_blocks(df: pd.DataFrame, *, lookback: int = 60) -> list[dict]:
    """Bullish / bearish order blocks from displacement bars.

    An OB is the last OPPOSITE candle before a displacement bar
    (body > 1.5x the 20-bar mean body).  The OB band is that
    candle's full range.  Returns unmitigated + mitigated blocks
    (newest first); mitigation = a later close inside the band
    that then pushes 0.5 ATR beyond it.
    """
    if df is None or len(df) < 25:
        return []
    sub = df.tail(lookback)
    o, h, l, c = (_arr(sub, k) for k in ("open", "high", "low", "close"))
    atr = _atr(df)
    if atr <= 0:
        return []
    bodies = np.abs(c - o)
    mean_body = float(np.mean(bodies[:-1])) if len(bodies) > 1 else 0.0
    out: list[dict] = []
    for i in range(1, len(sub) - 1):
        if bodies[i] <= 1.5 * mean_body or bodies[i] <= 0:
            continue
        direction = 1 if c[i] > o[i] else -1
        # OB = last opposite candle before the displacement bar.
        j = i - 1
        if j < 0 or (c[j] > o[j]) == (direction > 0):
            continue
        top, bottom = float(h[j]), float(l[j])
        mitigated = False
        for k in range(i + 1, len(sub)):
            if bottom <= c[k] <= top:
                # Inside close: mitigated only if it then pushes
                # 0.5 ATR beyond the band.
                later = c[k + 1:] if k + 1 < len(sub) else np.array([])
                if direction > 0 and len(later) \
                        and float(np.max(later)) > top + 0.5 * atr:
                    mitigated = True
                if direction < 0 and len(later) \
                        and float(np.min(later)) < bottom - 0.5 * atr:
                    mitigated = True
                break
        out.append({"kind": "bullish_ob" if direction > 0
                    else "bearish_ob", "direction": direction,
                    "top": top, "bottom": bottom,
                    "level": (top + bottom) / 2.0,
                    "bar_index": int(sub.index[i]),
                    "ts": sub.index[i], "mitigated": mitigated,
                    "score": 55.0 if not mitigated else 20.0})
    out.sort(key=lambda b: b["bar_index"], reverse=True)
    return out


def breaker_blocks(df: pd.DataFrame, *, lookback: int = 60) -> list[dict]:
    """Breakers: an OB that was BROKEN (close beyond it + BOS) and
    now acts as flipped support/resistance.

    A breaker is an order block whose band was closed through in
    the opposite direction and then respected from the other side.
    """
    obs = order_blocks(df, lookback=lookback)
    if not obs:
        return []
    c = _arr(df.tail(lookback), "close")
    out = []
    for ob in obs:
        if ob["mitigated"]:
            continue
        # Broken = a later close beyond the band opposite the OB.
        if ob["direction"] > 0:
            broken = bool(np.any(c > ob["top"]))
        else:
            broken = bool(np.any(c < ob["bottom"]))
        if broken:
            out.append({**ob, "kind": "breaker",
                        "direction": -ob["direction"],
                        "score": 65.0})
    return out


def mitigation_blocks(df: pd.DataFrame, *, lookback: int = 60) -> list[dict]:
    """Mitigation blocks: OBs that were mitigated and now act as
    the fresh OB for the next leg (the mitigation itself becomes
    the level)."""
    return [ob for ob in order_blocks(df, lookback=lookback)
            if ob["mitigated"]]


# ---------------------------------------------------------------- FVG

def fvg_list(df: pd.DataFrame, *, lookback: int = 120) -> list[dict]:
    """All FVGs in the window with fill / inversion state.

    Bullish FVG at bar i: low[i] > high[i-2] (gap up, unfilled
    imbalance).  Bearish: high[i] < low[i-2].  CE = 50% of the
    gap.  Inversion: the gap was filled (close through the far
    edge) and price then reversed — the filled FVG flips to
    support/resistance.  BPR: a bullish + bearish FVG overlapping
    (balanced price range = strong level).
    """
    if df is None or len(df) < 5:
        return []
    sub = df.tail(lookback)
    h, l, c = (_arr(sub, k) for k in ("high", "low", "close"))
    atr = _atr(df)
    out: list[dict] = []
    for i in range(2, len(sub)):
        if h[i] - l[i] <= FVG_MIN_ATR * atr and atr > 0:
            continue
        if l[i] > h[i - 2]:
            gap = (float(h[i - 2]), float(l[i]))
            direction = 1
        elif h[i] < l[i - 2]:
            gap = (float(h[i]), float(l[i - 2]))
            direction = -1
        else:
            continue
        top, bottom = max(gap), min(gap)
        filled = False
        inverted = False
        for k in range(i + 1, len(sub)):
            if bottom <= c[k] <= top:
                filled = True
            if filled and ((direction > 0 and c[k] < bottom)
                           or (direction < 0 and c[k] > top)):
                inverted = True
                break
        out.append({"kind": "bullish_fvg" if direction > 0
                    else "bearish_fvg", "direction": direction,
                    "top": top, "bottom": bottom,
                    "ce": (top + bottom) / 2.0,
                    "bar_index": int(sub.index[i]), "ts": sub.index[i],
                    "filled": filled, "inverted": inverted,
                    "score": 60.0 if not filled else (
                        65.0 if inverted else 25.0)})
    # BPR: overlapping bull + bear FVGs.
    for a in out:
        for b in out:
            if a is b or a["direction"] == b["direction"]:
                continue
            if min(a["top"], b["top"]) > max(a["bottom"], b["bottom"]):
                a["bpr"] = True
                a["score"] = max(a["score"], 75.0)
    out.sort(key=lambda f: f["bar_index"], reverse=True)
    return out


def unfilled_fvgs(df: pd.DataFrame, *, direction: int | None = None,
                  lookback: int = 120) -> list[dict]:
    """Unfilled FVGs (optionally one direction) — the entry levels."""
    return [f for f in fvg_list(df, lookback=lookback)
            if not f["filled"] and (direction is None
                                    or f["direction"] == direction)]


# ---------------------------------------------------------------- liquidity

def liquidity_pools(df: pd.DataFrame, *, tol_atr: float = 0.5,
                    min_points: int = 3, lookback: int = 240) -> list[dict]:
    """Equal highs / equal lows clusters = liquidity pools.

    3+ swing highs (or lows) within tol_atr x ATR of each other.
    Buy-side liquidity sits above equal highs; sell-side below
    equal lows.  Pools are where stops rest — and where sweeps hunt.
    """
    if df is None or len(df) < 10:
        return []
    atr = _atr(df)
    if atr <= 0:
        return []
    sub = df.tail(lookback)
    highs, lows = find_swings(_arr(sub, "high"), _arr(sub, "low"),
                              SWING_N)
    pools: list[dict] = []
    for side, swings in (("buy_side", highs), ("sell_side", lows)):
        levels = [p for _, p in swings]
        used = [False] * len(levels)
        for i, lv in enumerate(levels):
            if used[i]:
                continue
            group = [lv]
            used[i] = True
            for j in range(i + 1, len(levels)):
                if not used[j] and abs(levels[j] - lv) <= tol_atr * atr:
                    group.append(levels[j])
                    used[j] = True
            if len(group) >= min_points:
                pools.append({"side": side, "level": float(np.mean(group)),
                              "points": len(group), "score": 70.0,
                              "detail": f"{len(group)} equal "
                              f"{'highs' if side == 'buy_side' else 'lows'}"})
    return pools


def detect_sweep(df: pd.DataFrame, level: float, *, direction: int,
                 atr: float | None = None) -> dict:
    """Liquidity sweep of ``level`` on the last bars.

    Sweep = a bar wicks BEYOND the level by >= SWEEP_MIN_ATR x ATR
    and CLOSES BACK on the inside.  ``direction`` is the intended
    trade direction AFTER the sweep (opposite the sweep push):
    a sweep above a high (+1 push) sets up a SHORT (-1).
    A wick alone (no close back inside) is not a sweep.
    """
    if df is None or len(df) < 2:
        return _absent("not enough bars for sweep")
    a = atr if atr is not None else _atr(df)
    if a <= 0:
        return _absent("ATR unmeasurable")
    h, l, c = (_arr(df, k) for k in ("high", "low", "close"))
    for i in range(len(df) - 1, max(0, len(df) - 6), -1):
        if direction < 0:  # sweep above a high -> short
            if h[i] >= level + SWEEP_MIN_ATR * a and c[i] < level:
                return {"present": True, "score": 70.0,
                        "invalidation": float(h[i]),
                        "detail": f"sweep above {level:.1f}: wick "
                        f"{h[i]:.1f}, close back {c[i]:.1f}",
                        "wick": float(h[i]), "close": float(c[i]),
                        "bar_index": int(df.index[i])}
        else:  # sweep below a low -> long
            if l[i] <= level - SWEEP_MIN_ATR * a and c[i] > level:
                return {"present": True, "score": 70.0,
                        "invalidation": float(l[i]),
                        "detail": f"sweep below {level:.1f}: wick "
                        f"{l[i]:.1f}, close back {c[i]:.1f}",
                        "wick": float(l[i]), "close": float(c[i]),
                        "bar_index": int(df.index[i])}
    return _absent("no sweep wick + close-back-inside in last 5 bars")


def detect_displacement(df: pd.DataFrame, *, direction: int,
                        window: int = 3,
                        min_atr: float = DISPLACEMENT_MIN_ATR) -> dict:
    """Violent move: >= min_atr x ATR in <= window bars, with an
    FVG left behind.  Displacement without an FVG is drift, not
    institutional sponsorship."""
    if df is None or len(df) < window + 2:
        return _absent("not enough bars for displacement")
    atr = _atr(df)
    if atr <= 0:
        return _absent("ATR unmeasurable")
    c = _arr(df, "close")
    start = float(c[-window - 1])
    end = float(c[-1])
    move = (end - start) * direction
    if move < min_atr * atr:
        return _absent(f"move {move:.1f} < {min_atr}x ATR "
                       f"({min_atr * atr:.1f}) in {window} bars")
    fvgs = [f for f in fvg_list(df, lookback=window + 6)
            if f["direction"] == direction and not f["filled"]]
    if not fvgs:
        return _absent("displacement without an FVG — drift, not "
                       "sponsorship")
    return {"present": True, "score": 75.0,
            "invalidation": float(min(c[-window - 1:]))
            if direction > 0 else float(max(c[-window - 1:])),
            "detail": f"displacement {move:.1f} = "
            f"{move / atr:.1f}x ATR in {window} bars, FVG left",
            "move_atr": float(move / atr), "fvg": fvgs[0]}


# ---------------------------------------------------------------- QM / premium / OTE

def detect_qm(df: pd.DataFrame, *, lookback: int = 120) -> dict:
    """Quasimodo: head-and-shoulders structure, entry at the QM
    shoulder / OB.  Short QM: higher high (head), lower high
    (right shoulder), break below the neckline (low between head
    and right shoulder).  Mirror for longs.  Returns the most
    recent completed QM, or absent."""
    if df is None or len(df) < 30:
        return _absent("not enough bars for QM")
    sub = df.tail(lookback)
    highs, lows = find_swings(_arr(sub, "high"), _arr(sub, "low"),
                              SWING_N)
    if len(highs) < 3 or len(lows) < 2:
        return _absent("fewer than 3 swing highs / 2 swing lows")
    # Short QM: head = max high, shoulders on either side, the
    # right shoulder below the head, neckline = low between head
    # and right shoulder; trigger = close below neckline.
    for d in (-1, 1):
        if d < 0:
            head_i, head_p = max(highs, key=lambda p: p[1])
            later = [p for p in highs if p[0] > head_i]
            earlier = [p for p in highs if p[0] < head_i]
            if not later or not earlier:
                continue
            shoulder_i, shoulder_p = later[-1]
            if shoulder_p >= head_p:
                continue
            neck = [p for p in lows
                    if head_i < p[0] < shoulder_i]
            if not neck:
                continue
            neckline = min(p for _, p in neck)
            closes = _arr(sub, "close")
            trig = [i for i in range(shoulder_i + 1, len(sub))
                    if closes[i] < neckline]
            if trig:
                return {"present": True, "score": 70.0,
                        "invalidation": float(head_p),
                        "detail": f"short QM head {head_p:.1f} "
                        f"shoulder {shoulder_p:.1f} neckline "
                        f"{neckline:.1f}", "direction": -1,
                        "neckline": float(neckline),
                        "head": float(head_p),
                        "shoulder": float(shoulder_p)}
        else:
            head_i, head_p = min(lows, key=lambda p: p[1])
            later = [p for p in lows if p[0] > head_i]
            earlier = [p for p in lows if p[0] < head_i]
            if not later or not earlier:
                continue
            shoulder_i, shoulder_p = later[-1]
            if shoulder_p <= head_p:
                continue
            neck = [p for p in highs
                    if head_i < p[0] < shoulder_i]
            if not neck:
                continue
            neckline = max(p for _, p in neck)
            closes = _arr(sub, "close")
            trig = [i for i in range(shoulder_i + 1, len(sub))
                    if closes[i] > neckline]
            if trig:
                return {"present": True, "score": 70.0,
                        "invalidation": float(head_p),
                        "detail": f"long QM head {head_p:.1f} "
                        f"shoulder {shoulder_p:.1f} neckline "
                        f"{neckline:.1f}", "direction": 1,
                        "neckline": float(neckline),
                        "head": float(head_p),
                        "shoulder": float(shoulder_p)}
    return _absent("no completed QM in window")


def premium_discount(df: pd.DataFrame, *, lookback: int = 120) -> dict:
    """Equilibrium (50% of the range), premium / discount, OTE.

    Premium (above EQ) = shorts only.  Discount (below EQ) =
    longs only.  OTE = 62-79% Fibonacci retracement of the last
    displacement leg — the optimal trade entry on the retrace.
    """
    if df is None or len(df) < 10:
        return {"eq": None, "detail": "not enough bars"}
    sub = df.tail(lookback)
    hi, lo = float(sub["high"].max()), float(sub["low"].min())
    eq = (hi + lo) / 2.0
    last = float(sub["close"].iloc[-1])
    # Last displacement leg: the largest 5-bar move in the window.
    c = _arr(sub, "close")
    best = (0.0, 0, 0)
    for i in range(len(c) - 5):
        mv = abs(float(c[i + 5]) - float(c[i]))
        if mv > best[0]:
            best = (mv, i, i + 5)
    _, a, b = best
    leg_lo, leg_hi = min(float(c[a]), float(c[b])), \
        max(float(c[a]), float(c[b]))
    leg_dir = 1 if float(c[b]) > float(c[a]) else -1
    span = leg_hi - leg_lo
    ote = {"low": leg_hi - 0.79 * span, "high": leg_hi - 0.62 * span} \
        if leg_dir > 0 else \
        {"low": leg_lo + 0.62 * span, "high": leg_lo + 0.79 * span}
    return {"eq": eq, "range_high": hi, "range_low": lo,
            "zone": "premium" if last > eq else "discount",
            "allowed_direction": -1 if last > eq else 1,
            "ote": ote, "ote_50": (leg_hi + leg_lo) / 2.0,
            "detail": f"EQ {eq:.1f}, price {last:.1f} in "
            f"{'premium (shorts only)' if last > eq else 'discount (longs only)'}"}


# ---------------------------------------------------------------- AMD / inducement / turtle soup

def amd_phase(df: pd.DataFrame, *, lookback: int = 120) -> dict:
    """Accumulation -> Manipulation -> Distribution, from bars.

    Accumulation: range + compression (ATR contracting, inside
    bars).  Manipulation: a sweep in the last N bars.  Distribution:
    displacement + BOS after a sweep.  AMD is the NARRATIVE the
    state machine maps to — never a gate on its own.
    """
    if df is None or len(df) < 30:
        return {"phase": "unknown", "detail": "not enough bars"}
    sub = df.tail(lookback)
    atr = _atr(df)
    atrs = sub["high"].rolling(14).max() - sub["low"].rolling(14).min()
    compressed = False
    if len(atrs) >= 28:
        recent = float(np.nanmean(atrs.tail(14)))
        older = float(np.nanmean(atrs.iloc[-28:-14]))
        compressed = older > 0 and recent < 0.75 * older
    h, l, c = (_arr(sub, k) for k in ("high", "low", "close"))
    hi, lo = float(np.max(h)), float(np.min(l))
    swept = None
    for i in range(len(sub) - 1, max(0, len(sub) - 20), -1):
        if h[i] >= hi + 0.25 * atr and c[i] < hi:
            swept = ("high", float(h[i]), int(sub.index[i]))
            break
        if l[i] <= lo - 0.25 * atr and c[i] > lo:
            swept = ("low", float(l[i]), int(sub.index[i]))
            break
    if swept is not None:
        # Displacement after the sweep?
        tail = sub.iloc[swept[2] + 1:]
        if len(tail):
            mv = float(tail["close"].iloc[-1] - tail["close"].iloc[0])
            if abs(mv) >= DISPLACEMENT_MIN_ATR * atr:
                return {"phase": "distribution",
                        "detail": f"sweep {swept[0]} then "
                        f"displacement {mv:.1f}",
                        "sweep": swept}
        return {"phase": "manipulation",
                "detail": f"sweep {swept[0]} {swept[1]:.1f}, "
                "awaiting displacement", "sweep": swept}
    if compressed:
        return {"phase": "accumulation",
                "detail": "range + ATR compression"}
    return {"phase": "unknown",
            "detail": "no accumulation / sweep signature"}


def inducement_level(df: pd.DataFrame, *, direction: int) -> dict:
    """IDM: the last swing before the sweep — the inducement that
    must be taken before the true move.  Longs: the last lower
    low swept (sell-side inducement).  Shorts: the last higher
    high swept (buy-side inducement)."""
    if df is None or len(df) < 10:
        return _absent("not enough bars for inducement")
    highs, lows = _swings(df)
    if direction > 0:
        if not lows:
            return _absent("no swing lows")
        lv = lows[-1][1]
        return {"present": True, "score": 50.0, "invalidation": None,
                "detail": f"sell-side inducement low {lv:.1f}",
                "level": float(lv), "direction": 1}
    if not highs:
        return _absent("no swing highs")
    lv = highs[-1][1]
    return {"present": True, "score": 50.0, "invalidation": None,
            "detail": f"buy-side inducement high {lv:.1f}",
            "level": float(lv), "direction": -1}


def turtle_soup(df: pd.DataFrame, level: float, *,
                direction: int) -> dict:
    """Turtle soup: sweep of a prior high/low + reclaim close back
    inside within 2 bars.  The classic failed-breakout fade."""
    sweep = detect_sweep(df, level, direction=direction)
    if not sweep["present"]:
        return _absent("no sweep for turtle soup")
    return {"present": True, "score": 72.0,
            "invalidation": sweep["invalidation"],
            "detail": "turtle soup: " + sweep["detail"],
            "sweep": sweep}


# ---------------------------------------------------------------- SMT / sessions / volume / news

def smt_divergence(xau: pd.DataFrame, other: pd.DataFrame,
                   *, name: str = "DXY") -> dict:
    """Smart Money Technique: correlated assets diverge at swings.

    XAU makes a lower low while the correlated asset makes a
    higher low (or vice versa) at the same swing — divergence at
    the turning point.  Returns direction implied for XAU.
    """
    if xau is None or other is None or len(xau) < 10 \
            or len(other) < 10:
        return _absent("not enough bars for SMT")
    _, x_lows = _swings(xau)
    _, o_lows = _swings(other)
    _, x_highs = _swings(xau)
    _, o_highs = _swings(other)
    if len(x_lows) >= 2 and len(o_lows) >= 2:
        if x_lows[-1][1] < x_lows[-2][1] \
                and o_lows[-1][1] > o_lows[-2][1]:
            return {"present": True, "score": 65.0,
                    "invalidation": None,
                    "detail": f"SMT bullish divergence vs {name}: "
                    "XAU lower low, other higher low",
                    "direction": 1}
    if len(x_highs) >= 2 and len(o_highs) >= 2:
        if x_highs[-1][1] > x_highs[-2][1] \
                and o_highs[-1][1] < o_highs[-2][1]:
            return {"present": True, "score": 65.0,
                    "invalidation": None,
                    "detail": f"SMT bearish divergence vs {name}: "
                    "XAU higher high, other lower high",
                    "direction": -1}
    return _absent(f"no SMT divergence vs {name}")


def session_label(ts) -> dict:
    """Session + killzone for a timestamp (NY time).

    London killzone 02:00-05:00 NY, NY killzone 07:00-10:00 NY,
    silver bullet 10:00 NY, London close 11:00 NY, NY close
    15:00 NY, Asia range overnight.  Uses the reader's session
    clock when available, UTC-hour fallback otherwise.
    """
    try:
        from ..reader.session_timeframes import session_of
        sess, _ = session_of(pd.Timestamp(ts))
    except Exception:
        sess = "unknown"
    try:
        ny = pd.Timestamp(ts).tz_convert("America/New_York") \
            if pd.Timestamp(ts).tzinfo is not None else \
            pd.Timestamp(ts).tz_localize("UTC").tz_convert(
                "America/New_York")
        hh = ny.hour + ny.minute / 60.0
    except Exception:
        hh = -1.0
    killzone = None
    if 2.0 <= hh < 5.0:
        killzone = "london_killzone"
    elif 7.0 <= hh < 10.0:
        killzone = "ny_killzone"
    elif 9.9 <= hh < 10.3:
        killzone = "silver_bullet"
    elif 11.0 <= hh < 12.0:
        killzone = "london_close"
    elif 15.0 <= hh < 16.0:
        killzone = "ny_close"
    return {"session": sess, "killzone": killzone,
            "in_killzone": killzone is not None,
            "detail": f"session {sess}, killzone {killzone or 'none'}"}


def volume_confirm(df: pd.DataFrame, *, ratio_min: float = 1.5) -> dict:
    """Volume spike on the last bar: tick volume >= ratio x the
    20-bar mean.  Confirmation context, never a gate."""
    if df is None or "volume" not in df.columns or len(df) < 21:
        return {"present": False, "score": 0.0,
                "detail": "no volume data"}
    v = _arr(df.tail(21), "volume")
    mean = float(np.mean(v[:-1]))
    if mean <= 0:
        return {"present": False, "score": 0.0,
                "detail": "zero mean volume"}
    ratio = float(v[-1]) / mean
    return {"present": ratio >= ratio_min, "score": 10.0,
            "invalidation": None,
            "detail": f"volume ratio {ratio:.2f} "
            f"({'surge' if ratio >= ratio_min else 'no surge'})",
            "ratio": ratio}


def news_pause(calendar, now) -> dict:
    """High-impact news pause: no NEW entries 15 min before to
    45 min after a high-impact event (management only in the
    window).  Delegates to the existing NewsCalendar — this is
    the pattern-library surface for it."""
    if calendar is None:
        return {"paused": False, "detail": "no news calendar"}
    try:
        ex = calendar.execution_state(now)
        return {"paused": bool(ex.paused), "regime": ex.regime,
                "size_mult": float(ex.size_mult),
                "detail": f"news {ex.regime}, paused={ex.paused}"}
    except Exception as exc:
        return {"paused": False, "detail": f"news check failed: {exc}"}
