"""Imbalance (FVG) deep mining — Phase B measurements on 2y GC=F H1.

Mined claims under test (see experiments/IMBALANCE.md, Phase A):
  1. MAGNET: unfilled FVGs get revisited ("price returns to fill inefficiency").
  2. CE: on return, the 50% midline (consequent encroachment) holds better
     than the gap edge (IOFED).
  3. IFVG: a body-close through the gap inverts it; continuation follows in
     the inversion direction.
  4. STACKING: legs with 2+ stacked FVGs make stronger entry zones.

All geometry is recomputed causally from bars.  FVG rule (repo's):
bullish FVG is known at the close of bar t when low[t] > high[t-2];
the gap is (high[t-2], low[t]).  Bearish mirrors.
Mitigation (reader's rule): bullish dies on any low[j] <= bottom for
j in (created_at, upto].  Inversion: close[j] < bottom (body close
through) — a strict subset of mitigations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.data import load_gold_h1
from ..backtest.harness import COST_PER_OZ

STOP_BUFFER_ATR = 0.25
MIN_RISK_ATR = 0.25
MAX_RISK_ATR = 4.0
TRADE_HORIZON = 72


@dataclass
class Gap:
    """One 3-candle imbalance, fully described for causal replays."""

    direction: int          # +1 bullish (BISI) | -1 bearish (SIBI)
    top: float
    bottom: float
    created_at: int         # index of candle t (the third candle)
    size: float             # top - bottom
    atr_at: float
    disp_body_ratio: float  # body/range of the displacement candle (t-1)
    disp_vol_pct: float     # tick-volume percentile of candle t-1 (0-1)
    daily_bias: int         # completed-D1 bias sign at creation (+1/-1/0)

    @property
    def mid(self) -> float:
        return 0.5 * (self.top + self.bottom)


@dataclass
class GapLife:
    """What happened to a gap after creation (all bars > created_at)."""

    gap: Gap
    end_index: int | None        # first bar ending the gap's active life
    end_kind: str | None         # 'wick' | 'inversion' | None (survived)
    first_touch_index: int | None  # first bar touching the gap zone


def detect_gaps(df: pd.DataFrame) -> list[Gap]:
    """Every 3-candle FVG on the series, with context.  Causal by
    construction: gap at t uses only bars <= t."""
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy() if "volume" in df.columns else np.zeros(len(df))
    n = len(df)
    atr = _atr(h, l, c, 14)
    # rolling volume percentile of the displacement candle: rank of v[t-1]
    # within the trailing 100 bars.
    gaps: list[Gap] = []
    for t in range(2, n):
        bull = l[t] > h[t - 2] and l[t] > l[t - 1]
        bear = h[t] < l[t - 2] and h[t] < h[t - 1]
        if not (bull or bear):
            continue
        d = 1 if bull else -1
        if d == 1:
            top, bottom = l[t], h[t - 2]
        else:
            top, bottom = l[t - 2], h[t]
        if top <= bottom:
            continue
        size = top - bottom
        body = abs(c[t - 1] - o[t - 1])
        rng = h[t - 1] - l[t - 1]
        body_ratio = body / rng if rng > 0 else 0.0
        lo = max(0, t - 100)
        vol_pct = float(np.searchsorted(np.sort(v[lo:t]), v[t - 1]) / max(1, t - lo))
        gaps.append(Gap(direction=d, top=top, bottom=bottom, created_at=t,
                        size=size, atr_at=atr[t],
                        disp_body_ratio=body_ratio, disp_vol_pct=vol_pct,
                        daily_bias=_daily_bias(c, t)))
    return gaps


def replay_lives(df: pd.DataFrame, gaps: list[Gap],
                 max_age: int = 240) -> list[GapLife]:
    """For each gap, find first touch, and the first bar ending its life
    (wick touch vs body-close inversion), within max_age bars."""
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)
    lives: list[GapLife] = []
    for g in gaps:
        first_touch = None
        end_index = None
        end_kind = None
        stop = min(n - 1, g.created_at + max_age)
        for j in range(g.created_at + 1, stop + 1):
            touched = (l[j] <= g.bottom) if g.direction == 1 else (h[j] >= g.top)
            if touched and first_touch is None:
                first_touch = j
                inverted = (c[j] < g.bottom) if g.direction == 1 else (c[j] > g.top)
                end_index = j
                end_kind = "inversion" if inverted else "wick"
                break
        lives.append(GapLife(gap=g, end_index=end_index, end_kind=end_kind,
                             first_touch_index=first_touch))
    return lives


# ------------------------------------------------------------ test 1: magnet
def magnet_stats(lives: list[GapLife]) -> dict:
    """Fill (revisit) rate by age, size, and leg context."""
    out: dict = {}
    ages = [12, 24, 48, 72, 168]
    for a in ages:
        hit = sum(1 for lf in lives
                  if lf.first_touch_index is not None
                  and lf.first_touch_index - lf.gap.created_at <= a)
        out[f"fill_within_{a}"] = (hit, len(lives), hit / len(lives) if lives else 0.0)
    # size terciles (gap / ATR)
    ratios = np.array([lf.gap.size / lf.gap.atr_at if lf.gap.atr_at > 0 else 0.0
                       for lf in lives])
    if len(ratios):
        q1, q2 = np.quantile(ratios, [1 / 3, 2 / 3])
        for name, sel in (("small", ratios <= q1),
                          ("mid", (ratios > q1) & (ratios <= q2)),
                          ("large", ratios > q2)):
            sub = [lf for lf, s in zip(lives, sel) if s]
            hit = sum(1 for lf in sub if lf.first_touch_index is not None)
            out[f"size_{name}"] = (hit, len(sub), hit / len(sub) if sub else 0.0)
    # leg context: gap direction vs daily bias
    for name, pred in (("with_bias", lambda g: g.daily_bias == g.direction),
                       ("against_bias", lambda g: g.daily_bias == -g.direction),
                       ("no_bias", lambda g: g.daily_bias == 0)):
        sub = [lf for lf in lives if pred(lf.gap)]
        hit = sum(1 for lf in sub if lf.first_touch_index is not None)
        out[f"ctx_{name}"] = (hit, len(sub), hit / len(sub) if sub else 0.0)
    return out


def placebo_stats(df: pd.DataFrame, gaps: list[Gap], seed: int = 7) -> dict:
    """Control: same-size zones placed at random bars AT THE SAME DISTANCE
    from price as the real gaps (a gap sits away from price after
    displacement — a zone centered on the close would touch trivially).
    For each real gap we record its signed distance from the close to the
    near edge; the placebo replays that distance at a random bar."""
    rng = np.random.default_rng(seed)
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)
    out: dict = {}
    for a in (24, 72, 168):
        hit = 0
        m = 0
        for g in gaps:
            t = int(rng.integers(2, n - a - 1))
            # real gap's signed distance from close to near edge
            near = g.bottom if g.direction == 1 else g.top
            dist = near - c[g.created_at]
            half = g.size / 2
            center = c[t] + dist
            lo, hi = center - half, center + half
            touched = np.any(l[t + 1:t + 1 + a] <= lo) if g.direction == 1 else \
                np.any(h[t + 1:t + 1 + a] >= hi)
            hit += int(touched)
            m += 1
        out[f"placebo_fill_within_{a}"] = (hit, m, hit / m)
    return out


# ------------------------------------------------- test 2: CE vs edge entry
@dataclass
class EntryResult:
    entry_kind: str       # 'mid' | 'edge'
    direction: int
    r_mult: float
    won: bool
    bounced_1r: bool      # reached +1R before -1R


def ce_vs_edge(df: pd.DataFrame, lives: list[GapLife],
               atr: np.ndarray) -> list[EntryResult]:
    """At the FIRST touch of an unfilled gap, enter at midline vs edge.
    Stop beyond the far edge + 0.25 ATR buffer; fixed 2R target; stop
    wins ties; $0.45/oz cost.  Only gaps whose first touch is a wick
    touch (not an inversion) qualify — an inversion is a different
    trade (test 3)."""
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)
    results: list[EntryResult] = []
    for lf in lives:
        g = lf.gap
        if lf.end_kind != "wick" or lf.first_touch_index is None:
            continue
        t = lf.first_touch_index
        buf = STOP_BUFFER_ATR * atr[t]
        if g.direction == 1:
            entries = {"mid": g.mid, "edge": g.bottom}
            stop = g.top + buf          # far edge for a long pullback entry
        else:
            entries = {"mid": g.mid, "edge": g.top}
            stop = g.bottom - buf
        for kind, entry in entries.items():
            risk = abs(entry - stop)
            if risk <= 0 or not (MIN_RISK_ATR * atr[t] <= risk <= MAX_RISK_ATR * atr[t]):
                continue
            tp = entry + 2 * risk * g.direction
            res = _walk_trade(o, h, l, t, entry, stop, tp, g.direction, risk)
            results.append(EntryResult(entry_kind=kind, direction=g.direction,
                                       r_mult=res[0], won=res[0] > 0,
                                       bounced_1r=res[1]))
    return results


def _walk_trade(o, h, l, t, entry, stop, tp, direction, risk):
    """Walk bars t+1.. for first touch of stop/tp.  Returns (R, hit_1r)."""
    cost_r = COST_PER_OZ / risk if risk > 0 else 0.0
    hit_1r = False
    stop = min(stop, entry) if direction == 1 else max(stop, entry)
    for j in range(t + 1, min(len(o), t + 1 + TRADE_HORIZON)):
        lo, hi = l[j], h[j]
        if direction == 1:
            stop_hit = lo <= stop
            tp_hit = hi >= tp
            r1_hit = hi >= entry + risk
        else:
            stop_hit = hi >= stop
            tp_hit = lo <= tp
            r1_hit = lo <= entry - risk
        hit_1r = hit_1r or r1_hit
        if stop_hit and tp_hit:
            return -1.0 - cost_r, hit_1r   # stop wins ties
        if stop_hit:
            return -1.0 - cost_r, hit_1r
        if tp_hit:
            return 2.0 - cost_r, hit_1r
    # time stop: mark at close
    from math import isnan  # noqa
    return 0.0 - cost_r, hit_1r


# ------------------------------------------------------------ test 3: IFVG
def ifvg_drift(df: pd.DataFrame, lives: list[GapLife]) -> dict:
    """After a body-close inversion, measure drift in the inversion
    direction over 12/24/48 bars vs the all-bar base rate."""
    c = df["close"].to_numpy()
    atr = _atr(df["high"].to_numpy(), df["low"].to_numpy(), c, 14)
    n = len(df)
    inv = [(lf.end_index, -lf.gap.direction) for lf in lives
           if lf.end_kind == "inversion" and lf.end_index is not None]
    out: dict = {"n_inversions": len(inv)}
    for horizon in (12, 24, 48):
        ds = []
        for idx, d in inv:
            if idx + horizon >= n:
                continue
            move = (c[idx + horizon] - c[idx]) * d
            ds.append(move / atr[idx] if atr[idx] > 0 else 0.0)
        # base rate: same-horizon drift of all bars, signed randomly is
        # meaningless — use absolute comparison: mean drift magnitude of
        # a random direction vs the inversion direction.
        base = []
        for idx in range(100, n - horizon, 7):
            base.append(abs(c[idx + horizon] - c[idx]) / atr[idx] if atr[idx] > 0 else 0.0)
        out[f"h{horizon}"] = {
            "n": len(ds),
            "mean_drift_atr": float(np.mean(ds)) if ds else 0.0,
            "hit_rate": float(np.mean([x > 0 for x in ds])) if ds else 0.0,
            "base_abs_drift_atr": float(np.mean(base)) if base else 0.0,
        }
    return out


# ---------------------------------------------------------- test 4: stacking
def stacking_stats(df: pd.DataFrame, gaps: list[Gap],
                   lives: list[GapLife]) -> dict:
    """Legs with 2+ overlapping same-direction gaps vs single gaps:
    when price first returns to the zone, does it hold (wick touch,
    not inversion) more often?"""
    by_life = {id(lf.gap): lf for lf in lives}
    out: dict = {}
    for g in gaps:
        # stacked = another same-direction gap whose zone overlaps, created
        # within 24 bars before
        stacked = any(
            o.direction == g.direction and o.created_at < g.created_at
            and o.created_at >= g.created_at - 24
            and not (o.top < g.bottom or o.bottom > g.top)
            for o in gaps)
        lf = by_life.get(id(g))
        if lf is None or lf.end_kind is None:
            continue
        key = "stacked" if stacked else "single"
        b = out.setdefault(key, {"n": 0, "holds": 0})
        b["n"] += 1
        b["holds"] += 1 if lf.end_kind == "wick" else 0
    for k, b in out.items():
        b["hold_rate"] = b["holds"] / b["n"] if b["n"] else 0.0
    return out


# --------------------------------------------------------------- internals
def _atr(h, l, c, period=14):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)),
                                      np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    out = np.zeros_like(tr)
    out[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _daily_bias(c: np.ndarray, t: int) -> int:
    """Sign of the last completed daily candle's direction, approximated
    from 24-bar blocks.  Coarse but causal."""
    if t < 48:
        return 0
    day_open = c[t - 24]
    day_close = c[t - 1]
    if day_close > day_open * 1.0005:
        return 1
    if day_close < day_open * 0.9995:
        return -1
    return 0


def summarize(entries: list[EntryResult]) -> dict:
    out: dict = {}
    for kind in ("mid", "edge"):
        sub = [e for e in entries if e.entry_kind == kind]
        rs = [e.r_mult for e in sub]
        out[kind] = {
            "n": len(sub),
            "avg_r": float(np.mean(rs)) if rs else 0.0,
            "win_rate": float(np.mean([r > 0 for r in rs])) if rs else 0.0,
            "bounce_1r": float(np.mean([e.bounced_1r for e in sub])) if sub else 0.0,
        }
    return out
