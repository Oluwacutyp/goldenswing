"""Candlestick context map: which candle works WHERE.

Phase A (the mined claims) is documented in experiments/CANDLES.md.
Phase B (this module): every candle signal is a pure geometric event on
confirmed bars; the trade is evaluated per signal x context cell on 2y
GC=F H1 with repo conventions: entry at the next bar's open, structural
stop (0.25 ATR buffer), fixed 2R target, 72-bar horizon, $0.45/oz.

The deliverable is the context map, not a book: for each signal we report
n / win% / avg R overall and inside each context cell (location, trend
alignment, killzone, ATR regime).  The owner's rule governs: a candle is
not good or bad — it is good or bad WHERE it fires.

Mined claims under test (sources in CANDLES.md):
- Bulkowski: bearish engulfing 79% reversal; morning star 78%;
  three white soldiers 82% and three black crows 78% (as REVERSALS,
  i.e. the push exhausts); bullish engulfing 63% reversal but weak
  overall (rank 84/103); doji ~55% (coin flip); pattern potency
  decays 3-5 bars after completion.
- ICT/SMC: displacement candle = large body (>=1.5x ATR), minimal
  wicks, body-close through structure (wick-only breaks are NOT valid);
  valid MSS = displacement + structure break + FVG; rejection block =
  zone defined by WICKS at a PD array; indecision candle marks zones;
  mitigation blocks are one-touch only.
- Our own history: only bearish pin bars at supply carried edge
  (+$1,755/31 trades); bullish candle triggers lost.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.data import load_gold_h1
from ..backtest.harness import COST_PER_OZ
from ..features import read_and_featurize
from ..reader.reader import MarketReader

# ---------------------------------------------------------------- constants
#: Stop buffer beyond the signal extreme (Phase-2 convention).
STOP_BUFFER_ATR = 0.25
#: Stop-distance sanity band, in ATR (repo convention).
MIN_RISK_ATR = 0.25
MAX_RISK_ATR = 4.0
#: Fixed target isolates the candle signal (exits held constant).
FIXED_TP_R = 2.0
#: Max hold, bars.
TRADE_HORIZON = 72
#: ICT displacement: body must be this multiple of ATR.
DISP_BODY_ATR = 1.5
#: Displacement wicks: each wick at most this fraction of the range.
DISP_WICK_FRAC = 0.25
#: Pin bar: dominant wick at least this multiple of the body.
PIN_WICK_BODY = 2.0
#: Pin bar: dominant wick at least this fraction of the range.
PIN_WICK_RANGE = 0.5
#: Doji: body at most this fraction of the range.
DOJI_BODY_FRAC = 0.10
#: Zone/AOC/pool proximity thresholds, in ATR.
ZONE_TOL_ATR = 0.5
AOC_TOL_ATR = 0.75
POOL_TOL_ATR = 1.0
#: ATR regime cutoffs (percentile).
ATR_HI = 70.0
ATR_LO = 30.0

SIGNALS = ("pin", "engulf", "displace", "inside_break", "doji", "push_fade")


# ------------------------------------------------------------- geometry
def _o_h_l_c(df: pd.DataFrame):
    return (df["open"].to_numpy(dtype=float), df["high"].to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float), df["close"].to_numpy(dtype=float))


def pin_bar(o, h, l, c, i: int) -> int:
    """Rejection candle: dominant wick >= 2x body and >= 50% of range.

    Returns +1 (bullish, lower-wick rejection), -1 (bearish, upper-wick
    rejection), 0 (no pin).  Pure geometry, no context.
    """
    rng = h[i] - l[i]
    if rng <= 0:
        return 0
    body = abs(c[i] - o[i])
    upper = h[i] - max(o[i], c[i])
    lower = min(o[i], c[i]) - l[i]
    if lower >= PIN_WICK_BODY * body and lower >= PIN_WICK_RANGE * rng \
            and upper < lower:
        return 1
    if upper >= PIN_WICK_BODY * body and upper >= PIN_WICK_RANGE * rng \
            and lower < upper:
        return -1
    return 0


def engulfing(o, h, l, c, i: int) -> int:
    """Body of bar i fully engulfs body of bar i-1, opposite colours."""
    if i < 1:
        return 0
    b0, b1 = c[i - 1] - o[i - 1], c[i] - o[i]
    if b0 == 0 or b1 == 0 or (b0 > 0) == (b1 > 0):
        return 0
    lo0, hi0 = min(o[i - 1], c[i - 1]), max(o[i - 1], c[i - 1])
    lo1, hi1 = min(o[i], c[i]), max(o[i], c[i])
    if lo1 <= lo0 and hi1 >= hi0:
        return 1 if b1 > 0 else -1
    return 0


def displacement(o, h, l, c, atr: float, i: int,
                 mult: float = DISP_BODY_ATR) -> int:
    """ICT displacement: body >= mult x ATR, minimal wicks, body direction.

    Wick-only structure breaks are NOT displacement (MSS rule) — this is
    enforced by the wick cap, not by structure (the caller gates on it).
    """
    rng = h[i] - l[i]
    if rng <= 0 or atr <= 0:
        return 0
    body = c[i] - o[i]
    if abs(body) < mult * atr:
        return 0
    upper = h[i] - max(o[i], c[i])
    lower = min(o[i], c[i]) - l[i]
    if upper > DISP_WICK_FRAC * rng or lower > DISP_WICK_FRAC * rng:
        return 0
    return 1 if body > 0 else -1


def inside_break(o, h, l, c, i: int) -> int:
    """Bar i-1 inside bar i-2, bar i closes beyond bar i-1's extreme."""
    if i < 2:
        return 0
    if h[i - 1] <= h[i - 2] and l[i - 1] >= l[i - 2]:
        if c[i] > h[i - 1]:
            return 1
        if c[i] < l[i - 1]:
            return -1
    return 0


def is_doji(o, h, l, c, atr: float, i: int) -> bool:
    """Indecision: tiny body on a non-trivial range."""
    rng = h[i] - l[i]
    if rng <= 0 or atr <= 0 or rng < 0.5 * atr:
        return False
    return abs(c[i] - o[i]) <= DOJI_BODY_FRAC * rng


def three_push(o, h, l, c, i: int) -> int:
    """Three consecutive same-direction closes with real bodies.

    Returns +1 for three up-closes, -1 for three down-closes, 0 otherwise.
    Bulkowski's stats say soldiers/crows act as REVERSALS (the push
    exhausts), so the traded signal fades the push (see ``push_fade``).
    """
    if i < 2:
        return 0
    dirs = []
    for k in (i - 2, i - 1, i):
        rng = h[k] - l[k]
        if rng <= 0:
            return 0
        b = c[k] - o[k]
        if abs(b) < 0.5 * rng:
            return 0
        dirs.append(1 if b > 0 else -1)
    if dirs[0] == dirs[1] == dirs[2]:
        return dirs[0]
    return 0


# ---------------------------------------------------------------- context
@dataclass
class CandleSetup:
    """A candle signal at the close of bar ``index``."""
    index: int
    signal: str          # one of SIGNALS
    direction: int       # +1 long | -1 short
    stop: float          # buffered structural stop
    risk: float          # |entry_ref - stop|, price units (refined at fill)
    atr: float
    ctx: dict = field(default_factory=dict)


def _zone_near(zones, supply: bool, tol_atr: float) -> bool:
    """Is close at/inside the relevant zone or within ``tol_atr`` of it?"""
    if zones is None:
        return False
    if supply:
        if zones.in_supply:
            return True
        d = zones.supply_dist_atr
    else:
        if zones.in_demand:
            return True
        d = zones.demand_dist_atr
    return bool(d is not None and not np.isnan(d) and d <= tol_atr)


def _location(state, direction: int) -> str:
    """Where the candle fired: at_zone / at_aoc / at_pool / midrange."""
    fs = getattr(state, "features", None)
    zones = getattr(fs, "zones", None) if fs else None
    aoc = getattr(fs, "aoc", None) if fs else None
    atr = state.atr if state.atr > 0 else 1e-9
    if direction == 1:
        if _zone_near(zones, supply=False, tol_atr=ZONE_TOL_ATR):
            return "at_zone"
        if aoc is not None and aoc.below is not None \
                and not np.isnan(aoc.below_dist_atr) \
                and aoc.below_dist_atr <= AOC_TOL_ATR:
            return "at_aoc"
        if any((p.price - state.close) <= POOL_TOL_ATR * atr
               for p in state.pools_below):
            return "at_pool"
    else:
        if _zone_near(zones, supply=True, tol_atr=ZONE_TOL_ATR):
            return "at_zone"
        if aoc is not None and aoc.above is not None \
                and not np.isnan(aoc.above_dist_atr) \
                and aoc.above_dist_atr <= AOC_TOL_ATR:
            return "at_aoc"
        if any((state.close - p.price) <= POOL_TOL_ATR * atr
               for p in state.pools_above):
            return "at_pool"
    return "midrange"


def _context(state, direction: int) -> dict:
    """The how/where/when cell for one signal."""
    if state.bias == 0:
        align = "range"
    elif state.bias == direction:
        align = "with"
    else:
        align = "against"
    ap = state.atr_percentile
    regime = "hi" if ap >= ATR_HI else ("lo" if ap <= ATR_LO else "mid")
    return {
        "location": _location(state, direction),
        "trend": state.trend,
        "trend_align": align,
        "killzone": bool(state.killzone),
        "atr_regime": regime,
    }


# ------------------------------------------------------------------- scan
def _atr(df: pd.DataFrame) -> np.ndarray:
    """Causal ATR(14) via Wilder smoothing on confirmed bars."""
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.empty(n)
    atr[:14] = np.nan
    if n > 14:
        atr[13] = tr[:14].mean()
        for i in range(14, n):
            atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    return atr


def scan(df: pd.DataFrame, states: list) -> list[CandleSetup]:
    """Find every candle signal on confirmed bars.  Causal: geometry uses
    bars <= i, context uses the state's confirmed features at bar i."""
    o, h, l, c = _o_h_l_c(df)
    atr = _atr(df)
    n = len(df)
    setups: list[CandleSetup] = []

    def _stop(direction: int, extreme: float, atr_i: float):
        return extreme - direction * STOP_BUFFER_ATR * atr_i

    def _add(i: int, signal: str, direction: int, extreme: float):
        atr_i = atr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            return
        stop = _stop(direction, extreme, atr_i)
        # risk measured from the signal bar close (refined at fill)
        risk = (c[i] - stop) * direction
        if risk < MIN_RISK_ATR * atr_i or risk > MAX_RISK_ATR * atr_i:
            return
        ctx = _context(states[i], direction)
        setups.append(CandleSetup(index=i, signal=signal, direction=direction,
                                  stop=stop, risk=risk, atr=atr_i, ctx=ctx))

    for i in range(3, n - 1):          # need i+1 for the fill bar
        st = states[i]
        if getattr(st, "atr", 0) is None:
            continue
        d = pin_bar(o, h, l, c, i)
        if d:
            ext = l[i] if d == 1 else h[i]
            _add(i, "pin", d, ext)
        d = engulfing(o, h, l, c, i)
        if d:
            ext = min(l[i - 1], l[i]) if d == 1 else max(h[i - 1], h[i])
            _add(i, "engulf", d, ext)
        d = displacement(o, h, l, c, atr[i], i)
        if d:
            ext = l[i] if d == 1 else h[i]
            _add(i, "displace", d, ext)
        d = inside_break(o, h, l, c, i)
        if d:
            ext = l[i - 2] if d == 1 else h[i - 2]
            _add(i, "inside_break", d, ext)
        if is_doji(o, h, l, c, atr[i], i):
            # direction from location: rejection AWAY from the zone.
            # long off demand, short off supply; muted elsewhere.
            fs = getattr(st, "features", None)
            zones = getattr(fs, "zones", None) if fs else None
            dz = None
            if _zone_near(zones, supply=False, tol_atr=ZONE_TOL_ATR):
                dz = 1
            elif _zone_near(zones, supply=True, tol_atr=ZONE_TOL_ATR):
                dz = -1
            if dz:
                ext = l[i] if dz == 1 else h[i]
                _add(i, "doji", dz, ext)
        push = three_push(o, h, l, c, i)
        if push:
            # Bulkowski: soldiers/crows act as REVERSALS — fade the push.
            # The stop sits beyond the push extreme (the level the fade
            # says will hold).
            d = -push
            ext = (max(h[i - 2], h[i - 1], h[i]) if d == -1
                   else min(l[i - 2], l[i - 1], l[i]))
            _add(i, "push_fade", d, ext)
    return setups


# --------------------------------------------------------------- simulate
@dataclass
class CandleOutcome:
    index: int
    signal: str
    direction: int
    entry_index: int
    exit_index: int
    exit_reason: str            # "target" | "stop" | "timeout"
    r_net: float
    hold_bars: int
    ctx: dict = field(default_factory=dict)


def simulate(df: pd.DataFrame, s: CandleSetup) -> CandleOutcome | None:
    """Honest bar walk, v1 conventions: entry at the next bar's open,
    stop wins ties, gap opens fill at the adverse open, $0.45/oz cost,
    72-bar horizon.  Entry is NOT the candle itself — it is the
    confirmation that the candle's read was tradeable."""
    t = s.index
    n = len(df)
    if t + 1 >= n:
        return None
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    d = s.direction
    entry = o[t + 1]
    risk = (entry - s.stop) * d
    if risk <= 0:
        return None
    target = entry + d * FIXED_TP_R * risk

    last = min(t + TRADE_HORIZON, n - 1)
    exit_price, exit_idx, reason = cl[last], last, "timeout"
    for k in range(t + 1, last + 1):
        if d == 1:
            stop_hit = l[k] <= s.stop
            tgt_hit = h[k] >= target
            if stop_hit and tgt_hit:
                # same bar: conservative — the stop is nearer, take -1R
                # only when the open is already past the stop.
                if o[k] <= s.stop:
                    exit_price = o[k]
                else:
                    exit_price = s.stop
                exit_idx, reason = k, "stop"
                break
            if stop_hit:
                exit_price = s.stop if o[k] > s.stop else o[k]
                exit_idx, reason = k, "stop"
                break
            if tgt_hit:
                exit_price = target if o[k] < target else o[k]
                exit_idx, reason = k, "target"
                break
        else:
            stop_hit = h[k] >= s.stop
            tgt_hit = l[k] <= target
            if stop_hit and tgt_hit:
                if o[k] >= s.stop:
                    exit_price = o[k]
                else:
                    exit_price = s.stop
                exit_idx, reason = k, "stop"
                break
            if stop_hit:
                exit_price = s.stop if o[k] < s.stop else o[k]
                exit_idx, reason = k, "stop"
                break
            if tgt_hit:
                exit_price = target if o[k] > target else o[k]
                exit_idx, reason = k, "target"
                break
    r_gross = d * (exit_price - entry) / risk
    r_net = r_gross - COST_PER_OZ / risk
    return CandleOutcome(index=t, signal=s.signal, direction=d,
                         entry_index=t + 1, exit_index=exit_idx,
                         exit_reason=reason, r_net=float(r_net),
                         hold_bars=exit_idx - t, ctx=dict(s.ctx))


# ------------------------------------------------------------ context map
def _cell_stats(outcomes: list[CandleOutcome]) -> dict:
    rs = np.array([o.r_net for o in outcomes])
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) \
        and losses.sum() != 0 else float("inf")
    return {"n": int(len(rs)), "win_rate": float(np.mean(rs > 0) * 100.0),
            "avg_r": float(rs.mean()), "pf": pf}


def context_map(outcomes: list[CandleOutcome]) -> dict:
    """Per signal: overall + one row per context cell."""
    by_sig: dict[str, list[CandleOutcome]] = {}
    for oc in outcomes:
        by_sig.setdefault(oc.signal, []).append(oc)
    table: dict[str, dict] = {}
    for sig, ocs in by_sig.items():
        rows = {"overall": _cell_stats(ocs)}
        for dim in ("location", "trend_align", "atr_regime"):
            cells: dict[str, list[CandleOutcome]] = {}
            for oc in ocs:
                cells.setdefault(str(oc.ctx.get(dim, "?")), []).append(oc)
            for cell, cos in sorted(cells.items()):
                rows[f"{dim}={cell}"] = _cell_stats(cos)
        # killzone is boolean
        for kz in (True, False):
            cos = [oc for oc in ocs if oc.ctx.get("killzone") is kz]
            if cos:
                rows[f"killzone={kz}"] = _cell_stats(cos)
        # the interaction the owner asked for: location x trend_align
        pairs: dict[str, list[CandleOutcome]] = {}
        for oc in ocs:
            key = f"{oc.ctx.get('location')}|{oc.ctx.get('trend_align')}"
            pairs.setdefault(key, []).append(oc)
        for key, cos in sorted(pairs.items()):
            rows[f"loc|align={key}"] = _cell_stats(cos)
        table[sig] = rows
    return table


def run_experiment(df: pd.DataFrame | None = None,
                   peers: dict | None = None) -> dict:
    """Full Phase-B pipeline: read -> featurize -> scan -> simulate."""
    if df is None:
        df = load_gold_h1()
    if peers is None:
        from ..backtest.data import load_peers
        peers = load_peers()
    states = read_and_featurize(df, peers)
    setups = scan(df, states)
    outcomes = [oc for s in setups if (oc := simulate(df, s)) is not None]
    return {"setups": len(setups), "outcomes": outcomes,
            "map": context_map(outcomes)}
