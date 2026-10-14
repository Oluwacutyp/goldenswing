"""Placement review: the same market objects, tested in the roles
practitioners actually use them in — plus the INVERSE check.

The owner's doctrine: everything has a place — our kills tested each
item in ONE role (usually: standalone entry trigger, all contexts).
The inverse doctrine: for each killed directional signal, test whether
the OPPOSITE direction carries stable signal.  A consistently-wrong
bull signal is a bear signal — but "real" means it survives both window
halves AND parameter variants, not just a sign flip on the in-sample
number.  Asymmetry is the null hypothesis (bearish pins at supply
worked while bull pins didn't; bear_hid backs shorts while bear_reg
never positive), not the exception.

Three probes, one theme: **the failure is the signal**.

  T1 failed-pattern continuation — a double top / H&S whose breakdown
     fails back through the neckline (the Turtle Soup trade): the
     pattern marked the liquidity, the failure is the entry.
     INVERSE: trade WITH the pattern's bias at the reclaim (fade the
     failure).
  T2 pin-wick violation — a pin bar whose extreme is closed through:
     the rejection failed, trade the violation (stop-hunt continuation).
     INVERSE: trade with the pin's rejection at the violation bar.
  T3 failed zone-retest — the JeaFx resting retest, but traded when the
     zone FAILS (touch, then close through against the thesis): the ICT
     breaker.  INVERSE: fade the breaker, trade back with the original
     zone thesis.

Conventions (CANDLES.md Phase B, for comparability): entry at the next
bar's open, structural stop + 0.25 ATR, fixed 2R target, 72-bar horizon,
$0.45/oz cost, stop wins ties, adverse opens on gaps.  Everything is
causal: no bar beyond the signal bar is consulted for the decision.

Placement verdicts (not pardons): nothing is un-killed without measured
evidence.  A probe that fails keeps its kill, with the tested placement
added to the scope note.  Inverse claims face the same honesty bar:
both halves positive AND stable across parameter variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.data import load_gold_h1
from ..backtest.harness import COST_PER_OZ
from ..features.config import FeatureConfig
from ..features.engine import read_and_featurize
from ..features.patterns import compute as patterns_compute
from .candles import is_doji, pin_bar
from .retest_entry import ENTRY_EDGE, ORDER_EXPIRY_BARS, scan_setups

#: Bars after the trigger event inside which the failure must confirm.
#: Swept as a robustness parameter (8/12/16).
FAIL_BARS = 12
#: Minimum consecutive dojis to define a compression cluster (T4).
DOJI_CLUSTER_MIN = 2
#: An expansion bar must span this much ATR to count (T4) — a drift
#: through the cluster edge is not an expansion.
EXPANSION_MIN_RANGE_ATR = 0.75
#: Fixed target, in R (candle-map comparability).
FIXED_TP_R = 2.0
#: Stop buffer beyond the structural extreme.
STOP_BUFFER_ATR = 0.25
#: Max hold after entry, bars.
TRADE_HORIZON = 72


# ------------------------------------------------------------------ outcome
@dataclass
class PlacementOutcome:
    probe: str
    signal_index: int
    direction: int
    entry: float
    stop: float
    r_net: float
    exit_reason: str
    hold_bars: int


def _simulate(df: pd.DataFrame, signal_index: int, direction: int,
              stop: float, probe: str) -> PlacementOutcome | None:
    """Honest bar walk, v1 conventions (mirrors candles.simulate).

    Entry at the next bar's open; the walk starts on the entry bar, so a
    same-bar stop is possible and counted.  Stop wins ties; gap opens
    fill at the adverse open; flat $0.45/oz cost in R; 72-bar horizon.
    """
    t = signal_index
    n = len(df)
    if t + 1 >= n:
        return None
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    d = direction
    entry = o[t + 1]
    risk = (entry - stop) * d
    if risk <= 0:
        return None
    target = entry + d * FIXED_TP_R * risk
    last = min(t + TRADE_HORIZON, n - 1)
    exit_price, exit_idx, reason = cl[last], last, "timeout"
    for k in range(t + 1, last + 1):
        if d == 1:
            stop_hit = low[k] <= stop
            tgt_hit = h[k] >= target
            if stop_hit and tgt_hit:
                if o[k] <= stop:
                    exit_price = o[k]
                else:
                    exit_price = stop
                exit_idx, reason = k, "stop"
                break
            if stop_hit:
                exit_price = stop if o[k] > stop else o[k]
                exit_idx, reason = k, "stop"
                break
            if tgt_hit:
                exit_price = target if o[k] < target else o[k]
                exit_idx, reason = k, "target"
                break
        else:
            stop_hit = h[k] >= stop
            tgt_hit = low[k] <= target
            if stop_hit and tgt_hit:
                if o[k] >= stop:
                    exit_price = o[k]
                else:
                    exit_price = stop
                exit_idx, reason = k, "stop"
                break
            if stop_hit:
                exit_price = stop if o[k] < stop else o[k]
                exit_idx, reason = k, "stop"
                break
            if tgt_hit:
                exit_price = target if o[k] > target else o[k]
                exit_idx, reason = k, "target"
                break
    r_gross = d * (exit_price - entry) / risk
    r_net = r_gross - COST_PER_OZ / risk
    return PlacementOutcome(
        probe=probe, signal_index=t, direction=d, entry=entry, stop=stop,
        r_net=float(r_net), exit_reason=reason,
        hold_bars=exit_idx - (t + 1))


# ------------------------------------------------------- T1: failed pattern
def _t1_signals(df: pd.DataFrame, states,
                fail_bars: int = FAIL_BARS
                ) -> list[tuple[int, int, float, float]]:
    """Scan for failed pattern breakdowns/breakouts.

    Returns (signal_bar, bias_direction, stop_bias, stop_inverse).
    A bearish pattern (double_top / head_shoulders, direction -1) that
    closes below its neckline is the textbook short trigger; when price
    reclaims back above the neckline within ``fail_bars``, the breakdown
    failed — the trapped shorts fuel the reversal, so the failure is
    traded LONG (bias).  The INVERSE fades the failure: SHORT at the
    reclaim, with the pattern's original bias.  Mirror for bullish
    patterns.  Signals that hit the pattern's own invalidation before
    the reclaim are skipped (the pattern is dead, not failed).
    """
    cl = df["close"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    atr = np.array([s.atr if s.atr and s.atr > 0 else np.nan
                    for s in states], dtype=float)
    pat_states = patterns_compute(df, states, FeatureConfig())
    out: list[tuple[int, int, float, float]] = []
    n = len(df)
    for t in range(n):
        ev = pat_states[t].event
        if ev is None or ev.status != "broken":
            continue
        d_pat = ev.direction  # -1: breakdown short pattern | +1: breakout long
        if d_pat not in (-1, 1):
            continue
        last = min(t + fail_bars, n - 2)  # need bar j+1 for entry
        for j in range(t + 1, last + 1):
            # Pattern died on its own invalidation: not a failure, skip.
            if d_pat == -1 and cl[j] > ev.invalidation:
                break
            if d_pat == 1 and cl[j] < ev.invalidation:
                break
            reclaimed = (d_pat == -1 and cl[j] > ev.neckline) or \
                        (d_pat == 1 and cl[j] < ev.neckline)
            if not reclaimed:
                continue
            d = -d_pat  # bias: trade the failure, opposite the pattern
            a = atr[j]
            if not np.isfinite(a):
                break
            lo_min = float(np.min(low[t + 1: j + 1]))
            hi_max = float(np.max(h[t + 1: j + 1]))
            if d == 1:
                stop_bias = lo_min - STOP_BUFFER_ATR * a
                stop_inv = hi_max + STOP_BUFFER_ATR * a
            else:
                stop_bias = hi_max + STOP_BUFFER_ATR * a
                stop_inv = lo_min - STOP_BUFFER_ATR * a
            out.append((j, d, stop_bias, stop_inv))
            break  # first failure per pattern only
    return out


# --------------------------------------------------- T2: pin-wick violation
def _t2_signals(df: pd.DataFrame, states,
                fail_bars: int = FAIL_BARS
                ) -> list[tuple[int, int, float, float]]:
    """Scan for pin bars whose extreme is closed through.

    Returns (signal_bar, bias_direction, stop_bias, stop_inverse).
    A bearish pin (upper-wick rejection) marks sellers at the high; when
    a later bar CLOSES above the pin's high, the rejection failed — the
    pin-traders' stops are fuel, so the violation is traded LONG
    (stop-hunt continuation, bias).  The INVERSE trades with the pin's
    rejection at the violation bar.  Mirror for bullish pins.  First
    violation per pin only.
    """
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    atr = np.array([s.atr if s.atr and s.atr > 0 else np.nan
                    for s in states], dtype=float)
    out: list[tuple[int, int, float, float]] = []
    n = len(df)
    for i in range(n):
        r = pin_bar(o, h, low, cl, i)
        if r == 0:
            continue
        last = min(i + fail_bars, n - 2)
        for j in range(i + 1, last + 1):
            a = atr[j]
            if not np.isfinite(a):
                break
            if r == -1 and cl[j] > h[i]:
                # Bias long (violation); inverse short (pin's rejection).
                out.append((j, 1,
                            float(low[i]) - STOP_BUFFER_ATR * a,
                            float(h[i]) + STOP_BUFFER_ATR * a))
                break
            if r == 1 and cl[j] < low[i]:
                out.append((j, -1,
                            float(h[i]) + STOP_BUFFER_ATR * a,
                            float(low[i]) - STOP_BUFFER_ATR * a))
                break
    return out


# ------------------------------------------------- T3: failed zone retest
def _t3_signals(df: pd.DataFrame, states,
                fail_bars: int = FAIL_BARS
                ) -> list[tuple[int, int, float, float]]:
    """Scan JeaFx-style retest setups for zone failure (the breaker).

    Returns (signal_bar, bias_direction, stop_bias, stop_inverse).
    Reuses the retest experiment's own setup scan (edge mode): a setup is
    a long/short thesis at a supply/demand zone after sweep->zone->BoS.
    The placement tested here is the thesis's INVALIDATION: price
    touches the zone (the retest happens), then closes THROUGH it against
    the thesis within the order window — the zone is consumed, trade the
    continuation through it (ICT breaker, bias).  The INVERSE fades the
    breaker: trade back with the original zone thesis at the violation
    bar.  Touch-then-violate only; zones price never reaches are not
    failures.  NOTE: ``states`` must be featurized (read_and_featurize),
    plain reader states carry no supply/demand zones.
    """
    low = df["low"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    atr = np.array([s.atr if s.atr and s.atr > 0 else np.nan
                    for s in states], dtype=float)
    out: list[tuple[int, int, float, float]] = []
    n = len(df)
    for s in scan_setups(df, states, entry=ENTRY_EDGE):
        t, d, zb, zt = s.index, s.direction, s.zone_bottom, s.zone_top
        window_end = min(t + ORDER_EXPIRY_BARS, n - 2)
        # Phase 1: the retest must actually happen (wick into the zone).
        touch = None
        for j1 in range(t + 1, window_end + 1):
            if d == 1 and low[j1] <= zt:
                touch = j1
                break
            if d == -1 and h[j1] >= zb:
                touch = j1
                break
        if touch is None:
            continue
        # Phase 2: close through the zone against the thesis = failure.
        for j2 in range(touch + 1, window_end + 1):
            a = atr[j2]
            if not np.isfinite(a):
                break
            if d == 1 and cl[j2] < zb:
                # Bias: short the broken demand. Inverse: long the thesis.
                out.append((j2, -1,
                            float(zt) + STOP_BUFFER_ATR * a,
                            float(zb) - STOP_BUFFER_ATR * a))
                break
            if d == -1 and cl[j2] > zt:
                out.append((j2, 1,
                            float(zb) - STOP_BUFFER_ATR * a,
                            float(zt) + STOP_BUFFER_ATR * a))
                break
    return out


# ------------------------------------------------- T4: doji compression
def _t4_signals(df: pd.DataFrame, states,
                fail_bars: int = FAIL_BARS
                ) -> list[tuple[int, int, float, float]]:
    """Scan doji-cluster compression for expansion breaks.

    Returns (signal_bar, bias_direction, stop_bias, stop_inverse).
    Practitioner placement for the doji (ICT: "indecision marks zones"):
    2+ consecutive dojis define a compression range; the first bar that
    CLOSES beyond the cluster extreme with real range (>= 0.75 ATR) is
    the expansion — trade the break direction (bias).  The INVERSE fades
    the expansion.  One signal per cluster; clusters do not overlap.
    """
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    atr = np.array([s.atr if s.atr and s.atr > 0 else np.nan
                    for s in states], dtype=float)
    out: list[tuple[int, int, float, float]] = []
    n = len(df)
    i = 0
    while i < n - 2:
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or not is_doji(o, h, low, cl, a, i):
            i += 1
            continue
        j = i
        while j + 1 < n:
            aj = atr[j + 1]
            if not np.isfinite(aj) or aj <= 0:
                break
            if not is_doji(o, h, low, cl, aj, j + 1):
                break
            j += 1
        if j - i + 1 < DOJI_CLUSTER_MIN:
            i = j + 1
            continue
        c_hi = float(np.max(h[i:j + 1]))
        c_lo = float(np.min(low[i:j + 1]))
        last = min(j + fail_bars, n - 2)
        for k in range(j + 1, last + 1):
            ak = atr[k]
            if not np.isfinite(ak) or ak <= 0:
                break
            if h[k] - low[k] < EXPANSION_MIN_RANGE_ATR * ak:
                continue  # drift through the edge is not expansion
            if cl[k] > c_hi:
                out.append((k, 1,
                            c_lo - STOP_BUFFER_ATR * ak,
                            c_hi + STOP_BUFFER_ATR * ak))
                break
            if cl[k] < c_lo:
                out.append((k, -1,
                            c_hi + STOP_BUFFER_ATR * ak,
                            c_lo - STOP_BUFFER_ATR * ak))
                break
        i = j + 1
    return out


# ------------------------------------------- wedge breakout-direction drift
def wedge_breakout_drift(df: pd.DataFrame, states) -> dict:
    """Placement probe for #30: trade the breakout DIRECTION, not the
    prior trend.

    The #30 kill measured drift in the prior-trend direction (-0.250
    ATR vs +0.252 base).  Practitioner placement: a wedge is a
    compression/liquidity structure — the resolution direction is the
    information.  Returns drift in the breakout direction for every
    detected breakout, split by in-trend/counter kind.  Drift diagnostic
    only (n is thin); a book follows only if drift justifies it.
    """
    from .trendline_claims import (WEDGE_DRIFT_BARS, build_lines,
                                   detect_wedges, scan_wedge_breakouts)
    o = df["open"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    n = len(df)
    line_states = build_lines(df, states, FeatureConfig(), mode="shadow")
    wedges = detect_wedges(line_states, states, df)
    brks = scan_wedge_breakouts(wedges, df, states)
    recs: list[dict] = []
    for b in brks:
        k, bdir, atr = b["break_bar"], b["bdir"], b["atr"]
        kk = min(k + WEDGE_DRIFT_BARS, n - 1)
        if k + 1 >= n or atr <= 0:
            continue
        drift = bdir * (cl[kk] - o[k + 1]) / atr
        recs.append({"kind": b["kind"], "drift_r": float(drift)})
    def _agg(rs: list[float]) -> dict:
        a = np.array(rs)
        if len(a) == 0:
            return {"n": 0, "mean_drift": 0.0, "win_rate": 0.0}
        return {"n": len(a), "mean_drift": float(a.mean()),
                "win_rate": float(np.mean(a > 0) * 100.0)}
    all_r = [r["drift_r"] for r in recs]
    in_r = [r["drift_r"] for r in recs if r["kind"] == "in_trend"]
    co_r = [r["drift_r"] for r in recs if r["kind"] == "counter"]
    return {"n_wedges": len(wedges), "all": _agg(all_r),
            "in_trend": _agg(in_r), "counter": _agg(co_r)}


# -------------------------------------------------------------------- run
@dataclass
class ProbeResult:
    name: str
    n_signals: int
    n_trades: int
    win_rate: float
    avg_r: float
    pf: float
    h1_avg_r: float
    h2_avg_r: float
    notes: str = ""


def _stats(outcomes: list[PlacementOutcome], name: str,
           notes: str = "") -> ProbeResult:
    n = len(outcomes)
    if n == 0:
        return ProbeResult(name, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, notes)
    rs = np.array([o.r_net for o in outcomes])
    wins = rs[rs > 0].sum()
    losses = -rs[rs < 0].sum()
    mid = n // 2
    return ProbeResult(
        name=name, n_signals=n, n_trades=n,
        win_rate=float((rs > 0).mean()),
        avg_r=float(rs.mean()),
        pf=float(wins / losses) if losses > 0 else float("inf"),
        h1_avg_r=float(rs[:mid].mean()) if mid else 0.0,
        h2_avg_r=float(rs[mid:].mean()) if n - mid else 0.0,
        notes=notes)


@dataclass
class PlacementReport:
    bars: int
    start: str
    end: str
    fail_bars: int
    probes: list[ProbeResult] = field(default_factory=list)


def _dedupe(signals: list[tuple[int, int, float, float]]
          ) -> list[tuple[int, int, float, float]]:
    """One signal per (bar, direction): a trader takes one position.

    Multiple setups can flag the same bar (e.g. several zones failing
    together); counting them all would multiply-count a single trade.
    Keeps the first signal per key.
    """
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int, float, float]] = []
    for sig in signals:
        key = (sig[0], sig[1])
        if key in seen:
            continue
        seen.add(key)
        out.append(sig)
    return out


def run_placement(df: pd.DataFrame | None = None,
                  fail_bars: int = FAIL_BARS) -> PlacementReport:
    """Run the three placement probes (+ inverses) on 2y GC=F H1.

    States are featurized (read_and_featurize): T3's setup scan needs the
    supply/demand zones, which plain reader states do not carry.
    """
    if df is None:
        df = load_gold_h1()
    states = read_and_featurize(df)
    scanners = [
        ("T1_failed_pattern", _t1_signals),
        ("T2_pin_violation", _t2_signals),
        ("T3_zone_breaker", _t3_signals),
        ("T4_doji_expansion", _t4_signals),
    ]
    report = PlacementReport(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        fail_bars=fail_bars)
    for name, scanner in scanners:
        for variant, vname in (("bias", ""), ("inverse", "_inv")):
            outcomes: list[PlacementOutcome] = []
            for sig_bar, bias_dir, stop_bias, stop_inv in _dedupe(
                    scanner(df, states, fail_bars=fail_bars)):
                if variant == "bias":
                    d, stop = bias_dir, stop_bias
                else:
                    d, stop = -bias_dir, stop_inv
                oc = _simulate(df, sig_bar, d, stop, name + vname)
                if oc is not None:
                    outcomes.append(oc)
            report.probes.append(_stats(outcomes, name + vname))
    return report
