"""Bollinger trend states, measured.

``goldenswing/features/bollinger.py`` computes three states the desk
actually uses — squeeze, walk-the-band, band-rejection wick — and the
detector notes them as factors, but they were never MEASURED.  This
experiment asks the three practitioner questions (Carter's squeeze,
band-walk trend riding, band-rejection reversals) on 2y GC=F H1.

All knobs below are pre-committed, not fitted.

Q1 SQUEEZE -> EXPANSION
  - A squeeze episode = a maximal run of squeeze==True with length >= 5.
  - Episode ends at bar e (last squeeze bar).  Expansion window = e+1..e+24.
  - (a) Magnitude: |close[e+24]-close[e]| / atr[e] vs a seeded random-bar
    baseline.  Is there actually an expansion?
  - (b) Breakout direction: the first bar b in the window whose close
    breaks the 20-bar range (max high / min low of e-19..e).  Drift over
    the next 24 bars in the breakout direction, in ATR.
  - (c) Context direction: does the HTF bias (reader daily_bias, causal)
    predict the 24-bar sign?  Hit rate vs 50%.
  - (d) The practitioner's real question — is it worth trading?  A small
    breakout book: enter at the open after the break bar, stop at the
    violated range extreme +/- 0.25 ATR, fixed 2R target, $0.45/oz,
    stop wins ties, 72-bar horizon.  Reported with window halves.

Q2 WALK-THE-BAND as continuation (entry context, not a standalone trigger)
  - The anchor signals (Combo A + aoc, score >= 60) simulated standalone;
    split by whether states[sig].features.bollinger.walk == sig.direction.
  - Shadow avg R / PF / win% for walk-with vs walk-without.

Q3 BAND-REJECTION WICK as reversal
  - Event at t: rejection != 0 (wick poked through a band, close back
    inside).  Drift over 8/12/24 bars in the rejection direction, in ATR,
    vs zero and vs the unconditional base rate.  Halves reported.

Everything is causal: every field used at bar t comes from bars <= t.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.ablation import P2_FLAGS
from ..backtest.data import load_gold_h1
from ..backtest.harness import (
    COST_PER_OZ,
    HORIZON,
    Outcome,
    book_metrics,
    build_book,
    simulate_signal_v2,
)
from ..backtest.validation import COMBO_A
from ..features.engine import read_and_featurize
from ..golden.detector import (
    DetectorConfig,
    GoldenSignal,
    GoldenSwingDetector,
)

# ------------------------------------------------------------------ knobs
SQUEEZE_MIN_BARS = 5      # min run length to count as a squeeze episode
EXPANSION_BARS = 24       # forward window for magnitude/direction
RANGE_LOOKBACK = 20       # breakout range bars before the episode end
STOP_BUFFER_ATR = 0.25    # repo convention
FIXED_TP_R = 2.0          # breakout book target
BASELINE_SEED = 7         # seeded random-bar baseline

ANCHOR_FAMS = COMBO_A | {"aoc"}


# ------------------------------------------------------------- pure utils
def _pf(rs: list[float]) -> float:
    wins = sum(r for r in rs if r > 0)
    losses = -sum(r for r in rs if r < 0)
    return wins / losses if losses > 0 else float("inf")


def _halves(n: int) -> tuple[slice, slice]:
    mid = n // 2
    return slice(0, mid), slice(mid, n)


# ------------------------------------------------------------- Q1 squeeze
@dataclass
class SqueezeEpisode:
    start: int
    end: int          # last squeeze bar
    length: int


def squeeze_episodes(squeeze: np.ndarray) -> list[SqueezeEpisode]:
    """Maximal runs of squeeze==True with length >= SQUEEZE_MIN_BARS."""
    out: list[SqueezeEpisode] = []
    n = len(squeeze)
    t = 0
    while t < n:
        if squeeze[t]:
            s = t
            while t < n and squeeze[t]:
                t += 1
            if t - s >= SQUEEZE_MIN_BARS:
                out.append(SqueezeEpisode(start=s, end=t - 1,
                                          length=t - s))
        else:
            t += 1
    return out


@dataclass
class SqueezeResult:
    n_episodes: int
    mag_mean_atr: float          # mean |24-bar move| / ATR
    mag_base_mean_atr: float     # seeded random-bar baseline
    n_breakouts: int
    brk_drift_mean_atr: float   # 24-bar drift in breakout direction
    brk_win_rate: float          # drift > 0 share
    ctx_n: int                  # episodes with nonzero daily bias
    ctx_hit_rate: float         # 24-bar sign == daily_bias share
    book: dict = field(default_factory=dict)
    book_h1: dict = field(default_factory=dict)
    book_h2: dict = field(default_factory=dict)
    breakout_n_h1: int = 0
    breakout_n_h2: int = 0


def _simulate_breakout(df: pd.DataFrame, entry_idx: int, direction: int,
                       stop: float) -> Outcome | None:
    """Fixed-2R breakout trade walk, repo conventions."""
    n = len(df)
    if entry_idx + 1 >= n:
        return None
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    entry = open_[entry_idx + 1]
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    target = entry + direction * FIXED_TP_R * risk
    sig = GoldenSignal(index=entry_idx, timestamp=None,
                       direction=direction, score=0.0,
                       kind="squeeze_breakout", entry_ref=entry,
                       invalidation=stop, targets=[target], atr=risk)
    last = min(entry_idx + 1 + HORIZON, n - 1)
    exit_price, exit_idx, reason = close[last], last, "timeout"
    for k in range(entry_idx + 2, last + 1):
        if direction == 1:
            if low[k] <= stop:
                exit_price = stop if open_[k] > stop else open_[k]
                exit_idx, reason = k, "stop"
                break
            if high[k] >= target:
                exit_price = target if open_[k] < target else open_[k]
                exit_idx, reason = k, "target"
                break
        else:
            if high[k] >= stop:
                exit_price = stop if open_[k] < stop else open_[k]
                exit_idx, reason = k, "stop"
                break
            if low[k] <= target:
                exit_price = target if open_[k] > target else open_[k]
                exit_idx, reason = k, "target"
                break
    r_net = direction * (exit_price - entry) / risk - COST_PER_OZ / risk
    return Outcome(signal=sig, entry_index=entry_idx + 1,
                   exit_index=exit_idx, exit_reason=reason,
                   r_net=float(r_net), hold_bars=exit_idx - entry_idx - 1)


def run_squeeze(df: pd.DataFrame, states) -> SqueezeResult:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    squeeze = np.array([bool(s.features.bollinger.squeeze)
                        for s in states])
    atr = np.array([s.atr for s in states], dtype=float)
    episodes = [ep for ep in squeeze_episodes(squeeze)
                if ep.end + EXPANSION_BARS < n
                and ep.end - RANGE_LOOKBACK + 1 >= 0
                and atr[ep.end] > 0]

    mags: list[float] = []
    drifts: list[float] = []
    ctx_hits = 0
    ctx_n = 0
    outcomes: list[Outcome] = []
    h1, h2 = _halves(n)
    brk_n_h1 = 0
    brk_n_h2 = 0

    for ep in episodes:
        e = ep.end
        a = atr[e]
        # (a) magnitude vs baseline
        mags.append(abs(close[e + EXPANSION_BARS] - close[e]) / a)
        # (b) breakout direction: first close outside the 20-bar range
        rng_hi = high[e - RANGE_LOOKBACK + 1: e + 1].max()
        rng_lo = low[e - RANGE_LOOKBACK + 1: e + 1].min()
        b, direction = -1, 0
        for k in range(1, EXPANSION_BARS + 1):
            up = close[e + k] > rng_hi
            dn = close[e + k] < rng_lo
            if up and dn:      # ambiguous bar: skip the breakout test
                b, direction = -1, 0
                break
            if up or dn:
                b, direction = e + k, (1 if up else -1)
                break
        if direction != 0 and b + EXPANSION_BARS < n and atr[b] > 0:
            drift = ((close[b + EXPANSION_BARS] - close[b]) * direction
                     / atr[b])
            drifts.append(drift)
            if b < h1.stop:
                brk_n_h1 += 1
            else:
                brk_n_h2 += 1
            # (d) the tradable version: stop at the violated extreme
            stop = (rng_lo - STOP_BUFFER_ATR * atr[b] if direction == 1
                    else rng_hi + STOP_BUFFER_ATR * atr[b])
            o = _simulate_breakout(df, b, direction, stop)
            if o is not None:
                outcomes.append(o)
        # (c) context direction: does the HTF bias predict the sign?
        bias = states[e].daily_bias
        if bias != 0:
            ctx_n += 1
            if np.sign(close[e + EXPANSION_BARS] - close[e]) == bias:
                ctx_hits += 1

    # seeded random-bar magnitude baseline
    rng = np.random.default_rng(BASELINE_SEED)
    valid = [t for t in range(RANGE_LOOKBACK, n - EXPANSION_BARS)
             if atr[t] > 0]
    base_idx = rng.choice(valid, size=min(2000, len(valid)), replace=False)
    base_mags = [abs(close[t + EXPANSION_BARS] - close[t]) / atr[t]
                 for t in base_idx]

    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    book = book_metrics(build_book(outcomes, 0.0), years)
    mid = n // 2
    o1 = [o for o in outcomes if o.entry_index < mid]
    o2 = [o for o in outcomes if o.entry_index >= mid]

    return SqueezeResult(
        n_episodes=len(episodes),
        mag_mean_atr=float(np.mean(mags)) if mags else float("nan"),
        mag_base_mean_atr=float(np.mean(base_mags)),
        n_breakouts=len(drifts),
        brk_drift_mean_atr=float(np.mean(drifts)) if drifts
        else float("nan"),
        brk_win_rate=float(np.mean([d > 0 for d in drifts])) if drifts
        else float("nan"),
        ctx_n=ctx_n,
        ctx_hit_rate=ctx_hits / ctx_n if ctx_n else float("nan"),
        book=book,
        book_h1=book_metrics(build_book(o1, 0.0), years / 2),
        book_h2=book_metrics(build_book(o2, 0.0), years / 2),
        breakout_n_h1=brk_n_h1,
        breakout_n_h2=brk_n_h2,
    )


# ------------------------------------------------------- Q2 walk-the-band
@dataclass
class WalkResult:
    n_with: int
    n_without: int
    avg_r_with: float
    avg_r_without: float
    pf_with: float
    pf_without: float
    win_with: float
    win_without: float
    avg_r_with_h1: float = float("nan")
    avg_r_with_h2: float = float("nan")


def run_walk(df: pd.DataFrame, states) -> WalkResult:
    """Anchor signals (shadow) split by walk==direction at the signal bar."""
    detector = GoldenSwingDetector(DetectorConfig(
        emit_floor=35.0, golden_threshold=60.0,
        feature_families=ANCHOR_FAMS, **P2_FLAGS))
    n = len(df)
    mid = n // 2
    with_r, without_r = [], []
    with_w, without_w = [], []
    with_h1, with_h2 = [], []
    for sig in detector.scan(states):
        o = simulate_signal_v2(df, sig)
        if o is None:
            continue
        walk = states[sig.index].features.bollinger.walk
        if walk == sig.direction and walk != 0:
            with_r.append(o.r_net)
            with_w.append(1.0 if o.r_net > 0 else 0.0)
            (with_h1 if sig.index < mid else with_h2).append(o.r_net)
        else:
            without_r.append(o.r_net)
            without_w.append(1.0 if o.r_net > 0 else 0.0)

    def avg(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return WalkResult(
        n_with=len(with_r), n_without=len(without_r),
        avg_r_with=avg(with_r), avg_r_without=avg(without_r),
        pf_with=_pf(with_r), pf_without=_pf(without_r),
        win_with=avg(with_w), win_without=avg(without_w),
        avg_r_with_h1=avg(with_h1), avg_r_with_h2=avg(with_h2),
    )


# ------------------------------------------------- Q3 band-rejection wick
@dataclass
class RejectionResult:
    n_events: int
    drift_8: float
    drift_12: float
    drift_24: float
    base_24: float
    drift_24_h1: float
    drift_24_h2: float
    win_24: float


def run_rejection(df: pd.DataFrame, states) -> RejectionResult:
    """Drift after a band-rejection wick, in ATR, in the rejection dir."""
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    atr = np.array([s.atr for s in states], dtype=float)
    d8, d12, d24 = [], [], []
    mid = n // 2
    h1_d, h2_d = [], []
    for t in range(n - 24):
        rej = states[t].features.bollinger.rejection
        if rej == 0 or atr[t] <= 0:
            continue
        d = (close[t + 8] - close[t]) * rej / atr[t]
        d8.append(d)
        d = (close[t + 12] - close[t]) * rej / atr[t]
        d12.append(d)
        d = (close[t + 24] - close[t]) * rej / atr[t]
        d24.append(d)
        (h1_d if t < mid else h2_d).append(d)
    # unconditional base rate: 24-bar drift magnitude-agnostic mean
    base = [abs(close[t + 24] - close[t]) / atr[t]
            for t in range(n - 24) if atr[t] > 0]

    def avg(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return RejectionResult(
        n_events=len(d24),
        drift_8=avg(d8), drift_12=avg(d12), drift_24=avg(d24),
        base_24=avg(base),
        drift_24_h1=avg(h1_d), drift_24_h2=avg(h2_d),
        win_24=float(np.mean([d > 0 for d in d24])) if d24
        else float("nan"),
    )


# ------------------------------------------------------------------- run
@dataclass
class BollingerReport:
    bars: int
    start: str
    end: str
    years: float
    squeeze: SqueezeResult
    walk: WalkResult
    rejection: RejectionResult


def run_bollinger(refresh: bool = False) -> BollingerReport:
    df = load_gold_h1(refresh=refresh)
    states = read_and_featurize(df)
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    return BollingerReport(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        years=years,
        squeeze=run_squeeze(df, states),
        walk=run_walk(df, states),
        rejection=run_rejection(df, states),
    )
