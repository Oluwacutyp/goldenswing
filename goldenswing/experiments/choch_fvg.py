"""CHoCH -> FVG-midline -> opposing-OB sleeve (user-contributed, LuxAlgo).

The user's ruleset, in their words:
  1. Wait for a bullish CHoCH (bearish for shorts).
  2. Enter at the MIDLINE of a bullish FVG (the 50% level of the gap).
  3. Exit at the next opposing ("red") order block, scaled by risk:reward.

This is a shadow experiment: it never touches the detector.  Everything
here is causal — FVGs, OBs and CHoCH events are known at the signal bar's
close or recomputed from bars <= the decision bar with the reader's own
mitigation rules.

Mechanics:
- Setup at close of bar t: the reader's last structure event is a CHoCH
  in direction d, age <= CHOCH_FRESH_BARS, and there is an active FVG in
  direction d created around the CHoCH leg whose midline is untraded
  (no touch since creation) and sits on the pullback side of price.
- A resting limit is placed at the midline, live from bar t+1, expiring
  after ORDER_EXPIRY_BARS.  Fill = first bar whose wick touches the
  midline; a gap through it fills at the (adverse) open.  If the FVG is
  mitigated before the fill, the order is cancelled.  If a bar trades
  through BOTH the midline and the stop, the fill is taken and the stop
  exits it the same bar (conservative: -1R).
- Stop = the swing level the CHoCH broke, buffered 0.25x ATR (repo
  convention).  Skipped when the stop is < 0.25 or > 4 ATR away (noise /
  too wide to be a swing trade).
- Variant A (the user's exit): target = nearest edge of the nearest
  ACTIVE opposing OB at the fill bar that leaves >= 1R of room; the OB
  must have existed at the fill bar.  No qualifying OB -> no trade.
- Variant B (control): same entry/stop, fixed 2R target — isolates
  whether the OB target adds anything over a plain R:R exit.
- Trade walk mirrors the v1 harness conventions exactly: stop wins ties,
  adverse opens on gaps, $0.45/oz cost, 72-bar time stop from the fill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.data import load_gold_h1
from ..backtest.harness import (COST_PER_OZ, Outcome, book_metrics,
                                build_book, slice_stats)
from ..golden.detector import GoldenSignal
from ..reader.reader import MarketReader
from ..reader.state import MarketState, OrderBlock

# ---------------------------------------------------------------- constants
#: CHoCH must be this fresh at the signal bar (repo confirm-window scale).
CHOCH_FRESH_BARS = 24
#: Resting midline limit lives this many bars after the signal bar.
ORDER_EXPIRY_BARS = 24
#: The FVG must belong to the CHoCH leg: created within this many bars
#: before the CHoCH bar (the repo's 8-bar confirm window scale).
FVG_PAIR_BACK_BARS = 8
#: Stop buffer beyond the broken swing (Phase-2 convention).
STOP_BUFFER_ATR = 0.25
#: Stop-distance sanity band, in ATR.
MIN_RISK_ATR = 0.25
MAX_RISK_ATR = 4.0
#: Variant A: the opposing OB must leave at least this much room.
OB_TARGET_MIN_R = 1.0
#: Variant B control: fixed target in R.
FIXED_TP_R = 2.0
#: Max hold after the fill, bars.
TRADE_HORIZON = 72


# ------------------------------------------------------------------ events
@dataclass
class SleeveSetup:
    """A resting-limit setup, decided at the close of bar ``index``."""

    index: int
    direction: int            # +1 long | -1 short
    mid: float                # FVG midline = limit price
    stop: float               # buffered structural stop
    risk: float               # |mid - stop|, in price
    atr: float
    fvg_top: float
    fvg_bottom: float
    fvg_created_at: int
    choch_index: int
    choch_level: float


@dataclass
class SleeveFill:
    setup: SleeveSetup
    fill_index: int
    entry: float
    same_bar_stop: bool = False


# ------------------------------------------------------------- pure helpers
def _fvg_alive(df_low: np.ndarray, df_high: np.ndarray, top: float,
               bottom: float, direction: int, created_at: int,
               upto: int) -> bool:
    """The reader's FVG mitigation rule, recomputed from bars.

    Only bars (created_at, upto] can mitigate — mirrors reader.py, which
    skips levels with ``created_at >= t`` in the mitigation pass.
    """
    for j in range(created_at + 1, upto + 1):
        if direction == 1 and df_low[j] <= bottom:
            return False
        if direction == -1 and df_high[j] >= top:
            return False
    return True


def _obs_active_at(df_close: np.ndarray,
                   states: list[MarketState], upto: int) -> list[OrderBlock]:
    """Every OB that is live at bar ``upto``, with the reader's
    close-through mitigation recomputed from bars.

    The per-state snapshot is capped at the 8 newest OBs, so the full
    live set is reconstructed from the union of snapshots — deduplicated
    on identity.  Causal: only bars <= ``upto`` are consulted.
    """
    seen: dict[tuple, OrderBlock] = {}
    for s in states[: upto + 1]:
        for ob in s.active_obs:
            key = (ob.created_at, ob.candle_index, ob.top, ob.bottom,
                   ob.direction)
            seen.setdefault(key, ob)
    out: list[OrderBlock] = []
    for ob in seen.values():
        if ob.created_at > upto:
            continue
        dead = False
        for j in range(ob.created_at + 1, upto + 1):
            if ob.direction == 1 and df_close[j] < ob.bottom:
                dead = True
                break
            if ob.direction == -1 and df_close[j] > ob.top:
                dead = True
                break
        if not dead:
            out.append(ob)
    return out


def _fill_price(direction: int, mid: float, open_: float) -> float:
    """Resting-limit fill: midline, or the adverse open on a gap through."""
    if direction == 1:
        return mid if open_ > mid else open_
    return mid if open_ < mid else open_


# ------------------------------------------------------------------- scan
def scan_setups(df: pd.DataFrame,
                states: list[MarketState]) -> list[SleeveSetup]:
    """Find resting-limit setups.  Pure function of bars <= each signal bar.

    Every field of a returned setup is knowable at the close of
    ``setup.index``: the CHoCH, the FVG, the midline-touch history and the
    stop all come from completed bars.
    """
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    setups: list[SleeveSetup] = []
    for t, st in enumerate(states):
        ev = st.last_structure_event
        if ev is None or ev.kind != "CHOCH":
            continue
        if st.structure_event_age is None or \
                st.structure_event_age > CHOCH_FRESH_BARS:
            continue
        d = ev.direction
        atr = st.atr
        if not (atr and atr > 0 and np.isfinite(atr)):
            continue
        best = None
        for g in st.active_fvgs:
            if g.direction != d:
                continue
            if g.created_at > t or g.created_at < ev.index - FVG_PAIR_BACK_BARS:
                continue
            mid = g.midpoint
            # The limit must be a pullback: price on the favourable side.
            if d == 1 and not close[t] > mid:
                continue
            if d == -1 and not close[t] < mid:
                continue
            # Midline untraded since the FVG was created.
            seg = low[g.created_at: t + 1] if d == 1 \
                else high[g.created_at: t + 1]
            if d == 1 and bool((seg <= mid).any()):
                continue
            if d == -1 and bool((seg >= mid).any()):
                continue
            if best is None or g.created_at > best.created_at:
                best = g
        if best is None:
            continue
        mid = best.midpoint
        stop = ev.level - d * STOP_BUFFER_ATR * atr
        risk = (mid - stop) * d
        if risk < MIN_RISK_ATR * atr or risk > MAX_RISK_ATR * atr:
            continue
        setups.append(SleeveSetup(
            index=t, direction=d, mid=mid, stop=stop, risk=risk, atr=atr,
            fvg_top=best.top, fvg_bottom=best.bottom,
            fvg_created_at=best.created_at,
            choch_index=ev.index, choch_level=ev.level))
    return setups


# ------------------------------------------------------------------- fill
def find_fill(df: pd.DataFrame, setup: SleeveSetup) -> SleeveFill | None:
    """Work the resting limit from bar t+1.  Returns None when the order
    expires unfilled or the FVG dies first (cancelled, not a trade)."""
    t = setup.index
    d = setup.direction
    mid, stop = setup.mid, setup.stop
    n = len(df)
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    last = min(t + ORDER_EXPIRY_BARS, n - 1)
    for k in range(t + 1, last + 1):
        # FVG death cancels the order BEFORE any fill logic (a dead level
        # is not a level; the reader's rule, recomputed from bars <= k).
        if not _fvg_alive(low, high, setup.fvg_top, setup.fvg_bottom, d,
                          setup.fvg_created_at, k):
            return None
        entry = _fill_price(d, mid, open_[k])
        if d == 1:
            if low[k] <= stop:          # traded through fill AND stop
                return SleeveFill(setup, k, entry, same_bar_stop=True)
            if low[k] <= mid:
                return SleeveFill(setup, k, entry)
        else:
            if high[k] >= stop:
                return SleeveFill(setup, k, entry, same_bar_stop=True)
            if high[k] >= mid:
                return SleeveFill(setup, k, entry)
    return None


# ------------------------------------------------------------- variant A
def select_ob_target(df: pd.DataFrame, states: list[MarketState],
                     fill: SleeveFill) -> float | None:
    """Nearest edge of the nearest active opposing OB with >= 1R of room.

    The OB must have existed at the fill bar (created_at <= fill) and be
    live under the reader's close-through rule.  Returns None when no OB
    qualifies — the trade is skipped, not forced.
    """
    d = fill.setup.direction
    entry, risk = fill.entry, fill.setup.risk
    close = df["close"].to_numpy(dtype=float)
    best: float | None = None
    for ob in _obs_active_at(close, states, fill.fill_index):
        if ob.direction == d:           # opposing ("red" for longs) only
            continue
        if d == 1:
            if ob.bottom < entry + OB_TARGET_MIN_R * risk:
                continue
            if best is None or ob.bottom < best:
                best = ob.bottom
        else:
            if ob.top > entry - OB_TARGET_MIN_R * risk:
                continue
            if best is None or ob.top > best:
                best = ob.top
    return best


# --------------------------------------------------------------- simulate
def _to_signal(fill: SleeveFill, target: float) -> GoldenSignal:
    s = fill.setup
    return GoldenSignal(
        index=fill.fill_index, timestamp=None, direction=s.direction,
        score=100.0, kind="sleeve_choch_fvg", entry_ref=fill.entry,
        invalidation=s.stop, targets=[target], atr=s.atr)


def simulate_trade(df: pd.DataFrame, fill: SleeveFill, target: float,
                   cost: float = COST_PER_OZ) -> Outcome:
    """Trade walk with the v1 harness conventions: stop wins ties, adverse
    opens on gaps, flat cost in R, 72-bar time stop from the fill bar."""
    s = fill.setup
    d = s.direction
    entry, stop, risk = fill.entry, s.stop, s.risk
    n = len(df)
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    sig = _to_signal(fill, target)
    if fill.same_bar_stop:
        r_net = -1.0 - cost / risk
        return Outcome(signal=sig, entry_index=fill.fill_index,
                       exit_index=fill.fill_index, exit_reason="stop",
                       r_net=float(r_net), hold_bars=0)

    f = fill.fill_index
    last = min(f + TRADE_HORIZON, n - 1)
    exit_price, exit_idx, reason = close[last], last, "timeout"
    for k in range(f + 1, last + 1):
        if d == 1:
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
    r_gross = d * (exit_price - entry) / risk
    r_net = r_gross - cost / risk
    return Outcome(signal=sig, entry_index=f, exit_index=exit_idx,
                   exit_reason=reason, r_net=float(r_net),
                   hold_bars=exit_idx - f)


# -------------------------------------------------------------------- run
@dataclass
class SleeveResult:
    bars: int
    start: str
    end: str
    years: float
    n_setups: int
    n_fills: int
    n_cancelled: int
    n_skipped_no_ob: int
    variant_a: dict = field(default_factory=dict)   # OB target
    variant_b: dict = field(default_factory=dict)   # fixed 2R
    shadow_a: dict = field(default_factory=dict)
    shadow_b: dict = field(default_factory=dict)


def run_sleeve(refresh: bool = False) -> SleeveResult:
    """Full experiment: scan, fill, both variants, books + shadow."""
    df = load_gold_h1(refresh=refresh)
    states = MarketReader().read(df)
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25

    setups = scan_setups(df, states)
    fills: list[SleeveFill] = []
    n_cancelled = 0
    for s in setups:
        fl = find_fill(df, s)
        if fl is None:
            n_cancelled += 1
        else:
            fills.append(fl)

    outcomes_a: list[Outcome] = []
    n_skipped_no_ob = 0
    for fl in fills:
        target = select_ob_target(df, states, fl)
        if target is None:
            n_skipped_no_ob += 1
            continue
        outcomes_a.append(simulate_trade(df, fl, target))

    outcomes_b: list[Outcome] = []
    for fl in fills:
        d = fl.setup.direction
        outcomes_b.append(
            simulate_trade(df, fl, fl.entry + d * FIXED_TP_R * fl.setup.risk))

    return SleeveResult(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        years=years, n_setups=len(setups), n_fills=len(fills),
        n_cancelled=n_cancelled, n_skipped_no_ob=n_skipped_no_ob,
        variant_a=book_metrics(build_book(outcomes_a, 0.0), years),
        variant_b=book_metrics(build_book(outcomes_b, 0.0), years),
        shadow_a=slice_stats(outcomes_a),
        shadow_b=slice_stats(outcomes_b),
    )
