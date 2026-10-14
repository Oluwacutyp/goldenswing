"""JeaFx retest-entry sleeve (user-contributed, via video narration).

The ruleset, from the narration of the JeaFx short
("Simple and Profitable Forex Trading Strategy for EURUSD"):

  1. Context: price was in a downtrend (lower lows + lower highs), swept
     the highs, then traded down into a higher-timeframe demand zone.
  2. Confirmation: a strong push away from demand breaks structure up
     (a BoS rally).
  3. Entry: a resting limit at the demand zone's level on the retest.
  4. Stop: beneath the low (zone bottom, buffered).
  5. Target: the next highs (nearest liquidity pool ahead).
  Mirror for shorts.  JeaFx closes: "pretty much every trade he takes
  follows this same format" — i.e. this is his confirmation-entry
  playbook.

This is a shadow experiment: it never touches the detector.  Everything
is causal — sweeps, zones, BoS events and pools are known at the signal
bar's close or recomputed from bars <= the decision bar with the
reader's own mitigation rules.

Mechanics:
- Setup at close of bar t: the reader's last structure event is a BOS
  in direction d, age <= BOS_FRESH_BARS, and the broken level sits on
  the far side of a FRESH demand (d=+1) / supply (d=-1) zone the zone
  family marked BEFORE the break.  Before the break, price swept the
  opposite-side liquidity (sweep direction == -d, sweep bar before the
  BOS bar).  Daily bias agrees with d (the detector's zone convention).
- A resting limit is placed at the zone MIDLINE, live from bar t+1,
  expiring after ORDER_EXPIRY_BARS.  Fill = first bar whose wick
  touches the midline; a gap through it fills at the (adverse) open.
  The zone is mitigated when a CLOSE passes through its far edge: a
  bar that fills AND closes through the level is booked as filled then
  stopped (conservative: -1R).  A bar that trades through the fill and
  the stop is likewise filled then stopped.
- Stop = the zone's far edge buffered 0.25x ATR (repo convention).
  Skipped when the stop is < 0.25 or > 4 ATR away (noise / too wide).
- Variant A (the user's exit): target = nearest liquidity pool ahead
  at the fill bar that leaves >= 1R of room; the pool must have been
  known at the fill bar.  No qualifying pool -> no trade.
- Variant B (control): same entry/stop, fixed 2R target — isolates
  whether the "next highs" target adds anything over a plain R:R exit.
- Trade walk mirrors the v1 harness conventions exactly: stop wins
  ties, adverse opens on gaps, $0.45/oz cost, 72-bar time stop from
  the fill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.data import load_gold_h1
from ..backtest.harness import (COST_PER_OZ, Outcome, book_metrics,
                                build_book, slice_stats)
from ..features.engine import read_and_featurize
from ..golden.detector import GoldenSignal
from ..reader.state import LiquidityPool

# ---------------------------------------------------------------- constants
#: BoS must be this fresh at the signal bar (repo confirm-window scale).
BOS_FRESH_BARS = 24
#: The failed opposite-side sweep must sit inside this window before the
#: BoS confirmation.
SWEEP_BACK_BARS = 24
#: Resting zone limit lives this many bars after the signal bar.
ORDER_EXPIRY_BARS = 24
#: Stop buffer beyond the zone's far edge (Phase-2 convention).
STOP_BUFFER_ATR = 0.25
#: Stop-distance sanity band, in ATR.
MIN_RISK_ATR = 0.25
MAX_RISK_ATR = 4.0
#: Variant A: the target pool must leave at least this much room.
POOL_TARGET_MIN_R = 1.0
#: Variant B control: fixed target in R.
FIXED_TP_R = 2.0
#: Max hold after the fill, bars.
TRADE_HORIZON = 72
#: Entry level of the resting limit: "mid" = zone midline (the FVG-midline
#: experiment's convention), "edge" = the zone's first-touched edge (top
#: for longs / bottom for shorts — the literal reading of JeaFx's
#: "bought when price retested that level").
ENTRY_MID = "mid"
ENTRY_EDGE = "edge"


# ------------------------------------------------------------------ events
@dataclass
class RetestSetup:
    """A resting-limit setup, decided at the close of bar ``index``."""

    index: int
    direction: int            # +1 long | -1 short
    mid: float                # limit price (zone midline or first-touched
                              # edge, per entry_mode)
    stop: float               # buffered structural stop
    risk: float               # |mid - stop|, in price
    atr: float
    entry_mode: str           # "mid" | "edge"
    zone_top: float
    zone_bottom: float
    zone_created_at: int
    bos_index: int
    bos_level: float
    sweep_index: int


@dataclass
class RetestFill:
    setup: RetestSetup
    fill_index: int
    entry: float
    same_bar_stop: bool = False


# ------------------------------------------------------------- pure helpers
def _fill_price(direction: int, level: float, open_: float) -> float:
    """Resting-limit fill: the level, or the adverse open on a gap through."""
    if direction == 1:
        return level if open_ > level else open_
    return level if open_ < level else open_


# ------------------------------------------------------------------- scan
def scan_setups(df: pd.DataFrame, states,
                entry: str = ENTRY_MID) -> list[RetestSetup]:
    """Find resting-limit setups.  Pure function of bars <= each signal bar.

    Every field of a returned setup is knowable at the close of
    ``setup.index``: the sweep, the BoS, the zone (created before the
    break, fresh, daily-bias aligned) and the stop all come from
    completed bars.  ``entry`` selects the resting-limit level: the zone
    midline ("mid") or the first-touched edge ("edge").
    """
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    setups: list[RetestSetup] = []
    for t, st in enumerate(states):
        ev = st.last_structure_event
        if ev is None or ev.kind != "BOS":
            continue
        if st.structure_event_age is None or \
                st.structure_event_age > BOS_FRESH_BARS:
            continue
        d = ev.direction
        atr = st.atr
        if not (atr and atr > 0 and np.isfinite(atr)):
            continue
        # Context: the move swept the OPPOSITE side first (failed raid),
        # then reversed through the zone.
        sw = st.last_sweep
        if sw is None or sw.direction != -d:
            continue
        if st.sweep_age is None or st.sweep_age > SWEEP_BACK_BARS:
            continue
        if sw.index >= ev.index:
            continue
        # Daily-bias gate, the detector's zone convention.
        if st.daily_bias != d:
            continue
        fs = st.features
        if fs is None or fs.zones is None:
            continue
        z = fs.zones.nearest_demand if d == 1 else fs.zones.nearest_supply
        if z is None or z.direction != d:
            continue
        if z.freshness != "fresh":
            continue
        # The zone was marked BEFORE the BoS break (it is the level the
        # push departed from, not something drawn after the fact).
        if z.created_at >= ev.index:
            continue
        # The break pushed AWAY from the zone: the broken level sits on
        # the zone's far side.
        if d == 1 and not ev.level > z.top:
            continue
        if d == -1 and not ev.level < z.bottom:
            continue
        # Resting-limit level: midline, or the first-touched edge — the
        # literal "retest that level" reading (zone top for longs, zone
        # bottom for shorts).
        if entry == ENTRY_EDGE:
            level = z.top if d == 1 else z.bottom
        else:
            level = 0.5 * (z.top + z.bottom)
        # The limit must be a pullback: price on the favourable side of
        # the level at the signal bar.
        if d == 1 and not close[t] > level:
            continue
        if d == -1 and not close[t] < level:
            continue
        # Level untraded since the zone was created.  A fresh zone is
        # untraded at its edges by construction (the family counts a
        # touch as any wick into the zone); the midline needs its own
        # recompute from bars.
        if entry == ENTRY_MID:
            seg = low[z.created_at: t + 1] if d == 1 \
                else high[z.created_at: t + 1]
            if d == 1 and bool((seg <= level).any()):
                continue
            if d == -1 and bool((seg >= level).any()):
                continue
        stop = (z.bottom - STOP_BUFFER_ATR * atr) if d == 1 \
            else (z.top + STOP_BUFFER_ATR * atr)
        risk = (level - stop) * d
        if risk < MIN_RISK_ATR * atr or risk > MAX_RISK_ATR * atr:
            continue
        setups.append(RetestSetup(
            index=t, direction=d, mid=level, stop=stop, risk=risk, atr=atr,
            entry_mode=entry,
            zone_top=z.top, zone_bottom=z.bottom,
            zone_created_at=z.created_at,
            bos_index=ev.index, bos_level=ev.level,
            sweep_index=sw.index))
    return setups


# ------------------------------------------------------------------- fill
def find_fill(df: pd.DataFrame, setup: RetestSetup) -> RetestFill | None:
    """Work the resting limit from bar t+1.  Returns None when the order
    expires unfilled (cancelled, not a trade).

    Zone mitigation is close-based (the family's own rule): a close
    through the far edge kills the level.  A bar that fills AND kills
    the level is booked as filled-then-stopped (-1R) — the conservative
    same-bar convention.  ``setup.mid`` carries the limit price (the
    entry level for the setup's entry mode).
    """
    t = setup.index
    d = setup.direction
    mid, stop = setup.mid, setup.stop
    n = len(df)
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    ztop, zbot = setup.zone_top, setup.zone_bottom
    last = min(t + ORDER_EXPIRY_BARS, n - 1)
    for k in range(t + 1, last + 1):
        entry = _fill_price(d, mid, open_[k])
        if d == 1:
            touched = low[k] <= mid
            mitigated = close[k] < zbot
            if mitigated:
                # The level died: filled-then-stopped if the bar touched
                # the fill level on its way through.
                return (RetestFill(setup, k, entry, same_bar_stop=True)
                        if touched else None)
            if low[k] <= stop:          # traded through fill AND stop
                return RetestFill(setup, k, entry, same_bar_stop=True)
            if touched:
                return RetestFill(setup, k, entry)
        else:
            touched = high[k] >= mid
            mitigated = close[k] > ztop
            if mitigated:
                return (RetestFill(setup, k, entry, same_bar_stop=True)
                        if touched else None)
            if high[k] >= stop:
                return RetestFill(setup, k, entry, same_bar_stop=True)
            if touched:
                return RetestFill(setup, k, entry)
    return None


# ------------------------------------------------------------- variant A
def select_pool_target(states, fill: RetestFill) -> float | None:
    """Nearest liquidity pool ahead at the fill bar with >= 1R of room.

    Pools are the reader's nearest-first snapshots; the pool must have
    been created on or before the fill bar.  Returns None when no pool
    qualifies — the trade is skipped, not forced.
    """
    d = fill.setup.direction
    entry, risk = fill.entry, fill.setup.risk
    f = fill.fill_index
    pools: list[LiquidityPool] = (
        states[f].pools_above if d == 1 else states[f].pools_below)
    for p in pools:
        if p.created_at > f:
            continue
        if d == 1:
            if p.price < entry + POOL_TARGET_MIN_R * risk:
                continue
            return p.price
        if p.price > entry - POOL_TARGET_MIN_R * risk:
            continue
        return p.price
    return None


# --------------------------------------------------------------- simulate
def _to_signal(fill: RetestFill, target: float) -> GoldenSignal:
    s = fill.setup
    return GoldenSignal(
        index=fill.fill_index, timestamp=None, direction=s.direction,
        score=100.0, kind="sleeve_retest", entry_ref=fill.entry,
        invalidation=s.stop, targets=[target], atr=s.atr)


def simulate_trade(df: pd.DataFrame, fill: RetestFill, target: float,
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
class RetestResult:
    bars: int
    start: str
    end: str
    years: float
    entry_mode: str
    n_setups: int
    n_fills: int
    n_cancelled: int
    n_skipped_no_pool: int
    variant_a: dict = field(default_factory=dict)   # next-highs target
    variant_b: dict = field(default_factory=dict)   # fixed 2R
    shadow_a: dict = field(default_factory=dict)
    shadow_b: dict = field(default_factory=dict)


def run_sleeve(refresh: bool = False,
               entry: str = ENTRY_MID) -> RetestResult:
    """Full experiment: scan, fill, both variants, books + shadow."""
    df = load_gold_h1(refresh=refresh)
    states = read_and_featurize(df)
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25

    setups = scan_setups(df, states, entry=entry)
    fills: list[RetestFill] = []
    n_cancelled = 0
    for s in setups:
        fl = find_fill(df, s)
        if fl is None:
            n_cancelled += 1
        else:
            fills.append(fl)

    outcomes_a: list[Outcome] = []
    n_skipped_no_pool = 0
    for fl in fills:
        target = select_pool_target(states, fl)
        if target is None:
            n_skipped_no_pool += 1
            continue
        outcomes_a.append(simulate_trade(df, fl, target))

    outcomes_b: list[Outcome] = []
    for fl in fills:
        d = fl.setup.direction
        outcomes_b.append(
            simulate_trade(df, fl, fl.entry + d * FIXED_TP_R * fl.setup.risk))

    return RetestResult(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        years=years, entry_mode=entry,
        n_setups=len(setups), n_fills=len(fills),
        n_cancelled=n_cancelled, n_skipped_no_pool=n_skipped_no_pool,
        variant_a=book_metrics(build_book(outcomes_a, 0.0), years),
        variant_b=book_metrics(build_book(outcomes_b, 0.0), years),
        shadow_a=slice_stats(outcomes_a),
        shadow_b=slice_stats(outcomes_b),
    )
