"""OTE as the entry zone (claim #60, mined from ICT MMXM).

The MMXM model: accumulation -> manipulation (liquidity sweep + impulsive
displacement leg) -> retracement into the Optimal Trade Entry zone
(62-79% Fibonacci of the manipulation leg), confluent with the order
block formed by the displacement leg.  A resting limit sits at the OTE
zone instead of chasing a confirmation trigger.

Our system has Fib/OTE only as a context gate; this tests it as the
ENTRY MECHANIC, head-to-head against the sweep-reversal playbook's
displacement+CHoCH trigger on the SAME setups (entry mechanic vs entry
mechanic, exits held constant at Phase-2).

Mechanics (all causal — fib computed from confirmed leg extremes only):
- Setup at close of bar t: a detector sweep_reversal signal (the
  confirmed universe: sweep + displacement + CHoCH/BOS, daily veto,
  score >= 60).  The manipulation leg = sweep extreme -> displacement
  extreme (extreme in the trade direction over (sweep.index, t]).
- OTE zone = 62-79% retracement of that leg.  Limit price swept at
  62 / 70 / 79.
- Confluence: the zone must overlap an ACTIVE order block in the trade
  direction (the displacement OB), else no trade.
- Pullback requirement: the whole zone must sit on the pullback side of
  the signal-bar close (else the setup is stale — price already there).
- A resting limit is placed at the fib price, live from bar t+1,
  expiring after ORDER_EXPIRY_BARS.  Fill = first bar whose wick touches
  the limit; a gap through it fills at the (adverse) open.  Cancelled
  when price closes THROUGH the zone (beyond the far edge) before the
  fill — the pullback went too deep.  If a bar trades through BOTH the
  limit and the stop, the fill is taken and the stop exits it the same
  bar (conservative: -1R).
- Stop = beyond the manipulation extreme + STOP_BUFFER_ATR (repo
  convention).  Skipped when the stop is < 0.25 or > 4 ATR away.
- Exits: Phase-2 machinery (buffered structural stop already in the
  stop, 50% partial at first pool >= 1R from the signal's own targets,
  trail after +1.5R, 72-bar horizon, $0.45/oz) — identical exit code
  path to the anchor, so the ENTRY is the isolated variable.

The question: does resting at OTE beat chasing the confirmation?  A
limit at the fib zone should get better prices but miss the runners
that never pull back — fill rate and missed-trade cost are measured
explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.data import load_gold_h1, load_peers
from ..backtest.harness import (COST_PER_OZ, ExitConfig, Outcome,
                                book_metrics, build_book, slice_stats)
from ..backtest.validation import COMBO_A
from ..features.engine import read_and_featurize
from ..golden.detector import DetectorConfig, GoldenSignal, GoldenSwingDetector
from ..reader.state import MarketState, OrderBlock

# ---------------------------------------------------------------- constants
#: Fib retracement levels swept as the resting-limit price.
FIB_LEVELS = (0.62, 0.70, 0.79)
#: OTE zone bounds: 62% (near edge) .. 79% (far edge) of the leg.
OTE_NEAR, OTE_FAR = 0.62, 0.79
#: Resting OTE limit lives this many bars after the signal bar.
ORDER_EXPIRY_BARS = 24
#: Stop buffer beyond the manipulation extreme (Phase-2 convention).
STOP_BUFFER_ATR = 0.25
#: Stop-distance sanity band, in ATR (sleeve-experiment convention).
MIN_RISK_ATR = 0.25
MAX_RISK_ATR = 4.0
#: The manipulation leg must span at least this (else the fib is noise).
MIN_LEG_ATR = 1.0
#: Max hold after the fill, bars (Phase-2 horizon).
TRADE_HORIZON = 72

#: Anchor detector config — byte-identical to the SweepReversalPlaybook.
ANCHOR_FAMILIES = COMBO_A | {"aoc"}
ANCHOR_FLAGS = dict(require_sweep_confirmation=True, daily_veto=True)


# ------------------------------------------------------------------ events
@dataclass
class OTESetup:
    """An OTE resting-limit setup, decided at the close of ``index``."""

    index: int               # signal bar t
    direction: int           # +1 long | -1 short
    zone_lo: float           # OTE zone bounds (retracement of the leg)
    zone_hi: float
    stop: float              # buffered beyond the manipulation extreme
    atr: float
    sweep_index: int
    sweep_extreme: float
    leg_lo: float            # manipulation leg extremes
    leg_hi: float
    targets: list[float]     # pools ahead (the signal's own targets)
    score: float             # the detector signal's score
    ob_top: float            # confluent OB bounds
    ob_bottom: float


@dataclass
class OTEFill:
    setup: OTESetup
    fib: float               # fib level used as the limit price
    limit: float
    fill_index: int
    entry: float
    risk: float              # recomputed from the actual fill
    same_bar_stop: bool = False


# ------------------------------------------------------------- pure helpers
def _obs_active_at(df_close: np.ndarray,
                   states: list[MarketState], upto: int) -> list[OrderBlock]:
    """Every OB live at bar ``upto``, reader close-through rule.

    Mirrors goldenswing.experiments.choch_fvg: the per-state snapshot is
    capped, so the live set is reconstructed from the union of snapshots
    up to ``upto`` — causal, bars <= ``upto`` only.
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


def _ote_zone(d: int, leg_lo: float, leg_hi: float) -> tuple[float, float]:
    """(zone_lo, zone_hi): 62-79% retracement of the manipulation leg."""
    rng = leg_hi - leg_lo
    if d == 1:
        # leg runs low -> high; retracement measured down from the high
        return leg_hi - OTE_FAR * rng, leg_hi - OTE_NEAR * rng
    # leg runs high -> low; retracement measured up from the low
    return leg_lo + OTE_NEAR * rng, leg_lo + OTE_FAR * rng


def _fill_price(direction: int, limit: float, open_: float) -> float:
    """Resting-limit fill: the limit, or the adverse open on a gap."""
    if direction == 1:
        return limit if open_ > limit else open_
    return limit if open_ < limit else open_


# ------------------------------------------------------------------- scan
def scan_setups(df: pd.DataFrame, states: list[MarketState],
                signals: list[GoldenSignal]) -> list[OTESetup]:
    """Find OTE resting-limit setups, one per sweep_reversal signal.

    Pure function of bars <= each signal bar: the sweep, the leg
    extremes, the fib zone, the OB confluence and the stop are all
    knowable at the close of ``setup.index``.
    """
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    setups: list[OTESetup] = []
    for sig in signals:
        if sig.kind != "sweep_reversal":
            continue
        t = sig.index
        d = sig.direction
        st = states[t]
        atr = st.atr
        if not (atr and atr > 0 and np.isfinite(atr)):
            continue
        sweep = st.last_sweep
        # The signal must be built on this sweep (detector invariant).
        if sweep is None or sweep.direction != d:
            continue
        if sig.sweep_index is not None and sweep.index != sig.sweep_index:
            continue
        # Manipulation leg: sweep extreme -> displacement extreme, i.e.
        # the extreme in the trade direction over (sweep.index, t].
        seg_lo = low[sweep.index: t + 1]
        seg_hi = high[sweep.index: t + 1]
        if d == 1:
            leg_lo, leg_hi = float(sweep.extreme), float(seg_hi.max())
        else:
            leg_lo, leg_hi = float(seg_lo.min()), float(sweep.extreme)
        rng = leg_hi - leg_lo
        if rng < MIN_LEG_ATR * atr:
            continue
        zone_lo, zone_hi = _ote_zone(d, leg_lo, leg_hi)
        # Pullback requirement: the whole zone on the pullback side of
        # the signal-bar close, else the setup is stale.
        if d == 1 and not close[t] > zone_hi:
            continue
        if d == -1 and not close[t] < zone_lo:
            continue
        # OB confluence: an active OB in the trade direction overlapping
        # the OTE zone (the displacement OB).
        confluent = None
        for ob in _obs_active_at(close, states, t):
            if ob.direction != d:
                continue
            if ob.bottom <= zone_hi and ob.top >= zone_lo:
                confluent = ob
                break
        if confluent is None:
            continue
        stop = sweep.extreme - d * STOP_BUFFER_ATR * atr
        setups.append(OTESetup(
            index=t, direction=d, zone_lo=zone_lo, zone_hi=zone_hi,
            stop=stop, atr=atr, sweep_index=sweep.index,
            sweep_extreme=float(sweep.extreme), leg_lo=leg_lo, leg_hi=leg_hi,
            targets=list(sig.targets), score=sig.score,
            ob_top=float(confluent.top), ob_bottom=float(confluent.bottom)))
    return setups


# ------------------------------------------------------------------- fill
def find_fill(df: pd.DataFrame, setup: OTESetup, fib: float) -> OTEFill | None:
    """Work the resting OTE limit from bar t+1.

    Returns None when the order expires unfilled, the zone is violated
    (close through the far edge) before the fill, or the recomputed
    risk falls outside the sanity band.
    """
    t = setup.index
    d = setup.direction
    rng = setup.leg_hi - setup.leg_lo
    limit = setup.leg_hi - fib * rng if d == 1 \
        else setup.leg_lo + fib * rng
    stop = setup.stop
    risk = (limit - stop) * d
    if risk < MIN_RISK_ATR * setup.atr or risk > MAX_RISK_ATR * setup.atr:
        return None
    n = len(df)
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    last = min(t + ORDER_EXPIRY_BARS, n - 1)
    for k in range(t + 1, last + 1):
        # Zone violation cancels the order BEFORE any fill logic: a
        # close through the far edge means the pullback went too deep.
        if d == 1 and close[k] < setup.zone_lo:
            return None
        if d == -1 and close[k] > setup.zone_hi:
            return None
        entry = _fill_price(d, limit, open_[k])
        if d == 1:
            if low[k] <= stop:          # traded through fill AND stop
                r = (entry - stop) * d
                return OTEFill(setup, fib, limit, k, entry, r,
                               same_bar_stop=True)
            if low[k] <= limit:
                r = (entry - stop) * d
                return OTEFill(setup, fib, limit, k, entry, r)
        else:
            if high[k] >= stop:
                r = (entry - stop) * d
                return OTEFill(setup, fib, limit, k, entry, r,
                               same_bar_stop=True)
            if high[k] >= limit:
                r = (entry - stop) * d
                return OTEFill(setup, fib, limit, k, entry, r)
    return None


# --------------------------------------------------------------- simulate
def _to_signal(fill: OTEFill) -> GoldenSignal:
    s = fill.setup
    return GoldenSignal(
        index=fill.fill_index, timestamp=None, direction=s.direction,
        score=s.score, kind="ote_entry", entry_ref=fill.entry,
        invalidation=s.stop, targets=list(s.targets), atr=s.atr)


def simulate_ote_trade(df: pd.DataFrame, fill: OTEFill,
                       exits: ExitConfig | None = None,
                       cost: float = COST_PER_OZ) -> Outcome:
    """Phase-2 exit walk with an explicit entry price.

    Exit logic mirrors ``simulate_signal_v2`` exactly (stop first,
    partial at the first pool >= 1R, trail after +1.5R, 72-bar horizon,
    stop wins ties, adverse opens, flat cost in R) — the only
    difference is the entry: the OTE fill price at the fill bar instead
    of the signal bar's next open.  The stop is already buffered
    (sweep extreme +/- 0.25 ATR), so no second buffer is applied.
    """
    exits = exits or ExitConfig()
    s = fill.setup
    d = s.direction
    entry, stop, risk = fill.entry, s.stop, fill.risk
    n = len(df)
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    sig = _to_signal(fill)
    if fill.same_bar_stop or risk <= 0:
        r_net = -1.0 - cost / max(risk, 1e-9)
        return Outcome(signal=sig, entry_index=fill.fill_index,
                       exit_index=fill.fill_index, exit_reason="stop",
                       r_net=float(r_net), hold_bars=0)

    # partial target: nearest pool ahead at least partial_min_r away
    partial_price: float | None = None
    ahead = sorted(
        (p for p in s.targets if (p - entry) * d >= exits.partial_min_r * risk),
        key=lambda p: abs(p - entry))
    if ahead and exits.partial_frac > 0:
        partial_price = float(ahead[0])

    f = fill.fill_index
    last = min(f + TRADE_HORIZON, n - 1)
    r_banked = 0.0
    rest_frac = 1.0
    partial_done = False
    best_r = 0.0
    trail_level: float | None = None
    exit_price, exit_idx, reason = float(close[last]), last, "timeout"
    for k in range(f + 1, last + 1):
        active_stop = stop
        stop_is_trail = False
        if trail_level is not None:
            if (d == 1 and trail_level > active_stop) or \
                    (d == -1 and trail_level < active_stop):
                active_stop, stop_is_trail = trail_level, True
        stop_hit = low[k] <= active_stop if d == 1 else high[k] >= active_stop
        if stop_hit:
            if d == 1:
                exit_price = active_stop if open_[k] > active_stop \
                    else open_[k]
            else:
                exit_price = active_stop if open_[k] < active_stop \
                    else open_[k]
            exit_idx = k
            reason = "trail" if stop_is_trail else "stop"
            break
        if not partial_done and partial_price is not None:
            crossed = high[k] >= partial_price if d == 1 \
                else low[k] <= partial_price
            if crossed:
                fill_p = partial_price
                if (d == 1 and open_[k] > partial_price) or \
                        (d == -1 and open_[k] < partial_price):
                    fill_p = open_[k]
                r_banked += exits.partial_frac * d * (fill_p - entry) / risk
                rest_frac -= exits.partial_frac
                partial_done = True
        fav = high[k] if d == 1 else low[k]
        best_r = max(best_r, d * (fav - entry) / risk)
        if best_r >= exits.trail_trigger_r:
            lvl = entry + d * (best_r - exits.trail_dist_r) * risk
            if trail_level is None or (d == 1 and lvl > trail_level) or \
                    (d == -1 and lvl < trail_level):
                trail_level = lvl
    else:
        exit_price, exit_idx, reason = float(close[last]), last, "timeout"

    r_net = r_banked + rest_frac * d * (exit_price - entry) / risk \
        - exits.cost / risk
    return Outcome(signal=sig, entry_index=f, exit_index=exit_idx,
                   exit_reason=reason, r_net=float(r_net),
                   hold_bars=exit_idx - f)


# -------------------------------------------------------------------- run
@dataclass
class OTEResult:
    bars: int
    start: str
    end: str
    years: float
    n_signals: int            # sweep_reversal signals @ >=60 (anchor universe)
    n_setups: int             # with OTE zone + OB confluence + pullback side
    setup_indices: list = field(default_factory=list)  # signal bars with setups
    anchor_book: dict = field(default_factory=dict)
    anchor_shadow: dict = field(default_factory=dict)
    anchor_outcomes: list = field(default_factory=list)
    fib: dict = field(default_factory=dict)   # fib -> result dict


def _anchor_detector() -> GoldenSwingDetector:
    return GoldenSwingDetector(DetectorConfig(
        emit_floor=35.0, golden_threshold=60.0,
        feature_families=ANCHOR_FAMILIES, **ANCHOR_FLAGS))


def run_ote(fib_levels: tuple[float, ...] = FIB_LEVELS,
            refresh: bool = False) -> OTEResult:
    """Full experiment: anchor book + OTE books per fib level."""
    from ..backtest.harness import simulate_signal_v2

    df = load_gold_h1(refresh=refresh)
    peers = load_peers()
    states = read_and_featurize(df, peers)
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25

    detector = _anchor_detector()
    signals = [s for s in detector.scan(states)
               if s.kind == "sweep_reversal" and s.score >= 60.0]

    # Anchor: the playbook's own machinery on the same setups.
    anchor_outcomes = []
    for sig in signals:
        oc = simulate_signal_v2(df, sig)
        if oc is not None:
            anchor_outcomes.append(oc)
    anchor_book = book_metrics(build_book(anchor_outcomes, 60.0), years)

    setups = scan_setups(df, states, signals)

    fib_results: dict[float, dict] = {}
    for fib in fib_levels:
        fills: list[OTEFill] = []
        n_expired = 0
        n_risk_skip = 0
        for setup in setups:
            fl = find_fill(df, setup, fib)
            if fl is None:
                # risk-band skip vs expiry/violation: recompute the
                # setup-time risk to classify (diagnostic only)
                rng = setup.leg_hi - setup.leg_lo
                limit = setup.leg_hi - fib * rng if setup.direction == 1 \
                    else setup.leg_lo + fib * rng
                r0 = (limit - setup.stop) * setup.direction
                if r0 < MIN_RISK_ATR * setup.atr \
                        or r0 > MAX_RISK_ATR * setup.atr:
                    n_risk_skip += 1
                else:
                    n_expired += 1
            else:
                fills.append(fl)
        outcomes = [simulate_ote_trade(df, fl) for fl in fills]
        book = build_book(outcomes, 60.0)
        # Missed-trade cost: what did the anchor do with the setups the
        # OTE limit never filled?  (setup bars that produced no fill)
        filled_setup_idx = {fl.setup.index for fl in fills}
        missed_r = [o.r_net for o in anchor_outcomes
                    if o.signal.index not in filled_setup_idx]
        # Fill-price improvement vs the anchor's next-bar-open entry,
        # in OTE-risk units (positive = OTE got the better price).
        open_ = df["open"].to_numpy(dtype=float)
        improv = []
        for fl in fills:
            t = fl.setup.index
            if t + 1 < len(df):
                improv.append(
                    fl.setup.direction * (open_[t + 1] - fl.entry)
                    / max(fl.risk, 1e-9))
        fib_results[fib] = dict(
            n_fills=len(fills), n_expired=n_expired,
            n_risk_skip=n_risk_skip,
            fill_rate=len(fills) / max(len(setups), 1),
            book=book_metrics(book, years),
            shadow=slice_stats(outcomes),
            outcomes=outcomes,
            missed_n=len(missed_r),
            missed_avg_r=float(np.mean(missed_r)) if missed_r else 0.0,
            missed_total_r=float(np.sum(missed_r)) if missed_r else 0.0,
            fill_improve_avg_r=float(np.mean(improv)) if improv else 0.0,
        )

    return OTEResult(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        years=years, n_signals=len(signals), n_setups=len(setups),
        setup_indices=[s.index for s in setups],
        anchor_book=anchor_book,
        anchor_shadow=slice_stats(anchor_outcomes),
        anchor_outcomes=anchor_outcomes,
        fib=fib_results,
    )
