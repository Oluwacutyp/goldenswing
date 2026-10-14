"""Marco's sweep-entry vs our confirmation (Phase-2 pillar challenge).

Source: Marco Trades via Chart Fanatics (user-mined 2026-10-11).  The
claim: for liquidity sweeps, enter AT/AFTER the sweep with NO
displacement+CHoCH wait -- "the edge is the level, not entry precision."
The defense is level selection: only "engineered liquidity" qualifies --
highs that respect prior highs from the left (sellers building above),
i.e. a stricter level filter than any pool.

Variant A (ours): sweep of a QUALIFIED level -> displacement + CHoCH/BOS
    within 8 bars (the detector's exact ``_sweep_confirmed``), first
    trigger bar per sweep.  Phase-2 exits.  This is the anchor: it must
    reproduce the engulf_trigger experiment's variant-A sweep universe.
Variant B (Marco's): sweep of a QUALIFIED level -> enter at next bar
    open (no confirmation wait).  Stop just beyond the swept LEVEL +
    0.25 ATR buffer ("covering the level").  Target = nearest opposing
    QUALIFIED level.  Marco exits: roll stop to breakeven at +1R, hold
    to the target, 72-bar time stop.
Variant C (ablation): sweep of ANY pool (the reader's normal pool
    definition, no engineered filter) -> immediate entry, Marco exits,
    target = nearest opposing pool.  B vs C isolates the level filter;
    C is Phase-1's unconfirmed entry with Marco's exits.

Identical non-score gates for all variants (daily veto, room-to-pool,
stop-distance band) -- the detector's gates, no score gate.  One
position at a time in the books.

The question: does the engineered-liquidity filter rescue the
unconfirmed entry?  If B beats A, the Phase-2 pillar needs revision.
If B loses but C loses worse, the level filter has merit as context.
If all lose, the pillar stands.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.data import load_gold_h1
from ..backtest.harness import (COST_PER_OZ, HORIZON, Outcome, book_metrics,
                                build_book, simulate_signal_v2, slice_stats)
from ..features.engine import read_and_featurize
from ..golden.detector import (DetectorConfig, GoldenSignal,
                               GoldenSwingDetector)
from ..reader.state import SweepEvent
from .engulf_trigger import (_invalidation, _passes_gates, _sweeps_in_order)

# ---------------------------------------------------------------- constants
#: Bars after a sweep in which a confirmation may fire (Phase-2 window).
CONFIRM_WINDOW = 8
#: Bars of history scanned for level "respect" (touches from the left).
LOOKBACK = 200
#: Touch tolerance: wick within this of the level, in ATR.
TOUCH_TOL_ATR = 0.25
#: Minimum distinct approaches for a level to be "engineered".
MIN_TOUCHES = 2
#: Marco exit: roll stop to breakeven once this excursion is seen.
BE_TRIGGER_R = 1.0
#: Stop buffer beyond the structural invalidation, in ATR (Phase-2 conv).
STOP_BUFFER_ATR = 0.25

VARIANTS = ("A", "B", "C")


@dataclass
class MarcoHit:
    """One variant's trigger on one sweep event."""
    signal: GoldenSignal
    sweep: SweepEvent
    trigger_bar: int
    lag_bars: int      # trigger_bar - sweep.index
    qualified: bool     # swept level passed the engineered filter


def _count_touches(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   level: float, atr: float, si: int, side: str,
                   lookback: int = LOOKBACK,
                   tol_atr: float = TOUCH_TOL_ATR) -> int:
    """Count distinct approaches that *respected* the level, causally.

    Only bars ``[si - lookback, si)`` are read -- never the sweep bar
    itself, never anything after it.  A touch = the wick reached within
    ``tol_atr`` of the level while the body did NOT close through it
    (a close-through is a break, not respect).  Consecutive touch bars
    count as one approach (run-deduped).
    """
    if atr is None or atr <= 0:
        return 0
    tol = tol_atr * atr
    lo = max(0, si - lookback)
    touches = 0
    in_run = False
    for i in range(lo, si):
        if side == "bsl":      # level above price: highs respect it
            touched = high[i] >= level - tol and close[i] < level
        else:                  # "ssl": level below price: lows respect it
            touched = low[i] <= level + tol and close[i] > level
        if touched and not in_run:
            touches += 1
        in_run = touched
    return touches


def _engineered(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                level: float, atr: float, si: int, side: str,
                min_touches: int = MIN_TOUCHES) -> bool:
    """Marco's level filter: the level was respected from the left at
    least ``min_touches`` times before the sweep."""
    return _count_touches(high, low, close, level, atr, si, side) \
        >= min_touches


def _level_side(d: int) -> str:
    """The swept level sits below price for longs (ssl), above for shorts."""
    return "ssl" if d == 1 else "bsl"


def _nearest_qualified_opposing(high: np.ndarray, low: np.ndarray,
                                close: np.ndarray, s, t: int,
                                d: int, min_touches: int = MIN_TOUCHES
                                ) -> float | None:
    """Nearest opposing pool (nearest first) that passes the engineered
    filter with history strictly before bar ``t``.  None if there is none."""
    pools = s.pools_above if d == 1 else s.pools_below
    atr = s.atr
    opp_side = "bsl" if d == 1 else "ssl"
    for p in pools:
        if _engineered(high, low, close, p.price, atr, t, opp_side,
                       min_touches):
            if (p.price - s.close) * d > 0.1 * atr:
                return float(p.price)
    return None


def _make_signal(s, d: int, sweep: SweepEvent, inv: float,
                 targets: list[float]) -> GoldenSignal:
    return GoldenSignal(
        index=s.index, timestamp=s.timestamp, direction=d, score=100.0,
        kind="sweep_reversal", invalidation=float(inv),
        targets=[float(p) for p in targets],
        atr=float(s.atr), sweep_index=sweep.index,
        entry_ref=float(s.close))


def scan_variant(df: pd.DataFrame, states, det: GoldenSwingDetector,
                 mode: str, min_touches: int = MIN_TOUCHES) -> list[MarcoHit]:
    """Trigger bars per variant.  Fully causal."""
    assert mode in VARIANTS
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    hits: list[MarcoHit] = []
    for sw in _sweeps_in_order(states):
        d = sw.direction
        if d not in (1, -1):
            continue
        si = sw.index
        if si + 1 >= n - 1:
            continue
        s0 = states[si]
        atr = s0.atr
        if atr is None or atr <= 0:
            continue
        qualified = _engineered(h, low, c, sw.level, atr, si,
                                _level_side(d), min_touches)
        if mode in ("A", "B") and not qualified:
            continue
        if mode == "A":
            inv = _invalidation(sw, atr, d)
            hi = min(si + CONFIRM_WINDOW, n - 2)
            for t in range(si, hi + 1):
                s = states[t]
                if s.last_sweep is None or s.last_sweep.index != sw.index:
                    break
                if det._sweep_confirmed(s, d) and \
                        _passes_gates(det, s, d, sw, inv):
                    sig = _make_signal(
                        s, d, sw, inv,
                        [p.price for p in
                         (s.pools_above if d == 1 else s.pools_below)[:3]])
                    hits.append(MarcoHit(signal=sig, sweep=sw,
                                         trigger_bar=t, lag_bars=t - si,
                                         qualified=qualified))
                    break
        else:
            # B / C: immediate entry at next bar open after the sweep.
            s = s0
            inv = sw.level          # Marco: stop covers the LEVEL
            if not _passes_gates(det, s, d, sw, inv):
                continue
            if mode == "B":
                target = _nearest_qualified_opposing(h, low, c, s, si, d,
                                                     min_touches)
            else:
                pools = s.pools_above if d == 1 else s.pools_below
                target = None
                for p in pools:
                    if (p.price - s.close) * d > 0.1 * atr:
                        target = float(p.price)
                        break
            if target is None:
                continue
            sig = _make_signal(s, d, sw, inv, [target])
            hits.append(MarcoHit(signal=sig, sweep=sw, trigger_bar=si,
                                 lag_bars=0, qualified=qualified))
    return hits


def simulate_marco(df: pd.DataFrame, sig: GoldenSignal,
                   horizon: int = HORIZON,
                   be_trigger_r: float = BE_TRIGGER_R,
                   cost: float = COST_PER_OZ) -> Outcome | None:
    """Marco's exit style: hold to the opposing-liquidity target, roll the
    stop to breakeven once +``be_trigger_r`` prints, 72-bar time stop.
    Conservative intrabar order (stops before targets), BE armed from
    completed bars only -- the same honesty conventions as the harness.
    R is measured on the buffered stop, so it stays comparable to v2.
    """
    t = sig.index
    n = len(df)
    if t + 1 >= n or not sig.targets:
        return None
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    d = sig.direction
    entry = open_[t + 1]
    inv = sig.invalidation
    buffer = STOP_BUFFER_ATR * sig.atr if sig.atr > 0 else 0.0
    stop0 = inv - d * buffer
    risk = (entry - stop0) * d
    if risk <= 0:
        return None
    target = float(sig.targets[0])
    if (target - entry) * d <= 0:
        return None

    last = min(t + horizon, n - 1)
    be_armed = False
    best_r = 0.0
    wicked = False
    exit_price, exit_idx, reason = float(close[last]), last, "timeout"

    for k in range(t + 1, last + 1):
        if (d == 1 and low[k] < inv) or (d == -1 and high[k] > inv):
            wicked = True
        # 1. stops first (conservative): the hard stop, tightened to
        #    breakeven once armed -- the BE stop dominates stop0, so it
        #    is the active stop (same pattern as the harness trail).
        active_stop, be_active = stop0, False
        if be_armed and ((d == 1 and entry > stop0) or
                         (d == -1 and entry < stop0)):
            active_stop, be_active = entry, True
        stop_hit = low[k] <= active_stop if d == 1 else high[k] >= active_stop
        if stop_hit:
            if d == 1:
                exit_price = active_stop if open_[k] > active_stop \
                    else open_[k]
            else:
                exit_price = active_stop if open_[k] < active_stop \
                    else open_[k]
            exit_idx, reason = k, "be" if be_active else "stop"
            break
        # 2. the opposing-liquidity target.
        tgt_hit = high[k] >= target if d == 1 else low[k] <= target
        if tgt_hit:
            fill = target
            if (d == 1 and open_[k] > target) or \
                    (d == -1 and open_[k] < target):
                fill = open_[k]      # gapped through: take the open
            exit_price, exit_idx, reason = float(fill), k, "target"
            break
        # 3. arm BE from completed bars only.
        fav = high[k] if d == 1 else low[k]
        best_r = max(best_r, d * (fav - entry) / risk)
        if best_r >= be_trigger_r:
            be_armed = True
    else:
        exit_price, exit_idx, reason = float(close[last]), last, "timeout"

    r_net = d * (exit_price - entry) / risk - cost / risk
    return Outcome(signal=sig, entry_index=t + 1, exit_index=exit_idx,
                   exit_reason=reason, r_net=float(r_net),
                   hold_bars=exit_idx - t, wicked_inv=wicked)


def simulate_hits(df: pd.DataFrame,
                  hits: list[MarcoHit], mode: str) -> list[Outcome]:
    out: list[Outcome] = []
    for hit in hits:
        oc = simulate_signal_v2(df, hit.signal) if mode == "A" \
            else simulate_marco(df, hit.signal)
        if oc is not None:
            out.append(oc)
    return out


def _years(df: pd.DataFrame) -> float:
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    return span_days / 365.25


def sensitivity_b(df: pd.DataFrame, states, det: GoldenSwingDetector,
                  thresholds: tuple[int, ...] = (2, 3, 4, 5, 6)) -> list[dict]:
    """Variant B books at stricter engineered-touch thresholds.  Shares
    the already-computed states; only the B scan + simulation re-runs."""
    yrs = _years(df)
    n = len(df)
    out: list[dict] = []
    for mt in thresholds:
        hits = scan_variant(df, states, det, "B", min_touches=mt)
        outcomes = simulate_hits(df, hits, "B")
        book = build_book(outcomes, 0.0)
        out.append({"min_touches": mt, "hits": len(hits),
                    "shadow": slice_stats(outcomes),
                    "book": book_metrics(book, yrs)})
    return out


def run_experiment(df: pd.DataFrame | None = None,
                   states=None,
                   refresh: bool = False) -> dict:
    """Full pipeline: read -> featurize -> scan A/B/C -> simulate.

    Pass ``df`` and ``states`` to share an already-loaded read (the
    sensitivity runner does this); otherwise they are loaded here.
    """
    if df is None:
        df = load_gold_h1(refresh=refresh)
    if states is None:
        states = read_and_featurize(df)
    det = GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True))
    yrs = _years(df)
    n = len(df)
    res: dict = {"bars": n, "years": yrs,
                 "start": str(df.index[0]), "end": str(df.index[-1])}
    # qualification coverage: what fraction of sweeps are "engineered"?
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    sweeps = [sw for sw in _sweeps_in_order(states)
              if sw.direction in (1, -1) and sw.index + 1 < n - 1]
    qual = 0
    for sw in sweeps:
        a = states[sw.index].atr
        if a and a > 0 and _engineered(h, low, c, sw.level, a, sw.index,
                                       _level_side(sw.direction)):
            qual += 1
    res["qualification"] = {"sweeps": len(sweeps), "qualified": qual}

    for mode in VARIANTS:
        hits = scan_variant(df, states, det, mode)
        outcomes = simulate_hits(df, hits, mode)
        book = build_book(outcomes, 0.0)  # scores are 100: keep everything
        half = n // 2
        h1 = [oc for oc in outcomes if oc.signal.index < half]
        h2 = [oc for oc in outcomes if oc.signal.index >= half]
        res[mode] = {
            "hits": hits,
            "outcomes": outcomes,
            "shadow": slice_stats(outcomes),
            "book": book_metrics(book, yrs),
            "book_h1": book_metrics(build_book(h1, 0.0), yrs / 2),
            "book_h2": book_metrics(build_book(h2, 0.0), yrs / 2),
        }
    # ---- anchor validation: the real detector's sweep-reversal slice ----
    det_anchor = GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True, emit_floor=35.0))
    anchor = [s for s in det_anchor.scan(states)
              if s.kind == "sweep_reversal"]
    a_bars = {h.trigger_bar for h in res["A"]["hits"]}
    anchor_bars = {s.index for s in anchor}
    res["anchor"] = {
        "n_detector_signals": len(anchor),
        "n_variant_a": len(res["A"]["hits"]),
        "overlap": len(a_bars & anchor_bars),
    }
    # ---- exit mix for Marco variants ------------------------------------
    for mode in ("B", "C"):
        mix: dict[str, int] = {}
        for oc in res[mode]["outcomes"]:
            mix[oc.exit_reason] = mix.get(oc.exit_reason, 0) + 1
        res[mode]["exit_mix"] = mix
    return res
