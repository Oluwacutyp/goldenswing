"""Sweep-candle body-break as the sweep-reversal confirmation.

Source: SMC Liquidity Sweep Scalper (RanaAli878, MQL5 CodeBase, user-mined
2026-10-11).  The claim: after a liquidity sweep, wait N bars for a
confirmation break back through THE SWEEP CANDLE'S BODY (not just any
displacement/CHoCH) and enter at market.  This is the THIRD confirmation
definition for the identical sweep setup -- ours is displacement+CHoCH/BOS
(the Phase-2 pillar), the Tsutsumi video's was the engulfing close
(killed, #27).  Same sweep universe, same gates, one variable isolated.

Variant A (ours): sweep -> displacement + CHoCH/BOS confirmation within 8
    bars -- the detector's exact Phase-2 trigger (``_sweep_confirmed``) --
    first trigger bar per sweep, with the detector's non-score gates
    (daily veto, room-to-pool, stop-distance band).  No score gate: scores
    are a separate, already-validated system; this experiment isolates
    the confirmation definition.
Variant B5 / B10 (theirs): sweep -> first bar within 5 / 10 bars whose
    CLOSE crosses back through the sweep candle's BODY in the reversal
    direction (bullish: close > max(open, close) of the sweep bar;
    bearish: close < min(open, close)), same gates, enter next-bar-open.
Variant C (theirs + quality gates): B10 plus the author's stated gates --
    the confirmation candle must show momentum (body > 0.5x range, not
    weak drift) and the stop must be meaningful vs ATR (>= 0.5 ATR; the
    detector's own gate is 0.25).

Everything else identical across variants: next-bar-open entry (harness
convention), structural stop = sweep extreme +/- 0.15 ATR (detector's
invalidation) + 0.25 ATR (harness buffer), Phase-2 exits (50% partial at
first pool >= 1R, trail after +1.5R, 72-bar horizon), $0.45/oz cost.

Shadow experiment: never touches the detector's scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.data import load_gold_h1
from ..backtest.harness import book_metrics, build_book, slice_stats
from ..features.engine import read_and_featurize
from ..golden.detector import (DetectorConfig, GoldenSignal,
                               GoldenSwingDetector)
from ..reader.state import SweepEvent
from .engulf_trigger import (_invalidation, _make_signal, _passes_gates,
                             _sweeps_in_order, simulate_hits)

# ---------------------------------------------------------------- constants
#: Phase-2 confirm window (the anchor's definition).
CONFIRM_WINDOW_A = 8
#: Author's confirmation windows to test (5 and 10 bars).
CONFIRM_WINDOW_B5 = 5
CONFIRM_WINDOW_B10 = 10
#: Variant C: confirmation candle momentum gate (body fraction of range).
MOMENTUM_BODY_FRAC = 0.5
#: Variant C: stop must be meaningful vs ATR (author's gate; detector's
#: own gate is 0.25).
MIN_STOP_ATR_C = 0.5

VARIANTS = ("A", "B5", "B10", "C")


@dataclass
class TriggerHit:
    """One variant's trigger on one sweep event."""
    signal: GoldenSignal
    sweep: SweepEvent
    trigger_bar: int
    lag_bars: int  # trigger_bar - sweep.index


def body_break_trigger(o: np.ndarray, c: np.ndarray, sw_index: int,
                       t: int, d: int) -> bool:
    """Does bar ``t`` close back through the sweep bar's BODY in
    direction ``d``?  Bullish (d=+1): close above the sweep candle's body
    top.  Bearish (d=-1): close below the sweep candle's body bottom.
    Pure function of bars <= t (the sweep bar is in the past)."""
    body_top = max(o[sw_index], c[sw_index])
    body_bot = min(o[sw_index], c[sw_index])
    if d == 1:
        return bool(c[t] > body_top)
    return bool(c[t] < body_bot)


def _momentum_ok(o: np.ndarray, h: np.ndarray, low: np.ndarray,
                 c: np.ndarray, t: int) -> bool:
    """The author's momentum gate: the confirmation candle's body must
    exceed half its range -- real momentum, not weak drift."""
    rng = h[t] - low[t]
    if rng <= 0:
        return False
    return bool(abs(c[t] - o[t]) > MOMENTUM_BODY_FRAC * rng)


def scan_variant(df: pd.DataFrame, states, det: GoldenSwingDetector,
                 mode: str) -> list[TriggerHit]:
    """First trigger bar per sweep for each variant.  Fully causal."""
    assert mode in VARIANTS
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    window = {"A": CONFIRM_WINDOW_A, "B5": CONFIRM_WINDOW_B5,
              "B10": CONFIRM_WINDOW_B10, "C": CONFIRM_WINDOW_B10}[mode]
    hits: list[TriggerHit] = []
    for sw in _sweeps_in_order(states):
        d = sw.direction
        if d not in (1, -1):
            continue
        hi = min(sw.index + window, n - 2)  # need t+1 for entry
        for t in range(sw.index, hi + 1):
            s = states[t]
            # Sweep superseded by a newer one: the reader (and the
            # detector) no longer sees this setup.
            if s.last_sweep is None or s.last_sweep.index != sw.index:
                break
            atr = s.atr
            if atr is None or atr <= 0:
                continue
            inv = _invalidation(sw, atr, d)
            confirmed = det._sweep_confirmed(s, d)
            body_break = body_break_trigger(o, c, sw.index, t, d)
            if mode == "A":
                fire = confirmed
            elif mode in ("B5", "B10"):
                fire = body_break
            else:  # C: body-break + momentum + meaningful stop
                fire = (body_break and _momentum_ok(o, h, low, c, t)
                        and abs(c[t] - inv) / atr >= MIN_STOP_ATR_C)
            if fire and _passes_gates(det, s, d, sw, inv):
                hits.append(TriggerHit(
                    signal=_make_signal(s, d, sw, inv), sweep=sw,
                    trigger_bar=t, lag_bars=t - sw.index))
                break
    return hits


def _years(df: pd.DataFrame) -> float:
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    return span_days / 365.25


def run_experiment(df: pd.DataFrame | None = None,
                   refresh: bool = False) -> dict:
    """Full pipeline: read -> featurize -> scan A/B5/B10/C -> simulate."""
    if df is None:
        df = load_gold_h1(refresh=refresh)
    states = read_and_featurize(df)
    det = GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True))
    yrs = _years(df)
    n = len(df)
    res: dict = {"bars": n, "years": yrs,
                 "start": str(df.index[0]), "end": str(df.index[-1])}
    for mode in VARIANTS:
        hits = scan_variant(df, states, det, mode)
        sims = simulate_hits(df, hits)
        outcomes = [oc for oc, _ in sims]
        book = build_book(outcomes, 0.0)  # scores are 100: keep everything
        half = n // 2
        h1 = [oc for oc, hit in sims if hit.signal.index < half]
        h2 = [oc for oc, hit in sims if hit.signal.index >= half]
        res[mode] = {
            "hits": hits,
            "sims": sims,
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
    # ---- timing: sweeps that fired in both A and a B/C variant ----------
    lags = []
    for other in ("B5", "B10", "C"):
        by_sweep = {}
        for mode in ("A", other):
            for hit in res[mode]["hits"]:
                by_sweep.setdefault(hit.sweep.index, {})[mode] = hit
        pairs = [(v["A"], v[other]) for v in by_sweep.values()
                 if "A" in v and other in v]
        lag_a = np.mean([p[0].lag_bars for p in pairs]) \
            if pairs else float("nan")
        lag_x = np.mean([p[1].lag_bars for p in pairs]) \
            if pairs else float("nan")
        lags.append({"pair": f"A vs {other}", "n": len(pairs),
                     "mean_lag_a": float(lag_a),
                     "mean_lag_x": float(lag_x)})
    res["timing"] = lags
    return res
