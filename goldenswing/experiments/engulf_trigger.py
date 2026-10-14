"""Engulfing-close as the sweep-reversal trigger.

Source: Pro Trading School "Tsutsumi" video (user-mined).  The claim: for
sweep reversals, enter at the CLOSE of the engulfing candle instead of on
displacement + CHoCH/BOS confirmation.  Never trade engulfings mid-range --
only at edges/zones.

Variant A (ours): sweep -> displacement + CHoCH/BOS confirmation within 8
    bars -- the detector's exact Phase-2 trigger (``_sweep_confirmed``) --
    first trigger bar per sweep, with the detector's non-score gates
    (daily veto, room-to-pool, stop-distance band).  No score gate: scores
    are a separate, already-validated system; this experiment isolates
    the trigger.
Variant B (theirs): sweep -> first engulfing candle (body engulfs prior
    body) closing in the reversal direction within 8 bars, located at a
    pool/zone/AOC (never mid-range -- the video's own rule), same gates.
Variant C (confluence): first bar where BOTH the engulfing-at-level and
    the displacement + CHoCH confirmation hold.

Everything else identical across variants: next-bar-open entry (harness
convention -- the video says enter at the engulfing close; on H1 the
close -> open gap is noise and the shared harness keeps the comparison
about the trigger, not the fill), structural stop = sweep extreme +/-
0.15 ATR (detector's invalidation) + 0.25 ATR (harness buffer), Phase-2
exits (50% partial at first pool >= 1R, trail after +1.5R, 72-bar
horizon), $0.45/oz cost.

Shadow experiment: never touches the detector's scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.data import load_gold_h1
from ..backtest.harness import (ExitConfig, Outcome, book_metrics, build_book,
                                simulate_signal_v2, slice_stats)
from ..features.engine import read_and_featurize
from ..golden.detector import (DetectorConfig, GoldenSignal,
                               GoldenSwingDetector)
from ..reader.state import SweepEvent
from .candles import _location as candle_location
from .candles import engulfing

# ---------------------------------------------------------------- constants
#: Bars after a sweep in which a trigger may fire (Phase-2 confirm window).
CONFIRM_WINDOW = 8
#: Detector's invalidation buffer at the sweep extreme, in ATR.
INV_BUFFER_ATR = 0.15
#: Room gate when no pool is mapped ahead (detector convention).
ROOM_NO_POOL_ATR = 2.5

VARIANTS = ("A", "B", "C")


@dataclass
class TriggerHit:
    """One variant's trigger on one sweep event."""
    signal: GoldenSignal
    sweep: SweepEvent
    trigger_bar: int
    lag_bars: int  # trigger_bar - sweep.index


def _sweeps_in_order(states) -> list[SweepEvent]:
    """Unique sweep events, chronological.  One trade opportunity each."""
    out: list[SweepEvent] = []
    seen: set[int] = set()
    for s in states:
        sw = s.last_sweep
        if sw is not None and sw.index not in seen:
            seen.add(sw.index)
            out.append(sw)
    return out


def _invalidation(sweep: SweepEvent, atr: float, d: int) -> float:
    """Detector's structural invalidation at the sweep extreme."""
    return sweep.extreme - INV_BUFFER_ATR * atr if d == 1 \
        else sweep.extreme + INV_BUFFER_ATR * atr


def _passes_gates(det: GoldenSwingDetector, s, d: int,
                  sweep: SweepEvent, inv: float) -> bool:
    """The detector's non-score gates: daily veto, room, stop band."""
    cfg = det.cfg
    atr = s.atr
    if atr is None or atr <= 0:
        return False
    if cfg.daily_veto and s.daily_bias != 0 and d == -s.daily_bias:
        if not det._daily_exception(s, d, "sweep_reversal", sweep):
            return False
    pools_ahead = s.pools_above if d == 1 else s.pools_below
    room_atr = abs(pools_ahead[0].price - s.close) / atr if pools_ahead \
        else ROOM_NO_POOL_ATR
    if room_atr < cfg.min_room_atr:
        return False
    stop_atr = abs(s.close - inv) / atr
    if not (cfg.min_stop_atr <= stop_atr <= cfg.max_stop_atr):
        return False
    return True


def _make_signal(s, d: int, sweep: SweepEvent, inv: float) -> GoldenSignal:
    pools_ahead = s.pools_above if d == 1 else s.pools_below
    return GoldenSignal(
        index=s.index, timestamp=s.timestamp, direction=d, score=100.0,
        kind="sweep_reversal", invalidation=float(inv),
        targets=[float(p.price) for p in pools_ahead[:3]],
        atr=float(s.atr), sweep_index=sweep.index,
        entry_ref=float(s.close))


def scan_variant(df: pd.DataFrame, states, det: GoldenSwingDetector,
                 mode: str) -> list[TriggerHit]:
    """First trigger bar per sweep for variant A/B/C.  Fully causal."""
    assert mode in VARIANTS
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    hits: list[TriggerHit] = []
    for sw in _sweeps_in_order(states):
        d = sw.direction
        if d not in (1, -1):
            continue
        hi = min(sw.index + CONFIRM_WINDOW, n - 2)  # need t+1 for entry
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
            eng = engulfing(o, h, low, c, t) == d
            at_level = candle_location(s, d) != "midrange"
            confirmed = det._sweep_confirmed(s, d)
            fire = {"A": confirmed,
                    "B": eng and at_level,
                    "C": eng and at_level and confirmed}[mode]
            if fire and _passes_gates(det, s, d, sw, inv):
                hits.append(TriggerHit(
                    signal=_make_signal(s, d, sw, inv), sweep=sw,
                    trigger_bar=t, lag_bars=t - sw.index))
                break
    return hits


def simulate_hits(df: pd.DataFrame,
                  hits: list[TriggerHit]) -> list[tuple[Outcome, TriggerHit]]:
    out: list[tuple[Outcome, TriggerHit]] = []
    for hit in hits:
        oc = simulate_signal_v2(df, hit.signal)
        if oc is not None:
            out.append((oc, hit))
    return out


def _years(df: pd.DataFrame) -> float:
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    return span_days / 365.25


def run_experiment(df: pd.DataFrame | None = None,
                   refresh: bool = False) -> dict:
    """Full pipeline: read -> featurize -> scan A/B/C -> simulate."""
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
    # ---- timing: sweeps that fired in both A and B ----------------------
    lags = []
    for mode_pair in (("A", "B"), ("A", "C")):
        by_sweep = {}
        for mode in mode_pair:
            for hit in res[mode]["hits"]:
                by_sweep.setdefault(hit.sweep.index, {})[mode] = hit
        pairs = [(v["A"], v[mode_pair[1]]) for v in by_sweep.values()
                 if "A" in v and mode_pair[1] in v]
        lag_a = np.mean([p[0].lag_bars for p in pairs]) if pairs else float("nan")
        lag_x = np.mean([p[1].lag_bars for p in pairs]) if pairs else float("nan")
        lags.append({"pair": f"A vs {mode_pair[1]}", "n": len(pairs),
                     "mean_lag_a": float(lag_a), "mean_lag_x": float(lag_x)})
    res["timing"] = lags
    return res
