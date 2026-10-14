"""Body-anchored stops for sweep reversals.

Source: Faiz SMC's ICT turtle soup video (user-mined 2026-10-11).  The
claim: for sweep reversals, anchor the stop at the sweep candle's BODY
(not the wick extreme) -- deliberately tighter, accepting more
stop-outs for better R:R.  The author explicitly weighs the wider-stop
trade-off, so both a buffered and an unbuffered body variant are tested.

Variant A (ours): the detector's exact sweep-reversal trigger
    (displacement + CHoCH/BOS within 8 bars), structural invalidation at
    the sweep wick extreme +/- 0.15 ATR, harness stop buffer 0.25 ATR.
Variant B (theirs): IDENTICAL trigger bars, invalidation anchored at the
    sweep candle's body extreme (min(open,close) for longs,
    max(open,close) for shorts) +/- 0.15 ATR, same 0.25 ATR harness
    buffer.  The anchor point is the ONLY change.
Variant B2 (pure body): invalidation at the body extreme exactly, zero
    buffer anywhere (ExitConfig(stop_buffer_atr=0.0)).

Everything else identical across variants: next-bar-open entry (harness
convention), Phase-2 exits (50% partial at first pool >= 1R, trail after
+1.5R, 72-bar horizon), $0.45/oz cost.  The detector's non-score gates
(daily veto, room-to-pool, stop-distance band) are evaluated against
each variant's own invalidation, so the gates see the stop the trade
would actually carry.

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
from .engulf_trigger import (_invalidation, _make_signal, _passes_gates,
                             _sweeps_in_order)

# ---------------------------------------------------------------- constants
#: Bars after a sweep in which the confirmation trigger may fire (Phase-2).
CONFIRM_WINDOW = 8
#: Detector's invalidation offset from the anchor, in ATR (kept constant
#: across A and B so the anchor point is the only change).
INV_BUFFER_ATR = 0.15

VARIANTS = ("A", "B", "B2")


@dataclass
class TriggerHit:
    """One variant's trigger on one sweep event."""
    signal: GoldenSignal
    sweep: SweepEvent
    trigger_bar: int
    lag_bars: int  # trigger_bar - sweep.index


def body_anchor(o: np.ndarray, c: np.ndarray, sw_index: int,
                d: int) -> float:
    """The sweep candle's body extreme on the stop side.

    Longs (d=+1, swept the lows): the stop sits below, so the anchor is
    the body bottom, min(open, close) of the sweep bar.  Shorts (d=-1):
    the body top, max(open, close).  Pure function of the sweep bar,
    which is always in the past at decision time.
    """
    if d == 1:
        return float(min(o[sw_index], c[sw_index]))
    return float(max(o[sw_index], c[sw_index]))


def body_invalidation(o: np.ndarray, c: np.ndarray, sw: SweepEvent,
                      atr: float, d: int) -> float:
    """Variant B invalidation: body anchor +/- the detector's 0.15 ATR
    offset -- same structure as A, different anchor point."""
    anchor = body_anchor(o, c, sw.index, d)
    return anchor - INV_BUFFER_ATR * atr if d == 1 \
        else anchor + INV_BUFFER_ATR * atr


def variant_invalidation(o: np.ndarray, c: np.ndarray, sw: SweepEvent,
                         atr: float, d: int, mode: str) -> float:
    """The structural invalidation each variant carries."""
    assert mode in VARIANTS
    if mode == "A":
        return _invalidation(sw, atr, d)
    if mode == "B":
        return body_invalidation(o, c, sw, atr, d)
    # B2: the pure body extreme, no offset at all.
    return body_anchor(o, c, sw.index, d)


def variant_exits(mode: str) -> ExitConfig:
    """B2 runs with zero stop buffer; A and B use the repo default."""
    assert mode in VARIANTS
    if mode == "B2":
        return ExitConfig(stop_buffer_atr=0.0)
    return ExitConfig()


def scan_variant(df: pd.DataFrame, states, det: GoldenSwingDetector,
                 mode: str) -> list[TriggerHit]:
    """First confirmation-trigger bar per sweep, with each variant's own
    invalidation evaluated by the gates.  Fully causal."""
    assert mode in VARIANTS
    o = df["open"].to_numpy(dtype=float)
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
            inv = variant_invalidation(o, c, sw, atr, d, mode)
            if det._sweep_confirmed(s, d) and \
                    _passes_gates(det, s, d, sw, inv):
                hits.append(TriggerHit(
                    signal=_make_signal(s, d, sw, inv), sweep=sw,
                    trigger_bar=t, lag_bars=t - sw.index))
                break
    return hits


def simulate_hits(df: pd.DataFrame, hits: list[TriggerHit],
                  exits: ExitConfig) -> list[tuple[Outcome, TriggerHit]]:
    out: list[tuple[Outcome, TriggerHit]] = []
    for hit in hits:
        oc = simulate_signal_v2(df, hit.signal, exits)
        if oc is not None:
            out.append((oc, hit))
    return out


def exit_mix(outcomes: list[Outcome]) -> dict:
    """Exit-reason counts plus stop-out and wick-through rates."""
    n = len(outcomes)
    mix = {"stop": 0, "trail": 0, "timeout": 0, "other": 0}
    wicked = 0
    for oc in outcomes:
        mix[oc.exit_reason if oc.exit_reason in mix else "other"] += 1
        wicked += int(oc.wicked_inv)
    return {
        "n": n,
        "stop": mix["stop"],
        "trail": mix["trail"],
        "timeout": mix["timeout"],
        "other": mix["other"],
        "stop_rate": mix["stop"] / n if n else 0.0,
        "wicked_inv_rate": wicked / n if n else 0.0,
    }


def _years(df: pd.DataFrame) -> float:
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    return span_days / 365.25


def run_experiment(df: pd.DataFrame | None = None,
                   refresh: bool = False) -> dict:
    """Full pipeline: read -> featurize -> scan A/B/B2 -> simulate."""
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
        exits = variant_exits(mode)
        hits = scan_variant(df, states, det, mode)
        sims = simulate_hits(df, hits, exits)
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
            "exit_mix": exit_mix(outcomes),
        }
    # ---- anchor validation: variant A must match engulf_trigger's A ----
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
    return res
