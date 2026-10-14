"""Independent per-mode harness runs + the swing routing study.

Independence contract: every run builds its OWN data frame, reader
pass, feature attach and detector instance.  Nothing mutable is
shared between modes, and the routing study's OFF and ON runs are
separate reads too — the only thing that crosses from OFF to ON is
the measured RouteTable itself, which is the method, not leakage.

The swing run uses the canonical Phase 1–4 caches (``load_gold_h1`` /
``load_peers``) so the Combo A book must reproduce VALIDATION.md
exactly (+25.62% / PF 1.22 / 199 trades); the sub-hourly modes fetch
their own 60-day caches via modes.data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.data import load_gold_h1, load_peers
from ..backtest.harness import (ExitConfig, book_metrics, build_book,
                                simulate_signal_v2, slice_stats)
from ..features.engine import read_and_featurize
from ..golden.detector import GoldenSignal, GoldenSwingDetector
from ..reader.state import MarketState
from .config import SWING, ModeConfig
from .data import load_mode_universe
from .routing import RouteTable, build_matrix, derive_route_table
from .simulate import simulate_scalp


def signal_passes_mode(mode: ModeConfig, state: MarketState,
                       sig: GoldenSignal) -> bool:
    """The mode's entry filters, applied to a scanned signal.

    Pure function of (mode, state at the signal bar, signal) — the
    killzone tag, setup kind and displacement strength are all known
    at the signal bar's close, so filtering leaks nothing.
    """
    if mode.killzone_only and not state.killzone:
        return False
    if mode.allowed_kinds is not None and sig.kind not in mode.allowed_kinds:
        return False
    if mode.min_displacement > 0.0:
        disp = state.bull_disp_3 if sig.direction == 1 else state.bear_disp_3
        if disp < mode.min_displacement:
            return False
    return True


def _years(df: pd.DataFrame) -> float:
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    return span_days / 365.25


def _simulate(mode: ModeConfig, df: pd.DataFrame, signals, cost: float):
    out = []
    for sig in signals:
        if mode.exit_style == "swing_v2":
            o = simulate_signal_v2(df, sig, ExitConfig(cost=cost))
        else:
            o = simulate_scalp(df, sig, horizon=mode.horizon, cost=cost,
                               stop_buffer_atr=mode.stop_buffer_atr)
        if o is not None:
            out.append(o)
    return out


@dataclass
class ModeRun:
    mode: ModeConfig
    bars: int
    start: str
    end: str
    years: float
    peers_available: dict = field(default_factory=dict)
    n_scanned: int = 0        # signals out of the detector scan
    n_filtered: int = 0       # after the mode's entry filters
    n_simulated: int = 0      # outcomes actually produced
    book: dict = field(default_factory=dict)          # @ threshold, real cost
    book_zero_cost: dict = field(default_factory=dict)
    shadow: dict = field(default_factory=dict)        # all outcomes
    shadow60: dict = field(default_factory=dict)
    cost_r_mean: float | None = None    # scalp modes only (per-outcome)
    cost_r_median: float | None = None
    cost_r_p90: float | None = None
    exit_mix: dict = field(default_factory=dict)


def run_mode(mode: ModeConfig, refresh: bool = False) -> ModeRun:
    if mode.interval == "1h":
        df = load_gold_h1(refresh=refresh)
        peers = load_peers(refresh=refresh)
    else:
        df, peers = load_mode_universe(mode.interval, mode.period,
                                       refresh=refresh)
    years = _years(df)
    states = read_and_featurize(df, peers)
    detector = GoldenSwingDetector(mode.detector_config())
    scanned = detector.scan(states)
    signals = [s for s in scanned
               if signal_passes_mode(mode, states[s.index], s)]

    outcomes = _simulate(mode, df, signals, mode.cost_per_oz)
    outcomes0 = _simulate(mode, df, signals, 0.0)

    run = ModeRun(
        mode=mode, bars=len(df), start=str(df.index[0]),
        end=str(df.index[-1]), years=years,
        peers_available={k: bool(v is not None and len(v) > 100)
                         for k, v in (peers or {}).items()},
        n_scanned=len(scanned), n_filtered=len(signals),
        n_simulated=len(outcomes),
        book=book_metrics(build_book(outcomes, mode.threshold), years),
        book_zero_cost=book_metrics(
            build_book(outcomes0, mode.threshold), years),
        shadow=slice_stats(outcomes),
        shadow60=slice_stats([o for o in outcomes
                              if o.signal.score >= mode.threshold]),
    )
    if outcomes and hasattr(outcomes[0], "cost_r"):
        crs = np.array([o.cost_r for o in outcomes])
        run.cost_r_mean = float(crs.mean())
        run.cost_r_median = float(np.median(crs))
        run.cost_r_p90 = float(np.percentile(crs, 90))
        book = build_book(outcomes, mode.threshold)
        if book:
            breasons = [o.exit_reason for o in book]
            run.exit_mix = {r: 100.0 * breasons.count(r) / len(breasons)
                            for r in ("target", "stop", "timeout")
                            if breasons.count(r)}
    return run


# ------------------------------------------------------------ routing study
@dataclass
class RoutingStudy:
    bars: int
    start: str
    end: str
    years: float
    cells: dict = field(default_factory=dict)
    table: RouteTable | None = None
    n_signals_off: int = 0
    n_signals_on: int = 0
    off_book60: dict = field(default_factory=dict)
    off_book50: dict = field(default_factory=dict)
    off_shadow60: dict = field(default_factory=dict)
    on_book60: dict = field(default_factory=dict)
    on_book50: dict = field(default_factory=dict)
    on_shadow60: dict = field(default_factory=dict)


def run_swing_routing(refresh: bool = False) -> RoutingStudy:
    """OFF run (Combo A, no routing) → matrix from its shadow book →
    route table → ON run (fresh read, same config + table)."""
    df = load_gold_h1(refresh=refresh)
    peers = load_peers(refresh=refresh)
    years = _years(df)

    # OFF — an independent read, exactly the validated Combo A run.
    states_off = read_and_featurize(df, peers)
    det_off = GoldenSwingDetector(SWING.detector_config())
    signals_off = det_off.scan(states_off)
    outcomes_off = _simulate(SWING, df, signals_off, SWING.cost_per_oz)

    cells = build_matrix(outcomes_off, states_off)
    table = derive_route_table(cells, golden_threshold=SWING.threshold)

    # ON — a second, independent read; the ONLY difference is the
    # measured route table riding on the detector config.
    states_on = read_and_featurize(df, peers)
    det_on = GoldenSwingDetector(SWING.detector_config(route_table=table))
    signals_on = det_on.scan(states_on)
    outcomes_on = _simulate(SWING, df, signals_on, SWING.cost_per_oz)

    return RoutingStudy(
        bars=len(df), start=str(df.index[0]), end=str(df.index[-1]),
        years=years, cells=cells, table=table,
        n_signals_off=len(outcomes_off), n_signals_on=len(outcomes_on),
        off_book60=book_metrics(build_book(outcomes_off, 60.0), years),
        off_book50=book_metrics(build_book(outcomes_off, 50.0), years),
        off_shadow60=slice_stats([o for o in outcomes_off
                                  if o.signal.score >= 60.0]),
        on_book60=book_metrics(build_book(outcomes_on, 60.0), years),
        on_book50=book_metrics(build_book(outcomes_on, 50.0), years),
        on_shadow60=slice_stats([o for o in outcomes_on
                                 if o.signal.score >= 60.0]),
    )
