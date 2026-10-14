"""Feature-layer configuration.  Defaults compute every family; the
detector decides independently which families it *scores* with."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureConfig:
    # compute switches
    bollinger: bool = True
    fib: bool = True
    ichimoku: bool = True
    relstr: bool = True
    patterns: bool = True
    zones: bool = True
    phases: bool = True
    footprints: bool = True
    volume: bool = True
    mtf: bool = True
    trendlines: bool = True
    divergence: bool = True

    # Bollinger
    bb_period: int = 20
    bb_mult: float = 2.0
    bb_pct_window: int = 252
    bb_squeeze_pct: float = 20.0

    # Fibonacci
    fib_at_tol_atr: float = 0.15

    # Ichimoku
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_b_period: int = 52
    displacement: int = 26          # standard cloud shift

    # Relative strength
    rs_roc_bars: int = 24
    rs_deadband: float = 0.002      # ratio ROC deadband (|roc| below = flat)
    rs_pct_window: int = 252

    # Classical patterns (distances in ATR units, ATR from the reader)
    pat_tol_atr: float = 0.30       # double-top/bottom twin tolerance
    pat_shoulder_tol_atr: float = 0.50
    pat_shoulder_frac: float = 0.35  # of head height above neckline
    pat_min_depth_atr: float = 1.0   # head above neckline, min
    pat_min_valley_atr: float = 0.5  # double-top valley depth, min
    pat_max_age: int = 150          # bars a formed pattern stays live

    # Supply / demand zones
    sd_base_max_bars: int = 3
    sd_base_body_atr: float = 0.5   # max |body|/ATR for a base candle
    sd_depart_body_atr: float = 1.1  # min |body|/ATR for the departure
    sd_pre_bars: int = 5            # context bars for drop/rally into base
    sd_max_age: int = 400
    sd_max_zones: int = 24

    # Phase (AMD) state machine
    amd_range_bars: int = 48
    amd_base_width_atr: float = 5.0   # trailing range width to count as base
    amd_base_drift_atr: float = 2.0   # max net drift across the range
    amd_prior_bars: int = 48
    amd_prior_move_atr: float = 1.5   # min prior move to label accum/dist
    amd_exit_buffer_atr: float = 0.3  # close beyond range extreme to exit
    amd_manip_bars: int = 10          # manipulation episode memory
    amd_expansion_bars: int = 24      # expansion -> trend relabel delay
    # Missed-manipulation fallback (phases.py): when the direct sweep
    # check misses (sweep_age > 0, or the sweep was of an interior
    # pool), BOS + recent same-direction sweep + fresh FVG together
    # infer manipulation. All three legs required — a standing pool
    # alone was measured to fire on noise (72 bars vs 18 sweep-leg
    # bars per 3000 H1 bars), so pools are NOT a fallback leg.
    amd_fallback_bos_age: int = 3     # max age of the BOS/CHoCH event
    amd_fallback_sweep_age: int = 10  # max age of the same-direction sweep
    amd_fallback_fvg_age: int = 10    # max age of the fresh FVG
    amd_fallback_confidence: float = 0.55  # inferred, below direct 0.80
    amd_direct_confidence: float = 0.80    # witnessed sweep
    amd_base_confidence: float = 0.60      # directly observed range
    amd_expansion_confidence: float = 0.70  # exit + displacement

    # Volume
    vol_window: int = 20
    vol_climax_rvol: float = 2.0
    vol_dry_rvol: float = 0.6

    # Footprints
    fp_sweep_window: int = 6
    fp_inst_rvol: float = 1.6       # volume climax at the displacement bar
    fp_inst_max_retrace_atr: float = 0.5
    fp_chase_body_atr: float = 1.2
    fp_chase_momentum_atr: float = 2.5
    fp_chase_rvol: float = 1.5
    fp_trap_bars: int = 3
    fp_cta_lookback: int = 10
    fp_cta_min_side: int = 8        # of lookback closes on one VWAP side

    # Trendlines (Phase 4)
    tl_min_sep: int = 5             # min bars between the two anchors
    tl_tol_atr: float = 0.20        # touch / violation tolerance (ATR)
    tl_slope_min_atr: float = 0.002  # per-bar slope bounds (ATR units):
    tl_slope_max_atr: float = 0.35   # near-flat and near-vertical excluded
    tl_max_age: int = 300           # bars a line may live
    tl_max_drift_atr: float = 6.0   # expire when line is this far away
    tl_max_lines: int = 6           # live lines kept per direction
    tl_rearm_bars: int = 2          # bars price must spend away (> tol)
                                    # before another touch can register

    # AOC — area of concentration (Phase 4.5)
    aoc: bool = True                # compute switch (engine)
    aoc_tol_atr: float = 0.35       # cluster band: levels this close
                                    # (in ATR) belong to one AOC
    aoc_min_families: int = 2       # distinct families for an AOC
    aoc_major_families: int = 3     # distinct families for a MAJOR
                                    # pivot zone (patterns major test)
    aoc_max_zones: int = 8          # top-N clusters kept per bar
    aoc_recency_half_life: float = 96.0   # bars for the recency decay
                                          # to halve (floored at 0.5)

    # RSI divergence (mined 2026-10-11, TradingLab video)
    div_rsi_period: int = 14       # video uses 9; 14 is the classic
    div_pivot_left: int = 3        # fractal left bars
    div_pivot_right: int = 2       # fractal right bars = confirmation lag
    div_min_sep: int = 5           # min bars between the two pivots
    div_max_sep: int = 120         # max bars between the two pivots
    div_min_drsi: float = 2.0      # min |RSI_b - RSI_a| in RSI points
    div_max_age: int = 30          # bars a confirmed divergence stays live
