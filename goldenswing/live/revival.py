"""Revived strategies — one source of truth for all shadow/candidate variants.

SOURCE: ``ABANDONED_STRATEGIES_VERIFIED.md`` (commit 13b3031). Nothing here
is a new invention: each variant reuses the ORIGINAL experiment's pure
helper (OTE fib, engulfing, body anchor, body-break, retest edge,
breaker, Marco sweep-entry) and wraps it in the CURRENT rules:

- Hard veto (``check_hard_veto`` / ``check_vetoes``): Intraday H4 + D1
  veto, Scalp H1 + H4 veto (D1 second), Swing D1 + W1 veto. Veto only
  on STRONG disagreement, flat = no veto. No veto = no revive.
- Real-trader logic: sweep -> displacement -> retracement -> LIMIT AT
  POI. Gap out = skip. No retracement = no trade. AMD Manipulation ->
  Distribution only. Structural stop, opposing-liquidity target.
- Exness timing: $12, 0.01 lot, 1:2000/unlimited, spread 0.2,
  commission 0. No R-grind, no 1% risk.

VARIANTS (``variant`` param on ``run_exness_backtest``):
- ``base`` — existing Exness timing logic, unchanged.
- ``ote`` — POI = OTE 62-79% retracement of the sweep->displacement
  leg (ICT MMXM, ``experiments/ote_entry.py``). LIMIT at the OTE 70%
  mid. Same sweep/displacement/retracement/exit machinery otherwise.
- ``engulf`` — trigger ALSO requires an engulfing close in the trade
  direction (``experiments/candles.py::engulfing``) on the displacement
  bar or the bar before it. Tsutsumi variant B, but with LIMIT at POI
  (not market at engulfing close) and hard veto.
- ``retest`` — LIMIT at the POI near EDGE (not mid), stop = far edge +
  buffer (JeaFx retest-entry, ``experiments/retest_entry.py``). Same
  sweep/displacement/BOS otherwise. Tests whether edge fill beats mid.
- ``breaker_t3`` — Placement T3 (ICT breaker): the retest TOUCHES the
  zone, then CLOSES THROUGH it against the thesis within the patience
  window. Trade the continuation THROUGH the zone (direction flips).
  Veto is re-checked against the BREAKER direction (a breaker is
  counter-bias by construction; it must not fight the veto TF).
- ``marco`` — Marco sweep-entry: enter at next-bar open after the sweep,
  NO displacement/retracement wait. Stop beyond the swept level +
  0.25 ATR. Target = opposing pool. HIGH RISK, Scalp only. The
  confirmation pillar is deliberately skipped here — that is the
  experiment, not an accident.
- ``body_stop`` — stop = sweep-candle BODY extreme +/- buffer
  (``experiments/body_stop.py``), not the wick extreme. Trigger/entry/
  target identical to base. Tests whether the tighter stop helps or
  just stops out more.
- ``bodybreak`` — displacement confirmation REPLACED by close-through
  of the sweep candle's BODY (``experiments/bodybreak_confirm.py``)
  within 10 bars. No displacement requirement. Same gates otherwise.

GODCE (``godce_filter=True``): regime filter ONLY, never an entry.
Uses the real GODCE router (``godce/router.py``) on a MINIMAL feat
frame (regime from the bias TF, volume/session OK, no caution). Full
GODCE indicators are NOT computed here, so the filter only blocks in
MOM mode against the regime — conservative by design. A full-indicator
GODCE filter is future work, not claimed here.

NOT REVIVED (deliberate, see the verified report):
- ``opening_range`` for XAU (QQQ only), Bollinger / divergence /
  candles / Ichimoku as standalone triggers (context only), ``min_rr``
  as an R filter (Exness uses $ move >= $1 instead).
- ``choch_fvg`` / ``trendline_claims`` are already live via playbooks;
  this module does NOT duplicate them. ``imbalance`` is an FVG audit,
  not a new entry.

All helpers here are pure functions of closed bars only. No lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..experiments.body_stop import body_anchor as _body_anchor
from ..experiments.bodybreak_confirm import body_break_trigger as _body_break
from ..experiments.candles import engulfing as _engulfing

# ------------------------------------------------------------------ OTE
OTE_NEAR, OTE_FAR, OTE_MID = 0.62, 0.79, 0.70


def ote_zone(sweep_extreme: float, disp_extreme: float,
             direction: int) -> tuple[float, float, float]:
    """OTE 62-79% retracement zone of the sweep->displacement leg.

    Long: leg = sweep low -> displacement high. Retracement back down
    into 62-79% of the leg is the OTE zone. Short: mirrored.
    Returns (zone_lo, zone_hi, limit_70pct). Pure, closed bars only.
    """
    lo = min(sweep_extreme, disp_extreme)
    hi = max(sweep_extreme, disp_extreme)
    span = hi - lo
    if span <= 0:
        raise ValueError("OTE leg must have positive span")
    if direction == 1:
        # retracement from the high back toward the low
        zone_hi = hi - OTE_NEAR * span
        zone_lo = hi - OTE_FAR * span
        limit = hi - OTE_MID * span
    else:
        zone_lo = lo + OTE_NEAR * span
        zone_hi = lo + OTE_FAR * span
        limit = lo + OTE_MID * span
    return float(zone_lo), float(zone_hi), float(limit)


# ------------------------------------------------------------------ engulf
def has_engulfing(o: np.ndarray, h: np.ndarray, l: np.ndarray,
                  c: np.ndarray, t: int, direction: int) -> bool:
    """Engulfing close in ``direction`` at bar ``t`` or ``t-1``.

    Uses the original ``experiments/candles.py::engulfing`` (body of bar
    engulfs prior body, opposite colours). Causal: bars <= t only.
    """
    for k in (t, t - 1):
        if k < 1 or k >= len(c):
            continue
        if _engulfing(o, h, l, c, k) == direction:
            return True
    return False


# ------------------------------------------------------------------ retest edge
def retest_edge_limit(poi_top: float, poi_bottom: float,
                      direction: int) -> tuple[float, float]:
    """Retest variant: LIMIT at the near edge, stop = far edge + buffer.

    Long: near edge = POI top (first touched from above on retracement),
    far edge = POI bottom. Short: mirrored. Buffer is applied by the
    caller (0.1 ATR in the Exness loop) so this stays a pure level fn.
    Returns (limit_price, far_edge).
    """
    if direction == 1:
        return float(poi_top), float(poi_bottom)
    return float(poi_bottom), float(poi_top)


# ------------------------------------------------------------------ breaker (T3)
def find_breaker(o: np.ndarray, h: np.ndarray, l: np.ndarray,
                 c: np.ndarray, start: int, end: int,
                 poi_top: float, poi_bottom: float,
                 thesis_direction: int) -> dict | None:
    """Placement T3: touch-then-close-through against the thesis.

    Scan bars (start, end]: first a wick touch of the zone (the retest
    happens), then a CLOSE through the zone against the thesis (the
    zone is consumed). Returns {violation_idx, direction, entry} with
    direction = -thesis_direction, entry = close of the violation bar,
    or None. Causal: only bars <= end are read, and the caller passes
    end <= current bar. The violation bar itself is the entry bar, so
    the exit scan must start at violation_idx + 1 (the caller does).
    """
    n = len(c)
    end = min(end, n - 1)
    touch = None
    for j in range(start + 1, end + 1):
        if thesis_direction == 1 and l[j] <= poi_top:
            touch = j
            break
        if thesis_direction == -1 and h[j] >= poi_bottom:
            touch = j
            break
    if touch is None:
        return None
    for j in range(touch + 1, end + 1):
        if thesis_direction == 1 and c[j] < poi_bottom:
            return {"violation_idx": int(j), "direction": -1,
                    "entry": float(c[j])}
        if thesis_direction == -1 and c[j] > poi_top:
            return {"violation_idx": int(j), "direction": 1,
                    "entry": float(c[j])}
    return None


# ------------------------------------------------------------------ marco
def marco_entry(open_next: float, sweep_extreme: float, direction: int,
                atr: float) -> tuple[float, float]:
    """Marco sweep-entry: entry = next-bar open, stop beyond the level.

    Stop = swept level +/- 0.25 ATR buffer ("covering the level").
    Returns (entry, stop). The caller still applies margin, target,
    and exit-scan rules; only the confirmation wait is skipped.
    """
    stop = (sweep_extreme - 0.25 * atr if direction == 1
            else sweep_extreme + 0.25 * atr)
    return float(open_next), float(stop)


# ------------------------------------------------------------------ body stop / bodybreak
def body_stop_level(o: np.ndarray, c: np.ndarray, sweep_bar_idx: int,
                    direction: int, atr: float) -> float:
    """Body-anchored stop: sweep-candle body extreme +/- 0.15 ATR.

    Reuses ``experiments/body_stop.py::body_anchor`` (the original).
    ``sweep_bar_idx`` is the ABSOLUTE trigger-frame index of the sweep
    bar. Long: body bottom - 0.15 ATR. Short: body top + 0.15 ATR.
    """
    anchor = _body_anchor(o, c, int(sweep_bar_idx), int(direction))
    return float(anchor - 0.15 * atr if direction == 1
                 else anchor + 0.15 * atr)


def has_bodybreak(o: np.ndarray, c: np.ndarray, sweep_bar_idx: int,
                  start: int, end: int, direction: int) -> bool:
    """Close-through of the sweep candle's BODY in ``direction``.

    Reuses ``experiments/bodybreak_confirm.py::body_break_trigger``.
    Scan bars (start, end] — the caller bounds end <= current bar.
    """
    for k in range(start + 1, end + 1):
        if k >= len(c):
            break
        if _body_break(o, c, int(sweep_bar_idx), int(k), int(direction)):
            return True
    return False


# ------------------------------------------------------------------ GODCE regime filter
def godce_allow(direction: int, bias_bars: pd.DataFrame) -> tuple[bool, str]:
    """GODCE regime filter: allow/block one direction. Filter ONLY.

    Builds a MINIMAL feat frame (regime from the bias TF via the same
    swing logic the core engine uses, volume/session OK, no caution)
    and runs the REAL GODCE router (``compute_candidate_mode`` +
    ``apply_dwell`` + ``mode_rules``). In MOM mode the router only
    allows the regime side; every other mode allows both sides, so
    this filter only blocks counter-regime MOM trades. Full GODCE
    indicators are NOT computed — that is a documented limitation,
    not a hidden approximation. Returns (allowed, mode_name).
    """
    try:
        from ..godce.params import DEFAULT, REGIME_TREND_BEAR, REGIME_TREND_BULL
        from ..godce.router import (apply_dwell, compute_candidate_mode,
                                     mode_rules)
        from ..live.multitf import evaluate_bias
    except Exception as exc:  # pragma: no cover - fail-open, never block
        return True, f"godce_unavailable:{exc}"
    if bias_bars is None or len(bias_bars) < 10:
        return True, "godce_no_data"
    bias = evaluate_bias(bias_bars, swing_n=3)
    regime = (REGIME_TREND_BULL if bias == 1 else
              REGIME_TREND_BEAR if bias == -1 else 0)
    n = len(bias_bars)
    feat = pd.DataFrame({
        "volOk": [True] * n, "sessionOk": [True] * n,
        "cautionMode": [False] * n, "squeeze30": [False] * n,
        "adxUp": [True] * n, "regime": [regime] * n,
        "atrRatio": [1.0] * n, "vBotTrig": [False] * n,
        "vTopTrig": [False] * n, "vBotBreak": [False] * n,
        "vTopBreak": [False] * n, "patBreakLong": [False] * n,
        "patBreakShort": [False] * n, "asiaBreakLong": [False] * n,
        "asiaBreakShort": [False] * n, "adxOk": [True] * n,
    }, index=bias_bars.index)
    cand = compute_candidate_mode(feat, DEFAULT)
    modes = apply_dwell(cand, DEFAULT)
    mode = int(modes.iloc[-1])
    regime_bull = regime == REGIME_TREND_BULL
    regime_bear = regime == REGIME_TREND_BEAR
    aL, aS, _ctr, _f1, _f2, _f3, _cap, _run, _off = mode_rules(
        mode, 1.5, 3.0, regime_bull, regime_bear)
    mode_names = {0: "FLAT", 1: "MOM", 2: "MEAN", 3: "POST", 4: "BRK",
                  5: "LEG"}
    name = mode_names.get(mode, f"mode_{mode}")
    allowed = bool(aL if direction == 1 else aS)
    return allowed, name


VARIANTS = ("base", "ote", "engulf", "retest", "breaker_t3", "marco",
            "body_stop", "bodybreak")
