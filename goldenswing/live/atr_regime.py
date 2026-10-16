"""Dynamic ATR Regime: the engine adapts to volatility,
never static thresholds.

Baseline ATR: rolling median H4 ATR14 over the last 100 H4
bars.  Current ATR: H4 ATR14 now.  Ratio = current /
baseline.

Volatility regimes (ratio):
- NORMAL: < 1.25
- ELEVATED: 1.25-1.5
- HIGH: > 1.5
- EXTREME: > 2.0

All ATR-based thresholds scale with CURRENT ATR, never
fixed $: POI band = 0.5 x current ATR, SL buffer = 0.25 x
current ATR, sweep wick >= 0.25 x current ATR, displacement
>= 1.5 x current ATR, FVG >= 0.25 x current ATR.  No
hardcoded $ stops.

HIGH volatility (> 1.5x) tightens the demand:
- Displacement threshold raises 1.5 -> 1.75 x current ATR
  (demand > 1.75x to separate institutional from retail
  panic).
- Volume validation raises 1.5x -> 2.0x average.
- Confirmation window M15 -> M5 fast path.
- POI band / SL buffer stay 0.5x / 0.25x current ATR
  (wider in $, same in ATR).

PRE-ENTRY (setup not yet filled), baseline $15 -> $25,
mid-risk entry:
- SL buffer = 0.25 x current ATR = 0.25 x $25 = $6.25
  (vs $3.75 at baseline).  SL = min(sweep, FVG, CHoCH) -
  $6.25 (long).  SL distance = entry-to-sweep + buffer.
  Size = risk / SL distance, so lots get SMALLER (risk
  fixed, wider SL -> fewer lots).
- Time-stop = baseline_time_stop / ratio, clamped:
  8 / 1.67 = 4.8 -> 5 H1 bars (mid-risk).  GREEN 12 ->
  7 H1.  RED 6 -> 4 H1.  Never below 3 H1, never above
  baseline.
- TPs in R stay the same (R is SL distance, R scales
  with ATR), but in $ they are wider.  TP before an
  opposing zone still respects opposing - 0.1 ATR buffer
  (buffer also scales with current ATR).
- Displacement 1.75x current ATR, volume >= 2.0x avg,
  M5 fast path.

IN-TRADE (already open, ATR jumps):
- SL NEVER widens / NEVER moves against the trade.  The
  original SL stays at its entry-time buffer.  Do NOT move
  SL further away to chase volatility.
- Time-stop SHORTENS dynamically: remaining = max(1,
  new_total - elapsed).  If elapsed >= new total, exit at
  the next tick.
- Trailing TIGHTENS: trailing step 0.5 ATR -> 0.25 ATR
  (current ATR), trail behind M5 structure (not H1).
- TPs stay (do not move TPs further away on an ATR jump).
- Breakeven EARLY: if ATR HIGH and the trade is +0.5R or
  more, move SL to breakeven early (not waiting +1R).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NORMAL_MAX = 1.25
ELEVATED_MAX = 1.5
HIGH_MAX = 2.0
BASELINE_BARS = 100
ATR_PERIOD = 14
DISP_MIN_NORMAL = 1.5
DISP_MIN_HIGH = 1.75
VOLUME_SPIKE_NORMAL = 1.5
VOLUME_SPIKE_HIGH = 2.0
TIME_STOP_MIN_H1 = 3
EARLY_BREAKEVEN_R = 0.5


def atr_series(frame) -> pd.Series:
    """Per-bar ATR14 series (true range, 14-bar mean)."""
    if frame is None or len(frame) < 2:
        return pd.Series(dtype=float)
    h = frame["high"].to_numpy(dtype=float)
    l = frame["low"].to_numpy(dtype=float)
    c = frame["close"].to_numpy(dtype=float)
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c),
                                      np.abs(l - prev_c)))
    return pd.Series(tr, index=frame.index).rolling(
        ATR_PERIOD).mean()


def baseline_atr(h4_frame) -> float:
    """Rolling median H4 ATR14 over the last 100 H4 bars."""
    s = atr_series(h4_frame).dropna()
    if len(s) == 0:
        return 0.0
    return float(s.tail(BASELINE_BARS).median())


def atr_regime(current_atr: float, baseline: float) -> dict:
    """Volatility regime from current vs baseline ATR."""
    if baseline <= 0 or current_atr <= 0:
        return {"ratio": 0.0, "regime": "NORMAL",
                "current": current_atr, "baseline": baseline,
                "log": "ATR regime: NORMAL (no baseline)"}
    ratio = current_atr / baseline
    if ratio > HIGH_MAX:
        regime = "EXTREME"
    elif ratio > ELEVATED_MAX:
        regime = "HIGH"
    elif ratio >= NORMAL_MAX:
        regime = "ELEVATED"
    else:
        regime = "NORMAL"
    return {"ratio": ratio, "regime": regime,
            "current": current_atr, "baseline": baseline,
            "log": (f"ATR regime: {regime}, ratio {ratio:.2f}x, "
                    f"current ${current_atr:.2f} vs baseline "
                    f"${baseline:.2f}")}


def scaled_thresholds(regime: str) -> dict:
    """HIGH/EXTREME demand more: displacement 1.75x, volume
    2.0x, M5 fast path.  NORMAL/ELEVATED keep 1.5x / 1.5x /
    M15."""
    high = regime in ("HIGH", "EXTREME")
    return {"disp_min_atr": DISP_MIN_HIGH if high
            else DISP_MIN_NORMAL,
            "volume_spike_min": VOLUME_SPIKE_HIGH if high
            else VOLUME_SPIKE_NORMAL,
            "fast_path": high}


def scaled_time_stop(baseline_h1: int, ratio: float) -> int:
    """Time-stop = baseline / ratio, clamped: never below 3
    H1, never above baseline.  8 / 1.67 -> 5, 12 / 1.67 ->
    7, 6 / 1.67 -> 4."""
    if ratio <= 1.0 or baseline_h1 <= 0:
        return int(baseline_h1)
    scaled = int(round(baseline_h1 / ratio))
    return max(TIME_STOP_MIN_H1, min(int(baseline_h1), scaled))


def remaining_time_stop(elapsed_h1: int, new_total_h1: int) -> int:
    """Remaining time-stop after an in-trade ATR jump: max(1,
    new_total - elapsed).  0 means the stop already passed
    the new total -> exit at the next tick."""
    if elapsed_h1 >= new_total_h1:
        return 0
    return max(1, int(new_total_h1 - elapsed_h1))


def apply_in_trade_atr(plan: dict, *, ratio: float,
                       current_r: float, elapsed_h1: int,
                       entry_sl: float) -> dict:
    """In-trade ATR-jump rules applied to a management-plan
    dict (from ``plan_management`` / ``asdict``).

    - SL NEVER widens: ``sl_locked`` keeps the original SL;
      the plan's SL is never moved further away.
    - Time-stop shortens: remaining = max(1, new_total -
      elapsed); 0 -> exit at the next tick.
    - Trailing tightens in HIGH: step 0.25 ATR, M5 structure.
    - Early breakeven: HIGH + >= +0.5R -> breakeven now.
    Returns the adjusted plan dict (never mutates input).
    """
    out = dict(plan)
    out["sl_locked"] = True
    out["sl"] = entry_sl
    regime = "HIGH" if ratio > ELEVATED_MAX else (
        "ELEVATED" if ratio >= NORMAL_MAX else "NORMAL")
    notes = []
    if ratio > 1.0:
        new_total = scaled_time_stop(
            int(plan.get("time_stop_h1", 12)), ratio)
        out["time_stop_h1"] = new_total
        remaining = remaining_time_stop(elapsed_h1, new_total)
        out["time_stop_remaining_h1"] = remaining
        if remaining == 0:
            out["exit_now"] = True
            notes.append("time-stop expired on ATR jump")
        else:
            notes.append(f"ATR HIGH in-trade, time-stop "
                         f"shortened, remaining {remaining} H1")
    if regime == "HIGH" and current_r >= EARLY_BREAKEVEN_R:
        out["breakeven_now"] = True
        notes.append("ATR HIGH, early breakeven at +0.5R")
    if regime == "HIGH":
        out["trail_step_atr"] = 0.25
        out["trail_tf"] = "M5"
        notes.append("trailing tightened: 0.25 ATR, M5 structure")
    out["in_trade_atr_log"] = ("ATR jumped in-trade, SL locked, "
                               "never widens"
                               + ("; " + "; ".join(notes)
                                  if notes else ""))
    return out
