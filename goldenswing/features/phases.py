"""Phase detection — Wyckoff-flavoured AMD state machine.

STATES, not predictions.  The machine classifies what the market is
*doing*, from range behaviour + sweep activity + displacement:

- A **base** exists when the trailing ``amd_range_bars`` range is
  narrow (<= ``amd_base_width_atr`` ATR) with little net drift.  The
  range extremes freeze at base entry.
- The base is labelled by the move that led INTO it: after a decline
  -> **accumulation**, after a rise -> **distribution**, otherwise
  **consolidation**.
- A liquidity sweep through a base extreme that closes back inside is
  **manipulation** (a spring in accumulation / upthrust in
  distribution when the swept side matches the base's story).
  Manipulation resolves either by displacement out of the range
  (-> **expansion**) or by drifting back inside for
  ``amd_manip_bars`` (-> back to the base label).
- A close beyond a base extreme by >= ``amd_exit_buffer_atr`` with a
  displacement-strength body is a **phase exit** -> expansion, which
  matures into **trend_up / trend_down** after
  ``amd_expansion_bars``.  A fresh base can form out of a trend.

MISSED-MANIPULATION FALLBACK (owner's "if it missed it" rule).  The
direct manipulation check needs a sweep ON THIS BAR (``sweep_age == 0``)
whose extreme aligns with the frozen base extreme.  It misses when
the sweep was of an interior pool, or when the base froze after the
sweep.  In that case, while still in a base, BOS/CHoCH + a recent
same-direction sweep + a fresh same-direction FVG together infer
manipulation (event ``manipulation_fallback``, ``via="fallback"``,
confidence 0.55 vs 0.80 direct).  All three legs are required.  A
standing pool alone is NOT a leg — measured on 3000 real XAUUSD H1
bars, a pool leg fires on 72 noise bars vs 18 for the sweep leg,
because pools cluster at base extremes by construction.  Fallback
manipulation matures through the SAME exit logic as direct
manipulation, so there is still one state machine, and fallback can
never label distribution without displacement + exit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import ohlcv_arrays, rolling_max, rolling_min
from .config import FeatureConfig
from .state import PhaseState

BASE_LABELS = ("accumulation", "distribution", "consolidation")


def _fallback_manipulation(s, t: int, cfg: FeatureConfig):
    """Missed-sweep inference: BOS + recent sweep + fresh FVG.

    Returns ``(direction, evidence)`` when all three legs hold, else
    ``None``.  Every input is a causal fact already on the MarketState:
    the reader's structure event, last sweep, and active FVG list.
    """
    ev = s.last_structure_event
    if ev is None or ev.kind not in ("BOS", "CHOCH"):
        return None
    if s.structure_event_age is None \
            or s.structure_event_age > cfg.amd_fallback_bos_age:
        return None
    d = ev.direction
    sw = s.last_sweep
    if sw is None or s.sweep_age is None \
            or s.sweep_age > cfg.amd_fallback_sweep_age \
            or sw.direction != d:
        return None
    fresh = [g for g in s.active_fvgs
             if g.direction == d
             and (t - g.created_at) <= cfg.amd_fallback_fvg_age]
    if not fresh:
        return None
    evidence = (
        f"{ev.kind} {d:+d} {s.structure_event_age} bar(s) ago",
        f"sweep {d:+d} {s.sweep_age} bar(s) ago",
        f"fresh FVG {d:+d} ({len(fresh)} active)",
    )
    return d, evidence


def compute(df: pd.DataFrame, states, cfg: FeatureConfig) -> list[PhaseState]:
    a = ohlcv_arrays(df)
    high, low, close = a["high"], a["low"], a["close"]
    n = len(df)
    w = cfg.amd_range_bars
    r_hi = rolling_max(high, w)
    r_lo = rolling_min(low, w)

    phase = "unknown"
    phase_start = 0
    base_label = "consolidation"
    base_hi = base_lo = float("nan")
    manip_dir = 0
    manip_left = 0
    exp_left = 0
    # Provenance of the current phase: how it was entered, and the
    # entry-bar evidence.  Evidence is emitted only on the entry bar;
    # continuation bars carry via/confidence with evidence=().
    cur_via = ""
    cur_conf = 0.0
    cur_evidence: tuple = ()

    out: list[PhaseState] = []
    for t in range(n):
        s = states[t]
        atr = s.atr if s.atr > 0 else 1e-9
        event: str | None = None
        event_dir = 0

        base_candidate = False
        if t >= w:
            width = (r_hi[t] - r_lo[t]) / atr
            drift = abs(close[t] - close[t - w]) / atr
            base_candidate = width <= cfg.amd_base_width_atr \
                and drift <= cfg.amd_base_drift_atr

        def enter_base() -> None:
            nonlocal phase, phase_start, base_label, base_hi, base_lo
            nonlocal cur_via, cur_conf, cur_evidence
            prior = (close[t - 1] - close[t - 1 - cfg.amd_prior_bars]) / atr \
                if t - 1 - cfg.amd_prior_bars >= 0 else 0.0
            if prior <= -cfg.amd_prior_move_atr:
                base_label = "accumulation"
            elif prior >= cfg.amd_prior_move_atr:
                base_label = "distribution"
            else:
                base_label = "consolidation"
            base_hi, base_lo = float(r_hi[t]), float(r_lo[t])
            phase, phase_start = base_label, t
            cur_via, cur_conf = "direct", cfg.amd_base_confidence
            cur_evidence = (f"base {base_label}: width "
                            f"{(base_hi - base_lo) / atr:.1f} ATR",)

        def exit_base(direction: int, evt: str) -> None:
            nonlocal phase, phase_start, exp_left
            nonlocal cur_via, cur_conf, cur_evidence
            phase = "expansion_up" if direction == 1 else "expansion_down"
            phase_start = t
            exp_left = cfg.amd_expansion_bars
            cur_via, cur_conf = "direct", cfg.amd_expansion_confidence
            cur_evidence = (f"{evt} {direction:+d}",)

        in_base = phase in BASE_LABELS
        if phase == "unknown":
            if base_candidate:
                enter_base()
                event = "base_start"
        elif in_base:
            # manipulation: a sweep through a base extreme this bar
            sw = s.last_sweep
            if sw is not None and s.sweep_age == 0:
                if sw.direction == 1 and sw.extreme <= base_lo + 0.25 * atr:
                    phase, phase_start = "manipulation", t
                    manip_dir, manip_left = 1, cfg.amd_manip_bars
                    event = "spring" if base_label == "accumulation" \
                        else "manipulation_ssl"
                    event_dir = 1
                    cur_via, cur_conf = "direct", cfg.amd_direct_confidence
                    cur_evidence = (f"sweep {1:+d} at {sw.extreme:.1f}",)
                elif sw.direction == -1 and sw.extreme >= base_hi - 0.25 * atr:
                    phase, phase_start = "manipulation", t
                    manip_dir, manip_left = -1, cfg.amd_manip_bars
                    event = "upthrust" if base_label == "distribution" \
                        else "manipulation_bsl"
                    event_dir = -1
                    cur_via, cur_conf = "direct", cfg.amd_direct_confidence
                    cur_evidence = (f"sweep {-1:+d} at {sw.extreme:.1f}",)
            if phase in BASE_LABELS:   # no direct manipulation triggered
                # Missed-manipulation fallback: BOS + recent sweep + FVG.
                fb = _fallback_manipulation(s, t, cfg)
                if fb is not None:
                    d, evidence = fb
                    phase, phase_start = "manipulation", t
                    manip_dir, manip_left = d, cfg.amd_manip_bars
                    event, event_dir = "manipulation_fallback", d
                    cur_via, cur_conf = "fallback", \
                        cfg.amd_fallback_confidence
                    cur_evidence = evidence
            if phase in BASE_LABELS:   # no manipulation triggered
                buf = cfg.amd_exit_buffer_atr * atr
                if close[t] > base_hi + buf and s.bull_disp_3 >= 1.0:
                    exit_base(1, "phase_exit_up")
                    event, event_dir = "phase_exit_up", 1
                elif close[t] < base_lo - buf and s.bear_disp_3 >= 1.0:
                    exit_base(-1, "phase_exit_down")
                    event, event_dir = "phase_exit_down", -1
        elif phase == "manipulation":
            buf = cfg.amd_exit_buffer_atr * atr
            if close[t] > base_hi + buf and s.bull_disp_3 >= 1.0:
                exit_base(1, "phase_exit_up")
                event, event_dir = "phase_exit_up", 1
            elif close[t] < base_lo - buf and s.bear_disp_3 >= 1.0:
                exit_base(-1, "phase_exit_down")
                event, event_dir = "phase_exit_down", -1
            else:
                # Expiry means "no confirming evidence". If the
                # fallback legs still hold in the SAME direction as
                # this episode, the evidence is present: refresh the
                # episode (now with fallback provenance) instead of
                # dropping back to the base label for one bar and
                # letting the legs age out. Bounded: all three legs
                # are aging windows, so the episode still expires.
                fb = _fallback_manipulation(s, t, cfg)
                if fb is not None and fb[0] == manip_dir:
                    manip_left = cfg.amd_manip_bars
                    event, event_dir = "manipulation_fallback", fb[0]
                    cur_via, cur_conf = "fallback", \
                        cfg.amd_fallback_confidence
                    cur_evidence = fb[1]
                    # Refresh in place; the episode continues.
                else:
                    manip_left -= 1
                if manip_left <= 0 and base_lo <= close[t] <= base_hi:
                    phase, phase_start = base_label, t
                    cur_via, cur_conf = "direct", cfg.amd_base_confidence
                    cur_evidence = ()
        elif phase in ("expansion_up", "expansion_down"):
            exp_left -= 1
            if base_candidate:
                enter_base()
                event = "base_start"
            elif exp_left <= 0:
                phase = "trend_up" if phase == "expansion_up" else "trend_down"
                phase_start = t
                # Trend inherits the expansion's direct provenance.
        elif phase in ("trend_up", "trend_down"):
            if base_candidate:
                enter_base()
                event = "base_start"

        in_base_now = phase in BASE_LABELS or phase == "manipulation"
        out.append(PhaseState(
            phase=phase, phase_age=t - phase_start,
            range_high=base_hi if in_base_now else float("nan"),
            range_low=base_lo if in_base_now else float("nan"),
            event=event, event_direction=event_dir,
            via=cur_via, confidence=cur_conf, evidence=cur_evidence,
        ))
        cur_evidence = ()  # evidence only on the entry bar
    return out
