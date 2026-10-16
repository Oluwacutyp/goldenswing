"""STATE 5 Entry + alternate scenarios.  Professional desk
execution: no chasing, no half-assed entries.

ENTRY TYPES:
- LIMIT (preferred, ~70%): price retraces into the FVG / OB /
  breaker / mitigation block.  Limit at the FVG CE (50% of
  the FVG) for longs/shorts, OB mid, or breaker.  Only with
  FVG unmitigated + BOS holds + CHoCH valid + retracement
  within window (3 M15 normal, 12 M15 complex).  Expiry: the
  limit pending expires after 4 H1 bars (existing
  ``pending_expiry``); unfilled -> cancel, log expired.
- MARKET (confirm close, ~30%): displacement + CHoCH + FVG
  confirm close, price does NOT retrace into the FVG within
  1 M15 bar (it runs away) AND price is within 0.25 ATR of
  the FVG edge (not chased > 1 ATR from the FVG).  Market on
  the confirm close (next tick after the CHoCH bar close).
  Price > 1.0 ATR from the FVG without retracement = DO NOT
  CHASE ("missed, price ran >1 ATR, no chase"), back to
  STATE 1.
- STOP (breakout path ONLY): breakout path (momentum breach),
  entry = stop beyond the breakout edge + 0.1 ATR, on retest
  hold / BOS confirm.  NEVER a stop for mean-reversion sweep
  setups.

SL — STRICT, relative to the sweep extreme (invalidation, not
arbitrary ATR):
- LONG: SL = min(sweep_low, FVG_low, CHoCH_extreme) - 0.25 ATR.
  NOT below the FVG low alone (FVG low is entry, sweep low is
  invalidation).  If the FVG low is below the sweep low, SL =
  FVG low - 0.25 ATR (farther of the two).  If the CHoCH
  extreme is below the sweep low, SL = CHoCH extreme - 0.25
  ATR (farther).  Rule: SL = min(...) - 0.25 ATR.
- SHORT: SL = max(sweep_high, FVG_high, CHoCH_extreme) + 0.25 ATR.
- SL distance < 0.5 ATR -> setup INVALID ("SL too tight").
  SL distance > 3.0 ATR -> setup INVALID ("SL too wide").
- SL never moves against the trade; only trailing up (long)
  / down (short) behind structure / FVG / BOS, after +1R
  breakeven (runner / tick management owns the trail).

TP:
- TP1 = +1R (breakeven + 50% partial), TP2 = +2R (25%
  partial) OR HTF opposite zone / liquidity pool, whichever
  closer, TP3 = +3R (25% runner) OR HTF opposite zone /
  major liquidity pool.
- R:R check: TP1/SL >= 1.0, TP2/SL >= 2.0, else skip
  ("R:R <2, skip").

SIZING: risk per trade = config risk (e.g. 1% equity or
fixed lots 0.01 on $12).  lots = risk / SL distance, capped
by the margin shield (runner-owned — this module returns the
uncapped plan + size multiplier; the runner applies margin),
by HTF zone score (MAJOR 1.0x, MINOR 0.5x), by SMT fail
(-20% size), by complex pullback (-10% size, wider window =
less conviction).  One trade per setup, one position per HTF
zone at a time (runner-owned — no position state lives here).

ALTERNATES:
- Complex Pullback (multi-leg / ABC): sweep + displacement,
  but price does NOT pull back into the FVG within 3 M15
  bars — instead a secondary consolidation / complex ABC
  correction.  As long as the FVG remains UNMITIGATED (no
  close inside) AND the HTF zone stays valid AND BOS holds
  (no close back through the CHoCH extreme + FVG mitigated),
  the retracement window EXTENDS to 12 M15 bars (3h).
  Secondary test allowed: price touches the FVG, wicks,
  closes away, then touches again — still valid if the FVG
  is unmitigated + BOS holds.  BOS breaks OR FVG mitigated
  -> invalidate, back to STATE 1.  Size -10%.
- SMT Divergence (Gold vs DXY / Silver): confluence, NEVER a
  veto.  DXY unavailable -> SMT neutral 0, never blocks.
  Gold CHoCH long should coincide with DXY CHoCH short /
  DXY sweep high + displacement down, or DXY failing to make
  a new high (divergence).  DXY making a new high while Gold
  prints CHoCH long = SMT FAIL (-10, require stronger confirm
  MSB + FVG + volume).  Silver: Gold + Silver agree on CHoCH
  direction = PASS +10; one fails = FAIL -10.
- Nested TF Execution: an M15 FVG too wide for a tight R:R
  stop.  Once inside the M15 FVG band, drop to M5 (primary) /
  M1 (surgical) for a sub-TF CHoCH + sub-FVG inside the M15
  FVG.  Sub-CHoCH: M5 close beyond the last M5 opposing swing
  inside the M15 FVG + sub-FVG (M5 FVG >= 0.1 ATR).  Entry
  at the sub-FVG CE, SL beyond the sub-CHoCH extreme
  (tighter), TP same HTF target (better R:R).  No sub-CHoCH
  in 6 M5 bars inside the M15 FVG -> fall back to the M15
  FVG CE entry (wider SL, lower size to keep risk same).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .multitf import find_swings
from .min_lot import MIN_LOT, evaluate_min_lot
from .opposing import TIER_SPECS
from .patterns import trend_bias
from .zones import SWING_N

SL_BUFFER_ATR = 0.25
SL_MIN_ATR = 0.5
SL_MAX_ATR = 3.0
LIMIT_EXPIRY_H1_BARS = 4
MARKET_EDGE_MAX_ATR = 0.25
NO_CHASE_ATR = 1.0
STOP_EDGE_BUFFER_ATR = 0.1
MINOR_SIZE_MULT = 0.5
SMT_FAIL_SIZE_MULT = 0.8
COMPLEX_SIZE_MULT = 0.9
LATE_CONFIRM_SIZE_MULT = 0.5
SMT_ADJ = 10.0
SUB_FVG_MIN_ATR = 0.1
SUB_CHOCH_MAX_BARS = 6
XAUUSD_OZ_PER_LOT = 100.0


@dataclass
class EntryPlan:
    """One STATE 5 entry plan (order intent, not the order)."""

    valid: bool = False
    entry_type: str = "none"  # limit | market | stop | none
    entry_price: float | None = None
    stop: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None
    risk_atr: float = 0.0
    rr_tp2: float = 0.0
    size_mult: float = 1.0
    lots: float | None = None
    expiry_h1_bars: int = LIMIT_EXPIRY_H1_BARS
    nested: bool = False
    nested_tf: str = ""
    nested_extreme: float | None = None
    nested_fvg: dict = field(default_factory=dict)
    smt: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    invalid_reason: str = ""
    log: str = ""
    min_lot: dict = field(default_factory=dict)


# ---------------------------------------------------------------- SMT

def evaluate_smt(direction: int, *, dxy_frame=None,
                 silver_frame=None) -> dict:
    """SMT divergence: confluence, NEVER a veto, never blocks.

    DXY: Gold CHoCH long wants DXY CHoCH short / DXY failing
    to make a new high (divergence).  DXY making a new high
    while Gold prints CHoCH long = FAIL (-10, require stronger
    confirm MSB + FVG + volume).  Silver: Gold + Silver agree
    on CHoCH direction = PASS (+10); disagree = FAIL (-10).
    Unavailable asset -> neutral 0.
    """
    out = {"status": "neutral", "score_adj": 0.0, "flags": [],
           "detail": [], "dxy": "neutral", "silver": "neutral"}
    # --- DXY ---
    if dxy_frame is not None and len(dxy_frame) >= 2 * SWING_N + 5:
        try:
            highs, lows = find_swings(
                dxy_frame["high"].to_numpy(dtype=float),
                dxy_frame["low"].to_numpy(dtype=float), SWING_N)
            close = float(dxy_frame["close"].iloc[-1])
            dxy = "neutral"
            if direction > 0:
                # Gold long wants DXY bearish / divergent.
                if lows and close < lows[-1][1]:
                    dxy = "pass"  # DXY CHoCH short
                elif len(highs) >= 2 and highs[-1][1] > highs[-2][1]:
                    dxy = "fail"  # DXY also bullish: fakeout risk
                elif len(highs) >= 2:
                    dxy = "pass"  # DXY fails to make a new high
            else:
                if highs and close > highs[-1][1]:
                    dxy = "pass"  # DXY CHoCH long
                elif len(lows) >= 2 and lows[-1][1] < lows[-2][1]:
                    dxy = "fail"  # DXY also bearish
                elif len(lows) >= 2:
                    dxy = "pass"  # DXY fails to make a new low
            out["dxy"] = dxy
            out["detail"].append(f"DXY {dxy}")
        except Exception:
            pass
    # --- Silver: agree on direction = pass. ---
    if silver_frame is not None and len(silver_frame) >= 10:
        try:
            bias = int(trend_bias(silver_frame)["bias"])
            if bias == direction:
                out["silver"] = "pass"
            elif bias == -direction:
                out["silver"] = "fail"
            out["detail"].append(f"Silver {out['silver']}")
        except Exception:
            pass
    adj = 0.0
    fails = 0
    for asset in ("dxy", "silver"):
        if out[asset] == "pass":
            adj += SMT_ADJ
        elif out[asset] == "fail":
            adj -= SMT_ADJ
            fails += 1
    out["score_adj"] = adj
    if fails:
        out["status"] = "fail"
        out["flags"].append("smt_fail_requires_msb_fvg_volume")
        out["detail"].append(
            "SMT fail: Gold CHoCH long, DXY also bullish, "
            "fakeout risk" if direction > 0 else
            "SMT fail: Gold CHoCH short, DXY also bearish, "
            "fakeout risk")
    elif adj > 0:
        out["status"] = "pass"
    return out


# ---------------------------------------------------------------- nested

def _sub_setup(frame, m15_fvg: dict, direction: int,
               atr: float) -> dict | None:
    """Sub-TF CHoCH + sub-FVG inside the M15 FVG band.

    Returns {entry, sub_choch_extreme, sub_fvg, bar_pos} or None.
    The sub-CHoCH must print within 6 sub-TF bars of price
    first entering the M15 FVG band; the sub-FVG (>= 0.1 ATR)
    must overlap the M15 FVG band.
    """
    if frame is None or len(frame) < 2 * SWING_N + 5 or atr <= 0:
        return None
    top, bottom = float(m15_fvg["top"]), float(m15_fvg["bottom"])
    n = len(frame)
    h = frame["high"].to_numpy(dtype=float)
    l = frame["low"].to_numpy(dtype=float)
    c = frame["close"].to_numpy(dtype=float)
    # First sub-TF bar whose range overlaps the M15 FVG band.
    entry_band = None
    for i in range(n):
        if float(h[i]) >= bottom and float(l[i]) <= top:
            entry_band = i
            break
    if entry_band is None:
        return None
    window_end = min(n - 1, entry_band + SUB_CHOCH_MAX_BARS)
    for pos in range(entry_band, window_end + 1):
        sub = frame.iloc[:pos + 1]
        try:
            highs, lows = find_swings(
                sub["high"].to_numpy(dtype=float),
                sub["low"].to_numpy(dtype=float), SWING_N)
        except Exception:
            continue
        close = float(c[pos])
        if direction > 0:
            if not highs or close <= highs[-1][1]:
                continue
            extreme = float(np.min(l[pos:pos + 1]))
        else:
            if not lows or close >= lows[-1][1]:
                continue
            extreme = float(np.max(h[pos:pos + 1]))
        # Sub-FVG formed at/after the sub-CHoCH bar, overlapping
        # the M15 FVG band, size >= 0.1 ATR, unmitigated.
        for j in range(max(2, pos), min(window_end, n - 1) + 1):
            if direction > 0 and float(l[j]) > float(h[j - 2]):
                stop_, sbottom = float(l[j]), float(h[j - 2])
            elif direction < 0 and float(h[j]) < float(l[j - 2]):
                stop_, sbottom = float(l[j - 2]), float(h[j])
            else:
                continue
            if stop_ - sbottom < SUB_FVG_MIN_ATR * atr:
                continue
            if stop_ < bottom or sbottom > top:
                continue  # sub-FVG outside the M15 FVG band
            filled = bool(np.any(
                (c[j + 1:window_end + 1] >= sbottom)
                & (c[j + 1:window_end + 1] <= stop_))) \
                if j + 1 <= window_end else False
            if filled:
                continue
            # Sub-CHoCH extreme: extreme since the sub-CHoCH bar.
            if direction > 0:
                extreme = float(np.min(l[pos:j + 1]))
            else:
                extreme = float(np.max(h[pos:j + 1]))
            return {"entry": (stop_ + sbottom) / 2.0,
                    "sub_choch_extreme": extreme,
                    "sub_fvg": {"top": stop_, "bottom": sbottom,
                                "ce": (stop_ + sbottom) / 2.0},
                    "bar_pos": j}
    return None


def _nested_entry(m5_frame, m1_frame, m15_fvg: dict,
                  direction: int, atr: float) -> dict | None:
    """M5 primary, M1 surgical fallback."""
    got = _sub_setup(m5_frame, m15_fvg, direction, atr)
    if got is not None:
        got["tf"] = "M5"
        return got
    got = _sub_setup(m1_frame, m15_fvg, direction, atr)
    if got is not None:
        got["tf"] = "M1"
        return got
    return None


# ---------------------------------------------------------------- entry

def _zf(zone, name, default=None):
    if isinstance(zone, dict):
        return zone.get(name, default)
    return getattr(zone, name, default)


def evaluate_entry(frame_m15, *, confirmation, displacement,
                   sweep, direction: int, atr: float,
                   zone=None, opposite_levels=None,
                   dxy_frame=None, silver_frame=None,
                   m5_frame=None, m1_frame=None,
                   risk_amount: float | None = None,
                   breakout_edge: float | None = None,
                   late_confirm: bool = False,
                   risk_tier: str | None = None,
                   equity: float | None = None,
                   mode: str = "PERCENT",
                   free_margin: float | None = None,
                   margin_required: float | None = None,
                   open_trades: int = 0) -> EntryPlan:
    """STATE 5: build the entry plan from a STATE 4 confirmation.

    ``late_confirm``: second-bar institutional confirm —
    0.5x size, less conviction.  ``risk_tier``: GREEN /
    YELLOW / RED entry-zone tier — size is min(confirm,
    tier); tiering never gates.
    """
    plan = EntryPlan()
    if frame_m15 is None or len(frame_m15) == 0 or atr <= 0:
        plan.invalid_reason = "no bars for entry"
        plan.log = "STATE 5: no bars — no entry"
        return plan
    conf = confirmation or {}
    conf_decision = conf.get("decision", "wait_structure") \
        if isinstance(conf, dict) else getattr(
            conf, "decision", "wait_structure")
    conf_kind = conf.get("kind", "none") if isinstance(conf, dict) \
        else getattr(conf, "kind", "none")
    fvg = dict((conf.get("fvg") or {}) if isinstance(conf, dict)
               else getattr(conf, "fvg", {}) or {})
    choch_extreme = conf.get("choch_extreme") if isinstance(
        conf, dict) else getattr(conf, "choch_extreme", None)
    complex_pullback = bool((conf.get("flags") or [])
                            and "complex_pullback" in
                            (conf.get("flags") or [])) \
        if isinstance(conf, dict) else bool(
            "complex_pullback" in (getattr(conf, "flags", []) or []))
    if conf_decision == "missed":
        plan.invalid_reason = "missed, price ran >1 ATR, no chase"
        plan.log = ("STATE 5: missed, price ran >1 ATR from FVG "
                    "without retracement — no chase, back to "
                    "STATE 1")
        return plan
    if conf_decision in ("wait_structure", "wait_fvg", "stale",
                         "mitigated", "invalid"):
        plan.invalid_reason = f"STATE 4 not confirmed: {conf_decision}"
        plan.log = f"STATE 5: STATE 4 {conf_decision} — no entry"
        return plan
    if not fvg and breakout_edge is None:
        plan.invalid_reason = "no FVG for entry"
        plan.log = "STATE 5: no FVG — no entry"
        return plan
    last_close = float(frame_m15["close"].iloc[-1])
    fvg_top = float(fvg.get("top", last_close)) if fvg else last_close
    fvg_bottom = float(fvg.get("bottom", last_close)) if fvg \
        else last_close
    fvg_ce = float(fvg.get("ce", (fvg_top + fvg_bottom) / 2.0)) \
        if fvg else last_close
    bars_since = conf.get("bars_since_choch", 0) \
        if isinstance(conf, dict) else getattr(
            conf, "bars_since_choch", 0)
    retraced = conf.get("retraced", False) if isinstance(
        conf, dict) else getattr(conf, "retraced", False)

    # --- Entry type. ---
    if breakout_edge is not None:
        plan.entry_type = "stop"
        plan.entry_price = float(breakout_edge) + direction \
            * STOP_EDGE_BUFFER_ATR * atr
        plan.expiry_h1_bars = LIMIT_EXPIRY_H1_BARS
        plan.flags.append("breakout_stop_entry")
    elif conf_decision in ("confirm", "wait_retracement",
                           "complex_pullback"):
        # MARKET on confirm close: no retracement within 1 M15
        # bar, price leaving the FVG edge (within 0.25 ATR),
        # not chased > 1 ATR from the FVG.
        if (not retraced and bars_since >= 1
                and conf_decision == "wait_retracement"):
            dist_edge = min(abs(last_close - fvg_top),
                            abs(last_close - fvg_bottom)) / atr
            if dist_edge <= MARKET_EDGE_MAX_ATR:
                # Leaving? last close farther from the FVG than
                # the CHoCH-bar close was.
                plan.entry_type = "market"
                plan.entry_price = last_close
                plan.flags.append("market_confirm_close")
        if plan.entry_type == "none":
            plan.entry_type = "limit"
            plan.entry_price = fvg_ce
    else:
        plan.invalid_reason = f"STATE 4 {conf_decision} — no entry"
        plan.log = f"STATE 5: STATE 4 {conf_decision} — no entry"
        return plan
    # No-chase guard for every type.
    if plan.entry_type in ("limit", "market"):
        dist_fvg = 0.0 if fvg_bottom <= last_close <= fvg_top else \
            min(abs(last_close - fvg_top),
                abs(last_close - fvg_bottom)) / atr
        if (not retraced and plan.entry_type == "market"
                and dist_fvg > NO_CHASE_ATR):
            plan.valid = False
            plan.entry_type = "none"
            plan.invalid_reason = "missed, price ran >1 ATR, no chase"
            plan.log = ("STATE 5: missed, price ran >1 ATR from "
                        "FVG without retracement — no chase")
            return plan
    entry = float(plan.entry_price)
    # --- Nested TF: M5 primary / M1 surgical inside the M15 FVG.
    # LIMIT entries only; market/stop never nest (they fire now).
    if plan.entry_type == "limit" and (m5_frame is not None
                                       or m1_frame is not None):
        nested = _nested_entry(m5_frame, m1_frame, fvg, direction,
                               atr)
        if nested is not None:
            plan.nested = True
            plan.nested_tf = nested["tf"]
            plan.nested_extreme = float(nested["sub_choch_extreme"])
            plan.nested_fvg = dict(nested["sub_fvg"])
            plan.entry_price = float(nested["entry"])
            entry = float(plan.entry_price)
            plan.flags.append(f"nested_{nested['tf']}")
        else:
            plan.flags.append("nested_fallback_m15")
    # --- SL: strict min/max rule around invalidation. ---
    # Nested: the surgical stop is already buffered (sub-CHoCH
    # extreme +/- 0.25 ATR) — it replaces the HTF min/max SL as
    # the placed stop; the HTF SL stays the hard invalidation.
    if plan.nested and plan.nested_extreme is not None:
        plan.stop = float(plan.nested_extreme) - direction \
            * SL_BUFFER_ATR * atr
        plan.flags.append("nested_tighter_sl")
        cands = []
    else:
        cands = []
        if sweep and sweep.get("wick") is not None:
            cands.append(float(sweep["wick"]))
        if fvg:
            cands.append(fvg_bottom if direction > 0 else fvg_top)
        if choch_extreme is not None:
            cands.append(float(choch_extreme))
        if breakout_edge is not None:
            cands.append(float(breakout_edge))
    if not cands and plan.stop is None:
        plan.invalid_reason = "no invalidation level for SL"
        plan.log = "STATE 5: no SL anchor — no entry"
        return plan
    if plan.stop is None:
        if direction > 0:
            plan.stop = min(cands) - SL_BUFFER_ATR * atr
        else:
            plan.stop = max(cands) + SL_BUFFER_ATR * atr
    risk = abs(entry - float(plan.stop))
    plan.risk_atr = risk / atr
    if plan.risk_atr < SL_MIN_ATR:
        plan.invalid_reason = "SL too tight, <0.5 ATR, invalid"
        plan.log = (f"STATE 5: SL too tight ({plan.risk_atr:.2f} "
                    "ATR < 0.5) — will get swept again, invalid")
        return plan
    if plan.risk_atr > SL_MAX_ATR:
        plan.invalid_reason = "SL too wide, >3 ATR, invalid, skip"
        plan.log = (f"STATE 5: SL too wide ({plan.risk_atr:.2f} "
                    "ATR > 3.0) — R:R broken, invalid, skip")
        return plan
    # --- TP: TP1 +1R, TP2 +2R or opposite zone (closer), TP3
    # +3R or opposite zone. ---
    opp = None
    for lv in (opposite_levels or []):
        if (direction > 0 and float(lv) > entry) or \
                (direction < 0 and float(lv) < entry):
            if opp is None or abs(float(lv) - entry) < abs(opp - entry):
                opp = float(lv)
    tp1 = entry + direction * risk
    two_r = entry + direction * 2 * risk
    three_r = entry + direction * 3 * risk
    if opp is None:
        tp2, tp3 = two_r, three_r
    elif abs(opp - entry) >= 2 * risk:
        tp2 = two_r
        tp3 = opp if abs(opp - entry) <= 3 * risk else three_r
    else:
        tp2, tp3 = opp, three_r
    plan.tp1, plan.tp2, plan.tp3 = tp1, tp2, tp3
    plan.rr_tp2 = abs(tp2 - entry) / risk if risk > 0 else 0.0
    if plan.rr_tp2 < 2.0:
        # Late confirm: R:R recalculated on the late entry —
        # if < 2, skip, never enter late with broken R:R.
        plan.invalid_reason = (
            "late confirm, R:R <2, skip" if late_confirm
            else "R:R <2, skip")
        plan.log = (f"STATE 5: R:R {plan.rr_tp2:.2f} < 2 — skip"
                    + (" (late confirm)" if late_confirm else ""))
        return plan
    # --- Sizing multipliers (margin shield is runner-owned). ---
    mult = 1.0
    tier = str(_zf(zone, "tier", "MAJOR") or "MAJOR")
    if tier == "MINOR":
        mult *= MINOR_SIZE_MULT
        plan.flags.append("minor_half_size")
    if complex_pullback:
        mult *= COMPLEX_SIZE_MULT
        plan.flags.append("complex_pullback_minus_10pct")
    if late_confirm:
        mult *= LATE_CONFIRM_SIZE_MULT
        plan.flags.append("late_confirm_half_size")
    if risk_tier in TIER_SPECS:
        mult = min(mult,
                   float(TIER_SPECS[risk_tier]["size_mult"]))
        plan.flags.append(f"risk_tier_{risk_tier.lower()}")
    smt = evaluate_smt(direction, dxy_frame=dxy_frame,
                       silver_frame=silver_frame)
    plan.smt = smt
    if smt["status"] == "fail":
        mult *= SMT_FAIL_SIZE_MULT
        plan.flags.append("smt_fail_minus_20pct")
    plan.size_mult = mult
    if risk_amount is not None and risk > 0:
        lots = float(risk_amount) / (risk * XAUUSD_OZ_PER_LOT) * mult
        if lots < 0.01:
            lots = 0.01
            plan.flags.append("lots_floored_0.01")
        plan.lots = round(lots, 2)
    # --- Min-lot floor (live/min_lot.py): the min order is
    # 0.01 lots; the SL is invalidation and is NEVER shrunk
    # to fit a risk target.  Only runs when equity + target
    # risk are known; otherwise the floor logic is skipped.
    if equity is not None and risk_amount is not None \
            and risk > 0:
        decision = evaluate_min_lot(
            sl_distance=risk, target_risk=float(risk_amount),
            equity=float(equity), mode=mode,
            free_margin=free_margin,
            margin_required=margin_required,
            open_trades=open_trades)
        plan.min_lot = decision
        if decision["decision"] == "skip":
            plan.valid = False
            plan.invalid_reason = decision["log"]
            plan.log = f"STATE 5: {decision['log']}"
            return plan
        if decision.get("lots") is not None:
            plan.lots = float(decision["lots"])
            plan.flags.append("min_lot_floor")
        for warn in decision.get("warnings", []):
            plan.flags.append(f"min_lot_warning: {warn}")
    plan.valid = True
    # --- Entry log: the full narrative. ---
    zone_txt = ""
    if zone is not None:
        zone_txt = (f"{_zf(zone, 'tf', '?')} "
                    f"{float(_zf(zone, 'level', 0.0)):.1f}")
    sweep_txt = ""
    if sweep and sweep.get("wick") is not None:
        sweep_txt = f"sweep {float(sweep['wick']):.1f}"
    disp = displacement or {}
    disp_txt = ""
    if isinstance(disp, dict) and disp.get("strength"):
        disp_txt = (f"displacement {disp.get('strength')} "
                    f"{float(disp.get('move_atr', 0.0)):.1f} ATR")
    elif hasattr(disp, "strength"):
        disp_txt = (f"displacement {disp.strength} "
                    f"{float(disp.move_atr):.1f} ATR")
    nested_txt = ""
    if plan.nested:
        nested_txt = (f"nested: M15 FVG {fvg_bottom:.1f}-"
                      f"{fvg_top:.1f}, {plan.nested_tf} sub-CHoCH "
                      f"{plan.nested_extreme:.1f} + sub-FVG "
                      f"{plan.nested_fvg.get('bottom', 0):.1f}-"
                      f"{plan.nested_fvg.get('top', 0):.1f}, "
                      f"surgical entry {entry:.1f}")
    plan.log = (
        f"STATE 5 Entry: HTF zone {zone_txt} -> {sweep_txt} -> "
        f"{disp_txt} -> {conf_kind.upper()}+FVG "
        f"{fvg_bottom:.1f}-{fvg_top:.1f} -> entry "
        f"{plan.entry_type} {entry:.1f} -> SL {plan.stop:.1f} "
        f"(sweep extreme + {SL_BUFFER_ATR} ATR buffer) -> "
        f"TP1 {tp1:.1f} / TP2 {tp2:.1f} / TP3 {tp3:.1f} -> "
        f"R:R {plan.rr_tp2:.1f} -> score_adj "
        f"{float(smt.get('score_adj', 0.0)):+.0f} SMT "
        f"{smt['status']} -> size_mult {mult:.2f} "
        + ("late confirm, 0.5x size" if late_confirm else "")
        + (f"lots {plan.lots}" if plan.lots is not None else "")
        + (f" -> {nested_txt}" if nested_txt else ""))
    return plan
