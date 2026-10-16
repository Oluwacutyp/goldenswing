"""STATE 3 displacement + inducement: velocity + structure math.

DISPLACEMENT (institutional sponsorship proof):
- Direction: AWAY from the sweep (sweep low -> displacement UP,
  sweep high -> displacement DOWN).
- Distance (HTF ATR, ``ATR_H4`` — never M15 ATR): leg move
  >= 1.5 ATR, measured from the sweep extreme (wick) to the
  displacement extreme (farthest close in displacement
  direction).  1.0-1.5 ATR = WEAK (flagged, needs stronger LTF
  confirm).  < 1.0 ATR = NO displacement -> invalidate sweep.
- Bar count: the move must complete within <= 3 M15 bars (45
  min) — M15 is primary; M5 (<= 6 bars, 30 min) is an advisory
  confirm when supplied.  A move that takes > 3 M15 bars is
  drift, not displacement.  1 bar = STRONG, 2 = NORMAL,
  3 = WEAK but valid.
- Candle structure: leg candles body-dominated (avg body/wick
  >= 1.5); candles whose ranges all overlap the previous by
  > 50% = chop, not displacement.  >= 2 consecutive closes in
  the displacement direction (1-bar legs exempt — a 1-bar
  >= 2.0 ATR leg is VERY STRONG by the table); no opposite
  close > 0.5 ATR inside the leg.
- FVG: the leg MUST leave >= 1 unmitigated FVG (bullish for up,
  bearish for down), size >= 0.25 ATR_H4, recorded high/low/CE.
  No FVG = no displacement.  The FVG must still be unmitigated
  at STATE 4 entry.
- BOS: the leg must break the last opposing swing on M15 (up
  breaks the last M15 lower high, down breaks the last M15
  higher low) — falling back to the swept edge when no prior
  swing is visible.  No BOS = not displacement.
- Volume (when available): >= 1.5x average volume on >= 1 leg
  bar, or average leg volume >= 1.3x average.  Low volume =
  WEAK, needs volume confirm on CHoCH.

INDUCEMENT (IDM):
- Last swing before the sweep (inducement high for shorts,
  inducement low for longs).  Price must take the inducement
  BEFORE the true move, or the inducement must be the sweep
  target itself.
- After the sweep, price retraces into the inducement / sweep
  FVG: retracement > 50% of the displacement leg invalidates
  the displacement (back to STATE 1).
- Inducement NOT taken -> still valid, flagged "inducement not
  taken, weaker", score -5, needs stronger FVG confirm.
- Inducement FVG: sweep-created FVG; retracement into FVG 50%
  (CE) is the ideal entry.

Noise vs displacement:
- < 1.0 ATR, > 3 M15 bars, no FVG, no BOS, overlapping
  candles = NOISE, invalidate.
- 1.0-1.5 ATR, 3 M15 bars, FVG >= 0.25 ATR, BOS yes,
  body-dominated = WEAK, valid but flagged, needs stronger
  STATE 4.
- >= 1.5 ATR, <= 3 M15 bars, FVG >= 0.25 ATR, BOS yes,
  body-dominated, volume >= 1.5x = STRONG.
- >= 2.0 ATR, 1 M15 bar, FVG >= 0.5 ATR, BOS yes, volume
  spike = VERY STRONG, +10.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .multitf import find_swings
from .zones import SWING_N
from .patterns import inducement_level

DISP_MIN_ATR = 1.5
DISP_WEAK_MIN_ATR = 1.0
DISP_VERY_STRONG_ATR = 2.0
DISP_MAX_M15_BARS = 3
DISP_MAX_M5_BARS = 6
DISP_BODY_WICK_MIN = 1.5
DISP_FVG_MIN_ATR = 0.25
DISP_FVG_VERY_STRONG_ATR = 0.5
DISP_VOLUME_SPIKE = 1.5
DISP_VOLUME_AVG = 1.3
DISP_MAX_RETRACE_FRAC = 0.5
DISP_OPPOSITE_CLOSE_MAX_ATR = 0.5
CHOP_OVERLAP = 0.5
IDM_NOT_TAKEN_ADJ = -5.0
VERY_STRONG_ADJ = 10.0


@dataclass
class DisplacementVerdict:
    """One STATE 3 evaluation."""

    valid: bool = False
    strength: str = "none"  # none | weak | strong | very_strong
    direction: int = 0
    move_atr: float = 0.0
    bars: int = 0
    # ``complete``: the full 3-M15-bar leg window is closed, so a
    # rejection is final.  ``hard``: a rejection that holds even
    # on an incomplete leg (retracement > 50%).  The narrative
    # only acts on a rejection when ``complete or hard`` —
    # otherwise it waits for more bars.
    complete: bool = False
    hard: bool = False
    fvg: dict = field(default_factory=dict)
    bos: bool = False
    bos_price: float | None = None
    body_wick_avg: float = 0.0
    volume_ok: bool | None = None
    volume_ratio: float | None = None
    inducement_taken: bool | None = None
    inducement_level: float | None = None
    retracement_frac: float = 0.0
    retracement_atr: float = 0.0
    score_adj: float = 0.0
    flags: list = field(default_factory=list)
    leg_start_pos: int = -1
    leg_end_pos: int = -1
    extreme: float = 0.0
    log: str = ""


def _fvg_in_leg(frame, start_pos: int, end_pos: int,
                direction: int, atr: float) -> dict | None:
    """First unmitigated same-direction FVG formed inside the leg.

    Bullish FVG at bar i: low[i] > high[i-2].  Bearish:
    high[i] < low[i-2].  Unmitigated = no close inside the gap
    between formation and the LEG END (later fills are STATE 4's
    problem — mitigated vs inversion — not STATE 3's).  Size
    >= 0.25 ATR.
    """
    if frame is None or len(frame) < 3 or atr <= 0:
        return None
    h = frame["high"].to_numpy(dtype=float)
    l = frame["low"].to_numpy(dtype=float)
    c = frame["close"].to_numpy(dtype=float)
    for i in range(max(2, start_pos), min(end_pos, len(frame) - 1) + 1):
        if direction > 0 and l[i] > h[i - 2]:
            top, bottom = float(l[i]), float(h[i - 2])
        elif direction < 0 and h[i] < l[i - 2]:
            top, bottom = float(l[i - 2]), float(h[i])
        else:
            continue
        if top - bottom < DISP_FVG_MIN_ATR * atr:
            continue
        filled = bool(np.any((c[i + 1:end_pos + 1] >= bottom)
                             & (c[i + 1:end_pos + 1] <= top))) \
            if i + 1 <= end_pos else False
        if filled:
            continue
        return {"top": top, "bottom": bottom,
                "ce": (top + bottom) / 2.0,
                "size_atr": (top - bottom) / atr,
                "bar_pos": i, "bar_ts": str(frame.index[i]),
                "direction": direction}
    return None


def _bos_level(frame, sweep_pos: int, direction: int,
               swept_edge: float | None) -> float | None:
    """Last opposing swing before the sweep (BOS level).

    Up displacement: last swing high (the lower high).  Down:
    last swing low (the higher low).  Falls back to the swept
    edge when no prior swing is visible — the sweep itself took
    a level, so displacement beyond it is a structure break.
    """
    if sweep_pos > 2 * SWING_N + 5:
        sub = frame.iloc[:sweep_pos]
        try:
            highs, lows = find_swings(
                sub["high"].to_numpy(dtype=float),
                sub["low"].to_numpy(dtype=float), SWING_N)
        except Exception:
            highs, lows = [], []
        if direction > 0 and highs:
            return float(highs[-1][1])
        if direction < 0 and lows:
            return float(lows[-1][1])
    return swept_edge


def _body_wick_avg(frame, start: int, end: int) -> float:
    ratios = []
    for i in range(start, end + 1):
        row = frame.iloc[i]
        o, h, l, c = (float(row["open"]), float(row["high"]),
                      float(row["low"]), float(row["close"]))
        body = abs(c - o)
        wick = (h - max(o, c)) + (min(o, c) - l)
        ratios.append(body / wick if wick > 0 else float("inf"))
    finite = [r for r in ratios if r != float("inf")]
    if not finite:
        return float("inf")
    return float(np.mean(finite))


def _is_chop(frame, start: int, end: int) -> bool:
    """All adjacent leg candles overlap the previous by > 50% of
    the smaller range = chop, not displacement."""
    if end - start < 1:
        return False
    for i in range(start + 1, end + 1):
        prev = frame.iloc[i - 1]
        cur = frame.iloc[i]
        lo = max(float(prev["low"]), float(cur["low"]))
        hi = min(float(prev["high"]), float(cur["high"]))
        inter = max(0.0, hi - lo)
        smaller = min(float(prev["high"]) - float(prev["low"]),
                      float(cur["high"]) - float(cur["low"]))
        if smaller <= 0 or inter / smaller <= CHOP_OVERLAP:
            return False
    return True


def evaluate_inducement(frame, sweep: dict, disp: dict,
                        direction: int, atr: float,
                        *, idm_level: float | None = None) -> dict:
    """IDM check for a displacement: taken? retraced > 50%? CE?"""
    out = {"taken": None, "level": idm_level, "retracement_frac": 0.0,
           "retracement_atr": 0.0, "ce": None, "flags": [],
           "log": ""}
    if idm_level is None:
        try:
            sweep_pos = int(sweep.get("bar_pos", -1))
            sub = frame.iloc[:sweep_pos + 1] if sweep_pos >= 0 \
                else frame
            idm = inducement_level(sub, direction=direction)
            if idm.get("present"):
                idm_level = float(idm["level"])
                out["level"] = idm_level
        except Exception:
            idm_level = None
    if idm_level is not None:
        wick = sweep.get("wick")
        if wick is not None:
            taken = (float(wick) <= idm_level) if direction > 0 \
                else (float(wick) >= idm_level)
        else:
            taken = False
        # Inducement is the sweep target itself: within 0.5 ATR
        # of the swept edge.
        edge = sweep.get("edge")
        if edge is not None and abs(float(edge) - idm_level) \
                <= 0.5 * atr:
            taken = True
        out["taken"] = bool(taken)
        out["log"] = ("inducement taken, valid" if taken else
                      "inducement not taken, weaker")
        if not taken:
            out["flags"].append("inducement_not_taken")
    # Retracement after the displacement extreme.
    move = float(disp.get("move", 0.0))
    extreme = disp.get("extreme")
    end_pos = disp.get("leg_end_pos", -1)
    if extreme is not None and end_pos >= 0 and move > 0 \
            and end_pos + 1 < len(frame):
        last_close = float(frame["close"].iloc[-1])
        retr = (float(extreme) - last_close) * direction / atr
        out["retracement_atr"] = max(0.0, retr)
        out["retracement_frac"] = max(0.0, retr) / (move / atr) \
            if atr > 0 else 0.0
    fvg = disp.get("fvg") or {}
    if fvg:
        out["ce"] = float(fvg.get("ce", 0.0))
    return out


def evaluate_displacement(frame, *, sweep: dict | None,
                          direction: int, atr: float,
                          m5_frame=None,
                          idm_level: float | None = None,
                          disp_min_atr: float = DISP_MIN_ATR,
                          volume_spike_min: float
                          = DISP_VOLUME_SPIKE
                          ) -> DisplacementVerdict:
    """STATE 3: displacement + inducement after a confirmed sweep.

    ``disp_min_atr`` / ``volume_spike_min`` come from the
    Dynamic ATR Regime (``live/atr_regime.py``): in HIGH
    volatility the displacement bar rises 1.5 -> 1.75x and
    the volume spike 1.5x -> 2.0x.  Defaults are NORMAL.
    """
    base = DisplacementVerdict(direction=direction)
    if frame is None or len(frame) < 2 or atr <= 0:
        base.log = "no bars for displacement check"
        return base
    if not sweep or int(sweep.get("bar_pos", -1)) < 0:
        base.log = "no confirmed sweep — STATE 3 needs a sweep"
        return base
    sweep_pos = int(sweep["bar_pos"])
    sweep_wick = sweep.get("wick")
    if sweep_wick is None:
        row = frame.iloc[sweep_pos]
        sweep_wick = float(row["low"]) if direction > 0 \
            else float(row["high"])
    sweep_wick = float(sweep_wick)
    n = len(frame)
    c = frame["close"].to_numpy(dtype=float)
    # Leg = bars after the sweep, up to 3 M15 bars.
    leg_end = min(sweep_pos + DISP_MAX_M15_BARS, n - 1)
    if leg_end <= sweep_pos:
        base.log = "awaiting bars after sweep for displacement"
        return base
    base.complete = (n - 1 - sweep_pos) >= DISP_MAX_M15_BARS
    # Farthest close in the displacement direction, per bar count.
    reached_k = None
    move_atr = 0.0
    extreme = None
    for k in range(1, leg_end - sweep_pos + 1):
        window = c[sweep_pos + 1:sweep_pos + 1 + k]
        far = float(np.max(window)) if direction > 0 \
            else float(np.min(window))
        mv = (far - sweep_wick) * direction / atr
        if mv > move_atr:
            move_atr, extreme = mv, far
        if mv >= disp_min_atr:
            reached_k = k
            break
    base.move_atr = move_atr
    base.extreme = extreme if extreme is not None else sweep_wick
    base.leg_start_pos = sweep_pos + 1
    base.leg_end_pos = sweep_pos + (reached_k or
                                    (leg_end - sweep_pos))
    if move_atr < DISP_WEAK_MIN_ATR:
        if not base.complete:
            base.log = (f"displacement building: move "
                        f"{move_atr:.2f} ATR, awaiting more bars")
            return base
        base.hard = True
        base.log = (f"sweep without displacement: move "
                    f"{move_atr:.2f} ATR < 1.0 ATR — invalidate "
                    "sweep, back to STATE 1")
        return base
    if reached_k is None:
        # 1.0-disp_min ATR inside 3 bars, never reached the
        # bar: WEAK.
        base.strength = "weak"
        base.bars = leg_end - sweep_pos
        base.flags.append("weak_displacement")
        base.valid = True
    else:
        base.bars = reached_k
        if reached_k == 1 and move_atr >= DISP_VERY_STRONG_ATR:
            base.strength = "very_strong"
        elif reached_k <= 2:
            base.strength = "strong"
        else:
            base.strength = "weak"
            base.flags.append("weak_speed_3_bars")
        base.valid = True
    # Candle structure over the leg.
    leg_s, leg_e = base.leg_start_pos, base.leg_end_pos
    bw = _body_wick_avg(frame, leg_s, leg_e)
    base.body_wick_avg = bw
    if bw < DISP_BODY_WICK_MIN:
        base.valid = False
        base.strength = "none"
        base.hard = base.complete
        base.log = (f"displacement rejected: leg body/wick "
                    f"{bw:.2f} < 1.5 — momentum without structure")
        return base
    if _is_chop(frame, leg_s, leg_e):
        base.valid = False
        base.strength = "none"
        base.hard = base.complete
        base.log = ("displacement rejected: leg candles overlap "
                    "> 50% — chop, not displacement")
        return base
    # Consecutive closes in direction (multi-bar legs only).
    if leg_e - leg_s >= 1:
        run, best_run = 0, 0
        prev = float(c[sweep_pos])
        for i in range(leg_s, leg_e + 1):
            if (float(c[i]) - prev) * direction > 0:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
            prev = float(c[i])
        if best_run < 2:
            base.valid = False
            base.strength = "none"
            base.hard = base.complete
            base.log = ("displacement rejected: fewer than 2 "
                        "consecutive closes in displacement "
                        "direction")
            return base
    # No opposite close > 0.5 ATR inside the leg.
    prev = float(c[sweep_pos])
    for i in range(leg_s, leg_e + 1):
        if (prev - float(c[i])) * direction / atr \
                > DISP_OPPOSITE_CLOSE_MAX_ATR:
            base.valid = False
            base.strength = "none"
            base.hard = base.complete
            base.log = (f"displacement rejected: opposite close "
                        f"> 0.5 ATR inside the leg at bar {i}")
            return base
        prev = float(c[i])
    # FVG from the leg (must exist, unmitigated, >= 0.25 ATR).
    fvg = _fvg_in_leg(frame, leg_s, leg_e, direction, atr)
    if fvg is None:
        base.valid = False
        base.strength = "none"
        base.hard = base.complete
        base.log = ("displacement rejected: no unmitigated FVG "
                    ">= 0.25 ATR left by the leg — drift, not "
                    "sponsorship")
        return base
    base.fvg = fvg
    # BOS: break of the last opposing swing (or the swept edge).
    bos_lvl = _bos_level(frame, sweep_pos, direction,
                         sweep.get("edge"))
    base.bos_price = bos_lvl
    if bos_lvl is not None:
        leg_closes = c[leg_s:leg_e + 1]
        base.bos = bool(np.any(leg_closes > bos_lvl)) \
            if direction > 0 else bool(np.any(leg_closes < bos_lvl))
    if not base.bos:
        base.valid = False
        base.strength = "none"
        base.hard = base.complete
        base.log = ("displacement rejected: no BOS — leg never "
                    "broke the last opposing swing")
        return base
    # Volume (when available).
    if "volume" in frame.columns:
        try:
            vols = frame["volume"].to_numpy(dtype=float)
            mean_v = float(np.mean(
                vols[max(0, sweep_pos - 20):sweep_pos])) \
                if sweep_pos > 0 else 0.0
            leg_vols = vols[leg_s:leg_e + 1]
            if mean_v > 0:
                base.volume_ratio = float(np.max(leg_vols)) / mean_v
                avg_ratio = float(np.mean(leg_vols)) / mean_v
                base.volume_ok = base.volume_ratio >= \
                    volume_spike_min or avg_ratio >= DISP_VOLUME_AVG
            else:
                base.volume_ok = None
        except Exception:
            base.volume_ok = None
    if base.volume_ok is False:
        base.flags.append("low_volume_weak")
        if base.strength == "strong":
            base.strength = "weak"
    # VERY STRONG needs FVG >= 0.5 ATR + volume spike.
    if base.strength == "very_strong":
        if fvg["size_atr"] >= DISP_FVG_VERY_STRONG_ATR and \
                base.volume_ok:
            base.score_adj += VERY_STRONG_ADJ
        else:
            base.strength = "strong"
    if base.strength == "weak":
        base.flags.append("needs_stronger_state4")
    # Inducement: taken? retraced > 50%?
    idm = evaluate_inducement(
        frame, sweep,
        {"move": move_atr * atr, "extreme": base.extreme,
         "leg_end_pos": leg_e, "fvg": fvg},
        direction, atr, idm_level=idm_level)
    base.inducement_taken = idm["taken"]
    base.inducement_level = idm["level"]
    base.retracement_frac = idm["retracement_frac"]
    base.retracement_atr = idm["retracement_atr"]
    if idm["taken"] is False:
        base.flags.append("inducement_not_taken")
        base.score_adj += IDM_NOT_TAKEN_ADJ
    if idm["retracement_frac"] > DISP_MAX_RETRACE_FRAC:
        base.valid = False
        base.strength = "none"
        base.hard = True
        base.log = (f"displacement retraced "
                    f"{idm['retracement_frac'] * 100:.0f}% > 50%, "
                    "invalidated — back to STATE 1")
        return base
    # M5 advisory confirm (M15 is primary).
    if m5_frame is not None and len(m5_frame) >= 2:
        try:
            sweep_ts = pd.Timestamp(frame.index[sweep_pos])
            m5 = m5_frame[m5_frame.index > sweep_ts]
            m5 = m5.iloc[:DISP_MAX_M5_BARS]
            if len(m5):
                mc = m5["close"].to_numpy(dtype=float)
                far = float(np.max(mc)) if direction > 0 \
                    else float(np.min(mc))
                if (far - sweep_wick) * direction / atr \
                        >= disp_min_atr:
                    base.flags.append("m5_confirmed")
                else:
                    base.flags.append("m5_not_confirmed")
        except Exception:
            pass
    base.log = (f"displacement {base.strength}: "
                f"{move_atr:.2f} ATR in {base.bars} M15 bars, "
                f"FVG {fvg['bottom']:.1f}-{fvg['top']:.1f}, BOS "
                f"{bos_lvl}, inducement "
                f"{'taken' if idm['taken'] else 'not taken'}")
    return base
