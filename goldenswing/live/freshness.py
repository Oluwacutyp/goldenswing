"""STATE 1 freshness: is an unmitigated zone from the 3-year DB
truly ACTIVE, or already CONSUMED?

The stored status in ``zones.db`` is a *claim*, never proof.  Every
STATE 1 scan re-checks each candidate against the bars — mitigation
/ break / touches / expiry / volume — and arms only zones that
pass ALL of the ACTIVE criteria.  This module is PURE: it never
mutates the store.  The store's own ``compute()`` stays the durable
writer; freshness is the arming-time verifier.

ACTIVE = ALL of:
- mitigation_status = UNMITIGATED (stored status ``active``)
- No close INSIDE the zone since formation that fully mitigated
  (wick into the zone is a touch, never mitigation)
- No break: no 2 consecutive closes BEYOND the zone + BOS in the
  opposite direction
- Not expired: HTF (D1/W1/MN1) never expires on age; H4 (MEDIUM)
  expires after 100 H4 bars untested; H1/M15/M5 (MINOR) after
  20 D1 bars untested
- Touches: >= 1 respected touch (wick in + close away) OR fresh
  (age < 5 D1 bars, no touch yet, displacement >= 2 ATR)
- Score >= 60 (MAJOR) — or 40-59 MINOR, which arms with an
  extra-confirm flag (STATE 1 logs MINOR, needs CHoCH+FVG+
  displacement; no entry on MINOR alone)

CONSUMED / MITIGATED = ANY of:
- FULL MITIGATION: close INSIDE + move >= 0.5 ATR beyond the zone
  in the mitigation direction (OB filled, liquidity taken).
  With volume data, the mitigation close must ALSO print
  >= average volume to count as full; a low-volume mitigation
  close is WEAKENED, not consumed.
- PARTIAL MITIGATION: close inside but no 0.5 ATR push beyond.
  WEAKENED, score -20, stays active but flagged — needs stronger
  confirm (CHoCH+FVG+displacement+volume).
- BREAK: 2 consecutive closes BEYOND + BOS.  BROKEN, drop.
  One close beyond then back inside within 2 bars = FALSE
  BREAK: stays ACTIVE, score -10, flagged.
- 3+ touches: 3rd touch without displacement (>= 1.5 ATR
  excursion away after the touch) = TIRED, score -15, flagged
  "3rd touch, tired, needs displacement".  4th touch =
  EXHAUSTED, drop.
- Volume (when available): mitigation close with above-average
  volume = stronger mitigation (full); low volume = weak.

Effective score = store score recomputed WITHOUT the store's
own weak_break penalty (freshness owns that adjustment) +
STATE 1 adjustments (weakened -20, tired -15, false break -10),
clamped to [0, 100].  Adjustments are evaluation-only, never
written back to the store.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .multitf import find_swings
from .zones import SWING_N, score_zone

MITIGATION_PUSH_ATR = 0.5
FALSE_BREAK_RETURN_BARS = 2
TIRED_TOUCH = 3
EXHAUSTED_TOUCH = 4
FRESH_AGE_D1 = 5.0
FRESH_DISP_ATR = 2.0
TIRED_DISP_ATR = 1.5
MEDIUM_EXPIRY_H4_BARS = 100
MINOR_EXPIRY_D1 = 20.0
WEAKENED_ADJ = -20.0
TIRED_ADJ = -15.0
FALSE_BREAK_ADJ = -10.0


@dataclass
class FreshnessVerdict:
    """One zone's STATE 1 verdict."""

    verdict: str  # active | consumed | broken | exhausted |
    # expired | dormant
    zone_tf: str = ""
    zone_level: float = 0.0
    effective_score: float = 0.0
    tier: str = "noise"
    flags: list = field(default_factory=list)
    touches: int = 0
    age_d1: float = 0.0
    displacement_atr: float = 0.0
    mitigation_ts: str | None = None
    mitigation_price: float | None = None
    mitigation_push_atr: float = 0.0
    log: str = ""

    @property
    def arms(self) -> bool:
        return self.verdict == "active"

    @property
    def needs_stronger_confirm(self) -> bool:
        return bool({"weakened", "tired", "minor_needs_confirm",
                     "false_break"} & set(self.flags))


# ---------------------------------------------------------------- helpers

def _zf(zone, name, default=None):
    """Zone field, StructuralZone attribute or dict key."""
    if isinstance(zone, dict):
        return zone.get(name, default)
    return getattr(zone, name, default)


def _fmt_age(age_d1: float) -> str:
    if age_d1 >= 365:
        return f"{age_d1 / 365.0:.1f}y"
    if age_d1 >= 30:
        return f"{age_d1 / 30.0:.1f}mo"
    return f"{age_d1:.0f}d"


def _bos_m15(m15, direction: int, up_to_ts) -> bool:
    """M15 BOS in ``direction`` using only bars <= ``up_to_ts``.

    No / short M15 -> True (closes alone decide; same documented
    fallback as the store's ``_bos`` — never fabricated).
    """
    if m15 is None or len(m15) < 2 * SWING_N + 5:
        return True
    try:
        sub = m15.loc[:pd.Timestamp(up_to_ts)]
    except Exception:
        return True
    if len(sub) < 2 * SWING_N + 5:
        return True
    try:
        highs, lows = find_swings(
            sub["high"].to_numpy(dtype=float),
            sub["low"].to_numpy(dtype=float), SWING_N)
        close = float(sub.iloc[-1]["close"])
    except Exception:
        return True
    if direction > 0:
        return bool(highs) and close > highs[-1][1]
    return bool(lows) and close < lows[-1][1]


def _bars_since(frame, ts) -> pd.DataFrame:
    """Bars with ts strictly after ``ts`` (empty when ts is None)."""
    if ts is None:
        return frame
    try:
        cut = pd.Timestamp(ts)
        return frame[frame.index > cut]
    except Exception:
        return frame


def _scan_lifecycle(frame, zone, atr: float, m15=None) -> dict:
    """Full lifecycle scan of one zone over its own-TF bars.

    Returns touches, full/partial mitigation (+ ts / price /
    push / volume ratio), break, false break, displacement after
    the 3rd touch, and events newer than ``last_tested`` (bars
    since the last DB update — the stale-DB re-check).
    """
    side = _zf(zone, "side")
    level = float(_zf(zone, "level", 0.0))
    top = float(_zf(zone, "top", level))
    bottom = float(_zf(zone, "bottom", level))
    swing_ts = _zf(zone, "swing_ts")
    last_tested = _zf(zone, "last_tested")
    out = {"touches": 0, "full_mitigation": False,
           "partial_mitigation": False, "mitigation_ts": None,
           "mitigation_price": None, "mitigation_push_atr": 0.0,
           "mitigation_volume_ratio": None, "break": False,
           "break_ts": None, "false_break": False,
           "disp_after_3rd_touch": 0.0, "events_since_update": []}
    if frame is None or len(frame) == 0 or atr <= 0:
        return out
    bars = _bars_since(frame, swing_ts)
    if len(bars) == 0:
        bars = frame
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    idx = bars.index
    vols = None
    if "volume" in bars.columns:
        try:
            vols = bars["volume"].to_numpy(dtype=float)
        except Exception:
            vols = None
    breakout_dir = 1 if side == "resistance" else -1
    far = top if side == "resistance" else bottom
    near = bottom if side == "resistance" else top
    consec_beyond = 0
    touch_count = 0
    third_touch_pos = None
    max_exc_after_third = 0.0
    i = 0
    while i < len(bars):
        close, high, low = float(c[i]), float(h[i]), float(l[i])
        ts = idx[i]
        new_event = last_tested is None or pd.Timestamp(ts) > \
            pd.Timestamp(last_tested)
        # Excursion away (for displacement-after-touch).
        if third_touch_pos is not None:
            exc = (level - low) / atr if side == "resistance" \
                else (high - level) / atr
            max_exc_after_third = max(max_exc_after_third, exc)
        inside = bottom <= close <= top
        beyond = (close > far) if side == "resistance" \
            else (close < far)
        if inside:
            # Mitigation candidate: push >= 0.5 ATR beyond in
            # the mitigation direction on LATER bars.
            push = 0.0
            push_ts = None
            for j in range(i + 1, len(bars)):
                if side == "resistance":
                    push = max(push, (near - float(c[j])) / atr)
                else:
                    push = max(push, (float(c[j]) - near) / atr)
                if push >= MITIGATION_PUSH_ATR:
                    push_ts = idx[j]
                    break
            vol_ratio = None
            if vols is not None and i > 0:
                mean_v = float(np.mean(vols[max(0, i - 20):i]))
                vol_ratio = (float(vols[i]) / mean_v) \
                    if mean_v > 0 else None
            out["mitigation_push_atr"] = max(
                out["mitigation_push_atr"], push)
            if push >= MITIGATION_PUSH_ATR:
                # Volume decides full vs weakened (no volume
                # data -> full, the store's historical behaviour).
                if vol_ratio is not None and vol_ratio < 1.0:
                    out["partial_mitigation"] = True
                    out["mitigation_volume_ratio"] = vol_ratio
                else:
                    out["full_mitigation"] = True
                    out["mitigation_ts"] = str(push_ts or ts)
                    out["mitigation_price"] = close
                    out["mitigation_volume_ratio"] = vol_ratio
                if new_event:
                    out["events_since_update"].append(
                        "mitigation" if out["full_mitigation"]
                        else "partial_mitigation")
                if out["full_mitigation"]:
                    break
            else:
                out["partial_mitigation"] = True
                out["mitigation_volume_ratio"] = vol_ratio
                if new_event:
                    out["events_since_update"].append(
                        "partial_mitigation")
            # An inside close resets the beyond counter; the
            # zone stays scannable for later touches.
            consec_beyond = 0
            i += 1
            continue
        if beyond:
            consec_beyond += 1
            if consec_beyond >= 2:
                if _bos_m15(m15, breakout_dir, ts):
                    out["break"] = True
                    out["break_ts"] = str(ts)
                    if new_event:
                        out["events_since_update"].append("break")
                    break
            i += 1
            continue
        # Away close: false-break check (a lone beyond close
        # followed by a return within 2 bars) + respected touch.
        if consec_beyond == 1:
            out["false_break"] = True
            if new_event:
                out["events_since_update"].append("false_break")
        consec_beyond = 0
        reached = (high >= near) if side == "resistance" \
            else (low <= near)
        if reached:
            touch_count += 1
            if touch_count == TIRED_TOUCH:
                third_touch_pos = i
            if new_event:
                out["events_since_update"].append("touch")
        i += 1
    out["touches"] = touch_count
    out["disp_after_3rd_touch"] = max_exc_after_third
    return out


# ---------------------------------------------------------------- verdict

def evaluate_zone_freshness(zone, frame, atr: float, *, m15=None,
                            others=None) -> FreshnessVerdict:
    """STATE 1 freshness verdict for one zone against its bars."""
    tf = str(_zf(zone, "tf", ""))
    level = float(_zf(zone, "level", 0.0))
    stored_status = str(_zf(zone, "status", "active"))
    stored_score = float(_zf(zone, "score", 0.0) or 0.0)
    stored_touches = int(_zf(zone, "touches", 0) or 0)
    stored_disp = float(_zf(zone, "displacement_atr", 0.0) or 0.0)
    swing_ts = _zf(zone, "swing_ts")
    last_ts = frame.index[-1] if frame is not None and len(frame) \
        else None
    try:
        age_d1 = float((pd.Timestamp(last_ts)
                        - pd.Timestamp(swing_ts)).total_seconds()
                        / 86400.0) if swing_ts is not None \
            and last_ts is not None else float(
                _zf(zone, "age_d1", 0.0) or 0.0)
    except Exception:
        age_d1 = float(_zf(zone, "age_d1", 0.0) or 0.0)

    def verdict(v, flags, log, **kw):
        return FreshnessVerdict(
            verdict=v, zone_tf=tf, zone_level=level,
            effective_score=kw.pop("score", stored_score),
            tier=kw.pop("tier", _zf(zone, "tier", "noise")),
            flags=list(flags), touches=kw.pop(
                "touches", stored_touches),
            age_d1=age_d1, displacement_atr=stored_disp,
            mitigation_ts=kw.pop("mitigation_ts", None),
            mitigation_price=kw.pop("mitigation_price", None),
            mitigation_push_atr=kw.pop("mitigation_push_atr", 0.0),
            log=log)

    if stored_status != "active":
        return verdict("consumed", ["stored_status_" + stored_status],
                       f"STATE 1 skip: {tf} {level:.1f} stored "
                       f"status {stored_status} — not unmitigated")
    if frame is None or len(frame) == 0 or atr <= 0:
        return verdict("dormant", ["no_bars_to_verify"],
                       f"STATE 1 skip: {tf} {level:.1f} no bars "
                       "to verify freshness — never arm blind")
    life = _scan_lifecycle(frame, zone, atr, m15=m15)
    touches = life["touches"] or stored_touches

    # Base score WITHOUT the store's weak_break penalty:
    # freshness owns the false-break adjustment (-10).
    try:
        import copy as _copy
        zc = _copy.copy(zone) if not isinstance(zone, dict) \
            else dict(zone)
        if isinstance(zc, dict):
            zc["weak_break"] = False
        else:
            zc.weak_break = False
        base, _ = score_zone(zc, others or [zone], age_d1=age_d1)
    except Exception:
        base = stored_score

    if life["full_mitigation"]:
        push = life["mitigation_push_atr"]
        return verdict(
            "consumed", ["full_mitigation"],
            f"STATE 1 skip: {tf} {level:.1f} mitigated "
            f"{life['mitigation_ts']}, close inside + "
            f"{push:.1f} ATR beyond, consumed",
            score=base, touches=touches,
            mitigation_ts=life["mitigation_ts"],
            mitigation_price=life["mitigation_price"],
            mitigation_push_atr=push)
    if life["break"]:
        return verdict(
            "consumed", ["break"],
            f"STATE 1 skip: {tf} {level:.1f} broken "
            f"{life['break_ts']}, 2 closes beyond + BOS, consumed",
            score=base, touches=touches)
    if touches >= EXHAUSTED_TOUCH:
        return verdict(
            "exhausted", ["exhausted"],
            f"STATE 1 skip: {tf} {level:.1f} exhausted, "
            f"{touches} touches, liquidity likely taken",
            score=base, touches=touches)
    # Expiry: HTF never; H4 after 100 H4 bars untested; MINOR
    # after 20 D1 bars untested.  "Untested" = no respected touch
    # and no inside close since formation.
    untested = touches == 0 and not life["partial_mitigation"]
    if tf == "H4" and untested:
        bars_since = len(_bars_since(frame, _zf(zone, "last_tested")
                                     or swing_ts))
        if bars_since > MEDIUM_EXPIRY_H4_BARS:
            return verdict(
                "expired", ["expired"],
                f"STATE 1 skip: {tf} {level:.1f} expired, "
                f"{bars_since} H4 bars untested",
                score=base, touches=touches)
    if tf in ("H1", "M15", "M5") and untested and age_d1 > MINOR_EXPIRY_D1:
        return verdict(
            "expired", ["expired"],
            f"STATE 1 skip: {tf} {level:.1f} expired, "
            f"{age_d1:.0f}d untested",
            score=base, touches=touches)
    # Touch / fresh rule.
    fresh = age_d1 < FRESH_AGE_D1 and touches == 0 \
        and stored_disp >= FRESH_DISP_ATR
    if touches == 0 and not fresh:
        return verdict(
            "dormant", ["no_touch_not_fresh"],
            f"STATE 1 skip: {tf} {level:.1f} dormant, no "
            f"respected touch, age {age_d1:.0f}d, not fresh",
            score=base, touches=touches)
    # ACTIVE, with STATE 1 adjustments (evaluation-only).
    flags: list[str] = []
    adj = 0.0
    if life["partial_mitigation"]:
        flags.append("weakened")
        adj += WEAKENED_ADJ
    if touches >= TIRED_TOUCH and \
            life["disp_after_3rd_touch"] < TIRED_DISP_ATR:
        flags.append("tired")
        adj += TIRED_ADJ
    if life["false_break"]:
        flags.append("false_break")
        adj += FALSE_BREAK_ADJ
    if fresh:
        flags.append("fresh")
    eff = max(0.0, min(100.0, base + adj))
    tier = "MAJOR" if eff >= 60.0 else (
        "MINOR" if eff >= 40.0 else "noise")
    if tier == "MINOR":
        flags.append("minor_needs_confirm")
    if tier == "noise":
        return verdict(
            "dormant", flags + ["score_noise"],
            f"STATE 1 skip: {tf} {level:.1f} dormant, effective "
            f"score {eff:.0f} < 40", score=eff, tier=tier,
            touches=touches)
    desc = (f"STATE 1 HTF_ZONE_APPROACH: {tf} {level:.1f}, "
            f"score {eff:.0f}, {touches} touches, unmitigated, "
            f"age {_fmt_age(age_d1)}"
            + (", fresh" if fresh else "")
            + (", WEAKENED (partial mitigation, needs stronger "
               "confirm)" if "weakened" in flags else "")
            + (", TIRED (3rd touch, needs displacement)"
               if "tired" in flags else "")
            + (", MINOR (needs CHoCH+FVG+displacement)"
               if "minor_needs_confirm" in flags else ""))
    return verdict("active", flags, desc, score=eff, tier=tier,
                   touches=touches)


# ---------------------------------------------------------------- scan

@dataclass
class State1Scan:
    """Result of a STATE 1 scan: armed zone (or None) + skip logs."""

    armed: FreshnessVerdict | None = None
    skipped: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)


def scan_state1(zones, frames: dict, price: float, atr: float, *,
                radius_atr: float = 2.0) -> State1Scan:
    """STATE 1 scan: UNMITIGATED zones within ``radius_atr`` of
    ``price``, score desc, each re-checked against bars SINCE
    the last DB update (bars after ``last_tested`` — stale DB
    status is never trusted).  First ACTIVE zone arms; consumed
    zones log a skip and the scan moves on."""
    out = State1Scan()
    if not zones or atr <= 0:
        return out
    cands = [z for z in zones
             if str(_zf(z, "status", "")) == "active"
             and abs(float(_zf(z, "level", 0.0)) - price)
             <= radius_atr * atr]
    cands.sort(key=lambda z: -float(_zf(z, "score", 0.0) or 0.0))
    m15 = (frames or {}).get("M15")
    for zone in cands:
        frame = (frames or {}).get(_zf(zone, "tf"))
        v = evaluate_zone_freshness(zone, frame, atr, m15=m15,
                                    others=cands)
        out.verdicts.append(v)
        if v.arms:
            dist = abs(v.zone_level - price)
            v.log = (v.log + f", distance {dist:.1f} "
                     f"({dist / atr:.1f} ATR)") \
                if "distance" not in v.log else v.log
            out.armed = v
            return out
        out.skipped.append(v.log)
    return out
