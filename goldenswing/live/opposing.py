"""Opposing zones + risk tiering: how real traders handle
barriers, without aborting the setup.

RISK TIERING — High / Mid / Safe, NOT for gating.  A tier
never blocks a trade alone; it only adjusts size and
management.  Even RED zones can trade, with tighter
management.

- GREEN Safe / Institutional Grade: multi-year W1/MN1 zones,
  heavy volume, structural OB, aligned LTF confirms,
  displacement >= 2 ATR, score >= 80.  (v2/A1: no touch-count
  requirement — freshness is strength under the consumption
  model, so a fresh W1 zone can be GREEN.)
  Full size 1.0x, TP 1R/2R/3R, time stop 12 H1, trailing
  normal, runner allowed.
- YELLOW Mid Risk: clean displacement + one prior respected
  touch, but lacking MTF alignment, score 60-79, or single
  TF, volume moderate.  Size 0.75x, TP1 1R 50%, TP2 1.8R
  30% (before opposing), TP3 2.5R 20% or exit at opposing
  zone, time stop 8 H1, trailing tighter, runner smaller.
- RED High Risk / Low Confluence: minor intraday zones,
  single-touch, zones formed during low-volume hours
  (Asian, weekend), score 40-59, no MTF alignment, no
  displacement.  Size 0.5x, TP1 0.75R 50%, TP2 1.5R 30%,
  TP3 2R 20% or exit all at opposing zone, time stop 6 H1,
  trailing M5 tight, no runner if opposing within 2R,
  aggressive profit-taking.

Tier + Footprint + State Machine together determine
management, never a single gate.  Entry tier decides SIZE;
the opposing zone decides TP, not size.

OPPOSING ZONES — how real traders handle them:
- Zone Sandwich: price trapped between a nearby macro
  resistance + immediate support (demand vs supply
  opposing).  Professional traders do NOT look for swing
  home runs there; they treat it as a range-bound scalping
  environment / interim liquidity target.  If price is
  between two opposing MAJOR zones with distance < 3 ATR,
  style = RANGE_SCALP, TP = opposite zone, not +3R,
  holding minutes-hours.
- Two zones opposing each other, both ACTIVE: the engine
  does NOT pick one blindly, does NOT cancel both.  It
  WAITS for price to reach one zone + sweep / displacement
  / CHoCH+FVG at THAT zone.  First zone to trigger the
  chain wins; the other zone becomes TP / opposing
  barrier.  Never trade both directions at the same time
  from the same sandwich (no hedge, no simultaneous
  long+short on the same zone pair, max 1 position per
  sandwich).
- Mid-range: price equidistant from both zones, > 1 ATR
  from either -> NO setup, "mid-range, no zone, no
  trade", wait for price to reach a zone.  Real traders
  don't trade mid-range; they wait for the zone.

OPPOSING BARRIER (TP adjustment, never abort):
- TP1 stays (it is before the opposing zone).
- TP2 MOVES: TP2 = opposing zone - 0.1 ATR buffer (e.g.
  +2R -> +1.7R when opposing sits at +1.8R), NOT +2R
  (which would run into supply and get wicked out).
- TP3 REMOVED / CAPPED: no runner beyond opposing unless
  opposing breaks with displacement + CHoCH + FVG.  The
  remaining % exits at TP2 as well, unless opposing is
  MID-RISK (YELLOW), in which case 20% runner is held
  ONLY if opposing breaks.
- Trailing tightens to behind opposing - 0.25 ATR; if
  price approaches opposing within 0.25 ATR, trail SL to
  breakeven + 0.5R (lock profit).
- Opposing GREEN SAFE (W1/MN1, score >= 80, unmitigated,
  heavy volume): TP2 = opposing - 0.1 ATR, TP3 = NONE,
  exit ALL at TP2.  Green opposing = hard barrier,
  respect fully.
- Opposing RED HIGH RISK (score < 60, single touch, low
  volume): TP2/TP3 UNCHANGED (ignore red opposing, it
  will likely break).
- General rule: TP must be BEFORE the opposing zone (TP =
  opposing - 0.1 ATR buffer), unless opposing is RED or
  opposing breaks with displacement + CHoCH.  First-target
  priority: primary TP is set before the opposing zone,
  never run +3R into unmitigated monthly supply.
"""

from __future__ import annotations

SANDWICH_MAX_ATR = 3.0
MID_RANGE_MIN_ATR = 1.0
TP_BUFFER_ATR = 0.1
TRAIL_CAP_ATR = 0.25
TRAIL_LOCK_R = 0.5

TIER_SPECS = {
    "GREEN": {"size_mult": 1.0, "tp_r": (1.0, 2.0, 3.0),
              "tp_pct": (0.50, 0.25, 0.25), "time_stop_h1": 12,
              "trail_tf": "H1", "trail_step_atr": 0.5,
              "allow_runner": True},
    "YELLOW": {"size_mult": 0.75, "tp_r": (1.0, 1.8, 2.5),
               "tp_pct": (0.50, 0.30, 0.20), "time_stop_h1": 8,
               "trail_tf": "H1", "trail_step_atr": 0.25,
               "allow_runner": True},
    "RED": {"size_mult": 0.5, "tp_r": (0.75, 1.5, 2.0),
            "tp_pct": (0.50, 0.30, 0.20), "time_stop_h1": 6,
            "trail_tf": "M5", "trail_step_atr": 0.25,
            "allow_runner": False},
}

TIER_NAMES = {"GREEN": "GREEN SAFE", "YELLOW": "YELLOW MID-RISK",
              "RED": "RED HIGH-RISK"}


def _get(zone, key, default=None):
    if zone is None:
        return default
    if isinstance(zone, dict):
        return zone.get(key, default)
    return getattr(zone, key, default)


def risk_tier(zone) -> str:
    """Risk tier of a zone: GREEN / YELLOW / RED.  Tiering
    never gates — it only adjusts size and management."""
    score = float(_get(zone, "score", 0.0) or 0.0)
    # A1 (v2): GREEN no longer requires >= 2 touches — under the
    # consumption model the FRESHEST zone is the strongest, and the
    # old rule made a fresh W1 zone un-GREEN-able (the contradiction
    # that surfaced the inverted sign).  Touch count no longer
    # appears in the tier rule at all: consumption is priced by the
    # corrected score (zones.py A1) and the lifecycle class (A2).
    disp = float(_get(zone, "displacement_atr", 0.0) or 0.0)
    tf = str(_get(zone, "tf", "") or "")
    vol = _get(zone, "volume_ratio")
    if score >= 80 and disp >= 2.0 and \
            tf in ("W1", "MN1"):
        if vol is not None and float(vol) < 1.0:
            return "YELLOW"
        return "GREEN"
    if score >= 60:
        return "YELLOW"
    return "RED"


def tier_spec(tier: str) -> dict:
    return dict(TIER_SPECS.get(tier, TIER_SPECS["YELLOW"]))


def _is_active_major(zone) -> bool:
    status = str(_get(zone, "status", "active") or "active")
    tier = str(_get(zone, "tier", "") or "")
    score = float(_get(zone, "score", 0.0) or 0.0)
    return status == "active" and (tier == "MAJOR" or score >= 60)


def detect_sandwich(zones, price: float, atr: float) -> dict:
    """Zone sandwich: price between two opposing ACTIVE MAJOR
    zones with distance < 3 ATR -> RANGE_SCALP, TP =
    opposite zone.  Price > 1 ATR from either zone ->
    mid-range, no trade."""
    out = {"present": False, "mid_range": False, "demand": None,
           "supply": None, "distance_atr": None,
           "style": None, "log": ""}
    if not zones or atr <= 0:
        return out
    supports = [z for z in zones
                if str(_get(z, "side", "")) == "support"
                and _is_active_major(z)
                and float(_get(z, "level", 0.0)) < price]
    resistances = [z for z in zones
                   if str(_get(z, "side", "")) == "resistance"
                   and _is_active_major(z)
                   and float(_get(z, "level", 0.0)) > price]
    demand = max(supports, key=lambda z: float(_get(z, "level")))
    supply = min(resistances,
                 key=lambda z: float(_get(z, "level"))) \
        if resistances else None
    if demand is None or supply is None:
        return out
    d_level = float(_get(demand, "level"))
    s_level = float(_get(supply, "level"))
    dist = (s_level - d_level) / atr
    near = min(abs(price - d_level), abs(price - s_level)) / atr
    if near > MID_RANGE_MIN_ATR:
        out.update({"mid_range": True,
                    "demand": demand, "supply": supply,
                    "distance_atr": dist,
                    "log": ("mid-range, no zone, no trade — "
                            "wait for price to reach a zone")})
        return out
    if dist < SANDWICH_MAX_ATR:
        out.update({"present": True, "demand": demand,
                    "supply": supply, "distance_atr": dist,
                    "style": "RANGE_SCALP",
                    "log": (f"zone sandwich: demand {d_level:.1f} "
                            f"vs supply {s_level:.1f}, distance "
                            f"{dist:.1f} ATR, range scalp, TP = "
                            "opposite zone")})
    return out


def opposing_zone_for(zones, *, entry_price: float,
                      direction: int) -> dict | None:
    """Nearest ACTIVE MAJOR opposing zone beyond the entry in
    the trade direction: for a long, the nearest resistance
    above; for a short, the nearest support below.  None when
    no opposing zone blocks the path."""
    if not zones or direction not in (1, -1):
        return None
    cands = []
    for z in zones:
        if not _is_active_major(z):
            continue
        lvl = float(_get(z, "level", 0.0))
        side = str(_get(z, "side", ""))
        if direction > 0 and side == "resistance" \
                and lvl > entry_price:
            cands.append(z)
        if direction < 0 and side == "support" \
                and lvl < entry_price:
            cands.append(z)
    if not cands:
        return None
    best = min(cands, key=lambda z: abs(
        float(_get(z, "level")) - entry_price))
    return {"level": float(_get(best, "level")),
            "tf": str(_get(best, "tf", "")),
            "score": float(_get(best, "score", 0.0) or 0.0),
            "tier": risk_tier(best),
            "touches": int(_get(best, "touches", 0) or 0),
            "displacement_atr": float(
                _get(best, "displacement_atr", 0.0) or 0.0),
            "age_d1": float(_get(best, "age_d1", 0.0) or 0.0),
            "mitigation_status": str(_get(best, "status",
                                          "active")),
            "last_tested": _get(best, "last_tested"),
            "last_pattern": _get(best, "last_pattern"),
            "side": str(_get(best, "side", ""))}


def adjust_tps(*, tp1: float, tp2: float, tp3: float | None,
               tp1_pct: float, tp2_pct: float, tp3_pct: float,
               entry_price: float, stop: float, direction: int,
               opposing: dict | None, atr: float) -> dict:
    """Adjust TP2/TP3 for an opposing zone (never abort the
    setup).  Entry tier decides size; opposing decides TP.

    Returns the adjusted TPs + flags + the desk log line.
    TP3 = None means removed (exit all at TP2).
    """
    out = {"tp1": tp1, "tp2": tp2, "tp3": tp3,
           "tp1_pct": tp1_pct, "tp2_pct": tp2_pct,
           "tp3_pct": tp3_pct, "exit_all_at_tp2": False,
           "tp3_runner_if_opposing_breaks": False,
           "allow_runner": True,
           "trail_cap_price": None, "trail_lock_r": None,
           "opposing_r": None, "moved": False, "log": ""}
    if not opposing or atr <= 0:
        out["log"] = "no opposing zone, TP unchanged"
        return out
    r = abs(entry_price - stop)
    if r <= 0 or direction not in (1, -1):
        return out
    o_level = float(opposing["level"])
    o_tier = str(opposing.get("tier", "YELLOW"))
    o_r = (o_level - entry_price) * direction / r
    out["opposing_r"] = o_r
    before = o_level - direction * TP_BUFFER_ATR * atr
    tp2_r = (tp2 - entry_price) * direction / r
    tp1_r = (tp1 - entry_price) * direction / r
    if o_tier == "RED":
        out["log"] = (f"opposing RED, weak, TP unchanged, "
                      "expect break")
        return out
    # TP1 itself runs into opposing -> collapse to the
    # barrier: exit all at TP1-before-opposing.  Equality
    # counts: a TP exactly ON the barrier is not BEFORE it.
    if o_r <= tp1_r:
        out["tp1"] = before
        out["tp1_pct"] = tp1_pct + tp2_pct + tp3_pct
        out["tp2"] = before
        out["tp2_pct"] = 0.0
        out["tp3"] = None
        out["tp3_pct"] = 0.0
        out["exit_all_at_tp2"] = True
        out["allow_runner"] = False
        out["moved"] = True
        out["log"] = (f"opposing {TIER_NAMES.get(o_tier, o_tier)} "
                      f"at +{o_r:.1f}R before TP1, TP1 moved to "
                      "opposing - 0.1 ATR, exit all there")
        return out
    if o_tier == "GREEN":
        out["tp2"] = before
        out["tp2_pct"] = tp2_pct + tp3_pct
        out["tp3"] = None
        out["tp3_pct"] = 0.0
        out["exit_all_at_tp2"] = True
        out["allow_runner"] = False
        out["moved"] = True
        out["trail_cap_price"] = o_level - direction \
            * TRAIL_CAP_ATR * atr
        out["trail_lock_r"] = TRAIL_LOCK_R
        out["log"] = (f"opposing GREEN SAFE at +{o_r:.1f}R, "
                      "exit all before barrier, no runner")
        return out
    # YELLOW (MID-RISK): TP2 moves if it would run into (or
    # land exactly on) opposing; TP3 runner held ONLY if
    # opposing breaks.
    if o_r <= tp2_r:
        old_tp2_r = tp2_r
        out["tp2"] = before
        out["moved"] = True
        new_tp2_r = (before - entry_price) * direction / r
        out["log"] = (f"TP2 moved: +{old_tp2_r:.1f}R -> "
                      f"+{new_tp2_r:.1f}R, opposing supply at "
                      f"+{o_r:.1f}R, exit before barrier")
    else:
        out["log"] = (f"opposing YELLOW MID-RISK at +{o_r:.1f}R, "
                      "beyond TP2, no move")
    out["tp3_runner_if_opposing_breaks"] = True
    out["trail_cap_price"] = o_level - direction \
        * TRAIL_CAP_ATR * atr
    out["trail_lock_r"] = TRAIL_LOCK_R
    if out["moved"]:
        out["log"] += ("; TP3: no runner beyond opposing "
                       "unless opposing breaks with "
                       "displacement + CHoCH")
    else:
        out["log"] += ("; TP3 runner only if opposing breaks")
    return out


def apply_tier_to_management(plan: dict, tier: str, *,
                             entry_price: float, stop: float,
                             direction: int) -> dict:
    """Combine a confirm-ladder plan (clean/late) with the
    entry zone's risk tier.  The most conservative of the two
    ladders wins per TP; size = min(confirm, tier); time
    stop / trail step = min; runner needs both to allow it.
    Tiering never gates — it only adjusts."""
    spec = tier_spec(tier)
    out = dict(plan)
    out["size_mult"] = min(float(plan.get("size_mult", 1.0)),
                           float(spec["size_mult"]))
    r = abs(entry_price - stop)
    if r > 0 and direction in (1, -1):
        for i, key in enumerate(("tp1_r", "tp2_r", "tp3_r")):
            plan_r = float(plan.get(key, 0.0))
            tier_r = float(spec["tp_r"][i])
            if tier_r < plan_r:
                out[key] = tier_r
                out[f"tp{i + 1}_pct"] = float(spec["tp_pct"][i])
            out[f"tp{i + 1}"] = entry_price + direction * r \
                * float(out[key])
        out["max_r"] = min(float(plan.get("max_r", 3.0)),
                           float(spec["tp_r"][2]))
    out["time_stop_h1"] = min(int(plan.get("time_stop_h1", 12)),
                              int(spec["time_stop_h1"]))
    out["trail_step_atr"] = min(
        float(plan.get("trail_step_atr", 0.5)),
        float(spec["trail_step_atr"]))
    if spec["trail_tf"] == "M5":
        out["trail_tf"] = "M5"
    out["allow_runner"] = bool(plan.get("allow_runner", True)) \
        and bool(spec["allow_runner"])
    out["entry_tier"] = tier
    return out


def describe_opposing(opposing: dict | None,
                      adjustment: dict | None) -> str:
    """One /status line for the opposing barrier + the TP
    adjustment it caused."""
    if not opposing:
        return "Opposing: none"
    o_r = adjustment.get("opposing_r") if adjustment else None
    r_txt = f" at +{o_r:.1f}R" if o_r is not None else ""
    adj = ""
    if adjustment:
        if adjustment.get("exit_all_at_tp2"):
            adj = ", TP2 moved, TP3 removed, exit all at TP2"
        elif adjustment.get("moved"):
            adj = ", TP2 moved before barrier"
        elif adjustment.get("tp3_runner_if_opposing_breaks"):
            adj = ", TP3 runner only if opposing breaks"
    return (f"Opposing: {opposing.get('side', '')} "
            f"{opposing['level']:.1f}{r_txt}, "
            f"{TIER_NAMES.get(opposing.get('tier'), opposing.get('tier'))}"
            f"{adj}")
