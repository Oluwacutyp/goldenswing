"""Whole-market read for /status: the MarketSituation snapshot.

SPEC §9, Pass 1 demo proof: "/status shows the whole-market
read (per-TF states + relationships)".  The situation engine
(live/situation.py + live/relationships.py) was built and
journaled by Passes 0–2 and is read by later passes' organs,
but no surface ever rendered the snapshot itself — the /status
Market Read section is the older zone-book read, which stays.
This module is the render half, in market_read.py's shape: a
pure function from the latest snapshot dict to plain text.
The status builder appends it as the ``Market Situation``
section; the runner's ``situation_line`` supplies the snapshot
from the SituationKeeper.

Honesty contract (the situation's own, SPEC §3.1): absent
stays absent.  An unmeasured timeframe renders its readiness,
never a fabricated state; an unmeasured pair renders
unmeasured with its reason; a situation-level block the
runner could not supply renders ``not produced`` and is
named in the data-quality missing list the situation itself
carries.  Relative strength vs silver/DXY has no live feed
in the runner — the situation carries it None, and this
render says so in words.  It never prints a number the
snapshot does not carry, and never 0 for an absent one.
"""

from __future__ import annotations

_TFS = ("D1", "H4", "H1", "M30", "M15", "M5")


def _num(value, digits: int = 1) -> str | None:
    """A carried number, formatted — or None when absent."""
    if value is None:
        return None
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return None


def _bias_word(bias) -> str:
    if bias is None:
        return "bias unmeasured"
    if bias > 0:
        return "bull"
    if bias < 0:
        return "bear"
    return "flat"


def _tf_line(tf: str, state: dict | None) -> str:
    if not isinstance(state, dict):
        return f"  {tf}: not produced"
    readiness = state.get("readiness")
    bars = int(state.get("bars") or 0)
    if readiness != "ready":
        if bars:
            return (f"  {tf}: collecting ({bars} bars — state "
                    f"not measured yet)")
        return f"  {tf}: not ready (no bars — state not measured yet)"
    parts = [_bias_word(state.get("bias"))]
    phase = state.get("structure_phase")
    direction = state.get("direction_word")
    if phase and direction:
        parts.append(f"{phase}/{direction}")
    elif phase:
        parts.append(str(phase))
    vol = state.get("volatility") or {}
    atr = _num(vol.get("atr"))
    if atr is not None:
        seg = f"atr {atr}"
        ratio = _num(vol.get("atr_ratio"), 2)
        if ratio is not None:
            seg += f" ×{ratio}"
        if vol.get("atr_state"):
            seg += f" {vol['atr_state']}"
        parts.append(seg)
    event = state.get("last_structure_event")
    if isinstance(event, dict) and event.get("kind"):
        seg = f"last event {event['kind']}"
        direction_n = event.get("direction") or 0
        if direction_n:
            seg += " ↑" if direction_n > 0 else " ↓"
        level = _num(event.get("level"))
        if level is not None:
            seg += f" {level}"
        if event.get("against_bias"):
            seg += " (against bias)"
        parts.append(seg)
    swings = state.get("swings") or {}
    swing_parts = []
    high = swings.get("confirmed_high") or {}
    low = swings.get("confirmed_low") or {}
    if _num(high.get("price")) is not None:
        swing_parts.append(f"H {_num(high.get('price'))}")
    if _num(low.get("price")) is not None:
        swing_parts.append(f"L {_num(low.get('price'))}")
    if swing_parts:
        parts.append("swings " + " ".join(swing_parts))
    fresh = _num(state.get("freshness_s"), 0)
    if fresh is not None:
        parts.append(f"fresh {fresh}s")
    return f"  {tf}: " + " · ".join(parts)


def _relationship_line(rel: dict) -> str:
    prefix = "hierarchy " if rel.get("source") == "hierarchy" else ""
    pair = f"{rel.get('htf')}→{rel.get('ltf')}"
    kind = rel.get("kind")
    reason = str(rel.get("reason") or "")
    if kind is None:
        if reason == "insufficient_state":
            return f"  {prefix}{pair}: unmeasured (insufficient state)"
        if reason:
            return f"  {prefix}{pair}: unmeasured — {reason}"
        return f"  {prefix}{pair}: unmeasured"
    if reason:
        return f"  {prefix}{pair}: {kind} — {reason}"
    return f"  {prefix}{pair}: {kind}"


def _regime_line(regime: dict | None) -> str:
    if not isinstance(regime, dict):
        return "  regime: not produced"
    parts = []
    label = regime.get("bedrock_label_h1")
    if label:
        parts.append(f"bedrock H1: {label}")
    day = regime.get("day")
    if isinstance(day, dict):
        seg = []
        if day.get("day_type"):
            seg.append(str(day["day_type"]))
        adr = _num(day.get("adr_pct"), 0)
        if adr is not None:
            adr_seg = f"ADR {adr}%"
            if day.get("adr_state"):
                adr_seg += f" ({day['adr_state']})"
            seg.append(adr_seg)
        if day.get("opening_type"):
            seg.append(f"open {day['opening_type']}")
        if seg:
            parts.append("day: " + " · ".join(seg))
    if not parts:
        return "  regime: (no label measured yet)"
    return "  regime: " + " · ".join(parts)


def _location_lines(location: dict | None) -> list[str]:
    if not isinstance(location, dict):
        return ["  location: not produced"]
    parts = []
    price = _num(location.get("price"))
    if price is not None:
        parts.append(f"price {price}")
    value = location.get("value")
    if isinstance(value, dict):
        poc = _num(value.get("poc"))
        vah = _num(value.get("vah"))
        val = _num(value.get("val"))
        if poc is not None and vah is not None and val is not None:
            seg = f"value POC {poc} · VA {val}–{vah}"
            if value.get("developing"):
                seg += " (developing)"
            parts.append(seg)
    pd_read = location.get("premium_discount")
    if isinstance(pd_read, dict) and pd_read.get("zone"):
        seg = f"{pd_read['zone']} side"
        eq = _num(pd_read.get("equilibrium"))
        if eq is not None:
            seg += f" · eq {eq}"
        parts.append(seg)
    lines = []
    if parts:
        lines.append("  location: " + " · ".join(parts))
    else:
        lines.append("  location: (nothing measured yet)")
    parts = []
    levels = location.get("day_levels")
    if isinstance(levels, dict):
        ib_lo = _num(levels.get("ib_low"))
        ib_hi = _num(levels.get("ib_high"))
        if ib_lo is not None and ib_hi is not None:
            parts.append(f"IB {ib_lo}–{ib_hi}")
        on_lo = _num(levels.get("overnight_low"))
        on_hi = _num(levels.get("overnight_high"))
        if on_lo is not None and on_hi is not None:
            parts.append(f"overnight {on_lo}–{on_hi}")
    opening = location.get("opening_range")
    if isinstance(opening, dict):
        o_lo = _num(opening.get("low"))
        o_hi = _num(opening.get("high"))
        if o_lo is not None and o_hi is not None:
            seg = f"opening range {o_lo}–{o_hi}"
            width = _num(opening.get("width_atr"))
            if width is not None:
                seg += f" ({width} ATR wide)"
            parts.append(seg)
    pools_above = location.get("pools_above") or []
    pools_below = location.get("pools_below") or []
    if pools_above or pools_below:
        parts.append(f"pools {len(pools_above)} above / "
                     f"{len(pools_below)} below")
    magnet = _num(location.get("magnet_room_r"))
    if magnet is not None:
        parts.append(f"magnet room {magnet}R (measurement only)")
    extension = _num(location.get("extension_atr"))
    if extension is not None:
        parts.append(f"extension {extension} ATR vs developing POC")
    if parts:
        lines.append("    " + " · ".join(parts))
    return lines


def _activity_line(activity: dict | None) -> str:
    if not isinstance(activity, dict):
        return "  activity: not produced"
    parts = []
    if activity.get("session"):
        parts.append(str(activity["session"]))
    if activity.get("killzone"):
        parts.append(f"killzone {activity['killzone']}")
    spread = _num(activity.get("spread"), 2)
    if spread is not None:
        seg = f"spread {spread}"
        ratio = _num(activity.get("spread_ratio"), 2)
        if ratio is not None:
            seg += f" ×{ratio}"
        parts.append(seg)
    rvol = activity.get("rvol")
    if isinstance(rvol, dict):
        h1_rvol = _num(rvol.get("H1"))
        if h1_rvol is not None:
            parts.append(f"rvol H1 {h1_rvol}")
    vwap = _num(activity.get("vwap_session"))
    if vwap is not None:
        parts.append(f"VWAP {vwap}")
    blackout = activity.get("news_blackout")
    if blackout is True:
        parts.append("news blackout ON")
    elif blackout is False:
        parts.append("news clear")
    if not parts:
        return "  activity: (nothing measured yet)"
    return "  activity: " + " · ".join(parts)


def _rs_segment(name: str, rs) -> str:
    """One relative-strength read: the situation carries None
    (no live feed in the runner) — said in words, never a
    number.  A measured dict, if one ever exists, renders its
    trend."""
    if not isinstance(rs, dict):
        return f"{name} not produced (no live feed)"
    trend = rs.get("trend")
    word = {1: "up", -1: "down", 0: "flat"}.get(trend, "unmeasured")
    return f"{name} trend {word}"


def _cross_line(cross: dict | None) -> str:
    if not isinstance(cross, dict):
        return "  cross-asset: not produced"
    parts = []
    dxy = cross.get("dxy")
    if isinstance(dxy, dict):
        if dxy.get("available"):
            corr = _num(dxy.get("correlation"), 2)
            seg = f"DXY corr {corr}" if corr is not None \
                else "DXY available"
            bonus = _num(dxy.get("bonus"), 2)
            if bonus is not None:
                seg += f" (bonus {bonus})"
            parts.append(seg)
        elif dxy.get("status"):
            parts.append(f"DXY unavailable ({dxy['status']})")
        else:
            parts.append("DXY unavailable")
    smt = cross.get("smt_event")
    if isinstance(smt, dict) and smt.get("kind"):
        seg = f"SMT {smt['kind']}"
        direction = smt.get("direction") or 0
        if direction:
            seg += " ↑" if direction > 0 else " ↓"
        parts.append(seg)
    else:
        parts.append("SMT none")
    parts.append(_rs_segment("silver RS", cross.get("silver_rs")))
    parts.append(_rs_segment("DXY RS", cross.get("dxy_rs")))
    return "  cross-asset: " + " · ".join(parts)


def _quality_line(quality: dict | None) -> str:
    if not isinstance(quality, dict):
        return "  data quality: not produced"
    parts = []
    health = quality.get("health")
    if health:
        seg = f"health {health}"
        reason = quality.get("reason")
        if reason and health != "healthy":
            seg += f" — {reason}"
        parts.append(seg)
    missing = quality.get("missing") or []
    if missing:
        parts.append("missing: " + ", ".join(str(m) for m in missing))
    else:
        parts.append("missing: none")
    return "  data quality: " + " · ".join(parts)


def render_situation_read(situation: dict | None) -> str:
    """Render the standing whole-market read from the latest
    MarketSituation snapshot dict (``SituationKeeper.latest()``).

    ``None`` — no snapshot yet — renders the explicit
    not-produced line, never an empty section.
    """
    if not isinstance(situation, dict):
        return ("market situation: not produced yet "
                "(no snapshot — the read builds on the "
                "first timeframe close)")
    lines: list[str] = []
    created = situation.get("created_at")
    symbol = situation.get("symbol") or ""
    head = f"as of {created}" if created else "as of —"
    if symbol:
        head += f" · {symbol}"
    lines.append(head)
    per_tf = situation.get("per_tf") or {}
    for tf in _TFS:
        lines.append(_tf_line(tf, per_tf.get(tf)))
    for rel in situation.get("relationships") or []:
        if isinstance(rel, dict):
            lines.append(_relationship_line(rel))
    lines.append(_regime_line(situation.get("regime")))
    lines.extend(_location_lines(situation.get("location")))
    lines.append(_activity_line(situation.get("activity")))
    lines.append(_cross_line(situation.get("cross_asset")))
    lines.append(_quality_line(situation.get("data_quality")))
    return "\n".join(lines)
