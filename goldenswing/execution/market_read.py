"""Standing market read for /status: trend + key zones + scenarios.

Owner request (2026-10-26): the analysis message (overall trend,
key zones, bullish/bearish/neutral scenarios) WITH the engine status
attached — one message carrying both.  This module is the read half:
a pure function from engine state to plain text.  The status builder
in :mod:`goldenswing.execution.mt5_commands` appends it as the
``Market Read`` section; the runner's ``market_read_line`` supplies
the inputs from the live zone cache + D1 structure state.

Honesty contract (same as the assessment renderer): every level
printed here comes from a zone band or the D1 hold state — never
invented, never copied from an analyst's message.  A section whose
inputs are missing renders an explicit ``not produced`` line, never
a fabricated read.  The trend phrase is the engine's own
rule-based read (confirmed-swing sequence + pending swings +
flip requirements); it is NOT a copy of any human analyst's read
and will differ from one whenever the rules disagree.
"""

from __future__ import annotations

_ABOVE_CAP = 3
_BELOW_CAP = 2


def _swing_phrase(highs: list, lows: list) -> str:
    """Structure phrase from the last two confirmed swings per side."""
    if len(highs) < 2 or len(lows) < 2:
        return (f"insufficient confirmed swings "
                f"({len(highs)} highs, {len(lows)} lows)")
    lh = highs[-1] < highs[-2]
    hh = highs[-1] > highs[-2]
    ll = lows[-1] < lows[-2]
    hl = lows[-1] > lows[-2]
    if lh and ll:
        return "bearish — lower highs and lower lows"
    if hh and hl:
        return "bullish — higher highs and higher lows"
    if lh and hl:
        return "mixed — lower highs, higher lows (compression)"
    if hh and ll:
        return "mixed — higher highs, lower lows (expansion)"
    return "mixed structure"


def _band(zone: dict) -> str:
    lo = min(float(zone["bottom"]), float(zone["top"]))
    hi = max(float(zone["bottom"]), float(zone["top"]))
    return f"{lo:.1f}-{hi:.1f}"


def _zone_line(zone: dict, role: str) -> str:
    tf = f"{zone['tf']} " if zone.get("tf") else ""
    return (f"🔵 {tf}{_band(zone)}: {role} · "
            f"score {float(zone.get('score', 0)):.0f} · "
            f"{int(zone.get('touches', 0))} touches · "
            f"{zone.get('tier', 'noise')}")


def render_market_read(*, price: float | None, hold: dict | None,
                       zones: list) -> str:
    """Render the standing market read.

    ``price`` is the last tick mid (None when no tick yet).
    ``hold`` is a :class:`D1BiasState` ``to_dict()`` (None when no D1
    structure state).  ``zones`` is a list of active-zone dicts with
    ``side`` (``resistance``/``support``), ``level``, ``top``,
    ``bottom``, ``score``, ``tier``, ``touches`` (``tf`` optional).
    """
    lines: list[str] = []
    # ---- trend: the engine's own structure read -------------------
    if hold is None:
        lines.append("trend: not produced (no D1 structure state)")
    else:
        phrase = _swing_phrase(list(hold.get("confirmed_highs") or []),
                               list(hold.get("confirmed_lows") or []))
        trend = f"trend: {phrase}"
        pend: list[str] = []
        swing_n = hold.get("swing_n", 3)
        for label, cand in (("low", hold.get("pending_low")),
                            ("high", hold.get("pending_high"))):
            if cand is None:
                continue
            pend.append(
                f"pending swing {label} {float(cand['price']):.1f} "
                f"({int(cand.get('bars_confirmed', 0))}/{swing_n} bars "
                f"confirmed)")
        if pend:
            trend += " · " + ", ".join(pend)
        if hold.get("flip_needed"):
            trend += f" · {hold['flip_needed']}"
        lines.append(trend)
    # ---- key zones: bands price actually faces --------------------
    if price is None:
        lines.append("key zones: not produced (no last price)")
        lines.append("scenarios: not produced (no last price)")
        return "\n".join(lines)
    inside: list = []
    above_all: list = []
    below_all: list = []
    for z in zones:
        lo = float(z["bottom"])
        hi = float(z["top"])
        lvl = float(z["level"])
        if lo <= price <= hi:
            inside.append(z)
        elif z.get("side") == "resistance" and lvl > price:
            above_all.append(z)
        elif z.get("side") == "support" and lvl < price:
            below_all.append(z)
        # A zone on the wrong side of price (former resistance now
        # underneath, or vice versa) is not a level price faces
        # next — it is skipped, never relabelled.
    above = sorted(above_all,
                   key=lambda z: float(z["level"]))[:_ABOVE_CAP]
    below = sorted(below_all,
                   key=lambda z: -float(z["level"]))[:_BELOW_CAP]
    lines.append("key zones:")
    if inside:
        for z in inside:
            lines.append(_zone_line(z, "price inside"))
    for z in above:
        lines.append(_zone_line(z, "overhead supply"))
    for z in below:
        lines.append(_zone_line(z, "demand"))
    if not inside and not above and not below:
        lines.append("  no active zones in book")
    # ---- scenarios: triggers are zone bands, never inventions ------
    lines.append("scenarios:")
    if above:
        nxt = _band(above[1]) if len(above) > 1 \
            else "no further supply in book"
        lines.append(
            f"  bullish: a sustained break and hold above "
            f"{float(above[0]['top']):.1f} could extend toward {nxt}")
    else:
        lines.append("  bullish: not produced "
                     "(no overhead supply in book)")
    if below:
        tgt = _band(below[1]) if len(below) > 1 \
            else "no further demand in book"
        if above:
            lines.append(
                f"  bearish: rejection at {_band(above[0])} followed by "
                f"a break under {float(below[0]['bottom']):.1f} would "
                f"favor a retest of {tgt}")
        else:
            lines.append(
                f"  bearish: a break under "
                f"{float(below[0]['bottom']):.1f} would favor a "
                f"retest of {tgt}")
    else:
        lines.append("  bearish: not produced "
                     "(no demand below in book)")
    if inside:
        lines.append(
            f"  neutral: price inside {_band(inside[0])} — "
            f"consolidation inside the zone")
    elif above and below:
        lines.append(
            f"  neutral: consolidation between {_band(below[0])} and "
            f"{_band(above[0])} would indicate a pause before the "
            f"next directional move")
    else:
        lines.append("  neutral: not produced (one-sided book)")
    return "\n".join(lines)
