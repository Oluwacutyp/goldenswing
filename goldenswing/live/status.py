"""Professional desk /status overview for an actively
managed trade under opposing-zone rules.

One output, all data points together, no missing fields.
Plain text, one section per block, no JSON dump, human
readable — like a professional desk.

``build_snapshot`` assembles the snapshot from the machine's
own objects (narrative state, management plan, zones,
footprint).  Feed / margin / journal come from the runner
(they are live data, passed in as dicts).  ``format_trade_status``
renders the snapshot.  Every field has an "n/a" fallback —
the formatter never crashes on a partial snapshot — but a
complete snapshot renders every field in the spec.

Sections:
1. REGIME + STATE MACHINE
2. ZONES (persistent 3-year store)
3. TRADE STATE (trade, size, SL, TPs, current, footprint,
   management, pending orders)
4. FEED + EXECUTION (feed, margin, journal)

The runner wires this into the /status command by passing
its live feed / margin / journal dicts; the snapshot
builder covers everything the machine itself knows.
"""

from __future__ import annotations

STATE_NAMES = {
    0: "IDLE", 1: "HTF_ZONE_APPROACH", 2: "LIQUIDITY_SWEEP",
    3: "INDUCEMENT_DISPLACEMENT", 4: "LTF_CONFIRMATION",
    5: "ENTRY", 6: "MANAGEMENT", 7: "EXIT",
}


def _v(value, fmt="{:.1f}", default="n/a"):
    if value is None or value == "":
        return default
    try:
        return fmt.format(value)
    except Exception:
        return str(value)


def _money(value):
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


def _r(value):
    if value is None:
        return "n/a"
    return f"{value:+.1f}R" if value >= 0 else f"{value:.1f}R"


def _zone_line(label: str, zone) -> str:
    if not zone:
        return f"{label}: none"
    tier = zone.get("tier", "n/a")
    return (f"{label}: {zone.get('tf', 'n/a')} "
            f"{_v(zone.get('level'))} "
            f"(score {_v(zone.get('score'), '{:.0f}')} tier {tier}, "
            f"touches {_v(zone.get('touches'), '{:.0f}')}, "
            f"displacement {_v(zone.get('displacement_atr'))} ATR, "
            f"age {_v(zone.get('age_d1'), '{:.1f}')}d, "
            f"{zone.get('mitigation_status', 'n/a')}, "
            f"last tested {zone.get('last_tested', 'n/a')}, "
            f"last pattern {zone.get('last_pattern', 'n/a')})")


def build_snapshot(*, narrative=None, management=None,
                   entry_zone=None, opposing=None,
                   tp_adjustment=None, footprint=None,
                   feed=None, margin=None, journal=None,
                   zone_cache=None, sandwich=None,
                   nearest_zones=None, pending_orders=None,
                   current=None, trade=None,
                   min_lot=None) -> dict:
    """Assemble a snapshot.  ``narrative`` is a NarrativeState
    (or its payload dict); everything else is an optional
    dict.  Missing pieces stay None and render as n/a."""
    snap: dict = {"trade": dict(trade or {}),
                  "current": dict(current or {}),
                  "feed": dict(feed or {}),
                  "margin": dict(margin or {}),
                  "journal": dict(journal or {}),
                  "footprint": dict(footprint or {}),
                  "zone_cache": dict(zone_cache or {}),
                  "pending_orders": dict(pending_orders or {})}
    st = narrative
    if st is not None and not isinstance(st, dict):
        try:
            st = st.payload()
        except Exception:
            st = {}
    st = dict(st or {})
    snap["regime"] = st.get("regime", "unknown")
    snap["regime_since"] = st.get("regime_since")
    snap["regime_lock_remaining_h1"] = st.get(
        "regime_lock_remaining_h1")
    snap["breakout_confirm_count"] = st.get(
        "breakout_confirm_count")
    snap["regime_state_file"] = st.get("regime_state_file")
    snap["regime_state_last_saved"] = st.get(
        "regime_state_last_saved")
    snap["regime_state_version"] = st.get("regime_state_version")
    snap["state"] = st.get("state")
    snap["state_since"] = st.get("state_since")
    snap["state_narrative"] = st.get("state_narrative")
    snap["narrative_log"] = list(st.get("narrative") or [])
    snap["management"] = dict(management or
                             st.get("management") or {})
    snap["entry_zone"] = dict(entry_zone or st.get("zone") or {})
    snap["opposing"] = dict(opposing or st.get("opposing") or {})
    snap["tp_adjustment"] = dict(tp_adjustment or {})
    snap["sandwich"] = dict(sandwich or {})
    snap["nearest_zones"] = list(nearest_zones or [])
    snap["min_lot"] = dict(min_lot or
                           (snap["trade"].get("plan") or {})
                           .get("min_lot") or {})
    # Trade basics from the narrative entry when the caller
    # did not pass an explicit trade dict.
    entry = st.get("entry") or {}
    if entry and not snap["trade"]:
        snap["trade"] = {
            "direction": st.get("direction"),
            "entry_type": entry.get("entry_type"),
            "entry_price": entry.get("price"),
            "stop": entry.get("stop"),
            "targets": entry.get("targets"),
            "invalidation": entry.get("invalidation"),
            "plan": entry.get("plan") or {},
        }
    return snap


def format_trade_status(snap: dict) -> str:
    """Render the complete desk overview (plain text)."""
    s = dict(snap or {})
    lines = ["===== GOLDENSWING — ACTIVE TRADE DESK =====", ""]
    # 1. REGIME + STATE MACHINE
    lines.append("--- 1. REGIME + STATE MACHINE ---")
    lines.append(f"Regime: {s.get('regime', 'n/a')}")
    lines.append(f"Regime since: {s.get('regime_since', 'n/a')}")
    lines.append("Regime lock remaining: "
                 f"{_v(s.get('regime_lock_remaining_h1'), '{:.0f}')} H1 bars")
    lines.append("Breakout confirm count: "
                 f"{_v(s.get('breakout_confirm_count'), '{:.0f}')}/2")
    lines.append("Regime state: persisted "
                 f"({s.get('regime_state_file', 'n/a')}, last saved "
                 f"{s.get('regime_state_last_saved', 'n/a')}, "
                 f"version {_v(s.get('regime_state_version'), '{:.0f}')})")
    st = s.get("state")
    st_name = STATE_NAMES.get(st, "n/a") if isinstance(st, int) \
        else str(st or "n/a")
    lines.append(f"State: STATE {st} {st_name}")
    lines.append(f"State since: {s.get('state_since', 'n/a')}")
    lines.append("Narrative: "
                 f"{s.get('state_narrative') or ' '.join(s.get('narrative_log') or []) or 'n/a'}")
    lines.append("")
    # 2. ZONES
    lines.append("--- 2. ZONES (persistent 3-year store) ---")
    lines.append(_zone_line("Entry zone", s.get("entry_zone")))
    opp = s.get("opposing") or {}
    adj = s.get("tp_adjustment") or {}
    if opp:
        lines.append(_zone_line("Opposing zone", opp))
        lines.append(f"Opposing distance: "
                     f"{_r(adj.get('opposing_r'))} "
                     f"({_v(adj.get('opposing_distance_price'))} price, "
                     f"{_v(adj.get('opposing_distance_atr'))} ATR)")
        lines.append(f"Opposing TP adjustment: "
                     f"{adj.get('log', 'n/a')}")
    else:
        lines.append("Opposing zone: none")
    nearest = s.get("nearest_zones") or []
    if nearest:
        lines.append("Nearest MAJOR zones:")
        for i, z in enumerate(nearest[:3], 1):
            lines.append(f"  {i}. {_zone_line('', z).replace(': ', '', 1)}"
                         f" — distance {_v(z.get('distance_atr'))} ATR")
    else:
        lines.append("Nearest MAJOR zones: n/a")
    zc = s.get("zone_cache") or {}
    lines.append(f"Zone cache: {_v(zc.get('active'), '{:.0f}')} active zones, "
                 f"{_v(zc.get('nearby'), '{:.0f}')} nearby, last refresh "
                 f"{zc.get('last_refresh', 'n/a')}, query p95 "
                 f"{_v(zc.get('query_p95_ms'))}ms")
    sw = s.get("sandwich") or {}
    if sw.get("present"):
        lines.append(f"Zone sandwich: demand "
                     f"{_v(sw.get('demand_level'))} vs supply "
                     f"{_v(sw.get('supply_level'))}, distance "
                     f"{_v(sw.get('distance_atr'))} ATR, style "
                     "RANGE_SCALP")
    elif sw.get("mid_range"):
        lines.append("Zone sandwich: none (mid-range, no zone, "
                     "no trade)")
    else:
        lines.append("Zone sandwich: none")
    lines.append("")
    # 3. TRADE STATE
    lines.append("--- 3. TRADE STATE ---")
    tr = s.get("trade") or {}
    direction = tr.get("direction")
    d_txt = "LONG" if direction == 1 else (
        "SHORT" if direction == -1 else "n/a")
    mgmt = s.get("management") or {}
    plan = tr.get("plan") or {}
    style = tr.get("style") or mgmt.get("style") or "n/a"
    lines.append(f"Trade: {d_txt} {_v(tr.get('lots'))} @ "
                 f"{_v(tr.get('entry_price'))}, "
                 f"entry type {tr.get('entry_type', 'n/a')}, "
                 f"entry time {tr.get('entry_time', 'n/a')}, "
                 f"fill {tr.get('fill_status', 'n/a')}, "
                 f"style {style}")
    lines.append(f"Size: lots {_v(tr.get('lots'))}, multiplier "
                 f"{_v(mgmt.get('size_mult') or plan.get('size_mult'))}x, "
                 f"reason {tr.get('size_reason', mgmt.get('confirm_kind', 'n/a'))}, "
                 f"$ risk {_money(tr.get('risk_usd'))}, "
                 f"risk {_v(tr.get('risk_pct'))}% equity")
    # Min-lot floor: the min order is 0.01 lots; SL is
    # invalidation, never shrunk to fit a risk target.
    ml = s.get("min_lot") or {}
    if ml:
        from .min_lot import describe_min_lot
        lines.append(describe_min_lot(
            ml, equity=float(
                (s.get("margin") or {}).get("equity") or 0.0)))
    lines.append(f"SL: {_v(tr.get('stop'))} "
                 f"({_v(tr.get('sl_distance_atr'))} ATR, "
                 f"{_r(tr.get('sl_distance_r'))}), basis "
                 f"{tr.get('sl_basis', 'min(sweep, FVG, CHoCH) - buffer')}, "
                 f"moved: {tr.get('sl_moved', 'n/a')}, "
                 f"trailing level {_v(tr.get('trailing_level'))}")
    # TPs: prefer the management plan (it carries the adjusted
    # ladder); fall back to the entry plan targets.
    tp_rows = []
    for i, key in enumerate(("tp1", "tp2", "tp3"), 1):
        price = mgmt.get(key) if mgmt.get(key) is not None \
            else (tr.get("targets") or [None, None, None])[i - 1] \
            if tr.get("targets") else None
        tp_rows.append((i, price,
                        mgmt.get(f"tp{i}_r"),
                        mgmt.get(f"tp{i}_pct"),
                        tr.get(f"tp{i}_status", "pending"),
                        tr.get(f"tp{i}_adjustment", "")))
    for i, price, r_mult, pct, status, adj_reason in tp_rows:
        if price is None and i == 3 and adj.get("tp3") is None \
                and opp:
            lines.append(f"TP{i}: NONE (removed, opposing "
                         f"{opp.get('tier', '')})")
            continue
        pct_txt = f"{pct * 100:.0f}%" if isinstance(pct, (int, float)) else "n/a"
        extra = f" ({adj_reason})" if adj_reason else ""
        if i == 2 and adj.get("moved"):
            extra = (f" (moved from "
                     f"+{_v(tr.get('tp2_original_r'))}R, "
                     f"{adj.get('log', '')})")
        lines.append(f"TP{i}: {_v(price)} {_r(r_mult)} {pct_txt} "
                     f"{status}{extra}")
    cur = s.get("current") or {}
    lines.append(f"Current: price {_v(cur.get('price'))}, "
                 f"distance to SL {_v(cur.get('distance_to_sl'))}, "
                 f"to TP1 {_v(cur.get('distance_to_tp1'))}, "
                 f"to TP2 {_v(cur.get('distance_to_tp2'))}, "
                 f"to opposing {_v(cur.get('distance_to_opposing'))}, "
                 f"unrealized P&L {_money(cur.get('pnl_usd'))} "
                 f"({_r(cur.get('pnl_r'))}), R multiple "
                 f"{_r(cur.get('r_multiple'))}, time in trade "
                 f"{cur.get('time_in_trade', 'n/a')}, time stop "
                 f"remaining {_v(cur.get('time_stop_remaining_h1'), '{:.0f}')} H1 bars")
    fp = s.get("footprint") or {}
    lines.append(f"Footprint: score {_v(fp.get('score'), '{:.0f}')} "
                 f"({fp.get('classification', 'n/a')}), "
                 f"SMT {fp.get('smt', 'n/a')}, "
                 f"AMD {fp.get('amd', 'n/a')}, "
                 f"Judas {fp.get('judas', 'n/a')}, "
                 f"CTA {fp.get('cta', 'n/a')}, "
                 f"retail trap {fp.get('retail_trap', 'n/a')}")
    lines.append(f"Management: breakeven {mgmt.get('breakeven', tr.get('breakeven', 'n/a'))}, "
                 f"trailing {mgmt.get('trailing', tr.get('trailing', 'n/a'))}, "
                 f"partials taken {tr.get('partials_taken', 'n/a')}, "
                 f"runner {mgmt.get('runner', tr.get('runner', 'n/a'))}, "
                 f"invalidation { _v(tr.get('invalidation'))}, "
                 f"next action: {tr.get('next_action', mgmt.get('next_action', 'n/a'))}")
    po = s.get("pending_orders") or {}
    lines.append(f"Pending orders: audit {po.get('audit_status', 'n/a')}, "
                 f"tickets {po.get('tickets', 'n/a')}, "
                 f"expiry {po.get('expiry_bars', 'n/a')} bars, "
                 f"decay {po.get('decay_bars', 'n/a')} bars")
    lines.append("")
    # 4. FEED + EXECUTION
    lines.append("--- 4. FEED + EXECUTION ---")
    fd = s.get("feed") or {}
    lines.append(f"Feed: {fd.get('status', 'n/a')}, last tick "
                 f"{fd.get('last_tick', 'n/a')}, last bars "
                 f"{fd.get('last_bars', 'n/a')}")
    lines.append(f"D1 bias: structure {fd.get('d1_bias_structure', 'n/a')}, "
                 f"EMA {fd.get('d1_bias_ema', 'n/a')}, "
                 f"hold {fd.get('d1_hold_state', 'n/a')}")
    lines.append(f"H4 bias: {fd.get('h4_bias', 'n/a')}, macro "
                 f"alignment {fd.get('macro_alignment', 'n/a')}")
    mg = s.get("margin") or {}
    lines.append(f"Margin: equity {_money(mg.get('equity'))}, "
                 f"used {_money(mg.get('margin_used'))}, free "
                 f"{_money(mg.get('free_margin'))}, shield "
                 f"{mg.get('shield', 'n/a')}")
    jn = s.get("journal") or {}
    lines.append(f"Journal: last entry {jn.get('last_entry', 'n/a')}, "
                 f"trades today {_v(jn.get('trades_today'), '{:.0f}')}, "
                 f"daily P&L {_money(jn.get('daily_pnl'))}, "
                 f"daily {_r(jn.get('daily_r'))}")
    return "\n".join(lines)
