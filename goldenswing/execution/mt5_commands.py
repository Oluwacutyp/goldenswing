"""Telegram command surface for the MT5 engine.

Separate registry from the OANDA loop's :data:`COMMANDS` — the MT5
engine answers /status /report /db /trades /harvest /ticks from its
OWN journal/state (``data/mt5_*``), never the OANDA loop's
``data/paper_state/`` / ``trade_journal.jsonl``.  Each process owns
its registry, so importing this module never clobbers the OANDA
handlers in-process.

Polling: the MT5 engine owns ``getUpdates`` by default (see
``MT5RunnerConfig.poll_commands``).  The OANDA loop's poller must be
OFF at cutover (``GS_TELEGRAM_POLL=0``) — one getUpdates consumer
only, or the two pollers steal each other's updates.

/harvest scopes (same caps as the OANDA harvest: 20MB/file, 45MB
ZIP, slim fallback):

    /harvest           — all (journal + state + tick CSVs + results)
    /harvest journal   — MT5 journal only
    /harvest state     — MT5 risk state only
    /harvest ticks     — tick CSVs only (the backtest feed)
    /harvest results   — RESULTS.md + manifest only (no raw files)

Chart PNGs are EXCLUDED from the bundle by design (they already
ride the Telegram alerts, and they would blow the ZIP cap).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..feed.tick_log import export_tick_history
from ..feed.mt5 import MT5Feed
from ..alerts.commands import (CommandContext, CommandResult,
                               CommandRouter, build_zip_bundle,
                               fmt_code, fmt_feed_line, fmt_next_cta,
                               fmt_none, fmt_pnl, fmt_progress, fmt_r,
                               fmt_section)
from .mt5_config import (EXECUTE_PLAYBOOK_NAMES, SHADOW_PLAYBOOK_NAMES,
                        describe_change, parse_set)

#: name -> (description, handler).  Populated by @mt5_command.
MT5_COMMANDS: dict = {}


def mt5_command(name: str, description: str):
    """Register an MT5-engine ``/command`` handler."""
    if not name.startswith("/"):
        raise ValueError(f"command name must start with '/': {name!r}")

    def deco(fn):
        MT5_COMMANDS[name] = (description, fn)
        return fn

    return deco


def mt5_router(owner_chat_id) -> CommandRouter:
    """CommandRouter bound to the MT5 registry (owner-gated), with
    the persistent-menu button texts aliased to slash commands."""
    return CommandRouter(owner_chat_id=owner_chat_id,
                         registry=MT5_COMMANDS, aliases=MENU_ALIASES)


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# ------------------------------------------------------- inline keyboards
# Callback data is ``mt5:<action>`` / ``mt5:confirm:<token>`` /
# ``mt5:cancel:<token>`` — short (Telegram caps callback_data at 64
# bytes) and namespaced so the MT5 poller never mistakes a legacy
# ``gs:<action>:<alert_id>`` button for its own.

MT5_CALLBACK_PREFIX = "mt5"


def mt5_control_keyboard() -> dict:
    """Pause / Resume / Trim / Cancel pending / Refresh (+ Config)."""
    return {"inline_keyboard": [
        [{"text": "⏸ Pause", "callback_data": "mt5:pause"},
         {"text": "▶ Resume", "callback_data": "mt5:resume"}],
        [{"text": "✂️ Trim", "callback_data": "mt5:trim"},
         {"text": "❌ Cancel pending",
          "callback_data": "mt5:cancel_pending"}],
        [{"text": "🔄 Refresh", "callback_data": "mt5:refresh"},
         {"text": "⚙️ Config", "callback_data": "mt5:config"}],
    ]}


def mt5_confirm_keyboard(token: str) -> dict:
    """Confirm / Cancel for one pending mutating action."""
    return {"inline_keyboard": [[
        {"text": "✅ Confirm", "callback_data": f"mt5:confirm:{token}"},
        {"text": "❌ Cancel", "callback_data": f"mt5:cancel:{token}"},
    ]]}


# ------------------------------------------------- persistent reply menu
# The always-visible bottom menu (reply keyboard, not inline): sent
# on /start and /menu, then KEPT by Telegram (is_persistent) across
# every later message.  It is NOT re-sent with every reply — a
# message carries exactly one reply_markup, and the inline buttons
# on status/position/pending messages need theirs; re-sending the
# menu there would clobber the inline keyboards.  Sending a message
# with an inline keyboard (or with no markup) leaves the reply
# keyboard in place, so both surfaces coexist.
MENU_KEYBOARD: dict = {
    "keyboard": [
        [{"text": "📊 Status"}, {"text": "📈 Positions"},
         {"text": "⏳ Pendings"}],
        [{"text": "🎯 Playbooks"}, {"text": "⚙️ Config"},
         {"text": "📊 Trades"}],
        [{"text": "📦 Harvest"}, {"text": "⏸️ Pause"},
         {"text": "▶️ Resume"}],
    ],
    "resize_keyboard": True,
    "one_time_keyboard": False,
    "is_persistent": True,
}

#: Reply-keyboard button text (case-insensitive) -> slash command.
#: Pause / Resume go through the same confirm flow as /pause /resume.
#: Both the plain labels ("Status") and the emoji labels ("📊 Status")
#: resolve — the button text Telegram sends back carries the emoji.
MENU_ALIASES: dict[str, str] = {
    "status": "/status",
    "positions": "/position",
    "pendings": "/pending",
    "playbooks": "/playbooks",
    "config": "/config",
    "trades": "/trades",
    "harvest": "/harvest",
    "pause": "/pause",
    "resume": "/resume",
    "📊 status": "/status",
    "📈 positions": "/position",
    "⏳ pendings": "/pending",
    "🎯 playbooks": "/playbooks",
    "⚙️ config": "/config",
    "📊 trades": "/trades",
    "📦 harvest": "/harvest",
    "⏸️ pause": "/pause",
    "▶️ resume": "/resume",
}


def parse_mt5_callback(data: str) -> tuple[str, str | None] | None:
    """Parse ``mt5:<action>[:<arg>]``; None when not an MT5 callback."""
    if not isinstance(data, str) or not data.startswith("mt5:"):
        return None
    parts = data.split(":", 2)
    if len(parts) < 2 or not parts[1]:
        return None
    return parts[1], (parts[2] if len(parts) > 2 else None)


def _control(ctx: CommandContext):
    return getattr(ctx, "mt5_control", None)


def _feed_truth(ctx: CommandContext) -> dict:
    """The runner's reconciled feed truth (Runtime Truth item 1),
    or {} when no runner / an old runner is attached — the feed
    line then falls back to the raw poll status."""
    ctrl = _control(ctx)
    if ctrl is None:
        return {}
    fn = getattr(ctrl, "feed_truth", None)
    if not callable(fn):
        return {}
    try:
        return fn() or {}
    except Exception:
        return {}


def _confirm_result(ctx: CommandContext, action: str, label: str,
                    arg=None) -> CommandResult:
    """Ask for confirmation before a mutating action (never acts here)."""
    ctrl = _control(ctx)
    if ctrl is None:
        return CommandResult(
            text=f"{label} unavailable (no live runner attached)")
    token = ctrl.request_confirm(action, arg)
    return CommandResult(
        text=f"{label} — tap Confirm to act, Cancel to abort.",
        reply_markup=mt5_confirm_keyboard(token))


def _position_lines(ctx: CommandContext) -> list[str]:
    """Open positions with layers (live terminal truth wins)."""
    live = list(getattr(ctx, "live_positions", None) or [])
    lines: list[str] = []
    if live:
        for p in live:
            side = ("LONG 🟢" if (p.get("direction") or 0) > 0
                    else "SHORT 🔴")
            entry = float(p.get("entry_price") or p.get("entry") or 0)
            stop = float(p.get("sl") or p.get("stop") or 0)
            risk = abs(entry - stop)
            r_now = "?"
            if ctx.last_price and risk > 0:
                r_now = fmt_r((p.get('direction') or 0)
                              * (ctx.last_price - entry) / risk)
            lines.append(f"#{p.get('ticket', '?')} {p.get('playbook', '?')} "
                         f"{side} @ {fmt_code(f'{entry:,.1f}')} "
                         f"sl {fmt_code(f'{stop:,.1f}')} now ≈{r_now}")
            layers = p.get("layers") or []
            if len(layers) > 1:
                legs = " + ".join(
                    f"{ly.get('playbook', '?')} @ "
                    f"{float(ly.get('entry') or 0):,.1f} "
                    f"({float(ly.get('volume') or 0)} lots)"
                    for ly in layers)
                lines.append(f"   🧱 layers ({len(layers)}): {legs}")
        return lines
    from ..execution.reporting import open_positions
    journal_opens = open_positions(ctx.journal) if ctx.journal else []
    if not journal_opens:
        return [fmt_none()]
    for o in journal_opens:
        side = ("LONG 🟢" if (o.get("direction") or 0) > 0
                else "SHORT 🔴")
        entry = float(o.get("entry") or 0)
        stop = float(o.get("stop") or 0)
        risk = abs(entry - stop)
        r_now = "?"
        if ctx.last_price and risk > 0:
            r_now = fmt_r((o.get('direction') or 0)
                          * (ctx.last_price - entry) / risk)
        lines.append(f"#{o.get('ticket', '?')} {o.get('playbook', '?')} {side} "
                     f"@ {fmt_code(f'{entry:,.1f}')} "
                     f"sl {fmt_code(f'{stop:,.1f}')} now ≈{r_now}")
    return lines


def _pending_lines(ctx: CommandContext) -> list[str]:
    """Resting LIMIT / STOP orders the engine is responsible for."""
    pendings = list(getattr(ctx, "pending_orders", None) or [])
    if not pendings:
        return [fmt_none()]
    lines = []
    for p in pendings:
        side = ("LONG 🟢" if (p.get("direction") or 0) > 0
                else "SHORT 🔴")
        level = float(p.get("level") or 0)
        sl = float(p.get("sl") or 0)
        lines.append(f"#{p.get('ticket', '?')} "
                     f"{p.get('kind', '?').upper()} "
                     f"{p.get('playbook', '?')} {side} "
                     f"@ {fmt_code(f'{level:,.1f}')} "
                     f"({float(p.get('volume') or 0)} lots) "
                     f"SL {fmt_code(f'{sl:,.1f}')} "
                     f"expires H1 bar {fmt_code(p.get('expiry_bar', '?'))}")
    return lines


def _config_text(ctx: CommandContext) -> str:
    """The safe live config only — never tokens, paths, credentials,
    or live-trading permission."""
    ctrl = _control(ctx)
    if ctrl is None:
        return ("live config unavailable (no live runner attached).\n"
                "Safe keys: max_layers, manipulation_filter, "
                "pending_expiry_bars, playbook.<name>.")
    cfg = ctrl.get_live_config()
    _k = cfg.get('risk_pct')
    lines = [fmt_section("⚙️", "MT5 LIVE CONFIG (safe keys only):"), "",
             f"`sizing` = "
             f"{fmt_code('dynamic: lots = max(broker min, floor((k x equity x C x S) / (stop_ticks x tick_value))); k = ' + (str(_k) if _k is not None else '?') + ' (proposed coefficient, not a gate); C = 1.0 (confidence definition pending)')} ",
             f"`max_layers` = {fmt_code(cfg.get('max_layers'))}",
             f"`manipulation_filter` = "
             f"{fmt_code('on' if cfg.get('manipulation_filter') else 'off')}",
             f"`pending_expiry_bars` = "
             f"{fmt_code(cfg.get('pending_expiry_bars'))}",
             f"`paused` = {fmt_code('yes' if cfg.get('paused') else 'no')}",
             "", fmt_section("🎯", "Playbooks:")]
    for name, enabled in (cfg.get("playbooks") or {}).items():
        mark = "✅" if enabled else "⏸️"
        lines.append(f"  {mark} {name}: {'on' if enabled else 'off'}")
    lines += ["", "Change with `/set <key> <value>` (confirm required).",
              "Pause is not a `/set` key — use ⏸️ Pause / ▶️ Resume.",
              fmt_next_cta("/status", "/playbooks", "/set")]
    return "\n".join(lines)


# ------------------------------------------------------------------ commands

@mt5_command("/start", "show the persistent bottom menu")
def _mt5_start(ctx: CommandContext, args: list[str]) -> CommandResult:
    return CommandResult(
        text=("🤖 GoldenSwing MT5 engine — use the menu below, or /help "
              "for the full command list.  Status / position / pending "
              "messages carry their own action buttons.\n"
              + fmt_next_cta("/status", "/playbooks", "/help")),
        reply_markup=MENU_KEYBOARD)


@mt5_command("/menu", "re-show the persistent bottom menu")
def _mt5_menu(ctx: CommandContext, args: list[str]) -> CommandResult:
    return CommandResult(
        text=("📋 MT5 menu — 📊 Status / 📈 Positions / ⏳ Pendings / "
              "🎯 Playbooks / ⚙️ Config / 📊 Trades / 📦 Harvest / "
              "⏸️ Pause / ▶️ Resume."),
        reply_markup=MENU_KEYBOARD)


@mt5_command("/playbooks", "per-playbook state: executing vs shadow, on/off, signal counts")
def _mt5_playbooks(ctx: CommandContext, args: list[str]) -> CommandResult:
    ctrl = _control(ctx)
    enabled: dict[str, bool] = {}
    counts: dict[str, int] = {}
    if ctrl is not None:
        enabled = (ctrl.get_live_config().get("playbooks") or {})
        counts = (ctrl.blocked_stats().get("signal_counts") or {})
    lines = [fmt_section("🎯", "MT5 PLAYBOOKS"),
             "executing = may order · shadow = evaluated + logged, "
             "never ordered", ""]
    for name in EXECUTE_PLAYBOOK_NAMES:
        on = enabled.get(name, True)
        mark = "✅" if on else "⏸️"
        lines.append(f"  {mark} {name}: executing, "
                     f"{'on' if on else 'off'}, "
                     f"signals {counts.get(name, 0)}")
    for name in SHADOW_PLAYBOOK_NAMES:
        on = enabled.get(name, True)
        mark = "✅" if on else "⏸️"
        lines.append(f"  {mark} {name}: shadow, "
                     f"{'on' if on else 'off'}, "
                     f"signals {counts.get(name, 0)}")
    lines += ["", "Toggle with `/set playbook.<name> on/off` "
                  "(confirm required).",
              fmt_next_cta("/status", "/config", "/set")]
    return CommandResult(text="\n".join(lines),
                         reply_markup=mt5_control_keyboard())


@mt5_command("/help", "list the MT5 engine commands")
def _mt5_help(ctx: CommandContext, args: list[str]) -> CommandResult:
    lines = [fmt_section("📋", "MT5 engine commands"),
             "This engine's own journal/state:", ""]
    for name in sorted(MT5_COMMANDS):
        desc, _ = MT5_COMMANDS[name]
        lines.append(f"`{name}` — {desc}")
    lines += ["", fmt_next_cta("/status", "/playbooks", "/health")]
    return CommandResult(text="\n".join(lines))


@mt5_command("/status", "MT5 engine health: positions+layers, pendings, blocked-leg stats, today's P&L")
def _mt5_status(ctx: CommandContext, args: list[str]) -> CommandResult:
    from ..execution.reporting import (iter_paper_trades, open_positions)
    fs = ctx.feed_status or {}
    ctrl = _control(ctx)
    cfg = ctrl.get_live_config() if ctrl is not None else None

    lines = [fmt_section("📊", "MT5 ENGINE STATUS"), ""]
    # Feed — reconciled with the Health section below from ONE
    # source (Runtime Truth item 1): the runner's feed_truth()
    # carries market state + tick state, so this line can no
    # longer claim a bare OK while Health reports a stale-tick
    # breach on a closed market.  Guarded — old runners lack it.
    lines.append(fmt_section("📡", "Feed"))
    _truth = _feed_truth(ctx)
    lines.append(fmt_feed_line(
        fs, ctx.last_bar_time, ctx.last_price,
        market_open=_truth.get("market_open"),
        tick_state=_truth.get("tick_state"),
        tick_age_s=_truth.get("tick_age_s")))
    if ctrl is not None:
        # Per-TF last closed bars (guarded — old runners lack it).
        # The old /status showed only the H1 last bar, which read
        # as "feed is H1-only": the stack carries all six TFs.
        stack_line = getattr(ctrl, "stack_line", None)
        if callable(stack_line):
            lines.append(stack_line())
    lines.append("")
    # Live setup: armed / not-armed + why.  Guarded — old runners
    # lack it.  This is the answer to "why no trades" when no
    # playbook has been reached yet (blocked legs stay empty
    # until a trigger fires, so they cannot answer it alone).
    if ctrl is not None:
        setup_line = getattr(ctrl, "setup_line", None)
        if callable(setup_line):
            lines.append(fmt_section("🎯", "Live Setup"))
            lines.append(setup_line())
            lines.append("")
    # Edge boundary (sweep vs breakout) + persisted hysteresis
    # clock + regime pause + pending-audit summary.  Guarded —
    # old runners lack it.  Never hidden: a corrupt state /
    # pause shows here, not behind a healthy-looking setup.
    if ctrl is not None:
        boundary_line = getattr(ctrl, "boundary_line", None)
        if callable(boundary_line):
            lines.append(fmt_section("🧱", "Boundary"))
            lines.append(boundary_line())
            lines.append("")
    # Multi-day structural zones (top MAJOR).  Guarded — old
    # runners lack it.
    if ctrl is not None:
        zones_line = getattr(ctrl, "zones_line", None)
        if callable(zones_line):
            lines.append(fmt_section("🏛️", "Structural Zones"))
            lines.append(zones_line())
            lines.append("")
    # P3 day line: the trading day as an object (day type, ADR
    # consumption, IB state, posture, session, NY checkpoint).
    # Guarded — old runners lack it.
    if ctrl is not None:
        day_line = getattr(ctrl, "day_line", None)
        if callable(day_line):
            lines.append(fmt_section("🗓️", "Day"))
            lines.append(day_line())
            lines.append("")
    # Engine state
    lines.append(fmt_section("⚙️", "Engine"))
    if cfg is not None and cfg.get("paused"):
        lines.append("⏸️ engine: PAUSED — new signals blocked, tick "
                     "management (breakeven / trail / partial) continues")
    else:
        lines.append("🟢 engine: RUNNING")
    lines.append("")
    if ctrl is not None:
        # MacroTrend alignment (D1/W1/MN1): sizing + Golden Swing
        # confirmation state.  Guarded — old runners lack it.
        alignment = getattr(ctrl, "alignment_line", None)
        if callable(alignment):
            lines.append(fmt_section("🌍", "Macro Alignment"))
            lines.append(alignment())
            lines.append("")
        # DXY correlation: confluence context, never a gate.
        # Guarded — old runners lack it.
        dxy = getattr(ctrl, "dxy_line", None)
        if callable(dxy):
            lines.append(fmt_section("💵", "DXY"))
            lines.append(dxy())
            lines.append("")
        # Execution health: tick freshness / order latency /
        # slippage.  Guarded — old runners lack it.
        health = getattr(ctrl, "health_line", None)
        if callable(health):
            lines.append(fmt_section("❤️", "Health"))
            lines.append(health())
            lines.append("")
    if cfg is not None:
        lines.append(fmt_section("🔧", "Config"))
        lines.append(
            f"config: `sizing=`"
            f"{fmt_code('dynamic k=' + str(cfg.get('risk_pct')) + ' x equity x C x S, floored at broker min; C=1.0 pending' if cfg.get('risk_pct') is not None else 'dynamic')} "
            f"`max_layers=`{fmt_code(cfg.get('max_layers'))} "
            f"`manipulation_filter=`"
            f"{fmt_code('on' if cfg.get('manipulation_filter') else 'off')} "
            f"`pending_expiry=`{fmt_code(cfg.get('pending_expiry_bars'))} bars")
        lines.append("")
    # Live positions from the terminal (MT5 = source of truth) win
    # over journal opens: they include restart-adopted positions that
    # have no signal record.  Journal opens are the fallback.
    lines.append(fmt_section("📈", "Open Positions"))
    lines += _position_lines(ctx)
    lines.append("")
    lines.append(fmt_section("⏳", "Pending Orders"))
    lines += _pending_lines(ctx)

    trades = iter_paper_trades(ctx.journal) if ctx.journal else []
    today = _today_utc()
    todays = [t for t in trades if (t["closed_ts"] or "")[:10] == today]
    day_r = sum(t["r_net"] for t in todays)
    day_pnl = sum(t["pnl_money"] for t in todays)
    lines += ["", fmt_section("📅", "Today (UTC)"),
              f"today (UTC): {len(todays)} closed, "
              f"{fmt_r(day_r)} ({fmt_pnl(day_pnl)})"]
    opens = open_positions(ctx.journal) if ctx.journal else []
    if opens:
        last_open = max(opens, key=lambda r: r.get("ts") or "")
        lines.append(f"last signal: {last_open.get('playbook', '?')} "
                     f"{last_open.get('ts', '?')}")
    else:
        lines.append("last signal: none journaled yet")

    # Why no trades: blocked-leg stats + per-playbook signal counts.
    if ctrl is not None:
        stats = ctrl.blocked_stats()
        lines.append("")
        lines.append(fmt_section("🚫", "why no trades — blocked legs:"))
        legs = stats.get("legs") or {}
        if not legs and not stats.get("manipulation_blocked"):
            lines.append(f"  {fmt_none()} (nothing blocked yet — "
                         f"see Live Setup above for why no trigger "
                         f"has fired)")
        for leg, count in sorted(legs.items(), key=lambda kv: -kv[1]):
            lines.append(f"  ⛔ `{leg}`: {count}")
        if stats.get("manipulation_blocked"):
            lines.append(f"  ⛔ `manipulation_filter`: "
                         f"{stats['manipulation_blocked']}")
        lines.append(f"  last blocked: {stats.get('last') or '(none)'}")
        # Runtime Truth item 2: when the order path blocked
        # nothing but the funnel shows candidates ending earlier,
        # SAY SO here — the EC2 status showed "31/31 + 10/10
        # ineligible" in one section and "(nothing blocked)" in
        # this one, with no bridge between them.
        if not legs:
            _ftot = getattr(ctrl, "funnel_totals", None)
            if callable(_ftot):
                try:
                    _t = _ftot() or {}
                except Exception:
                    _t = {}
                if _t.get("ineligible") or _t.get("refused"):
                    lines.append(
                        f"  ➖ order path blocked nothing — "
                        f"candidates ended earlier: "
                        f"{_t.get('ineligible', 0)} ineligible · "
                        f"{_t.get('refused', 0)} risk-refused "
                        f"(reason codes in Decision Funnel below)")
        counts = stats.get("signal_counts") or {}
        lines.append("")
        lines.append(fmt_section("🎯", "Per-Playbook Signals"))
        lines.append("per-playbook signals (passed gate+score):")
        if not counts:
            lines.append(f"  {fmt_none()} yet")
        for name in sorted(counts):
            lines.append(f"  🎯 `{name}`: {counts[name]}")
    # FUNNEL (§1): the decision funnel — which playbooks were
    # EVALUATED (per clock), where candidates disappeared, and the
    # armed plan's last disposition.  This is the section that
    # answers "why no trades" at the correct stage instead of by
    # guesswork (directive §3 exit criterion).  Guarded — old
    # runners lack it.
    if ctrl is not None:
        funnel = getattr(ctrl, "funnel_line", None)
        if callable(funnel):
            lines.append("")
            lines.append(fmt_section("🧭", "Decision Funnel"))
            lines.append(funnel())
    lines.append(fmt_next_cta("/playbooks", "/health", "/trades"))
    return CommandResult(text="\n".join(lines),
                         reply_markup=mt5_control_keyboard())


@mt5_command("/assess", "the engine's latest market assessment: /assess [playbook|all]")
def _mt5_assess(ctx: CommandContext, args: list[str]) -> CommandResult:
    """Visible market reading (owner milestone, 2026-10-23):
    render the LATEST funnel assessment per playbook from THIS
    engine's journal — structure read, itemized evidence,
    competing scenarios, verdict + plain-language reason, with
    absent capabilities marked, never invented.  ``/assess``
    shows the newest assessment plus a per-playbook verdict
    index; ``/assess <playbook>`` one playbook in full;
    ``/assess all`` the whole roster (a document when the text
    would exceed a Telegram message, the /recap pattern)."""
    from .assessment_render import (is_shadow_record,
                                    latest_evaluations,
                                    record_provenance,
                                    render_assessment)
    if ctx.journal is None:
        return CommandResult(text="no journal attached.")
    try:
        records = ctx.journal.records()
    except Exception:
        records = []
    latest = latest_evaluations(records)
    if not latest:
        return CommandResult(
            text="no playbook evaluations journaled yet — the "
                 "engine has not assessed anything on this "
                 "journal.")
    ctrl = _control(ctx)
    symbol = getattr(getattr(ctrl, "cfg", None), "symbol", None)

    def _text_of(rec: dict) -> str:
        return rec.get("assessment_text") or render_assessment(
            rec, symbol=symbol)

    def _index_line(name: str, rec: dict) -> str:
        a = rec.get("assessment") or {}
        verdict = str(a.get("verdict") or "?").upper()
        mark = " (SHADOW)" if is_shadow_record(rec) else ""
        line = (f"  {name}: {verdict}{mark} · "
                f"bar {rec.get('last_bar_ts') or '?'}")
        # Runtime Truth item 3: the index carries the freshness
        # facts too — a record whose bar is an old touch must read
        # as history on the index itself, not as the playbook's
        # current read (the qm_d1 2025-07-30 defect).
        prov = record_provenance(rec)
        if prov:
            bits = []
            if prov.get("touch_age_bars") is not None:
                bits.append(f"touch age {prov['touch_age_bars']} "
                            f"bar(s)")
            if prov.get("source"):
                bits.append(f"source {prov['source']}")
            if bits:
                line += " · " + " · ".join(bits)
        if verdict == "EXPIRED":
            line += " — STALE history, not a current setup"
        return line

    arg = (args[0].lower() if args else "")
    if arg and arg != "all":
        rec = latest.get(arg)
        if rec is None:
            known = ", ".join(sorted(latest))
            return CommandResult(
                text=f"no evaluation journaled for {arg!r} yet.\n"
                     f"assessed so far: {known}\n"
                     f"usage: /assess [playbook|all]")
        return CommandResult(
            text=_text_of(rec)
            + "\n" + fmt_next_cta("/assess all", "/status"))
    if arg == "all":
        blocks = [_text_of(latest[name]) for name in sorted(latest)]
        text = ("\n\n" + "—" * 24 + "\n\n").join(blocks)
        if len(text) <= 3500 or not ctx.reports_dir:
            return CommandResult(
                text=text + "\n" + fmt_next_cta("/assess", "/status"))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = Path(ctx.reports_dir) / f"mt5-assess-{stamp}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return CommandResult(
            document=path,
            caption=f"🧭 MT5 market assessments — latest per "
                    f"playbook ({len(blocks)} playbooks)")
    newest_name = max(latest, key=lambda n: latest[n].get("ts") or "")
    lines = [_text_of(latest[newest_name]), "",
             fmt_section("🧭", "Latest per playbook")]
    for name in sorted(latest):
        lines.append(_index_line(name, latest[name]))
    lines.append(fmt_next_cta("/assess <playbook>", "/assess all",
                              "/status"))
    return CommandResult(text="\n".join(lines))


@mt5_command("/position", "open MT5 positions with layers")
def _mt5_position(ctx: CommandContext, args: list[str]) -> CommandResult:
    lines = [fmt_section("📈", "MT5 OPEN POSITIONS:"), ""]
    lines += _position_lines(ctx)
    lines.append(fmt_next_cta("/status", "/pending", "/trades"))
    return CommandResult(text="\n".join(lines),
                         reply_markup=mt5_control_keyboard())


@mt5_command("/pending", "resting MT5 pending LIMIT / STOP orders")
def _mt5_pending(ctx: CommandContext, args: list[str]) -> CommandResult:
    lines = [fmt_section("⏳", "MT5 PENDING ORDERS:"), ""]
    lines += _pending_lines(ctx)
    lines.append(fmt_next_cta("/status", "/position", "/cancel_pending"))
    return CommandResult(text="\n".join(lines),
                         reply_markup=mt5_control_keyboard())


@mt5_command("/pause", "pause new signals (management continues) — confirm required")
def _mt5_pause(ctx: CommandContext, args: list[str]) -> CommandResult:
    return _confirm_result(ctx, "pause", "⏸ Pause engine")


@mt5_command("/resume", "resume new signals — confirm required")
def _mt5_resume(ctx: CommandContext, args: list[str]) -> CommandResult:
    return _confirm_result(ctx, "resume", "▶ Resume engine")


@mt5_command("/cancel_pending", "cancel ALL resting pending orders — confirm required")
def _mt5_cancel_pending(ctx: CommandContext, args: list[str]) -> CommandResult:
    return _confirm_result(ctx, "cancel_pending",
                           "❌ Cancel ALL pending orders")


@mt5_command("/trim", "close the top (most recent) added layer early; base rides — confirm required")
def _mt5_trim(ctx: CommandContext, args: list[str]) -> CommandResult:
    return _confirm_result(ctx, "trim", "✂️ Trim top layer")


@mt5_command("/config", "show the live MT5 config (safe keys only)")
def _mt5_config(ctx: CommandContext, args: list[str]) -> CommandResult:
    return CommandResult(text=_config_text(ctx),
                         reply_markup=mt5_control_keyboard())


@mt5_command("/set", "change a safe live config key: /set <key> <value> — confirm required")
def _mt5_set(ctx: CommandContext, args: list[str]) -> CommandResult:
    if len(args) < 2:
        return CommandResult(
            text="usage: /set <key> <value>\n"
                 "safe keys: max_layers (0–3), "
                 "manipulation_filter (on/off), pending_expiry_bars (1–24), "
                 "playbook.<name> (on/off).  Sizing is dynamic-only "
                 "(risk rule) — there is no lots key.  Pause is not "
                 "a /set key.")
    if len(args) > 2:
        return CommandResult(text="/set takes exactly one key and one value.")
    key, raw = args[0], args[1]
    change, error = parse_set(key, raw)
    if error is not None:
        return CommandResult(text=f"/set rejected: {error}")
    ctrl = _control(ctx)
    if ctrl is None:
        return CommandResult(
            text="/set unavailable (no live runner attached)")
    token = ctrl.request_confirm("set", change)
    return CommandResult(
        text=f"Apply {describe_change(change)}? "
             f"Tap Confirm to apply, Cancel to abort.",
        reply_markup=mt5_confirm_keyboard(token))


@mt5_command("/trades", "last N MT5 trades with R multiples (default 5)")
def _mt5_trades(ctx: CommandContext, args: list[str]) -> CommandResult:
    from ..execution.reporting import iter_paper_trades
    try:
        n = max(1, min(20, int(args[0]))) if args else 5
    except (ValueError, IndexError):
        return CommandResult(text="usage: /trades [n]  (1–20)")
    if ctx.journal is None:
        return CommandResult(text="no journal attached.")
    trades = iter_paper_trades(ctx.journal)[-n:]
    if not trades:
        return CommandResult(text="no closed MT5 trades yet.")
    lines = [fmt_section("📊", f"last {len(trades)} MT5 trades:"), ""]
    for t in trades:
        side = "LONG 🟢" if t["direction"] > 0 else "SHORT 🔴"
        ts = (t["closed_ts"] or "?")[:16].replace("T", " ")
        lines.append(f"{ts} {t['playbook']} {side} "
                     f"{fmt_r(t['r_net'])} ({t['reason']})")
    lines.append(fmt_next_cta("/status", "/report", "/recap"))
    return CommandResult(text="\n".join(lines))


@mt5_command("/db", "send the MT5 journal as a file")
def _mt5_db(ctx: CommandContext, args: list[str]) -> CommandResult:
    path = Path(ctx.journal_path) if ctx.journal_path else None
    if path is None or not path.exists():
        return CommandResult(text="MT5 journal file not found.")
    size_mb = path.stat().st_size / 1e6
    note = (f" — {size_mb:.1f}MB; Telegram bot documents cap at 50MB"
            if size_mb > 40 else "")
    return CommandResult(
        document=path,
        caption=f"📓 MT5 journal ({path.name}){note}")


@mt5_command("/report", "send the latest MT5 results as a file")
def _mt5_report(ctx: CommandContext, args: list[str]) -> CommandResult:
    from ..execution.reporting import build_report_file
    if ctx.journal is None:
        return CommandResult(text="no journal attached.")
    try:
        path = build_report_file(
            ctx.journal, out_dir=ctx.reports_dir,
            measured_oos=ctx.measured_oos or None)
    except Exception as exc:
        return CommandResult(
            text=f"report build failed: {type(exc).__name__}: {exc}")
    return CommandResult(
        document=path,
        caption="📊 MT5 engine report (fresh from the MT5 journal)")


@mt5_command("/recap", "daily / weekly recap — every closed trade, "
                        "wins AND losses, nothing hidden")
def _mt5_recap(ctx: CommandContext, args: list[str]) -> CommandResult:
    from ..execution.reporting import recap_for_period
    period = (args[0].lower() if args else "daily")
    if period not in ("daily", "weekly"):
        return CommandResult(text="usage: /recap [daily|weekly]")
    if ctx.journal is None:
        return CommandResult(text="no journal attached.")
    try:
        text = recap_for_period(ctx.journal, period)
    except Exception as exc:
        return CommandResult(
            text=f"recap build failed: {type(exc).__name__}: {exc}")
    # Short recaps go out as text; a long week goes out as a file
    # (Telegram truncates long messages — the file loses nothing).
    if len(text) <= 3500 or not ctx.reports_dir:
        header = fmt_section("📅", f"MT5 {period} recap")
        bar = ""
        try:
            from ..execution.reporting import recap_trades
            now = datetime.now(timezone.utc)
            window = (timedelta(days=7) if period == "weekly"
                      else timedelta(days=1))
            window_trades = recap_trades(ctx.journal, since=now - window,
                                         until=now)
            if window_trades:
                wins = sum(1 for t in window_trades
                           if t["r_net"] > 0)
                bar = (f"\n🏆 Win rate: "
                       f"{fmt_progress(wins / len(window_trades))} "
                       f"({wins}/{len(window_trades)} wins)")
        except Exception:
            bar = ""
        return CommandResult(
            text=header + "\n\n" + text + bar
            + fmt_next_cta("/status", "/trades", "/report"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = Path(ctx.reports_dir) / f"mt5-recap-{period}-{stamp}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return CommandResult(
        document=path,
        caption=f"📅 MT5 {period} recap (every closed trade, wins and losses)")


@mt5_command("/ticks", "export the last N days of MT5 tick history as CSV (default 1, max 7)")
def _mt5_ticks(ctx: CommandContext, args: list[str]) -> CommandResult:
    try:
        days = max(1, min(7, int(args[0]))) if args else 1
    except (ValueError, IndexError):
        return CommandResult(text="usage: /ticks [days]  (1–7)")
    if ctx.feed is None:
        return CommandResult(
            text="tick history export needs the live MT5 feed "
                 "(not available in this context).")
    out_dir = Path(ctx.reports_dir) if ctx.reports_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    stamp = end.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"mt5-ticks-{days}d-{stamp}.csv"
    try:
        path, rows = export_tick_history(ctx.feed, start, end, out_path)
    except Exception as exc:
        return CommandResult(
            text=f"tick export failed: {type(exc).__name__}: {exc}")
    return CommandResult(
        document=path,
        caption=f"📈 MT5 tick history, last {days} day(s): {rows} ticks "
                f"(exactly as seen — no resampling)")


#: /d1 defaults — mirror the feed's warm-start bar count for D1.
D1_DEFAULT_BARS = 2000
D1_MAX_BARS = 5000
D1_COLUMNS = ["time", "open", "high", "low", "close", "volume",
              "real_volume", "spread"]


@mt5_command("/d1", "export D1 bars as CSV from the live MT5 feed (default 2000, max 5000)")
def _mt5_d1(ctx: CommandContext, args: list[str]) -> CommandResult:
    try:
        bars_n = int(args[0]) if args else D1_DEFAULT_BARS
    except (ValueError, IndexError):
        return CommandResult(text="usage: /d1 [bars]  (1–5000, default 2000)")
    if bars_n <= 0:
        return CommandResult(text="usage: /d1 [bars]  (1–5000, default 2000)")
    bars_n = min(bars_n, D1_MAX_BARS)
    if ctx.feed is None or not hasattr(ctx.feed, "bars"):
        return CommandResult(
            text="D1 export needs the live MT5 feed "
                 "(not available in this context).")
    out_dir = Path(ctx.reports_dir) if ctx.reports_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"mt5-d1-{bars_n}bars-{stamp}.csv"
    try:
        # Real MT5 D1 bars via copy_rates_from_pos(TIMEFRAME_D1) —
        # never resampled, never faked. Closed bars only: the
        # still-forming daily bar is incomplete and would corrupt
        # validation — the engine trades closed bars.
        frame = ctx.feed.bars("D1", bars_n)
        frame = MT5Feed._closed_only(frame, "D1",
                                     datetime.now(timezone.utc))
        out = frame.reset_index(names="time")
        out = out[D1_COLUMNS]
        out.to_csv(out_path, index=False)
    except Exception as exc:
        return CommandResult(
            text=f"D1 export failed: {type(exc).__name__}: {exc}")
    if len(out):
        span = f"{str(out['time'].iloc[0])[:10]} → " \
               f"{str(out['time'].iloc[-1])[:10]}"
    else:
        span = "no closed bars"
    return CommandResult(
        document=out_path,
        caption=f"📅 MT5 D1 bars: {len(out)} closed bars, {span} "
                f"(real MT5 bars — no resampling)")


# ------------------------------------------------------------------ /harvest
MT5_HARVEST_SCOPES = ("all", "journal", "state", "ticks", "results")


def _mt5_candidates(ctx: CommandContext) -> tuple[Path, list[Path]]:
    """(data_dir, candidate files in stable order)."""
    if ctx.journal_path:
        data_dir = Path(ctx.journal_path).parent
    else:
        data_dir = Path.cwd() / "data"
    names: list[Path] = []
    if ctx.journal_path:
        names.append(Path(ctx.journal_path))
    names.append(data_dir / "mt5_risk_state.json")
    if data_dir.is_dir():
        names.extend(sorted(data_dir.glob("mt5_ticks_*.csv")))
    seen: set[str] = set()
    out: list[Path] = []
    for p in names:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            out.append(p)
    return data_dir, out


def _mt5_harvest_filter(paths: list[Path], scope: str) -> list[Path]:
    if scope == "journal":
        return [p for p in paths if "journal" in p.name]
    if scope == "state":
        return [p for p in paths if p.name == "mt5_risk_state.json"]
    if scope == "ticks":
        return [p for p in paths if p.name.startswith("mt5_ticks_")]
    if scope == "results":
        return []
    return list(paths)


def _mt5_results_md(ctx: CommandContext, scope: str, manifest: dict) -> str:
    """Human-readable RESULTS.md — the summary inside the ZIP."""
    from ..execution.reporting import (iter_paper_trades, open_positions,
                                       paper_book_stats)
    lines = ["# GoldenSwing MT5 harvest", "",
             f"harvested_utc: {manifest['harvested_utc']}",
             f"scope: {scope}",
             f"journal: {manifest.get('journal', '(none)')}",
             f"feed: {manifest.get('feed', '(no status)')}",
             f"last_bar: {manifest.get('last_bar_time') or '(unknown)'}",
             f"last_price: {manifest.get('last_price') or '(unknown)'}", ""]
    if ctx.journal is None:
        lines += ["## Results", "", "(no journal attached — no trade stats)", ""]
    else:
        stats = paper_book_stats(ctx.journal)
        book = stats.get("book", {}) or {}
        lines += ["## Results (from MT5 journal)", "",
                  f"trades: {book.get('trades', 0)}",
                  f"win_rate: {book.get('win_rate', 0):.1f}%",
                  f"total_R: {book.get('total_r', 0):+.2f}R",
                  f"avg_R: {book.get('avg_r', 0):+.2f}R",
                  f"profit_factor: {book.get('pf', 0)}",
                  f"total_pnl: {book.get('total_pnl', 0):+.2f}",
                  f"open_positions: {stats.get('open_count', 0)}", "",
                  "### Per playbook", ""]
        for pb, s in (stats.get("per_playbook") or {}).items():
            lines.append(f"- {pb}: {s.get('trades', 0)} trades, "
                         f"{s.get('win_rate', 0):.1f}% win, "
                         f"{s.get('total_r', 0):+.2f}R, "
                         f"PnL {s.get('total_pnl', 0):+.2f}")
        trades = iter_paper_trades(ctx.journal)
        today = datetime.now(timezone.utc).date().isoformat()
        todays = [t for t in trades if (t.get("closed_ts") or "")[:10] == today]
        lines += ["", f"### Today (UTC {today})", "",
                  f"closed: {len(todays)}, "
                  f"total_R: {sum(t['r_net'] for t in todays):+.2f}R, "
                  f"PnL: {sum(t['pnl_money'] for t in todays):+.2f}", "",
                  "### Last 10 closed trades", ""]
        for t in trades[-10:]:
            side = "LONG" if t["direction"] > 0 else "SHORT"
            lines.append(f"- {(t.get('closed_ts') or '?')[:16]} "
                         f"{t['playbook']} {side} {t['r_net']:+.2f}R "
                         f"({t['reason']})")
        opens = open_positions(ctx.journal)
        lines += ["", "### Open positions", ""]
        if not opens:
            lines.append("(none)")
        for o in opens:
            side = "LONG" if (o.get("direction") or 0) > 0 else "SHORT"
            lines.append(f"- {o.get('alert_id', '?')} {o.get('playbook', '?')} "
                         f"{side} entry {o.get('entry', '?')} "
                         f"stop {o.get('stop', '?')}")
    lines += ["", "## Tick CSVs in this bundle", ""]
    tick_files = [f for f in manifest["files"]
                  if f["arcname"].startswith("mt5_ticks_")]
    if not tick_files:
        lines.append("(none — no tick CSVs in this scope)")
    for f in tick_files:
        lines.append(f"- {f['arcname']} ({f['bytes']} bytes, "
                     f"sha256 {f['sha256'][:12]}…)")
    lines += ["", "## Files in this bundle", ""]
    for f in manifest["files"]:
        lines.append(f"- {f['arcname']} ({f['bytes']} bytes, "
                     f"sha256 {f['sha256'][:12]}…)")
    if manifest.get("skipped"):
        lines += ["", "## Skipped", ""]
        for s in manifest["skipped"]:
            lines.append(f"- {s['arcname']}: {s['reason']}")
    if manifest.get("fallback_slim"):
        lines += ["", "> Note: full bundle exceeded the 45MB ZIP cap, "
                       "so this is the slim fallback (journal + "
                       "RESULTS.md + manifest only)."]
    lines += ["", "## Excluded by design",
              "", "- chart PNGs (already ride the Telegram alerts)",
              "- *.pkl / *.joblib binaries, .venv, wheels",
              "- use /db for the single journal file, /ticks for a",
              "  fresh tick-history export.", ""]
    return "\n".join(lines)


def build_mt5_harvest_bundle(ctx: CommandContext, scope: str = "all"
                             ) -> tuple[Path, dict]:
    """Build the MT5 harvest ZIP. Returns (zip_path, manifest)."""
    if scope not in MT5_HARVEST_SCOPES:
        raise ValueError(f"unknown scope {scope!r}; use one of {MT5_HARVEST_SCOPES}")
    data_dir, candidates = _mt5_candidates(ctx)
    chosen = _mt5_harvest_filter(candidates, scope)
    out_dir = Path(ctx.reports_dir) if ctx.reports_dir else data_dir / "reports"
    harvested_utc = datetime.now(timezone.utc).isoformat()
    manifest_base: dict = {
        "harvested_utc": harvested_utc,
        "scope": scope,
        "engine": "mt5",
        "journal": str(ctx.journal_path) if ctx.journal_path else None,
        "feed": str(ctx.feed_status) if ctx.feed_status else None,
        "last_bar_time": ctx.last_bar_time,
        "last_price": ctx.last_price,
    }
    journal_only = [p for p in chosen if "journal" in p.name]
    return build_zip_bundle(
        out_dir=out_dir, zip_prefix="goldenswing-mt5-harvest",
        files=chosen, data_dir=data_dir, manifest_base=manifest_base,
        results_md_fn=lambda m: _mt5_results_md(ctx, scope, m),
        slim_files=journal_only)


@mt5_command("/harvest", "harvest MT5 journal + state + tick CSVs + results as one ZIP")
def _mt5_harvest(ctx: CommandContext, args: list[str]) -> CommandResult:
    scope = (args[0].lower() if args else "all")
    if scope not in MT5_HARVEST_SCOPES:
        return CommandResult(
            text="usage: /harvest [all|journal|state|ticks|results]\n"
                 "all = journal + state + tick CSVs + results (default). "
                 "Chart PNGs excluded by design.")
    if scope != "state" and ctx.journal is None:
        return CommandResult(text="no journal attached.")
    try:
        zip_path, manifest = build_mt5_harvest_bundle(ctx, scope)
    except Exception as exc:
        return CommandResult(
            text=f"harvest failed: {type(exc).__name__}: {exc}")
    files = manifest["files"]
    n_trades = "?"
    total_r = ""
    if ctx.journal is not None:
        try:
            from ..execution.reporting import paper_book_stats
            book = paper_book_stats(ctx.journal).get("book", {}) or {}
            n_trades = book.get("trades", 0)
            total_r = f", {book.get('total_r', 0):+.2f}R"
        except Exception:
            pass
    caption = (f"📦 MT5 harvest {scope} — {n_trades} trades{total_r}, "
               f"{len(files)} files, {manifest['zip_bytes'] // 1024}KB — "
               f"RESULTS.md inside has the full breakdown")
    return CommandResult(document=zip_path, caption=caption)


# ------------------------------------------------- engine / backtest commands
# Same contract as the OANDA loop's /backtest family, bound to the MT5
# registry so the MT5 engine (the getUpdates owner at cutover) answers
# them from its own journal.  Delegates to the shared implementation
# in goldenswing.alerts.commands — one implementation, two registries.

@mt5_command("/backtest", "run a real whole-engine backtest (async)")
def _mt5_backtest(ctx: CommandContext, args: list[str]) -> CommandResult:
    from ..alerts.commands import start_backtest_from_args
    return start_backtest_from_args(ctx, args, drop_back=False)


@mt5_command("/backtest_send", "run a real backtest, send file to drop back")
def _mt5_backtest_send(ctx: CommandContext, args: list[str]) -> CommandResult:
    from ..alerts.commands import start_backtest_from_args
    return start_backtest_from_args(ctx, args, drop_back=True)


@mt5_command("/backtest_export", "export paper-journal trades as a report file")
def _mt5_backtest_export(ctx: CommandContext, args: list[str]) -> CommandResult:
    from ..alerts.commands import build_backtest_export_result
    return build_backtest_export_result(ctx, drop_back=False)


@mt5_command("/journal_export", "export paper-journal trades as a report file")
def _mt5_journal_export(ctx: CommandContext, args: list[str]) -> CommandResult:
    from ..alerts.commands import build_backtest_export_result
    return build_backtest_export_result(ctx, drop_back=False)


@mt5_command("/journal", "journal head / record count / last record")
def _mt5_journal(ctx: CommandContext, args: list[str]) -> CommandResult:
    if ctx.journal is None:
        return CommandResult(text="no journal attached.")
    try:
        records = ctx.journal.records()
    except Exception:
        records = []
    lines = [fmt_section("📓", "MT5 Journal"),
             f"journal: {fmt_code(ctx.journal_path or '(unknown path)')}",
             f"records: {len(records)}"]
    if records:
        last = records[-1]
        lines.append(
            f"last: seq={fmt_code(last.get('seq', '?'))} "
            f"type={fmt_code(last.get('type', '?'))} "
            f"ts={fmt_code(last.get('ts', '?'))}")
    lines.append(fmt_next_cta("/snapshot", "/health", "/status"))
    return CommandResult(text="\n".join(lines))


@mt5_command("/snapshot", "deterministic-engine snapshot manifest (N / N_prev)")
def _mt5_snapshot(ctx: CommandContext, args: list[str]) -> CommandResult:
    from ..alerts.commands import _snapshot as _shared_snapshot
    return _shared_snapshot(ctx, args)


@mt5_command("/reconcile", "reconciliation status: unresolved / open vs closed")
def _mt5_reconcile(ctx: CommandContext, args: list[str]) -> CommandResult:
    from ..alerts.commands import _reconcile as _shared_reconcile
    return _shared_reconcile(ctx, args)


@mt5_command("/health", "feed + journal + open positions health")
def _mt5_health(ctx: CommandContext, args: list[str]) -> CommandResult:
    fs = ctx.feed_status or {}
    _truth = _feed_truth(ctx)
    lines = [fmt_section("❤️", "MT5 HEALTH"), "",
             fmt_section("📡", "Feed"),
             fmt_feed_line(fs, ctx.last_bar_time, ctx.last_price,
                           market_open=_truth.get("market_open"),
                           tick_state=_truth.get("tick_state"),
                           tick_age_s=_truth.get("tick_age_s"))]
    if ctx.journal is not None:
        try:
            lines += ["", fmt_section("📓", "Journal"),
                      f"journal records: {len(ctx.journal.records())}"]
        except Exception:
            pass
    lines += ["", fmt_section("📈", "Open Positions"),
              f"open MT5 positions: {len(ctx.live_positions)}",
              "", fmt_section("⏳", "Pending Orders"),
              f"pending orders: {len(ctx.pending_orders)}"]
    ctrl = getattr(ctx, "mt5_control", None)
    if ctrl is not None and hasattr(ctrl, "deterministic_status"):
        try:
            det = ctrl.deterministic_status()
            lines += ["", fmt_section("🧱", "Deterministic Engine")]
            if det.get("enabled"):
                lines.append(
                    f"deterministic: 🟢 {det.get('phase')} "
                    f"head={fmt_code(det.get('head'))} "
                    f"commit={fmt_code(det.get('commit_index'))} "
                    f"unresolved={det.get('unresolved')}")
            else:
                lines.append(
                    f"deterministic: ⚠️ disabled ({det.get('error')})")
        except Exception:
            pass
    lines.append(fmt_next_cta("/status", "/reconcile", "/journal"))
    return CommandResult(text="\n".join(lines))
