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
from ..alerts.commands import (CommandContext, CommandResult,
                              CommandRouter, build_zip_bundle)

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
    """CommandRouter bound to the MT5 registry (owner-gated)."""
    return CommandRouter(owner_chat_id=owner_chat_id, registry=MT5_COMMANDS)


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# ------------------------------------------------------------------ commands

@mt5_command("/help", "list the MT5 engine commands")
def _mt5_help(ctx: CommandContext, args: list[str]) -> CommandResult:
    lines = ["MT5 engine commands (this engine's own journal/state):"]
    for name in sorted(MT5_COMMANDS):
        desc, _ = MT5_COMMANDS[name]
        lines.append(f"{name} — {desc}")
    return CommandResult(text="\n".join(lines))


@mt5_command("/status", "MT5 engine health: terminal, open positions, today's P&L")
def _mt5_status(ctx: CommandContext, args: list[str]) -> CommandResult:
    from ..execution.reporting import (iter_paper_trades, open_positions)
    fs = ctx.feed_status or {}
    feed_line = ("feed: OK (MT5 terminal connected)" if fs.get("ok")
                 else f"feed: STALE — {fs.get('reason', 'no status yet')}")
    if ctx.last_bar_time:
        feed_line += f" (last bar {ctx.last_bar_time})"
    if ctx.last_price:
        feed_line += f" last price {ctx.last_price:,.1f}"

    lines = ["MT5 ENGINE STATUS", feed_line, "",
             "open MT5 positions:"]
    # Live positions from the terminal (MT5 = source of truth) win
    # over journal opens: they include restart-adopted positions that
    # have no signal record.  Journal opens are the fallback.
    live = list(getattr(ctx, "live_positions", None) or [])
    journal_opens = open_positions(ctx.journal) if ctx.journal else []
    if live:
        for p in live:
            side = "LONG" if (p.get("direction") or 0) > 0 else "SHORT"
            entry = float(p.get("entry_price") or p.get("entry") or 0)
            stop = float(p.get("sl") or p.get("stop") or 0)
            risk = abs(entry - stop)
            r_now = "?"
            if ctx.last_price and risk > 0:
                r_now = (f"{(p.get('direction') or 0) * (ctx.last_price - entry) / risk:+.2f}R")
            lines.append(f"#{p.get('ticket', '?')} {p.get('playbook', '?')} "
                         f"{side} @ {entry:,.1f} sl {stop:,.1f} now ≈{r_now}")
    else:
        if not journal_opens:
            lines.append("(none)")
        for o in journal_opens:
            side = "LONG" if (o.get("direction") or 0) > 0 else "SHORT"
            entry = float(o.get("entry") or 0)
            stop = float(o.get("stop") or 0)
            risk = abs(entry - stop)
            r_now = "?"
            if ctx.last_price and risk > 0:
                r_now = (f"{(o.get('direction') or 0) * (ctx.last_price - entry) / risk:+.2f}R")
            lines.append(f"#{o.get('ticket', '?')} {o.get('playbook', '?')} {side} "
                         f"@ {entry:,.1f} sl {stop:,.1f} now ≈{r_now}")
    opens = journal_opens

    trades = iter_paper_trades(ctx.journal) if ctx.journal else []
    today = _today_utc()
    todays = [t for t in trades if (t["closed_ts"] or "")[:10] == today]
    lines += ["", f"today (UTC): {len(todays)} closed, "
                  f"{sum(t['r_net'] for t in todays):+.2f}R "
                  f"({sum(t['pnl_money'] for t in todays):+.2f})"]
    if opens:
        last_open = max(opens, key=lambda r: r.get("ts") or "")
        lines.append(f"last signal: {last_open.get('playbook', '?')} "
                     f"{last_open.get('ts', '?')}")
    else:
        lines.append("last signal: none journaled yet")
    return CommandResult(text="\n".join(lines))


@mt5_command("/trim", "close the top (most recent) added layer early; base rides")
def _mt5_trim(ctx: CommandContext, args: list[str]) -> CommandResult:
    cb = getattr(ctx, "trim_callback", None)
    if cb is None:
        return CommandResult(text="trim unavailable (no live runner)")
    try:
        return CommandResult(text=str(cb()))
    except Exception as exc:
        return CommandResult(text=f"trim FAILED: {exc}")


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
    lines = [f"last {len(trades)} MT5 trades:"]
    for t in trades:
        side = "LONG" if t["direction"] > 0 else "SHORT"
        ts = (t["closed_ts"] or "?")[:16].replace("T", " ")
        lines.append(f"{ts} {t['playbook']} {side} "
                     f"{t['r_net']:+.2f}R ({t['reason']})")
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
        caption=f"MT5 journal ({path.name}){note}")


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
        caption="MT5 engine report (fresh from the MT5 journal)")


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
        caption=f"MT5 tick history, last {days} day(s): {rows} ticks "
                f"(exactly as seen — no resampling)")


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
    caption = (f"MT5 harvest {scope} — {n_trades} trades{total_r}, "
               f"{len(files)} files, {manifest['zip_bytes'] // 1024}KB — "
               f"RESULTS.md inside has the full breakdown")
    return CommandResult(document=zip_path, caption=caption)
