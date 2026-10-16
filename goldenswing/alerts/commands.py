"""Telegram command surface — on-demand control for the owner.

One consumer rule: commands arrive as ``message`` updates through the
SAME ``getUpdates`` poll the button taps use (see
``LiveRunner._poll_telegram``).  Never create a second poller — two
consumers race on the update offset and one of them starves.

Owner-only: every command is answered only when the update's
``chat.id`` matches the configured owner chat id
(``GS_TELEGRAM_CHAT_ID``, the same id alerts go to).  Anything else —
other chats, groups, channels, non-text updates — is ignored SILENTLY:
no reply, no error, no log line an attacker can probe.

Adding a new command (the owner asked for "other features I might not
have thought of" — this is the whole extension contract)::

    from goldenswing.alerts.commands import command, CommandContext, CommandResult

    @command("/mysignal", "one-line description for /help")
    def my_signal(ctx: CommandContext, args: list[str]) -> CommandResult:
        # ctx.journal   — TradeJournal (append-only)
        # ctx.paper     — PaperExecutor | None (open positions, balance)
        # ctx.feed_status — dict | None (last PollingFeed.poll_status())
        # ctx.last_price / ctx.last_bar_time — latest closed bar, if known
        # ctx.measured_oos  — versioned measured-OOS rows
        # ctx.reports_dir   — where generated documents go
        #
        # Return CommandResult(text=...) for a text reply, or
        # CommandResult(document=path, caption=...) to send a file.
        return CommandResult(text="hello from the loop")

That is the entire contract: one decorated function, no wiring
elsewhere.  ``/help`` picks it up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Callable

#: name -> (description, handler).  Populated by @command.
COMMANDS: dict[str, tuple[str, Callable]] = {}


def command(name: str, description: str):
    """Register a ``/command`` handler.  See module docstring."""
    if not name.startswith("/"):
        raise ValueError(f"command name must start with '/': {name!r}")

    def deco(fn: Callable) -> Callable:
        COMMANDS[name] = (description, fn)
        return fn

    return deco


@dataclass
class CommandResult:
    """What the runner sends back for one command."""
    text: str | None = None
    document: str | Path | None = None   # sent via sendDocument
    caption: str | None = None
    reply_markup: dict | None = None     # inline keyboard, if any


@dataclass
class CommandContext:
    journal: object = None               # TradeJournal
    journal_path: Path | None = None
    paper: object = None                 # PaperExecutor | None
    feed_status: dict | None = None      # last PollingFeed.poll_status()
    last_price: float | None = None      # latest closed bar's close
    last_bar_time: str | None = None
    measured_oos: list[dict] = field(default_factory=list)
    reports_dir: Path | None = None
    feed: object = None                  # live feed (MT5 /ticks export)
    live_positions: list = field(default_factory=list)  # MT5 truth: open positions
    trim_callback: object = None         # MT5 runner.trim_top_layer (bound)
    mt5_control: object = None           # MT5 runner (pause/config/confirm)
    pending_orders: list = field(default_factory=list)  # MT5 pendings
    reconciler: object = None            # engine Reconciler | None (/reconcile)
    deterministic: object = None        # DeterministicRuntime | None (/health)


def _is_owner(update: dict, owner_chat_id: str | int | None) -> bool:
    if owner_chat_id in (None, ""):
        return False                     # fail closed: unconfigured = nobody
    msg = update.get("message") or {}
    chat = msg.get("chat") or {}
    return str(chat.get("id")) == str(owner_chat_id)


def is_owner_callback(update: dict, owner_chat_id: str | int | None) -> bool:
    """Owner check for a ``callback_query`` update: the TAPPER
    (``callback_query.from.id``) must be the owner, and when the
    source message carries a chat it must be the owner chat too.
    Fail closed, same as :func:`_is_owner`."""
    if owner_chat_id in (None, ""):
        return False
    query = update.get("callback_query") or {}
    from_id = (query.get("from") or {}).get("id")
    if str(from_id) != str(owner_chat_id):
        return False
    msg = query.get("message") or {}
    chat = msg.get("chat") or {}
    if chat and str(chat.get("id")) != str(owner_chat_id):
        return False
    return True


class CommandRouter:
    """Parse + gate + dispatch one raw getUpdates update.

    ``registry`` defaults to the global :data:`COMMANDS` (the OANDA
    loop).  The MT5 engine passes its own registry so the two loops
    answer different commands from their own journal/state without
    clobbering each other's handlers in-process.
    """

    def __init__(self, owner_chat_id: str | int | None = None,
                 registry: dict | None = None,
                 aliases: dict | None = None):
        self.owner_chat_id = owner_chat_id
        self.registry = COMMANDS if registry is None else registry
        # Plain-text aliases for reply-keyboard menu buttons
        # (``"Status"`` -> ``"/status"``): button taps arrive as
        # ordinary message text, not slash commands.  Keys are
        # matched case-insensitively; the owner check below still
        # applies exactly as for slash commands.
        self.aliases = {str(k).lower(): v for k, v in (aliases or {}).items()}

    def dispatch(self, update: dict, ctx: CommandContext
                 ) -> tuple[CommandResult | None, str | None]:
        """Return (result, reply_chat_id).  ``(None, None)`` = ignore
        silently (not a command, not the owner, nothing to say)."""
        msg = update.get("message") or {}
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            alias = self.aliases.get(text.lower())
            if alias is None:
                return None, None
            text = alias
        if not text.startswith("/"):
            return None, None
        if not _is_owner(update, self.owner_chat_id):
            return None, None            # silent: never confirm we exist
        chat_id = str((msg.get("chat") or {}).get("id"))
        parts = text.split()
        name, args = parts[0].split("@")[0].lower(), parts[1:]
        entry = self.registry.get(name)
        if entry is None:
            return CommandResult(
                text=f"unknown command {name} — /help lists the commands."), \
                chat_id
        _, handler = entry
        try:
            return handler(ctx, args), chat_id
        except Exception as exc:         # a command never breaks the loop
            return CommandResult(
                text=f"{name} failed: {type(exc).__name__}: {exc}"), chat_id


# ------------------------------------------------------------------ helpers

def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _open_position_lines(ctx: CommandContext) -> list[str]:
    """Open paper positions with approximate current R."""
    if ctx.paper is None:
        return ["(paper executor not attached)"]
    positions = ctx.paper.positions()
    if not positions:
        return ["(none)"]
    pb_by_alert: dict[str, str] = {}
    if ctx.journal is not None:
        for r in ctx.journal.records():
            if r.get("type") == "open" and r.get("alert_id"):
                pb_by_alert[r["alert_id"]] = r.get("playbook", "?")
    lines = []
    for p in positions:
        side = "LONG" if p.direction > 0 else "SHORT"
        pb = pb_by_alert.get(p.alert_id, "")
        risk_dist = abs((p.entry_price or 0) - (p.sl or 0))
        r_now = "?"
        if ctx.last_price and risk_dist > 0:
            r_val = p.direction * (ctx.last_price - p.entry_price) \
                / risk_dist
            r_now = f"{r_val:+.2f}R"
        lines.append(
            f"#{p.ticket} {pb} {side} {p.symbol} {p.volume:.2f} lots "
            f"@ {p.entry_price:,.1f} sl {p.sl:,.1f} now ≈{r_now}")
    return lines


# ------------------------------------------------------------------ commands

@command("/help", "list the commands")
def _help(ctx: CommandContext, args: list[str]) -> CommandResult:
    lines = ["GoldenSwing commands:"]
    for name in sorted(COMMANDS):
        desc, _ = COMMANDS[name]
        lines.append(f"{name} — {desc}")
    return CommandResult(text="\n".join(lines))


@command("/status", "loop health: feed, open positions, today's P&L, last signal")
def _status(ctx: CommandContext, args: list[str]) -> CommandResult:
    from goldenswing.execution.reporting import iter_paper_trades
    fs = ctx.feed_status or {}
    feed_line = ("feed: OK" if fs.get("ok")
                 else f"feed: STALE — {fs.get('reason', 'no status yet')}")
    if ctx.last_bar_time:
        feed_line += f" (last bar {ctx.last_bar_time})"

    lines = ["PAPER LOOP STATUS", feed_line, "",
             "open paper positions:"]
    lines += _open_position_lines(ctx)

    trades = iter_paper_trades(ctx.journal) if ctx.journal else []
    today = _today_utc()
    todays = [t for t in trades
              if (t["closed_ts"] or "")[:10] == today]
    day_r = sum(t["r_net"] for t in todays)
    day_pnl = sum(t["pnl_money"] for t in todays)
    lines += ["", f"today (UTC): {len(todays)} closed, "
                  f"{day_r:+.2f}R ({day_pnl:+.2f})"]
    last_open = None
    if ctx.journal:
        opens = [r for r in ctx.journal.records()
                 if r.get("type") == "open"]
        if opens:
            last_open = max(opens, key=lambda r: r.get("ts") or "")
    if last_open:
        lines.append(f"last signal: {last_open.get('playbook', '?')} "
                     f"{last_open.get('ts', '?')}")
    else:
        lines.append("last signal: none journaled yet")
    return CommandResult(text="\n".join(lines))


@command("/trades", "last N paper trades with R multiples (default 5)")
def _trades(ctx: CommandContext, args: list[str]) -> CommandResult:
    from goldenswing.execution.reporting import iter_paper_trades
    try:
        n = max(1, min(20, int(args[0]))) if args else 5
    except (ValueError, IndexError):
        return CommandResult(text="usage: /trades [n]  (1–20)")
    if ctx.journal is None:
        return CommandResult(text="no journal attached.")
    trades = iter_paper_trades(ctx.journal)[-n:]
    if not trades:
        return CommandResult(text="no closed paper trades yet.")
    lines = [f"last {len(trades)} paper trades:"]
    for t in trades:
        side = "LONG" if t["direction"] > 0 else "SHORT"
        ts = (t["closed_ts"] or "?")[:16].replace("T", " ")
        lines.append(f"{ts} {t['playbook']} {side} "
                     f"{t['r_net']:+.2f}R ({t['reason']})")
    return CommandResult(text="\n".join(lines))


@command("/db", "send the paper journal database as a file")
def _db(ctx: CommandContext, args: list[str]) -> CommandResult:
    path = Path(ctx.journal_path) if ctx.journal_path else None
    if path is None or not path.exists():
        return CommandResult(text="journal file not found.")
    size_mb = path.stat().st_size / 1e6
    note = (f" — {size_mb:.1f}MB; Telegram bot documents cap at 50MB"
            if size_mb > 40 else "")
    return CommandResult(
        document=path,
        caption=f"paper journal ({path.name}){note}")


@command("/report", "send the latest backtest/paper results as a file")
def _report(ctx: CommandContext, args: list[str]) -> CommandResult:
    from goldenswing.execution.reporting import build_report_file
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
        caption="GoldenSwing paper-book report (fresh from the journal)")


# ------------------------------------------------------------------ /harvest
# One command that pulls EVERYTHING the signal bot owns — database +
# journal + state + results — into a single ZIP the owner can download
# from Telegram.  /db sends only the journal, /report sends only the
# report; /harvest is the full bundle for backup / audit / moving the
# bot to another machine (e.g. AWS).
#
# Scopes:
#   /harvest           — all (journal + state + results)
#   /harvest journal   — journal files only (*journal*.jsonl)
#   /harvest state     — paper_state + risk_state only (no journal)
#   /harvest results   — RESULTS.md + manifest + measured-OOS / modes /
#                        validation (no raw journal rows)
#
# Deliberately EXCLUDED (never in the bundle): market-data CSVs
# (hundreds of MB), *.pkl / *.joblib binaries, chart PNGs from
# alert_outbox, .venv, wheels.  Those are reproducible / bulky, not
# evidence.  A per-file 20MB cap + a 45MB ZIP cap (under Telegram's
# 50MB sendDocument limit) keep the send deliverable; oversize
# falls back to a slim bundle and says so in the manifest.

HARVEST_MAX_FILE_BYTES = 20 * 1024 * 1024
HARVEST_MAX_ZIP_BYTES = 45 * 1024 * 1024
HARVEST_SCOPES = ("all", "journal", "state", "results")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _harvest_candidates(ctx: CommandContext) -> tuple[Path, list[Path]]:
    """(data_dir, candidate files in stable order)."""
    if ctx.journal_path:
        data_dir = Path(ctx.journal_path).parent
    else:
        data_dir = Path.cwd() / "data"
    names: list[Path] = []
    if ctx.journal_path:
        names.append(Path(ctx.journal_path))
    for rel in (
        "trade_journal.jsonl",
        "confluence_journal.jsonl",
        "risk_state.json",
        "measured_oos.json",
        "modes_results.json",
        "validation_results.json",
    ):
        names.append(data_dir / rel)
    paper_dir = data_dir / "paper_state"
    names.append(paper_dir / "paper.json")
    names.append(paper_dir / "meta.json")
    if paper_dir.is_dir():
        names.extend(sorted(paper_dir.glob("playbooks/*.json")))
    # De-dupe, keep order, keep only real files.
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


def _harvest_filter(paths: list[Path], scope: str) -> list[Path]:
    if scope == "journal":
        return [p for p in paths if "journal" in p.name]
    if scope == "state":
        return [p for p in paths
                if p.name in ("paper.json", "meta.json", "risk_state.json")
                or "playbooks" in p.parts]
    if scope == "results":
        return [p for p in paths
                if p.name in ("measured_oos.json", "modes_results.json",
                              "validation_results.json")]
    return list(paths)


def _harvest_results_md(ctx: CommandContext, scope: str,
                        manifest: dict) -> str:
    """Human-readable RESULTS.md — the summary inside the ZIP."""
    from goldenswing.execution.reporting import (
        iter_paper_trades, open_positions, paper_book_stats)
    lines = ["# GoldenSwing harvest", "",
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
        lines += ["## Results (from journal)", "",
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
    # Paper state (balance etc.) when the file exists — best effort.
    paper_state = manifest.get("paper_state")
    if paper_state:
        lines += ["", "## Paper state (paper_state/paper.json)", "",
                  f"balance: {paper_state.get('balance', '?')}",
                  f"pending_orders: {len(paper_state.get('pending_orders') or [])}",
                  f"tracked: {len(paper_state.get('tracked') or {})}",
                  f"next_ticket: {paper_state.get('next_ticket', '?')}"]
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
              "", "- market-data CSVs (reproducible, hundreds of MB)",
              "- *.pkl / *.joblib binaries, chart PNGs, .venv, wheels",
              "- use /db for the single journal file, /report for the",
              "  report-only document.", ""]
    return "\n".join(lines)


def build_zip_bundle(*, out_dir: Path, zip_prefix: str,
                     files: list[Path], data_dir: Path,
                     manifest_base: dict, results_md_fn,
                     slim_files: list[Path] | None = None
                     ) -> tuple[Path, dict]:
    """Generic bundle-ZIP writer shared by the OANDA + MT5 /harvest.

    ``results_md_fn(manifest)`` returns the RESULTS.md text and is
    called with ``files``/``skipped`` already filled in.  Same caps as
    the OANDA /harvest: 20MB/file, 45MB ZIP; oversize rebuilds the
    slim fallback (``slim_files`` + RESULTS.md + manifest) once.
    Returns (zip_path, manifest).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    manifest: dict = {**manifest_base, "files": [], "skipped": [],
                      "fallback_slim": False}

    def _write_zip(zip_path: Path, file_list: list[Path]) -> dict:
        entries: list[dict] = []
        skipped: list[dict] = []
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in file_list:
                try:
                    arc = str(p.relative_to(data_dir))
                except ValueError:
                    arc = p.name
                size = p.stat().st_size
                if size > HARVEST_MAX_FILE_BYTES:
                    skipped.append({"arcname": arc,
                                    "reason": f"{size} bytes > 20MB cap"})
                    continue
                zf.write(p, arc)
                entries.append({"arcname": arc, "bytes": size,
                                "sha256": _sha256_file(p)})
            results_md = results_md_fn(
                {**manifest, "files": entries, "skipped": skipped})
            zf.writestr("RESULTS.md", results_md)
            full_manifest = {**manifest, "files": entries,
                             "skipped": skipped,
                             "zip_bytes": 0}
            zf.writestr("manifest.json",
                         json.dumps(full_manifest, indent=2, default=str))
        return {"files": entries, "skipped": skipped}

    zip_path = out_dir / f"{zip_prefix}-{stamp}.zip"
    outcome = _write_zip(zip_path, files)
    manifest["files"] = outcome["files"]
    manifest["skipped"] = outcome["skipped"]
    if zip_path.stat().st_size > HARVEST_MAX_ZIP_BYTES:
        if slim_files is None:
            zip_path.unlink(missing_ok=True)
            raise ValueError("bundle exceeds the 45MB ZIP cap "
                             "and no slim fallback was provided")
        zip_path.unlink(missing_ok=True)
        slim_path = out_dir / f"{zip_prefix}-{stamp}-slim.zip"
        manifest["fallback_slim"] = True
        outcome = _write_zip(slim_path, slim_files)
        manifest["files"] = outcome["files"]
        manifest["skipped"] = outcome["skipped"] + [{
            "arcname": "(full bundle)",
            "reason": f"ZIP > {HARVEST_MAX_ZIP_BYTES} bytes — slim fallback"}]
        zip_path = slim_path
        if zip_path.stat().st_size > HARVEST_MAX_ZIP_BYTES:
            zip_path.unlink(missing_ok=True)
            raise ValueError("even the slim harvest exceeds the 45MB ZIP cap")
    manifest["zip_bytes"] = zip_path.stat().st_size
    manifest["zip_name"] = zip_path.name
    return zip_path, manifest


def build_harvest_bundle(ctx: CommandContext, scope: str = "all") -> tuple[Path, dict]:
    """Build the harvest ZIP. Returns (zip_path, manifest)."""
    if scope not in HARVEST_SCOPES:
        raise ValueError(f"unknown scope {scope!r}; use one of {HARVEST_SCOPES}")
    data_dir, candidates = _harvest_candidates(ctx)
    chosen = _harvest_filter(candidates, scope)
    out_dir = Path(ctx.reports_dir) if ctx.reports_dir else data_dir / "reports"
    harvested_utc = datetime.now(timezone.utc).isoformat()

    # Paper state snapshot for RESULTS.md (best effort, never fatal).
    paper_state = None
    paper_path = data_dir / "paper_state" / "paper.json"
    if paper_path.is_file():
        try:
            paper_state = json.loads(paper_path.read_text())
        except (json.JSONDecodeError, OSError):
            paper_state = None

    manifest_base: dict = {
        "harvested_utc": harvested_utc,
        "scope": scope,
        "journal": str(ctx.journal_path) if ctx.journal_path else None,
        "feed": str(ctx.feed_status) if ctx.feed_status else None,
        "last_bar_time": ctx.last_bar_time,
        "last_price": ctx.last_price,
        "paper_state": paper_state,
    }
    journal_only = [p for p in chosen if "journal" in p.name]
    return build_zip_bundle(
        out_dir=out_dir, zip_prefix="goldenswing-harvest",
        files=chosen, data_dir=data_dir, manifest_base=manifest_base,
        results_md_fn=lambda m: _harvest_results_md(ctx, scope, m),
        slim_files=journal_only)


# ------------------------------------------------------------------ engine /
# backtest export commands (deterministic engine + Telegram wiring).
#
# /backtest      — build a backtest report from the attached journal
#                  and send it as a FILE + concise summary caption.
#                  The file is the contract: the owner drops it back
#                  to Muse for review.  The caption carries trades /
#                  win% / R / PF / MaxDD / period.  Never silent: no
#                  journal means an explicit text reply, not nothing.
# /backtest_send — same file + caption, with the explicit instruction
#                  to drop the file back to Muse.  (Alias of /backtest
#                  with the drop-back instruction in the caption.)
# /journal       — journal head / record count / last record.
# /snapshot      — deterministic-engine snapshot manifest (N / N_prev)
#                  when an engine dir exists, else an honest "none yet".
# /reconcile     — reconciliation status from the journal (open vs
#                  closed); the live Reconciler is reported when the
#                  runner attaches one on ctx (ctx.reconciler).
# /health        — feed + journal + open positions in one reply.

def _backtest_rows_from_journal(ctx: CommandContext) -> list[dict]:
    """Trade rows for the backtest exporter, from the paper journal."""
    from goldenswing.execution.reporting import iter_paper_trades
    if ctx.journal is None:
        return []
    rows: list[dict] = []
    for t in iter_paper_trades(ctx.journal):
        rows.append({
            "playbook": t.get("playbook", "?"),
            "direction": t.get("direction", 0),
            "r_net": t.get("r_net", 0.0),
            "pnl": t.get("pnl_money", 0.0),
            "reason": t.get("reason", ""),
        })
    return rows


def _d1_coverage(ctx: CommandContext) -> tuple[int | None, bool]:
    """(d1_bars, d1_stale) from feed_status when available.

    The paper loop reports hierarchy-blocked D1 staleness via
    feed_status; when it says the D1 feed is stale the backtest
    summary must say so too — never claim clean on stale D1 data.
    """
    fs = ctx.feed_status or {}
    if not fs:
        return None, False
    stale = not bool(fs.get("ok", True))
    return None, stale


def build_backtest_result(ctx: CommandContext, *,
                          drop_back: bool = False) -> CommandResult:
    """Shared /backtest + /backtest_send implementation (both loops)."""
    from goldenswing.engine.backtest_export import (
        format_summary, write_report)
    if ctx.journal is None:
        return CommandResult(
            text="no journal attached — nothing to export. "
                 "Run the paper loop or a backtest first.")
    rows = _backtest_rows_from_journal(ctx)
    if not rows:
        return CommandResult(
            text="no closed trades in the journal yet — "
                 "nothing to export. Run the paper loop first.")
    d1_bars, d1_stale = _d1_coverage(ctx)
    out_dir = Path(ctx.reports_dir) if ctx.reports_dir else \
        Path.cwd() / "data" / "reports"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path, _summary = write_report(
        rows, out_dir / f"backtest-{stamp}.md",
        period="paper journal", d1_bars=d1_bars, d1_stale=d1_stale)
    summary = format_summary(_summary)
    caption = summary
    if drop_back:
        caption += "\nDrop this file back to Muse for review."
    # Caption limit is 1024 chars in the sender; keep it concise.
    if len(caption) > 1000:
        caption = caption[:997] + "…"
    return CommandResult(document=path, caption=caption)


@command("/backtest", "build + send backtest report file + summary")
def _backtest(ctx: CommandContext, args: list[str]) -> CommandResult:
    return build_backtest_result(ctx, drop_back=False)


@command("/backtest_send", "send backtest results file to drop back to Muse")
def _backtest_send(ctx: CommandContext, args: list[str]) -> CommandResult:
    return build_backtest_result(ctx, drop_back=True)


@command("/journal", "journal head / record count / last record")
def _journal(ctx: CommandContext, args: list[str]) -> CommandResult:
    if ctx.journal is None:
        return CommandResult(text="no journal attached.")
    try:
        records = ctx.journal.records()
    except Exception:
        records = []
    lines = [f"journal: {ctx.journal_path or '(unknown path)'}",
             f"records: {len(records)}"]
    if records:
        last = records[-1]
        lines.append(
            f"last: seq={last.get('seq', '?')} "
            f"type={last.get('type', '?')} ts={last.get('ts', '?')}")
    return CommandResult(text="\n".join(lines))


@command("/snapshot", "deterministic-engine snapshot manifest (N / N_prev)")
def _snapshot(ctx: CommandContext, args: list[str]) -> CommandResult:
    # Engine dir: alongside the journal (data/engine/) when present.
    base = Path(ctx.journal_path).parent / "engine" \
        if ctx.journal_path else Path.cwd() / "data" / "engine"
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return CommandResult(
            text="no deterministic-engine snapshots yet — "
                 "the engine has not written a snapshot manifest. "
                 "Paper journal snapshots are separate (see /db).")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return CommandResult(
            text=f"snapshot manifest unreadable: {exc}")
    snaps = manifest.get("snapshots", [])
    if not snaps:
        return CommandResult(text="snapshot manifest is empty.")
    lines = ["snapshots (newest first):"]
    for s in snaps:
        lines.append(
            f"N={s.get('watermark')} root={str(s.get('snapshot_root'))[:12]}… "
            f"chain={str(s.get('chain_n'))[:12]}…")
    return CommandResult(text="\n".join(lines))


@command("/reconcile", "reconciliation status: unresolved / open vs closed")
def _reconcile(ctx: CommandContext, args: list[str]) -> CommandResult:
    reconciler = getattr(ctx, "reconciler", None)
    if reconciler is not None:
        unresolved = reconciler.unresolved()
        return CommandResult(
            text=f"reconciler phase: {reconciler.phase}\n"
                 f"unresolved: {len(unresolved)}\n"
                 f"reconciliation_complete: "
                 f"{reconciler.reconciliation_complete}\n"
                 f"holding queue: {len(reconciler.holding)}")
    if ctx.journal is None:
        return CommandResult(text="no journal attached.")
    from goldenswing.execution.reporting import open_positions
    try:
        opens = open_positions(ctx.journal)
    except Exception:
        opens = []
    return CommandResult(
        text="no live reconciler attached to this loop.\n"
             f"journal open positions (unclosed): {len(opens)}\n"
             "Run the deterministic engine for live "
             "UNCONFIRMED/UNKNOWN reconciliation status.")


@command("/health", "feed + journal + open positions health")
def _health(ctx: CommandContext, args: list[str]) -> CommandResult:
    fs = ctx.feed_status or {}
    feed_line = ("feed: OK" if fs.get("ok")
                 else f"feed: STALE — {fs.get('reason', 'no status yet')}")
    lines = ["HEALTH", feed_line]
    if getattr(ctx, "deterministic", None) is not None:
        try:
            det = ctx.deterministic.status()
            lines.append(
                f"deterministic: {det.get('phase')} head={det.get('head')} "
                f"commit={det.get('commit_index')} "
                f"unresolved={det.get('unresolved')}")
        except Exception:
            pass
    if ctx.journal is not None:
        try:
            lines.append(f"journal records: {len(ctx.journal.records())}")
        except Exception:
            pass
    if ctx.paper is not None:
        try:
            lines.append(f"open paper positions: {len(ctx.paper.positions())}")
        except Exception:
            pass
    return CommandResult(text="\n".join(lines))


@command("/harvest", "harvest database + journal + state + results as one ZIP")
def _harvest(ctx: CommandContext, args: list[str]) -> CommandResult:
    scope = (args[0].lower() if args else "all")
    if scope not in HARVEST_SCOPES:
        return CommandResult(
            text="usage: /harvest [all|journal|state|results]\n"
                 "all = journal + state + results (default). "
                 "CSVs / pkl / PNGs excluded by design.")
    if scope != "state" and ctx.journal is None:
        return CommandResult(text="no journal attached.")
    try:
        zip_path, manifest = build_harvest_bundle(ctx, scope)
    except Exception as exc:
        return CommandResult(
            text=f"harvest failed: {type(exc).__name__}: {exc}")
    files = manifest["files"]
    n_trades = "?"
    total_r = ""
    if ctx.journal is not None:
        try:
            from goldenswing.execution.reporting import paper_book_stats
            book = paper_book_stats(ctx.journal).get("book", {}) or {}
            n_trades = book.get("trades", 0)
            total_r = f", {book.get('total_r', 0):+.2f}R"
        except Exception:
            pass
    caption = (f"harvest {scope} — {n_trades} trades{total_r}, "
               f"{len(files)} files, {manifest['zip_bytes'] // 1024}KB — "
               f"RESULTS.md inside has the full breakdown")
    return CommandResult(document=zip_path, caption=caption)
