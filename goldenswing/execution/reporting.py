"""Shared paper-book stats + report-file generation.

The numbers behind the Telegram ``/report`` command and
``scripts/paper_report.py`` (stdout) come from exactly one place: this
module reads the append-only journal and the versioned
``data/measured_oos.json`` records.  Nothing is hardcoded — a report is
only as fresh as the journal it was generated from.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .learning import TradeJournal

#: Where /report writes its document (overwritten each run).
REPORTS_DIRNAME = "reports"


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def iter_paper_trades(journal: TradeJournal) -> list[dict]:
    """Closed paper trades, each joined to its open record.

    Every record carries: alert_id, ticket, playbook, direction, score,
    entry, stop, volume, risk_money, exit, reason, pnl_money, r_net,
    hold_bars, opened_ts, closed_ts.  Sorted oldest-first.
    """
    records = journal.records()
    opens = {r["alert_id"]: r for r in records
             if r.get("type") == "open"}
    trades = []
    for r in records:
        if r.get("type") != "close" or r.get("alert_id") not in opens:
            continue
        o = opens[r["alert_id"]]
        trades.append({
            "alert_id": r["alert_id"],
            "ticket": r.get("ticket") or o.get("ticket"),
            "playbook": o.get("playbook", "?"),
            "direction": int(o.get("direction") or 0),
            "score": _to_float(o.get("score")),
            "entry": _to_float(o.get("entry")),
            "stop": _to_float(o.get("stop")),
            "volume": _to_float(o.get("volume")),
            "risk_money": _to_float(o.get("risk_money")),
            "exit": _to_float(r.get("exit")),
            "reason": r.get("reason", "?"),
            "pnl_money": _to_float(r.get("pnl_money")),
            "r_net": _to_float(r.get("r_net")),
            "hold_bars": int(r.get("hold_bars") or 0),
            "opened_ts": o.get("ts"),
            "closed_ts": r.get("ts"),
        })
    trades.sort(key=lambda t: t["closed_ts"] or "")
    return trades


def open_positions(journal: TradeJournal) -> list[dict]:
    """Opens with no matching close — the paper book's live risk."""
    records = journal.records()
    opens = {r["alert_id"]: r for r in records
             if r.get("type") == "open"}
    closed_ids = {r["alert_id"] for r in records
                  if r.get("type") == "close" and r.get("alert_id") in opens}
    return [opens[a] for a in opens if a not in closed_ids]


def _stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"trades": 0}
    rs = [r["r_net"] for r in rows]
    wins = [r for r in rs if r > 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(r for r in rs if r <= 0))
    return {
        "trades": n,
        "win_rate": 100.0 * len(wins) / n,
        "total_r": sum(rs),
        "avg_r": sum(rs) / n,
        "pf": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "total_pnl": sum(r["pnl_money"] for r in rows),
    }


def paper_book_stats(journal: TradeJournal) -> dict:
    """Per-playbook + whole-book stats from closed paper trades."""
    by_pb: dict[str, list[dict]] = defaultdict(list)
    for t in iter_paper_trades(journal):
        by_pb[t["playbook"]].append(t)
    per_playbook = {pb: _stats(rows) for pb, rows in sorted(by_pb.items())}
    return {
        "per_playbook": per_playbook,
        "book": _stats([t for rows in by_pb.values() for t in rows]),
        "open_count": len(open_positions(journal)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "journal": str(journal.path),
    }


def load_measured_oos(path: str | Path | None = None) -> list[dict]:
    """Versioned measured OOS records (never hardcoded in logic)."""
    p = Path(path) if path else \
        Path(__file__).resolve().parents[2] / "data" / "measured_oos.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("rows", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def _fmt(value, fmt="{:+.2f}") -> str:
    return "n/a" if value is None else fmt.format(value)


def render_report_markdown(stats: dict, measured_oos: list[dict] | None,
                           trades: list[dict] | None = None) -> str:
    """The /report document body."""
    lines = [
        "# GoldenSwing paper book — report",
        "",
        f"Generated (UTC): {stats.get('generated_utc', '?')}",
        f"Journal: {stats.get('journal', '?')}",
        "",
        "## Paper book (journal, closed trades)",
        "",
        "| playbook | trades | win% | +R | avgR | PF | pnl |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pb, s in stats["per_playbook"].items():
        if s["trades"] == 0:
            continue
        lines.append(
            f"| {pb} | {s['trades']} | {s['win_rate']:.0f} | "
            f"{s['total_r']:+.2f} | {s['avg_r']:+.2f} | "
            f"{s['pf']:.2f} | {s['total_pnl']:+.2f} |")
    b = stats["book"]
    if b["trades"]:
        lines.append(
            f"| **book** | **{b['trades']}** | **{b['win_rate']:.0f}** | "
            f"**{b['total_r']:+.2f}** | **{b['avg_r']:+.2f}** | "
            f"**{b['pf']:.2f}** | **{b['total_pnl']:+.2f}** |")
    else:
        lines.append("| (no closed paper trades yet) | | | | | | |")
    lines += [
        "",
        f"Open positions: {stats.get('open_count', 0)}",
        "",
        "## Measured out-of-sample (experiment records)",
        "",
        "From `data/measured_oos.json` — provenance on each row:",
        "",
    ]
    for row in measured_oos or []:
        ret = row.get("oos_return_pct")
        ret_txt = f"{ret:+.2f}%" if ret is not None else \
            f"{row.get('oos_total_r', 0):+.1f}R"
        lines.append(
            f"- **{row['playbook']}** ({row.get('variant', '?')}): "
            f"{row.get('oos_trades', '?')} OOS trades, {ret_txt}, "
            f"PF {_fmt(row.get('oos_pf'))} — {row.get('status', '?')} "
            f"[{row.get('source', '?')}]")
    if not (measured_oos or []):
        lines.append("(no measured-OOS records found)")
    if trades:
        lines += ["", "## Last closed trades", ""]
        for t in trades[-10:]:
            side = "LONG" if t["direction"] > 0 else "SHORT"
            lines.append(
                f"- {t['closed_ts'] or '?'} {t['playbook']} {side} "
                f"{t['r_net']:+.2f}R ({t['reason']})")
    lines += [
        "",
        "---",
        "Promotion bar: ≥50 OOS trades per sleeve + stability "
        "(CONFLUENCE.md checklist).",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------- daily/weekly recap
# A recap is the disciplined caller's update: EVERY closed trade in
# the window, wins AND losses, listed individually — never a
# wins-only highlight reel.  The journal already records all trades;
# the recap must not hide the losers.


def _parse_ts(value) -> datetime | None:
    """Journal ts (ISO) -> aware datetime; None when unparseable."""
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def recap_trades(journal: TradeJournal, *, since: datetime,
                 until: datetime) -> list[dict]:
    """Closed trades whose close timestamp falls in [since, until)."""
    out = []
    for t in iter_paper_trades(journal):
        ts = _parse_ts(t.get("closed_ts"))
        if ts is not None and since <= ts < until:
            out.append(t)
    return out


def _pf_fmt(pf: float) -> str:
    if pf == float("inf"):
        return "∞"
    return f"{pf:.2f}"


def render_recap_markdown(trades: list[dict], *, title: str,
                          window_label: str) -> str:
    """Recap document: totals + per-playbook + EVERY trade.

    Losses are listed with the wins, in the same table, with the
    same detail — a recap that hides losers is a lie by omission.
    """
    lines = [
        f"# GoldenSwing recap — {title}",
        "",
        f"Window (UTC): {window_label}",
        "",
    ]
    stats = _stats(trades)
    if stats["trades"] == 0:
        lines.append("(no closed trades in this window)")
        lines.append("")
        return "\n".join(lines)
    wins = sum(1 for t in trades if t["r_net"] > 0)
    losses = stats["trades"] - wins
    lines += [
        f"Trades: {stats['trades']} · Wins: {wins} · "
        f"Losses: {losses} · Win%: {stats['win_rate']:.0f} · "
        f"Total: {stats['total_r']:+.2f}R · PF: "
        f"{_pf_fmt(stats['pf'])} · PnL: {stats['total_pnl']:+.2f}",
        "",
        "## Per playbook",
        "",
        "| playbook | trades | win% | +R | avgR | PF |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_pb: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_pb[t["playbook"]].append(t)
    for pb, rows in sorted(by_pb.items()):
        s = _stats(rows)
        lines.append(
            f"| {pb} | {s['trades']} | {s['win_rate']:.0f} | "
            f"{s['total_r']:+.2f} | {s['avg_r']:+.2f} | "
            f"{_pf_fmt(s['pf'])} |")
    lines += [
        "",
        "## All closed trades — wins AND losses, nothing hidden",
        "",
    ]
    for t in trades:
        side = "LONG" if t["direction"] > 0 else "SHORT"
        mark = "✅" if t["r_net"] > 0 else "❌"
        lines.append(
            f"- {mark} {t['closed_ts'] or '?'} {t['playbook']} {side} "
            f"{t['r_net']:+.2f}R ({t['reason']}) "
            f"pnl {t['pnl_money']:+.2f}")
    lines.append("")
    return "\n".join(lines)


def build_daily_recap(journal: TradeJournal, day) -> str:
    """Recap for one UTC calendar day (``day``: a ``date``)."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    trades = recap_trades(journal, since=start, until=end)
    return render_recap_markdown(
        trades, title=f"daily — {day.isoformat()}",
        window_label=f"{start.date().isoformat()} (UTC calendar day)")


def build_weekly_recap(journal: TradeJournal, week_start) -> str:
    """Recap for one Mon–Sun UTC week (``week_start``: Monday date)."""
    start = datetime(week_start.year, week_start.month, week_start.day,
                     tzinfo=timezone.utc)
    end = start + timedelta(days=7)
    trades = recap_trades(journal, since=start, until=end)
    return render_recap_markdown(
        trades, title=f"weekly — {start.date().isoformat()}",
        window_label=f"{start.date().isoformat()} -> "
                     f"{(end - timedelta(days=1)).date().isoformat()} "
                     f"(UTC Mon–Sun week)")


def recap_for_period(journal: TradeJournal, period: str) -> str:
    """Recap for the trailing window: ``daily`` = last 24h,
    ``weekly`` = last 7d (used by /recap)."""
    now = datetime.now(timezone.utc)
    if period == "weekly":
        trades = recap_trades(journal, since=now - timedelta(days=7),
                              until=now)
        return render_recap_markdown(
            trades, title="weekly — last 7 days",
            window_label="trailing 7 days to now (UTC)")
    trades = recap_trades(journal, since=now - timedelta(days=1),
                          until=now)
    return render_recap_markdown(
        trades, title="daily — last 24 hours",
        window_label="trailing 24 hours to now (UTC)")


def build_report_file(journal: TradeJournal, *,
                      out_dir: str | Path | None = None,
                      measured_oos: list[dict] | None = None,
                      name: str | None = None) -> Path:
    """Write the /report markdown document; return its path."""
    root = Path(__file__).resolve().parents[2]
    out = Path(out_dir) if out_dir else root / "data" / REPORTS_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out / (name or f"goldenswing-report-{stamp}.md")
    stats = paper_book_stats(journal)
    trades = iter_paper_trades(journal)
    if measured_oos is None:
        measured_oos = load_measured_oos()
    path.write_text(render_report_markdown(stats, measured_oos, trades))
    return path
