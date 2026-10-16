"""Backtest result export for Telegram.

A backtest result is a file + a concise summary, never a silent
write and never an unbounded text dump.  :func:`summarize` computes
trades / win% / total R / PF / MaxDD / period from a list of trade
rows (each with ``r_net`` and ``pnl``).  :func:`write_report`
writes a markdown report file.  :func:`chunk_text` splits long
summaries into Telegram-sized chunks (4096 chars) so the bot can
send them without truncation.  The owner drops the exported file
back to Muse for review — the file is the contract, the chat
summary is only a pointer.
"""
from __future__ import annotations

import json
from pathlib import Path

TELEGRAM_MESSAGE_LIMIT = 4096


def summarize(trades: list[dict], *, period: str = "",
              d1_bars: int | None = None,
              d1_stale: bool = False) -> dict:
    """Summarise trade rows.  Honest about D1 coverage: if D1 bars
    are insufficient/stale, ``d1_coverage_ok`` is False and the
    summary says so — never claim a clean backtest on stale data."""
    rs = [float(t.get("r_net", 0.0) or 0.0) for t in trades]
    wins = [r for r in rs if r > 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(r for r in rs if r <= 0))
    # MaxDD in R: worst peak-to-trough of the cumulative R curve.
    peak = 0.0
    maxdd = 0.0
    running = 0.0
    for r in rs:
        running += r
        peak = max(peak, running)
        maxdd = max(maxdd, peak - running)
    return {
        "trades": len(rs),
        "win_rate_pct": (100.0 * len(wins) / len(rs)) if rs else 0.0,
        "total_r": sum(rs),
        "avg_r": (sum(rs) / len(rs)) if rs else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0
        else (float("inf") if gross_win > 0 else 0.0),
        "max_drawdown_r": maxdd,
        "total_pnl": sum(float(t.get("pnl", 0.0) or 0.0) for t in trades),
        "period": period,
        "d1_bars": d1_bars,
        "d1_stale": bool(d1_stale),
        "d1_coverage_ok": (not d1_stale) and (d1_bars is None or d1_bars >= 2),
    }


def format_summary(summary: dict) -> str:
    lines = [
        "GoldenSwing backtest",
        f"period: {summary.get('period') or '(unspecified)'}",
        f"trades: {summary['trades']}",
        f"win%: {summary['win_rate_pct']:.1f}%",
        f"total R: {summary['total_r']:+.2f}R",
        f"avg R: {summary['avg_r']:+.2f}R",
        f"profit factor: {summary['profit_factor']:.2f}"
        if summary['profit_factor'] != float('inf') else "profit factor: inf",
        f"max drawdown: {summary['max_drawdown_r']:.2f}R",
        f"total PnL: {summary['total_pnl']:+.2f}",
    ]
    if summary.get("d1_bars") is not None:
        lines.append(f"D1 bars: {summary['d1_bars']}")
    if not summary.get("d1_coverage_ok", True):
        lines.append("WARNING: D1 coverage insufficient/stale — "
                     "backtest is NOT clean, do not promote on it.")
    return "\n".join(lines)


def write_report(trades: list[dict], path: str | Path, *,
                 period: str = "", d1_bars: int | None = None,
                 d1_stale: bool = False) -> tuple[Path, dict]:
    """Write a markdown backtest report.  Returns (path, summary)."""
    summary = summarize(trades, period=period, d1_bars=d1_bars,
                        d1_stale=d1_stale)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# GoldenSwing backtest report", "",
             f"period: {period or '(unspecified)'}", "",
             "## Summary", "", format_summary(summary), "",
             "## Trades", ""]
    for i, trade in enumerate(trades, 1):
        lines.append(
            f"{i}. {trade.get('playbook', '?')} "
            f"{trade.get('direction', '')} "
            f"r={float(trade.get('r_net', 0.0) or 0.0):+.2f}R "
            f"pnl={float(trade.get('pnl', 0.0) or 0.0):+.2f} "
            f"({trade.get('reason', '')})")
    lines += ["", "## Raw", "", "```json",
              json.dumps({"summary": summary, "trades": trades},
                         indent=2, default=str), "```", ""]
    path.write_text("\n".join(lines))
    return path, summary


def chunk_text(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split ``text`` into Telegram-sized chunks on line boundaries.

    A single line longer than ``limit`` is hard-split into
    limit-sized pieces — otherwise one long line would produce one
    over-limit chunk.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    def _flush():
        nonlocal current, size
        if current:
            chunks.append("".join(current))
            current, size = [], 0

    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            room = limit - size
            if room <= 0:
                _flush()
                room = limit
            take = line[:room]
            current.append(take)
            size += len(take)
            line = line[room:]
            if size >= limit:
                _flush()
        if line:
            if size + len(line) > limit:
                _flush()
            current.append(line)
            size += len(line)
    _flush()
    return chunks
