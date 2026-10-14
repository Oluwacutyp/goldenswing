"""Daily summary message: the desk's end-of-day report.

Built from the journal (opens/closes/skips for the day), the risk
guards' snapshot (halted? drawdown?), and the account state.  Sent
silent — it's a report, not an alarm.  Mined from the v18 bot's
``notify_balance`` / status chatter, rebuilt on our journal so the
numbers are the audited ones, not a side counter's.
"""
from __future__ import annotations

from datetime import date


def _day_of(rec: dict) -> str | None:
    ts = rec.get("ts") or rec.get("timestamp") or ""
    return str(ts)[:10] or None


def build_daily_summary(*, day: str | None = None,
                        records: list[dict] | None = None,
                        halted: bool = False, halt_reason: str = "",
                        balance: float | None = None,
                        drawdown_pct: float | None = None,
                        open_positions: int = 0,
                        alerts_sent: int = 0) -> str:
    """Render the day's summary as plain text.

    ``day`` defaults to today (UTC).  ``records`` are journal records;
    only those stamped with ``day`` are counted.
    """
    day = day or date.today().isoformat()
    records = records or []
    todays = [r for r in records if _day_of(r) == day]
    opens = [r for r in todays if r.get("type") == "open"]
    closes = [r for r in todays if r.get("type") == "close"]
    skips = [r for r in todays if r.get("type") == "skip"]
    wins = [c for c in closes if (c.get("pnl_money") or 0) >= 0]
    pnl = sum(c.get("pnl_money") or 0 for c in closes)
    r_net = sum(c.get("r_net") or 0 for c in closes)

    icon = "🟢" if pnl >= 0 else "🔴"
    lines = [
        f"📊 DAILY SUMMARY — {day}",
        "",
        f"{icon} Net: {pnl:+.2f}  ({r_net:+.2f}R)",
        f"Trades closed: {len(closes)} "
        f"({len(wins)}W / {len(closes) - len(wins)}L)",
        f"Opened: {len(opens)} · skipped: {len(skips)} · "
        f"alerts sent: {alerts_sent}",
    ]
    if open_positions:
        lines.append(f"Open positions: {open_positions}")
    if balance is not None:
        lines.append(f"Balance: {balance:,.2f}")
    if drawdown_pct is not None:
        lines.append(f"Drawdown from peak: {drawdown_pct:.1%}")
    if halted:
        lines.append("")
        lines.append(f"⛔ RISK HALT ACTIVE — {halt_reason or 'no reason given'}")
        lines.append("new risk denied until a human resets the halt")
    elif closes:
        wr = len(wins) / len(closes)
        lines.append(f"Win rate today: {wr:.0%}")
    return "\n".join(lines)
