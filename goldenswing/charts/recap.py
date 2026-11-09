"""Bento recap image — the daily recap as ONE glanceable tile image.

Design absorption (owner's design-sources mining, 2026-11-07;
absorption list #5): Magic UI's Bento Grid + the Number Ticker
typography rule, translated into the matplotlib PNG organ the
alert charts already live in.  The P&L figure is the largest thing
on the image (tabular numerals, one number per tile + one label),
supporting stats step down in fixed ratio, and there is exactly
ONE accent treatment — the P&L's sign colour.  Everything else is
the classic chart theme's neutral ink, so the image survives
Telegram size the way the signal charts do.

Honesty contract (the /health vitals doctrine): a value the caller
could not read arrives as ``None`` and renders as an unavailable
tile (a dash + the reason in words) — never as 0.00.  A win rate
with no closed trades is undefined, so it is unavailable, not 0%.
The day's realized P&L with no closes IS a real zero (nothing
closed, nothing realized) and renders as +0.00 saying so.

Determinism matches the chart engine: fixed figure size/DPI, the
day string is data (not a wall clock), pinned PNG metadata — the
same stats dict renders byte-identical output.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from .tiles import BG as _BG  # noqa: E402
from .tiles import DOWN as _DOWN  # noqa: E402
from .tiles import MONO as _MONO  # noqa: E402
from .tiles import MUTED as _MUTED  # noqa: E402
from .tiles import TEXT as _TEXT  # noqa: E402
from .tiles import UP as _UP  # noqa: E402
from .tiles import drawdown_bar  # noqa: E402
from .tiles import tile as _tile  # noqa: E402
from .tiles import tile_label as _label  # noqa: E402
from .tiles import unavailable as _unavailable  # noqa: E402

_FIGSIZE = (12, 7)           # at 100 dpi -> 1200x700 px
_DPI = 100


def render_daily_recap(stats: dict) -> bytes:
    """Render the bento daily-recap image from a stats dict.

    The dict is built by
    :func:`goldenswing.execution.reporting.daily_recap_image_stats`
    (journal numbers identical to the text recap's) — this function
    only draws.  Keys: day, pnl, r_net, trades, wins, losses,
    win_rate, open_positions, equity, balance, currency,
    drawdown_pct, drawdown_limit, halted, halt_reason.
    """
    day = str(stats.get("day") or "")
    pnl = float(stats.get("pnl") or 0.0)
    r_net = float(stats.get("r_net") or 0.0)
    trades = int(stats.get("trades") or 0)
    wins = int(stats.get("wins") or 0)
    losses = int(stats.get("losses") or 0)
    win_rate = stats.get("win_rate")
    open_positions = stats.get("open_positions")
    equity = stats.get("equity")
    balance = stats.get("balance")
    currency = str(stats.get("currency") or "")
    drawdown = stats.get("drawdown_pct")
    limit = float(stats.get("drawdown_limit") or 0.15)
    halted = bool(stats.get("halted"))
    halt_reason = str(stats.get("halt_reason") or "")

    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI)
    fig.patch.set_facecolor(_BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------------------------------------------------- header
    ax.text(0.02, 0.965, "GOLDENSWING — DAILY RECAP", fontsize=15,
            fontweight="bold", color=_TEXT, ha="left", va="top")
    ax.text(0.98, 0.965, day, fontsize=12.5, family=_MONO,
            color=_MUTED, ha="right", va="top")
    if halted:
        badge = "RISK HALT ACTIVE"
        if halt_reason:
            badge = badge + " — " + halt_reason[:80]
        ax.text(0.5, 0.905, badge, fontsize=11, fontweight="bold",
                color=_DOWN, ha="center", va="top")

    # ---------------------------------------------------------- P&L tile
    _tile(ax, 0.02, 0.30, 0.56, 0.56)
    _label(ax, 0.045, 0.825, "NET P&L — CLOSED TRADES")
    accent = _UP if pnl > 0 else (_DOWN if pnl < 0 else _MUTED)
    pnl_s = f"{pnl:+,.2f}"
    ax.text(0.30, 0.60, pnl_s, fontsize=42, fontweight="bold",
            family=_MONO, color=accent, ha="center", va="center")
    if trades:
        sub = f"{r_net:+.2f}R · {trades} closed · {wins}W / {losses}L"
    else:
        sub = "no closed trades — nothing realized"
    if currency:
        sub = currency + " · " + sub
    ax.text(0.30, 0.375, sub, fontsize=11.5, color=_TEXT,
            ha="center", va="center")

    # ---------------------------------------------------------- win rate
    _tile(ax, 0.60, 0.60, 0.38, 0.26)
    _label(ax, 0.625, 0.825, "WIN RATE")
    if win_rate is None:
        _unavailable(ax, 0.79, 0.715, "no closed trades this day")
    else:
        wr_s = f"{float(win_rate):.0f}%"
        ax.text(0.79, 0.745, wr_s, fontsize=30, fontweight="bold",
                family=_MONO, color=_TEXT, ha="center", va="center")
        sub = f"{wins}W / {losses}L of {trades} closed"
        ax.text(0.79, 0.645, sub, fontsize=10.5, color=_MUTED,
                ha="center", va="center")

    # ------------------------------------------------- open positions
    _tile(ax, 0.60, 0.32, 0.145, 0.26)
    _label(ax, 0.617, 0.545, "OPEN")
    if open_positions is None:
        _unavailable(ax, 0.6725, 0.43, "read n/a", value_size=20)
    else:
        n_s = str(int(open_positions))
        ax.text(0.6725, 0.445, n_s, fontsize=24, fontweight="bold",
                family=_MONO, color=_TEXT, ha="center", va="center")
        ax.text(0.6725, 0.355, "positions", fontsize=10,
                color=_MUTED, ha="center", va="center")

    # ---------------------------------------------------------- equity
    _tile(ax, 0.765, 0.32, 0.215, 0.26)
    _label(ax, 0.782, 0.545, "EQUITY")
    if equity is None:
        _unavailable(ax, 0.8725, 0.43, "account read n/a",
                     value_size=20)
    else:
        eq_s = f"{float(equity):,.2f}"
        ax.text(0.8725, 0.45, eq_s, fontsize=17, fontweight="bold",
                family=_MONO, color=_TEXT, ha="center", va="center")
        if balance is not None:
            bal = f"bal {float(balance):,.2f}"
            if currency:
                bal = bal + " " + currency
            ax.text(0.8725, 0.36, bal, fontsize=9.5, color=_MUTED,
                    ha="center", va="center")

    # ------------------------------------------- drawdown vs daily halt
    _tile(ax, 0.02, 0.05, 0.96, 0.19)
    limit_s = f"{limit:.0%}"
    _label(ax, 0.045, 0.205, "DAY DRAWDOWN vs " + limit_s + " DAILY HALT")
    if drawdown is None:
        ax.text(0.5, 0.115,
                "drawdown unavailable — no day-start balance recorded",
                fontsize=11.5, color=_MUTED, ha="center", va="center")
    else:
        dd = float(drawdown)
        breached = dd >= limit
        drawdown_bar(ax, x0=0.045, x1=0.70, y=0.085, h=0.045,
                     drawdown=dd, limit=limit, halted=halted)
        dd_s = f"{dd:.1%}"
        ax.text(0.845, 0.135, dd_s, fontsize=15, fontweight="bold",
                family=_MONO, color=_TEXT, ha="center", va="center")
        if halted:
            state = "HALTED"
            state_color = _DOWN
        elif breached:
            state = "at / over the halt"
            state_color = _DOWN
        else:
            room = limit - dd
            state = f"{room:.1%} of room"
            state_color = _MUTED
        ax.text(0.845, 0.075, state, fontsize=9.5, color=state_color,
                ha="center", va="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=_BG,
                metadata={"Software": "GoldenSwing charts"})
    plt.close(fig)
    return buf.getvalue()
