"""Desk card image — the public face of the trading day.

Design absorption (owner's design-sources mining, 2026-11-07;
absorption list #12): the daily desk / LinkedIn image as a
repeatable template — Magic UI's Bento Grid + Number Ticker
typography, one-accent discipline.  This is the bento recap's
brand-facing sibling, NOT a second recap: the recap image
(charts/recap.py) is the bot's operational day-end card, sent to
the owner at rollover; the desk card is the polished public
image of the SAME day, rendered on demand (/desk) at LinkedIn's
1200x627 shape — wordmark, the day's hero number, and the
"Powered by Peacethefirst" stamp the typography law puts on
presented output.

Same numbers, by construction: the stats dict comes from
:func:`goldenswing.execution.reporting.daily_recap_image_stats`
— the journal numbers the text recap renders, vitals from the
runner's ``account_vitals()`` — and the tiles/furniture come
from charts/tiles.py, shared with the recap.  Desk, recap and
/health can never disagree.

Honesty contract (identical to the recap's): a value the caller
could not read arrives as ``None`` and renders as an unavailable
tile (a dash + the reason in words) — never as 0.00.  A win rate
with no closed trades is undefined, not 0%.  The day's realized
P&L with no closes IS a real zero and renders as +0.00 saying so.

Determinism matches the chart engine: fixed figure size/DPI,
the day string is data (not a wall clock), pinned PNG metadata —
the same stats dict renders byte-identical output.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .tiles import BG as _BG  # noqa: E402
from .tiles import DOWN as _DOWN  # noqa: E402
from .tiles import EDGE as _EDGE  # noqa: E402
from .tiles import MONO as _MONO  # noqa: E402
from .tiles import MUTED as _MUTED  # noqa: E402
from .tiles import TEXT as _TEXT  # noqa: E402
from .tiles import UP as _UP  # noqa: E402
from .tiles import drawdown_bar  # noqa: E402
from .tiles import tile as _tile  # noqa: E402
from .tiles import tile_label as _label  # noqa: E402
from .tiles import unavailable as _unavailable  # noqa: E402

#: The brand furniture: the product wordmark in the header and
#: the typography law's stamp in the footer.
WORDMARK = "GOLDENSWING"
STAMP = "Powered by Peacethefirst"

_FIGSIZE = (12, 6.27)        # at 100 dpi -> 1200x627 px (LinkedIn)
_DPI = 100


def render_desk_card(stats: dict) -> bytes:
    """Render the desk card image from a stats dict.

    The dict is built by
    :func:`goldenswing.execution.reporting.daily_recap_image_stats`
    (the recap's builder — journal numbers identical to the text
    recap's, vitals from ``account_vitals()``); this function
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
    ax.text(0.02, 0.965, WORDMARK, fontsize=18, fontweight="bold",
            color=_TEXT, ha="left", va="top")
    ax.text(0.30, 0.958, "DAILY DESK", fontsize=11.5,
            fontweight="bold", color=_MUTED, ha="left", va="top")
    ax.text(0.98, 0.965, day, fontsize=12.5, family=_MONO,
            color=_MUTED, ha="right", va="top")
    ax.add_patch(Rectangle((0.02, 0.878), 0.96, 0.0035,
                           facecolor=_EDGE, edgecolor="none"))
    if halted:
        badge = "RISK HALT ACTIVE"
        if halt_reason:
            badge = badge + " — " + halt_reason[:80]
        ax.text(0.5, 0.848, badge, fontsize=10.5, fontweight="bold",
                color=_DOWN, ha="center", va="top")

    # ---------------------------------------------------------- P&L tile
    _tile(ax, 0.02, 0.27, 0.56, 0.545)
    _label(ax, 0.045, 0.78, "NET P&L — CLOSED TRADES")
    accent = _UP if pnl > 0 else (_DOWN if pnl < 0 else _MUTED)
    pnl_s = f"{pnl:+,.2f}"
    ax.text(0.30, 0.545, pnl_s, fontsize=44, fontweight="bold",
            family=_MONO, color=accent, ha="center", va="center")
    if trades:
        sub = f"{r_net:+.2f}R · {trades} closed · {wins}W / {losses}L"
    else:
        sub = "no closed trades — nothing realized"
    if currency:
        sub = currency + " · " + sub
    ax.text(0.30, 0.345, sub, fontsize=11.5, color=_TEXT,
            ha="center", va="center")

    # ---------------------------------------------------------- win rate
    _tile(ax, 0.60, 0.555, 0.38, 0.26)
    _label(ax, 0.625, 0.78, "WIN RATE")
    if win_rate is None:
        _unavailable(ax, 0.79, 0.685, "no closed trades this day")
    else:
        wr_s = f"{float(win_rate):.0f}%"
        ax.text(0.79, 0.715, wr_s, fontsize=30, fontweight="bold",
                family=_MONO, color=_TEXT, ha="center", va="center")
        sub = f"{wins}W / {losses}L of {trades} closed"
        ax.text(0.79, 0.615, sub, fontsize=10.5, color=_MUTED,
                ha="center", va="center")

    # ------------------------------------------------------------ trades
    _tile(ax, 0.60, 0.27, 0.145, 0.26)
    _label(ax, 0.617, 0.495, "TRADES")
    n_s = str(trades)
    ax.text(0.6725, 0.415, n_s, fontsize=26, fontweight="bold",
            family=_MONO, color=_TEXT, ha="center", va="center")
    if trades:
        sub = "closed"
        if open_positions is not None:
            sub = sub + f" · {int(open_positions)} open"
    else:
        sub = "nothing closed"
    ax.text(0.6725, 0.315, sub, fontsize=10, color=_MUTED,
            ha="center", va="center")

    # ------------------------------------------------------------ equity
    _tile(ax, 0.765, 0.27, 0.215, 0.26)
    _label(ax, 0.782, 0.495, "EQUITY")
    if equity is None:
        _unavailable(ax, 0.8725, 0.40, "account read n/a",
                     value_size=20)
    else:
        eq_s = f"{float(equity):,.2f}"
        ax.text(0.8725, 0.42, eq_s, fontsize=17, fontweight="bold",
                family=_MONO, color=_TEXT, ha="center", va="center")
        if balance is not None:
            bal = f"bal {float(balance):,.2f}"
            if currency:
                bal = bal + " " + currency
            ax.text(0.8725, 0.325, bal, fontsize=9.5, color=_MUTED,
                    ha="center", va="center")

    # ------------------------------------------- drawdown vs daily halt
    _tile(ax, 0.02, 0.075, 0.96, 0.155)
    limit_s = f"{limit:.0%}"
    _label(ax, 0.045, 0.195, "DAY DRAWDOWN vs " + limit_s + " DAILY HALT")
    if drawdown is None:
        ax.text(0.5, 0.135,
                "drawdown unavailable — no day-start balance recorded",
                fontsize=11.5, color=_MUTED, ha="center", va="center")
    else:
        dd = float(drawdown)
        breached = dd >= limit
        drawdown_bar(ax, x0=0.045, x1=0.70, y=0.105, h=0.04,
                     drawdown=dd, limit=limit, halted=halted)
        dd_s = f"{dd:.1%}"
        ax.text(0.845, 0.16, dd_s, fontsize=15, fontweight="bold",
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
        ax.text(0.845, 0.105, state, fontsize=9.5, color=state_color,
                ha="center", va="center")

    # ---------------------------------------------------------- footer
    ax.text(0.02, 0.038, STAMP, fontsize=10.5, color=_MUTED,
            ha="left", va="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=_BG,
                metadata={"Software": "GoldenSwing charts"})
    plt.close(fig)
    return buf.getvalue()
