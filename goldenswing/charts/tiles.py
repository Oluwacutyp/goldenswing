"""Shared bento furniture for the chart image organ.

The tile theme + drawing primitives both PNG cards use — the
daily recap (charts/recap.py) and the desk card (charts/desk.py)
— so the two renderers can never drift apart in ink, tile shape
or the unavailable-value language.  The charts' honesty contract
lives here too: :func:`unavailable` is the ONLY way a missing
value is drawn (a dash + the reason in words, never a fabricated
number standing in for an unread one).

Number hierarchy (design absorption, Magic UI's Number Ticker as
a typography rule): the hero figure is the largest thing on the
card, supporting stats step down in fixed ratio, and there is
exactly ONE accent treatment per image — each renderer spends it
on its P&L figure's sign colour.
"""
from __future__ import annotations

from matplotlib.patches import FancyBboxPatch, Rectangle

#: The classic chart theme (charts/engine.py) + tile neutrals.
BG = "#ffffff"
TILE = "#f7f8fa"
EDGE = "#e6e8ee"
TEXT = "#131722"
MUTED = "#6e7686"
UP = "#26a69a"
DOWN = "#ef5350"
BAR_IDLE = "#9aa3b2"
MONO = "DejaVu Sans Mono"


def tile(ax, x: float, y: float, w: float, h: float) -> None:
    """One rounded bento tile at axes fraction (x, y, w, h)."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=1.0, edgecolor=EDGE, facecolor=TILE))


def tile_label(ax, x: float, y: float, text: str) -> None:
    """A tile's small caps label, top-left inside the tile."""
    ax.text(x, y, text, fontsize=11, fontweight="bold",
            color=MUTED, ha="left", va="top")


def unavailable(ax, cx: float, cy: float, reason: str, *,
                value_size: float = 26) -> None:
    """An unread value's tile body: a dash + the reason in words.

    Never a number: a value the caller could not read is not 0,
    and the tile says why instead of dressing the gap up.
    """
    ax.text(cx, cy + 0.045, "—", fontsize=value_size, fontweight="bold",
            family=MONO, color=MUTED, ha="center", va="center")
    ax.text(cx, cy - 0.075, reason, fontsize=10.5,
            color=MUTED, ha="center", va="center")


def drawdown_bar(ax, *, x0: float, x1: float, y: float, h: float,
                 drawdown: float, limit: float, halted: bool) -> None:
    """The drawdown-vs-daily-halt bar: grey track, fill up to the
    halt tick standing at ``x1``.

    The fill stays the idle grey while the day has room; it turns
    red only when the halt is breached or active — inside this
    tile red means the halt, nowhere else (the image's one accent
    is spent on the P&L sign).  ``drawdown`` / ``limit`` are
    fractions (0.212 vs a 0.15 limit).
    """
    frac = max(0.0, min(1.0, drawdown / limit)) if limit > 0 else 0.0
    breached = drawdown >= limit
    bar_color = DOWN if (breached or halted) else BAR_IDLE
    ax.add_patch(Rectangle((x0, y), x1 - x0, h,
                           facecolor=EDGE, edgecolor="none"))
    if frac > 0:
        ax.add_patch(Rectangle((x0, y), (x1 - x0) * frac, h,
                               facecolor=bar_color, edgecolor="none"))
    ax.plot([x1, x1], [y - 0.012, y + h + 0.012],
            color=DOWN, linewidth=1.6)
    ax.text(x1, y + h + 0.028, "HALT", fontsize=8.5,
            fontweight="bold", color=DOWN, ha="center", va="bottom")
