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

Optional market ticker strip (the desk build's named remainder,
closed when the runner gained the ``desk_ticker`` seam): when
the stats dict carries a ``ticker`` dict — the symbol's last
tick mid with its freshness class, the DXY's last closed H1
close, the session — the card draws a one-line market strip
above the drawdown tile and the middle tiles compress to make
room, the month target strip's own pattern.  Without the dict
the strip is not drawn and every coordinate is the card's
original layout, byte-identical to the card that shipped
before the strip existed.  The strip obeys the same honesty
contract: a price the feed could not supply is a dash + its
reason in words, and a price older than the engine's live
class is NAMED stale / market-closed with its age — never
dressed as live.
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
from .tiles import target_bar as _target_bar  # noqa: E402
from .tiles import tile as _tile  # noqa: E402
from .tiles import tile_label as _label  # noqa: E402
from .tiles import unavailable as _unavailable  # noqa: E402

#: The brand furniture: the product wordmark in the header and
#: the typography law's stamp in the footer.
WORDMARK = "GOLDENSWING"
STAMP = "Powered by Peacethefirst"

#: Session words for the ticker strip — the engine's session
#: vocabulary (live/regime.session_of) in the card's voice.
_SESSION_WORDS = {
    "asia": "Asia session",
    "london": "London session",
    "overlap": "London/NY overlap",
    "ny": "New York session",
    "offhours": "off-hours",
}


def _age_words(age) -> str:
    """A tick age in words: seconds, then minutes, then hours."""
    if age is None:
        return "age unknown"
    secs = float(age)
    if secs < 90:
        return f"{secs:.0f}s ago"
    if secs < 5400:
        return f"{secs / 60:.0f}m ago"
    return f"{secs / 3600:.1f}h ago"


def ticker_line(ticker: dict) -> str:
    """The market ticker strip's one line, from the runner's
    ``desk_ticker`` dict (see
    :meth:`goldenswing.execution.mt5_runner.MT5Runner.desk_ticker`).

    Segments: the symbol's last price with its freshness, the
    DXY's last closed H1 close, the session — the day-log spec
    for the strip (XAUUSD/DXY/session).  A value the runner
    could not supply renders as a dash + its reason in words —
    never a number standing in for an unread one; a price
    outside the engine's live class is named with its state
    and age, never presented bare as if live.  Pure text: the
    card draws this line and the /desk caption carries it, so
    the two can never disagree.
    """
    parts: list[str] = []
    symbol = str(ticker.get("symbol") or "XAUUSD")
    price = ticker.get("price")
    state = str(ticker.get("tick_state") or "no_ticks")
    age = ticker.get("tick_age_s")
    if price is None:
        reason = ("market closed — no tick yet"
                  if state == "closed_expected" else "no tick yet")
        parts.append(symbol + " — " + reason)
    else:
        seg = f"{symbol} {float(price):,.2f}"
        if state == "stale":
            seg = seg + f" (stale — last tick {_age_words(age)})"
        elif state == "closed_expected":
            seg = seg + (f" (market closed — last tick "
                         f"{_age_words(age)})")
        parts.append(seg)
    dxy = ticker.get("dxy")
    if dxy is not None:
        parts.append(f"DXY {float(dxy):,.2f}")
    else:
        status = str(ticker.get("dxy_status") or "unavailable")
        if status.startswith("fetch failed"):
            status = "feed fetch failed"
        parts.append("DXY — " + status)
    label = _SESSION_WORDS.get(str(ticker.get("session") or ""))
    if label:
        parts.append(label)
    return " · ".join(parts)

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

    Optional month-target keys (reporting.monthly_target_progress,
    present only when the owner has set the display-only
    ``monthly_target`` live-config key): monthly_target,
    month_pnl, month_pct, month_label.  When monthly_target is
    absent/None the strip is not drawn and every other coordinate
    is the card's original layout — an unset target renders
    byte-identical to the card that shipped before targets
    existed.  The target never recolours: the strip follows the
    drawdown bar's neutral grammar (the one accent stays the
    P&L sign).

    Optional market ticker (the runner's ``desk_ticker`` dict,
    merged by the /desk assembly when the runner carries the
    seam): ``ticker`` with symbol, price, tick_state,
    tick_age_s, dxy, dxy_status, session.  When absent the
    ticker strip is not drawn and the layout is the card's
    original (or target-only) geometry — byte-identical to the
    card that shipped before the strip existed.  When present
    the strip draws even with every market value unavailable:
    a dead feed is named on the strip, never silently absent.
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
    target = stats.get("monthly_target")
    has_target = target is not None
    month_pnl = stats.get("month_pnl")
    month_pct = stats.get("month_pct")
    month_label = str(stats.get("month_label") or "")
    ticker = stats.get("ticker")
    has_ticker = isinstance(ticker, dict)

    # Vertical rhythm: each optional strip (month target, market
    # ticker) steps the middle rows up to make room for it; with
    # no strip, every coordinate below is the card's original
    # layout (the byte-identical unset render above).  `level`
    # counts the strips present: 0 = original, 1 = one strip,
    # 2 = both — levels 0/1 reproduce the shipped constants
    # exactly.
    level = (1 if has_target else 0) + (1 if has_ticker else 0)
    pnl_y = (0.27, 0.335, 0.405)[level]
    pnl_h = (0.545, 0.48, 0.41)[level]
    pnl_num_y = (0.545, 0.575, 0.605)[level]
    pnl_sub_y = (0.345, 0.41, 0.475)[level]
    wr_y = (0.555, 0.585, 0.615)[level]
    wr_h = (0.26, 0.23, 0.20)[level]
    wr_val_y = (0.715, 0.725, 0.735)[level]
    wr_sub_y = (0.615, 0.63, 0.645)[level]
    wr_unavail_y = (0.685, 0.70, 0.715)[level]
    row_y = pnl_y
    row_h = (0.26, 0.23, 0.20)[level]
    row_lab_y = (0.495, 0.53, 0.565)[level]
    tr_val_y = (0.415, 0.45, 0.485)[level]
    tr_sub_y = (0.315, 0.375, 0.435)[level]
    eq_val_y = (0.42, 0.455, 0.49)[level]
    eq_sub_y = (0.325, 0.385, 0.445)[level]
    eq_unavail_y = (0.40, 0.45, 0.50)[level]

    # Strip slots above the drawdown tile, bottom-up: the ticker
    # takes the first slot; the target strip moves up one slot
    # when the ticker is also present (its content shifts by the
    # same step).  A lone strip sits in the slot the target strip
    # shipped in.
    ticker_tile_y = 0.245
    ticker_tile_h = 0.065 if has_target else 0.07
    ticker_cy = 0.278 if has_target else 0.28
    target_dy = 0.085 if has_ticker else 0.0

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
    _tile(ax, 0.02, pnl_y, 0.56, pnl_h)
    _label(ax, 0.045, 0.78, "NET P&L — CLOSED TRADES")
    accent = _UP if pnl > 0 else (_DOWN if pnl < 0 else _MUTED)
    pnl_s = f"{pnl:+,.2f}"
    ax.text(0.30, pnl_num_y, pnl_s, fontsize=44, fontweight="bold",
            family=_MONO, color=accent, ha="center", va="center")
    if trades:
        sub = f"{r_net:+.2f}R · {trades} closed · {wins}W / {losses}L"
    else:
        sub = "no closed trades — nothing realized"
    if currency:
        sub = currency + " · " + sub
    ax.text(0.30, pnl_sub_y, sub, fontsize=11.5, color=_TEXT,
            ha="center", va="center")

    # ---------------------------------------------------------- win rate
    _tile(ax, 0.60, wr_y, 0.38, wr_h)
    _label(ax, 0.625, 0.78, "WIN RATE")
    if win_rate is None:
        _unavailable(ax, 0.79, wr_unavail_y, "no closed trades this day")
    else:
        wr_s = f"{float(win_rate):.0f}%"
        ax.text(0.79, wr_val_y, wr_s, fontsize=30, fontweight="bold",
                family=_MONO, color=_TEXT, ha="center", va="center")
        sub = f"{wins}W / {losses}L of {trades} closed"
        ax.text(0.79, wr_sub_y, sub, fontsize=10.5, color=_MUTED,
                ha="center", va="center")

    # ------------------------------------------------------------ trades
    _tile(ax, 0.60, row_y, 0.145, row_h)
    _label(ax, 0.617, row_lab_y, "TRADES")
    n_s = str(trades)
    ax.text(0.6725, tr_val_y, n_s, fontsize=26, fontweight="bold",
            family=_MONO, color=_TEXT, ha="center", va="center")
    if trades:
        sub = "closed"
        if open_positions is not None:
            sub = sub + f" · {int(open_positions)} open"
    else:
        sub = "nothing closed"
    ax.text(0.6725, tr_sub_y, sub, fontsize=10, color=_MUTED,
            ha="center", va="center")

    # ------------------------------------------------------------ equity
    _tile(ax, 0.765, row_y, 0.215, row_h)
    _label(ax, 0.782, row_lab_y, "EQUITY")
    if equity is None:
        _unavailable(ax, 0.8725, eq_unavail_y, "account read n/a",
                     value_size=20)
    else:
        eq_s = f"{float(equity):,.2f}"
        ax.text(0.8725, eq_val_y, eq_s, fontsize=17, fontweight="bold",
                family=_MONO, color=_TEXT, ha="center", va="center")
        if balance is not None:
            bal = f"bal {float(balance):,.2f}"
            if currency:
                bal = bal + " " + currency
            ax.text(0.8725, eq_sub_y, bal, fontsize=9.5, color=_MUTED,
                    ha="center", va="center")

    # --------------------------------------- month target (owner-set)
    if has_target:
        _tile(ax, 0.02, 0.245 + target_dy, 0.96,
              0.065 if has_ticker else 0.07)
        lab = "MONTH TARGET"
        if month_label:
            lab = lab + " — " + month_label.upper()
        ax.text(0.045, 0.28 + target_dy, lab, fontsize=10.5,
                fontweight="bold", color=_MUTED, ha="left",
                va="center")
        if month_pnl is None or month_pct is None:
            ax.text(0.72, 0.28 + target_dy,
                    "— month-to-date P&L unavailable",
                    fontsize=10.5, color=_MUTED, ha="center",
                    va="center")
        else:
            _target_bar(ax, x0=0.315, x1=0.60, y=0.262 + target_dy,
                        h=0.022, pct=float(month_pct), tick_label="")
            pct_s = f"{float(month_pct):+.0f}%"
            ax.text(0.635, 0.28 + target_dy, pct_s, fontsize=13,
                    fontweight="bold", family=_MONO, color=_TEXT,
                    ha="left", va="center")
            amt = (f"{float(month_pnl):+,.2f} of "
                   f"{float(target):,.2f}")
            if currency:
                amt = amt + " " + currency
            ax.text(0.955, 0.28 + target_dy, amt, fontsize=10.5,
                    family=_MONO, color=_MUTED, ha="right",
                    va="center")

    # ------------------------------------- market ticker (runner-fed)
    if has_ticker:
        _tile(ax, 0.02, ticker_tile_y, 0.96, ticker_tile_h)
        ax.text(0.5, ticker_cy, ticker_line(ticker), fontsize=11.5,
                family=_MONO, color=_TEXT, ha="center", va="center")

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
