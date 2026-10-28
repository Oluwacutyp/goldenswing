"""Matplotlib chart engine — TradingView-style signal charts.

One public entry point::

    png: bytes = render(df, state, signal, style="classic")

``df`` is the full OHLCV frame the states were read from, ``state`` the
:class:`~goldenswing.reader.state.MarketState` at the signal bar and
``signal`` the :class:`~goldenswing.golden.detector.GoldenSignal` (may be
``None`` — the chart then shows the market read without trade levels).

Four styles ("chart classes") — plus the phone-first alert style:

- ``classic``     — TradingView-light candlesticks.
- ``heikin_ashi`` — Heikin-Ashi candles on the same light theme (levels
                    and zones still read off the real state).
- ``line``        — close line/area with full structure overlay.
- ``blueprint``   — dark study view: candles subdued, levels/zones at
                    full strength.  The "read the thesis" chart.
- ``signal``      — the ALERT chart, phone-first: portrait 1200x1500,
                    a big-type header band (direction / score / entry /
                    stop / targets — the trade, first), then clean
                    candles with ONLY the trade levels (entry zone,
                    entry, stop, targets, 2R).  No zones, pools, fibs,
                    trendlines, swing labels, killzone shading, or
                    annotation box — the thesis text already rides in
                    the alert caption; the chart must survive phone
                    width without zooming.

Determinism: fixed figure size/DPI, no wall-clock text anywhere in the
figure, pinned PNG metadata — the same (df, state, signal, style)
renders byte-identical output.

What gets drawn (when the state carries it): candles + volume panel,
swing markers with HH/HL/LH/LL labels, liquidity pools (dashed,
labelled BSL/SSL), FVG and order-block zones (shaded from creation bar),
premium/discount shading of the dealing range + EQ line, killzone
session shading, trendlines (Phase-4 family state), Bollinger bands and
the Ichimoku cloud (when the aligned FeatureSet history is passed via
``features``), Fib levels (when price is at/near them, always on
blueprint), entry zone + entry/stop/target lines, and the annotation
box carrying the full thesis: direction, score, entry/stop/targets and
the detector's backing-factor list.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..reader import session_timeframes as sessions  # Step 3: DST-aware authority (was fixed-UTC sessions)
from .signal_model import (  # Step 7: one shared signal-view model
    decollide_label_ys,
    panel_role,
    score_text,
)

STYLES = ("classic", "heikin_ashi", "line", "blueprint", "signal")

_FIGSIZE = (16, 10)          # at 100 dpi -> 1600x1000 px
_DPI = 100
_SIGNAL_FIGSIZE = (12, 15)   # phone-first alert chart -> 1200x1500 px

_THEMES = {
    "classic": dict(bg="#ffffff", grid="#e6e8ee", text="#131722",
                    up="#26a69a", down="#ef5350", candle_alpha=1.0),
    "heikin_ashi": dict(bg="#ffffff", grid="#e6e8ee", text="#131722",
                        up="#26a69a", down="#ef5350", candle_alpha=1.0),
    "line": dict(bg="#ffffff", grid="#e6e8ee", text="#131722",
                 up="#26a69a", down="#ef5350", candle_alpha=1.0),
    "blueprint": dict(bg="#131722", grid="#232a38", text="#d1d4dc",
                      up="#26a69a", down="#ef5350", candle_alpha=0.38),
    "signal": dict(bg="#ffffff", grid="#e6e8ee", text="#131722",
                   up="#1b7f75", down="#d43d3a", candle_alpha=1.0),
}


@dataclass(frozen=True)
class ForecastNote:
    """Optional ensemble-forecaster context for the annotation box."""

    prob_direction: float | None = None  # P(big move in signal direction)
    rank_pct: float | None = None        # percentile of that prob vs
                                         # history (0-100, higher =
                                         # forecaster likes it more)
    text: str | None = None

    def line(self) -> str | None:
        parts = []
        if self.text:
            parts.append(self.text)
        if self.prob_direction is not None:
            parts.append(f"P(move) {self.prob_direction:.2f}")
        if self.rank_pct is not None:
            top = max(1.0, 100.0 - self.rank_pct)
            parts.append(f"rank: top {top:.0f}% of bars")
        if not parts:
            return None
        return "Forecaster: " + " · ".join(parts)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi candles for a frame with open/high/low/close."""
    o, h, l, c = (df[k].to_numpy(dtype=float)
                  for k in ("open", "high", "low", "close"))
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    if len(ha_c):
        ha_o[0] = (o[0] + c[0]) / 2.0
        for i in range(1, len(ha_c)):
            ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    ha_h = np.maximum.reduce([h, ha_o, ha_c])
    ha_l = np.minimum.reduce([l, ha_o, ha_c])
    return pd.DataFrame(
        {"open": ha_o, "high": ha_h, "low": ha_l, "close": ha_c},
        index=df.index)


# ------------------------------------------------------------------ helpers
def _swing_labels(points, kind: str) -> list[tuple[int, float, str]]:
    """(index, price, label) for a swing series, labelled vs the
    previous swing of the same kind: HH/LH for highs, HL/LL lows."""
    out = []
    prev = None
    for p in sorted(points, key=lambda q: q.index):
        if prev is None:
            lab = "H" if kind == "high" else "L"
        elif kind == "high":
            lab = "HH" if p.price > prev else "LH"
        else:
            lab = "HL" if p.price > prev else "LL"
        out.append((p.index, p.price, lab))
        prev = p.price
    return out


def _fmt(x: float) -> str:
    return f"{x:,.1f}"


# ------------------------------------------------------------------- render
def render(df: pd.DataFrame, state, signal=None, *, style: str = "classic",
           window: int = 120, margin: int = 12, features=None,
           forecast: ForecastNote | None = None, instrument: str = "XAUUSD",
           timeframe: str = "H1", mode: str | None = None) -> bytes:
    """Render the signal chart to PNG bytes.

    The classic family renders 1600x1000; the ``signal`` style renders
    1200x1500 (phone-first portrait).  See the module docstring.

    ``features``: optional full list[FeatureSet] aligned 1:1 with ``df``
    — enables the Bollinger / Ichimoku series overlays (a single
    point-in-time FeatureSet only carries the current bar's values).
    """
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")
    if style == "signal":
        return _render_signal(df, state, signal, window=window,
                             margin=margin, instrument=instrument,
                             timeframe=timeframe, mode=mode)
    theme = _THEMES[style]
    sig_i = state.index
    start = max(0, sig_i - window + 1)
    end = sig_i + 1                       # bars drawn: [start, sig_i]
    x_lo, x_hi = start - 0.6, sig_i + margin + 0.4
    view = df.iloc[start:end]

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=_FIGSIZE, dpi=_DPI, sharex=True,
        gridspec_kw=dict(height_ratios=[4.2, 1.0], hspace=0.04))
    fig.patch.set_facecolor(theme["bg"])
    for a in (ax, axv):
        a.set_facecolor(theme["bg"])
        a.tick_params(colors=theme["text"], labelsize=8)
        for spine in a.spines.values():
            spine.set_color(theme["grid"])
        a.grid(True, color=theme["grid"], linewidth=0.5, alpha=0.7)
        a.set_xlim(x_lo, x_hi)

    ts = df.index
    # ---------- session shading (killzones) -------------------------------
    run_start = None
    for i in range(start, end):
        _name, kz = sessions.session_of(ts[i])
        if kz and run_start is None:
            run_start = i
        if (not kz or i == end - 1) and run_start is not None:
            run_end = i if not kz else i + 1
            ax.axvspan(run_start - 0.5, run_end - 0.5,
                       color="#f7b924", alpha=0.07, zorder=0)
            axv.axvspan(run_start - 0.5, run_end - 0.5,
                        color="#f7b924", alpha=0.07, zorder=0)
            run_start = None

    # ---------- premium / discount --------------------------------------
    rh, rl = state.range_high, state.range_low
    if np.isfinite(rh) and np.isfinite(rl) and rh > rl:
        eq = 0.5 * (rh + rl)
        ax.axhspan(eq, rh, color="#ef5350", alpha=0.045, zorder=0)
        ax.axhspan(rl, eq, color="#26a69a", alpha=0.045, zorder=0)
        ax.axhline(eq, color=theme["text"], linewidth=0.7,
                   linestyle=(0, (4, 3)), alpha=0.45)
        ax.text(x_lo + 0.4, rh, " PREMIUM", va="top", fontsize=7.5,
                color="#ef5350", alpha=0.8)
        ax.text(x_lo + 0.4, eq, " EQ", va="bottom", fontsize=7.5,
                color=theme["text"], alpha=0.7)
        ax.text(x_lo + 0.4, rl, " DISCOUNT", va="bottom", fontsize=7.5,
                color="#26a69a", alpha=0.8)

    o = view["open"].to_numpy(dtype=float)
    h = view["high"].to_numpy(dtype=float)
    l = view["low"].to_numpy(dtype=float)  # noqa: E741
    c = view["close"].to_numpy(dtype=float)
    xs = np.arange(start, end, dtype=float)

    # ---------- Ichimoku cloud + Bollinger (feature history) -------------
    if features is not None:
        fwin = features[start:end]
        ct = np.array([f.ichimoku.cloud_top for f in fwin], dtype=float)
        cb = np.array([f.ichimoku.cloud_bottom for f in fwin], dtype=float)
        ok = np.isfinite(ct) & np.isfinite(cb)
        if ok.sum() > 1:
            ax.fill_between(xs[ok], cb[ok], ct[ok],
                            where=ct[ok] >= cb[ok],
                            color="#26a69a", alpha=0.10, zorder=1,
                            interpolate=True)
            ax.fill_between(xs[ok], cb[ok], ct[ok],
                            where=ct[ok] < cb[ok],
                            color="#ef5350", alpha=0.10, zorder=1,
                            interpolate=True)
        bu = np.array([f.bollinger.upper for f in fwin], dtype=float)
        bm = np.array([f.bollinger.mid for f in fwin], dtype=float)
        bl = np.array([f.bollinger.lower for f in fwin], dtype=float)
        for arr, ls in ((bu, "-"), (bm, ":"), (bl, "-")):
            okb = np.isfinite(arr)
            if okb.sum() > 1:
                ax.plot(xs[okb], arr[okb], color="#5b8def", linewidth=0.8,
                        linestyle=ls, alpha=0.65, zorder=2)

    # ---------- zones: FVG + order blocks --------------------------------
    for g in state.active_fvgs:
        col = theme["up"] if g.direction == 1 else theme["down"]
        ax.add_patch(Rectangle(
            (max(g.created_at, start) - 0.5, g.bottom),
            x_hi - max(g.created_at, start), g.top - g.bottom,
            facecolor=col, alpha=0.16, edgecolor=col, linewidth=0.6,
            zorder=1))
        ax.text(x_hi - 0.3, g.midpoint, "FVG ", ha="right", va="center",
                fontsize=7.5, color=col)
    for ob in state.active_obs:
        col = theme["up"] if ob.direction == 1 else theme["down"]
        ax.add_patch(Rectangle(
            (max(ob.created_at, start) - 0.5, ob.bottom),
            x_hi - max(ob.created_at, start), ob.top - ob.bottom,
            facecolor=col, alpha=0.07, edgecolor=col, linewidth=1.1,
            zorder=1))
        ax.text(x_hi - 0.3, 0.5 * (ob.top + ob.bottom), "OB ", ha="right",
                va="center", fontsize=7.5, color=col)

    # ---------- candles / line -------------------------------------------
    if style == "line":
        ax.fill_between(xs, c, l.min(), color="#2962ff", alpha=0.07,
                        zorder=2)
        ax.plot(xs, c, color="#2962ff", linewidth=1.6, zorder=4)
    else:
        if style == "heikin_ashi":
            ha = heikin_ashi(view)
            bo, bh, bl, bc = (ha[k].to_numpy(dtype=float)
                              for k in ("open", "high", "low", "close"))
        else:
            bo, bh, bl, bc = o, h, l, c
        alpha = theme["candle_alpha"]
        for i in range(len(xs)):
            up = bc[i] >= bo[i]
            col = theme["up"] if up else theme["down"]
            ax.plot([xs[i], xs[i]], [bl[i], bh[i]], color=col,
                    linewidth=0.9, alpha=alpha, zorder=3,
                    solid_capstyle="round")
            lo_b, hi_b = min(bo[i], bc[i]), max(bo[i], bc[i])
            ax.add_patch(Rectangle(
                (xs[i] - 0.32, lo_b), 0.64, max(hi_b - lo_b, 1e-9),
                facecolor=col, edgecolor=col, alpha=alpha, zorder=3))

    # ---------- volume -----------------------------------------------------
    if "volume" in view.columns:
        vol = view["volume"].to_numpy(dtype=float)
        vcol = [theme["up"] if cc >= oo else theme["down"]
                for oo, cc in zip(o, c)]
        axv.bar(xs, vol, width=0.64, color=vcol,
                alpha=0.75 if style != "blueprint" else 0.45)
        axv.set_ylabel("Vol", fontsize=8, color=theme["text"])
        axv.set_ylim(0, (np.nanmax(vol) * 1.15) if len(vol) else 1.0)

    # ---------- swings --------------------------------------------------
    for pts, kind in ((state.recent_swing_highs, "high"),
                      (state.recent_swing_lows, "low")):
        for idx, price, lab in _swing_labels(pts, kind):
            if not (start <= idx <= sig_i):
                continue
            above = kind == "high"
            ax.plot([idx], [price], marker="v" if above else "^",
                    markersize=5, color=theme["text"], alpha=0.75, zorder=5)
            ax.annotate(lab, (idx, price),
                        xytext=(0, 7 if above else -13),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color=theme["text"], alpha=0.85,
                        zorder=5)

    # ---------- liquidity pools -----------------------------------------
    for p in state.pools_above:
        ax.axhline(p.price, color="#e08a00", linewidth=1.0,
                   linestyle=(0, (6, 3)), alpha=0.85, zorder=2)
        ax.text(x_lo + 0.4, p.price,
                f" BSL {_fmt(p.price)} x{p.strength}", va="bottom",
                fontsize=7.5, color="#e08a00")
    for p in state.pools_below:
        ax.axhline(p.price, color="#00838f", linewidth=1.0,
                   linestyle=(0, (6, 3)), alpha=0.85, zorder=2)
        ax.text(x_lo + 0.4, p.price,
                f" SSL {_fmt(p.price)} x{p.strength}", va="top",
                fontsize=7.5, color="#00838f")

    # ---------- trendlines ------------------------------------------------
    fs = getattr(state, "features", None)
    if fs is not None:
        for line in fs.trendlines.active:
            xa = max(start, line.anchor1_index)
            xb = sig_i + margin
            col = "#7e57c2" if line.direction == 1 else "#5c6bc0"
            ax.plot([xa, xb], [line.price_at(xa), line.price_at(xb)],
                    color=col, linewidth=1.2,
                    linestyle="-" if line.state == "ACTIVE" else (0, (5, 3)),
                    alpha=0.9, zorder=2)
        # ---------- Fib (when relevant; always on blueprint) -------------
        fib = fs.fib
        relevant = fib.at_fib or fib.in_ote_long or fib.in_ote_short
        if fib.levels and (relevant or style == "blueprint"):
            for frac, price in fib.levels:
                ax.axhline(price, color="#8d6e00", linewidth=0.6,
                           linestyle=(0, (2, 3)), alpha=0.55, zorder=1)
                ax.text(x_hi - 0.3, price, f"{frac:.3f} ", ha="right",
                        va="bottom", fontsize=7, color="#8d6e00")

    # ---------- trade levels ----------------------------------------------
    if signal is not None:
        d = signal.direction
        dir_col = theme["up"] if d == 1 else theme["down"]
        zlo, zhi = signal.entry_zone
        ax.add_patch(Rectangle(
            (sig_i - 1.5, zlo), margin + 2.0, zhi - zlo,
            facecolor="#f7b924", alpha=0.30, edgecolor="#f7b924",
            linewidth=0.8, zorder=2))
        ax.axhline(signal.entry_ref, color="#f7b924", linewidth=1.4,
                   zorder=4)
        ax.text(x_hi - 0.3, signal.entry_ref,
                f" ENTRY {_fmt(signal.entry_ref)}", ha="right",
                va="bottom" if d == 1 else "top", fontsize=8.5,
                color="#b8860b", fontweight="bold")
        ax.axhline(signal.invalidation, color="#d32f2f", linewidth=1.4,
                   zorder=4)
        ax.text(x_hi - 0.3, signal.invalidation, " STOP ", ha="right",
                va="top" if d == 1 else "bottom", fontsize=8.5,
                color="#d32f2f", fontweight="bold")
        risk = abs(signal.entry_ref - signal.invalidation)
        for j, t in enumerate(signal.targets):
            ax.axhline(t, color="#2e7d32", linewidth=1.1,
                       linestyle=(0, (6, 3)), zorder=3)
            ax.text(x_hi - 0.3, t, f" T{j + 1} {_fmt(t)}", ha="right",
                    va="bottom" if d == 1 else "top", fontsize=8.5,
                    color="#2e7d32")
        if risk > 0:
            t2r = signal.entry_ref + d * 2.0 * risk
            ax.axhline(t2r, color="#2e7d32", linewidth=0.8,
                       linestyle=(0, (2, 2)), alpha=0.7, zorder=3)
            ax.text(x_hi - 0.3, t2r, " 2R ", ha="right",
                    va="bottom" if d == 1 else "top", fontsize=8,
                    color="#2e7d32")
        ax.plot([sig_i], [signal.entry_ref], marker="o", markersize=7,
                color="#f7b924", markeredgecolor=theme["text"], zorder=6)

    # ---------- y limits (include visible levels, clamped) ---------------
    lo_v, hi_v = float(np.nanmin(l)), float(np.nanmax(h))
    span = max(hi_v - lo_v, 1e-9)
    cand = [lo_v, hi_v]
    if signal is not None:
        cand += [signal.invalidation, signal.entry_ref] + list(signal.targets)
    for v in cand:
        if lo_v - 1.6 * span <= v <= hi_v + 1.6 * span:
            lo_v, hi_v = min(lo_v, v), max(hi_v, v)
    pad = 0.07 * (hi_v - lo_v)
    ax.set_ylim(lo_v - pad, hi_v + pad)

    # ---------- annotation box --------------------------------------------
    lines = _annotation_lines(state, signal, forecast, instrument, mode)
    ax.text(0.008, 0.992, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8.6, family="monospace",
            color=theme["text"], zorder=7,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=theme["bg"],
                      edgecolor=theme["text"], alpha=0.88))

    # ---------- axes polish -------------------------------------------------
    ticks = np.linspace(start, sig_i, 6).astype(int)
    axv.set_xticks(ticks)
    axv.set_xticklabels(
        [ts[i].strftime("%m-%d %H:%M") if hasattr(ts[i], "strftime")
         else str(ts[i]) for i in ticks], fontsize=8)
    ax.set_ylabel("Price", fontsize=9, color=theme["text"])
    style_label = {"classic": "Candles", "heikin_ashi": "Heikin-Ashi",
                   "line": "Line + structure",
                   "blueprint": "Blueprint (study)",
                   "signal": "Signal (alert)"}[style]
    ts_txt = state.timestamp
    ts_txt = ts_txt.strftime("%Y-%m-%d %H:%M UTC") \
        if hasattr(ts_txt, "strftime") else str(ts_txt)
    ax.set_title(f"{instrument} · {timeframe} · {style_label} · {ts_txt}",
                 fontsize=11, color=theme["text"], loc="left")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=theme["bg"],
                metadata={"Software": "GoldenSwing charts"})
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------ signal
# The phone-first alert chart.  Hierarchy: the TRADE first (direction,
# score, entry, stop, targets as big type in a header band), then clean
# price action with only the trade's own levels.  The "why" (pools,
# zones, fibs, backing factors) rides in the alert caption text — this
# chart must read at phone width without zooming, so it draws nothing
# else.  Deterministic like the other styles: fixed size/DPI, no
# wall-clock text, pinned PNG metadata.
def _render_signal(df: pd.DataFrame, state, signal=None, *, window: int = 64,
                   margin: int = 12, instrument: str = "XAUUSD",
                   timeframe: str = "H1", mode: str | None = None) -> bytes:
    theme = _THEMES["signal"]
    sig_i = state.index
    start = max(0, sig_i - window + 1)
    x_lo, x_hi = start - 0.6, sig_i + margin + 0.4
    view = df.iloc[start:sig_i + 1]

    fig = plt.figure(figsize=_SIGNAL_FIGSIZE, dpi=_DPI)
    fig.patch.set_facecolor(theme["bg"])
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 8.0], hspace=0.03,
                          left=0.055, right=0.965, top=0.985, bottom=0.045)
    axh = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    for a in (axh, ax):
        a.set_facecolor(theme["bg"])
    axh.axis("off")

    d = signal.direction if signal is not None else 0
    dir_col = theme["up"] if d == 1 else theme["down"] if d == -1 \
        else theme["text"]
    # tinted header: direction readable before a single word is parsed
    axh.set_facecolor("#e9f6f3" if d == 1 else "#fdecea" if d == -1
                      else theme["bg"])

    ts = state.timestamp
    ts_txt = ts.strftime("%Y-%m-%d %H:%M UTC") if hasattr(ts, "strftime") \
        else str(ts)
    if signal is not None:
        arrow = "▲ LONG" if d == 1 else "▼ SHORT"
        golden = "  ★ GOLDEN" if signal.is_golden else ""
        axh.text(0.015, 0.88, f"{arrow} {instrument}{golden}",
                 transform=axh.transAxes, va="top", ha="left",
                 fontsize=24, fontweight="bold", color=dir_col)
        axh.text(0.985, 0.88, f"score {signal.score:.0f}",
                 transform=axh.transAxes, va="top", ha="right",
                 fontsize=20, fontweight="bold", color=theme["text"])
        zlo, zhi = signal.entry_zone
        risk = abs(signal.entry_ref - signal.invalidation)
        axh.text(0.015, 0.60,
                 f"ENTRY  {_fmt(signal.entry_ref)}",
                 transform=axh.transAxes, va="top", ha="left",
                 fontsize=19, fontweight="bold", color="#8a6d00")
        axh.text(0.52, 0.60, f"zone {_fmt(zlo)}–{_fmt(zhi)}",
                 transform=axh.transAxes, va="top", ha="left",
                 fontsize=14, color=theme["text"])
        risk_txt = f"STOP  {_fmt(signal.invalidation)}"
        if signal.atr > 0 and risk > 0:
            risk_txt += f"    risk {risk:.1f} ({risk / signal.atr:.2f}× ATR)"
        axh.text(0.015, 0.36, risk_txt,
                 transform=axh.transAxes, va="top", ha="left",
                 fontsize=17, fontweight="bold", color=theme["down"])
        if signal.targets:
            tg = "    ".join(
                f"T{j + 1} {_fmt(t)}"
                for j, t in enumerate(signal.targets))
            axh.text(0.015, 0.13, tg,
                     transform=axh.transAxes, va="top", ha="left",
                     fontsize=15, fontweight="bold", color=theme["up"])
        axh.text(0.985, 0.06, f"{signal.kind} · {ts_txt}",
                 transform=axh.transAxes, va="bottom", ha="right",
                 fontsize=11, color=theme["text"], alpha=0.75)
    else:
        axh.text(0.015, 0.80, f"{instrument} · {timeframe}",
                 transform=axh.transAxes, va="top", ha="left",
                 fontsize=22, fontweight="bold", color=theme["text"])
        axh.text(0.015, 0.45, f"market read · {ts_txt}",
                 transform=axh.transAxes, va="top", ha="left",
                 fontsize=14, color=theme["text"], alpha=0.8)

    # ---------- price: candles + the trade's own levels, nothing else ---
    ax.tick_params(colors=theme["text"], labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(theme["grid"])
    ax.grid(True, color=theme["grid"], linewidth=0.6, alpha=0.7)
    ax.set_xlim(x_lo, x_hi)

    o = view["open"].to_numpy(dtype=float)
    h = view["high"].to_numpy(dtype=float)
    l = view["low"].to_numpy(dtype=float)  # noqa: E741
    c = view["close"].to_numpy(dtype=float)
    xs = np.arange(start, sig_i + 1, dtype=float)
    for i in range(len(xs)):
        up = c[i] >= o[i]
        col = theme["up"] if up else theme["down"]
        ax.plot([xs[i], xs[i]], [l[i], h[i]], color=col, linewidth=1.7,
                zorder=3, solid_capstyle="round")
        lo_b, hi_b = min(o[i], c[i]), max(o[i], c[i])
        ax.add_patch(Rectangle(
            (xs[i] - 0.36, lo_b), 0.72, max(hi_b - lo_b, 1e-9),
            facecolor=col, edgecolor=col, zorder=3))

    if signal is not None:
        zlo, zhi = signal.entry_zone
        ax.add_patch(Rectangle(
            (sig_i - 1.5, zlo), margin + 2.0, zhi - zlo,
            facecolor="#f7b924", alpha=0.28, edgecolor="#f7b924",
            linewidth=1.0, zorder=2))
        ax.axhline(signal.entry_ref, color="#c99700", linewidth=2.6,
                   zorder=4)
        ax.text(x_hi - 0.3, signal.entry_ref,
                f" ENTRY {_fmt(signal.entry_ref)}", ha="right",
                va="bottom" if d == 1 else "top", fontsize=13,
                fontweight="bold", color="#8a6d00",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff8e1",
                          edgecolor="#c99700", alpha=0.95), zorder=6)
        ax.axhline(signal.invalidation, color=theme["down"], linewidth=2.6,
                   zorder=4)
        ax.text(x_hi - 0.3, signal.invalidation, " STOP ", ha="right",
                va="top" if d == 1 else "bottom", fontsize=13,
                fontweight="bold", color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=theme["down"], edgecolor="none",
                          alpha=0.95), zorder=6)
        risk = abs(signal.entry_ref - signal.invalidation)
        # target labels: merge ones that would collide into a single line
        tg_sorted = sorted(signal.targets)
        groups: list[list[tuple[int, float]]] = []
        min_sep = 0.30 * risk if risk > 0 else 0.0
        for j, t in enumerate(tg_sorted):
            if groups and min_sep > 0 and t - groups[-1][-1][1] < min_sep:
                groups[-1].append((j, t))
            else:
                groups.append([(j, t)])
        # recover original T-numbers: targets were enumerated in signal order
        order = {t: j for j, t in enumerate(signal.targets)}
        for grp in groups:
            mid = sum(t for _, t in grp) / len(grp)
            txt = " · ".join(
                f"T{order[t] + 1} {_fmt(t)}"
                for _, t in sorted(grp, key=lambda p: order[p[1]]))
            for _, t in grp:
                ax.axhline(t, color=theme["up"], linewidth=1.8,
                           linestyle=(0, (7, 3)), zorder=3)
            ax.text(x_hi - 0.3, mid, f" {txt}", ha="right",
                    va="center", fontsize=12.5,
                    fontweight="bold", color=theme["up"], zorder=6)
        if risk > 0:
            t2r = signal.entry_ref + d * 2.0 * risk
            ax.axhline(t2r, color=theme["up"], linewidth=1.0,
                       linestyle=(0, (2, 2)), alpha=0.75, zorder=3)
            # skip the 2R tag when a target line sits on top of it
            if not any(abs(t2r - t) < 0.25 * risk for t in signal.targets):
                ax.text(x_hi - 0.3, t2r, " 2R ", ha="right",
                        va="bottom" if d == 1 else "top", fontsize=11,
                        color=theme["up"], zorder=6)
        ax.plot([sig_i], [signal.entry_ref], marker="o", markersize=10,
                color="#f7b924", markeredgecolor=theme["text"],
                markeredgewidth=1.2, zorder=7)
        lo_v, hi_v = float(np.nanmin(l)), float(np.nanmax(h))
        for v in [signal.invalidation, signal.entry_ref,
                  *signal.targets]:
            if lo_v - 1.2 * (hi_v - lo_v) <= v <= hi_v + 1.2 * (hi_v - lo_v):
                lo_v, hi_v = min(lo_v, v), max(hi_v, v)
    else:
        lo_v, hi_v = float(np.nanmin(l)), float(np.nanmax(h))
    pad = 0.06 * max(hi_v - lo_v, 1e-9)
    ax.set_ylim(lo_v - pad, hi_v + pad)

    ticks = np.linspace(start, sig_i, 5).astype(int)
    ts_idx = df.index
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [ts_idx[i].strftime("%m-%d %H:%M") if hasattr(ts_idx[i], "strftime")
         else str(ts_idx[i]) for i in ticks], fontsize=10.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=theme["bg"],
                metadata={"Software": "GoldenSwing charts"})
    plt.close(fig)
    return buf.getvalue()


def _annotation_lines(state, signal, forecast, instrument, mode) -> list[str]:
    ts = state.timestamp
    ts_txt = ts.strftime("%Y-%m-%d %H:%M UTC") if hasattr(ts, "strftime") \
        else str(ts)
    if signal is None:
        return [f"{instrument} — market read @ {ts_txt}",
                f"trend {state.trend} · {state.vol_regime} vol · "
                f"{state.session}"]
    d = signal.direction
    arrow = "LONG ▲" if d == 1 else "SHORT ▼"
    golden = " · ★ GOLDEN" if signal.is_golden else ""
    lines = [
        f"{arrow} {instrument} — score {signal.score:.0f}{golden}",
        f"{signal.kind}" + (f" · mode: {mode}" if mode else "")
        + f" · {ts_txt}",
        f"Entry {_fmt(signal.entry_ref)} "
        f"(zone {_fmt(signal.entry_zone[0])}–{_fmt(signal.entry_zone[1])})",
        f"Stop  {_fmt(signal.invalidation)}",
    ]
    if signal.targets:
        lines.append("Targets " + " · ".join(
            f"T{j + 1} {_fmt(t)}" for j, t in enumerate(signal.targets)))
    risk = abs(signal.entry_ref - signal.invalidation)
    if signal.atr > 0:
        lines.append(f"Risk dist {risk:.1f} ({risk / signal.atr:.2f}x ATR)")
    lines.append("Backing:")
    for f in signal.factors[:8]:
        lines.append(f" • {f}")
    if len(signal.factors) > 8:
        lines.append(f" • … +{len(signal.factors) - 8} more")
    if forecast is not None:
        fl = forecast.line()
        if fl:
            lines.append(fl)
    return lines


def render_to_file(path, df, state, signal=None, **kwargs) -> str:
    """Render and write the PNG to ``path``; returns the path."""
    data = render(df, state, signal, **kwargs)
    with open(path, "wb") as fh:
        fh.write(data)
    return str(path)


# ------------------------------------------------------------------ MT5
# Multi-timeframe signal chart for the MT5 engine: the WHOLE hierarchy
# stack, never H1-only.  Panels, top to bottom:
#   H4  — structure: H4 candles + H4 POI zone band + D1 bias note.
#   H1  — trigger: H1 candles + trigger bar marked (BOS/MSCH/CHoCH).
#   M15/M30 — entry: entry-clock candles + entry bar marked.
# Every panel shares the same entry / stop / targets / 2R / fill lines,
# so the trade reads identically at every timeframe.  A missing frame
# renders its panel with an explicit "unavailable" label — bars are
# never invented.  Every level drawn comes from the caller.
_MT5_STYLES = ("classic", "blueprint")


def _mt5_candles(ax, view, xs, theme) -> None:
    """Candlesticks for one panel's window view."""
    o = view["open"].to_numpy(dtype=float)
    h = view["high"].to_numpy(dtype=float)
    l = view["low"].to_numpy(dtype=float)  # noqa: E741
    c = view["close"].to_numpy(dtype=float)
    for i in range(len(xs)):
        up = c[i] >= o[i]
        col = theme["up"] if up else theme["down"]
        ax.plot([xs[i], xs[i]], [l[i], h[i]], color=col, linewidth=0.9,
                zorder=3, solid_capstyle="round")
        lo_b, hi_b = min(o[i], c[i]), max(o[i], c[i])
        ax.add_patch(Rectangle(
            (xs[i] - 0.32, lo_b), 0.64, max(hi_b - lo_b, 1e-9),
            facecolor=col, edgecolor=col, zorder=3))


def _mt5_trade_levels(ax, x_hi, *, entry: float, stop: float,
                      broker_stop: float | None,
                      targets: list[float], direction: int, risk: float,
                      fill_price: float | None, theme) -> list[dict]:
    """Shared entry / stops / targets / 2R / fill LINES on one panel.

    Draws the lines now and returns the label specs. Labels are drawn
    later (``_mt5_edge_labels``), after a canvas draw finalizes axes
    positions, so de-collision works in real pixels — entry and stop
    a couple of points apart no longer stack into one illegible blob
    at the right edge. ``stop`` is the structural invalidation;
    ``broker_stop`` is the broker stop when one is known, else None.
    """
    specs: list[dict] = []

    def _line(value, color, width, style, label, bold=False, size=9):
        ax.axhline(value, color=color, linewidth=width,
                   linestyle=style, zorder=4 if bold else 3)
        specs.append({"value": float(value), "text": label,
                      "color": color, "bold": bold, "size": size})

    _line(entry, "#f7b924", 1.6, "-", f" ENTRY {_fmt(entry)}", bold=True)
    _line(stop, "#d32f2f", 1.6, "-",
          f" STOP {_fmt(stop)} (structural)", bold=True)
    if broker_stop is not None:
        _line(broker_stop, "#ff7043", 1.6, "-",
              f" BROKER SL {_fmt(broker_stop)}", bold=True)
    for j, t in enumerate(targets):
        _line(t, "#2e7d32", 1.1, (0, (6, 3)), f" T{j + 1} {_fmt(t)}")
    if risk > 0:
        _line(entry + direction * 2.0 * risk, "#2e7d32", 0.8,
              (0, (2, 2)), " 2R ", size=8)
    if fill_price is not None:
        _line(fill_price, "#7e57c2", 1.4, (0, (8, 3)),
              f" FILL {_fmt(fill_price)}", bold=True)
    return specs


def _mt5_edge_labels(ax, x_hi, specs: list[dict]) -> None:
    """Draw right-edge level labels, de-collided in display pixels.

    Call only after a canvas draw: ``transData`` needs finalized
    axes positions. This is the ONE right-edge label pass: trade
    levels, nearby-zone levels, Fib levels and the OTE label all
    arrive here as specs, so no label set can land on another —
    the step-7 pass covered trade labels only, and study labels
    drawn raw at the same edge still piled into the trade labels
    (the 2026-10-28 qm_m5 chart). Labels keep their line's color
    and value; a shifted label gets a short vertical leader back
    to its line.
    """
    if not specs:
        return
    orig = [ax.transData.transform((0, s["value"]))[1] for s in specs]
    bbox = ax.bbox
    adj = decollide_label_ys(orig, min_gap=18.0,
                             top=bbox.y1 - 4, bottom=bbox.y0 + 4)
    for s, o_px, a_px in zip(specs, orig, adj):
        a_val = ax.transData.inverted().transform((0, a_px))[1]
        if abs(a_px - o_px) > 1.0:
            ax.plot([x_hi - 0.3, x_hi - 0.3], [s["value"], a_val],
                    color=s["color"], linewidth=0.7, zorder=5)
        ax.text(x_hi - 0.3, a_val, s["text"], ha="right", va="center",
                fontsize=s["size"], color=s["color"],
                fontweight="bold" if s["bold"] else "normal", zorder=6)


def _mt5_panel_ylim(view, *, entry: float, stop: float,
                    broker_stop: float | None = None,
                    targets: list[float], fill_price: float | None,
                    zone: dict | None) -> tuple[float, float]:
    """Y limits from the panel's own candles + the shared levels."""
    l = view["low"].to_numpy(dtype=float)
    h = view["high"].to_numpy(dtype=float)
    lo_v, hi_v = float(np.nanmin(l)), float(np.nanmax(h))
    span = max(hi_v - lo_v, 1e-9)
    cand = [lo_v, hi_v, stop, entry] + list(targets)
    if broker_stop is not None:
        cand.append(broker_stop)
    if fill_price is not None:
        cand.append(fill_price)
    if zone is not None and zone.get("top") and zone.get("bottom"):
        cand += [float(zone["top"]), float(zone["bottom"])]
    for v in cand:
        if lo_v - 1.6 * span <= v <= hi_v + 1.6 * span:
            lo_v, hi_v = min(lo_v, v), max(hi_v, v)
    pad = 0.07 * max(hi_v - lo_v, 1e-9)
    return lo_v - pad, hi_v + pad


def zones_to_study_dicts(zones, entry: float, signal_zone, cap: int = 8) -> list[dict]:
    """ZoneCache zones -> chart-ready dicts, nearest-first, capped.

    Zones whose level falls inside the signal zone's band are skipped
    (the band already shows that ground). Only dicts come out, so the
    renderer never imports the cache's types.
    """
    sig_lo = sig_hi = 0.0
    if signal_zone:
        try:
            sig_lo = float(signal_zone.get("bottom", 0) or 0)
            sig_hi = float(signal_zone.get("top", 0) or 0)
        except Exception:
            sig_lo = sig_hi = 0.0
    out = []
    for z in zones or []:
        try:
            lvl = float(getattr(z, "level", 0) or 0)
        except Exception:
            continue
        if not lvl:
            continue
        if sig_lo and sig_hi and sig_lo <= lvl <= sig_hi:
            continue
        out.append({"level": lvl,
                    "tier": str(getattr(z, "tier", "") or ""),
                    "side": str(getattr(z, "side", "") or "")})
    out.sort(key=lambda d: abs(d["level"] - entry))
    return out[:cap]


def _mt5_sessions(ax, view, theme) -> None:
    """Killzone shading for one panel, from its own bar timestamps."""
    ts = view.index
    run_start = None
    n = len(view)
    for i in range(n):
        try:
            _name, kz = sessions.session_of(ts[i])
        except Exception:
            continue
        if kz and run_start is None:
            run_start = i
        if (not kz or i == n - 1) and run_start is not None:
            run_end = i if not kz else i + 1
            ax.axvspan(run_start - 0.5, run_end - 0.5,
                       color="#f7b924", alpha=0.07, zorder=0)
            run_start = None


def _mt5_premium_discount(ax, x_lo, x_hi, range_low, range_high, theme) -> None:
    """Dealing-range premium/discount shading + equilibrium line."""
    eq = 0.5 * (range_low + range_high)
    ax.axhspan(eq, range_high, color="#ef5350", alpha=0.045, zorder=0)
    ax.axhspan(range_low, eq, color="#26a69a", alpha=0.045, zorder=0)
    ax.axhline(eq, color=theme["text"], linewidth=0.7,
               linestyle=(0, (4, 3)), alpha=0.45, zorder=1)
    ax.text(x_lo + 0.4, range_high, " PREMIUM", va="top", fontsize=7.5,
            color="#ef5350", alpha=0.8)
    ax.text(x_lo + 0.4, eq, " EQ", va="bottom", fontsize=7.5,
            color=theme["text"], alpha=0.7)
    ax.text(x_lo + 0.4, range_low, " DISCOUNT", va="bottom", fontsize=7.5,
            color="#26a69a", alpha=0.8)


def _mt5_nearby_lines(ax, x_hi, nearby_zones, theme) -> list[dict]:
    """Cache zone levels as horizontal lines (MAJOR solid, rest dashed).

    Lines are drawn now; labels come back as specs for the shared
    post-draw de-collision pass (``_mt5_edge_labels``) — a zone label
    drawn raw at the right edge lands on top of a trade label
    whenever the zone sits near the entry, which is the common case.
    """
    specs: list[dict] = []
    for z in nearby_zones or []:
        lvl = float(z["level"])
        side = z.get("side", "")
        tier = z.get("tier", "")
        col = (theme["up"] if side == "support"
               else theme["down"] if side == "resistance"
               else theme["text"])
        major = tier == "MAJOR"
        ax.axhline(lvl, color=col, linewidth=1.0 if major else 0.6,
                   linestyle="-" if major else (0, (6, 3)),
                   alpha=0.8, zorder=2)
        specs.append({"value": lvl, "text": f" {tier} {lvl:.1f} ",
                      "color": col, "bold": False, "size": 7.5})
    return specs


def _mt5_fib(ax, x_hi, fib_levels, ote_band, theme) -> list[dict]:
    """Auto-Fib retracement lines + the OTE band, on the H1 panel.

    Lines and the OTE span are drawn now; labels come back as specs
    for the shared post-draw de-collision pass (``_mt5_edge_labels``),
    same reason as ``_mt5_nearby_lines``: raw right-edge labels
    collide with the trade labels they sit next to.
    """
    specs: list[dict] = []
    for frac, price in fib_levels or []:
        ax.axhline(price, color="#8d6e00", linewidth=0.6,
                   linestyle=(0, (4, 3)), zorder=2)
        specs.append({"value": float(price), "text": f" {frac:.3f} ",
                      "color": "#8d6e00", "bold": False, "size": 7.5})
    if ote_band is not None:
        lo, hi = ote_band
        if hi > lo:
            ax.axhspan(lo, hi, color="#8d6e00", alpha=0.10, zorder=1)
            specs.append({"value": float(hi), "text": " OTE ",
                          "color": "#8d6e00", "bold": True, "size": 7.5})
    return specs


def render_mt5_signal(h1: pd.DataFrame, *, zone: dict | None,
                       entry: float, stop: float,
                       targets: list[float] | None = None,
                       direction: int = 0, verdict: dict | None = None,
                       playbook: str = "", score: float | None = None,
                       decision_score: float | None = None,
                       broker_stop: float | None = None,
                       instrument: str = "XAUUSD",
                       fill_price: float | None = None,
                       window: int = 120,
                       h4: pd.DataFrame | None = None,
                       entry_frame: pd.DataFrame | None = None,
                       entry_clock: str | None = None,
                       style: str = "classic",
                       h4_window: int = 60,
                       entry_window: int = 60,
                       nearby_zones: list[dict] | None = None,
                       fib_levels: list[tuple[float, float]] | None = None,
                       fib_range: tuple[float, float] | None = None,
                       ote_band: tuple[float, float] | None = None) -> bytes:
    """Render the MT5 multi-timeframe signal chart to PNG bytes.

    Three stacked panels (H4 structure / H1 trigger / entry-TF entry),
    all sharing the same trade levels.  ``h1`` is required; ``h4`` and
    ``entry_frame`` are optional — a missing frame renders its panel
    with an explicit "unavailable" label, never with invented bars.
    ``entry_clock`` defaults to the verdict's entry clock (M15/M30).
    ``style`` is ``"classic"`` or ``"blueprint"`` (the two MT5 chart
    classes).  Every level drawn comes from the caller — nothing is
    inferred, estimated, or defaulted.  Step 7: ``decision_score``
    (the score recorded for the signal) is what the page shows;
    ``score`` is only the legacy fallback for old callers, and no
    score at all renders as "Score unavailable", never a placeholder.
    ``broker_stop`` is the broker stop when one is known (fill /
    pending charts); signal charts pass None and the page says so.
    Panel roles come from the route + entry clock (see
    ``charts.signal_model``): on native routes H1 is context, not
    the trigger.

    Studies (at most four, all caller-supplied except sessions,
    which shade the panels' own bar timestamps):
    - signal zone band (H4 panel) + ``nearby_zones`` cache levels
      (all panels);
    - ``fib_levels`` auto-Fib retracements + ``ote_band`` (H1 panel),
      computed by the engine's feature pipeline on the H1 frame;
    - ``fib_range`` dealing-range premium/discount shading (H1 panel);
    - killzone session shading (all panels, from bar timestamps).
    """
    if style not in _MT5_STYLES:
        raise ValueError(
            f"unknown MT5 chart style {style!r}; expected one of "
            f"{_MT5_STYLES}")
    theme = _THEMES[style]
    if h1 is None or len(h1) == 0:
        raise ValueError("render_mt5_signal needs a non-empty H1 frame")
    targets = list(targets or [])
    risk = (abs(entry - broker_stop) if broker_stop is not None
            else abs(entry - stop))
    shown_score = score_text(decision_score, fallback=score)
    clock = entry_clock or (verdict or {}).get("entry_clock") or "M15"
    bias_d1 = (verdict or {}).get("bias_d1", 0)
    bias_h4 = (verdict or {}).get("bias_h4", 0)
    trigger = (verdict or {}).get("trigger") or "?"
    arrow = "▲ LONG" if direction == 1 else "▼ SHORT" if direction == -1 \
        else "FLAT"

    fig = plt.figure(figsize=(16, 13), dpi=_DPI)
    fig.patch.set_facecolor(theme["bg"])
    fig.suptitle(f"{instrument} · MT5 signal · {playbook} · "
                 f"{shown_score}",
                 fontsize=13, color=theme["text"], x=0.02, ha="left")
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.2, 1.8],
                          hspace=0.45, left=0.05, right=0.97,
                          top=0.93, bottom=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    h1_role = panel_role("H1", playbook=playbook, entry_clock=clock)
    h1_title = f"H1 — {h1_role}"
    if trigger != "?":
        h1_title += f": {trigger}"
    panels = [
        # (frame, window, title, zone band?, marker label, is H1 panel?)
        (h4, h4_window,
         f"H4 — structure · D1 bias {bias_d1:+d} · H4 bias {bias_h4:+d}",
         True, None, False),
        (h1, window, h1_title, False,
         f"TRIGGER {trigger}" if trigger != "?" else None, True),
        (entry_frame, entry_window, f"{clock} — entry", False, "ENTRY",
         False),
    ]
    verdict_box_done = False
    label_specs: list[tuple] = []
    for ax, (frame, win, title, show_zone, marker, is_h1) in zip(axes, panels):
        ax.set_facecolor(theme["bg"])
        ax.tick_params(colors=theme["text"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(theme["grid"])
        ax.grid(True, color=theme["grid"], linewidth=0.5, alpha=0.7)
        ax.set_title(title, fontsize=11, color=theme["text"], loc="left")
        if frame is None or len(frame) == 0:
            # Missing frame: label the gap, never invent bars.
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 1)
            ax.text(5, 0.5, f"{title.split(' —')[0]} frame unavailable — "
                    "panel not charted (no bars invented)",
                    ha="center", va="center", fontsize=10,
                    color=theme["text"], alpha=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        view = frame.iloc[-win:]
        xs = np.arange(len(view), dtype=float)
        x_lo, x_hi = -0.6, len(view) - 1 + 12.4
        ax.set_xlim(x_lo, x_hi)
        _mt5_candles(ax, view, xs, theme)
        _mt5_sessions(ax, view, theme)
        if show_zone and zone is not None:
            ztop = float(zone.get("top", 0) or 0)
            zbot = float(zone.get("bottom", 0) or 0)
            if ztop > zbot > 0:
                ax.axhspan(zbot, ztop, color="#f7b924", alpha=0.25,
                           zorder=1)
                ax.axhline(ztop, color="#b8860b", linewidth=0.9,
                           linestyle=(0, (4, 3)), zorder=2)
                ax.axhline(zbot, color="#b8860b", linewidth=0.9,
                           linestyle=(0, (4, 3)), zorder=2)
                ax.text(x_lo + 0.4, ztop,
                        f" H4 ZONE {zone.get('kind', '')} "
                        f"{_fmt(zbot)}–{_fmt(ztop)}",
                        va="bottom", fontsize=8.5, color="#8a6d00",
                        fontweight="bold")
        study_specs = _mt5_nearby_lines(ax, x_hi, nearby_zones, theme)
        if is_h1:
            if fib_range is not None:
                _mt5_premium_discount(ax, x_lo, x_hi, fib_range[0],
                                      fib_range[1], theme)
            study_specs += _mt5_fib(ax, x_hi, fib_levels, ote_band,
                                    theme)
        lo_v, hi_v = _mt5_panel_ylim(
            view, entry=entry, stop=stop, broker_stop=broker_stop,
            targets=targets, fill_price=fill_price,
            zone=zone if show_zone else None)
        ax.set_ylim(lo_v, hi_v)
        label_specs.append((ax, x_hi, _mt5_trade_levels(
            ax, x_hi, entry=entry, stop=stop, broker_stop=broker_stop,
            targets=targets, direction=direction, risk=risk,
            fill_price=fill_price, theme=theme) + study_specs))
        if marker is not None:
            last = len(view) - 1
            ax.axvspan(last - 0.5, last + 0.5, color="#7e57c2",
                       alpha=0.12, zorder=1)
            ax.text(last, hi_v - 0.02 * (hi_v - lo_v), f" {marker} ",
                    ha="center", va="top", fontsize=8.5,
                    color="#ffffff", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#7e57c2", edgecolor="none",
                              alpha=0.95), zorder=6)
            if fill_price is not None:
                ax.plot([last], [fill_price], marker="o", markersize=7,
                        color="#7e57c2",
                        markeredgecolor=theme["text"], zorder=6)
        # Verdict box on the first available panel (H4 preferred).
        if not verdict_box_done:
            vlines = [f"{arrow} {instrument} — {playbook} · "
                      f"{shown_score}"]
            if verdict is not None:
                vlines.append(
                    f"D1 {bias_d1:+d} / H4 {bias_h4:+d}"
                    + (f" / {trigger}" if trigger != "?" else "")
                    + f" / {clock}")
                vparts = []
                amd = verdict.get("amd_phase")
                if amd not in (None, "?"):
                    vparts.append(f"AMD: {amd}")
                qm_status = (verdict.get("qm") or {}).get("status")
                if qm_status not in (None, "?"):
                    vparts.append(f"QM: {qm_status}")
                vol = (verdict.get("volume") or {}).get("volume_ratio")
                if vol not in (None, "?"):
                    vparts.append(f"vol: {vol}")
                if vparts:
                    vlines.append(" · ".join(vparts))
            stop_line = (f"Entry {_fmt(entry)} · Stop {_fmt(stop)} "
                         "(structural)")
            if broker_stop is not None:
                stop_line += f" · Broker SL {_fmt(broker_stop)}"
            else:
                stop_line += " · Broker SL pending at placement"
            if fill_price is not None:
                stop_line += f" · Fill {_fmt(fill_price)}"
            vlines.append(stop_line)
            ax.text(0.008, 0.992, "\n".join(vlines),
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=8.6, family="monospace",
                    color=theme["text"], zorder=7,
                    bbox=dict(boxstyle="round,pad=0.45",
                              facecolor=theme["bg"],
                              edgecolor=theme["text"], alpha=0.88))
            verdict_box_done = True
        ticks = np.linspace(0, len(view) - 1, 5).astype(int)
        ts = frame.index[-len(view):]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [ts[i].strftime("%m-%d %H:%M")
             if hasattr(ts[i], "strftime") else str(ts[i])
             for i in ticks], fontsize=8)
        ax.set_ylabel("Price", fontsize=9, color=theme["text"])

    # Labels last: a canvas draw finalizes axes positions, so the
    # de-collision pass measures real pixels, not estimates.
    fig.canvas.draw()
    for ax, x_hi_panel, specs in label_specs:
        _mt5_edge_labels(ax, x_hi_panel, specs)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=theme["bg"],
                metadata={"Software": "GoldenSwing charts"})
    plt.close(fig)
    return buf.getvalue()
