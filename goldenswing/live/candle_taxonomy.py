"""Named candlestick taxonomy — the standard named patterns as
first-class, causal bar-geometry reads.

The Phase 2 capability census recorded this capability as
ABSENT: the engine measured candle geometry (body/wick ratios in
live/confirmation.py, QM pattern tags) but no detector anywhere
named a hammer, a doji or an engulfing-by-name.  This module is
that detector.  It is an EVIDENCE source: the census-wiring
consumer (live/census_evidence.py) records its reads on live
evaluation records; nothing here gates, scores or decides.

Definitions are the textbook ones, built where possible on the
repo's own canonical geometry (experiments/candles.py — the
context-map study the definitions were mined for):

- **doji** — repo ``is_doji``: body <= 10% of range on a range
  >= 0.5 ATR (indecision).  Subtypes replace the plain name when
  they qualify (one name per bar-shape, never stacked):
  **dragonfly_doji** (lower wick >= 60% of range, upper <= 15% —
  rejection of lows, +1), **gravestone_doji** (mirror, -1),
  **long_legged_doji** (both wicks >= 30% of range, 0).
- **pin family** — repo ``pin_bar`` geometry (dominant wick
  >= 2x body and >= 50% of range), named by the causal trend
  context of the bars BEFORE the pattern bar (net close move
  over the prior 3 bars): lower-wick rejection after a decline
  = **hammer** (+1); the same shape after an advance =
  **hanging_man** (-1); upper-wick rejection after an advance
  = **shooting_star** (-1); after a decline =
  **inverted_hammer** (+1); with no clear prior trend the
  generic **pin_bar_bullish** / **pin_bar_bearish** names are
  used, so every pin event carries exactly one name.
- **bullish_engulfing / bearish_engulfing** — repo ``engulfing``
  (bar body fully engulfs the prior body, opposite colours),
  named by its sign.
- **piercing_line** (+1) / **dark_cloud_cover** (-1) — two-bar:
  a strong prior body; the pattern bar opens beyond the prior
  CLOSE and closes back past the prior body's midpoint without
  fully engulfing it.  (Gap conditions are stated against the
  prior close, the standard continuous-market adaptation —
  24h FX/CFD bars rarely gap beyond the prior extreme.)
- **morning_star** (+1) / **evening_star** (-1) — three-bar: a
  strong first body, a small-bodied star bar closing beyond
  the first bar's close, and a confirming bar closing past
  the first body's midpoint.

Causality: a pattern completing at bar ``t`` is a function of
bars <= t only (trend context uses bars < t).  The truncation
property is test-pinned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..experiments.candles import engulfing as _engulfing
from ..experiments.candles import is_doji as _is_doji
from ..experiments.candles import pin_bar as _pin_bar
from ..reader.indicators import atr as _atr_series

#: Every name this detector can emit (the census-named set:
#: hammer / doji family / engulfing-by-name, plus the standard
#: single/two/three-bar canon built from the same geometry).
PATTERN_NAMES = (
    "doji", "dragonfly_doji", "gravestone_doji",
    "long_legged_doji",
    "hammer", "hanging_man", "inverted_hammer", "shooting_star",
    "pin_bar_bullish", "pin_bar_bearish",
    "bullish_engulfing", "bearish_engulfing",
    "piercing_line", "dark_cloud_cover",
    "morning_star", "evening_star",
)

# Doji subtype wick shares (fractions of the bar's range).
_DRAGONFLY_LOWER_MIN = 0.60
_DRAGONFLY_UPPER_MAX = 0.15
_LONG_LEGGED_MIN = 0.30
# A "strong" body for piercing / dark-cloud / star first bars:
# at least this share of the bar's own range.
_STRONG_BODY_FRAC = 0.50
# Star bar: body at most this share of its own range.
_STAR_BODY_FRAC = 0.30


def _trend_before(c: np.ndarray, i: int) -> int:
    """Causal trend context from bars strictly before bar ``i``:
    the net close move over the prior 3 bars.  +1 advance,
    -1 decline, 0 no clear trend (or not enough history)."""
    if i < 4:
        return 0
    if c[i - 1] > c[i - 4]:
        return 1
    if c[i - 1] < c[i - 4]:
        return -1
    return 0


def patterns_at(o, h, l, c, atr_v: float, i: int) -> list[tuple[str, int]]:
    """Every named pattern COMPLETING at bar ``i`` (a function
    of bars <= i only).  Returns [(name, direction)] with
    direction +1 bullish / -1 bearish / 0 indecision."""
    out: list[tuple[str, int]] = []
    if i < 1:
        return out
    rng = h[i] - l[i]
    if rng <= 0:
        return out
    body = c[i] - o[i]
    upper = h[i] - max(o[i], c[i])
    lower = min(o[i], c[i]) - l[i]

    # --- doji family (repo is_doji definition) ----------------
    if _is_doji(o, h, l, c, atr_v, i):
        if lower >= _DRAGONFLY_LOWER_MIN * rng \
                and upper <= _DRAGONFLY_UPPER_MAX * rng:
            out.append(("dragonfly_doji", 1))
        elif upper >= _DRAGONFLY_LOWER_MIN * rng \
                and lower <= _DRAGONFLY_UPPER_MAX * rng:
            out.append(("gravestone_doji", -1))
        elif lower >= _LONG_LEGGED_MIN * rng \
                and upper >= _LONG_LEGGED_MIN * rng:
            out.append(("long_legged_doji", 0))
        else:
            out.append(("doji", 0))
    else:
        # --- pin family (repo pin_bar geometry, trend-named) --
        pin = _pin_bar(o, h, l, c, i)
        if pin:
            trend = _trend_before(c, i)
            if pin > 0:      # lower-wick rejection
                name = ("hammer" if trend < 0 else
                        "hanging_man" if trend > 0 else
                        "pin_bar_bullish")
                out.append((name, 1 if trend <= 0 else -1))
            else:            # upper-wick rejection
                name = ("shooting_star" if trend > 0 else
                        "inverted_hammer" if trend < 0 else
                        "pin_bar_bearish")
                out.append((name, -1 if trend >= 0 else 1))

    # --- engulfing-by-name (repo engulfing geometry) ----------
    eng = _engulfing(o, h, l, c, i)
    if eng > 0:
        out.append(("bullish_engulfing", 1))
    elif eng < 0:
        out.append(("bearish_engulfing", -1))

    # --- piercing line / dark cloud cover ----------------------
    b_prev = c[i - 1] - o[i - 1]
    rng_prev = h[i - 1] - l[i - 1]
    if rng_prev > 0 and abs(b_prev) >= _STRONG_BODY_FRAC * rng_prev:
        mid_prev = (o[i - 1] + c[i - 1]) / 2.0
        if b_prev < 0 and body > 0 and not eng:
            # Bullish: opens at/below the prior close, closes
            # past the prior body's midpoint, short of the open.
            if o[i] <= c[i - 1] and mid_prev < c[i] < o[i - 1]:
                out.append(("piercing_line", 1))
        elif b_prev > 0 and body < 0 and not eng:
            if o[i] >= c[i - 1] and o[i - 1] < c[i] < mid_prev:
                out.append(("dark_cloud_cover", -1))

    # --- morning / evening star --------------------------------
    if i >= 2:
        b0 = c[i - 2] - o[i - 2]
        rng0 = h[i - 2] - l[i - 2]
        b1 = c[i - 1] - o[i - 1]
        rng1 = h[i - 1] - l[i - 1]
        if rng0 > 0 and rng1 > 0 \
                and abs(b0) >= _STRONG_BODY_FRAC * rng0 \
                and abs(b1) <= _STAR_BODY_FRAC * rng1:
            mid0 = (o[i - 2] + c[i - 2]) / 2.0
            if b0 < 0 and c[i - 1] < c[i - 2] and body > 0 \
                    and c[i] > mid0:
                out.append(("morning_star", 1))
            elif b0 > 0 and c[i - 1] > c[i - 2] and body < 0 \
                    and c[i] < mid0:
                out.append(("evening_star", -1))
    return out


def detect_named_patterns(df: pd.DataFrame, *,
                          window: int = 10) -> list[dict]:
    """Named patterns completing on the last ``window`` bars of
    ``df`` (the frame as the caller holds it — closed bars on
    the live path).  Each entry: name, direction, the
    completing bar's positional index + timestamp, and its age
    in bars relative to the frame's last bar.  Oldest first."""
    if df is None or len(df) < 2:
        return []
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    atr_s = _atr_series(df).to_numpy(dtype=float)
    n = len(df)
    out: list[dict] = []
    for i in range(max(1, n - window), n):
        atr_v = float(atr_s[i]) if np.isfinite(atr_s[i]) else 0.0
        for name, direction in patterns_at(o, h, l, c, atr_v, i):
            out.append({
                "name": name, "direction": direction,
                "bar_index": int(i),
                "bar_ts": str(df.index[i]),
                "bars_ago": int(n - 1 - i)})
    return out
