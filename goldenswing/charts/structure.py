"""Workspace structure: causal measurements on a bars frame.

Every function here measures what the bars themselves show — swings,
breaks, liquidity sweeps, fair-value gaps, named session levels,
session ranges, volatility regime, and the expected path to the
session target.  Nothing is inferred from outside the frame, nothing
is synthesized, and nothing uses a bar that does not exist yet:

- a swing is reported only once ``right`` bars exist after its
  extreme (unconfirmed swings in the last ``right`` bars are
  invisible, by construction);
- a session counts as completed only when the frame holds a bar
  from a later day;
- a break / sweep / FVG is timestamped at the bar that confirmed
  it, never at the bar it anticipates.

Callers that replay history slice the frame to the as-of time
FIRST and then call :func:`measure_workspace` — the measurement
cannot see the future because the future is not in the frame.

All payloads are JSON-safe (ISO timestamps, plain floats/ints).
``missing`` names everything that could not be measured and why —
a missing overlay is stated, never faked.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

__all__ = [
    "SwingPoint",
    "swing_points",
    "label_swings",
    "detect_breaks",
    "detect_liquidity",
    "detect_fvgs",
    "session_levels",
    "session_range_band",
    "volatility_regime",
    "expected_path",
    "measure_workspace",
    "mtf_state",
]

_SWING_CAP = 40
_BREAK_CAP = 20
_SWEEP_CAP = 20
_FVG_CAP = 12


@dataclass(frozen=True)
class SwingPoint:
    """One confirmed fractal swing: bar position, time, price."""

    index: int
    ts: pd.Timestamp
    price: float
    kind: str  # "high" | "low"


def _frame(frame) -> pd.DataFrame:
    """A sorted, validated copy of the bars frame."""
    if frame is None or len(frame) == 0:
        raise ValueError("workspace structure needs a non-empty frame")
    for col in ("open", "high", "low", "close"):
        if col not in frame:
            raise ValueError(f"frame is missing column {col!r}")
    out = frame.sort_index()
    return out


def _iso(ts) -> str:
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return str(ts)


# ------------------------------------------------------------------ swings
def swing_points(frame, *, left: int = 2, right: int = 2
                 ) -> list[SwingPoint]:
    """Confirmed fractal swings, in time order.

    A high at bar ``i`` is a swing high iff it is the maximum of
    ``high[i-left : i+right+1]`` and strictly above its left
    neighbour; a low is the mirror.  Bars in the last ``right``
    positions can never confirm, so they never yield a swing —
    that is the causality guarantee, not a bug.  Consecutive
    swings of the same kind collapse to the more extreme one.
    """
    df = _frame(frame)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    raw: list[SwingPoint] = []
    for i in range(left, n - right):
        window_h = high[i - left:i + right + 1]
        window_l = low[i - left:i + right + 1]
        if high[i] == window_h.max() and high[i] > high[i - 1]:
            raw.append(SwingPoint(i, df.index[i], float(high[i]),
                                  "high"))
        elif low[i] == window_l.min() and low[i] < low[i - 1]:
            raw.append(SwingPoint(i, df.index[i], float(low[i]),
                                  "low"))
    out: list[SwingPoint] = []
    for p in raw:
        if out and out[-1].kind == p.kind:
            keep = (p if (p.price > out[-1].price
                          if p.kind == "high"
                          else p.price < out[-1].price)
                    else out[-1])
            out[-1] = keep
        else:
            out.append(p)
    return out


def label_swings(points) -> list[dict]:
    """HH/LH for highs, HL/LL for lows, vs the previous swing of
    the same kind — the same rule as ``engine._swing_labels``."""
    out = []
    prev: dict[str, float] = {}
    for p in sorted(points, key=lambda q: q.index):
        kind = p.kind
        if kind not in prev:
            lab = "H" if kind == "high" else "L"
        elif kind == "high":
            lab = "HH" if p.price > prev[kind] else "LH"
        else:
            lab = "HL" if p.price > prev[kind] else "LL"
        out.append({"bar": int(p.index), "ts": _iso(p.ts),
                    "price": float(p.price), "kind": kind,
                    "label": lab})
        prev[kind] = p.price
    return out


# ------------------------------------------------------- breaks / liquidity
def detect_breaks(frame, swings, *, fail_bars: int = 5
                  ) -> list[dict]:
    """Breaks of swing extremes, and the breaks that failed.

    A break is the first close beyond a prior swing extreme
    (above a swing high, below a swing low), timestamped at that
    close's bar.  A failed break is a break followed within
    ``fail_bars`` bars by a close back inside the level — the
    break is still recorded (it happened), the failure is a
    second, linked event.  Returns the most recent
    ``_BREAK_CAP`` events, oldest first.
    """
    df = _frame(frame)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    events: list[dict] = []
    for s in swings:
        tail = close[s.index + 1:]
        if len(tail) == 0:
            continue
        if s.kind == "high":
            above = tail > s.price
            if not above.any():
                continue
            j = s.index + 1 + int(np.argmax(above))
            events.append({"kind": "break", "direction": 1,
                           "level": float(s.price), "bar": int(j),
                           "ts": _iso(df.index[j]),
                           "swing_bar": int(s.index)})
            window = close[j + 1: j + 1 + fail_bars]
            back = window < s.price
            if back.any():
                k = j + 1 + int(np.argmax(back))
                events.append({"kind": "failed_break",
                               "direction": 1,
                               "level": float(s.price),
                               "bar": int(k),
                               "ts": _iso(df.index[k]),
                               "swing_bar": int(s.index),
                               "break_bar": int(j)})
        else:
            below = tail < s.price
            if not below.any():
                continue
            j = s.index + 1 + int(np.argmax(below))
            events.append({"kind": "break", "direction": -1,
                           "level": float(s.price), "bar": int(j),
                           "ts": _iso(df.index[j]),
                           "swing_bar": int(s.index)})
            window = close[j + 1: j + 1 + fail_bars]
            back = window > s.price
            if back.any():
                k = j + 1 + int(np.argmax(back))
                events.append({"kind": "failed_break",
                               "direction": -1,
                               "level": float(s.price),
                               "bar": int(k),
                               "ts": _iso(df.index[k]),
                               "swing_bar": int(s.index),
                               "break_bar": int(j)})
    events.sort(key=lambda e: (e["bar"], e["kind"] != "break"))
    return events[-_BREAK_CAP:]


def detect_liquidity(frame, swings) -> list[dict]:
    """Liquidity sweeps: a wick past a prior swing extreme with
    the close back inside it, on the same bar.

    Above-side: high beyond the swing high, close below it (a
    sell-side pool swept).  Below-side is the mirror.  Each sweep
    bar is its own event; the most recent ``_SWEEP_CAP`` are
    returned, oldest first.
    """
    df = _frame(frame)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    events: list[dict] = []
    for s in swings:
        if s.kind == "high":
            hits = np.where(
                (high[s.index + 1:] > s.price)
                & (close[s.index + 1:] < s.price))[0]
            for h in hits:
                j = s.index + 1 + int(h)
                events.append({"kind": "sweep", "side": "above",
                               "level": float(s.price),
                               "bar": int(j),
                               "ts": _iso(df.index[j]),
                               "wick": float(high[j] - s.price),
                               "swing_bar": int(s.index)})
        else:
            hits = np.where(
                (low[s.index + 1:] < s.price)
                & (close[s.index + 1:] > s.price))[0]
            for h in hits:
                j = s.index + 1 + int(h)
                events.append({"kind": "sweep", "side": "below",
                               "level": float(s.price),
                               "bar": int(j),
                               "ts": _iso(df.index[j]),
                               "wick": float(s.price - low[j]),
                               "swing_bar": int(s.index)})
    events.sort(key=lambda e: e["bar"])
    return events[-_SWEEP_CAP:]


# --------------------------------------------------------------------- FVGs
def detect_fvgs(frame) -> list[dict]:
    """Fair-value gaps (3-bar) with causal mitigation.

    Bullish: low[i] > high[i-2]; bearish: high[i] < low[i-2].
    A gap is mitigated at the first later bar whose range trades
    into it (low <= bottom for bullish, high >= top for bearish);
    otherwise it stays open.  The most recent ``_FVG_CAP`` gaps,
    oldest first.
    """
    df = _frame(frame)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    gaps: list[dict] = []
    for i in range(2, n):
        if low[i] > high[i - 2]:
            gaps.append({"direction": 1,
                         "top": float(low[i]),
                         "bottom": float(high[i - 2]),
                         "created_bar": int(i),
                         "created_ts": _iso(df.index[i]),
                         "mitigated_bar": None,
                         "mitigated_ts": None,
                         "status": "open"})
        elif high[i] < low[i - 2]:
            gaps.append({"direction": -1,
                         "top": float(low[i - 2]),
                         "bottom": float(high[i]),
                         "created_bar": int(i),
                         "created_ts": _iso(df.index[i]),
                         "mitigated_bar": None,
                         "mitigated_ts": None,
                         "status": "open"})
    for g in gaps[-30:]:
        for j in range(g["created_bar"] + 1, n):
            if g["direction"] == 1 and low[j] <= g["bottom"]:
                g["mitigated_bar"] = int(j)
                g["mitigated_ts"] = _iso(df.index[j])
                g["status"] = "mitigated"
                break
            if g["direction"] == -1 and high[j] >= g["top"]:
                g["mitigated_bar"] = int(j)
                g["mitigated_ts"] = _iso(df.index[j])
                g["status"] = "mitigated"
                break
    return gaps[-_FVG_CAP:]


# ------------------------------------------------------------ session levels
def _day_groups(frame) -> list[tuple[object, int, int]]:
    """(UTC date, first bar, last bar) per calendar day."""
    idx = frame.index
    days: list[tuple[object, int, int]] = []
    cur = None
    for i, ts in enumerate(idx):
        d = pd.Timestamp(ts).date()
        if d != cur:
            days.append((d, i, i))
            cur = d
        else:
            days[-1] = (d, days[-1][1], i)
    return days


def session_levels(frame, *, count: int = 2) -> list[dict]:
    """Named levels of the last ``count`` COMPLETED sessions.

    A UTC day is completed only when the frame holds a bar from a
    later day — the levels of the day still forming are never
    presented as finished.  Each level carries the weekday name
    the owner reads on the chart ("Friday H", "Thursday L").
    """
    df = _frame(frame)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    days = _day_groups(df)
    completed = days[:-1] if len(days) > 1 else []
    out: list[dict] = []
    for date, first, last in completed[-count:]:
        hi_bar = first + int(np.argmax(high[first:last + 1]))
        lo_bar = first + int(np.argmin(low[first:last + 1]))
        weekday = date.strftime("%A")
        out.append({"date": date.isoformat(), "weekday": weekday,
                    "label_high": f"{weekday} H",
                    "label_low": f"{weekday} L",
                    "high": float(high[hi_bar]),
                    "low": float(low[lo_bar]),
                    "high_bar": int(hi_bar), "low_bar": int(lo_bar),
                    "high_ts": _iso(df.index[hi_bar]),
                    "low_ts": _iso(df.index[lo_bar]),
                    "start_bar": int(first), "end_bar": int(last)})
    return out


def session_range_band(frame) -> dict:
    """The session range band: the most recent completed session
    as the band, plus the developing range of the day in progress.

    ``prior`` is None when no session has completed yet;
    ``current`` is None only for an empty frame (rejected above).
    """
    df = _frame(frame)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    days = _day_groups(df)
    prior = None
    if len(days) > 1:
        date, first, last = days[-2]
        prior = {"top": float(high[first:last + 1].max()),
                 "bottom": float(low[first:last + 1].min()),
                 "label": f"{date.strftime('%A')} range",
                 "date": date.isoformat(),
                 "start_bar": int(first), "end_bar": int(last)}
    date, first, last = days[-1]
    current = {"top": float(high[first:last + 1].max()),
               "bottom": float(low[first:last + 1].min()),
               "label": f"{date.strftime('%A')} range (forming)",
               "date": date.isoformat(),
               "start_bar": int(first), "end_bar": int(last),
               "developing": True}
    return {"prior": prior, "current": current}


# ------------------------------------------------------- volatility regime
def volatility_regime(frame, *, period: int = 14, window: int = 50,
                      expand: float = 1.5, contract: float = 0.7
                      ) -> dict:
    """ATR vs its own trailing median: expansion / contraction /
    normal, with the current run's start.

    ATR is Wilder's EMA of true range (past bars only); the
    median runs over the trailing ``window`` ATR values.  The run
    walks back from the last bar while the regime label holds.
    """
    df = _frame(frame)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev = np.empty_like(close)
    prev[0] = close[0]
    prev[1:] = close[:-1]
    tr = np.maximum.reduce([high - low, np.abs(high - prev),
                            np.abs(low - prev)])
    atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False,
                            min_periods=1).mean().to_numpy(dtype=float)
    med = pd.Series(atr).rolling(window, min_periods=1).median() \
        .to_numpy(dtype=float)
    ratio = atr / np.where(med > 0, med, np.nan)
    last = float(ratio[-1]) if len(ratio) else float("nan")

    def _label(r: float) -> str:
        if not np.isfinite(r):
            return "normal"
        if r >= expand:
            return "expansion"
        if r <= contract:
            return "contraction"
        return "normal"

    regime = _label(last)
    run_start = len(df) - 1
    while run_start > 0 and _label(float(ratio[run_start - 1])) == regime:
        run_start -= 1
    return {"regime": regime, "atr": float(atr[-1]),
            "median_atr": float(med[-1]), "ratio": last,
            "run_start_bar": int(run_start),
            "run_start_ts": _iso(df.index[run_start]),
            "run_bars": int(len(df) - run_start)}


# ------------------------------------------------------------ expected path
def expected_path(frame, levels, *, target: str = "prior_low",
                  swing_cap: int = 3) -> dict | None:
    """The drawn path to the session target, from measured points.

    Waypoints are real: the last bar's close, then the most
    recent swing points inside the current session (at most
    ``swing_cap``), in time order.  The endpoint is the named
    session level (the prior session's low, or high) at a
    projected time — marked ``projected: true`` — three median
    bar intervals past the last bar.  The path is an annotation
    joining measured structure to a measured level, not a
    forecast: it says where the target sits, never that price
    will go there.  Returns None when no session level exists.
    """
    if not levels:
        return None
    df = _frame(frame)
    prior = levels[-1]
    if target == "prior_high":
        target_price = float(prior["high"])
        target_label = prior["label_high"]
    else:
        target_price = float(prior["low"])
        target_label = prior["label_low"]
    swings = swing_points(df)
    current_start = prior["end_bar"] + 1
    recent = [s for s in swings if s.index >= current_start]
    recent = recent[-swing_cap:]
    points: list[dict] = []
    last_i = len(df) - 1
    points.append({"bar": int(last_i), "ts": _iso(df.index[last_i]),
                   "price": float(df["close"].iloc[last_i]),
                   "kind": "close", "projected": False})
    for s in recent:
        points.append({"bar": int(s.index), "ts": _iso(s.ts),
                       "price": float(s.price), "kind": "swing",
                       "projected": False})
    points.sort(key=lambda p: (p["bar"] if p["bar"] is not None
                               else 1 << 30))
    idx = df.index
    if len(idx) > 1:
        deltas = np.diff(idx.asi8) / 1e9
        step = float(np.median(deltas[deltas > 0])) \
            if np.any(deltas > 0) else 3600.0
    else:
        step = 3600.0
    end_ts = pd.Timestamp(idx[-1]) + pd.Timedelta(seconds=3 * step)
    points.append({"bar": None, "ts": _iso(end_ts),
                   "price": target_price, "kind": "target",
                   "projected": True})
    return {"target": target, "target_label": target_label,
            "target_price": target_price, "marker": "$",
            "points": points}


# ------------------------------------------------------------------- whole
def measure_workspace(frame, *, session_count: int = 2) -> dict:
    """The whole workspace measurement for one frame.

    ``missing`` names every overlay that could not be measured
    and why.  Bars are capped in the docstrings of the pieces;
    the payload carries the most recent of each.
    """
    df = _frame(frame)
    missing: list[dict] = []
    swings = swing_points(df)
    labeled = label_swings(swings)
    levels = session_levels(df, count=session_count)
    band = session_range_band(df)
    vol = volatility_regime(df)
    path = expected_path(df, levels)
    if len(df) < 5:
        missing.append({"what": "swings",
                        "reason": f"only {len(df)} bars; need 5+"})
    if not levels:
        missing.append({"what": "session_levels",
                        "reason": "no completed session in the frame"})
    if band["prior"] is None:
        missing.append({"what": "session_range",
                        "reason": "no completed session in the frame"})
    if path is None:
        missing.append({"what": "expected_path",
                        "reason": "no session level to target"})
    vol_col = next((c for c in ("tick_volume", "volume",
                                "real_volume") if c in df), None)
    if vol_col is None:
        missing.append({"what": "volume",
                        "reason": "frame carries no volume column"})
    return {"as_of": _iso(df.index[-1]), "bars": int(len(df)),
            "swings": labeled[-_SWING_CAP:],
            "breaks": detect_breaks(df, swings),
            "liquidity": detect_liquidity(df, swings),
            "fvgs": detect_fvgs(df),
            "session_levels": levels,
            "session_range": band,
            "volatility": vol,
            "expected_path": path,
            "missing": missing}


def mtf_state(frame, *, tf: str) -> dict:
    """One timeframe's measured state, from ITS OWN frame.

    Trend comes from that TF's own swing labels (last two highs
    and last two lows); regime from its own ATR; range position
    from its own prior session range.  Nothing is borrowed from
    another timeframe.
    """
    try:
        df = _frame(frame)
    except ValueError as exc:
        return {"tf": tf, "available": False,
                "reason": str(exc)}
    swings = label_swings(swing_points(df))
    highs = [s for s in swings if s["kind"] == "high"]
    lows = [s for s in swings if s["kind"] == "low"]
    if len(highs) >= 2 and len(lows) >= 2:
        up = (highs[-1]["price"] > highs[-2]["price"]
              and lows[-1]["price"] > lows[-2]["price"])
        down = (highs[-1]["price"] < highs[-2]["price"]
                and lows[-1]["price"] < lows[-2]["price"])
        trend = "up" if up else "down" if down else "mixed"
    elif swings:
        trend = "flat"
    else:
        trend = "unknown"
    levels = session_levels(df, count=1)
    band = session_range_band(df)
    pos = None
    if band["prior"] is not None:
        top = band["prior"]["top"]
        bot = band["prior"]["bottom"]
        if top > bot:
            pos = (float(df["close"].iloc[-1]) - bot) / (top - bot)
            pos = max(0.0, min(1.0, pos))
    vol = volatility_regime(df)
    return {"tf": tf, "available": True, "bars": int(len(df)),
            "last_ts": _iso(df.index[-1]),
            "trend": trend,
            "last_swing": swings[-1] if swings else None,
            "regime": vol["regime"],
            "range_position": pos,
            "session": levels[-1] if levels else None}
