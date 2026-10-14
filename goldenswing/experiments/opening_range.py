"""Opening-range breakout + retest on QQQ (user-contributed, comment section).

The ruleset, from the user's words (a trading comment section):

  1. Mark the high and low of the first 5-minute candle at 09:35 ET
     (the 09:30-09:35 bar of the NY stock session) -> the opening range.
  2. Drop to the 1-minute timeframe; wait for a break above or below
     either level.
  3. Wait PATIENTLY for a retest of the broken level, then enter on the
     retest.
  4. An FVG (3-candle rule) can confirm the entry.
  5. Check higher timeframes before NY open; QQQ itself is the bias
     read for the true direction.
  6. TP at 1:2 R:R, or at critical levels (PDH/PDL, PMH/PML).
  7. Range-width filter (a second comment): "works better when the range
     is wide enough — otherwise R:R is not good enough and the
     probability is lower."  Skip narrow opening ranges.

This is the project's first non-gold market AND first intraday session
experiment, so it is a standalone shadow experiment: it never touches
the gold detector or any reader state.  Everything is causal — the
opening range comes from the first bar of the day, breaks/retests/FVGs
from closed bars only, daily bias and PDH/PDL from completed daily
bars, and the range gate from prior days' ranges.

Mechanics (per regular-session day, America/New_York):
- Opening range (OR): high/low of the 09:30-09:35 bar (first 5m bar;
  on M1 grids, high/low of the first five 1m bars).
- Break: first bar (after the OR bar) whose CLOSE crosses the level
  (close > OR_hi -> long bias / close < OR_lo -> short).  A wick
  through without a close is not a break.
- Retest: a resting limit at the broken level, live for RETEST_EXPIRY
  bars after the break bar.  Fill = first bar whose wick touches the
  level; a gap through it fills at the adverse open.  A bar that fills
  AND closes through the level (break failure) is booked as filled
  then stopped (-1R) — the conservative same-bar convention from the
  other sleeves.
- FVG confirmation (variant): a 3-candle fair value gap in the
  breakout direction formed within FVG_LOOKBACK bars ending at the
  retest bar.  Bullish FVG at bar i: low[i] > high[i-2].
- Bias filter (variant): QQQ daily trend — long setups only when the
  prior close >= its 20-day SMA (completed bars), shorts only when <=.
- Range gate (variant): skip the day when the OR width is below
  RANGE_GATE_RATIO x the median of the prior MIN_GATE_HISTORY days'
  OR widths.
- Stop: beyond the broken level by STOP_BUFFER x OR width.  Risk is
  therefore STOP_BUFFER x OR width by construction.
- Targets: variant "rr2" = entry +/- 2R (the comment's 1:2); variant
  "pdh_pdl" = the day's PDH (long) / PDL (short), skipped when it
  leaves less than 1R of room.
- Trade walk: stop wins ties, adverse opens on gaps, flat $0.02/share
  round-trip cost (stated assumption), time stop = last regular-session
  bar (timeout exit at its close).  One trade per day maximum.

Costs are per share, so cost_in_R = 0.02 / risk_per_share.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.data import REPO_ROOT
from ..backtest.harness import (Outcome, book_metrics, build_book,
                                slice_stats)
from ..golden.detector import GoldenSignal

# ---------------------------------------------------------------- constants
#: yfinance server caps: M5 -> ~60d, M1 -> ~7d (verified 2026-10-11).
QQQ_5M_CACHE = REPO_ROOT / "data" / "QQQ_5M_60d.csv"
QQQ_1M_CACHE = REPO_ROOT / "data" / "QQQ_1M_7d.csv"
QQQ_1D_CACHE = REPO_ROOT / "data" / "QQQ_1D_2y.csv"

NY = "America/New_York"
SESSION_OPEN = "09:30"
SESSION_CLOSE = "16:00"          # bars start before this; last bar 15:55
#: Resting retest limit lives this many bars after the break bar.
RETEST_EXPIRY_M5 = 12
RETEST_EXPIRY_M1 = 30
#: Stop sits this fraction of the OR width beyond the broken level.
STOP_BUFFER_OR = 0.5
#: Fixed target, the comment's 1:2.
FIXED_TP_R = 2.0
#: PDH/PDL variant: target must leave at least this much room.
LEVEL_TARGET_MIN_R = 1.0
#: Range gate: skip when OR width < ratio x median of prior widths.
RANGE_GATE_RATIO = 0.5
#: Prior OR widths needed before the gate can judge (else allow).
MIN_GATE_HISTORY = 10
#: FVG 3-candle confirmation: gap must form within this many bars ending
#: at the retest bar.
FVG_LOOKBACK = 5
#: Stated cost assumption, dollars per share round trip.
COST_PER_SHARE = 0.02


# ------------------------------------------------------------------- data
def load_qqq(cache_path: Path) -> pd.DataFrame:
    """Load a cached QQQ frame (UTC index).  Raises if the cache is
    missing — the fetch used chunked overlapping yfinance pulls
    (see the experiment report); deeper intraday history will come from
    the MT5 terminal's own data once deployed."""
    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def regular_sessions(df: pd.DataFrame):
    """Yield (et_date, day_frame) for each ET calendar day with regular-
    session bars, in chronological order.  ``pos`` holds each bar's
    integer position in the full frame; ``ts`` its UTC timestamp."""
    from datetime import time as _time
    et = df.tz_convert(NY)
    dates = et.index.date
    out = []
    for d in sorted(set(dates)):
        full_pos = np.flatnonzero(dates == d)
        times = et.index[full_pos].time
        reg = np.array([(t >= _time(9, 30)) and (t < _time(16, 0))
                        for t in times])
        if not reg.any():
            continue
        keep = full_pos[reg]
        sub = df.iloc[keep].copy()
        sub["pos"] = keep
        sub["ts"] = sub.index
        out.append((d, sub.reset_index(drop=True)))
    return out


# ------------------------------------------------------------------ events
@dataclass
class ORSetup:
    """A decided opening-range setup for one day."""

    day: object
    direction: int                 # +1 long | -1 short
    or_hi: float
    or_lo: float
    or_width: float
    break_pos: int                 # integer position in the full frame
    expiry: int                    # bars the retest limit lives
    bias_ok: bool = True
    gate_ok: bool = True
    grid: str = "5m"


@dataclass
class ORFill:
    setup: ORSetup
    fill_pos: int
    entry: float
    same_bar_stop: bool = False


# ------------------------------------------------------------- pure helpers
def opening_range(day_df: pd.DataFrame, grid: str):
    """High/low of the first 5m candle (09:30-09:35).  On M1 grids this
    is the high/low of the first five 1m bars.  Returns
    (hi, lo, first_signal_offset) where the offset is the first bar
    AFTER the OR window."""
    n = 5 if grid == "1m" else 1
    hi = float(day_df["high"].iloc[:n].max())
    lo = float(day_df["low"].iloc[:n].min())
    return hi, lo, n


def has_fvg(high: np.ndarray, low: np.ndarray, end: int,
            direction: int, lookback: int = FVG_LOOKBACK) -> bool:
    """3-candle fair value gap in ``direction`` formed within
    ``lookback`` bars ending at ``end`` (all closed bars).  Bullish FVG
    at bar i: low[i] > high[i-2]; bearish: high[i] < low[i-2]."""
    lo = max(2, end - lookback + 1)
    for i in range(lo, end + 1):
        if direction == 1 and low[i] > high[i - 2]:
            return True
        if direction == -1 and high[i] < low[i - 2]:
            return True
    return False


def _fill_price(direction: int, level: float, open_: float) -> float:
    if direction == 1:
        return level if open_ > level else open_
    return level if open_ < level else open_


# ------------------------------------------------------------------- scan
def scan_day(day_df: pd.DataFrame, or_hi: float, or_lo: float,
             or_width: float, grid: str, first_sig: int,
             prev_or_widths: list[float],
             bias: int | None,
             use_gate: bool, use_bias: bool) -> list[ORSetup]:
    """Find the day's break->retest setup.  At most one setup per day:
    the FIRST close-through break of either OR level.

    ``prev_or_widths`` holds completed prior days' OR widths (causal).
    ``bias`` is the daily-trend bias decided before the open (+1/-1/0).
    """
    high = day_df["high"].to_numpy(dtype=float)
    low = day_df["low"].to_numpy(dtype=float)
    close = day_df["close"].to_numpy(dtype=float)
    pos = day_df["pos"].to_numpy(dtype=int)
    expiry = RETEST_EXPIRY_M1 if grid == "1m" else RETEST_EXPIRY_M5

    gate_ok = True
    if use_gate and len(prev_or_widths) >= MIN_GATE_HISTORY:
        gate_ok = or_width >= RANGE_GATE_RATIO * float(
            np.median(prev_or_widths))

    for k in range(first_sig, len(day_df)):
        d = 0
        if close[k] > or_hi:
            d = 1
        elif close[k] < or_lo:
            d = -1
        if d == 0:
            continue
        bias_ok = True
        if use_bias and bias is not None:
            bias_ok = (bias == d)
        level = or_hi if d == 1 else or_lo
        return [ORSetup(day=day_df["ts"].iloc[0].date(), direction=d,
                        or_hi=or_hi, or_lo=or_lo, or_width=or_width,
                        break_pos=int(pos[k]), expiry=expiry,
                        bias_ok=bias_ok, gate_ok=gate_ok, grid=grid)]
    return []


# ------------------------------------------------------------------- fill
def find_retest_fill(df: pd.DataFrame, setup: ORSetup,
                     use_fvg: bool) -> ORFill | None:
    """Work the resting limit at the broken level after the break bar.

    Returns None when the order expires unfilled (cancelled) or when
    the FVG confirmation variant is on and no qualifying gap exists at
    the retest bar.  A bar that fills AND closes back through the level
    (break failure) is booked filled-then-stopped.
    """
    d = setup.direction
    level = setup.or_hi if d == 1 else setup.or_lo
    or_width = setup.or_width
    stop = (level - STOP_BUFFER_OR * or_width) if d == 1 \
        else (level + STOP_BUFFER_OR * or_width)
    n = len(df)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    last = min(setup.break_pos + setup.expiry, n - 1)
    for k in range(setup.break_pos + 1, last + 1):
        entry = _fill_price(d, level, open_[k])
        if d == 1:
            touched = low[k] <= level
            failed = close[k] < level      # closed back below the level
            if failed:
                return (ORFill(setup, k, entry, same_bar_stop=True)
                        if touched else None)
            if low[k] <= stop:             # traded through fill and stop
                return ORFill(setup, k, entry, same_bar_stop=True)
            if touched:
                if use_fvg and not has_fvg(high, low, k, d):
                    return None            # no confirmation: pass the day
                return ORFill(setup, k, entry)
        else:
            touched = high[k] >= level
            failed = close[k] > level
            if failed:
                return (ORFill(setup, k, entry, same_bar_stop=True)
                        if touched else None)
            if high[k] >= stop:
                return ORFill(setup, k, entry, same_bar_stop=True)
            if touched:
                if use_fvg and not has_fvg(high, low, k, d):
                    return None
                return ORFill(setup, k, entry)
    return None


# --------------------------------------------------------------- simulate
def _to_signal(fill: ORFill, target: float) -> GoldenSignal:
    s = fill.setup
    atr_proxy = s.or_width  # OR width stands in for ATR on this grid
    return GoldenSignal(
        index=fill.fill_pos, timestamp=None, direction=s.direction,
        score=100.0, kind="sleeve_opening_range", entry_ref=fill.entry,
        invalidation=(s.or_hi - STOP_BUFFER_OR * s.or_width
                      if s.direction == 1
                      else s.or_lo + STOP_BUFFER_OR * s.or_width),
        targets=[target], atr=atr_proxy)


def simulate_trade(df: pd.DataFrame, day_end_pos: int, fill: ORFill,
                   target: float | None,
                   cost: float = COST_PER_SHARE) -> Outcome | None:
    """Intraday trade walk: stop wins ties, adverse opens, flat per-share
    cost, time stop at the last regular-session bar (timeout at close).
    ``target=None`` means no qualifying level target -> skip the trade."""
    if target is None:
        return None
    s = fill.setup
    d = s.direction
    or_width = s.or_width
    level = s.or_hi if d == 1 else s.or_lo
    stop = (level - STOP_BUFFER_OR * or_width) if d == 1 \
        else (level + STOP_BUFFER_OR * or_width)
    entry = fill.entry
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    sig = _to_signal(fill, target)

    def _net(exit_price: float, idx: int, reason: str, hold: int):
        r = d * (exit_price - entry) / risk - cost / risk
        return Outcome(signal=sig, entry_index=fill.fill_pos,
                       exit_index=idx, exit_reason=reason, r_net=float(r),
                       hold_bars=hold)

    if fill.same_bar_stop:
        return _net(entry, fill.fill_pos, "stop", 0)

    n = len(df)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    last = min(day_end_pos, n - 1)
    f = fill.fill_pos
    for k in range(f + 1, last + 1):
        if d == 1:
            if low[k] <= stop:
                px = stop if open_[k] > stop else open_[k]
                return _net(px, k, "stop", k - f)
            if high[k] >= target:
                px = target if open_[k] < target else open_[k]
                return _net(px, k, "target", k - f)
        else:
            if high[k] >= stop:
                px = stop if open_[k] < stop else open_[k]
                return _net(px, k, "stop", k - f)
            if low[k] <= target:
                px = target if open_[k] > target else open_[k]
                return _net(px, k, "target", k - f)
    return _net(close[last], last, "timeout", last - f)

# -------------------------------------------------------------------- run
@dataclass
class ORResult:
    grid: str
    bars: int
    start: str
    end: str
    n_sessions: int
    n_setups: int
    n_fills: int
    n_cancelled: int
    n_gated: int
    n_bias_blocked: int
    n_skipped_level: int
    n_fills_fvg: int = 0
    variants: dict = field(default_factory=dict)


def _daily_bias(daily: pd.DataFrame, day) -> int | None:
    """QQQ daily-trend bias decided BEFORE the open: prior close vs its
    20-day SMA, both from completed daily bars.  None when the history
    is too short."""
    et = daily.tz_convert(NY)
    dates = et.index.date
    idxs = [i for i, dd in enumerate(dates) if dd < day]
    if len(idxs) < 21:
        return None
    closes = daily["close"].to_numpy(dtype=float)
    prev = closes[idxs[-1]]
    sma = float(np.mean(closes[idxs[-21]:-1]))
    if prev > sma:
        return 1
    if prev < sma:
        return -1
    return 0


def _pdh_pdl(daily: pd.DataFrame, day):
    """Prior completed day's regular high/low.  yfinance daily bars are
    regular-session aggregates, so high/low are the PDH/PDL."""
    et = daily.tz_convert(NY)
    dates = et.index.date
    idxs = [i for i, dd in enumerate(dates) if dd < day]
    if not idxs:
        return None, None
    i = idxs[-1]
    return (float(daily["high"].iloc[i]), float(daily["low"].iloc[i]))


def _level_target(setup: ORSetup, entry: float, pdh: float | None,
                  pdl: float | None) -> float | None:
    """PDH (long) / PDL (short) target, skipped when it leaves less
    than 1R of room."""
    d = setup.direction
    level = setup.or_hi if d == 1 else setup.or_lo
    risk = STOP_BUFFER_OR * setup.or_width
    if d == 1:
        if pdh is None or pdh - entry < LEVEL_TARGET_MIN_R * risk:
            return None
        return pdh
    if pdl is None or entry - pdl < LEVEL_TARGET_MIN_R * risk:
        return None
    return pdl


def run_experiment(grid: str = "5m", refresh: bool = False) -> ORResult:
    """Full experiment on one grid.  Variants: FVG confirmation on/off,
    bias filter on/off, range gate on/off, target fixed-2R vs PDH/PDL."""
    cache = QQQ_5M_CACHE if grid == "5m" else QQQ_1M_CACHE
    df = load_qqq(cache_path=cache) if not refresh else None
    if df is None or refresh:
        raise RuntimeError(f"QQQ {grid} cache missing: {cache}")
    daily = load_qqq(QQQ_1D_CACHE)
    sessions = regular_sessions(df)

    prev_widths: list[float] = []
    setups_all: list[tuple[ORSetup, ORFill | None]] = []
    n_gated = n_bias = 0

    for day, day_df in sessions:
        if len(day_df) < (6 if grid == "1m" else 2):
            continue
        # First bar must be the 09:30 bar (regular session start).
        first_ts = day_df["ts"].iloc[0].tz_convert(NY)
        if not (first_ts.hour == 9 and first_ts.minute == 30):
            continue
        or_hi, or_lo, first_sig = opening_range(day_df, grid)
        or_width = or_hi - or_lo
        if or_width <= 0:
            prev_widths.append(or_width)
            continue
        bias = _daily_bias(daily, day)
        pdh, pdl = _pdh_pdl(daily, day)
        found = scan_day(day_df, or_hi, or_lo, or_width, grid,
                         first_sig, prev_widths, bias,
                         use_gate=True, use_bias=True)
        prev_widths.append(or_width)
        if not found:
            continue
        setup = found[0]
        # NOTE: gate/bias flags are stored; the gate ON/OFF and bias
        # ON/OFF comparison happens in the variant grid below.
        for use_fvg in (False, True):
            fill = find_retest_fill(df, setup, use_fvg=use_fvg)
            setups_all.append((setup, fill, use_fvg))

    def _collect(use_gate: bool, use_bias: bool, use_fvg: bool):
        outs: dict[str, list[Outcome]] = {"rr2": [], "pdh_pdl": []}
        skipped = 0
        for setup, fill, fvg in setups_all:
            if fvg != use_fvg or fill is None:
                continue
            if use_gate and not setup.gate_ok:
                continue
            if use_bias and not setup.bias_ok:
                continue
            sday = setup.day
            et = df.tz_convert(NY)
            day_end = int(np.flatnonzero(et.index.date == sday)[-1])
            entry = fill.entry
            d = setup.direction
            risk = STOP_BUFFER_OR * setup.or_width
            outs["rr2"].append(simulate_trade(
                df, day_end, fill, entry + d * FIXED_TP_R * risk))
            pdh, pdl = _pdh_pdl(daily, sday)
            tgt = _level_target(setup, entry, pdh, pdl)
            o = simulate_trade(df, day_end, fill, tgt)
            if o is None:
                skipped += 1
            else:
                outs["pdh_pdl"].append(o)
        return outs, skipped

    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    variants: dict[str, dict] = {}
    for g in (False, True):
        for b in (False, True):
            for f in (False, True):
                outs, skipped = _collect(g, b, f)
                for tname in ("rr2", "pdh_pdl"):
                    book = build_book(
                        [o for o in outs[tname] if o is not None], 0.0)
                    variants[
                        f"gate{'on' if g else 'off'}_"
                        f"bias{'on' if b else 'off'}_"
                        f"fvg{'on' if f else 'off'}_{tname}"] = {
                        **book_metrics(book, years),
                        "shadow": slice_stats(
                            [o for o in outs[tname] if o is not None]),
                        "n_skipped_level":
                            skipped if tname == "pdh_pdl" else 0,
                    }

    n_setups = len(setups_all) // 2
    fills_plain = sum(1 for _, fl, f in setups_all
                      if fl is not None and not f)
    fills_fvg = sum(1 for _, fl, f in setups_all
                    if fl is not None and f)
    uniq = [s for s, _, f in setups_all if not f]
    return ORResult(
        grid=grid, bars=len(df), start=str(df.index[0]),
        end=str(df.index[-1]), n_sessions=len(sessions),
        n_setups=n_setups, n_fills=fills_plain,
        n_cancelled=n_setups - fills_plain,
        n_gated=sum(1 for s in uniq if not s.gate_ok),
        n_bias_blocked=sum(1 for s in uniq if not s.bias_ok),
        n_skipped_level=0, variants=variants,
        n_fills_fvg=fills_fvg)
