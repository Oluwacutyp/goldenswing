"""The trading DAY as a first-class object (P3, spec v3 S2/S13).

Before P3 the engine had no day: no prior-day value area, no
Initial Balance, no opening type, no fuel gauge, no gap state —
the P2 scenario matrix's day-level rows (1-3, 8, 12) had nothing
to read.  This module computes ONE :class:`DayContext` per
session from the frames the runner already holds, refreshed on
bar closes.  Everything here is pure: frames in, context out.

Conventions (documented, desk-standard):

- The trading day is keyed to New York time and starts at the
  18:00 ET reopen (post-rollover): bars from 18:00 ET Monday to
  17:59 ET Tuesday belong to Tuesday's trading day.  Sunday's
  18:00 ET reopen opens Monday's day (and the week).
- Prior-day value (VAH / VAL / POC) comes from the prior day's
  H1 bars through the desk's existing TPO volume profile
  (``live/volume_profile.py``, 24 bins, 70% value area) — value
  is an INVARIANT (mining section 4); the bin count is a
  parameter of that module, unchanged.
- Initial Balance = the first H1 bar of the trading day (the
  18:00 ET hour).  Dalton's IB is the first hour of trade; on a
  24h market the first hour of the NEW day is the honest anchor.
- ADR / AWR are FUEL GAUGES (S8/F12): they price continuation
  and bank runners; they never veto.  Trend days regularly
  print 120-150% of ADR — the gauge de-rates, it does not block.
- Opening types are classified per Dalton at the London and NY
  checkpoints from the open's position vs prior value and the
  first period's behaviour, and RE-TYPED as the session prints
  more bars (S13): a label is a working hypothesis, not a stamp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .multitf import atr14
from .volume_profile import volume_profile

NY_TZ = "America/New_York"

# Opening-type labels (Dalton).
OPEN_DRIVE = "open_drive"
OPEN_TEST_DRIVE = "open_test_drive"
OPEN_REJECTION_REVERSE = "open_rejection_reverse"
OPEN_AUCTION = "open_auction"
OPEN_UNKNOWN = "unknown"

# Day types (profile day taxonomy, S2).
DAY_TREND = "trend"
DAY_NEUTRAL = "neutral"
DAY_DOUBLE_DIST = "double_distribution"
DAY_RANGE = "range"
DAY_NONTREND = "nontrend"
DAY_DEVELOPING = "developing"

# ADR consumption states (S8): fresh < 50%, working 50-85%,
# spent 85-100%, extended 100-150% (trend days live here),
# exhausted > 150%.
ADR_FRESH = "fresh"
ADR_WORKING = "working"
ADR_SPENT = "spent"
ADR_EXTENDED = "extended"
ADR_EXHAUSTED = "exhausted"


def trading_day_key(ts) -> "pd.Timestamp":
    """The NY trading-day date a timestamp belongs to.

    Bars at/after 18:00 ET belong to the NEXT calendar date's
    trading day (the day session they open); earlier bars belong
    to their own date.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    ny = t.tz_convert(NY_TZ)
    return (ny + pd.Timedelta(hours=6)).normalize()


def group_days(h1: pd.DataFrame) -> dict:
    """Group an H1 frame into {trading-day-key: sub-frame}."""
    if h1 is None or not len(h1):
        return {}
    keys = [trading_day_key(ts) for ts in h1.index]
    out: dict = {}
    for key, (_, row) in zip(keys, h1.iterrows()):
        out.setdefault(key, []).append(row)
    return {k: pd.DataFrame(rows) for k, rows in out.items()}


def _day_levels(day: pd.DataFrame) -> dict:
    """OHLC + value area for one day's H1 bars."""
    vp = volume_profile(day, bins=24)
    return {
        "high": float(day["high"].max()),
        "low": float(day["low"].min()),
        "open": float(day["open"].iloc[0]),
        "close": float(day["close"].iloc[-1]),
        "vah": float(vp["vah"] or 0.0), "val": float(vp["val"] or 0.0),
        "poc": float(vp["poc"] or 0.0),
        "range": float(day["high"].max() - day["low"].min()),
        "bars": int(len(day)),
    }


def adr_of(d1: pd.DataFrame, *, n: int = 14) -> float:
    """Average daily range over the last ``n`` completed D1 bars."""
    if d1 is None or len(d1) < 2:
        return 0.0
    ranges = (d1["high"] - d1["low"]).iloc[-(n + 1):-1]
    if not len(ranges):
        ranges = (d1["high"] - d1["low"]).iloc[-n:]
    return float(ranges.mean()) if len(ranges) else 0.0


def awr_of(w1: pd.DataFrame, *, n: int = 14) -> float:
    """Average weekly range over the last ``n`` completed weeks."""
    if w1 is None or len(w1) < 2:
        return 0.0
    ranges = (w1["high"] - w1["low"]).iloc[-(n + 1):-1]
    if not len(ranges):
        ranges = (w1["high"] - w1["low"]).iloc[-n:]
    return float(ranges.mean()) if len(ranges) else 0.0


def adr_state(consumed_pct: float) -> str:
    if consumed_pct < 50.0:
        return ADR_FRESH
    if consumed_pct < 85.0:
        return ADR_WORKING
    if consumed_pct < 100.0:
        return ADR_SPENT
    if consumed_pct <= 150.0:
        return ADR_EXTENDED
    return ADR_EXHAUSTED


def classify_open(*, open_price: float, first_closes: list[float],
                  period_high: float, period_low: float,
                  prior_vah: float, prior_val: float,
                  ib_width: float) -> str:
    """Dalton opening type from the open vs prior value + the
    first period's behaviour.  Pure; called at a checkpoint with
    however many first-period closes exist (re-typed as more
    print).  Thresholds are in IB-width units — a parameter, the
    classification structure is the invariant.
    """
    if not first_closes or ib_width <= 0:
        return OPEN_UNKNOWN
    last = float(first_closes[-1])
    opened_above = open_price > prior_vah
    opened_below = open_price < prior_val
    opened_inside = not opened_above and not opened_below
    # Open-drive: opened at/near an extreme and the first period
    # drove away from the open in one direction, closing in the
    # driving direction's half with real range extension.
    drive_up = opened_above or (opened_inside
                                and open_price >= prior_val)
    if drive_up and last > open_price + 0.5 * ib_width \
            and period_low >= open_price - 0.25 * ib_width:
        return OPEN_DRIVE
    if (opened_below or (opened_inside
                         and open_price <= prior_vah)) \
            and last < open_price - 0.5 * ib_width \
            and period_high <= open_price + 0.25 * ib_width:
        return OPEN_DRIVE
    # Open-rejection-reverse: opened OUTSIDE prior value and the
    # first period rejected it back inside.
    if opened_above and last < prior_vah:
        return OPEN_REJECTION_REVERSE
    if opened_below and last > prior_val:
        return OPEN_REJECTION_REVERSE
    # Open-test-drive: opened inside/near value, probed one way
    # (beyond the open by > 0.5 IB), then drove the OTHER way
    # through the open.
    probed_up = period_high > open_price + 0.5 * ib_width
    probed_dn = period_low < open_price - 0.5 * ib_width
    if probed_up and last < open_price:
        return OPEN_TEST_DRIVE
    if probed_dn and last > open_price:
        return OPEN_TEST_DRIVE
    if opened_inside:
        return OPEN_AUCTION
    return OPEN_UNKNOWN


def classify_day_type(*, ib_width: float, day_high: float,
                      day_low: float, ib_high: float,
                      ib_low: float, adr: float,
                      poc: float, late_break: bool) -> str:
    """Profile day type from IB + extensions + value position.

    Extensions are measured in IB-width units (Dalton): a trend
    day extends >= 2x IB one way with value migrating; a neutral
    day extends both ways; a wide-IB day that goes nowhere is a
    rotation (range) day; a narrow-IB day that goes nowhere is a
    nontrend day (no business done — the coil).  A range day
    that breaks late into a second distribution is a
    double-distribution trend day.
    """
    if ib_width <= 0:
        return DAY_DEVELOPING
    up_ext = (day_high - ib_high) / ib_width
    dn_ext = (ib_low - day_low) / ib_width
    ib_frac = ib_width / adr if adr > 0 else 0.0
    one_sided = (up_ext >= 2.0 and dn_ext <= 0.5) or \
        (dn_ext >= 2.0 and up_ext <= 0.5)
    if one_sided:
        return DAY_DOUBLE_DIST if late_break else DAY_TREND
    if up_ext >= 1.0 and dn_ext >= 1.0:
        return DAY_NEUTRAL
    if up_ext < 0.5 and dn_ext < 0.5:
        if ib_frac >= 0.45:
            return DAY_RANGE
        if ib_frac <= 0.25:
            return DAY_NONTREND
        return DAY_RANGE if poc and ib_low <= poc <= ib_high \
            else DAY_DEVELOPING
    return DAY_DEVELOPING


@dataclass
class DayContext:
    """One trading day, computed.  Refreshed on bar closes."""
    day_key: str = ""
    prior: dict = field(default_factory=dict)     # prior day levels
    prior_week: dict = field(default_factory=dict)
    naked_pocs: list = field(default_factory=list)
    ib_high: float = 0.0
    ib_low: float = 0.0
    ib_width: float = 0.0
    ib_complete: bool = False
    day_high: float = 0.0
    day_low: float = 0.0
    day_open: float = 0.0
    poc_now: float = 0.0
    vah_now: float = 0.0
    val_now: float = 0.0
    day_type: str = DAY_DEVELOPING
    day_type_history: list = field(default_factory=list)
    london_open_type: str = OPEN_UNKNOWN
    ny_open_type: str = OPEN_UNKNOWN
    adr: float = 0.0
    adr_days: int = 0        # completed days the ADR sampled —
    #                          the gauge is only priced when the
    #                          sample is real (>= 10 days)
    adr_consumed_pct: float = 0.0
    adr_state: str = ADR_FRESH
    awr: float = 0.0
    week_consumed_pct: float = 0.0
    gap_size_atr: float = 0.0
    gap_direction: int = 0
    gap_state: str = "none"      # none | open | filled | accepted
    eighty_rule: str = ""        # "" | "traverse_up" | "traverse_down"
    session_levels: dict = field(default_factory=dict)
    inventory: str = ""          # "" | "long" | "short" (one-sided so far)

    def summary(self) -> dict:
        return {
            "day": self.day_key, "type": self.day_type,
            "adr_pct": round(self.adr_consumed_pct, 1),
            "adr_state": self.adr_state,
            "ib": (round(self.ib_low, 2), round(self.ib_high, 2))
            if self.ib_complete else None,
            "london_open": self.london_open_type,
            "ny_open": self.ny_open_type,
            "gap": self.gap_state,
            "eighty": self.eighty_rule or None,
        }


def _session_extremes(day_h1: pd.DataFrame) -> dict:
    """Asia / London ranges within the day (unswept extremes are
    liquidity objects the Location block can draw to)."""
    from ..reader.session_timeframes import session_of
    out: dict = {}
    for sess in ("asia", "london", "newyork"):
        rows = [r for ts, r in day_h1.iterrows()
                if session_of(ts)[0] == sess]
        if rows:
            out[f"{sess}_high"] = max(float(r["high"]) for r in rows)
            out[f"{sess}_low"] = min(float(r["low"]) for r in rows)
    return out


def _gap_state(h1: pd.DataFrame, atr_h1: float) -> tuple:
    """Sunday/weekend gap: Friday close vs this week's first
    open.  Returns (size_atr, direction, state)."""
    if h1 is None or len(h1) < 10 or atr_h1 <= 0:
        return 0.0, 0, "none"
    ny_idx = [pd.Timestamp(ts).tz_convert(NY_TZ) for ts in h1.index]
    # The current week starts at the most recent Sunday 18:00 ET
    # reopen: find the largest index gap > 24h in the last 200
    # bars (the weekend hole).
    idx = h1.index
    start = max(0, len(idx) - 200)
    gap_i = None
    for i in range(len(idx) - 1, start, -1):
        if (idx[i] - idx[i - 1]) > pd.Timedelta(hours=24):
            gap_i = i
            break
    if gap_i is None or gap_i == 0:
        return 0.0, 0, "none"
    fri_close = float(h1["close"].iloc[gap_i - 1])
    week_open = float(h1["open"].iloc[gap_i])
    size = (week_open - fri_close) / atr_h1
    if abs(size) < 0.25:
        return 0.0, 0, "none"
    direction = 1 if size > 0 else -1
    since = h1.iloc[gap_i:]
    # Filled: price traded back through the Friday close.
    if direction > 0:
        filled = bool((since["low"] <= fri_close).any())
    else:
        filled = bool((since["high"] >= fri_close).any())
    if filled:
        return abs(size), direction, "filled"
    closes_beyond = since["close"] - fri_close
    accepted = int((closes_beyond * direction > 0).sum()) >= 2 \
        and bool((closes_beyond.iloc[-1] * direction) > 0) \
        if len(since) else False
    return abs(size), direction, "accepted" if accepted else "open"


def compute_day_context(frames: dict, now=None,
                        prior_history: list | None = None
                        ) -> DayContext:
    """Build the DayContext from the runner's frames.

    ``frames``: {"H1": ..., "D1": ..., "W1": ...} (W1 optional).
    ``prior_history``: the previous context's day_type_history so
    re-typing accumulates across refreshes of the same day.
    """
    ctx = DayContext()
    h1 = frames.get("H1")
    d1 = frames.get("D1")
    if h1 is None or not len(h1):
        return ctx
    days = group_days(h1)
    if not days:
        return ctx
    keys = sorted(days.keys())
    cur_key = keys[-1]
    cur = days[cur_key]
    ctx.day_key = str(cur_key.date())
    ctx.day_open = float(cur["open"].iloc[0])
    ctx.day_high = float(cur["high"].max())
    ctx.day_low = float(cur["low"].min())
    # Initial Balance: the first H1 bar of the day.
    ctx.ib_high = float(cur["high"].iloc[0])
    ctx.ib_low = float(cur["low"].iloc[0])
    ctx.ib_width = ctx.ib_high - ctx.ib_low
    ctx.ib_complete = len(cur) >= 2
    # Intraday value so far.  A degenerate (single-price) day can
    # leave the profile edges undefined — the day object reports
    # 0.0 rather than crashing the refresh.
    vp_now = volume_profile(cur, bins=24)
    ctx.poc_now = float(vp_now["poc"] or 0.0)
    ctx.vah_now = float(vp_now["vah"] or 0.0)
    ctx.val_now = float(vp_now["val"] or 0.0)
    # Fuel gauges.
    ctx.adr = adr_of(d1) if d1 is not None else 0.0
    if d1 is not None and ctx.adr > 0:
        ctx.adr_days = int(min(len(d1), 14))
    if ctx.adr <= 0 and len(keys) >= 3:
        # No D1 frame: ADR from the grouped H1 days themselves
        # (prior completed days only, the current day excluded).
        prior_ranges = [
            float(days[k]["high"].max() - days[k]["low"].min())
            for k in keys[-15:-1]]
        if prior_ranges:
            ctx.adr = float(sum(prior_ranges) / len(prior_ranges))
            ctx.adr_days = len(prior_ranges)
    # Gauge sanity: a real daily range spans many H1 bars — an
    # ADR below half the H1 ATR cannot be a real daily range
    # (broken/stub D1 sample), and a gauge that cannot be read is
    # never priced (adr_days = 0 disarms every consumer).
    if ctx.adr > 0:
        # Guard source is live.multitf.atr14, the engine's own
        # ATR.  A previous revision imported it from
        # features.indicators — a module that never existed —
        # and the bare except below it swallowed the
        # ModuleNotFoundError, so this guard silently NEVER ran
        # and a stub D1 sample read thousands of percent
        # consumed.  The import is now top-level, so any future
        # breakage fails loudly at module load instead of
        # disarming the guard quietly.
        _a = float(atr14(h1)) if h1 is not None else 0.0
        if _a > 0 and ctx.adr < 0.5 * _a:
            ctx.adr_days = 0
    day_range = ctx.day_high - ctx.day_low
    ctx.adr_consumed_pct = (100.0 * day_range / ctx.adr) \
        if ctx.adr > 0 else 0.0
    ctx.adr_state = adr_state(ctx.adr_consumed_pct)
    w1 = frames.get("W1")
    ctx.awr = awr_of(w1) if w1 is not None else 0.0
    if w1 is not None and len(w1):
        week_range = float(w1["high"].iloc[-1] - w1["low"].iloc[-1])
        ctx.week_consumed_pct = (100.0 * week_range / ctx.awr) \
            if ctx.awr > 0 else 0.0
    # Prior day + week.
    if len(keys) >= 2:
        ctx.prior = _day_levels(days[keys[-2]])
    if w1 is not None and len(w1) >= 2:
        ctx.prior_week = {
            "high": float(w1["high"].iloc[-2]),
            "low": float(w1["low"].iloc[-2]),
            "close": float(w1["close"].iloc[-2]),
        }
    # Naked POCs: POCs of up to 5 prior days that no later day
    # has traded through (price never reached the POC since).
    naked: list = []
    day_keys = keys[:-1][-5:]
    for j, k in enumerate(day_keys):
        lv = _day_levels(days[k])
        poc = lv["poc"]
        traded = False
        for k2 in day_keys[j + 1:] + [cur_key]:
            d2 = days[k2]
            if float(d2["low"].min()) <= poc <= float(
                    d2["high"].max()):
                traded = True
                break
        if not traded:
            naked.append(round(poc, 2))
    ctx.naked_pocs = naked
    # Day type (with late-break detection for double
    # distribution: the one-sided extension first reached 2x IB
    # only after the day had printed >= 6 bars range-like).
    late_break = False
    if ctx.ib_width > 0 and len(cur) >= 6:
        mid = cur.iloc[: max(2, len(cur) // 2)]
        mid_up = (float(mid["high"].max()) - ctx.ib_high) \
            / ctx.ib_width
        mid_dn = (ctx.ib_low - float(mid["low"].min())) \
            / ctx.ib_width
        now_up = (ctx.day_high - ctx.ib_high) / ctx.ib_width
        now_dn = (ctx.ib_low - ctx.day_low) / ctx.ib_width
        late_break = (max(mid_up, mid_dn) < 1.0) and \
            (max(now_up, now_dn) >= 2.0)
    ctx.day_type = classify_day_type(
        ib_width=ctx.ib_width, day_high=ctx.day_high,
        day_low=ctx.day_low, ib_high=ctx.ib_high,
        ib_low=ctx.ib_low, adr=ctx.adr, poc=ctx.poc_now,
        late_break=late_break)
    hist = list(prior_history or [])
    if not hist or hist[-1] != ctx.day_type:
        hist.append(ctx.day_type)
    ctx.day_type_history = hist
    # Opening types at the London and NY checkpoints.
    if ctx.prior:
        from ..reader.session_timeframes import session_of
        for sess_name, attr in (("london", "london_open_type"),
                                ("newyork", "ny_open_type")):
            rows = [(ts, r) for ts, r in cur.iterrows()
                    if session_of(ts)[0] == sess_name]
            if rows:
                seg = pd.DataFrame([r for _, r in rows])
                setattr(ctx, attr, classify_open(
                    open_price=float(seg["open"].iloc[0]),
                    first_closes=[float(c) for c in
                                  seg["close"].iloc[:2]],
                    period_high=float(seg["high"].max()),
                    period_low=float(seg["low"].min()),
                    prior_vah=ctx.prior["vah"],
                    prior_val=ctx.prior["val"],
                    ib_width=ctx.ib_width))
    # 80% rule: opened outside prior VA, re-entered within the
    # first two bars -> expect a traverse to the far VA edge.
    if ctx.prior and len(cur) >= 2:
        op = ctx.day_open
        first2 = cur.iloc[:2]
        if op > ctx.prior["vah"] and bool(
                (first2["close"] < ctx.prior["vah"]).any()):
            ctx.eighty_rule = "traverse_down"
        elif op < ctx.prior["val"] and bool(
                (first2["close"] > ctx.prior["val"]).any()):
            ctx.eighty_rule = "traverse_up"
    # Gap state.
    try:
        atr_h1 = float((h1["high"] - h1["low"]).tail(20).mean())
    except Exception:
        atr_h1 = 0.0
    g = _gap_state(h1, atr_h1)
    ctx.gap_size_atr, ctx.gap_direction, ctx.gap_state = g
    # Session extremes + inventory colour.
    try:
        ctx.session_levels = _session_extremes(cur)
    except Exception:
        ctx.session_levels = {}
    if ctx.prior:
        # One-sided overnight inventory: the day opened and held
        # outside prior value on one side.
        if ctx.day_open > ctx.prior["vah"] and \
                ctx.day_low > ctx.prior["val"]:
            ctx.inventory = "long"
        elif ctx.day_open < ctx.prior["val"] and \
                ctx.day_high < ctx.prior["vah"]:
            ctx.inventory = "short"
    return ctx
