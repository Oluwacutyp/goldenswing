"""RANGE regime: D1 flat is a range, not a shutdown.

Contract (2026-10-16, owner: "traders don't wait, they read every
candle"):

- D1 structure bias != 0 -> TREND regime.  The existing trend
  playbooks run, D1 supplies direction, H4 supplies the POI zone.
- D1 structure bias == 0 -> RANGE regime.  The engine is NOT blind:
  it looks for a defined H4/D1 range (clear high + low, >= 2 touches
  each side, width >= 1.5x ATR14, still intact) and arms the RANGE
  playbook set at the edges.  Only when NO defined range exists does
  the engine HOLD, with reason ``no defined range``.
- Range playbooks NEVER fire in TREND regime; trend playbooks NEVER
  fire in RANGE regime.  Regime gating lives in
  ``confluence.ConfluenceEngine.gate_live_signal`` (one choke point)
  and is enforced on the ``regime_scope`` each playbook declares.
- Range trades use the SAME gates as trend trades: score >= 60,
  confluence, manipulation filter, cost gate, margin/shield.  Nothing
  is loosened — the playbook set changes, not the bar.  Range size is
  0.5x (choppy regime = smaller), applied at order placement via the
  verdict's ``size_mult``.

Range edges come from CONFIRMED H4 swings (same ``find_swings`` /
``swing_n`` as the trend POI zones), so a range edge is the same
kind of object as a trend POI — a marked level, never a fitted line.
Touches are counted on H1 bars (the trigger TF): a touch is an H1
bar whose high/low came within 0.25x ATR14(H1) of the edge level.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .multitf import atr14, find_swings

MIN_TOUCHES = 2          # touches per side for a "defined" range
MIN_WIDTH_ATR = 1.5      # range width must clear this many ATR14(H1)
TOUCH_TOL_ATR = 0.25     # H1 bar counts as a touch within this of edge
MAX_OUTSIDE_CLOSES = 2   # H1 closes outside the band (last 60) before
#                          the range is declared broken / expanding
LOOKBACK_H1 = 120        # H1 bars scanned for touches (~5 days)
COMPRESS_ATR_RATIO = 0.75  # recent ATR < this x longer ATR = compressed
RANGE_SIZE_MULT = 0.5    # range trades are half size (choppy regime)


@dataclass(frozen=True)
class RangeDef:
    """A defined range: edges, touches, width, compression."""

    high: float
    low: float
    mid: float
    width: float
    atr: float  # ATR14(H1) — the measuring unit for the range
    touches_high: int
    touches_low: int
    compressed: bool
    high_index: int
    low_index: int

    def to_dict(self) -> dict:
        return {
            "high": self.high, "low": self.low, "mid": self.mid,
            "width": self.width, "atr": self.atr,
            "touches_high": self.touches_high,
            "touches_low": self.touches_low,
            "compressed": self.compressed,
            "high_index": self.high_index, "low_index": self.low_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RangeDef":
        return cls(high=float(d["high"]), low=float(d["low"]),
                   mid=float(d["mid"]), width=float(d["width"]),
                   atr=float(d["atr"]),
                   touches_high=int(d["touches_high"]),
                   touches_low=int(d["touches_low"]),
                   compressed=bool(d["compressed"]),
                   high_index=int(d["high_index"]),
                   low_index=int(d["low_index"]))

    def describe(self) -> str:
        return (
            f"RANGE {self.low:.1f}–{self.high:.1f} "
            f"(width {self.width:.1f} = {self.width / self.atr:.1f}x ATR, "
            f"touches H{self.touches_high}/L{self.touches_low}, "
            f"mid {self.mid:.1f}"
            + (", compressed" if self.compressed else "") + ")"
        )

    def edge_zone(self, side: str) -> dict:
        """POI band for one edge (same construction as trend POI zones:
        level +/- 0.5x ATR).  ``side`` is ``"high"`` (resistance,
        shorts back to mid) or ``"low"`` (support, longs to mid)."""
        if side == "high":
            level, direction = self.high, -1
            kind = "resistance"
        elif side == "low":
            level, direction = self.low, +1
            kind = "support"
        else:
            raise ValueError(f"side must be 'high'/'low', got {side!r}")
        half = 0.5 * self.atr
        return {"kind": kind, "level": float(level),
                "top": float(level + half),
                "bottom": float(level - half),
                "direction": direction, "side": side,
                "target": float(self.mid)}

    def edges(self) -> list[dict]:
        return [self.edge_zone("high"), self.edge_zone("low")]


def _inside_bars(frame: pd.DataFrame, n: int = 10) -> int:
    """Count inside bars among the last ``n`` bars of ``frame``."""
    tail = frame.tail(n)
    highs = tail["high"].to_numpy(dtype=float)
    lows = tail["low"].to_numpy(dtype=float)
    count = 0
    for i in range(1, len(highs)):
        if highs[i] <= highs[i - 1] and lows[i] >= lows[i - 1]:
            count += 1
    return count


def detect_range(d1, h4, h1, *, swing_n: int = 3,
                 min_touches: int = MIN_TOUCHES,
                 min_width_atr: float = MIN_WIDTH_ATR) -> RangeDef | None:
    """Detect a defined range, or None.

    Edges are the most recent CONFIRMED H4 swing high / swing low
    (``find_swings``, ``swing_n``).  A range is defined iff:

    - both edges exist, high > low;
    - width >= ``min_width_atr`` x ATR14(H1);
    - >= ``min_touches`` H1 touches per side in the last
      ``LOOKBACK_H1`` H1 bars (bar extreme within ``TOUCH_TOL_ATR``
      x ATR of the edge level);
    - still intact: at most ``MAX_OUTSIDE_CLOSES`` H1 closes outside
      the [low, high] band in the last 60 H1 bars (more = expanding
      / broken, not a range).

    ``compressed`` is True when recent ATR (last 20 H1 bars) is below
    ``COMPRESS_ATR_RATIO`` x ATR14(full H1 frame), or >= 3 inside
    bars sit in the last 10 H1 bars.  Compression is a FACT about
    the range, not a trade — the compression-breakout playbook
    reads it.

    ``d1`` is accepted for signature symmetry with the trend path
    (callers pass the whole stack); the range itself is an H4/H1
    object, so D1 is not read.  ``h4`` may be None when only H1 is
    available (playbook ``live_signal``): it is resampled from H1,
    which derives from the same ticks as the real H4 bars.
    """
    if h1 is None or len(h1) < 30:
        return None
    if h4 is None:
        try:
            h4 = h1.resample("4h").agg(
                {"open": "first", "high": "max",
                 "low": "min", "close": "last"})
        except Exception:
            return None
    if h4 is None or len(h4) < 2 * swing_n + 5:
        return None
    highs, lows = find_swings(h4["high"].to_numpy(),
                              h4["low"].to_numpy(), swing_n)
    if not highs or not lows:
        return None
    high_idx, high = highs[-1]
    low_idx, low = lows[-1]
    if not high > low:
        return None
    atr_h1 = atr14(h1)
    if atr_h1 <= 0:
        return None
    width = float(high - low)
    if width < min_width_atr * atr_h1:
        return None
    # Touches on H1: bar extreme within tolerance of the edge level.
    tail = h1.tail(LOOKBACK_H1)
    tol = TOUCH_TOL_ATR * atr_h1
    touches_high = int(
        (tail["high"].to_numpy(dtype=float) >= high - tol).sum())
    touches_low = int(
        (tail["low"].to_numpy(dtype=float) <= low + tol).sum())
    if touches_high < min_touches or touches_low < min_touches:
        return None
    # Still intact: few closes outside the band recently.
    recent = h1.tail(60)
    outside = int(((recent["close"] > high) |
                   (recent["close"] < low)).sum())
    if outside > MAX_OUTSIDE_CLOSES:
        return None
    # Compression: ATR contracting, or a cluster of inside bars.
    recent_atr = atr14(h1.tail(20))
    compressed = (recent_atr < COMPRESS_ATR_RATIO * atr_h1) \
        or _inside_bars(h1, 10) >= 3
    return RangeDef(high=float(high), low=float(low),
                    mid=float((high + low) / 2.0), width=width,
                    atr=float(atr_h1),
                    touches_high=touches_high, touches_low=touches_low,
                    compressed=bool(compressed),
                    high_index=int(high_idx), low_index=int(low_idx))


def resolve_range_touch(range_def: RangeDef,
                        mid_price: float) -> dict | None:
    """Edge whose POI band contains ``mid_price``, or None.

    The band is level +/- 0.5x ATR — the same band the trend trigger
    uses.  A price inside a band is AT that edge; anything else is
    between edges, waiting.
    """
    for edge in range_def.edges():
        if edge["bottom"] <= mid_price <= edge["top"]:
            return edge
    return None


def range_setup(range_def: RangeDef, *, bias_h4: int = 0,
                h4_context: str = "", ref_level: float | None = None,
                atr_h1: float | None = None, buf: float | None = None,
                h4_highs: list | None = None,
                h4_lows: list | None = None,
                swing_n: int = 3) -> dict:
    """A live-setup dict for a RANGE regime (``live_setup`` shape).

    ``direction`` is 0 — a range has no single direction; the tick
    clock resolves the touched edge (``resolve_range_touch``) into
    a per-edge direction/zone before arming the pending.  ``zone``
    is None for the same reason; ``edges`` carries both bands.
    ``size_mult`` is ``RANGE_SIZE_MULT`` (0.5x).
    """
    atr = float(atr_h1 if atr_h1 is not None else range_def.atr)
    return {
        "regime": "range", "bias_d1": 0, "bias_h4": int(bias_h4),
        "h4_context": h4_context, "direction": 0, "zone": None,
        "edges": range_def.edges(), "range": range_def.to_dict(),
        "ref_level": ref_level, "atr_h1": atr,
        "buf": float(buf if buf is not None else 0.5 * atr),
        "h4_highs": list(h4_highs or []),
        "h4_lows": list(h4_lows or []),
        "swing_n": int(swing_n), "size_mult": RANGE_SIZE_MULT,
    }


# ------------------------------------------------------- boundary machine
# Sweep vs breakout at a range edge — the exact boundary moment.
#
# A wick beyond an edge is NOT a trade.  A single close beyond is NOT
# a regime switch.  The machine below is the single place that
# decides, on CLOSED bars only, in sequence:
#
# FAILED SWEEP (mean-reversion): an M5/M15 bar wicks beyond an edge
# by >= SWEEP_WICK_ATR x ATR14 but closes back INSIDE the range, AND
# a LATER M5/M15 bar closes inside + BOS reclaim in the
# mean-reversion direction + an FVG confirm.  Only then is the edge
# sweep-confirmed and the range edge sweep playbook may fire.
# Wick alone = no trade (event ``sweep_wick``, log "sweep wick,
# awaiting reclaim close").
#
# SUCCESSFUL BREAKOUT (regime switch): an H1 bar closes beyond an edge
# by >= BREAKOUT_CLOSE_ATR x ATR14, AND the next H1 bar closes
# beyond again (2 consecutive closes beyond), AND M15 BOS in the
# breakout direction.  Only then does the regime switch RANGE ->
# BREAKOUT_TREND (D1 still flat, so it is NOT a D1 TREND — trend
# playbooks arm in the breakout direction, range playbooks stop).
# Size is 0.5x until a retest of the broken edge holds (price touches
# the broken edge and closes back in the breakout direction), then
# 1.0x.  No retest within RETEST_WINDOW_H1 H1 bars: the breakout
# stands on the 2nd close + BOS alone, still 0.5x until a retest
# holds (a later retest still upgrades the size).
#
# ANTI-WHIPSAW / HYSTERESIS:
# - Regime lock: once the regime switches, it holds a minimum of
#   REGIME_LOCK_H1 H1 bars before it can switch back.  No flip-flop
#   on a single bar.
# - Buffer band: edge +/- BUFFER_ATR x ATR14 is a NO-TRADE zone for
#   a regime switch — only a sweep wick counts there.  A breakout
#   must CLOSE beyond the buffer (beyond + 0.5 ATR), not just touch.
# - False breakout guard: price closes beyond, then closes back
#   inside within the decision window (failed breakout) -> log
#   "false breakout, back to RANGE", regime reverts to RANGE after
#   the lock expires, and the mean-reversion playbook arms on the
#   reclaim (the reclaim close inside IS the sweep confirm).
# - Cooldown: after a false breakout or a Close-Rule invalidation,
#   COOLDOWN_H1 H1 bars before the next range edge trade on the
#   same edge.
#
# All counters are RELATIVE (bars remaining), and cursors are bar
# CLOSE TIMESTAMPS, so the whole machine round-trips through
# ``RegimeState`` (``live/regime_state.py``) and survives a restart
# with the hysteresis clock intact.  A restart never re-counts a
# bar: ``update()`` skips bars at or before the persisted
# ``last_processed_*_close`` timestamps.
SWEEP_WICK_ATR = 0.25     # wick beyond edge to count as a sweep probe
BREAKOUT_CLOSE_ATR = 0.5  # H1 close beyond edge to count as breakout
BUFFER_ATR = 0.25         # no-trade band around an edge (regime switch)
REGIME_LOCK_H1 = 4        # min H1 bars a regime holds before switching
RETEST_WINDOW_H1 = 4      # H1 bars to see a retest after breakout
COOLDOWN_H1 = 2           # H1 bars cooldown per edge after false breakout
SWEEP_WICK_EXPIRY_M15 = 12  # M15 bars a wick waits for reclaim
SWEEP_CONFIRM_EXPIRY_M15 = 12  # M15 bars a sweep confirm stays armed
BREAKOUT_DECIDE_H1 = 2    # H1 bars to decide a beyond-1 close
FVG_LOOKBACK_M15 = 5      # M15 bars scanned for a confirming FVG

EDGE_SIDES = ("high", "low")


@dataclass
class BoundaryEvent:
    """One boundary decision, in order.  ``kind`` values:

    sweep_wick / sweep_confirmed / sweep_expired /
    breakout_pending (1/2 closes, or 2/2 awaiting M15 BOS) /
    breakout_confirmed (regime switch) / breakout_stalled /
    false_breakout / retest_held / retest_missed /
    regime_reverted / cooldown_started.
    """

    kind: str
    edge: str  # "high" / "low" / "" (regime-wide)
    detail: str
    h1_ts: object = None  # H1 bar close timestamp of the deciding bar


def _edge_state() -> dict:
    return {
        "state": "inside",  # inside | wick | sweep_confirmed |
        # beyond1 | beyond2_no_bos | awaiting_retest |
        # retest_held | breakout_no_retest
        "wick_price": None,
        "wick_ts": None,
        "wick_pos": -1,      # positional index of the wick M15 bar
        "wick_age_m15": 0,
        "confirm_age_m15": 0,
        "reclaim_bars": 0,   # M15 bars since wick closing inside
        "beyond_closes": 0,  # consecutive H1 closes beyond the edge
        "decide_remaining": 0,  # H1 bars left to decide a beyond-1
    }


class BoundaryTracker:
    """Sweep-vs-breakout state machine for one defined range.

    Fed with closed H1 + M15 frames (``update``).  The frames are
    the FULL current frames; only bars newer than the last
    processed close timestamp are consumed, so re-feeding the same
    frames (or restarting from persisted state) never double-counts
    a bar.  Pure bar logic — no feed, no orders, no clock.
    """

    def __init__(self, range_def: RangeDef):
        self.range_def = range_def
        self.atr = float(range_def.atr)
        self.regime = "range"  # "range" | "breakout_trend"
        self.breakout_direction = 0
        self.breakout_edge: str | None = None
        self.breakout_id = 0  # increments per confirmed breakout
        self.lock_remaining = 0
        self.retest_remaining = 0
        self.retest_held = False
        self.pending_revert = False
        self.regime_since_ts = None
        self.breakout_first_close_ts = None
        self.sweep_confirmed: set = set()
        self.cooldown_remaining: dict = {"high": 0, "low": 0}
        self.edge: dict = {s: _edge_state() for s in EDGE_SIDES}
        self.last_h1_ts = None
        self.last_m15_ts = None

    # ---------------------------------------------------------- queries
    @property
    def size_mult(self) -> float:
        """Range 0.5x; breakout 0.5x until the retest holds, then 1.0x."""
        if self.regime == "range":
            return RANGE_SIZE_MULT
        return 1.0 if self.retest_held else 0.5

    @property
    def breakout_pending(self) -> bool:
        return any(
            self.edge[s]["state"] in ("beyond1", "beyond2_no_bos")
            for s in EDGE_SIDES)

    def sweep_armed(self, side: str) -> bool:
        return side in self.sweep_confirmed

    def arm_allowed(self, side: str) -> bool:
        """May a range edge trade arm on ``side`` right now?"""
        return self.cooldown_remaining.get(side, 0) <= 0

    def snapshot(self) -> dict:
        """Runner / /status / verdict view of the machine."""
        return {
            "regime": self.regime,
            "breakout_direction": self.breakout_direction,
            "breakout_edge": self.breakout_edge,
            "breakout_id": self.breakout_id,
            "size_mult": self.size_mult,
            "retest_held": self.retest_held,
            "retest_remaining_h1": int(self.retest_remaining),
            "lock_remaining_h1": int(self.lock_remaining),
            "pending_revert": bool(self.pending_revert),
            "sweep_confirmed_edges": sorted(self.sweep_confirmed),
            "cooldown_edges": sorted(
                s for s in EDGE_SIDES
                if self.cooldown_remaining.get(s, 0) > 0),
            "cooldown_remaining_h1": dict(self.cooldown_remaining),
            "edge_states": {
                s: {"state": self.edge[s]["state"],
                    "beyond_closes": self.edge[s]["beyond_closes"],
                    "wick_age_m15": self.edge[s]["wick_age_m15"],
                    "reclaim_bars": self.edge[s]["reclaim_bars"]}
                for s in EDGE_SIDES},
            "range": self.range_def.to_dict(),
        }

    # ------------------------------------------------------- persistence
    def to_regime_state(self):
        """Serialize the hysteresis clock for ``regime_state.py``."""
        from .regime_state import RegimeState

        sweep_pending = []
        for side in EDGE_SIDES:
            st = self.edge[side]
            if st["state"] in ("wick", "sweep_confirmed"):
                sweep_pending.append({
                    "edge": side, "wick_price": st["wick_price"],
                    "wick_ts": (st["wick_ts"].isoformat()
                                if st["wick_ts"] is not None else None),
                    "wick_age_m15": int(st["wick_age_m15"]),
                    "reclaim_bars_confirmed": int(st["reclaim_bars"]),
                    "confirmed": st["state"] == "sweep_confirmed"})
        return RegimeState(
            regime=self.regime,
            regime_since_ts=(
                self.regime_since_ts.isoformat()
                if self.regime_since_ts is not None else None),
            regime_since_bar=(
                self.breakout_first_close_ts.isoformat()
                if self.breakout_first_close_ts is not None else None),
            lock_remaining_h1=int(self.lock_remaining),
            breakout_confirm_count=int(
                self.edge[self.breakout_edge]["beyond_closes"]
                if self.breakout_edge else 0),
            breakout_direction=int(self.breakout_direction),
            breakout_edge=self.breakout_edge,
            breakout_first_close_time=(
                self.breakout_first_close_ts.isoformat()
                if self.breakout_first_close_ts is not None else None),
            sweep_pending=sweep_pending,
            cooldown_remaining_h1=dict(self.cooldown_remaining),
            last_range_high=float(self.range_def.high),
            last_range_low=float(self.range_def.low),
            last_range_mid=float(self.range_def.mid),
            last_range_atr=float(self.range_def.atr),
            last_processed_bar_close=(
                self.last_h1_ts.isoformat()
                if self.last_h1_ts is not None else None),
            last_processed_m15_close=(
                self.last_m15_ts.isoformat()
                if self.last_m15_ts is not None else None),
        )

    def apply_regime_state(self, state) -> None:
        """Restore the hysteresis clock from a verified RegimeState.

        Counters resume EXACTLY (no reset, no fabrication).  Bar
        cursors resume from the persisted last-processed closes,
        so ``update()`` skips already-counted bars.
        """
        self.lock_remaining = int(state.lock_remaining_h1 or 0)
        self.breakout_direction = int(state.breakout_direction or 0)
        self.breakout_edge = state.breakout_edge
        self.regime = str(state.regime or "range")
        self.retest_held = False
        self.pending_revert = False
        self.retest_remaining = 0
        if self.regime == "breakout_trend":
            # A restored breakout is still awaiting its retest
            # unless the persist happened after it held — the
            # retest state itself is not persisted (it resolves
            # on the next H1 bar from price action, conservatively
            # re-checking the touch rather than trusting a flag).
            self.retest_remaining = RETEST_WINDOW_H1
            self.breakout_id = 1  # restored breakout already fired
        if state.breakout_edge in EDGE_SIDES:
            self.edge[state.breakout_edge]["beyond_closes"] = int(
                state.breakout_confirm_count or 0)
            if self.regime == "breakout_trend":
                self.edge[state.breakout_edge]["state"] = \
                    "awaiting_retest"
        for key, val in (state.cooldown_remaining_h1 or {}).items():
            if key in EDGE_SIDES:
                self.cooldown_remaining[key] = int(val or 0)
        for entry in state.sweep_pending or []:
            side = entry.get("edge")
            if side not in EDGE_SIDES:
                continue
            st = self.edge[side]
            st["wick_price"] = entry.get("wick_price")
            st["wick_age_m15"] = int(entry.get("wick_age_m15") or 0)
            st["reclaim_bars"] = int(
                entry.get("reclaim_bars_confirmed") or 0)
            st["state"] = "sweep_confirmed" if entry.get("confirmed") \
                else "wick"
            if entry.get("confirmed"):
                self.sweep_confirmed.add(side)
            if entry.get("wick_ts"):
                try:
                    st["wick_ts"] = pd.Timestamp(entry["wick_ts"])
                except Exception:
                    st["wick_ts"] = None
        if state.last_processed_bar_close:
            try:
                self.last_h1_ts = pd.Timestamp(
                    state.last_processed_bar_close)
            except Exception:
                self.last_h1_ts = None
        if state.last_processed_m15_close:
            try:
                self.last_m15_ts = pd.Timestamp(
                    state.last_processed_m15_close)
            except Exception:
                self.last_m15_ts = None
        if state.breakout_first_close_time:
            try:
                self.breakout_first_close_ts = pd.Timestamp(
                    state.breakout_first_close_time)
            except Exception:
                self.breakout_first_close_ts = None
        if state.regime_since_ts:
            try:
                self.regime_since_ts = pd.Timestamp(
                    state.regime_since_ts)
            except Exception:
                self.regime_since_ts = None

    # ------------------------------------------------------------ feeding
    def update(self, h1, m15) -> list[BoundaryEvent]:
        """Consume newly closed H1 + M15 bars, in time order.

        M15 bars first (sweep wick / reclaim / BOS live on M15),
        then H1 bars (breakout closes / lock / cooldown / retest
        live on H1).  Bars at or before the last-processed close
        are skipped — idempotent across restarts and re-feeds.
        """
        events: list[BoundaryEvent] = []
        if m15 is not None and len(m15):
            events.extend(self._process_m15(m15))
        if h1 is not None and len(h1):
            events.extend(self._process_h1(h1, m15))
        return events

    def _new_bars(self, frame, last_ts):
        """(positions, timestamps) of bars newer than ``last_ts``."""
        out = []
        for pos in range(len(frame)):
            ts = frame.index[pos]
            if last_ts is None or ts > last_ts:
                out.append((pos, ts))
        return out

    def _process_m15(self, m15) -> list[BoundaryEvent]:
        events: list[BoundaryEvent] = []
        atr = self.atr
        for pos, ts in self._new_bars(m15, self.last_m15_ts):
            self.last_m15_ts = ts
            row = m15.iloc[pos]
            try:
                close = float(row["close"])
                high = float(row["high"])
                low = float(row["low"])
            except Exception:
                continue
            prev_close = None
            if pos > 0:
                try:
                    prev_close = float(m15.iloc[pos - 1]["close"])
                except Exception:
                    prev_close = None
            for side in EDGE_SIDES:
                if self.cooldown_remaining.get(side, 0) > 0:
                    continue  # cooling edge: no new probes
                if self.regime != "range" \
                        and side != self.breakout_edge:
                    continue  # breakout: only the broken edge matters
                st = self.edge[side]
                edge_level = self.range_def.high \
                    if side == "high" else self.range_def.low
                if self.regime == "range":
                    # --- sweep wick: probe beyond + close inside.
                    if st["state"] in ("inside",):
                        if side == "high" and prev_close is not None \
                                and prev_close <= edge_level \
                                and high > edge_level \
                                + SWEEP_WICK_ATR * atr \
                                and close <= edge_level:
                            st["state"] = "wick"
                            st["wick_price"] = high
                            st["wick_ts"] = ts
                            st["wick_pos"] = pos
                            st["wick_age_m15"] = 0
                            st["reclaim_bars"] = 0
                            events.append(BoundaryEvent(
                                "sweep_wick", side,
                                "sweep wick, awaiting reclaim close",
                                self.last_h1_ts))
                    # --- reclaim: a LATER bar closes inside + BOS + FVG.
                    elif st["state"] == "wick" and pos > st["wick_pos"]:
                        st["wick_age_m15"] += 1
                        inside = (close <= edge_level) \
                            if side == "high" \
                            else (close >= edge_level)
                        if inside:
                            st["reclaim_bars"] += 1
                            direction = -1 if side == "high" else +1
                            bos = self._m15_bos(
                                m15, direction, close_price=close,
                                upto_pos=st["wick_pos"])
                            fvg = self._has_fvg(m15, direction,
                                                upto_pos=pos)
                            if bos and fvg:
                                st["state"] = "sweep_confirmed"
                                st["confirm_age_m15"] = 0
                                self.sweep_confirmed.add(side)
                                events.append(BoundaryEvent(
                                    "sweep_confirmed", side,
                                    "sweep reclaim confirmed "
                                    "(M15 BOS + FVG)", self.last_h1_ts))
                        if st["wick_age_m15"] > SWEEP_WICK_EXPIRY_M15 \
                                and st["state"] == "wick":
                            st["state"] = "inside"
                            st["wick_price"] = None
                            events.append(BoundaryEvent(
                                "sweep_expired", side,
                                "sweep wick expired, no reclaim",
                                self.last_h1_ts))
                    elif st["state"] == "sweep_confirmed":
                        st["confirm_age_m15"] += 1
                        if st["confirm_age_m15"] \
                                > SWEEP_CONFIRM_EXPIRY_M15:
                            st["state"] = "inside"
                            st["wick_price"] = None
                            self.sweep_confirmed.discard(side)
                            events.append(BoundaryEvent(
                                "sweep_expired", side,
                                "sweep confirm expired",
                                self.last_h1_ts))
                elif self.regime == "breakout_trend" \
                        and side == self.breakout_edge \
                        and st["state"] == "beyond2_no_bos":
                    # 2nd H1 close beyond already printed; the M15
                    # BOS is the last confirmation piece.
                    if self._m15_bos(m15, self.breakout_direction,
                                     close_price=close,
                                     upto_pos=pos):
                        self._confirm_breakout(ts, events)
        return events

    def _process_h1(self, h1, m15) -> list[BoundaryEvent]:
        events: list[BoundaryEvent] = []
        atr = self.atr
        for pos, ts in self._new_bars(h1, self.last_h1_ts):
            self.last_h1_ts = ts
            row = h1.iloc[pos]
            try:
                close = float(row["close"])
                high = float(row["high"])
                low = float(row["low"])
            except Exception:
                continue
            # --- hysteresis clocks tick on H1 closes.
            if self.lock_remaining > 0:
                self.lock_remaining -= 1
            for side in EDGE_SIDES:
                if self.cooldown_remaining.get(side, 0) > 0:
                    self.cooldown_remaining[side] -= 1
            if self.pending_revert and self.lock_remaining <= 0:
                self._revert_to_range(ts, events,
                                      "lock expired after false "
                                      "breakout")
            if self.regime == "breakout_trend":
                self._process_h1_breakout(h1, pos, ts, close, high,
                                          low, events)
            else:
                self._process_h1_range(ts, close, high, low, events,
                                       m15=m15)
            # Retest window: no retest within RETEST_WINDOW_H1 bars
            # -> the breakout stands on 2nd close + BOS alone, still
            # 0.5x until a retest holds (a later retest upgrades).
            if self.regime == "breakout_trend" \
                    and not self.retest_held \
                    and self.retest_remaining > 0:
                self.retest_remaining -= 1
                if self.retest_remaining == 0:
                    st = self.edge[self.breakout_edge]
                    if st["state"] == "awaiting_retest":
                        st["state"] = "breakout_no_retest"
                    events.append(BoundaryEvent(
                        "retest_missed", self.breakout_edge or "",
                        "no retest within 4 H1 bars — breakout "
                        "stands on 2nd close + BOS, size stays 0.5x",
                        ts))
        return events

    def _process_h1_range(self, ts, close, high, low, events,
                          *, m15=None) -> None:
        """H1 logic while regime == RANGE: breakout counting."""
        atr = self.atr
        for side in EDGE_SIDES:
            st = self.edge[side]
            edge_level = self.range_def.high \
                if side == "high" else self.range_def.low
            beyond = (close > edge_level + BREAKOUT_CLOSE_ATR * atr) \
                if side == "high" else \
                (close < edge_level - BREAKOUT_CLOSE_ATR * atr)
            inside = (close <= edge_level) if side == "high" \
                else (close >= edge_level)
            if st["state"] in ("inside", "wick", "sweep_confirmed"):
                if beyond:
                    st["state"] = "beyond1"
                    st["beyond_closes"] = 1
                    st["decide_remaining"] = BREAKOUT_DECIDE_H1
                    if self.breakout_first_close_ts is None:
                        self.breakout_first_close_ts = ts
                    events.append(BoundaryEvent(
                        "breakout_pending", side,
                        "close beyond 1/2, awaiting 2nd close", ts))
            elif st["state"] == "beyond1":
                if beyond:
                    st["beyond_closes"] = 2
                    # M15 BOS decides now if already printed (the
                    # M15 bars of this same update ran first), else
                    # the machine waits in beyond2_no_bos.
                    bos = False
                    if m15 is not None and len(m15):
                        bos = self._m15_bos(
                            m15, +1 if side == "high" else -1,
                            close_price=float(m15.iloc[-1]["close"]),
                            upto_pos=len(m15) - 1)
                    if bos:
                        st["state"] = "beyond2_no_bos"
                        self._confirm_breakout(ts, events)
                    else:
                        st["state"] = "beyond2_no_bos"
                        st["decide_remaining"] = BREAKOUT_DECIDE_H1
                        events.append(BoundaryEvent(
                            "breakout_pending", side,
                            "close beyond 2/2, awaiting M15 BOS", ts))
                elif inside:
                    self._false_breakout(side, ts, events,
                                        "close back inside after "
                                        "1 beyond close")
                else:
                    # Close inside the buffer band: neither beyond
                    # nor inside — no decision, window shrinks.
                    st["decide_remaining"] -= 1
                    if st["decide_remaining"] <= 0:
                        st["state"] = "inside"
                        st["beyond_closes"] = 0
                        events.append(BoundaryEvent(
                            "breakout_stalled", side,
                            "beyond-1 close never confirmed "
                            "within window", ts))
            elif st["state"] == "beyond2_no_bos":
                if inside:
                    self._false_breakout(side, ts, events,
                                        "close back inside before "
                                        "M15 BOS")
                else:
                    st["decide_remaining"] -= 1
                    if st["decide_remaining"] <= 0:
                        st["state"] = "inside"
                        st["beyond_closes"] = 0
                        events.append(BoundaryEvent(
                            "breakout_stalled", side,
                            "M15 BOS never arrived within window",
                            ts))

    def _process_h1_breakout(self, h1, pos, ts, close, high, low,
                             events) -> None:
        """H1 logic while regime == BREAKOUT_TREND: retest / failure."""
        atr = self.atr
        side = self.breakout_edge
        if side not in EDGE_SIDES:
            return
        st = self.edge[side]
        edge_level = self.range_def.high \
            if side == "high" else self.range_def.low
        if self.retest_held:
            return  # retest held: breakout genuine, no revert
        touches_band = (low <= edge_level + BUFFER_ATR * atr
                        and high >= edge_level - BUFFER_ATR * atr)
        closes_breakout_way = (close > edge_level) \
            if side == "high" else (close < edge_level)
        closes_inside = (close < edge_level - BUFFER_ATR * atr) \
            if side == "high" else \
            (close > edge_level + BUFFER_ATR * atr)
        if touches_band and closes_breakout_way:
            self.retest_held = True
            st["state"] = "retest_held"
            events.append(BoundaryEvent(
                "retest_held", side,
                "retest of broken edge held — size 0.5x -> 1.0x",
                ts))
        elif closes_inside:
            self._false_breakout(side, ts, events,
                                 "close back inside after breakout")

    def _confirm_breakout(self, ts, events) -> None:
        """2 closes beyond + M15 BOS: RANGE -> BREAKOUT_TREND."""
        side = None
        for s in EDGE_SIDES:
            if self.edge[s]["state"] == "beyond2_no_bos":
                side = s
                break
        if side is None:
            return
        direction = +1 if side == "high" else -1
        self.regime = "breakout_trend"
        self.breakout_direction = direction
        self.breakout_edge = side
        self.breakout_id += 1
        self.lock_remaining = REGIME_LOCK_H1
        self.retest_remaining = RETEST_WINDOW_H1
        self.retest_held = False
        self.pending_revert = False
        self.regime_since_ts = ts
        self.edge[side]["state"] = "awaiting_retest"
        self.sweep_confirmed.discard(side)
        events.append(BoundaryEvent(
            "breakout_confirmed", side,
            f"RANGE -> BREAKOUT_TREND {direction:+d} "
            f"(2 closes beyond + M15 BOS), lock {REGIME_LOCK_H1} H1, "
            f"size 0.5x until retest", ts))

    def _false_breakout(self, side: str, ts, events,
                        reason: str) -> None:
        """Close beyond, then close back inside: failed breakout."""
        self.cooldown_remaining[side] = COOLDOWN_H1
        # The reclaim close inside IS the sweep confirm: the
        # mean-reversion playbook arms on the reclaim (sweep logic).
        self.sweep_confirmed.add(side)
        st = self.edge[side]
        st["state"] = "inside"
        st["beyond_closes"] = 0
        st["decide_remaining"] = 0
        st["wick_price"] = None
        events.append(BoundaryEvent(
            "false_breakout", side,
            f"false breakout, back to RANGE ({reason}); "
            f"cooldown {COOLDOWN_H1} H1 on this edge, "
            f"mean-reversion arms on reclaim", ts))
        events.append(BoundaryEvent(
            "cooldown_started", side,
            f"cooldown {COOLDOWN_H1} H1 on {side} edge", ts))
        if self.regime == "breakout_trend":
            if self.lock_remaining > 0:
                # Lock holds the regime: revert when it expires.
                self.pending_revert = True
            else:
                self._revert_to_range(ts, events, reason)

    def _revert_to_range(self, ts, events, reason: str) -> None:
        self.regime = "range"
        self.breakout_direction = 0
        self.breakout_edge = None
        self.retest_held = False
        self.retest_remaining = 0
        self.pending_revert = False
        self.breakout_first_close_ts = None
        events.append(BoundaryEvent(
            "regime_reverted", "",
            f"false breakout, back to RANGE ({reason})", ts))

    def note_invalidation(self, side: str) -> BoundaryEvent | None:
        """Close-Rule invalidation on an edge: cooldown, sweep dead.

        Called by the runner when an armed touch on this edge fails
        the Close Rule.  Returns the cooldown event (or None for a
        bad side).  The invalidation itself is journalled by the
        caller; this only arms the boundary cooldown.
        """
        if side not in EDGE_SIDES:
            return None
        self.cooldown_remaining[side] = COOLDOWN_H1
        self.sweep_confirmed.discard(side)
        self.edge[side]["state"] = "inside"
        self.edge[side]["wick_price"] = None
        return BoundaryEvent(
            "cooldown_started", side,
            f"cooldown {COOLDOWN_H1} H1 on {side} edge "
            f"(Close-Rule invalidation)", self.last_h1_ts)

    # ------------------------------------------------------------ helpers
    def _m15_bos(self, m15, direction: int, *, close_price: float,
                 upto_pos: int) -> bool:
        """M15 break of structure in ``direction`` at ``close_price``.

        The reference swing is the last CONFIRMED M15 swing from
        bars up to ``upto_pos`` (the wick bar for a reclaim, the
        latest bar for a breakout).  ``find_swings`` only returns
        confirmed swings (3 bars each side), so the reference can
        never be the forming / unconfirmed extreme.
        """
        if upto_pos is None or upto_pos < 2 * 3 + 5:
            return False
        try:
            sub = m15.iloc[:upto_pos + 1]
            highs, lows = find_swings(
                sub["high"].to_numpy(dtype=float),
                sub["low"].to_numpy(dtype=float), 3)
        except Exception:
            return False
        if direction > 0:
            return bool(highs) and close_price > highs[-1][1]
        return bool(lows) and close_price < lows[-1][1]

    def _has_fvg(self, m15, direction: int, *, upto_pos: int) -> bool:
        """Unfilled fair-value gap in ``direction`` ending at ``upto_pos``.

        Bearish (direction -1): a bar's low sits above a later bar's
        high (gap down, ``low[i-1] > high[i+1]``).  Bullish: the
        mirror (``high[i-1] < low[i+1]``).  Scanned over the last
        FVG_LOOKBACK_M15 M15 bars — the gap must belong to the
        reclaim move, not ancient history.
        """
        start = max(0, upto_pos - (FVG_LOOKBACK_M15 - 1))
        try:
            window = m15.iloc[start:upto_pos + 1]
            highs = window["high"].to_numpy(dtype=float)
            lows = window["low"].to_numpy(dtype=float)
        except Exception:
            return False
        for i in range(1, len(highs) - 1):
            if direction < 0 and lows[i - 1] > highs[i + 1]:
                return True
            if direction > 0 and highs[i - 1] < lows[i + 1]:
                return True
        return False
