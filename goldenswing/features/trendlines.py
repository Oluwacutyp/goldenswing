"""Auto-trendlines from CONFIRMED fractal swings (Phase 4).

A support line is drawn through two HIGHER swing lows, a resistance
line through two LOWER swing highs — the geometry a trader draws by
hand.  A line becomes *known* on the confirmation bar of its second
anchor, so everything downstream is causal: anchors are confirmed
swings, the between-anchor validation only inspects bars up to the
second anchor, and touches/breaks are evaluated bar-by-bar from the
creation bar onward.

Lifecycle (one state machine per line)::

    FORMING --(retest touch)--> ACTIVE --(close through)--> BROKEN
       |                          |
       +----(age / drift)---------+----> EXPIRED

- **FORMING**: two anchors, never retested.  Two points are a guess.
- **ACTIVE**: at least one retest touch since creation — the third
  contact is what validates a line in trader practice.
- **Touch / bounce**: the bar's wick reaches the line (within
  ``tl_tol_atr``) and the close stays on the live side.  A touch whose
  close decisively rejects the line (>= 0.25 ATR back on the live
  side) is a **bounce**.  Touches re-arm only after price spends
  ``tl_rearm_bars`` away from the line, so one approach = one touch.
- **BROKEN**: a close through the line by more than the tolerance.
  Terminal; the break is an EVENT on that bar (with displacement it
  is a structure event — see gating.py).
- **EXPIRED**: older than ``tl_max_age`` bars, or the line has
  drifted more than ``tl_max_drift_atr`` from price.  Silent.

Near-parallel duplicates are suppressed at creation (same direction,
line price within 0.25 ATR at the creation bar and slope within 25%):
the older line keeps ownership of that geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import ohlcv_arrays
from .config import FeatureConfig
from .state import Trendline, TrendlineEvent, TrendlinesState


@dataclass
class _Line:
    direction: int
    i1: int
    p1: float
    i2: int
    p2: float
    slope: float
    created_at: int
    touches: int = 0
    last_touch_at: int | None = None
    state: str = "FORMING"
    bars_away: int = 10 ** 9     # bars since price was last near

    def price_at(self, t: int) -> float:
        return self.p1 + self.slope * (t - self.i1)

    def snapshot(self, state: str | None = None) -> Trendline:
        return Trendline(
            direction=self.direction, anchor1_index=self.i1,
            anchor1_price=self.p1, anchor2_index=self.i2,
            anchor2_price=self.p2, slope=self.slope,
            created_at=self.created_at, touches=self.touches,
            last_touch_at=self.last_touch_at,
            state=state or self.state)


class _Builder:
    """Streaming trendline builder/manager (creation and lifecycle
    kept separate, mirroring the MQL5 builder/manager split)."""

    def __init__(self, df: pd.DataFrame, cfg: FeatureConfig):
        self.cfg = cfg
        a = ohlcv_arrays(df)
        self.high, self.low, self.close = a["high"], a["low"], a["close"]
        self.stream: list[tuple] = []        # (kind, price, index, conf)
        self.seen: set[tuple] = set()
        self.lines: list[_Line] = []

    # ------------------------------------------------------------- ingest
    def _ingest(self, s) -> list[tuple]:
        """Newly confirmed swings at state ``s`` (returned so the
        builder can try to anchor lines on them)."""
        new = []
        for sp in list(s.recent_swing_highs) + list(s.recent_swing_lows):
            key = (sp.kind, sp.index)
            if key not in self.seen and sp.confirmed_at <= s.index:
                self.seen.add(key)
                entry = (sp.kind, float(sp.price), int(sp.index),
                         int(sp.confirmed_at))
                self.stream.append(entry)
                new.append(entry)
        return new

    def extend(self, df: pd.DataFrame) -> None:
        """Point the OHLC arrays at a longer frame (live loop).

        The builder snapshots numpy arrays at construction; a growing
        live frame needs them re-pointed before ``update`` is called
        for the new bars.  Purely additive — batch construction is
        untouched.
        """
        a = ohlcv_arrays(df)
        self.high, self.low, self.close = a["high"], a["low"], a["close"]
        if hasattr(self, "open"):
            self.open = a["open"]

    # ----------------------------------------------------------- creation
    def _try_create(self, t: int, swing: tuple, atr: float) -> None:
        cfg = self.cfg
        kind, p2, i2, _conf = swing
        d = 1 if kind == "low" else -1
        priors = [x for x in self.stream
                  if x[0] == kind and x[2] < i2][-12:]
        for _k, p1, i1, _c in reversed(priors):
            if i2 - i1 < cfg.tl_min_sep:
                continue
            if d == 1 and not p2 > p1:
                continue                    # support needs HIGHER lows
            if d == -1 and not p2 < p1:
                continue                    # resistance needs LOWER highs
            slope = (p2 - p1) / (i2 - i1)
            if not (cfg.tl_slope_min_atr * atr <= abs(slope)
                    <= cfg.tl_slope_max_atr * atr):
                continue
            if not self._valid_between(i1, p1, slope, d, i2, atr):
                continue
            if self._is_duplicate(t, d, i1, p1, slope, atr):
                continue
            self.lines.append(_Line(
                direction=d, i1=i1, p1=p1, i2=i2, p2=p2, slope=slope,
                created_at=t))
            self._cap_lines(d)
            return                          # one line per new swing

    def _valid_between(self, i1: int, p1: float, slope: float, d: int,
                       i2: int, atr: float) -> bool:
        """No close (and no extreme) may violate the line between the
        anchors — a line price already cut through was never a line."""
        tol = self.cfg.tl_tol_atr * atr
        for k in range(i1 + 1, i2):
            lp = p1 + slope * (k - i1)
            if d == 1 and (self.low[k] < lp - tol
                           or self.close[k] < lp - tol):
                return False
            if d == -1 and (self.high[k] > lp + tol
                            or self.close[k] > lp + tol):
                return False
        return True

    def _is_duplicate(self, t: int, d: int, i1: int, p1: float,
                      slope: float, atr: float) -> bool:
        """Near-parallel check: an existing live line in the same
        direction whose price at bar ``t`` is within 0.25 ATR of the
        candidate's AND whose slope is within 25% owns that geometry
        already — the candidate is suppressed."""
        cand_price = p1 + slope * (t - i1)
        for ln in self.lines:
            if ln.direction != d:
                continue
            if abs(ln.price_at(t) - cand_price) <= 0.25 * atr \
                    and abs(ln.slope - slope) \
                    <= 0.25 * max(abs(slope), 1e-12):
                return True
        return False

    def _cap_lines(self, d: int) -> None:
        same = [ln for ln in self.lines if ln.direction == d]
        if len(same) <= self.cfg.tl_max_lines:
            return
        same.sort(key=lambda ln: (ln.touches, ln.created_at),
                  reverse=True)
        keep = set(id(ln) for ln in same[:self.cfg.tl_max_lines])
        self.lines = [ln for ln in self.lines
                      if ln.direction != d or id(ln) in keep]

    # ---------------------------------------------------------- lifecycle
    def update(self, t: int, s) -> TrendlinesState:
        atr = s.atr if s.atr > 0 else 1e-9
        for swing in self._ingest(s):
            self._try_create(t, swing, atr)

        tol = self.cfg.tl_tol_atr * atr
        events: list[TrendlineEvent] = []
        snapshot: list[Trendline] = []
        survivors: list[_Line] = []
        for ln in self.lines:
            lp = ln.price_at(t)
            if t - ln.created_at > self.cfg.tl_max_age \
                    or abs(self.close[t] - lp) > self.cfg.tl_max_drift_atr * atr:
                continue                    # EXPIRED, silently
            if ln.direction == 1:
                broken = self.close[t] < lp - tol
                near = self.low[t] <= lp + tol and not broken
            else:
                broken = self.close[t] > lp + tol
                near = self.high[t] >= lp - tol and not broken
            if broken:
                events.append(TrendlineEvent(
                    kind="break", direction=ln.direction, line_price=lp,
                    touches=ln.touches))
                snapshot.append(ln.snapshot(state="BROKEN"))
                continue
            if near:
                if ln.bars_away >= self.cfg.tl_rearm_bars \
                        and t > ln.created_at:
                    ln.touches += 1
                    ln.last_touch_at = t
                    ln.state = "ACTIVE"
                    rejection = (self.close[t] - lp) * ln.direction
                    kind = "bounce" if rejection >= 0.25 * atr else "touch"
                    events.append(TrendlineEvent(
                        kind=kind, direction=ln.direction,
                        line_price=lp, touches=ln.touches))
                ln.bars_away = 0
            else:
                ln.bars_away += 1
            survivors.append(ln)
            snapshot.append(ln.snapshot())
        self.lines = survivors
        return TrendlinesState(active=tuple(snapshot),
                               events=tuple(events))


def compute(df: pd.DataFrame, states, cfg: FeatureConfig
            ) -> list[TrendlinesState]:
    builder = _Builder(df, cfg)
    return [builder.update(t, s) for t, s in enumerate(states)]
