"""Classical chart patterns from CONFIRMED fractal swings only.

Definitions (swing-based, tolerances in ATR units):

- **Double top / bottom** — two consecutive swing highs within
  ``pat_tol_atr`` of each other, a swing low between them at least
  ``pat_min_valley_atr`` below the lower high.  Neckline = that low.
  Mirror for bottoms.
- **Head & shoulders** — three consecutive swing highs H1, H2, H3 with
  H2 (head) strictly highest, head at least ``pat_min_depth_atr``
  above the neckline, and shoulders symmetric:
  ``|H1 - H3| <= max(pat_shoulder_tol_atr * ATR, pat_shoulder_frac *
  (H2 - neckline))``.  Neckline = the lower of the two intervening
  swing lows (horizontal-neckline simplification, documented).
  Mirror for inverse H&S.

A pattern is *formed* on the bar its final shoulder/top confirms
(reader confirmation bar — causal), becomes *broken* when a close
crosses the neckline in the pattern's direction, and dies on a close
through its invalidation (the head/twin extreme) or after
``pat_max_age`` bars.  Target = measured move (pattern height
projected from the neckline).
"""
from __future__ import annotations

import pandas as pd

from .config import FeatureConfig
from .state import PatternInstance, PatternsState


class _Tracker:
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self.stream: list[tuple] = []   # (kind, price, index, confirmed_at)
        self.seen_swings: set[tuple] = set()
        self.formed_keys: set[tuple] = set()
        self.live: list[dict] = []      # mutable pattern records

    # -- swing stream ----------------------------------------------------
    def _ingest(self, s) -> None:
        # Swings confirm in index order (uniform fractal lag), so plain
        # appends keep the stream sorted by swing index.
        for sp in list(s.recent_swing_highs) + list(s.recent_swing_lows):
            key = (sp.kind, sp.index)
            if key not in self.seen_swings and sp.confirmed_at <= s.index:
                self.seen_swings.add(key)
                self.stream.append((sp.kind, sp.price, sp.index,
                                    sp.confirmed_at))

    def _last(self, kind: str, k: int) -> list[tuple]:
        out: list[tuple] = []
        for x in reversed(self.stream):
            if x[0] == kind:
                out.append(x)
                if len(out) == k:
                    break
        return out[::-1]

    def _highs(self):
        return self._last("high", 3)

    def _lows(self):
        return self._last("low", 3)

    def _valley(self, i1: int, i2: int, kind: str):
        """Extreme swing of ``kind`` strictly between swing indices
        i1 and i2 (deepest low / highest high)."""
        best = None
        for x in reversed(self.stream):
            if x[2] <= i1:
                break
            if x[0] == kind and x[2] < i2:
                if best is None or (x[1] < best[1] if kind == "low"
                                    else x[1] > best[1]):
                    best = x
        return best

    # -- formation ---------------------------------------------------------
    def _try_form(self, t: int, atr: float) -> PatternInstance | None:
        cfg = self.cfg
        highs, lows = self._highs(), self._lows()
        formed: PatternInstance | None = None

        def consider(kind, direction, key, neckline, extreme, formed_at):
            nonlocal formed
            if key in self.formed_keys or formed_at != t:
                return
            self.formed_keys.add(key)
            height = abs(extreme - neckline)
            inst = PatternInstance(
                kind=kind, direction=direction, status="active",
                formed_at=formed_at, neckline=float(neckline),
                extreme=float(extreme), invalidation=float(extreme),
                target=float(neckline + direction * height))
            self.live.append({"inst": inst, "broken": False})
            formed = inst

        if len(highs) >= 3:
            h1, h2, h3 = highs[-3], highs[-2], highs[-1]
            la = self._valley(h1[2], h2[2], "low")
            lb = self._valley(h2[2], h3[2], "low")
            if la is not None and lb is not None:
                neck = min(la[1], lb[1])
                depth = h2[1] - neck
                sym_tol = max(cfg.pat_shoulder_tol_atr * atr,
                              cfg.pat_shoulder_frac * depth)
                if h2[1] > h1[1] and h2[1] > h3[1] \
                        and depth >= cfg.pat_min_depth_atr * atr \
                        and abs(h1[1] - h3[1]) <= sym_tol:
                    consider("head_shoulders", -1,
                             ("hs", h1[2], h2[2], h3[2]), neck, h2[1], h3[3])
            # double top on the last two highs
            v = self._valley(h2[2], h3[2], "low")
            if v is not None \
                    and abs(h2[1] - h3[1]) <= cfg.pat_tol_atr * atr \
                    and min(h2[1], h3[1]) - v[1] >= cfg.pat_min_valley_atr * atr:
                consider("double_top", -1, ("dt", h2[2], h3[2]),
                         v[1], max(h2[1], h3[1]), h3[3])
        elif len(highs) == 2:
            h1, h2 = highs
            v = self._valley(h1[2], h2[2], "low")
            if v is not None \
                    and abs(h1[1] - h2[1]) <= cfg.pat_tol_atr * atr \
                    and min(h1[1], h2[1]) - v[1] >= cfg.pat_min_valley_atr * atr:
                consider("double_top", -1, ("dt", h1[2], h2[2]),
                         v[1], max(h1[1], h2[1]), h2[3])

        if len(lows) >= 3:
            l1, l2, l3 = lows[-3], lows[-2], lows[-1]
            ha = self._valley(l1[2], l2[2], "high")
            hb = self._valley(l2[2], l3[2], "high")
            if ha is not None and hb is not None:
                neck = max(ha[1], hb[1])
                depth = neck - l2[1]
                sym_tol = max(cfg.pat_shoulder_tol_atr * atr,
                              cfg.pat_shoulder_frac * depth)
                if l2[1] < l1[1] and l2[1] < l3[1] \
                        and depth >= cfg.pat_min_depth_atr * atr \
                        and abs(l1[1] - l3[1]) <= sym_tol:
                    consider("inv_head_shoulders", 1,
                             ("ihs", l1[2], l2[2], l3[2]), neck, l2[1], l3[3])
            v = self._valley(l2[2], l3[2], "high")
            if v is not None \
                    and abs(l2[1] - l3[1]) <= cfg.pat_tol_atr * atr \
                    and v[1] - max(l2[1], l3[1]) >= cfg.pat_min_valley_atr * atr:
                consider("double_bottom", 1, ("db", l2[2], l3[2]),
                         v[1], min(l2[1], l3[1]), l3[3])
        elif len(lows) == 2:
            l1, l2 = lows
            v = self._valley(l1[2], l2[2], "high")
            if v is not None \
                    and abs(l1[1] - l2[1]) <= cfg.pat_tol_atr * atr \
                    and v[1] - max(l1[1], l2[1]) >= cfg.pat_min_valley_atr * atr:
                consider("double_bottom", 1, ("db", l1[2], l2[2]),
                         v[1], min(l1[1], l2[1]), l2[3])
        return formed

    # -- per-bar update ----------------------------------------------------
    def update(self, t: int, s, close_t: float) -> PatternsState:
        self._ingest(s)
        event = self._try_form(t, s.atr)
        survivors: list[dict] = []
        for rec in self.live:
            inst: PatternInstance = rec["inst"]
            dead = False
            if t - inst.formed_at > self.cfg.pat_max_age:
                dead = True
            elif inst.direction == -1 and close_t > inst.invalidation:
                dead = True
            elif inst.direction == 1 and close_t < inst.invalidation:
                dead = True
            if dead:
                continue
            if not rec["broken"]:
                crossed = (inst.direction == -1 and close_t < inst.neckline) \
                    or (inst.direction == 1 and close_t > inst.neckline)
                if crossed and t > inst.formed_at:
                    rec["broken"] = True
                    broken = PatternInstance(
                        kind=inst.kind, direction=inst.direction,
                        status="broken", formed_at=inst.formed_at,
                        neckline=inst.neckline, extreme=inst.extreme,
                        invalidation=inst.invalidation, target=inst.target)
                    rec["inst"] = broken
                    event = broken
            survivors.append(rec)
        self.live = survivors
        return PatternsState(
            active=tuple(r["inst"] for r in self.live), event=event)


def compute(df: pd.DataFrame, states, cfg: FeatureConfig) -> list[PatternsState]:
    tracker = _Tracker(cfg)
    close = df["close"].to_numpy(dtype=float)
    return [tracker.update(t, s, float(close[t]))
            for t, s in enumerate(states)]
