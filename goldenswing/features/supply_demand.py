"""Supply / demand zones: rally-base-drop (supply) and
drop-base-rally (demand).

Detection (all causal — a zone is *known* at its departure bar):

1. A **departure** candle: |body| >= ``sd_depart_body_atr`` x ATR, in
   the zone's direction (up for demand, down for supply), closing
   beyond the base.
2. The **base**: the 1..``sd_base_max_bars`` candles immediately before
   the departure, each with |body| <= ``sd_base_body_atr`` x ATR.
   Zone = [min low, max high] of the base candles.
3. The **approach**: price moved INTO the base from the opposite side
   — for demand, the close ``sd_pre_bars`` before the base sits above
   the base top (it dropped in); for supply, below the base bottom.
   (Skipped only during warm-up when those bars don't exist.)

Freshness: a zone is **fresh** until price trades back into it
(0 touches), then **tested** (touches >= 1), and **mitigated** —
removed — when a close passes through its far edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .common import ohlcv_arrays
from .config import FeatureConfig
from .state import SDZone, ZonesState


@dataclass
class _Zone:
    top: float
    bottom: float
    direction: int
    created_at: int
    base_bars: int
    departure: float
    touches: int = 0
    inside: bool = False

    def snapshot(self) -> SDZone:
        return SDZone(top=self.top, bottom=self.bottom,
                      direction=self.direction, created_at=self.created_at,
                      base_bars=self.base_bars, departure=self.departure,
                      touches=self.touches)


def compute(df: pd.DataFrame, states, cfg: FeatureConfig) -> list[ZonesState]:
    a = ohlcv_arrays(df)
    open_, high, low, close = a["open"], a["high"], a["low"], a["close"]
    n = len(df)
    zones: list[_Zone] = []
    out: list[ZonesState] = []

    for t in range(n):
        atr_t = states[t].atr if states[t].atr > 0 else 1e-9

        # ---- update live zones with bar t ------------------------------
        survivors: list[_Zone] = []
        for z in zones:
            if t - z.created_at > cfg.sd_max_age:
                continue
            if z.direction == 1:      # demand: support below/around price
                if close[t] < z.bottom:
                    continue          # mitigated
                now_in = low[t] <= z.top
            else:                     # supply
                if close[t] > z.top:
                    continue
                now_in = high[t] >= z.bottom
            if now_in and not z.inside:
                z.touches += 1
            z.inside = now_in
            survivors.append(z)
        zones = survivors

        # ---- detect a new zone departing at bar t ----------------------
        body = close[t] - open_[t]
        if abs(body) >= cfg.sd_depart_body_atr * atr_t:
            d = 1 if body > 0 else -1
            base: list[int] = []
            j = t - 1
            while j >= 0 and len(base) < cfg.sd_base_max_bars:
                b_atr = states[j].atr if states[j].atr > 0 else 1e-9
                if abs(close[j] - open_[j]) <= cfg.sd_base_body_atr * b_atr:
                    base.append(j)
                    j -= 1
                else:
                    break
            if base:
                base_lo = min(float(low[k]) for k in base)
                base_hi = max(float(high[k]) for k in base)
                closes_beyond = (close[t] > base_hi) if d == 1 \
                    else (close[t] < base_lo)
                approach_ok = True
                ref = base[-1] - cfg.sd_pre_bars   # oldest base bar - pre
                if ref >= 0:
                    approach_ok = (close[ref] > base_hi) if d == 1 \
                        else (close[ref] < base_lo)
                overlap = any(
                    z.direction == d
                    and min(z.top, base_hi) - max(z.bottom, base_lo)
                    > 0.5 * (base_hi - base_lo)
                    for z in zones)
                if closes_beyond and approach_ok and not overlap:
                    zones.append(_Zone(
                        top=base_hi, bottom=base_lo, direction=d,
                        created_at=t, base_bars=len(base),
                        departure=abs(body) / atr_t))
                    zones = zones[-cfg.sd_max_zones:]

        # ---- snapshot ----------------------------------------------------
        c = float(close[t])
        sup = [z for z in zones if z.direction == -1]
        dem = [z for z in zones if z.direction == 1]
        n_sup = min(sup, key=lambda z: _edge_dist(c, z), default=None)
        n_dem = min(dem, key=lambda z: _edge_dist(c, z), default=None)
        out.append(ZonesState(
            nearest_supply=n_sup.snapshot() if n_sup else None,
            nearest_demand=n_dem.snapshot() if n_dem else None,
            supply_dist_atr=(_edge_dist(c, n_sup) / atr_t
                             if n_sup else float("nan")),
            demand_dist_atr=(_edge_dist(c, n_dem) / atr_t
                             if n_dem else float("nan")),
            in_supply=bool(n_sup and n_sup.bottom <= c <= n_sup.top),
            in_demand=bool(n_dem and n_dem.bottom <= c <= n_dem.top),
            # full live list — the AOC family clusters these (Phase 4.5)
            active=tuple(z.snapshot() for z in zones),
        ))
    return out


def _edge_dist(close: float, z: _Zone) -> float:
    if close > z.top:
        return close - z.top
    if close < z.bottom:
        return z.bottom - close
    return 0.0
