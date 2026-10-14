"""Participant footprints — character classification, honest heuristics.

No magic "smart money" labels: every tag carries the evidence that
fired it, and each is a *characterisation of behaviour*, not proof of
who traded.

- **Institutional / bank-entry signature**: a liquidity sweep, then a
  displacement candle on climax volume (RVOL >= ``fp_inst_rvol`` at
  the displacement bar), then a shallow retracement (<=
  ``fp_inst_max_retrace_atr`` ATR given back) — the footprint of size
  entering through a level and defending it.
- **CTA character**: slow, persistent trend — at least
  ``fp_cta_min_side`` of the last ``fp_cta_lookback`` closes on one
  side of session VWAP, with the H1 trend AND the daily bias pointing
  the same way.  Momentum-fund flow looks like this: it does not chase
  single candles, it leans for days.
- **Retail character**: the late chase — a climax-body candle AT a
  range extreme after an already-extended move, on high volume; and
  its sequel, the **trap**: price closing back through the chase
  candle's midpoint within ``fp_trap_bars`` (the chase crowd is
  suddenly underwater).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import ohlcv_arrays
from .config import FeatureConfig
from .state import FootprintsState


def compute(df: pd.DataFrame, states, cfg: FeatureConfig,
            rvol: np.ndarray, vwap: np.ndarray) -> list[FootprintsState]:
    a = ohlcv_arrays(df)
    close, open_ = a["close"], a["open"]
    n = len(df)

    inst_active = False
    inst_dir = 0
    inst_since: int | None = None
    last_chase: tuple[int, int, float] | None = None   # (bar, dir, midpoint)
    trap_fired_for: int | None = None

    out: list[FootprintsState] = []
    for t in range(n):
        s = states[t]
        atr = s.atr if s.atr > 0 else 1e-9
        tags: list[str] = []
        evidence: list[str] = []

        # ---- institutional signature -----------------------------------
        fired_now = False
        sw = s.last_sweep
        if sw is not None and s.sweep_age is not None \
                and s.sweep_age <= cfg.fp_sweep_window:
            d = sw.direction
            disp_age = s.bull_disp_age if d == 1 else s.bear_disp_age
            if disp_age is not None and disp_age <= s.sweep_age:
                disp_bar = t - disp_age
                rv = rvol[disp_bar] if disp_bar >= 0 else np.nan
                if np.isfinite(rv) and rv >= cfg.fp_inst_rvol:
                    peak = float(np.max(close[disp_bar:t + 1])) if d == 1 \
                        else float(np.min(close[disp_bar:t + 1]))
                    giveback = d * (peak - close[t])
                    if giveback <= cfg.fp_inst_max_retrace_atr * atr:
                        fired_now = True
                        if not (inst_active and inst_dir == d):
                            inst_since = t
                        inst_active, inst_dir = True, d
                        tags.append("institutional_entry")
                        evidence.append(
                            f"sweep {s.sweep_age}b ago + displacement on "
                            f"RVOL {rv:.1f} + retracement "
                            f"{giveback / atr:.2f} ATR")
        if not fired_now:
            inst_active, inst_dir, inst_since = False, 0, None

        # ---- CTA character ----------------------------------------------
        cta = 0
        if t >= cfg.fp_cta_lookback - 1:
            side_up = side_dn = 0
            for k in range(t - cfg.fp_cta_lookback + 1, t + 1):
                if np.isfinite(vwap[k]):
                    if close[k] > vwap[k]:
                        side_up += 1
                    elif close[k] < vwap[k]:
                        side_dn += 1
            if side_up >= cfg.fp_cta_min_side and s.bias == 1 \
                    and s.daily_bias == 1:
                cta = 1
            elif side_dn >= cfg.fp_cta_min_side and s.bias == -1 \
                    and s.daily_bias == -1:
                cta = -1
        if cta != 0:
            tags.append("cta_trend")
            evidence.append(
                f"{max(side_up if cta == 1 else side_dn, 0)}/"
                f"{cfg.fp_cta_lookback} closes one side of VWAP, "
                f"H1+daily aligned")

        # ---- retail chase + trap ----------------------------------------
        chase = 0
        body = s.body_ratio
        if abs(body) >= cfg.fp_chase_body_atr:
            d = 1 if body > 0 else -1
            at_extreme = (d == 1 and s.eq_position >= 0.88) or \
                (d == -1 and s.eq_position <= 0.12)
            rv = rvol[t]
            if at_extreme and abs(s.momentum) >= cfg.fp_chase_momentum_atr \
                    and np.isfinite(rv) and rv >= cfg.fp_chase_rvol:
                chase = d
                last_chase = (t, d, 0.5 * (open_[t] + close[t]))
                tags.append("retail_chase")
                evidence.append(
                    f"body {body:+.1f} ATR at eq {s.eq_position:.2f} after "
                    f"momentum {s.momentum:+.1f} ATR, RVOL {rv:.1f}")

        trap = 0
        if last_chase is not None and trap_fired_for != last_chase[0]:
            cbar, cdir, mid = last_chase
            if cbar < t <= cbar + cfg.fp_trap_bars:
                if (cdir == 1 and close[t] < mid) or \
                        (cdir == -1 and close[t] > mid):
                    trap = -cdir
                    trap_fired_for = cbar
                    tags.append("retail_trap")
                    evidence.append(
                        f"close back through chase midpoint "
                        f"{t - cbar}b after the chase")

        out.append(FootprintsState(
            tags=tuple(tags), evidence=tuple(evidence),
            institutional_dir=inst_dir if inst_active else 0,
            institutional_age=(t - inst_since)
            if inst_active and inst_since is not None else None,
            cta_trend=cta, retail_chase=chase, retail_trap=trap,
        ))
    return out
