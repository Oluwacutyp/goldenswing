"""Big-move labels for the forecaster — horizon-general (Phase 6.5).

Definition (per bar ``t``, reference = close[t], scale = ATR(t)):

- ``y_up[t] = 1`` iff, within bars ``t+1 .. t+horizon``, the running
  favourable excursion (high - close[t]) reaches ``fav_mult`` x ATR
  BEFORE the running adverse excursion (close[t] - low) reaches
  ``adv_mult`` x ATR.
- ``y_dn[t]`` is the mirror image.
- If neither barrier resolves inside the window, both labels are 0
  ("no clean big move").
- If both barriers are touched in the SAME bar, the ADVERSE barrier
  wins (the same conservative convention as the backtest harness:
  "stop wins ties") — so a bar whose range spans both levels resolves
  to 0 for that direction.
- ``y_up`` and ``y_dn`` are mutually exclusive by construction: the
  first of the two adverse barriers to be touched kills the opposite
  direction's label.
- Bars without a FULL window ahead of them (``t + horizon > n - 1``)
  or with a non-positive/non-finite ATR get NaN labels — unknown,
  never silently 0.

The label at ``t`` uses only bars ``> t`` for the OUTCOME and only
ATR(t) (known at ``t``) for the scale.  Training folds purge any
sample whose label window overlaps the test window (see
walkforward.py) — that embargo is what keeps the future outcome of a
training row from leaking into a test bar's past.

Horizons (Phase 6.5 — the owner's "not limited to 3 days"): the
Phase-6 definition is the 72-bar row of ``HORIZON_SPECS`` and is
reproduced EXACTLY by the defaults below (``HORIZON`` / ``FAV_MULT``
/ ``ADV_MULT`` are kept as the legacy names other modules import).
Barrier multiples grow with the horizon: a longer window gives price
more time to wander, so a "big move" on a 2-week clock must clear a
wider favourable barrier (and is allowed a wider adverse excursion
before it counts as killed) than one on a half-day clock.  The
fav:adv ratio stays 2:1 at every horizon — the house geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON = 72          # bars (~3 trading days of H1) — Phase-6 default
FAV_MULT = 2.0        # favourable excursion, in ATRs
ADV_MULT = 1.0        # adverse excursion, in ATRs


@dataclass(frozen=True)
class HorizonSpec:
    """One forecast horizon: window length + barrier geometry."""
    horizon: int        # bars ahead the label looks
    fav_mult: float     # favourable barrier, in ATRs
    adv_mult: float     # adverse barrier, in ATRs
    tag: str            # short label for alerts ("12h", "1d", ...)
    name: str           # human name ("half-day", "1 day", ...)


#: Every horizon the multi-horizon forecaster evaluates on H1.
#: fav/adv in ATRs — see the module docstring for the scaling logic.
HORIZON_SPECS: dict[int, HorizonSpec] = {
    12: HorizonSpec(12, 1.5, 0.75, "12h", "half-day"),
    24: HorizonSpec(24, 2.0, 1.0, "1d", "1 day"),
    72: HorizonSpec(72, 2.0, 1.0, "3d", "3 days"),
    168: HorizonSpec(168, 3.0, 1.5, "1w", "1 week"),
    360: HorizonSpec(360, 4.0, 2.0, "2w", "2 weeks"),
}

#: Horizons evaluated, shortest first.
HORIZONS: tuple[int, ...] = tuple(sorted(HORIZON_SPECS))


def spec_for(horizon: int) -> HorizonSpec:
    """The registered spec for ``horizon`` bars (KeyError if the
    horizon was never registered — unregistered horizons have no
    agreed barrier geometry and must not be labelled ad hoc)."""
    return HORIZON_SPECS[horizon]


def build_labels(df: pd.DataFrame, atr: np.ndarray,
                 horizon: int = HORIZON, fav_mult: float = FAV_MULT,
                 adv_mult: float = ADV_MULT) -> pd.DataFrame:
    """Return a frame indexed like ``df`` with float columns
    ``y_up`` / ``y_dn`` in {0.0, 1.0, NaN}."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    y_up = np.full(n, np.nan)
    y_dn = np.full(n, np.nan)

    for t in range(n):
        if t + horizon > n - 1:
            break                      # no full window ahead: unknown
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        ref = close[t]
        up_fav = ref + fav_mult * a    # high reaching this = up wins
        up_adv = ref - adv_mult * a    # low reaching this = up dead
        dn_fav = ref - fav_mult * a    # low reaching this = down wins
        dn_adv = ref + adv_mult * a    # high reaching this = down dead
        up: float | None = None
        dn: float | None = None
        for k in range(t + 1, t + horizon + 1):
            h, l = high[k], low[k]
            if up is None:
                if l <= up_adv:        # adverse first (or same bar)
                    up = 0.0
                elif h >= up_fav:
                    up = 1.0
            if dn is None:
                if h >= dn_adv:        # adverse first (or same bar)
                    dn = 0.0
                elif l <= dn_fav:
                    dn = 1.0
            if up is not None and dn is not None:
                break
        y_up[t] = 0.0 if up is None else up
        y_dn[t] = 0.0 if dn is None else dn

    return pd.DataFrame({"y_up": y_up, "y_dn": y_dn}, index=df.index)


def build_labels_for_horizon(df: pd.DataFrame, atr: np.ndarray,
                             horizon: int) -> pd.DataFrame:
    """Labels under the registered geometry for ``horizon`` bars."""
    spec = spec_for(horizon)
    return build_labels(df, atr, horizon=spec.horizon,
                        fav_mult=spec.fav_mult, adv_mult=spec.adv_mult)
