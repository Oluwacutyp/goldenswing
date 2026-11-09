"""Bill Williams — the measurement core (Pass 8, SPEC
§6.6).  One strategy's organs, implemented as causal
measurements over one prepared frame:

- **Alligator**: jaw / teeth / lips — SMMA of the
  median price at Williams' periods (13 / 8 / 5), built
  from ``reader/indicators.py``'s EMA machinery (its
  alpha = 1/period EWM *is* the SMMA/Wilder recursion)
  and read at Williams' forward shifts (8 / 5 / 3):
  the line value used at bar ``j`` is the smoothed
  value from ``shift`` bars earlier — the shift is a
  placement convention, never a peek forward.
- **Fractals**: Williams' 5-bar pattern, via the
  family's own thin wrapper over
  ``live/multitf.find_swings`` with ``n=2`` (the same
  geometry, pinned in one place).  A fractal centered
  at bar ``k`` is CONFIRMED at bar ``k+2`` — before
  that bar closes it does not exist for any read.
- **AO / AC**: the median-price momentum pair.  AO =
  SMA(median, 5) − SMA(median, 34) (Williams' own
  definition); AC = AO − SMA(AO, 5).
- **Gator**: the jaw–teeth and teeth–lips distance
  histograms of the shifted lines, with the phase read
  as a causal state machine over their deltas.
- **BW-MFI**: market facilitation = bar range /
  volume, classified against the previous bar
  (green / fake / fade / squat).  MT5 volume on this
  feed is TICK VOLUME: every MFI read carries
  ``volume_basis`` and the caveat string.  A frame
  with no usable volume yields absent MFI reads —
  absent stays absent, never fabricated.

Alligator state words (SPEC §6.6): ``sleeping`` (the
lines are intertwined — no directional order),
``awakening`` (directional order has formed and the
mouth is opening, order young), ``eating`` (order
held, mouth still opening, order established),
``satiated`` (order held but the mouth is closing).
Before the lines can be read at all the state is
``unformed`` and every dependent read says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...live.multitf import find_swings
from ...reader import indicators as _ind

__all__ = [
    "AC_PERIOD",
    "AO_FAST",
    "AO_SLOW",
    "AWAKEN_ORDER_BARS",
    "JAW_PERIOD",
    "JAW_SHIFT",
    "LIPS_PERIOD",
    "LIPS_SHIFT",
    "TEETH_PERIOD",
    "TEETH_SHIFT",
    "VOLUME_BASIS_ABSENT",
    "VOLUME_BASIS_TICK",
    "VOLUME_CAVEAT_TICK",
    "FractalSeries",
    "PreparedWilliams",
    "annotate",
    "smma",
]

#: Alligator construction (Williams' standard).
JAW_PERIOD, JAW_SHIFT = 13, 8
TEETH_PERIOD, TEETH_SHIFT = 8, 5
LIPS_PERIOD, LIPS_SHIFT = 5, 3

#: AO / AC construction (Williams' standard).
AO_FAST, AO_SLOW = 5, 34
AC_PERIOD = 5

#: A directional order this young (in bars) with the
#: mouth opening is "awakening"; older is "eating".
AWAKEN_ORDER_BARS = 5

#: BW-MFI provenance.  Tick volume is a trade COUNT,
#: used as the facilitation denominator; it is not
#: traded volume and no record dresses it as such.
VOLUME_BASIS_TICK = "tick_volume"
VOLUME_BASIS_ABSENT = "absent"
VOLUME_CAVEAT_TICK = (
    "MT5 volume on this feed is tick volume (a trade "
    "count), used as the BW-MFI denominator — facilitation "
    "readings carry that provenance, never traded volume")


def smma(values: np.ndarray, period: int) -> np.ndarray:
    """Smoothed MA via the reader's EMA machinery
    (alpha = 1/period, adjust=False — the Wilder/SMMA
    recursion, the repo's one smoothing organ)."""
    return _ind.ema(np.asarray(values, dtype=float), period)


def _sma(values: np.ndarray, period: int) -> np.ndarray:
    """Plain SMA, NaN until ``period`` values exist
    (Williams' AO is defined on simple averages)."""
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rolling(period, min_periods=period).mean() \
        .to_numpy(dtype=float)


def _sign(v: float | None) -> int | None:
    if v is None or not np.isfinite(v):
        return None
    return 1 if v > 0 else (-1 if v < 0 else 0)


class FractalSeries:
    """Williams 5-bar fractals over one frame, with
    confirmation times.  The thin wrapper over
    ``find_swings(n=2)`` that pins the family's
    definition in one place: an up fractal is a bar
    whose high is strictly greater than the two bars
    on each side; a down fractal mirrors on lows.
    Ties are never fractals (find_swings is strict).
    A fractal centered at ``k`` is confirmed at
    ``k + 2`` — the first bar whose close can know it.
    """

    def __init__(self, frame: pd.DataFrame):
        highs, lows = find_swings(
            frame["high"].to_numpy(dtype=float),
            frame["low"].to_numpy(dtype=float), 2)
        self.up = [{"center": int(p), "price": float(v),
                    "confirmed_at": int(p) + 2}
                   for p, v in highs]
        self.down = [{"center": int(p), "price": float(v),
                      "confirmed_at": int(p) + 2}
                     for p, v in lows]

    def known_at(self, j: int, direction: int) -> dict | None:
        """The latest fractal in ``direction`` (+1 up,
        −1 down) CONFIRMED by bar ``j`` — or None.
        A fractal whose right side has not closed by
        ``j`` is not returned: it is not yet a fact."""
        src = self.up if direction > 0 else self.down
        out = None
        for f in src:
            if f["confirmed_at"] <= j:
                out = f
            else:
                break
        return dict(out) if out is not None else None

    def confirmed_at_bar(self, j: int) -> list[dict]:
        """Fractals whose confirmation bar IS ``j``
        (each with its ``direction`` attached)."""
        out = []
        for d, src in ((1, self.up), (-1, self.down)):
            for f in src:
                if f["confirmed_at"] == j:
                    out.append({**f, "direction": d})
        return out


class PreparedWilliams:
    """One frame's Williams organs, prepared once:
    every series below is causal by construction, so a
    read at bar ``j`` can never disagree with the same
    read taken on the frame truncated at ``j``."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.n = len(frame)
        self.h = frame["high"].to_numpy(dtype=float)
        self.l = frame["low"].to_numpy(dtype=float)
        self.c = frame["close"].to_numpy(dtype=float)
        median = (self.h + self.l) / 2.0
        self._jaw_raw = smma(median, JAW_PERIOD)
        self._teeth_raw = smma(median, TEETH_PERIOD)
        self._lips_raw = smma(median, LIPS_PERIOD)
        self.fractals = FractalSeries(frame)
        ao_fast = _sma(median, AO_FAST)
        ao_slow = _sma(median, AO_SLOW)
        self._ao = ao_fast - ao_slow
        self._ac = self._ao - _sma(self._ao, AC_PERIOD)
        # Volume / BW-MFI.
        if "volume" in frame.columns:
            self._vol = frame["volume"].to_numpy(dtype=float)
            self.volume_basis = VOLUME_BASIS_TICK
        else:
            self._vol = None
            self.volume_basis = VOLUME_BASIS_ABSENT
        self._mfi = self._mfi_series()
        # Gator phase is a state machine over the whole
        # series — computed once, causally (phase at j
        # is a function of deltas up to j).
        self._gator_phase = self._gator_phases()

    # -- raw line reads --------------------------------

    def _line(self, raw: np.ndarray, shift: int,
              j: int) -> float | None:
        k = j - shift
        if k < 0 or k >= self.n:
            return None
        v = float(raw[k])
        return v if np.isfinite(v) else None

    def lines_at(self, j: int) -> dict | None:
        """The Alligator lines AS READ at bar ``j``
        (Williams' shifts applied), or None while any
        line is unformed."""
        jaw = self._line(self._jaw_raw, JAW_SHIFT, j)
        teeth = self._line(self._teeth_raw, TEETH_SHIFT, j)
        lips = self._line(self._lips_raw, LIPS_SHIFT, j)
        if jaw is None or teeth is None or lips is None:
            return None
        return {"jaw": jaw, "teeth": teeth, "lips": lips}

    def _order_dir(self, lines: dict) -> int:
        if lines["lips"] > lines["teeth"] > lines["jaw"]:
            return 1
        if lines["jaw"] > lines["teeth"] > lines["lips"]:
            return -1
        return 0

    def _order_age(self, j: int, direction: int) -> int:
        age = 0
        k = j
        while k >= 0:
            lines = self.lines_at(k)
            if lines is None or self._order_dir(lines) != direction:
                break
            age += 1
            k -= 1
        return age

    def alligator_at(self, j: int) -> dict:
        """The Alligator read at bar ``j``: state word,
        direction (+1 lips>teeth>jaw / −1 mirrored / 0
        intertwined), the lines, the mouth spread and
        its change, and the close's distance beyond the
        lips in the order's direction."""
        lines = self.lines_at(j)
        if lines is None:
            return {"state": "unformed", "direction": 0,
                    "jaw": None, "teeth": None, "lips": None,
                    "spread": None, "spread_delta": None,
                    "order_age": 0, "close_beyond_lips": None}
        spread = max(lines.values()) - min(lines.values())
        prev = self.lines_at(j - 1) if j > 0 else None
        spread_delta = None
        if prev is not None:
            spread_delta = spread - (
                max(prev.values()) - min(prev.values()))
        direction = self._order_dir(lines)
        age = self._order_age(j, direction) if direction else 0
        if direction == 0:
            state = "sleeping"
        elif spread_delta is None:
            state = "awakening"
        elif spread_delta > 0:
            state = "awakening" if age <= AWAKEN_ORDER_BARS \
                else "eating"
        else:
            state = "satiated"
        beyond = None
        if direction:
            beyond = (self.c[j] - lines["lips"]) * direction
        return {"state": state, "direction": direction,
                "jaw": lines["jaw"], "teeth": lines["teeth"],
                "lips": lines["lips"], "spread": spread,
                "spread_delta": spread_delta,
                "order_age": age, "close_beyond_lips": beyond}

    # -- AO / AC ----------------------------------------

    def _osc_at(self, series: np.ndarray, j: int) -> dict:
        v = float(series[j]) if 0 <= j < self.n else None
        if v is not None and not np.isfinite(v):
            v = None
        prev = None
        if v is not None and j > 0:
            p = float(series[j - 1])
            prev = p if np.isfinite(p) else None
        return {"value": v, "sign": _sign(v),
                "rising": (v > prev) if (v is not None
                                         and prev is not None)
                else None}

    def ao_at(self, j: int) -> dict:
        return self._osc_at(self._ao, j)

    def ac_at(self, j: int) -> dict:
        return self._osc_at(self._ac, j)

    # -- Gator ------------------------------------------

    def _gator_distances(self, j: int) -> tuple | None:
        lines = self.lines_at(j)
        if lines is None:
            return None
        return (abs(lines["jaw"] - lines["teeth"]),
                abs(lines["teeth"] - lines["lips"]))

    def _gator_phases(self) -> list:
        phases: list = []
        prev_phase = None
        prev_dist = None
        for j in range(self.n):
            dist = self._gator_distances(j)
            if dist is None or prev_dist is None:
                phases.append(prev_phase)
                if dist is not None:
                    prev_dist = dist
                continue
            du = dist[0] - prev_dist[0]
            dl = dist[1] - prev_dist[1]
            if du > 0 and dl > 0:
                phase = "eating"
            elif du < 0 and dl < 0:
                phase = "satiated" if prev_phase == "eating" \
                    else "sleeping"
            elif du == 0 and dl == 0:
                phase = prev_phase
            else:  # mixed — one side opens, one closes
                phase = "satiated" if prev_phase == "eating" \
                    else "awakening"
            phases.append(phase)
            prev_phase = phase
            prev_dist = dist
        return phases

    def gator_at(self, j: int) -> dict:
        dist = self._gator_distances(j)
        if dist is None:
            return {"upper": None, "lower": None,
                    "phase": None}
        return {"upper": dist[0], "lower": dist[1],
                "phase": self._gator_phase[j]}

    # -- BW-MFI ------------------------------------------

    def _mfi_series(self) -> np.ndarray:
        out = np.full(self.n, np.nan)
        if self._vol is None:
            return out
        rng = self.h - self.l
        for j in range(self.n):
            v = self._vol[j]
            if np.isfinite(v) and v > 0:
                out[j] = rng[j] / v
        return out

    def mfi_at(self, j: int) -> dict:
        """BW-MFI at bar ``j`` with its Williams class
        against bar ``j−1``.  Absent (value/class None)
        whenever volume is missing or zero — the basis
        and caveat ride every read either way."""
        base = {"value": None, "class": None,
                "volume_basis": self.volume_basis,
                "volume_caveat": VOLUME_CAVEAT_TICK
                if self.volume_basis == VOLUME_BASIS_TICK
                else None}
        if self._vol is None or not (0 <= j < self.n):
            return base
        v = float(self._mfi[j])
        if not np.isfinite(v):
            return base
        base["value"] = v
        if j == 0:
            return base
        prev = float(self._mfi[j - 1])
        if not np.isfinite(prev):
            return base
        dm = v - prev
        dv = float(self._vol[j]) - float(self._vol[j - 1])
        if dm == 0 or dv == 0:
            cls = "flat"
        elif dm > 0 and dv > 0:
            cls = "green"
        elif dm > 0:
            cls = "fake"
        elif dv > 0:
            cls = "squat"
        else:
            cls = "fade"
        base["class"] = cls
        return base


def annotate(prep: PreparedWilliams, j: int, *,
             timeframe: str,
             situation_id: str | None = None) -> dict:
    """One bar's full research read — the payload of a
    ``bw_read`` journal record (the harness adds the
    envelope).  Pure: the same (prep, j) always yields
    the same read, on the full frame or any truncation
    that still contains bar ``j``."""
    frame = prep.frame
    ts = None
    try:
        ts = str(frame.index[j])
    except Exception:
        ts = None
    return {
        "timeframe": timeframe,
        "bar": j,
        "ts": ts,
        "situation_id": situation_id,
        "alligator": prep.alligator_at(j),
        "fractal_up": prep.fractals.known_at(j, 1),
        "fractal_down": prep.fractals.known_at(j, -1),
        "ao": prep.ao_at(j),
        "ac": prep.ac_at(j),
        "gator": prep.gator_at(j),
        "mfi": prep.mfi_at(j),
    }
