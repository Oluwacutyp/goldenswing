"""Developing value area — the §6.3 EXTENSION of
``live/volume_profile.py`` (MarketSituation Pass 7c).

The desk's value layer today describes COMPLETED value:
``volume_profile`` over a finished window, and the day
object's prior-day VA.  The auction/value family needs
value as it BUILDS — the developing (intraday, cumulative)
value area: POC / VAH / VAL recomputed causally over the
current trading day's bars as each new bar prints, so the
migration of value (the auction re-pricing itself, or
refusing to) is a measurable series rather than a story.

This module is that computation, and it is the SHARED
value-state layer: the situation's ``location.value``
carries its current state (the runner builds it here),
the auction/value family reads its per-bar history here,
and Pass 7d's mean-reversion family will read the same
state — one implementation of developing value, never a
private re-derivation per family.

Construction parity is a pinned contract: the histogram
is ``volume_profile``'s own (24 bins over the slice's
range, each bar's volume spread uniformly across the
bins its range touches, value area = the top bins
holding 70% of volume), so the developing state at a
day's last bar EQUALS ``volume_profile(day)`` on the
same bars — a test pins it.  Day boundaries are the
desk's own (``day_context.trading_day_key`` — the NY
trading day), so the family's "today" is the day
object's "today".

VOLUME PROVENANCE, named never dressed: MT5's ``volume``
column is TICK VOLUME (the count of ticks per bar), not
traded volume — ``volume_profile``'s docstring has always
said so.  Every state this module returns carries
``volume_basis = "tick_volume"`` and every journal record
the family writes repeats it: the profile is a tick-
activity profile, and its levels are read as such.

Absent stays absent (the situation engine's honesty
law): a day with fewer than MIN_BARS bars, a flat range
or zero volume yields ``status`` naming why and None
levels — never a synthesized POC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import day_context
from .volume_profile import MIN_BARS, PROFILE_BINS, VALUE_AREA_PCT

#: The volume basis of every profile in this engine:
#: MT5's volume column is tick volume.  Named on every
#: state and every journal record that carries a value
#: read (see the module docstring).
VOLUME_BASIS = "tick_volume"

#: The family's clocks (SPEC §6.3): the developing area
#: builds on the day's H1 bars; the auction family's
#: trigger clock is M15.  Stated here because the value
#: layer is where the day's bars are defined.
FORMATION_TF = "H1"
TRIGGER_TF = "M15"

#: An HVN (high-volume node) is a local-maximum bin
#: holding at least this share of the POC bin's volume —
#: a real shelf of activity, not a rounding bump.
HVN_MIN_SHARE = 0.50

#: An LVN (low-volume node) is a local-minimum bin
#: holding at most this share of the POC bin's volume,
#: flanked by HVNs on BOTH sides — thin air price can
#: traverse fast, with a shelf behind and a shelf ahead
#: (the traversal hypothesis's geometry, SPEC §6.3).
LVN_MAX_SHARE = 0.35

#: Migration dead-band, in bin widths: an edge or POC
#: that moved less than this between consecutive bars
#: reads "steady", not a direction — bin re-spanning as
#: the day's range grows makes sub-band moves arithmetic,
#: not auction facts.
MIGRATION_DEADBAND_BINS = 0.25


# ------------------------------------------------------------------
# histogram + value area (volume_profile's construction, verbatim)
# ------------------------------------------------------------------

def profile_histogram(frame: pd.DataFrame, *,
                      bins: int = PROFILE_BINS):
    """(centers, bin_vol, edges) for ``frame`` — the exact
    binning ``volume_profile`` computes (24 bins over the
    frame's range; each bar's volume spread uniformly
    across the bins its range touches).  None when the
    frame cannot carry a profile (the caller names the
    status)."""
    if frame is None or len(frame) < MIN_BARS:
        return None
    if not {"high", "low", "volume"}.issubset(frame.columns):
        return None
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    if not np.isfinite(high).all() or not np.isfinite(low).all():
        return None
    lo, hi = float(low.min()), float(high.max())
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_vol = np.zeros(bins)
    for i in range(len(frame)):
        touched = np.where((centers >= low[i])
                           & (centers <= high[i]))[0]
        if len(touched):
            bin_vol[touched] += volume[i] / len(touched)
    if float(bin_vol.sum()) <= 0:
        return None
    return centers, bin_vol, edges


def value_area(centers: np.ndarray, bin_vol: np.ndarray, *,
               va_pct: float = VALUE_AREA_PCT) -> dict:
    """POC / VAH / VAL from a histogram — ``volume_profile``'s
    accumulation (top bins by volume until the value-area
    share is held), so the two modules can never disagree
    on the same histogram."""
    total = float(bin_vol.sum())
    poc = float(centers[int(np.argmax(bin_vol))])
    order = np.argsort(bin_vol)[::-1]
    va_bins, acc = [], 0.0
    for idx in order:
        if acc >= total * va_pct:
            break
        va_bins.append(idx)
        acc += bin_vol[idx]
    va_prices = centers[va_bins]
    return {"poc": poc,
            "vah": float(va_prices.max()) if len(va_prices)
            else poc,
            "val": float(va_prices.min()) if len(va_prices)
            else poc}


def nodes(centers: np.ndarray, bin_vol: np.ndarray,
          edges: np.ndarray) -> dict:
    """High / low volume nodes from a histogram.

    Extrema are read on PLATEAU-MERGED runs (consecutive
    equal-volume bins are one shelf, read at its middle
    bin): an HVN is a run that is a local maximum holding
    >= HVN_MIN_SHARE of the POC bin's volume; an LVN is a
    run that is a local minimum holding <= LVN_MAX_SHARE,
    with an HVN run on both sides — its ``below_hvn`` /
    ``above_hvn`` are those shelves' prices (the
    traversal's near and far nodes)."""
    n = len(bin_vol)
    runs: list[tuple[int, int, float]] = []
    start = 0
    for i in range(1, n + 1):
        if i == n or bin_vol[i] != bin_vol[start]:
            runs.append((start, i - 1, float(bin_vol[start])))
            start = i
    poc_vol = float(bin_vol.max()) if n else 0.0

    def _is_max(k: int) -> bool:
        v = runs[k][2]
        left = runs[k - 1][2] if k > 0 else None
        right = runs[k + 1][2] if k + 1 < len(runs) else None
        return (left is None or v > left) \
            and (right is None or v > right)

    def _is_min(k: int) -> bool:
        v = runs[k][2]
        left = runs[k - 1][2] if k > 0 else None
        right = runs[k + 1][2] if k + 1 < len(runs) else None
        return (left is None or v < left) \
            and (right is None or v < right)

    def _price(k: int) -> float:
        s, e, _v = runs[k]
        return float(centers[(s + e) // 2])

    hvn_runs = [k for k in range(len(runs))
                if _is_max(k)
                and runs[k][2] >= HVN_MIN_SHARE * poc_vol]
    hvns = [_price(k) for k in hvn_runs]
    lvns: list[dict] = []
    for k in range(len(runs)):
        if not _is_min(k) \
                or runs[k][2] > LVN_MAX_SHARE * poc_vol:
            continue
        below = [r for r in hvn_runs if r < k]
        above = [r for r in hvn_runs if r > k]
        if not below or not above:
            continue
        s, e, _v = runs[k]
        lvns.append({
            "price": _price(k),
            "low": float(edges[s]),
            "high": float(edges[e + 1]),
            "below_hvn": _price(below[-1]),
            "above_hvn": _price(above[0]),
        })
    return {"hvns": hvns, "lvns": lvns}


# ------------------------------------------------------------------
# the causal developing series over one frame
# ------------------------------------------------------------------

def _migration_word(delta: float | None,
                    bin_width: float | None) -> str | None:
    if delta is None or not bin_width:
        return None
    band = MIGRATION_DEADBAND_BINS * bin_width
    if delta > band:
        return "rising"
    if delta < -band:
        return "falling"
    return "steady"


class DevelopingSeries:
    """The developing value area of a frame's days, read
    causally per bar: ``at(j)`` is a function of bars
    ``<= j`` only (the day slice ends at ``j``), so a read
    at bar ``j`` on the full frame equals the read on the
    frame truncated at ``j`` — judged by what was
    observable at the time, never hindsight.

    One series per scan; states are cached per bar (a
    day's slice is re-profiled per read — at H1 cadence
    the day holds <= ~24 bars and the histogram is 24
    bins, the same order of work ``day_context`` already
    does per refresh; it must not creep to faster clocks,
    SPEC §10.2 risk 3)."""

    def __init__(self, frame: pd.DataFrame | None):
        self.frame = frame
        self.n = len(frame) if frame is not None else 0
        if frame is not None and isinstance(
                frame.index, pd.DatetimeIndex):
            self._keys = [day_context.trading_day_key(ts)
                          for ts in frame.index]
        else:
            # A non-datetime frame is one undated day —
            # the read still works, the day_key says so.
            self._keys = [None] * self.n
        self._cache: dict[int, dict] = {}

    def day_positions(self, j: int) -> list[int]:
        """Frame positions of bar ``j``'s trading day, up
        to and including ``j``."""
        key = self._keys[j]
        return [i for i in range(j + 1) if self._keys[i] == key]

    def at(self, j: int) -> dict:
        """The developing value state as of bar ``j``."""
        if j in self._cache:
            return self._cache[j]
        state = self._compute(j)
        self._cache[j] = state
        return state

    def _compute(self, j: int) -> dict:
        frame = self.frame
        key = self._keys[j]
        day_key = (key.date().isoformat()
                   if hasattr(key, "date") else None)
        base = {"day_key": day_key, "bars": 0, "poc": None,
                "vah": None, "val": None, "bin_width": None,
                "volume_basis": VOLUME_BASIS,
                "hvns": [], "lvns": [], "migration": None,
                "prev": None}
        if frame is None or j < 0 or j >= self.n:
            return {**base, "status": "no frame"}
        pos = self.day_positions(j)
        day = frame.iloc[pos]
        base["bars"] = len(day)
        if len(day) < MIN_BARS:
            return {**base, "status": "insufficient bars"}
        if not {"high", "low", "volume"}.issubset(
                frame.columns):
            return {**base, "status": "missing columns"}
        hist = profile_histogram(day)
        if hist is None:
            high = day["high"].to_numpy(dtype=float)
            low = day["low"].to_numpy(dtype=float)
            if not (np.isfinite(high).all()
                    and np.isfinite(low).all()):
                status = "non-finite prices"
            elif float(low.min()) >= float(high.max()):
                status = "flat range"
            else:
                status = "no volume"
            return {**base, "status": status}
        centers, bin_vol, edges = hist
        va = value_area(centers, bin_vol)
        nd = nodes(centers, bin_vol, edges)
        bin_width = float(edges[1] - edges[0])
        state = {**base, "status": "ok",
                 "poc": va["poc"], "vah": va["vah"],
                 "val": va["val"], "bin_width": bin_width,
                 "hvns": nd["hvns"], "lvns": nd["lvns"]}
        # Migration vs the SAME DAY's previous bar (a day's
        # first measurable state has no predecessor — the
        # migration block is absent, never zero-filled).
        if pos and pos[0] < j:
            prev = self.at(j - 1)
            if prev.get("status") == "ok" \
                    and prev.get("day_key") == day_key:
                deltas = {
                    f"{k}_delta": float(state[k] - prev[k])
                    for k in ("poc", "vah", "val")}
                state["prev"] = {
                    k: prev[k] for k in ("poc", "vah", "val")}
                state["migration"] = {
                    **deltas,
                    **{k: _migration_word(deltas[f"{k}_delta"],
                                          bin_width)
                       for k in ("poc", "vah", "val")}}
        return state


def developing_state(frame: pd.DataFrame | None,
                     j: int) -> dict:
    """One-shot causal read (builds a series; scans that
    read many bars hold a DevelopingSeries instead)."""
    if frame is None or j < 0 or j >= len(frame):
        return {"status": "no frame", "day_key": None,
                "bars": 0, "poc": None, "vah": None,
                "val": None, "bin_width": None,
                "volume_basis": VOLUME_BASIS, "hvns": [],
                "lvns": [], "migration": None, "prev": None}
    return DevelopingSeries(frame).at(j)


def current_developing(frame: pd.DataFrame | None) -> dict:
    """The developing state at the frame's last bar — the
    situation block's read (the day so far, as it stands)."""
    if frame is None or not len(frame):
        return developing_state(frame, -1)
    return developing_state(frame, len(frame) - 1)


def migrated_toward(state: dict, ref: dict | None,
                    direction: int) -> bool:
    """Has value migrated TOWARD ``direction`` between the
    reference state and ``state`` — the POC or the probed
    edge (VAH up / VAL down) moved in the direction by
    more than the migration dead-band.  The acceptance
    read's migration leg (SPEC §6.3: closes beyond value
    + value migrating toward price = the auction
    re-pricing)."""
    if not ref or state.get("status") != "ok" \
            or ref.get("status") not in ("ok", None):
        return False
    band = MIGRATION_DEADBAND_BINS * float(
        state.get("bin_width") or 0.0)
    edge = "vah" if direction > 0 else "val"
    for key in ("poc", edge):
        now, was = state.get(key), ref.get(key)
        if now is None or was is None:
            continue
        if (float(now) - float(was)) * direction > band:
            return True
    return False
