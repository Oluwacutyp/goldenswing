"""Volatility expansion — the family's measurement core
(MarketSituation Pass 7a, SPEC §6.2).

The behavior: a market that has coiled (volatility compressed
across the timeframe stack) escapes the coil, and the escape
either ACCEPTS beyond the coil (closes holding outside — the
follow-through is the trade) or FAILS (a close back inside —
the escape was rejected, and the fade toward the opposite
edge is the trade).  Two hypotheses, one state machine:

    coil ──trigger──▶ escape armed ──acceptance close──▶ ESCAPE
                          │
                          └──rejection close──▶ FAILED EXPANSION

Compression here is NOT a single-indicator read.  A coil is
measured on the formation timeframe by TWO of the desk's own
state organs agreeing — the Bollinger squeeze state
(``features/bollinger.py``: bandwidth percentile below the
squeeze floor) held for a run of bars, AND the ATR regime
method the situation engine itself uses (``atr_regime``:
ATR14 against its own trailing-100 median, "contracting"
below ``regime.ATR_CONTRACTING``) — and a detection only
becomes a hypothesis when a THIRD read corroborates from the
market's multi-timeframe state (the situation engine's
TFStates: another timeframe also contracting, or the
formation frame's own bedrock structure phase reading
"compression").  The corroboration rule lives in the
playbook layer (it owns the situation threading); this
module is the pure bar math.

Distinctness (SPEC §6 discipline): ``range_compression_
breakout`` trades a RangeDetector range OBJECT on H4 inside
the D1-flat range regime, entering on the breakout close
itself.  This family reads NO range object and NO regime
verdict: the coil is the volatility state's own geometry
(the squeeze run's high/low envelope) on whatever clock the
squeeze formed on, and the entry is the ACCEPTANCE close —
the bar AFTER the trigger whose whole range holds beyond
the coil — never the breakout print.  A level is a question;
the candle sequence is the answer.

Every read is causal: the state at bar ``j`` is a function
of bars ``<= j`` only (judged by what was observable at the
time, never hindsight).  The series accessors below are the
features/atr organs' own formulas, evaluated per bar with
the same trailing windows ``features.bollinger.compute``
and ``live.situation.build_tf_state`` use, so this module
can never disagree with the situation engine on identical
bars (parity is test-pinned).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.common import rolling_mean, rolling_std
from ..features.config import FeatureConfig
from .atr_regime import atr_series as _atr_series
from .displacement import DISP_BODY_WICK_MIN
from .regime import ATR_CONTRACTING

#: The family's formation clocks (SPEC §6.2: formation on
#: the squeeze TF, trigger same TF, entry one down).  The
#: situation's intraday stack is the family's home; D1
#: squeezes belong to the swing families' clocks, M5/M1 to
#: the scalp sleeve's own stream.
FORMATION_TFS: tuple[str, ...] = ("H4", "H1", "M15")

#: Entry clock one step down from each formation clock.
ENTRY_TF_DOWN: dict[str, str] = {
    "H4": "H1", "H1": "M15", "M15": "M5"}

#: A coil is a RUN of squeeze bars, not a single coiled
#: print (the family's stated definition; one squeezed bar
#: is weather, five is a coil).
MIN_COIL_BARS = 5

#: Longest coil run the walk-back will measure.  A longer
#: squeeze is still a coil — its measured window is the
#: most recent bars of the run (the geometry that matters
#: is the tight part price is escaping from).
MAX_COIL_BARS = 200

#: Bars after the trigger in which acceptance / rejection
#: must resolve.  The displacement doctrine's shape: a move
#: that takes longer than this to resolve is drift, not
#: expansion (cf. DISP_MAX_M15_BARS in live/displacement).
ACCEPT_WINDOW_BARS = 3

#: The trigger close must clear the coil edge by this much
#: of the coil's own ATR — the desk's standing buffer scale
#: (playbooks/range_regime.STOP_BUFFER_ATR), applied to the
#: breakout playbook's penetration read.
TRIGGER_PENETRATION_ATR = 0.25

#: The structural stop sits this far INSIDE the coil edge
#: (the breakout playbook's "stop just inside the edge";
#: a close back through it is the escape voided).
STOP_BUFFER_ATR = 0.25


# ------------------------------------------------------------------
# series preparation (the organs' own formulas, per bar)
# ------------------------------------------------------------------

class PreparedFrame:
    """One frame's detector series, prepared once per scan.

    Bandwidth / ATR are computed over the full frame with
    the organs' vectorised formulas (cheap); the percentile
    and ratio reads — trailing-window statistics — are
    evaluated per bar on demand and memoised, so a scan
    pays for the bars it actually inspects (a bounded
    lookback around each candidate bar), never O(n·window)
    for the whole history.
    """

    def __init__(self, frame: pd.DataFrame, *,
                 cfg: FeatureConfig | None = None):
        cfg = cfg or FeatureConfig()
        self.frame = frame
        self.n = len(frame)
        self.o = frame["open"].to_numpy(dtype=float)
        self.h = frame["high"].to_numpy(dtype=float)
        self.l = frame["low"].to_numpy(dtype=float)
        self.c = frame["close"].to_numpy(dtype=float)
        # Bollinger bandwidth (features/bollinger.compute's
        # own expressions): width / mid over (period, mult).
        mid = rolling_mean(self.c, cfg.bb_period)
        sd = rolling_std(self.c, cfg.bb_period)
        upper = mid + cfg.bb_mult * sd
        lower = mid - cfg.bb_mult * sd
        with np.errstate(invalid="ignore", divide="ignore"):
            bandwidth = (upper - lower) / mid
        # rolling_percentile's input: NaN bandwidth counts
        # as 0.0 there (np.nan_to_num in bollinger.compute).
        self._bw_input = np.nan_to_num(bandwidth, nan=0.0)
        self._bb_window = int(cfg.bb_pct_window)
        self._squeeze_pct = float(cfg.bb_squeeze_pct)
        self._pct_cache: dict[int, float] = {}
        # ATR, in build_tf_state's own construction — the
        # parity contract in the module docstring is exact,
        # so both halves are the situation engine's:
        # the numerator series is multitf.atr14's Wilder
        # ATR (EWMA alpha=1/14, adjust=False, min_periods=
        # period; its too-short fallback is the plain TR
        # mean) evaluated causally per bar, and the ratio's
        # baseline series is atr_regime.atr_series
        # (verbatim), whose trailing-100 median at each bar
        # is the denominator build_tf_state divides by.
        prev_c = np.concatenate(([self.c[0]], self.c[:-1])) \
            if self.n else np.zeros(0)
        tr = np.maximum(
            self.h - self.l,
            np.maximum(np.abs(self.h - prev_c),
                       np.abs(self.l - prev_c))) \
            if self.n else np.zeros(0)
        wilder = pd.Series(tr).ewm(
            alpha=1.0 / 14, adjust=False, min_periods=14
        ).mean().to_numpy(dtype=float, copy=True) if self.n \
            else np.zeros(0)
        for j in range(min(13, self.n)):
            if not np.isfinite(wilder[j]):
                wilder[j] = float(np.mean(tr[:j + 1])) \
                    if j >= 0 else np.nan
        self._atr = wilder
        series = _atr_series(frame)
        self._atr_base = (series.to_numpy(dtype=float)
                          if len(series) == self.n
                          else np.full(self.n, np.nan))
        self._ratio_cache: dict[int, float | None] = {}

    # -- per-bar reads --------------------------------------

    def bw_percentile(self, j: int) -> float:
        """Bandwidth percentile at bar ``j`` — reader/
        indicators.rolling_percentile's formula over the
        trailing window, NaN-bandwidth-as-zero input."""
        cached = self._pct_cache.get(j)
        if cached is not None:
            return cached
        start = max(0, j - self._bb_window + 1)
        w = self._bw_input[start:j + 1]
        value = self._bw_input[j]
        pct = 100.0 * float(np.mean(w <= value)) if len(w) \
            else 50.0
        self._pct_cache[j] = pct
        return pct

    def squeeze(self, j: int) -> bool:
        """The features organ's squeeze state at bar ``j``
        (bandwidth percentile below the squeeze floor)."""
        if j < 0 or j >= self.n:
            return False
        return bool(self.bw_percentile(j) < self._squeeze_pct)

    def atr(self, j: int) -> float | None:
        value = self._atr[j] if 0 <= j < self.n else np.nan
        return float(value) if np.isfinite(value) else None

    def atr_ratio(self, j: int) -> float | None:
        """ATR vs its trailing-100 median baseline at bar
        ``j`` — build_tf_state's ratio applied causally at
        ``j``: this bar's Wilder ATR over the median of the
        last 100 finite atr_series values up to and
        including ``j``."""
        if j in self._ratio_cache:
            return self._ratio_cache[j]
        ratio: float | None = None
        atr = self.atr(j)
        if atr is not None and atr > 0:
            window = self._atr_base[:j + 1]
            window = window[np.isfinite(window)][-100:]
            if len(window):
                baseline = float(np.median(window))
                if baseline > 0:
                    ratio = atr / baseline
        self._ratio_cache[j] = ratio
        return ratio

    def atr_state(self, j: int) -> str | None:
        """The situation engine's word at bar ``j``:
        "contracting" below ATR_CONTRACTING (the only word
        this family reads; the full regime vocabulary stays
        atr_regime's)."""
        ratio = self.atr_ratio(j)
        if ratio is None:
            return None
        return "contracting" if ratio < ATR_CONTRACTING \
            else "not_contracting"

    def body_dominated(self, j: int, direction: int) -> bool:
        """The displacement doctrine's candle read
        (DISP_BODY_WICK_MIN): the bar's body in
        ``direction`` is at least 1.5x its total wick."""
        o, h, l, c = self.o[j], self.h[j], self.l[j], self.c[j]
        body = (c - o) * direction
        if body <= 0:
            return False
        wick = (h - max(o, c)) + (min(o, c) - l)
        return body >= DISP_BODY_WICK_MIN * max(wick, 0.0)


# ------------------------------------------------------------------
# the coil
# ------------------------------------------------------------------

def coil_ending_at(prep: PreparedFrame, end: int) -> dict | None:
    """The coil found by the squeeze run reaching bar ``end``.

    A coil exists when the squeeze state held for at least
    MIN_COIL_BARS consecutive bars AND the ATR state at the
    coil's end is contracting.  The returned ``end`` is the
    coil's TRUE last bar, which can precede ``end``: the
    run is truncated at the first bar whose close escapes
    the envelope the run's earlier bars formed (that bar is
    the escape, not a coil bar — see the truncation note in
    the body).  Geometry is the run's own high/low envelope
    — measured from the volatility state, never from a
    range object."""
    if end < 0 or end >= prep.n or not prep.squeeze(end):
        return None
    start = end
    floor = max(0, end - MAX_COIL_BARS + 1)
    while start - 1 >= floor and prep.squeeze(start - 1):
        start -= 1
    # Envelope truncation: the squeeze STATE (a bandwidth
    # percentile against a long history) stays compressed
    # for several bars INTO an escape — those bars' rolling
    # windows are still coil-dominated — so a raw squeeze
    # run swallows the escape itself and the coil's edge
    # runs away from price.  A bar whose CLOSE escapes the
    # envelope the run's earlier bars formed (by the trigger
    # penetration) is not a coil bar; it is the escape.  The
    # coil ends the bar before the first escape.
    env_high = float(prep.h[start])
    env_low = float(prep.l[start])
    true_end = end
    for k in range(start + 1, end + 1):
        atr_k = prep.atr(k)
        pen = TRIGGER_PENETRATION_ATR * atr_k if atr_k else 0.0
        if prep.c[k] > env_high + pen \
                or prep.c[k] < env_low - pen:
            true_end = k - 1
            break
        env_high = max(env_high, float(prep.h[k]))
        env_low = min(env_low, float(prep.l[k]))
    end = true_end
    run = end - start + 1
    if run < MIN_COIL_BARS:
        return None
    if prep.atr_state(end) != "contracting":
        return None
    atr = prep.atr(end)
    if atr is None:
        return None
    high = float(np.max(prep.h[start:end + 1]))
    low = float(np.min(prep.l[start:end + 1]))
    if not high > low:
        return None
    return {
        "start": int(start), "end": int(end), "bars": int(run),
        "high": high, "low": low,
        "mid": (high + low) / 2.0, "width": high - low,
        "atr": float(atr),
        "atr_ratio": prep.atr_ratio(end),
        "bw_percentile": prep.bw_percentile(end),
    }


# ------------------------------------------------------------------
# the sequence: coil → trigger → acceptance / rejection
# ------------------------------------------------------------------

def _is_trigger(prep: PreparedFrame, g: int, coil: dict,
                direction: int) -> bool:
    """A displacement close beyond the coil: the close
    clears the edge by the penetration buffer (of the
    coil's own ATR) on a body-dominated bar."""
    edge = coil["high"] if direction > 0 else coil["low"]
    close = prep.c[g]
    if direction > 0:
        if not close > edge + TRIGGER_PENETRATION_ATR \
                * coil["atr"]:
            return False
    else:
        if not close < edge - TRIGGER_PENETRATION_ATR \
                * coil["atr"]:
            return False
    return prep.body_dominated(g, direction)


def _is_acceptance(prep: PreparedFrame, a: int, coil: dict,
                   direction: int) -> bool:
    """Acceptance-by-range: the bar closes beyond the coil
    edge and its WHOLE range held beyond it — price did not
    trade back inside the coil at all on the acceptance
    bar (the reaction doctrine: closes accepting beyond
    the range, not a poke through it)."""
    edge = coil["high"] if direction > 0 else coil["low"]
    if direction > 0:
        return bool(prep.c[a] > edge and prep.l[a] >= edge)
    return bool(prep.c[a] < edge and prep.h[a] <= edge)


def _is_rejection(prep: PreparedFrame, a: int, coil: dict,
                  direction: int) -> bool:
    """The escape's rejection: a close back INSIDE the
    coil after the trigger — the market refused the new
    prices and returned to the coil's own ground.  A close
    that punches through the coil's OPPOSITE edge is not a
    rejection (it is an escape the other way, or chaos);
    it does not resolve the sequence — the window simply
    runs, and expiry without acceptance or rejection is
    drift, no hypothesis."""
    return bool(coil["low"] < prep.c[a] < coil["high"])


def sequence_at(prep: PreparedFrame, j: int) -> dict | None:
    """The family's state machine read as of bar ``j``.

    Returns None when no coil sequence is live or recently
    resolved at ``j``; otherwise a dict:

    - ``phase``: "coil" (coiled, awaiting a trigger),
      "triggered" (displacement close printed, awaiting
      acceptance/rejection), "accepted" (the escape's
      entry bar is ``event_bar``), "rejected" (the failed
      expansion's entry bar is ``event_bar``), "expired"
      (a trigger neither accepted nor rejected inside the
      window — drift, no hypothesis).
    - ``coil``: the coil dict (see coil_ending_at).
    - ``direction``: the ESCAPE's direction (the failed
      expansion trades against it).
    - ``trigger_bar`` / ``event_bar``: bar indices.
    - ``extreme``: the escape's excursion extreme through
      the resolution bar (the failed expansion's stop
      anchor).

    The most recent coil owns the read: an older coil is
    never resurrected once a newer one has formed.
    """
    if j < 1 or j >= prep.n:
        return None
    coil = _latest_coil(prep, j)
    if coil is None:
        return None
    e = int(coil["end"])
    if e == j:
        # Still coiling at j: the run ending at j IS the state.
        return {"phase": "coil", "coil": coil,
                "direction": 0, "trigger_bar": None,
                "event_bar": None, "extreme": None}
    # Trigger scan: the first displacement close beyond
    # either edge inside the window after the coil.
    trigger_bar = None
    direction = 0
    g_last = min(j, e + ACCEPT_WINDOW_BARS)
    for g in range(e + 1, g_last + 1):
        for d in (1, -1):
            if _is_trigger(prep, g, coil, d):
                trigger_bar, direction = g, d
                break
        if trigger_bar is not None:
            break
    if trigger_bar is None:
        # Armed while the trigger window is open; beyond it
        # the sequence is over (the coil premise decayed
        # without an escape — no record beyond the scans'
        # abstentions).
        if j <= e + ACCEPT_WINDOW_BARS:
            return {"phase": "coil", "coil": coil,
                    "direction": 0, "trigger_bar": None,
                    "event_bar": None, "extreme": None}
        return None
    # Resolution scan: acceptance and rejection race; the
    # first resolving close wins.
    a_last = min(j, trigger_bar + ACCEPT_WINDOW_BARS)
    extreme = None
    for a in range(trigger_bar, a_last + 1):
        seg_h = float(np.max(prep.h[trigger_bar:a + 1]))
        seg_l = float(np.min(prep.l[trigger_bar:a + 1]))
        extreme = seg_h if direction > 0 else seg_l
        if a == trigger_bar:
            continue
        if _is_acceptance(prep, a, coil, direction):
            return {"phase": "accepted", "coil": coil,
                    "direction": direction,
                    "trigger_bar": trigger_bar,
                    "event_bar": a, "extreme": extreme}
        if _is_rejection(prep, a, coil, direction):
            return {"phase": "rejected", "coil": coil,
                    "direction": direction,
                    "trigger_bar": trigger_bar,
                    "event_bar": a, "extreme": extreme}
    phase = ("triggered"
             if j <= trigger_bar + ACCEPT_WINDOW_BARS
             else "expired")
    return {"phase": phase, "coil": coil,
            "direction": direction,
            "trigger_bar": trigger_bar,
            "event_bar": None, "extreme": extreme}


def _latest_coil(prep: PreparedFrame, j: int) -> dict | None:
    """The coil that owns bar ``j``'s read: the coil ending
    at ``j`` itself when one is live there; otherwise the
    most recent coil whose (truncated) end is still inside
    the sequence windows reaching ``j``.  Older coils are
    never resurrected once a newer sequence has run."""
    coil = coil_ending_at(prep, j)
    if coil is not None:
        return coil
    floor_e = max(0, j - ACCEPT_WINDOW_BARS - 3)
    for e in range(j - 1, floor_e - 1, -1):
        coil = coil_ending_at(prep, e)
        if coil is not None:
            return coil
    return None


# ------------------------------------------------------------------
# geometry (the hypotheses' tradable shape)
# ------------------------------------------------------------------

def escape_geometry(prep: PreparedFrame, seq: dict) -> dict:
    """The accepted escape's geometry: entry at the
    acceptance close; stop just inside the coil edge (the
    declared invalidation — a close back inside voids the
    escape); target the measured move — the coil's width
    projected from the ENTRY actually transacted (the
    classic edge-projected objective can already be behind
    a late acceptance entry after a violent displacement;
    a target behind the entry is not an objective)."""
    coil = seq["coil"]
    d = int(seq["direction"])
    a = int(seq["event_bar"])
    edge = coil["high"] if d > 0 else coil["low"]
    entry = float(prep.c[a])
    return {
        "entry": entry,
        "stop": float(edge - d * STOP_BUFFER_ATR * coil["atr"]),
        "targets": [float(entry + d * coil["width"])],
        "atr": prep.atr(a) or coil["atr"],
        "edge": float(edge),
    }


def failed_geometry(prep: PreparedFrame, seq: dict) -> dict:
    """The failed expansion's geometry (traded AGAINST the
    escape's direction): entry at the rejection close;
    stop beyond the escape's excursion extreme + the
    standing buffer (a close back through it voids the
    rejection); targets the coil mid, then the opposite
    edge (the range family's fade convention) — each kept
    only when it is actually AHEAD of the entry (a
    rejection closing in the coil's far half has already
    traversed the mid; a target behind the entry is not an
    objective).  The opposite edge is ahead by construction
    (the rejection close is inside the coil), so the list
    is never empty."""
    coil = seq["coil"]
    d = int(seq["direction"])          # the escape's direction
    r = int(seq["event_bar"])
    extreme = float(seq["extreme"])
    entry = float(prep.c[r])
    fade = -d                          # the trade's direction
    raw = [float(coil["mid"]),
           float(coil["low"] if d > 0 else coil["high"])]
    targets = [t for t in raw if (t - entry) * fade > 0]
    return {
        "entry": entry,
        "stop": float(extreme + d * STOP_BUFFER_ATR
                      * coil["atr"]),
        "targets": targets,
        "atr": prep.atr(r) or coil["atr"],
        "edge": float(coil["high"] if d > 0 else coil["low"]),
        "extreme": extreme,
    }
