"""Trend / momentum — the family's measurement core
(MarketSituation Pass 7b, SPEC §6.1).

The behavior: directional continuation as a STATE, not a
pattern.  A trend is a relationship fact — the formation
timeframe's structure direction WITH the trigger timeframe's
participation — and momentum is the market's follow-through
quality: displacement sustained by closes, or momentum
decaying into absorption (price moving without progressing —
the follow-through being eaten).  Two hypotheses, one trend
sequence:

    impulse ──▶ pullback (holds the impulse's structure)
                    │
                    ├── LTF structure turn ──▶ IMPULSE PULLBACK
                    ├── formation close beyond the impulse
                    │   origin ──▶ INVALIDATED (the trend
                    │   claim died on its own timeframe)
                    └── too deep / too old ──▶ EXPIRED

    momentum_acceleration is a per-bar continuation read on
    the same trend state: velocity rising through a stated
    percentile of its own distribution while closes accept
    beyond the last confirmed swing.

Measurement discipline (the situation engine's honesty law,
inherited): the trend STATE is read from the MarketSituation
(per-TF bias / structure phase / relationship kinds — the
playbook layer owns that read, live/trend_momentum's sibling
in playbooks/); THIS module is the pure bar math the
situation does not carry — structure swings, momentum
quality, velocity — computed with the desk's own organs'
formulas (multitf.find_swings for structure, the Wilder ATR
construction build_tf_state uses, displacement's body/wick
doctrine for displacement closes), so a read here can never
disagree with the situation engine on identical bars.

Phase is read from STRUCTURE, never from an EMA: the
impulse is a close beyond a confirmed swing; the pullback's
life and death are closes against confirmed swings.

Distinctness (SPEC §6 discipline): ``trendline_retest`` and
the ``fvg_*`` playbooks are LEVEL-RETURN patterns (price
comes back to a drawn level / gap and reacts).  This family
reads no level: its trigger is the momentum state
transition — the pullback's structure turn on the trigger
clock — and its invalidation is the trend's structural
death point, not a level's far edge.  ``momentum_
acceleration`` is not a breakout playbook either: it fires
only inside an already-established trend state whose
velocity is rising through its own distribution — a
breakout in a non-trend state is another family's ground
(or nobody's).

Every read is causal: the state at bar ``j`` is a function
of bars ``<= j`` only (swings are confirmed fractal swings —
a swing at position p is known only from bar p + SWING_N;
the trigger read at formation bar j uses only trigger bars
closed by formation bar j's close).  Judged by what was
observable at the time, never hindsight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import multitf
from .displacement import DISP_BODY_WICK_MIN

#: The family's formation clocks (SPEC §6.1: formation on
#: the horizon TF — H4 intraday / D1 swing; trigger one TF
#: down).  M15/M5 formations belong to the scalp sleeve's
#: own clocks; H1-formed trend reads are the roster's home
#: ground, not this family's.
FORMATION_TFS: tuple[str, ...] = ("D1", "H4")

#: Trigger clock one step down from each formation clock.
#: The entry transacts on the trigger clock (the LTF turn
#: IS the entry event), so ENTRY == TRIGGER for this
#: family — stated here, not assumed downstream.
TRIGGER_TF_DOWN: dict[str, str] = {"D1": "H4", "H4": "H1"}

#: The inverse ladder: the formation clock a scan on this
#: trigger clock belongs to (the runner's threading reads
#: it — a scan on H1 is an H4-formed trend read, a scan on
#: H4 is a D1-formed one).
FORMATION_TF_UP: dict[str, str] = {
    v: k for k, v in TRIGGER_TF_DOWN.items()}

#: The clock above each formation clock (the relationship
#: context pair; D1 has none — it is the stack's top).
ABOVE_TF: dict[str, str | None] = {"D1": None, "H4": "D1"}

#: Bar durations, for causal cross-clock slicing (a trigger
#: bar is visible at a formation bar's close only when the
#: trigger bar itself has closed by then).
TF_DELTAS: dict[str, pd.Timedelta] = {
    "D1": pd.Timedelta(days=1), "H4": pd.Timedelta(hours=4),
    "H1": pd.Timedelta(hours=1), "M30": pd.Timedelta(minutes=30),
    "M15": pd.Timedelta(minutes=15), "M5": pd.Timedelta(minutes=5),
    "M1": pd.Timedelta(minutes=1),
}

#: Confirmed-swing geometry on the formation clock — the
#: situation engine's own swing_n (build_tf_state), so the
#: family's structure read IS the situation's structure
#: read on identical bars.
SWING_N = 3

#: Minor-swing geometry on the trigger clock: the LTF turn
#: is a minor structure shift (the fractal n=2 geometry the
#: desk's BW/minor reads use) — the pullback's own clock
#: turning, not the formation clock's major swings.
TRIGGER_SWING_N = 2

#: The structural stop sits this far beyond the defended
#: swing (the desk's standing buffer scale —
#: playbooks/range_regime.STOP_BUFFER_ATR, the vol
#: family's coil stops): the invalidation PRICE is the
#: structure point; this is only the execution buffer
#: around it, never a fixed-distance stop.
STOP_BUFFER_ATR = 0.25

#: A counter-move is a PULLBACK of the impulse leg inside
#: this retrace band (fraction of the leg, extreme to
#: extreme).  Shallower is a pause (the leg is still
#: extending — the acceleration read's ground); deeper has
#: consumed the leg the trend claim stands on.
MIN_RETRACE_FRAC = 0.15
MAX_RETRACE_FRAC = 0.85

#: Formation bars a pullback may run without turning
#: before the sequence is spent (the thesis is the turn;
#: a pullback that never turns is a reversal in progress).
MAX_PULLBACK_BARS = 12

#: Momentum quality is read over this trailing formation
#: window (velocity / efficiency / displacement closes).
MOMENTUM_WINDOW = 8

#: Velocity: ATR-normalised close velocity over this many
#: formation bars (the situation schema's velocity_atr
#: construction — close displacement per bar, in ATR).
VELOCITY_BARS = 6

#: The trailing window a velocity read is percentiled
#: against (its OWN distribution — SPEC §6.1's stated
#: construction for the acceleration hypothesis).
VELOCITY_DIST_WINDOW = 100

#: Acceleration requires velocity at or above this
#: percentile of its own trailing distribution.
VELOCITY_PERCENTILE = 80.0

#: Minimum velocity samples before a percentile is a
#: measurement rather than a guess (absent stays absent).
VELOCITY_MIN_SAMPLES = 20

#: Momentum state words (this module's vocabulary, carried
#: on every signal / arm record the family produces).
MOMENTUM_SUSTAINED = "sustained"
MOMENTUM_DECAYING = "decaying"
MOMENTUM_ABSORBED = "absorbed"


# ------------------------------------------------------------------
# series preparation (the organs' own formulas, per bar)
# ------------------------------------------------------------------

class PreparedFrame:
    """One frame's detector series, prepared once per scan.

    ATR is build_tf_state's own construction (multitf's
    Wilder ATR14 — EWMA alpha=1/14 — with the too-short
    fallback to the plain TR mean), evaluated causally per
    bar, so this module's ATR at bar ``j`` equals the
    situation engine's on the frame truncated at ``j``.
    Swings are multitf.find_swings over the full frame; a
    swing at position ``p`` is only READ at bars
    ``>= p + swing_n`` (its confirmation bar) — the
    no-lookahead rule find_swings itself is built on.
    """

    def __init__(self, frame: pd.DataFrame, *,
                 swing_n: int = SWING_N):
        self.frame = frame
        self.n = len(frame)
        self.swing_n = int(swing_n)
        self.o = frame["open"].to_numpy(dtype=float)
        self.h = frame["high"].to_numpy(dtype=float)
        self.l = frame["low"].to_numpy(dtype=float)
        self.c = frame["close"].to_numpy(dtype=float)
        prev_c = np.concatenate(([self.c[0]], self.c[:-1])) \
            if self.n else np.zeros(0)
        tr = np.maximum(
            self.h - self.l,
            np.maximum(np.abs(self.h - prev_c),
                       np.abs(self.l - prev_c))) \
            if self.n else np.zeros(0)
        wilder = pd.Series(tr).ewm(
            alpha=1.0 / 14, adjust=False, min_periods=14
        ).mean().to_numpy(dtype=float) if self.n \
            else np.zeros(0)
        for j in range(min(13, self.n)):
            if not np.isfinite(wilder[j]):
                wilder[j] = float(np.mean(tr[:j + 1])) \
                    if j >= 0 else np.nan
        self._atr = wilder
        try:
            highs, lows = multitf.find_swings(
                frame["high"], frame["low"], self.swing_n)
        except Exception:  # noqa: BLE001 — malformed frame
            highs, lows = [], []  # degrades to absent swings
        self.swing_highs: list[tuple[int, float]] = list(highs)
        self.swing_lows: list[tuple[int, float]] = list(lows)
        self._vel_cache: dict[tuple[int, int], float | None] = {}
        self._vpct_cache: dict[tuple[int, int], float | None] = {}

    # -- per-bar reads --------------------------------------

    def atr(self, j: int) -> float | None:
        value = self._atr[j] if 0 <= j < self.n else np.nan
        return float(value) if np.isfinite(value) else None

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

    def swings_known(self, j: int, *, highs: bool
                     ) -> list[tuple[int, float]]:
        """Confirmed swings READABLE at bar ``j``: swing
        position + swing_n <= j (the confirmation rule)."""
        src = self.swing_highs if highs else self.swing_lows
        return [(p, v) for p, v in src
                if p + self.swing_n <= j]

    def velocity(self, j: int, direction: int) -> float | None:
        """ATR-normalised close velocity in ``direction``
        over VELOCITY_BARS ending at ``j`` (signed: positive
        = progressing with the direction)."""
        key = (j, direction)
        if key in self._vel_cache:
            return self._vel_cache[key]
        value: float | None = None
        atr = self.atr(j)
        if atr and atr > 0 and j - VELOCITY_BARS >= 0:
            value = float(
                (self.c[j] - self.c[j - VELOCITY_BARS])
                * direction / (VELOCITY_BARS * atr))
        self._vel_cache[key] = value  # type: ignore[assignment]
        return value

    def velocity_percentile(self, j: int,
                            direction: int) -> float | None:
        """Where bar ``j``'s velocity sits in its own
        trailing distribution (VELOCITY_DIST_WINDOW values
        ending at ``j``) — reader/indicators'
        rolling_percentile formula, as the vol core's
        bandwidth percentile.  None below the sample floor:
        an unmeasured percentile is absent, never guessed."""
        key = (j, direction)
        if key in self._vpct_cache:
            return self._vpct_cache[key]
        pct: float | None = None
        values: list[float] = []
        start = max(VELOCITY_BARS, j - VELOCITY_DIST_WINDOW + 1)
        for i in range(start, j + 1):
            v = self.velocity(i, direction)
            if v is not None:
                values.append(v)
        if len(values) >= VELOCITY_MIN_SAMPLES:
            current = self.velocity(j, direction)
            if current is not None:
                pct = 100.0 * float(np.mean(
                    np.asarray(values) <= current))
        self._vpct_cache[key] = pct
        return pct

    # -- momentum quality -----------------------------------

    def momentum_at(self, j: int, direction: int) -> dict:
        """The follow-through quality read at bar ``j`` in
        ``direction``: net progress vs path (efficiency),
        velocity, and displacement-close counts over the
        trailing MOMENTUM_WINDOW bars.

        - ``sustained``: real net progress (>= 1 ATR over
          the window) at trend efficiency, with at least as
          many displacement closes FOR as against — the
          displacement is being sustained by closes.
        - ``absorbed``: churn — movement without progress
          (|net| <= 0.3 ATR at low efficiency), or counter
          displacement closes dominating while net progress
          dies: the follow-through is being absorbed.
        - ``decaying``: progress exists but the quality is
          fading between the two — evidence, weighed by the
          hypotheses, never a silent middle.
        """
        w = min(MOMENTUM_WINDOW, j)
        atr = self.atr(j)
        out = {"state": MOMENTUM_DECAYING, "net_atr": None,
               "efficiency": None, "velocity": None,
               "disp_for": 0, "disp_against": 0,
               "window": int(w)}
        if atr is None or atr <= 0 or w < 2:
            return out
        net = float((self.c[j] - self.c[j - w]) * direction
                    / atr)
        path = float(np.sum(np.abs(np.diff(
            self.c[j - w:j + 1])))) / atr
        eff = abs(net) / path if path > 0 else 0.0
        disp_for = sum(1 for i in range(j - w + 1, j + 1)
                       if self.body_dominated(i, direction))
        disp_against = sum(1 for i in range(j - w + 1, j + 1)
                           if self.body_dominated(i, -direction))
        out.update({"net_atr": net, "efficiency": float(eff),
                    "velocity": self.velocity(j, direction),
                    "disp_for": int(disp_for),
                    "disp_against": int(disp_against)})
        if net >= 1.0 and eff >= 0.30 \
                and disp_for >= disp_against:
            out["state"] = MOMENTUM_SUSTAINED
        elif (abs(net) <= 0.30 and eff < 0.20) or (
                disp_against >= disp_for + 2 and net <= 0.30):
            out["state"] = MOMENTUM_ABSORBED
        return out


# ------------------------------------------------------------------
# cross-clock visibility
# ------------------------------------------------------------------

def last_visible_bar(f_frame: pd.DataFrame, f_idx: int,
                     t_frame: pd.DataFrame, *,
                     formation_tf: str,
                     trigger_tf: str) -> int:
    """The last trigger-frame bar CLOSED by formation bar
    ``f_idx``'s close (bar end = index + the TF's delta).
    -1 when none is closed yet.  Non-datetime indices (a
    research frame without a clock) fall back to the
    trigger frame's last bar — the caller's fixtures carry
    real clocks wherever causality is being pinned."""
    if t_frame is None or len(t_frame) == 0:
        return -1
    f_idx_ts = f_frame.index[f_idx]
    t_index = t_frame.index
    if not isinstance(f_idx_ts, pd.Timestamp) \
            or not isinstance(t_index, pd.DatetimeIndex):
        return len(t_frame) - 1
    f_end = f_idx_ts + TF_DELTAS[formation_tf]
    t_delta = TF_DELTAS[trigger_tf]
    ends = t_index + t_delta
    pos = int(ends.searchsorted(f_end, side="right")) - 1
    return pos


def last_formation_bar(t_frame: pd.DataFrame, m: int,
                       f_frame: pd.DataFrame, *,
                       formation_tf: str,
                       trigger_tf: str) -> int:
    """The last formation-frame bar CLOSED by trigger bar
    ``m``'s close — the scan direction's visibility rule
    (the family scans on the trigger clock; the formation
    state it reads is the last formation bar that had
    closed when the trigger bar closed).  -1 when no
    formation bar has closed yet.  Same non-datetime
    fallback as last_visible_bar."""
    if f_frame is None or len(f_frame) == 0:
        return -1
    t_ts = t_frame.index[m]
    f_index = f_frame.index
    if not isinstance(t_ts, pd.Timestamp) \
            or not isinstance(f_index, pd.DatetimeIndex):
        return len(f_frame) - 1
    t_end = t_ts + TF_DELTAS[trigger_tf]
    ends = f_index + TF_DELTAS[formation_tf]
    return int(ends.searchsorted(t_end, side="right")) - 1


def first_bar_after(f_frame: pd.DataFrame, f_idx: int,
                    t_frame: pd.DataFrame, *,
                    formation_tf: str,
                    trigger_tf: str) -> int:
    """The first trigger-frame bar that OPENS at or after
    formation bar ``f_idx``'s close — the pullback window's
    first trigger bar (the extreme bar itself is formation
    ground; the turn hunt starts when it has closed)."""
    if t_frame is None or len(t_frame) == 0:
        return 0
    f_idx_ts = f_frame.index[f_idx]
    t_index = t_frame.index
    if not isinstance(f_idx_ts, pd.Timestamp) \
            or not isinstance(t_index, pd.DatetimeIndex):
        return 0
    f_end = f_idx_ts + TF_DELTAS[formation_tf]
    return int(t_index.searchsorted(f_end, side="left"))


# ------------------------------------------------------------------
# the trend sequence: impulse -> pullback -> turn / death / expiry
# ------------------------------------------------------------------

def _latest_impulse(prep: PreparedFrame, j: int) -> dict | None:
    """The most recent impulse leg readable at bar ``j``:
    a formation close beyond the then-last confirmed swing
    in direction d, whose origin is the last confirmed
    opposite swing known at the break, and whose follow-
    through at its extreme read SUSTAINED (a break without
    sustained follow-through is drift, not an impulse —
    the sequence never starts).  A later opposing break
    supersedes: the walk keeps only the newest leg whose
    direction still owns the structure at ``j``."""
    best: dict | None = None
    for d in (1, -1):
        highs = prep.swings_known(j, highs=True)
        lows = prep.swings_known(j, highs=False)
        # Break candidates: bars b whose close cleared the
        # last swing (in direction d) confirmed BEFORE b.
        swing_src = highs if d > 0 else lows
        origin_src = lows if d > 0 else highs
        for (p, level) in swing_src:
            # First close beyond this swing after it was
            # confirmed and before the next swing on the
            # same side was confirmed (that swing replaces
            # it as the level to break).
            start = p + prep.swing_n
            later = [q for (q, _v) in swing_src if q > p]
            stop = min(later) + prep.swing_n if later else j + 1
            stop = min(stop, j + 1)
            b = None
            for k in range(start, stop):
                if (prep.c[k] - level) * d > 0:
                    b = k
                    break
            if b is None:
                continue
            origins = [(q, v) for (q, v) in origin_src
                       if q + prep.swing_n <= b]
            if not origins:
                continue
            o_pos, o_price = origins[-1]
            # The extreme of the leg through j.
            seg = prep.h[b:j + 1] if d > 0 else prep.l[b:j + 1]
            e = b + int(np.argmax(seg)) if d > 0 \
                else b + int(np.argmin(seg))
            extreme = float(prep.h[e]) if d > 0 \
                else float(prep.l[e])
            if (extreme - o_price) * d <= 0:
                continue
            # Momentum is read at the leg's extreme CLOSE,
            # not at the price-extreme bar: the bar after a
            # top opens at the top, so its high can edge out
            # the true extreme while its close already
            # belongs to the pullback — reading momentum
            # there would judge the leg by its first
            # counter bar.  Geometry keeps the price
            # extreme; the momentum bar rides alongside.
            cseg = prep.c[b:j + 1]
            e_c = b + int(np.argmax(cseg)) if d > 0 \
                else b + int(np.argmin(cseg))
            mom = prep.momentum_at(e_c, d)
            if mom["state"] != MOMENTUM_SUSTAINED:
                continue
            cand = {"direction": d, "break_bar": int(b),
                    "break_level": float(level),
                    "origin_pos": int(o_pos),
                    "origin_price": float(o_price),
                    "extreme_bar": int(e),
                    "extreme_price": extreme,
                    "momentum_bar": int(e_c),
                    "leg": float(abs(extreme - o_price)),
                    "momentum_at_extreme": mom}
            if best is None or cand["break_bar"] \
                    > best["break_bar"]:
                best = cand
    return best


def _counter_sustained(prep: PreparedFrame, imp: dict,
                       j: int) -> bool:
    """Did the pullback itself displace AGAINST the trend —
    sustained counter follow-through (the pullback is a
    counter-impulse, the trend's follow-through absorbed
    by the other side)?  Measured over the pullback span:
    counter progress at trend efficiency covering most of
    the leg, on displacement closes."""
    d = imp["direction"]
    # The pullback span starts at the leg's extreme CLOSE
    # (the momentum bar): the price-extreme bar can be the
    # pullback's own first bar (it opens at the extreme),
    # and spanning from its close would erase the drop's
    # first formation bar from the counter measure.
    e = imp.get("momentum_bar", imp["extreme_bar"])
    if j <= e:
        return False
    atr = prep.atr(j)
    if atr is None or atr <= 0:
        return False
    drop = float((prep.c[e] - prep.c[j]) * d) / atr
    path = float(np.sum(np.abs(np.diff(prep.c[e:j + 1])))) / atr
    if path <= 0:
        return False
    eff = drop / path
    counter = sum(1 for i in range(e + 1, j + 1)
                  if prep.body_dominated(i, -d))
    leg_atr = imp["leg"] / atr
    return bool(eff >= 0.50 and drop >= 0.60 * leg_atr
                and counter >= 2)


def _trigger_turn(prep_f: PreparedFrame, prep_t: PreparedFrame,
                  imp: dict, j: int, m: int, *,
                  formation_tf: str,
                  trigger_tf: str) -> dict | None:
    """The pullback's turn on the trigger clock, evaluated
    AT trigger bar ``m`` (the scan bar — the turn must be
    THIS bar's close; a turn bars ago is history the entry
    missed): after the pullback's trigger-clock extreme,
    the first confirmed minor swing (TRIGGER_SWING_N) in
    the trend's direction forms, and bar ``m`` closes
    beyond it.  The minor structure turned; the close is
    the entry."""
    d = imp["direction"]
    if m < 0 or m >= prep_t.n:
        return None
    start = first_bar_after(
        prep_f.frame, imp["extreme_bar"], prep_t.frame,
        formation_tf=formation_tf, trigger_tf=trigger_tf)
    if start > m:
        return None
    window = prep_t.l[start:m + 1] if d > 0 \
        else prep_t.h[start:m + 1]
    if len(window) == 0:
        return None
    x = start + (int(np.argmin(window)) if d > 0
                 else int(np.argmax(window)))
    # Confirmed minor swings in the trend's direction
    # formed at or after the pullback extreme bar.
    src = prep_t.swing_highs if d > 0 else prep_t.swing_lows
    levels = [(p, v) for (p, v) in src
              if p >= x and p + prep_t.swing_n <= m]
    if not levels:
        return None
    p, level = levels[-1]
    if (prep_t.c[m] - level) * d <= 0:
        return None
    # The turn close must be the FIRST close beyond that
    # swing since it was confirmed (no stale turns).
    for k in range(p + prep_t.swing_n, m):
        if (prep_t.c[k] - level) * d > 0:
            return None
    return {"bar": int(m), "swing_pos": int(p),
            "swing_price": float(level),
            "extreme_bar": int(x),
            "close": float(prep_t.c[m])}


def sequence_at(prep_f: PreparedFrame, prep_t: PreparedFrame,
                j: int, m: int, *, formation_tf: str,
                trigger_tf: str) -> dict | None:
    """The family's trend-sequence read as of formation bar
    ``j`` / trigger bar ``m`` (both causal: formation bars
    <= j, trigger bars <= m, and ``m`` no later than the
    last trigger bar closed by formation bar ``j``).

    Returns None when no impulse sequence is live at ``j``;
    otherwise a dict with ``phase``:

    - ``impulse``: the leg is live and no qualifying
      pullback has formed yet (pause or still extending);
    - ``pullback``: a qualifying pullback is in progress,
      structure intact, awaiting the trigger turn;
    - ``triggered``: trigger bar ``m`` IS the turn close
      (``event_bar`` == m; the hypothesis' entry bar);
    - ``invalidated``: a formation close beyond the
      impulse origin during the pullback — the trend claim
      died on its declaring timeframe (``break_bar``);
    - ``expired``: the pullback ran past its age or depth
      bound without turning — spent, no hypothesis.

    Plus the impulse geometry (origin / extreme / leg /
    break), the pullback read (extreme, depth fraction,
    counter-sustained flag) and, when triggered, the turn.
    The most recent impulse owns the read.
    """
    if j < 1 or j >= prep_f.n:
        return None
    imp = _latest_impulse(prep_f, j)
    if imp is None:
        return None
    d = imp["direction"]
    e = imp["extreme_bar"]
    out: dict = {"impulse": imp, "direction": d,
                 "phase": "impulse", "pullback": None,
                 "turn": None, "break_bar": None}
    if e >= j:
        return out
    # Pullback geometry through j (extremes for depth,
    # closes for the structural death test).
    if d > 0:
        p_idx = e + int(np.argmin(prep_f.l[e + 1:j + 1])) + 1 \
            if j > e else e
        p_price = float(prep_f.l[p_idx])
    else:
        p_idx = e + int(np.argmax(prep_f.h[e + 1:j + 1])) + 1 \
            if j > e else e
        p_price = float(prep_f.h[p_idx])
    depth = float((imp["extreme_price"] - p_price) * d
                  / imp["leg"]) if imp["leg"] > 0 else 0.0
    out["pullback"] = {
        "extreme_bar": int(p_idx), "extreme_price": p_price,
        "depth_frac": depth,
        "counter_sustained": _counter_sustained(prep_f, imp, j)}
    # Structural death: a formation CLOSE beyond the
    # impulse origin during the pullback.
    for k in range(e + 1, j + 1):
        if (prep_f.c[k] - imp["origin_price"]) * d < 0:
            out["phase"] = "invalidated"
            out["break_bar"] = int(k)
            return out
    if depth < MIN_RETRACE_FRAC:
        return out                      # a pause, not a pullback
    if depth > MAX_RETRACE_FRAC or (j - e) > MAX_PULLBACK_BARS:
        out["phase"] = "expired"
        return out
    out["phase"] = "pullback"
    # The turn, on the trigger clock, at the scan bar.
    if m >= 0:
        turn = _trigger_turn(prep_f, prep_t, imp, j, m,
                             formation_tf=formation_tf,
                             trigger_tf=trigger_tf)
        if turn is not None:
            # The pullback entry must sit on the pullback's
            # side of the impulse extreme — a turn close
            # already beyond the extreme is the leg
            # extending (the acceleration read's ground),
            # not a pullback entry.
            if (turn["close"] - imp["extreme_price"]) * d < 0:
                out["phase"] = "triggered"
                out["turn"] = turn
    return out


# ------------------------------------------------------------------
# geometry (the hypotheses' tradable shape)
# ------------------------------------------------------------------

def pullback_geometry(prep_f: PreparedFrame, seq: dict,
                      turn: dict, *, at_bar: int) -> dict:
    """The impulse-pullback trade: entry at the trigger
    turn's close (the trigger clock's price — the entry
    transacts where the turn printed); stop beyond the
    pullback's defended swing on the FORMATION clock (the
    pullback extreme — a formation close beyond it after
    the turn is the structure break that kills the trend
    claim) plus the standing buffer; targets the impulse
    extreme first (the retest the continuation must win),
    then the leg's measured projection from the extreme —
    each kept only when actually ahead of the entry."""
    imp = seq["impulse"]
    d = imp["direction"]
    atr = prep_f.atr(at_bar) or prep_f.atr(prep_f.n - 1) or 0.0
    entry = float(turn["close"])
    defended = float(seq["pullback"]["extreme_price"])
    raw = [float(imp["extreme_price"]),
           float(imp["extreme_price"] + d * imp["leg"])]
    targets = [t for t in raw if (t - entry) * d > 0]
    return {
        "entry": entry,
        "stop": float(defended - d * STOP_BUFFER_ATR * atr),
        "targets": targets,
        "atr": float(atr),
        "defended_swing": defended,
        "origin_price": float(imp["origin_price"]),
    }


def acceleration_at(prep_f: PreparedFrame, j: int) -> dict | None:
    """The momentum-acceleration read at formation bar
    ``j``: velocity at/above VELOCITY_PERCENTILE of its own
    trailing distribution AND rising, momentum sustained,
    and a FRESH acceptance — a close beyond the last
    confirmed swing in the trend's direction within the
    last two bars, every close since holding beyond it.
    Direction comes from the fresh break itself.  None
    when any leg is unmeasured or unmet (the situation's
    participation gate is the playbook layer's)."""
    if j < 2 or j >= prep_f.n:
        return None
    for d in (1, -1):
        mom = prep_f.momentum_at(j, d)
        if mom["state"] != MOMENTUM_SUSTAINED:
            continue
        pct = prep_f.velocity_percentile(j, d)
        if pct is None or pct < VELOCITY_PERCENTILE:
            continue
        v_now = prep_f.velocity(j, d)
        v_prev = prep_f.velocity(j - 3, d) if j >= 3 else None
        if v_now is None or v_prev is None or not v_now > v_prev:
            continue
        swings = prep_f.swings_known(
            j, highs=(d > 0))
        if not swings:
            continue
        s_pos, s_price = swings[-1]
        # Fresh break: the first close beyond the swing
        # happened at j-1 or j, and no close since fell
        # back beyond it.
        b = None
        for k in range(max(s_pos + prep_f.swing_n, j - 1),
                       j + 1):
            if (prep_f.c[k] - s_price) * d > 0:
                b = k
                break
        if b is None:
            continue
        if any((prep_f.c[k] - s_price) * d <= 0
               for k in range(b, j + 1)):
            continue
        # The leg being extended: from the last confirmed
        # opposite swing before the broken one to the
        # broken swing itself.
        opp = [(p, v) for (p, v) in prep_f.swings_known(
            j, highs=(d < 0)) if p < s_pos]
        if not opp:
            continue
        o_pos, o_price = opp[-1]
        leg = float(abs(s_price - o_price))
        if leg <= 0:
            continue
        atr = prep_f.atr(j)
        if atr is None:
            continue
        entry = float(prep_f.c[j])
        return {
            "direction": d, "bar": int(j),
            "velocity": float(v_now),
            "velocity_percentile": float(pct),
            "momentum": mom,
            "broken_swing": {"pos": int(s_pos),
                             "price": float(s_price)},
            "origin": {"pos": int(o_pos), "price": o_price},
            "leg": leg,
            "entry": entry,
            "stop": float(s_price - d * STOP_BUFFER_ATR * atr),
            "targets": [float(entry + d * leg)],
            "atr": float(atr),
        }
    return None
