"""Auction / value — the family's measurement core
(MarketSituation Pass 7c, SPEC §6.3).

The behavior: the auction itself.  Price probes beyond
the developing value area and the market answers in one
of two ways — it ACCEPTS the probe (closes hold outside
the value that was probed, and developing value migrates
toward price: the auction is re-pricing) or it REJECTS
it (closes return inside, the probe's extreme left
standing untested-by-acceptance: a failed auction, and
the rotation back through value is the trade).  And
where value is THIN — a low-volume node between two
shelves — price traverses fast to the far shelf.

    probe (close beyond the pre-probe value edge)
        │
        ├── >= 2 closes holding outside + value
        │   migrating toward the probe ──▶ ACCEPTED
        │   (entry: the M15 retest that holds outside
        │   the probed edge)
        ├── a close back inside before acceptance
        │   ──▶ REJECTED (entry: rotation toward POC,
        │   then the opposite edge)
        └── neither inside the window ──▶ EXPIRED
    accepted + close back inside with the POC no
    longer migrating ──▶ INVALIDATED (the re-pricing
    did not hold — the thesis died on its own clock)
    rejected + a close beyond the probe extreme
    ──▶ INVALIDATED (the auction re-accepted the
    probe's ground after all)

Measurement discipline: acceptance / rejection are read
from CLOSES against the PRE-PROBE developing value (the
value the probe escaped from — a fixed reference for the
sequence's life) plus the migration of the CURRENT
developing state toward the probe — never from a single
wick (a wick beyond the edge with the close inside forms
no probe at all).  The value states are
``live/developing_value.py``'s — the shared §6.3
extension the situation itself carries — so a read here
can never disagree with the situation's value block on
identical bars.  Value here is a TICK-VOLUME profile
(MT5's volume column); the basis rides every state as
``volume_basis`` and is named, never dressed.

Distinctness (SPEC §6 discipline): the range family
trades a RangeDetector range OBJECT's edges in a D1-flat
regime; this family reads no range object and no regime
verdict — its levels are the developing value area's own,
its trigger is the auction's answer (acceptance /
rejection / traversal), and its invalidation is the
auction fact dying.  ``range_mean_reversion`` fades a
range edge re-entry; a failed auction here is defined by
the probe-and-answer sequence against migrating value,
with the rotation's targets read from value (POC, then
the far edge), not from a range mid.

Every read is causal: the sequence at formation bar
``j`` is a function of bars ``<= j``; the trigger read
at bar ``m`` sees only formation bars closed by trigger
bar ``m``'s close.  Judged by what was observable at
the time, never hindsight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import developing_value as _dv
from .displacement import DISP_BODY_WICK_MIN
from .trend_momentum import TF_DELTAS, first_bar_after, last_formation_bar

__all__ = [
    "ACCEPT_MIN_CLOSES",
    "FORMATION_TFS",
    "FORMATION_TF_UP",
    "RESOLUTION_WINDOW",
    "RETEST_MAX_TRIGGER_BARS",
    "RETEST_TOL_ATR",
    "STOP_BUFFER_ATR",
    "TF_DELTAS",
    "TRIGGER_TF_DOWN",
    "PreparedAuction",
    "auction_sequence_at",
    "body_dominated",
    "first_bar_after",
    "last_formation_bar",
    "lvn_crossing_at",
]

#: The family's clocks (SPEC §6.3): formation on H1 (the
#: developing VA's own clock), trigger M15.  One
#: formation clock — the value area is a day object, and
#: the day's bars are H1 bars by the extension's
#: definition (SPEC §6.3 / §10.2 risk 3: the developing
#: profile must not creep to faster clocks).
FORMATION_TFS: tuple[str, ...] = ("H1",)

#: Trigger clock one step down from the formation clock.
#: The entry transacts on the trigger clock (the retest
#: hold / the rejection read IS an M15 event), so
#: ENTRY == TRIGGER for this family — stated, not
#: assumed downstream.
TRIGGER_TF_DOWN: dict[str, str] = {"H1": "M15"}

#: The inverse ladder (the runner's threading reads it —
#: a scan on M15 is an H1-formed auction read).
FORMATION_TF_UP: dict[str, str] = {
    v: k for k, v in TRIGGER_TF_DOWN.items()}

#: The structural stop sits this far beyond the family's
#: level (the desk's standing buffer scale —
#: playbooks/range_regime.STOP_BUFFER_ATR, the vol and
#: trend families' stops): the invalidation PRICE is the
#: auction fact's level; this is only the execution
#: buffer around it, never a fixed-distance stop.
STOP_BUFFER_ATR = 0.25

#: Acceptance needs at least this many closes holding
#: outside the probed value (the probe close itself
#: counts) — one close outside is a visit, two is the
#: market staying there while value follows.
ACCEPT_MIN_CLOSES = 2

#: Formation bars after the probe close in which the
#: probe must resolve (accept or reject).  A probe that
#: is neither accepted nor rejected inside the window
#: EXPIRES — the auction gave no answer, and no answer
#: is not a hypothesis.
RESOLUTION_WINDOW = 3

#: The acceptance retest: an M15 bar whose range comes
#: within this ATR fraction of the probed edge and whose
#: close holds outside it is the retest that held.
RETEST_TOL_ATR = 0.10

#: Trigger bars the retest may take to arrive after
#: acceptance before the setup is stale (16 M15 bars =
#: the 4 hours after the accepting H1 close).
RETEST_MAX_TRIGGER_BARS = 16

#: Sequence phases (this module's vocabulary, carried on
#: every signal / arm record the family produces).
PHASE_PROBING = "probing"
PHASE_ACCEPTED = "accepted"
PHASE_REJECTED = "rejected"
PHASE_EXPIRED = "expired"
PHASE_INVALIDATED = "invalidated"


def body_dominated(o: float, h: float, l: float, c: float,
                   direction: int) -> bool:
    """The displacement doctrine's candle read
    (DISP_BODY_WICK_MIN): the bar's body in ``direction``
    is at least 1.5x its total wick."""
    body = (c - o) * direction
    if body <= 0:
        return False
    wick = (h - max(o, c)) + (min(o, c) - l)
    return body >= DISP_BODY_WICK_MIN * max(wick, 0.0)


class PreparedAuction:
    """One formation frame's detector series, prepared
    once per scan: the developing value series (the
    shared §6.3 extension) plus the causal Wilder ATR
    (build_tf_state's construction — EWMA alpha=1/14
    with the too-short TR-mean fallback — evaluated per
    bar, so this module's ATR at bar ``j`` equals the
    situation engine's on the frame truncated at
    ``j``)."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.n = len(frame)
        self.o = frame["open"].to_numpy(dtype=float)
        self.h = frame["high"].to_numpy(dtype=float)
        self.l = frame["low"].to_numpy(dtype=float)
        self.c = frame["close"].to_numpy(dtype=float)
        self.values = _dv.DevelopingSeries(frame)
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
                wilder[j] = float(np.mean(tr[:j + 1]))
        self._atr = wilder

    def atr(self, j: int) -> float | None:
        value = self._atr[j] if 0 <= j < self.n else np.nan
        return float(value) if np.isfinite(value) else None

    def value_at(self, j: int) -> dict:
        return self.values.at(j)


def _ts(frame, k):
    try:
        return str(frame.index[k])
    except Exception:
        return None


def auction_sequence_at(prep: PreparedAuction,
                        j: int) -> dict | None:
    """The day's latest auction sequence as of formation
    bar ``j`` (sequences are day-bound: developing value
    resets with the trading day, so a probe's life never
    spans days).  None when no probe has formed by ``j``.

    The walk is over the day's bars up to ``j`` only —
    the sequence state at ``j`` is a function of what
    was observable at ``j`` (each bar's value state is
    the developing state AT that bar, itself causal)."""
    if prep is None or j < 0 or j >= prep.n:
        return None
    frame = prep.frame
    pos = prep.values.day_positions(j)
    seq: dict | None = None      # live (probing) sequence
    done: dict | None = None     # latest resolved sequence
    prev_inside = False
    prev_state: dict | None = None
    for i in pos:
        st = prep.value_at(i)
        if st.get("status") != "ok":
            # Value unmeasurable at this bar: the chain of
            # reads breaks honestly — no probe can form
            # across an unmeasured bar, and a live probe's
            # window keeps running on the clock (its bars
            # still age it toward expiry below).
            if seq is not None and i - seq["probe_bar"] \
                    >= RESOLUTION_WINDOW:
                seq = {**seq, "phase": PHASE_EXPIRED,
                       "resolution_bar": i,
                       "resolution_ts": _ts(frame, i)}
                done, seq = seq, None
            prev_inside, prev_state = False, None
            continue
        c = float(prep.c[i])
        inside = st["val"] <= c <= st["vah"]
        died_here = False
        # --- post-resolution death reads (the arm layer's
        # invalidations; a dead sequence stays dead).
        if done is not None and seq is None \
                and done["phase"] in (PHASE_ACCEPTED,
                                      PHASE_REJECTED):
            d = done["direction"]
            if done["phase"] == PHASE_ACCEPTED:
                edge = done["ref"]["vah"] if d > 0 \
                    else done["ref"]["val"]
                band = _dv.MIGRATION_DEADBAND_BINS * float(
                    st.get("bin_width") or 0.0)
                poc_stalled = (float(st["poc"])
                               - done["accept_poc"]) * d <= band
                if (c - edge) * d <= 0 and poc_stalled:
                    done = {**done, "phase": PHASE_INVALIDATED,
                            "death_bar": i,
                            "death_ts": _ts(frame, i)}
                    died_here = True
            else:  # rejected: a close beyond the probe
                # extreme = the auction re-accepted the
                # probe's ground after all.
                if (c - done["extreme_price"]) * d > 0:
                    done = {**done, "phase": PHASE_INVALIDATED,
                            "death_bar": i,
                            "death_ts": _ts(frame, i)}
                    died_here = True
        if seq is None:
            # A probe starts when a close leaves the value
            # that existed BEFORE this bar (the previous
            # bar's developing state — the pre-probe value
            # the sequence is judged against for its whole
            # life), having been inside it at the previous
            # close.  The current bar's own state never
            # judges its own probe: its volume is already
            # in that profile.  A bar that just killed the
            # previous sequence starts no new probe: the
            # thrust beyond the dead probe's extreme IS
            # the re-acceptance event, not a fresh probe
            # with a fresh reference value.
            if prev_inside and prev_state is not None \
                    and not died_here:
                d = 1 if c > prev_state["vah"] else \
                    -1 if c < prev_state["val"] else 0
                if d:
                    seq = {
                        "phase": PHASE_PROBING,
                        "direction": d,
                        "probe_bar": i,
                        "probe_ts": _ts(frame, i),
                        "extreme_price": float(
                            prep.h[i] if d > 0 else prep.l[i]),
                        "extreme_bar": i,
                        "ref": {
                            "poc": prev_state["poc"],
                            "vah": prev_state["vah"],
                            "val": prev_state["val"],
                            "day_key": prev_state["day_key"],
                            "bars": prev_state["bars"]},
                        "outside_closes": 1,
                        "resolution_bar": None,
                        "resolution_ts": None,
                        "accept_edge": None,
                        "accept_poc": None,
                        "death_bar": None,
                        "death_ts": None,
                        "identity": "|".join([
                            "H1", str(_ts(frame, i)),
                            f"{d:+d}"]),
                    }
                    done = None
            prev_inside, prev_state = inside, st
            continue
        # --- a live probe: extreme, then resolution.
        d = seq["direction"]
        ext = float(prep.h[i] if d > 0 else prep.l[i])
        if (ext - seq["extreme_price"]) * d > 0:
            seq = {**seq, "extreme_price": ext,
                   "extreme_bar": i}
        ref = seq["ref"]
        ref_edge = ref["vah"] if d > 0 else ref["val"]
        if (c - ref_edge) * d <= 0:
            # A close back inside the probed value before
            # acceptance: the probe was rejected.
            seq = {**seq, "phase": PHASE_REJECTED,
                   "resolution_bar": i,
                   "resolution_ts": _ts(frame, i)}
            done, seq = seq, None
            prev_inside, prev_state = inside, st
            continue
        seq = {**seq,
               "outside_closes": seq["outside_closes"] + 1}
        if seq["outside_closes"] >= ACCEPT_MIN_CLOSES \
                and _dv.migrated_toward(st, ref, d):
            seq = {**seq, "phase": PHASE_ACCEPTED,
                   "resolution_bar": i,
                   "resolution_ts": _ts(frame, i),
                   "accept_edge": float(ref_edge),
                   "accept_poc": float(st["poc"])}
            done, seq = seq, None
            prev_inside, prev_state = inside, st
            continue
        if i - seq["probe_bar"] >= RESOLUTION_WINDOW:
            seq = {**seq, "phase": PHASE_EXPIRED,
                   "resolution_bar": i,
                   "resolution_ts": _ts(frame, i)}
            done, seq = seq, None
            prev_inside, prev_state = inside, st
            continue
        prev_inside, prev_state = inside, st
    out = seq if seq is not None else done
    if out is None:
        return None
    return {**out, "value": prep.value_at(j)}


def lvn_crossing_at(prep: PreparedAuction, j: int,
                    prev_close: float,
                    close: float) -> dict | None:
    """A traversal entry read: the trigger close
    ``close`` (previous trigger close ``prev_close``)
    has just entered a low-volume node's span FROM one
    side, on the formation value state at bar ``j``.

    Returns {"direction", "lvn", "value"} — direction is
    the direction of travel (entered from below, heading
    for the above shelf = +1; from above toward the below
    shelf = -1).  A close that jumps the whole span is
    not an entry (the traversal already happened — the
    read never chases it), and a close that was already
    inside the span is not a crossing."""
    st = prep.value_at(j)
    if st.get("status") != "ok":
        return None
    for lvn in st.get("lvns") or []:
        if prev_close < lvn["low"] \
                and lvn["low"] <= close <= lvn["high"]:
            return {"direction": 1, "lvn": lvn, "value": st}
        if prev_close > lvn["high"] \
                and lvn["low"] <= close <= lvn["high"]:
            return {"direction": -1, "lvn": lvn, "value": st}
    return None
