"""Mean reversion — the family's measurement core
(MarketSituation Pass 7d, SPEC §6.4).

The behavior: EXHAUSTION of an extension away from
value.  Price has travelled a statistically unusual
distance beyond the developing value area, and the
travel is dying — value has stopped migrating after
it, momentum at the extreme is decaying or absorbed,
and closes have stopped making progress beyond the
extreme.  The fade back toward value is the trade:
POC first, the opposite value edge second (the
rotation is the thesis, staged).

    close beyond the anchored value edge by
    >= EXTENSION_MIN_ATR ──▶ EXTENDED (armed)
        │
        ├── no close progress for >= STALL_MIN_BARS
        │   + momentum no longer sustained
        │   + value NOT migrated toward price since
        │   the extreme ──▶ STALLED (the fade's ground:
        │   entry on the trigger clock's structure
        │   turn back toward value)
        ├── formation close through the extension's
        │   DEFENDED SWING ──▶ BROKE (the failed-
        │   continuation ground: the trend claim died
        │   on its own timeframe, at extension)
        ├── (from STALLED / BROKE) a formation close
        │   beyond the extreme close ──▶ INVALIDATED
        │   (the auction accepted beyond the extreme
        │   after all, or the stall was a pause: the
        │   fade's premise is dead — this is 7c's
        │   acceptance fact arriving, and the two
        │   families never fight over it)
        ├── stall never confirms inside the window
        │   ──▶ EXPIRED
        └── a close back inside the reference value
            ──▶ CLOSED (the rotation completed, or the
            extension unwound quietly — either way the
            episode is over)

Measurement discipline: the extension is judged
against the ANCHORED value — the developing value
state at the last bar whose close sat inside it,
fixed for the episode's life (a value area that
chases price bar by bar can never be extended from;
7c's pre-probe reference rule, generalised).  The
stall's migration leg reads the LIVE developing
states (extreme bar vs current bar, ``migrated_
toward``) — value following price is the auction
accepting the extension, and an accepted extension
is not exhaustion.  Momentum is the trend family's
own read (``live/trend_momentum.PreparedFrame.
momentum_at`` — sustained / decaying / absorbed on
the formation frame), never a private re-derivation;
value is ``live/developing_value``'s shared state.
Value here is a TICK-VOLUME profile (MT5's volume
column); the basis rides every state as
``volume_basis`` and is named, never dressed.

Distinctness (SPEC §6 discipline): 7c trades the
auction's answer to a PROBE at the value edge
(acceptance continues, rejection rotates from the
edge).  This family trades the FAR side of the move:
an extension that has already travelled multiples
of ATR beyond value and is exhausting — different
object (the extension episode, not the probe),
different entry (the stall + trigger turn, not the
retest / first read), different death (acceptance
beyond the extreme kills it — 7c's continuation
fact is this family's invalidation).  ``range_mean_
reversion`` fades a range edge re-entry inside a
RangeDetector range in a D1-flat regime; this
family reads no range object — its levels are
developing value's own and its gate is the
situation's structure phase.  ``trend_exhaustion_
fade`` is NOT in the trend family (SPEC §6.1 names
it here): exhaustion is this family's one subject.

The REGIME GATE is the family's honesty, and it
lives at the playbook layer (it reads the
situation): this family does not fade a clean
trend.  The core's stall is necessary, never
sufficient.

Every read is causal: the episode at formation bar
``j`` is a function of bars ``<= j`` (each bar's
value state is itself causal; swings are confirmed
swings, known only from their confirmation bar).
Episodes are day-bound — developing value resets
with the trading day, so an extension's life never
spans days.  Judged by what was observable at the
time, never hindsight.
"""
from __future__ import annotations

import pandas as pd

from . import developing_value as _dv
from . import trend_momentum as _tm
from .trend_momentum import TF_DELTAS, first_bar_after, last_formation_bar

__all__ = [
    "EXTENSION_MAX_BARS",
    "EXTENSION_MIN_ATR",
    "FORMATION_TFS",
    "FORMATION_TF_UP",
    "PHASE_BROKE",
    "PHASE_CLOSED",
    "PHASE_EXPIRED",
    "PHASE_EXTENDED",
    "PHASE_INVALIDATED",
    "PHASE_STALLED",
    "POST_EXHAUSTION_MIN_BARS",
    "STALL_MIN_BARS",
    "STOP_BUFFER_ATR",
    "TF_DELTAS",
    "TRIGGER_TFS",
    "TURN_WINDOW_BARS",
    "PreparedReversion",
    "first_bar_after",
    "last_formation_bar",
    "reversion_episode_at",
]

#: The family's formation clock (SPEC §6.4): H1 — the
#: developing value area's own clock (the day's bars;
#: SPEC §10.2 risk 3: the developing profile must not
#: creep to faster clocks).
FORMATION_TFS: tuple[str, ...] = ("H1",)

#: Trigger clocks (SPEC §6.4: trigger M15 / M5): the
#: fade's turn is read one or two steps down from the
#: formation clock.  The trigger clock decides the
#: instance's horizon (M15 intraday / M5 scalp).
TRIGGER_TFS: tuple[str, ...] = ("M15", "M5")

#: The inverse ladder (the runner's threading reads
#: it — a scan on M15 or M5 is an H1-formed reversion
#: read).
FORMATION_TF_UP: dict[str, str] = {
    tf: "H1" for tf in TRIGGER_TFS}

#: An extension is a formation CLOSE beyond the
#: anchored value edge by at least this many formation
#: ATRs.  Inside ~1 ATR beyond the edge is probe /
#: acceptance territory (7c's ground); two ATRs beyond
#: the edge is a statistically unusual excursion —
#: the SPEC §6.4 "stated ATR multiple".
EXTENSION_MIN_ATR = 2.0

#: Formation bars after the extreme bar with no close
#: progress before a stall may confirm — a FLOOR only:
#: the binding stall evidence is the momentum read
#: leaving "sustained" and value not following (the
#: momentum window needs bars to roll the extension
#: leg out of itself; one quiet bar is a pause, not
#: exhaustion).
STALL_MIN_BARS = 2

#: Bars without close progress the stall must show
#: (with momentum ABSORBED) before the regime gate
#: will name a clean-trend extension post-exhausted
#: and let the fade pass (the playbook layer's
#: override rule — a longer, deader stall than the
#: rotational-state floor).
POST_EXHAUSTION_MIN_BARS = 4

#: Formation bars from the extension's first close in
#: which the stall must confirm.  An extension that
#: neither stalls nor returns inside value inside the
#: window EXPIRES — the move may still be travelling,
#: and a travelling move is not this family's trade.
EXTENSION_MAX_BARS = 12

#: The structural stop sits this far beyond the
#: extension extreme (the desk's standing buffer
#: scale — the vol / trend / auction families' stops):
#: the invalidation PRICE is the extreme itself; this
#: is only the execution buffer around it.
STOP_BUFFER_ATR = 0.25

#: Trigger bars the fade's turn may take to arrive
#: after the stall confirms, per trigger clock (16
#: M15 bars = the 4 hours after the stall close; 24
#: M5 bars = 2 hours — the scalp turn comes faster or
#: the scalp thesis is stale).
TURN_WINDOW_BARS: dict[str, int] = {"M15": 16, "M5": 24}

#: Episode phases (this module's vocabulary, carried
#: on every signal / arm record the family produces).
PHASE_EXTENDED = "extended"
PHASE_STALLED = "stalled"
PHASE_BROKE = "broke"
PHASE_INVALIDATED = "invalidated"
PHASE_EXPIRED = "expired"
PHASE_CLOSED = "closed"


class PreparedReversion:
    """One formation frame's detector series, prepared
    once per scan: the trend core's prepared frame
    (causal Wilder ATR, confirmed swings, the momentum
    quality read — the situation engine's own formulas)
    plus the developing value series (the shared §6.3
    extension).  One preparation, both organs, so a
    read here can never disagree with the situation,
    the trend family or the auction family on identical
    bars."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.n = len(frame)
        self.f = _tm.PreparedFrame(frame)
        self.values = _dv.DevelopingSeries(frame)

    def atr(self, j: int) -> float | None:
        return self.f.atr(j)

    def value_at(self, j: int) -> dict:
        return self.values.at(j)


def _ts(frame, k):
    try:
        return str(frame.index[k])
    except Exception:
        return None


def _ref_block(state: dict) -> dict:
    return {"poc": state["poc"], "vah": state["vah"],
            "val": state["val"],
            "day_key": state.get("day_key"),
            "bars": state.get("bars")}


def _defended_swing(prep: PreparedReversion, j: int,
                    direction: int,
                    extreme_close: float):
    """The continuation's defended swing as of bar
    ``j``: the most recent CONFIRMED swing on the
    counter side known at ``j`` (n=3, the situation's
    geometry) whose price sits on the value side of
    the extreme close — for an up-extension, the last
    confirmed swing low below the extreme close (the
    low the up-move's structure defends; a formation
    close through it is the structure break).  None
    when no such swing is confirmed yet — the failed-
    continuation hypothesis then has no break level
    for this episode, stated, never invented."""
    known = prep.f.swings_known(j, highs=direction < 0)
    best = None
    for pos, price in known:
        if (extreme_close - price) * direction <= 0:
            continue
        if best is None or pos > best[0]:
            best = (pos, price)
    if best is None:
        return None, None
    return float(best[1]), int(best[0])


def reversion_episode_at(prep: PreparedReversion,
                         j: int) -> dict | None:
    """The day's latest extension episode as of
    formation bar ``j`` — live (extended / stalled /
    broke) or the latest terminal one (invalidated /
    expired / closed).  None when no extension has
    formed by ``j``.  The walk is over the day's bars
    up to ``j`` only, each read a function of what was
    observable at that bar."""
    if prep is None or j < 0 or j >= prep.n:
        return None
    frame = prep.frame
    pos = prep.values.day_positions(j)
    anchor: dict | None = None   # value at last inside close
    ep: dict | None = None       # live episode
    done: dict | None = None     # latest terminal episode
    for i in pos:
        st = prep.value_at(i)
        if st.get("status") != "ok":
            # Value unmeasurable at this bar: the chain
            # of reads breaks honestly — a live episode's
            # window keeps running on the clock (its bars
            # still age it toward expiry below).
            if ep is not None \
                    and ep["phase"] == PHASE_EXTENDED \
                    and i - ep["first_bar"] \
                    >= EXTENSION_MAX_BARS:
                ep = {**ep, "phase": PHASE_EXPIRED,
                      "resolution_bar": i,
                      "resolution_ts": _ts(frame, i)}
                done, ep = ep, None
            continue
        c = float(prep.f.c[i])
        if ep is None:
            # --- start: a close beyond the ANCHORED
            # value edge by the stated ATR multiple.
            atr = prep.atr(i)
            d = 0
            if anchor is not None and atr and atr > 0:
                if (c - float(anchor["vah"])) / atr \
                        >= EXTENSION_MIN_ATR:
                    d = 1
                elif (float(anchor["val"]) - c) / atr \
                        >= EXTENSION_MIN_ATR:
                    d = -1
            if d:
                defended, defended_pos = _defended_swing(
                    prep, i, d, c)
                ep = {
                    "phase": PHASE_EXTENDED,
                    "direction": d,
                    "first_bar": i,
                    "first_ts": _ts(frame, i),
                    "ref": _ref_block(anchor),
                    "extension_atr_first": float(
                        (c - (anchor["vah"] if d > 0
                              else anchor["val"])) * d / atr),
                    "extreme_close": c,
                    "extreme_price": float(
                        prep.f.h[i] if d > 0 else prep.f.l[i]),
                    "extreme_bar": i,
                    "extreme_ts": _ts(frame, i),
                    "extreme_state": dict(st),
                    "defended_swing": defended,
                    "defended_pos": defended_pos,
                    "stall_bar": None,
                    "stall_ts": None,
                    "momentum_at_stall": None,
                    "stall_state": None,
                    "break_bar": None,
                    "break_ts": None,
                    "death_bar": None,
                    "death_ts": None,
                    "death_kind": None,
                    "resolution_bar": None,
                    "resolution_ts": None,
                    "identity": "|".join([
                        "H1", str(_ts(frame, i)),
                        f"{d:+d}"]),
                }
                done = None
        else:
            d = ep["direction"]
            ref_edge = ep["ref"]["vah"] if d > 0 \
                else ep["ref"]["val"]
            # --- a close back inside the reference
            # value ends the episode in every phase:
            # the rotation completed (or the extension
            # unwound quietly into value).
            if (c - ref_edge) * d <= 0:
                ep = {**ep, "phase": PHASE_CLOSED,
                      "resolution_bar": i,
                      "resolution_ts": _ts(frame, i)}
                done, ep = ep, None
            elif ep["phase"] == PHASE_EXTENDED \
                    and i - ep["first_bar"] \
                    >= EXTENSION_MAX_BARS:
                # The extension outlived its window
                # without stalling — expired, whether or
                # not this bar made a new extreme: an
                # extension that keeps extending for the
                # full window IS a trend repricing, and
                # the exhaustion premise is dead on the
                # clock, not only on a stall that never
                # came.  (Checked before the extreme
                # update so a still-climbing extension
                # cannot renew itself forever.)
                ep = {**ep, "phase": PHASE_EXPIRED,
                      "resolution_bar": i,
                      "resolution_ts": _ts(frame, i)}
                done, ep = ep, None
            else:
                new_extreme = \
                    (c - ep["extreme_close"]) * d > 0
                if ep["phase"] in (PHASE_STALLED,
                                   PHASE_BROKE) \
                        and new_extreme:
                    # The stall was a pause / the break
                    # failed: price accepted (or ground
                    # back) beyond the extreme.  Death
                    # kind names which fact arrived —
                    # 7c-style acceptance (value followed
                    # beyond the extreme) vs a bare
                    # resumption / re-acceptance close.
                    if ep["phase"] == PHASE_STALLED and \
                            _dv.migrated_toward(
                                st, ep["extreme_state"], d):
                        kind = "acceptance"
                    elif ep["phase"] == PHASE_BROKE:
                        kind = "re_acceptance"
                    else:
                        kind = "resumption"
                    ep = {**ep, "phase": PHASE_INVALIDATED,
                          "death_bar": i,
                          "death_ts": _ts(frame, i),
                          "death_kind": kind}
                    done, ep = ep, None
                elif new_extreme:
                    # Still extending: the extreme (and
                    # its defended swing, and the stall's
                    # migration baseline) move with it.
                    defended, defended_pos = \
                        _defended_swing(prep, i, d, c)
                    ep = {**ep,
                          "extreme_close": c,
                          "extreme_price": float(
                              prep.f.h[i] if d > 0
                              else prep.f.l[i]),
                          "extreme_bar": i,
                          "extreme_ts": _ts(frame, i),
                          "extreme_state": dict(st),
                          "defended_swing": defended,
                          "defended_pos": defended_pos}
                elif ep["phase"] in (PHASE_EXTENDED,
                                     PHASE_STALLED) \
                        and ep["defended_swing"] is not None \
                        and (c - ep["defended_swing"]) * d < 0:
                    # The continuation's structure died on
                    # its own timeframe, at extension:
                    # the failed-continuation ground.
                    ep = {**ep, "phase": PHASE_BROKE,
                          "break_bar": i,
                          "break_ts": _ts(frame, i)}
                elif ep["phase"] == PHASE_EXTENDED:
                    # Expiry was adjudicated above (the
                    # outer window check); what remains for
                    # a live extended episode is the stall.
                    mom = prep.f.momentum_at(i, d)
                    if i - ep["extreme_bar"] \
                            >= STALL_MIN_BARS \
                            and mom["state"] \
                            != _tm.MOMENTUM_SUSTAINED \
                            and not _dv.migrated_toward(
                                st, ep["extreme_state"],
                                d):
                        ep = {**ep,
                              "phase": PHASE_STALLED,
                              "stall_bar": i,
                              "stall_ts": _ts(frame, i),
                              "momentum_at_stall": mom,
                              "stall_state": dict(st)}
        # The anchor follows the last close that sat
        # inside the CURRENT developing value — but only
        # while no episode is live (an episode's
        # reference is fixed for its life).
        if ep is None and st["val"] <= c <= st["vah"]:
            anchor = st
    out = ep if ep is not None else done
    if out is None:
        return None
    atr = prep.atr(j)
    d = out["direction"]
    ref_edge = out["ref"]["vah"] if d > 0 \
        else out["ref"]["val"]
    extension_atr = None
    if atr and atr > 0:
        extension_atr = float(
            (float(prep.f.c[j]) - ref_edge) * d / atr)
    return {**out, "value": prep.value_at(j),
            "extension_atr": extension_atr}
