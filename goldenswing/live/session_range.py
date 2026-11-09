"""Session / opening range — the family's measurement
core (MarketSituation Pass 7e, SPEC §6.5).

The behavior: the day has a SHAPE, and the shape is
set in its opening.  The trading day's first hour —
the Initial Balance, the day object's own anchor
(``live/day_context.py``: the first H1 bar of the NY
trading day) — is the day's opening range.  Once the
second bar of the day prints, that range is FROZEN:
nothing that happens later can widen it, and every
read in this module is taken against the frozen range,
causally, on the day's own bars.

    IB frozen (the day's second bar exists)
        │
        ├── formation CLOSE beyond an edge ──▶ PROBING
        │       ├── >= 2 closes beyond the edge AND
        │       │   follow-through (a displacement
        │       │   close among them, or closes
        │       │   progressing beyond the probe
        │       │   close) ──▶ ACCEPTED (the breakout
        │       │   is the market staying, not visiting)
        │       ├── a close back inside before
        │       │   acceptance ──▶ REJECTED (the
        │       │   failed auction at the day's own
        │       │   scale: rotation to the IB mid,
        │       │   then the opposite edge)
        │       └── neither inside the window
        │           ──▶ EXPIRED (no answer is not a
        │           hypothesis)
        ├── London's first bars probe beyond the
        │   ASIA range and close back inside ──▶
        │   SWEPT (the day's first value was raided,
        │   not repriced: rotation to the Asia mid,
        │   then the opposite Asia edge)
        └── the day OPENED outside prior value (a
            drive premise) and London sweeps the
            OVERNIGHT (Asia) extreme and rejects ──▶
            DRIVE REJECTED (fade toward the IB mid,
            then the prior value edge the drive left)

    Deaths, read after resolution on the same clock:
    an accepted breakout whose closes return inside
    the range has FAILED (death_kind
    acceptance_failed); a rejected probe the auction
    later accepts beyond the probe extreme was not a
    failure (re_acceptance); a sweep or a rejected
    drive whose extreme is later accepted by a close
    was a probe, not a rejection (acceptance).

The session facts are the engine's own, reused, never
re-derived: the trading day and its IB from
``live/day_context.py`` (NY day, first H1 bar), the
session partition and killzones from
``reader/session_timeframes.session_of`` (DST-aware —
the same function the situation's activity block
consults, so the family and the situation can never
disagree about which session a bar belongs to), the
prior day's value area from ``live/volume_profile.py``
(the day object's own construction — a TICK-VOLUME
profile, named as such wherever it rides).

TIME HONESTY is part of the measurement, not a
wrapper: entries exist only while the day is in its
tradable sessions (London / overlap / New York — the
``ENTRY_SESSIONS`` below); the day's Asia hours build
the ranges and the overnight extremes, and the day's
late off-hours close the book on it.  A breakout
accepted after the cutoff is still a FACT (the arm
stream journals it) — it is simply not this family's
trade, and the family's playbooks refuse it by the
entry bar's own session label.  A day whose opening
bar is flat (zero-width IB) or unmeasurable forms NO
range: the read says ``no_range`` with its reason, and
the arm stream journals the abstention — Pass 9
grades what this family refused as well as what it
took.

Distinctness (SPEC §6 discipline + §6.5's own
boundary): ``killzone_open`` trades a PRE-SESSION
range break at the London / NY opens on the H1 clock
— displacement, then entry on the RETEST of the
broken edge, 1.5R target.  This family trades the
DAY's frozen IB (and the Asia range the day built),
formed on H1 and entered on M15: the breakout enters
on ACCEPTANCE (closes holding beyond + follow-
through — no retest is waited for), the failure is a
rotation killzone_open does not have, and the sweep
trades the REJECTION of a London raid where
killzone_open trades the break's continuation.  7c's
auction family reads probes against DEVELOPING VALUE
(the day's migrating profile); this family's
references are the session's own frozen objects (IB,
Asia range, prior value), which never migrate.

Every read is causal: the day read at formation bar
``j`` is a function of bars ``<= j`` — the frozen
range is computed from the day's first bar only, the
Asia range from Asia bars already closed, acceptance
from closes already printed.  Judged by what was
observable at the time, never hindsight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..reader.session_timeframes import session_of
from . import day_context as _dc
from . import trend_momentum as _tm
from .trend_momentum import TF_DELTAS, first_bar_after, last_formation_bar
from .volume_profile import volume_profile

__all__ = [
    "ACCEPT_MIN_CLOSES",
    "DRIVE_LONDON_BARS",
    "ENTRY_SESSIONS",
    "FORMATION_TFS",
    "FORMATION_TF_UP",
    "PHASE_ACCEPTED",
    "PHASE_EXPIRED",
    "PHASE_INVALIDATED",
    "PHASE_PROBING",
    "PHASE_REJECTED",
    "RESOLUTION_WINDOW",
    "STOP_BUFFER_ATR",
    "SWEEP_LONDON_BARS",
    "TF_DELTAS",
    "TRIGGER_TF_DOWN",
    "PreparedSession",
    "first_bar_after",
    "last_formation_bar",
    "session_day_at",
]

#: The family's clocks (SPEC §6.5): formation on the
#: session clock (H1 — the day object's own bars; the
#: IB is an H1 object by the day object's definition),
#: trigger M15.
FORMATION_TFS: tuple[str, ...] = ("H1",)

#: Trigger clock one step down from the formation
#: clock.  The entry transacts on the trigger clock
#: (the first read after the formation fact IS an M15
#: event), so ENTRY == TRIGGER for this family —
#: stated, not assumed downstream.
TRIGGER_TF_DOWN: dict[str, str] = {"H1": "M15"}

#: The inverse ladder (the runner's threading reads
#: it — a scan on M15 is an H1-formed session read).
FORMATION_TF_UP: dict[str, str] = {
    v: k for k, v in TRIGGER_TF_DOWN.items()}

#: The sessions an entry may print in: the day's
#: tradable body (London open through the New York
#: close), by ``session_of``'s DST-aware partition —
#: the same labels the situation's activity block
#: carries.  Asia builds the ranges; the off hours
#: are the day winding down.  An entry read outside
#: these sessions is refused by the playbooks and the
#: refusal is the declared cutoff (see the module
#: docstring's time-honesty paragraph).
ENTRY_SESSIONS: tuple[str, ...] = ("london", "overlap",
                                   "newyork")

#: Acceptance needs at least this many closes holding
#: beyond the probed IB edge (the probe close itself
#: counts) — one close beyond is a visit (the first
#: poke); two holding, with follow-through, is the
#: market staying there.
ACCEPT_MIN_CLOSES = 2

#: Formation bars after the probe close in which the
#: probe must resolve (accept or reject).  A probe
#: that is neither accepted nor rejected inside the
#: window EXPIRES — the auction at the range edge
#: gave no answer in the session's first hours, and
#: no answer is not a hypothesis.
RESOLUTION_WINDOW = 6

#: The London sweep is a London-OPEN event: the sweep
#: bar must be among the first this many London-
#: segment bars of the day.  A probe beyond the Asia
#: edge deeper into London is a breakout test (the IB
#: machine's ground, or nothing), not the raid this
#: family fades.
SWEEP_LONDON_BARS = 3

#: The opening drive's rejection is likewise an
#: opening event: the sweep of the overnight extreme
#: must print within the first this many London-
#: segment bars of the day.
DRIVE_LONDON_BARS = 2

#: The structural stop sits this far beyond the
#: family's level (the desk's standing buffer scale —
#: the vol / trend / auction / reversion families'
#: stops): the invalidation PRICE is the session
#: fact's level; this is only the execution buffer
#: around it, never a fixed-distance stop.
STOP_BUFFER_ATR = 0.25

#: Sequence phases (this module's vocabulary, carried
#: on every signal / arm record the family produces).
PHASE_PROBING = "probing"
PHASE_ACCEPTED = "accepted"
PHASE_REJECTED = "rejected"
PHASE_EXPIRED = "expired"
PHASE_INVALIDATED = "invalidated"

#: Sweep / drive phases (their own short vocabularies:
#: a sweep either stands rejected or was accepted
#: after all).
PHASE_SWEPT = "swept"
PHASE_DRIVE_REJECTED = "drive_rejected"


class PreparedSession:
    """One formation frame's detector series, prepared
    once per scan: the trend core's prepared frame
    (causal Wilder ATR, confirmed swings, the momentum
    quality read — the situation engine's own formulas)
    plus the day partition (``day_context``'s NY
    trading-day keys) and the session partition
    (``session_of``'s DST-aware labels), so a read here
    can never disagree with the situation, the day
    object or the other families on identical bars."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.n = len(frame)
        self.f = _tm.PreparedFrame(frame)
        self.o = self.f.o
        self.h = self.f.h
        self.l = self.f.l
        self.c = self.f.c
        self._keys = [_dc.trading_day_key(ts)
                      for ts in frame.index]
        self._key_strs = [str(k.date()) for k in self._keys]
        self._sessions = [session_of(ts)[0]
                          for ts in frame.index]
        days: dict[str, list[int]] = {}
        for i, ks in enumerate(self._key_strs):
            days.setdefault(ks, []).append(i)
        self._days = days
        self._prior_cache: dict[str, dict | None] = {}

    def atr(self, j: int) -> float | None:
        return self.f.atr(j)

    def day_key(self, j: int) -> str:
        return self._key_strs[j]

    def day_positions(self, j: int) -> list[int]:
        """All frame positions of bar ``j``'s trading
        day, ascending (the day object's partition)."""
        return list(self._days.get(self._key_strs[j], []))

    def session_at(self, j: int) -> str:
        return self._sessions[j]

    def prior_value(self, day_first_pos: int) -> dict | None:
        """The PRIOR trading day's value area (the day
        object's own construction: ``volume_profile``
        over the prior day's H1 bars, 24 bins / 70% —
        a tick-volume profile, basis named on the
        block).  None when the frame carries no prior
        day or its profile is unmeasurable — absent
        stays absent, and the drive premise (which
        needs it) simply does not exist that day."""
        key = self._key_strs[day_first_pos]
        if key in self._prior_cache:
            return self._prior_cache[key]
        out: dict | None = None
        keys = sorted(self._days.keys())
        idx = keys.index(key) if key in keys else -1
        if idx > 0:
            prior_pos = self._days[keys[idx - 1]]
            if len(prior_pos) >= 5:
                sub = self.frame.iloc[prior_pos]
                try:
                    vp = volume_profile(sub, bins=24)
                except Exception:  # noqa: BLE001 — unmeasurable
                    vp = {}
                if vp.get("poc") is not None \
                        and vp.get("vah") is not None \
                        and vp.get("val") is not None:
                    out = {"poc": float(vp["poc"]),
                           "vah": float(vp["vah"]),
                           "val": float(vp["val"]),
                           "day_key": keys[idx - 1],
                           "bars": len(prior_pos),
                           "volume_basis": "tick_volume"}
        self._prior_cache[key] = out
        return out


def _ts(frame, k):
    try:
        return str(frame.index[k])
    except Exception:  # noqa: BLE001 — unreadable index
        return None


def _finite(*vals) -> bool:
    return all(v is not None and np.isfinite(v)
               for v in vals)


def session_day_at(prep: PreparedSession,
                   j: int) -> dict | None:
    """The trading day's session read as of formation
    bar ``j``: the frozen opening range (IB), the Asia
    range, the prior-value position of the open, and
    the three sequence machines' latest states (the IB
    breakout probe, the London Asia sweep, the opening
    drive's rejection) — each live or resolved, each a
    function of bars ``<= j`` only.  None when bar
    ``j``'s day has not printed its first bar (never
    in practice: ``j`` is in its own day)."""
    if prep is None or j < 0 or j >= prep.n:
        return None
    frame = prep.frame
    pos_all = prep.day_positions(j)
    pos = [i for i in pos_all if i <= j]
    if not pos:
        return None
    first = pos[0]
    day_open = float(prep.o[first])
    prior = prep.prior_value(first)
    opened_outside: int | None = None
    if prior is not None:
        if day_open > prior["vah"]:
            opened_outside = 1
        elif day_open < prior["val"]:
            opened_outside = -1
        else:
            opened_outside = 0
    # --- the opening range: the day's first bar, frozen
    # the moment the day's second bar exists (the day
    # object's IB rule).  A flat or unmeasurable first
    # bar forms NO range — the day is honestly rangeless
    # for this family, stated with its reason.
    ib: dict | None = None
    no_range: dict | None = None
    h0, l0 = float(prep.h[first]), float(prep.l[first])
    if not _finite(h0, l0, day_open):
        no_range = {"reason": "unmeasurable_open",
                    "detail": "the day's first bar carries "
                              "non-finite prices — no "
                              "opening range can be read"}
    elif h0 - l0 <= 0:
        no_range = {"reason": "flat_open",
                    "detail": "the day's first bar is a "
                              "single price (zero-width "
                              "Initial Balance) — a holiday "
                              "/ flat open forms no range"}
    elif len(pos) < 2:
        no_range = {"reason": "window_open",
                    "detail": "the opening window has not "
                              "closed yet — the range is "
                              "not frozen"}
    else:
        ib = {"high": h0, "low": l0, "width": h0 - l0,
              "mid": (h0 + l0) / 2.0, "bar": first,
              "ts": _ts(frame, first)}
    # --- the Asia range (the day's first value, and the
    # day's overnight extremes in the situation's own
    # naming): Asia-segment bars of the day up to ``j``.
    asia: dict | None = None
    for i in pos:
        if prep.session_at(i) != "asia":
            continue
        hh, ll = float(prep.h[i]), float(prep.l[i])
        if not _finite(hh, ll):
            continue
        if asia is None:
            asia = {"high": hh, "low": ll, "bars": 1,
                    "first_bar": i, "last_bar": i}
        else:
            asia = {**asia,
                    "high": max(asia["high"], hh),
                    "low": min(asia["low"], ll),
                    "bars": asia["bars"] + 1,
                    "last_bar": i}
    if asia is not None:
        asia = {**asia, "mid": (asia["high"] + asia["low"])
                / 2.0,
                "width": asia["high"] - asia["low"]}
    out: dict = {
        "day_key": prep.day_key(j),
        "day_open": day_open,
        "open_ts": _ts(frame, first),
        "ib": ib,
        "no_range": no_range,
        "asia": asia,
        "prior_value": prior,
        "opened_outside_value": opened_outside,
        "breakout": None,
        "sweep": None,
        "drive": None,
        "london_seen": any(
            prep.session_at(i) == "london" for i in pos),
        "last_bar": j,
        "last_ts": _ts(frame, j),
        "last_session": prep.session_at(j),
    }
    if ib is None:
        return out
    brk: dict | None = None     # live IB probe
    brk_done: dict | None = None  # resolved IB sequence
    sweep: dict | None = None
    sweep_done: dict | None = None
    drive: dict | None = None
    drive_done: dict | None = None
    edge_closed: dict[int, bool] = {1: False, -1: False}
    london_count = 0
    for i in pos[1:]:
        c = float(prep.c[i])
        sess = prep.session_at(i)
        if sess == "london":
            london_count += 1
        # ---------- the IB breakout machine ----------
        if brk_done is not None and brk is None:
            d = brk_done["direction"]
            edge = ib["high"] if d > 0 else ib["low"]
            if brk_done["phase"] == PHASE_ACCEPTED:
                if (c - edge) * d <= 0:
                    brk_done = {
                        **brk_done, "phase": PHASE_INVALIDATED,
                        "death_bar": i, "death_ts": _ts(frame, i),
                        "death_kind": "acceptance_failed"}
            elif brk_done["phase"] == PHASE_REJECTED:
                if (c - brk_done["extreme_price"]) * d > 0:
                    brk_done = {
                        **brk_done, "phase": PHASE_INVALIDATED,
                        "death_bar": i, "death_ts": _ts(frame, i),
                        "death_kind": "re_acceptance"}
        if brk is None and (brk_done is None
                            or brk_done["phase"]
                            == PHASE_REJECTED):
            # A fresh probe: the day's first, or — after
            # a REJECTED probe — the opposite side's
            # answer (the traverse-then-accept shape:
            # the failed auction rotated through the
            # range and the far edge is now being
            # tested).  After an ACCEPTED probe (standing
            # or failed) or an EXPIRED one the day has
            # told its story at the range and this
            # machine asks no second question.  A rejected
            # sequence's death watch (above) pauses while
            # the second probe is live — its position-
            # level invalidation is the stop at the probe
            # extreme, carried on the failure signal's
            # own terms.
            d = 1 if c > ib["high"] else \
                -1 if c < ib["low"] else 0
            if brk_done is not None \
                    and d != -brk_done["direction"]:
                d = 0
            if d:
                brk = {
                    "phase": PHASE_PROBING,
                    "direction": d,
                    "probe_bar": i,
                    "probe_ts": _ts(frame, i),
                    "extreme_price": float(
                        prep.h[i] if d > 0 else prep.l[i]),
                    "extreme_bar": i,
                    "outside_closes": 1,
                    "first_close": c,
                    "displacement": bool(
                        prep.f.body_dominated(i, d)),
                    "progressed": False,
                    "resolution_bar": None,
                    "resolution_ts": None,
                    "death_bar": None,
                    "death_ts": None,
                    "death_kind": None,
                    "identity": "|".join([
                        "H1", prep.day_key(j),
                        str(_ts(frame, i)), f"{d:+d}",
                        "ib_breakout"]),
                }
        elif brk is not None:
            d = brk["direction"]
            edge = ib["high"] if d > 0 else ib["low"]
            ext = float(prep.h[i] if d > 0 else prep.l[i])
            if (ext - brk["extreme_price"]) * d > 0:
                brk = {**brk, "extreme_price": ext,
                       "extreme_bar": i}
            if (c - edge) * d <= 0:
                brk = {**brk, "phase": PHASE_REJECTED,
                       "resolution_bar": i,
                       "resolution_ts": _ts(frame, i)}
                brk_done, brk = brk, None
            else:
                outside = brk["outside_closes"] + 1
                disp = brk["displacement"] or bool(
                    prep.f.body_dominated(i, d))
                progressed = brk["progressed"] or \
                    (c - brk["first_close"]) * d > 0
                brk = {**brk, "outside_closes": outside,
                       "displacement": disp,
                       "progressed": progressed}
                if outside >= ACCEPT_MIN_CLOSES \
                        and (disp or progressed):
                    brk = {**brk, "phase": PHASE_ACCEPTED,
                           "resolution_bar": i,
                           "resolution_ts": _ts(frame, i)}
                    brk_done, brk = brk, None
                elif i - brk["probe_bar"] >= RESOLUTION_WINDOW:
                    brk = {**brk, "phase": PHASE_EXPIRED,
                           "resolution_bar": i,
                           "resolution_ts": _ts(frame, i)}
                    brk_done, brk = brk, None
        # ---------- the London Asia sweep machine ------
        # Formation is a London-open event; the death
        # watch (a close accepting beyond the sweep
        # extreme) runs on every later bar, any session.
        if sweep is not None:
            d = sweep["direction"]
            if (c - sweep["extreme_price"]) * d > 0:
                sweep = {
                    **sweep, "phase": PHASE_INVALIDATED,
                    "death_bar": i, "death_ts": _ts(frame, i),
                    "death_kind": "acceptance"}
                sweep_done, sweep = sweep, None
        if asia is not None and sess == "london":
            for d in (1, -1):
                a_edge = asia["high"] if d > 0 \
                    else asia["low"]
                if (c - a_edge) * d > 0:
                    edge_closed[d] = True
            if sweep is None and sweep_done is None \
                    and london_count <= SWEEP_LONDON_BARS:
                for d in (1, -1):
                    if edge_closed[d]:
                        continue
                    a_edge = asia["high"] if d > 0 \
                        else asia["low"]
                    ext = float(prep.h[i] if d > 0
                                else prep.l[i])
                    if (ext - a_edge) * d > 0 \
                            and (c - a_edge) * d <= 0:
                        sweep = {
                            "phase": PHASE_SWEPT,
                            "direction": d,
                            "sweep_bar": i,
                            "sweep_ts": _ts(frame, i),
                            "extreme_price": ext,
                            "edge": a_edge,
                            "death_bar": None,
                            "death_ts": None,
                            "death_kind": None,
                            "identity": "|".join([
                                "H1", prep.day_key(j),
                                str(_ts(frame, i)),
                                f"{d:+d}", "asia_sweep"]),
                        }
                        break
        # ---------- the opening drive machine ----------
        # Same shape: the rejection forms at the London
        # open (the drive premise is the day's open
        # outside prior value); its death watch runs on
        # every later bar.
        if drive is not None:
            d = drive["direction"]
            if (c - drive["extreme_price"]) * d > 0:
                drive = {
                    **drive, "phase": PHASE_INVALIDATED,
                    "death_bar": i, "death_ts": _ts(frame, i),
                    "death_kind": "acceptance"}
                drive_done, drive = drive, None
        if opened_outside in (1, -1) and asia is not None \
                and sess == "london" \
                and drive is None and drive_done is None \
                and london_count <= DRIVE_LONDON_BARS:
            d = int(opened_outside)
            a_edge = asia["high"] if d > 0 else asia["low"]
            ext = float(prep.h[i] if d > 0 else prep.l[i])
            if (ext - a_edge) * d > 0 \
                    and (c - a_edge) * d <= 0:
                drive = {
                    "phase": PHASE_DRIVE_REJECTED,
                    "direction": d,
                    "sweep_bar": i,
                    "sweep_ts": _ts(frame, i),
                    "extreme_price": ext,
                    "edge": a_edge,
                    "death_bar": None,
                    "death_ts": None,
                    "death_kind": None,
                    "identity": "|".join([
                        "H1", prep.day_key(j),
                        str(_ts(frame, i)), f"{d:+d}",
                        "drive_reversal"]),
                }
    out["breakout"] = brk if brk is not None else brk_done
    out["sweep"] = sweep if sweep is not None else sweep_done
    out["drive"] = drive if drive is not None else drive_done
    # --- the cutoff read: the day's entry window is past
    # when London has been seen and the read bar sits in
    # a non-entry session (the day's off hours).
    out["past_cutoff"] = bool(
        out["london_seen"]
        and out["last_session"] not in ENTRY_SESSIONS)
    return out
