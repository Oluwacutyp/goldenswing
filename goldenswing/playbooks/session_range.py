"""Session / opening range family (MarketSituation
Pass 7e, SPEC §6.5) — the day's opening structures:

- ``session_opening_breakout``: the frozen Initial
  Balance breaks and the break is ACCEPTED — closes
  holding beyond the edge with follow-through (a
  displacement close among them, or closes
  progressing beyond the probe close) → continuation
  toward the measured extension (the IB height
  projected from the broken edge), then the next
  situation zone beyond it.  Entry at the first
  trigger close after the accepting formation close,
  while that close still holds beyond the edge.
- ``session_opening_failure``: a probe beyond the IB
  edge that closes back inside before acceptance —
  the failed auction at the session's own scale →
  rotation to the IB mid first, then the opposite
  edge (staged); entry at the first trigger close
  after the rejection close, while price is still
  inside the range; stop beyond the probe extreme.
- ``session_drive_reversal``: the day OPENED outside
  prior value (the drive premise — Dalton's open
  drive ground) and London swept the overnight
  (Asia) extreme and rejected → fade toward the IB
  mid, then the prior value edge the drive left.
- ``session_asia_sweep``: on a day that did NOT open
  outside prior value, London's raid of the Asia
  range (the day's first value) closes back inside →
  rotation to the Asia mid, then the opposite Asia
  edge.  The drive playbook owns the sweep when the
  drive premise stands; this one owns every other
  day's raid — the two can never double-fire.

TIME HONESTY is the family's declared rule, enforced
here at the entry read: an entry may print only in
the day's tradable sessions (London / overlap / New
York, ``live/session_range.ENTRY_SESSIONS`` — the
DST-aware labels the situation's activity block
carries).  A formation fact that confirms in the
day's off hours (or in Asia) is journaled by the arm
stream as the fact it is and refused here as an
entry — a breakout accepted at 21:00 is a different
bet from the morning acceptance, and this family
does not silently trade it as the morning shape.  A
day that forms no range abstains the same way, with
the reason on the arm record.

The measurement core is ``live/session_range.py``
over the engine's own session organs (day_context's
NY day + IB, session_timeframes' DST-aware session
partition, volume_profile's prior-day value — a
TICK-VOLUME profile, named on every record that
carries it, never dressed as traded volume).

Boundary (SPEC §6.5): ``killzone_open`` stays where
it is — a pre-session range break at the London / NY
opens on the H1 clock, entered on the RETEST.  This
family's references are the day's own frozen objects
(IB / Asia range / prior value), its formation clock
is H1 with entries on M15, its breakout enters on
acceptance rather than a retest, and its failure /
sweep / drive hypotheses trade rejections
killzone_open never takes.  7c's auction family
reads probes against MIGRATING developing value;
these references never migrate.

The runner threads two facts in per scan, in the
``set_h4_frame`` idiom:

- ``set_formation(frame, tf)`` — the formation clock
  (H1) and its frame (the scan itself runs on the
  TRIGGER clock: the first read after a formation
  fact is a trigger-clock event);
- ``set_formation_tf(tf)`` — the scan's own clock,
  which for this family IS its trigger clock (M15);
- ``set_situation`` — the current MarketSituation
  snapshot (zones for the breakout's second target,
  day levels for arm corroboration, the opening type
  as recorded evidence).

``live = False``: the family's candidates are scored
and logged, never in the live P&L book — the
scheduled route is shadow (journal + scenario book
only) until the family's own promotion evaluation
earns a separate owner decision (SPEC §6 common
contract).
"""
from __future__ import annotations

from ..backtest.harness import simulate_signal_v2
from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..live import session_range as _sr
from ..reader.session_timeframes import session_of
from .base import Candidate, EngineContext, Playbook

FAMILY = "session_range"

FORMATION_TF = "H1"
TRIGGER_TF = "M15"


def _ts(frame, k):
    try:
        return str(frame.index[k])
    except Exception:  # noqa: BLE001 — unreadable index
        return None


def _range_block(ib: dict | None) -> dict | None:
    if not ib:
        return None
    return {"high": ib["high"], "low": ib["low"],
            "width": ib["width"], "mid": ib["mid"],
            "bar_ts": ib["ts"]}


def _asia_block(asia: dict | None) -> dict | None:
    if not asia:
        return None
    return {"high": asia["high"], "low": asia["low"],
            "mid": asia["mid"], "width": asia["width"],
            "bars": asia["bars"]}


class _SessionRangeBase(Playbook):
    """Shared plumbing for the four hypotheses.  The
    scan frame is the TRIGGER frame (M15); the
    formation frame (H1) + its clock are threaded per
    scan."""

    regime_scope = "both"
    session_scope = "any"
    hunting_ground = HuntingGround(excluded=set(), bonuses={})
    live = False
    live_threshold = 60.0

    def __init__(self) -> None:
        self._processed = -1  # last TRIGGER bar scanned
        # Threaded by the runner per scan (see the module
        # docstring).  Defaults are the bare/research
        # path: formation H1, trigger M15, no formation
        # frame, no snapshot.
        self._formation_tf = FORMATION_TF
        self._formation_frame = None
        self._trigger_tf = TRIGGER_TF
        self._situation = None

    # -- threading (the set_h4_frame idiom) -------------------

    def set_formation(self, frame, tf) -> None:
        """The formation clock + its frame for the scan
        about to run (the scan's own frame is the
        trigger clock's)."""
        if tf:
            self._formation_tf = str(tf)
        self._formation_frame = frame

    def set_formation_tf(self, tf) -> None:
        """The scan's own clock.  For this family the
        scan runs on the TRIGGER clock, so this names
        the instance's trigger (M15)."""
        if tf == TRIGGER_TF:
            self._trigger_tf = str(tf)

    def set_situation(self, snapshot) -> None:
        """Thread the current MarketSituation snapshot
        (a dict in live/situation.py's schema, or
        None)."""
        self._situation = snapshot

    # -- situation reads ---------------------------------------

    def _situation_id(self):
        snap = self._situation
        if isinstance(snap, dict):
            return snap.get("situation_id")
        return None

    def _opening_type(self):
        """The day's opening type as the SITUATION
        recorded it (regime.day — the day object's
        Dalton classification at the London / NY
        checkpoints): recorded evidence on the family's
        signals, never re-derived and never a gate."""
        snap = self._situation
        if not isinstance(snap, dict):
            return None
        day = (snap.get("regime") or {}).get("day") or {}
        return day.get("opening_type")

    def _zone_beyond(self, entry: float,
                     direction: int) -> float | None:
        """The nearest situation zone level strictly
        beyond ``entry`` in ``direction`` (its proximal
        edge — the edge price meets first), from the
        snapshot's location block.  None when no
        snapshot / no such zone: absent stays absent,
        the measured projection stands alone."""
        snap = self._situation
        if not isinstance(snap, dict):
            return None
        location = snap.get("location") or {}
        zones = location.get(
            "zones_above" if direction > 0
            else "zones_below") or []
        best: float | None = None
        for z in zones:
            if not isinstance(z, dict):
                continue
            level = z.get("bottom" if direction > 0
                          else "top")
            if level is None:
                continue
            level = float(level)
            if (level - entry) * direction <= 0:
                continue
            if best is None or (level - best) * direction < 0:
                best = level
        return best

    # -- the cutoff (the family's declared time rule) --------

    def _entry_session(self, df, m: int) -> str | None:
        """The entry bar's session label (session_of —
        the same DST-aware partition the situation's
        activity block carries), or None when it cannot
        be read.  Entries exist only in ENTRY_SESSIONS;
        every hypothesis checks this before it exists
        as a signal."""
        try:
            return session_of(df.index[m])[0]
        except Exception:  # noqa: BLE001 — unreadable clock
            return None

    def _entry_allowed(self, df, m: int) -> bool:
        return self._entry_session(df, m) in _sr.ENTRY_SESSIONS

    # -- arming (the runner's detection journal) --------------

    def pending_state(self, df) -> dict | None:
        """The family's live ARM as of the trigger
        frame's last closed bar (read off the BREAKOUT
        playbook — the day machine is shared by all four
        hypotheses, so one arm stream serves the
        family): the day's standing session fact —
        a live probe (``probing``), a resolution
        (``accepted`` / ``failed`` / ``swept`` /
        ``drive_rejected``), a death (``invalidated``),
        or one of the family's two standing ABSTENTIONS
        (``no_range`` — the day formed no range, with
        its reason; ``past_cutoff`` — the range stands
        but the day's entry window has closed with no
        resolution taken).  None when the day is still
        inside its opening window with nothing to say.
        Pure.

        Corroboration: the frozen range the machine
        read must be the range the SITUATION's day
        levels carry (location.day_levels — the same
        day object, read by the runner); a snapshot
        whose day levels name a different range does
        not corroborate the arm.  With no snapshot
        threaded the frame read stands alone (the
        research path).  Abstentions carry the flag
        but journal regardless — they are the family's
        refusals, and Pass 9 grades refusals."""
        if df is None or self._formation_frame is None \
                or len(df) < 2:
            return None
        m = len(df) - 1
        prep = _sr.PreparedSession(self._formation_frame)
        f_idx = _sr.last_formation_bar(
            df, m, self._formation_frame,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)
        if f_idx < 0:
            return None
        day = _sr.session_day_at(prep, f_idx)
        if day is None:
            return None
        phase: str | None = None
        direction: int | None = None
        seq: dict | None = None
        seq_kind: str | None = None
        brk = day.get("breakout")
        sweep = day.get("sweep")
        drive = day.get("drive")
        candidates = [
            (brk, "breakout"), (sweep, "sweep"),
            (drive, "drive")]
        dead = None
        for s, kind in candidates:
            if s and s["phase"] == _sr.PHASE_INVALIDATED:
                dead = (s, kind)
                break
        if dead is not None:
            seq, seq_kind = dead
            phase = "invalidated"
            direction = int(seq["direction"])
        else:
            for s, kind in candidates:
                if s and s["phase"] in (
                        _sr.PHASE_ACCEPTED, _sr.PHASE_REJECTED,
                        _sr.PHASE_SWEPT,
                        _sr.PHASE_DRIVE_REJECTED):
                    seq, seq_kind = s, kind
                    phase = ("failed"
                             if s["phase"] == _sr.PHASE_REJECTED
                             else s["phase"])
                    direction = int(s["direction"])
                    break
            if phase is None:
                for s, kind in candidates:
                    if s and s["phase"] == _sr.PHASE_PROBING:
                        seq, seq_kind = s, kind
                        phase = "probing"
                        direction = int(s["direction"])
                        break
        if phase is None:
            if day.get("no_range") is not None:
                if day["no_range"]["reason"] == "window_open":
                    return None
                phase = "no_range"
            elif day.get("past_cutoff"):
                phase = "past_cutoff"
            elif day.get("ib") is not None:
                phase = "range_set"
            else:
                return None
        corroborated = False
        if day.get("ib") is not None:
            corroborated = True
            snap = self._situation
            if isinstance(snap, dict):
                levels = (snap.get("location") or {}).get(
                    "day_levels") or {}
                snap_high = levels.get("ib_high")
                snap_low = levels.get("ib_low")
                if snap_high is None or snap_low is None \
                        or abs(float(snap_high)
                               - day["ib"]["high"]) > 1e-6 \
                        or abs(float(snap_low)
                               - day["ib"]["low"]) > 1e-6:
                    corroborated = False
        identity_seq = (seq or {}).get("identity") or "|".join(
            ["H1", day["day_key"], "day"])
        return {
            "phase": phase,
            "seq_phase": (seq or {}).get("phase"),
            "seq_kind": seq_kind,
            "formation_tf": self._formation_tf,
            "trigger_tf": self._trigger_tf,
            "direction": direction,
            "day": {
                "day_key": day["day_key"],
                "open_ts": day["open_ts"],
                "range": _range_block(day.get("ib")),
                "asia": _asia_block(day.get("asia")),
                "opened_outside_value":
                    day.get("opened_outside_value"),
                "no_range": day.get("no_range"),
                "last_session": day.get("last_session"),
                "past_cutoff": bool(day.get("past_cutoff"))},
            "sequence": seq,
            "corroborated": corroborated,
            "situation_id": self._situation_id(),
            "identity": f"{identity_seq}|{phase}",
        }

    # -- detection -------------------------------------------

    def _detect_at(self, prep_f, prep_t, df, m: int,
                   f_idx: int):
        raise NotImplementedError

    def _scan(self, df, start: int, end: int):
        """Yield (m, signal) for trigger bars start..end.
        One causal code path for live_signal and
        candidates: each trigger bar's read sees
        formation bars closed by that trigger bar's
        close, and trigger bars <= itself."""
        if self._formation_frame is None:
            return
        prep_f = _sr.PreparedSession(self._formation_frame)
        prep_t = _sr._tm.PreparedFrame(
            df, swing_n=_sr._tm.TRIGGER_SWING_N)
        for m in range(start, end + 1):
            f_idx = _sr.last_formation_bar(
                df, m, self._formation_frame,
                formation_tf=self._formation_tf,
                trigger_tf=self._trigger_tf)
            if f_idx < 0:
                continue
            sig = self._detect_at(prep_f, prep_t, df, m, f_idx)
            if sig is not None:
                yield m, sig

    def live_signal(self, states: list, t: int, df=None) -> list:
        if df is None or t < 0 or t >= len(df):
            return []
        out = [sig for _m, sig in
               self._scan(df, self._processed + 1, t)]
        self._processed = t
        return out

    def candidates(self, ctx: EngineContext) -> list[Candidate]:
        """The research path: ctx.df is the TRIGGER frame
        and the formation frame must be threaded via
        set_formation (the family is a two-clock
        detector; unthreaded, it honestly produces
        nothing)."""
        df = ctx.df
        if df is None or len(df) < 30 \
                or self._formation_frame is None:
            return []
        out: list[Candidate] = []
        for m, sig in self._scan(df, 0, len(df) - 1):
            oc = simulate_signal_v2(df, sig)
            if oc is None:
                continue
            side = "long" if sig.direction > 0 else "short"
            out.append(Candidate(
                playbook=self.name, outcome=oc,
                base_score=float(sig.score),
                thesis=(f"{self.name} {side} @ "
                        f"{sig.entry_ref:.1f} (bar {m}): "
                        f"{'; '.join(sig.factors)}"),
                context=dict(sig.live_context or {})))
        return out

    def export_state(self) -> dict:
        return {"processed": self._processed}

    def import_state(self, state: dict) -> None:
        self._processed = int((state or {}).get("processed", -1))

    # -- shared signal construction ---------------------------

    def _context(self, *, hypothesis, day, terms,
                 extra) -> dict:
        formation_tf = self._formation_tf
        trigger_tf = self._trigger_tf
        ctx = {
            "family": FAMILY,
            "hypothesis": hypothesis,
            "formation_tf": formation_tf,
            "trigger_tf": trigger_tf,
            "entry_tf": trigger_tf,
            # The additive wiring facts the runner's shared
            # builders read (management_block_for's context
            # clock; the book's per-scenario clocks).  The
            # family's horizon is CONSTANT (intraday,
            # SPEC §6.5 — every sequence is a session
            # object), so no management_horizon key is
            # carried: the registry default resolves it,
            # and the management clock is the formation
            # clock — the day thesis is managed against
            # the session it formed in.
            "management_context_timeframe": formation_tf,
            "scenario_clocks": {
                "formation_tf": formation_tf,
                "refinement_tf": None,
                "trigger_tf": trigger_tf,
                "entry_tf": trigger_tf,
                "management_tf": formation_tf},
            "situation_id": self._situation_id(),
            "opening_type": self._opening_type(),
            "day": {
                "day_key": day["day_key"],
                "range": _range_block(day.get("ib")),
                "asia": _asia_block(day.get("asia")),
                "opened_outside_value":
                    day.get("opened_outside_value"),
                "prior_value": day.get("prior_value")},
            "invalidation_terms": list(terms),
        }
        ctx.update(extra)
        return ctx

    def _signal(self, *, index, timestamp, direction, score,
                factors, entry_ref, entry_zone, invalidation,
                targets, atr, context) -> GoldenSignal:
        sig = GoldenSignal(
            index=index, timestamp=timestamp,
            direction=direction, score=float(score),
            kind=self.name, factors=list(factors),
            entry_zone=(float(entry_zone[0]),
                        float(entry_zone[1])),
            entry_ref=float(entry_ref),
            invalidation=float(invalidation),
            targets=[float(t) for t in targets],
            atr=float(atr))
        sig.live_context = dict(context)
        return sig

    def _first_trigger_after(self, df, formation_bar: int) -> int:
        return _sr.first_bar_after(
            self._formation_frame, formation_bar, df,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)


class SessionOpeningBreakoutPlaybook(_SessionRangeBase):
    name = "session_opening_breakout"
    description = (
        "Session opening-range breakout: the day's "
        "frozen Initial Balance (first H1 bar) breaks "
        "and the break is ACCEPTED — at least two "
        "formation closes beyond the edge with "
        "follow-through (a displacement close among "
        "them, or closes progressing beyond the probe "
        "close).  Entry at the first trigger close "
        "after the accepting close, while that close "
        "still holds beyond the edge and prints in a "
        "tradable session; stop back inside the range "
        "beyond the broken edge; targets the measured "
        "extension (the IB height projected), then the "
        "next situation zone beyond it."
    )
    context_description = (
        "The range is the day object's IB (live/"
        "day_context.py — the trading day's first H1 "
        "bar), frozen when the day's second bar prints; "
        "later bars never widen it.  A single close "
        "beyond the edge is a probe, not a breakout: "
        "acceptance needs the second holding close AND "
        "follow-through — the reaction doctrine's "
        "acceptance-by-closes read, never the first "
        "poke.  Distinct from killzone_open (a "
        "pre-session range break entered on the RETEST, "
        "H1 clock): this entry is the acceptance itself, "
        "on the M15 read after it."
    )

    def _detect_at(self, prep_f, prep_t, df, m, f_idx):
        day = _sr.session_day_at(prep_f, f_idx)
        if day is None or day.get("ib") is None:
            return None
        seq = day.get("breakout")
        if seq is None or seq["phase"] != _sr.PHASE_ACCEPTED:
            return None
        accept_bar = seq["resolution_bar"]
        if accept_bar is None:
            return None
        # Entry at the FIRST trigger close after the
        # accepting formation close — one acceptance,
        # one entry read; later trigger bars are the
        # trade running, not new entries.
        if m != self._first_trigger_after(df, accept_bar):
            return None
        if not self._entry_allowed(df, m):
            return None
        d = int(seq["direction"])
        ib = day["ib"]
        edge = ib["high"] if d > 0 else ib["low"]
        entry = float(df["close"].iloc[m])
        if (entry - edge) * d <= 0:
            return None
        atr_f = prep_f.atr(accept_bar)
        if not atr_f:
            return None
        stop = edge - d * _sr.STOP_BUFFER_ATR * atr_f
        # Targets: the measured extension first (the IB
        # height projected from the broken edge), then
        # the next situation zone beyond it — filtered
        # to levels strictly beyond the entry (an entry
        # that already ran past the projection has
        # consumed it; nothing left = no signal, the
        # honest abstention).
        measured = edge + d * ib["width"]
        targets = [t for t in (measured,)
                   if (t - entry) * d > 0]
        zone = self._zone_beyond(entry, d)
        if zone is not None and all(
                (zone - t) * d > 0 for t in targets):
            targets.append(zone)
        if not targets:
            return None
        score = 64.0 + (4.0 if prep_t.body_dominated(
            m, d) else 0.0)
        terms = [
            (f"{self._formation_tf} close back inside the "
             f"opening range (beyond {edge:.2f}) — the "
             "acceptance failed; the breakout thesis is "
             "dead (thesis exit, Pass 6 plan)"),
            ("the day's entry window closing (the off "
             "hours) ends the session thesis by its own "
             "clock"),
        ]
        context = self._context(
            hypothesis="opening_range_breakout", day=day,
            terms=terms,
            extra={
                "breakout": {
                    "probe_ts": seq["probe_ts"],
                    "accept_ts": seq["resolution_ts"],
                    "outside_closes": seq["outside_closes"],
                    "displacement": seq["displacement"],
                    "progressed": seq["progressed"],
                    "extreme_price": seq["extreme_price"]},
                "entry_session": self._entry_session(df, m)})
        return self._signal(
            index=m, timestamp=df.index[m], direction=d,
            score=score,
            factors=["session_opening_breakout",
                     "ib_break_accepted",
                     "follow_through_closes"],
            entry_ref=entry,
            entry_zone=(min(entry, edge), max(entry, edge)),
            invalidation=stop, targets=targets,
            atr=atr_f, context=context)


class SessionOpeningFailurePlaybook(_SessionRangeBase):
    name = "session_opening_failure"
    description = (
        "Session opening-range failure: a probe beyond "
        "the frozen IB edge that closes back inside "
        "before acceptance — the failed auction at the "
        "session's own scale.  Entry at the first "
        "trigger close after the rejection close, while "
        "price is still inside the range; rotation "
        "targets the IB mid first, then the opposite "
        "edge (staged); stop beyond the probe extreme "
        "(acceptance forming beyond it is the failure "
        "failing)."
    )
    context_description = (
        "The rejection is the probe sequence's own "
        "fact (live/session_range.py): the probe close "
        "left the frozen range and a later formation "
        "close returned inside BEFORE the acceptance "
        "read (two holding closes + follow-through) "
        "could form.  The rotation is the thesis — mid "
        "first because the traverse pays there if it "
        "pays at all, the far edge as the full traverse "
        "(the day's 80%-style completion is a stated "
        "expectation for the shadow book to grade, "
        "never assumed in the detector).  Distinct "
        "from 7c's value rejection: that reference is "
        "migrating developing value; this one is the "
        "frozen session range."
    )

    def _detect_at(self, prep_f, prep_t, df, m, f_idx):
        day = _sr.session_day_at(prep_f, f_idx)
        if day is None or day.get("ib") is None:
            return None
        seq = day.get("breakout")
        if seq is None or seq["phase"] != _sr.PHASE_REJECTED:
            return None
        resolution_bar = seq["resolution_bar"]
        if resolution_bar is None:
            return None
        if m != self._first_trigger_after(df, resolution_bar):
            return None
        if not self._entry_allowed(df, m):
            return None
        d = int(seq["direction"])       # the probe's side
        fade_d = -d
        ib = day["ib"]
        entry = float(df["close"].iloc[m])
        if not (ib["low"] <= entry <= ib["high"]):
            return None
        targets = [t for t in (ib["mid"],
                               ib["low"] if d > 0
                               else ib["high"])
                   if (t - entry) * fade_d > 0]
        if not targets:
            return None
        atr_f = prep_f.atr(resolution_bar)
        if not atr_f:
            return None
        extreme = float(seq["extreme_price"])
        stop = extreme + d * _sr.STOP_BUFFER_ATR * atr_f
        score = 63.0 + (4.0 if prep_t.body_dominated(
            m, fade_d) else 0.0)
        terms = [
            (f"{self._formation_tf} close beyond the probe "
             f"extreme {extreme:.2f} — the auction accepted "
             "beyond the probe after all; the failure "
             "thesis is dead (thesis exit, Pass 6 plan)"),
            (f"{self._formation_tf} close beyond the probed "
             "edge and holding (acceptance forming) ends "
             "the rotation premise"),
        ]
        context = self._context(
            hypothesis="opening_range_failure", day=day,
            terms=terms,
            extra={
                "failure": {
                    "probe_ts": seq["probe_ts"],
                    "rejection_ts": seq["resolution_ts"],
                    "extreme_price": extreme},
                "rotation": {
                    "first_target": "ib_mid",
                    "second_target": "opposite_edge"},
                "entry_session": self._entry_session(df, m)})
        return self._signal(
            index=m, timestamp=df.index[m], direction=fade_d,
            score=score,
            factors=["session_opening_failure",
                     "probe_rejected_at_range_edge",
                     "rotation_to_mid"],
            entry_ref=entry,
            entry_zone=(min(entry, ib["mid"]),
                        max(entry, ib["mid"])),
            invalidation=stop, targets=targets,
            atr=atr_f, context=context)


class SessionDriveReversalPlaybook(_SessionRangeBase):
    name = "session_drive_reversal"
    description = (
        "Session opening-drive reversal (SPEC §6.5's "
        "opening_drive_reversal): the day OPENED "
        "outside prior value — the drive premise — and "
        "London's opening bars swept the overnight "
        "(Asia) extreme in the drive's direction and "
        "rejected (extreme beyond, close back inside "
        "the Asia range).  Entry at the first trigger "
        "close after the rejection close; fade toward "
        "the IB mid first, then the prior value edge "
        "the drive left; stop beyond the sweep extreme."
    )
    context_description = (
        "The premise is the day object's own account: "
        "the open vs the PRIOR day's value area (live/"
        "volume_profile.py over the prior day's H1 "
        "bars — a tick-volume profile, named on the "
        "record) and Dalton's opening-type vocabulary "
        "(the situation's regime.day opening_type rides "
        "as recorded evidence).  The sweep is of the "
        "OVERNIGHT extreme — the day's Asia range, the "
        "situation's overnight levels — inside London's "
        "first two bars: an opening drive that gets "
        "rejected at the overnight extreme is the "
        "open-rejection-reverse shape, and the fade is "
        "its trade.  This playbook owns the Asia sweep "
        "whenever the drive premise stands (the sweep "
        "playbook abstains on those days) — one event, "
        "one owner."
    )

    def _detect_at(self, prep_f, prep_t, df, m, f_idx):
        day = _sr.session_day_at(prep_f, f_idx)
        if day is None or day.get("ib") is None \
                or day.get("asia") is None \
                or day.get("prior_value") is None:
            return None
        seq = day.get("drive")
        if seq is None or seq["phase"] != _sr.PHASE_DRIVE_REJECTED:
            return None
        sweep_bar = seq["sweep_bar"]
        if m != self._first_trigger_after(df, sweep_bar):
            return None
        if not self._entry_allowed(df, m):
            return None
        d = int(seq["direction"])       # the drive's side
        fade_d = -d
        asia = day["asia"]
        ib = day["ib"]
        entry = float(df["close"].iloc[m])
        if not (asia["low"] <= entry <= asia["high"]):
            return None
        prior_edge = (day["prior_value"]["vah"] if d > 0
                      else day["prior_value"]["val"])
        targets = [t for t in (ib["mid"], prior_edge)
                   if (t - entry) * fade_d > 0]
        if not targets:
            return None
        atr_f = prep_f.atr(sweep_bar)
        if not atr_f:
            return None
        extreme = float(seq["extreme_price"])
        stop = extreme + d * _sr.STOP_BUFFER_ATR * atr_f
        score = 63.0 + (4.0 if prep_t.body_dominated(
            m, fade_d) else 0.0)
        terms = [
            (f"{self._formation_tf} close beyond the sweep "
             f"extreme {extreme:.2f} — the drive was "
             "accepted after all; the reversal thesis is "
             "dead (thesis exit, Pass 6 plan)"),
            (f"{self._formation_tf} close back beyond the "
             "overnight edge and holding — the drive "
             "resumes; the fade's premise is gone"),
        ]
        context = self._context(
            hypothesis="opening_drive_reversal", day=day,
            terms=terms,
            extra={
                "drive": {
                    "opened_outside_value":
                        day.get("opened_outside_value"),
                    "sweep_ts": seq["sweep_ts"],
                    "extreme_price": extreme,
                    "overnight_edge": seq["edge"]},
                "entry_session": self._entry_session(df, m)})
        return self._signal(
            index=m, timestamp=df.index[m], direction=fade_d,
            score=score,
            factors=["session_drive_reversal",
                     "opened_outside_prior_value",
                     "overnight_sweep_rejected"],
            entry_ref=entry,
            entry_zone=(min(entry, ib["mid"]),
                        max(entry, ib["mid"])),
            invalidation=stop, targets=targets,
            atr=atr_f, context=context)


class SessionAsiaSweepPlaybook(_SessionRangeBase):
    name = "session_asia_sweep"
    description = (
        "Session Asia-range play (SPEC §6.5's "
        "asia_range_play): the Asia range is the day's "
        "first value; London's opening bars sweep one "
        "edge (extreme beyond, close back inside) on a "
        "day that did NOT open outside prior value → "
        "rotation to the Asia mid first, then the "
        "opposite Asia edge (staged).  Entry at the "
        "first trigger close after the sweep close; "
        "stop beyond the sweep extreme."
    )
    context_description = (
        "A sweep is a raid, not a repricing: the "
        "London bar's extreme takes the Asia edge and "
        "its close comes home.  A London close BEYOND "
        "the edge is acceptance territory — the raid "
        "read dies for that side (the breakout family "
        "and killzone_open read breaks; this playbook "
        "reads only rejections).  Days that opened "
        "outside prior value belong to the drive "
        "reversal playbook (the same event under a "
        "stronger premise) — this one abstains there, "
        "so the family never double-fires one sweep."
    )

    def _detect_at(self, prep_f, prep_t, df, m, f_idx):
        day = _sr.session_day_at(prep_f, f_idx)
        if day is None or day.get("asia") is None:
            return None
        # The drive premise owns the day when it stands:
        # opened outside prior value -> the drive
        # reversal playbook's ground, never this one's.
        if day.get("opened_outside_value") in (1, -1):
            return None
        seq = day.get("sweep")
        if seq is None or seq["phase"] != _sr.PHASE_SWEPT:
            return None
        sweep_bar = seq["sweep_bar"]
        if m != self._first_trigger_after(df, sweep_bar):
            return None
        if not self._entry_allowed(df, m):
            return None
        d = int(seq["direction"])       # the raid's side
        fade_d = -d
        asia = day["asia"]
        entry = float(df["close"].iloc[m])
        if not (asia["low"] <= entry <= asia["high"]):
            return None
        far_edge = asia["low"] if d > 0 else asia["high"]
        targets = [t for t in (asia["mid"], far_edge)
                   if (t - entry) * fade_d > 0]
        if not targets:
            return None
        atr_f = prep_f.atr(sweep_bar)
        if not atr_f:
            return None
        extreme = float(seq["extreme_price"])
        stop = extreme + d * _sr.STOP_BUFFER_ATR * atr_f
        score = 62.0 + (4.0 if prep_t.body_dominated(
            m, fade_d) else 0.0)
        terms = [
            (f"{self._formation_tf} close beyond the sweep "
             f"extreme {extreme:.2f} — the raid was "
             "accepted after all; the rotation thesis is "
             "dead (thesis exit, Pass 6 plan)"),
            (f"{self._formation_tf} close beyond the Asia "
             "edge and holding — the sweep becomes a "
             "break; the fade's premise is gone"),
        ]
        context = self._context(
            hypothesis="asia_range_play", day=day,
            terms=terms,
            extra={
                "sweep": {
                    "sweep_ts": seq["sweep_ts"],
                    "extreme_price": extreme,
                    "edge": seq["edge"]},
                "rotation": {
                    "first_target": "asia_mid",
                    "second_target": "opposite_edge"},
                "entry_session": self._entry_session(df, m)})
        return self._signal(
            index=m, timestamp=df.index[m], direction=fade_d,
            score=score,
            factors=["session_asia_sweep",
                     "london_sweep_of_asia_edge",
                     "rotation_to_mid"],
            entry_ref=entry,
            entry_zone=(min(entry, asia["mid"]),
                        max(entry, asia["mid"])),
            invalidation=stop, targets=targets,
            atr=atr_f, context=context)


SESSION_PLAYBOOKS = (
    SessionOpeningBreakoutPlaybook,
    SessionOpeningFailurePlaybook,
    SessionDriveReversalPlaybook,
    SessionAsiaSweepPlaybook,
)
