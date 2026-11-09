"""Mean reversion family (MarketSituation Pass 7d,
SPEC §6.4) — the two hypotheses of the extension:

- ``mean_reversion_extension_fade``: price extended a
  stated ATR multiple beyond the anchored developing
  value and the extension STALLED (value stopped
  migrating after it, momentum at the extreme decaying
  or absorbed, closes no longer progressing) → entry
  on the trigger clock's structure turn back toward
  value; targets the POC, then the opposite value
  edge (the rotation, staged); stop beyond the
  extension extreme.
- ``mean_reversion_failed_continuation``: the mirror
  the SPEC registers in this family so the bookkeeping
  lives in one place — a continuation thesis (any
  family's) whose structure DIED at extension: the
  formation close through the extension's defended
  swing.  Entry at the first trigger close after the
  break, while the fade still has room to value; same
  staged targets; same stop beyond the extreme.  The
  two hypotheses are temporally disjoint by
  construction: the fade's turn hunt ends where the
  break begins (a stalled episode that breaks hands
  its ground to the mirror), so they can never
  double-fire on one episode.

THE REGIME GATE is this family's honesty, stated in
SPEC §6.4 as an exclusion: the family does NOT fade a
clean trend.  The gate reads the formation TF's
structure phase (the situation's, when a snapshot is
threaded; the formation frame's own bedrock read on
the research path — source stated on every record):

- phase clean_trend / breakout → REFUSED, unless the
  episode is the named post-exhaustion state: stalled
  with momentum ABSORBED and >= POST_EXHAUSTION_MIN_
  BARS without close progress — then the gate passes
  and names the override (a blow-off top whose
  follow-through the market itself absorbed is not
  "a clean trend" any more; the reads say so);
- any other phase (range / compression / choppy /
  reversal / unclear / transition) → rotational
  ground, the gate passes on the stall's own
  evidence;
- a BROKE episode passes on the break itself: the
  continuation claim died on its own timeframe,
  which is the fact the gate exists to wait for.

A refusal is a journaled fact, not a silent absence:
the family's arm stream carries a ``regime_refused``
record with the gate's reason (Pass 9 grades what
this family refused and why).

Divergence (SPEC §6.4 names it an input) rides as
RECORDED EVIDENCE from the situation's per-TF
momentum block when a snapshot is threaded — never
a detector gate: re-deriving the feature inside the
detector would fork it, and the stall trio (value
not following + momentum quality + no close
progress) is the family's stated evidence.

The measurement core is ``live/mean_reversion.py``
over ``live/developing_value.py`` (the shared §6.3
extension — value is a TICK-VOLUME profile, named on
every record, never dressed as traded volume) and
``live/trend_momentum.py``'s momentum read.

The runner threads three facts in per scan, in the
``set_h4_frame`` idiom:

- ``set_formation(frame, tf)`` — the formation clock
  (H1) and its frame (the scan itself runs on the
  TRIGGER clock: the turn and the first post-break
  read are trigger-clock events);
- ``set_formation_tf(tf)`` — the scan's own clock,
  which for this family IS its trigger clock (M15
  intraday / M5 scalp): the instance's horizon
  follows it;
- ``set_situation`` — the current MarketSituation
  snapshot.

``live = False``: the family's candidates are scored
and logged, never in the live P&L book — the scheduled
route is shadow (journal + scenario book only) until
the family's own promotion evaluation earns a
separate owner decision (SPEC §6 common contract).
"""
from __future__ import annotations

from ..backtest.harness import simulate_signal_v2
from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..live import mean_reversion as _mr
from ..live import multitf
from ..live import trend_momentum as _tm
from ..live.regime import classify_regime
from .base import Candidate, EngineContext, Playbook

FAMILY = "mean_reversion"

FORMATION_TF = "H1"
TRIGGER_TFS = ("M15", "M5")

#: Bedrock structure phases that count as trend-phased
#: (live/relationships.py's TREND_PHASES — one
#: vocabulary for the situation and the families).
TREND_PHASES = frozenset({"clean_trend", "breakout"})

#: Arm phase the runner journals when a stalled
#: episode's fade is refused by the regime gate (the
#: refusal record — see the module docstring).
PHASE_REGIME_REFUSED = "regime_refused"


def _ts(frame, k):
    try:
        return str(frame.index[k])
    except Exception:
        return None


def _value_block(state: dict | None) -> dict | None:
    """The developing-value summary carried on contexts
    and arm records: levels, day, migration — and the
    volume basis, named on every record (tick volume,
    never dressed as traded volume)."""
    if not state or state.get("status") != "ok":
        return None
    return {"poc": state["poc"], "vah": state["vah"],
            "val": state["val"],
            "day_key": state.get("day_key"),
            "bars": state.get("bars"),
            "volume_basis": state.get("volume_basis"),
            "migration": state.get("migration")}


class _MeanReversionBase(Playbook):
    """Shared plumbing for the two hypotheses.  The
    scan frame is the TRIGGER frame (M15 / M5); the
    formation frame (H1) + its clock are threaded per
    scan, and the scan's own clock (the trigger clock)
    with them."""

    regime_scope = "both"
    session_scope = "any"
    hunting_ground = HuntingGround(excluded=set(), bonuses={})
    live = False
    live_threshold = 60.0

    def __init__(self) -> None:
        self._processed = -1  # last TRIGGER bar scanned
        # Threaded by the runner per scan (see the module
        # docstring).  Defaults are the bare/research path:
        # formation H1, trigger M15 (the family's intraday
        # home trigger), no formation frame, no snapshot.
        self._formation_tf = FORMATION_TF
        self._formation_frame = None
        self._trigger_tf = "M15"
        self._situation = None

    # -- threading (the set_h4_frame idiom) -------------------

    def set_formation(self, frame, tf) -> None:
        """The formation clock + its frame for the scan
        about to run (the scan's own frame is the trigger
        clock's)."""
        if tf:
            self._formation_tf = str(tf)
        self._formation_frame = frame

    def set_formation_tf(self, tf) -> None:
        """The scan's own clock.  For this family the
        scan runs on the TRIGGER clock, so this names
        the instance's trigger (M15 / M5) — and with it
        the instance's horizon."""
        if tf in TRIGGER_TFS:
            self._trigger_tf = str(tf)

    def set_situation(self, snapshot) -> None:
        """Thread the current MarketSituation snapshot (a
        dict in live/situation.py's schema, or None)."""
        self._situation = snapshot

    @property
    def _horizon(self) -> str:
        """SPEC §6.4: intraday, scalp for M5-triggered —
        declared per instance on the signal (the registry
        spec carries the intraday default)."""
        return "scalp" if self._trigger_tf == "M5" \
            else "intraday"

    # -- situation reads ---------------------------------------

    def _situation_id(self):
        snap = self._situation
        if isinstance(snap, dict):
            return snap.get("situation_id")
        return None

    def _pair_kinds(self, snap: dict) -> dict:
        kinds: dict = {}
        for rel in snap.get("relationships") or []:
            if rel.get("source") == "engine":
                kinds[(rel.get("htf"), rel.get("ltf"))] = \
                    rel.get("kind")
        if not kinds:
            for rel in snap.get("relationships") or []:
                kinds.setdefault(
                    (rel.get("htf"), rel.get("ltf")),
                    rel.get("kind"))
        return kinds

    def _divergence_evidence(self) -> dict | None:
        """The situation's divergence read for the
        formation TF (SPEC §6.4 input) — recorded
        evidence on the signal's context when a snapshot
        is threaded; absent (None) otherwise, never
        synthesized by the detector."""
        snap = self._situation
        if not isinstance(snap, dict):
            return None
        own = (snap.get("per_tf") or {}).get(
            self._formation_tf) or {}
        div = (own.get("momentum") or {}).get("divergence")
        return dict(div) if isinstance(div, dict) else None

    # -- the regime gate (the family's honesty) ---------------

    def _regime_gate(self, prep: _mr.PreparedReversion,
                     ep: dict, f_idx: int) -> dict:
        """The gate read for episode ``ep`` as of
        formation bar ``f_idx`` (see the module
        docstring for the rules).  Returns the gate
        block carried on signals, arm records and
        refusal records: verdict pass / refused, the
        basis that decided it, the phase read and its
        source — stated, never hidden."""
        formation_tf = self._formation_tf
        d = int(ep["direction"])
        phase = None
        bias = None
        pair = None
        source = None
        snap = self._situation
        if isinstance(snap, dict):
            per_tf = snap.get("per_tf") or {}
            own = per_tf.get(formation_tf) or {}
            phase = own.get("structure_phase")
            bias = own.get("bias")
            pair = self._pair_kinds(snap).get(
                (formation_tf, self._trigger_tf))
            source = "situation"
        if phase is None:
            # Research path: the formation frame's own
            # bedrock read, causally at f_idx (the same
            # classifier the situation engine runs per
            # TF — classify_regime + canonical bias).
            frame = self._formation_frame
            if frame is not None and 0 <= f_idx < len(frame):
                sliced = frame.iloc[:f_idx + 1]
                try:
                    regime = classify_regime(sliced)
                    phase = regime.structure \
                        if regime is not None else None
                    bias = int(multitf.evaluate_bias(sliced))
                except Exception:  # noqa: BLE001 — read degrades
                    phase, bias = None, None
            source = "frame_structure"
        base = {"structure_phase": phase,
                "formation_bias": bias,
                "pair_kind": pair,
                "source": source,
                "situation_id": self._situation_id()}
        if ep["phase"] == _mr.PHASE_BROKE:
            return {**base, "verdict": "pass",
                    "basis": "structure_break",
                    "reason": (
                        "the extension's defended swing "
                        "broke on the formation clock — the "
                        "continuation claim died on its own "
                        "timeframe; the state the gate "
                        "protects against no longer stands")}
        if phase in TREND_PHASES:
            mom = ep.get("momentum_at_stall") or {}
            # Absorption is the exhaustion fact and can
            # land a bar or two AFTER the stall confirms
            # (the stall records the first non-sustained
            # read), so the override reads momentum at the
            # decision bar as well as at the stall, and
            # no-progress is measured to the decision bar.
            mom_now = {}
            try:
                mom_now = prep.f.momentum_at(f_idx, d) or {}
            except Exception:  # noqa: BLE001 — read degrades
                mom_now = {}
            no_progress = int(f_idx - ep["extreme_bar"])
            absorbed = (
                mom.get("state") == _tm.MOMENTUM_ABSORBED
                or mom_now.get("state")
                == _tm.MOMENTUM_ABSORBED)
            trend_block = {
                "momentum_at_stall": mom.get("state"),
                "momentum_now": mom_now.get("state"),
                "no_progress_bars": no_progress}
            if ep["phase"] == _mr.PHASE_STALLED \
                    and absorbed \
                    and no_progress \
                    >= _mr.POST_EXHAUSTION_MIN_BARS:
                return {**base, **trend_block,
                        "verdict": "pass",
                        "basis": "post_exhaustion",
                        "reason": (
                            f"formation TF reads {phase}, "
                            "but the extension is the named "
                            "post-exhaustion state: momentum "
                            "absorbed at the extreme and "
                            f"{no_progress} formation bars "
                            "without close progress — the "
                            "gate's override, named")}
            return {**base, **trend_block,
                    "verdict": "refused",
                    "basis": "clean_trend",
                    "reason": (
                        f"formation TF reads {phase} — this "
                        "family does not fade a clean trend "
                        "(an extension inside a live trend "
                        "may be the trend repricing, not "
                        "exhausting; fading it is how this "
                        "family dies)")}
        return {**base, "verdict": "pass",
                "basis": "rotational_state",
                "reason": (
                    f"formation TF reads "
                    f"{phase or 'unmeasured'} — balance / "
                    "rotation ground; the stall's own "
                    "evidence carries the hypothesis")}

    # -- arming (the runner's detection journal) --------------

    def pending_state(self, df) -> dict | None:
        """The family's live ARM as of the trigger frame's
        last closed bar (read off the EXTENSION FADE
        playbook — the episode machine is shared by both
        hypotheses, so one arm stream serves the family):
        the episode's phase (extended / stalled / broke /
        invalidated) with its geometry, the developing-
        value read, and the regime gate's verdict — a
        stalled episode the gate refuses reports as
        ``regime_refused`` so the refusal journals as
        its own record.  None when no episode is live or
        resolved-and-standing.  Pure."""
        if df is None or self._formation_frame is None \
                or len(df) < 2:
            return None
        m = len(df) - 1
        prep = _mr.PreparedReversion(self._formation_frame)
        f_idx = _mr.last_formation_bar(
            df, m, self._formation_frame,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)
        if f_idx < 0:
            return None
        ep = _mr.reversion_episode_at(prep, f_idx)
        if ep is None or ep["phase"] not in (
                _mr.PHASE_EXTENDED, _mr.PHASE_STALLED,
                _mr.PHASE_BROKE, _mr.PHASE_INVALIDATED):
            return None
        gate = self._regime_gate(prep, ep, f_idx)
        phase = ep["phase"]
        if phase == _mr.PHASE_STALLED \
                and gate["verdict"] == "refused":
            phase = PHASE_REGIME_REFUSED
        state = prep.value_at(f_idx)
        corroborated = state.get("status") == "ok"
        snap = self._situation
        if corroborated and isinstance(snap, dict):
            snap_value = (snap.get("location") or {}).get(
                "value") or {}
            developing = snap_value.get("developing_state") \
                or {}
            if developing.get("day_key") not in (
                    None, state.get("day_key")) or not snap_value:
                corroborated = False
        return {
            "phase": phase,
            "episode_phase": ep["phase"],
            "formation_tf": self._formation_tf,
            "trigger_tf": self._trigger_tf,
            "direction": int(ep["direction"]),
            "episode": {
                "first_ts": ep["first_ts"],
                "ref": dict(ep["ref"]),
                "extension_atr_first":
                    ep["extension_atr_first"],
                "extreme_close": ep["extreme_close"],
                "extreme_price": ep["extreme_price"],
                "extreme_ts": ep["extreme_ts"],
                "defended_swing": ep["defended_swing"],
                "stall_ts": ep["stall_ts"],
                "momentum_at_stall":
                    ep["momentum_at_stall"],
                "break_ts": ep["break_ts"],
                "death_ts": ep["death_ts"],
                "death_kind": ep["death_kind"]},
            "extension_atr": ep["extension_atr"],
            "value": _value_block(state),
            "gate": gate,
            "corroborated": corroborated,
            "situation_id": self._situation_id(),
            "identity": "|".join([
                ep["identity"], phase]),
        }

    # -- detection -------------------------------------------

    def _detect_at(self, prep_f, prep_t, df, m: int,
                   f_idx: int) -> GoldenSignal | None:
        raise NotImplementedError

    def _scan(self, df, start: int, end: int):
        """Yield (m, signal) for trigger bars start..end.
        One causal code path for live_signal and
        candidates: each trigger bar's read sees
        formation bars closed by that trigger bar's
        close, and trigger bars <= itself."""
        if self._formation_frame is None:
            return
        prep_f = _mr.PreparedReversion(self._formation_frame)
        prep_t = _tm.PreparedFrame(
            df, swing_n=_tm.TRIGGER_SWING_N)
        for m in range(start, end + 1):
            f_idx = _mr.last_formation_bar(
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
        set_formation (the family is a two-clock detector;
        unthreaded, it honestly produces nothing)."""
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

    def _targets(self, ep: dict, entry: float):
        """The staged rotation targets (POC first, the
        opposite value edge second), filtered to levels
        strictly beyond the entry in the fade's
        direction: a target the entry has already passed
        is consumed, not a target — if POC itself is
        behind the entry the rotation already ran
        without us and there is no signal."""
        d = int(ep["direction"])
        poc = float(ep["ref"]["poc"])
        far_edge = float(ep["ref"]["val"] if d > 0
                         else ep["ref"]["vah"])
        targets = [t for t in (poc, far_edge)
                   if (t - entry) * -d > 0]
        return targets

    def _context(self, *, hypothesis, ep, state, gate,
                 terms, extra) -> dict:
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
            # clock + declared horizon; the book's
            # per-scenario clocks + horizon).  The instance
            # declares its horizon because it VARIES with
            # the trigger clock (M15 intraday / M5 scalp,
            # SPEC §6.4); the management clock is the
            # formation clock — the fade thesis is managed
            # against the value it extended from.
            "management_context_timeframe": formation_tf,
            "management_horizon": self._horizon,
            "scenario_clocks": {
                "formation_tf": formation_tf,
                "refinement_tf": None,
                "trigger_tf": trigger_tf,
                "entry_tf": trigger_tf,
                "management_tf": formation_tf},
            "situation_id": self._situation_id(),
            "volume_basis": (state or {}).get(
                "volume_basis"),
            "value": _value_block(state),
            "extension": {
                "first_ts": ep["first_ts"],
                "ref": dict(ep["ref"]),
                "extension_atr_first":
                    ep["extension_atr_first"],
                "extreme_close": ep["extreme_close"],
                "extreme_price": ep["extreme_price"],
                "extreme_ts": ep["extreme_ts"],
                "defended_swing": ep["defended_swing"]},
            "regime_gate": gate,
            "divergence": self._divergence_evidence(),
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


class MeanReversionExtensionFadePlaybook(_MeanReversionBase):
    name = "mean_reversion_extension_fade"
    description = (
        "Mean reversion extension fade: price extended "
        ">= 2 formation ATRs beyond the anchored "
        "developing value edge and the extension STALLED "
        "— value stopped migrating after it, momentum at "
        "the extreme decaying or absorbed, closes no "
        "longer progressing beyond the extreme.  Entry "
        "on the trigger clock's structure turn back "
        "toward value; rotation targets the POC, then "
        "the opposite value edge; stop beyond the "
        "extension extreme.  The regime gate refuses the "
        "fade inside a clean trend (named post-exhaustion "
        "states excepted)."
    )
    context_description = (
        "Extension and stall from the shared layers "
        "(live/developing_value.py — tick-volume basis, "
        "named; live/trend_momentum.py's momentum read): "
        "the reference value is the developing state at "
        "the last inside close, fixed for the episode; "
        "the stall needs >= 2 formation bars without "
        "close progress past the extreme, the trailing "
        "momentum read off 'sustained', and the current "
        "developing state NOT migrated toward price "
        "since the extreme bar.  The turn is the trigger "
        "clock's first minor-swing (n=2) break in the "
        "fade's direction after the stall confirms, on "
        "a swing formed after the extreme.  A formation "
        "close beyond the extreme after the stall kills "
        "the episode (acceptance / resumption); a "
        "formation close through the defended swing "
        "hands the ground to the failed-continuation "
        "hypothesis."
    )

    def _turn_bar(self, prep_t, df, m: int, m0: int,
                  m_ext: int, fade_d: int) -> bool:
        """Is trigger bar ``m`` THE turn bar — the first
        bar at/after the stall window's open (``m0``)
        whose close breaks the last confirmed minor
        swing (n=2) in the fade's direction, the swing
        formed at/after the extreme's first trigger bar
        (``m_ext``)?  Earlier turn bars inside the
        window disqualify later ones (one episode, one
        turn — the first break is the entry; bars after
        it are the trade running)."""
        closes = df["close"].to_numpy(dtype=float)
        for k in range(m0, m + 1):
            known = [(p, v) for p, v in
                     prep_t.swings_known(
                         k, highs=fade_d > 0)
                     if p >= m_ext]
            if not known:
                continue
            level = known[-1][1]
            if (closes[k] - level) * fade_d > 0:
                return k == m
        return False

    def _detect_at(self, prep_f, prep_t, df, m, f_idx):
        ep = _mr.reversion_episode_at(prep_f, f_idx)
        if ep is None or ep["phase"] != _mr.PHASE_STALLED:
            return None
        d = int(ep["direction"])
        fade_d = -d
        gate = self._regime_gate(prep_f, ep, f_idx)
        if gate["verdict"] == "refused":
            return None
        stall_bar = ep["stall_bar"]
        if stall_bar is None:
            return None
        m0 = _mr.first_bar_after(
            self._formation_frame, stall_bar, df,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)
        if m < m0 or m - m0 \
                >= _mr.TURN_WINDOW_BARS[self._trigger_tf]:
            return None
        m_ext = _mr.first_bar_after(
            self._formation_frame, ep["extreme_bar"], df,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)
        if not self._turn_bar(prep_t, df, m, m0, m_ext,
                              fade_d):
            return None
        entry = float(df["close"].iloc[m])
        targets = self._targets(ep, entry)
        if not targets:
            return None
        atr_f = prep_f.atr(ep["stall_bar"])
        if not atr_f:
            return None
        extreme = float(ep["extreme_price"])
        stop = extreme + d * _mr.STOP_BUFFER_ATR * atr_f
        mom = ep["momentum_at_stall"] or {}
        score = 64.0 + (4.0 if prep_t.body_dominated(
            m, fade_d) else 0.0)
        terms = [
            f"{self._formation_tf} close beyond the "
            f"extension extreme {extreme:.2f} — the "
            "auction accepted beyond the extreme; the "
            "fade's premise is dead (thesis exit, "
            "Pass 6 plan)",
            f"{self._formation_tf} close back inside "
            "the reference value completes the rotation "
            "— the thesis is done, not extended",
        ]
        state = prep_f.value_at(ep["stall_bar"])
        context = self._context(
            hypothesis="extension_fade", ep=ep, state=state,
            gate=gate, terms=terms,
            extra={
                "stall": {
                    "bar_ts": ep["stall_ts"],
                    "bars_since_extreme": int(
                        ep["stall_bar"] - ep["extreme_bar"]),
                    "momentum_state": mom.get("state"),
                    "momentum_net_atr": mom.get("net_atr"),
                    "momentum_efficiency":
                        mom.get("efficiency"),
                    "value_migrated_toward": False},
                "turn": {
                    "bar_ts": _ts(df, m),
                    "trigger_tf": self._trigger_tf},
                "rotation": {
                    "first_target": "poc",
                    "second_target": "opposite_edge"}})
        return self._signal(
            index=m, timestamp=df.index[m], direction=fade_d,
            score=score,
            factors=["mean_reversion_extension_fade",
                     "extension_beyond_value", "stall_confirmed",
                     "ltf_structure_turn"],
            entry_ref=entry,
            entry_zone=(min(entry, float(ep["ref"]["poc"])),
                        max(entry, float(ep["ref"]["poc"]))),
            invalidation=stop, targets=targets,
            atr=atr_f, context=context)


class MeanReversionFailedContinuationPlaybook(
        _MeanReversionBase):
    name = "mean_reversion_failed_continuation"
    description = (
        "Mean reversion failed continuation (SPEC §6.4's "
        "mirror, registered in this family): a "
        "continuation thesis died at extension — the "
        "formation close through the extended move's "
        "defended swing.  Entry at the first trigger "
        "close after the break, while price is still on "
        "the extension side of the POC (the fade still "
        "has its rotation); targets the POC, then the "
        "opposite value edge; stop beyond the extension "
        "extreme (a close back beyond it is the auction "
        "re-accepting the extension — the fade is dead)."
    )
    context_description = (
        "The break is the extension episode's own fact "
        "(live/mean_reversion.py): the defended swing is "
        "the most recent confirmed counter-swing (n=3) "
        "known at the extreme bar, and the break is a "
        "formation close through it after the extreme. "
        "The hypothesis exists only after that close — "
        "there is no pre-break failed continuation — "
        "which is its regime honesty: it never fades a "
        "standing trend, it trades the trend claim's "
        "structural death at extension.  A formation "
        "close beyond the extreme after the break "
        "invalidates the episode before any entry."
    )

    def _detect_at(self, prep_f, prep_t, df, m, f_idx):
        ep = _mr.reversion_episode_at(prep_f, f_idx)
        if ep is None or ep["phase"] != _mr.PHASE_BROKE:
            return None
        d = int(ep["direction"])
        fade_d = -d
        break_bar = ep["break_bar"]
        if break_bar is None:
            return None
        # Entry at the FIRST trigger close after the
        # break formation close (the trend family's
        # acceleration / the auction rejection's
        # arrangement: the formation close is the fact,
        # the first trigger close after it is the entry
        # read) — and only while price is still on the
        # extension side of the POC (a first read already
        # at/past the POC has missed the rotation's
        # premise).
        m0 = _mr.first_bar_after(
            self._formation_frame, break_bar, df,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)
        if m != m0:
            return None
        entry = float(df["close"].iloc[m])
        poc = float(ep["ref"]["poc"])
        if (entry - poc) * d <= 0:
            return None
        targets = self._targets(ep, entry)
        if not targets:
            return None
        atr_f = prep_f.atr(break_bar)
        if not atr_f:
            return None
        extreme = float(ep["extreme_price"])
        stop = extreme + d * _mr.STOP_BUFFER_ATR * atr_f
        gate = self._regime_gate(prep_f, ep, f_idx)
        defended = float(ep["defended_swing"])
        score = 63.0 + (4.0 if prep_t.body_dominated(
            m, fade_d) else 0.0)
        terms = [
            f"{self._formation_tf} close beyond the "
            f"extension extreme {extreme:.2f} — the "
            "auction re-accepted the extension after "
            "the break; the fade's premise is dead "
            "(thesis exit, Pass 6 plan)",
            f"{self._formation_tf} close back above "
            f"the defended swing {defended:.2f} — the "
            "break failed; the continuation claim "
            "stands again",
        ]
        state = prep_f.value_at(break_bar)
        context = self._context(
            hypothesis="failed_continuation", ep=ep,
            state=state, gate=gate, terms=terms,
            extra={
                "break": {
                    "bar_ts": ep["break_ts"],
                    "defended_swing": defended},
                "rotation": {
                    "first_target": "poc",
                    "second_target": "opposite_edge"}})
        return self._signal(
            index=m, timestamp=df.index[m], direction=fade_d,
            score=score,
            factors=["mean_reversion_failed_continuation",
                     "continuation_structure_broke",
                     "extension_unwind"],
            entry_ref=entry,
            entry_zone=(min(entry, poc), max(entry, poc)),
            invalidation=stop, targets=targets,
            atr=atr_f, context=context)


REVERSION_PLAYBOOKS = (
    MeanReversionExtensionFadePlaybook,
    MeanReversionFailedContinuationPlaybook,
)
