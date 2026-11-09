"""Trend / momentum family (MarketSituation Pass 7b,
SPEC §6.1) — the two hypotheses of the trend state:

- ``trend_impulse_pullback``: impulse → pullback that
  holds the impulse's structure → continuation, entered
  on the pullback's LTF structure turn (the trigger
  clock's minor swing break in the trend's direction).
  Stop beyond the pullback's defended swing on the
  formation clock; targets the impulse extreme, then the
  leg's measured projection.
- ``trend_momentum_acceleration``: inside an established
  trend state, velocity rising through a stated percentile
  of its own distribution while closes accept beyond the
  last confirmed swing — the continuation entry.  Stop
  back beyond the broken swing; target the leg's measured
  projection from the entry.

The measurement core is ``live/trend_momentum.py`` (pure,
causal).  This module is the playbook plumbing plus the
family's defining read: the TREND STATE.  A trend here is
a relationship fact, read from the MarketSituation — the
formation timeframe trend-phased (bedrock structure phase
clean_trend / breakout) with its own canonical bias in
the trade's direction, and the formation→trigger pair in
a participating relationship (aligned / correcting /
nested for the pullback — the pullback inside an aligned
pair is this family's home ground; aligned only for
acceleration, which trades WITH live participation).
A conflicting pair is the market stating the opposite
structure fact: no hypothesis forms (the scan's no_setup
record is the abstention; the counter-fact itself rides
the situation's relationship records).  With no snapshot
threaded (the research path), the read is the formation
frame's own organs — canonical bias + bedrock structure
phase on the formation bars themselves.  Which source
decided rides on every signal's context — stated, never
hidden.

The runner threads three facts in per scan, in the
``set_h4_frame`` idiom:

- ``set_formation(frame, tf)`` — the formation clock and
  its frame (the scan itself runs on the TRIGGER clock:
  entries fire when the LTF turn closes, not up to a
  formation bar later);
- ``set_situation`` — the current MarketSituation
  snapshot.

``live = False``: the family's candidates are scored and
logged, never in the live P&L book — the scheduled route
is shadow (journal + scenario book only) until the
family's own promotion evaluation earns a separate owner
decision (SPEC §6 common contract).
"""
from __future__ import annotations

from ..backtest.harness import simulate_signal_v2
from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..live import multitf
from ..live import trend_momentum as _tm
from ..live.regime import classify_regime
from .base import Candidate, EngineContext, Playbook

FAMILY = "trend_momentum"

#: Bedrock structure phases that count as trend-phased
#: (live/relationships.py's own TREND_PHASES — one
#: vocabulary for the situation and the family).
TREND_PHASES = frozenset({"clean_trend", "breakout"})

#: Pair kinds that count as participation for the
#: pullback hypothesis (SPEC §6.1: a pullback inside an
#: ``aligned`` pair is the family's home ground; the
#: correcting pair IS the pullback state at the pair
#: level; nested = the trigger scenario sits inside the
#: formation structure's ground).
_PULLBACK_PAIRS = frozenset({"aligned", "correcting", "nested"})
_ACCEL_PAIRS = frozenset({"aligned"})


def _ts(frame, k):
    try:
        return str(frame.index[k])
    except Exception:
        return None


class _TrendMomentumBase(Playbook):
    """Shared trend-state plumbing for the two hypotheses.
    The scan frame is the TRIGGER frame; the formation
    frame + clock are threaded per scan."""

    regime_scope = "both"
    session_scope = "any"
    hunting_ground = HuntingGround(excluded=set(), bonuses={})
    live = False
    live_threshold = 60.0

    def __init__(self) -> None:
        self._processed = -1  # last TRIGGER bar scanned
        # Threaded by the runner per scan (see the module
        # docstring).  Defaults are the bare/research path:
        # formation on the family's intraday home clock,
        # no formation frame, no snapshot.
        self._formation_tf = "H4"
        self._formation_frame = None
        self._situation = None

    # -- threading (the set_h4_frame idiom) -------------------

    def set_formation(self, frame, tf) -> None:
        """The formation clock + its frame for the scan
        about to run (the scan's own frame is the trigger
        clock's)."""
        if tf:
            self._formation_tf = str(tf)
        self._formation_frame = frame

    def set_situation(self, snapshot) -> None:
        """Thread the current MarketSituation snapshot (a
        dict in live/situation.py's schema, or None)."""
        self._situation = snapshot

    @property
    def _trigger_tf(self) -> str:
        return _tm.TRIGGER_TF_DOWN[self._formation_tf]

    @property
    def _horizon(self) -> str:
        """SPEC §6.1: intraday default, swing for
        D1-formed — declared per instance on the signal
        (the registry spec carries the intraday default)."""
        return "swing" if self._formation_tf == "D1" \
            else "intraday"

    # -- the trend state (the family's defining read) ---------

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

    def _trend_state(self, prep_f: _tm.PreparedFrame,
                     f_idx: int, d: int, *,
                     pairs_ok: frozenset) -> dict | None:
        """The trend state for direction ``d`` as of
        formation bar ``f_idx``: the relationship fact the
        family trades on (see the module docstring).  None
        = the market's wider state does not state this
        trend — no hypothesis, and the scan's no_setup
        record is the abstention."""
        formation_tf = self._formation_tf
        trigger_tf = self._trigger_tf
        snap = self._situation
        if isinstance(snap, dict):
            per_tf = snap.get("per_tf") or {}
            own = per_tf.get(formation_tf) or {}
            if own.get("bias") != d:
                return None
            if own.get("structure_phase") not in TREND_PHASES:
                return None
            kinds = self._pair_kinds(snap)
            pair = kinds.get((formation_tf, trigger_tf))
            if pair not in pairs_ok:
                return None
            above = _tm.ABOVE_TF[formation_tf]
            return {"source": "situation",
                    "pair_kind": pair,
                    "htf_pair_kind": (
                        kinds.get((above, formation_tf))
                        if above else None),
                    "formation_phase":
                        own.get("structure_phase"),
                    "situation_id": snap.get("situation_id")}
        # Research path: the formation frame's own organs,
        # read causally at f_idx (canonical bias + bedrock
        # structure — the same two reads the situation
        # engine assembles, on the same bars).
        frame = self._formation_frame
        if frame is None or f_idx < 0 or f_idx >= len(frame):
            return None
        sliced = frame.iloc[:f_idx + 1]
        try:
            bias = int(multitf.evaluate_bias(sliced))
            regime = classify_regime(sliced)
        except Exception:
            return None
        if bias != d or regime is None \
                or regime.structure not in TREND_PHASES:
            return None
        return {"source": "frame_structure", "pair_kind": None,
                "htf_pair_kind": None,
                "formation_phase": regime.structure,
                "situation_id": None}

    # -- arming (the runner's detection journal) --------------

    def pending_state(self, df) -> dict | None:
        """The family's live ARM as of the trigger frame's
        last closed bar: the trend sequence's phase
        (impulse / pullback / invalidated) with its
        geometry and the trend-state read attached.  None
        when no sequence is live.  Pure."""
        if df is None or self._formation_frame is None \
                or len(df) < 2:
            return None
        m = len(df) - 1
        prep_f = _tm.PreparedFrame(self._formation_frame)
        prep_t = _tm.PreparedFrame(df, swing_n=_tm.TRIGGER_SWING_N)
        f_idx = _tm.last_formation_bar(
            df, m, self._formation_frame,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)
        if f_idx < 0:
            return None
        seq = _tm.sequence_at(
            prep_f, prep_t, f_idx, m,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)
        if seq is None or seq["phase"] not in (
                "impulse", "pullback", "invalidated"):
            return None
        imp = seq["impulse"]
        d = int(seq["direction"])
        corr = self._trend_state(prep_f, f_idx, d,
                                 pairs_ok=_PULLBACK_PAIRS)
        f_frame = self._formation_frame
        out = {
            "phase": seq["phase"],
            "formation_tf": self._formation_tf,
            "trigger_tf": self._trigger_tf,
            "direction": d,
            "impulse": {
                "origin_price": imp["origin_price"],
                "origin_ts": _ts(f_frame, imp["origin_pos"]),
                "break_ts": _ts(f_frame, imp["break_bar"]),
                "break_level": imp["break_level"],
                "extreme_price": imp["extreme_price"],
                "extreme_ts": _ts(f_frame, imp["extreme_bar"]),
                "leg": imp["leg"]},
            "pullback": seq["pullback"],
            "break_ts": (_ts(f_frame, seq["break_bar"])
                         if seq["break_bar"] is not None
                         else None),
            "corroborated": corr is not None,
            "corroboration": corr,
            "identity": "|".join([
                self._formation_tf, seq["phase"],
                str(_ts(f_frame, imp["origin_pos"])),
                str(_ts(f_frame, imp["extreme_bar"])),
                str(_ts(f_frame, seq["break_bar"])
                    if seq["break_bar"] is not None else "")]),
        }
        return out

    # -- detection -------------------------------------------

    def _detect_at(self, prep_f, prep_t, df, m: int,
                   f_idx: int) -> GoldenSignal | None:
        raise NotImplementedError

    def _scan(self, df, start: int, end: int):
        """Yield (m, signal) for trigger bars start..end.
        One causal code path for live_signal and
        candidates: each trigger bar's read sees formation
        bars closed by that trigger bar's close, and
        trigger bars <= itself."""
        if self._formation_frame is None:
            return
        prep_f = _tm.PreparedFrame(self._formation_frame)
        prep_t = _tm.PreparedFrame(
            df, swing_n=_tm.TRIGGER_SWING_N)
        for m in range(start, end + 1):
            f_idx = _tm.last_formation_bar(
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

    def _context(self, *, hypothesis, trend_state, terms,
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
            # clock + declared horizon; the book's
            # per-scenario clocks + horizon).  The instance
            # declares them because its formation clock —
            # and with §6.1 its horizon — varies per
            # instance.  No other family's signals carry the
            # horizon key; their blocks are untouched.
            "management_context_timeframe": formation_tf,
            "management_horizon": self._horizon,
            "scenario_clocks": {
                "formation_tf": formation_tf,
                "refinement_tf": None,
                "trigger_tf": trigger_tf,
                "entry_tf": trigger_tf,
                "management_tf": formation_tf},
            "trend_state": dict(trend_state),
            "situation_id": trend_state.get("situation_id"),
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


class TrendImpulsePullbackPlaybook(_TrendMomentumBase):
    name = "trend_impulse_pullback"
    description = (
        "Trend impulse pullback: an impulse leg whose "
        "follow-through read sustained breaks structure on "
        "the formation clock (H4 intraday / D1 swing); the "
        "pullback retraces inside the leg's band WITHOUT a "
        "formation close beyond the impulse origin; entry "
        "on the pullback's structure turn on the trigger "
        "clock (minor swing break in the trend's "
        "direction).  Stop beyond the pullback's defended "
        "swing on the formation clock; targets the impulse "
        "extreme, then the leg's measured projection."
    )
    context_description = (
        "Trend state from the MarketSituation: formation TF "
        "trend-phased (clean_trend / breakout) with "
        "canonical bias in the trade's direction and the "
        "formation->trigger pair aligned / correcting / "
        "nested.  Impulse: formation close beyond a "
        "confirmed swing with momentum sustained at the "
        "leg's extreme.  Pullback: retrace 0.15-0.85 of the "
        "leg, no formation close beyond the impulse origin, "
        "counter-momentum never sustained, formation "
        "momentum never absorbed at the turn.  A formation "
        "close beyond the origin invalidates the sequence."
    )

    def _detect_at(self, prep_f, prep_t, df, m, f_idx):
        seq = _tm.sequence_at(
            prep_f, prep_t, f_idx, m,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)
        if seq is None or seq["phase"] != "triggered":
            return None
        d = int(seq["direction"])
        corr = self._trend_state(prep_f, f_idx, d,
                                 pairs_ok=_PULLBACK_PAIRS)
        if corr is None:
            return None
        # The abstention gates (SPEC §6.1's momentum
        # doctrine): follow-through absorbed at the turn,
        # or the pullback itself displaced against the
        # trend (a counter-impulse, not a pullback) — no
        # hypothesis either way.
        mom = prep_f.momentum_at(f_idx, d)
        if mom["state"] == _tm.MOMENTUM_ABSORBED:
            return None
        if seq["pullback"]["counter_sustained"]:
            return None
        turn = seq["turn"]
        geo = _tm.pullback_geometry(prep_f, seq, turn,
                                    at_bar=f_idx)
        if not geo["targets"]:
            return None
        imp = seq["impulse"]
        pb = seq["pullback"]
        f_frame = self._formation_frame
        score = 64.0 + (4.0 if prep_t.body_dominated(m, d)
                        else 0.0)
        terms = [
            f"{self._formation_tf} close beyond the "
            f"pullback's defended swing "
            f"{geo['defended_swing']:.2f} after the turn "
            "breaks the trend's structure on the declaring "
            "timeframe — the thesis exit (Pass 6 plan)",
            f"{self._formation_tf} close beyond the impulse "
            f"origin {geo['origin_price']:.2f} negates the "
            "impulse itself",
        ]
        extra = {
            "impulse": {
                "origin_price": imp["origin_price"],
                "origin_ts": _ts(f_frame, imp["origin_pos"]),
                "break_ts": _ts(f_frame, imp["break_bar"]),
                "break_level": imp["break_level"],
                "extreme_price": imp["extreme_price"],
                "extreme_ts": _ts(f_frame, imp["extreme_bar"]),
                "leg": imp["leg"],
                "momentum_at_extreme":
                    imp["momentum_at_extreme"]["state"]},
            "pullback": {
                "extreme_price": pb["extreme_price"],
                "extreme_ts": _ts(f_frame, pb["extreme_bar"]),
                "depth_frac": pb["depth_frac"],
                "counter_sustained": pb["counter_sustained"]},
            "turn": {
                "bar_ts": _ts(df, m),
                "swing_price": turn["swing_price"],
                "close": turn["close"]},
            "momentum": {"state": mom["state"],
                         "net_atr": mom["net_atr"],
                         "efficiency": mom["efficiency"]},
        }
        context = self._context(
            hypothesis="impulse_pullback", trend_state=corr,
            terms=terms, extra=extra)
        zone = (min(pb["extreme_price"], turn["swing_price"]),
                max(pb["extreme_price"], turn["swing_price"]))
        return self._signal(
            index=m, timestamp=df.index[m], direction=d,
            score=score,
            factors=["trend_impulse_pullback", "impulse_leg",
                     "pullback_holds_structure",
                     "ltf_structure_turn"],
            entry_ref=geo["entry"], entry_zone=zone,
            invalidation=geo["stop"], targets=geo["targets"],
            atr=geo["atr"], context=context)


class TrendMomentumAccelerationPlaybook(_TrendMomentumBase):
    name = "trend_momentum_acceleration"
    description = (
        "Trend momentum acceleration: inside an established "
        "trend state (formation TF trend-phased, the "
        "formation->trigger pair ALIGNED — live "
        "participation), velocity rises through the 80th "
        "percentile of its own trailing distribution while "
        "closes accept beyond the last confirmed swing.  "
        "Entry at the formation close that confirms the "
        "break, read at the first trigger close after it; "
        "stop back beyond the broken swing; target the "
        "leg's measured projection."
    )
    context_description = (
        "Continuation, not breakout: the trend state must "
        "already exist (situation pair aligned) and the "
        "formation momentum must read sustained with "
        "velocity rising and >= the 80th percentile of its "
        "own last-100 distribution.  The acceptance is a "
        "close beyond the last confirmed formation swing "
        "within the last two formation bars, every close "
        "since holding beyond it."
    )

    def _detect_at(self, prep_f, prep_t, df, m, f_idx):
        # Fire ONCE per formation bar: at the first trigger
        # close whose visible formation bar is f_idx (the
        # formation bar has just closed; later trigger
        # bars reading the same formation bar are the same
        # moment, not new entries).
        if m > 0:
            prev_f = _tm.last_formation_bar(
                df, m - 1, self._formation_frame,
                formation_tf=self._formation_tf,
                trigger_tf=self._trigger_tf)
            if prev_f >= f_idx:
                return None
        acc = _tm.acceleration_at(prep_f, f_idx)
        if acc is None:
            return None
        if _tm.acceleration_at(prep_f, f_idx - 1) is not None:
            # A qualifying run fires on its FIRST bar only:
            # one acceleration event, one scenario, however
            # many bars the velocity tail keeps the read lit
            # (the break stays "fresh" for two bars by the
            # core's own window).
            return None
        d = int(acc["direction"])
        corr = self._trend_state(prep_f, f_idx, d,
                                 pairs_ok=_ACCEL_PAIRS)
        if corr is None:
            return None
        score = 63.0 + (4.0 if prep_f.body_dominated(
            acc["bar"], d) else 0.0)
        broken = acc["broken_swing"]["price"]
        terms = [
            f"{self._formation_tf} close back beyond the "
            f"broken swing {broken:.2f} = the continuation "
            "failed — acceptance was false (thesis exit, "
            "Pass 6 plan)",
            "velocity collapsing with formation momentum "
            "absorbed ends the acceleration premise",
        ]
        extra = {
            "acceleration": {
                "velocity": acc["velocity"],
                "velocity_percentile":
                    acc["velocity_percentile"],
                "broken_swing": broken,
                "broken_ts": _ts(self._formation_frame,
                                 acc["broken_swing"]["pos"]),
                "origin_price": acc["origin"]["price"],
                "leg": acc["leg"],
                "bar_ts": _ts(self._formation_frame,
                              acc["bar"])},
            "momentum": {"state": acc["momentum"]["state"],
                         "net_atr": acc["momentum"]["net_atr"],
                         "efficiency":
                             acc["momentum"]["efficiency"]},
        }
        context = self._context(
            hypothesis="momentum_acceleration",
            trend_state=corr, terms=terms, extra=extra)
        zone = (min(broken, acc["entry"]),
                max(broken, acc["entry"]))
        return self._signal(
            index=m, timestamp=df.index[m], direction=d,
            score=score,
            factors=["trend_momentum_acceleration",
                     "velocity_percentile_breakout",
                     "acceptance_beyond_swing",
                     "aligned_participation"],
            entry_ref=acc["entry"], entry_zone=zone,
            invalidation=acc["stop"], targets=acc["targets"],
            atr=acc["atr"], context=context)


TREND_PLAYBOOKS = (
    TrendImpulsePullbackPlaybook,
    TrendMomentumAccelerationPlaybook,
)
