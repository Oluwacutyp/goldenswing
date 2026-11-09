"""Auction / value family (MarketSituation Pass 7c,
SPEC §6.3) — the three hypotheses of the auction:

- ``auction_value_acceptance``: price probes beyond the
  developing value area and is ACCEPTED — closes hold
  outside the probed value while developing value
  migrates toward price (the auction re-pricing).
  Entry on the M15 retest that holds outside the probed
  edge; targets the next zone beyond (the situation's
  zones when threaded, else the value-height
  projection); stop beyond the probed edge.
- ``auction_value_rejection``: the probe is REJECTED —
  a close back inside the probed value before
  acceptance (a failed auction).  Entry at the first
  M15 close after the rejection, rotation toward the
  POC, then the opposite edge; stop beyond the probe's
  extreme.  The classic rotation-completion rate (the
  "80% rule") is carried as a STATED hypothesis this
  family's own shadow book must validate — it is never
  assumed in the detector (SPEC §6.3).
- ``auction_lvn_traversal``: price enters a low-volume
  node between two high-volume shelves → fast traversal
  to the far shelf.  Entry on the M15 close that crosses
  into the node; tight invalidation beyond the near
  shelf.

The measurement core is ``live/auction_value.py`` over
``live/developing_value.py`` — the §6.3 developing-value
extension the MarketSituation itself carries in
``location.value`` (one implementation of developing
value: the situation's current state and this family's
per-bar history are the same computation on the same
bars, so they cannot disagree).  The situation snapshot
the runner threads in contributes what the snapshot
owns: the situation_id stamp, the zones the acceptance
continuation targets, and the session VWAP context —
all recorded on the signal's context, source stated.

The runner threads two facts in per scan, in the
``set_h4_frame`` idiom:

- ``set_formation(frame, tf)`` — the formation clock
  (H1) and its frame (the scan itself runs on the
  TRIGGER clock, M15: the retest hold and the rejection
  read are M15 events);
- ``set_situation`` — the current MarketSituation
  snapshot.

``live = False``: the family's candidates are scored
and logged, never in the live P&L book — the scheduled
route is shadow (journal + scenario book only) until
the family's own promotion evaluation earns a separate
owner decision (SPEC §6 common contract).
"""
from __future__ import annotations

from ..backtest.harness import simulate_signal_v2
from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..live import auction_value as _av
from ..live import developing_value as _dv
from .base import Candidate, EngineContext, Playbook

FAMILY = "auction_value"

FORMATION_TF = "H1"
TRIGGER_TF = "M15"


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


class _AuctionValueBase(Playbook):
    """Shared plumbing for the three hypotheses.  The
    scan frame is the TRIGGER frame (M15); the formation
    frame (H1) + its clock are threaded per scan."""

    regime_scope = "both"
    session_scope = "any"
    hunting_ground = HuntingGround(excluded=set(), bonuses={})
    live = False
    live_threshold = 60.0

    def __init__(self) -> None:
        self._processed = -1  # last TRIGGER bar scanned
        # Threaded by the runner per scan (see the module
        # docstring).  Defaults are the bare/research path:
        # formation on the family's only clock, no
        # formation frame, no snapshot.
        self._formation_tf = FORMATION_TF
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
        return _av.TRIGGER_TF_DOWN[self._formation_tf]

    @property
    def _horizon(self) -> str:
        """SPEC §6.3: intraday, for every instance (the
        registry spec carries the same declaration; the
        family's sequences are day-bound by
        construction)."""
        return "intraday"

    # -- situation reads ---------------------------------------

    def _situation_id(self):
        snap = self._situation
        if isinstance(snap, dict):
            return snap.get("situation_id")
        return None

    def _next_zone(self, entry: float, direction: int):
        """The nearest situation zone beyond ``entry`` in
        ``direction`` (its near edge) — the acceptance
        continuation's first target when the snapshot is
        threaded.  None when no snapshot / no zone lies
        beyond (the caller falls back to the measured
        projection and says so)."""
        snap = self._situation
        if not isinstance(snap, dict):
            return None
        location = snap.get("location") or {}
        zones = location.get(
            "zones_above" if direction > 0 else "zones_below")
        best = None
        for z in zones or []:
            near = z.get("bottom") if direction > 0 \
                else z.get("top")
            if near is None:
                continue
            near = float(near)
            if (near - entry) * direction <= 0:
                continue
            if best is None or (near - best) * direction < 0:
                best = near
        return best

    # -- arming (the runner's detection journal) --------------

    def pending_state(self, df) -> dict | None:
        """The family's live ARM as of the trigger frame's
        last closed bar: the auction sequence's phase
        (probing / accepted / rejected / invalidated)
        with its geometry and the developing-value read
        attached.  None when no sequence is live or
        resolved-and-standing.  Pure.

        Corroboration: the sequence stands on a MEASURED
        developing value area (status ok — a sequence
        cannot exist otherwise), and when a snapshot is
        threaded its location block must carry a value
        read for the same trading day; a snapshot that
        cannot see value does not corroborate an arm
        (the scans' no_setup records carry the unarmed
        reads).  With no snapshot threaded the frame read
        stands alone (the research path)."""
        if df is None or self._formation_frame is None \
                or len(df) < 2:
            return None
        m = len(df) - 1
        prep = _av.PreparedAuction(self._formation_frame)
        f_idx = _av.last_formation_bar(
            df, m, self._formation_frame,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)
        if f_idx < 0:
            return None
        seq = _av.auction_sequence_at(prep, f_idx)
        if seq is None or seq["phase"] not in (
                _av.PHASE_PROBING, _av.PHASE_ACCEPTED,
                _av.PHASE_REJECTED, _av.PHASE_INVALIDATED):
            return None
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
            "phase": seq["phase"],
            "formation_tf": self._formation_tf,
            "trigger_tf": self._trigger_tf,
            "direction": seq["direction"],
            "probe": {
                "probe_ts": seq["probe_ts"],
                "extreme_price": seq["extreme_price"],
                "ref": dict(seq["ref"])},
            "resolution_ts": seq["resolution_ts"],
            "death_ts": seq["death_ts"],
            "value": _value_block(state),
            "corroborated": corroborated,
            "situation_id": self._situation_id(),
            "identity": "|".join([
                seq["identity"], seq["phase"]]),
        }

    # -- detection -------------------------------------------

    def _detect_at(self, prep, df, m: int,
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
        prep = _av.PreparedAuction(self._formation_frame)
        for m in range(start, end + 1):
            f_idx = _av.last_formation_bar(
                df, m, self._formation_frame,
                formation_tf=self._formation_tf,
                trigger_tf=self._trigger_tf)
            if f_idx < 0:
                continue
            sig = self._detect_at(prep, df, m, f_idx)
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

    def _context(self, *, hypothesis, seq, state, terms,
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
            # family's management clock is its formation
            # clock — the auction thesis is managed
            # against the value area it formed on (Pass
            # 6's own-clock rule).  Horizon is the
            # registry's intraday for every instance, so
            # no horizon key is carried (the 7b key exists
            # because that family's horizon VARIES per
            # instance; this one's cannot).
            "management_context_timeframe": formation_tf,
            "scenario_clocks": {
                "formation_tf": formation_tf,
                "refinement_tf": None,
                "trigger_tf": trigger_tf,
                "entry_tf": trigger_tf,
                "management_tf": formation_tf},
            "situation_id": self._situation_id(),
            "volume_basis": _dv.VOLUME_BASIS,
            "value": _value_block(state),
            "probe": {
                "probe_ts": seq["probe_ts"],
                "extreme_price": seq["extreme_price"],
                "ref": dict(seq["ref"]),
                "outside_closes": seq["outside_closes"]},
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


class AuctionValueAcceptancePlaybook(_AuctionValueBase):
    name = "auction_value_acceptance"
    description = (
        "Auction value acceptance: price probes beyond "
        "the developing value area and is ACCEPTED — at "
        "least two closes hold outside the probed value "
        "while developing value migrates toward the probe "
        "(the auction re-pricing).  Entry on the M15 "
        "retest that holds outside the probed edge; stop "
        "beyond the probed edge; target the next zone "
        "beyond, then the value-height projection."
    )
    context_description = (
        "Developing value from the shared extension "
        "(live/developing_value.py — the situation's own "
        "value layer, tick-volume basis, named): a probe "
        "is a close beyond the pre-probe VAH/VAL after a "
        "close inside it; acceptance is >= 2 closes "
        "holding beyond the probed edge with the POC or "
        "the probed edge migrated toward the probe beyond "
        "the migration dead-band.  The retest: an M15 bar "
        "that comes back within tolerance of the probed "
        "edge and closes outside it, before any M15 close "
        "back inside and while the accepted sequence is "
        "still the day's latest.  A formation close back "
        "inside with the POC no longer migrating "
        "invalidates the sequence."
    )

    def _detect_at(self, prep, df, m, f_idx):
        seq = _av.auction_sequence_at(prep, f_idx)
        if seq is None or seq["phase"] != _av.PHASE_ACCEPTED:
            return None
        d = int(seq["direction"])
        acc_bar = seq["resolution_bar"]
        if acc_bar is None:
            return None
        # The retest window opens at the first trigger
        # bar after the acceptance close.
        m0 = _av.first_bar_after(
            self._formation_frame, acc_bar, df,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)
        if m < m0 or m - m0 > _av.RETEST_MAX_TRIGGER_BARS:
            return None
        edge = float(seq["accept_edge"])
        atr_f = prep.atr(acc_bar)
        if not atr_f:
            return None
        tol = _av.RETEST_TOL_ATR * atr_f
        closes = df["close"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)

        def _touched(k: int) -> bool:
            # The bar's range came back to the probed
            # edge (within tolerance): from above for a
            # long acceptance (the low reaches down to
            # the edge), from below for a short.
            return lows[k] <= edge + tol if d > 0 \
                else highs[k] >= edge - tol

        # Any trigger close back inside the probed edge
        # before this bar voids the retest (the hold
        # failed before it formed); the FIRST touch-and-
        # hold bar is the entry — later bars are the
        # trade running, not new entries.
        for k in range(m0, m):
            if (closes[k] - edge) * d < 0:
                return None
            if _touched(k) and (closes[k] - edge) * d > 0:
                return None  # an earlier bar already held
        if not _touched(m) or (closes[m] - edge) * d <= 0:
            return None
        entry = float(closes[m])
        stop = edge - d * _av.STOP_BUFFER_ATR * atr_f
        zone = self._next_zone(entry, d)
        poc = float(seq["accept_poc"])
        height = abs(edge - poc)
        # Targets are LEVELS beyond the entry: a measured
        # projection the retest fill has already passed
        # is consumed, not a target — if every projection
        # is behind the entry the move is spent and there
        # is no signal (an honest abstention, not a stale
        # target list).
        projections = [edge + d * k * height
                       for k in (1.0, 2.0, 3.0)]
        projections = [t for t in projections
                       if (t - entry) * d > 0]
        if zone is not None:
            targets = [zone] + projections[:1]
            target_source = "situation_zone"
        else:
            targets = projections[:2]
            target_source = "measured_projection"
        if not targets:
            return None
        score = 64.0 + (4.0 if _av.body_dominated(
            float(df["open"].iloc[m]), float(highs[m]),
            float(lows[m]), entry, d) else 0.0)
        terms = [
            f"{self._formation_tf} close back inside the "
            f"probed edge {edge:.2f} with the POC no "
            "longer migrating toward the probe — the "
            "re-pricing did not hold (thesis exit, "
            "Pass 6 plan)",
            f"{self._trigger_tf} close back inside "
            f"{edge:.2f} before the retest voids the "
            "entry read",
        ]
        state = prep.value_at(acc_bar)
        context = self._context(
            hypothesis="value_acceptance", seq=seq,
            state=state, terms=terms,
            extra={
                "acceptance": {
                    "bar_ts": seq["resolution_ts"],
                    "edge": edge,
                    "poc_at_acceptance": poc,
                    "outside_closes": seq["outside_closes"]},
                "retest": {
                    "bar_ts": _ts(df, m),
                    "edge": edge,
                    "tolerance": tol},
                "target_source": target_source})
        return self._signal(
            index=m, timestamp=df.index[m], direction=d,
            score=score,
            factors=["auction_value_acceptance",
                     "probe_accepted", "value_migrated",
                     "retest_held_outside"],
            entry_ref=entry,
            entry_zone=(min(edge, entry), max(edge, entry)),
            invalidation=stop, targets=targets,
            atr=atr_f, context=context)


class AuctionValueRejectionPlaybook(_AuctionValueBase):
    name = "auction_value_rejection"
    description = (
        "Auction value rejection (the failed auction): a "
        "probe beyond the developing value area is "
        "REJECTED — a close back inside the probed value "
        "before acceptance forms.  Entry at the first "
        "M15 close after the rejection close, while price "
        "is still inside value; rotation targets the POC, "
        "then the opposite edge; stop beyond the probe's "
        "extreme."
    )
    context_description = (
        "The probe and its reference value are the "
        "acceptance hypothesis's own reads (one sequence "
        "machine, two resolutions — the pair cannot "
        "double-fire: acceptance needs two outside closes "
        "plus migration, rejection is the first close "
        "back inside before that).  A formation close "
        "beyond the probe extreme after rejection means "
        "the auction re-accepted the probe's ground — "
        "the rotation thesis is invalidated.  The "
        "rotation-completion rate is a stated hypothesis "
        "for this family's shadow book to validate, "
        "never an assumption in the detector."
    )

    def _detect_at(self, prep, df, m, f_idx):
        seq = _av.auction_sequence_at(prep, f_idx)
        if seq is None or seq["phase"] != _av.PHASE_REJECTED:
            return None
        d = int(seq["direction"])
        rej_bar = seq["resolution_bar"]
        if rej_bar is None:
            return None
        # Entry is read at the FIRST trigger close after
        # the rejection formation close (the trend
        # family's acceleration arrangement: the formation
        # close is the fact, the first trigger close
        # after it is the entry read) — and only while
        # price is still inside the rejection bar's value
        # (rotation ground; a first read already beyond
        # the POC has missed the rotation's premise).
        m0 = _av.first_bar_after(
            self._formation_frame, rej_bar, df,
            formation_tf=self._formation_tf,
            trigger_tf=self._trigger_tf)
        if m != m0:
            return None
        state = prep.value_at(rej_bar)
        if state.get("status") != "ok":
            return None
        entry = float(df["close"].iloc[m])
        if not (state["val"] <= entry <= state["vah"]):
            return None
        atr_f = prep.atr(rej_bar)
        if not atr_f:
            return None
        extreme = float(seq["extreme_price"])
        stop = extreme + d * _av.STOP_BUFFER_ATR * atr_f
        # Rotation: against the probe's direction — POC
        # first, then the opposite edge of the probed
        # value.
        poc = float(state["poc"])
        far_edge = float(seq["ref"]["val"] if d > 0
                         else seq["ref"]["vah"])
        targets = [poc, far_edge]
        score = 63.0 + (4.0 if _av.body_dominated(
            float(df["open"].iloc[m]),
            float(df["high"].iloc[m]),
            float(df["low"].iloc[m]), entry, -d) else 0.0)
        terms = [
            f"{self._formation_tf} close beyond the "
            f"probe extreme {extreme:.2f} — the auction "
            "re-accepted the probe's ground; the rotation "
            "thesis is dead (thesis exit, Pass 6 plan)",
        ]
        context = self._context(
            hypothesis="value_rejection_rotation", seq=seq,
            state=state, terms=terms,
            extra={
                "rejection": {
                    "bar_ts": seq["resolution_ts"],
                    "extreme_price": extreme,
                    "poc_at_rejection": poc},
                "rotation": {
                    "first_target": "poc",
                    "second_target": "opposite_edge",
                    "completion_rate": "stated hypothesis "
                                       "— validated by "
                                       "this family's "
                                       "shadow book, "
                                       "never assumed"}})
        return self._signal(
            index=m, timestamp=df.index[m], direction=-d,
            score=score,
            factors=["auction_value_rejection",
                     "failed_auction", "probe_rejected",
                     "rotation_to_value"],
            entry_ref=entry,
            entry_zone=(min(entry, poc), max(entry, poc)),
            invalidation=stop, targets=targets,
            atr=atr_f, context=context)


class AuctionLVNTraversalPlaybook(_AuctionValueBase):
    name = "auction_lvn_traversal"
    description = (
        "Auction LVN traversal: price enters a low-volume "
        "node between two high-volume shelves (the "
        "developing profile's thin air) → fast traversal "
        "to the far shelf.  Entry on the M15 close that "
        "crosses into the node; stop beyond the near "
        "shelf (tight invalidation at the near node); "
        "target the far shelf."
    )
    context_description = (
        "Nodes are the developing profile's own (local "
        "extrema of the tick-volume histogram, plateau-"
        "merged): an LVN is a local-minimum bin at or "
        "below the stated share of the POC bin, flanked "
        "by HVN shelves on both sides.  The crossing is "
        "an M15 close entering the node's span from "
        "outside it, read against the formation value "
        "state at the last closed H1 bar — a close that "
        "jumps the whole span is the traversal already "
        "done and is never chased."
    )

    def _detect_at(self, prep, df, m, f_idx):
        if m < 1:
            return None
        prev_close = float(df["close"].iloc[m - 1])
        cross = _av.lvn_crossing_at(
            prep, f_idx, prev_close,
            float(df["close"].iloc[m]))
        if cross is None:
            return None
        # One traversal per contiguous visit to the thin
        # region: adjacent LVN bins between the same two
        # shelves are ONE thin region split by binning,
        # so a descent that crosses two of them in
        # consecutive bars is one event, not two entries.
        # The previous close must have been OUTSIDE every
        # LVN span (in a shelf / thick region, or outside
        # value entirely) for this crossing to be a new
        # traversal.
        for other in cross["value"].get("lvns") or []:
            if other is not cross["lvn"] and \
                    other["low"] <= prev_close <= other["high"]:
                return None
        d = int(cross["direction"])
        lvn = cross["lvn"]
        atr_f = prep.atr(f_idx)
        if not atr_f:
            return None
        entry = float(df["close"].iloc[m])
        near = float(lvn["below_hvn"] if d > 0
                     else lvn["above_hvn"])
        far = float(lvn["above_hvn"] if d > 0
                    else lvn["below_hvn"])
        stop = near - d * _av.STOP_BUFFER_ATR * atr_f
        score = 62.0 + (4.0 if _av.body_dominated(
            float(df["open"].iloc[m]),
            float(df["high"].iloc[m]),
            float(df["low"].iloc[m]), entry, d) else 0.0)
        terms = [
            f"{self._trigger_tf} close back beyond the "
            f"near shelf {near:.2f} — the node rejected "
            "the traversal at its near node (thesis "
            "exit, Pass 6 plan)",
        ]
        state = cross["value"]
        context = self._context(
            hypothesis="lvn_traversal",
            seq={"probe_ts": _ts(df, m), "extreme_price":
                 entry, "ref": {
                     "poc": state["poc"], "vah": state["vah"],
                     "val": state["val"],
                     "day_key": state.get("day_key"),
                     "bars": state.get("bars")},
                 "outside_closes": 0},
            state=state, terms=terms,
            extra={
                "lvn": {"price": lvn["price"],
                        "low": lvn["low"],
                        "high": lvn["high"],
                        "near_hvn": near,
                        "far_hvn": far}})
        return self._signal(
            index=m, timestamp=df.index[m], direction=d,
            score=score,
            factors=["auction_lvn_traversal",
                     "thin_value_node", "shelf_to_shelf"],
            entry_ref=entry,
            entry_zone=(lvn["low"], lvn["high"]),
            invalidation=stop, targets=[far],
            atr=atr_f, context=context)


AUCTION_PLAYBOOKS = (
    AuctionValueAcceptancePlaybook,
    AuctionValueRejectionPlaybook,
    AuctionLVNTraversalPlaybook,
)
