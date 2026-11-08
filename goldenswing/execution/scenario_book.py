"""Scenario book (MarketSituation program, Pass 2 — SPEC §4).

The book is the union of the two machines it generalises:

- ``judgment_thesis.ThesisKeeper`` — ONE market thesis with
  versions, children, invalidation terms evaluated as data and
  an atomic state snapshot beside the journal (WS4 pattern);
- ``setup_lifecycle.SetupLifecycleStore`` — MANY persisted
  setups with stable identity (``candidate_lifecycle.
  instance_key`` lineage) and journaled transitions.

A SCENARIO is the thesis superset (SPEC §4.1): one hypothesis —
from a playbook candidate, a §G shadow scan, or the S2 market
thesis itself — carrying its family, direction, horizon, its
OWN clocks (formation / refinement / trigger / entry /
management), geometry, invalidation terms, validity window,
evidence groups, a parent link, a version history, and — once
its trade is open — its own frozen management context.

The book holds MANY scenarios at once: long and short, same or
different timeframes, parented and independent (SPEC §4.3).
Opposing scenarios coexist; the only pair the book singles out
is the COMPETING pair (same horizon, same region, opposite
directions), which it MARKS — resolving competing pairs is
Judgment v2's job (Pass 3), never the book's.

SHADOW DISCIPLINE (Pass 2): the book proposes, relates, scores
and journals BESIDE the live paths.  Judgment's governing
decision path is untouched: governed routes still read the
legacy thesis path, the lifecycle store still owns setup
transitions (the book MIRRORS them — it never originates one),
and no order path reads the book.  Scores journaled here are
the legacy machinery's own outputs — the five-scenario
assessor's probabilities (``judgment_thesis.assess_market``)
mapped onto each scenario's side, and the S1 verdict
(``judgment.evaluate_judgment``) of the scenario's originating
candidate, attached by the runner from the funnel's own tap.
Nothing is re-derived and nothing is invented: a scenario whose
candidate was never S1-evaluated carries ``s1 = None``.

Persistence is the SetupLifecycleStore pattern: one atomic JSON
snapshot beside the journal (tmp + fsync + rename), written on
every mutation.  A state file that cannot be trusted raises
``ScenarioBookStateError`` — the runner disables the book for
the run with a loud note rather than starting from a
guessed-empty book while a state file exists (the WS4 rule).
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .judgment_thesis import (
    DEFAULT_RECOVER_BAND,
    DEFAULT_WEAKEN_DROP,
)
from .playbook_registry import spec_for
from .setup_lifecycle import (
    STAGE_ARMED,
    STAGE_DETECTED,
    STAGE_EVALUATED,
    STAGE_EXPIRED,
    STAGE_FILLED,
    STAGE_INVALIDATED,
    STAGE_REJECTED,
    STAGE_SUBMITTED,
    STAGE_TRIGGER_FIRED,
    STAGE_TRIGGER_OBSERVED,
    TERMINAL_STAGES,
    TrackedSetup,
)

#: Durable book state envelope (the ThesisStateStore rule: a
#: file that is not exactly this kind/version is not book state
#: and is never guessed at).
STATE_KIND = "scenario_book_state"
STATE_VERSION = 1
STATE_KEYS = ("seq", "scenarios", "closed", "relationships")

#: Book states (SPEC §4.1).
SCENARIO_STATES = ("proposed", "forming", "armed", "triggered",
                   "entered", "managing", "closed")
#: States in which a scenario is a live hypothesis.
LIVE_STATES = frozenset({"proposed", "forming", "armed",
                         "triggered", "entered", "managing"})
#: States that count as OPEN EXPOSURE for the cycle record
#: (SPEC §4.3: the book reports, it grants no capacity).
POSITION_STATES = frozenset({"entered", "managing"})

#: Lifecycle stage -> book state.  Stages absent here are
#: journal-only dispositions (they extend the scenario's
#: history without moving its state): execution_evaluated and
#: order_submitted happen INSIDE "triggered"; trigger_observed
#: is the WS3 shadow marker, not a stage at all.
STAGE_TO_STATE = {
    STAGE_DETECTED: "proposed",
    STAGE_ARMED: "armed",
    STAGE_TRIGGER_FIRED: "triggered",
    STAGE_FILLED: "entered",
}
#: Terminal lifecycle stage -> close_reason (SPEC §4.1's enum,
#: extended with the lifecycle's own "rejected" — §4.1 adopts
#: the lifecycle vocabulary wholesale, and "rejected" is one of
#: its terminal names).
STAGE_TO_CLOSE_REASON = {
    STAGE_FILLED: None,  # a fill ENTERS; it does not close
    STAGE_REJECTED: "rejected",
    STAGE_EXPIRED: "expired",
    STAGE_INVALIDATED: "invalidated",
}

#: The S2 market thesis mirrored as a book scenario: its family
#: name and declared clocks (the thesis cycle is H1-driven; its
#: plan's management horizon is a policy id, not a horizon word,
#: so the scenario declares the cycle's own horizon).
THESIS_FAMILY = "judgment_thesis"
THESIS_HORIZON = "intraday"
THESIS_CLOCKS = {"formation_tf": "H1", "refinement_tf": None,
                 "trigger_tf": "H1", "entry_tf": "H1",
                 "management_tf": "H1"}
#: Thesis close status -> book close_reason.
THESIS_CLOSE_REASONS = {"invalidated": "invalidated",
                        "expired": "expired",
                        "superseded": "superseded"}


class ScenarioBookStateError(RuntimeError):
    """Book state cannot be safely read, written or trusted.

    Raised with the specific gap named — the runner journals it
    and disables the book for the run rather than starting from
    a guessed-empty book."""


class ScenarioBookStore:
    """Durable book state: one atomic JSON snapshot beside the
    journal — the SetupLifecycleStore / ThesisStateStore
    pattern (tmp write + fsync + rename, so a crash mid-write
    never leaves a torn file as the state of record).  The book
    writes on every mutation; this class owns only the file and
    its envelope."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, state: dict) -> None:
        payload = dict(state)
        payload["saved_ts"] = time.time()
        try:
            text = json.dumps(payload, default=str)
        except (TypeError, ValueError) as exc:
            raise ScenarioBookStateError(
                f"book state is not serializable: {exc}"
            ) from exc
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(
                self.path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
            try:  # the rename itself durable, where supported
                dfd = os.open(str(self.path.parent), os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
        except OSError as exc:
            raise ScenarioBookStateError(
                f"book state write failed: {exc}") from exc

    def load(self) -> dict:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ScenarioBookStateError(
                f"book state unreadable: {exc}") from exc
        try:
            state = json.loads(raw)
        except ValueError as exc:
            raise ScenarioBookStateError(
                f"book state is torn (invalid JSON: {exc})"
            ) from exc
        if not isinstance(state, dict):
            raise ScenarioBookStateError(
                "book state is not a state object "
                f"(got {type(state).__name__})")
        if state.get("kind") != STATE_KIND:
            raise ScenarioBookStateError(
                f"book state kind is {state.get('kind')!r}, "
                f"expected {STATE_KIND!r}")
        if state.get("version") != STATE_VERSION:
            raise ScenarioBookStateError(
                f"book state version is "
                f"{state.get('version')!r}, expected "
                f"{STATE_VERSION}")
        missing = [k for k in STATE_KEYS if k not in state]
        if missing:
            raise ScenarioBookStateError(
                "book state is missing keys: "
                + ", ".join(missing))
        return state


@dataclass(frozen=True)
class ScenarioBookConfig:
    """Standing discipline, mirroring the thesis keeper's
    stated constants: a scenario whose assessed standing erodes
    this far from its peak is WEAKENED (journaled, new version,
    scenario stands); recovery to within the band of the peak
    clears the weakened state."""
    weaken_drop: float = DEFAULT_WEAKEN_DROP
    recover_band: float = DEFAULT_RECOVER_BAND


def _span(scenario: dict) -> tuple[float, float] | None:
    """The scenario's price span [lo, hi] over entry, stop and
    targets — the region its trade occupies.  Incomplete
    geometry -> None (the pair is independent unless a parent
    link declares otherwise; absent stays absent)."""
    geo = scenario.get("geometry") or {}
    points = [geo.get("entry"), geo.get("stop")]
    points += list(geo.get("targets") or [])
    vals = []
    for raw in points:
        try:
            if raw is not None:
                vals.append(float(raw))
        except (TypeError, ValueError):
            continue
    if len(vals) < 2:
        return None
    return (min(vals), max(vals))


def _spans_overlap(a: tuple[float, float],
                   b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def pair_relationship(a: dict, b: dict) -> tuple[str, dict]:
    """The deterministic relationship between two live
    scenarios (SPEC §4.2(b)), with the evidence fields that
    decided it.  Priority: a DECLARED parent link outranks
    geometry; competing (same horizon + same region + opposite
    directions) outranks conflicting (opposite directions,
    shared region, different horizons)."""
    span_a, span_b = _span(a), _span(b)
    overlap = bool(span_a and span_b
                   and _spans_overlap(span_a, span_b))
    zone_a = (a.get("geometry") or {}).get("zone_id")
    zone_b = (b.get("geometry") or {}).get("zone_id")
    shared_zone = bool(zone_a) and zone_a == zone_b
    region = overlap or shared_zone
    dir_a, dir_b = int(a.get("direction") or 0), \
        int(b.get("direction") or 0)
    evidence = {
        "span_a": list(span_a) if span_a else None,
        "span_b": list(span_b) if span_b else None,
        "overlap": overlap,
        "shared_zone": shared_zone,
        "directions": [dir_a, dir_b],
        "horizons": [a.get("horizon"), b.get("horizon")],
    }
    if a.get("parent_scenario_id") == b.get("scenario_id") \
            or b.get("parent_scenario_id") == a.get("scenario_id"):
        return "parent_child", evidence
    if dir_a and dir_b and dir_a == -dir_b:
        if not region:
            return "independent", evidence
        if a.get("horizon") == b.get("horizon"):
            return "competing", evidence
        return "conflicting", evidence
    if dir_a and dir_a == dir_b:
        if not region:
            return "independent", evidence
        entry_b = (b.get("geometry") or {}).get("entry")
        entry_a = (a.get("geometry") or {}).get("entry")
        clock_a = (a.get("clocks") or {}).get("formation_tf")
        clock_b = (b.get("clocks") or {}).get("formation_tf")
        nested = False
        try:
            if span_a and entry_b is not None \
                    and span_a[0] <= float(entry_b) <= span_a[1] \
                    and clock_a != clock_b:
                nested = True
            if span_b and entry_a is not None \
                    and span_b[0] <= float(entry_a) <= span_b[1] \
                    and clock_a != clock_b:
                nested = True
        except (TypeError, ValueError):
            nested = False
        return ("nested" if nested else "aligned"), evidence
    return "independent", evidence


@dataclass
class ScenarioBook:
    """The generalized keeper: many concurrent scenarios,
    each with its own clocks, related to each other, journaled
    end to end.  Methods return the journal records they
    produced; the runner appends them verbatim (the
    ThesisKeeper contract).  The book never originates a
    lifecycle transition and never reads a decision path — it
    mirrors, relates, scores and reports."""

    store: ScenarioBookStore | None = field(
        default=None, repr=False, compare=False)
    config: ScenarioBookConfig = field(
        default_factory=ScenarioBookConfig)
    scenarios: dict = field(default_factory=dict)
    closed: list = field(default_factory=list)
    relationships: dict = field(default_factory=dict)
    _seq: int = 0

    def __post_init__(self):
        if self.store is not None and self.store.exists():
            self.restore(self.store.load())

    # -- persistence -------------------------------------------

    def snapshot(self) -> dict:
        return {
            "kind": STATE_KIND,
            "version": STATE_VERSION,
            "seq": self._seq,
            "scenarios": copy.deepcopy(self.scenarios),
            "closed": copy.deepcopy(self.closed),
            "relationships": copy.deepcopy(self.relationships),
        }

    @staticmethod
    def _validate_state(state: dict) -> None:
        def gap(msg: str) -> None:
            raise ScenarioBookStateError(
                f"book state invalid: {msg}")

        seq = state.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) \
                or seq < 0:
            gap(f"'seq' is not a non-negative int (got {seq!r})")
        scenarios = state.get("scenarios")
        if not isinstance(scenarios, dict):
            gap("'scenarios' is not a mapping")
        for sid, sc in scenarios.items():
            if not isinstance(sc, dict) \
                    or sc.get("scenario_id") != sid:
                gap(f"scenario {sid!r} is not a record keyed "
                    "by its own id")
            for key in ("family", "direction", "state",
                        "clocks", "geometry", "version",
                        "history", "standing"):
                if key not in sc:
                    gap(f"scenario {sid!r} missing '{key}'")
            if sc.get("state") not in LIVE_STATES:
                gap(f"scenario {sid!r} state is "
                    f"{sc.get('state')!r}, not a live state")
        closed = state.get("closed")
        if not isinstance(closed, list) \
                or any(not isinstance(c, dict)
                       or not c.get("scenario_id")
                       for c in closed):
            gap("'closed' is not a list of scenario records")
        if not isinstance(state.get("relationships"), dict):
            gap("'relationships' is not a mapping")

    def restore(self, state: dict) -> dict:
        """Rebuild the book from a validated snapshot.  A
        closed scenario restores as HISTORY — it never comes
        back live — and live scenarios restore with their
        versions, standing and frozen management contexts
        intact."""
        self._validate_state(state)
        self._seq = int(state["seq"])
        self.scenarios = copy.deepcopy(state["scenarios"])
        self.closed = copy.deepcopy(state["closed"])
        self.relationships = copy.deepcopy(
            state["relationships"])
        return {
            "seq": self._seq,
            "live": len(self.scenarios),
            "closed_total": len(self.closed),
            "live_ids": sorted(self.scenarios),
            "state_saved_ts": state.get("saved_ts"),
        }

    def _persist(self) -> None:
        if self.store is not None:
            self.store.save(self.snapshot())

    # -- lookups -------------------------------------------------

    def get(self, scenario_id: str) -> dict | None:
        sc = self.scenarios.get(scenario_id)
        return copy.deepcopy(sc) if sc is not None else None

    def live(self) -> list[dict]:
        return [copy.deepcopy(self.scenarios[sid])
                for sid in sorted(self.scenarios)]

    # -- internals ------------------------------------------------

    def _bump(self, scenario: dict, ts, kind: str,
              detail: str = "", **extra) -> None:
        """One material change: a new version + a history entry
        (prior versions' facts are never overwritten — the
        history IS the version chain)."""
        scenario["version"] = int(scenario.get("version") or 1) + 1
        scenario["updated_ts"] = ts
        entry = {"version": scenario["version"], "ts": ts,
                 "kind": kind, "detail": detail}
        entry.update(extra)
        scenario["history"].append(entry)

    def _transition_record(self, scenario: dict, *, ts,
                           from_state: str, stage: str,
                           detail: str,
                           situation_id) -> dict:
        return {
            "type": "scenario_transition",
            "ts": ts,
            "scenario_id": scenario["scenario_id"],
            "family": scenario.get("family"),
            "from_state": from_state,
            "to_state": scenario["state"],
            "close_reason": scenario.get("close_reason"),
            "stage": stage,
            "detail": detail,
            "version": scenario["version"],
            "situation_id": situation_id,
        }

    def _close(self, scenario: dict, *, ts, close_reason: str,
               stage: str, detail: str,
               situation_id) -> dict:
        from_state = scenario["state"]
        scenario["state"] = "closed"
        scenario["close_reason"] = close_reason
        scenario["closed_ts"] = ts
        self._bump(scenario, ts, "closed",
                   detail or close_reason, stage=stage)
        self.closed.append(scenario)
        del self.scenarios[scenario["scenario_id"]]
        return self._transition_record(
            scenario, ts=ts, from_state=from_state, stage=stage,
            detail=detail, situation_id=situation_id)

    # -- proposals (the deterministic bridge) ---------------------

    def _clocks_for_setup(self, setup: TrackedSetup) -> dict:
        spec = spec_for(setup.playbook)
        mgmt_tf = (setup.management or {}).get(
            "management_context_timeframe") or None
        clocks = {
            "formation_tf": (spec.native_tf if spec is not None
                             else mgmt_tf or setup.trigger_tf),
            "refinement_tf": (spec.refinement_tf
                              if spec is not None else None),
            "trigger_tf": setup.trigger_tf,
            "entry_tf": setup.trigger_tf,
            "management_tf": mgmt_tf or setup.trigger_tf,
        }
        if spec is not None and spec.clocks:
            clocks.update(dict(spec.clocks))
        return clocks

    def propose_from_setup(self, setup: TrackedSetup, *, ts,
                           situation_id=None,
                           authority: str = "shadow",
                           evidence_groups=None, s1=None
                           ) -> list[dict]:
        """A tracked setup becomes a book scenario (identity =
        the setup's instance key — the same setup is the same
        scenario across rescans and restarts; re-proposal of a
        live scenario is a refresh, never a duplicate)."""
        existing = self.scenarios.get(setup.setup_id)
        if existing is not None:
            return self._refresh_from_setup(
                existing, setup, ts=ts,
                situation_id=situation_id)
        if any(c.get("scenario_id") == setup.setup_id
               for c in self.closed):
            # A terminal life stays as the audit of the first
            # life (the lifecycle store's own rule).
            return []
        spec = spec_for(setup.playbook)
        self._seq += 1
        scenario = {
            "scenario_id": setup.setup_id,
            "family": setup.playbook,
            "route": setup.route,
            "authority": authority,
            "direction": int(setup.direction),
            "horizon": (spec.management_horizon
                        if spec is not None else "intraday"),
            "clocks": self._clocks_for_setup(setup),
            "state": "proposed",
            "close_reason": None,
            "geometry": {
                "entry": float(setup.entry_ref),
                "stop": float(setup.invalidation),
                "targets": [float(x) for x in setup.targets],
                "zone_id": None},
            "invalidation": {
                "price": float(setup.invalidation),
                "terms": [f"close beyond invalidation "
                          f"{float(setup.invalidation):.2f} "
                          f"(setup lifecycle upkeep)"]},
            "validity": {
                "formed_bar_ts": setup.formation_ts,
                "touch_ts": setup.touch_ts,
                "expires_ts": setup.expiry_ts,
                "validity_bars": int(setup.validity_bars)},
            "formed_situation_id": situation_id,
            "parent_scenario_id": None,
            "thesis_id": setup.thesis_id,
            "thesis_version": setup.thesis_version,
            "evidence_groups": list(evidence_groups or []),
            "counter_evidence_groups": [],
            "management": copy.deepcopy(setup.management or {}),
            "management_context": None,
            "ticket": (int(setup.ticket)
                       if setup.ticket is not None else None),
            "alert_id": setup.alert_id,
            "setup_stage": setup.stage,
            "score": float(setup.score or 0.0),
            "standing": {"p": None, "peak_p": None,
                         "state": "intact"},
            "s1": copy.deepcopy(s1) if s1 is not None else None,
            "version": 1,
            "history": [{"version": 1, "ts": ts,
                         "kind": "proposed",
                         "detail": "scenario proposed from "
                                   "tracked setup"}],
            "relationships": [],
            "created_ts": ts,
            "updated_ts": ts,
        }
        self.scenarios[setup.setup_id] = scenario
        self._resolve_parent(scenario)
        records = [self._transition_record(
            scenario, ts=ts, from_state=None,
            stage=STAGE_DETECTED,
            detail="scenario proposed from tracked setup",
            situation_id=situation_id)]
        self._persist()
        return records

    def _resolve_parent(self, scenario: dict) -> None:
        """Parent identity: a scenario born under the live S2
        thesis (its setup carries the WS3 thesis link) parents
        to the thesis's book scenario when the book holds it."""
        thesis_id = scenario.get("thesis_id")
        if thesis_id and thesis_id in self.scenarios:
            scenario["parent_scenario_id"] = thesis_id
        elif scenario.get("parent_scenario_id") \
                and scenario["parent_scenario_id"] \
                not in self.scenarios \
                and not any(
                    c.get("scenario_id")
                    == scenario["parent_scenario_id"]
                    for c in self.closed):
            scenario["parent_scenario_id"] = None

    def _refresh_from_setup(self, scenario: dict,
                            setup: TrackedSetup, *, ts,
                            situation_id=None) -> list[dict]:
        changed = False
        if scenario.get("thesis_id") != setup.thesis_id:
            scenario["thesis_id"] = setup.thesis_id
            scenario["thesis_version"] = setup.thesis_version
            self._resolve_parent(scenario)
            changed = True
        if setup.ticket is not None and scenario.get("ticket") \
                != int(setup.ticket):
            scenario["ticket"] = int(setup.ticket)
            changed = True
        if setup.alert_id is not None \
                and scenario.get("alert_id") != setup.alert_id:
            scenario["alert_id"] = setup.alert_id
            changed = True
        if changed:
            self._bump(scenario, ts, "refreshed",
                       "setup facts refreshed (thesis link / "
                       "ticket)")
            self._persist()
        return []

    def sync_setup(self, setup: TrackedSetup, *, ts,
                   stage: str | None = None, detail: str = "",
                   situation_id=None) -> list[dict]:
        """Mirror one lifecycle transition (or fact refresh)
        into the book.  The lifecycle store owns the transition;
        the book follows it — a stage the store refused (unknown
        / already terminal) never moves the book either, because
        the runner calls this only with the store's accepted
        event."""
        scenario = self.scenarios.get(setup.setup_id)
        if scenario is None:
            return self.propose_from_setup(
                setup, ts=ts, situation_id=situation_id)
        records = self._refresh_from_setup(
            scenario, setup, ts=ts, situation_id=situation_id)
        stage = stage or setup.stage
        scenario["setup_stage"] = stage
        if stage in TERMINAL_STAGES:
            reason = STAGE_TO_CLOSE_REASON.get(stage)
            if reason is not None:
                records.append(self._close(
                    scenario, ts=ts, close_reason=reason,
                    stage=stage, detail=detail,
                    situation_id=situation_id))
                self._persist()
                return records
        target = STAGE_TO_STATE.get(stage)
        if target is not None and target != scenario["state"]:
            from_state = scenario["state"]
            scenario["state"] = target
            if target == "entered":
                self._freeze_management_context(scenario, ts)
            self._bump(scenario, ts, "transition", detail,
                       stage=stage)
            records.append(self._transition_record(
                scenario, ts=ts, from_state=from_state,
                stage=stage, detail=detail,
                situation_id=situation_id))
            self._persist()
        elif stage not in STAGE_TO_STATE:
            # Journal-only disposition (evaluated / submitted /
            # trigger observed): history carries it, state
            # stands.
            scenario["history"].append(
                {"version": scenario["version"], "ts": ts,
                 "kind": "stage", "stage": stage,
                 "detail": detail})
            scenario["updated_ts"] = ts
            self._persist()
        return records

    def _freeze_management_context(self, scenario: dict,
                                   ts) -> None:
        """At entry the scenario's management context becomes
        ITS OWN: a frozen copy of the declared management block
        plus the fill facts.  From here on nothing outside this
        scenario — a thesis flip, a rival scenario, a book
        cycle — may rewrite it (SPEC §4.1 / A2 capability 7);
        only the scenario's own state advances."""
        if scenario.get("management_context") is not None:
            return
        scenario["management_context"] = {
            **copy.deepcopy(scenario.get("management") or {}),
            "ticket": scenario.get("ticket"),
            "alert_id": scenario.get("alert_id"),
            "entry": (scenario.get("geometry") or {})
            .get("entry"),
            "frozen_ts": ts,
            "frozen_from_scenario_version":
                scenario.get("version"),
        }

    def note_judgment(self, scenario_id: str, *, ts, s1=None,
                      groups_for=None, groups_against=None
                      ) -> list[dict]:
        """Attach the funnel's own S1 verdict for this
        scenario's originating candidate (the runner's
        judgment tap) plus the correlation groups S1's weighing
        assigned to each side.  Enrichment, journaled in the
        scenario's history — never a state change."""
        scenario = self.scenarios.get(scenario_id)
        if scenario is None:
            return []
        if s1 is not None:
            scenario["s1"] = copy.deepcopy(s1)
        if groups_for is not None:
            scenario["evidence_groups"] = list(groups_for)
        if groups_against is not None:
            scenario["counter_evidence_groups"] = list(
                groups_against)
        self._bump(scenario, ts, "judgment_noted",
                   "S1 verdict + evidence groups attached from "
                   "the funnel tap")
        self._persist()
        return []

    # -- the S2 thesis, mirrored -----------------------------------

    def sync_thesis(self, active: dict | None, *, ts,
                    closed_theses=(),
                    situation_id=None) -> list[dict]:
        """Mirror the ThesisKeeper's live thesis as a book
        scenario and close the mirrors of theses that closed
        this cycle.  The keeper stays the authority for the
        thesis lifecycle (its order-boundary gate is untouched);
        the book's mirror is how OTHER scenarios parent to it
        and how the book's relationships see it."""
        records: list[dict] = []
        for thesis in closed_theses or ():
            scenario = self.scenarios.get(thesis.get("id"))
            if scenario is None:
                continue
            reason = THESIS_CLOSE_REASONS.get(
                thesis.get("status"), "invalidated")
            records.append(self._close(
                scenario, ts=ts, close_reason=reason,
                stage=f"thesis_{thesis.get('status')}",
                detail=str(thesis.get("close_reason") or ""),
                situation_id=situation_id))
        if active is not None:
            tid = active.get("id")
            scenario = self.scenarios.get(tid)
            if scenario is None:
                self._seq += 1
                plan = active.get("plan") or {}
                scenario = {
                    "scenario_id": tid,
                    "family": THESIS_FAMILY,
                    "route": "judgment_thesis",
                    "authority": "shadow",
                    "direction": int(active.get("direction")
                                     or 0),
                    "horizon": THESIS_HORIZON,
                    "clocks": dict(THESIS_CLOCKS),
                    "state": "forming",
                    "close_reason": None,
                    "geometry": {
                        "entry": plan.get("entry"),
                        "stop": plan.get("invalidation_price"),
                        "targets": ([plan.get("target")]
                                    if plan.get("target")
                                    is not None else []),
                        "zone_id": None},
                    "invalidation": {
                        "price": plan.get("invalidation_price"),
                        "terms": [c.get("text") for c in
                                  active.get("invalidation")
                                  or []]},
                    "validity": {
                        "formed_bar_ts": None,
                        "touch_ts": None,
                        "expires_ts": None,
                        "validity_bars": None},
                    "formed_situation_id": situation_id,
                    "parent_scenario_id":
                        active.get("supersedes"),
                    "thesis_id": tid,
                    "thesis_scenario":
                        active.get("scenario"),
                    "thesis_version":
                        active.get("version"),
                    "evidence_groups": [],
                    "counter_evidence_groups": [],
                    "management": {},
                    "management_context": None,
                    "ticket": None,
                    "alert_id": None,
                    "setup_stage": None,
                    "score": 0.0,
                    "standing": {
                        "p": active.get("winner_p"),
                        "peak_p": active.get("peak_p"),
                        "state": active.get("state")
                        or "intact"},
                    "s1": None,
                    "version": int(active.get("version") or 1),
                    "history": [{
                        "version": int(active.get("version")
                                       or 1),
                        "ts": ts, "kind": "proposed",
                        "detail": "S2 market thesis mirrored "
                                  "into the book"}],
                    "relationships": [],
                    "created_ts": ts,
                    "updated_ts": ts,
                }
                self.scenarios[tid] = scenario
                records.append(self._transition_record(
                    scenario, ts=ts, from_state=None,
                    stage="thesis_created",
                    detail="S2 market thesis mirrored into "
                           "the book",
                    situation_id=situation_id))
                # Children proposed before their parent was
                # mirrored adopt it now.
                for other in self.scenarios.values():
                    if other.get("thesis_id") == tid \
                            and other["scenario_id"] != tid:
                        other["parent_scenario_id"] = tid
            else:
                standing = scenario["standing"]
                scenario["thesis_scenario"] = \
                    active.get("scenario")
                standing["p"] = active.get("winner_p")
                standing["peak_p"] = active.get("peak_p")
                standing["state"] = active.get("state") \
                    or standing.get("state") or "intact"
                scenario["thesis_version"] = \
                    active.get("version")
                if int(active.get("version") or 1) \
                        != int(scenario.get("version") or 1):
                    scenario["version"] = int(
                        active.get("version") or 1)
                    scenario["history"].append({
                        "version": scenario["version"],
                        "ts": ts, "kind": "thesis_revision",
                        "detail": "thesis version advanced "
                                  "(keeper revision)"})
                scenario["updated_ts"] = ts
        self._persist()
        return records

    # -- relationships ----------------------------------------------

    def _pair_key(self, a: str, b: str) -> str:
        return "|".join(sorted((a, b)))

    def refresh_relationships(self, *, ts,
                              situation_id=None) -> list[dict]:
        """Recompute scenario-pair relationships over the live
        book (SPEC §4.2(b)); a kind CHANGE journals a
        ``scenario_relationship`` record with both ids and the
        deciding evidence.  Pairs involving a scenario that has
        since closed journal one final ``ended`` record."""
        records: list[dict] = []
        live_ids = sorted(self.scenarios)
        current: dict[str, tuple[str, str, dict]] = {}
        for i, a_id in enumerate(live_ids):
            for b_id in live_ids[i + 1:]:
                kind, evidence = pair_relationship(
                    self.scenarios[a_id], self.scenarios[b_id])
                current[self._pair_key(a_id, b_id)] = (
                    a_id, b_id, kind, evidence)
        for key, (a_id, b_id, kind, evidence) in current.items():
            previous = self.relationships.get(key)
            if previous is None or previous.get("kind") != kind:
                records.append({
                    "type": "scenario_relationship",
                    "ts": ts, "a": a_id, "b": b_id,
                    "kind": kind,
                    "previous_kind": (previous or {}).get(
                        "kind"),
                    "evidence": evidence,
                    "situation_id": situation_id})
            self.relationships[key] = {
                "a": a_id, "b": b_id, "kind": kind}
        for key in list(self.relationships):
            if key not in current:
                old = self.relationships.pop(key)
                records.append({
                    "type": "scenario_relationship",
                    "ts": ts, "a": old.get("a"),
                    "b": old.get("b"), "kind": "ended",
                    "previous_kind": old.get("kind"),
                    "evidence": {"reason": "a scenario in the "
                                           "pair left the live "
                                           "book"},
                    "situation_id": situation_id})
        # Each scenario carries its side of every live pair.
        for scenario in self.scenarios.values():
            scenario["relationships"] = []
        for (a_id, b_id, kind, _ev) in current.values():
            self.scenarios[a_id]["relationships"].append(
                {"other_id": b_id, "kind": kind})
            self.scenarios[b_id]["relationships"].append(
                {"other_id": a_id, "kind": kind})
        if records:
            self._persist()
        return records

    # -- the cycle: positions, shadow scores, summary ----------------

    def _note_positions(self, open_tickets, *, ts,
                        situation_id) -> list[dict]:
        records: list[dict] = []
        open_set = {int(t) for t in open_tickets or ()}
        for scenario in list(self.scenarios.values()):
            if scenario["state"] not in POSITION_STATES:
                continue
            ticket = scenario.get("ticket")
            if ticket is None:
                # No broker identity recorded — the book cannot
                # verify this position either way, so it does
                # not move the scenario on absence of evidence.
                continue
            if int(ticket) in open_set:
                if scenario["state"] == "entered":
                    scenario["state"] = "managing"
                    self._bump(scenario, ts, "transition",
                               "broker position open — scenario "
                               "is managing its trade",
                               stage="position_open")
                    records.append(self._transition_record(
                        scenario, ts=ts, from_state="entered",
                        stage="position_open", detail="",
                        situation_id=situation_id))
            else:
                records.append(self._close(
                    scenario, ts=ts, close_reason="filled",
                    stage="position_closed",
                    detail="broker position no longer open — "
                           "the scenario's trade is complete",
                    situation_id=situation_id))
        if records:
            self._persist()
        return records

    def _score_scenarios(self, assessment: dict, *, ts,
                         situation_id) -> list[dict]:
        """The legacy five-scenario assessor scores every live
        book scenario: each scenario's standing is the
        assessor's probability for ITS side (bullish for long,
        bearish for short, range for directionless), tracked
        against its own peak with the thesis keeper's
        weaken/recover discipline — one scenario may weaken
        while another strengthens in the same cycle, and both
        facts land on the record."""
        records: list[dict] = []
        probs = {s.get("name"): float(s.get("p") or 0.0)
                 for s in assessment.get("scenarios") or []}
        scores = []
        for scenario in sorted(
                self.scenarios.values(),
                key=lambda s: s["scenario_id"]):
            if scenario["state"] not in LIVE_STATES:
                continue
            direction = int(scenario.get("direction") or 0)
            side = scenario.get("thesis_scenario") or (
                "bullish" if direction > 0
                else "bearish" if direction < 0
                else "range")
            p = probs.get(side)
            standing = scenario["standing"]
            p_before = standing.get("p")
            trend = "flat"
            if p is not None and p_before is not None:
                if p > p_before + 1e-9:
                    trend = "up"
                elif p < p_before - 1e-9:
                    trend = "down"
            if p is not None:
                peak = standing.get("peak_p")
                peak = p if peak is None else max(peak, p)
                standing["peak_p"] = peak
                if standing.get("state") == "intact" \
                        and p_before is not None \
                        and peak - p >= self.config.weaken_drop:
                    standing["state"] = "weakened"
                    standing["p"] = p
                    facts = [
                        f"assessed standing eroded: peak "
                        f"{peak:.4f} -> {p:.4f} (drop "
                        f"{peak - p:.4f} >= weaken drop "
                        f"{self.config.weaken_drop})"]
                    self._bump(
                        scenario, ts, "weakened",
                        "; ".join(facts))
                    records.append({
                        "type": "scenario_weakened",
                        "ts": ts,
                        "scenario_id": scenario["scenario_id"],
                        "family": scenario.get("family"),
                        "version": scenario["version"],
                        "evidence": {
                            "p_before": p_before,
                            "p_after": p,
                            "peak_p": peak,
                            "facts": facts},
                        "situation_id": situation_id})
                elif standing.get("state") == "weakened" \
                        and p >= peak - self.config.recover_band:
                    standing["state"] = "intact"
                    standing["p"] = p
                    scenario["history"].append({
                        "version": scenario["version"],
                        "ts": ts, "kind": "recovered",
                        "p_after": p, "peak_p": peak})
                else:
                    standing["p"] = p
            scores.append({
                "scenario_id": scenario["scenario_id"],
                "family": scenario.get("family"),
                "state": scenario["state"],
                "direction": direction,
                "side": side,
                "p": p,
                "p_before": p_before,
                "trend": trend,
                "standing_state": standing.get("state"),
                "s1": copy.deepcopy(scenario.get("s1")),
            })
        records.append({
            "type": "scenario_scores",
            "ts": ts,
            "situation_id": situation_id,
            "assessment": {
                "winner": assessment.get("winner"),
                "winner_p": assessment.get("winner_p"),
                "margin": assessment.get("margin"),
                "direction": assessment.get("direction"),
                "probabilities": probs},
            "scores": scores})
        self._persist()
        return records

    def _summary_record(self, *, ts, situation_id) -> dict:
        by_state: dict[str, int] = {}
        by_horizon: dict[str, int] = {}
        by_authority: dict[str, int] = {}
        exposure: dict[str, dict] = {}
        group_funders: dict[str, list[str]] = {}
        rows = []
        for scenario in self.live():
            by_state[scenario["state"]] = \
                by_state.get(scenario["state"], 0) + 1
            by_horizon[scenario.get("horizon")] = \
                by_horizon.get(scenario.get("horizon"), 0) + 1
            by_authority[scenario.get("authority")] = \
                by_authority.get(scenario.get("authority"),
                                 0) + 1
            if scenario["state"] in POSITION_STATES:
                slot = exposure.setdefault(
                    scenario.get("horizon"),
                    {"long": 0, "short": 0})
                if int(scenario.get("direction") or 0) > 0:
                    slot["long"] += 1
                elif int(scenario.get("direction") or 0) < 0:
                    slot["short"] += 1
            for group in scenario.get("evidence_groups") or []:
                group_funders.setdefault(group, []).append(
                    scenario["scenario_id"])
            rows.append({
                "scenario_id": scenario["scenario_id"],
                "family": scenario.get("family"),
                "state": scenario["state"],
                "direction": scenario.get("direction"),
                "horizon": scenario.get("horizon"),
                "authority": scenario.get("authority"),
                "parent_scenario_id":
                    scenario.get("parent_scenario_id"),
                "version": scenario.get("version")})
        competing = [
            {"a": rel["a"], "b": rel["b"]}
            for rel in self.relationships.values()
            if rel.get("kind") == "competing"]
        return {
            "type": "scenario_book",
            "ts": ts,
            "situation_id": situation_id,
            "open": len(rows),
            "closed_total": len(self.closed),
            "by_state": by_state,
            "by_horizon": by_horizon,
            "by_authority": by_authority,
            "exposure_by_horizon": exposure,
            "competing_pairs": competing,
            # One vote per correlation group, extended across
            # the book (SPEC §5.2 rule 3): the summary counts
            # each group ONCE however many scenarios it funds,
            # and names the sharing so double-counting is
            # visible, never silent.
            "evidence_groups": {
                "distinct": len(group_funders),
                "shared": {g: ids for g, ids
                           in group_funders.items()
                           if len(ids) > 1}},
            "scenarios": rows}

    def cycle(self, *, ts, situation_id=None, assessment=None,
              open_tickets=None) -> list[dict]:
        """One book cycle (H1 close, per SPEC §5.1's cadence):
        reconcile open positions, score every live scenario
        with the assessor (shadow), refresh pair relationships,
        and journal the book summary.  Returns every record the
        cycle produced, in production order."""
        records: list[dict] = []
        if open_tickets is not None:
            records += self._note_positions(
                open_tickets, ts=ts, situation_id=situation_id)
        if assessment is not None:
            records += self._score_scenarios(
                assessment, ts=ts, situation_id=situation_id)
        records += self.refresh_relationships(
            ts=ts, situation_id=situation_id)
        records.append(self._summary_record(
            ts=ts, situation_id=situation_id))
        return records
