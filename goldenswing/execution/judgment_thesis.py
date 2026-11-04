"""Judgment core S2 (owner spec approved 2026-11-02): the
five-scenario market read, the trade plan, and thesis maintenance.

S1 (judgment.py) judges ONE candidate at a time.  S2 sits above
it: on every evaluation cycle the core reads the WHOLE market —
the bedrock regime, posture, day type, ADR state, SMT and the
desk's own candidate flow — and maintains ONE live thesis:

- five competing scenarios are priced every cycle: **bullish**,
  **bearish**, **range**, **reversal**, **continuation**;
- the winner is chosen openly: it carries the evidence lines that
  fund it and the named conditions that would invalidate it;
- a directional winner becomes a THESIS with an actionable plan
  (entry conditions, invalidation, target, execution method,
  management horizon — seeded from the freshest aligned candidate
  when the desk has one, scenario-only otherwise);
- the thesis is MAINTAINED across cycles: refreshed while it
  keeps winning, invalidated the cycle its named conditions are
  observed, superseded when a rival scenario wins decisively,
  expired on its horizon or after repeated range challenges.
  Every transition is a journal event; history is preserved,
  never silently overwritten.

REVISION (owner approval 2026-11-04, the missed-move program):
maintenance above was not enough — the Nov 3 bearish thesis
stated its own invalidation price (3944.42) and then journaled
nothing while price ran through it to 4046, because the keeper
never saw price and erosion short of a named trigger was
invisible.  The lifecycle now REVISES:

- every surviving cycle re-prices the thesis's own scenario
  against the fresh assessment; a material erosion of its
  probability from its peak (structure intact, no trigger
  fired) journals ``thesis_weakened`` — a new thesis VERSION
  (version + supersedes pointer + the exact evidence: scenario
  probabilities before/after and the facts that moved) — and
  the thesis stands, weakened, still parenting its setups;
- the thesis's stated invalidation terms are evaluated AS DATA
  every cycle: the named triggers (regime flip, structure
  break) AND the plan's ``invalidation_price`` — a price cross
  invalidates exactly like a named trigger;
- a decisive rival REPLACES the thesis: the old one closes
  ``superseded`` and the new thesis record points back at it
  (``supersedes``), so version history is a chain, not a pile;
- candidates aligned with the live thesis are adopted as its
  CHILD setups (registered on the thesis, refreshed on
  re-emission); when a thesis closes, its unfilled children are
  journaled ``thesis_orphaned`` — never silently dropped.

DEPLOYMENT (owner ruling 2026-11-04): S1 stays shadow-only.
This S2 lifecycle is DEMO-ACTIVE for its own children: the
runner's shared order boundary (``_place_live_order``) consults
``gate_check`` and cuts a child whose parent thesis has closed
(inside the orphan window) — children of a live thesis and
independent, never-parented flow pass untouched.  The keeper
still never sizes, never steers independent playbooks, and the
friction gate / risk safeguards stay outside it, untouched.
Analytical confidence and evidence quality are carried as
SEPARATE fields, never blended.  Like S1, this is the
deterministic baseline — stated priors and response constants,
no fitted probabilities — that a later learned rung must beat
on chronological walk-forward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

SCENARIOS = ("bullish", "bearish", "range", "reversal",
             "continuation")
DIRECTIONAL = ("bullish", "bearish", "reversal", "continuation")

#: Scenario priors (stated, not fitted).
PRIORS = {"bullish": 0.22, "bearish": 0.22, "range": 0.24,
          "reversal": 0.14, "continuation": 0.18}

#: Scenario TTL / challenge defaults.
DEFAULT_TTL_HOURS = 24.0
DEFAULT_MIN_MARGIN = 0.08
DEFAULT_SWITCH_MARGIN = 0.05
DEFAULT_RANGE_CHALLENGE_LIMIT = 3
HEARTBEAT_SECONDS = 1800.0

#: Revision defaults (stated constants, like the priors): a
#: thesis whose own scenario probability falls this far from
#: its peak is WEAKENED (journaled, new version, thesis stands);
#: recovery to within the band of the peak clears the weakened
#: state.  Neither touches structure — a structural break is an
#: invalidation, never a weakening.
DEFAULT_WEAKEN_DROP = 0.10
DEFAULT_RECOVER_BAND = 0.03

#: How long after a thesis closes its orphaned children are
#: still recognised at the order boundary (the armed window: a
#: setup registered under the thesis can only be mid-flight
#: briefly — pendings arm inside an H1 window).  After it, a
#: re-emitted setup is the desk's own fresh candidate again.
ORPHAN_WINDOW_SECONDS = 3600.0

#: Child matching at the boundary uses the same at-level
#: tolerance as the Step-15 thesis-twin duplicate guard.
CHILD_MATCH_ATR_FRACTION = 0.25


@dataclass(frozen=True)
class ThesisConfig:
    ttl_hours: float = DEFAULT_TTL_HOURS
    min_margin: float = DEFAULT_MIN_MARGIN
    switch_margin: float = DEFAULT_SWITCH_MARGIN
    range_challenge_limit: int = DEFAULT_RANGE_CHALLENGE_LIMIT
    heartbeat_seconds: float = HEARTBEAT_SECONDS
    weaken_drop: float = DEFAULT_WEAKEN_DROP
    recover_band: float = DEFAULT_RECOVER_BAND


def _parse_regime_label(label) -> dict:
    out = {"direction": None, "structure": None,
           "volatility": None, "phase": None, "session": None}
    if not label:
        return out
    parts = str(label).split("/")
    for key, part in zip(out, parts):
        out[key] = part or None
    return out


def _fund(scores: dict, why: dict, name: str, amount: float,
          line: str) -> None:
    scores[name] += amount
    if amount > 0:
        why[name].append(line)


def assess_market(*, context: dict | None,
                  flow: dict | None = None) -> dict:
    """Price the five scenarios for one evaluation cycle.

    ``context`` is the runner's decision context (regime label,
    posture, day type, ADR, session, SMT).  ``flow`` summarises
    the desk's recent candidate flow:
    ``{"evaluated_long": n, "evaluated_short": n,
       "qualified_long": n, "qualified_short": n}`` — the desk's
    own engines voting with their feet.  Absent inputs abstain.
    """
    context = context or {}
    flow = flow or {}
    regime = _parse_regime_label(context.get("regime"))
    direction_word = regime.get("direction")
    regime_dir = {"up": 1, "down": -1}.get(str(direction_word or ""))
    structure = str(regime.get("structure") or "").lower()
    phase = str(regime.get("phase") or "").lower()
    volatility = str(regime.get("volatility") or "").lower()
    posture = str(context.get("posture") or "").lower()
    adr_state = str(context.get("adr_state") or "").lower()
    adr_pct = context.get("adr_pct")
    smt = str(context.get("smt") or "").lower()

    scores = {name: 0.0 for name in SCENARIOS}
    why: dict[str, list[str]] = {name: [] for name in SCENARIOS}
    evidence_lines = 0

    if regime_dir:
        evidence_lines += 1
        side = "bullish" if regime_dir > 0 else "bearish"
        _fund(scores, why, side, 1.2,
              f"regime direction is {direction_word}")
        if structure in ("clean_trend", "breakout", "trend"):
            _fund(scores, why, "continuation", 0.9,
                  f"structure {structure} carries the direction")
    if structure in ("clean_trend", "breakout", "trend"):
        evidence_lines += 1
        _fund(scores, why, "range", -1.0, "")
    elif structure in ("range", "compression", "choppy"):
        evidence_lines += 1
        _fund(scores, why, "range", 1.2,
              f"structure is {structure}")
        _fund(scores, why, "continuation", -0.6, "")
    elif structure == "reversal":
        evidence_lines += 1
        _fund(scores, why, "reversal", 1.1,
              "structure is classified reversal")
    if phase == "manipulation":
        evidence_lines += 1
        _fund(scores, why, "reversal", 0.5,
              "phase is manipulation — moves are suspect")
    elif phase == "distribution":
        evidence_lines += 1
        _fund(scores, why, "bearish", 0.4,
              "phase is distribution")
        _fund(scores, why, "reversal", 0.3, "")
    elif phase == "accumulation":
        evidence_lines += 1
        _fund(scores, why, "bullish", 0.4,
              "phase is accumulation")
    exhausted = adr_state == "exhausted" or (
        isinstance(adr_pct, (int, float)) and adr_pct >= 90.0)
    if exhausted:
        evidence_lines += 1
        _fund(scores, why, "reversal", 0.6,
              "ADR is exhausted — the day's range is spent")
        _fund(scores, why, "continuation", -0.5, "")
        _fund(scores, why, "range", 0.3, "")
    if volatility == "spike":
        evidence_lines += 1
        _fund(scores, why, "reversal", 0.3,
              "volatility is spiking")
        _fund(scores, why, "range", -0.4, "")
    if smt:
        evidence_lines += 1
        if "diverg" in smt:
            _fund(scores, why, "reversal", 0.5,
                  f"SMT divergence ({smt})")
        elif "confirm" in smt:
            _fund(scores, why, "continuation", 0.4,
                  f"SMT confirmation ({smt})")
    if "defensive" in posture or "risk_off" in posture:
        evidence_lines += 1
        _fund(scores, why, "range", 0.3,
              f"posture is {posture}")

    ev_long = int(flow.get("evaluated_long") or 0)
    ev_short = int(flow.get("evaluated_short") or 0)
    q_long = int(flow.get("qualified_long") or 0)
    q_short = int(flow.get("qualified_short") or 0)
    total_ev = ev_long + ev_short
    if total_ev >= 3:
        evidence_lines += 1
        skew = (ev_long - ev_short) / total_ev
        if abs(skew) >= 0.2:
            side = "bullish" if skew > 0 else "bearish"
            _fund(scores, why, side, 0.25 * abs(skew) * 2,
                  f"candidate flow skews {side} "
                  f"({ev_long} long vs {ev_short} short evaluated)")
    total_q = q_long + q_short
    if total_q >= 1:
        evidence_lines += 1
        qskew = (q_long - q_short) / max(total_q, 1)
        if qskew:
            side = "bullish" if qskew > 0 else "bearish"
            _fund(scores, why, side, 0.5 * abs(qskew),
                  f"qualified flow skews {side} "
                  f"({q_long} long vs {q_short} short qualified)")

    weights = {name: PRIORS[name] * math.exp(scores[name])
               for name in SCENARIOS}
    total_w = sum(weights.values())
    probs = {name: weights[name] / total_w for name in SCENARIOS}
    ranked = sorted(SCENARIOS, key=lambda n: probs[n], reverse=True)
    winner = ranked[0]
    runner_up = ranked[1]
    margin = probs[winner] - probs[runner_up]

    if winner in ("bullish", "continuation"):
        win_dir = regime_dir if regime_dir else (
            1 if winner == "bullish" else 0)
        if winner == "bullish" and not regime_dir:
            win_dir = 1
    elif winner == "bearish":
        win_dir = regime_dir if regime_dir else -1
        if not regime_dir:
            win_dir = -1
    elif winner == "reversal":
        win_dir = -regime_dir if regime_dir else 0
    else:
        win_dir = 0

    invalidation = _invalidation_for(winner, win_dir)
    confidence = min(1.0, 0.25 + 0.6 * margin + 0.05 * evidence_lines)
    quality = min(1.0, evidence_lines / 6.0)

    return {
        "scenarios": [
            {"name": name, "p": round(probs[name], 4),
             "why": why[name]} for name in ranked],
        "winner": winner,
        "winner_p": round(probs[winner], 4),
        "margin": round(margin, 4),
        "direction": win_dir,
        "invalidation": invalidation,
        "analytical_confidence": round(confidence, 4),
        "evidence_quality": round(quality, 4),
        "inputs": {
            "regime": context.get("regime"),
            "posture": context.get("posture"),
            "day_type": context.get("day_type"),
            "adr_pct": adr_pct,
            "adr_state": context.get("adr_state"),
            "smt": context.get("smt"),
            "flow": dict(flow),
        },
    }


def _invalidation_for(winner: str, win_dir: int) -> list[dict]:
    """Named invalidation conditions for a scenario: human text
    plus the machine-checkable trigger the keeper watches."""
    if winner == "bullish":
        return [
            {"text": "regime direction turns down",
             "trigger": "regime_direction_down"},
            {"text": "structure breaks into reversal against longs",
             "trigger": "structure_reversal_against"},
        ]
    if winner == "bearish":
        return [
            {"text": "regime direction turns up",
             "trigger": "regime_direction_up"},
            {"text": "structure breaks into reversal against shorts",
             "trigger": "structure_reversal_against"},
        ]
    if winner == "continuation":
        if win_dir < 0:
            return [
                {"text": "regime direction turns up",
                 "trigger": "regime_direction_up"},
                {"text": "structure decays into range/compression",
                 "trigger": "structure_range"},
            ]
        return [
            {"text": "regime direction turns down",
             "trigger": "regime_direction_down"},
            {"text": "structure decays into range/compression",
             "trigger": "structure_range"},
        ]
    if winner == "reversal":
        return [
            {"text": "the original direction reasserts with trend "
                     "structure",
             "trigger": "trend_reasserts"},
            {"text": "structure decays into range/compression",
             "trigger": "structure_range"},
        ]
    return [
        {"text": "structure breaks out with a direction",
         "trigger": "structure_breakout"},
    ]


def _triggers_fired(thesis: dict, assessment: dict) -> list[str]:
    """Which of the thesis's invalidation triggers the new
    assessment's inputs satisfy."""
    inputs = assessment.get("inputs") or {}
    regime = _parse_regime_label(inputs.get("regime"))
    direction_word = str(regime.get("direction") or "")
    structure = str(regime.get("structure") or "").lower()
    fired = []
    for cond in thesis.get("invalidation") or []:
        trig = cond.get("trigger")
        if trig == "regime_direction_down" \
                and direction_word == "down":
            fired.append(cond["text"])
        elif trig == "regime_direction_up" \
                and direction_word == "up":
            fired.append(cond["text"])
        elif trig == "structure_reversal_against" \
                and structure == "reversal":
            fired.append(cond["text"])
        elif trig == "structure_range" \
                and structure in ("range", "compression", "choppy"):
            fired.append(cond["text"])
        elif trig == "structure_breakout" \
                and structure in ("breakout", "clean_trend", "trend"):
            fired.append(cond["text"])
        elif trig == "trend_reasserts":
            orig = thesis.get("against_direction") or 0
            if orig and direction_word == ("up" if orig > 0
                                           else "down") \
                    and structure in ("clean_trend", "breakout",
                                      "trend"):
                fired.append(cond["text"])
    return fired


_ENTRY_CONDITIONS = {
    "bullish": [
        "longs only while the regime direction stays up",
        "enter on an entry-TF zone retest that holds, or on "
        "displacement through the session high",
        "no long into a fresh bearish displacement",
    ],
    "bearish": [
        "shorts only while the regime direction stays down",
        "enter on an entry-TF zone retest that rejects, or on "
        "displacement through the session low",
        "no short into a fresh bullish displacement",
    ],
    "continuation": [
        "trade with the prevailing direction on pullbacks to the "
        "entry-TF zone",
        "stand down on acceptance against the trend",
    ],
    "reversal": [
        "counter-trend only on a confirmed sweep + reclaim (or "
        "CHoCH) at the level",
        "half conviction until structure confirms the turn",
    ],
}


def build_plan(*, assessment: dict, candidate: dict | None,
               playbook: str = "") -> dict:
    """The actionable plan for a directional thesis.  Prices and
    method come from the freshest ALIGNED candidate when the desk
    has one; without one the plan is scenario-only and says so —
    levels are never invented."""
    winner = assessment.get("winner")
    direction = int(assessment.get("direction") or 0)
    plan: dict = {
        "direction": direction,
        "entry_conditions": list(
            _ENTRY_CONDITIONS.get(winner or "", [])),
        "invalidation": [c["text"] for c in
                         assessment.get("invalidation") or []],
        "entry": None, "invalidation_price": None, "target": None,
        "execution_method": "per playbook intent",
        "management_horizon": "standard_v1",
        "playbook": "",
        "basis": "scenario only — no aligned candidate this cycle",
    }
    cand = candidate or {}
    if cand and int(cand.get("direction") or 0) == direction \
            and direction != 0:
        plan["entry"] = cand.get("entry")
        plan["invalidation_price"] = cand.get("stop")
        targets = cand.get("targets") or []
        plan["target"] = targets[0] if targets else None
        plan["playbook"] = playbook or str(
            cand.get("playbook") or "")
        intent = str(cand.get("intent") or "")
        if intent in ("market", "limit"):
            plan["execution_method"] = intent
        if plan["playbook"] == "qm_m5":
            plan["management_horizon"] = "fixed_bracket_v1"
        plan["basis"] = (f"aligned candidate "
                         f"{plan['playbook'] or 'unnamed'}")
    return plan


def _scenario_probs(assessment: dict) -> dict:
    """Scenario name -> probability from one assessment."""
    return {s.get("name"): float(s.get("p") or 0.0)
            for s in assessment.get("scenarios") or []}


def _standing_snapshot(assessment: dict) -> dict:
    """The keeper's per-cycle standing read of the market: what
    the fresh assessment says, kept on the thesis so the NEXT
    cycle's revision evidence can quote before/after exactly."""
    inputs = assessment.get("inputs") or {}
    return {"scenarios": _scenario_probs(assessment),
            "winner": assessment.get("winner"),
            "margin": assessment.get("margin"),
            "regime": inputs.get("regime")}


@dataclass
class ThesisKeeper:
    """Maintains the one live thesis across evaluation cycles.

    ``update`` returns the journal events this cycle produced
    (possibly none — a maintained thesis is quiet between
    heartbeats).  Events are dicts the runner appends verbatim
    as ``judgment_thesis`` records.
    """

    config: ThesisConfig = field(default_factory=ThesisConfig)
    active: dict | None = None
    history: list[dict] = field(default_factory=list)
    _seq: int = 0
    _last_emit: float = 0.0
    _range_challenges: int = 0
    _last_orphans: list[dict] = field(default_factory=list)
    _orphan_close: dict = field(default_factory=dict)
    _pending_supersedes: str | None = None

    def _new_thesis(self, assessment: dict, ts: float,
                    candidate: dict | None,
                    playbook: str) -> dict:
        self._seq += 1
        winner = assessment["winner"]
        thesis = {
            "id": f"thesis-{self._seq}",
            "created_ts": ts,
            "updated_ts": ts,
            "scenario": winner,
            "direction": int(assessment.get("direction") or 0),
            "against_direction": (
                -int(assessment.get("direction") or 0)
                if winner == "reversal" else 0),
            "winner_p": assessment.get("winner_p"),
            "invalidation": assessment.get("invalidation") or [],
            "plan": build_plan(assessment=assessment,
                               candidate=candidate,
                               playbook=playbook),
            "status": "active",
            # Revision lineage (2026-11-04): a thesis is a chain
            # of versions; ``revisions`` is its own history, and
            # ``supersedes`` points at the thesis this one
            # replaced (None for a fresh line).
            "version": 1,
            "state": "intact",
            "supersedes": None,
            "revisions": [],
            "children": [],
            "peak_p": float(assessment.get("winner_p") or 0.0),
            "standing_p": float(assessment.get("winner_p")
                                or 0.0),
            "standing": _standing_snapshot(assessment),
        }
        return thesis

    def _close(self, status: str, ts: float, reason: str,
               evidence: dict | None = None) -> dict:
        thesis = self.active or {}
        thesis["status"] = status
        thesis["closed_ts"] = ts
        thesis["close_reason"] = reason
        self.history.append(thesis)
        self.active = None
        self._range_challenges = 0
        # The closed thesis's children become the recognisable
        # orphans at the order boundary (latest close only — a
        # newer close replaces the set).
        self._last_orphans = [dict(c)
                              for c in thesis.get("children")
                              or []]
        self._orphan_close = {
            "thesis_id": thesis.get("id"), "status": status,
            "reason": reason, "ts": ts}
        event = {"event": f"thesis_{status}", "ts": ts,
                 "thesis_id": thesis.get("id"),
                 "scenario": thesis.get("scenario"),
                 "direction": thesis.get("direction"),
                 "version": thesis.get("version"),
                 "reason": reason, "thesis": thesis}
        if evidence is not None:
            event["evidence"] = evidence
        return event

    def _orphan_event(self, events: list[dict], close_event: dict,
                      ts: float) -> None:
        """Journal the closed thesis's unfilled children as
        orphaned — they lose their parent here, on the record,
        never by silence.  (Shadow children are never filled by
        construction; the order boundary is where a mid-flight
        one is cut — see gate_check.)"""
        thesis = close_event.get("thesis") or {}
        children = thesis.get("children") or []
        if not children:
            return
        for child in children:
            child["orphaned_ts"] = ts
            child["orphan_status"] = thesis.get("status")
        events.append({
            "event": "thesis_orphaned", "ts": ts,
            "thesis_id": thesis.get("id"),
            "version": thesis.get("version"),
            "scenario": thesis.get("scenario"),
            "direction": thesis.get("direction"),
            "close_status": thesis.get("status"),
            "reason": thesis.get("close_reason"),
            "children": children, "count": len(children)})

    def _close_evidence(self, assessment: dict,
                        fired: list[str] | None = None,
                        price: float | None = None) -> dict:
        """The exact evidence a terminal revision carries:
        scenario probabilities before (the thesis's standing
        read) and after (the fresh assessment), plus the
        triggering facts when the close is an invalidation."""
        thesis = self.active or {}
        evidence = {
            "scenarios_before": (thesis.get("standing") or {})
            .get("scenarios"),
            "scenarios_after": _scenario_probs(assessment),
            "winner": assessment.get("winner"),
            "winner_p": assessment.get("winner_p"),
            "margin": assessment.get("margin"),
            "inputs": assessment.get("inputs"),
        }
        if fired:
            evidence["triggers_fired"] = list(fired)
        if price is not None:
            evidence["price"] = price
            evidence["invalidation_price"] = (
                (thesis.get("plan") or {})
                .get("invalidation_price"))
        return evidence

    def _price_fact(self, price: float | None) -> str | None:
        """The thesis's own stated invalidation price, evaluated
        as data: for a short thesis, price at/above the level
        breaks the premise; for a long, at/below.  No plan price
        (scenario-only thesis) -> no price term, exactly as
        before."""
        if price is None or self.active is None:
            return None
        plan = self.active.get("plan") or {}
        level = plan.get("invalidation_price")
        direction = int(self.active.get("direction") or 0)
        if level is None or direction == 0:
            return None
        try:
            px = float(price)
            lv = float(level)
        except (TypeError, ValueError):
            return None
        if (direction < 0 and px >= lv) \
                or (direction > 0 and px <= lv):
            return (f"price {px:.2f} crossed the thesis "
                    f"invalidation price {lv:.2f} "
                    f"(plan: {plan.get('basis') or 'scenario'})")
        return None

    def _register_child(self, candidate: dict | None,
                        playbook: str, ts: float) -> tuple:
        """Adopt an aligned candidate as a child of the live
        thesis.  Returns (child, is_new); a re-emission of the
        same setup refreshes its registration silently.  A
        counter-thesis or directionless candidate is never a
        child (weighed by the desk, not parented)."""
        if self.active is None or not candidate:
            return None, False
        direction = int(candidate.get("direction") or 0)
        if direction == 0 \
                or direction != int(self.active["direction"]):
            return None, False
        name = playbook or str(candidate.get("playbook") or "")
        entry = candidate.get("entry")
        stop = candidate.get("stop")
        for child in self.active["children"]:
            if child["playbook"] == name \
                    and child["direction"] == direction \
                    and child.get("entry") == entry \
                    and child.get("stop") == stop:
                child["registered_ts"] = ts
                return child, False
        child = {"playbook": name, "direction": direction,
                 "entry": entry, "stop": stop,
                 "targets": list(candidate.get("targets") or []),
                 "registered_ts": ts,
                 "thesis_id": self.active["id"],
                 "thesis_version": self.active["version"]}
        self.active["children"].append(child)
        return child, True

    def adopt_child(self, candidate: dict | None, playbook: str,
                    ts: float) -> dict | None:
        """Adopt an aligned candidate as a child of the live
        thesis and return the ``thesis_child_registered`` event
        for a NEW child (None when there is no live thesis, the
        candidate is not a child, or the child was already
        registered).  The update cycle journals the event
        itself; WS3's scan path calls this directly so a
        persisted setup instance can carry its parent link from
        birth — a later cycle re-registration of the same child
        is the silent refresh, exactly as before, so the event
        is journaled exactly once either way."""
        child, is_new = self._register_child(candidate, playbook,
                                             ts)
        if child is None or not is_new or self.active is None:
            return None
        return {
            "event": "thesis_child_registered", "ts": ts,
            "thesis_id": self.active["id"],
            "version": self.active["version"],
            "scenario": self.active["scenario"],
            "direction": self.active["direction"],
            "child": child}

    def child_link(self, *, playbook: str = "", direction: int = 0,
                   entry=None, stop=None) -> dict | None:
        """The parent link for a candidate the live thesis has
        already adopted as a child — the same identity
        ``_register_child`` dedupes on (playbook, direction,
        entry, stop).  Returns {"thesis_id", "thesis_version"}
        or None.  Read-only: this never registers anything."""
        if self.active is None:
            return None
        name = playbook
        for child in self.active.get("children") or []:
            if child.get("playbook") == name \
                    and int(child.get("direction") or 0) \
                    == int(direction or 0) \
                    and child.get("entry") == entry \
                    and child.get("stop") == stop:
                return {"thesis_id": self.active["id"],
                        "thesis_version": self.active["version"]}
        return None

    def _adopt(self, events: list[dict],
               candidate: dict | None, playbook: str,
               ts: float) -> None:
        """Register the cycle's candidate under the live thesis
        (if any); a NEW child is journaled so the decision trace
        thesis -> child exists on the record."""
        event = self.adopt_child(candidate, playbook, ts)
        if event is not None:
            events.append(event)

    def _revise_standing(self, assessment: dict, ts: float,
                         events: list[dict]) -> None:
        """Revision bookkeeping for a SURVIVING thesis: track
        its scenario's standing/peak probability; on material
        erosion with structure intact, journal the weakening as
        a new version with the exact evidence.  A recovery to
        within the band of the peak clears the weakened state
        (recorded on the thesis's own revision history)."""
        cfg = self.config
        thesis = self.active
        if thesis is None:
            return
        probs = _scenario_probs(assessment)
        scenario = thesis["scenario"]
        p_now = float(probs.get(scenario) or 0.0)
        peak = float(thesis.get("peak_p") or 0.0)
        if thesis.get("state") == "intact" \
                and peak - p_now >= cfg.weaken_drop:
            facts = [
                f"{scenario} probability eroded: peak "
                f"{peak:.4f} -> {p_now:.4f} (drop "
                f"{peak - p_now:.4f} >= weaken drop "
                f"{cfg.weaken_drop})"]
            winner = assessment.get("winner")
            if winner != scenario:
                why = next((s.get("why") for s in
                            assessment.get("scenarios") or []
                            if s.get("name") == winner), None)
                facts.append(
                    f"{winner} now leads at "
                    f"p={assessment.get('winner_p')}"
                    + (f" — {why[0]}" if why else ""))
            old_regime = (thesis.get("standing") or {}) \
                .get("regime")
            new_regime = (assessment.get("inputs") or {}) \
                .get("regime")
            if old_regime != new_regime:
                facts.append(f"regime read changed: "
                             f"{old_regime} -> {new_regime}")
            thesis["version"] = int(thesis.get("version")
                                     or 1) + 1
            thesis["state"] = "weakened"
            evidence = {
                "p_before": thesis.get("standing_p"),
                "p_after": p_now, "peak_p": peak,
                "scenarios_before": (thesis.get("standing")
                                     or {}).get("scenarios"),
                "scenarios_after": probs,
                "winner": winner,
                "winner_p": assessment.get("winner_p"),
                "margin": assessment.get("margin"),
                "inputs": assessment.get("inputs"),
                "facts": facts,
                "analytical_confidence":
                    assessment.get("analytical_confidence"),
                "evidence_quality":
                    assessment.get("evidence_quality"),
            }
            thesis["revisions"].append({
                "version": thesis["version"], "ts": ts,
                "kind": "weakened", "evidence": evidence})
            events.append({
                "event": "thesis_weakened", "ts": ts,
                "thesis_id": thesis["id"],
                "version": thesis["version"],
                "supersedes_version": thesis["version"] - 1,
                "scenario": scenario,
                "direction": thesis["direction"],
                "reason": "; ".join(facts),
                "evidence": evidence})
        elif thesis.get("state") == "weakened" \
                and p_now >= peak - cfg.recover_band:
            thesis["state"] = "intact"
            thesis["revisions"].append({
                "version": thesis["version"], "ts": ts,
                "kind": "recovered", "p_after": p_now,
                "peak_p": peak})
        thesis["standing"] = _standing_snapshot(assessment)
        thesis["standing_p"] = p_now
        thesis["peak_p"] = max(peak, p_now)

    @staticmethod
    def _match_child(children: list[dict], playbook: str,
                     direction: int, entry, atr) -> dict | None:
        if entry is None:
            return None
        try:
            level = float(entry)
        except (TypeError, ValueError):
            return None
        tol = CHILD_MATCH_ATR_FRACTION * float(atr or 0.0)
        for child in children:
            if int(child.get("direction") or 0) != direction:
                continue
            if playbook and child.get("playbook") \
                    and child["playbook"] != playbook:
                continue
            child_entry = child.get("entry")
            if child_entry is None:
                continue
            if abs(float(child_entry) - level) <= tol:
                return child
        return None

    def gate_check(self, *, playbook: str = "",
                   direction: int = 0, entry=None, atr=None,
                   ts: float | None = None) -> dict:
        """The order boundary's question (demo-active ruling,
        2026-11-04): is this setup a child of the thesis
        lifecycle, and may it flow?

        - ``child``: a registered child of the LIVE thesis, or
          aligned by direction while the thesis stands — flows
          through the shared order path as today.
        - ``orphaned``: registered under a thesis that has since
          closed and still inside the orphan window — the cut.
          An orphan match outranks mere alignment with a newer
          thesis: the setup was born under the dead thesis, and
          a new thesis must ADOPT it through a fresh
          registration cycle (rule order below) before it can
          flow as that thesis's child.
        - ``independent``: never parented by a thesis — the
          lifecycle abstains; the desk's own path decides.
        """
        direction = int(direction or 0)
        if self.active is not None and direction:
            child = self._match_child(
                self.active.get("children") or [], playbook,
                direction, entry, atr)
            if child is not None:
                return {"verdict": "child",
                        "thesis_id": self.active["id"],
                        "version": self.active.get("version"),
                        "state": self.active.get("state"),
                        "registered": True,
                        "child": child}
        close = self._orphan_close
        if direction and close and self._last_orphans:
            if ts is None or ts - float(close.get("ts") or 0.0) \
                    <= ORPHAN_WINDOW_SECONDS:
                child = self._match_child(
                    self._last_orphans, playbook, direction,
                    entry, atr)
                if child is not None:
                    return {"verdict": "orphaned",
                            "thesis_id": close.get("thesis_id"),
                            "close_status": close.get("status"),
                            "close_reason": close.get("reason"),
                            "child": child}
        if self.active is not None and direction \
                and direction == int(self.active.get("direction")
                                     or 0):
            return {"verdict": "child",
                    "thesis_id": self.active["id"],
                    "version": self.active.get("version"),
                    "state": self.active.get("state"),
                    "registered": False,
                    "child": None}
        return {"verdict": "independent", "thesis_id": None}

    def update(self, assessment: dict, *, ts: float,
               candidate: dict | None = None,
               playbook: str = "",
               price: float | None = None) -> list[dict]:
        cfg = self.config
        events: list[dict] = []
        winner = assessment.get("winner")
        margin = float(assessment.get("margin") or 0.0)

        if self.active is not None:
            age_h = (ts - float(self.active["created_ts"])) / 3600.0
            if age_h > cfg.ttl_hours:
                ev = self._close(
                    "expired", ts,
                    f"horizon reached ({age_h:.1f}h > "
                    f"{cfg.ttl_hours:.0f}h)",
                    evidence=self._close_evidence(assessment))
                events.append(ev)
                self._orphan_event(events, ev, ts)
            else:
                fired = _triggers_fired(self.active, assessment)
                price_fact = self._price_fact(price)
                if price_fact:
                    fired = fired + [price_fact]
                if fired:
                    ev = self._close(
                        "invalidated", ts,
                        "invalidation observed: "
                        + "; ".join(fired),
                        evidence=self._close_evidence(
                            assessment, fired=fired, price=price))
                    events.append(ev)
                    self._orphan_event(events, ev, ts)
        if self.active is not None:
            # Revision: a surviving thesis is re-priced against
            # the fresh assessment EVERY cycle — erosion short
            # of a structural break is a weakening, journaled,
            # never silence (the Nov 3 failure mode).
            self._revise_standing(assessment, ts, events)
            if winner == self.active["scenario"]:
                self._range_challenges = 0
                self.active["updated_ts"] = ts
                self.active["winner_p"] = assessment.get("winner_p")
                if ts - self._last_emit >= cfg.heartbeat_seconds:
                    self._last_emit = ts
                    events.append({
                        "event": "thesis_maintained", "ts": ts,
                        "thesis_id": self.active["id"],
                        "scenario": winner,
                        "direction": self.active["direction"],
                        "winner_p": assessment.get("winner_p"),
                        "version": self.active["version"],
                        "state": self.active["state"],
                        "assessment": assessment})
                self._adopt(events, candidate, playbook, ts)
                return events
            if winner == "range":
                self._range_challenges += 1
                if self._range_challenges >= \
                        cfg.range_challenge_limit:
                    ev = self._close(
                        "expired", ts,
                        "faded: repeated range reads — the "
                        "directional thesis no longer describes "
                        "the market",
                        evidence=self._close_evidence(assessment))
                    events.append(ev)
                    self._orphan_event(events, ev, ts)
                    # fall through: a fresh thesis may form below
                else:
                    self.active["updated_ts"] = ts
                    self._adopt(events, candidate, playbook, ts)
                    return events
            elif winner in DIRECTIONAL and margin >= cfg.min_margin:
                active_p = float(self.active.get("winner_p") or 0.0)
                if float(assessment.get("winner_p") or 0.0) \
                        >= active_p + cfg.switch_margin:
                    old = self.active
                    ev = self._close(
                        "superseded", ts,
                        f"{winner} won decisively "
                        f"(p={assessment.get('winner_p')} vs "
                        f"thesis {old.get('scenario')} "
                        f"p={active_p})",
                        evidence=self._close_evidence(assessment))
                    events.append(ev)
                    self._orphan_event(events, ev, ts)
                    self._pending_supersedes = old.get("id")
                else:
                    self.active["updated_ts"] = ts
                    self._adopt(events, candidate, playbook, ts)
                    return events
            else:
                self.active["updated_ts"] = ts
                self._adopt(events, candidate, playbook, ts)
                return events

        if self.active is None and winner in DIRECTIONAL \
                and margin >= cfg.min_margin:
            thesis = self._new_thesis(assessment, ts, candidate,
                                      playbook)
            if self._pending_supersedes:
                thesis["supersedes"] = self._pending_supersedes
                self._pending_supersedes = None
            self.active = thesis
            self._last_emit = ts
            self._range_challenges = 0
            created = {
                "event": "thesis_created", "ts": ts,
                "thesis_id": thesis["id"], "scenario": winner,
                "direction": thesis["direction"],
                "winner_p": assessment.get("winner_p"),
                "version": thesis["version"],
                "reason": "; ".join(
                    (assessment.get("scenarios") or [{}])[0]
                    .get("why") or []) or winner,
                "assessment": assessment, "thesis": thesis}
            if thesis["supersedes"]:
                created["supersedes"] = thesis["supersedes"]
            events.append(created)
            # The seeding candidate is the thesis's first child.
            self._adopt(events, candidate, playbook, ts)
        return events

    def status(self) -> dict:
        if self.active is None:
            return {"active": False, "closed_total": len(self.history)}
        return {"active": True, **{
            k: self.active.get(k) for k in (
                "id", "scenario", "direction", "winner_p",
                "created_ts", "updated_ts", "status",
                "version", "state")},
            "plan": self.active.get("plan"),
            "children_open": len(self.active.get("children")
                                 or []),
            "closed_total": len(self.history)}
