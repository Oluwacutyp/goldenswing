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

Still SHADOW (the standing ruling): nothing here steers, sizes,
gates or vetoes an order.  Analytical confidence and evidence
quality are carried as SEPARATE fields, never blended.  Like S1,
this is the deterministic baseline — stated priors and response
constants, no fitted probabilities — that a later learned rung
must beat on chronological walk-forward.
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


@dataclass(frozen=True)
class ThesisConfig:
    ttl_hours: float = DEFAULT_TTL_HOURS
    min_margin: float = DEFAULT_MIN_MARGIN
    switch_margin: float = DEFAULT_SWITCH_MARGIN
    range_challenge_limit: int = DEFAULT_RANGE_CHALLENGE_LIMIT
    heartbeat_seconds: float = HEARTBEAT_SECONDS


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
        }
        return thesis

    def _close(self, status: str, ts: float, reason: str) -> dict:
        thesis = self.active or {}
        thesis["status"] = status
        thesis["closed_ts"] = ts
        thesis["close_reason"] = reason
        self.history.append(thesis)
        self.active = None
        self._range_challenges = 0
        return {"event": f"thesis_{status}", "ts": ts,
                "thesis_id": thesis.get("id"),
                "scenario": thesis.get("scenario"),
                "direction": thesis.get("direction"),
                "reason": reason, "thesis": thesis}

    def update(self, assessment: dict, *, ts: float,
               candidate: dict | None = None,
               playbook: str = "") -> list[dict]:
        cfg = self.config
        events: list[dict] = []
        winner = assessment.get("winner")
        margin = float(assessment.get("margin") or 0.0)

        if self.active is not None:
            age_h = (ts - float(self.active["created_ts"])) / 3600.0
            if age_h > cfg.ttl_hours:
                events.append(self._close(
                    "expired", ts,
                    f"horizon reached ({age_h:.1f}h > "
                    f"{cfg.ttl_hours:.0f}h)"))
            else:
                fired = _triggers_fired(self.active, assessment)
                if fired:
                    events.append(self._close(
                        "invalidated", ts,
                        "invalidation observed: "
                        + "; ".join(fired)))
        if self.active is not None:
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
                        "assessment": assessment})
                return events
            if winner == "range":
                self._range_challenges += 1
                if self._range_challenges >= \
                        cfg.range_challenge_limit:
                    events.append(self._close(
                        "expired", ts,
                        "faded: repeated range reads — the "
                        "directional thesis no longer describes "
                        "the market"))
                    # fall through: a fresh thesis may form below
                else:
                    self.active["updated_ts"] = ts
                    return events
            elif winner in DIRECTIONAL and margin >= cfg.min_margin:
                active_p = float(self.active.get("winner_p") or 0.0)
                if float(assessment.get("winner_p") or 0.0) \
                        >= active_p + cfg.switch_margin:
                    old = self.active
                    events.append(self._close(
                        "superseded", ts,
                        f"{winner} won decisively "
                        f"(p={assessment.get('winner_p')} vs "
                        f"thesis {old.get('scenario')} "
                        f"p={active_p})"))
                else:
                    self.active["updated_ts"] = ts
                    return events
            else:
                self.active["updated_ts"] = ts
                return events

        if self.active is None and winner in DIRECTIONAL \
                and margin >= cfg.min_margin:
            thesis = self._new_thesis(assessment, ts, candidate,
                                      playbook)
            self.active = thesis
            self._last_emit = ts
            self._range_challenges = 0
            events.append({
                "event": "thesis_created", "ts": ts,
                "thesis_id": thesis["id"], "scenario": winner,
                "direction": thesis["direction"],
                "winner_p": assessment.get("winner_p"),
                "reason": "; ".join(
                    (assessment.get("scenarios") or [{}])[0]
                    .get("why") or []) or winner,
                "assessment": assessment, "thesis": thesis})
        return events

    def status(self) -> dict:
        if self.active is None:
            return {"active": False, "closed_total": len(self.history)}
        return {"active": True, **{
            k: self.active.get(k) for k in (
                "id", "scenario", "direction", "winner_p",
                "created_ts", "updated_ts", "status")},
            "plan": self.active.get("plan"),
            "closed_total": len(self.history)}
