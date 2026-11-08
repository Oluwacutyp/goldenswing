"""Judgment over the scenario book (MarketSituation Pass 3 —
SPEC §5.1/§5.2 as scoped by the pass brief, plus AMENDMENTS-1
§A6 coverage gaps).

Pass 2 built the book: many coexisting scenarios, competing
pairs MARKED, never resolved.  This module is the reader the
spec always intended for it — a book-level cycle that runs
once per H1 close, beside the governing paths, and journals
what Judgment WOULD do with the whole book:

- every COMPETING pair is resolved on the record: the higher
  S1 expected_r wins the pair (a tie inside the stated epsilon
  abstains both), the winner must still clear the same stated
  bars S1 uses (``min_expected_r`` / ``min_confidence`` — the
  runner's judgment config, unchanged), the loser is capped
  at MONITOR with the reason ``competing_pair_lost``, and a
  pair neither side of which clears the bars ABSTAINS with
  the closest scenario named (§5.2 rules 2 + 4);
- every TRIGGERED scenario standing alone (in no competing
  pair) is judged on its own S1 verdict: TAKE when S1 took
  it, ABSTAIN otherwise, with S1's own reason carried
  verbatim — the assessor's vocabulary, not a paraphrase;
- the book's evidence-group discipline (§5.2 rule 3) is
  cited, never double-counted: when both sides of a pair are
  funded by the same correlation group, the record says so —
  one vote, not two.

And the coverage instrument (AMENDMENTS-1 §A6): situation
states and relationship configurations that persist across
the stated window (``COVERAGE_WINDOW_CYCLES`` consecutive
book cycles) with NO live scenario proposing into them
journal a ``coverage_gap`` record naming the configuration
concretely.  The cycle names the strongest uncovered
configuration for the book summary — the "nineteenth
opportunity" made visible: what none of the specialists
described.

SHADOW DISCIPLINE (Pass 3): every record here is evidence.
Governed routes keep their legacy authority until Pass 5;
nothing in this module is read by any decision path, and the
forecast E3 gate is untouched — forecasts are not an input
here at all.  A scenario with no S1 verdict on record is
UNJUDGEABLE and the record says so: absent stays absent,
never invented.

Determinism (D9): this module is a pure function of the
book's recorded state, the latest situation and the tracker
state — same inputs, same records out.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

#: Two scenarios whose S1 expected_r values sit within this
#: distance of each other are a TIE for pair resolution
#: (§5.2 rule 2's "stated epsilon"): the pair abstains both
#: rather than letting noise pick a side.
PAIR_EPSILON = 0.02

#: The stated persistence window for a coverage gap
#: (AMENDMENTS-1 §A6): a situation configuration must be
#: present — and uncovered — on this many CONSECUTIVE book
#: cycles (the book cycle is the H1 close, so the default is
#: three hours of unchanged, unaddressed market structure)
#: before it journals as a gap.
COVERAGE_WINDOW_CYCLES = 3

#: Timeframe weight for "strongest configuration" ranking:
#: a gap on a slower clock outranks one on a faster clock.
_TF_WEIGHT = {"D1": 6, "H4": 5, "H1": 4, "M30": 3,
              "M15": 2, "M5": 1, "M1": 0}
_TF_ORDER = ("D1", "H4", "H1", "M30", "M15", "M5", "M1")


@dataclass(frozen=True)
class BookJudgmentConfig:
    """The stated bars, shared with S1 by construction: the
    runner builds this from the same config fields
    ``_judgment_evaluate`` reads (``judgment_min_expected_r``
    / ``judgment_min_confidence``), so the book judges against
    the desk's bar, never a second one."""
    min_expected_r: float = 0.05
    min_confidence: float = 0.15
    pair_epsilon: float = PAIR_EPSILON
    coverage_window: int = COVERAGE_WINDOW_CYCLES


# ------------------------------------------------------------ inputs

def _s1_view(scenario: dict) -> dict | None:
    """The scenario's S1 verdict, in the fields the book
    judges with — or None when the scenario was never
    S1-evaluated (absent stays absent)."""
    s1 = scenario.get("s1")
    if not isinstance(s1, dict):
        return None
    try:
        expected_r = s1.get("expected_r")
        confidence = s1.get("confidence")
        return {
            "expected_r": (float(expected_r)
                           if expected_r is not None else None),
            "confidence": (float(confidence)
                           if confidence is not None else None),
            "decision": s1.get("decision"),
            "reason": s1.get("reason"),
        }
    except (TypeError, ValueError):
        return None


def _side_of(scenario: dict) -> str:
    """The assessor side this scenario stands on — the same
    rule the book's scorer uses (the thesis scenario name for
    the mirrored thesis; bullish / bearish / range by
    direction)."""
    return scenario.get("thesis_scenario") or (
        "bullish" if int(scenario.get("direction") or 0) > 0
        else "bearish" if int(scenario.get("direction") or 0) < 0
        else "range")


def _inputs_block(scenario: dict) -> dict:
    """What the judgment read off one scenario — cited on the
    record so the verdict is reconstructable from the journal
    alone (the version judged is named: a scenario revised
    after the judgment never rewrites what was judged)."""
    view = _s1_view(scenario)
    standing = scenario.get("standing") or {}
    return {
        "scenario_id": scenario.get("scenario_id"),
        "family": scenario.get("family"),
        "version": scenario.get("version"),
        "state": scenario.get("state"),
        "direction": scenario.get("direction"),
        "expected_r": (view or {}).get("expected_r"),
        "confidence": (view or {}).get("confidence"),
        "s1_decision": (view or {}).get("decision"),
        "assessor_side": _side_of(scenario),
        "assessor_p": standing.get("p"),
        "standing_state": standing.get("state"),
        "evidence_groups": list(
            scenario.get("evidence_groups") or []),
    }


def _bars_text(cfg: BookJudgmentConfig) -> str:
    return (f"stated bars: expected_r >= "
            f"{cfg.min_expected_r:+.2f}R, confidence >= "
            f"{cfg.min_confidence:.2f}")


# ------------------------------------------------------------ pairs

def judge_pair(a: dict, b: dict, *, ts, situation_id,
               cfg: BookJudgmentConfig) -> dict:
    """Resolve one competing pair (§5.2 rule 2 + rule 4).

    Higher S1 expected_r wins the pair; a tie inside the
    epsilon abstains both; the winner must clear the stated
    bars or the pair abstains with the winner named as the
    closest.  The loser of a taken pair is capped at MONITOR
    with reason ``competing_pair_lost`` — marked, never
    silently dropped."""
    va, vb = _s1_view(a), _s1_view(b)
    a_id, b_id = a["scenario_id"], b["scenario_id"]
    shared = sorted(set(a.get("evidence_groups") or [])
                    & set(b.get("evidence_groups") or []))
    reasons: list[str] = []
    if shared:
        reasons.append(
            "shared evidence groups fund both sides: "
            + ", ".join(shared)
            + " — one vote per group across the book, not two "
              "(§5.2 rule 3)")
    verdicts: dict[str, str] = {}
    take_id = None
    closest_id = None
    for sc in (a, b):
        standing = sc.get("standing") or {}
        if standing.get("state") in ("weakened",
                                     "invalidated"):
            reasons.append(
                f"assessor standing: {sc['scenario_id']} "
                f"{standing['state']} (p "
                f"{standing.get('p')}, peak "
                f"{standing.get('peak_p')}) — the assessor's "
                f"own read of this scenario's thesis")
    if va is None or vb is None:
        missing = [sid for sid, v in ((a_id, va), (b_id, vb))
                   if v is None]
        verdict = "abstain"
        verdicts = {a_id: "abstain", b_id: "abstain"}
        reasons.append(
            "unscored: " + ", ".join(missing)
            + " carries no S1 verdict on record — the pair "
              "cannot be resolved from absent evidence")
    elif va["expected_r"] is None or vb["expected_r"] is None:
        verdict = "abstain"
        verdicts = {a_id: "abstain", b_id: "abstain"}
        reasons.append(
            "an S1 verdict without an expected_r (invalid "
            "geometry at judgment time) cannot win a pair — "
            "abstain both")
    else:
        ea, eb = va["expected_r"], vb["expected_r"]
        reasons.append(
            f"expected_r: {a_id} {ea:+.2f}R vs {b_id} "
            f"{eb:+.2f}R (Δ {abs(ea - eb):.2f}R, pair epsilon "
            f"{cfg.pair_epsilon:.2f}R)")
        reasons.append(
            f"confidence: {a_id} {va['confidence']:.2f} vs "
            f"{b_id} {vb['confidence']:.2f}")
        if abs(ea - eb) <= cfg.pair_epsilon:
            verdict = "abstain"
            verdicts = {a_id: "abstain", b_id: "abstain"}
            closest_id = a_id if ea >= eb else b_id
            reasons.append(
                "tie inside the pair epsilon — abstain both "
                "rather than let noise pick a side (§5.2 "
                "rule 2)")
        else:
            winner, loser = (a, b) if ea > eb else (b, a)
            w_view = va if winner is a else vb
            w_id, l_id = winner["scenario_id"], \
                loser["scenario_id"]
            closest_id = w_id
            failed = []
            if w_view["expected_r"] < cfg.min_expected_r:
                failed.append(
                    f"expected_r {w_view['expected_r']:+.2f}R "
                    f"< {cfg.min_expected_r:+.2f}R")
            if w_view["confidence"] < cfg.min_confidence:
                failed.append(
                    f"confidence {w_view['confidence']:.2f} "
                    f"< {cfg.min_confidence:.2f}")
            if failed:
                verdict = "abstain"
                verdicts = {a_id: "abstain", b_id: "abstain"}
                reasons.append(
                    f"pair winner {w_id} does not clear the "
                    f"book's bars ({'; '.join(failed)}) — "
                    f"abstain both; {w_id} named closest")
            else:
                verdict = "take"
                take_id = w_id
                verdicts = {w_id: "take", l_id: "monitor"}
                reasons.append(
                    f"{w_id} wins the pair and clears the "
                    f"book's bars ({_bars_text(cfg)}); {l_id} "
                    f"is capped at MONITOR "
                    f"(competing_pair_lost)")
    return {
        "type": "book_judgment",
        "ts": ts,
        "scope": "competing_pair",
        "pair": [a_id, b_id],
        "verdict": verdict,
        "take_scenario_id": take_id,
        "closest_scenario_id": closest_id,
        "verdicts": verdicts,
        "loser_reason": ("competing_pair_lost"
                         if take_id is not None else None),
        "reasons": reasons,
        "inputs": {a_id: _inputs_block(a),
                   b_id: _inputs_block(b)},
        "relationship_kinds": ["competing"],
        "thresholds": {
            "min_expected_r": cfg.min_expected_r,
            "min_confidence": cfg.min_confidence,
            "pair_epsilon": cfg.pair_epsilon},
        "situation_id": situation_id,
    }


def judge_lone_triggered(scenario: dict, *, ts, situation_id,
                         cfg: BookJudgmentConfig) -> dict:
    """Judge one TRIGGERED scenario that stands in no
    competing pair: its own S1 verdict is the evidence — TAKE
    when S1 took it (the bars were cleared inside S1's own
    evaluation, against these same thresholds), ABSTAIN
    otherwise, with S1's reason carried verbatim."""
    sid = scenario["scenario_id"]
    view = _s1_view(scenario)
    reasons: list[str] = []
    standing = scenario.get("standing") or {}
    if standing.get("state") in ("weakened", "invalidated"):
        reasons.append(
            f"assessor standing: {sid} {standing['state']} "
            f"(p {standing.get('p')}, peak "
            f"{standing.get('peak_p')}) — the assessor's own "
            f"read of this scenario's thesis")
    closest_id = None
    if view is None:
        verdict = "abstain"
        reasons.append(
            "no S1 verdict on record — this scenario was "
            "never S1-evaluated; absent stays absent")
    elif view["decision"] == "take":
        verdict = "take"
        closest_id = sid
        if view.get("reason"):
            reasons.append(f"S1: {view['reason']}")
        reasons.append(
            f"triggered and standing alone; clears the book's "
            f"bars ({_bars_text(cfg)}) on its own S1 verdict")
    else:
        verdict = "abstain"
        closest_id = sid
        if view.get("reason"):
            reasons.append(f"S1: {view['reason']}")
        reasons.append(
            "triggered and standing alone, but its S1 verdict "
            "did not take it — the book abstains with it named "
            "closest")
    return {
        "type": "book_judgment",
        "ts": ts,
        "scope": "lone_triggered",
        "pair": None,
        "verdict": verdict,
        "take_scenario_id": sid if verdict == "take" else None,
        "closest_scenario_id": closest_id,
        "verdicts": {sid: verdict},
        "loser_reason": None,
        "reasons": reasons,
        "inputs": {sid: _inputs_block(scenario)},
        "relationship_kinds": [],
        "thresholds": {
            "min_expected_r": cfg.min_expected_r,
            "min_confidence": cfg.min_confidence,
            "pair_epsilon": cfg.pair_epsilon},
        "situation_id": situation_id,
    }


# ------------------------------------------------------------ coverage

def configurations_of(situation: dict) -> list[dict]:
    """The market configurations the current situation
    presents, in the concrete vocabulary a gap record names:
    per-TF measured states (ready TFs with a measured bias)
    and the relationship engine's pair kinds (the hierarchy
    comparison row is not a configuration of the engine's own
    read; ``independent`` is the absence of a relationship —
    there is nothing to propose INTO it)."""
    out: list[dict] = []
    per_tf = (situation or {}).get("per_tf") or {}
    for tf in _TF_ORDER:
        state = per_tf.get(tf)
        if not state or state.get("readiness") != "ready":
            continue
        bias = state.get("bias")
        if bias is None:
            continue
        phase = state.get("structure_phase")
        out.append({
            "key": f"tf:{tf}:bias={int(bias):+d}"
                   f":phase={phase}",
            "kind": "tf",
            "tf": tf, "bias": int(bias), "phase": phase,
            "label": f"{tf} measured state: bias "
                     f"{int(bias):+d}, structure phase "
                     f"{phase}"})
    seen_pairs = set()
    for rel in (situation or {}).get("relationships") or []:
        if rel.get("source") not in (None, "engine"):
            continue
        kind = rel.get("kind")
        if not kind or kind == "independent":
            continue
        pair = (rel.get("htf"), rel.get("ltf"))
        if pair in seen_pairs or None in pair:
            continue
        seen_pairs.add(pair)
        out.append({
            "key": f"rel:{pair[0]}->{pair[1]}:{kind}",
            "kind": "relationship",
            "htf": pair[0], "ltf": pair[1], "rel_kind": kind,
            "label": f"{pair[0]}->{pair[1]} relationship: "
                     f"{kind}"})
    return out


def configuration_covered(config: dict, live: list[dict]
                            ) -> bool:
    """A configuration is covered when a LIVE scenario
    proposes into it: for a TF state, a scenario formed (or
    triggered) on that TF in the state's direction (any
    direction covers a flat read — a range scenario is a
    proposal into a range); for a relationship, a scenario
    whose clocks trade across the pair (formed on the slower
    TF, entered/triggered on the faster)."""
    for scenario in live:
        clocks = scenario.get("clocks") or {}
        if config["kind"] == "tf":
            if config["tf"] not in (clocks.get("formation_tf"),
                                    clocks.get("trigger_tf")):
                continue
            if config["bias"] == 0 or int(
                    scenario.get("direction") or 0) \
                    == config["bias"]:
                return True
        else:
            if clocks.get("formation_tf") != config["htf"]:
                continue
            if config["ltf"] in (clocks.get("entry_tf"),
                                 clocks.get("trigger_tf")):
                return True
    return False


def _strength(config: dict, cycles: int) -> tuple:
    """Strongest-first ordering for uncovered configurations:
    relationship configurations outrank single-TF states (the
    cross-timeframe read is the program's point), longer
    persistence outranks shorter, slower clocks outrank
    faster.  Lower tuple = stronger."""
    if config["kind"] == "relationship":
        return (0, -cycles, -_TF_WEIGHT.get(config["htf"], 0),
                config["key"])
    return (1, -cycles, -_TF_WEIGHT.get(config["tf"], 0),
            config["key"])


class CoverageTracker:
    """Per-configuration persistence across book cycles.

    The tracker is the cycle's memory: a configuration earns
    a ``coverage_gap`` record when it has been present AND
    uncovered for the stated window of consecutive cycles,
    and fires ONCE per episode — the episode ends when the
    configuration breaks (leaves the situation) or becomes
    covered.  State is in-memory (held by the runner): a
    restart restarts the window, which is the honest posture
    for an observation instrument — the gap record itself,
    once journaled, is durable."""

    def __init__(self):
        self._state: dict[str, dict] = {}

    def observe(self, situation: dict, live: list[dict], *,
                ts, situation_id,
                cfg: BookJudgmentConfig
                ) -> tuple[list[dict], dict | None]:
        """One book cycle's coverage read.  Returns the NEW
        gap records fired this cycle and the strongest ACTIVE
        gap (fired, still persisting) for the book summary —
        or None when no gap is active."""
        configs = configurations_of(situation)
        present: dict[str, dict] = {}
        for config in configs:
            if configuration_covered(config, live):
                # Covered: any episode for this configuration
                # ends — silently; the gap record (if fired)
                # stands as the history of the blind spot.
                self._state.pop(config["key"], None)
                continue
            entry = self._state.get(config["key"])
            if entry is None:
                entry = {"count": 0, "first_ts": ts,
                         "first_situation_id": situation_id,
                         "fired": False, "config": config}
                self._state[config["key"]] = entry
            entry["count"] += 1
            entry["config"] = config
            present[config["key"]] = entry
        # A configuration that left the situation ends its
        # episode without firing.
        for key in list(self._state):
            if key not in present:
                self._state.pop(key, None)
        records: list[dict] = []
        active: list[tuple[tuple, dict]] = []
        for key, entry in present.items():
            if entry["count"] < cfg.coverage_window:
                continue
            if not entry["fired"]:
                entry["fired"] = True
                records.append({
                    "type": "coverage_gap",
                    "ts": ts,
                    "config_key": key,
                    "configuration": copy.deepcopy(
                        entry["config"]),
                    "label": entry["config"]["label"],
                    "first_seen_ts": entry["first_ts"],
                    "first_situation_id":
                        entry["first_situation_id"],
                    "cycles": entry["count"],
                    "window": cfg.coverage_window,
                    "situation_id": situation_id,
                    "detail": "this configuration has "
                              "persisted uncovered across "
                              "the stated window — no live "
                              "scenario proposes into it",
                })
            active.append((_strength(entry["config"],
                                      entry["count"]), entry))
        strongest = None
        if active:
            active.sort(key=lambda item: item[0])
            entry = active[0][1]
            strongest = {
                "config_key": entry["config"]["key"],
                "label": entry["config"]["label"],
                "first_seen_ts": entry["first_ts"],
                "first_situation_id":
                    entry["first_situation_id"],
                "cycles": entry["count"],
            }
        return records, strongest


# ------------------------------------------------------------ the cycle

def evaluate_book(book, *, ts, situation_id=None,
                  situation=None, tracker=None,
                  cfg: BookJudgmentConfig | None = None
                  ) -> tuple[list[dict], dict]:
    """One book-judgment cycle (H1 close): resolve every
    competing pair, judge every lone triggered scenario, and
    read coverage against the current situation.  Returns
    (records, summary_extras) — the extras ride the book's
    own summary record (``scenario_book``), so the cycle's
    outcome counts and its strongest gap are part of the
    book's per-H1 account, not a side channel."""
    cfg = cfg or BookJudgmentConfig()
    live = book.live()
    by_id = {sc["scenario_id"]: sc for sc in live}
    records: list[dict] = []
    counts = {"pairs": 0, "pairs_taken": 0, "pairs_abstained": 0,
              "lone_triggered": 0, "lone_taken": 0,
              "lone_abstained": 0}
    paired_ids: set[str] = set()
    for key in sorted(book.relationships):
        rel = book.relationships[key]
        if rel.get("kind") != "competing":
            continue
        a = by_id.get(rel.get("a"))
        b = by_id.get(rel.get("b"))
        if a is None or b is None:
            continue
        paired_ids.update((a["scenario_id"], b["scenario_id"]))
        record = judge_pair(a, b, ts=ts,
                            situation_id=situation_id, cfg=cfg)
        records.append(record)
        counts["pairs"] += 1
        counts["pairs_taken" if record["verdict"] == "take"
               else "pairs_abstained"] += 1
    for scenario in live:
        if scenario.get("state") != "triggered":
            continue
        if scenario["scenario_id"] in paired_ids:
            continue
        record = judge_lone_triggered(
            scenario, ts=ts, situation_id=situation_id, cfg=cfg)
        records.append(record)
        counts["lone_triggered"] += 1
        counts["lone_taken" if record["verdict"] == "take"
               else "lone_abstained"] += 1
    extras: dict = {"judgment": counts}
    if tracker is not None and situation is not None:
        gap_records, strongest = tracker.observe(
            situation, live, ts=ts,
            situation_id=situation_id, cfg=cfg)
        records += gap_records
        if strongest is not None:
            extras["coverage_gap"] = strongest
    return records, extras
