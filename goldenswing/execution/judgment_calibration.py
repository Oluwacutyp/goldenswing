"""Judgment Core — measured priors (MarketSituation Pass 4;
SPEC §5.3 + AMENDMENTS-1 §A3).

The S1 core's scenario priors were stated constants
(continuation 0.42 / reversal 0.38 / chop 0.20 —
``judgment.PRIOR_*``) with the module's own docstring naming
the debt: "the shadow journal is the dataset a later fitted
rung must beat".  This module is that rung's measurement half:

- From the S3 reconciliation book (judgment verdicts paired to
  forward outcomes in the candidate's own geometry — the
  ``judgment_validation.build_pairs`` join, reused, never
  re-implemented) it computes empirical continuation /
  reversal / chop rates per cell = (structure class x
  candidate direction relative to the market direction the
  judgment record itself journaled).
- Shrinkage, stated (SPEC §5.3): a cell's prior is
  ``(n * measured + N0 * stated) / (n + N0)`` with
  ``N0 = 100`` — with the §A3 amendment that ``n`` is the
  cell's EFFECTIVE sample size, never the raw count.
- The measured set runs behind a gate with the house's
  promotion shape: FIXED constants by default; measured priors
  in force only when config allows (``judgment_priors =
  "calibrated"``) AND the locked-holdout comparison
  (``compare_prior_sets``) has journaled a WIN for measured.
  No verdict, or a loss, and the stated constants rule — the
  prior of last resort, never deleted.

§A3 statistical-honesty pack (each rule pinned by tests):

1. UNIVERSE = every judged candidate — verdict-take,
   verdict-abstain AND shadow-evaluated (a verdict that never
   carried authority is still a judged candidate with a
   forward outcome).  The measurement is built from
   judgment_shadow ⋈ judgment_reconciliation pairs; the gate
   stack's execution result is journaled on each pair for the
   record, never used as a filter.  A rate computed from
   executed trades only is impossible by construction here:
   no code path in this module subsets on execution.
2. EFFECTIVE SAMPLE SIZE: candidates judged inside one
   market episode share most of their forward price path, so
   their outcomes are related observations, not independent
   ones.  The episode key is the situation_id the shadow
   record carries (Pass 0's stamp — every candidate judged
   under one situation snapshot is one episode); records
   predating the stamp fall back to the judged hour.  Inside
   a cell, each episode contributes ONE observation (the mean
   of its pairs' outcome indicators); a cell reports raw n
   AND effective n, and shrinkage blends on effective n.
   (Episodes closer than the +4h horizon still share path —
   the episode rule is the stated, conservative grain, named
   here rather than hidden.)
3. CELLS: the (structure x direction) grid is the base.
   Family (playbook) / timeframe (intended_tf) / regime
   (volatility class) sub-cells are computed and journaled
   beside it, and the provider resolves through one ONLY
   where its effective n clears ``SUBCELL_EFF_N_FLOOR``;
   below the floor the pair's evidence pools upward — it is
   in the base cell by construction, and resolution falls
   back to the base cell.
4. BLENDING is the shrinkage above, on effective n, toward
   the stated constants, which remain the prior of last
   resort forever.

Outcome classification (stated): a judged candidate's forward
outcome, in its own geometry at the S3 horizon (+4h):
realized_r >= +0.5 -> continuation; realized_r <= -0.5 ->
reversal; in between -> chop.  Half the risk unit is the
desk's own "progress" fraction (the profit-lock doctrine's
0.50); a move that reached neither half-way to target-ward
nor half-way to the stop within the horizon is chop for
decision purposes.  Pairs whose realized_r is absent
(pending, direction-only, degenerate-R excluded by the S3/S4
guard) do not classify and are counted as unclassed, never
silently dropped.

Direction-relative cell axis (stated): the judgment record
journals ``market_state.regime_direction`` ("up"/"down"/
"flat" — the market direction its own estimation derived)
and the reconciliation journals the candidate's direction;
the pair's axis is with_bias / against_bias / no_bias from
those two journaled facts.  (The raw bias_d1 the core weighed
is not journaled on the shadow record; the regime direction
is the journaled market direction, and this module measures
only what the journal carries.)

The holdout comparison scores PROBABILITY QUALITY, the thing
priors are: on the locked holdout (the same chronological
last-30% split judgment_validation uses), each pair's
journaled scenario probabilities were computed under the
stated priors, and the evidence likelihoods are prior-free
(``w_i = prior_i * L_i``), so the measured-prior probabilities
for the identical evidence are recovered exactly:
``p'_i ~ measured_i * (p_i / stated_i)``, renormalised.  Both
sides are scored by mean log-loss (Brier beside it) against
the classified outcome.  Measured WINS only by strictly
lower holdout log-loss with at least BAR_MIN_HOLDOUT classed
holdout pairs — a tie or a thin holdout holds the fixed
priors.  Deterministic throughout (D9): same journal in,
same numbers out.
"""
from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

from .judgment import (
    PRIOR_CHOP,
    PRIOR_CONTINUATION,
    PRIOR_REVERSAL,
)
from .judgment_validation import (
    BAR_MIN_HOLDOUT, build_pairs, gate_class)

#: Record types this module journals.
CALIBRATION_RECORD = "judgment_priors"
VERDICT_RECORD = "judgment_priors_verdict"

#: Shrinkage prior strength, paired outcomes (SPEC §5.3).
PRIOR_STRENGTH_N0 = 100.0

#: Outcome classification band, in R of the candidate's own
#: geometry (see module docstring).
OUTCOME_BAND_R = 0.5

#: A sub-cell separates from its parent cell only at or above
#: this effective sample size (§A3.3's stated floor).
SUBCELL_EFF_N_FLOOR = 25.0

#: Measurement window: pairs older than this are pruned from
#: the standing state (the stated constants predate it; the
#: measured set describes the recent desk).
TRAILING_DAYS = 90

#: Pairs complete at the S3 grading horizon (build_pairs'
#: default; the desk's decision horizon).
HORIZON_H = 4

#: Standing-state caps (the audit state's own discipline).
_PAIRS_CAP = 6000
_SHADOWS_CAP = 2000
#: A shadow whose reconciliation has not landed within this
#: many hours is stale evidence; the horizon is +4h and the
#: audit cycle lands it within a cycle or two.
_SHADOW_STALE_HOURS = 48

STATED_PRIORS = {
    "continuation": PRIOR_CONTINUATION,
    "reversal": PRIOR_REVERSAL,
    "chop": PRIOR_CHOP,
}
OUTCOMES = ("continuation", "reversal", "chop")

VERDICT_MEASURED_WINS = "measured_wins"
VERDICT_FIXED_HOLDS = "fixed_holds"
VERDICT_INSUFFICIENT = "insufficient_data"


# ------------------------------------------------------------ derivations
def rel_direction(regime_direction, direction) -> str:
    """The pair's direction axis from two journaled facts:
    the judgment record's market direction word and the
    candidate's signed direction."""
    word = {"up": 1, "down": -1}.get(
        str(regime_direction or "").lower())
    try:
        sign = int(direction or 0)
    except (TypeError, ValueError):
        sign = 0
    if word is None or sign not in (1, -1):
        return "no_bias"
    return "with_bias" if sign == word else "against_bias"


def cell_key(structure, rel_dir) -> str:
    return f"{structure or 'unknown'}|{rel_dir or 'no_bias'}"


def classify_outcome(realized_r) -> str | None:
    """The forward outcome's scenario class, or None when the
    pair carries no classable R (pending / direction-only /
    degenerate-excluded)."""
    if realized_r is None:
        return None
    try:
        r = float(realized_r)
    except (TypeError, ValueError):
        return None
    if r >= OUTCOME_BAND_R:
        return "continuation"
    if r <= -OUTCOME_BAND_R:
        return "reversal"
    return "chop"


def episode_key(pair: dict) -> str:
    """The correlation key (§A3.2): one market episode = one
    effective observation."""
    sit = pair.get("situation_id")
    if sit:
        return f"sit:{sit}"
    return f"hour:{str(pair.get('ts') or '')[:13]}"


# ------------------------------------------------------------ pairs
def build_calibration_pairs(records: list[dict],
                            horizon_h: int = HORIZON_H
                            ) -> list[dict]:
    """The S3 pairs, extended with the calibration fields.

    The join, the degenerate-R exclusion and the gate
    classification are judgment_validation.build_pairs' own —
    reused verbatim.  This layer joins the judgment record's
    journaled facts (structure, market direction, scenario
    probabilities, authority) and the reconciliation's
    candidate direction onto each pair by the same
    (ts, playbook) key, and derives the cell + outcome class.
    """
    base = build_pairs(records, horizon_h=horizon_h)
    shadows: dict[tuple[str, str], dict] = {}
    directions: dict[tuple[str, str], Any] = {}
    for row in records:
        key = (str(row.get("ts") or row.get("judged_ts") or ""),
               str(row.get("playbook") or ""))
        if row.get("type") == "judgment_shadow":
            shadows.setdefault(key, row)
        elif row.get("type") == "judgment_reconciliation" \
                and int(row.get("horizon_h") or 0) == int(horizon_h):
            directions[key] = row.get("direction")
    out: list[dict] = []
    for pair in base:
        key = (pair["ts"], pair["playbook"])
        shadow = shadows.get(key) or {}
        judgment = shadow.get("judgment") or {}
        market = judgment.get("market_state") or {}
        direction = directions.get(key)
        structure = str(market.get("structure") or "unknown")
        rel = rel_direction(market.get("regime_direction"),
                            direction)
        scenarios = {}
        for scen in judgment.get("scenarios") or []:
            if isinstance(scen, dict) and scen.get("name"):
                scenarios[str(scen["name"])] = scen.get("p")
        out.append({
            **pair,
            "direction": direction,
            "structure": structure,
            "regime_direction": market.get("regime_direction"),
            "volatility": market.get("volatility"),
            "rel_dir": rel,
            "cell": cell_key(structure, rel),
            "outcome": classify_outcome(pair.get("realized_r")),
            "mode": judgment.get("mode"),
            "authority": judgment.get("authority"),
            "shadow_only": (
                judgment.get("mode") == "shadow"
                or judgment.get("authority") == "shadow_declared"),
            "situation_id": shadow.get("situation_id"),
            "scenarios": scenarios,
        })
    return out


# ------------------------------------------------------------ measurement
def _cell_stats(pairs: list[dict]) -> dict[str, Any]:
    """Episode-collapsed rates + shrinkage for one pair set."""
    episodes: dict[str, list[dict]] = {}
    for pair in pairs:
        episodes.setdefault(episode_key(pair), []).append(pair)
    eff_n = float(len(episodes))
    totals = {name: 0.0 for name in OUTCOMES}
    for rows in episodes.values():
        share = 1.0 / len(rows)
        for row in rows:
            totals[str(row["outcome"])] += share
    measured = {name: (round(totals[name] / eff_n, 6) if eff_n
                       else 0.0) for name in OUTCOMES}
    priors = {}
    denom = eff_n + PRIOR_STRENGTH_N0
    for name in OUTCOMES:
        priors[name] = round(
            (totals[name]
             + PRIOR_STRENGTH_N0 * STATED_PRIORS[name]) / denom, 6)
    return {"n": len(pairs), "eff_n": eff_n,
            "episodes": len(episodes),
            "measured": measured, "priors": priors}


def calibrate(pairs: list[dict]) -> dict[str, Any]:
    """The calibration body over a pair set (the caller adds
    type/bar/run_id).  Pure: same pairs in, same body out."""
    classed = [p for p in pairs if p.get("outcome")]
    cells: dict[str, Any] = {}
    by_cell: dict[str, list[dict]] = {}
    for pair in classed:
        by_cell.setdefault(str(pair["cell"]), []).append(pair)
    for key in sorted(by_cell):
        cells[key] = _cell_stats(by_cell[key])
    sub_cells: dict[str, Any] = {}
    dims = {"family": "playbook", "tf": "intended_tf",
            "regime": "volatility"}
    for dim, field in dims.items():
        groups: dict[str, list[dict]] = {}
        for pair in classed:
            value = pair.get(field)
            if value is None or value == "":
                continue
            groups.setdefault(
                f"{pair['cell']}@{dim}={value}", []).append(pair)
        for key in sorted(groups):
            stats = _cell_stats(groups[key])
            stats["cleared"] = stats["eff_n"] >= SUBCELL_EFF_N_FLOOR
            stats["pooled_into"] = key.split("@", 1)[0]
            sub_cells[key] = stats
    universe = {
        "n_judged": len(pairs),
        "n_with_outcome": sum(
            1 for p in pairs if p.get("realized_r") is not None),
        "n_classed": len(classed),
        "n_unclassed": len(pairs) - len(classed),
        "verdict_take": sum(
            1 for p in pairs
            if p.get("judgment_decision") == "take"),
        "verdict_abstain": sum(
            1 for p in pairs
            if p.get("judgment_decision") == "abstain"),
        "gate_executed": sum(
            1 for p in pairs if p.get("gate") == "executed"),
        "shadow_only": sum(
            1 for p in pairs if p.get("shadow_only")),
        "excluded_degenerate_r": sum(
            1 for p in pairs if p.get("r_excluded")),
    }
    return {
        "window": {"trailing_days": TRAILING_DAYS,
                   "horizon_h": HORIZON_H,
                   "n0": PRIOR_STRENGTH_N0,
                   "outcome_band_r": OUTCOME_BAND_R,
                   "subcell_eff_n_floor": SUBCELL_EFF_N_FLOOR},
        "universe": universe,
        "stated_priors": dict(STATED_PRIORS),
        "cells": cells,
        "sub_cells": sub_cells,
        "note": (
            "Measured over EVERY judged candidate with a "
            "classable +4h outcome (taken, abstained and "
            "shadow-evaluated; execution is never a filter). "
            "Rates are episode-collapsed: one situation = one "
            "effective observation; shrinkage blends on "
            "effective n toward the stated constants (N0=100), "
            "which remain the prior of last resort."),
    }


# ------------------------------------------------------------ the holdout gate
def _rebase(scenarios: dict, priors: dict) -> dict | None:
    """Measured-prior probabilities for the identical evidence
    whose stated-prior probabilities were journaled."""
    weights = {}
    for name in OUTCOMES:
        p = scenarios.get(name)
        if p is None:
            return None
        try:
            p = float(p)
        except (TypeError, ValueError):
            return None
        weights[name] = float(priors[name]) * (
            p / STATED_PRIORS[name])
    total = sum(weights.values())
    if total <= 0:
        return None
    return {name: w / total for name, w in weights.items()}


def _scores(rows: list[tuple[dict, str]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "log_loss": None, "brier": None}
    log_loss = 0.0
    brier = 0.0
    for probs, outcome in rows:
        log_loss += -math.log(max(probs[outcome], 1e-9))
        brier += sum(
            (probs[name] - (1.0 if name == outcome else 0.0)) ** 2
            for name in OUTCOMES)
    n = len(rows)
    return {"n": n, "log_loss": round(log_loss / n, 6),
            "brier": round(brier / n, 6)}


def compare_prior_sets(pairs: list[dict]) -> dict[str, Any]:
    """The locked-holdout comparison over calibration pairs.

    Chronology is the honesty mechanism (judgment_validation's
    own split): the final 30% of classed pairs by judged time
    is the holdout; the calibration the measured side prices
    from is built ONLY from pairs at or before the boundary.
    """
    classed = sorted(
        (p for p in pairs
         if p.get("outcome") and p.get("scenarios")),
        key=lambda p: str(p.get("ts") or ""))
    split = int(len(classed) * (1.0 - 0.30))
    if split:
        boundary = str(classed[split - 1].get("ts") or "")
        eval_pairs = [p for p in classed
                      if str(p.get("ts") or "") <= boundary]
        holdout = [p for p in classed
                   if str(p.get("ts") or "") > boundary]
    else:
        eval_pairs, holdout = [], list(classed)
    calibration = calibrate(eval_pairs)
    provider = PriorsProvider(
        calibration=calibration,
        verdict={"verdict": VERDICT_MEASURED_WINS},
        mode="calibrated")
    stated_rows: list[tuple[dict, str]] = []
    measured_rows: list[tuple[dict, str]] = []
    for pair in holdout:
        scen = {name: pair["scenarios"].get(name)
                for name in OUTCOMES}
        if any(v is None for v in scen.values()):
            continue
        stated_probs = {name: float(scen[name]) for name in OUTCOMES}
        total = sum(stated_probs.values())
        if total <= 0:
            continue
        stated_probs = {name: v / total
                        for name, v in stated_probs.items()}
        resolved = provider.for_cell(
            pair["structure"], pair["rel_dir"],
            playbook=pair.get("playbook"),
            intended_tf=pair.get("intended_tf"),
            volatility=pair.get("volatility"))
        measured_probs = _rebase(scen, resolved)
        if measured_probs is None:
            continue
        outcome = str(pair["outcome"])
        stated_rows.append((stated_probs, outcome))
        measured_rows.append((measured_probs, outcome))
    stated = _scores(stated_rows)
    measured = _scores(measured_rows)
    if measured["n"] < BAR_MIN_HOLDOUT:
        verdict = VERDICT_INSUFFICIENT
    elif measured["log_loss"] < stated["log_loss"]:
        verdict = VERDICT_MEASURED_WINS
    else:
        verdict = VERDICT_FIXED_HOLDS
    return {
        "verdict": verdict,
        "holdout_n": measured["n"],
        "eval_pairs": len(eval_pairs),
        "stated": stated,
        "measured": measured,
        "bar": {"min_holdout": BAR_MIN_HOLDOUT,
                "rule": "measured mean log-loss strictly below "
                        "stated on the locked chronological "
                        "holdout; a tie or a thin holdout holds "
                        "the fixed priors"},
        "note": (
            "Probability quality on identical evidence: the "
            "journaled stated-prior probabilities carry the "
            "evidence likelihoods (prior-free), so the measured "
            "side is the same evidence re-priced under the "
            "measured priors — recovered exactly, never "
            "re-estimated. The measured priors were calibrated "
            "only on pairs at or before the holdout boundary."),
    }


# ------------------------------------------------------------ the provider
class PriorsProvider:
    """The one priors source the assessor consults.

    In force: MEASURED only when mode == "calibrated", a
    calibration exists, and the latest holdout verdict is a
    measured win.  Every other state resolves to the stated
    constants — with the candidate's cell and that cell's
    effective n still reported, so every verdict can journal
    its priors trace under both settings (SPEC §5.3).
    """

    def __init__(self, *, calibration: dict | None = None,
                 verdict: dict | None = None,
                 mode: str = "stated",
                 version: Any = None):
        self._calibration = calibration or {}
        self._verdict = verdict or {}
        self._mode = str(mode or "stated")
        self._version = version

    @property
    def in_force(self) -> str:
        if self._mode == "calibrated" \
                and self._calibration.get("cells") \
                and self._verdict.get("verdict") \
                == VERDICT_MEASURED_WINS:
            return "measured"
        return "stated"

    def for_cell(self, structure, rel_dir, *, playbook=None,
                 intended_tf=None, volatility=None) -> dict:
        base_key = cell_key(structure, rel_dir)
        cells = self._calibration.get("cells") or {}
        base = cells.get(base_key) or {}
        if self.in_force == "measured":
            subs = self._calibration.get("sub_cells") or {}
            for dim, value in (("family", playbook),
                               ("tf", intended_tf),
                               ("regime", volatility)):
                if value is None or value == "":
                    continue
                sub = subs.get(f"{base_key}@{dim}={value}")
                if sub and sub.get("cleared"):
                    return {**sub["priors"], "source": "measured",
                            "version": self._version,
                            "cell": f"{base_key}@{dim}={value}",
                            "n": sub["eff_n"]}
            if base:
                return {**base["priors"], "source": "measured",
                        "version": self._version, "cell": base_key,
                        "n": base["eff_n"]}
        return {**STATED_PRIORS, "source": "stated",
                "version": "stated", "cell": base_key,
                "n": float(base.get("eff_n") or 0.0)}


# ------------------------------------------------------------ standing state
def new_state() -> dict:
    """The standing calibration state (rides inside the S3
    audit state file, under its own key — the range-promotion
    arrangement)."""
    return {
        "cycles": 0,
        "damaged_cycles": 0,
        "shadows": {},
        "pairs": [],
        "version": 0,
        "last_calibration": None,
        "last_journaled_calibration": None,
        "last_verdict": None,
        "last_journaled_verdict": None,
        "last_cycle_bar": None,
    }


def _compact_pair(shadow: dict, recon: dict) -> dict | None:
    judgment = shadow.get("judgment") or {}
    market = judgment.get("market_state") or {}
    realized = recon.get("realized_r")
    if recon.get("r_excluded"):
        realized = None
    direction = recon.get("direction")
    structure = str(market.get("structure") or "unknown")
    rel = rel_direction(market.get("regime_direction"), direction)
    scenarios = {}
    for scen in judgment.get("scenarios") or []:
        if isinstance(scen, dict) and scen.get("name"):
            scenarios[str(scen["name"])] = scen.get("p")
    return {
        "ts": str(shadow.get("ts") or ""),
        "playbook": str(shadow.get("playbook") or ""),
        "intended_tf": shadow.get("intended_tf"),
        "direction": direction,
        "structure": structure,
        "regime_direction": market.get("regime_direction"),
        "volatility": market.get("volatility"),
        "rel_dir": rel,
        "cell": cell_key(structure, rel),
        "judgment_decision": judgment.get("decision"),
        "expected_r": judgment.get("expected_r"),
        "confidence": judgment.get("confidence"),
        "gate": gate_class(shadow),
        "realized_r": realized,
        "r_excluded": recon.get("r_excluded"),
        "hit": recon.get("hit"),
        "outcome": classify_outcome(realized),
        "mode": judgment.get("mode"),
        "authority": judgment.get("authority"),
        "shadow_only": (
            judgment.get("mode") == "shadow"
            or judgment.get("authority")
            == "shadow_declared"),
        "situation_id": shadow.get("situation_id"),
        "scenarios": scenarios,
    }


def _fold(state: dict, records: list[dict]) -> None:
    shadows = state["shadows"]
    for row in records:
        if row.get("type") != "judgment_shadow":
            continue
        key = f"{row.get('ts') or ''}|{row.get('playbook') or ''}"
        if key not in shadows:
            shadows[key] = row
    done_keys = []
    known = {f"{p['ts']}|{p['playbook']}" for p in state["pairs"]}
    for row in records:
        if row.get("type") != "judgment_reconciliation":
            continue
        if int(row.get("horizon_h") or 0) != HORIZON_H:
            continue
        key = f"{row.get('judged_ts') or ''}|{row.get('playbook') or ''}"
        if key in known:
            continue
        shadow = shadows.get(key)
        if shadow is None:
            continue
        pair = _compact_pair(shadow, row)
        if pair is not None:
            state["pairs"].append(pair)
            known.add(key)
            done_keys.append(key)
    for key in done_keys:
        shadows.pop(key, None)
    overflow = len(state["pairs"]) - _PAIRS_CAP
    if overflow > 0:
        del state["pairs"][:overflow]
    overflow = len(shadows) - _SHADOWS_CAP
    if overflow > 0:
        for key in list(shadows)[:overflow]:
            del shadows[key]


def _prune(state: dict, now) -> None:
    try:
        cutoff = str(now - timedelta(days=TRAILING_DAYS))[:10]
    except TypeError:
        return
    state["pairs"] = [
        p for p in state["pairs"]
        if str(p.get("ts") or "")[:10] >= cutoff]
    try:
        stale = str(now - timedelta(
            hours=_SHADOW_STALE_HOURS))[:13]
    except TypeError:
        return
    state["shadows"] = {
        key: row for key, row in state["shadows"].items()
        if str(row.get("ts") or "")[:13] >= stale}


def cycle(state: dict, records: list[dict], *, now=None,
          damaged_lines: int = 0, window: dict | None = None,
          run_id=None, bar=None) -> list[dict]:
    """One calibration cycle over one journal window (the
    self-audit's contract): mutate ``state``, return the
    journal records to append — the calibration record and/or
    the verdict record, each only when its content changed, so
    a quiet hour journals nothing.  The FIRST cycle always
    journals both: those records are the deployment markers
    stating the measurement window has opened.  A damaged
    window degrades the cycle (counted, nothing folded)."""
    for key, value in new_state().items():
        state.setdefault(key, value)
    if damaged_lines:
        state["damaged_cycles"] = int(
            state.get("damaged_cycles") or 0) + 1
        return []
    _fold(state, records)
    if now is not None:
        _prune(state, now)
    state["last_cycle_bar"] = bar
    out: list[dict] = []
    body = calibrate(state["pairs"])
    if body != state.get("last_journaled_calibration"):
        state["version"] = int(state.get("version") or 0) + 1
        record = {"type": CALIBRATION_RECORD,
                  "version": state["version"], "bar": bar,
                  "run_id": run_id, "window_meta": window, **body}
        state["last_calibration"] = record
        state["last_journaled_calibration"] = body
        out.append(record)
    elif state.get("last_calibration") is None:
        state["last_calibration"] = {
            "type": CALIBRATION_RECORD,
            "version": int(state.get("version") or 0), **body}
    verdict_body = compare_prior_sets(state["pairs"])
    if verdict_body != state.get("last_journaled_verdict"):
        record = {"type": VERDICT_RECORD, "bar": bar,
                  "run_id": run_id,
                  "calibration_version": state["version"],
                  **verdict_body}
        state["last_verdict"] = record
        state["last_journaled_verdict"] = verdict_body
        out.append(record)
    elif state.get("last_verdict") is None:
        state["last_verdict"] = {
            "type": VERDICT_RECORD,
            "calibration_version": state["version"],
            **verdict_body}
    state["cycles"] = int(state.get("cycles") or 0) + 1
    return out


def provider_from_state(state: dict, *, mode: str) -> PriorsProvider:
    """The live provider over the standing state."""
    calibration = None
    last = state.get("last_calibration")
    if isinstance(last, dict):
        calibration = last
    return PriorsProvider(
        calibration=calibration,
        verdict=state.get("last_verdict") or {},
        mode=mode,
        version=(last or {}).get("version"))


def evaluate_records(records: list[dict]) -> dict:
    """One-shot measurement over a complete record list (an
    exported journal): calibration + holdout verdict, the same
    pure functions the live cycle runs."""
    pairs = build_calibration_pairs(records)
    return {"calibration": calibrate(pairs),
            "verdict": compare_prior_sets(pairs),
            "pairs": len(pairs)}
