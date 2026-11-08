"""§G range family — PROMOTION EVALUATION (owner program
2026-11-05, the follow-on completion-02 named but did not
build).

The four range playbooks (range_sweep_reclaim, range_qm,
range_mean_reversion, range_compression_breakout) run under
the §G contract: scheduled H4-generation / H1-refinement scans
that journal candidates and hold NO order authority
(playbook_registry.ROUTE_RANGE_H4_H1).  Completion-02 declared
them ``shadow_declared`` judgment routes, so every §G candidate
now carries a route-stamped ``judgment_shadow`` verdict and an
S3 reconciliation, exactly like the other connected routes.
What was missing is any machinery that MEASURES that shadow
history and says whether a route deserves promotion.  This
module is that measurement.  It promotes NOTHING: a
``promotion_ready`` verdict is a recommendation to the owner,
and promotion itself stays the separate explicit step it has
always been (config ``judgment_route_modes`` naming the route
"governing" — the Judgment Core promotion precedent,
2026-11-05).

Method — the evidence pack's discipline, mechanized:

- The dataset is the §G routes' OWN shadow history only:
  ``judgment_shadow`` records whose judgment block carries the
  route stamp (``authority_route``).  Records without the stamp
  predate the wiring (or belong to another route — the same
  playbooks also fire through the governing h1_trigger roster,
  whose verdicts are stamped with THAT route and are a
  different question).  The evaluation window therefore starts
  at this wiring's deployment, stated on every record: no
  pre-wiring history is retroactively evaluated.
- Outcomes are the S4 pairing (judgment_validation.build_pairs)
  at the house grading horizon (+4h): each judged candidate
  pairs to its S3 reconciliation in its own geometry.  Exact
  duplicate journal rows are deduped first (the pack's key:
  ts + playbook + decision + expected_r); two stamped verdicts
  sharing one (ts, playbook) key cannot be told apart by the
  pairing machinery itself, so the first counts and the rest
  are counted as duplicate-key records, never double-graded.
- Degenerate-R pairs are excluded from every statistic and
  counted beside them (completion-04/05 rule): a reconciliation
  the S3 cycle already marked ``r_excluded``, or one whose
  journaled R implies a degenerate stop under
  judgment_audit.degenerate_stop (build_pairs' re-derivation),
  never grades an arm.
- Per route and arm (would-take / would-abstain): n paired,
  hit rate, MEDIAN realized R.  Medians and hit rates, never
  means alone — the pack showed a mean can be carried by one
  degenerate record while the typical outcome says the
  opposite.
- Discrimination is read BOTH ways the pack taught: a route
  whose would-abstains outperform its would-takes (higher
  median R, or higher hit rate) reads INVERTED, and an inverted
  route can never read promotion-ready.

Verdicts (per route):

- ``insufficient_data`` — below the readable floors: either no
  stamped candidates observed at all (stated as such — a route
  that never fires is a different fact from a route that fires
  and fails), or fewer than MIN_PAIRED_PER_ARM paired outcomes
  in either arm, with the current n named.
- ``not_ready`` — measured and failing: would-take median R
  negative, or the discrimination inverted.
- ``promotion_ready`` — measured, right-way discrimination on
  both reads, non-negative would-take median.  A
  RECOMMENDATION, never an action.

The cycle host is the unified self-audit (the runner feeds this
module the same journal windows, inside the same S3 cadence,
persisted in the same state file).  Findings feed back
nowhere: no decision input, no parameter change, no demotion —
the record, /status and the manifest are the whole surface.
"""
from __future__ import annotations

import statistics
from typing import Any

from .judgment_validation import build_pairs
from .playbook_registry import RANGE_SPECS
from .self_audit import SHADOW_MAX_AGE_H

#: The judgment route each range playbook's §G shadow history
#: journals under.  Route name == playbook name (the
#: completion-02 precedent for connected non-roster paths:
#: hot_scalp_promotion / resting_limit are likewise one route
#: per connected path, and the journal's shadow records are
#: keyed by playbook).  The runner's JUDGMENT_ROUTE_STATES
#: declares each of these "shadow_declared"; a test pins the
#: two tables to each other.
RANGE_JUDGMENT_ROUTES: dict[str, str] = {
    spec.name: spec.name for spec in RANGE_SPECS}

#: The §G playbooks, registry order.
RANGE_PLAYBOOKS: tuple[str, ...] = tuple(RANGE_JUDGMENT_ROUTES)

#: The grading horizon (hours).  The house S4 horizon: the
#: self-audit grades at +4h, build_pairs defaults to it, and
#: the promotion evidence pack's 4h read is the one that held
#: up.  Stated on every record — a verdict without its horizon
#: is not a verdict.
HORIZON_H = 4

#: Paired outcomes required PER ARM before discrimination is
#: readable at all.  Mirrors the house floors (E3 trust earns
#: at >= 20 resolutions; the S4 holdout bar is 20): below it a
#: median is a coin flip wearing a number.
MIN_PAIRED_PER_ARM = 20

#: Journal record type for one evaluation snapshot.
RECORD_TYPE = "range_promotion_evaluation"

#: Standing-state caps (the state file is JSON and rides every
#: S3 save): paired R values kept per arm (medians need the
#: distribution; the cap is 50x the readable floor), pending
#: shadows carried across windows, folded keys remembered so a
#: duplicate journal row can never grade twice.
_R_CAP = 1000
_PENDING_CAP = 500
_KEY_CAP = 4000

VERDICT_INSUFFICIENT = "insufficient_data"
VERDICT_NOT_READY = "not_ready"
VERDICT_READY = "promotion_ready"

_WINDOW_BASIS = (
    "§G route-stamped judgment_shadow records only "
    "(judgment.authority_route naming the route).  Records "
    "without the stamp predate the shadow wiring or belong to "
    "another route and are not evaluated: the window starts "
    "at this wiring's deployment.")


# ------------------------------------------------------------ state

def new_state() -> dict:
    """The standing evaluation state (rides inside the S3
    audit state file, under its own key — the self-audit's
    arrangement)."""
    return {
        "cycles": 0,
        "damaged_cycles": 0,
        "window_start": None,
        "pending": [],
        "folded_keys": [],
        "routes": {},
        "last_snapshot": None,
        "last_journaled": None,
        "last_cycle_bar": None,
    }


def _route_slot(state: dict, playbook: str) -> dict:
    slot = state["routes"].setdefault(playbook, {
        "evaluated": 0,
        "take_r": [], "take_hits": 0,
        "abstain_r": [], "abstain_hits": 0,
        "excluded_degenerate_r": 0,
        "duplicate_key_records": 0,
        "unmeasurable_expired": 0,
        "dropped_pending": 0,
    })
    return slot


def _ts(value) -> Any:
    """Tolerant timestamp parse (the self-audit's rule: an
    unreadable clock is a missing fact, never a guessed one)."""
    if value is None:
        return None
    try:
        import pandas as pd

        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else ts
    except Exception:
        return None


def _key(ts: str, playbook: str) -> str:
    return f"{ts}|{playbook}"


# ------------------------------------------------------------ verdict

def _arm(r_values: list, hits: int) -> dict:
    n = len(r_values)
    return {
        "paired": n,
        "hits": int(hits),
        "hit_rate": (round(int(hits) / n, 4) if n else None),
        "median_r": (round(float(statistics.median(r_values)), 4)
                     if n else None),
    }


def verdict_for(slot: dict, *, family: str = "§G",
                binds: bool = False) -> tuple[str, str]:
    """One route's verdict from its accumulated arms — the
    rule, exactly as coded (see the module docstring).

    ``family`` names the route family in the no-candidates
    reason (the range family is §G; route_promotion passes its
    own label).  ``binds`` (Pass 5, route_promotion only)
    changes the promotion_ready reason's tail: a binding
    evaluation's verdict is an authority effect under the
    owner's 2026-11-08 ruling, not a recommendation — the
    BAR itself (floors, medians, discrimination) is identical
    either way; one evaluator, one rule."""
    evaluated = int(slot.get("evaluated") or 0)
    if evaluated == 0:
        return VERDICT_INSUFFICIENT, (
            f"no candidates observed — the {family} route has "
            "journaled no core-evaluated candidates since the "
            "evaluation window opened")
    take = _arm(slot.get("take_r") or [],
                slot.get("take_hits") or 0)
    abstain = _arm(slot.get("abstain_r") or [],
                   slot.get("abstain_hits") or 0)
    floor = MIN_PAIRED_PER_ARM
    if take["paired"] < floor or abstain["paired"] < floor:
        return VERDICT_INSUFFICIENT, (
            f"paired outcomes below the readable floor of "
            f"{floor} per arm at +{HORIZON_H}h: would-takes "
            f"{take['paired']}/{floor}, would-abstains "
            f"{abstain['paired']}/{floor} (degenerate-R pairs "
            f"excluded, outcomes still landing are not counted)")
    take_med = float(take["median_r"])
    abstain_med = float(abstain["median_r"])
    if take_med < 0:
        return VERDICT_NOT_READY, (
            f"measured and failing: would-take median "
            f"{take_med:+.2f}R is negative over "
            f"{take['paired']} paired outcomes — the core's "
            f"takes on this route lose on the typical outcome")
    inverted = abstain_med > take_med \
        or float(abstain["hit_rate"]) > float(take["hit_rate"])
    if inverted:
        return VERDICT_NOT_READY, (
            f"measured and failing: INVERTED discrimination — "
            f"would-abstains (median {abstain_med:+.2f}R, hit "
            f"rate {abstain['hit_rate']:.0%}) outperform "
            f"would-takes (median {take_med:+.2f}R, hit rate "
            f"{take['hit_rate']:.0%}); a route whose abstains "
            f"beat its takes can never read promotion-ready")
    if binds:
        return VERDICT_READY, (
            f"measured, right-way discrimination on both "
            f"reads: would-takes median {take_med:+.2f}R / "
            f"hit rate {take['hit_rate']:.0%} over "
            f"{take['paired']} paired, would-abstains median "
            f"{abstain_med:+.2f}R / hit rate "
            f"{abstain['hit_rate']:.0%} over "
            f"{abstain['paired']} paired — under the owner's "
            f"2026-11-08 ruling this verdict BINDS: the route "
            f"is promoted to governing through the judgment "
            f"seam until a fresh evaluation or an invariant "
            f"violation demotes it")
    return VERDICT_READY, (
        f"measured, right-way discrimination on both reads: "
        f"would-takes median {take_med:+.2f}R / hit rate "
        f"{take['hit_rate']:.0%} over {take['paired']} paired, "
        f"would-abstains median {abstain_med:+.2f}R / hit "
        f"rate {abstain['hit_rate']:.0%} over "
        f"{abstain['paired']} paired — a RECOMMENDATION to the "
        f"owner; promotion is a separate explicit decision "
        f"(config judgment_route_modes), never this record's "
        f"action")


def snapshot(state: dict, *, routes_map: dict | None = None,
             playbooks: tuple | None = None,
             basis: str | None = None, family: str = "§G",
             binds: bool = False) -> dict:
    """The per-route verdict table, as journaled / rendered /
    manifested.  Every registered range route is present —
    absence of data is a stated verdict, never a missing row.

    The keyword parameters are the Pass-5 generalisation
    (execution/route_promotion.py reuses this ONE evaluator
    for the hot_scalp_promotion / resting_limit routes): a
    different playbook→route map, playbook set, window basis
    and verdict voice — the computation is unchanged."""
    routes_map = routes_map or RANGE_JUDGMENT_ROUTES
    playbooks = playbooks or RANGE_PLAYBOOKS
    routes: dict[str, Any] = {}
    pending_counts: dict[str, int] = {}
    for entry in state.get("pending") or []:
        pb = str(entry.get("playbook") or "")
        pending_counts[pb] = pending_counts.get(pb, 0) + 1
    for playbook in playbooks:
        slot = (state.get("routes") or {}).get(playbook) or {}
        verdict, reason = verdict_for(
            slot, family=family, binds=binds)
        routes[playbook] = {
            "route": routes_map[playbook],
            "playbook": playbook,
            "authority": "shadow_declared",
            "evaluated": int(slot.get("evaluated") or 0),
            "pending_outcomes": pending_counts.get(playbook, 0),
            "take": _arm(slot.get("take_r") or [],
                         slot.get("take_hits") or 0),
            "abstain": _arm(slot.get("abstain_r") or [],
                            slot.get("abstain_hits") or 0),
            "excluded_degenerate_r": int(
                slot.get("excluded_degenerate_r") or 0),
            "duplicate_key_records": int(
                slot.get("duplicate_key_records") or 0),
            "unmeasurable_expired": int(
                slot.get("unmeasurable_expired") or 0),
            "verdict": verdict,
            "verdict_reason": reason,
        }
    return {
        "horizon_h": HORIZON_H,
        "floors": {"min_paired_per_arm": MIN_PAIRED_PER_ARM},
        "window_start": state.get("window_start"),
        "window_basis": basis or _WINDOW_BASIS,
        "routes": routes,
    }


# ------------------------------------------------------------ accumulate

def _stamped_shadow(row: dict, routes_map: dict | None = None
                    ) -> str | None:
    """The §G route a journal row belongs to, or None: the row
    must be a judgment_shadow whose judgment block carries the
    route stamp for one of the range routes.  ``routes_map``
    is the Pass-5 generalisation (playbook -> route); the
    default is the range family's map."""
    routes_map = routes_map or RANGE_JUDGMENT_ROUTES
    if row.get("type") != "judgment_shadow":
        return None
    playbook = str(row.get("playbook") or "")
    route = routes_map.get(playbook)
    if route is None:
        return None
    judgment = row.get("judgment") or {}
    if str(judgment.get("authority_route") or "") != route:
        return None
    return playbook


def _shadow_row(row: dict) -> dict:
    """The minimal shadow row build_pairs needs, kept in the
    standing state across windows (JSON-shaped)."""
    judgment = row.get("judgment") or {}
    return {
        "type": "judgment_shadow",
        "ts": str(row.get("ts") or ""),
        "playbook": str(row.get("playbook") or ""),
        "intended_tf": row.get("intended_tf"),
        "gate_result": row.get("gate_result"),
        "gate_order_result": row.get("gate_order_result"),
        "gate_reason_code": row.get("gate_reason_code"),
        "judgment": {
            "decision": judgment.get("decision"),
            "expected_r": judgment.get("expected_r"),
            "confidence": judgment.get("confidence"),
        },
    }


def accumulate(state: dict, records: list[dict], *,
               now=None, routes_map: dict | None = None,
               playbooks: tuple | None = None,
               basis: str | None = None, family: str = "§G",
               binds: bool = False) -> dict:
    """Fold one journal window into the standing state and
    return the new snapshot.

    Mirrors the self-audit's carry discipline: stamped shadows
    are carried in ``pending`` across windows until their +4h
    reconciliation lands; a shadow whose outcome never lands
    within SHADOW_MAX_AGE_H is unmeasurable, not pending
    forever.  Folding is keyed and remembered, so re-reading a
    window (or a duplicated journal row) can never grade twice.

    The keyword parameters are the Pass-5 generalisation (see
    snapshot): route_promotion folds the hot_scalp /
    arrival_zone histories through this same machinery.
    """
    for key, value in new_state().items():
        state.setdefault(key, value)
    now_ts = _ts(now) if now is not None else None

    pending = list(state.get("pending") or [])
    pending_keys = {_key(str(e.get("ts") or ""),
                         str(e.get("playbook") or ""))
                    for e in pending}
    folded = set(state.get("folded_keys") or [])
    fresh: list[dict] = []
    seen_exact: set[tuple] = set()
    for e in pending:
        seen_exact.add((
            str(e.get("ts") or ""), str(e.get("playbook") or ""),
            str((e.get("judgment") or {}).get("decision")),
            str((e.get("judgment") or {}).get("expected_r"))))
    for row in records:
        playbook = _stamped_shadow(row, routes_map)
        if playbook is None:
            continue
        judgment = row.get("judgment") or {}
        exact = (str(row.get("ts") or ""), playbook,
                 str(judgment.get("decision")),
                 str(judgment.get("expected_r")))
        if exact in seen_exact:
            # The pack's dedupe: an exact duplicate journal row
            # is not a second candidate.
            continue
        seen_exact.add(exact)
        key = _key(str(row.get("ts") or ""), playbook)
        slot = _route_slot(state, playbook)
        if key in pending_keys or key in folded:
            # A DISTINCT verdict under a key the pairing
            # machinery itself cannot split (h4_generation and
            # h1_refinement can stamp the same tick): counted,
            # never double-graded.
            slot["duplicate_key_records"] = int(
                slot.get("duplicate_key_records") or 0) + 1
            continue
        fresh.append(_shadow_row(row))
        pending_keys.add(key)
        slot["evaluated"] = int(slot.get("evaluated") or 0) + 1
        ts = str(row.get("ts") or "")
        if ts and (state.get("window_start") is None
                   or ts < str(state["window_start"])):
            state["window_start"] = ts

    # S4 pairing over everything carried + this window's rows.
    # First pair wins per key: duplicate-key shadows (counted
    # at intake, never carried) can produce a second pair for
    # the same key out of the raw window rows, and every pair
    # for one key shares the same outcome row by construction
    # (build_pairs keys outcomes on (ts, playbook)) — so the
    # carried/first-seen verdict is the one that grades.
    pairs = build_pairs(pending + fresh + list(records),
                        horizon_h=HORIZON_H)
    resolved: dict[str, dict] = {}
    for pair in pairs:
        key = _key(pair["ts"], pair["playbook"])
        if key in resolved:
            continue
        if pair.get("r_excluded") \
                or pair.get("realized_r") is not None:
            resolved[key] = pair
    # S3-native degenerate marks: a reconciliation row that
    # carries r_excluded itself (its realized_r was never
    # journaled, so build_pairs leaves the pair unresolved) —
    # the exclusion is still the pair's verdict.
    for row in records:
        if row.get("type") != "judgment_reconciliation":
            continue
        if int(row.get("horizon_h") or 0) != HORIZON_H:
            continue
        if not row.get("r_excluded"):
            continue
        key = _key(str(row.get("judged_ts") or ""),
                   str(row.get("playbook") or ""))
        if key not in resolved:
            resolved[key] = {"ts": str(row.get("judged_ts")
                                      or ""),
                             "playbook": str(
                                 row.get("playbook") or ""),
                             "r_excluded": row.get("r_excluded"),
                             "realized_r": None,
                             "hit": row.get("hit"),
                             "judgment_decision": row.get(
                                 "decision")}

    still_pending: list[dict] = []
    for entry in pending + fresh:
        key = _key(str(entry.get("ts") or ""),
                   str(entry.get("playbook") or ""))
        playbook = str(entry.get("playbook") or "")
        slot = _route_slot(state, playbook)
        pair = resolved.get(key)
        if pair is None:
            judged = _ts(entry.get("ts"))
            if judged is not None and now_ts is not None and (
                    now_ts - judged).total_seconds() \
                    > SHADOW_MAX_AGE_H * 3600:
                slot["unmeasurable_expired"] = int(
                    slot.get("unmeasurable_expired") or 0) + 1
                folded.add(key)
            else:
                still_pending.append(entry)
            continue
        folded.add(key)
        if pair.get("r_excluded"):
            slot["excluded_degenerate_r"] = int(
                slot.get("excluded_degenerate_r") or 0) + 1
            continue
        decision = pair.get("judgment_decision")
        realized = float(pair["realized_r"])
        if decision == "take":
            values = slot["take_r"]
            values.append(realized)
            del values[:-_R_CAP]
            if pair.get("hit"):
                slot["take_hits"] = int(
                    slot.get("take_hits") or 0) + 1
        elif decision == "abstain":
            values = slot["abstain_r"]
            values.append(realized)
            del values[:-_R_CAP]
            if pair.get("hit"):
                slot["abstain_hits"] = int(
                    slot.get("abstain_hits") or 0) + 1
    overflow = len(still_pending) - _PENDING_CAP
    if overflow > 0:
        for entry in still_pending[:overflow]:
            slot = _route_slot(
                state, str(entry.get("playbook") or ""))
            slot["dropped_pending"] = int(
                slot.get("dropped_pending") or 0) + 1
            folded.add(_key(str(entry.get("ts") or ""),
                            str(entry.get("playbook") or "")))
        del still_pending[:overflow]
    state["pending"] = still_pending
    folded_list = sorted(folded)
    state["folded_keys"] = folded_list[-_KEY_CAP:]
    return snapshot(state, routes_map=routes_map,
                    playbooks=playbooks, basis=basis,
                    family=family, binds=binds)


# ------------------------------------------------------------ the cycle

def cycle(*, state: dict, records: list[dict], now=None,
          damaged_lines: int = 0, window: dict | None = None,
          run_id: str | None = None,
          bar: str | None = None) -> dict | None:
    """One evaluation cycle over one journal window (the
    self-audit's contract): mutate ``state``, return the
    journal record to append — or None when the snapshot did
    not change, so a quiet hour journals nothing.  The FIRST
    cycle always journals: that record is the deployment
    marker stating the evaluation window has opened.  A
    damaged window degrades the cycle (counted, nothing
    folded), never a guessed grading."""
    for key, value in new_state().items():
        state.setdefault(key, value)
    if damaged_lines:
        state["damaged_cycles"] = int(
            state.get("damaged_cycles") or 0) + 1
        return None
    snap = accumulate(state, records, now=now)
    state["last_snapshot"] = snap
    state["last_cycle_bar"] = bar
    if snap == state.get("last_journaled"):
        return None
    state["cycles"] = int(state.get("cycles") or 0) + 1
    state["last_journaled"] = snap
    return {
        "type": RECORD_TYPE, "bar": bar, "run_id": run_id,
        "window": window, **snap,
        "note": (
            "Promotion evaluation over the §G routes' own "
            "shadow history (S4 pairing, +4h, degenerate-R "
            "excluded). A promotion_ready verdict is a "
            "recommendation to the owner — this record "
            "promotes nothing, demotes nothing, and feeds no "
            "decision."),
    }


def evaluate_records(records: list[dict], *, now=None) -> dict:
    """One-shot evaluation over a complete record list (an
    exported journal): the same accumulate + snapshot the live
    cycle runs, over a fresh state.  The rerunnable evidence
    report (scripts/range_promotion_report.py) is this
    function plus rendering."""
    state = new_state()
    if now is None:
        stamps = [str(r.get("ts") or "") for r in records
                  if r.get("type") == "judgment_shadow"]
        now = max(stamps) if stamps else None
    return accumulate(state, records, now=now)


# ------------------------------------------------------------ rendering

_VERDICT_PLAIN = {
    VERDICT_READY: "PROMOTION READY (recommendation only)",
    VERDICT_NOT_READY: "not ready",
    VERDICT_INSUFFICIENT: "not enough data yet",
}


def _r(value) -> str:
    return "—" if value is None else f"{float(value):+.2f}R"


def _pct(value) -> str:
    return "—" if value is None else f"{float(value):.0%}"


def render_lines(state: dict) -> list[str]:
    """The /status block, plain words first: one line per
    route with its verdict and the numbers behind it."""
    snap = state.get("last_snapshot")
    if not snap:
        return ["range promotion (§G): no evaluation cycle "
                "has run yet — per-route verdicts (promotion "
                "ready / not ready / not enough data) land here "
                "once the self-audit cycle journals them"]
    lines = [
        f"range promotion (§G): graded at +{HORIZON_H}h, "
        f"floor {MIN_PAIRED_PER_ARM} paired outcomes per arm; "
        f"window opened {snap.get('window_start') or '—'} "
        f"(route-stamped shadow history only)"]
    damaged = int(state.get("damaged_cycles") or 0)
    if damaged:
        lines.append(
            f"  journal damage degraded {damaged} cycle(s) — "
            "those windows were not graded, never guessed")
    for playbook in RANGE_PLAYBOOKS:
        row = (snap.get("routes") or {}).get(playbook) or {}
        verdict = row.get("verdict") or VERDICT_INSUFFICIENT
        take = row.get("take") or {}
        abstain = row.get("abstain") or {}
        head = (f"  {playbook}: "
                f"{_VERDICT_PLAIN.get(verdict, verdict)}")
        if row.get("evaluated"):
            head += (
                f" — takes median {_r(take.get('median_r'))} "
                f"(hit {_pct(take.get('hit_rate'))}, "
                f"{take.get('paired', 0)} paired) vs abstains "
                f"median {_r(abstain.get('median_r'))} "
                f"(hit {_pct(abstain.get('hit_rate'))}, "
                f"{abstain.get('paired', 0)} paired) · "
                f"{row.get('evaluated', 0)} evaluated")
            if row.get("pending_outcomes"):
                head += (f" · {row['pending_outcomes']} "
                         "awaiting outcomes")
            if row.get("excluded_degenerate_r"):
                head += (f" · {row['excluded_degenerate_r']} "
                         "degenerate-R excluded")
        lines.append(head)
        lines.append(f"    {row.get('verdict_reason') or ''}")
    return lines
