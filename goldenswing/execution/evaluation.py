"""The whole-book EVALUATION instrument (S4 pairing at +4h).

One shared evaluator, extracted from the deleted
execution/range_promotion.py (the §G promotion module): it
grades a route's judged history — per route and arm
(would-take / would-abstain): n paired, hit rate, MEDIAN
realized R — and reads discrimination both ways (a route
whose would-abstains outperform its would-takes reads
INVERTED and can never read ready).  Degenerate-R pairs
are excluded from every statistic and counted beside them.

The dataset is the routes' OWN route-stamped judgment
history only: ``judgment`` records whose judgment block
carries the route stamp (``authority_route``).  The
pre-rework record type "judgment_shadow" is accepted for
old journals.  The evaluation window starts at the route
wiring's deployment, stated on every record: no pre-wiring
history is retroactively evaluated.

Verdicts (per route): ``insufficient_data`` (below the
readable floors), ``not_ready`` (measured and failing:
would-take median R negative, or the discrimination
inverted), ``ready`` (measured, right-way discrimination
on both reads, non-negative would-take median).  A verdict
is a MEASUREMENT only — it grants, promotes, demotes and
changes nothing.  The promotion gate is deleted (owner
order 2026-11-25/27); there is no promotion for a verdict
to recommend.
"""
from __future__ import annotations

import statistics
from typing import Any

from .judgment_validation import build_pairs
from .playbook_registry import RANGE_SPECS
from .self_audit import JUDGMENT_MAX_AGE_H

#: Judgment record types the evaluator reads: the live type
#: "judgment" and the pre-rework type "judgment_shadow"
#: (old journals on the box still grade).
JUDGMENT_RECORD_TYPES: tuple[str, ...] = (
    "judgment", "judgment_shadow")

#: The judgment route each range playbook's history journals
#: under.  Route name == playbook name (one route per
#: connected path; the journal's judgment records are keyed
#: by playbook).
RANGE_JUDGMENT_ROUTES: dict[str, str] = {
    spec.name: spec.name for spec in RANGE_SPECS}

#: The range playbooks, registry order.
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


#: Standing-state caps (the state file is JSON and rides every
#: S3 save): paired R values kept per arm (medians need the
#: distribution; the cap is 50x the readable floor), pending
#: judgments carried across windows, folded keys remembered so a
#: duplicate journal row can never grade twice.
_R_CAP = 1000
_PENDING_CAP = 500
_KEY_CAP = 4000

VERDICT_INSUFFICIENT = "insufficient_data"
VERDICT_NOT_READY = "not_ready"
VERDICT_READY = "ready"

_WINDOW_BASIS = (
    "route-stamped judgment records only "
    "(judgment.authority_route naming the route).  Records "
    "without the stamp predate the route wiring or belong to "
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
    except (ValueError, TypeError):
        # Unparseable clock: a missing fact, never a guessed one.
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


def _r(value) -> str:
    return "—" if value is None else f"{float(value):+.2f}R"


def _pct(value) -> str:
    return "—" if value is None else f"{float(value):.0%}"


def verdict_for(slot: dict, *, family: str = "§G"
                ) -> tuple[str, str]:
    """One route's verdict from its accumulated arms — the
    rule, exactly as coded (see the module docstring).

    ``family`` names the route family in the no-candidates
    reason (the range family is §G).  The verdict is a
    measurement only: it grants, promotes, demotes and
    changes nothing — the record, /status and the manifest
    are the whole surface."""
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
            f"beat its takes can never read ready")
    return VERDICT_READY, (
        f"measured, right-way discrimination on both reads: "
        f"would-takes median {take_med:+.2f}R / hit rate "
        f"{take['hit_rate']:.0%} over {take['paired']} paired, "
        f"would-abstains median {abstain_med:+.2f}R / hit "
        f"rate {abstain['hit_rate']:.0%} over "
        f"{abstain['paired']} paired — a measurement, never "
        f"an action")


def snapshot(state: dict, *, routes_map: dict | None = None,
             playbooks: tuple | None = None,
             basis: str | None = None, family: str = "§G"
             ) -> dict:
    """The per-route verdict table, as journaled / rendered /
    manifested.  Every registered route in the map is present —
    absence of data is a stated verdict, never a missing row.

    The keyword parameters let callers grade a different
    playbook→route map, playbook set, window basis and
    verdict voice — the computation is unchanged."""
    routes_map = routes_map or RANGE_JUDGMENT_ROUTES
    playbooks = playbooks or RANGE_PLAYBOOKS
    routes: dict[str, Any] = {}
    pending_counts: dict[str, int] = {}
    for entry in state.get("pending") or []:
        pb = str(entry.get("playbook") or "")
        pending_counts[pb] = pending_counts.get(pb, 0) + 1
    for playbook in playbooks:
        slot = (state.get("routes") or {}).get(playbook) or {}
        verdict, reason = verdict_for(slot, family=family)
        routes[playbook] = {
            "route": routes_map[playbook],
            "playbook": playbook,
            "authority": "live",
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

def _stamped_judgment(row: dict, routes_map: dict | None = None
                      ) -> str | None:
    """The route a journal row belongs to, or None: the row
    must be a judgment record whose judgment block carries
    the route stamp for one of the routes in the map.
    ``routes_map`` is the playbook -> route map; the default
    is the range family's map.  The pre-rework record type
    "judgment_shadow" is accepted for old journals."""
    routes_map = routes_map or RANGE_JUDGMENT_ROUTES
    if row.get("type") not in JUDGMENT_RECORD_TYPES:
        return None
    playbook = str(row.get("playbook") or "")
    route = routes_map.get(playbook)
    if route is None:
        return None
    judgment = row.get("judgment") or {}
    if str(judgment.get("authority_route") or "") != route:
        return None
    return playbook


def _judgment_row(row: dict) -> dict:
    """The minimal judgment row build_pairs needs, kept in the
    standing state across windows (JSON-shaped)."""
    judgment = row.get("judgment") or {}
    return {
        "type": "judgment",
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
               basis: str | None = None, family: str = "§G"
               ) -> dict:
    """Fold one journal window into the standing state and
    return the new snapshot.

    Mirrors the self-audit's carry discipline: stamped
    judgments are carried in ``pending`` across windows until
    their +4h reconciliation lands; a judgment whose outcome
    never lands within JUDGMENT_MAX_AGE_H is unmeasurable,
    not pending forever.  Folding is keyed and remembered, so
    re-reading a window (or a duplicated journal row) can
    never grade twice.
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
        playbook = _stamped_judgment(row, routes_map)
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
        fresh.append(_judgment_row(row))
        pending_keys.add(key)
        slot["evaluated"] = int(slot.get("evaluated") or 0) + 1
        ts = str(row.get("ts") or "")
        if ts and (state.get("window_start") is None
                   or ts < str(state["window_start"])):
            state["window_start"] = ts

    # S4 pairing over everything carried + this window's rows.
    # First pair wins per key: duplicate-key judgments (counted
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
                    > JUDGMENT_MAX_AGE_H * 3600:
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
                    family=family)




def evaluate_records(records: list[dict], *, now=None,
                     routes_map: dict | None = None,
                     playbooks: tuple | None = None) -> dict:
    """One-shot evaluation over a complete record list (an
    exported journal): the same accumulate + snapshot the live
    cycle runs, over a fresh state."""
    state = new_state()
    if now is None:
        stamps = [str(r.get("ts") or "") for r in records
                  if r.get("type") in JUDGMENT_RECORD_TYPES]
        now = max(stamps) if stamps else None
    return accumulate(state, records, now=now,
                      routes_map=routes_map,
                      playbooks=playbooks)
