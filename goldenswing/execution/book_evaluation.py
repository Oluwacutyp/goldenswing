"""Whole-book SELF-EVALUATION (MarketSituation Pass 9 —
SPEC §2.2's validation row + the §G discipline, generalised
from per-family verdicts to one comparison surface).

Every family and route grades on ONE shared instrument
(execution/evaluation.py — the S4 pairing at +4h, the same
degenerate-R exclusions, the same per-arm floors): the five
families' rows (range, volatility expansion, trend /
momentum, auction / value, mean reversion, session /
opening range) and the connected routes (hot_scalp,
arrival_zone) and the roster / native routes
(h1_roster, qm_m5_native, qm_d1_native).  Each group's
POOLED read (its routes' paired outcomes combined) is
verdict_for over the pooled arms — one evaluator, one
rule, applied to the pool: inverted discrimination can
never read ready at any level.

POSTURE: evaluation and reporting only.  This record
grants, promotes, demotes and changes NO authority — the
promotion gate is deleted (owner order 2026-11-25/27),
so a verdict is a measurement, never a recommendation
for a promotion that no longer exists.  Every route is
live; the families' rows are their measured history, the
same as the governing routes'.

The cycle host is the unified self-audit (the runner feeds
this module the same journal windows, inside the same S3
cadence, persisted in the same state file under its own
key).  Findings feed back nowhere.
"""
from __future__ import annotations

from . import evaluation as _eval
from .playbook_registry import (
    AUCTION_SPECS, NATIVE_SPECS, RANGE_SPECS, REVERSION_SPECS,
    ROSTER_SPECS, SESSION_SPECS, TREND_SPECS, VOL_SPECS,
    order_route_for as _order_route_for)

#: Journal record type for one comparison snapshot.
RECORD_TYPE = "book_evaluation"

#: The families, in program order: (group key, plain label,
#: the registry specs whose playbooks form the group).  Each
#: group's route map is its playbooks' own live routes
#: (playbook_registry.order_route_for) — one truth, derived.
FAMILY_GROUPS: tuple[tuple[str, str, tuple], ...] = (
    ("range", "§G range", RANGE_SPECS),
    ("vol", "volatility expansion", VOL_SPECS),
    ("trend", "trend / momentum", TREND_SPECS),
    ("auction", "auction / value", AUCTION_SPECS),
    ("reversion", "mean reversion", REVERSION_SPECS),
    ("session", "session / opening range", SESSION_SPECS),
)

#: Connected routes (the two sleeves outside the registry's
#: playbook specs): hot_scalp and arrival_zone, each on its
#: own live route.
CONNECTED_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("connected", "connected routes",
     ("hot_scalp", "arrival_zone")),
)


def _routes_map_for(specs) -> dict[str, str]:
    """Playbook -> live route for a spec group (the registry's
    order_route_for, one truth)."""
    return {spec.name: (_order_route_for(spec.name) or spec.name)
            for spec in specs}


def _playbooks_for(specs) -> tuple[str, ...]:
    return tuple(spec.name for spec in specs)

#: The roster / native routes and the playbooks whose judged
#: history is stamped with them: the roster playbooks ride
#: the h1_roster route, the two natives ride their own
#: native routes.  (The same playbooks' family-route history
#: is a different question and is graded in those families'
#: own rows, never here.)
GOVERNING_ROUTES: tuple[str, ...] = (
    "h1_roster", "qm_m5_native", "qm_d1_native")
GOVERNING_JUDGMENT_ROUTES: dict[str, str] = {
    spec.name: (_order_route_for(spec.name) or spec.name)
    for spec in tuple(ROSTER_SPECS) + tuple(NATIVE_SPECS)
    if (_order_route_for(spec.name) or spec.name)
    in GOVERNING_ROUTES}
GOVERNING_PLAYBOOKS: tuple[str, ...] = tuple(
    GOVERNING_JUDGMENT_ROUTES)

_GOVERNING_BASIS = (
    "roster / native routes' own route-stamped judgment "
    "records only (judgment.authority_route naming the "
    "route: h1_roster for the roster, qm_m5_native / "
    "qm_d1_native for the native routes).")

_COMPARISON_BASIS = (
    "one instrument for every row: S4 pairing at +4h, "
    "degenerate-R pairs excluded and counted, per-arm "
    "paired floor of 20, medians and hit rates (never "
    "means alone), inverted discrimination can never read "
    "ready.  Family rows are each family's own evaluation "
    "snapshot; roster / native rows are folded from the "
    "journal on the same machinery; each group's pooled "
    "read combines its routes' paired outcomes and applies "
    "the same verdict rule to the pool.")


# ------------------------------------------------------------ state

def new_state() -> dict:
    """The standing state (rides inside the S3 audit state
    file, under its own key): the shared evaluator's block
    for the governing fold, plus the comparison's own
    journaling bookkeeping."""
    state = _eval.new_state()
    state.update({
        "last_comparison": None,
        "last_journaled_comparison": None,
    })
    return state


# ------------------------------------------------------------ pooling

def _pooled_slot(route_slots: dict) -> dict:
    """Combine per-route evaluator slots into one pooled
    slot (the group's combined paired history)."""
    slot = {"evaluated": 0, "take_r": [], "take_hits": 0,
            "abstain_r": [], "abstain_hits": 0,
            "excluded_degenerate_r": 0,
            "duplicate_key_records": 0,
            "unmeasurable_expired": 0,
            "pending_outcomes": 0}
    for sub in (route_slots or {}).values():
        if not isinstance(sub, dict):
            continue
        slot["evaluated"] += int(sub.get("evaluated") or 0)
        slot["take_r"] += list(sub.get("take_r") or [])
        slot["take_hits"] += int(sub.get("take_hits") or 0)
        slot["abstain_r"] += list(sub.get("abstain_r") or [])
        slot["abstain_hits"] += int(
            sub.get("abstain_hits") or 0)
        slot["excluded_degenerate_r"] += int(
            sub.get("excluded_degenerate_r") or 0)
        slot["duplicate_key_records"] += int(
            sub.get("duplicate_key_records") or 0)
        slot["unmeasurable_expired"] += int(
            sub.get("unmeasurable_expired") or 0)
    return slot


def _pooled_read(slot: dict, *, family: str) -> dict:
    """The verdict rule over one pooled slot — the same
    verdict_for every route faces, applied to the pool."""
    verdict, reason = _eval.verdict_for(slot, family=family)
    return {
        "binds": False,
        "evaluated": int(slot.get("evaluated") or 0),
        "take": _eval._arm(slot.get("take_r") or [],
                           slot.get("take_hits") or 0),
        "abstain": _eval._arm(slot.get("abstain_r") or [],
                              slot.get("abstain_hits") or 0),
        "excluded_degenerate_r": int(
            slot.get("excluded_degenerate_r") or 0),
        "verdict": verdict,
        "verdict_reason": reason,
    }


def _verdict_counts(rows: dict) -> dict:
    counts = {_eval.VERDICT_READY: 0, _eval.VERDICT_NOT_READY: 0,
              _eval.VERDICT_INSUFFICIENT: 0}
    for row in (rows or {}).values():
        verdict = (row or {}).get("verdict") \
            or _eval.VERDICT_INSUFFICIENT
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def _group(key: str, label: str, posture: str, snap: dict,
           pool_slots: dict) -> dict:
    """One comparison group: the source snapshot's route
    rows verbatim, plus the pooled read over its routes'
    combined paired history."""
    rows = (snap or {}).get("routes") or {}
    pooled = _pooled_read(_pooled_slot(pool_slots),
                          family=label)
    return {
        "key": key,
        "label": label,
        "authority_posture": posture,
        "window_start": (snap or {}).get("window_start"),
        "routes": rows,
        "route_verdicts": _verdict_counts(rows),
        "pooled": pooled,
    }


# ------------------------------------------------------------ snapshot

def comparison(*, family_states: dict | None,
               governing_state: dict) -> dict:
    """The whole-book comparison over the families' standing
    states + the governing fold's standing state.  Pure: the
    same states in, the same table out.

    ``family_states`` maps group keys ("range", "vol", …,
    "connected") to their evaluator standing states (each
    accumulated over the journal on the shared instrument).
    A group whose state has not been supplied reads as an
    empty snapshot — stated, never invented.  Every group
    is live: the promotion gate is deleted, so no group
    carries a shadow or governing posture anymore."""
    family_states = family_states or {}
    groups: list[dict] = []
    for key, label, specs in FAMILY_GROUPS:
        sub = family_states.get(key)
        routes_map = _routes_map_for(specs)
        playbooks = _playbooks_for(specs)
        if isinstance(sub, dict):
            snap = _eval.snapshot(
                sub, routes_map=routes_map, playbooks=playbooks,
                family=label)
            pool = sub.get("routes") or {}
        else:
            snap = _eval.snapshot(
                _eval.new_state(), routes_map=routes_map,
                playbooks=playbooks, family=label)
            pool = {}
        groups.append(_group(key, label, "live", snap, pool))
    connected_map = {"hot_scalp": "hot_scalp_promotion",
                     "arrival_zone": "resting_limit"}
    connected_playbooks = ("hot_scalp", "arrival_zone")
    connected_sub = family_states.get("connected")
    if isinstance(connected_sub, dict):
        connected_snap = _eval.snapshot(
            connected_sub, routes_map=connected_map,
            playbooks=connected_playbooks,
            family="connected routes")
        connected_pool = connected_sub.get("routes") or {}
    else:
        connected_snap = _eval.snapshot(
            _eval.new_state(), routes_map=connected_map,
            playbooks=connected_playbooks,
            family="connected routes")
        connected_pool = {}
    groups.append(_group(
        "connected", "connected routes",
        "live", connected_snap, connected_pool))
    governing_snap = _eval.snapshot(
        governing_state, routes_map=GOVERNING_JUDGMENT_ROUTES,
        playbooks=GOVERNING_PLAYBOOKS, basis=_GOVERNING_BASIS,
        family="governing routes")
    for row in governing_snap["routes"].values():
        row["authority"] = "live"
    groups.append(_group(
        "governing", "governing routes (measured)",
        "live", governing_snap,
        governing_state.get("routes") or {}))
    return {
        "horizon_h": _eval.HORIZON_H,
        "floors": {"min_paired_per_arm":
                   _eval.MIN_PAIRED_PER_ARM},
        "window_basis": _COMPARISON_BASIS,
        "groups": groups,
    }


# ------------------------------------------------------------ the cycle

def cycle(*, state: dict, records: list[dict],
          now=None, damaged_lines: int = 0,
          window: dict | None = None, run_id: str | None = None,
          bar: str | None = None) -> dict | None:
    """One comparison cycle over one journal window (the
    self-audit's contract): fold the window into the
    governing routes' standing state AND into each family's
    own standing state (kept here, under ``family_<key>`` —
    the promotion modules that used to hold them are
    deleted), and return the journal record when the
    comparison changed — or None.  The FIRST cycle always
    journals: that record is the deployment marker stating
    the comparison has opened.  A damaged window degrades
    the cycle (counted, nothing folded)."""
    for key, value in new_state().items():
        state.setdefault(key, value)
    if damaged_lines:
        state["damaged_cycles"] = int(
            state.get("damaged_cycles") or 0) + 1
        return None
    _eval.accumulate(
        state, records, now=now,
        routes_map=GOVERNING_JUDGMENT_ROUTES,
        playbooks=GOVERNING_PLAYBOOKS, basis=_GOVERNING_BASIS,
        family="governing routes")
    family_states: dict[str, dict] = {}
    for key, label, specs in FAMILY_GROUPS:
        sub = state.setdefault(f"family_{key}", _eval.new_state())
        _eval.accumulate(
            sub, records, now=now,
            routes_map=_routes_map_for(specs),
            playbooks=_playbooks_for(specs), family=label)
        family_states[key] = sub
    connected_sub = state.setdefault(
        "family_connected", _eval.new_state())
    _eval.accumulate(
        connected_sub, records, now=now,
        routes_map={"hot_scalp": "hot_scalp_promotion",
                    "arrival_zone": "resting_limit"},
        playbooks=("hot_scalp", "arrival_zone"),
        family="connected routes")
    family_states["connected"] = connected_sub
    table = comparison(family_states=family_states,
                       governing_state=state)
    state["last_comparison"] = table
    state["last_cycle_bar"] = bar
    if table == state.get("last_journaled_comparison"):
        return None
    state["cycles"] = int(state.get("cycles") or 0) + 1
    state["last_journaled_comparison"] = table
    return {
        "type": RECORD_TYPE, "bar": bar, "run_id": run_id,
        "window": window, **table,
        "note": (
            "Whole-book self-evaluation: every family's and "
            "the governing routes' measured history on one "
            "instrument (S4 pairing, +4h, degenerate-R "
            "excluded).  This record grants, promotes, "
            "demotes and changes no authority — every route "
            "is live, and the verdicts here are measurements, "
            "never actions."),
    }


def evaluate_records(records: list[dict], *, now=None) -> dict:
    """One-shot comparison over a complete record list (an
    exported journal): every family's own accumulate over
    fresh states, the connected routes' accumulate, and the
    governing fold — the same computation the live cycle
    performs."""
    if now is None:
        stamps = [str(r.get("ts") or "") for r in records
                  if r.get("type") in _eval.JUDGMENT_RECORD_TYPES]
        now = max(stamps) if stamps else None
    family_states: dict[str, dict] = {}
    for key, label, specs in FAMILY_GROUPS:
        sub = _eval.new_state()
        _eval.accumulate(
            sub, records, now=now,
            routes_map=_routes_map_for(specs),
            playbooks=_playbooks_for(specs),
            basis=None, family=label)
        family_states[key] = sub
    connected_sub = _eval.new_state()
    _eval.accumulate(
        connected_sub, records, now=now,
        routes_map={"hot_scalp": "hot_scalp_promotion",
                    "arrival_zone": "resting_limit"},
        playbooks=("hot_scalp", "arrival_zone"),
        basis=None, family="connected routes")
    family_states["connected"] = connected_sub
    governing = new_state()
    _eval.accumulate(
        governing, records, now=now,
        routes_map=GOVERNING_JUDGMENT_ROUTES,
        playbooks=GOVERNING_PLAYBOOKS, basis=_GOVERNING_BASIS,
        family="governing routes")
    return comparison(family_states=family_states,
                      governing_state=governing)


# ------------------------------------------------------------ rendering

_VERDICT_PLAIN = {
    _eval.VERDICT_READY: "ready (measured)",
    _eval.VERDICT_NOT_READY: "not ready",
    _eval.VERDICT_INSUFFICIENT: "not enough data yet",
}


def render_lines(state: dict) -> list[str]:
    """The /status block, plain words first: one line per
    group with its pooled read and its routes' verdict
    counts — the comparison on one page."""
    table = state.get("last_comparison")
    if not table:
        return [("book evaluation (Pass 9): no comparison "
                 "cycle has run yet — the whole-book table "
                 "(every family + the governing routes on one "
                 "instrument) lands here once the self-audit "
                 "cycle journals it")]
    lines = [
        (f"book evaluation (Pass 9): every family and the "
         f"governing routes graded at +{_eval.HORIZON_H}h, "
         f"floor {_eval.MIN_PAIRED_PER_ARM} paired outcomes "
         f"per arm, degenerate-R excluded — one instrument, "
         f"no authority moved by this table")]
    damaged = int(state.get("damaged_cycles") or 0)
    if damaged:
        lines.append(
            f"  journal damage degraded {damaged} cycle(s) "
            "— those windows were not graded, never guessed")
    for group in table.get("groups") or []:
        pooled = group.get("pooled") or {}
        take = pooled.get("take") or {}
        abstain = pooled.get("abstain") or {}
        counts = group.get("route_verdicts") or {}
        verdict = pooled.get("verdict") \
            or _eval.VERDICT_INSUFFICIENT
        head = (f"  {group.get('label')}: pooled "
                f"{_VERDICT_PLAIN.get(verdict, verdict)}")
        if pooled.get("evaluated"):
            head += (
                f" — takes median "
                f"{_eval._r(take.get('median_r'))} "
                f"(hit {_eval._pct(take.get('hit_rate'))}, "
                f"{take.get('paired', 0)} paired) vs abstains "
                f"median {_eval._r(abstain.get('median_r'))} "
                f"(hit {_eval._pct(abstain.get('hit_rate'))}, "
                f"{abstain.get('paired', 0)} paired) · "
                f"{pooled.get('evaluated', 0)} evaluated")
        head += (
            f" · routes: {counts.get(_eval.VERDICT_READY, 0)} "
            f"ready / {counts.get(_eval.VERDICT_NOT_READY, 0)} "
            f"not ready / "
            f"{counts.get(_eval.VERDICT_INSUFFICIENT, 0)} "
            f"insufficient")
        lines.append(head)
        # Each route WITH history gets its own row with its
        # own n — the comparison is per route first, pooled
        # second (routes with no history are counted in the
        # group line, not padded out as empty rows).
        for name, row in sorted(
                (group.get("routes") or {}).items()):
            if not (row or {}).get("evaluated"):
                continue
            r_take = row.get("take") or {}
            r_abstain = row.get("abstain") or {}
            r_verdict = row.get("verdict") \
                or _eval.VERDICT_INSUFFICIENT
            lines.append(
                f"    {name}: "
                f"{_VERDICT_PLAIN.get(r_verdict, r_verdict)}"
                f" — takes median "
                f"{_eval._r(r_take.get('median_r'))} "
                f"(hit {_eval._pct(r_take.get('hit_rate'))}, "
                f"{r_take.get('paired', 0)} paired) vs "
                f"abstains median "
                f"{_eval._r(r_abstain.get('median_r'))} "
                f"(hit {_eval._pct(r_abstain.get('hit_rate'))}, "
                f"{r_abstain.get('paired', 0)} paired) · "
                f"{row.get('evaluated', 0)} evaluated")
    return lines
