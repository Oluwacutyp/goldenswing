"""Whole-book SELF-EVALUATION (MarketSituation Pass 9 —
SPEC §2.2's validation row + the §G discipline, generalised
from per-family verdicts to one comparison surface).

Passes 5 and 7a–7e each built a promotion evaluation for one
family or route pair: the §G range routes, volatility
expansion, trend / momentum, auction / value, mean
reversion, session / opening range, and the two binding
connected routes (hot_scalp_promotion / resting_limit).
Each reads ONLY its own route-stamped shadow history
through the ONE shared evaluator (execution/
range_promotion.py).  What none of them does is put the
whole book on one page: which families are measured at
all, which discriminate the right way, and how the
GOVERNING routes' own judged history (h1_trigger,
qm_m5_native, qm_d1_native — every candidate on them also
journals a route-stamped judgment_shadow verdict) grades
on the identical instrument.

This module is that page.  It re-implements NOTHING:

- the shadow families' rows are their own modules'
  snapshots, verbatim (execution/range_promotion.py,
  vol_promotion.py, trend_promotion.py, auction_promotion.py,
  reversion_promotion.py, session_promotion.py) and the
  connected routes' rows are route_promotion.py's standing
  evaluation, verbatim;
- the governing routes' rows are folded from the journal
  through range_promotion's accumulate with a governing
  routes map — the same S4 pairing at +4h, the same
  degenerate-R exclusions, the same per-arm floors;
- each group's POOLED read (its routes' paired outcomes
  combined) is verdict_for over the pooled arms — one
  evaluator, one rule, applied to the pool: inverted
  discrimination can never read ready at any level.

POSTURE: evaluation and reporting only.  This record
grants, demotes and changes NO authority.  The shadow
families' promotion_ready verdicts remain recommendations
to the owner; the connected routes' binding verdicts live
in route_promotion's own records and are neither re-issued
nor overridden here (their pooled read in this table is a
comparison, binds=False); the governing routes are measured
for the record — a governing route's pooled read is a
performance fact, not a promotion question.

The cycle host is the unified self-audit (the runner feeds
this module the same journal windows, inside the same S3
cadence, persisted in the same state file under its own
key, AFTER every family phase has folded the window — the
comparison therefore reads this cycle's family snapshots,
not last cycle's).  Findings feed back nowhere.
"""
from __future__ import annotations

from . import auction_promotion as _auction
from . import range_promotion as _eval
from . import reversion_promotion as _reversion
from . import route_promotion as _route
from . import session_promotion as _session
from . import trend_promotion as _trend
from . import vol_promotion as _vol

#: Journal record type for one comparison snapshot.
RECORD_TYPE = "book_evaluation"

#: The shadow families, in program order: (group key, plain
#: label, module).  Each module's snapshot(state) is the
#: family's own verdict table — consumed, never recomputed.
FAMILY_GROUPS: tuple[tuple[str, str, object], ...] = (
    ("range", "§G range", _eval),
    ("vol", "volatility expansion", _vol),
    ("trend", "trend / momentum", _trend),
    ("auction", "auction / value", _auction),
    ("reversion", "mean reversion", _reversion),
    ("session", "session / opening range", _session),
)

#: The governing routes and the playbooks whose judged
#: history is stamped with them (route_promotion's own
#: order-attribution table, minus the two connected routes
#: whose binding evaluation is its own group below).
GOVERNING_ROUTES: tuple[str, ...] = (
    "h1_trigger", "qm_m5_native", "qm_d1_native")
GOVERNING_JUDGMENT_ROUTES: dict[str, str] = {
    playbook: route
    for playbook, route in _route.ORDER_ROUTE_BY_PLAYBOOK.items()
    if route in GOVERNING_ROUTES}
GOVERNING_PLAYBOOKS: tuple[str, ...] = tuple(
    GOVERNING_JUDGMENT_ROUTES)

_GOVERNING_BASIS = (
    "governing routes' own route-stamped judgment_shadow "
    "records only (judgment.authority_route naming the "
    "route: h1_trigger for the roster + range playbooks' "
    "roster firings, qm_m5_native / qm_d1_native for the "
    "native routes).  The same playbooks' family-route "
    "history (§G and Pass 7 stamps) is a different question "
    "and is graded in those families' own rows, never here.")

_COMPARISON_BASIS = (
    "one instrument for every row: S4 pairing at +4h, "
    "degenerate-R pairs excluded and counted, per-arm "
    "paired floor of 20, medians and hit rates (never "
    "means alone), inverted discrimination can never read "
    "promotion_ready.  Family rows are each family's own "
    "evaluation snapshot; governing rows are folded from "
    "the journal on the same machinery; each group's "
    "pooled read combines its routes' paired outcomes and "
    "applies the same verdict rule to the pool.")


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
    verdict, reason = _eval.verdict_for(
        slot, family=family, binds=False)
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

    ``family_states`` maps audit-state keys to standing
    states: the six shadow families under their own keys
    ("range_promotion" … "session_promotion") and the
    connected routes under "route_promotion".  A family
    whose state has not been supplied reads as an empty
    snapshot — stated, never invented."""
    family_states = family_states or {}
    groups: list[dict] = []
    for key, label, module in FAMILY_GROUPS:
        sub = family_states.get(f"{key}_promotion")
        if isinstance(sub, dict):
            snap = module.snapshot(sub)
            pool = sub.get("routes") or {}
        else:
            snap = module.snapshot(module.new_state())
            pool = {}
        groups.append(_group(
            key, label, "shadow_declared", snap, pool))
    route_sub = family_states.get("route_promotion")
    if isinstance(route_sub, dict):
        route_snap = route_sub.get("last_evaluation") or {}
        route_pool = route_sub.get("routes") or {}
    else:
        route_snap, route_pool = {}, {}
    groups.append(_group(
        "connected", "connected routes (binding evaluation)",
        "shadow_declared", route_snap, route_pool))
    governing_snap = _eval.snapshot(
        governing_state, routes_map=GOVERNING_JUDGMENT_ROUTES,
        playbooks=GOVERNING_PLAYBOOKS, basis=_GOVERNING_BASIS,
        family="governing routes", binds=False)
    for row in governing_snap["routes"].values():
        row["authority"] = "governing"
    groups.append(_group(
        "governing", "governing routes (measured)",
        "governing", governing_snap,
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
          family_states: dict | None = None, now=None,
          damaged_lines: int = 0, window: dict | None = None,
          run_id: str | None = None,
          bar: str | None = None) -> dict | None:
    """One comparison cycle over one journal window (the
    self-audit's contract): fold the window into the
    governing routes' standing state, read every family's
    current snapshot, and return the journal record when
    the comparison changed — or None.  The FIRST cycle
    always journals: that record is the deployment marker
    stating the comparison has opened.  A damaged window
    degrades the cycle (counted, nothing folded)."""
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
        family="governing routes", binds=False)
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
            "Whole-book self-evaluation: every family's "
            "promotion evaluation and the governing routes' "
            "measured history on one instrument (S4 pairing, "
            "+4h, degenerate-R excluded).  This record "
            "grants, demotes and changes no authority — "
            "shadow verdicts remain recommendations, the "
            "connected routes' binding verdicts live in "
            "route_promotion's own records, and governing "
            "routes are measured, not promoted or demoted, "
            "here."),
    }


def evaluate_records(records: list[dict], *, now=None) -> dict:
    """One-shot comparison over a complete record list (an
    exported journal): every family's own accumulate over
    fresh states, the connected routes' one-shot evaluation,
    and the governing fold — the same computation the live
    cycle performs."""
    if now is None:
        stamps = [str(r.get("ts") or "") for r in records
                  if r.get("type") == "judgment_shadow"]
        now = max(stamps) if stamps else None
    family_states: dict[str, dict] = {}
    for key, _label, module in FAMILY_GROUPS:
        sub = module.new_state()
        _eval.accumulate(
            sub, records, now=now,
            routes_map=getattr(
                module, _ROUTES_ATTR[key]),
            playbooks=getattr(module, _PLAYBOOKS_ATTR[key]),
            basis=None, family=_label, binds=False)
        family_states[f"{key}_promotion"] = sub
    route_eval = _route.evaluate_records(records, now=now)
    route_sub = _route.new_state()
    _eval.accumulate(
        route_sub, records, now=now,
        routes_map=_route.ROUTE_JUDGMENT_ROUTES,
        playbooks=_route.PROMOTION_PLAYBOOKS, basis=None,
        family="connected routes", binds=True)
    family_states["route_promotion"] = {
        "last_evaluation": route_eval,
        "routes": route_sub.get("routes") or {}}
    governing = new_state()
    _eval.accumulate(
        governing, records, now=now,
        routes_map=GOVERNING_JUDGMENT_ROUTES,
        playbooks=GOVERNING_PLAYBOOKS, basis=_GOVERNING_BASIS,
        family="governing routes", binds=False)
    table = comparison(family_states=family_states,
                       governing_state=governing)
    # The one-shot route evaluation is computed fresh (the
    # standing-state shortcut above cannot reproduce its
    # criteria-bound snapshot); splice its rows in.
    for group in table["groups"]:
        if group["key"] == "connected":
            group["routes"] = route_eval.get("routes") or {}
            group["route_verdicts"] = _verdict_counts(
                group["routes"])
            group["window_start"] = route_eval.get(
                "window_start")
    return table


#: Each family module's routes-map / playbooks constants
#: (the one-shot path folds through the shared evaluator
#: with the family's own tables, exactly as its cycle does).
_ROUTES_ATTR = {
    "range": "RANGE_JUDGMENT_ROUTES",
    "vol": "VOL_JUDGMENT_ROUTES",
    "trend": "TREND_JUDGMENT_ROUTES",
    "auction": "AUCTION_JUDGMENT_ROUTES",
    "reversion": "REVERSION_JUDGMENT_ROUTES",
    "session": "SESSION_JUDGMENT_ROUTES",
}
_PLAYBOOKS_ATTR = {
    "range": "RANGE_PLAYBOOKS",
    "vol": "VOL_PLAYBOOKS",
    "trend": "TREND_PLAYBOOKS",
    "auction": "AUCTION_PLAYBOOKS",
    "reversion": "REVERSION_PLAYBOOKS",
    "session": "SESSION_PLAYBOOKS",
}


# ------------------------------------------------------------ rendering

_VERDICT_PLAIN = {
    _eval.VERDICT_READY: "PROMOTION READY (recommendation only)",
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
