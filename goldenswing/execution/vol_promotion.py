"""Volatility-expansion family — PROMOTION EVALUATION
(MarketSituation Pass 7a, SPEC §6 common contract + §9
Pass 7).

The two Pass 7a playbooks (vol_expansion_escape,
vol_failed_expansion) run on the family's scheduled route
(playbook_registry.ROUTE_VOLATILITY_EXPANSION): scheduled
formation-clock scans that journal candidates and hold NO
order authority.  Pass 5 declared them ``shadow_declared``
judgment routes, so every family candidate carries a
route-stamped ``judgment_shadow`` verdict and an S3
reconciliation, exactly like the §G range routes.  This
module is the measurement of that shadow history — the
family's own promotion evaluation, in the §G shape.

The evaluator is NOT re-implemented: execution/
range_promotion.py's accumulate/snapshot machinery (Pass 5
generalised it for exactly this reuse) folds the family's
route-stamped history with the family's route map, window
basis and verdict voice.  One evaluator, one rule: paired
outcomes at +4h per arm, degenerate-R excluded, medians
and hit rates, inverted discrimination can never read
ready, floors identical to every other family's.

It promotes NOTHING: a ``promotion_ready`` verdict is a
recommendation to the owner, and promotion itself stays
the separate explicit step it has always been (config
``judgment_route_modes`` naming the route "governing" —
though the family has no order path to govern until a
future pass builds one; the verdict is the evidence such
a pass would be decided on).

The cycle host is the unified self-audit (the runner feeds
this module the same journal windows, inside the same S3
cadence, persisted in the same state file under its own
key).  Findings feed back nowhere.
"""
from __future__ import annotations

from . import range_promotion as _eval
from .playbook_registry import VOL_SPECS

#: The judgment route each vol playbook's shadow history
#: journals under.  Route name == playbook name (the §G
#: precedent).  The runner's JUDGMENT_ROUTE_STATES declares
#: each of these "shadow_declared"; a test pins the two
#: tables to each other.
VOL_JUDGMENT_ROUTES: dict[str, str] = {
    spec.name: spec.name for spec in VOL_SPECS}

#: The family's playbooks, registry order.
VOL_PLAYBOOKS: tuple[str, ...] = tuple(VOL_JUDGMENT_ROUTES)

#: Journal record type for one evaluation snapshot.
RECORD_TYPE = "vol_promotion_evaluation"

FAMILY_LABEL = "volatility expansion"

_WINDOW_BASIS = (
    "volatility-expansion route-stamped judgment_shadow "
    "records only (judgment.authority_route naming the "
    "route).  Records without the stamp predate the Pass "
    "7a wiring or belong to another route and are not "
    "evaluated: the window starts at this wiring's "
    "deployment.")


def new_state() -> dict:
    """The standing evaluation state (rides inside the S3
    audit state file, under its own key)."""
    return _eval.new_state()


def snapshot(state: dict) -> dict:
    """The per-route verdict table over the standing
    state, with the family's routes, basis and voice."""
    return _eval.snapshot(
        state, routes_map=VOL_JUDGMENT_ROUTES,
        playbooks=VOL_PLAYBOOKS, basis=_WINDOW_BASIS,
        family=FAMILY_LABEL, binds=False)


def cycle(*, state: dict, records: list[dict], now=None,
          damaged_lines: int = 0, window: dict | None = None,
          run_id: str | None = None,
          bar: str | None = None) -> dict | None:
    """One evaluation cycle over one journal window (the
    self-audit's contract, §G-identical): mutate ``state``,
    return the journal record to append — or None when the
    snapshot did not change.  The FIRST cycle always
    journals: that record is the deployment marker stating
    the evaluation window has opened.  A damaged window
    degrades the cycle (counted, nothing folded)."""
    for key, value in new_state().items():
        state.setdefault(key, value)
    if damaged_lines:
        state["damaged_cycles"] = int(
            state.get("damaged_cycles") or 0) + 1
        return None
    snap = _eval.accumulate(
        state, records, now=now,
        routes_map=VOL_JUDGMENT_ROUTES,
        playbooks=VOL_PLAYBOOKS, basis=_WINDOW_BASIS,
        family=FAMILY_LABEL, binds=False)
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
            "Promotion evaluation over the volatility-"
            "expansion routes' own shadow history (S4 "
            "pairing, +4h, degenerate-R excluded). A "
            "promotion_ready verdict is a recommendation "
            "to the owner — this record promotes nothing, "
            "demotes nothing, and feeds no decision."),
    }


def evaluate_records(records: list[dict], *, now=None) -> dict:
    """One-shot evaluation over a complete record list (an
    exported journal): the same accumulate + snapshot the
    live cycle runs, over a fresh state."""
    state = new_state()
    if now is None:
        stamps = [str(r.get("ts") or "") for r in records
                  if r.get("type") == "judgment_shadow"]
        now = max(stamps) if stamps else None
    return _eval.accumulate(
        state, records, now=now,
        routes_map=VOL_JUDGMENT_ROUTES,
        playbooks=VOL_PLAYBOOKS, basis=_WINDOW_BASIS,
        family=FAMILY_LABEL, binds=False)


# ------------------------------------------------------------ rendering

_VERDICT_PLAIN = {
    _eval.VERDICT_READY: "PROMOTION READY (recommendation only)",
    _eval.VERDICT_NOT_READY: "not ready",
    _eval.VERDICT_INSUFFICIENT: "not enough data yet",
}


def render_lines(state: dict) -> list[str]:
    """The /status block, plain words first: one line per
    route with its verdict and the numbers behind it (the
    §G block's shape, the family's own routes)."""
    snap = state.get("last_snapshot")
    if not snap:
        return ["vol promotion (Pass 7a): no evaluation "
                "cycle has run yet — per-route verdicts "
                "(promotion ready / not ready / not enough "
                "data) land here once the self-audit cycle "
                "journals them"]
    lines = [
        f"vol promotion (Pass 7a): graded at "
        f"+{_eval.HORIZON_H}h, floor "
        f"{_eval.MIN_PAIRED_PER_ARM} paired outcomes per "
        f"arm; window opened "
        f"{snap.get('window_start') or '—'} (route-stamped "
        f"shadow history only)"]
    damaged = int(state.get("damaged_cycles") or 0)
    if damaged:
        lines.append(
            f"  journal damage degraded {damaged} cycle(s) "
            "— those windows were not graded, never guessed")
    for playbook in VOL_PLAYBOOKS:
        row = (snap.get("routes") or {}).get(playbook) or {}
        verdict = row.get("verdict") \
            or _eval.VERDICT_INSUFFICIENT
        take = row.get("take") or {}
        abstain = row.get("abstain") or {}
        head = (f"  {playbook}: "
                f"{_VERDICT_PLAIN.get(verdict, verdict)}")
        if row.get("evaluated"):
            head += (
                f" — takes median "
                f"{_eval._r(take.get('median_r'))} "
                f"(hit {_eval._pct(take.get('hit_rate'))}, "
                f"{take.get('paired', 0)} paired) vs abstains "
                f"median {_eval._r(abstain.get('median_r'))} "
                f"(hit {_eval._pct(abstain.get('hit_rate'))}, "
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
