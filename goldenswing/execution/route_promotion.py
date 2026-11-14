"""Route PROMOTION with binding verdicts (MarketSituation
Pass 5; owner ruling 2026-11-08, AMENDMENTS-1 §A5).

The two connected non-roster routes — ``hot_scalp_promotion``
(playbook hot_scalp) and ``resting_limit`` (playbook
arrival_zone) — were declared ``shadow_declared`` by
completion-02: their candidates are funnel-evaluated and
core-judged, a shadow history accrues, and their own gate
stacks decided orders.  This module is the measurement +
authority machinery Pass 5 adds on top of that history, and
it differs from the §G range evaluation
(execution/range_promotion.py) in exactly one way: **the
verdict binds**.

- The EVALUATION is the range evaluator, shared — this module
  folds the routes' own route-stamped ``judgment_shadow``
  history through range_promotion's accumulate/verdict
  machinery (one evaluator, one bar set: S4 pairing at +4h,
  degenerate-R exclusions, per-arm paired floors, medians and
  hit rates, inverted discrimination can never read ready).
  A route's verdict is computed ONLY from its own stamped
  history: hot_scalp verdicts under the
  ``hot_scalp_promotion`` stamp, arrival_zone verdicts under
  the ``resting_limit`` stamp — the same playbooks' roster
  verdicts (h1_trigger) are a different question and are
  never read here.
- The CRITERIA are pre-registered (A5a): before any
  evaluation under them reads outcomes, a
  ``route_promotion_criteria`` record journals the bars
  (floors, horizon, the verdict rule verbatim, the criteria
  version).  Every evaluation record cites the criteria id +
  registration timestamp it was computed under, and a verdict
  with no preceding criteria record can never apply
  (``applicable_verdict`` refuses it) — the bars may not move
  after results are seen.
- The EFFECT (the runner applies it; this module stays
  pure): a ``promotion_ready`` verdict promotes the route to
  governing — its orders then flow through the
  ``_judgment_authority`` seam like every governed route,
  never around it.  A ``not_ready`` verdict demotes a
  verdict-promoted route back to shadow.  ``insufficient_data``
  changes nothing: absence of evidence strips no authority.
  Under the owner's strict boundary (2026-11-08), a route
  that does not govern submits NO independent demo orders —
  its candidates are preserved as shadow (the runner's
  boundary gates journal them) and keep accruing the history
  this evaluation reads.
- AUTO-ROLLBACK (A5b): ``scan_violations`` reads the same
  journal windows for invariant violations attributed to a
  route — a friction-gate breach on a PLACED order (a gate
  refusal is the gate working, never a violation), a broker
  reconciliation divergence, a duplicate order for one setup
  identity, or an explicit ``route_invariant_violation``
  record (the forward contract any producer may journal).
  The runner demotes a governing route on any of them,
  journals the evidence, and the route stays shadow until a
  FRESH evaluation (new outcomes folded after the demotion)
  reads promotion_ready again, or the owner's explicit
  config step releases it.

The cycle host is the unified self-audit (the runner feeds
this module the same journal windows, inside the same S3
cadence, persisted in the same state file) — no new
scheduler.
"""
from __future__ import annotations

from typing import Any

from . import range_promotion as _rp
from .friction_gate import FRICTION_RATIO_LIMIT
from .playbook_registry import ROSTER_SPECS, spec_for
from .range_promotion import (HORIZON_H, MIN_PAIRED_PER_ARM,
                              VERDICT_INSUFFICIENT,
                              VERDICT_NOT_READY, VERDICT_READY)

#: The routes this evaluation governs, route -> the playbook
#: whose route-stamped shadow history is the route's dataset
#: (the runner's JUDGMENT_ROUTE_PLAYBOOK carries the same
#: pairs for its manifest counts; a test pins the two).
#: Route name == playbook name for the per-playbook routes
#: (the §G range family's pattern): each family member is
#: evaluated on its OWN shadow history and promoted on its
#: OWN verdict — a strong member never carries a weak one,
#: and a weak one never holds a strong one back.
PROMOTION_ROUTES: dict[str, str] = {
    "hot_scalp_promotion": "hot_scalp",
    "resting_limit": "arrival_zone",
    # The 13 new-family members (owner promotion program
    # 2026-11-14): the volatility-expansion, trend-momentum,
    # auction-value, mean-reversion and session-range
    # families.  Their routes are already declared shadow in
    # the runner's JUDGMENT_ROUTE_STATES and carried in its
    # JUDGMENT_ROUTE_PLAYBOOK; registering them here is what
    # lets the binding evaluation actually judge them — a
    # family with no paired outcomes reads insufficient_data
    # (changes nothing), and only a route whose would-take
    # median R and discrimination both pass earns
    # promotion_ready.
    "vol_expansion_escape": "vol_expansion_escape",
    "vol_failed_expansion": "vol_failed_expansion",
    "trend_impulse_pullback": "trend_impulse_pullback",
    "trend_momentum_acceleration": "trend_momentum_acceleration",
    "auction_value_acceptance": "auction_value_acceptance",
    "auction_value_rejection": "auction_value_rejection",
    "auction_lvn_traversal": "auction_lvn_traversal",
    "mean_reversion_extension_fade":
        "mean_reversion_extension_fade",
    "mean_reversion_failed_continuation":
        "mean_reversion_failed_continuation",
    "session_opening_breakout": "session_opening_breakout",
    "session_opening_failure": "session_opening_failure",
    "session_drive_reversal": "session_drive_reversal",
    "session_asia_sweep": "session_asia_sweep",
}

#: The shared evaluator's convention: playbook -> route.
ROUTE_JUDGMENT_ROUTES: dict[str, str] = {
    playbook: route for route, playbook in PROMOTION_ROUTES.items()}

#: The evaluated playbooks, route-table order.
PROMOTION_PLAYBOOKS: tuple[str, ...] = tuple(
    PROMOTION_ROUTES.values())

#: Journal record types.
CRITERIA_RECORD_TYPE = "route_promotion_criteria"
EVALUATION_RECORD_TYPE = "route_promotion_evaluation"
VIOLATION_RECORD_TYPE = "route_invariant_violation"

#: The pre-registered criteria version.  Changing any bar
#: (floor, horizon, verdict rule) bumps this, and the bump
#: itself journals a fresh criteria record before the next
#: evaluation — the record IS the registration.
CRITERIA_VERSION = 1

#: Invariant-violation kinds (A5b).
KIND_FRICTION_BREACH = "friction_gate_breach"
KIND_RECONCILIATION_MISMATCH = "reconciliation_mismatch"
KIND_DUPLICATE_ORDER = "duplicate_order"

_WINDOW_BASIS = (
    "route-stamped judgment_shadow records only "
    "(judgment.authority_route naming the route).  Records "
    "without the stamp predate the shadow wiring or belong "
    "to another route and are not evaluated: the window "
    "starts at this wiring's deployment.")

#: Order attribution for the violation scan: playbook -> the
#: route its ORDERS ride.  Roster playbooks (incl. the range
#: family, whose §G scans hold no order authority) order
#: under h1_trigger; the native playbooks under their native
#: routes; the two connected paths under their own routes.
ORDER_ROUTE_BY_PLAYBOOK: dict[str, str] = {
    **{spec.name: "h1_trigger" for spec in ROSTER_SPECS},
    "qm_m5": "qm_m5_native",
    "qm_d1": "qm_d1_native",
    **{playbook: route
       for route, playbook in PROMOTION_ROUTES.items()},
}


def _registry_route(playbook: str) -> str | None:
    """Fallback attribution from the registry itself, so a
    playbook registered after this table was written still
    attributes by its route family rather than vanishing."""
    spec = spec_for(playbook)
    if spec is None:
        return None
    if spec.route == "native":
        return f"{playbook}_native"
    return "h1_trigger"


# ------------------------------------------------------------ criteria

def criteria_id_for(route: str) -> str:
    return f"route-promotion:{route}:v{CRITERIA_VERSION}"


def criteria_record(route: str, *, ts, run_id: str | None = None,
                    bar: str | None = None) -> dict:
    """The pre-registered bars for one route (A5a), as
    journaled BEFORE any evaluation under them reads
    outcomes.  The verdict rule is stated verbatim so a later
    reader can check a verdict against the bars that were in
    force when its window opened — never against bars chosen
    after the fact."""
    return {
        "type": CRITERIA_RECORD_TYPE,
        "criteria_id": criteria_id_for(route),
        "criteria_version": CRITERIA_VERSION,
        "route": route,
        "playbook": PROMOTION_ROUTES[route],
        "registered_at": ts,
        "bar": bar,
        "run_id": run_id,
        "horizon_h": HORIZON_H,
        "floors": {"min_paired_per_arm": MIN_PAIRED_PER_ARM},
        "verdict_rule": {
            "insufficient_data": (
                "fewer than min_paired_per_arm paired outcomes "
                "in either arm at the grading horizon, or no "
                "candidates observed at all (stated as the "
                "different fact it is)"),
            "not_ready": (
                "would-take median R negative, or "
                "discrimination inverted on either read "
                "(would-abstain median R or hit rate above "
                "would-takes)"),
            "promotion_ready": (
                "floors met in both arms, would-take median R "
                ">= 0, discrimination right-way on both reads"),
        },
        "universe": (
            "the route's OWN route-stamped judgment_shadow "
            "history only (judgment.authority_route naming "
            "the route), S4 pairing at the grading horizon, "
            "degenerate-R pairs excluded and counted"),
        "effect": (
            "the verdict BINDS (owner ruling 2026-11-08): "
            "promotion_ready promotes the route to governing "
            "through the judgment seam; not_ready demotes a "
            "verdict-promoted route; insufficient_data "
            "changes nothing; an invariant violation demotes "
            "any governing route until a fresh evaluation "
            "re-promotes it"),
        "note": (
            "registered before any evaluation under these "
            "criteria reads outcomes — the bars may not move "
            "after results are seen (AMENDMENTS-1 §A5a)"),
    }


# ------------------------------------------------------------ state

def new_state() -> dict:
    """The standing state (rides inside the S3 audit state
    file, under its own key — the self-audit's arrangement):
    the shared evaluator's standing block plus this module's
    criteria registrations, applied-verdict bookkeeping,
    duplicate-scan memory and last journaled evaluation."""
    state = _rp.new_state()
    state.update({
        "criteria": {},
        "applied": {},
        "filled_identities": {},
        "last_evaluation": None,
        "last_journaled_evaluation": None,
    })
    return state


def ensure_criteria(state: dict, route: str, *, ts,
                    run_id: str | None = None,
                    bar: str | None = None) -> dict | None:
    """Register one route's criteria into the standing state
    if not already registered at the current version;
    returns the journal record when newly registered, else
    None.  Registration is a state write BEFORE the cycle's
    accumulation runs — the ordering the pin enforces."""
    registered = state.setdefault("criteria", {})
    existing = registered.get(route)
    if isinstance(existing, dict) \
            and existing.get("criteria_version") \
            == CRITERIA_VERSION:
        return None
    record = criteria_record(route, ts=ts, run_id=run_id,
                             bar=bar)
    registered[route] = record
    return record


# ------------------------------------------------------------ evaluation

def _evaluation_snapshot(state: dict, snap: dict) -> dict:
    """The route-keyed evaluation view over the shared
    evaluator's playbook-keyed snapshot, with each route's
    verdict bound to the criteria it was computed under."""
    routes: dict[str, Any] = {}
    for route, playbook in PROMOTION_ROUTES.items():
        row = (snap.get("routes") or {}).get(playbook) or {}
        crit = (state.get("criteria") or {}).get(route) or {}
        routes[route] = {
            "route": route,
            "playbook": playbook,
            "criteria_id": crit.get("criteria_id"),
            "criteria_registered_at": crit.get("registered_at"),
            "evaluated": int(row.get("evaluated") or 0),
            "pending_outcomes": int(
                row.get("pending_outcomes") or 0),
            "take": row.get("take") or {},
            "abstain": row.get("abstain") or {},
            "excluded_degenerate_r": int(
                row.get("excluded_degenerate_r") or 0),
            "duplicate_key_records": int(
                row.get("duplicate_key_records") or 0),
            "unmeasurable_expired": int(
                row.get("unmeasurable_expired") or 0),
            "verdict": row.get("verdict")
            or VERDICT_INSUFFICIENT,
            "verdict_reason": row.get("verdict_reason") or "",
        }
    return {
        "horizon_h": HORIZON_H,
        "floors": {"min_paired_per_arm": MIN_PAIRED_PER_ARM},
        "window_start": snap.get("window_start"),
        "window_basis": _WINDOW_BASIS,
        "routes": routes,
    }


def applicable_verdict(state: dict, route: str) -> dict | None:
    """The route's current verdict row IF it may bind, else
    None.  A verdict binds only when it was computed under a
    criteria record registered for the route (present in the
    standing state) and cites that record's id — a verdict
    computed without a preceding criteria record is refused
    here, by construction (A5a's ordering pin)."""
    crit = (state.get("criteria") or {}).get(route)
    if not isinstance(crit, dict) or not crit.get("criteria_id"):
        return None
    snap = state.get("last_evaluation") or {}
    row = (snap.get("routes") or {}).get(route)
    if not isinstance(row, dict):
        return None
    if row.get("criteria_id") != crit.get("criteria_id"):
        return None
    return row


def evaluate_records(records: list[dict], *, now=None) -> dict:
    """One-shot evaluation over a complete record list (an
    exported journal): criteria registered into a fresh
    state, then the same accumulate + snapshot the live cycle
    runs.  The rerunnable evidence view of a route's standing
    verdict."""
    state = new_state()
    if now is None:
        stamps = [str(r.get("ts") or "") for r in records
                  if r.get("type") == "judgment_shadow"]
        now = max(stamps) if stamps else None
    for route in PROMOTION_ROUTES:
        ensure_criteria(state, route, ts=now)
    snap = _rp.accumulate(
        state, records, now=now,
        routes_map=ROUTE_JUDGMENT_ROUTES,
        playbooks=PROMOTION_PLAYBOOKS, basis=_WINDOW_BASIS,
        family="promotion", binds=True)
    state["last_evaluation"] = _evaluation_snapshot(state, snap)
    return state["last_evaluation"]


# ------------------------------------------------------------ the cycle

def cycle(*, state: dict, records: list[dict], now=None,
          damaged_lines: int = 0, window: dict | None = None,
          run_id: str | None = None,
          bar: str | None = None) -> list[dict]:
    """One evaluation cycle over one journal window (the
    self-audit's contract): mutate ``state``, return the
    journal records to append — criteria registrations FIRST
    (always ahead of any verdict computed under them), then
    the evaluation record when the snapshot changed.  The
    FIRST cycle always journals an evaluation record: that
    record is the deployment marker stating the binding
    evaluation has opened.  A damaged window degrades the
    cycle (counted, nothing folded), never a guessed
    grading."""
    for key, value in new_state().items():
        state.setdefault(key, value)
    if damaged_lines:
        state["damaged_cycles"] = int(
            state.get("damaged_cycles") or 0) + 1
        return []
    out: list[dict] = []
    ts = bar if bar is not None else (
        str(now) if now is not None else None)
    for route in PROMOTION_ROUTES:
        record = ensure_criteria(
            state, route, ts=ts, run_id=run_id, bar=bar)
        if record is not None:
            out.append(record)
    snap = _rp.accumulate(
        state, records, now=now,
        routes_map=ROUTE_JUDGMENT_ROUTES,
        playbooks=PROMOTION_PLAYBOOKS, basis=_WINDOW_BASIS,
        family="promotion", binds=True)
    evaluation = _evaluation_snapshot(state, snap)
    state["last_evaluation"] = evaluation
    state["last_cycle_bar"] = bar
    if evaluation == state.get("last_journaled_evaluation") \
            and not out:
        return out
    state["cycles"] = int(state.get("cycles") or 0) + 1
    state["last_journaled_evaluation"] = evaluation
    out.append({
        "type": EVALUATION_RECORD_TYPE, "bar": bar,
        "run_id": run_id, "window": window, **evaluation,
        "note": (
            "Binding promotion evaluation over the routes' "
            "own route-stamped shadow history (S4 pairing, "
            f"+{HORIZON_H}h, degenerate-R excluded), under "
            "criteria registered before the window's outcomes "
            "were read.  promotion_ready promotes the route "
            "through the judgment seam; not_ready demotes a "
            "verdict-promoted route; invariant violations "
            "demote any governing route (owner ruling "
            "2026-11-08)."),
    })
    return out


# ------------------------------------------------------------ violations

def _row_route(row: dict) -> str | None:
    """The route a journal row attributes to: an explicit
    route field, the judgment block's authority stamp, or —
    for order records — the route the row's playbook orders
    under."""
    for key in ("route", "authority_route"):
        value = row.get(key)
        if value:
            return str(value)
    judgment = row.get("judgment") or {}
    if judgment.get("authority_route"):
        return str(judgment["authority_route"])
    playbook = str(row.get("playbook") or "")
    if playbook:
        return ORDER_ROUTE_BY_PLAYBOOK.get(playbook) \
            or _registry_route(playbook)
    return None


def _friction_ratio_of(row: dict, decision: dict) -> float | None:
    for source in (row, decision):
        value = source.get("friction_ratio")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def scan_violations(records: list[dict], *,
                    filled_identities: dict | None = None,
                    ) -> tuple[list[dict], dict]:
    """Scan one journal window for invariant violations
    (A5b).  Returns (violations, filled_identities): the
    standing map of placed-order identities
    (``route|setup_id`` -> first placement) the duplicate
    detector remembers across windows; pass the standing map
    in and persist the returned one.

    Recognised shapes, each attributed to a route:

    - an explicit ``route_invariant_violation`` record (the
      forward contract: any producer may journal one with a
      route + a kind from this module's vocabulary);
    - a broker ``reconciliation`` record whose event is a
      position divergence, attributed by its own route /
      playbook fields;
    - an order record (evaluation / fill-shaped) that PLACED
      (order_result filled / placed_pending) carrying a
      friction ratio above the gate's limit — the gate
      refused nothing here; an order went through priced
      beyond the invariant (a friction REFUSAL record is the
      gate working and never appears here);
    - a second PLACED fill for one setup identity under one
      route (a duplicate order — the working-order guard
      refusing a twin is, likewise, the guard working).
    """
    filled = filled_identities if filled_identities \
        is not None else {}
    violations: list[dict] = []

    def _add(route, kind, evidence, ts, source):
        violations.append({
            "route": route, "kind": kind,
            "evidence": evidence, "ts": ts, "source": source})

    for row in records:
        if not isinstance(row, dict):
            continue
        rtype = row.get("type")
        if rtype == VIOLATION_RECORD_TYPE:
            # The runner's phase journals its own normalized
            # copy of every processed violation (carrying the
            # demotion_applied / scan_source marks); that copy
            # lands in the NEXT window and must never re-fire
            # as a fresh signal — one violation, one demotion,
            # and a later fresh verdict can release it.
            if "demotion_applied" in row \
                    or "scan_source" in row:
                continue
            route = _row_route(row)
            kind = str(row.get("kind") or "")
            if route and kind in (
                    KIND_FRICTION_BREACH,
                    KIND_RECONCILIATION_MISMATCH,
                    KIND_DUPLICATE_ORDER):
                _add(route, kind, row.get("evidence") or {},
                     row.get("ts"), "explicit_record")
            continue
        if rtype == "reconciliation" \
                and row.get("event") \
                == "divergence_position_mismatch":
            route = _row_route(row)
            if route:
                _add(route, KIND_RECONCILIATION_MISMATCH, {
                    "trade_id": row.get("trade_id"),
                    "event": row.get("event"),
                    "note": row.get("note")},
                    row.get("detected_at")
                    or row.get("broker_ts"),
                    "reconciliation_record")
            continue
        decision = row.get("decision")
        if not isinstance(decision, dict):
            continue
        order_result = decision.get("order_result")
        if order_result not in ("filled", "placed_pending"):
            continue
        route = _row_route(row)
        if not route:
            continue
        ratio = _friction_ratio_of(row, decision)
        if ratio is not None and ratio > FRICTION_RATIO_LIMIT:
            _add(route, KIND_FRICTION_BREACH, {
                "friction_ratio": ratio,
                "friction_limit": FRICTION_RATIO_LIMIT,
                "order_result": order_result,
                "playbook": row.get("playbook")},
                row.get("ts"), "order_record")
        if order_result != "filled":
            continue
        candidate = row.get("candidate") or {}
        identity = candidate.get("setup_id") \
            or row.get("alert_id") \
            or candidate.get("alert_id")
        if not identity:
            continue
        key = f"{route}|{identity}"
        first = filled.get(key)
        if first is not None:
            _add(route, KIND_DUPLICATE_ORDER, {
                "setup_id": str(identity),
                "first_ts": first.get("ts"),
                "duplicate_ts": row.get("ts")},
                row.get("ts"), "order_record")
        else:
            filled[key] = {"route": route,
                           "ts": row.get("ts")}
    return violations, filled


# ------------------------------------------------------------ rendering

_VERDICT_PLAIN = {
    VERDICT_READY: "PROMOTION READY — route governs through "
                   "the judgment seam",
    VERDICT_NOT_READY: "not ready — route stays shadow",
    VERDICT_INSUFFICIENT: "not enough data yet — route stays "
                          "shadow",
}


def _r(value) -> str:
    return "—" if value is None else f"{float(value):+.2f}R"


def _pct(value) -> str:
    return "—" if value is None else f"{float(value):.0%}"


def render_lines(state: dict) -> list[str]:
    """The /status block for the two binding routes, plain
    words first: each route's standing verdict, the criteria
    it was computed under, and the numbers behind it."""
    snap = state.get("last_evaluation")
    if not snap:
        return ["route promotion (binding): no evaluation "
                "cycle has run yet — hot_scalp_promotion and "
                "resting_limit stay shadow (no independent "
                "demo orders) until a cycle journals a verdict "
                "under pre-registered criteria"]
    lines = [
        f"route promotion (binding): graded at +{HORIZON_H}h, "
        f"floor {MIN_PAIRED_PER_ARM} paired outcomes per arm; "
        f"window opened {snap.get('window_start') or '—'} "
        f"(route-stamped shadow history only); verdicts bind "
        f"under the owner's 2026-11-08 ruling"]
    damaged = int(state.get("damaged_cycles") or 0)
    if damaged:
        lines.append(
            f"  journal damage degraded {damaged} cycle(s) — "
            "those windows were not graded, never guessed")
    applied = state.get("applied") or {}
    for route in PROMOTION_ROUTES:
        row = (snap.get("routes") or {}).get(route) or {}
        verdict = row.get("verdict") or VERDICT_INSUFFICIENT
        take = row.get("take") or {}
        abstain = row.get("abstain") or {}
        head = (f"  {route}: "
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
        lines.append(
            f"    criteria {row.get('criteria_id') or '—'} "
            f"(registered "
            f"{row.get('criteria_registered_at') or '—'}) · "
            f"{row.get('verdict_reason') or ''}")
        effect = applied.get(route) or {}
        if effect.get("verdict"):
            lines.append(
                f"    applied: {effect['verdict']} under "
                f"{effect.get('criteria_id') or '—'} at "
                f"{effect.get('evaluation_ts') or '—'}")
    return lines
