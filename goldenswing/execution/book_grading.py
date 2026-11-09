"""Book-vs-S1 PAIRED GRADING (MarketSituation Pass 9).

Every scenario in the book is born stamped with the
situation it was formed under (``formed_situation_id`` —
Pass 0/1's snapshot id) and dies with a journaled close
reason.  Until this pass, nothing ever paired the two:
the book knew what it proposed and how each thesis ended,
but not WHICH market pictures produced theses that
survived.  Judgment's belief about that was assertion.

This module is the pairing.  For every closed scenario it
reads, from the journal's own records:

- the scenario's terminal fact (its ``close_reason``,
  verbatim from the book): ``invalidated`` — the thesis
  died by its own stated terms; ``filled`` — the scenario
  entered and its trade completed; ``expired`` — its
  validity window ran out unresolved; ``superseded`` /
  ``rejected`` / ``merged`` — closed by bookkeeping, and
  any other reason reads ``other``, counted, never
  silently renamed;
- the situation snapshot it was formed under, folded from
  the journal's ``situation`` snapshot records by id, and
  the CELL that snapshot describes for this scenario: its
  formation timeframe's structure phase, the scenario's
  direction against that timeframe's measured bias
  (with / against / flat / neutral), and the relationship
  kind live between its formation and trigger clocks.

Grades land per family and per cell:

- THESIS SURVIVAL = the share of graded closes that were
  NOT invalidated — a thesis survives when the market
  never proved it wrong inside its own terms (it filled,
  expired standing, or was superseded), and dies only on
  ``invalidated``.  The full outcome distribution rides
  beside the rate, so the definition is checkable from
  the record itself.
- A family or cell below the stated floor (20 graded
  closes) reads ``insufficient_data`` — its distribution
  is still reported, but no survival rate is dressed up
  as measured.
- Degenerate scenarios (geometry with no measurable risk:
  entry or stop absent, entry == stop, no targets) are
  excluded from every grade and counted beside them —
  their survival is undefined, not zero.
- A scenario whose formation situation was never folded
  (it predates this grading's window, or its snapshot
  scrolled past the carried set) is UNPAIRED: counted in
  its family's totals and in the unpaired count, graded
  into its family's outcome distribution, but into NO
  cell — the pairing is the mechanism, and an unpaired
  scenario is a stated fact, never a dropped row.
- Per family, the S1 cross-tab: of the graded scenarios
  S1 (the per-candidate Judgment core) took vs abstained
  on, how many were invalidated — the book's outcomes
  graded against S1's own calls, paired per scenario.

POSTURE: grading only.  Nothing here feeds a decision,
moves a parameter, or touches authority — the record is
evidence for the owner and for later passes, in the §G
discipline.  The cycle host is the unified self-audit
(same windows, same cadence, same state file, own key);
the runner feeds it the book's closed scenarios (the
book's persisted state is their durable form) and it
dedupes by scenario identity, so re-feeding is harmless.
"""
from __future__ import annotations

#: Journal record type for one grading snapshot.
RECORD_TYPE = "book_grading"

#: Graded closes required before a family or cell survival
#: rate reads as measured (the house floor — E3 trust and
#: the promotion evaluations use the same 20).
MIN_GRADED = 20

#: Carried formation snapshots (one situation per H1 close:
#: the cap is months of snapshots) and folded scenario ids.
_SITUATION_CAP = 1500
_FOLDED_CAP = 8000

#: The book's terminal vocabulary (scenario_book's close
#: reasons).  A reason outside this set grades as "other".
KNOWN_OUTCOMES = ("invalidated", "expired", "filled",
                  "superseded", "rejected", "merged")

_PAIRING_BASIS = (
    "each closed scenario paired by its "
    "formed_situation_id to the journal's situation "
    "snapshot record of that id; the cell is the "
    "snapshot's formation-timeframe structure phase, the "
    "scenario's direction against that timeframe's "
    "measured bias, and the engine relationship kind "
    "between its formation and trigger clocks.  Thesis "
    "survival = graded closes not invalidated, over all "
    "graded closes; degenerate geometry excluded and "
    "counted; scenarios whose snapshot was never folded "
    "are unpaired — counted, graded into family totals, "
    "into no cell")


# ------------------------------------------------------------ state

def new_state() -> dict:
    """The standing state (rides inside the S3 audit state
    file, under its own key): carried formation snapshots,
    folded scenario identities, and the per-family /
    per-cell aggregates (counters only — the distributions
    are the state)."""
    return {
        "cycles": 0,
        "damaged_cycles": 0,
        "situations": {},
        "folded": [],
        "families": {},
        "cells": {},
        "totals": {
            "closed_seen": 0, "graded": 0,
            "degenerate_excluded": 0, "unpaired": 0,
            "cell_unknown": 0},
        "window_start": None,
        "last_snapshot": None,
        "last_journaled": None,
        "last_cycle_bar": None,
    }


def _aggregate() -> dict:
    return {"graded": 0,
            "outcomes": {name: 0 for name in
                         KNOWN_OUTCOMES + ("other",)}}


def _family_slot(state: dict, family: str) -> dict:
    slot = state["families"].setdefault(family, {
        **_aggregate(),
        "degenerate_excluded": 0, "unpaired": 0,
        "cell_unknown": 0,
        "s1": {"take": {"graded": 0, "invalidated": 0},
               "abstain": {"graded": 0, "invalidated": 0},
               "none": 0},
    })
    return slot


# ------------------------------------------------------------ pairing

def _fold_situations(state: dict, records: list[dict]
                     ) -> None:
    """Carry the compact cell-source of each journaled
    situation snapshot, keyed by its id (the full snapshot
    is the journal's; the grading needs only what a cell
    is made of)."""
    situations: dict = state["situations"]
    for row in records or []:
        if row.get("type") != "situation" \
                or row.get("kind") != "situation_snapshot":
            continue
        situation = row.get("situation") or {}
        sid = row.get("situation_id") \
            or situation.get("situation_id")
        if not sid:
            continue
        per_tf = {}
        for tf, tf_state in (situation.get("per_tf")
                             or {}).items():
            if isinstance(tf_state, dict):
                per_tf[tf] = {
                    "bias": tf_state.get("bias"),
                    "structure_phase":
                        tf_state.get("structure_phase")}
        rels = []
        for rel in situation.get("relationships") or []:
            if isinstance(rel, dict):
                rels.append({
                    "htf": rel.get("htf"),
                    "ltf": rel.get("ltf"),
                    "kind": rel.get("kind"),
                    "source": rel.get("source")})
        situations[str(sid)] = {"per_tf": per_tf,
                                "relationships": rels}
    overflow = len(situations) - _SITUATION_CAP
    if overflow > 0:
        for key in list(situations)[:overflow]:
            del situations[key]


def _cell_of(scenario: dict, snap: dict) -> str | None:
    """The situation cell a scenario was formed in, from
    its OWN declared clocks and the paired snapshot — or
    None when the snapshot carries no state for the
    scenario's formation timeframe (the cell is then
    unmeasurable, stated as such)."""
    clocks = scenario.get("clocks") or {}
    ftf = clocks.get("formation_tf")
    ttf = clocks.get("trigger_tf")
    tf_state = (snap.get("per_tf") or {}).get(ftf)
    if tf_state is None:
        return None
    bias = tf_state.get("bias")
    direction = int(scenario.get("direction") or 0)
    if direction == 0:
        alignment = "neutral"
    elif bias is None:
        alignment = "unknown"
    elif int(bias) == 0:
        alignment = "flat"
    elif int(bias) == direction:
        alignment = "with"
    else:
        alignment = "against"
    rel_kind = "none"
    for rel in snap.get("relationships") or []:
        if rel.get("source") not in (None, "engine"):
            continue
        if rel.get("htf") == ftf and rel.get("ltf") == ttf \
                and rel.get("kind"):
            rel_kind = str(rel["kind"])
            break
    return (f"{ftf}:{tf_state.get('structure_phase')}:"
            f"{alignment}:rel={rel_kind}")


def _is_degenerate(scenario: dict) -> bool:
    """Geometry with no measurable risk cannot be graded:
    a survival read needs an invalidation price distinct
    from the entry and at least one target."""
    geometry = scenario.get("geometry") or {}
    entry = geometry.get("entry")
    stop = geometry.get("stop")
    if entry is None or stop is None:
        return True
    try:
        if float(entry) == float(stop):
            return True
    except (TypeError, ValueError):
        return True
    return not geometry.get("targets")


def _grade_one(state: dict, scenario: dict) -> None:
    family = str(scenario.get("family") or "?")
    slot = _family_slot(state, family)
    totals = state["totals"]
    totals["closed_seen"] = int(totals.get("closed_seen")
                                or 0) + 1
    closed_ts = scenario.get("closed_ts")
    if closed_ts and (state.get("window_start") is None
                      or str(closed_ts)
                      < str(state["window_start"])):
        state["window_start"] = str(closed_ts)
    if _is_degenerate(scenario):
        totals["degenerate_excluded"] = int(
            totals.get("degenerate_excluded") or 0) + 1
        slot["degenerate_excluded"] = int(
            slot.get("degenerate_excluded") or 0) + 1
        return
    reason = str(scenario.get("close_reason") or "")
    outcome = reason if reason in KNOWN_OUTCOMES else "other"
    slot["graded"] += 1
    slot["outcomes"][outcome] += 1
    totals["graded"] = int(totals.get("graded") or 0) + 1
    # S1 cross-tab: what the per-candidate core said about
    # this scenario, graded by what the book recorded.
    s1 = scenario.get("s1")
    decision = (s1 or {}).get("decision") \
        if isinstance(s1, dict) else None
    if decision in ("take", "abstain"):
        arm = slot["s1"][decision]
        arm["graded"] += 1
        if outcome == "invalidated":
            arm["invalidated"] += 1
    else:
        slot["s1"]["none"] = int(slot["s1"].get("none")
                                 or 0) + 1
    # The situation pairing.
    sid = scenario.get("formed_situation_id")
    snap = state["situations"].get(str(sid)) if sid else None
    if snap is None:
        totals["unpaired"] = int(totals.get("unpaired")
                                 or 0) + 1
        slot["unpaired"] = int(slot.get("unpaired") or 0) + 1
        return
    cell = _cell_of(scenario, snap)
    if cell is None:
        totals["cell_unknown"] = int(
            totals.get("cell_unknown") or 0) + 1
        slot["cell_unknown"] = int(
            slot.get("cell_unknown") or 0) + 1
        return
    cell_slot = state["cells"].setdefault(cell, _aggregate())
    cell_slot["graded"] += 1
    cell_slot["outcomes"][outcome] += 1


# ------------------------------------------------------------ snapshot

def _row(slot: dict) -> dict:
    graded = int(slot.get("graded") or 0)
    invalidated = int((slot.get("outcomes") or {}).get(
        "invalidated") or 0)
    return {
        "graded": graded,
        "outcomes": dict(slot.get("outcomes") or {}),
        "invalidated": invalidated,
        "survival_rate": (
            round((graded - invalidated) / graded, 4)
            if graded else None),
        "status": ("measured" if graded >= MIN_GRADED
                   else "insufficient_data"),
    }


def snapshot(state: dict) -> dict:
    """The grading table over the standing state: per
    family (with its S1 cross-tab and pairing counts) and
    per situation cell."""
    families: dict[str, dict] = {}
    for family, slot in sorted(
            (state.get("families") or {}).items()):
        row = _row(slot)
        row.update({
            "degenerate_excluded": int(
                slot.get("degenerate_excluded") or 0),
            "unpaired": int(slot.get("unpaired") or 0),
            "cell_unknown": int(slot.get("cell_unknown")
                                or 0),
            "s1": slot.get("s1") or {},
        })
        families[family] = row
    cells = {key: _row(slot) for key, slot in sorted(
        (state.get("cells") or {}).items())}
    return {
        "floors": {"min_graded": MIN_GRADED},
        "pairing_basis": _PAIRING_BASIS,
        "window_start": state.get("window_start"),
        "totals": dict(state.get("totals") or {}),
        "families": families,
        "cells": cells,
    }


# ------------------------------------------------------------ the cycle

def cycle(*, state: dict, records: list[dict],
          closed_scenarios: list[dict] | None = None,
          now=None, damaged_lines: int = 0,
          window: dict | None = None,
          run_id: str | None = None,
          bar: str | None = None) -> dict | None:
    """One grading cycle (the self-audit's contract): fold
    the window's situation snapshots, grade every closed
    scenario not yet folded, and return the journal record
    when the table changed — or None.  The FIRST cycle
    always journals.  A damaged window degrades the cycle
    (counted, nothing folded)."""
    for key, value in new_state().items():
        state.setdefault(key, value)
    if damaged_lines:
        state["damaged_cycles"] = int(
            state.get("damaged_cycles") or 0) + 1
        return None
    _fold_situations(state, records)
    folded = set(state.get("folded") or [])
    for scenario in closed_scenarios or []:
        sid = scenario.get("scenario_id")
        if sid is None or str(sid) in folded:
            continue
        folded.add(str(sid))
        _grade_one(state, scenario)
    folded_list = sorted(folded)
    state["folded"] = folded_list[-_FOLDED_CAP:]
    snap = snapshot(state)
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
            "Book-vs-S1 paired grading: every closed "
            "scenario's outcome paired to the situation "
            "snapshot it was formed under — thesis survival "
            "per family and per situation cell, degenerate "
            "geometry excluded and counted, unpaired "
            "scenarios counted, never dropped.  Grading "
            "only: this record feeds no decision and moves "
            "no authority."),
    }


def evaluate_closed(records: list[dict],
                    closed_scenarios: list[dict], *,
                    now=None) -> dict:
    """One-shot grading over a complete record list + the
    book's closed scenarios: the same fold the live cycle
    performs, over a fresh state."""
    state = new_state()
    cycle(state=state, records=records,
          closed_scenarios=closed_scenarios, now=now)
    return state["last_snapshot"]


# ------------------------------------------------------------ rendering

def _pct(value) -> str:
    return "—" if value is None else f"{float(value):.0%}"


def render_lines(state: dict) -> list[str]:
    """The /status block, plain words first: per family,
    the graded count, the invalidated count, and the
    survival rate — or insufficient_data with the n
    named."""
    snap = state.get("last_snapshot")
    if not snap:
        return [("book grading (Pass 9): no grading cycle "
                 "has run yet — thesis survival per family "
                 "and per situation cell lands here once the "
                 "self-audit cycle journals it")]
    totals = snap.get("totals") or {}
    lines = [
        (f"book grading (Pass 9): {totals.get('graded', 0)} "
         f"closed scenarios graded against the situation "
         f"they were formed under · "
         f"{totals.get('unpaired', 0)} unpaired (snapshot "
         f"never folded) · "
         f"{totals.get('degenerate_excluded', 0)} degenerate "
         f"excluded · floor {MIN_GRADED} graded for a "
         f"measured rate")]
    damaged = int(state.get("damaged_cycles") or 0)
    if damaged:
        lines.append(
            f"  journal damage degraded {damaged} cycle(s) "
            "— those windows were not graded, never guessed")
    for family, row in (snap.get("families") or {}).items():
        outcomes = row.get("outcomes") or {}
        if row.get("status") == "measured":
            head = (f"  {family}: survival "
                    f"{_pct(row.get('survival_rate'))} over "
                    f"{row.get('graded')} graded "
                    f"(invalidated {outcomes.get('invalidated', 0)}, "
                    f"filled {outcomes.get('filled', 0)}, "
                    f"expired {outcomes.get('expired', 0)})")
        else:
            head = (f"  {family}: not enough data yet — "
                    f"{row.get('graded', 0)}/{MIN_GRADED} "
                    f"graded closes (invalidated "
                    f"{outcomes.get('invalidated', 0)}, filled "
                    f"{outcomes.get('filled', 0)}, expired "
                    f"{outcomes.get('expired', 0)})")
        s1 = row.get("s1") or {}
        take = (s1.get("take") or {})
        abstain = (s1.get("abstain") or {})
        if take.get("graded") or abstain.get("graded"):
            head += (
                f" · S1 took {take.get('graded', 0)} "
                f"({take.get('invalidated', 0)} invalidated), "
                f"abstained {abstain.get('graded', 0)} "
                f"({abstain.get('invalidated', 0)} "
                f"invalidated)")
        lines.append(head)
    measured_cells = [key for key, row in
                      (snap.get("cells") or {}).items()
                      if row.get("status") == "measured"]
    thin_cells = len(snap.get("cells") or {}) \
        - len(measured_cells)
    if measured_cells:
        best = max(measured_cells, key=lambda key: (
            snap["cells"][key].get("survival_rate") or 0.0,
            snap["cells"][key].get("graded") or 0))
        row = snap["cells"][best]
        lines.append(
            f"  most durable cell: {best} — survival "
            f"{_pct(row.get('survival_rate'))} over "
            f"{row.get('graded')} graded")
    if thin_cells:
        lines.append(
            f"  {thin_cells} situation cell(s) below the "
            f"floor — distributions in the journal record, "
            f"rates not dressed up as measured")
    return lines
