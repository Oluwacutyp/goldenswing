"""Coverage-gap CONSUMPTION (MarketSituation Pass 9 —
AMENDMENTS-1 §A6's second half).

Pass 3's book cycle detects blind spots: a situation
configuration (a measured per-TF state, or a cross-TF
relationship kind) that persists uncovered across the
stated window journals ONE ``coverage_gap`` record per
episode (execution/judgment_book.py's CoverageTracker).
Detection alone leaves the fact scattered — one record per
episode, no standing account of which gaps are still open,
how old they are, or whether a gap is the system standing
aside ON PURPOSE versus a hole nobody is watching.

This module is the standing account.  It folds the
``coverage_gap`` records into episodes, adjudicates each
open episode against the LATEST situation and the book's
live scenarios every cycle (an episode closes when its
configuration is covered — reusing judgment_book's own
``configuration_covered`` — or leaves the situation), and
classifies every open gap from the situation's own state:

- ``deliberate_data`` — the situation itself says the read
  is not there: the data-quality gate reads ``halt``, or
  the configuration's timeframe is not ``ready`` in the
  latest situation (or is named in its missing list).  The
  engine does not propose into a market it cannot see;
  that is the fail-closed design working, not a hole.
- ``deliberate_news`` — the situation's activity block
  reports a news blackout: the desk's standing rule is to
  stand aside, and the situation says so.
- ``unwatched`` — no stand-aside fact in the situation
  explains the absence.  This is the class that matters:
  a market configuration the whole book can see and none
  of its families describes.

Classification is read from situation fields, named on
every gap row with the field that decided it — never
guessed, never inferred from the gap's age or shape.  A
classification can change as the situation changes (a gap
that was unwatched becomes deliberate the hour the feed
dies); the current read is what the report states, with
the episode's origin kept beside it.

POSTURE: a gap is a REPORT, never an auto-build order.
Findings feed back nowhere automatically — no family is
spawned, no parameter moves, no decision reads this
record (§G discipline).  The cycle host is the unified
self-audit, in the standing arrangement (same windows,
same cadence, same state file, own key).
"""
from __future__ import annotations

import copy

from . import judgment_book as _jb
from . import range_promotion as _eval

#: Journal record type for one standing coverage report.
RECORD_TYPE = "coverage_report"

CLASS_DELIBERATE_DATA = "deliberate_data"
CLASS_DELIBERATE_NEWS = "deliberate_news"
CLASS_UNWATCHED = "unwatched"

_CLASSIFICATION_BASIS = (
    "each open gap is classified from the LATEST "
    "situation's own fields: deliberate_data when the "
    "data-quality gate reads halt or the configuration's "
    "timeframe is not ready / named missing; "
    "deliberate_news when the activity block reports a "
    "news blackout; otherwise unwatched — no stand-aside "
    "fact in the situation explains the absence")


# ------------------------------------------------------------ state

def new_state() -> dict:
    """The standing state (rides inside the S3 audit state
    file, under its own key): the episode ledger plus the
    journaling bookkeeping."""
    return {
        "cycles": 0,
        "damaged_cycles": 0,
        "episodes": {},
        "episodes_total": 0,
        "closed_total": 0,
        "closed_by_class": {},
        "last_snapshot": None,
        "last_journaled": None,
        "last_cycle_bar": None,
    }


# ------------------------------------------------------------ classification

def _config_tfs(config: dict) -> list[str]:
    if config.get("kind") == "relationship":
        return [tf for tf in (config.get("htf"),
                              config.get("ltf")) if tf]
    return [config["tf"]] if config.get("tf") else []


def classify_gap(config: dict, situation: dict | None
                 ) -> tuple[str, str]:
    """One open gap's classification from the situation's
    own state — (class, basis) with the deciding field
    named.  Absent situation: the gap stands as first
    classified (the caller keeps the stored class); this
    function is only called with a situation in hand."""
    situation = situation or {}
    dq = situation.get("data_quality") or {}
    if dq.get("health") == "halt":
        return CLASS_DELIBERATE_DATA, (
            "data_quality.health is 'halt' — the data "
            "gate has halted market reads (fail-closed); "
            "no family proposes into a halted read")
    per_tf = situation.get("per_tf") or {}
    missing = [str(m) for m in dq.get("missing") or []]
    for tf in _config_tfs(config):
        state = per_tf.get(tf) or {}
        readiness = state.get("readiness")
        if readiness is not None and readiness != "ready":
            return CLASS_DELIBERATE_DATA, (
                f"per_tf.{tf}.readiness is "
                f"'{readiness}' in the latest situation — "
                "the configuration's own timeframe cannot "
                "currently be read")
        if any(m == tf or m.startswith(f"{tf}:")
               for m in missing):
            return CLASS_DELIBERATE_DATA, (
                f"data_quality.missing names {tf} in the "
                "latest situation — the input is absent, "
                "and absent stays absent")
    activity = situation.get("activity") or {}
    if activity.get("news_blackout") is True:
        return CLASS_DELIBERATE_NEWS, (
            "activity.news_blackout is true in the latest "
            "situation — the desk stands aside around the "
            "news by its own rule")
    return CLASS_UNWATCHED, (
        "no stand-aside fact in the latest situation "
        "(data gate not halted, the configuration's "
        "timeframes ready, no news blackout) explains "
        "the absence — the book can see this configuration "
        "and no family describes it")


def _age_hours(first_seen_ts, now_ts) -> float | None:
    first = _eval._ts(first_seen_ts)
    if first is None or now_ts is None:
        return None
    return round(
        (now_ts - first).total_seconds() / 3600.0, 2)


# ------------------------------------------------------------ snapshot

def _snapshot(state: dict, *, situation: dict | None,
              live: list[dict] | None, now_ts) -> dict:
    episodes = state.get("episodes") or {}
    open_rows: list[dict] = []
    counts = {CLASS_DELIBERATE_DATA: 0,
              CLASS_DELIBERATE_NEWS: 0, CLASS_UNWATCHED: 0}
    for key in sorted(episodes):
        ep = episodes[key]
        if ep.get("status") != "open":
            continue
        counts[ep["classification"]] = \
            counts.get(ep["classification"], 0) + 1
        open_rows.append({
            "config_key": key,
            "label": ep.get("label"),
            "configuration": ep.get("configuration"),
            "first_seen_ts": ep.get("first_seen_ts"),
            "fired_ts": ep.get("fired_ts"),
            "age_hours": _age_hours(
                ep.get("first_seen_ts"), now_ts),
            "cycles_at_fire": ep.get("cycles_at_fire"),
            "classification": ep.get("classification"),
            "classification_basis": ep.get(
                "classification_basis"),
        })
    open_rows.sort(key=lambda row: (
        -(row["age_hours"] or 0.0), row["config_key"]))
    coverage = None
    if situation is not None:
        configs = _jb.configurations_of(situation)
        live = live or []
        rows = []
        covered_n = 0
        for config in configs:
            covering = sorted({
                str(sc.get("family") or "?")
                for sc in live
                if _jb.configuration_covered(config, [sc])})
            covered = bool(covering)
            covered_n += 1 if covered else 0
            rows.append({
                "key": config["key"], "label": config["label"],
                "covered": covered, "covered_by": covering})
        coverage = {
            "situation_id": situation.get("situation_id"),
            "configurations": len(configs),
            "covered": covered_n,
            "uncovered": len(configs) - covered_n,
            "rows": rows,
            "live_families": sorted({
                str(sc.get("family") or "?") for sc in live}),
        }
    return {
        "classification_basis": _CLASSIFICATION_BASIS,
        "coverage": coverage,
        "open_gaps": open_rows,
        "open_by_class": counts,
        "episodes_total": int(state.get("episodes_total")
                              or 0),
        "closed_total": int(state.get("closed_total") or 0),
        "closed_by_class": dict(state.get("closed_by_class")
                                or {}),
    }


# ------------------------------------------------------------ the cycle

def cycle(*, state: dict, records: list[dict], now=None,
          situation: dict | None = None,
          live: list[dict] | None = None,
          damaged_lines: int = 0, window: dict | None = None,
          run_id: str | None = None,
          bar: str | None = None) -> dict | None:
    """One coverage cycle (the self-audit's contract): fold
    new ``coverage_gap`` records into episodes, adjudicate
    open episodes against the latest situation + live book,
    and return the report record when the standing account
    changed — or None.  The FIRST cycle always journals.
    A damaged window degrades the cycle (counted, nothing
    folded)."""
    for key, value in new_state().items():
        state.setdefault(key, value)
    if damaged_lines:
        state["damaged_cycles"] = int(
            state.get("damaged_cycles") or 0) + 1
        return None
    now_ts = _eval._ts(now) if now is not None else None
    episodes: dict = state["episodes"]

    # --- fold new gap episodes.
    for row in records or []:
        if row.get("type") != "coverage_gap":
            continue
        key = str(row.get("config_key") or "")
        if not key:
            continue
        existing = episodes.get(key)
        if existing is not None \
                and existing.get("status") == "open":
            continue  # one open episode per configuration
        config = copy.deepcopy(row.get("configuration")
                               or {})
        classification, basis = (
            classify_gap(config, situation)
            if situation is not None
            else (CLASS_UNWATCHED,
                  ("no situation read accompanied the gap's "
                   "arrival; classified unwatched until a "
                   "situation read adjudicates it")))
        episodes[key] = {
            "status": "open",
            "label": row.get("label"),
            "configuration": config,
            "first_seen_ts": row.get("first_seen_ts"),
            "fired_ts": row.get("ts"),
            "first_situation_id": row.get(
                "first_situation_id"),
            "cycles_at_fire": row.get("cycles"),
            "classification": classification,
            "classification_basis": basis,
        }
        state["episodes_total"] = int(
            state.get("episodes_total") or 0) + 1

    # --- adjudicate open episodes against the latest read.
    if situation is not None:
        present = {config["key"]: config
                   for config in _jb.configurations_of(
                       situation)}
        live = live or []
        for key, ep in list(episodes.items()):
            if ep.get("status") != "open":
                continue
            config = present.get(key)
            if config is None or _jb.configuration_covered(
                    config, live):
                ep["status"] = "closed"
                ep["closed_ts"] = str(now) \
                    if now is not None else None
                ep["closed_reason"] = (
                    "configuration covered by a live "
                    "scenario" if config is not None else
                    "configuration left the situation")
                state["closed_total"] = int(
                    state.get("closed_total") or 0) + 1
                by_class = state.setdefault(
                    "closed_by_class", {})
                cls = ep.get("classification") \
                    or CLASS_UNWATCHED
                by_class[cls] = int(by_class.get(cls) or 0) + 1
                continue
            classification, basis = classify_gap(
                ep.get("configuration") or {}, situation)
            ep["classification"] = classification
            ep["classification_basis"] = basis

    snap = _snapshot(state, situation=situation, live=live,
                     now_ts=now_ts)
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
            "Standing coverage account (AMENDMENTS-1 §A6 "
            "consumed): the book's configurations, which "
            "are covered and by whom, and every open "
            "coverage gap with its age and its "
            "classification read from the situation's own "
            "state.  A gap is a report — this record builds "
            "nothing, moves no parameter and feeds no "
            "decision."),
    }


def evaluate_records(records: list[dict], *,
                     situation: dict | None = None,
                     live: list[dict] | None = None,
                     now=None) -> dict:
    """One-shot standing account over a complete record
    list: the same fold + adjudication the live cycle
    performs, over a fresh state."""
    state = new_state()
    if now is None:
        stamps = [str(r.get("ts") or "") for r in records
                  if r.get("type") == "coverage_gap"]
        now = max(stamps) if stamps else None
    cycle(state=state, records=records, now=now,
          situation=situation, live=live)
    return state["last_snapshot"]


# ------------------------------------------------------------ rendering

_CLASS_PLAIN = {
    CLASS_DELIBERATE_DATA: "deliberate (data)",
    CLASS_DELIBERATE_NEWS: "deliberate (news)",
    CLASS_UNWATCHED: "UNWATCHED",
}


def render_lines(state: dict) -> list[str]:
    """The /status block, plain words first: the coverage
    count, then every open gap with its age and class."""
    snap = state.get("last_snapshot")
    if not snap:
        return [("coverage (Pass 9): no coverage cycle has "
                 "run yet — the standing account of covered "
                 "configurations and open gaps lands here "
                 "once the self-audit cycle journals it")]
    lines = []
    coverage = snap.get("coverage")
    if coverage:
        lines.append(
            f"coverage (Pass 9): {coverage.get('covered')} "
            f"of {coverage.get('configurations')} situation "
            f"configurations covered by a live scenario · "
            f"{coverage.get('uncovered')} uncovered")
    else:
        lines.append(
            "coverage (Pass 9): no situation read on the "
            "last cycle — coverage of configurations "
            "unmeasured this cycle, gap episodes carried")
    damaged = int(state.get("damaged_cycles") or 0)
    if damaged:
        lines.append(
            f"  journal damage degraded {damaged} cycle(s) "
            "— those windows were not read, never guessed")
    counts = snap.get("open_by_class") or {}
    lines.append(
        f"  open gaps: {len(snap.get('open_gaps') or [])} "
        f"({counts.get(CLASS_UNWATCHED, 0)} unwatched, "
        f"{counts.get(CLASS_DELIBERATE_DATA, 0)} deliberate "
        f"data, {counts.get(CLASS_DELIBERATE_NEWS, 0)} "
        f"deliberate news) · episodes closed so far: "
        f"{snap.get('closed_total', 0)} of "
        f"{snap.get('episodes_total', 0)}")
    for gap in snap.get("open_gaps") or []:
        age = gap.get("age_hours")
        age_txt = f"{age:.1f}h old" if age is not None \
            else "age unmeasured"
        lines.append(
            f"  {gap.get('label')}: "
            f"{_CLASS_PLAIN.get(gap.get('classification'), gap.get('classification'))}"
            f" · {age_txt}")
        lines.append(
            f"    {gap.get('classification_basis') or ''}")
    return lines
