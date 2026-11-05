"""Completion-05 — the unified self-audit (owner completion
program, item 05; S3/S4 completed).

S3 reconciles shadow judgments and theses against forward
prices, and flags material bars nobody evaluated.  S4 grades
the judgment core against the gate stack on paired history.
What neither did was audit the DESK as a whole, every cycle,
in the five shapes a trading operation actually loses money
in:

- MISSED OPPORTUNITIES — setups that were detected or armed
  and never became attempts, each with the reason it died;
- WRONG JUDGMENTS — core takes that lost and abstentions that
  would have won, graded on the S4 pairing (clean set: the
  completion-04 degenerate-R exclusions apply unchanged);
- WEAK SETUPS — attempts the gates executed on a thin
  evidence vector, naming the evidence groups that never
  voted;
- PREMATURE EXITS — closed positions whose own target was
  reached only after the exit, inside the measurement
  window;
- EXECUTION FAILURES — attempts refused at the execution
  boundary, fills that differ materially from intent, and
  candidate decisions that never reached a terminal bucket.

This module is the pure core: one cycle is a function of
(journal records since the watermark, the standing state,
the bars the engine already stores, the cycle clock).  The
runner owns the cadence (it rides the S3 H1 cycle), the
state file and the journal append.  Shadow posture
throughout, exactly like S3/S4: the audit MEASURES the desk,
it never steers it.  Findings feed back as journaled
evidence (per-route rates the judgment records can cite);
no parameter changes, no auto-demotion — promotion and
demotion stay the owner's decisions, like graduation.

Honesty mechanics (the category contract):

- every category distinguishes "none found" (measured, zero)
  from "could not measure": a category whose data is missing
  reports ``state == "unmeasurable"`` with the count of
  unmeasurable ITEMS beside the found count — never a silent
  zero dressed as a clean bill;
- a corrupt journal region (a complete line that does not
  parse, mid-window) degrades EVERY category of that cycle
  to unmeasurable: counts are withheld rather than guessed,
  the watermark still advances (one damaged region degrades
  exactly one cycle), and the damage is counted standing.
  A torn FINAL line (crash mid-append, no trailing newline)
  is held for the next read — the S4 precedent — and is not
  damage;
- the cycle is idempotent over the same journal window:
  the byte watermark plus carried pending state mean a
  re-run over the same bytes finds nothing new to count.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .funnel import (
    CATEGORY_EXECUTION,
    ORDERED_RESULTS,
    TERMINAL_ORDER_REFUSALS,
    category_for_code,
    decision_bucket,
)
from .judgment import GROUP_WEIGHTS
from .judgment_validation import GATE_EXECUTED, build_pairs, gate_class

#: The five audit categories, in report order.
CATEGORIES = ("missed_opportunities", "wrong_judgments",
              "weak_setups", "premature_exits",
              "execution_failures")

#: Worst offenders itemized per category per cycle.
ITEM_LIMIT = 5

#: A candidate whose weighed evidence engaged at most this
#: many of the known correlation groups (score channel
#: included) was judged on a thin vector.
WEAK_MAX_GROUPS = 2

#: Post-exit measurement window (hours): how long after an
#: exit the audit watches for the position's own target —
#: the S3/S4 outer horizon, the setup's own horizon class.
PREMATURE_WINDOW_H = 4

#: A fill whose adverse slippage eats more than this fraction
#: of the trade's own stop distance differs materially from
#: intent (the friction audit's shape, as a finding).
SLIPPAGE_STOP_FRAC = 0.25

#: Carried-state bounds (the S3 pattern: overflow is dropped
#: AND counted, never silent).
PENDING_SHADOW_CAP = 500
PENDING_OPEN_CAP = 500
PENDING_EXIT_CAP = 200

#: A shadow judgment whose reconciliation never lands within
#: this many hours is unmeasurable, not pending forever.
SHADOW_MAX_AGE_H = 48
#: An open record whose close never lands within this many
#: hours stops being carried.
OPEN_MAX_AGE_H = 24 * 7


# ------------------------------------------------------------ state
def new_state() -> dict:
    """The standing self-audit state (rides inside the S3
    audit state file, under its own key)."""
    return {
        "journal_offset": 0,
        "cycles": 0,
        "damaged_cycles": 0,
        "pending_shadows": [],
        "pending_opens": {},
        "pending_exits": [],
        "totals": {cat: {"found": 0, "unmeasurable": 0}
                   for cat in CATEGORIES},
        "wrong_judgments_excluded_degenerate_r": 0,
        "dropped": {"shadows": 0, "opens": 0, "exits": 0},
        "routes": {},
        "last_cycle": None,
    }


def _route_slot(state: dict, playbook: str) -> dict:
    return state["routes"].setdefault(playbook, {
        "takes": 0, "take_losses": 0, "abstains": 0,
        "abstain_wins": 0, "attempts": 0, "weak": 0})


def route_feedback(state: dict) -> dict:
    """Per-route rates from the standing counters — the named
    evidence block judgment records cite (req: findings feed
    back as evidence, never as a gate).  Routes with no graded
    history are absent, never zero-filled."""
    out: dict[str, Any] = {}
    for playbook, slot in sorted(
            (state.get("routes") or {}).items()):
        takes = int(slot.get("takes") or 0)
        abstains = int(slot.get("abstains") or 0)
        attempts = int(slot.get("attempts") or 0)
        if not (takes or abstains or attempts):
            continue
        out[playbook] = {
            "wrong_take_rate": (round(
                int(slot.get("take_losses") or 0) / takes, 4)
                if takes else None),
            "wrong_abstain_rate": (round(
                int(slot.get("abstain_wins") or 0) / abstains, 4)
                if abstains else None),
            "weak_setup_rate": (round(
                int(slot.get("weak") or 0) / attempts, 4)
                if attempts else None),
            "takes": takes, "abstains": abstains,
            "attempts": attempts,
        }
    return out


# ------------------------------------------------------------ journal window
def read_window(path: str | Path,
                offset: int) -> tuple[list[dict], int, int]:
    """Read the journal bytes appended since ``offset``.

    Returns (records, new_offset, damaged_lines).  Only
    newline-terminated lines are consumed: a torn final line
    (the writer was mid-append) is held — the offset stops
    before it — and is not damage.  A complete line that does
    not parse IS damage (counted); parsing continues so the
    caller can still see the window's shape, but the cycle
    contract treats a damaged window as unmeasurable.  A file
    that shrank below the offset (replaced/rotated) restarts
    at 0 with the loss counted as one damaged line."""
    p = Path(path)
    if not p.exists():
        return [], int(offset), 0
    data = p.read_bytes()
    damaged = 0
    start = int(offset)
    if len(data) < start:
        damaged += 1
        start = 0
    chunk = data[start:]
    last_nl = chunk.rfind(b"\n")
    if last_nl < 0:
        return [], start, damaged
    complete = chunk[:last_nl + 1]
    new_offset = start + len(complete)
    records: list[dict] = []
    for raw in complete.split(b"\n"):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            damaged += 1
            continue
        if isinstance(row, dict):
            records.append(row)
        else:
            damaged += 1
    return records, new_offset, damaged


# ------------------------------------------------------------ helpers
def _ts(value) -> Any:
    """Tolerant timestamp parse (journal stamps are ISO with
    offset; funnel stamps are pandas strings).  None when the
    value cannot be read — an unreadable clock is a missing
    fact, never a guessed one."""
    if value is None:
        return None
    try:
        import pandas as pd

        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else ts
    except Exception:
        return None


def _f(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _category_of_decision(decision: dict) -> int | None:
    cat = decision.get("reason_category")
    if isinstance(cat, int):
        return cat
    return category_for_code(decision.get("reason_code"),
                             decision.get("result"))


# ------------------------------------------------------------ categories
def _missed_opportunities(records: list[dict]) -> dict:
    """(a) Setups detected/armed that never became attempts.

    Three journal shapes: funnel evaluation records whose
    candidate died before the order path (terminal bucket
    refused / ineligible / no_order / unresolved) — the
    reason it died is the record's own reason code; trigger
    lifecycle records for armed setups that expired or were
    invalidated; and the S3 bar-level missed-opportunity
    events (a material move nobody evaluated at all)."""
    found = 0
    unmeasurable = 0
    by_reason: dict[tuple[str, str], int] = {}
    bar_misses: list[dict] = []

    def _note(playbook: str, code: str) -> None:
        key = (playbook, code)
        by_reason[key] = by_reason.get(key, 0) + 1

    for row in records:
        rtype = row.get("type")
        if rtype == "judgment_audit_event" \
                and row.get("kind") == "missed_opportunity":
            found += 1
            bar_misses.append({
                "bar": row.get("bar"),
                "direction": row.get("direction"),
                "move_atr": row.get("move_atr"),
                "reason_code": row.get("reason_code")})
            continue
        if rtype != "evaluation":
            continue
        kind = row.get("kind")
        if kind == "trigger_event":
            if row.get("trigger_stage") in ("expired",
                                            "invalidated"):
                found += 1
                _note("__trigger__",
                      str(row.get("reason_code")
                          or "trigger_" + str(
                              row.get("trigger_stage"))))
            continue
        if kind == "trigger_binding":
            continue
        if row.get("outcome") != "candidate_emitted":
            continue
        decision = row.get("decision")
        if not isinstance(decision, dict):
            # A candidate with no decision block at all: it
            # died somewhere the funnel did not record — the
            # death is a fact, its reason is not measurable.
            unmeasurable += 1
            continue
        bucket = decision_bucket(decision)
        if bucket == "ordered":
            continue
        found += 1
        _note(str(row.get("playbook") or "?"),
              str(decision.get("reason_code")
                  or row.get("reason_code")
                  or "no_code_recorded"))
    items = [{"playbook": pb, "reason_code": code, "n": n}
             for (pb, code), n in sorted(
                 by_reason.items(), key=lambda kv: -kv[1])]
    bar_misses.sort(key=lambda m: -(m.get("move_atr") or 0))
    return {"state": "measured", "found": found,
            "unmeasurable": unmeasurable,
            "items": items[:ITEM_LIMIT],
            "bar_misses": bar_misses[:ITEM_LIMIT]}


def _wrong_judgments(state: dict, records: list[dict],
                     now) -> dict:
    """(b) Takes that lost + abstains that would have won,
    graded by the S4 pairing machinery (build_pairs joins
    judgment_shadow records to their S3 reconciliations on
    the clean set — degenerate-R pairs arrive excluded and
    are counted, never graded).  Shadows whose outcome has
    not landed yet are carried in the standing state across
    windows; a shadow older than SHADOW_MAX_AGE_H with no
    reconciliation is unmeasurable, not pending forever."""
    pending = list(state.get("pending_shadows") or [])
    pending_keys = {(str(s.get("ts") or ""),
                     str(s.get("playbook") or ""))
                    for s in pending}
    fresh_shadows = []
    for row in records:
        if row.get("type") != "judgment_shadow":
            continue
        key = (str(row.get("ts") or ""),
               str(row.get("playbook") or ""))
        if key not in pending_keys:
            fresh_shadows.append(row)
            pending_keys.add(key)
    pairs = build_pairs(pending + fresh_shadows + list(records),
                        horizon_h=4)
    resolved: dict[tuple[str, str], dict] = {}
    for pair in pairs:
        key = (pair["ts"], pair["playbook"])
        if pair.get("r_excluded") or pair["realized_r"] is not None:
            resolved[key] = pair
    found_items: list[dict] = []
    found = 0
    unmeasurable = 0
    excluded = 0
    still_pending: list[dict] = []
    for shadow in pending + fresh_shadows:
        key = (str(shadow.get("ts") or ""),
               str(shadow.get("playbook") or ""))
        pair = resolved.get(key)
        if pair is None:
            judged = _ts(shadow.get("ts"))
            if judged is not None and now is not None and (
                    now - judged).total_seconds() \
                    > SHADOW_MAX_AGE_H * 3600:
                unmeasurable += 1
            else:
                still_pending.append(shadow)
            continue
        if pair.get("r_excluded"):
            excluded += 1
            continue
        realized = float(pair["realized_r"])
        playbook = pair["playbook"] or "?"
        slot = _route_slot(state, playbook)
        decision = pair.get("judgment_decision")
        wrong = False
        if decision == "take":
            slot["takes"] += 1
            if realized < 0:
                slot["take_losses"] += 1
                wrong = True
        elif decision == "abstain":
            slot["abstains"] += 1
            if realized > 0:
                slot["abstain_wins"] += 1
                wrong = True
        if wrong:
            found += 1
            found_items.append({
                "ts": pair["ts"], "playbook": playbook,
                "decision": decision,
                "realized_r": realized,
                "expected_r": pair.get("expected_r")})
    # Shadows resolved this cycle leave the pending set;
    # anything in `resolved` that was never pending (both
    # halves landed in this same window) was graded above via
    # fresh_shadows — nothing double-counts because a resolved
    # shadow is never carried forward.
    overflow = len(still_pending) - PENDING_SHADOW_CAP
    if overflow > 0:
        del still_pending[:overflow]
        state["dropped"]["shadows"] += overflow
    state["pending_shadows"] = still_pending
    state["wrong_judgments_excluded_degenerate_r"] = int(
        state.get("wrong_judgments_excluded_degenerate_r") or 0) \
        + excluded
    found_items.sort(key=lambda i: -abs(i["realized_r"]))
    return {"state": "measured", "found": found,
            "unmeasurable": unmeasurable,
            "excluded_degenerate_r": excluded,
            "pending": len(still_pending),
            "items": found_items[:ITEM_LIMIT]}


def _weak_setups(state: dict, records: list[dict]) -> dict:
    """(c) Attempts executed on a thin evidence vector.

    An attempt is a judgment_shadow whose gate class is
    EXECUTED.  The evidence vector is the journaled judgment
    block itself — never reconstructed: the correlation
    groups that voted (evidence_balance groups_for/against)
    plus the score channel when a score was present.  Thin =
    at most WEAK_MAX_GROUPS of the known groups engaged, or
    confidence under the record's own stated bar.  The
    finding names the known groups that never voted."""
    known = set(GROUP_WEIGHTS)
    found = 0
    unmeasurable = 0
    items: list[dict] = []
    for row in records:
        if row.get("type") != "judgment_shadow":
            continue
        if gate_class(row) != GATE_EXECUTED:
            continue
        playbook = str(row.get("playbook") or "?")
        slot = _route_slot(state, playbook)
        slot["attempts"] += 1
        judgment = row.get("judgment") or {}
        balance = judgment.get("evidence_balance")
        if not isinstance(balance, dict):
            unmeasurable += 1
            continue
        present = set(balance.get("groups_for") or []) \
            | set(balance.get("groups_against") or [])
        score = judgment.get("score") or {}
        if isinstance(score, dict) \
                and score.get("total") is not None:
            present.add("asymmetry_score")
        engaged = present & known
        absent = sorted(known - present)
        thresholds = judgment.get("thresholds") or {}
        min_conf = _f(thresholds.get("min_confidence"))
        confidence = _f(judgment.get("confidence"))
        thin = len(engaged) <= WEAK_MAX_GROUPS
        below_bar = (min_conf is not None
                     and confidence is not None
                     and confidence < min_conf)
        if thin or below_bar:
            found += 1
            slot["weak"] += 1
            items.append({
                "ts": row.get("ts"), "playbook": playbook,
                "engaged_groups": sorted(engaged),
                "absent_groups": absent,
                "confidence": confidence,
                "expected_r": judgment.get("expected_r")})
    items.sort(key=lambda i: (len(i["engaged_groups"]),
                              i["confidence"] or 0.0))
    return {"state": "measured", "found": found,
            "unmeasurable": unmeasurable,
            "items": items[:ITEM_LIMIT]}


def _carry_opens(state: dict, records: list[dict],
                 now) -> None:
    """Join material for premature exits: open records and
    the signal records that carry their targets, keyed by
    alert_id, carried across windows until the close lands
    (or the entry ages out, counted as dropped)."""
    opens = state["pending_opens"]
    for row in records:
        rtype = row.get("type")
        if rtype == "signal":
            targets = row.get("targets") or []
            if targets:
                slot = opens.setdefault(
                    str(row.get("signal_id") or ""), {})
                slot["targets"] = [_f(t) for t in targets]
        elif rtype == "open":
            slot = opens.setdefault(
                str(row.get("alert_id") or ""), {})
            slot.update({
                "ts": row.get("ts"),
                "ticket": row.get("ticket"),
                "playbook": row.get("playbook"),
                "direction": row.get("direction"),
                "entry": row.get("entry")})
    if now is not None:
        stale = []
        for key, slot in opens.items():
            ts = _ts(slot.get("ts"))
            if ts is not None and (
                    now - ts).total_seconds() \
                    > OPEN_MAX_AGE_H * 3600:
                stale.append(key)
        for key in stale:
            del opens[key]
            state["dropped"]["opens"] += 1
    overflow = len(opens) - PENDING_OPEN_CAP
    if overflow > 0:
        for key in list(opens)[:overflow]:
            del opens[key]
        state["dropped"]["opens"] += overflow


def _register_closes(state: dict, records: list[dict]) -> int:
    """Closes become pending exits (due one window later);
    a close whose target/open cannot be joined is
    unmeasurable on the spot, with the reason named."""
    unmeasurable = 0
    opens = state["pending_opens"]
    for row in records:
        if row.get("type") != "close":
            continue
        alert_id = str(row.get("alert_id") or "")
        opened = opens.pop(alert_id, None)
        exit_ts = _ts(row.get("ts"))
        if opened is None or opened.get("entry") is None \
                or opened.get("direction") not in (1, -1):
            unmeasurable += 1
            continue
        targets = [t for t in (opened.get("targets") or [])
                   if t is not None]
        if not targets:
            unmeasurable += 1
            continue
        if exit_ts is None:
            unmeasurable += 1
            continue
        state["pending_exits"].append({
            "alert_id": alert_id,
            "ticket": row.get("ticket"),
            "playbook": opened.get("playbook"),
            "direction": int(opened["direction"]),
            "entry": opened.get("entry"),
            "open_ts": opened.get("ts"),
            "exit_price": row.get("exit"),
            "exit_ts": row.get("ts"),
            "r_net": row.get("r_net"),
            "partial_taken": bool(row.get("partial_taken")),
            "target": targets[0],
            "due_ts": str(exit_ts + _hours(PREMATURE_WINDOW_H))})
    overflow = len(state["pending_exits"]) - PENDING_EXIT_CAP
    if overflow > 0:
        del state["pending_exits"][:overflow]
        state["dropped"]["exits"] += overflow
    return unmeasurable


def _hours(n: int):
    import pandas as pd

    return pd.Timedelta(hours=n)


def _measure_exits(state: dict, bars: list[dict],
                   now) -> dict:
    """(d) Premature exits, measured from stored bars.

    An exit is due one PREMATURE_WINDOW_H after it closed.
    Due + bars covering [open, due]: the path is a fact —
    target touched while held means the exit was not
    premature (management's ground); first touched only
    after the exit means it was, and the item carries how
    long after.  Due + bars NOT covering the window:
    unmeasurable, named — a path is never estimated.  A
    partial-taken close is management banking by design and
    is never flagged."""
    found = 0
    unmeasurable = 0
    items: list[dict] = []
    still: list[dict] = []
    bar_rows = []
    for b in bars or []:
        ts = _ts(b.get("ts"))
        if ts is None:
            continue
        bar_rows.append((ts, _f(b.get("high")), _f(b.get("low"))))
    bar_rows.sort(key=lambda r: r[0])
    one_h = _hours(1)
    for item in state["pending_exits"]:
        due = _ts(item.get("due_ts"))
        if due is None or now is None or now < due:
            still.append(item)
            continue
        open_ts = _ts(item.get("open_ts"))
        exit_ts = _ts(item.get("exit_ts"))
        covered = bool(bar_rows) and open_ts is not None \
            and exit_ts is not None \
            and bar_rows[0][0] <= open_ts \
            and bar_rows[-1][0] + one_h >= due
        if not covered:
            unmeasurable += 1
            continue
        direction = int(item["direction"])
        target = float(item["target"])

        def _touched(ts_lo, ts_hi) -> Any:
            for b_ts, high, low in bar_rows:
                if b_ts + one_h <= ts_lo or b_ts >= ts_hi:
                    continue
                if direction == 1 and high is not None \
                        and high >= target:
                    return b_ts
                if direction == -1 and low is not None \
                        and low <= target:
                    return b_ts
            return None

        if item.get("partial_taken"):
            continue  # measured, not premature — banked by design
        if _touched(open_ts, exit_ts) is not None:
            continue  # target reached while held — not premature
        hit = _touched(exit_ts, due)
        if hit is not None:
            found += 1
            items.append({
                "ticket": item.get("ticket"),
                "alert_id": item.get("alert_id"),
                "playbook": item.get("playbook"),
                "exit_ts": item.get("exit_ts"),
                "exit_price": item.get("exit_price"),
                "target": target,
                "r_net": item.get("r_net"),
                "minutes_to_target_after_exit": int(max(
                    0.0, (hit - exit_ts).total_seconds()
                    // 60))})
    state["pending_exits"] = still
    items.sort(key=lambda i: i["minutes_to_target_after_exit"])
    return {"state": "measured", "found": found,
            "unmeasurable": unmeasurable,
            "pending": len(still),
            "items": items[:ITEM_LIMIT]}


def _execution_failures(records: list[dict]) -> dict:
    """(e) Execution failures, three journal shapes:
    attempts refused at the execution boundary (terminal
    order-path refusal or an execution-category decision —
    the friction gate's refusal at this boundary is the
    canonical member); fills differing materially from
    intent (the open record's own slippage vs its stop
    distance); and candidate decisions that never reached a
    terminal bucket (the funnel's own named defect class —
    an unfilled, unaccounted attempt)."""
    found = 0
    unmeasurable = 0
    items: list[dict] = []
    for row in records:
        rtype = row.get("type")
        if rtype == "open":
            slippage = _f(row.get("slippage"))
            stop_distance = _f(row.get("stop_distance"))
            intended = _f(row.get("intended_entry"))
            if slippage is None or stop_distance is None \
                    or intended is None:
                continue
            if stop_distance > 0 and slippage \
                    > SLIPPAGE_STOP_FRAC * stop_distance:
                found += 1
                items.append({
                    "kind": "fill_differs_from_intent",
                    "ticket": row.get("ticket"),
                    "alert_id": row.get("alert_id"),
                    "playbook": row.get("playbook"),
                    "intended_entry": intended,
                    "fill_entry": row.get("entry"),
                    "slippage": round(slippage, 4),
                    "slippage_frac_of_stop": round(
                        slippage / stop_distance, 4)})
            continue
        if rtype != "evaluation" \
                or row.get("kind") in ("trigger_event",
                                       "trigger_binding"):
            continue
        if row.get("outcome") != "candidate_emitted":
            continue
        decision = row.get("decision")
        if not isinstance(decision, dict):
            continue
        bucket = decision_bucket(decision)
        playbook = str(row.get("playbook") or "?")
        code = str(decision.get("reason_code")
                   or row.get("reason_code")
                   or "no_code_recorded")
        if bucket == "unresolved":
            found += 1
            items.append({
                "kind": "unfilled_unaccounted",
                "ts": row.get("ts"), "playbook": playbook,
                "reason_code": code})
        elif bucket == "refused":
            category = _category_of_decision(decision)
            order_result = decision.get("order_result")
            if category == CATEGORY_EXECUTION \
                    or order_result in TERMINAL_ORDER_REFUSALS:
                found += 1
                items.append({
                    "kind": "refused_at_execution_boundary",
                    "ts": row.get("ts"), "playbook": playbook,
                    "reason_code": code,
                    "reason_category": category,
                    "order_result": order_result})
    return {"state": "measured", "found": found,
            "unmeasurable": unmeasurable,
            "items": items[:ITEM_LIMIT]}


# ------------------------------------------------------------ the cycle
#: Record types the audit consumes — a window containing
#: ONLY other types (notes, forecasts, the audit's own
#: records) produces no audit record: the cycle journals on
#: evidence, not on the clock alone.
_CONSUMED_TYPES = ("evaluation", "judgment_shadow",
                   "judgment_reconciliation",
                   "judgment_audit_event", "signal", "open",
                   "close")


def audit_cycle(*, state: dict, records: list[dict],
                bars: list[dict], now,
                damaged_lines: int = 0,
                window: dict | None = None,
                run_id: str | None = None,
                bar: str | None = None) -> dict | None:
    """One unified self-audit cycle over one journal window.

    Mutates ``state`` (the standing block) and returns the
    ``self_audit`` journal record to append, or None when the
    window carried nothing the audit consumes and nothing
    came due — re-running over the same window therefore
    adds nothing (idempotent by watermark + state)."""
    for key, value in new_state().items():
        state.setdefault(key, value)
    now_ts = _ts(now) if not hasattr(now, "total_seconds") \
        else now
    if damaged_lines:
        state["damaged_cycles"] = int(
            state.get("damaged_cycles") or 0) + 1
        state["cycles"] = int(state.get("cycles") or 0) + 1
        record = {
            "type": "self_audit", "bar": bar, "run_id": run_id,
            "window": window,
            "journal_damage": {"damaged_lines": int(
                damaged_lines)},
            "categories": {
                cat: {"state": "unmeasurable", "found": None,
                      "unmeasurable": None,
                      "reason": "journal_damage_in_window"}
                for cat in CATEGORIES},
            "route_feedback": route_feedback(state),
            "totals": {c: dict(t) for c, t in
                       state["totals"].items()},
        }
        state["last_cycle"] = record
        return record
    consumed = [r for r in records
                if r.get("type") in _CONSUMED_TYPES]
    # Carry joins first: opens/signals feed closes, closes
    # register exits, exits measure when due.
    _carry_opens(state, records, now_ts)
    close_unmeasurable = _register_closes(state, records)
    exits = _measure_exits(state, bars, now_ts)
    exits["unmeasurable"] += close_unmeasurable
    missed = _missed_opportunities(records)
    wrong = _wrong_judgments(state, records, now_ts)
    weak = _weak_setups(state, records)
    execution = _execution_failures(records)
    categories = {
        "missed_opportunities": missed,
        "wrong_judgments": wrong,
        "weak_setups": weak,
        "premature_exits": exits,
        "execution_failures": execution,
    }
    due_activity = exits["found"] or exits["unmeasurable"] \
        or wrong["found"] or wrong["unmeasurable"]
    if not consumed and not due_activity:
        return None
    state["cycles"] = int(state.get("cycles") or 0) + 1
    for cat, body in categories.items():
        slot = state["totals"][cat]
        slot["found"] = int(slot.get("found") or 0) \
            + int(body.get("found") or 0)
        slot["unmeasurable"] = int(slot.get("unmeasurable")
                                   or 0) \
            + int(body.get("unmeasurable") or 0)
    record = {
        "type": "self_audit", "bar": bar, "run_id": run_id,
        "window": window,
        "journal_damage": None,
        "categories": categories,
        "route_feedback": route_feedback(state),
        "totals": {c: dict(t) for c, t in
                   state["totals"].items()},
    }
    state["last_cycle"] = record
    return record


# ------------------------------------------------------------ rendering
_PLAIN = {
    "missed_opportunities": "missed opportunities",
    "wrong_judgments": "wrong judgments",
    "weak_setups": "weak setups (thin evidence)",
    "premature_exits": "premature exits",
    "execution_failures": "execution failures",
}


def render_lines(state: dict) -> list[str]:
    """The /status block, plain words first (the owner reads
    this surface): last cycle's findings, the running totals,
    and — always distinguished — what could not be measured."""
    cycles = int(state.get("cycles") or 0)
    if not cycles:
        return ["self-audit: no cycle has run yet — the first "
                "audited hour lands here (missed moves, wrong "
                "calls, weak setups, early exits, execution "
                "failures, each counted or named unmeasurable)"]
    last = state.get("last_cycle") or {}
    cats = last.get("categories") or {}
    lines = [f"self-audit: {cycles} cycle(s) run"
             + (f", last on bar {last.get('bar')}"
                if last.get("bar") else "")]
    if last.get("journal_damage"):
        lines.append(
            "  last cycle: journal damage in the window — "
            "every category read as unmeasurable rather than "
            "guessed")
    for cat in CATEGORIES:
        body = cats.get(cat) or {}
        total = (state.get("totals") or {}).get(cat) or {}
        if body.get("state") == "unmeasurable":
            lines.append(
                f"  {_PLAIN[cat]}: unmeasurable last cycle · "
                f"running {total.get('found', 0)} found / "
                f"{total.get('unmeasurable', 0)} unmeasurable")
        else:
            lines.append(
                f"  {_PLAIN[cat]}: {body.get('found', 0)} last "
                f"cycle ({body.get('unmeasurable', 0)} could "
                f"not be measured) · running "
                f"{total.get('found', 0)}")
    damaged = int(state.get("damaged_cycles") or 0)
    if damaged:
        lines.append(
            f"  journal damage degraded {damaged} cycle(s) "
            "so far — those windows were reported unmeasurable, "
            "never cleaned up on paper")
    excluded = int(state.get(
        "wrong_judgments_excluded_degenerate_r") or 0)
    if excluded:
        lines.append(
            f"  degenerate-R records excluded from judgment "
            f"grading: {excluded} (direction counted, R never)")
    for playbook, fb in route_feedback(state).items():
        bits = []
        if fb["wrong_take_rate"] is not None:
            bits.append(
                f"takes lost {round(fb['wrong_take_rate'] * fb['takes'])}"
                f" of {fb['takes']}")
        if fb["wrong_abstain_rate"] is not None:
            bits.append(
                f"abstains that would have won "
                f"{round(fb['wrong_abstain_rate'] * fb['abstains'])}"
                f" of {fb['abstains']}")
        if fb["weak_setup_rate"] is not None:
            bits.append(
                f"weak setups {round(fb['weak_setup_rate'] * fb['attempts'])}"
                f" of {fb['attempts']} attempts")
        if bits:
            lines.append(
                f"  route `{playbook}`: " + " · ".join(bits))
    return lines
