"""Broker reconciliation: align the journal with reality after a blind window.

Core principle: the BROKER is the source of truth for what HAPPENED to
positions; the JOURNAL is the source of truth for what WE DECIDED. When the
two diverge (e.g., a stop executed during a feed gap we never observed),
reconciliation writes the broker's facts as explicitly-labeled events.

What reconciliation NEVER does:
  - Invent bars, ticks, or price paths for the blind window. The FVGTracker
    never sees fabricated data; unobserved history stays unknown.
  - Rewrite or delete journal lines. Append-only, always.
  - Confuse fills with reconciliations. A fill is a broker-confirmed execution
    we ordered AND observed. A reconciliation is an execution we did NOT
    observe. They are different record types with different semantics.

Zone handling: for a broker close at price P, price necessarily traversed the
range between P and the last observed price. Zones intersecting that range may
have been interacted with unobserved -> marked SUSPECT (excluded from signals
until a fresh observed bar re-validates them). Zones outside the range are
untouched — we have no evidence either way, and absence of evidence is not
evidence of interaction.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from goldenswing.live.monitor import (
    JOURNAL_RECORD_VERSION,
    append_trade_to_journal,
)
from goldenswing.live.journal_lock import journal_transaction, append_journal_record

# Reconciliation event types.
STOP_EXECUTED = "broker_stop_executed"
TARGET_EXECUTED = "broker_target_executed"
BROKER_AMENDED = "broker_amended"          # qty/stop changed out-of-band
BROKER_CLOSED_UNKNOWN = "broker_closed_unknown"  # gone, no close record
# Journal says closed but broker shows open (or vice versa): a genuine
# disagreement about reality. Distinct from BROKER_CLOSED_UNKNOWN because
# it blocks engine start until a human resolves it (see startup.py) —
# sizing new entries against a wrong heat model is how overleverage happens.
DIVERGENCE = "divergence_position_mismatch"
# Operator clearance for a divergence (see resolve_divergence).
DIVERGENCE_CLEARED = "divergence_cleared"


def append_reconciliation_to_journal(
    journal_path: str,
    trade_id: str,
    event: str,
    broker_ts: str,
    blind_window: tuple[str, str],
    close_price: float | None = None,
    close_qty: float | None = None,
    traversed_range: tuple[float, float] | None = None,
    needs_review: bool = False,
    note: str = "",
    now: datetime | None = None,
) -> int:
    """Journal one out-of-band broker event. Append-only; never rewrites.

    ticks_observed / bars_observed are always False here — that is the point:
    the record explicitly states what was NOT seen, so no downstream reader
    can mistake it for normally-observed data.
    """
    now = now or datetime.now(timezone.utc)
    record = {
        "v": JOURNAL_RECORD_VERSION,
        "type": "reconciliation",
        "trade_id": trade_id,
        "event": event,
        "broker_ts": broker_ts,       # when the broker says it happened
        "detected_at": now.isoformat(),  # when WE learned about it
        "blind_window": list(blind_window),
        "close_price": close_price,
        "close_qty": close_qty,
        "traversed_range": list(traversed_range) if traversed_range else None,
        "ticks_observed": False,
        "bars_observed": False,
        "needs_review": needs_review,
        "note": note,
    }
    return append_journal_record(journal_path, record)


@dataclass
class ReconciliationReport:
    events: list = None          # journal seqs written
    trades_closed: list = None   # trade_ids closed via reconciliation
    zones_suspected: int = 0
    needs_review: list = None    # trade_ids requiring human eyes
    heat_pct: float = 0.0        # portfolio heat AFTER reconciliation
    bars_backfilled: int = 0
    backfill_complete: bool = True

    def __post_init__(self):
        self.events = self.events or []
        self.trades_closed = self.trades_closed or []
        self.needs_review = self.needs_review or []


def _open_trade_ids(journal_path: str) -> dict:
    """Derive open positions: trade_ids with fills but no trade summary.

    Returns {trade_id: {"fill_seqs": [...], "qty": total, "direction": ...}}.
    """
    from goldenswing.live.monitor import upcast_trade_record
    opens: dict = {}
    closed: set = set()
    if not os.path.exists(journal_path):
        return opens
    with open(journal_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = upcast_trade_record(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
            t = d.get("type", "trade")
            tid = d.get("trade_id")
            if not tid:
                continue
            if t == "fill":
                e = opens.setdefault(tid, {"fill_seqs": [], "qty": 0.0,
                                           "direction": 1 if d.get("side") == "buy" else -1})
                e["fill_seqs"].append(d["seq"])
                e["qty"] += float(d.get("qty", 0) or 0)
            elif t == "trade":
                closed.add(tid)
    return {tid: e for tid, e in opens.items() if tid not in closed}


def reconcile_positions(journal_path: str,
                        broker_open: list[dict],
                        broker_closed: list[dict],
                        tracker,
                        gap_start: str,
                        gap_end: str,
                        last_observed_price: float,
                        now: datetime | None = None) -> ReconciliationReport:
    """Reconcile journal vs broker after a blind window.

    broker_open:   [{trade_id, qty}] currently open at broker.
    broker_closed: [{trade_id, close_price, close_qty, close_time, reason,
                     r_multiple}] closed during the gap. reason is "stop" |
                     "target" | "unknown"; r_multiple is the caller's honest
                     computation from the position's known stop distance
                     (0.0 if genuinely unknown — never fabricated precision).

    Cases:
      journal-open + broker-closed -> reconciliation event + trade summary
        (exit_provenance="broker_reconciliation"); traversed zones -> SUSPECT.
      journal-open + broker-open, qty differs -> BROKER_AMENDED event.
      journal-closed + broker-open -> DIVERGENCE: needs_review, no auto-action.
      broker-closed + journal never knew the trade -> needs_review.
    """
    now = now or datetime.now(timezone.utc)
    report = ReconciliationReport()
    journal_open = _open_trade_ids(journal_path)
    broker_open_ids = {p["trade_id"]: p for p in broker_open}
    blind = (gap_start, gap_end)

    for bc in broker_closed:
        tid = bc["trade_id"]
        if tid not in journal_open:
            # Broker closed something we never tracked: human must resolve.
            # Idempotent: skip if a previous partial run already recorded it.
            if not _has_reconciliation_event(journal_path, tid, gap_start,
                                             BROKER_CLOSED_UNKNOWN):
                seq = append_reconciliation_to_journal(
                    journal_path, tid, BROKER_CLOSED_UNKNOWN,
                    bc.get("close_time", now.isoformat()), blind,
                    close_price=bc.get("close_price"),
                    close_qty=bc.get("close_qty"),
                    needs_review=True,
                    note="broker closed an untracked position",
                    now=now)
                report.events.append(seq)
            report.needs_review.append(tid)
            continue

        jo = journal_open[tid]
        reason = bc.get("reason", "unknown")
        event = STOP_EXECUTED if reason == "stop" else (
            TARGET_EXECUTED if reason == "target" else BROKER_CLOSED_UNKNOWN)
        cp = float(bc["close_price"])
        lo, hi = (cp, last_observed_price) if cp <= last_observed_price else (last_observed_price, cp)

        # Idempotent: a crashed run may have written the event but not the
        # trade summary. Never duplicate the event; the summary below is
        # still needed (journal_open excludes summarized trades, so reaching
        # here means it is missing).
        if not _has_reconciliation_event(journal_path, tid, gap_start, event):
            seq = append_reconciliation_to_journal(
                journal_path, tid, event, bc.get("close_time", now.isoformat()),
                blind, close_price=cp, close_qty=bc.get("close_qty"),
                traversed_range=(lo, hi),
                needs_review=(event == BROKER_CLOSED_UNKNOWN),
                note=f"broker {reason} during blind window; ticks not observed",
                now=now)
            report.events.append(seq)

        # Zone marking is UNCONDITIONAL (not tied to the event write): a
        # crashed run may have written the event without persisting the
        # tracker, losing the in-memory suspect flags. Re-marking is
        # idempotent — only False->True transitions are counted.
        for z in tracker.zones:
            if (z.status != "FILLED" and not z.suspect
                    and not (z.top < lo or z.bottom > hi)):
                z.suspect = True
                report.zones_suspected += 1

        # Trade summary: P&L on broker facts, provenance flagged.
        # exit_provenance="broker_reconciliation" marks the exit price as
        # broker fact with unobserved path — excluded from execution-cost
        # stats (unknown, not zero) and visible in audits.
        direction = jo["direction"]
        r_mult = float(bc.get("r_multiple", 0.0))
        append_trade_to_journal(
            journal_path, r_mult, direction=direction,
            entry=last_observed_price, exit_=cp, trade_id=tid,
            qty=jo["qty"], fill_seqs=jo["fill_seqs"],
            r_theoretical=r_mult,  # unmeasured exit -> no fabricated cost
            state_at_signal="unknown",
            exit_provenance="broker_reconciliation",
            ts_closed=now.isoformat())
        report.trades_closed.append(tid)
        if event == BROKER_CLOSED_UNKNOWN:
            report.needs_review.append(tid)

    # Amendments: still open but qty changed out-of-band.
    for tid, bo in broker_open_ids.items():
        if tid in journal_open:
            jq = journal_open[tid]["qty"]
            bq = float(bo.get("qty", 0) or 0)
            if abs(jq - bq) > 1e-9 and not _has_reconciliation_event(
                    journal_path, tid, gap_start, BROKER_AMENDED):
                seq = append_reconciliation_to_journal(
                    journal_path, tid, BROKER_AMENDED,
                    now.isoformat(), blind, close_qty=bq - jq,
                    needs_review=True,
                    note=f"qty drift: journal {jq} vs broker {bq}",
                    now=now)
                report.events.append(seq)
                report.needs_review.append(tid)

    # Divergence: journal thinks closed, broker shows open. This is the one
    # reconciliation case that blocks engine start until resolved — see
    # startup.py. Sizing entries against a wrong heat model is overleverage.
    journal_closed_ids = _closed_trade_ids(journal_path)
    for tid in journal_closed_ids:
        if tid in broker_open_ids and not _has_reconciliation_event(
                journal_path, tid, gap_start, DIVERGENCE):
            seq = append_reconciliation_to_journal(
                journal_path, tid, DIVERGENCE,
                now.isoformat(), blind, needs_review=True,
                note="DIVERGENCE: journal closed but broker shows open — "
                     "no auto-action, human must resolve before entries resume",
                now=now)
            report.events.append(seq)
            report.needs_review.append(tid)

    return report


def _has_reconciliation_event(journal_path: str, trade_id: str,
                              gap_start: str, event: str) -> bool:
    """Idempotency probe: was this trade already reconciled for this gap?

    Keyed on (trade_id, gap_start, event). gap_start is stable across
    restarts (persisted by GapTracker); gap_end may shift by seconds between
    a crashed run and its retry, so it is NOT part of the key.
    """
    from goldenswing.live.monitor import upcast_trade_record
    if not os.path.exists(journal_path):
        return False
    with open(journal_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = upcast_trade_record(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
            if (d.get("type") == "reconciliation"
                    and d.get("trade_id") == trade_id
                    and d.get("event") == event
                    and isinstance(d.get("blind_window"), list)
                    and d["blind_window"]
                    and d["blind_window"][0] == gap_start):
                return True
    return False


def _has_reconnection_marker(journal_path: str, gap_start: str) -> bool:
    """Has a reconnection sequence already committed for this gap?"""
    from goldenswing.live.monitor import upcast_trade_record
    if not os.path.exists(journal_path):
        return False
    with open(journal_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = upcast_trade_record(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
            if (d.get("type") == "reconnection"
                    and d.get("event") == "reconnection_complete"
                    and d.get("gap_start") == gap_start):
                return True
    return False


def append_reconnection_complete(journal_path: str, gap_start: str,
                                 gap_end: str, report: "ReconciliationReport",
                                 now: datetime | None = None) -> int:
    """Commit point of the reconnection sequence.

    Written AFTER all reconcile appends, BEFORE the monitor check. Crash
    semantics: if the marker exists, the batch committed — a restart skips
    the sequence (the check may have been missed, which is just a skipped
    cycle). If it does not exist, the batch did not commit — a restart
    safely re-runs the sequence (reconciliation is idempotent).
    """
    now = now or datetime.now(timezone.utc)
    record = {
        "v": JOURNAL_RECORD_VERSION,
        "type": "reconnection",
        "event": "reconnection_complete",
        "gap_start": gap_start,
        "gap_end": gap_end,
        "detected_at": now.isoformat(),
        "trades_closed": report.trades_closed,
        "zones_suspected": report.zones_suspected,
        "heat_pct": report.heat_pct,
        "bars_backfilled": report.bars_backfilled,
        "backfill_complete": report.backfill_complete,
        "needs_review": report.needs_review,
    }
    return append_journal_record(journal_path, record)


def resolve_divergence(journal_path: str, trade_id: str, note: str,
                       now: datetime | None = None) -> int:
    """Operator clearance for a divergence. Append-only: the original
    divergence event stays in the log; this records that a human reviewed
    and resolved it. Startup treats a divergence with a later clearance as
    resolved and will not block on it again.

    Usage (one-liner for the operator):
        resolve_divergence(jp, "t-0007", "closed stray broker position manually")
    """
    now = now or datetime.now(timezone.utc)
    record = {
        "v": JOURNAL_RECORD_VERSION,
        "type": "review",
        "event": DIVERGENCE_CLEARED,
        "trade_id": trade_id,
        "cleared_at": now.isoformat(),
        "cleared_by": "human",
        "note": note,
    }
    return append_journal_record(journal_path, record)


def _unresolved_divergences(journal_path: str) -> dict:
    """{trade_id: event} for needs_review reconciliations with no later
    divergence_cleared. The journal is append-only, so a resolved divergence
    must not block every future startup — the clearance is the release."""
    from goldenswing.live.monitor import upcast_trade_record
    pending: dict[str, dict] = {}
    if not os.path.exists(journal_path):
        return {}
    with open(journal_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = upcast_trade_record(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
            t = d.get("type")
            if t == "reconciliation" and d.get("needs_review"):
                pending[d["trade_id"]] = d
            elif (t == "review" and d.get("event") == DIVERGENCE_CLEARED
                    and d.get("trade_id") in pending):
                del pending[d["trade_id"]]
    return {tid: d.get("event") for tid, d in pending.items()}


def _closed_trade_ids(journal_path: str) -> set:
    from goldenswing.live.monitor import upcast_trade_record
    closed = set()
    if not os.path.exists(journal_path):
        return closed
    with open(journal_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = upcast_trade_record(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
            if d.get("type", "trade") == "trade" and d.get("trade_id"):
                closed.add(d["trade_id"])
    return closed


def compute_heat(broker_open: list[dict]) -> float:
    """Portfolio heat (% risk) from post-reconciliation open positions.

    Each broker position dict should carry its risk_pct (set at entry by the
    execution layer). Missing risk_pct defaults to 0.0 — we never invent a
    risk number for a position we didn't size.
    """
    return sum(float(p.get("risk_pct", 0) or 0) for p in broker_open)


def run_reconnection_sequence(journal_path: str,
                              tracker,
                              monitor,
                              gap_start: str,
                              gap_end: str,
                              last_observed_price: float,
                              fetch_broker_state,
                              fetch_missing_bars,
                              tier1_fn=None,
                              tier2_fn=None,
                              tracker_state_path: str | None = None,
                              now: datetime | None = None) -> tuple:
    """Full recovery after a feed gap, in strict order.

    WHEN STATE MAY MOVE — the core rule:
      Reconciliation updates DATA immediately (journal, heat, suspect zones).
      State TRANSITIONS happen only at evaluation boundaries. Reconnection
      completion — after broker reconciliation AND ordered bar backfill — IS
      an evaluation boundary: the monitor runs exactly ONCE here, on the
      fully-reconciled picture. Not per-trade during reconciliation, and not
      deferred to the next bar close.

    Why not per-trade: reconciliation is a batch. Evaluating after each
    position would let partially-reconciled intermediate states drive
    transitions (thrash), and would invent a finer evaluation cadence that
    corrupts sustained-window counting (HALT needs 3 sustained windows;
    probation graduation needs 2 consecutive evaluations).

    Why not wait for the next bar: after backfill + reconciliation the
    information state is exactly what the next normal cycle would see.
    Waiting a full timeframe (4h on H4) to act on known facts buys nothing.

    Heat: recomputed from post-reconciliation opens as soon as broker state
    arrives (it is a portfolio fact), but it does not TRIGGER evaluation —
    and no entries occur during this sequence anyway (the caller resumes
    entries only after this returns), so the only heat that ever gates an
    entry is the fully-reconciled one.

    Steps:
      1. Fetch broker state (facts about what happened while blind).
      2. reconcile_positions: journal events + trade summaries + suspect
         zones + heat. The batch is the atomic unit.
      3. Backfill missing bars IN ORDER through tracker.on_bar. These are
         real historical bars from the broker — never fabricated. Idx jumps
         inside the backfill get gap_before=True via existing machinery.
         Fresh bars may re-validate suspect zones (observed touch clears).
      4. Re-run T1/T2 on fresh data (optional callbacks).
      5. ONE monitor.check() on LiveStats.from_journal (includes reconciled
         trades — they are real completed trades and belong in the window;
         skipped gap cycles neither advance nor reset streaks).
      6. Return (status, checks, report). Caller resumes entries after.

    fetch_broker_state() -> (broker_open, broker_closed) — see
      reconcile_positions for dict shapes.
    fetch_missing_bars(gap_start, gap_end) -> (bars, complete) where bars
      is [(high, low, idx), ...] chronological, complete True when the
      broker had every bar in the window.
    """
    from goldenswing.live.monitor import LiveStats
    now = now or datetime.now(timezone.utc)

    with journal_transaction(journal_path):
        if _has_reconnection_marker(journal_path, gap_start):
            # Already committed (previous run finished or crashed after the
            # commit point). No-op: safe to call on every startup.
            return None, [], ReconciliationReport(), {"t1": None, "t2": None}

        # 1. Broker facts + real historical bars. Fetched before the first
        #    write: a fetch failure raises with nothing written.
        broker_open, broker_closed = fetch_broker_state()
        bars, complete = fetch_missing_bars(gap_start, gap_end)

        # 2. Reconcile as one idempotent batch.
        report = reconcile_positions(
            journal_path, broker_open, broker_closed, tracker,
            gap_start, gap_end, last_observed_price, now=now)
        report.heat_pct = compute_heat(broker_open)

        # 3. Ordered backfill of REAL historical bars. Fresh bars may
        #    re-validate suspect zones via observed touches.
        report.bars_backfilled = len(bars)
        report.backfill_complete = bool(complete)
        prev_idx = None
        for high, low, idx in bars:
            gap_before = prev_idx is not None and idx > prev_idx + 1
            tracker.on_bar(float(high), float(low), int(idx),
                           gap_before=gap_before)
            prev_idx = idx

        # 4. Persist tracker atomically INSIDE the same transaction as the
        #    journal appends — the two can never disagree about what
        #    committed.
        if tracker_state_path:
            tmp = tracker_state_path + ".tmp"
            os.makedirs(os.path.dirname(os.path.abspath(tracker_state_path)),
                        exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(tracker.to_dict(), f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, tracker_state_path)

        # 5. Commit point. After this, a restart skips the sequence.
        append_reconnection_complete(journal_path, gap_start, gap_end,
                                     report, now=now)

        # 6. Fresh T1/T2 on reconciled state.
        t1 = tier1_fn() if tier1_fn else None
        t2 = tier2_fn(t1) if tier2_fn else None

        # 7. The single evaluation boundary.
        live = LiveStats.from_journal(journal_path)
        status, checks = monitor.check(live)

    # 8. Caller resumes entries after this returns.
    return status, checks, report, {"t1": t1, "t2": t2}


def recover_if_needed(journal_path: str, tracker, monitor, gap_state_dir: str,
                      tracker_state_path: str | None,
                      fetch_broker_state, fetch_missing_bars,
                      last_observed_price_fn, tier1_fn=None, tier2_fn=None,
                      now: datetime | None = None) -> list:
    """Startup hook: finish any gap recovery a crashed run left behind.

    Scans data_gaps.jsonl for ended gaps with no reconnection_complete
    marker and runs the sequence for each, oldest first. Returns a list of
    (gap_start, status, report) per recovered gap. Pure journal derivation —
    no extra state files to go stale.
    """
    now = now or datetime.now(timezone.utc)
    gaps_path = os.path.join(gap_state_dir, "data_gaps.jsonl")
    ended = []
    if os.path.exists(gaps_path):
        with open(gaps_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("kind") == "ended" and e.get("started_at"):
                    ended.append(e)
    ended.sort(key=lambda e: e["started_at"])
    results = []
    for g in ended:
        gs = g["started_at"]
        if _has_reconnection_marker(journal_path, gs):
            continue
        status, _, report, _ = run_reconnection_sequence(
            journal_path, tracker, monitor, gs, g.get("ended_at") or gs,
            last_observed_price_fn(), fetch_broker_state, fetch_missing_bars,
            tier1_fn=tier1_fn, tier2_fn=tier2_fn,
            tracker_state_path=tracker_state_path, now=now)
        results.append((gs, status, report))
    return results
