"""Structured raw-signal persistence: every emission, every decision.

The engine already journals human-readable notes for gated/accepted
signals, and keeps blocked-leg counters in memory only (a restart
wipes them).  That answers "what happened" in prose, but it cannot
answer the audit question mechanically: *what was seen, what blocked
it, and which playbook produced it?*

This module is the structured half of that answer — the owner's
bot's ``raw_signals`` table (signal + rejected_reason + filters
passed/failed + link to the accepted signal) translated into this
engine's append-only JSONL journal.  No parallel store: records go
through :class:`TradeJournal.append` with ``type == "signal"``,
carrying the same ``alert_id`` the open record will carry, so a
signal joins to its trade exactly like opens join to closes.

Decisions recorded: ``accepted`` (ordered), ``shadow`` (evaluated,
never ordered), ``gated`` (confluence/hierarchy gate), ``below_floor``
(score floor), ``manipulation_blocked``, ``invalidated`` (order
selection).  Path-level blocks (paused, market-closed) and playbook
crashes produce no emission, so they produce no signal record —
there is nothing to attribute; the text notes already cover them.

``summarize_signals`` aggregates the records back into filter
stats: per-playbook counts by decision, and per-reason counts for
every gated/invalidated record.  It reads journal records, so the
stats survive restarts — unlike the in-memory blocked-leg counters
(which stay for /status; this is the durable record, not a
replacement for the live counters).
"""
from __future__ import annotations


#: Decisions a raw signal record can carry.
DECISIONS = (
    "accepted", "shadow", "gated", "below_floor",
    "manipulation_blocked", "invalidated",
)


def log_signal(journal, *, signal_id: str, playbook: str,
               direction: int, score: float, decision: str,
               reason: str = "", confluence: float | None = None,
               bar_ts=None, entry: float | None = None,
               stop: float | None = None,
               targets: list | tuple | None = None,
               factors: list | tuple | None = None,
               dxy_corr: float | None = None,
               urgency: float | None = None,
               hierarchy: dict | None = None,
               reasoning: str | None = None,
               study: dict | None = None) -> dict:
    """Append one structured signal record.  Never raises: a logging
    failure must not break the signal path (the caller treats the
    record as best-effort audit, like the text notes)."""
    if decision not in DECISIONS:
        decision = "gated"
        reason = f"unknown decision recorded as gated; was: {decision!r}"
    try:
        return journal.append({
            "type": "signal",
            "signal_id": signal_id,
            "playbook": playbook,
            "direction": int(direction),
            "score": float(score),
            "decision": decision,
            "reason": reason or "",
            "confluence": confluence,
            "bar_ts": str(bar_ts) if bar_ts is not None else None,
            "entry": entry, "stop": stop,
            "targets": [float(x) for x in (targets or [])],
            "factors": list(factors or []),
            "dxy_corr": dxy_corr,
            "urgency": urgency,
            "hierarchy": hierarchy,
            # Dynamic reasoning + study package (accepted signals):
            # the why + the key levels/data behind it, on the record.
            "reasoning": reasoning,
            "study": study,
        })
    except Exception:
        return {}


def signal_records(records: list[dict]) -> list[dict]:
    """All structured signal records in a journal read."""
    return [r for r in records if r.get("type") == "signal"]


def summarize_signals(records: list[dict]) -> dict:
    """Filter stats from signal records: per-playbook counts by
    decision, and per-reason counts for blocked records.

    Returns ``{"by_playbook": {name: {decision: n, ...}}, "by_reason":
    {reason: n}}``.  Empty input -> empty maps, never a crash.
    """
    by_playbook: dict[str, dict[str, int]] = {}
    by_reason: dict[str, int] = {}
    for r in signal_records(records):
        pb = r.get("playbook", "?")
        dec = r.get("decision", "?")
        slot = by_playbook.setdefault(pb, {})
        slot[dec] = slot.get(dec, 0) + 1
        slot["total"] = slot.get("total", 0) + 1
        reason = r.get("reason") or ""
        if dec in ("gated", "invalidated", "manipulation_blocked",
                   "below_floor") and reason:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    return {"by_playbook": by_playbook, "by_reason": by_reason}
