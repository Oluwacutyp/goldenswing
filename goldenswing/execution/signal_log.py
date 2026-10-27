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
never ordered — HISTORICAL: the MT5 path no longer emits it since
the A14 unshadow; the paper runner's records may still carry it),
``gated`` (confluence/hierarchy gate), ``below_floor``
(score floor), ``manipulation_blocked``, ``invalidated`` (order
selection), ``watch`` (rework §5: runaway from the zone — the
setup is parked in PullbackWatch, not reset).  Path-level blocks
(paused, market-closed) and playbook crashes produce no emission,
so they produce no signal record — there is nothing to
attribute; the text notes already cover them.

Rework records (§6) additionally carry the Asymmetry Score
breakdown (``score_breakdown``), the trigger source
(``trigger_source``: resting_limit | arrival_score | funding_layer)
and the RiskPool state (``pool``) — per-decision, per-playbook
evidence; under A14 this journal IS the per-playbook confirmation
record (fills, management and R outcomes join on alert_id).

``summarize_signals`` aggregates the records back into filter
stats: per-playbook counts by decision, and per-reason counts for
every gated/invalidated record.  It reads journal records, so the
stats survive restarts — unlike the in-memory blocked-leg counters
(which stay for /status; this is the durable record, not a
replacement for the live counters).
"""
from __future__ import annotations


#: Decisions a raw signal record can carry.  "management" (P3)
#: covers position-management events journaled against an open
#: trade (be_earned / progress / drift / tp_zone / pre_print) —
#: the reason string names the action.
DECISIONS = (
    "accepted", "shadow", "gated", "below_floor",
    "manipulation_blocked", "invalidated", "watch", "management",
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
               study: dict | None = None,
               score_breakdown: dict | None = None,
               trigger_source: str | None = None,
               pool: dict | None = None,
               context: dict | None = None) -> dict:
    """Append one structured signal record.  Never raises: a logging
    failure must not break the signal path (the caller treats the
    record as best-effort audit, like the text notes)."""
    if decision not in DECISIONS:
        decision = "gated"
        reason = f"unknown decision recorded as gated; was: {decision!r}"
    try:
        record = {
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
            # Rework §6 fields (None on legacy-path records).
            "score_breakdown": score_breakdown,
            "trigger_source": trigger_source,
            "pool": pool,
            # P3 scenario context (spec v3 §12 J): regime label +
            # posture, day type, ADR state, session posture, SMT,
            # trigger clock — the scenario the decision was made
            # IN, on the record.  None on legacy-path records.
            "context": context,
        }
        # The common evidence ledger is deliberately a shadow-only addition.
        # Existing playbook decisions and execution gates are unchanged until
        # chronological, cost-aware testing supports promotion. A provider is
        # attached by the MT5 runner; standalone/paper users remain compatible.
        provider = getattr(journal, "shared_evidence_provider", None)
        if decision != "management" and callable(provider):
            try:
                record["shared_evidence"] = provider(
                    direction=int(direction), playbook=str(playbook),
                    entry=entry, bar_ts=bar_ts, context=context or {})
            except Exception as exc:  # observational failure never blocks a signal
                record["shared_evidence"] = {
                    "status": "unavailable",
                    "reason_code": "shared_evidence.provider_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "execution_gate": False,
                }
        return journal.append(record)
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
