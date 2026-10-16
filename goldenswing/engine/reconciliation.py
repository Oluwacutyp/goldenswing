"""Broker reconciliation + LIVE fence.

States: an order that times out after send is UNCONFIRMED, never
rejected.  A broker query that times out is UNKNOWN — a third state,
neither found nor not found.  No resubmit, no cancel, no adoption on
UNKNOWN.

Reconciliation (by original idempotency key, coalesced single-flight):
- Working -> adopt as working, attach protection.
- Filled -> adopt fill + protection, dedup by broker_fill_id.
- Rejected / Cancelled -> resolve + free the slot.
- Confirmed absent -> resubmit SAME key only if still valid.
- Broker unavailable / UNKNOWN after bounded retries -> RECOVERY_HOLD.

``reconciliation_complete`` is a quorum-committed fence before LIVE.
LIVE is a replicated phase fence, not a node-local flag.  After the
fence, the holding queue drains bounded/paced, management first,
with TTL / invalidation / margin / setup / costs revalidated before
sequence assignment.

Ambiguous timeouts: bounded retries (3, backoff + jitter, ~5 min
deadline), then RECOVERY_HOLD + slow retries + an authenticated,
idempotent admin override via the admin queue (command_id, target
still-UNKNOWN check, override_resolved frame).  Mode flips to LIVE
only when ALL items are resolved — never forced by a command.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field


@dataclass
class OutboxEntry:
    idempotency_key: str
    data: dict
    status: str = "PENDING"  # PENDING | UNCONFIRMED | UNKNOWN | RESOLVED
    broker_order_id: str | None = None
    broker_fill_id: str | None = None
    attempts: int = 0
    last_error: str = ""


@dataclass
class HoldingEntry:
    idempotency_key: str
    data: dict
    priority: int = 0  # higher = management / risk-reducing first
    enqueued_at: float = field(default_factory=time.time)


class Reconciler:
    """Pure reconciliation state machine (no sockets, no broker calls).

    The caller supplies ``broker_query(key)`` returning one of:
    ``working`` | ``filled`` | ``rejected`` | ``cancelled`` |
    ``absent`` | ``unknown``.  This class owns the transitions,
    dedup, fence and holding-queue drain.
    """

    MAX_RETRIES = 3
    RECONCILE_DEADLINE_S = 300.0

    def __init__(self):
        self.outbox: dict[str, OutboxEntry] = {}
        self.seen_fill_ids: set[str] = set()
        self.holding: list[HoldingEntry] = []
        self.holding_capacity = 1024
        self.processed_override_ids: set[str] = set()
        self.phase = "RECONCILING"
        self.reconciliation_complete = False
        self.started_at = time.time()

    # ---------------------------------------------------------- outbox
    def submit(self, key: str, data: dict) -> OutboxEntry:
        entry = OutboxEntry(idempotency_key=key, data=dict(data))
        self.outbox[key] = entry
        return entry

    def mark_unconfirmed(self, key: str) -> None:
        if key in self.outbox:
            self.outbox[key].status = "UNCONFIRMED"

    def unresolved(self) -> list[OutboxEntry]:
        return [e for e in self.outbox.values()
                if e.status in ("UNCONFIRMED", "UNKNOWN", "PENDING")]

    # ---------------------------------------------------------- reconcile
    def reconcile_one(self, key: str, broker_result: str,
                      broker_detail: dict | None = None) -> dict:
        """Reconcile one entry against broker truth.  Returns an outcome
        dict: {action, key, ...}.  ``action`` is one of adopt_working,
        adopt_fill, resolve, resubmit_same_key, hold_unknown, duplicate_fill.
        """
        entry = self.outbox.get(key)
        if entry is None:
            return {"action": "not_found", "key": key}
        detail = broker_detail or {}
        if broker_result == "working":
            entry.status = "RESOLVED"
            entry.broker_order_id = detail.get("broker_order_id")
            return {"action": "adopt_working", "key": key}
        if broker_result == "filled":
            fill_id = str(detail.get("broker_fill_id") or "")
            if fill_id and fill_id in self.seen_fill_ids:
                entry.status = "RESOLVED"
                return {"action": "duplicate_fill", "key": key,
                        "broker_fill_id": fill_id}
            if fill_id:
                self.seen_fill_ids.add(fill_id)
            entry.status = "RESOLVED"
            entry.broker_fill_id = fill_id or None
            entry.broker_order_id = detail.get("broker_order_id")
            return {"action": "adopt_fill", "key": key,
                    "broker_fill_id": fill_id}
        if broker_result in ("rejected", "cancelled"):
            entry.status = "RESOLVED"
            return {"action": "resolve", "key": key,
                    "outcome": broker_result}
        if broker_result == "absent":
            entry.status = "RESOLVED"
            return {"action": "resubmit_same_key", "key": key}
        # unknown / timeout / broker unavailable
        entry.attempts += 1
        entry.last_error = str(detail.get("error", "timeout"))
        entry.status = "UNKNOWN"
        return {"action": "hold_unknown", "key": key,
                "attempts": entry.attempts}

    def reconcile_all(self, broker_query) -> list[dict]:
        """Reconcile every unresolved entry via ``broker_query(key)``.

        ``broker_query`` returns (result, detail).  Single-flight is
        the caller's job (one call per key here); this method never
        calls it twice for the same key in one pass.
        """
        outcomes: list[dict] = []
        for entry in list(self.unresolved()):
            if entry.status == "PENDING":
                continue  # never sent: nothing to reconcile
            result, detail = broker_query(entry.idempotency_key)
            outcomes.append(self.reconcile_one(
                entry.idempotency_key, result, detail))
        return outcomes

    def blocked_by_unknown(self) -> bool:
        return any(e.status == "UNKNOWN" for e in self.outbox.values())

    def retries_exhausted(self, key: str) -> bool:
        entry = self.outbox.get(key)
        return bool(entry and entry.attempts >= self.MAX_RETRIES)

    # ---------------------------------------------------------- admin override
    def admin_override(self, *, command_id: str, key: str,
                       resolution: str, detail: dict | None = None,
                       reason: str = "") -> dict:
        """Authenticated admin override for a still-UNKNOWN entry.

        ``resolution`` is "mark_absent" or "adopt".  Idempotent on
        ``command_id``; rejected unless the target is still UNKNOWN.
        Returns a frame dict the caller journals as override_resolved.
        """
        if command_id in self.processed_override_ids:
            return {"action": "duplicate_override", "command_id": command_id}
        entry = self.outbox.get(key)
        if entry is None or entry.status != "UNKNOWN":
            return {"action": "rejected",
                    "reason": "target is not currently UNKNOWN",
                    "key": key}
        if resolution == "adopt" and not (detail or {}).get("broker_fill_id"):
            return {"action": "rejected",
                    "reason": "adopt requires full broker details",
                    "key": key}
        if resolution not in ("mark_absent", "adopt"):
            return {"action": "rejected",
                    "reason": f"unknown resolution {resolution!r}", "key": key}
        self.processed_override_ids.add(command_id)
        entry.status = "RESOLVED"
        if resolution == "adopt" and detail:
            fill_id = str(detail.get("broker_fill_id") or "")
            if fill_id:
                self.seen_fill_ids.add(fill_id)
        frame = {"type": "override_resolved", "command_id": command_id,
                 "key": key, "resolution": resolution,
                 "reason": reason, "detail": detail or {}}
        frame["frame_hash"] = hashlib.sha256(
            json.dumps(frame, sort_keys=True).encode()).hexdigest()
        return {"action": "override_resolved", "frame": frame}

    # ---------------------------------------------------------- fence + drain
    def try_complete(self) -> bool:
        """Return True and flip to LIVE only when ALL items resolved."""
        if self.unresolved():
            return False
        if self.blocked_by_unknown():
            return False
        self.reconciliation_complete = True
        self.phase = "LIVE"
        return True

    def enqueue_holding(self, key: str, data: dict, priority: int = 0) -> dict:
        if len(self.holding) >= self.holding_capacity:
            return {"action": "shed", "key": key,
                    "reason": "holding_queue_full"}
        self.holding.append(HoldingEntry(
            idempotency_key=key, data=dict(data), priority=priority))
        return {"action": "held", "key": key}

    def drain_holding(self, *, batch_size: int = 16,
                      revalidate=None) -> list[dict]:
        """Drain bounded/paced, management (high priority) first.

        ``revalidate(entry)`` returns True if the entry is still
        valid (TTL / invalidation / margin / setup / costs).  Invalid
        entries are dropped with an explicit outcome, never silently.
        """
        if self.phase != "LIVE":
            return []
        ordered = sorted(self.holding, key=lambda e: (-e.priority, e.enqueued_at))
        batch, rest = ordered[:batch_size], ordered[batch_size:]
        self.holding = rest
        out: list[dict] = []
        for entry in batch:
            if revalidate is not None and not revalidate(entry):
                out.append({"action": "dropped_stale", "key": entry.idempotency_key})
                continue
            out.append({"action": "drain", "key": entry.idempotency_key})
        return out
