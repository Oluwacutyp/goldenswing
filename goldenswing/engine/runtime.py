"""Deterministic runtime: binds journal + snapshot + replication +
reconciliation into the REPLAY -> RECONCILING -> LIVE lifecycle.

This is the bridge the live MT5 runner uses.  The runner keeps its
existing TradeJournal (human-readable JSONL audit) AND writes every
decision-critical frame to the deterministic journal (binary chained,
fenced, idempotent).  The deterministic journal is the authority for
seq / chain / fencing / idempotency; the legacy journal is the audit
view.  Dual-write failures never break the live loop: the legacy
path continues and the failure is noted.

Fencing secret: loaded from GS_ENGINE_SECRET, else from
<data_dir>/.engine_secret (0600, created once).  Changing the secret
after frames exist makes old frames fail validation on a fresh
authority, so the file is created once and never rotated by code.
Single-node default: cluster_size=1, leader is this node.  Multi-node
quorum uses the same Leader class with real follower acks.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from .journal import (DeterministicJournal, FencedAuthority,
                      IdempotencyConflict, payload_hash)
from .reconciliation import Reconciler
from .replication import Leader
from .snapshot import SnapshotStore


def load_or_create_secret(path: str | Path) -> bytes:
    path = Path(path)
    env = os.environ.get("GS_ENGINE_SECRET", "")
    if env:
        return env.encode()
    if path.exists():
        return path.read_bytes()
    secret = secrets.token_hex(32).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(secret)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return secret


class DeterministicRuntime:
    """REPLAY -> RECONCILING -> LIVE over the deterministic journal."""

    def __init__(self, data_dir: str | Path, *, node_id: str = "goldenswing-1",
                 epoch: int = 1, term: int = 1, cluster_size: int = 1,
                 secret: bytes | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.node_id = node_id
        self.secret = secret if secret is not None else \
            load_or_create_secret(self.data_dir / ".engine_secret")
        self.authority = FencedAuthority(epoch=epoch, term=term,
                                         leader_id=node_id, token_version=1,
                                         secret=self.secret)
        self.journal = DeterministicJournal(
            str(self.data_dir / "deterministic.gsj"), self.authority)
        self.snapshots = SnapshotStore(self.data_dir / "deterministic_snapshots")
        self.reconciler = Reconciler()
        self.leader = Leader(node_id=node_id, cluster_size=cluster_size,
                             term=term, epoch=epoch, head=self.journal.head)
        for frame in self.journal.frames():
            self.leader.record_chain(frame["seq"], frame["chain"].hex())
            self.leader.record_term(frame["seq"], frame["term"])
        self.phase = "REPLAY"
        self._replayed = 0

    # ---------------------------------------------------------- helpers
    def _token(self) -> str:
        return self.authority.token()

    def _append(self, data: dict, *, idempotency_key: str | None = None) -> dict:
        meta = self.journal.append(
            epoch=self.authority.epoch, term=self.authority.term,
            leader_id=self.node_id, token_version=self.authority.token_version,
            token=self._token(), data=data, idempotency_key=idempotency_key)
        self.leader.head = meta["seq"]
        self.leader.record_chain(meta["seq"], meta["chain"].hex())
        self.leader.record_term(meta["seq"], meta["term"])
        # Single-node: commit advances to head once the frame is fsynced.
        if self.leader.cluster_size <= 1:
            self.leader.commit_index = meta["seq"]
        return meta

    # ---------------------------------------------------------- lifecycle
    def replay(self) -> dict:
        """REPLAY: re-read committed frames with effects suppressed.

        The journal already verified the chain on load.  Replay here
        counts frames and restores leader state; no broker calls.
        """
        self._replayed = len(self.journal.frames())
        self.phase = "RECONCILING"
        self.reconciler.phase = "RECONCILING"
        return {"phase": self.phase, "replayed_frames": self._replayed,
                "head": self.journal.head,
                "head_chain": self.journal.head_chain_hex}

    def reconcile(self, broker_query=None) -> dict:
        """RECONCILING: reconcile unresolved outbox via broker truth."""
        if broker_query is not None:
            outcomes = self.reconciler.reconcile_all(broker_query)
        else:
            outcomes = []
        completed = self.reconciler.try_complete()
        if completed:
            self._append({"type": "reconciliation_complete",
                          "head": self.journal.head})
            self.phase = "LIVE"
        else:
            self.phase = "RECONCILING"
        return {"phase": self.phase, "outcomes": outcomes,
                "reconciliation_complete": completed,
                "unresolved": len(self.reconciler.unresolved())}

    def begin_live(self, broker_query=None) -> dict:
        """REPLAY -> RECONCILING -> LIVE (no-op reconcile when empty)."""
        replayed = self.replay()
        result = self.reconcile(broker_query)
        return {**replayed, **result}

    # ---------------------------------------------------------- intents
    def submit_intent(self, key: str, data: dict) -> dict:
        """Journal-first order intent. Returns {action, seq?, ...}."""
        # Journal stores the wrapped frame {"type": "order_intent",
        # "data": data}, so the idempotency hash must be over the
        # same wrapped payload the journal will store.
        wrapped = {"type": "order_intent", "data": data}
        try:
            action, rec = self.journal.check_idempotency(
                key, payload_hash(wrapped))
        except IdempotencyConflict as exc:
            return {"action": "conflict", "key": key,
                    "committed_seq": exc.committed_seq}
        if action == "replay":
            return {"action": "replay", "key": key,
                    "committed_seq": rec["seq"]}
        meta = self._append({"type": "order_intent", "data": data},
                            idempotency_key=key)
        self.reconciler.submit(key, data)
        return {"action": "new", "key": key, "seq": meta["seq"]}

    def journal_event(self, kind: str, data: dict,
                      *, idempotency_key: str | None = None) -> dict | None:
        """Journal a decision-critical event. Returns meta or None on
        fencing/idempotency rejection (caller falls back to legacy)."""
        try:
            return self._append({"type": kind, "data": data},
                                idempotency_key=idempotency_key)
        except Exception:
            return None

    def mark_unconfirmed(self, key: str) -> None:
        self.reconciler.mark_unconfirmed(key)

    def snapshot(self) -> dict:
        """Write a snapshot of head state (pages = one JSON page)."""
        import json as _json
        payload = _json.dumps(
            {"head": self.journal.head,
             "head_chain": self.journal.head_chain_hex,
             "phase": self.phase}, sort_keys=True).encode()
        page = payload + b"\x00" * (4096 - len(payload) % 4096) \
            if len(payload) % 4096 else payload
        pages = [page[i:i + 4096] for i in range(0, len(page), 4096)]
        manifest = self.snapshots.write_snapshot(
            self.journal.head,
            bytes.fromhex(self.journal.head_chain_hex)
            if self.journal.head else b"\x00" * 32, pages)
        return {"watermark": self.journal.head, "manifest": manifest}

    def status(self) -> dict:
        return {"phase": self.phase, "head": self.journal.head,
                "head_chain": self.journal.head_chain_hex[:16],
                "commit_index": self.leader.commit_index,
                "unresolved": len(self.reconciler.unresolved()),
                "holding": len(self.reconciler.holding)}
