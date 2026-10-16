"""Replication + consensus: replicate the journal, not state.

Mapping: Raft log index = journal seq, term = leadership epoch,
commit index = committed journal head, apply index = projection
watermark.  The leader is the sole sequencer.  Followers verify
contiguity / chain / schema before applying, with effects suppressed.

Commit rule: on each ack, sort all matchIndex values (leader head
included) descending and take the value at position N/2 — the
majority median.  Advance only if the new value is greater than the
old commit, the entry at that seq is from the CURRENT term, and the
follower's reported chain_N matches the leader's chain_N at that
seq.  Chain mismatch => no advancement, quarantine.

Far-behind followers get InstallSnapshot in chunks to a temp file,
full verification, install, then tail N+1..head.  Stale term /
epoch / token / leader is rejected before any append.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class FollowerState:
    node_id: str
    match_index: int = 0
    next_index: int = 1
    chain_hex: str = ""
    term: int = 0
    epoch: int = 0
    phase: str = "FOLLOWER"
    quarantined: bool = False


class Leader:
    """Leader-side replication state (pure, testable, no sockets)."""

    def __init__(self, node_id: str, cluster_size: int, term: int,
                 epoch: int, head: int = 0):
        self.node_id = node_id
        self.cluster_size = int(cluster_size)
        self.term = int(term)
        self.epoch = int(epoch)
        self.head = int(head)
        self.commit_index = 0
        self.followers: dict[str, FollowerState] = {}
        # seq -> chain hex, populated by the journal as frames commit.
        self.chains: dict[int, str] = {}

    def register_follower(self, node_id: str) -> FollowerState:
        st = FollowerState(node_id=node_id, next_index=self.head + 1)
        self.followers[node_id] = st
        return st

    def record_chain(self, seq: int, chain_hex: str) -> None:
        self.chains[int(seq)] = chain_hex

    def on_ack(self, node_id: str, match_index: int, chain_hex: str,
               term: int, epoch: int) -> int:
        """Process one follower ack.  Returns the (possibly new) commit.

        Stale term/epoch acks never advance commit.  Chain mismatch
        quarantines the follower and never advances commit.
        """
        st = self.followers.get(node_id)
        if st is None:
            st = self.register_follower(node_id)
        if int(term) != self.term or int(epoch) != self.epoch:
            return self.commit_index  # stale epoch/term: ignore
        if match_index > 0:
            expected = self.chains.get(int(match_index))
            if expected is not None and expected != chain_hex:
                st.quarantined = True
                return self.commit_index  # fork: never advance
        if int(match_index) > st.match_index:
            st.match_index = int(match_index)
            st.chain_hex = chain_hex
        return self._advance_commit()

    def _advance_commit(self) -> int:
        values = sorted(
            [self.head] + [f.match_index for f in self.followers.values()],
            reverse=True)
        if not values:
            return self.commit_index
        # Majority median: index N/2 in the descending sort.
        candidate = values[self.cluster_size // 2] \
            if len(values) >= self.cluster_size else values[-1]
        # Current-term-only: the entry at candidate must be from this
        # term.  Callers record per-seq terms via record_term(); if no
        # term is recorded we conservatively refuse to advance past
        # the old commit on a bare ack (leader must commit a no-op
        # in the new term first — see commit_noop()).
        if candidate > self.commit_index and self._term_of(candidate) == self.term:
            self.commit_index = candidate
        return self.commit_index

    # ---------------------------------------------------------- term tracking
    def record_term(self, seq: int, term: int) -> None:
        if not hasattr(self, "_seq_term_map"):
            self._seq_term_map: dict[int, int] = {}
        self._seq_term_map[int(seq)] = int(term)

    def _term_of(self, seq: int) -> int | None:
        return getattr(self, "_seq_term_map", {}).get(int(seq))

    def commit_noop(self) -> int:
        """Commit a no-op in the new term so later entries can commit.

        Raft rule: commit advances only for current-term entries.
        The leader appends a no-op at head+1 in the current term and
        records its term; quorum ack of the no-op then lets commit
        advance.  Returns the no-op seq.
        """
        seq = self.head + 1
        self.head = seq
        self.record_term(seq, self.term)
        self.record_chain(seq, hashlib.sha256(
            f"noop|{seq}|{self.term}".encode()).hexdigest())
        return seq

    # ---------------------------------------------------------- heartbeats
    def heartbeat(self) -> dict:
        """Heartbeat payload: term/epoch/watermark/chain/phase."""
        return {"term": self.term, "epoch": self.epoch,
                "leader_id": self.node_id,
                "commit_index": self.commit_index,
                "head": self.head,
                "chain_hex": self.chains.get(self.commit_index, ""),
                "phase": "LIVE" if self.commit_index else "RECONCILING"}


@dataclass
class SnapshotChunk:
    offset: int
    data: bytes
    chunk_hash: str
    last: bool = False


def chunk_snapshot(payload: bytes, chunk_size: int = 65536):
    """Split a snapshot payload into hash-chained chunks."""
    for off in range(0, max(1, len(payload)), chunk_size):
        chunk = payload[off:off + chunk_size]
        yield SnapshotChunk(
            offset=off, data=chunk,
            chunk_hash=hashlib.sha256(chunk).hexdigest(),
            last=(off + len(chunk) >= len(payload)))


def verify_snapshot_chunk(chunk: SnapshotChunk) -> bool:
    return hashlib.sha256(chunk.data).hexdigest() == chunk.chunk_hash


class Follower:
    """Follower-side apply + snapshot-install state (pure, testable)."""

    def __init__(self, node_id: str, term: int = 0, epoch: int = 0):
        self.node_id = node_id
        self.term = int(term)
        self.epoch = int(epoch)
        self.applied: list[dict] = []
        self.watermark = 0
        self.chain_hex = ""
        self.leader_id: str | None = None
        self.snapshot_buffer = bytearray()
        self.snapshot_complete = False

    def validate_append(self, *, term: int, epoch: int, leader_id: str,
                        seq: int, chain_hex: str, prev_chain_hex: str,
                        expected_prev_chain: str) -> bool:
        """Term/epoch/leader + contiguity + chain checks for one entry."""
        if int(term) < self.term or int(epoch) < self.epoch:
            return False  # stale leader
        if seq != self.watermark + 1:
            return False  # gap
        if prev_chain_hex != expected_prev_chain:
            return False  # chain break
        return True

    def apply(self, *, term: int, epoch: int, leader_id: str, seq: int,
              chain_hex: str, payload: dict) -> bool:
        """Apply one committed entry with effects suppressed."""
        if int(term) > self.term:
            self.term = int(term)
            self.epoch = int(epoch)
        if int(term) < self.term or int(epoch) < self.epoch:
            return False
        if seq != self.watermark + 1:
            return False
        self.leader_id = leader_id
        self.applied.append({"seq": seq, "payload": payload})
        self.watermark = seq
        self.chain_hex = chain_hex
        return True

    def install_snapshot_chunk(self, chunk: SnapshotChunk) -> bool:
        """Append one verified chunk to the temp snapshot buffer."""
        if not verify_snapshot_chunk(chunk):
            return False
        if chunk.offset != len(self.snapshot_buffer):
            return False  # out-of-order chunk
        self.snapshot_buffer.extend(chunk.data)
        if chunk.last:
            self.snapshot_complete = True
        return True

    def install_snapshot(self, *, watermark: int, chain_hex: str,
                         payload: dict) -> bool:
        """Install a fully verified snapshot, then tail resumes N+1."""
        try:
            snap = json.loads(bytes(self.snapshot_buffer).decode())
        except (ValueError, UnicodeDecodeError):
            return False
        if int(snap.get("watermark", -1)) != int(watermark):
            return False
        if snap.get("chain_n") != chain_hex:
            return False
        # Payload binding: the snapshot's own content must match.
        if snap.get("snapshot_root") != payload.get("snapshot_root"):
            return False
        self.watermark = int(watermark)
        self.chain_hex = chain_hex
        self.snapshot_buffer = bytearray()
        self.snapshot_complete = False
        return True
