"""Pragmatic Anchor v1: per-epoch Merkle anchoring of the journal.

What this is
------------
Each epoch covers a batch of journal anchor hashes (one leaf per journal
record, or per checkpoint — the caller decides the batching). The epoch's
Merkle root is chained into the previous epoch root, so tampering with ANY
earlier epoch breaks every later link. The epoch record is threshold-signed
and appended to an append-only JSONL anchor log.

Merkle construction (documented, deterministic)
------------------------------------------------
- Leaf: ``leaf_hash(anchor_seq, chain_head_hash, timestamp)`` =
  ``sha256(f"{anchor_seq}|{chain_head_hash}|{timestamp}")``.
  The leaf binds the journal seq, the journal chain head at anchor time,
  and the anchor timestamp — a leaf cannot be transplanted to another seq.
- Tree: leaves are hashed pairwise bottom-up. Within each pair the two
  children are SORTED lexicographically before concatenation, so the tree
  is independent of left/right insertion order and a proof verifier only
  needs the sibling + a flag. An ODD node at any level is DUPLICATED
  (hashed with itself) to form a complete pair — never promoted bare, so
  every internal node is always sha256 of exactly two children.
- A single leaf is its own root. An empty batch hashes to sha256(b"").

Epoch chain
-----------
``epoch_root_i = sha256(epoch_root_{i-1} || merkle_root_i)`` (raw bytes),
with ``epoch_root_0 = sha256(b"goldenswing-genesis")``. The first epoch's
``prev_epoch_root`` is the genesis root.

Threshold signing (SIMULATED M-of-N on a single node)
------------------------------------------------------
``ThresholdSigner`` is an interface shaped like a real N-provider DKG
setup: N signer identities, M required. For the pragmatic single-node
build each signer's key is HMAC-SHA256 key material derived from a seed
via hashlib — NOT real secret storage, NOT a distributed key. The interface
(sign over a canonical message, verify requires >= M distinct valid
signers) is kept identical so a later real DKG/VSS implementation can swap
in without changing AnchorStore or the anchor log format. Simulation must
never be mistaken for distributed trust: all N keys live on this node.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone

GENESIS_SEED = b"goldenswing-genesis"
GENESIS_EPOCH_ROOT = hashlib.sha256(GENESIS_SEED).hexdigest()

DEFAULT_N = 3
DEFAULT_M = 2
SIGNER_IDS = ("signer-1", "signer-2", "signer-3")


# ---------------------------------------------------------------------------
# Leaves / Merkle tree
# ---------------------------------------------------------------------------

def leaf_hash(anchor_seq: int, chain_head_hash: str, timestamp: str) -> str:
    """Bind one journal anchor point into a leaf hash.

    ``sha256(f"{anchor_seq}|{chain_head_hash}|{timestamp}")`` (UTF-8).
    """
    payload = f"{anchor_seq}|{chain_head_hash}|{timestamp}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def leaves_from_entries(entries: list[dict]) -> list[str]:
    """Build leaf hashes from entry dicts.

    Each entry needs ``anchor_seq``, ``chain_head_hash`` and ``timestamp``.
    """
    out = []
    for e in entries:
        out.append(leaf_hash(int(e["anchor_seq"]),
                             str(e["chain_head_hash"]),
                             str(e["timestamp"])))
    return out


def _hash_pair(left_hex: str, right_hex: str) -> str:
    """sha256 of a pair, children sorted lexicographically first."""
    a, b = sorted((left_hex, right_hex))
    return hashlib.sha256(bytes.fromhex(a) + bytes.fromhex(b)).hexdigest()


def merkle_root(leaves: list[str]) -> str:
    """Merkle root over leaf hashes.

    Sorted-pair rule + duplicate-last-odd-node rule (see module docstring).
    Empty batch -> sha256(b""). Single leaf -> the leaf itself.
    """
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = [str(leaf).lower() for leaf in leaves]
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left  # duplicate odd
            nxt.append(_hash_pair(left, right))
        level = nxt
    return level[0]


def merkle_proof(leaves: list[str], idx: int) -> list[tuple[str, bool]]:
    """Inclusion proof for the leaf at ``idx``.

    Returns [(sibling_hash, is_left), ...] bottom-up, where ``is_left`` is
    True when the sibling sorts to the LEFT of the current node in the
    sorted pair (i.e. the pair hashes as sibling || current).
    """
    if not leaves:
        raise ValueError("cannot prove membership in an empty tree")
    if not 0 <= idx < len(leaves):
        raise IndexError(f"leaf index {idx} out of range for {len(leaves)} leaves")
    level = [str(leaf).lower() for leaf in leaves]
    pos = idx
    proof: list[tuple[str, bool]] = []
    while len(level) > 1:
        if pos % 2 == 0:
            sib_pos = pos + 1 if pos + 1 < len(level) else pos  # duplicated odd
        else:
            sib_pos = pos - 1
        sibling = level[sib_pos]
        cur = level[pos]
        # In the sorted pair, the smaller hash is left.
        is_left = sibling <= cur
        proof.append((sibling, is_left))
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(_hash_pair(left, right))
        level = nxt
        pos //= 2
    return proof


def verify_proof(leaf: str, proof: list[tuple[str, bool]],
                 expected_root: str) -> bool:
    """Verify an inclusion proof against ``expected_root``."""
    try:
        cur = str(leaf).lower()
        for sibling, is_left in proof:
            sib = str(sibling).lower()
            if is_left:
                cur = hashlib.sha256(bytes.fromhex(sib) + bytes.fromhex(cur)).hexdigest()
            else:
                cur = hashlib.sha256(bytes.fromhex(cur) + bytes.fromhex(sib)).hexdigest()
        return cur == str(expected_root).lower()
    except (ValueError, TypeError):
        return False


def chain_epoch_root(prev_epoch_root: str, merkle_root_hex: str) -> str:
    """``epoch_root_i = sha256(epoch_root_{i-1} || merkle_root_i)`` (bytes)."""
    return hashlib.sha256(
        bytes.fromhex(prev_epoch_root) + bytes.fromhex(merkle_root_hex)
    ).hexdigest()


# ---------------------------------------------------------------------------
# Threshold signing (SIMULATED M-of-N — see module docstring)
# ---------------------------------------------------------------------------

class ThresholdSigner:
    """Simulated M-of-N threshold signer with N-provider shape.

    Each signer id gets HMAC-SHA256 key material derived from a seed:
    ``key_i = sha256(seed || "|" || signer_id)``. This is a simulation for
    the pragmatic single-node build — all keys live on this node — but the
    sign/verify interface (M distinct signatures required) is what a real
    DKG implementation must satisfy, so callers never depend on simulation
    details.
    """

    def __init__(self, seed: bytes = b"goldenswing-anchor-sim",
                 n: int = DEFAULT_N, m: int = DEFAULT_M,
                 signer_ids: tuple[str, ...] | None = None):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not 1 <= m <= n:
            raise ValueError(f"m must satisfy 1 <= m <= n, got m={m} n={n}")
        self.seed = bytes(seed)
        self.n = n
        self.m = m
        if signer_ids is None:
            if n != DEFAULT_N:
                signer_ids = tuple(f"signer-{i + 1}" for i in range(n))
            else:
                signer_ids = SIGNER_IDS
        if len(signer_ids) != n or len(set(signer_ids)) != n:
            raise ValueError("signer_ids must be n distinct ids")
        self.signer_ids = tuple(signer_ids)
        self._keys = {
            sid: hashlib.sha256(self.seed + b"|" + sid.encode("utf-8")).digest()
            for sid in self.signer_ids
        }

    def sign(self, message: bytes, signer_ids: list[str] | None = None) -> dict:
        """Produce signatures for ``message``.

        Defaults to the first M signers. Raises ValueError if fewer than M
        distinct known signer ids are requested — a sub-threshold signature
        set must never be produced as if it were valid.
        """
        ids = list(self.signer_ids[: self.m]) if signer_ids is None else list(signer_ids)
        unique = list(dict.fromkeys(ids))
        unknown = [s for s in unique if s not in self._keys]
        if unknown:
            raise ValueError(f"unknown signer(s): {unknown}")
        if len(unique) < self.m:
            raise ValueError(
                f"threshold signing requires >= {self.m} distinct signers, "
                f"got {len(unique)}")
        return {
            sid: hmac.new(self._keys[sid], bytes(message), hashlib.sha256).hexdigest()
            for sid in unique
        }

    def verify(self, message: bytes, signatures: dict) -> bool:
        """True iff >= M DISTINCT known signers have valid signatures."""
        if not isinstance(signatures, dict):
            return False
        valid: set[str] = set()
        for sid, sig in signatures.items():
            key = self._keys.get(sid)
            if key is None:
                continue  # unknown signer never counts toward threshold
            try:
                expect = hmac.new(key, bytes(message), hashlib.sha256).hexdigest()
            except (TypeError, ValueError):
                continue
            if isinstance(sig, str) and hmac.compare_digest(expect, sig):
                valid.add(sid)
        return len(valid) >= self.m


# ---------------------------------------------------------------------------
# Anchor store: append-only JSONL anchor log
# ---------------------------------------------------------------------------

def _signing_message(record: dict) -> bytes:
    """Canonical message the threshold signatures cover.

    Covers every chain-relevant field EXCEPT the signatures themselves, so
    verify_chain() can recompute it from the stored record alone.
    """
    canonical = "|".join([
        str(record["anchor_seq"]),
        str(record["merkle_root"]),
        str(record["epoch_root"]),
        str(record["prev_epoch_root"]),
        str(record["timestamp"]),
        str(record["leaf_count"]),
        ",".join(record["leaves"]),
    ])
    return canonical.encode("utf-8")


class AnchorStore:
    """Append-only anchor log.

    Each line is one epoch record:
    ``{anchor_seq, merkle_root, epoch_root, prev_epoch_root, timestamp,
    signatures, leaf_count, leaves}``.

    Append-only discipline: new epochs are APPENDED + fsynced, never
    rewritten. There is no update/delete path — a tampered log is detected
    by verify_chain(), never silently repaired.
    """

    def __init__(self, log_path: str, signer: ThresholdSigner | None = None):
        self.log_path = log_path
        self.signer = signer or ThresholdSigner()
        parent = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(parent, exist_ok=True)

    def _read_all(self) -> list[dict]:
        if not os.path.exists(self.log_path):
            return []
        records = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def last_record(self) -> dict | None:
        records = self._read_all()
        return records[-1] if records else None

    def append_epoch(self, leaves: list[str] | list[dict],
                     timestamp: str | None = None) -> dict:
        """Anchor one epoch and append its record to the log.

        ``leaves`` may be leaf-hash hex strings, or entry dicts (converted
        via leaves_from_entries). Returns the epoch record.
        """
        if leaves and isinstance(leaves[0], dict):
            leaf_hashes = leaves_from_entries(leaves)  # type: ignore[arg-type]
        else:
            leaf_hashes = [str(leaf).lower() for leaf in leaves]  # type: ignore[union-attr]
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        prev = self.last_record()
        prev_root = prev["epoch_root"] if prev else GENESIS_EPOCH_ROOT
        seq = (prev["anchor_seq"] + 1) if prev else 1
        root = merkle_root(leaf_hashes)
        epoch_root = chain_epoch_root(prev_root, root)
        record = {
            "anchor_seq": seq,
            "merkle_root": root,
            "epoch_root": epoch_root,
            "prev_epoch_root": prev_root,
            "timestamp": ts,
            "leaf_count": len(leaf_hashes),
            "leaves": leaf_hashes,
        }
        record["signatures"] = self.signer.sign(_signing_message(record))
        # Append-only: open in append mode, write one line, fsync. Never
        # rewrite the file — a rewrite would destroy the tamper evidence
        # that verify_chain() relies on.
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def verify_chain(self) -> tuple[bool, str]:
        """Recompute EVERYTHING and report the first break, if any.

        Checks, in order per epoch: seq monotonic from 1, merkle root
        recomputed from stored leaves, prev_epoch_root links to the prior
        epoch (genesis for epoch 1), epoch_root recomputed from the chain
        rule, and >= M valid threshold signatures over the canonical
        message. Any mismatch -> (False, reason). Never auto-picks a
        "correct" side — the caller decides what a break means.
        """
        try:
            records = self._read_all()
        except (OSError, ValueError) as exc:
            return False, f"anchor log unreadable: {exc}"
        prev_root = GENESIS_EPOCH_ROOT
        for i, rec in enumerate(records):
            expect_seq = i + 1
            if rec.get("anchor_seq") != expect_seq:
                return False, (
                    f"epoch {i + 1}: anchor_seq {rec.get('anchor_seq')!r} "
                    f"!= expected {expect_seq} (seq not monotonic)")
            leaves = rec.get("leaves")
            if not isinstance(leaves, list):
                return False, f"epoch {expect_seq}: leaves missing or not a list"
            recomputed_root = merkle_root(leaves)
            if rec.get("merkle_root") != recomputed_root:
                return False, (
                    f"epoch {expect_seq}: merkle_root mismatch "
                    f"(stored {rec.get('merkle_root')!r} != "
                    f"recomputed {recomputed_root!r})")
            if rec.get("prev_epoch_root") != prev_root:
                return False, (
                    f"epoch {expect_seq}: prev_epoch_root mismatch "
                    f"(stored {rec.get('prev_epoch_root')!r} != "
                    f"expected {prev_root!r})")
            recomputed_epoch = chain_epoch_root(prev_root, recomputed_root)
            if rec.get("epoch_root") != recomputed_epoch:
                return False, (
                    f"epoch {expect_seq}: epoch_root mismatch "
                    f"(stored {rec.get('epoch_root')!r} != "
                    f"recomputed {recomputed_epoch!r})")
            try:
                msg = _signing_message(rec)
            except KeyError as exc:
                return False, f"epoch {expect_seq}: record missing field {exc}"
            if not self.signer.verify(msg, rec.get("signatures") or {}):
                return False, (
                    f"epoch {expect_seq}: threshold signature check failed "
                    f"(need >= {self.signer.m} valid signatures)")
            prev_root = rec["epoch_root"]
        return True, f"chain valid: {len(records)} epoch(s) verified"


# ---------------------------------------------------------------------------
# Publisher: local stand-in for S3 Object Lock
# ---------------------------------------------------------------------------

class AnchorPublisher:
    """Fan-out for anchored epoch records.

    ``publish(epoch_record)`` writes the record to the local provider dir
    (``<base_dir>/<provider>/epochs.jsonl``). ``publish_to_provider`` is
    the hook a real S3 Object Lock implementation plugs into — same
    interface, same record shape — without this module ever attempting a
    real AWS call. Local files are NOT Object Lock: they are a stand-in
    for interface parity, not for immutability guarantees.
    """

    def __init__(self, base_dir: str, default_provider: str = "local"):
        self.base_dir = base_dir
        self.default_provider = default_provider

    def publish_to_provider(self, provider_name: str, record: dict) -> str:
        if not provider_name or "/" in provider_name or "\\" in provider_name \
                or provider_name in (".", ".."):
            raise ValueError(f"invalid provider name: {provider_name!r}")
        out_dir = os.path.join(self.base_dir, provider_name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "epochs.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return path

    def publish(self, epoch_record: dict) -> str:
        """Publish one epoch record to the default provider. Returns path."""
        return self.publish_to_provider(self.default_provider, epoch_record)
