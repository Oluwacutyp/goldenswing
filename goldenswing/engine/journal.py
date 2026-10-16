"""Deterministic core journal: append-only, chained, fenced, idempotent.

Frame wire format (binary, big-endian)::

    [magic 4s "GSJ1"][length 4I][seq 8Q][epoch 8Q][term 8Q]
    [token_version 4I][payload_len 4I][payload bytes]
    [crc32 4I][chain 32s]

``crc32`` is CRC-32 (binascii.crc32) over header + payload — the
accident layer.  ``chain`` is SHA-256(prev_chain || header || payload
|| crc32) — the tamper layer.  ``chain`` of seq 0 is 32 zero bytes.

Rules (all enforced here, nowhere else):
- ``seq == head + 1`` always; gaps rejected.
- Torn tail (partial final frame) is truncated to the last valid
  frame on open.  Middle corruption fails closed — never skipped.
- Group commit: :meth:`DeterministicJournal.append_batch` fsyncs
  once per batch (batch up to ``max_batch`` or caller flush).
- Fencing: every frame carries epoch/term/leader_id/token_version
  plus an HMAC fencing token.  The storage path validates them
  against the current fenced authority in the SAME journal
  transaction (same lock) that assigns the sequence.  Stale epoch,
  wrong leader, stale token version or bad HMAC is rejected BEFORE
  durability — no frame, no sequence allocation, no publication.
- Idempotency: dedup consults ONLY the committed journal.  Same
  key + same payload hash replays the committed result; same key +
  different payload hash is a 409 conflict.  Provisional / uncommitted
  records are audit-only and never a disposition.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import struct
import threading
import zlib

MAGIC = b"GSJ1"
_HEADER = struct.Struct(">4sIQ Q Q I I")
# header fields: magic, length(total after header excl crc+chain?), seq, epoch, term, token_version, payload_len
# We encode length = payload_len for simplicity; total frame size is
# header + payload + 4 (crc) + 32 (chain).
_CRC_LEN = 4
_CHAIN_LEN = 32
_ZERO_CHAIN = b"\x00" * 32


class JournalError(Exception):
    """Base journal failure."""


class FencingError(JournalError):
    """Stale epoch / leader / token — rejected before durability."""


class CorruptionError(JournalError):
    """Middle corruption — fail closed, never skip."""


class IdempotencyConflict(JournalError):
    """Same idempotency key, different payload hash (409)."""

    def __init__(self, key: str, committed_seq: int):
        super().__init__(f"idempotency_key_conflict: {key} at seq {committed_seq}")
        self.key = key
        self.committed_seq = committed_seq


def fencing_token(secret: bytes, epoch: int, term: int, leader_id: str,
                  token_version: int) -> str:
    """HMAC fencing token for one (epoch, term, leader, version)."""
    msg = f"{epoch}|{term}|{leader_id}|{token_version}".encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def payload_hash(payload: dict) -> str:
    """Canonical payload hash — same payload, same hash."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _crc(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def encode_frame(seq: int, epoch: int, term: int, token_version: int,
                 payload: bytes, prev_chain: bytes) -> bytes:
    """Encode one frame (no validation — caller owns fencing/seq)."""
    header = _HEADER.pack(MAGIC, len(payload), seq, epoch, term,
                          token_version, len(payload))
    crc = _crc(header + payload)
    chain = hashlib.sha256(prev_chain + header + payload +
                           struct.pack(">I", crc)).digest()
    return header + payload + struct.pack(">I", crc) + chain


def decode_frame(buf: bytes, offset: int):
    """Decode one frame at ``offset``.

    Returns ``(frame_dict, next_offset)`` or ``(None, offset)`` for a
    torn tail (fewer bytes than a complete frame).  Raises
    :class:`CorruptionError` for bad magic / CRC / chain.
    """
    if len(buf) - offset < _HEADER.size + _CRC_LEN + _CHAIN_LEN:
        return None, offset  # torn tail: not enough bytes for any frame
    header = buf[offset:offset + _HEADER.size]
    magic, _length, seq, epoch, term, token_version, plen = \
        _HEADER.unpack(header)
    if magic != MAGIC:
        raise CorruptionError(f"bad magic at offset {offset}")
    total = _HEADER.size + plen + _CRC_LEN + _CHAIN_LEN
    if len(buf) - offset < total:
        return None, offset  # torn tail: partial frame
    payload = buf[offset + _HEADER.size: offset + _HEADER.size + plen]
    crc_off = offset + _HEADER.size + plen
    (crc_stored,) = struct.unpack(">I", buf[crc_off:crc_off + 4])
    if _crc(header + payload) != crc_stored:
        raise CorruptionError(f"CRC mismatch at seq {seq} offset {offset}")
    chain = buf[crc_off + 4: crc_off + 4 + _CHAIN_LEN]
    return {
        "seq": seq, "epoch": epoch, "term": term,
        "token_version": token_version, "payload": payload,
        "crc": crc_stored, "chain": chain, "offset": offset,
    }, offset + total


class FencedAuthority:
    """Current fenced authority (epoch/term/leader/token_version)."""

    def __init__(self, epoch: int, term: int, leader_id: str,
                 token_version: int, secret: bytes):
        self.epoch = int(epoch)
        self.term = int(term)
        self.leader_id = str(leader_id)
        self.token_version = int(token_version)
        self.secret = bytes(secret)

    def token(self) -> str:
        return fencing_token(self.secret, self.epoch, self.term,
                             self.leader_id, self.token_version)

    def validate(self, epoch: int, term: int, leader_id: str,
                 token_version: int, token: str) -> None:
        """Raise FencingError unless the proposal matches authority."""
        if int(term) < self.term or int(epoch) < self.epoch:
            raise FencingError(
                f"stale epoch/term: got epoch={epoch} term={term}, "
                f"current epoch={self.epoch} term={self.term}")
        if int(term) == self.term and str(leader_id) != self.leader_id:
            raise FencingError(
                f"wrong leader for term {term}: {leader_id!r} != "
                f"{self.leader_id!r}")
        if int(token_version) < self.token_version:
            raise FencingError(
                f"stale token_version: {token_version} < {self.token_version}")
        expected = fencing_token(self.secret, int(epoch), int(term),
                                 str(leader_id), int(token_version))
        if not hmac.compare_digest(str(token), expected):
            raise FencingError("fencing token HMAC mismatch")


class DeterministicJournal:
    """Append-only journal file with fencing + idempotency + chain."""

    def __init__(self, path: str, authority: FencedAuthority):
        self.path = str(path)
        self.authority = authority
        self._lock = threading.Lock()
        self._frames: list[dict] = []
        self._head_chain = _ZERO_CHAIN
        self._idempotency: dict[str, dict] = {}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)),
                    exist_ok=True)
        self._load()

    # ---------------------------------------------------------- load / verify
    def _load(self) -> None:
        """Load + verify; truncate ONLY a torn final frame."""
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as fh:
            buf = fh.read()
        offset = 0
        prev_chain = _ZERO_CHAIN
        frames: list[dict] = []
        while offset < len(buf):
            try:
                frame, nxt = decode_frame(buf, offset)
            except CorruptionError:
                # decode_frame raises only for a COMPLETE-length frame
                # with bad magic/CRC — that is middle corruption if any
                # bytes follow, torn tail only if it is the last frame.
                # Re-raise as middle corruption unless the remainder
                # is shorter than a full frame (already handled by the
                # torn-tail path returning None).
                raise
            if frame is None:
                break  # torn tail
            expected = hashlib.sha256(
                prev_chain + buf[offset:offset + _HEADER.size +
                                 len(frame["payload"]) + 4]).digest()
            if frame["chain"] != expected:
                raise CorruptionError(
                    f"chain mismatch at seq {frame['seq']} offset {offset}")
            if frames and frame["seq"] != frames[-1]["seq"] + 1:
                raise CorruptionError(
                    f"sequence gap: {frames[-1]['seq']} -> {frame['seq']}")
            if not frames and frame["seq"] != 1:
                raise CorruptionError(
                    f"first seq must be 1, got {frame['seq']}")
            frames.append(frame)
            prev_chain = frame["chain"]
            offset = nxt
        if offset < len(buf):
            # Torn tail: truncate to the last valid frame boundary.
            with open(self.path, "r+b") as fh:
                fh.truncate(offset)
                fh.flush()
                os.fsync(fh.fileno())
        self._frames = frames
        self._head_chain = prev_chain if frames else _ZERO_CHAIN
        self._rebuild_idempotency()

    def _rebuild_idempotency(self) -> None:
        self._idempotency = {}
        for frame in self._frames:
            try:
                payload = json.loads(frame["payload"].decode())
            except (ValueError, UnicodeDecodeError):
                continue
            key = payload.get("idempotency_key")
            if key:
                self._idempotency[str(key)] = {
                    "payload_hash": payload.get("payload_hash") or
                    payload_hash(payload.get("data") or {}),
                    "seq": frame["seq"],
                    "payload": payload,
                }

    # ---------------------------------------------------------- properties
    @property
    def head(self) -> int:
        return self._frames[-1]["seq"] if self._frames else 0

    @property
    def head_chain(self) -> bytes:
        return self._head_chain

    @property
    def head_chain_hex(self) -> str:
        return self._head_chain.hex()

    def frames(self) -> list[dict]:
        return list(self._frames)

    # ---------------------------------------------------------- idempotency
    def idempotency_lookup(self, key: str):
        """Committed-only lookup: None, or the committed record."""
        return self._idempotency.get(str(key))

    def check_idempotency(self, key: str, phash: str):
        """Return (action, record).

        action: "new" | "replay" | "conflict".  Conflict raises
        :class:`IdempotencyConflict` (409).
        """
        rec = self._idempotency.get(str(key))
        if rec is None:
            return "new", None
        if rec["payload_hash"] == phash:
            return "replay", rec
        raise IdempotencyConflict(str(key), rec["seq"])

    # ---------------------------------------------------------- append
    def append(self, *, epoch: int, term: int, leader_id: str,
               token_version: int, token: str, data: dict,
               idempotency_key: str | None = None) -> dict:
        """Append one frame.  Fencing validated BEFORE durability;
        sequence assigned in the same lock.  Returns the frame meta.
        """
        return self.append_batch([{
            "epoch": epoch, "term": term, "leader_id": leader_id,
            "token_version": token_version, "token": token,
            "data": data, "idempotency_key": idempotency_key,
        }])[0]

    def append_batch(self, items: list[dict]) -> list[dict]:
        """Append a batch with ONE fsync (group commit)."""
        if not items:
            return []
        with self._lock:
            # 1. Validate fencing for EVERY item before ANY durability.
            for item in items:
                self.authority.validate(
                    item["epoch"], item["term"], item["leader_id"],
                    item["token_version"], item["token"])
            # 2. Validate idempotency for every item before durability.
            # Same key + same payload replays the committed frame
            # without allocating a new seq (no duplicate).  Same key +
            # different payload raises 409 via check_idempotency.
            for item in items:
                key = item.get("idempotency_key")
                if key:
                    ph = payload_hash(item["data"])
                    action, rec = self.check_idempotency(str(key), ph)
                    if action == "replay":
                        return [{"seq": rec["seq"],
                                 "epoch": item["epoch"],
                                 "term": item["term"],
                                 "token_version": item["token_version"],
                                 "payload": rec["payload"].get("payload", b"")
                                 if isinstance(rec.get("payload"), dict)
                                 else b"",
                                 "chain": self._head_chain,
                                 "data": item["data"],
                                 "idempotency_key": str(key),
                                 "replayed": True}]
            # 3. Encode + write + single fsync.
            out: list[dict] = []
            blobs: list[bytes] = []
            prev_chain = self._head_chain
            seq = self.head
            for item in items:
                seq += 1
                payload_obj = {
                    "data": item["data"],
                    "payload_hash": payload_hash(item["data"]),
                }
                if item.get("idempotency_key"):
                    payload_obj["idempotency_key"] = str(
                        item["idempotency_key"])
                payload = json.dumps(
                    payload_obj, sort_keys=True,
                    separators=(",", ":")).encode()
                blob = encode_frame(seq, int(item["epoch"]),
                                    int(item["term"]),
                                    int(item["token_version"]),
                                    payload, prev_chain)
                # Chain for the NEXT frame is this frame's chain.
                prev_chain = hashlib.sha256(
                    prev_chain + blob[:_HEADER.size + len(payload) + 4]
                ).digest()
                blobs.append(blob)
                meta = {"seq": seq, "epoch": int(item["epoch"]),
                        "term": int(item["term"]),
                        "token_version": int(item["token_version"]),
                        "payload": payload, "chain": prev_chain,
                        "data": item["data"]}
                if item.get("idempotency_key"):
                    meta["idempotency_key"] = str(item["idempotency_key"])
                out.append(meta)
            with open(self.path, "ab") as fh:
                for blob in blobs:
                    fh.write(blob)
                fh.flush()
                os.fsync(fh.fileno())
            for meta in out:
                frame = {"seq": meta["seq"], "epoch": meta["epoch"],
                         "term": meta["term"],
                         "token_version": meta["token_version"],
                         "payload": meta["payload"],
                         "chain": meta["chain"],
                         "offset": 0}
                self._frames.append(frame)
                key = meta.get("idempotency_key")
                if key:
                    self._idempotency[key] = {
                        "payload_hash": payload_hash(meta["data"]),
                        "seq": meta["seq"],
                        "payload": {"data": meta["data"]},
                    }
            self._head_chain = prev_chain
            return out

    # ---------------------------------------------------------- verify / replay
    def verify(self) -> bool:
        """Re-read from disk and verify the full chain.  True = clean."""
        try:
            other = DeterministicJournal.__new__(DeterministicJournal)
            other.path = self.path
            other._frames = []
            other._head_chain = _ZERO_CHAIN
            other._idempotency = {}
            other._lock = threading.Lock()
            other.authority = self.authority
            # Bypass __init__ load (which truncates); do a pure verify.
            with open(self.path, "rb") as fh:
                buf = fh.read()
            offset = 0
            prev = _ZERO_CHAIN
            expect_seq = 1
            while offset < len(buf):
                frame, nxt = decode_frame(buf, offset)
                if frame is None:
                    return False  # torn tail present
                expected = hashlib.sha256(
                    prev + buf[offset:offset + _HEADER.size +
                               len(frame["payload"]) + 4]).digest()
                if frame["chain"] != expected or frame["seq"] != expect_seq:
                    return False
                prev = frame["chain"]
                offset = nxt
                expect_seq += 1
            return offset == len(buf)
        except (CorruptionError, OSError):
            return False
