"""Storage safety: O_DIRECT cold reads, mmap policy, ring models.

O_DIRECT constraints (kernel returns EINVAL otherwise):
- buffer address % block_size == 0 (4096 here),
- file offset % block_size == 0,
- transfer length % block_size == 0.
The reader below rounds the file offset DOWN and the length UP to
block boundaries, reads full blocks, then slices in memory to the
requested range.  Buffers come from ``posix_memalign`` via ctypes;
on filesystems where O_DIRECT is unsupported (tmpfs) it falls back
to plain pread — never an EINVAL loop.

mmap policy: NO mmap for cold snapshots.  Bounded mmap window only
for the hot snapshot N + tail.  Cold reads use O_DIRECT/pread +
``POSIX_FADV_DONTNEED`` / ``NOREUSE``; hot reads use
``POSIX_FADV_WILLNEED``.

Rings: :class:`Ring` models the io_uring SQ/CQ state machine the
engine relies on (SQPOLL NEED_WAKEUP, SQ-full backpressure, CQ
overflow via koverflow, SENDZC MORE/NOTIF) deterministically in
pure Python so every rule is testable without a kernel binding.
The same state machine drives the real submission path.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass, field

BLOCK_SIZE = 4096

# IORING flags (kernel uapi, stable values).
IORING_SQ_NEED_WAKEUP = 1
IORING_CQE_F_MORE = 1
IORING_CQE_F_NOTIF = 2


def aligned_buffer(size: int, alignment: int = BLOCK_SIZE) -> tuple[object, int]:
    """Allocate ``size`` bytes aligned to ``alignment``.

    Returns (ctypes buffer, address).  The caller must keep the buffer
    alive while the address is used.  ``size`` is rounded up to a
    multiple of ``alignment``.
    """
    libc_name = ctypes.util.find_library("c")
    libc = ctypes.CDLL(libc_name, use_errno=True)
    rounded = ((int(size) + alignment - 1) // alignment) * alignment
    ptr = ctypes.c_void_p()
    rc = libc.posix_memalign(ctypes.byref(ptr), alignment, rounded)
    if rc != 0:
        raise OSError(rc, f"posix_memalign failed for {rounded} bytes")
    buf = (ctypes.c_char * rounded).from_address(ptr.value)
    return buf, ptr.value


def fadvise_dontneed(fd: int, offset: int = 0, length: int = 0) -> None:
    """POSIX_FADV_DONTNEED | NOREUSE for a cold range (best effort)."""
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_NOREUSE)
    except (AttributeError, OSError):
        pass


def fadvise_willneed(fd: int, offset: int = 0, length: int = 0) -> None:
    """POSIX_FADV_WILLNEED for a hot range (best effort)."""
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_WILLNEED)
    except (AttributeError, OSError):
        pass


def read_cold_blocks(path: str, offset: int, length: int) -> bytes:
    """Read ``length`` bytes at ``offset`` without polluting page cache.

    O_DIRECT with 4096-aligned buffer/offset/length; unaligned
    requests are rounded to block boundaries and sliced in memory.
    Falls back to pread when O_DIRECT is unsupported (e.g. tmpfs).
    """
    if length <= 0:
        return b""
    aligned_off = (offset // BLOCK_SIZE) * BLOCK_SIZE
    lead = offset - aligned_off
    aligned_len = ((lead + length + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    use_direct = hasattr(os, "O_DIRECT")
    try:
        fd = os.open(path, os.O_RDONLY |
                     (os.O_DIRECT if use_direct else 0))
    except OSError:
        fd = os.open(path, os.O_RDONLY)
        use_direct = False
    try:
        if use_direct:
            try:
                # libc pread(fd, aligned_ptr, count, offset): the
                # aligned pointer is passed straight to the kernel,
                # which is what O_DIRECT requires.  os.read would
                # allocate its own (possibly unaligned) buffer, so it
                # must NOT be used on the O_DIRECT path.
                libc = ctypes.CDLL(ctypes.util.find_library("c"),
                                   use_errno=True)
                libc.pread.restype = ctypes.c_ssize_t
                libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.c_int64]
                buf, addr = aligned_buffer(aligned_len)
                got = 0
                while got < aligned_len:
                    n = libc.pread(fd, addr + got, aligned_len - got,
                                   aligned_off + got)
                    if n < 0:
                        err = ctypes.get_errno()
                        raise OSError(err, os.strerror(err))
                    if n == 0:
                        break
                    got += n
                data = bytes(buf[:got])
                fadvise_dontneed(fd, aligned_off, aligned_len)
                return data[lead:lead + length]
            except OSError:
                # O_DIRECT unsupported (tmpfs) or rejected: fall
                # through to the plain pread path below.
                pass
        # Plain pread fallback (no O_DIRECT): always correct.
        data = os.pread(fd, length, offset) if hasattr(os, "pread") else b""
        if not hasattr(os, "pread"):
            os.lseek(fd, offset, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = length
            while remaining > 0:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
        return data
    finally:
        os.close(fd)


@dataclass
class CQE:
    """Completion queue entry."""

    user_data: int
    res: int = 0
    flags: int = 0


class RingOverflowError(Exception):
    """CQ overflowed — koverflow > 0, fail closed and reconcile."""


@dataclass
class Ring:
    """Deterministic SQ/CQ ring model (per-thread SPSC in production).

    - ``sq_capacity`` / ``cq_capacity`` bound the queues.
    - SQ full -> the submission goes to a bounded local pending queue;
      the caller NEVER blocks.
    - CQ overflow increments ``koverflow``; the consumer must check
      it after every reap and fail closed + reconcile via the
      ``outstanding`` (user_data) map.
    - SQPOLL: after updating the SQ tail, read ``sq_flags``; call
      ``io_uring_enter(SQ_WAKEUP)`` ONLY if NEED_WAKEUP is set.
    - SENDZC: first CQE has MORE (buffer still pinned); the second
      CQE has NOTIF (buffer releasable).  Freeing on MORE is a
      use-after-free.
    """

    sq_capacity: int = 64
    cq_capacity: int = 128
    sq: list[dict] = field(default_factory=list)
    cq: list[CQE] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)
    pending_capacity: int = 256
    koverflow: int = 0
    sq_flags: int = 0  # IORING_SQ_NEED_WAKEUP when the kernel sleeps
    wakeups: int = 0
    outstanding: dict[int, dict] = field(default_factory=dict)

    # ---------------------------------------------------------- submit
    def submit(self, sqe: dict) -> bool:
        """Submit one SQE.  True = queued in the SQ; False = went to
        the bounded pending queue (SQ full).  Never blocks."""
        if len(self.sq) < self.sq_capacity:
            self.sq.append(sqe)
            self.outstanding[sqe["user_data"]] = sqe
            self._maybe_wakeup()
            return True
        if len(self.pending) >= self.pending_capacity:
            raise OverflowError("local pending queue full: shed with 429/503")
        self.pending.append(sqe)
        return False

    def _maybe_wakeup(self) -> None:
        """SQPOLL wakeup rule: wake ONLY if NEED_WAKEUP is set.

        Caller order (enforced by callers of submit): SQ tail updated
        first, then flags read.  The kernel sets NEED_WAKEUP before
        sleeping, so no wakeup is ever missed.
        """
        if self.sq_flags & IORING_SQ_NEED_WAKEUP:
            self.wakeups += 1  # models io_uring_enter(SQ_WAKEUP)

    def drain_pending(self) -> int:
        """Move pending SQEs into the SQ as slots free.  Returns count."""
        moved = 0
        while self.pending and len(self.sq) < self.sq_capacity:
            sqe = self.pending.pop(0)
            self.sq.append(sqe)
            self.outstanding[sqe["user_data"]] = sqe
            moved += 1
        if moved:
            self._maybe_wakeup()
        return moved

    # ---------------------------------------------------------- complete
    def complete(self, user_data: int, res: int = 0, flags: int = 0) -> None:
        """Kernel-side completion.  Overflow increments koverflow and
        drops the CQE — never silent, because koverflow is checked."""
        if len(self.cq) >= self.cq_capacity:
            self.koverflow += 1
            return
        self.cq.append(CQE(user_data=user_data, res=res, flags=flags))

    def reap(self) -> list[CQE]:
        """Reap all CQEs.  Raises RingOverflowError if koverflow > 0 —
        the caller must reconcile ``outstanding`` before proceeding."""
        out = list(self.cq)
        self.cq.clear()
        if self.koverflow:
            overflowed = self.koverflow
            self.koverflow = 0
            raise RingOverflowError(
                f"CQ overflowed: {overflowed} CQE(s) dropped; "
                "reconcile outstanding user_data before proceeding")
        for cqe in out:
            if not (cqe.flags & IORING_CQE_F_MORE):
                self.outstanding.pop(cqe.user_data, None)
        # Free SQ slots for completed work and drain pending.
        done = {c.user_data for c in out
                if not (c.flags & IORING_CQE_F_MORE)}
        self.sq = [s for s in self.sq if s["user_data"] not in done]
        self.drain_pending()
        return out

    # ---------------------------------------------------------- SENDZC
    def sendzc(self, user_data: int, payload: bytes) -> None:
        """Zero-copy send: MORE CQE first, NOTIF CQE second.

        The buffer must not be freed/mutated until the NOTIF CQE.
        This method models the kernel sequence; the caller owns the
        buffer lifecycle and must wait for NOTIF.
        """
        self.submit({"user_data": user_data, "op": "SENDZC",
                     "payload": payload})
        self.complete(user_data, res=len(payload),
                      flags=IORING_CQE_F_MORE)
        self.complete(user_data, res=0, flags=IORING_CQE_F_NOTIF)


def sendzc_buffer_releasable(cqes: list[CQE], user_data: int) -> bool:
    """True only after a NOTIF CQE for ``user_data``.  MORE is NOT enough."""
    return any(c.user_data == user_data and
               (c.flags & IORING_CQE_F_NOTIF) for c in cqes)
