"""Crash-safe mutual exclusion for the append-only journal.

The journal is a plain file and sequence numbers are derived from the line
count, so concurrent writers would duplicate seqs and interleave lines.
This module is the transaction boundary everything journal-related flows
through.

Design:
- Inter-process: fcntl.flock(LOCK_EX). The OS releases the lock when the
  holder dies, so a crashed process can NEVER leave a stale lock behind.
  (PID lockfiles cannot promise this.) Crash recovery is safe by
  construction: the next acquirer always gets a clean lock.
- Intra-process: re-entrant via a process-local hold count, so a helper
  that appends (and locks) can be called inside a larger transaction
  (which also locks) without self-deadlock. Thread-safe via a mutex.

Torn-tail recovery: if a previous writer died mid-append, the file may end
with a partial line (no trailing newline). On acquisition we truncate back
to the last newline. Rationale: we always terminate records with "\\n", so
a final line WITHOUT one was never a complete record — and letting the
next append glue onto it would destroy two records instead of zero.

Contract:
- Every journal WRITE happens inside journal_transaction().
- blocking=False lets a tick detect a contested journal (recovery in
  progress) and skip its cycle cleanly instead of queuing behind it.
- Reads tolerate a torn tail (the parser skips bad lines), but writers
  always repair it first, so seq derivation (line count + 1) is stable
  inside the transaction.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)

_state_lock = threading.Lock()
# (lock_path, thread_ident) -> [fileobj, hold_count]
# Thread-local: re-entrancy applies only within the SAME thread. A different
# thread is a genuinely different contender — it must go through flock
# (which serializes across file descriptions even in one process), so
# blocking=False correctly reports contention instead of barging in.
_holds: dict[tuple[str, int], list] = {}


def _key(lock_path: str) -> tuple[str, int]:
    return (lock_path, threading.get_ident())


def _repair_torn_tail(journal_path: str) -> None:
    """Truncate a partial final line left by a crashed writer."""
    if not os.path.exists(journal_path) or os.path.getsize(journal_path) == 0:
        return
    with open(journal_path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        # Check the last byte; scan back a bounded window for the last \n.
        window = min(size, 65536)
        f.seek(size - window)
        tail = f.read()
        if tail.endswith(b"\n"):
            return  # clean
        idx = tail.rfind(b"\n")
        if idx == -1:
            f.truncate(0)  # single torn line is the whole file
        else:
            f.truncate(size - window + idx + 1)


@contextmanager
def journal_transaction(journal_path: str, blocking: bool = True):
    """Exclusive transaction boundary for journal writes.

    blocking=True (default): wait for the lock. Use for the reconnection
      sequence and standalone appends.
    blocking=False: yields True if the lock was acquired, False if contested.
      Use for normal ticks: a contested journal means recovery is in
      progress, so the tick skips its cycle instead of evaluating
      mid-recovery state.

    The lock is held across the whole `with` body — the reconnection
    sequence keeps it from the first reconcile append through the final
    monitor check, making the batch one atomic unit. Crash-safe: the OS
    frees the lock if the holder dies.
    """
    lock_path = journal_path + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    key = _key(lock_path)

    with _state_lock:
        hold = _holds.get(key)
        if hold is not None:
            hold[1] += 1
            reentrant = True
        else:
            reentrant = False
    if reentrant:
        try:
            yield True
        finally:
            with _state_lock:
                h = _holds.get(key)
                if h is not None:
                    h[1] -= 1
        return

    f = open(lock_path, "w")
    try:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(f.fileno(), flags)
        except BlockingIOError:
            f.close()
            yield False
            return
    except Exception:
        f.close()
        raise

    with _state_lock:
        _holds[key] = [f, 1]
    try:
        _repair_torn_tail(journal_path)
        yield True
    finally:
        with _state_lock:
            _holds.pop(key, None)
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


# ---------------------------------------------------------------------------
# Timeout wrapper: bound blocking calls made while the journal lock is held.
#
# Health checks, broker fetches and backfills run inside the journal
# transaction (init holds it across the whole sequence; ticks hold it across
# the cycle). A hung network call must never hold the lock forever, so every
# such call goes through call_with_timeout().
# ---------------------------------------------------------------------------

class _CallOutcome:
    """One-shot box carrying a worker thread's result across the join.

    Single writer (the worker thread), single reader (the joining thread).
    Thread.join() on a finished thread is a memory barrier, so the box needs
    no lock — provided it is ONLY read after join() confirms the worker has
    finished. Reading it while the worker is alive would race a half-written
    box, which is why call_with_timeout() is strict about the read ordering.
    """
    __slots__ = ("settled", "value", "error")

    def __init__(self):
        self.settled = False
        self.value = None
        self.error = None


def call_with_timeout(fn, timeout_s, *, label="call"):
    """Run fn() in a daemon thread; return its value or re-raise its error.

    Raises TimeoutError if fn() is still running after timeout_s seconds.
    A timed-out worker is abandoned, never killed: fn must never take the
    journal lock (it runs in a different thread, so per-thread re-entrancy
    does not apply — a lock attempt would deadlock against the holder), and
    as a daemon thread the abandoned worker cannot block interpreter
    shutdown. (concurrent.futures.ThreadPoolExecutor is unsuitable here: its
    workers are non-daemon and its atexit hook joins them, so a hung future
    would wedge process exit.)

    Exception transparency: anything fn() raises — including
    KeyboardInterrupt/SystemExit — is re-raised in the calling thread.
    Python 3 appends the re-raise frame to the worker's original
    __traceback__, so the true origin is preserved, not masked.
    """
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
    outcome = _CallOutcome()

    def _target():
        try:
            outcome.value = fn()
        except BaseException as exc:
            outcome.error = exc
        finally:
            outcome.settled = True

    worker = threading.Thread(target=_target, daemon=True,
                              name=f"watchdog-{label}")
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"{label} timed out after {timeout_s:g}s")
    assert outcome.settled, (
        f"watchdog-{label}: thread finished without settling the outcome")
    if outcome.error is not None:
        raise outcome.error
    return outcome.value


# ---------------------------------------------------------------------------
# Atomic append path: the ONLY sanctioned way to add a journal record.
#
# Sequence numbers derive from the line count, so deriving them outside the
# lock is a TOCTOU race: two concurrent writers read the same count and emit
# colliding seqs (and worse, colliding seq-derived trade_ids). The count is
# only meaningful inside the lock anyway — acquisition repairs a torn tail
# first, and _journal_length counts non-blank lines, so a torn partial line
# observed outside the lock would corrupt the count by one.
# ---------------------------------------------------------------------------

def _journal_length(journal_path: str) -> int:
    """Number of complete records in the journal.

    MUST be called with the journal lock held: only then has any torn tail
    been repaired, making the count exact and stable for the critical section.
    """
    if not os.path.exists(journal_path):
        return 0
    with open(journal_path) as f:
        return sum(1 for line in f if line.strip())


def append_journal_record(journal_path: str, record: dict) -> int:
    """Append one record, assigning its sequence number atomically.

    Derives seq, stamps record["seq"], appends with fsync — all inside one
    journal_transaction. Returns the assigned seq. Callers must NEVER compute
    seq themselves; any seq-derived content (e.g. default trade_ids) has to be
    resolved inside this critical section too.
    """
    os.makedirs(os.path.dirname(os.path.abspath(journal_path)), exist_ok=True)
    with journal_transaction(journal_path):
        seq = _journal_length(journal_path) + 1
        record["seq"] = seq
        with open(journal_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return seq


# ---------------------------------------------------------------------------
# Tick policy: how normal cycles behave when the journal is contested.
#
# A contested journal means the reconnection sequence holds the lock, i.e.
# recovery is in progress. The tick must NOT evaluate mid-recovery state
# (partially reconciled journal, unmarked suspect zones, stale heat).
#
# Policy: skip-fast, skip-visibly, skip-healably.
# - Skip-fast: bounded retry for the just-missed-it case, then give up. Never
#   block indefinitely — a hung recovery must not wedge the tick loop, and
#   waiting workers must not pile up behind it.
# - Skip-visibly: every contention skip is recorded in <journal>.skips.json.
#   N consecutive contention skips fire on_alert once per streak (default 3)
#   — a lock held across multiple tick intervals is pathological (or a
#   genuinely huge backfill) and the operator should know. This mirrors the
#   gap-escalation philosophy: automatic mechanism, human on pathology.
# - Skip-healably: a skip defers work, it never loses it. CALLER CONTRACT:
#   tick_fn must process bars from tracker.last_idx + 1 (catch-up), never
#   assume one-bar-per-tick. The next tick then replays the deferred bars
#   deterministically. (Same principle as the journal itself.)
#
# Note: the reconnection sequence already runs one monitor.check() at its
# end, so a skipped tick misses no evaluation — it avoids a duplicate.
# ---------------------------------------------------------------------------

def _skip_state_path(journal_path: str) -> str:
    return journal_path + ".skips.json"


def _load_skip_state(path: str) -> dict:
    try:
        with open(path) as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_skip_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def _reset_contention_streak(journal_path: str) -> None:
    st = _load_skip_state(_skip_state_path(journal_path))
    if st.get("consecutive_contention_skips"):
        st["consecutive_contention_skips"] = 0
        st["alerted_for_streak"] = False
        _save_skip_state(_skip_state_path(journal_path), st)


def _record_contention_skip(journal_path: str, alert_after: int,
                            on_alert) -> None:
    p = _skip_state_path(journal_path)
    st = _load_skip_state(p)
    st["consecutive_contention_skips"] = st.get("consecutive_contention_skips", 0) + 1
    st["total_contention_skips"] = st.get("total_contention_skips", 0) + 1
    st["last_skip_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    n = st["consecutive_contention_skips"]
    if n >= alert_after and not st.get("alerted_for_streak"):
        st["alerted_for_streak"] = True
        _save_skip_state(p, st)
        msg = (f"journal contested for {n} consecutive ticks "
               f"({journal_path}): recovery may be stuck")
        log.warning(msg)
        if on_alert is not None:
            try:
                on_alert(msg)
            except Exception:
                log.exception("on_alert failed for contention skip")
        return
    _save_skip_state(p, st)


def run_tick_guarded(journal_path: str, tick_fn, *,
                     retry_attempts: int = 3, retry_delay_s: float = 5.0,
                     contention_alert_after: int = 3,
                     on_alert=None) -> tuple[str, object]:
    """Run one tick cycle, handling journal contention.

    Returns ("ran", tick_fn_result) or ("skipped", reason).

    tick_fn(journal_path) holds the journal lock for its whole body: health
    gate, T1–T4, journal appends, monitor check. It must be idempotent w.r.t.
    missed bars (process from tracker.last_idx + 1).

    On contention: retry a bounded number of times (the sequence may be
    seconds from finishing), then skip. A skip is routine — logged, counted,
    and healed by the next tick — not an error, so this returns normally
    rather than raising.
    """
    for attempt in range(max(1, retry_attempts)):
        with journal_transaction(journal_path, blocking=False) as acquired:
            if acquired:
                _reset_contention_streak(journal_path)
                return ("ran", tick_fn(journal_path))
        if attempt < retry_attempts - 1:
            time.sleep(retry_delay_s)
    log.info("tick skipped: journal contested (recovery in progress)")
    _record_contention_skip(journal_path, contention_alert_after, on_alert)
    return ("skipped", "contention: journal locked (recovery in progress)")
