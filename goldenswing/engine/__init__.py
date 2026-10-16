"""GoldenSwing deterministic fault-tolerant engine.

Slices (each independently testable):
- :mod:`goldenswing.engine.journal` — append-only chained journal,
  fencing, idempotency, group commit, torn-tail truncation.
- :mod:`goldenswing.engine.snapshot` — Merkle snapshots, atomic
  manifest, N/N_prev retention, orphan cleanup.
- :mod:`goldenswing.engine.storage` — O_DIRECT cold reads, mmap
  policy, ring (SQ/CQ/SENDZC) state machine.
- :mod:`goldenswing.engine.replication` — journal replication,
  quorum commit, InstallSnapshot, follower apply.
- :mod:`goldenswing.engine.reconciliation` — UNCONFIRMED/UNKNOWN,
  broker-truth reconciliation, LIVE fence, holding drain.
- :mod:`goldenswing.engine.backtest_export` — backtest file +
  summary for Telegram.

Phases: REPLAY -> RECONCILING -> LIVE.  Effects are suppressed in
REPLAY/RECONCILING and fire only after the quorum-committed
``reconciliation_complete`` fence.
"""
from .journal import (CorruptionError, DeterministicJournal,
                      FencedAuthority, FencingError, IdempotencyConflict,
                      fencing_token, payload_hash)
from .snapshot import (SnapshotStore, build_snapshot, verify_snapshot)
from .storage import (Ring, RingOverflowError, read_cold_blocks,
                      sendzc_buffer_releasable)
from .replication import Follower, Leader, chunk_snapshot
from .reconciliation import Reconciler
from .backtest_export import (chunk_text, format_summary, summarize,
                              write_report)
from .pipeline import aggregate_bars, merge_events, transition
from .runtime import DeterministicRuntime, load_or_create_secret

__all__ = [
    "CorruptionError", "DeterministicJournal", "FencedAuthority",
    "FencingError", "IdempotencyConflict", "fencing_token",
    "payload_hash", "SnapshotStore", "build_snapshot",
    "verify_snapshot", "Ring", "RingOverflowError",
    "read_cold_blocks", "sendzc_buffer_releasable",
    "Follower", "Leader", "chunk_snapshot",
    "Reconciler",
    "chunk_text", "format_summary", "summarize", "write_report",
    "DeterministicRuntime", "load_or_create_secret",
    "aggregate_bars", "merge_events", "transition",
]
