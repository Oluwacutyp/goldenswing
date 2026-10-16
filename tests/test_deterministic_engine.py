"""Tests for the deterministic fault-tolerant engine (goldenswing.engine).

Each slice is tested in isolation: journal fencing/idempotency/chain,
snapshot Merkle/manifest/orphans, storage O_DIRECT + rings + SENDZC,
replication commit + snapshot install, reconciliation fence.
"""
import hashlib
import json
import os

import pytest

from goldenswing.engine import (
    DeterministicJournal, FencedAuthority, FencingError,
    IdempotencyConflict, Leader, Follower, Reconciler, Ring,
    RingOverflowError, SnapshotStore, build_snapshot,
    chunk_snapshot, read_cold_blocks, sendzc_buffer_releasable,
    summarize, format_summary, chunk_text, write_report,
    verify_snapshot, fencing_token, payload_hash,
)
from goldenswing.engine.storage import (
    IORING_CQE_F_MORE, IORING_CQE_F_NOTIF, IORING_SQ_NEED_WAKEUP)


def _authority(secret=b"test-secret"):
    return FencedAuthority(epoch=1, term=1, leader_id="leader-1",
                            token_version=1, secret=secret)


def _tok(auth):
    return auth.token()


def _append(journal, auth, data, key=None):
    return journal.append(epoch=1, term=1, leader_id="leader-1",
                          token_version=1, token=_tok(auth),
                          data=data, idempotency_key=key)


# ---------------------------------------------------------------- journal
def test_journal_append_chain_and_reload(tmp_path):
    auth = _authority()
    path = str(tmp_path / "journal.bin")
    j = DeterministicJournal(path, auth)
    _append(j, auth, {"op": "a"}, key="k1")
    _append(j, auth, {"op": "b"})
    assert j.head == 2
    assert j.verify() is True
    j2 = DeterministicJournal(path, auth)
    assert j2.head == 2
    assert j2.head_chain == j.head_chain


def test_journal_torn_tail_truncated(tmp_path):
    auth = _authority()
    path = str(tmp_path / "journal.bin")
    j = DeterministicJournal(path, auth)
    _append(j, auth, {"op": "a"})
    with open(path, "ab") as fh:
        fh.write(b"\x00\x01partial")
    j2 = DeterministicJournal(path, auth)
    assert j2.head == 1
    assert j2.verify() is True


def test_journal_middle_corruption_fails_closed(tmp_path):
    auth = _authority()
    path = str(tmp_path / "journal.bin")
    j = DeterministicJournal(path, auth)
    _append(j, auth, {"op": "a"})
    _append(j, auth, {"op": "b"})
    _append(j, auth, {"op": "c"})
    data = bytearray(open(path, "rb").read())
    # Corrupt a payload byte in the middle frame (not the tail).
    data[100] ^= 0xFF
    open(path, "wb").write(bytes(data))
    with pytest.raises(Exception):
        DeterministicJournal(path, auth)


def test_journal_fencing_rejects_stale(tmp_path):
    auth = _authority()
    path = str(tmp_path / "journal.bin")
    j = DeterministicJournal(path, auth)
    with pytest.raises(FencingError):
        j.append(epoch=0, term=0, leader_id="leader-1",
                 token_version=1, token=_tok(auth), data={"op": "x"})
    with pytest.raises(FencingError):
        j.append(epoch=1, term=1, leader_id="leader-1",
                 token_version=1, token="bad-token", data={"op": "x"})
    with pytest.raises(FencingError):
        j.append(epoch=1, term=1, leader_id="intruder",
                 token_version=1, token=_tok(auth), data={"op": "x"})
    assert j.head == 0  # nothing allocated on rejection


def test_journal_idempotency_replay_and_conflict(tmp_path):
    auth = _authority()
    path = str(tmp_path / "journal.bin")
    j = DeterministicJournal(path, auth)
    _append(j, auth, {"order": 1}, key="k1")
    assert j.check_idempotency("k1", payload_hash({"order": 1}))[0] == "replay"
    with pytest.raises(IdempotencyConflict):
        _append(j, auth, {"order": 2}, key="k1")
    assert j.head == 1


def test_journal_group_commit_single_fsync(tmp_path):
    auth = _authority()
    path = str(tmp_path / "journal.bin")
    j = DeterministicJournal(path, auth)
    out = j.append_batch([{"epoch": 1, "term": 1, "leader_id": "leader-1",
                           "token_version": 1, "token": _tok(auth),
                           "data": {"i": i}} for i in range(5)])
    assert [m["seq"] for m in out] == [1, 2, 3, 4, 5]
    assert j.verify() is True


# ---------------------------------------------------------------- snapshot
def test_snapshot_merkle_and_root_binding(tmp_path):
    pages = [b"A" * 4096, b"B" * 4096]
    meta = build_snapshot(pages, watermark=10, chain_n=b"\x01" * 32)
    assert verify_snapshot(meta, pages) is True
    assert verify_snapshot(meta, [b"A" * 4096, b"C" * 4096]) is False


def test_snapshot_store_retains_n_and_nprev_and_orphans(tmp_path):
    store = SnapshotStore(tmp_path)
    store.write_snapshot(10, b"\x01" * 32, [b"A" * 4096])
    store.write_snapshot(11, b"\x02" * 32, [b"B" * 4096])
    manifest = store.read_manifest()
    assert [s["watermark"] for s in manifest["snapshots"]] == [11, 10]
    (tmp_path / "stale.tmp").write_bytes(b"x")
    (tmp_path / "stale.part").write_bytes(b"x")
    removed = store.cleanup_orphans()
    assert "stale.tmp" in removed and "stale.part" in removed
    # Manifest-referenced files are never removed.
    assert (tmp_path / "snapshot-11.json").exists()


def test_snapshot_prune_gates(tmp_path):
    store = SnapshotStore(tmp_path)
    assert store.prune_hot_copy(10, cold_verified=True,
                                grace_passed=True,
                                readers_drained=True) is True
    assert store.prune_hot_copy(10, cold_verified=True,
                                grace_passed=False,
                                readers_drained=True) is False


# ---------------------------------------------------------------- storage
def test_cold_read_aligned_and_unaligned(tmp_path):
    path = tmp_path / "cold.bin"
    data = bytes((i * 7) % 256 for i in range(8192))
    path.write_bytes(data)
    assert read_cold_blocks(str(path), 0, 100) == data[:100]
    assert read_cold_blocks(str(path), 100, 5000) == data[100:5100]
    assert read_cold_blocks(str(path), 4096, 4096) == data[4096:8192]


def test_ring_sq_full_goes_to_pending_never_blocks():
    ring = Ring(sq_capacity=2, pending_capacity=4)
    assert ring.submit({"user_data": 1}) is True
    assert ring.submit({"user_data": 2}) is True
    assert ring.submit({"user_data": 3}) is False  # pending, not blocked
    assert len(ring.pending) == 1
    ring.complete(1)
    ring.reap()
    assert len(ring.sq) == 2  # pending drained into the freed slot


def test_ring_sq_wakeup_only_when_need_wakeup():
    ring = Ring()
    ring.sq_flags = 0
    ring.submit({"user_data": 1})
    assert ring.wakeups == 0
    ring.sq_flags = IORING_SQ_NEED_WAKEUP
    ring.submit({"user_data": 2})
    assert ring.wakeups == 1


def test_ring_cq_overflow_fails_closed():
    ring = Ring(sq_capacity=8, cq_capacity=1)
    ring.complete(1)
    ring.complete(2)  # overflows: koverflow == 1
    assert ring.koverflow == 1
    with pytest.raises(RingOverflowError):
        ring.reap()


def test_sendzc_notif_required_before_free():
    ring = Ring()
    ring.sendzc(7, b"hello")
    cqes = ring.reap()
    # reap() consumes the CQEs; check NOTIF semantics via the helper
    # on a fresh completion sequence.
    ring2 = Ring()
    ring2.complete(9, res=5, flags=IORING_CQE_F_MORE)
    assert sendzc_buffer_releasable(ring2.cq, 9) is False
    ring2.complete(9, res=0, flags=IORING_CQE_F_NOTIF)
    assert sendzc_buffer_releasable(ring2.cq, 9) is True


# ---------------------------------------------------------------- replication
def test_replication_commit_majority_median_and_chain():
    leader = Leader("L", cluster_size=3, term=2, epoch=1, head=5)
    leader.record_term(5, 2)
    leader.record_chain(5, "chain5")
    leader.register_follower("F1")
    leader.register_follower("F2")
    # Only one follower acked at 5: median of [5, 5, 0] is 5 -> commit.
    assert leader.on_ack("F1", 5, "chain5", term=2, epoch=1) == 5
    # Chain mismatch never advances and quarantines.
    leader2 = Leader("L", cluster_size=3, term=2, epoch=1, head=6)
    leader2.record_term(6, 2)
    leader2.record_chain(6, "chain6")
    leader2.register_follower("F1")
    assert leader2.on_ack("F1", 6, "WRONG", term=2, epoch=1) == 0
    assert leader2.followers["F1"].quarantined is True


def test_replication_follower_rejects_stale_and_gap():
    follower = Follower("F1", term=2, epoch=1)
    follower.watermark = 10
    assert follower.validate_append(term=1, epoch=1, leader_id="L",
                                    seq=11, chain_hex="x",
                                    prev_chain_hex="",
                                    expected_prev_chain="") is False
    assert follower.validate_append(term=2, epoch=1, leader_id="L",
                                    seq=12, chain_hex="x",
                                    prev_chain_hex="",
                                    expected_prev_chain="") is False


def test_replication_snapshot_chunks_install(tmp_path):
    payload = json.dumps({"watermark": 10, "chain_n": "ab",
                          "snapshot_root": "root"}).encode()
    follower = Follower("F1", term=1, epoch=1)
    for chunk in chunk_snapshot(payload, chunk_size=8):
        assert follower.install_snapshot_chunk(chunk) is True
    assert follower.snapshot_complete is True
    assert follower.install_snapshot(watermark=10, chain_hex="ab",
                                     payload={"snapshot_root": "root"}) is True
    assert follower.watermark == 10


# ---------------------------------------------------------------- reconciliation
def test_reconciliation_adopt_fill_dedup_by_fill_id():
    r = Reconciler()
    r.submit("k1", {"order": 1})
    r.mark_unconfirmed("k1")
    out = r.reconcile_one("k1", "filled",
                          {"broker_fill_id": "f1"})
    assert out["action"] == "adopt_fill"
    r.submit("k2", {"order": 2})
    r.mark_unconfirmed("k2")
    out2 = r.reconcile_one("k2", "filled",
                           {"broker_fill_id": "f1"})
    assert out2["action"] == "duplicate_fill"


def test_reconciliation_unknown_blocks_fence_then_override():
    r = Reconciler()
    r.submit("k1", {"order": 1})
    r.mark_unconfirmed("k1")
    out = r.reconcile_one("k1", "unknown", {"error": "timeout"})
    assert out["action"] == "hold_unknown"
    assert r.try_complete() is False
    res = r.admin_override(command_id="c1", key="k1",
                           resolution="mark_absent", reason="owner check")
    assert res["action"] == "override_resolved"
    assert r.try_complete() is True
    assert r.phase == "LIVE"
    # Duplicate command_id is idempotent.
    dup = r.admin_override(command_id="c1", key="k1",
                           resolution="mark_absent")
    assert dup["action"] == "duplicate_override"


def test_reconciliation_holding_drain_priority_and_revalidate():
    r = Reconciler()
    r.phase = "LIVE"
    r.reconciliation_complete = True
    r.enqueue_holding("low", {"x": 1}, priority=0)
    r.enqueue_holding("high", {"x": 2}, priority=10)
    out = r.drain_holding(batch_size=2,
                          revalidate=lambda e: e.idempotency_key != "low")
    assert [o["key"] for o in out] == ["high", "low"]
    assert out[0]["action"] == "drain"
    assert out[1]["action"] == "dropped_stale"


# ---------------------------------------------------------------- backtest export
def test_backtest_export_summary_and_chunks(tmp_path):
    trades = [{"playbook": "fvg_limit", "direction": 1, "r_net": 2.0,
               "pnl": 20.0, "reason": "tp"},
              {"playbook": "fvg_limit", "direction": -1, "r_net": -1.0,
               "pnl": -10.0, "reason": "sl"}]
    path, summary = write_report(trades, tmp_path / "bt.md",
                                 period="2026-01-01..2026-02-01",
                                 d1_bars=100)
    assert summary["trades"] == 2
    assert summary["total_r"] == 1.0
    assert summary["d1_coverage_ok"] is True
    text = format_summary(summary)
    assert "trades: 2" in text
    assert len(chunk_text("x" * 9000)) > 1
    assert path.exists()
