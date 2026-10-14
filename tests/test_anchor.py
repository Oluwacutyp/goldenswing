"""Anchor v1: merkle tree, epoch chain, threshold sigs, chain verification."""
import json
import os

import pytest

from goldenswing.live.anchor import (
    AnchorPublisher,
    AnchorStore,
    ThresholdSigner,
    chain_epoch_root,
    leaf_hash,
    leaves_from_entries,
    merkle_proof,
    merkle_root,
    verify_proof,
    GENESIS_EPOCH_ROOT,
)


def _leaves(n, tag=""):
    return [leaf_hash(i + 1, f"head-{tag}-{i}", f"2026-10-13T00:00:0{i}Z")
            for i in range(n)]


def test_merkle_root_recomputation_is_deterministic():
    leaves = _leaves(5, "a")
    assert merkle_root(leaves) == merkle_root(leaves)
    assert merkle_root(leaves) != merkle_root(_leaves(5, "b"))
    # Odd count: duplicate-last rule — 3 leaves must equal 4 with last duped.
    three = _leaves(3, "c")
    assert merkle_root(three) == merkle_root(three + [three[-1]])


def test_inclusion_proof_valid_and_tampered_fails():
    leaves = _leaves(7, "p")
    root = merkle_root(leaves)
    for idx in range(len(leaves)):
        proof = merkle_proof(leaves, idx)
        assert verify_proof(leaves[idx], proof, root) is True
        # Tampered leaf fails.
        assert verify_proof(leaf_hash(999, "x", "y"), proof, root) is False
        # Tampered root fails.
        assert verify_proof(leaves[idx], proof, "00" * 32) is False
        # Tampered proof (replace first sibling hash) fails.
        bad = [(leaf_hash(999, "x", "y"), flag) if j == 0 else (s, flag)
               for j, (s, flag) in enumerate(proof)]
        assert verify_proof(leaves[idx], bad, root) is False
        # Tampered proof (flip the first flag) fails — unless the first
        # sibling IS the duplicated odd node (sibling == current), where
        # flipping is a genuine no-op: same hash either way.
        if proof and proof[0][0] != leaves[idx]:
            bad2 = [(s, (not f) if k == 0 else f)
                    for k, (s, f) in enumerate(proof)]
            assert verify_proof(leaves[idx], bad2, root) is False


def test_epoch_chain_valid_and_tampered_prev_breaks(tmp_path):
    store = AnchorStore(str(tmp_path / "anchors.jsonl"))
    e1 = store.append_epoch(_leaves(3, "e1"))
    e2 = store.append_epoch(_leaves(4, "e2"))
    assert e1["anchor_seq"] == 1
    assert e2["anchor_seq"] == 2
    assert e2["prev_epoch_root"] == e1["epoch_root"]
    assert e1["prev_epoch_root"] == GENESIS_EPOCH_ROOT
    assert e1["epoch_root"] == chain_epoch_root(GENESIS_EPOCH_ROOT, e1["merkle_root"])
    assert e2["epoch_root"] == chain_epoch_root(e1["epoch_root"], e2["merkle_root"])
    ok, reason = store.verify_chain()
    assert ok, reason

    # Tamper with epoch 2's prev link (rewrite file by hand — the test
    # simulating an attacker; the store itself never rewrites).
    lines = open(store.log_path).read().splitlines()
    rec = json.loads(lines[1])
    rec["prev_epoch_root"] = "ff" * 32
    lines[1] = json.dumps(rec)
    open(store.log_path, "w").write("\n".join(lines) + "\n")
    ok, reason = store.verify_chain()
    assert ok is False
    assert "prev_epoch_root" in reason


def test_threshold_sig_requires_m_and_one_sig_fails():
    signer = ThresholdSigner()
    msg = b"epoch-1"
    full = signer.sign(msg)
    assert len(full) == 2  # default: exactly M signers
    assert signer.verify(msg, full) is True
    one = {k: v for k, v in list(full.items())[:1]}
    assert signer.verify(msg, one) is False
    # Unknown signer never counts toward the threshold.
    forged = dict(full)
    forged["mallory"] = full[list(full)[0]]
    assert signer.verify(msg, forged) is True  # still M real ones
    assert signer.verify(msg, {"mallory": full[list(full)[0]]}) is False
    # Wrong message fails.
    assert signer.verify(b"epoch-2", full) is False
    # Sub-threshold sign() refuses to produce a signature set.
    with pytest.raises(ValueError):
        signer.sign(msg, signer_ids=["signer-1"])


def test_verify_chain_catches_tampered_anchor_log(tmp_path):
    store = AnchorStore(str(tmp_path / "anchors.jsonl"))
    store.append_epoch(_leaves(2, "t1"))
    store.append_epoch(_leaves(2, "t2"))
    ok, _ = store.verify_chain()
    assert ok

    # Tamper 1: a leaf inside epoch 1.
    lines = open(store.log_path).read().splitlines()
    rec = json.loads(lines[0])
    rec["leaves"][0] = leaf_hash(1, "forged", "2026-10-13T00:00:00Z")
    lines[0] = json.dumps(rec)
    open(store.log_path, "w").write("\n".join(lines) + "\n")
    ok, reason = store.verify_chain()
    assert ok is False
    assert "merkle_root" in reason

    # Tamper 2 (fresh log): drop a signature below threshold.
    store2 = AnchorStore(str(tmp_path / "anchors2.jsonl"))
    store2.append_epoch(_leaves(2, "s1"))
    lines = open(store2.log_path).read().splitlines()
    rec = json.loads(lines[0])
    keep = next(iter(rec["signatures"]))
    rec["signatures"] = {keep: rec["signatures"][keep]}
    lines[0] = json.dumps(rec)
    open(store2.log_path, "w").write("\n".join(lines) + "\n")
    ok, reason = store2.verify_chain()
    assert ok is False
    assert "signature" in reason


def test_append_epoch_accepts_entry_dicts_and_publisher_fans_out(tmp_path):
    entries = [{"anchor_seq": 1, "chain_head_hash": "abc",
                "timestamp": "2026-10-13T00:00:00Z"}]
    store = AnchorStore(str(tmp_path / "anchors.jsonl"))
    rec = store.append_epoch(entries)
    assert rec["leaves"] == leaves_from_entries(entries)
    assert rec["leaf_count"] == 1

    pub = AnchorPublisher(str(tmp_path / "anchors"))
    path = pub.publish(rec)
    assert path == os.path.join(str(tmp_path / "anchors"), "local", "epochs.jsonl")
    assert json.loads(open(path).readline())["anchor_seq"] == 1
    with pytest.raises(ValueError):
        pub.publish_to_provider("../escape", rec)
