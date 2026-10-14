"""Divergence endpoint: auth, validation, idempotency, CAS, claims, broker.

All state lives under tmp_path — never the real data/ journal.
"""
import fcntl
import json
import os
import threading

import pytest

from goldenswing.live.divergence_api import (
    BrokerUnavailable,
    IdempotencyKeyReused,
    NoActiveDivergence,
    PendingOperation,
    StaleDivergence,
    handle_resolve_request,
    resolve_divergence_provenance,
)
from goldenswing.live.journal_lock import append_journal_record
from goldenswing.live.monitor import JOURNAL_RECORD_VERSION
from goldenswing.live.reconcile import DIVERGENCE

TOKEN = "dashboard-token-123"
ACTOR = "operator-1"


def _verify(token):
    return ACTOR if token == TOKEN else None


def _headers(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def _body(**over):
    body = {"trade_id": "t-0007", "divergence_seq": 1,
            "reason": "closed stray broker position manually",
            "idempotency_key": "key-1"}
    body.update(over)
    return body


def _read(jp):
    with open(jp) as f:
        return [json.loads(line) for line in f if line.strip()]


def _seed_divergence(jp, trade_id="t-0007"):
    """Append a needs_review divergence; return its journal seq."""
    return append_journal_record(jp, {
        "v": JOURNAL_RECORD_VERSION, "type": "reconciliation",
        "event": DIVERGENCE, "trade_id": trade_id, "needs_review": True,
        "note": "journal closed but broker shows open"})


def _call(jp, claims, body=None, headers=None, broker=None):
    return handle_resolve_request(
        headers if headers is not None else _headers(),
        body if body is not None else _body(),
        journal_path=jp,
        fetch_broker_state=(lambda: broker) if broker is not None else (lambda: {"open": []}),
        actor_from_token=_verify,
        claims_dir=claims)


def test_unauth_401_and_no_journal_work(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)
    before = len(_read(jp))
    status, resp = _call(jp, claims, headers={})
    assert status == 401
    status, resp = _call(jp, claims, headers=_headers("wrong-token"))
    assert status == 401
    status, resp = _call(jp, claims, headers={"Authorization": "Token " + TOKEN})
    assert status == 401
    assert len(_read(jp)) == before  # no journal work on auth failure


def test_missing_field_400(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)
    body = _body()
    del body["reason"]
    status, _ = _call(jp, claims, body=body)
    assert status == 400


def test_extra_field_400(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)
    status, _ = _call(jp, claims, body=_body(surprise="nope"))
    assert status == 400


def test_stale_seq_409(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)  # active divergence is seq 1
    status, resp = _call(jp, claims, body=_body(divergence_seq=99))
    assert status == 409
    assert "stale" in resp["error"].lower()


def test_no_active_divergence_404(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    status, resp = _call(jp, claims, body=_body())
    assert status == 404


def test_duplicate_same_key_same_payload_no_new_append(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)
    status1, r1 = _call(jp, claims)
    assert status1 == 200
    assert r1["status"] == "resolved"
    n_after_first = len(_read(jp))
    status2, r2 = _call(jp, claims)
    assert status2 == 200
    assert r2["status"] == "duplicate"
    assert r2["journal_seq"] == r1["journal_seq"]
    assert len(_read(jp)) == n_after_first  # NO new append


def test_same_key_diff_payload_409(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)
    status1, _ = _call(jp, claims)
    assert status1 == 200
    status2, resp = _call(jp, claims, body=_body(reason="different reason"))
    assert status2 == 409
    assert "idempotency" in resp["error"].lower()


def test_broker_unavailable_503_no_clearance_claim_removed_key_retryable(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)

    def _boom():
        raise BrokerUnavailable("broker feed down")

    status, resp = handle_resolve_request(
        _headers(), _body(), journal_path=jp, fetch_broker_state=_boom,
        actor_from_token=_verify, claims_dir=claims)
    assert status == 503
    recs = _read(jp)
    assert not any(r.get("event") == "divergence_cleared" for r in recs)
    # Claim removed (no leftover claim file blocks the retry).
    assert not [f for f in os.listdir(claims) if f.endswith(".json")]
    # Key remains retryable: same key + same payload now succeeds.
    status2, r2 = _call(jp, claims)
    assert status2 == 200
    assert r2["status"] == "resolved"


def test_concurrent_threads_same_key_exactly_one_append(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)
    results = []
    errors = []

    def worker():
        try:
            status, resp = _call(jp, claims)
            results.append((status, resp))
        except Exception as exc:  # noqa: BLE001 — surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors
    assert len(results) == 10
    assert all(s == 200 for s, _ in results)
    seqs = {r["journal_seq"] for _, r in results}
    assert seqs == {2}  # all return the SAME journal_seq
    statuses = sorted(r["status"] for _, r in results)
    assert statuses.count("resolved") == 1
    assert statuses.count("duplicate") == 9
    recs = _read(jp)
    assert sum(1 for r in recs if r.get("event") == "divergence_cleared") == 1


def test_pending_claim_from_dead_holder_is_reclaimable(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)
    os.makedirs(claims, exist_ok=True)
    # Dead holder: claim file exists, but NOBODY holds the flock.
    with open(os.path.join(claims, "key-1.json"), "w") as f:
        json.dump({"owner_token": "dead-beef", "pid": 999999,
                   "fingerprint": "x",
                   "created_at": "2026-10-13T00:00:00Z"}, f)
    status, resp = _call(jp, claims)
    assert status == 200
    assert resp["status"] == "resolved"


def test_pending_claim_held_by_live_holder_returns_pending(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)
    os.makedirs(claims, exist_ok=True)
    # Live holder: holds the flock on the lock file.
    lock_path = os.path.join(claims, "key-1.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    with open(os.path.join(claims, "key-1.json"), "w") as f:
        json.dump({"owner_token": "live-holder", "pid": os.getpid(),
                   "fingerprint": "x",
                   "created_at": "2026-10-13T00:00:00Z"}, f)
    try:
        with pytest.raises(PendingOperation):
            resolve_divergence_provenance(
                jp, trade_id="t-0007", divergence_seq=1,
                reason="closed stray broker position manually",
                idempotency_key="key-1", actor=ACTOR,
                fetch_broker_state=lambda: {"open": []},
                claims_dir=claims, pending_timeout_s=0.3)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    # Live holder's claim was never executed: no clearance appended.
    assert not any(r.get("event") == "divergence_cleared" for r in _read(jp))


def test_core_rejects_empty_fields_and_legacy_key_collision(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)
    with pytest.raises(ValueError, match="reason"):
        resolve_divergence_provenance(
            jp, trade_id="t-0007", divergence_seq=1, reason="  ",
            idempotency_key="k", actor=ACTOR, claims_dir=claims)
    # Legacy clearance (old resolve_divergence, no fingerprint) under the
    # same key cannot be proven identical -> key reuse refused.
    append_journal_record(jp, {
        "v": JOURNAL_RECORD_VERSION, "type": "review",
        "event": "divergence_cleared", "trade_id": "t-0007",
        "idempotency_key": "legacy-key", "cleared_by": "human",
        "note": "old-style clearance"})
    with pytest.raises(IdempotencyKeyReused):
        resolve_divergence_provenance(
            jp, trade_id="t-0007", divergence_seq=1, reason="r",
            idempotency_key="legacy-key", actor=ACTOR, claims_dir=claims)


def test_second_divergence_after_first_cleared_needs_new_seq(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    claims = str(tmp_path / "claims")
    _seed_divergence(jp)  # seq 1
    status, _ = _call(jp, claims)
    assert status == 200
    # Old seq now has no active divergence.
    status, _ = _call(jp, claims, body=_body(idempotency_key="key-2"))
    assert status == 404
    # A fresh divergence gets a fresh seq and resolves under a new key.
    seq2 = _seed_divergence(jp)
    status, r = _call(jp, claims, body=_body(divergence_seq=seq2,
                                             idempotency_key="key-3"))
    assert status == 200
    assert r["journal_seq"] == seq2 + 1
