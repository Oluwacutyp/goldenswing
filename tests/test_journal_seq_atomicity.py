"""Seq assignment atomicity: concurrent appends get unique, gapless seqs.

Regression test for the TOCTOU race where _journal_length() was derived
outside journal_transaction: two writers could read the same line count and
emit colliding sequence numbers — and, worse, colliding seq-derived trade_ids
in append_trade_to_journal. The fix centralizes seq derivation inside the
lock (append_journal_record); this test fails deterministically against the
old pattern and must pass against the new one.
"""
import json
import threading

from goldenswing.live.journal_lock import append_journal_record
from goldenswing.live.monitor import append_trade_to_journal


def _read_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_concurrent_appends_get_unique_gapless_seqs(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    n = 25
    seen = []
    guard = threading.Lock()

    def writer(i):
        s = append_journal_record(jp, {"v": 2, "type": "probe", "i": i})
        with guard:
            seen.append(s)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert sorted(seen) == list(range(1, n + 1))
    records = _read_records(jp)
    assert len(records) == n
    # Each holder appends exactly one line before releasing, so file order
    # matches acquisition order and every seq appears exactly once.
    assert [r["seq"] for r in records] == list(range(1, n + 1))


def test_concurrent_trade_appends_get_unique_trade_ids(tmp_path):
    jp = str(tmp_path / "journal.jsonl")
    n = 15
    guard = threading.Lock()
    returned = []

    def writer():
        # trade_id=None exercises the seq-derived default inside the lock.
        s = append_trade_to_journal(jp, r_multiple=0.5, direction=1)
        with guard:
            returned.append(s)

    threads = [threading.Thread(target=writer) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    records = _read_records(jp)
    assert len(records) == n
    assert sorted(r["seq"] for r in records) == list(range(1, n + 1))
    trade_ids = [r["trade_id"] for r in records]
    assert len(set(trade_ids)) == n, f"colliding trade_ids: {trade_ids}"
    assert set(trade_ids) == {f"t-{s}" for s in range(1, n + 1)}
    assert sorted(returned) == list(range(1, n + 1))
