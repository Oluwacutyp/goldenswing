from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from goldenswing.live.setup_store import SetupStore

NOW = "2026-10-27T00:00:00+00:00"
SETUP = {
    "setup_id": "hs-test-unique-setup",
    "symbol": "XAUUSD",
    "strategy": "breakout_retest",
    "direction": 1,
    "formation_ts": NOW,
    "armed_ts": NOW,
    "expires_at": "2026-10-27T01:00:00+00:00",
    "level": 4000.0,
    "stop": 3995.0,
}


def test_same_setup_id_survives_new_store_instance(tmp_path):
    db = tmp_path / "shared" / "setups.sqlite"
    a, b = SetupStore(db), SetupStore(db)
    assert a.arm(SETUP)["inserted"] is True
    assert b.arm({**SETUP, "description": "re-scanned"})["inserted"] is False
    rows = a.active()
    assert len(rows) == 1
    assert rows[0]["setup_id"] == SETUP["setup_id"]


def test_atomic_trigger_claim_allows_only_one_process(tmp_path):
    db = tmp_path / "setups.sqlite"
    stores = [SetupStore(db), SetupStore(db)]
    stores[0].arm(SETUP)
    assert stores[0].mark_touched(SETUP["setup_id"], NOW)
    barrier = Barrier(2)

    def claim(i):
        barrier.wait()
        return stores[i].claim_trigger(SETUP["setup_id"], f"process-{i}", NOW)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, range(2)))
    claimed = [r for r in results if r is not None]
    assert len(claimed) == 1
    assert claimed[0]["fencing_token"] == 1


def test_claim_is_not_released_until_authoritative_reconciliation(tmp_path):
    store = SetupStore(tmp_path / "setups.sqlite")
    store.arm(SETUP)
    store.mark_touched(SETUP["setup_id"], NOW)
    claim = store.claim_trigger(SETUP["setup_id"], "old-process", NOW)
    assert claim is not None
    # Restarted process cannot claim a second time while the old claim is ambiguous.
    assert store.claim_trigger(SETUP["setup_id"], "new-process", NOW) is None
    with pytest.raises(ValueError):
        store.resolve_claim_after_broker_reconciliation(
            setup_id=SETUP["setup_id"], fencing_token=claim["fencing_token"],
            broker_order_exists=None, now=NOW)
    assert store.resolve_claim_after_broker_reconciliation(
        setup_id=SETUP["setup_id"], fencing_token=claim["fencing_token"],
        broker_order_exists=False, now=NOW)
    second = store.claim_trigger(SETUP["setup_id"], "new-process", NOW)
    assert second and second["fencing_token"] == 2
    assert store.resolve_claim_after_broker_reconciliation(
        setup_id=SETUP["setup_id"], fencing_token=second["fencing_token"],
        broker_order_exists=True, now=NOW)
    assert store.active() == []


def test_event_id_is_idempotent(tmp_path):
    store = SetupStore(tmp_path / "setups.sqlite")
    store.arm(SETUP)
    args = dict(event_id="armed:test", setup_id=SETUP["setup_id"],
                event_type="setup_armed", now=NOW, payload={"ok": True})
    assert store.write_event_once(**args) is True
    assert store.write_event_once(**args) is False
