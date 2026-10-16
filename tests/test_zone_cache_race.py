"""Atomic cache transactions: one writer, versioned copy-on-write,
one outcome per zone, cluster propagation, idempotency, crash
safety.  A volatile candle must never cause a race, a
double-count, or a torn cache.
"""
import threading

import numpy as np
import pandas as pd
import pytest

from goldenswing.live.zones import (
    ZoneCache, ZoneStore,
)


# ------------------------------------------------------------- builders

def _frame(rows, *, start="2024-01-01", freq="D"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    arr = np.array(rows, dtype=float)
    return pd.DataFrame(
        {"open": arr[:, 0], "high": arr[:, 1], "low": arr[:, 2],
         "close": arr[:, 3], "volume": arr[:, 4]}, index=idx)


def _flat(n, *, close=4200.0, high=4210.0, low=4190.0, vol=1000.0):
    return [[close, high, low, close, vol] for _ in range(n)]


def _m15_bos_up():
    """M15, last close above the last confirmed swing high (BOS +1).

    All bars precede the D1/H1 bars they confirm (causal slice).
    """
    m = []
    for i in range(45):
        c = 4200.0 + i
        m.append([c, c + 5, c - 5, c, 1000.0])
    m.append([4310.0, 4320.0, 4300.0, 4310.0, 1000.0])  # swing high
    for c in (4300.0, 4295.0, 4300.0):
        m.append([c, c + 5, c - 5, c, 1000.0])
    for c in (4315.0, 4330.0, 4340.0, 4350.0, 4360.0):
        m.append([c, c + 5, c - 5, c, 1000.0])
    return _frame(m, freq="15min")


def _m15_bos_down():
    """M15, last close below the last confirmed swing low (BOS -1)."""
    m = []
    for i in range(45):
        c = 4300.0 - i
        m.append([c, c + 5, c - 5, c, 1000.0])
    m.append([4090.0, 4100.0, 4080.0, 4090.0, 1000.0])  # swing low
    for c in (4100.0, 4105.0, 4100.0):
        m.append([c, c + 5, c - 5, c, 1000.0])
    for c in (4090.0, 4080.0, 4070.0, 4060.0, 4050.0):
        m.append([c, c + 5, c - 5, c, 1000.0])
    return _frame(m, freq="15min")


def _d1_break_frame():
    """D1 resistance 4300, then two closes beyond (4340, 4360)."""
    rows = _flat(40)
    rows[10] = [4250.0, 4300.0, 4190.0, 4250.0, 1000.0]
    rows[20] = [4340.0, 4350.0, 4200.0, 4340.0, 1000.0]
    rows[21] = [4360.0, 4370.0, 4200.0, 4360.0, 1000.0]
    return _frame(rows, freq="D")


def _h1_mitigate_frame(*, mitigate_close=4095.0, mitigate_low=4090.0):
    """H1 support 4100, then a bar closing inside with penetration.

    Close 4095 (not 4100): penetration vs 0.5x ATR must clear the
    threshold with margin, not sit on it.
    """
    rows = _flat(40)
    rows[10] = [4150.0, 4210.0, 4100.0, 4150.0, 1000.0]
    rows[20] = [4200.0, 4210.0, mitigate_low, mitigate_close, 1000.0]
    return _frame(rows, freq="h")


def _run(frames, tick_id):
    """One full transaction: store.apply + cache swap (the runner
    pattern).  Returns (store, cache, outcomes)."""
    store = ZoneStore(":memory:")
    out = store.apply(frames, tick_id)
    cache = ZoneCache()
    cache.swap(store.active_zones(), out, tick_id, reason="bar close")
    return store, cache, out


# ------------------------------------------------------------- same-tick outcomes

def test_same_tick_mitigation_and_break_across_tfs():
    """One transaction, one snapshot: D1 zone BROKEN + H1 zone
    MITIGATED — both applied, exactly one outcome per zone."""
    store, cache, out = _run(
        {"D1": _d1_break_frame(), "H1": _h1_mitigate_frame(),
         "M15": _m15_bos_up()}, "T1")
    by_key = {}
    for o in out:
        assert o.zone_key not in by_key, \
            f"zone {o.zone_key} got two outcomes in one transaction"
        by_key[o.zone_key] = o.outcome
    d1_key = next(k for k in by_key if k[0] == "D1"
                  and store._mirror[k].level == pytest.approx(4300.0))
    h1_key = next(k for k in by_key if k[0] == "H1")
    assert by_key[d1_key] == "BROKEN"
    assert by_key[h1_key] == "MITIGATED"
    assert store._mirror[d1_key].status == "broken"
    assert store._mirror[h1_key].status == "mitigated"
    # Cache published the new version with both applied.
    assert cache.stats()["version"] == 1
    summary = cache.last_outcomes_summary()
    assert "D1 BROKEN 4300" in summary
    assert "H1 MITIGATED 4100" in summary
    store.close()


def test_one_outcome_ambiguous_wick_close_inside():
    """Wick beyond the far edge + BOS present + close INSIDE ->
    MITIGATED only (the close governs), reason notes both
    conditions, never double-counted."""
    # Mitigation bar wicks BELOW the far edge (low 4080 < 4087ish)
    # but closes inside with penetration.
    store, _cache, out = _run(
        {"H1": _h1_mitigate_frame(mitigate_close=4095.0,
                                 mitigate_low=4080.0),
         "M15": _m15_bos_down()}, "T1")
    z = next(z for z in store._mirror.values()
             if z.tf == "H1" and abs(z.level - 4100.0) < 1e-9)
    assert z.status == "mitigated"
    mine = [o for o in out if o.zone_key == z.key]
    assert len(mine) == 1 and mine[0].outcome == "MITIGATED"
    assert "close governs" in mine[0].reason
    store.close()


def test_cluster_broken_propagates_to_subzone():
    """D1 cluster zone BROKEN + H1 sub-zone inside its band that
    would have MITIGATED -> H1 is BROKEN ('cluster broken,
    sub-zones broken with parent'), one outcome, not mitigated."""
    # H1 resistance 4302: flat bars, then a bar closing inside
    # with penetration (high 4320 puts its swing above the D1
    # band, so no extra sub-zone lands inside the parent band).
    rows = _flat(40)
    rows[10] = [4252.0, 4302.0, 4190.0, 4252.0, 1000.0]
    rows[20] = [4305.0, 4320.0, 4200.0, 4305.0, 1000.0]
    h1 = _frame(rows, freq="h")
    store, _cache, out = _run(
        {"D1": _d1_break_frame(), "H1": h1, "M15": _m15_bos_up()}, "T1")
    h1_zone = next(z for z in store._mirror.values()
                   if z.tf == "H1" and abs(z.level - 4302.0) < 1e-9)
    assert h1_zone.status == "broken"
    mine = [o for o in out if o.zone_key == h1_zone.key]
    assert len(mine) == 1 and mine[0].outcome == "BROKEN"
    assert "cluster broken" in mine[0].reason
    store.close()


# ------------------------------------------------------------- torn reads / idempotency

def test_no_torn_read_concurrent_readers():
    """200 swaps under 4 reader threads: every read sees exactly
    one committed snapshot (old or new), never a mix."""
    cache = ZoneCache()
    committed = {0: ()}
    stop = threading.Event()
    errors: list[str] = []

    def reader():
        while not stop.is_set():
            v, res = cache.nearby(4000.0, 10.0, radius_atr=100000.0)
            got = tuple(z.level for z in res)
            want = committed.get(v)
            if want is not None and got != want:
                errors.append(f"version {v}: torn read {got} != {want}")

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for i in range(200):
            zones = []
            for j in range(50):
                from goldenswing.live.zones import StructuralZone
                zones.append(StructuralZone(
                    tf="H1", side="resistance",
                    level=4000.0 + j * 10.0 + (i % 3),
                    top=4010.0, bottom=3990.0, atr=10.0,
                    score=50.0, tier="MINOR", status="active",
                    swing_ts=f"2024-01-01T00:{j:02d}:00+00:00"))
            v = cache.swap(zones, [], f"t{i}", reason="bar close")
            committed[v] = tuple(sorted(
                z.level for z in zones if z.status == "active"))
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert not errors, errors[:3]


def test_replay_same_tick_idempotent():
    """Same tick replayed -> nothing re-applied; a different tick
    id with no new bars -> nothing re-applied; touches counted
    exactly once."""
    rows = _flat(40)
    rows[10] = [4250.0, 4300.0, 4190.0, 4250.0, 1000.0]
    rows[20] = [4200.0, 4305.0, 4190.0, 4200.0, 1000.0]  # touch bar
    h1 = _frame(rows, freq="h")
    store = ZoneStore(":memory:")
    out1 = store.apply({"H1": h1}, "T1")
    z = next(z for z in store._mirror.values() if z.tf == "H1")
    assert any(o.outcome == "TOUCHED" for o in out1)
    assert store._mirror[z.key].touches == 1
    assert store.apply({"H1": h1}, "T1") == []  # same tick id
    assert store._mirror[z.key].touches == 1
    assert store.apply({"H1": h1}, "T2") == []  # no new bars
    assert store._mirror[z.key].touches == 1
    store.close()


def test_crash_mid_transaction_keeps_old_cache():
    """Compute crash -> nothing persisted, mirror untouched, cache
    version unchanged.  Persist crash -> same.  A clean re-apply
    afterwards applies exactly once."""
    rows = _flat(40)
    rows[10] = [4250.0, 4300.0, 4190.0, 4250.0, 1000.0]
    rows[20] = [4200.0, 4305.0, 4190.0, 4200.0, 1000.0]  # touch bar
    h1 = _frame(rows, freq="h")
    store = ZoneStore(":memory:")
    cache = ZoneCache()
    store.seed({"H1": h1[:20]})  # seed WITHOUT the touch bar
    cache.swap(store.active_zones(), [], "seed", reason="bar close")
    assert cache.stats()["version"] == 1
    z = next(z for z in store._mirror.values() if z.tf == "H1")
    assert store._mirror[z.key].touches == 0

    real_compute = store.compute
    store.compute = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("simulated crash in outcome computation"))
    with pytest.raises(RuntimeError):
        store.apply({"H1": h1}, "T-crash")
    store.compute = real_compute
    assert store._mirror[z.key].touches == 0  # mirror untouched
    assert cache.stats()["version"] == 1  # cache unswapped

    real_persist = store.persist
    store.persist = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("simulated crash in persist"))
    with pytest.raises(RuntimeError):
        store.apply({"H1": h1}, "T-crash2")
    store.persist = real_persist
    assert store._mirror[z.key].touches == 0
    assert cache.stats()["version"] == 1

    out = store.apply({"H1": h1}, "T-ok")
    assert any(o.outcome == "TOUCHED" for o in out)
    assert store._mirror[z.key].touches == 1  # exactly once
    store.close()


def test_runner_update_zones_never_raises_on_crash(tmp_path):
    """Runner-level guarantee: a zone compute crash inside
    _update_zones is logged, never raised, old cache kept."""
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    from goldenswing.live.hierarchy import TimeframeStack
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("stub",), news_pause_enabled=False)
    runner = MT5Runner(cfg, feed=None, executor=None)
    rows = _flat(40)
    rows[10] = [4250.0, 4300.0, 4190.0, 4250.0, 1000.0]
    h1 = _frame(rows, freq="h")
    runner._stack = TimeframeStack.from_frames(
        {"H1": h1, "D1": h1, "H4": h1, "M15": h1, "M5": h1,
         "W1": h1, "MN1": h1})
    runner._update_zones()  # first refresh seeds
    v0 = runner._zone_cache.stats()["version"]
    assert v0 == 1
    store = runner._get_zone_store()
    real_compute = store.compute
    store.compute = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("simulated crash"))
    try:
        runner._update_zones()  # must NOT raise
    finally:
        store.compute = real_compute
    assert runner._zone_cache.stats()["version"] == v0
