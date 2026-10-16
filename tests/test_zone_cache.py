"""In-memory zone cache: sub-millisecond nearby queries, no SQLite
on the tick path, versioned copy-on-write snapshots, R-tree
rebuild path, VACUUM/ANALYZE on large DBs.

SQLite is the DURABLE store, NOT the execution path.  The tick
loop bisects the cache (O(log n + k)); the cache holds no DB
handle at all.
"""
import numpy as np
import pandas as pd
import pytest

from goldenswing.live.zones import (
    ZoneCache, ZoneStore, StructuralZone,
)


def _bulk_zones(n, *, level0=3000.0, step=0.2):
    zones = []
    tfs = ["MN1", "W1", "D1", "H4", "H1", "M15", "M5"]
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(n):
        level = level0 + i * step
        major = (i % 2 == 0)
        zones.append(StructuralZone(
            tf=tfs[i % len(tfs)],
            side="resistance" if i % 3 else "support",
            level=level, top=level + 5.0, bottom=level - 5.0,
            score=80.0 if major else 50.0,
            tier="MAJOR" if major else "MINOR",
            status="active", touches=1, displacement_atr=2.0,
            atr=10.0,
            swing_ts=(base + pd.Timedelta(minutes=i)).isoformat()))
    return zones


def test_benchmark_10k_zones_10k_ticks_sub_ms_p95(capsys):
    """10k zones in the DB, 10k ticks, each nearby query < 1ms p95,
    ZERO SQLite queries during ticks (the store connection is
    CLOSED before the tick loop — any DB access would raise)."""
    store = ZoneStore(":memory:")
    store.bulk_insert(_bulk_zones(10000))
    cache = ZoneCache()
    cache.swap(store.active_zones(), [], "bench", reason="bar close")
    assert cache.stats()["n_zones"] == 10000
    store.close()  # tick path must not touch SQLite — prove it
    assert not hasattr(cache, "_conn")  # cache holds no DB handle
    rng = np.random.default_rng(42)
    prices = 3000.0 + rng.random(10000) * 2000.0
    for price in prices:
        _v, res = cache.nearby(float(price), 10.0)
        assert isinstance(res, list)
    stats = cache.stats()
    print(f"\nBENCHMARK 10k zones x 10k ticks: "
          f"p50={stats['p50_ms']:.4f}ms "
          f"p95={stats['p95_ms']:.4f}ms "
          f"p99={stats['p99_ms']:.4f}ms "
          f"queries={stats['queries']}")
    assert stats["queries"] == 10000
    assert stats["p95_ms"] < 1.0, \
        f"nearby p95 {stats['p95_ms']:.3f}ms must be sub-millisecond"
    assert stats["p99_ms"] < 1.0


def test_cache_refresh_only_on_bar_close():
    """Nearby calls never refresh the snapshot (version unchanged);
    only a swap (bar close) publishes a new version."""
    cache = ZoneCache()
    z1 = StructuralZone(tf="D1", side="resistance", level=4300.0,
                        top=4310.0, bottom=4290.0, atr=20.0,
                        score=80.0, tier="MAJOR", status="active")
    v1 = cache.swap([z1], [], "t1", reason="bar close")
    assert v1 == 1
    for _ in range(100):
        v, _res = cache.nearby(4300.0, 20.0)
        assert v == 1
    z2 = StructuralZone(tf="H1", side="support", level=4100.0,
                        top=4110.0, bottom=4090.0, atr=20.0,
                        score=50.0, tier="MINOR", status="active")
    v2 = cache.swap([z1, z2], [], "t2", reason="bar close")
    assert v2 == 2
    _v, res = cache.nearby(4100.0, 20.0)
    assert any(z.level == 4100.0 for z in res)


def test_precompute_major_within_5atr():
    """major_nearby serves the precomputed MAJOR list (built once
    per swap, bisected per call): MINOR zones never appear, MAJOR
    zones outside 5 ATR never appear."""
    cache = ZoneCache()
    zones = [
        StructuralZone(tf="D1", side="resistance", level=4200.0,
                       top=4210.0, bottom=4190.0, atr=10.0,
                       score=80.0, tier="MAJOR", status="active"),
        StructuralZone(tf="H1", side="support", level=4205.0,
                       top=4215.0, bottom=4195.0, atr=10.0,
                       score=50.0, tier="MINOR", status="active"),
        StructuralZone(tf="W1", side="resistance", level=4300.0,
                       top=4310.0, bottom=4290.0, atr=10.0,
                       score=90.0, tier="MAJOR", status="active"),
    ]
    cache.swap(zones, [], "t1", reason="bar close")
    _v, majors = cache.major_nearby(4200.0, 10.0, radius_atr=5.0)
    assert [z.level for z in majors] == [4200.0]
    _v, near = cache.nearby(4200.0, 10.0, radius_atr=2.0)
    assert sorted(z.level for z in near) == [4200.0, 4205.0]


def test_rtree_rebuild_matches_bisect_and_fallback():
    """load_active_via_rtree (rebuild path) returns the same set
    as the cache bisect — via R-tree AND via the B-tree fallback."""
    store = ZoneStore(":memory:")
    store.bulk_insert(_bulk_zones(200))
    cache = ZoneCache()
    cache.swap(store.active_zones(), [], "t1", reason="bar close")
    _v, bisected = cache.nearby(3500.0, 10.0)
    rtree = store.load_active_via_rtree(3500.0, 10.0)
    assert {z.key for z in rtree} == {z.key for z in bisected}
    store.rtree_available = False  # force the B-tree fallback
    fallback = store.load_active_via_rtree(3500.0, 10.0)
    assert {z.key for z in fallback} == {z.key for z in bisected}
    store.close()


def test_vacuum_on_large_db_reopen(tmp_path):
    """Opening a DB with >10k zones runs VACUUM + ANALYZE and
    preserves every zone."""
    path = str(tmp_path / "zones.db")
    store = ZoneStore(path)
    assert store.bulk_insert(_bulk_zones(10001)) == 10001
    store.close()
    store2 = ZoneStore(path)
    assert store2.vacuumed is True
    assert len(store2._mirror) == 10001
    assert len(store2.active_zones()) == 10001
    store2.close()
