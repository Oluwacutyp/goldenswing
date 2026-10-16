"""Persistent zone store: 3+ year horizon, decay vs permanence,
cluster/conflict, incremental updates, scoring.

The engine was STARVED of data (20 D1 bars, no multi-year zones,
no persistent zone memory) — bias/session were symptoms, not the
cause.  These tests pin the cure: zones cataloged once from 3+
years of bars, major HTF zones never expiring on age alone,
minors fading, clusters combining, conflicts waiting for
immediate confirm — and zones NEVER triggering a trade alone.
"""
import numpy as np
import pandas as pd
import pytest

from goldenswing.live.zones import (
    ZoneStore, find_clusters, score_zone, StructuralZone,
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


def _d1_with_swings(*, n=1000, swing_high=None, swing_low=None):
    """Flat D1 + engineered swings.

    swing_high = (idx, price), swing_low = (idx, price).  Flat
    bars have equal highs/lows, so find_swings finds ONLY the
    engineered swings (strict-greater / strict-less).
    """
    rows = _flat(n)
    if swing_high is not None:
        i, p = swing_high
        rows[i] = [p - 50.0, p, 4190.0, p - 50.0, 1000.0]
    if swing_low is not None:
        i, p = swing_low
        rows[i] = [p + 50.0, 4210.0, p, p + 50.0, 1000.0]
    return _frame(rows, freq="D")


# ------------------------------------------------------------- 3-year horizon

def test_3year_zone_persists_major_never_expires_on_age(tmp_path):
    """A D1 zone from ~2.5 years ago is still active, MAJOR,
    unmitigated — D1/W1/MN1 NEVER expire on time alone."""
    d1 = _d1_with_swings(n=1000, swing_high=(100, 4300.0),
                         swing_low=(300, 4100.0))
    # W1: 200 bars (~4y), one old swing high, untested.
    w1_rows = _flat(200)
    w1_rows[50] = [4350.0, 4400.0, 4190.0, 4350.0, 1000.0]
    w1 = _frame(w1_rows, freq="W")
    store = ZoneStore(str(tmp_path / "zones.db"))
    store.seed({"D1": d1, "W1": w1})
    act = {z.key: z for z in store.active_zones()}
    d1_hi = next(z for z in act.values()
                 if z.tf == "D1" and z.side == "resistance")
    d1_lo = next(z for z in act.values()
                 if z.tf == "D1" and z.side == "support")
    w1_hi = next(z for z in act.values() if z.tf == "W1")
    assert d1_hi.level == pytest.approx(4300.0)
    assert d1_hi.tier == "MAJOR" and d1_hi.status == "active"
    assert d1_hi.age_d1 > 800  # ~2.5y old, still active
    assert d1_lo.level == pytest.approx(4100.0)
    assert d1_lo.tier == "MAJOR"
    assert w1_hi.level == pytest.approx(4400.0)
    assert w1_hi.status == "active"  # W1 never expires on age
    store.close()


def test_minor_expires_after_20_d1_bars_untested(tmp_path):
    """An H1 MINOR zone (confluence-boosted to active) with zero
    touches expires after 20 D1 bars untested.  The D1 zone at
    the same level does NOT expire."""
    # D1 swing at 4300 (gives the H1 zone confluence).
    d1 = _d1_with_swings(n=60, swing_high=(10, 4300.0))
    # H1: swing at 4300 (idx 10), then flat bars — untested, but
    # only ~12 days old at seed (under the 20-day expiry).
    h1_rows = _flat(300, close=4200.0)
    h1_rows[10] = [4250.0, 4300.0, 4190.0, 4250.0, 1000.0]
    h1 = _frame(h1_rows, freq="h", start="2024-01-01")
    store = ZoneStore(str(tmp_path / "zones.db"))
    store.seed({"D1": d1, "H1": h1})
    h1_zone = next(z for z in store.active_zones() if z.tf == "H1")
    assert h1_zone.tier == "MINOR"  # confluence lifted it to active
    assert h1_zone.touches == 0
    # Extend H1 by 30 more days of flat bars (age >> 20 D1 bars).
    more = _flat(720, close=4200.0)
    h1b = pd.concat([h1, _frame(
        more, freq="h", start=h1.index[-1] + pd.Timedelta(hours=1))])
    out = store.apply({"H1": h1b}, str(h1b.index[-1]))
    expired = [o for o in out if o.outcome == "EXPIRED"
               and o.zone_key == h1_zone.key]
    assert expired, "untested H1 minor must expire after 20 D1 bars"
    assert h1_zone.key not in {z.key for z in store.active_zones()}
    # The D1 zone at the same level is untouched by expiry.
    assert any(z.tf == "D1" and z.status == "active"
               for z in store._mirror.values())
    store.close()


# ------------------------------------------------------------- lifecycle

def test_mitigated_drops(tmp_path):
    """Close INSIDE the zone + penetration >= 0.5 ATR -> MITIGATED,
    dropped from the active list."""
    rows = _flat(60)
    rows[10] = [4150.0, 4210.0, 4100.0, 4150.0, 1000.0]  # swing low
    rows[20] = [4200.0, 4210.0, 4090.0, 4095.0, 1000.0]  # deep inside
    h1 = _frame(rows, freq="h")
    store = ZoneStore(str(tmp_path / "zones.db"))
    store.seed({"H1": h1})
    z = next(z for z in store._mirror.values()
             if z.tf == "H1" and z.side == "support"
             and abs(z.level - 4100.0) < 1e-9)
    assert z.status == "mitigated"
    assert z.key not in {x.key for x in store.active_zones()}
    store.close()


def test_broken_drops_with_m15_bos(tmp_path):
    """2 closes beyond the far edge + M15 BOS -> BROKEN, dropped."""
    # D1: swing high 4300, then two closes beyond (4340, 4360).
    rows = _flat(40)
    rows[10] = [4250.0, 4300.0, 4190.0, 4250.0, 1000.0]
    rows[20] = [4340.0, 4350.0, 4200.0, 4340.0, 1000.0]
    rows[21] = [4360.0, 4370.0, 4200.0, 4360.0, 1000.0]
    d1 = _frame(rows, freq="D")
    # M15: rising closes, last close above last confirmed swing high.
    m = []
    for i in range(45):
        c = 4200.0 + i
        m.append([c, c + 5, c - 5, c, 1000.0])
    m.append([4310.0, 4320.0, 4300.0, 4310.0, 1000.0])  # swing high
    for c in (4300.0, 4295.0, 4300.0):
        m.append([c, c + 5, c - 5, c, 1000.0])
    for c in (4315.0, 4330.0, 4340.0, 4350.0, 4360.0):
        m.append([c, c + 5, c - 5, c, 1000.0])
    m15 = _frame(m, freq="15min")
    store = ZoneStore(str(tmp_path / "zones.db"))
    out = store.seed({"D1": d1, "M15": m15})
    z = next(z for z in store._mirror.values()
             if z.tf == "D1" and abs(z.level - 4300.0) < 1e-9)
    assert z.status == "broken"
    assert any(o.outcome == "BROKEN" and o.zone_key == z.key
               for o in out)
    assert z.key not in {x.key for x in store.active_zones()}
    store.close()


def test_unmitigated_persists_on_wick_only(tmp_path):
    """Wick into the zone + close away = respected touch, zone
    stays active (unmitigated)."""
    rows = _flat(60)
    rows[10] = [4250.0, 4300.0, 4190.0, 4250.0, 1000.0]  # swing high
    rows[20] = [4200.0, 4305.0, 4190.0, 4200.0, 1000.0]  # wick, close away
    h1 = _frame(rows, freq="h")
    store = ZoneStore(str(tmp_path / "zones.db"))
    store.seed({"H1": h1})
    z = next(z for z in store._mirror.values()
             if z.tf == "H1" and z.side == "resistance")
    assert z.status == "active"
    assert z.touches == 1
    assert z.last_pattern == "OB"
    store.close()


def test_chop_swing_below_40_ignored(tmp_path):
    """A chop swing (no displacement, no touches, not round) scores
    < 40 -> noise, never active."""
    rows = [[4206.0, 4208.0, 4205.0, 4206.0, 1000.0] for _ in range(60)]
    rows[10] = [4206.0, 4211.0, 4205.0, 4206.0, 1000.0]  # tiny swing
    h1 = _frame(rows, freq="h")
    store = ZoneStore(str(tmp_path / "zones.db"))
    store.seed({"H1": h1})
    zs = [z for z in store._mirror.values() if z.tf == "H1"]
    assert zs, "swing must be cataloged"
    assert all(z.tier == "noise" and z.status == "noise" for z in zs)
    assert not store.active_zones()
    store.close()


# ------------------------------------------------------------- cluster / conflict

def test_cluster_bonus_same_direction(tmp_path):
    """Two same-direction zones within 0.5 ATR -> CLUSTER,
    combined = max + 10 (cap 100)."""
    d1 = _d1_with_swings(n=200, swing_high=(100, 4300.0))
    h4_rows = _flat(500)
    # Late swing (idx 480): the H4 frame spans ~83 days, so an
    # early swing would already be past the 100-H4-bar expiry.
    h4_rows[480] = [4255.0, 4305.0, 4190.0, 4255.0, 1000.0]
    h4 = _frame(h4_rows, freq="4h")
    store = ZoneStore(str(tmp_path / "zones.db"))
    store.seed({"D1": d1, "H4": h4})
    clusters = find_clusters(store.active_zones())
    assert len(clusters) == 1
    cl = clusters[0]
    assert cl["same_direction"] and not cl["conflict"]
    assert cl["combined_score"] == pytest.approx(
        min(100.0, max(z.score for z in cl["zones"]) + 10.0))
    store.close()


def test_conflict_waits_for_confirm(tmp_path):
    """Opposite-direction zones near each other -> CONFLICT:
    detected, no auto-pick (combined_score None)."""
    d1 = _d1_with_swings(n=200, swing_high=(100, 4300.0))
    # H4 priced HIGH (~4350) with a dip to 4295: a valid swing LOW
    # needs neighbours' lows ABOVE it.  Late swing (idx 480) so it
    # is not past the 100-H4-bar expiry at seed.
    h4_rows = _flat(500, close=4350.0, high=4360.0, low=4340.0)
    h4_rows[480] = [4345.0, 4360.0, 4295.0, 4345.0, 1000.0]
    h4 = _frame(h4_rows, freq="4h")
    store = ZoneStore(str(tmp_path / "zones.db"))
    store.seed({"D1": d1, "H4": h4})
    clusters = find_clusters(store.active_zones())
    assert len(clusters) == 1
    assert clusters[0]["conflict"]
    assert clusters[0]["combined_score"] is None
    store.close()


# ------------------------------------------------------------- persistence / incremental

def test_persistent_db_survives_restart(tmp_path):
    """Seed, close, reopen the same path -> zones + counts + cursors
    preserved, no duplicates on re-apply."""
    d1 = _d1_with_swings(n=200, swing_high=(50, 4300.0),
                         swing_low=(100, 4100.0))
    path = str(tmp_path / "zones.db")
    store = ZoneStore(path)
    store.seed({"D1": d1})
    n_mirror = len(store._mirror)
    n_active = len(store.active_zones())
    store.close()
    store2 = ZoneStore(path)
    assert len(store2._mirror) == n_mirror
    assert len(store2.active_zones()) == n_active
    out = store2.apply({"D1": d1}, str(d1.index[-1]))
    assert out == [], "re-apply with no new bars must be a no-op"
    assert len(store2._mirror) == n_mirror
    store2.close()


def test_incremental_update_no_recalc(tmp_path):
    """Seed N bars; append ONE touch bar -> touches increment by
    exactly 1; replaying the same frame processes 0 bars."""
    rows = _flat(60)
    rows[10] = [4250.0, 4300.0, 4190.0, 4250.0, 1000.0]
    h1 = _frame(rows, freq="h")
    store = ZoneStore(str(tmp_path / "zones.db"))
    store.seed({"H1": h1})
    z = next(z for z in store._mirror.values() if z.tf == "H1")
    assert z.touches == 0
    touch = [4200.0, 4305.0, 4190.0, 4200.0, 1000.0]  # wick + close away
    h1b = pd.concat([h1, _frame(
        [touch], freq="h", start=h1.index[-1] + pd.Timedelta(hours=1))])
    out = store.apply({"H1": h1b}, str(h1b.index[-1]))
    assert any(o.outcome == "TOUCHED" and o.zone_key == z.key
               for o in out)
    assert store._mirror[z.key].touches == 1
    # Replay the same frame (new tick id, no new bars) -> no-op.
    out2 = store.apply({"H1": h1b}, "replay-different-tick-id")
    assert out2 == []
    assert store._mirror[z.key].touches == 1
    store.close()


# ------------------------------------------------------------- scoring

def test_scoring_formula():
    """Score = base + touches + displacement + confluence + pool +
    round + volume + recency (MAJOR track never negative)."""
    others = [
        StructuralZone(tf="D1", side="resistance", level=4300.0,
                       top=4310.0, bottom=4290.0, atr=20.0),
        StructuralZone(tf="H4", side="resistance", level=4302.0,
                       top=4312.0, bottom=4292.0, atr=20.0),
        StructuralZone(tf="H1", side="resistance", level=4298.0,
                       top=4308.0, bottom=4288.0, atr=20.0),
    ]
    z = StructuralZone(tf="W1", side="resistance", level=4300.0,
                       top=4310.0, bottom=4290.0, atr=20.0,
                       touches=3, displacement_atr=2.5,
                       volume_ratio=1.5)
    score, tier = score_zone(z, others + [z], age_d1=2.0)
    # 45 + 30 + 15 + 15 (confluence) + 10 (pool) + 5 (round)
    # + 10 (volume) = 130 provisional -> recency +10 -> clamp 100.
    assert score == pytest.approx(100.0) and tier == "MAJOR"
    # Fresh H4, one touch, modest displacement, no confluence,
    # not a round number (4513 is 13 away from 4500).
    z2 = StructuralZone(tf="H4", side="resistance", level=4513.0,
                        top=4523.0, bottom=4503.0, atr=20.0,
                        touches=1, displacement_atr=1.5)
    score2, tier2 = score_zone(z2, [z2], age_d1=3.0)
    # 25 + 10 + 5 + 0 + 0 + 0 = 40 provisional -> recency +10 = 50.
    assert score2 == pytest.approx(50.0) and tier2 == "MINOR"
    # Old minor fades: provisional < 60, age > 100 -> recency -10.
    # (4113 is not a round number: 13 away from 4100.)
    z3 = StructuralZone(tf="H1", side="support", level=4113.0,
                        top=4123.0, bottom=4103.0, atr=20.0,
                        touches=0, displacement_atr=0.5)
    score3, tier3 = score_zone(z3, [z3], age_d1=150.0)
    # 10 + 0 + 0 = 10 -> recency -10 -> 0.
    assert score3 == pytest.approx(0.0) and tier3 == "noise"
    # MAJOR track never penalised for age alone.
    z4 = StructuralZone(tf="D1", side="resistance", level=4300.0,
                        top=4310.0, bottom=4290.0, atr=20.0,
                        touches=3, displacement_atr=2.5)
    score4, _ = score_zone(z4, [z4], age_d1=900.0)
    # 40 + 30 + 15 + 5 (round) = 90 provisional -> recency 0.
    assert score4 == pytest.approx(90.0)


# ------------------------------------------------------------- zones never trigger

def test_zones_have_no_trigger_or_order_path():
    """Zones annotate; they never trigger, fire, or order.  No
    trigger/order method may exist on the store or the cache."""
    for cls in (ZoneStore,):
        for name in ("place_order", "trigger", "fire", "signal",
                     "enter", "buy", "sell"):
            assert not hasattr(cls, name), f"{cls.__name__}.{name}"
    from goldenswing.live.zones import ZoneCache
    for name in ("place_order", "trigger", "fire", "signal",
                 "enter", "buy", "sell"):
        assert not hasattr(ZoneCache, name), f"ZoneCache.{name}"
