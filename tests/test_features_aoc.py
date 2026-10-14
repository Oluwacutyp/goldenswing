"""AOC family (Phase 4.5): level harvest, clustering, concentration
score, causality, detector integration, and the patterns-gate repair."""
import math

import pytest

from goldenswing.features import FeatureConfig, FeatureEngine, gating
from goldenswing.features.aoc import cluster, harvest, _recency
from goldenswing.features.state import (
    AOCMember,
    AOCState,
    AOCZone,
    FeatureSet,
    FibState,
    PatternInstance,
    PatternsState,
    SDZone,
    Trendline,
    TrendlinesState,
    ZonesState,
)
from goldenswing.golden.detector import DetectorConfig, GoldenSwingDetector
from goldenswing.reader.state import (
    FairValueGap,
    LiquidityPool,
    MarketState,
    OrderBlock,
    StructureEvent,
    SweepEvent,
)

P2 = dict(require_sweep_confirmation=True, daily_veto=True)


def _member(family, price, weight=1.0, recency=1.0):
    return AOCMember(family=family, price=price, weight=weight,
                     recency=recency)


def _zone(center, families, score=3.0, width=4.0):
    members = tuple(_member(f, center) for f in families)
    return AOCZone(center=center, top=center + width / 2,
                   bottom=center - width / 2, width=width,
                   families=tuple(sorted(families)), score=score,
                   members=members)


def _aoc_state(*zones):
    return AOCState(zones=tuple(zones))


# ------------------------------------------------------------- unit: score
def test_recency_floors_at_half():
    assert _recency(0, 96.0) == pytest.approx(1.0)
    assert _recency(96, 96.0) == pytest.approx(0.75)
    assert _recency(10 ** 9, 96.0) == pytest.approx(0.5)
    assert _recency(-5, 96.0) == pytest.approx(1.0)


def test_cluster_groups_within_tol_and_caps_span():
    ms = [_member("pool", 2000.0), _member("ob", 2002.0),
          _member("fvg", 2004.0), _member("pool", 2010.0)]
    # tol 3.0: first three join; 2010 is 6 away from 2004 -> new cluster
    cl = cluster(ms, tol=3.0, max_span=10.0)
    assert len(cl) == 2
    assert [len(c) for c in cl] == [3, 1]
    # tight max_span forces a split even inside tolerance
    cl2 = cluster(ms[:3], tol=3.0, max_span=2.0)
    assert len(cl2) == 2


def test_cluster_dedupes_families_for_threshold():
    # two pools + one zone = 2 DISTINCT families, not 3
    cl = cluster([_member("pool", 2000.0), _member("pool", 2001.0),
                  _member("sd_zone", 2002.0)], tol=3.0, max_span=10.0)
    assert len(cl) == 1
    from goldenswing.features.aoc import _zone_of
    z = _zone_of(cl[0])
    assert z.families == ("pool", "sd_zone")
    assert len(z.members) == 3            # members kept, families deduped
    assert z.score == pytest.approx(3.0)  # score counts every member


def test_major_at_needs_three_families():
    ao = _aoc_state(_zone(2000.0, ("pool", "ob")),
                    _zone(2050.0, ("pool", "ob", "fib")))
    assert ao.major_at(2000.0, 4.0) is None       # 2 families: not major
    hit = ao.major_at(2050.0, 4.0)
    assert hit is not None and hit.center == 2050.0
    assert ao.major_at(1900.0, 4.0) is None       # nowhere near
    assert AOCState().major_at(2050.0, 4.0) is None


# ------------------------------------------------- harvest: only live ammo
def _harvest_state(**kw):
    # NOTE: per the reader's contract, the active_* lists hold ONLY
    # structures live at this bar — a swept pool or mitigated FVG
    # simply never appears here (the reader filters at snapshot
    # time).  harvest() trusts the lists for exactly this reason.
    base = dict(index=100, timestamp="t", close=2000.0, atr=10.0,
                trend="up", bias=1,
                pools_above=[LiquidityPool(price=2060.0, side="bsl",
                                           strength=2, created_at=40)],
                active_fvgs=[FairValueGap(top=2055.0, bottom=2050.0,
                                          direction=1, created_at=80)],
                active_obs=[OrderBlock(top=2052.0, bottom=2048.0,
                                       direction=1, created_at=82,
                                       candle_index=80)],
                daily_swing_high=2070.0, daily_swing_low=1930.0,
                range_high=2060.0, range_low=1940.0)
    base.update(kw)
    return MarketState(**base)


def _harvest_fs(zones_active=None, **kw):
    if zones_active is None:
        zones_active = (SDZone(top=2053.0, bottom=2049.0,
                               direction=1, created_at=78,
                               base_bars=2, departure=1.3),)
    fs = FeatureSet(
        index=100,
        zones=ZonesState(active=zones_active),
        trendlines=TrendlinesState(active=(Trendline(
            direction=1, anchor1_index=60, anchor1_price=1980.0,
            anchor2_index=80, anchor2_price=2010.0,
            slope=1.5, created_at=85, touches=1,
            last_touch_at=95, state="ACTIVE"),)),
        fib=FibState(levels=((0.618, 2051.0), (0.5, 2040.0)),
                     leg_direction=1),
        **kw)
    return fs


def test_harvest_collects_live_levels_only():
    s = _harvest_state()
    fs = _harvest_fs()
    got = harvest(s, fs, FeatureConfig(), 100)
    fams = sorted({m.family for m in got})
    assert fams == ["daily", "eq", "fib", "fvg", "ob", "pool",
                    "sd_zone", "trendline"]
    # one level per live structure — nothing invented, nothing doubled
    assert sum(1 for m in got if m.family == "pool") == 1
    assert sum(1 for m in got if m.family == "fvg") == 1
    assert sum(1 for m in got if m.family == "ob") == 1
    assert sum(1 for m in got if m.family == "sd_zone") == 1
    assert sum(1 for m in got if m.family == "daily") == 2
    assert sum(1 for m in got if m.family == "eq") == 1
    # trendline value is the line's price AT bar t
    tl = next(m for m in got if m.family == "trendline")
    assert tl.price == pytest.approx(1980.0 + 1.5 * (100 - 60))


def test_harvest_zone_freshness_weights():
    fs = _harvest_fs(zones_active=(
        SDZone(top=2053.0, bottom=2049.0, direction=1, created_at=78,
               base_bars=2, departure=1.3, touches=0),
        SDZone(top=1960.0, bottom=1956.0, direction=1, created_at=50,
               base_bars=2, departure=1.2, touches=2)))
    got = harvest(_harvest_state(), fs, FeatureConfig(), 100)
    zw = {m.price: m.weight for m in got if m.family == "sd_zone"}
    assert zw[2051.0] == pytest.approx(1.0)   # fresh
    assert zw[1958.0] == pytest.approx(0.6)    # tested


# --------------------------------- compute on synthetic full-feature states
def test_compute_builds_ranked_zones_with_above_below():
    s = _harvest_state()
    fs = _harvest_fs()
    from goldenswing.features.aoc import compute
    out = compute(None, [s], FeatureConfig(), [fs])
    assert len(out) == 1
    ao = out[0]
    assert len(ao.zones) >= 1
    # the 2048-2060 knot: pool(2060) + fvg(2052.5) + ob(2050) +
    # sd_zone(2051) + fib(2051) + tl(~2040? no) -> several families
    best = ao.zones[0]
    assert len(best.families) >= 2
    assert best.top >= best.center >= best.bottom
    assert best.width >= 0
    # above/below bookkeeping
    if ao.above is not None:
        assert ao.above.center > s.close
        assert ao.above_dist_atr == pytest.approx(
            (ao.above.center - s.close) / s.atr)
    if ao.below is not None:
        assert ao.below.center < s.close


# ------------------------------------------------- causality / truncation
def test_aoc_truncation_stable():
    import numpy as np
    import pandas as pd

    from conftest import make_df
    from goldenswing.reader import MarketReader

    rng = np.random.default_rng(3)
    n = 400
    steps = rng.normal(0, 1.0, n).cumsum()
    closes = 2000.0 + steps * 2.0
    rows = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        rows.append((float(o), float(max(o, c) + abs(rng.normal(0, 1.2))),
                     float(min(o, c) - abs(rng.normal(0, 1.2))),
                     float(c)))
    df = make_df(rows, start="2024-05-01")
    df["volume"] = (800 + 900 * np.abs(rng.normal(0, 1, n))).tolist()

    def digests(frame):
        states = MarketReader().read(frame)
        FeatureEngine().attach(frame, states)
        out = []
        for st in states:
            ao = st.features.aoc
            out.append(tuple(
                (round(z.center, 6), z.families, round(z.score, 6))
                for z in ao.zones))
        return out

    full = digests(df)
    cut = digests(df.iloc[:250])
    assert full[:250] == cut  # prefix must not change


# ------------------------------------------------- detector integration
def _det_state(aoc_state, **kw):
    # mirrors tests/test_features_detector.py::_state — the Phase-2
    # entry prerequisites (sweep + displacement + structure event) so
    # evaluate() actually emits a signal
    ev = StructureEvent(index=99, kind="CHOCH", direction=1, level=2005.0)
    base = dict(index=100, timestamp="t", close=2000.0, atr=10.0,
                trend="down", bias=-1,
                pools_above=[LiquidityPool(price=2060.0, side="bsl",
                                           strength=1, created_at=50)],
                pools_below=[LiquidityPool(price=1940.0, side="ssl",
                                           strength=1, created_at=50)],
                last_sweep=SweepEvent(index=98, level=1990.0,
                                      extreme=1984.0, direction=1),
                sweep_age=2, bull_disp_3=1.2, bull_disp_age=1,
                last_structure_event=ev,
                features=FeatureSet(index=100, aoc=aoc_state))
    base.update(kw)
    return MarketState(**base)


def _score(cfg, state, direction=1):
    return GoldenSwingDetector(cfg).evaluate(state)


def test_aoc_off_reproduces_phase2():
    s = _det_state(_aoc_state(_zone(2000.0, ("pool", "ob", "fib"))))
    base = _score(DetectorConfig(**P2), s).score
    on = _score(DetectorConfig(feature_families=frozenset({"aoc"}),
                               **P2), s).score
    # close inside a 3-family AOC acting as support -> +4.0
    assert on == pytest.approx(base + 4.0)


def test_aoc_silent_outside_context():
    # AOC far from the entry, no pool ahead -> nothing speaks
    ao = _aoc_state(_zone(2100.0, ("pool", "ob")))
    s = _det_state(ao, pools_above=[])
    base = _score(DetectorConfig(**P2), s).score
    on = _score(DetectorConfig(feature_families=frozenset({"aoc"}),
                               **P2), s).score
    assert on == base


def test_aoc_headwind_penalty_between_entry_and_target():
    # entry NOT in an AOC; an opposing concentration sits between
    # close (2000) and the first pool target (2060)
    ao = _aoc_state(_zone(2030.0, ("pool", "ob")))
    s = _det_state(ao)
    base = _score(DetectorConfig(**P2), s).score
    on = _score(DetectorConfig(feature_families=frozenset({"aoc"}),
                               **P2), s).score
    assert on == pytest.approx(base - 4.0)
    sig = _score(DetectorConfig(feature_families=frozenset({"aoc"}),
                                **P2), s)
    assert any("overhead" in f or "between entry" in f
               for f in sig.factors)


def test_aoc_empty_state_is_graceful():
    s = _det_state(AOCState())
    base = _score(DetectorConfig(**P2), s).score
    on = _score(DetectorConfig(feature_families=frozenset({"aoc"}),
                               **P2), s).score
    assert on == base


# ------------------------------------------------- patterns-gate repair
def _pattern_fs_repair(aoc_state, status="broken"):
    inst = PatternInstance(
        kind="double_top", direction=-1, status=status, formed_at=90,
        neckline=1990.0, extreme=2060.0, invalidation=2060.0,
        target=1920.0)
    return FeatureSet(index=100, patterns=PatternsState(active=(inst,)),
                      aoc=aoc_state)


def _pattern_state_repair(**kw):
    kw.setdefault("range_high", 2070.0)
    kw.setdefault("range_low", 1990.0)      # span 80 = 8 ATR (extended)
    kw.setdefault("bear_disp_3", 1.2)       # neckline break displaced
    base = dict(index=100, timestamp="t", close=2000.0, atr=10.0,
                trend="down", bias=-1)
    base.update(kw)
    return MarketState(**base)


def test_patterns_gate_fires_at_aoc_major_zone():
    ao = _aoc_state(_zone(2060.0, ("pool", "ob", "fib")))  # at extreme
    out = gating.patterns_gate(_pattern_state_repair(),
                               _pattern_fs_repair(ao), -1, "x")
    assert sum(d for d, _ in out) == pytest.approx(6.0)
    assert any("MAJOR" in f or "pivot" in f for _, f in out)


def test_patterns_gate_silent_without_three_family_aoc():
    # 2-family cluster at the extreme: major test fails
    ao = _aoc_state(_zone(2060.0, ("pool", "ob")))
    out = gating.patterns_gate(_pattern_state_repair(),
                               _pattern_fs_repair(ao), -1, "x")
    assert out == []
    # empty AOC state: silent too
    out2 = gating.patterns_gate(_pattern_state_repair(),
                                _pattern_fs_repair(AOCState()), -1, "x")
    assert out2 == []


def test_patterns_gate_neckline_also_counts():
    ao = _aoc_state(_zone(1990.0, ("sd_zone", "trendline", "fib")))
    out = gating.patterns_gate(_pattern_state_repair(),
                               _pattern_fs_repair(ao), -1, "x")
    assert sum(d for d, _ in out) == pytest.approx(6.0)
