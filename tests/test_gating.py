"""Context gates (Phase 4): each gated family fires ONLY inside its
trader context and is silent outside it."""
import pytest

from goldenswing.features import gating
from goldenswing.features.state import (
    FeatureSet,
    AOCMember,
    AOCState,
    AOCZone,
    FibState,
    FootprintsState,
    IchimokuState,
    MTFState,
    PatternInstance,
    PatternsState,
    PhaseState,
    SDZone,
    Trendline,
    TrendlineEvent,
    TrendlinesState,
    VolumeState,
    ZonesState,
)
from goldenswing.golden.detector import DetectorConfig, GoldenSwingDetector
from goldenswing.reader.state import (
    LiquidityPool,
    MarketState,
    OrderBlock,
    StructureEvent,
)


def _state(**kw) -> MarketState:
    base = dict(index=100, timestamp="t", close=2000.0, atr=10.0,
                trend="up", bias=1)
    base.update(kw)
    return MarketState(**base)


def _fs(**kw) -> FeatureSet:
    return FeatureSet(index=100, **kw)


def _deltas(contribs):
    return sum(d for d, _ in contribs)


# --------------------------------------------------------------------- fib
def _fib_fs(**fib_kw):
    fib = FibState(leg_direction=1, in_ote_long=True, **fib_kw)
    return _fs(fib=fib)


def _fib_state(**kw):
    kw.setdefault("last_structure_event",
                  StructureEvent(index=90, kind="BOS", direction=1,
                                 level=1990.0))
    kw.setdefault("daily_bias", 1)
    return _state(**kw)


def test_fib_gate_fires_in_ote_pullback_context():
    out = gating.fib_gate(_fib_state(), _fib_fs(), 1, "continuation")
    assert _deltas(out) == pytest.approx(6.0)


def test_fib_gate_silent_outside_context():
    fs = _fib_fs()
    # still extending (not a pullback)
    assert gating.fib_gate(_fib_state(momentum=1.0), fs, 1, "x") == []
    # HTF does not agree
    assert gating.fib_gate(_fib_state(daily_bias=0), fs, 1, "x") == []
    # leg never broke structure
    s = _fib_state(last_structure_event=StructureEvent(
        index=90, kind="BOS", direction=-1, level=2010.0))
    assert gating.fib_gate(s, fs, 1, "x") == []
    # not in the OTE zone
    fs2 = _fs(fib=FibState(leg_direction=1, in_ote_long=False))
    assert gating.fib_gate(_fib_state(), fs2, 1, "x") == []
    # structure flipped against the trade
    assert gating.fib_gate(_fib_state(bias=-1), fs, 1, "x") == []


# ------------------------------------------------------------------- zones
def _zone(direction=1, touches=0):
    return SDZone(top=2005.0, bottom=1995.0, direction=direction,
                  created_at=50, base_bars=2, departure=1.4,
                  touches=touches)


def test_zones_gate_fires_fresh_aligned_slowing():
    fs = _fs(zones=ZonesState(in_demand=True, nearest_demand=_zone(),
                              demand_dist_atr=0.0))
    s = _state(daily_bias=1, momentum=0.2, body_ratio=0.1, bear_disp_3=0.4)
    out = gating.zones_gate(s, fs, 1, "continuation")
    assert _deltas(out) == pytest.approx(6.0)


def test_zones_gate_silent_when_tested_misaligned_or_knifing():
    fresh = _fs(zones=ZonesState(in_demand=True, nearest_demand=_zone(),
                                 demand_dist_atr=0.0))
    calm = dict(momentum=0.2, body_ratio=0.1, bear_disp_3=0.4)
    # tested zone is spent
    tested = _fs(zones=ZonesState(
        in_demand=True, nearest_demand=_zone(touches=2),
        demand_dist_atr=0.0))
    assert gating.zones_gate(_state(daily_bias=1, **calm), tested,
                             1, "x") == []
    # against the daily bias
    assert gating.zones_gate(_state(daily_bias=-1, **calm), fresh,
                             1, "x") == []
    # knifing into the level is not an arrival
    knife = dict(momentum=0.2, body_ratio=0.1, bear_disp_3=1.6)
    assert gating.zones_gate(_state(daily_bias=1, **knife), fresh,
                             1, "x") == []


# --------------------------------------------------------------- ichimoku
def test_ichimoku_gate_regime_plus_expansion_confirmation():
    fs = _fs(ichimoku=IchimokuState(position="above", d1_position="above",
                                    tk_cross=1),
             phase=PhaseState(phase="expansion_up"))
    out = gating.ichimoku_gate(_state(), fs, 1, "continuation")
    assert _deltas(out) == pytest.approx(3.0 + 3.0 + 2.0)


def test_ichimoku_gate_silent_inside_daily_cloud():
    fs = _fs(ichimoku=IchimokuState(position="above",
                                    d1_position="inside", tk_cross=1),
             phase=PhaseState(phase="expansion_up"))
    assert gating.ichimoku_gate(_state(), fs, 1, "x") == []


def test_ichimoku_gate_no_h1_points_in_consolidation():
    fs = _fs(ichimoku=IchimokuState(position="above", d1_position="above",
                                    tk_cross=1),
             phase=PhaseState(phase="consolidation"))
    out = gating.ichimoku_gate(_state(), fs, 1, "x")
    assert _deltas(out) == pytest.approx(3.0)   # regime vote only


# --------------------------------------------------------------- patterns
def _aoc_zone_at(price, families):
    return AOCZone(center=price, top=price + 2.0, bottom=price - 2.0,
                   width=4.0, families=tuple(sorted(families)),
                   score=3.0,
                   members=tuple(AOCMember(family=f, price=price,
                                           weight=1.0, recency=1.0)
                                 for f in families))


def _pattern_fs(status="broken", aoc_major=True):
    inst = PatternInstance(
        kind="double_top", direction=-1, status=status, formed_at=90,
        neckline=1990.0, extreme=2060.0, invalidation=2060.0,
        target=1920.0)
    # Phase 4.5: the major-level test is an AOC pivot zone (>= 3
    # families).  The firing case carries one at the extreme.
    aoc = AOCState(zones=(_aoc_zone_at(2060.0, ("pool", "ob", "fib")),)) \
        if aoc_major else AOCState()
    return _fs(patterns=PatternsState(active=(inst,)), aoc=aoc)


def _pattern_state(**kw):
    kw.setdefault("pools_above", [LiquidityPool(
        price=2060.0, side="bsl", strength=2, created_at=40)])
    kw.setdefault("range_high", 2070.0)
    kw.setdefault("range_low", 1990.0)      # span 80 = 8 ATR (extended)
    kw.setdefault("bear_disp_3", 1.2)       # neckline break displaced
    return _state(**kw)


def test_patterns_gate_fires_broken_at_major_level():
    out = gating.patterns_gate(_pattern_state(), _pattern_fs(), -1, "x")
    assert _deltas(out) == pytest.approx(6.0)


def test_patterns_gate_silent_without_trigger_or_location():
    # formed but neckline intact — pattern alone never scores
    assert gating.patterns_gate(
        _pattern_state(), _pattern_fs(status="active"), -1, "x") == []
    # broken but no displacement / structure confirmation
    assert gating.patterns_gate(
        _pattern_state(bear_disp_3=0.2), _pattern_fs(), -1, "x") == []
    # broken with displacement but in the middle of nowhere
    assert gating.patterns_gate(
        _pattern_state(pools_above=[]),
        _pattern_fs(aoc_major=False), -1, "x") == []


# ------------------------------------------------------------- footprints
def test_footprints_gate_institutional_needs_a_level():
    fs = _fs(footprints=FootprintsState(institutional_dir=1))
    near = _state(active_obs=[OrderBlock(
        top=2005.0, bottom=1995.0, direction=1, created_at=90,
        candle_index=88)])
    assert _deltas(gating.footprints_gate(near, fs, 1, "x")) == \
        pytest.approx(7.0)
    far = _state(active_obs=[OrderBlock(
        top=2100.0, bottom=2090.0, direction=1, created_at=90,
        candle_index=88)])
    assert gating.footprints_gate(far, fs, 1, "x") == []


def test_footprints_gate_chase_caution_only_at_climax_extreme():
    fs = _fs(footprints=FootprintsState(retail_chase=1),
             volume=VolumeState(available=True, climax=True))
    hot = _state(eq_position=0.95)
    assert _deltas(gating.footprints_gate(hot, fs, 1, "x")) == \
        pytest.approx(-5.0)
    fs_noclimax = _fs(footprints=FootprintsState(retail_chase=1),
                      volume=VolumeState(available=True, climax=False))
    assert gating.footprints_gate(hot, fs_noclimax, 1, "x") == []
    mid = _state(eq_position=0.60)
    assert gating.footprints_gate(mid, fs, 1, "x") == []


# --------------------------------------------------------------------- mtf
def test_mtf_gate_confirms_modestly_and_cap_vetoes():
    both_for = _fs(mtf=MTFState(d1_bias=1, h4_bias=1))
    assert _deltas(gating.mtf_gate(_state(), both_for, 1, "x")) == \
        pytest.approx(3.0)
    assert gating.mtf_cap(_state(), both_for, 1) is None

    both_against = _fs(mtf=MTFState(d1_bias=-1, h4_bias=-1))
    assert gating.mtf_gate(_state(), both_against, 1, "x") == []
    assert gating.mtf_cap(_state(), both_against, 1) == 59.0


# -------------------------------------------------------------- trendlines
def test_trendlines_bounce_needs_trend_context():
    ev = TrendlineEvent(kind="bounce", direction=1, line_price=1990.0,
                        touches=2)
    fs = _fs(trendlines=TrendlinesState(events=(ev,)))
    assert _deltas(gating.trendlines_score(
        _state(bias=1), fs, 1, "continuation")) == pytest.approx(5.0)
    # same bounce, but the trade fights the trend — silent
    assert gating.trendlines_score(
        _state(bias=-1), fs, 1, "continuation") == []


def test_trendlines_break_needs_displacement():
    ev = TrendlineEvent(kind="break", direction=1, line_price=1990.0,
                        touches=2)
    fs = _fs(trendlines=TrendlinesState(events=(ev,)))
    hard = _state(bias=-1, bear_disp_3=1.3)
    assert _deltas(gating.trendlines_score(hard, fs, -1, "x")) == \
        pytest.approx(5.0)
    soft = _state(bias=-1, bear_disp_3=0.3)
    assert gating.trendlines_score(soft, fs, -1, "x") == []


# ------------------------------------------------- detector integration
def _continuation_state(features, daily_bias):
    return MarketState(
        index=100, timestamp="t", close=2000.0, atr=10.0,
        trend="up", bias=1, bull_disp_3=1.0, daily_bias=daily_bias,
        pools_above=[LiquidityPool(price=2060.0, side="bsl", strength=1,
                                   created_at=50)],
        last_structure_event=StructureEvent(
            index=97, kind="BOS", direction=1, level=1990.0),
        structure_event_age=3, features=features)


def test_detector_gated_fib_silent_then_speaks():
    fs = _fs(fib=FibState(leg_direction=1, in_ote_long=True))
    det = GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True,
        feature_families=frozenset({"fib_ctx"})))
    base = GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True))
    # HTF silent (daily + MTF D1 both neutral) -> identical to Phase 2
    s0 = _continuation_state(fs, daily_bias=0)
    assert det.evaluate(s0).score == base.evaluate(s0).score
    # HTF agrees -> the gate opens and OTE adds its +6
    s1 = _continuation_state(fs, daily_bias=1)
    assert det.evaluate(s1).score == pytest.approx(
        min(100.0, base.evaluate(s1).score + 6.0))


def test_detector_mtf_ctx_caps_below_golden():
    fs = _fs(mtf=MTFState(d1_bias=-1, h4_bias=-1))
    s = _continuation_state(fs, daily_bias=0)
    det = GoldenSwingDetector(DetectorConfig(
        require_sweep_confirmation=True, daily_veto=True,
        feature_families=frozenset({"mtf_ctx"})))
    sig = det.evaluate(s)
    assert sig is not None and sig.score <= 59.0
    assert any("MTF veto" in f for f in sig.factors)
