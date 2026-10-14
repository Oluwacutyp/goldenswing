"""Detector feature gating: OFF (None) and EMPTY family sets must
reproduce Phase-2 scoring exactly; enabling a family moves the score
only through that family's documented deltas."""
import pytest

from goldenswing.features.state import FeatureSet, FootprintsState, \
    IchimokuState
from goldenswing.golden.detector import DetectorConfig, GoldenSwingDetector
from goldenswing.reader.state import (
    LiquidityPool,
    MarketState,
    StructureEvent,
    SweepEvent,
)

P2 = dict(require_sweep_confirmation=True, daily_veto=True)


def _state(features=None) -> MarketState:
    ev = StructureEvent(index=99, kind="CHOCH", direction=1, level=2005.0)
    return MarketState(
        index=100, timestamp="t", close=2000.0, atr=10.0,
        trend="down", bias=-1,
        pools_above=[LiquidityPool(price=2060.0, side="bsl", strength=1,
                                   created_at=50)],
        pools_below=[LiquidityPool(price=1940.0, side="ssl", strength=1,
                                   created_at=50)],
        last_sweep=SweepEvent(index=98, level=1990.0, extreme=1984.0,
                              direction=1),
        sweep_age=2, bull_disp_3=1.2, bull_disp_age=1,
        last_structure_event=ev, features=features)


def _score(cfg: DetectorConfig, state: MarketState) -> float:
    sig = GoldenSwingDetector(cfg).evaluate(state)
    assert sig is not None
    return sig.score


def test_features_off_and_empty_set_match_phase2():
    s = _state(FeatureSet(index=100,
                          footprints=FootprintsState(institutional_dir=1)))
    base = _score(DetectorConfig(**P2), s)
    empty = _score(DetectorConfig(feature_families=frozenset(), **P2), s)
    assert empty == base


def test_no_features_attached_is_graceful():
    s = _state(None)
    base = _score(DetectorConfig(**P2), s)
    on = _score(DetectorConfig(
        feature_families=frozenset({"footprints"}), **P2), s)
    assert on == base        # nothing attached -> nothing changes


def test_footprints_family_moves_score_by_its_delta():
    fs = FeatureSet(index=100,
                    footprints=FootprintsState(institutional_dir=1))
    s = _state(fs)
    base = _score(DetectorConfig(**P2), s)
    on = _score(DetectorConfig(
        feature_families=frozenset({"footprints"}), **P2), s)
    assert on == pytest.approx(min(100.0, base + 7.0))
    sig = GoldenSwingDetector(DetectorConfig(
        feature_families=frozenset({"footprints"}), **P2)).evaluate(s)
    assert any("institutional entry signature" in f for f in sig.factors)


def test_unlisted_family_contributes_nothing():
    fs = FeatureSet(index=100,
                    ichimoku=IchimokuState(position="inside"))
    s = _state(fs)
    base = _score(DetectorConfig(**P2), s)
    # footprints on, ichimoku NOT listed -> the inside-cloud penalty
    # must not leak in
    on = _score(DetectorConfig(
        feature_families=frozenset({"footprints"}), **P2), s)
    assert on == base
    on2 = _score(DetectorConfig(
        feature_families=frozenset({"ichimoku"}), **P2), s)
    assert on2 == pytest.approx(base - 2.0)


def test_feat_phases_fallback_scores_below_direct():
    """Fallback manipulation (BOS+sweep+FVG, inferred) is real fuel
    but scores below the witnessed direct sweep."""
    from goldenswing.features.state import PhaseState
    from goldenswing.golden.detector import GoldenSwingDetector

    det = GoldenSwingDetector
    f: list[str] = []
    fb = det._feat_phases(
        PhaseState(event="manipulation_fallback", event_direction=1),
        1, f)
    assert fb == 3.0 and "manipulation inferred" in f[0]
    f2: list[str] = []
    direct = det._feat_phases(
        PhaseState(event="spring", event_direction=1), 1, f2)
    assert direct == 4.0
    f3: list[str] = []
    assert det._feat_phases(
        PhaseState(event="manipulation_fallback", event_direction=1),
        -1, f3) == 0.0
