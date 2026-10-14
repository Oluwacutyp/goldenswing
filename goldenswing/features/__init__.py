"""Phase-3 feature layer: ten causal per-bar feature families that sit
on top of the MarketReader's MarketState (see SPEC.md).

Families: Bollinger, Fibonacci/OTE, Ichimoku, relative strength,
classical patterns, supply/demand zones, AMD phase detection,
participant footprints, volume character, multi-timeframe context.
"""
from .config import FeatureConfig
from .engine import FeatureEngine, read_and_featurize
from .state import (
    FAMILIES,
    AOCMember,
    AOCState,
    AOCZone,
    BollingerState,
    FeatureSet,
    FibState,
    FootprintsState,
    IchimokuState,
    MTFState,
    PatternInstance,
    PatternsState,
    PeerRS,
    PhaseState,
    RSState,
    SDZone,
    Trendline,
    TrendlineEvent,
    TrendlinesState,
    VolumeState,
    ZonesState,
)

__all__ = [
    "FeatureConfig",
    "FeatureEngine",
    "read_and_featurize",
    "FAMILIES",
    "FeatureSet",
    "AOCMember",
    "AOCState",
    "AOCZone",
    "BollingerState",
    "FibState",
    "IchimokuState",
    "RSState",
    "PeerRS",
    "PatternInstance",
    "PatternsState",
    "SDZone",
    "ZonesState",
    "PhaseState",
    "VolumeState",
    "FootprintsState",
    "MTFState",
    "Trendline",
    "TrendlineEvent",
    "TrendlinesState",
]
