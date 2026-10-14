"""FeatureEngine — orchestrates the ten feature families into one
:class:`FeatureSet` per bar, aligned with the MarketReader's states.

``attach`` writes the FeatureSet onto each MarketState (the
``features`` field Phase 1-2 reserved for this).  States without an
attached FeatureSet — or a detector that hasn't opted in — behave
exactly as Phase 2.
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from ..reader.reader import MarketReader
from ..reader.state import MarketState
from . import (aoc, bollinger, divergence, fibonacci, footprints,
               ichimoku, mtf, patterns, phases, relstrength,
               supply_demand, trendlines, volume)
from .config import FeatureConfig
from .state import (BollingerState, DivergenceState, FeatureSet, FibState,
                    FootprintsState, IchimokuState, MTFState, PatternsState,
                    PhaseState, RSState, Trendline, TrendlineEvent,
                    TrendlinesState, VolumeState, ZonesState)


class FeatureEngine:
    def __init__(self, config: FeatureConfig | None = None):
        self.cfg = config or FeatureConfig()

    def compute(self, df: pd.DataFrame, states: list[MarketState],
                peers: dict | None = None) -> list[FeatureSet]:
        cfg = self.cfg
        n = len(df)
        if len(states) != n:
            raise ValueError("states must align 1:1 with df bars")

        bb = bollinger.compute(df, states, cfg) if cfg.bollinger \
            else [BollingerState()] * n
        fb = fibonacci.compute(df, states, cfg) if cfg.fib \
            else [FibState()] * n
        ich = ichimoku.compute(df, states, cfg) if cfg.ichimoku \
            else [IchimokuState()] * n
        rs = relstrength.compute(df, states, cfg, peers) if cfg.relstr \
            else [RSState()] * n
        pat = patterns.compute(df, states, cfg) if cfg.patterns \
            else [PatternsState()] * n
        zon = supply_demand.compute(df, states, cfg) if cfg.zones \
            else [ZonesState()] * n
        pha = phases.compute(df, states, cfg) if cfg.phases \
            else [PhaseState()] * n
        vol = volume.compute(df, states, cfg) if cfg.volume \
            else [VolumeState()] * n
        if cfg.footprints:
            rvol, vwap, _avail = volume.volume_arrays(df, cfg)
            fp = footprints.compute(df, states, cfg, rvol, vwap)
        else:
            fp = [FootprintsState()] * n
        mt = mtf.compute(df, states, cfg) if cfg.mtf \
            else [MTFState()] * n
        tl = trendlines.compute(df, states, cfg) if cfg.trendlines \
            else [TrendlinesState()] * n
        dv = divergence.compute(df, states, cfg) if cfg.divergence \
            else [DivergenceState()] * n

        sets = [FeatureSet(index=t, bollinger=bb[t], fib=fb[t],
                           ichimoku=ich[t], relstr=rs[t], patterns=pat[t],
                           zones=zon[t], phase=pha[t], volume=vol[t],
                           footprints=fp[t], mtf=mt[t], trendlines=tl[t],
                           divergence=dv[t])
                for t in range(n)]
        # AOC clusters the levels the other families just produced, so
        # it runs AFTER the partial FeatureSets exist (still causal:
        # aoc.compute is a pure function of bar t's confirmed state).
        if cfg.aoc:
            ao = aoc.compute(df, states, cfg, sets)
            sets = [dataclasses.replace(fs, aoc=a)
                    for fs, a in zip(sets, ao)]
        return sets

    def attach(self, df: pd.DataFrame, states: list[MarketState],
               peers: dict | None = None) -> list[MarketState]:
        """Compute features and write them onto the states in place."""
        for s, fs in zip(states, self.compute(df, states, peers)):
            s.features = fs
        return states


def read_and_featurize(df: pd.DataFrame, peers: dict | None = None,
                       feature_config: FeatureConfig | None = None,
                       ) -> list[MarketState]:
    """Convenience: MarketReader + FeatureEngine.attach in one call."""
    states = MarketReader().read(df)
    FeatureEngine(feature_config).attach(df, states, peers)
    return states
