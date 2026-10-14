"""Phase 5: execution modes + phase→playbook routing.

One market read (MarketReader + FeatureEngine), three execution modes
that differ only in timeframe, horizon, exit style and entry filters —
plus the routing layer that mutes (phase, playbook) cells the measured
shadow book says lose money.
"""
from .config import FAST_SCALP, SCALP, SWING, ModeConfig
from .routing import (MUTE_MIN_N, RouteTable, build_matrix,
                      derive_route_table, phase_group)

__all__ = [
    "ModeConfig", "SWING", "SCALP", "FAST_SCALP",
    "RouteTable", "build_matrix", "derive_route_table", "phase_group",
    "MUTE_MIN_N",
]
