"""Chart rendering for GoldenSwing signals.

:func:`render` turns one signal bar (bars + MarketState + optional
GoldenSignal) into a study-able PNG: the reader's whole thesis — swings,
liquidity pools, FVGs, order blocks, premium/discount, killzones, levels,
entry/stop/targets — drawn on the chart in one of four styles.
"""
from .desk import render_desk_card
from .engine import ForecastNote, STYLES, heikin_ashi, render, render_to_file
from .recap import render_daily_recap
from .structure import (expected_path, measure_workspace, mtf_state,
                        session_levels)

__all__ = ["ForecastNote", "STYLES", "expected_path", "heikin_ashi",
           "measure_workspace", "mtf_state", "render",
           "render_daily_recap", "render_desk_card", "render_to_file",
           "session_levels"]
