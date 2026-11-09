"""Chart rendering for GoldenSwing signals.

:func:`render` turns one signal bar (bars + MarketState + optional
GoldenSignal) into a study-able PNG: the reader's whole thesis — swings,
liquidity pools, FVGs, order blocks, premium/discount, killzones, levels,
entry/stop/targets — drawn on the chart in one of four styles.
"""
from .engine import ForecastNote, STYLES, heikin_ashi, render, render_to_file
from .recap import render_daily_recap

__all__ = ["ForecastNote", "STYLES", "heikin_ashi", "render",
           "render_daily_recap", "render_to_file"]
