"""GOD Conviction Engine — Python port of GOD-CE-SLIM.pine.

26-factor confluence scoring, factor gate, dynamic thresholds, strategy
router with 2-bar dwell, full entry-gate stack, Hydra trade state
machine, and the trap-intelligence layer.
"""
from .params import GodceParams, MODE_NAMES, REGIME_NAMES
from .backtest import run_backtest, walk_forward, summarize, map_breakdown

__all__ = [
    "GodceParams", "MODE_NAMES", "REGIME_NAMES",
    "run_backtest", "walk_forward", "summarize", "map_breakdown",
]
