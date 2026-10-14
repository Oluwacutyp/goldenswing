"""Fibonacci confluence for Version 2.

Measures retracements of impulse legs and identifies confluence zones
where Fib levels overlap with structure.

Levels:
- Retracements: 50%, 61.8%, 78.6% (entries)
- Extensions: 1.272, 1.618 (targets)

A Fib level alone is not a signal. It becomes significant when it
overlaps with structure (sr_flip), Volume Profile (HVN/POC), or
psychological levels — that's the confluence engine's job.
"""
from __future__ import annotations

from dataclasses import dataclass


# Standard retracement levels for entries
RETRACEMENT_LEVELS = [0.50, 0.618, 0.786]

# Extension levels for targets
EXTENSION_LEVELS = [1.272, 1.618]


@dataclass
class FibLevel:
    """A single Fibonacci level."""
    ratio: float          # 0.618, etc.
    price: float          # price at this level
    kind: str             # "retracement" | "extension"
    impulse_high: float   # high of the impulse leg
    impulse_low: float    # low of the impulse leg
    direction: int        # +1 bullish impulse, -1 bearish impulse


@dataclass
class FibZone:
    """A confluence zone around a Fib level."""
    level: FibLevel
    top: float            # zone top (level + tolerance)
    bottom: float         # zone bottom (level - tolerance)


def find_impulse_legs(high: list[float], low: list[float], close: list[float],
                      min_atr_mult: float = 3.0, atr: list[float] | None = None,
                      lookback: int = 100) -> list[tuple[int, int, int]]:
    """Find significant impulse legs (strong directional moves).

    Returns list of (start_idx, end_idx, direction) for legs with
    range > min_atr_mult * ATR.

    An impulse leg is a clean directional move — the kind professionals
    measure Fibonacci retracements from.
    """
    if atr is None:
        return []

    legs = []
    n = len(close)

    for end in range(lookback, n):
        start = max(0, end - lookback)
        # Find the most significant swing in the window
        window_high = max(high[start:end])
        window_low = min(low[start:end])
        hi_idx = high[start:end].index(window_high) + start if isinstance(high, list) else start
        lo_idx = low[start:end].index(window_low) + start if isinstance(low, list) else start

        # Determine direction and measure
        if hi_idx > lo_idx:
            # bullish impulse: low -> high
            move = window_high - window_low
            a = atr[end] if end < len(atr) else atr[-1]
            if a > 0 and move > min_atr_mult * a:
                legs.append((lo_idx, hi_idx, 1))
        else:
            # bearish impulse: high -> low
            move = window_high - window_low
            a = atr[end] if end < len(atr) else atr[-1]
            if a > 0 and move > min_atr_mult * a:
                legs.append((hi_idx, lo_idx, -1))

    # Deduplicate overlapping legs, keep the largest
    legs = sorted(set(legs))
    return legs[-5:] if len(legs) > 5 else legs  # keep recent


def retracement_levels(impulse_high: float, impulse_low: float,
                       direction: int) -> list[FibLevel]:
    """Calculate Fib retracement levels for an impulse leg.

    For bullish impulse (low->high): retracements are below the high.
    For bearish impulse (high->low): retracements are above the low.
    """
    levels = []
    move = impulse_high - impulse_low

    for ratio in RETRACEMENT_LEVELS:
        if direction == 1:  # bullish impulse, pullback down
            price = impulse_high - ratio * move
        else:  # bearish impulse, pullback up
            price = impulse_low + ratio * move
        levels.append(FibLevel(
            ratio=ratio, price=price, kind="retracement",
            impulse_high=impulse_high, impulse_low=impulse_low,
            direction=direction,
        ))
    return levels


def extension_levels(impulse_high: float, impulse_low: float,
                     direction: int) -> list[FibLevel]:
    """Calculate Fib extension levels for profit targets."""
    levels = []
    move = impulse_high - impulse_low

    for ratio in EXTENSION_LEVELS:
        if direction == 1:  # bullish, target above high
            price = impulse_high + (ratio - 1.0) * move
        else:  # bearish, target below low
            price = impulse_low - (ratio - 1.0) * move
        levels.append(FibLevel(
            ratio=ratio, price=price, kind="extension",
            impulse_high=impulse_high, impulse_low=impulse_low,
            direction=direction,
        ))
    return levels


def level_near_price(level_price: float, price: float, atr: float,
                     tol_atr: float = 0.25) -> bool:
    """Check if a Fib level is near a price (within tolerance)."""
    if atr <= 0:
        return False
    return abs(level_price - price) <= tol_atr * atr
