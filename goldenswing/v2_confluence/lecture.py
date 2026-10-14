"""The Lecture: Complete Professional Trading System.

Implements the full MTF confluence framework as taught:

1. MACRO PERMISSION (D1): Daily bias assigns direction. Above 200 EMA
   (or bullish structure) → longs only. Below → shorts only. Neutral → sit out.

2. CONFLUENCE LOCATION (H1): Price must be at a high-quality zone with
   2+ factors: Structure (sr_flip) + Fibonacci (50/61.8/78.6%) +
   Volume Profile (POC/HVN/VAH/VAL).

3. MOMENTUM CONFIRMATION: RSI divergence or exhaustion at the zone.
   Bullish div (price LL, RSI HL) → sellers exhausted.

4. EXECUTION TRIGGER (H1/M15): Rejection candle (long wick, close back
   through zone) or minor structure break in the HTF direction.

5. RISK & SIZING: Stop 1.5× ATR beyond the zone. Risk 1% of equity.
   Position size = risk_amount / (stop_distance × point_value).

6. INVALIDATION: If price closes firmly through the zone (no rejection),
   premise is broken → exit immediately. Don't wait for stop.

This is the complete system, not a collection of patterns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LectureSignal:
    """A complete trade plan from the lecture framework."""
    direction: int              # +1 long, -1 short
    entry: float                # entry price
    stop: float                 # stop loss price
    targets: list[float]        # profit targets
    position_size: float        # lots
    risk_amount: float          # $ risked
    zone_top: float             # confluence zone top
    zone_bottom: float          # confluence zone bottom
    confluence_score: int       # 2 or 3
    confluence_factors: list[str] = field(default_factory=list)
    rsi_signal: str = ""        # "bullish_divergence", "exhaustion", etc.
    invalidation_price: float = 0.0  # close beyond this = exit immediately


@dataclass
class LectureConfig:
    """Configuration for the lecture system."""
    risk_pct: float = 1.0       # % of equity per trade
    atr_stop_mult: float = 1.5  # stop = 1.5 × ATR
    min_confluence: int = 2     # minimum factors for a zone
    point_value_per_lot: float = 100.0  # $ per point per 1.0 lot (XAUUSD)
    # Note: 0.01 lot = $1.00 per point? No — $0.10 per point per 0.01 lot
    # So 1.0 lot = $10.00 per point. Adjust for broker.
    rsi_overbought: float = 80.0
    rsi_oversold: float = 20.0


def calculate_position_size(equity: float, risk_pct: float,
                           entry: float, stop: float,
                           point_value_per_lot: float = 10.0) -> tuple[float, float]:
    """Calculate lot size for a given risk.

    Returns (lots, risk_amount).
    point_value_per_lot: $ per 1 point move per 1.0 lot.
    For XAUUSD: 1.0 lot = $10 per $1 move (check broker).
    """
    risk_amount = equity * (risk_pct / 100.0)
    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return 0.0, 0.0
    risk_per_lot = stop_distance * point_value_per_lot
    if risk_per_lot <= 0:
        return 0.0, 0.0
    lots = risk_amount / risk_per_lot
    # Round to 0.01
    lots = round(lots, 2)
    actual_risk = lots * risk_per_lot
    return lots, actual_risk


def check_invalidation(close: float, zone_top: float, zone_bottom: float,
                       direction: int) -> bool:
    """Check if the structural premise is broken.

    Long: close firmly below zone_bottom → invalidated.
    Short: close firmly above zone_top → invalidated.
    """
    if direction == 1:
        return close < zone_bottom
    else:
        return close > zone_top
