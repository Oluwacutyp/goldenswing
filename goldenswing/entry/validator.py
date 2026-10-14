"""Entry validation: candles confirm, levels locate, MTF refines.

This is the entry system the project set out to build from the start:
- HTF gives bias (D1/H4)
- H1 marks the level (S/R flip, OB, liquidity pool)
- CANDLES validate the entry (rejection, engulfing, close-back)
- Entry type is chosen by setup (limit at zone vs market on confirmation)
- LTF refines the trigger (M15/M5 shows the rejection clearly)

A level without candle validation is a hope. A candle without a level
is noise. The two together are a trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EntryType(Enum):
    """How to enter the trade."""
    LIMIT_AT_ZONE = "limit_at_zone"      # patient: place limit at the level
    MARKET_ON_CONFIRM = "market_on_confirm"  # aggressive: market on candle close


@dataclass
class CandleValidation:
    """Result of validating a candle at a level."""
    valid: bool
    pattern: str          # "rejection_wick" | "engulfing" | "close_back" | "none"
    strength: float       # 0.0-1.0, how clean the validation is
    detail: str = ""


def validate_bearish_rejection(open_: float, high: float, low: float,
                               close: float, level: float,
                               atr: float) -> CandleValidation:
    """Check if a candle shows bearish rejection at a resistance level.

    Patterns (in order of strength):
    1. Rejection wick: upper wick > 0.5x ATR, close in lower half, level
       within the upper wick zone
    2. Bearish engulfing: body engulfs prior candle, closes below level
    3. Close-back: wicked above level but closed back below it
    """
    if atr <= 0:
        return CandleValidation(False, "none", 0.0, "no ATR")

    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    candle_range = high - low

    # level must be near the candle (within 0.5x ATR)
    if abs(high - level) > 0.5 * atr and abs(close - level) > 0.5 * atr:
        return CandleValidation(False, "none", 0.0, "level too far")

    # 1. rejection wick: long upper wick, close weak
    if upper_wick > 0.5 * atr and close < open_:
        # level should be in the wick zone (rejected from there)
        if low <= level <= high:
            strength = min(1.0, upper_wick / atr)
            return CandleValidation(True, "rejection_wick", strength,
                                    f"wick {upper_wick/atr:.2f} ATR")

    # 2. close-back: wicked above level, closed below
    if high > level and close < level and close < open_:
        strength = min(1.0, (high - level) / atr + 0.3)
        return CandleValidation(True, "close_back", strength,
                                "closed back below level")

    # 3. bearish body at resistance (weakest, needs other confluence)
    if close < open_ and body > 0.3 * atr and abs(close - level) < 0.3 * atr:
        return CandleValidation(True, "bearish_body", 0.4,
                                "bearish close at level")

    return CandleValidation(False, "none", 0.0, "no rejection")


def validate_bullish_rejection(open_: float, high: float, low: float,
                               close: float, level: float,
                               atr: float) -> CandleValidation:
    """Mirror of bearish: bullish rejection at a support level."""
    if atr <= 0:
        return CandleValidation(False, "none", 0.0, "no ATR")

    body = abs(close - open_)
    lower_wick = min(open_, close) - low

    if abs(low - level) > 0.5 * atr and abs(close - level) > 0.5 * atr:
        return CandleValidation(False, "none", 0.0, "level too far")

    # 1. rejection wick: long lower wick, close strong
    if lower_wick > 0.5 * atr and close > open_:
        if low <= level <= high:
            strength = min(1.0, lower_wick / atr)
            return CandleValidation(True, "rejection_wick", strength,
                                    f"wick {lower_wick/atr:.2f} ATR")

    # 2. close-back: wicked below level, closed above
    if low < level and close > level and close > open_:
        strength = min(1.0, (level - low) / atr + 0.3)
        return CandleValidation(True, "close_back", strength,
                                "closed back above level")

    # 3. bullish body at support
    if close > open_ and body > 0.3 * atr and abs(close - level) < 0.3 * atr:
        return CandleValidation(True, "bullish_body", 0.4,
                                "bullish close at level")

    return CandleValidation(False, "none", 0.0, "no rejection")


def choose_entry_type(setup_kind: str, validation: CandleValidation) -> EntryType:
    """Route to limit or market based on setup and validation.

    - S/R flip, OB retest: LIMIT at the zone (patient, price comes to you)
    - Breakout, MSS: MARKET on confirmation (aggressive, don't miss it)
    - Strong validation: can use limit (high confidence in the level)
    - Weak validation: market on confirm (need to see it happen)
    """
    patient_kinds = {"sr_flip", "ob_retest", "liquidity_retest"}
    aggressive_kinds = {"breakout", "turtle_soup", "v_reversal"}

    if setup_kind in patient_kinds:
        return EntryType.LIMIT_AT_ZONE
    if setup_kind in aggressive_kinds:
        return EntryType.MARKET_ON_CONFIRM
    # default: strong validation -> limit, weak -> market
    if validation.strength >= 0.7:
        return EntryType.LIMIT_AT_ZONE
    return EntryType.MARKET_ON_CONFIRM
