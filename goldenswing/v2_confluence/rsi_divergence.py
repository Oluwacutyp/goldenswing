"""RSI divergence detector for the lecture framework.

Bullish divergence: price makes lower low, RSI makes higher low
→ sellers exhausted, bounce likely.

Bearish divergence: price makes higher high, RSI makes lower high
→ buyers exhausted, drop likely.

Used at confluence zones to confirm institutional defense.
"""
from __future__ import annotations

import numpy as np


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.convolve(gain, np.ones(period) / period, mode='same')
    avg_loss = np.convolve(loss, np.ones(period) / period, mode='same')
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    return 100 - (100 / (1 + rs))


def find_swing_lows(low: np.ndarray, n: int = 3) -> list[int]:
    """Find fractal swing low indices."""
    out = []
    for i in range(n, len(low) - n):
        if low[i] == low[i-n:i+n+1].min():
            out.append(i)
    return out


def find_swing_highs(high: np.ndarray, n: int = 3) -> list[int]:
    """Find fractal swing high indices."""
    out = []
    for i in range(n, len(high) - n):
        if high[i] == high[i-n:i+n+1].max():
            out.append(i)
    return out


def bullish_divergence(low: np.ndarray, close: np.ndarray,
                       lookback: int = 50) -> bool:
    """Check for bullish RSI divergence in recent bars.

    Price: lower low. RSI: higher low. Sellers exhausted.
    """
    if len(close) < lookback + 14:
        return False
    r = rsi(close)
    lows = find_swing_lows(low[-lookback:])
    if len(lows) < 2:
        return False
    # Last two swing lows (in window coordinates)
    l1, l2 = lows[-2], lows[-1]
    # Convert to full array indices
    offset = len(close) - lookback
    i1, i2 = l1 + offset, l2 + offset
    # Price lower low, RSI higher low
    price_ll = low[i2] < low[i1]
    rsi_hl = r[i2] > r[i1]
    return price_ll and rsi_hl


def bearish_divergence(high: np.ndarray, close: np.ndarray,
                       lookback: int = 50) -> bool:
    """Check for bearish RSI divergence in recent bars.

    Price: higher high. RSI: lower high. Buyers exhausted.
    """
    if len(close) < lookback + 14:
        return False
    r = rsi(close)
    highs = find_swing_highs(high[-lookback:])
    if len(highs) < 2:
        return False
    h1, h2 = highs[-2], highs[-1]
    offset = len(close) - lookback
    i1, i2 = h1 + offset, h2 + offset
    price_hh = high[i2] > high[i1]
    rsi_lh = r[i2] < r[i1]
    return price_hh and rsi_lh


def rsi_exhaustion(close: np.ndarray, direction: int,
                  overbought: float = 80.0, oversold: float = 20.0) -> bool:
    """Check RSI exhaustion (for gold: 80/20 bands).

    Long: RSI was oversold (<20) and is curling up.
    Short: RSI was overbought (>80) and is curling down.
    """
    if len(close) < 20:
        return False
    r = rsi(close)
    if direction == 1:
        # Was oversold recently, now rising
        was_os = (r[-10:] < oversold).any()
        curling = r[-1] > r[-3]
        return was_os and curling
    else:
        was_ob = (r[-10:] > overbought).any()
        curling = r[-1] < r[-3]
        return was_ob and curling
