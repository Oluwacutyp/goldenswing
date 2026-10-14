#!/usr/bin/env python3
"""V-reversal (V-top) short experiment.

From practitioner mining: the parabolic-exhaustion short.
- Parabolic rise: price accelerates upward (5+ bars, >2x ATR gain)
- Climax bar: large range (>1.5x ATR), the blowoff top
- Trigger: next bar closes below the climax bar's low
- Entry: at the trigger close (or next open)
- Stop: above the climax high
- Target: 2R (or measured move)

This is a momentum-exhaustion pattern, distinct from liquidity-sweep
(Turtle Soup) and S/R-flip approaches. The user specifically flagged
the reverse-V as a candidate short strategy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.mtf_data import load_oanda

START = "2022-01-01T00:00:00Z"
END = "2024-10-01T00:00:00Z"


def find_v_tops(df: pd.DataFrame) -> list[dict]:
    """Find V-top reversal setups."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    # ATR(14) for scale
    tr = np.maximum(high - low,
                    np.maximum(abs(high - np.roll(close, 1)),
                               abs(low - np.roll(close, 1))))
    atr = pd.Series(tr).rolling(14).mean().values

    setups = []
    for i in range(10, len(df) - 2):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        # 1. parabolic rise: close[i] > close[i-5] by > 2x ATR
        #    and accelerating (last 2 bars gain > first 3 bars gain)
        gain_5 = close[i] - close[i - 5]
        if gain_5 < 2.0 * a:
            continue
        gain_last2 = close[i] - close[i - 2]
        gain_first3 = close[i - 2] - close[i - 5]
        if gain_last2 < gain_first3 * 1.2:  # must be accelerating
            continue
        # 2. climax bar: range > 1.5x ATR (blowoff)
        climax_range = high[i] - low[i]
        if climax_range < 1.5 * a:
            continue
        # 3. trigger: next bar closes below climax low
        if close[i + 1] < low[i]:
            setups.append({
                "index": i + 1,
                "entry": float(close[i + 1]),
                "stop": float(high[i] + 0.2 * a),  # above climax high
                "atr": float(a),
                "climax_high": float(high[i]),
                "climax_low": float(low[i]),
            })
    return setups


def main() -> None:
    print("loading H1 ...", flush=True)
    df = load_oanda("H1", START, END)
    setups = find_v_tops(df)
    print(f"{len(setups)} V-top setups", flush=True)

    results = []
    for s in setups:
        entry = s["entry"]
        stop = s["stop"]
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        r = None
        for k in range(s["index"], min(s["index"] + 72, len(df))):
            hi, lo = df["high"].iloc[k], df["low"].iloc[k]
            if hi >= stop:
                r = (entry - stop) / risk  # stopped = -1R
                break
            if lo <= entry - 2 * risk:  # 2R target
                r = 2.0
                break
        if r is None:
            r = (entry - df["close"].iloc[min(s["index"] + 71,
                                             len(df) - 1)]) / risk
        results.append(r - 0.02)  # cost

    if results:
        rs = np.array(results)
        net = (np.prod(1 + 0.01 * rs) - 1) * 100
        pf = rs[rs > 0].sum() / abs(rs[rs <= 0].sum()) \
            if (rs <= 0).any() else float("inf")
        print(f"V-TOP SHORT: n={len(rs)} net={net:.1f}% PF={pf:.2f} "
              f"win={np.mean(rs > 0):.0%} avgR={rs.mean():.3f}", flush=True)
    else:
        print("no V-top setups found", flush=True)


if __name__ == "__main__":
    main()
