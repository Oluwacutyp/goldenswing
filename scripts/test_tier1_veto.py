"""Tier 1 vs Tier 3: does volatility expansion veto FVG setups? Does Daily
alignment amplify them? H4 FVG zones split two ways:

  A. Regime at formation: EXPANSION (ATR/ATR20 > 2.0, hysteresis exit 1.7)
     vs NORMAL. Expectation: expansion kills zone respect -> Tier 1 veto justified.
  B. Daily alignment: H4 zone within 0.5 ATR of a D1 swing level vs isolated.
     Expectation: aligned zones hold better -> HTF bonus justified.

Outcome: touch -> hold 0.5 ATR (respect) or close-through (violate).
"""
import sys
sys.path.insert(0, '.')

import csv
from goldenswing.v2_confluence.fvg import FVGTracker


def load_h1(path):
    ts, o, h, l, c = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(row['ts'])
            o.append(float(row['open'])); h.append(float(row['high']))
            l.append(float(row['low'])); c.append(float(row['close']))
    return ts, o, h, l, c


def resample(ts, o, h, l, c, n):
    ts2, o2, h2, l2, c2 = [], [], [], [], []
    for i in range(0, len(c) - n + 1, n):
        ts2.append(ts[i]); o2.append(o[i])
        h2.append(max(h[i:i+n])); l2.append(min(l[i:i+n])); c2.append(c[i+n-1])
    return ts2, o2, h2, l2, c2


def atr(h, l, c, i, n=14):
    if i < n:
        return 0.0
    return sum(max(h[k]-l[k], abs(h[k]-c[k-1]), abs(l[k]-c[k-1]))
               for k in range(i-n+1, i+1)) / n


def atr_ratio(h, l, c, i, fast=14, slow=20):
    """ATR[i] / SMA(ATR, slow). Hysteresis applied by caller."""
    if i < slow + 14:
        return 1.0
    now = atr(h, l, c, i, fast)
    past = sum(atr(h, l, c, k, fast) for k in range(i-slow+1, i+1)) / slow
    return now / past if past > 0 else 1.0


def d1_swings(h, l, i, lookback=20):
    """D1 swing high/low over lookback bars ending at i (causal)."""
    if i < lookback:
        return []
    wh, wl = h[i-lookback:i], l[i-lookback:i]
    return [(max(wh), i-lookback+wh.index(max(wh))),
            (min(wl), i-lookback+wl.index(min(wl)))]


def main():
    ts, o, h, l, c = load_h1('data/OANDA_XAU_USD_H1_20241001_20261012.csv')
    _, o4, h4, l4, c4 = resample(ts, o, h, l, c, 4)
    _, od, hd, ld, cd = resample(ts, o, h, l, c, 24)
    n4, nd = len(c4), len(cd)

    tracker = FVGTracker(max_age_bars=60)
    zones = []  # (zone, formed_idx)
    for i in range(n4):
        for z in tracker.on_bar(h4[i], l4[i], i):
            zones.append((z, i))

    # regime state with hysteresis, evaluated at formation
    # Calibrated: H4 99th pct ATR-ratio ~= 1.57. Enter expansion at 1.6, exit 1.4.
    expansion = False
    regime_at = {}
    for i in range(n4):
        r = atr_ratio(h4, l4, c4, i)
        if not expansion and r > 1.6:
            expansion = True
        elif expansion and r < 1.4:
            expansion = False
        regime_at[i] = expansion

    # outcome per zone
    def outcome(z, fi):
        a = atr(h4, l4, c4, fi)
        if a <= 0:
            return None
        for k in range(fi+1, min(fi+22, n4)):
            if z.direction == 1:
                if c4[k] < z.bottom: return (False, a)
                if h4[k] >= z.top + 0.5*a: return (True, a)
            else:
                if c4[k] > z.top: return (False, a)
                if l4[k] <= z.bottom - 0.5*a: return (True, a)
        return None

    # A: regime split
    for name, pred in (("EXPANSION", lambda fi: regime_at[fi]),
                       ("NORMAL", lambda fi: not regime_at[fi])):
        holds = viol = 0
        for z, fi in zones:
            oc = outcome(z, fi)
            if oc is None or not pred(fi):
                continue
            holds, viol = (holds+1, viol) if oc[0] else (holds, viol+1)
        tot = holds + viol
        print(f"Regime {name}: {tot} zones, respect "
              f"{100*holds/tot:.1f}% ({holds}/{tot})" if tot else f"Regime {name}: no zones")

    # B: daily alignment split (map H4 idx -> D1 idx)
    holds_a = viol_a = holds_i = viol_i = 0
    for z, fi in zones:
        oc = outcome(z, fi)
        if oc is None:
            continue
        respected, a = oc
        di = fi // 6  # 6 H4 bars per D1
        aligned = False
        if di < nd:
            for px, _ in d1_swings(hd, ld, di):
                if abs(px - z.mid) <= 0.5 * a:
                    aligned = True
                    break
        if aligned:
            holds_a, viol_a = (holds_a+1, viol_a) if respected else (holds_a, viol_a+1)
        else:
            holds_i, viol_i = (holds_i+1, viol_i) if respected else (holds_i, viol_i+1)
    ta, ti = holds_a+viol_a, holds_i+viol_i
    print(f"Daily-aligned: {ta} zones, respect {100*holds_a/ta:.1f}%")
    print(f"Isolated 4H:   {ti} zones, respect {100*holds_i/ti:.1f}%")


if __name__ == '__main__':
    main()
