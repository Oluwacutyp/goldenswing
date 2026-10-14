"""FVG x Confluence backtest integration.

Architecture: the simulation loop feeds one closed bar at a time into
FVGTracker.on_bar(high, low, idx). After feeding bar i, tracker state
reflects everything known through bar i — so signal evaluation at bar i
is causal by construction. No lookahead possible.

At each bar:
  1. tracker.on_bar(h[i], l[i], i)          # update zones + lifecycle
  2. factors = fvg_factors(tracker)          # active zones -> Factor objects
  3. factors += structure_factors(...)       # swing levels (causal)
  4. zone = score_confluence(...)            # independence-aware scoring

Evaluation: for every FVG zone formed, did price respect it (touch zone
then move >= 0.5 ATR favorably within 24 bars) or violate it (close
through)? Split by confluence score.
"""
import sys
sys.path.insert(0, '.')

import csv
from goldenswing.v2_confluence.fvg import FVGTracker
from goldenswing.v2_confluence.engine import Factor, score_confluence


def load_ohlc(path):
    ts, o, h, l, c = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(row['ts'])
            o.append(float(row['open'])); h.append(float(row['high']))
            l.append(float(row['low'])); c.append(float(row['close']))
    return ts, o, h, l, c


def atr(high, low, close, i, n=14):
    if i < n:
        return 0.0
    trs = []
    for k in range(i - n + 1, i + 1):
        trs.append(max(high[k] - low[k],
                       abs(high[k] - close[k - 1]),
                       abs(low[k] - close[k - 1])))
    return sum(trs) / n


def swing_levels(high, low, i, lookback=50):
    """Causal swing highs/lows over lookback bars ending at i. Returns Factors."""
    if i < lookback:
        return []
    factors = []
    win_h, win_l = high[i - lookback:i], low[i - lookback:i]
    sh, sl = max(win_h), min(win_l)
    # anchor = bar where the extreme occurred
    ah = i - lookback + win_h.index(sh)
    al = i - lookback + win_l.index(sl)
    factors.append(Factor('structure', sh, ah, 'swing_high_resistance'))
    factors.append(Factor('structure', sl, al, 'swing_low_support'))
    return factors


def fvg_factors(tracker: FVGTracker):
    """Active FVG zones -> confluence Factors."""
    out = []
    for z in tracker.active():
        label = 'bullish' if z.direction == 1 else 'bearish'
        out.append(Factor('fvg', z.mid, z.anchor_idx, label))
    return out


def main():
    path = 'data/OANDA_XAU_USD_H1_20241001_20261012.csv'
    ts, o, h, l, c = load_ohlc(path)
    n = len(c)
    print(f"Loaded {n} H1 bars")

    tracker = FVGTracker(max_age_bars=100)
    results = []  # (score, respected: bool)

    # Track zones formed during the loop for outcome evaluation
    pending = {}  # formed_idx -> (zone, entry_bar)

    for i in range(n):
        # 1. Feed the closed bar — tracker state now causal through bar i
        new_zones = tracker.on_bar(h[i], l[i], i)

        # 2-4. Build factors and score confluence at current price
        a = atr(h, l, c, i)
        if a <= 0:
            continue
        factors = fvg_factors(tracker) + swing_levels(h, l, i)
        zone = score_confluence(
            price=c[i],
            structure=[f for f in factors if f.category == 'structure'],
            fvg=[f for f in factors if f.category == 'fvg'],
            atr=a,
        )

        # Record newly formed zones for outcome tracking
        for z in new_zones:
            # confluence score AT formation (causal), evaluated at zone mid
            fz = [Factor('fvg', z.mid, z.anchor_idx,
                         'bullish' if z.direction == 1 else 'bearish')]
            sz = swing_levels(h, l, i)
            zscore = score_confluence(price=z.mid, structure=sz, fvg=fz, atr=a)
            pending[z.formed_idx] = (z, zscore.score, i, False)  # 4th: touched?

        # 5. Evaluate pending zones:
        #    - must first TRADE INTO the zone (touch)
        #    - then: hold (favorable 0.5 ATR move) or break (close through)?
        for fidx in list(pending.keys()):
            z, score, entry, touched = pending[fidx]
            if i <= entry:
                continue
            if not touched:
                # check touch: bar traded into the zone
                if z.direction == 1 and l[i] <= z.top and h[i] >= z.bottom:
                    pending[fidx] = (z, score, entry, True)
                elif z.direction == -1 and h[i] >= z.bottom and l[i] <= z.top:
                    pending[fidx] = (z, score, entry, True)
                elif i - entry >= 48:
                    del pending[fidx]  # never touched: no trade, no outcome
                continue
            # touched: hold or break?
            a_now = atr(h, l, c, i) or a
            if z.direction == 1:
                if c[i] < z.bottom:
                    results.append((score, False)); del pending[fidx]
                elif h[i] >= z.top + 0.5 * a_now:
                    results.append((score, True)); del pending[fidx]
            else:
                if c[i] > z.top:
                    results.append((score, False)); del pending[fidx]
                elif l[i] <= z.bottom - 0.5 * a_now:
                    results.append((score, True)); del pending[fidx]
            if fidx in pending and i - entry >= 48:
                del pending[fidx]  # touched but neither: inconclusive

    # Report by confluence score
    from collections import defaultdict
    by_score = defaultdict(lambda: [0, 0])  # score -> [respected, violated]
    for score, respected in results:
        by_score[score][0 if respected else 1] += 1

    print(f"\nEvaluated {len(results)} touched FVG zones (hold=+0.5 ATR, break=close through):")
    print(f"{'Score':<8} {'N':<6} {'Respect%':<10}")
    for s in sorted(by_score):
        r, v = by_score[s]
        tot = r + v
        print(f"{s:<8} {tot:<6} {100*r/tot:<10.1f}")


if __name__ == '__main__':
    main()
