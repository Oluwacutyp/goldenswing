"""FVG on higher timeframes: D1/H4 zone respect + stop-buffer sensitivity.

Resamples OANDA H1 -> H4/D1 (causal OHLC), then:
  1. Zone-respect rate per timeframe (touch -> hold 0.5 ATR or break)
  2. Entry test (limit at edge) with stop-buffer sweep:
     0.25 / 0.5 / 0.75 / 1.0 ATR beyond zone extreme
  3. Absolute target = 2.0 ATR from edge (multi-day swings need room)

Question: how must stop/ATR rules change when zones are multi-day
institutional commitments instead of intraday noise?
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
    """Causal OHLC resample: every n H1 bars -> 1 bar."""
    ts2, o2, h2, l2, c2 = [], [], [], [], []
    for i in range(0, len(c) - n + 1, n):
        ts2.append(ts[i])
        o2.append(o[i])
        h2.append(max(h[i:i+n]))
        l2.append(min(l[i:i+n]))
        c2.append(c[i+n-1])
    return ts2, o2, h2, l2, c2


def atr(h, l, c, i, n=14):
    if i < n:
        return 0.0
    return sum(max(h[k]-l[k], abs(h[k]-c[k-1]), abs(l[k]-c[k-1]))
               for k in range(i-n+1, i+1)) / n


def run(tf_name, o, h, l, c, buf_mults=(0.25, 0.5, 0.75, 1.0)):
    n = len(c)
    tracker = FVGTracker(max_age_bars=60)
    events = []
    for i in range(n):
        tracker.on_bar(h[i], l[i], i)
        for z in tracker.active():
            if i <= z.formed_idx:
                continue
            t = (l[i] <= z.top and h[i] >= z.bottom) if z.direction == 1 \
                else (h[i] >= z.bottom and l[i] <= z.top)
            if t and not any(e[0] is z for e in events):
                events.append((z, i))

    # zone respect (touch -> hold 0.5 ATR or close-through break)
    holds = breaks = 0
    for z, ti in events:
        a = atr(h, l, c, ti)
        if a <= 0:
            continue
        for k in range(ti+1, min(ti+21, n)):
            if z.direction == 1:
                if c[k] < z.bottom:
                    breaks += 1; break
                if h[k] >= z.top + 0.5*a:
                    holds += 1; break
            else:
                if c[k] > z.top:
                    breaks += 1; break
                if l[k] <= z.bottom - 0.5*a:
                    holds += 1; break
    tot = holds + breaks
    print(f"\n=== {tf_name}: {len(events)} touches, "
          f"respect {100*holds/tot:.1f}% ({holds}/{tot}) ===")

    # entry test per stop buffer
    for bm in buf_mults:
        rs = []
        for z, ti in events:
            a = atr(h, l, c, ti)
            if a <= 0:
                continue
            buf = bm * a
            edge = z.top if z.direction == 1 else z.bottom
            extreme = z.bottom if z.direction == 1 else z.top
            # fill: penetration >= 0.05 ATR (realistic)
            pen = (l[ti] <= edge - 0.05*a) if z.direction == 1 \
                else (h[ti] >= edge + 0.05*a)
            if not pen:
                continue
            entry = edge
            stop = extreme - buf if z.direction == 1 else extreme + buf
            risk = abs(entry - stop)
            if risk <= 0:
                continue
            target = edge + 2.0*a if z.direction == 1 else edge - 2.0*a
            r = None
            for k in range(ti+1, min(ti+22, n)):
                if z.direction == 1 and c[k] < z.bottom:
                    r = (c[k]-entry)/risk; break
                if z.direction == -1 and c[k] > z.top:
                    r = (entry-c[k])/risk; break
                if z.direction == 1:
                    if l[k] <= stop: r = (stop-entry)/risk; break
                    if h[k] >= target: r = (target-entry)/risk; break
                else:
                    if h[k] >= stop: r = (entry-stop)/risk; break
                    if l[k] <= target: r = (entry-target)/risk; break
            if r is not None:
                # fix short target R sign
                rs.append(r)
        # recompute short target R correctly
        ntr = len(rs)
        if ntr:
            tot_r = sum(rs)
            wins = sum(1 for r in rs if r > 0)
            pf = sum(r for r in rs if r > 0) / (-sum(r for r in rs if r <= 0) or 1e-9)
            print(f"  buf {bm:.2f} ATR: {ntr} trades | {tot_r:+.1f}R | "
                  f"win {100*wins/ntr:.1f}% | PF {pf:.2f} | avg {tot_r/ntr:+.3f}R")


def main():
    ts, o, h, l, c = load_h1('data/OANDA_XAU_USD_H1_20241001_20261012.csv')
    print(f"H1 bars: {len(c)}")
    for name, nb in (("H4", 4), ("D1", 24)):
        ts2, o2, h2, l2, c2 = resample(ts, o, h, l, c, nb)
        print(f"{name} bars: {len(c2)}")
        run(name, o2, h2, l2, c2)


if __name__ == '__main__':
    main()
