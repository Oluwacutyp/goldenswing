"""FVG entry A/B test: limit-at-touch vs. confirmation.

Question: when price reaches an active FVG zone, fire immediately or wait?

  Entry A (limit):  on first touch, fill at zone edge. No confirmation.
  Entry B (confirm): on first touch, wait for a rejection signal —
                    next bar closes back above zone top (bullish) with a
                    lower wick into the zone. Enter at that bar's close.
                    No signal within 5 bars -> skip.

Shared (isolates the entry variable):
  Stop:        beyond zone extreme + 0.25 ATR (catastrophic backstop)
  Invalidation: close through the zone -> exit immediately at close
  Target:      1.5R fixed
  Risk:        1R per trade, results in R multiples

Causal throughout: tracker fed bar-by-bar, signals only from closed bars.
"""
import sys
sys.path.insert(0, '.')

import csv
from goldenswing.v2_confluence.fvg import FVGTracker


def load_ohlc(path):
    ts, o, h, l, c = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(row['ts'])
            o.append(float(row['open'])); h.append(float(row['high']))
            l.append(float(row['low'])); c.append(float(row['close']))
    return ts, o, h, l, c


def atr(h, l, c, i, n=14):
    if i < n:
        return 0.0
    return sum(max(h[k] - l[k], abs(h[k] - c[k-1]), abs(l[k] - c[k-1]))
               for k in range(i - n + 1, i + 1)) / n


def simulate(zones_events, o, h, l, c, mode, min_pen_atr=0.0, offset_atr=0.0):
    """zones_events: list of (zone, touch_idx). Returns list of R multiples.

    Fill realism via two mechanisms (use one, not both):
      min_pen_atr: limit at edge; fills ONLY if touch bar penetrates the edge
                   by >= min_pen_atr (queue-priority proxy).
      offset_atr:  limit placed offset_atr INSIDE the zone; fills iff the bar
                   trades at-or-through the limit price. The offset IS the
                   certainty — price demonstrably traded there.
    Fill price is always the limit price.
    """
    n = len(c)
    results = []
    for zone, touch_idx in zones_events:
        a = atr(h, l, c, touch_idx)
        if a <= 0:
            continue
        buf = 0.25 * a
        if zone.direction == 1:  # bullish
            edge, extreme = zone.top, zone.bottom
            limit = edge - offset_atr * a      # inside = below the top
        else:
            edge, extreme = zone.bottom, zone.top
            limit = edge + offset_atr * a      # inside = above the bottom

        # --- entry ---
        if mode == 'limit':
            if offset_atr > 0:
                filled = (l[touch_idx] <= limit) if zone.direction == 1 \
                    else (h[touch_idx] >= limit)
            else:
                min_pen = min_pen_atr * a
                filled = (l[touch_idx] <= edge - min_pen) if zone.direction == 1 \
                    else (h[touch_idx] >= edge + min_pen)
            if not filled:
                continue  # missed: queue never cleared / level never reached
            entry_px = limit
            entry_idx = touch_idx
        else:  # confirmation
            entry_px, entry_idx = None, None
            for j in range(touch_idx + 1, min(touch_idx + 6, n)):
                touched = (l[j] <= zone.top and h[j] >= zone.bottom) if zone.direction == 1 \
                    else (h[j] >= zone.bottom and l[j] <= zone.top)
                if not touched:
                    continue
                if zone.direction == 1:
                    # rejection: wick into zone, close back above top
                    if l[j] < zone.top and c[j] > zone.top and c[j] > o[j]:
                        entry_px, entry_idx = c[j], j
                        break
                else:
                    if h[j] > zone.bottom and c[j] < zone.bottom and c[j] < o[j]:
                        entry_px, entry_idx = c[j], j
                        break
            if entry_px is None:
                continue  # no confirmation: no trade

        # --- stop / target ---
        # ABSOLUTE geometry: stop and target fixed from the zone, not from entry.
        # This isolates the offset's true cost: worse entry = smaller R per
        # winner, larger R per loser, with stop always -1R by construction.
        # (R-defined targets would mechanically pull closer with worse entries,
        #  flattering the offset — a measurement artifact, not edge.)
        if zone.direction == 1:
            stop = extreme - buf
            target = edge + 1.0 * a
            risk = entry_px - stop
        else:
            stop = extreme + buf
            target = edge - 1.0 * a
            risk = stop - entry_px
        if risk <= 0:
            continue

        # --- walk forward ---
        r = None
        for k in range(entry_idx + 1, min(entry_idx + 49, n)):
            # invalidation first: close through the zone
            if zone.direction == 1 and c[k] < zone.bottom:
                r = (c[k] - entry_px) / risk
                break
            if zone.direction == -1 and c[k] > zone.top:
                r = (entry_px - c[k]) / risk
                break
            # stop / target (stop-first on ambiguity)
            if zone.direction == 1:
                if l[k] <= stop:
                    r = (stop - entry_px) / risk
                    break
                if h[k] >= target:
                    r = 1.5
                    break
            else:
                if h[k] >= stop:
                    r = (entry_px - stop) / risk
                    break
                if l[k] <= target:
                    r = 1.5
                    break
        if r is not None:
            results.append(r)
    return results


def summarize(name, rs):
    n = len(rs)
    if not n:
        print(f"{name}: no trades")
        return
    wins = [r for r in rs if r > 0]
    tot = sum(rs)
    pf_num = sum(wins)
    pf_den = -sum(r for r in rs if r <= 0) or 1e-9
    print(f"{name}: {n} trades | total {tot:+.1f}R | win {100*len(wins)/n:.1f}% | "
          f"PF {pf_num/pf_den:.2f} | avg {tot/n:+.3f}R")


def main():
    path = 'data/OANDA_XAU_USD_H1_20241001_20261012.csv'
    ts, o, h, l, c = load_ohlc(path)
    n = len(c)

    tracker = FVGTracker(max_age_bars=100)
    touch_events = []  # (zone, touch_idx) — first touch only

    for i in range(n):
        new_zones = tracker.on_bar(h[i], l[i], i)
        # detect first touch of any active zone
        for z in tracker.active():
            if i <= z.formed_idx:
                continue
            touched = (l[i] <= z.top and h[i] >= z.bottom) if z.direction == 1 \
                else (h[i] >= z.bottom and l[i] <= z.top)
            if touched and not any(ev[0] is z for ev in touch_events):
                touch_events.append((z, i))

    print(f"{len(touch_events)} first-touch events")
    print("--- fill-realism via penetration threshold (limit at edge) ---")
    for pen in (0.0, 0.05, 0.10, 0.20):
        rs = simulate(touch_events, o, h, l, c, 'limit', min_pen_atr=pen)
        summarize(f"pen>={pen:.2f} ATR", rs)
    print("--- offset inside zone (offset IS the fill certainty) ---")
    for off in (0.0, 0.02, 0.05, 0.10):
        rs = simulate(touch_events, o, h, l, c, 'limit', offset_atr=off)
        summarize(f"offset {off:.2f} ATR", rs)


if __name__ == '__main__':
    main()
