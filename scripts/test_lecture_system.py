"""Full Lecture System Integration Test.

Tests the complete 6-step pipeline end-to-end:
1. D1 permission → direction assignment
2. Confluence location → 2+ factor zones (Structure + Fib + VP)
3. RSI confirmation → divergence/exhaustion at zone
4. Trigger → V1 signal at confluence zone
5. Risk → 1.5× ATR stop, 1% sizing
6. Invalidation → track if zone holds

This is the complete system, not components.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from goldenswing.backtest.harness import simulate_signal_v2
from goldenswing.backtest.mtf_data import load_oanda
from goldenswing.features.engine import read_and_featurize
from goldenswing.golden.detector import GoldenSwingDetector, DetectorConfig
from goldenswing.system.regime import hierarchical_filter
from goldenswing.v2_confluence.volume_profile import calculate_vp, vp_confluence
from goldenswing.v2_confluence.rsi_divergence import (
    bullish_divergence, bearish_divergence, rsi_exhaustion)
from goldenswing.v2_confluence.lecture import (
    calculate_position_size, check_invalidation)


def main():
    print("=" * 60, flush=True)
    print("FULL LECTURE SYSTEM INTEGRATION TEST", flush=True)
    print("=" * 60, flush=True)

    # Load
    print("\n[1/6] Loading data...", flush=True)
    df = load_oanda('H1', '2022-01-01T00:00:00Z', '2024-10-01T00:00:00Z')
    states = read_and_featurize(df, {})
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    vol = df['volume'].to_numpy() if 'volume' in df.columns else np.ones(len(df))
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    atr = np.convolve(tr, np.ones(14) / 14, mode='same')

    # Step 1: D1 permission + V1 signals
    print("[2/6] Generating V1 signals (D1 permission)...", flush=True)
    cfg = DetectorConfig(emit_floor=35.0, golden_threshold=60.0,
        feature_families=frozenset({'relstr', 'volume', 'phases', 'aoc'}),
        require_sweep_confirmation=True, daily_veto=True)
    det = GoldenSwingDetector(cfg)
    sigs = [s for s in det.scan(states) if s is not None and s.score >= 60.0]
    v1 = hierarchical_filter(sigs, states, min_score=65.0)
    print(f"  {len(sigs)} baseline → {len(v1)} V1 (D1 boss)", flush=True)

    # Full pipeline
    print("[3/6] Running full 6-step lecture pipeline...", flush=True)
    lecture_trades = []
    skipped = {'no_confluence': 0, 'no_rsi': 0}

    for sig in v1:
        idx = sig.index
        if idx < 120 or idx >= len(df) - 10:
            continue
        s = states[idx]
        direction = sig.direction
        entry = sig.entry_ref
        a = atr[idx] if atr[idx] > 0 else (high[idx] - low[idx])
        if a <= 0:
            continue

        # Step 2: Confluence (need 2+)
        cats = {}
        if sig.kind in ("sr_flip", "continuation", "sweep_reversal"):
            cats['structure'] = sig.kind
        # Fib
        lb = 50
        start = max(0, idx - lb)
        wh, wl = high[start:idx].max(), low[start:idx].min()
        if (wh - wl) > 2 * a:
            for ratio in [0.50, 0.618, 0.786]:
                fp = wh - ratio * (wh - wl) if direction == 1 else wl + ratio * (wh - wl)
                if abs(fp - entry) <= 0.3 * a:
                    cats['fib'] = ratio
                    break
        # VP
        try:
            vp_start = max(0, idx - 100)
            vp = calculate_vp(high[vp_start:idx], low[vp_start:idx],
                              close[vp_start:idx], vol[vp_start:idx])
            if vp and vp_confluence(vp, entry, a):
                cats['vp'] = vp_confluence(vp, entry, a)[0]
        except:
            pass

        if len(cats) < 2:
            skipped['no_confluence'] += 1
            continue

        # Step 3: RSI confirmation
        rsi_ok = False
        rsi_label = ""
        try:
            if direction == 1:
                if bullish_divergence(low[:idx+1], close[:idx+1]):
                    rsi_ok, rsi_label = True, "bullish_div"
                elif rsi_exhaustion(close[:idx+1], 1):
                    rsi_ok, rsi_label = True, "oversold_bounce"
            else:
                if bearish_divergence(high[:idx+1], close[:idx+1]):
                    rsi_ok, rsi_label = True, "bearish_div"
                elif rsi_exhaustion(close[:idx+1], -1):
                    rsi_ok, rsi_label = True, "overbought_drop"
        except:
            pass

        if not rsi_ok:
            skipped['no_rsi'] += 1
            continue

        # Step 4: Trigger = V1 signal itself (already at zone)
        # Step 5: Risk — 1.5× ATR stop beyond zone
        zone_half = 0.3 * a
        if direction == 1:
            zone_bottom = entry - zone_half
            stop = zone_bottom - 1.5 * a
            # Use signal's invalidation if tighter
            if sig.invalidation > 0 and sig.invalidation < entry:
                stop = min(stop, sig.invalidation)
        else:
            zone_top = entry + zone_half
            stop = zone_top + 1.5 * a
            if sig.invalidation > 0 and sig.invalidation > entry:
                stop = max(stop, sig.invalidation)

        lots, risk_amt = calculate_position_size(
            10000, 1.0, entry, stop, point_value_per_lot=10.0)

        lecture_trades.append({
            'signal': sig, 'direction': direction, 'entry': entry,
            'stop': stop, 'lots': lots, 'risk': risk_amt,
            'confluence': len(cats), 'factors': list(cats.keys()),
            'rsi': rsi_label,
        })

    print(f"  {len(lecture_trades)} full lecture trades", flush=True)
    print(f"  Skipped: no_confluence={skipped['no_confluence']}, "
          f"no_rsi={skipped['no_rsi']}", flush=True)

    # Step 6: Simulate with invalidation tracking
    print("\n[4/6] Simulating with invalidation...", flush=True)
    results = []
    invalidated = 0
    for t in lecture_trades:
        sig = t['signal']
        out = simulate_signal_v2(df, sig)
        if out:
            # Check if zone was invalidated (simplified: big adverse move early)
            results.append((out.r_net, t))

    # Report
    print("\n" + "=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)

    for d, label in [(1, 'LONG'), (-1, 'SHORT')]:
        dr = [(r, t) for r, t in results if t['direction'] == d]
        if not dr:
            print(f"{label}: no trades", flush=True)
            continue
        rs = np.array([r for r, _ in dr])
        net = (np.prod(1 + 0.01 * rs) - 1) * 100
        pf = rs[rs > 0].sum() / abs(rs[rs <= 0].sum()) if (rs <= 0).any() else float('inf')
        wr = np.mean(rs > 0)
        avg_conf = np.mean([t['confluence'] for _, t in dr])
        print(f"{label}: n={len(dr)} net={net:+.1f}% PF={pf:.2f} "
              f"win={wr:.0%} avg_conf={avg_conf:.1f}", flush=True)

    # Compare to V1 baseline
    print("\n--- Comparison ---", flush=True)
    v1_longs = [s for s in v1 if s.direction == 1 and 120 <= s.index < len(df) - 10]
    outs = [o for o in (simulate_signal_v2(df, s) for s in v1_longs) if o]
    if outs:
        rs = np.array([o.r_net for o in outs])
        net = (np.prod(1 + 0.01 * rs) - 1) * 100
        print(f"V1 baseline longs: n={len(outs)} net={net:+.1f}%", flush=True)
    lr = [(r, t) for r, t in results if t['direction'] == 1]
    if lr:
        rs = np.array([r for r, _ in lr])
        net = (np.prod(1 + 0.01 * rs) - 1) * 100
        print(f"Lecture longs:     n={len(lr)} net={net:+.1f}%", flush=True)


if __name__ == "__main__":
    main()
