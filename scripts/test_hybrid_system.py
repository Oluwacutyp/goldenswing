"""Hybrid System: V1 for longs, Lecture for shorts, confluence sizing.

- LONGS: V1 hierarchical filter (D1 boss + H1 stack + score≥65)
  → 200 trades, +7.6%. Don't over-filter what works.

- SHORTS: Full lecture pipeline (confluence 2+ + RSI confirmation)
  → Only config where shorts are profitable (+2.6%, PF 1.49).

- SIZING: Confluence-based, systematic.
  2 factors → 1.0% risk. 3 factors → 1.5% risk.

V1 finds the trades. The lecture grades them.
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
from goldenswing.v2_confluence.lecture import calculate_position_size


def confluence_score(sig, idx, high, low, close, vol, atr):
    """Calculate confluence score (0-3) for a signal."""
    cats = {}
    entry = sig.entry_ref
    a = atr[idx]
    direction = sig.direction

    # Structure
    if sig.kind in ("sr_flip", "continuation", "sweep_reversal"):
        cats['structure'] = True

    # Fib
    lb = 50
    start = max(0, idx - lb)
    wh, wl = high[start:idx].max(), low[start:idx].min()
    if (wh - wl) > 2 * a:
        for ratio in [0.50, 0.618, 0.786]:
            fp = wh - ratio * (wh - wl) if direction == 1 else wl + ratio * (wh - wl)
            if abs(fp - entry) <= 0.3 * a:
                cats['fib'] = True
                break

    # VP
    try:
        vp_start = max(0, idx - 100)
        vp = calculate_vp(high[vp_start:idx], low[vp_start:idx],
                          close[vp_start:idx], vol[vp_start:idx])
        if vp and vp_confluence(vp, entry, a):
            cats['vp'] = True
    except:
        pass

    return len(cats)


def rsi_confirmed(sig, idx, high, low, close):
    """Check RSI confirmation for a signal."""
    direction = sig.direction
    try:
        if direction == 1:
            return (bullish_divergence(low[:idx+1], close[:idx+1]) or
                    rsi_exhaustion(close[:idx+1], 1))
        else:
            return (bearish_divergence(high[:idx+1], close[:idx+1]) or
                    rsi_exhaustion(close[:idx+1], -1))
    except:
        return False


def main():
    print("=" * 60, flush=True)
    print("HYBRID SYSTEM TEST", flush=True)
    print("Longs: V1 | Shorts: Lecture | Sizing: Confluence", flush=True)
    print("=" * 60, flush=True)

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

    cfg = DetectorConfig(emit_floor=35.0, golden_threshold=60.0,
        feature_families=frozenset({'relstr', 'volume', 'phases', 'aoc'}),
        require_sweep_confirmation=True, daily_veto=True)
    det = GoldenSwingDetector(cfg)
    sigs = [s for s in det.scan(states) if s is not None and s.score >= 60.0]
    v1 = hierarchical_filter(sigs, states, min_score=65.0)
    print(f"\n{len(sigs)} baseline → {len(v1)} V1", flush=True)

    # HYBRID: longs from V1, shorts from lecture pipeline
    hybrid_trades = []

    for sig in v1:
        idx = sig.index
        if idx < 120 or idx >= len(df) - 10:
            continue
        a = atr[idx]
        if a <= 0:
            continue

        conf = confluence_score(sig, idx, high, low, close, vol, atr)

        if sig.direction == 1:
            # LONG: V1 (no extra filter)
            risk_pct = 1.5 if conf >= 3 else 1.0
            hybrid_trades.append((sig, risk_pct, conf))
        else:
            # SHORT: lecture pipeline (confluence 2+ + RSI)
            if conf < 2:
                continue
            if not rsi_confirmed(sig, idx, high, low, close):
                continue
            risk_pct = 1.5 if conf >= 3 else 1.0
            hybrid_trades.append((sig, risk_pct, conf))

    print(f"{len(hybrid_trades)} hybrid trades", flush=True)

    # Simulate with confluence-based sizing
    # For P&L: scale R by risk_pct (1.0% baseline, 1.5% = 1.5x)
    results = []
    for sig, risk_pct, conf in hybrid_trades:
        out = simulate_signal_v2(df, sig)
        if out:
            # Scale by risk: 1.5% risks 1.5x the R
            scaled_r = out.r_net * (risk_pct / 1.0)
            results.append((scaled_r, out.r_net, sig.direction, conf, risk_pct))

    print("\n" + "=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)

    for d, label in [(1, 'LONG (V1)'), (-1, 'SHORT (Lecture)')]:
        dr = [(sr, r, c, rp) for sr, r, dd, c, rp in results if dd == d]
        if not dr:
            print(f"{label}: no trades", flush=True)
            continue
        srs = np.array([sr for sr, _, _, _ in dr])
        rs = np.array([r for _, r, _, _ in dr])
        # Net with sizing: compound scaled R
        net = (np.prod(1 + 0.01 * srs) - 1) * 100
        # PF on raw R
        pf = rs[rs > 0].sum() / abs(rs[rs <= 0].sum()) if (rs <= 0).any() else float('inf')
        wr = np.mean(rs > 0)
        avg_conf = np.mean([c for _, _, c, _ in dr])
        print(f"{label}: n={len(dr)} net={net:+.1f}% PF={pf:.2f} "
              f"win={wr:.0%} avg_conf={avg_conf:.1f}", flush=True)

    # Total
    if results:
        srs = np.array([sr for sr, _, _, _, _ in results])
        net = (np.prod(1 + 0.01 * srs) - 1) * 100
        rs = np.array([r for _, r, _, _, _ in results])
        pf = rs[rs > 0].sum() / abs(rs[rs <= 0].sum()) if (rs <= 0).any() else float('inf')
        print(f"\nTOTAL: n={len(results)} net={net:+.1f}% PF={pf:.2f}", flush=True)


if __name__ == "__main__":
    main()
