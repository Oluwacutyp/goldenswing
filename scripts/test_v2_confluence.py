"""Test V2 confluence: do 2+ factor zones outperform?

Compares Version 1 (hierarchical filter) signals split by confluence score:
- High confluence (2+ factors: Structure + Fib + VP)
- Low confluence (<2 factors)

If V2 works, high-confluence signals should outperform.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from goldenswing.backtest.harness import simulate_signal_v2
from goldenswing.backtest.mtf_data import load_oanda
from goldenswing.features.engine import read_and_featurize
from goldenswing.golden.detector import GoldenSwingDetector, DetectorConfig
from goldenswing.system.regime import hierarchical_filter
from goldenswing.v2_confluence.fibonacci import retracement_levels
from goldenswing.v2_confluence.volume_profile import calculate_vp, vp_confluence
from goldenswing.v2_confluence.engine import score_zone, is_high_quality


def main():
    print("Loading data...", flush=True)
    df = load_oanda('H1', '2022-01-01T00:00:00Z', '2024-10-01T00:00:00Z')
    states = read_and_featurize(df, {})

    print("Generating V1 signals...", flush=True)
    cfg = DetectorConfig(emit_floor=35.0, golden_threshold=60.0,
        feature_families=frozenset({'relstr','volume','phases','aoc'}),
        require_sweep_confirmation=True, daily_veto=True)
    det = GoldenSwingDetector(cfg)
    sigs = [s for s in det.scan(states) if s is not None and s.score >= 60.0]
    v1 = hierarchical_filter(sigs, states, min_score=65.0)
    longs = [s for s in v1 if s.direction == 1]
    print(f"{len(longs)} V1 long signals", flush=True)

    # For each signal, compute confluence at entry
    high_conf = []
    low_conf = []

    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    vol = df['volume'].to_numpy() if 'volume' in df.columns else np.ones(len(df))

    # Simple ATR for tolerance
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    atr = np.convolve(tr, np.ones(14)/14, mode='same')

    for sig in longs:
        idx = sig.index
        if idx < 100 or idx >= len(df) - 5:
            continue

        s = states[idx]
        entry = sig.entry_ref
        a = atr[idx] if atr[idx] > 0 else (high[idx] - low[idx])

        factors = []

        # 1. Structure: check if near sr_flip or swing
        # (simplified: use signal kind as structure proxy)
        if sig.kind in ("sr_flip", "continuation", "sweep_reversal"):
            factors.append(f"structure:{sig.kind}")

        # 2. Fib: find recent impulse, check retracement
        try:
            # Look back 50 bars for impulse
            lb = 50
            start = max(0, idx - lb)
            wh = high[start:idx].max()
            wl = low[start:idx].min()
            move = wh - wl
            if move > 2 * a:  # significant impulse
                # Bullish impulse assumed for longs
                for ratio in [0.50, 0.618, 0.786]:
                    fp = wh - ratio * move
                    if abs(fp - entry) <= 0.3 * a:
                        factors.append(f"fib:{ratio}")
                        break
        except:
            pass

        # 3. Volume Profile: calculate VP for lookback window
        try:
            vp_start = max(0, idx - 100)
            vp = calculate_vp(
                high[vp_start:idx], low[vp_start:idx],
                close[vp_start:idx], vol[vp_start:idx],
            )
            if vp:
                vp_hits = vp_confluence(vp, entry, a)
                for h in vp_hits:
                    factors.append(f"vp:{h}")
                    break
        except:
            pass

        # Deduplicate by category
        cats = {}
        for f in factors:
            cat = f.split(':')[0]
            if cat not in cats:
                cats[cat] = f
        score = len(cats)

        if score >= 2:
            high_conf.append(sig)
        else:
            low_conf.append(sig)

    print(f"\nHigh confluence (2+): {len(high_conf)}", flush=True)
    print(f"Low confluence (<2): {len(low_conf)}", flush=True)

    for label, sig_list in [("HIGH", high_conf), ("LOW", low_conf)]:
        outs = [o for o in (simulate_signal_v2(df, s) for s in sig_list) if o]
        if outs:
            rs = np.array([o.r_net for o in outs])
            net = (np.prod(1 + 0.01 * rs) - 1) * 100
            pf = rs[rs > 0].sum() / abs(rs[rs <= 0].sum()) if (rs <= 0).any() else float('inf')
            wr = np.mean(rs > 0)
            print(f"{label}: n={len(outs)} net={net:.1f}% PF={pf:.2f} win={wr:.0%}", flush=True)


if __name__ == "__main__":
    main()
