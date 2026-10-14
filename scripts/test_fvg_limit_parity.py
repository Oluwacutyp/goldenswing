"""Parity test for FVGLimitPlaybook.

Proves three things:
  1. The live path (incremental live_signal) emits EXACTLY the same
     signals as the batch path (candidates()).
  2. The batch path's R totals reproduce the validated A/B script
     (scripts/test_fvg_entry_ab.py, pen>=0.10 ATR): +32.0R.
  3. Export/import_state round-trips the streaming machine without
     changing the emitted signal set.

Run:  ~/workspace/goldenswing/.venv/bin/python scripts/test_fvg_limit_parity.py
"""
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import pandas as pd

from goldenswing.playbooks.base import EngineContext
from goldenswing.playbooks.fvg_limit import FVGLimitPlaybook
import test_fvg_entry_ab as ab


def main():
    path = 'data/OANDA_XAU_USD_H1_20241001_20261012.csv'
    ts, o_l, h_l, l_l, c_l = ab.load_ohlc(path)
    df = pd.DataFrame(
        {"open": o_l, "high": h_l, "low": l_l, "close": c_l},
        index=pd.to_datetime(ts),
    )
    n = len(df)
    print(f"bars: {n}")

    # --- reference: the validated A/B script's own machinery ---
    tracker = __import__(
        "goldenswing.v2_confluence.fvg",
        fromlist=["FVGTracker"]).FVGTracker(max_age_bars=100)
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    touch_events = []
    for i in range(n):
        tracker.on_bar(h[i], l[i], i)
        for z in tracker.active():
            if i <= z.formed_idx:
                continue
            touched = (l[i] <= z.top and h[i] >= z.bottom) if z.direction == 1 \
                else (h[i] >= z.bottom and l[i] <= z.top)
            if touched and not any(ev[0] is z for ev in touch_events):
                touch_events.append((z, i))
    ref_rs = ab.simulate(touch_events, o_l, h_l, l_l, c_l, 'limit',
                         min_pen_atr=0.10)
    print(f"A/B reference (pen>=0.10): {len(ref_rs)} trades, "
          f"{sum(ref_rs):+.1f}R")

    # --- batch path ---
    pb = FVGLimitPlaybook()
    ctx = EngineContext(df=df, states=[], years=2.0, instrument="XAUUSD")
    cands = pb.candidates(ctx)
    batch_keys = sorted((cd.signal_index, cd.direction) for cd in cands)
    batch_r = sum(cd.r_net for cd in cands)
    print(f"batch candidates: {len(cands)} trades, {batch_r:+.1f}R")

    # --- live path: incremental replay, emit only latest close ---
    pb2 = FVGLimitPlaybook()
    live_sigs = []
    for t in range(n):
        for s in pb2.live_signal([], t, df):
            live_sigs.append(s)
    live_keys = sorted((s.index, s.direction) for s in live_sigs)
    print(f"live signals:     {len(live_sigs)}")

    # --- export/import round-trip at the midpoint ---
    pb3 = FVGLimitPlaybook()
    mid = n // 2
    for t in range(mid):
        pb3.live_signal([], t, df)
    pb4 = FVGLimitPlaybook()
    pb4.import_state(pb3.export_state())
    resumed = []
    for t in range(mid, n):
        for s in pb4.live_signal([], t, df):
            resumed.append(s)
    resumed_keys = sorted((s.index, s.direction) for s in resumed)
    live_tail_keys = sorted((s.index, s.direction)
                            for s in live_sigs if s.index >= mid)

    # --- verdicts ---
    fails = []
    if batch_keys != live_keys:
        fails.append(f"batch/live signal sets differ "
                     f"({len(batch_keys)} vs {len(live_keys)})")
    if abs(batch_r - sum(ref_rs)) > 1e-6:
        fails.append(f"batch R {batch_r:+.4f} != A/B reference "
                     f"{sum(ref_rs):+.4f}")
    if live_tail_keys != resumed_keys:
        fails.append("state round-trip changed the signal set")

    # check batch context gates pass their own checker
    for cd in cands:
        ok, reason = pb.check_live_signal(cd.outcome.signal, ctx,
                                          cd.context)
        if not ok:
            fails.append(f"batch candidate failed own gate: {reason}")
            break

    print()
    if fails:
        print("PARITY FAILED:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print(f"PARITY OK: {len(cands)} signals identical batch/live, "
          f"{batch_r:+.1f}R reproduces the validated A/B total, "
          f"state round-trip clean.")

    # breakdown for the report
    longs = [cd for cd in cands if cd.direction == 1]
    shorts = [cd for cd in cands if cd.direction == -1]
    print(f"  longs : {len(longs)} trades, "
          f"{sum(cd.r_net for cd in longs):+.1f}R")
    print(f"  shorts: {len(shorts)} trades, "
          f"{sum(cd.r_net for cd in shorts):+.1f}R")


if __name__ == "__main__":
    main()
