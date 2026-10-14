#!/usr/bin/env python3
"""Hierarchical entry experiment: H1 setup + M5 trigger vs H1 market entry.

The user's hypothesis: the concepts were killed as H1-entry
implementations, not as concepts. This tests it directly.

Setup (identical for both arms):
  - GoldenSwingDetector (anchor config) over OANDA XAU/USD H1,
    2022-01-01 -> 2024-10-01 (the sacred held-out window).
  - Same gates: score >= 60, same signal set.

Arm A (baseline, the killed implementation):
  - simulate_signal_v2 on H1: enter at next H1 open, stop at
    invalidation +/- buffer. This replicates the held-out kill.

Arm B (hierarchical):
  - H1 marks the setup AND the entry_zone (bias + level, no timing).
  - Drop to M5: wait for price to touch the zone (max 24 H1 bars).
  - Enter at the next M5 open after the touch.
  - Stop: zone extreme +/- 0.3 * H1 ATR (tighter, structure-based).
  - Targets: the same HTF liquidity targets; same exit machinery
    (simulate_signal_v2 run on the M5 tape with the adjusted signal).

No tuning: the 24-bar timeout and 0.3x buffer are fixed a priori.
This window is sacred held-out data — the read is diagnostic only.

Writes experiments/HIERARCHICAL_ENTRY.md.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.harness import book_metrics, simulate_signal_v2
from goldenswing.backtest.mtf_data import fetch_ltf_window, load_oanda
from goldenswing.features.engine import read_and_featurize
from goldenswing.playbooks import SweepReversalPlaybook
from goldenswing.playbooks.base import EngineContext

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "HIERARCHICAL_ENTRY.md"

START = "2022-01-01T00:00:00Z"
END = "2024-10-01T00:00:00Z"

#: Max H1 bars to wait for the M5 zone touch before giving up.
MAX_WAIT_H1 = 24
#: Stop buffer beyond the zone extreme, in H1 ATRs.
ZONE_BUFFER_ATR = 0.3


def simulate_hierarchical(sig, df_h1) -> object | None:
    """M5 patient entry for one H1 signal. None if no zone touch."""
    d = sig.direction
    z_lo, z_hi = sorted(sig.entry_zone)
    if z_hi <= z_lo:
        return None
    t_sig = df_h1.index[sig.index]
    # the signal is known at the CLOSE of its H1 bar
    search_start = t_sig + pd.Timedelta(hours=1)
    m5 = fetch_ltf_window(t_sig + pd.Timedelta(hours=1), "M5",
                          bars_before=0,
                          bars_after=MAX_WAIT_H1 * 12 + 12)
    if len(m5) < 3:
        return None
    m5 = m5[m5.index >= search_start]
    if len(m5) < 3:
        return None

    low = m5["low"].to_numpy()
    high = m5["high"].to_numpy()
    # first M5 bar that touches the zone
    touch = None
    for i in range(len(m5)):
        if d == 1 and low[i] <= z_hi:
            touch = i
            break
        if d == -1 and high[i] >= z_lo:
            touch = i
            break
    if touch is None or touch + 1 >= len(m5):
        return None  # never came back — discipline: no trade

    entry_idx = touch + 1
    buf = ZONE_BUFFER_ATR * sig.atr if sig.atr > 0 else 0.0
    inv = (z_lo - buf) if d == 1 else (z_hi + buf)

    # same exit machinery, run on the M5 tape with the LTF entry
    sig2 = copy.copy(sig)
    sig2.index = entry_idx
    sig2.invalidation = inv
    # M5 ATR for the stop buffer inside simulate_signal_v2
    h, l, c = high, low, m5["close"].to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)),
                                      abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    sig2.atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
    return simulate_signal_v2(m5, sig2)


def main() -> None:
    print("loading OANDA H1/H4 ...", flush=True)
    df = load_oanda("H1", START, END)
    print(f"H1: {len(df)} bars", flush=True)

    print("featurizing ...", flush=True)
    states = read_and_featurize(df, {})
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    ctx = EngineContext(df=df, states=states, years=years,
                        instrument="XAUUSD")
    pb = SweepReversalPlaybook()

    print("scanning detector ...", flush=True)
    sigs = [s for s in pb.detector.scan(states)
            if s is not None and s.score >= 60.0]
    print(f"{len(sigs)} signals at score>=60", flush=True)

    base_out, hier_out = [], []
    hier_trades, hier_skipped = 0, 0
    for n, sig in enumerate(sigs):
        if n % 25 == 0:
            print(f"  signal {n}/{len(sigs)} ...", flush=True)
        oc_a = simulate_signal_v2(df, sig)
        if oc_a is not None:
            base_out.append(oc_a)
        try:
            oc_b = simulate_hierarchical(sig, df)
        except Exception as e:  # noqa: BLE001 — one bad window skips
            print(f"  warn: hierarchical sim failed on signal {n}: {e}")
            oc_b = None
        if oc_b is not None:
            hier_out.append(oc_b)
            hier_trades += 1
        else:
            hier_skipped += 1

    m_a = book_metrics(base_out, years)
    m_b = book_metrics(hier_out, years)

    # persist outcomes so metric tweaks never require a re-run
    import pickle
    with open(REPO / "experiments" / "hierarchical_outcomes.pkl",
              "wb") as f:
        pickle.dump({"base": base_out, "hier": hier_out,
                     "years": years}, f)

    def row(m):
        pf = f"{m['pf']:.2f}" if m['pf'] != float('inf') else "inf"
        return (f"{m['trades']:>5}  {m['return_pct']:>8.2f}%  "
                f"{pf:>6}  {m['win_rate']:>6.1f}%")

    md = f"""# Hierarchical entry experiment (H1 setup + M5 trigger)

## Design
- Detector: anchor GoldenSwingDetector, score >= 60, unchanged.
- Data: OANDA XAU/USD H1 2022-01-01 -> 2024-10-01 (sacred held-out).
- Arm A (baseline): H1 market entry at next open, H1 invalidation stop.
- Arm B (hierarchical): wait for M5 touch of the signal's entry_zone
  (max {MAX_WAIT_H1} H1 bars), enter next M5 open, stop at zone extreme
  +/- {ZONE_BUFFER_ATR}x H1 ATR. Same targets, same exit machinery.

## Results
| arm | trades | net | PF | win rate |
|---|---:|---:|---:|---:|
| A: H1 entry | {row(m_a)} |
| B: M5 zone entry | {row(m_b)} |

Hierarchical: {hier_trades} entries taken, {hier_skipped} signals never
touched the zone (no trade — by design).

## Read
{"TBD — fill after run."}
"""
    OUT.write_text(md)
    print(md)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
