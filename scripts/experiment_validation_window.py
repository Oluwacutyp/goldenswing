#!/usr/bin/env python3
"""Validation-window check: does the anchor replicate on OANDA spot
2024-10-01 -> 2026-10-11 (its own validation regime, different feed)?

If profitable: regime problem confirmed (works in bull, fails 2022-24).
If not: data/instrument mismatch or overfit — deeper problem.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.harness import book_metrics, simulate_signal_v2
from goldenswing.backtest.mtf_data import load_oanda
from goldenswing.features.engine import read_and_featurize
from goldenswing.playbooks import SweepReversalPlaybook
from goldenswing.playbooks.base import EngineContext

START = "2024-10-01T00:00:00Z"
END = "2026-10-12T00:00:00Z"


def main() -> None:
    print("loading ...", flush=True)
    df = load_oanda("H1", START, END)
    print(f"H1: {len(df)} bars", flush=True)
    states = read_and_featurize(df, {})
    years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    ctx = EngineContext(df=df, states=states, years=years,
                        instrument="XAUUSD")
    pb = SweepReversalPlaybook()
    sigs = [s for s in pb.detector.scan(states)
            if s is not None and s.score >= 60.0]
    print(f"{len(sigs)} signals", flush=True)
    outs = []
    for s in sigs:
        oc = simulate_signal_v2(df, s)
        if oc is not None:
            outs.append(oc)
    m = book_metrics(outs, years)
    longs = [o for o in outs if o.signal.direction == 1]
    shorts = [o for o in outs if o.signal.direction == -1]
    ml = book_metrics(longs, years)
    ms = book_metrics(shorts, years)
    print(f"ALL:   n={m['trades']} net={m['return_pct']:.2f}% "
          f"PF={m['pf']:.2f} win={m['win_rate']:.1f}%")
    print(f"LONG:  n={ml['trades']} net={ml['return_pct']:.2f}% "
          f"PF={ml['pf']:.2f} win={ml['win_rate']:.1f}%")
    print(f"SHORT: n={ms['trades']} net={ms['return_pct']:.2f}% "
          f"PF={ms['pf']:.2f} win={ms['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
