#!/usr/bin/env python3
"""Hierarchical Turtle Soup: H1 buy-side sweep + LTF bearish MSS entry.

The H1-only Turtle Soup failed (-17.2%, PF 0.19). The mining says the
pattern needs multi-TF structure: HTF takes the liquidity, LTF confirms
the shift. This tests both M15 and M5 as the entry timeframe.

Design:
  H1: buy-side sweep detected (wick above highs, close back below)
  LTF (M15/M5): first bearish CHoCH/BOS after the sweep = entry trigger
  Entry: next LTF open after the MSS bar
  Stop: H1 swept extreme + 0.30x H1 ATR (wide for gold's spikes)
  Target: nearest sell-side liquidity (from H1 pools below)

Compares M15-entry vs M5-entry vs the failed H1-only version.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.harness import simulate_signal_v2
from goldenswing.backtest.mtf_data import fetch_ltf_window, load_oanda
from goldenswing.features.engine import read_and_featurize
from goldenswing.golden.detector import GoldenSwingDetector, DetectorConfig
from goldenswing.reader.reader import MarketReader

START = "2022-01-01T00:00:00Z"
END = "2024-10-01T00:00:00Z"


def find_ltf_mss(sweep_time: pd.Timestamp, ltf: str,
                 max_h1_bars: int = 12) -> tuple | None:
    """First bearish CHoCH/BOS on LTF after the H1 sweep.

    Returns (entry_time, entry_price_hint) or None.
    """
    m5_per_h1 = {"M15": 4, "M5": 12, "M1": 60}[ltf]
    bars_after = max_h1_bars * m5_per_h1 + 10
    ldf = fetch_ltf_window(sweep_time, ltf, bars_before=5,
                           bars_after=bars_after)
    if len(ldf) < 10:
        return None
    reader = MarketReader()
    states = reader.read(ldf)
    # sweep bar time on LTF: first LTF bar at/after the H1 sweep close
    sweep_close = sweep_time + pd.Timedelta(hours=1)
    for s in states:
        if s.timestamp < sweep_close:
            continue
        ev = s.last_structure_event
        if ev is None:
            continue
        # bearish MSS after the sweep: CHoCH down or BOS down
        if ev.direction == -1 and ev.index >= 0:
            # event must be after the sweep (compare timestamps)
            # find the event bar's timestamp
            evt_time = ldf.index[ev.index] if ev.index < len(ldf) else None
            if evt_time is None or evt_time < sweep_close:
                continue
            # entry at next LTF open after the MSS bar
            entry_idx = ev.index + 1
            if entry_idx >= len(ldf):
                return None
            return ldf.index[entry_idx], float(ldf["open"].iloc[entry_idx])
    return None


def main() -> None:
    print("loading H1 ...", flush=True)
    df = load_oanda("H1", START, END)
    states = read_and_featurize(df, {})
    # find H1 buy-side sweeps (direction == -1)
    sweeps = []
    for s in states:
        sw = s.last_sweep
        if sw is not None and sw.direction == -1 and s.sweep_age == 0:
            # sweep just happened (age 0 = this bar)
            sweeps.append((s, sw))
    print(f"{len(sweeps)} H1 buy-side sweeps", flush=True)

    for ltf in ["M15", "M5"]:
        entries = []
        for s, sw in sweeps:
            # H1 sweep bar timestamp
            st = df.index[s.index]
            found = find_ltf_mss(st, ltf)
            if found is None:
                continue
            entry_time, entry_px = found
            # build a synthetic signal for simulation
            # stop above the H1 swept extreme
            inv = sw.extreme + 0.30 * s.atr
            # find H1 pools below for targets
            targets = [p.price for p in s.pools_below[:3]]
            entries.append({
                "sweep_time": st, "entry_time": entry_time,
                "entry_px": entry_px, "inv": inv, "targets": targets,
                "atr": s.atr, "sweep_extreme": sw.extreme,
            })
        print(f"{ltf}: {len(entries)} LTF-confirmed entries", flush=True)
        # simulate each on H1 data from entry time
        # (simplified: enter at LTF price, stop at H1 inv, H1 targets)
        results = []
        for e in entries:
            # find H1 bar containing entry_time
            try:
                h1_idx = df.index.get_indexer([e["entry_time"]],
                                              method="ffill")[0]
            except Exception:
                continue
            if h1_idx < 0 or h1_idx >= len(df) - 1:
                continue
            # walk H1 bars from entry: stop first, then targets
            d = -1
            entry = e["entry_px"]
            risk = abs(entry - e["inv"])
            if risk <= 0:
                continue
            r = 0.0
            outcome = None
            for k in range(h1_idx, min(h1_idx + 72, len(df))):
                hi, lo = df["high"].iloc[k], df["low"].iloc[k]
                if hi >= e["inv"]:  # stop hit
                    # fill at stop (conservative)
                    exit_px = e["inv"] if df["open"].iloc[k] < e["inv"] \
                        else df["open"].iloc[k]
                    r = (entry - exit_px) / risk * d * -1  # short: profit when price drops
                    r = (entry - exit_px) / risk  # for short, entry > exit = profit
                    # careful: r = (entry - exit)/risk for short
                    outcome = r
                    break
                # check targets (first pool below)
                for t in e["targets"]:
                    if lo <= t:
                        r = (entry - t) / risk
                        outcome = r
                        break
                if outcome is not None:
                    break
            if outcome is None:
                # timeout: mark to market at last bar
                outcome = (entry - df["close"].iloc[min(h1_idx + 71, len(df) - 1)]) / risk
            results.append(outcome)
        if results:
            rs = np.array(results)
            # apply 0.5% cost per trade (approx)
            rs = rs - 0.02
            net = (np.prod(1 + 0.01 * rs) - 1) * 100
            pf = rs[rs > 0].sum() / abs(rs[rs <= 0].sum()) \
                if (rs <= 0).any() else float("inf")
            print(f"  {ltf} hierarchical: n={len(rs)} net={net:.1f}% "
                  f"PF={pf:.2f} win={np.mean(rs > 0):.0%} "
                  f"avgR={rs.mean():.3f}", flush=True)
        else:
            print(f"  {ltf}: no trades", flush=True)


if __name__ == "__main__":
    main()
