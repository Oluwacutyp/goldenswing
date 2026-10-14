#!/usr/bin/env python3
"""Phase 6.5 multi-horizon forecaster run -> FORECAST_HORIZONS.md.

Runs the purged walk-forward at every registered horizon, grades
each (AUC / alert quality / calibration / standalone book /
permutation null), tests 24+72+168 agreement as a detector-book
filter, and checks the 72-bar row against FORECAST.md's Phase-6
anchors.

Usage: python3 scripts/forecast_horizons.py [--refresh-data]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.forecast.horizons import (build_horizons_report,  # noqa: E402
                                           run_horizons)


def main() -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing horizons")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()
    if args.refresh_data:
        from goldenswing.backtest.data import fetch_gold_h1
        fetch_gold_h1()
    r = run_horizons()
    report = build_horizons_report(r)
    out = Path(__file__).resolve().parents[1] / "FORECAST_HORIZONS.md"
    out.write_text(report)
    print(report)
    print(f"[written to {out}]")
    bad = [k for k, v in r.reproduction.items() if not v["ok"]]
    if bad:
        print(f"WARNING: Phase-6 reproduction FAILED for: {bad}")
        sys.exit(1)


if __name__ == "__main__":
    main()
