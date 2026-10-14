#!/usr/bin/env python3
"""Phase 6 end-to-end forecaster run -> FORECAST.md.

Usage: python3 scripts/forecast.py [--refresh-data]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.forecast.run import build_report, run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing forecaster")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()
    if args.refresh_data:
        from goldenswing.backtest.data import fetch_gold_h1
        fetch_gold_h1()
    r = run()
    report = build_report(r)
    out = Path(__file__).resolve().parents[1] / "FORECAST.md"
    out.write_text(report)
    print(report)
    print(f"[written to {out}]")


if __name__ == "__main__":
    main()
