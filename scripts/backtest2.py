#!/usr/bin/env python3
"""Run the GoldenSwing Phase-2 backtest (writes BACKTEST2.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.harness import main2

if __name__ == "__main__":
    main2()
