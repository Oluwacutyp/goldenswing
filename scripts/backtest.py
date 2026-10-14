#!/usr/bin/env python3
"""Run the GoldenSwing Phase-1 backtest (writes BACKTEST.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.harness import main

if __name__ == "__main__":
    main()
