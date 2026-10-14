#!/usr/bin/env python3
"""Run the GoldenSwing Phase-4.5 AOC ablation (writes AOC.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.aoc_ablation import main_aoc

if __name__ == "__main__":
    main_aoc()
