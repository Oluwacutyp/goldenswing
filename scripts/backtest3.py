#!/usr/bin/env python3
"""Run the GoldenSwing Phase-3 feature ablation (writes FEATURES.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.ablation import main3

if __name__ == "__main__":
    main3()
