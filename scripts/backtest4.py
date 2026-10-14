#!/usr/bin/env python3
"""Run the GoldenSwing Phase-4 context-gate ablation (writes CONTEXT.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.context_ablation import main4

if __name__ == "__main__":
    main4()
