#!/usr/bin/env python3
"""Run the GoldenSwing Phase-3.5 validation (writes VALIDATION.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.validation import main35

if __name__ == "__main__":
    main35()
