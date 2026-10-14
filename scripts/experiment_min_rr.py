#!/usr/bin/env python3
"""Run the GoldenSwing minimum-R:R qualification filter experiment
(writes experiments/MIN_RR.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.experiments.min_rr import main_min_rr

if __name__ == "__main__":
    main_min_rr()
