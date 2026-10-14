#!/usr/bin/env python3
"""Run the GoldenSwing Phase-5 modes + routing study (writes MODES.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.modes.report import main5

if __name__ == "__main__":
    main5()
