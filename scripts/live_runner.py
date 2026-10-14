#!/usr/bin/env python3
"""GoldenSwing live runner entry point.

Dry mode (default, works anywhere): replays the cached bars through
the full loop — detector -> alerts (outbox) -> risk pipeline ->
PaperExecutor -> journal.  With --auto, golden signals are executed
on the paper book through the real callback path.

    python3 scripts/live_runner.py [--max-bars N] [--auto]

Live mode (Windows host, MT5 terminal installed + logged in +
active):

    python3 scripts/live_runner.py --live [--auto]

Configuration is via GS_* env vars — see EXECUTION.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.execution.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
