"""Owner-rulings pins: the repository must remember the owner's settled
decisions, and the tests must prevent them from being silently undone.

Covers the load-bearing parts of OWNER_RULINGS.md (repo root). The
older doctrine pins live in tests/test_sizing_min_floor.py (minimum
floor) and tests/test_phase2_fixes.py (analytical verdict identical at
every balance; 15% drawdown halt); this file pins the gaps those do not:
the dead fixed-lots flag, the dead below-minimum abandonment, the absent
score parameter, and the rulings file itself.
"""

import inspect
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from goldenswing.execution.risk import gold_spec
from goldenswing.live.sizing import size_for_risk

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_rejects_fixed_lots_flag():
    """The runbook once documented `--fixed-lots 0.01`; the flag does
    not exist. The real CLI must reject it (argparse error), so the
    dead doc can never silently become a live option again."""
    proc = subprocess.run(
        [sys.executable, "scripts/run_mt5.py", "--fixed-lots", "0.01"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "unrecognized arguments" in proc.stderr


def test_size_for_risk_never_abandons_below_minimum():
    """Below-minimum size floors at the broker minimum and trades
    (owner ruling 2026-10-24). $12 equity at a $10 stop: risk $0.12,
    raw 0.00012 lots -> ok at 0.01 with min_override, never a
    below-minimum decline."""
    spec = gold_spec()
    r = size_for_risk(Decimal("0.12"), 10.0, spec)
    assert r.ok
    assert r.lots == Decimal("0.01")
    assert r.min_override
    assert r.verdict == "ok"


def test_size_for_risk_takes_no_score_parameter():
    """Score.total is a ranking total, not confidence: the sizing API
    must take no score, score-band, multiplier, or confidence
    parameter, so a score multiplier cannot be reintroduced through
    the sizing call."""
    params = set(inspect.signature(size_for_risk).parameters)
    assert params.isdisjoint(
        {"score", "score_band", "multiplier", "confidence", "confidence_band"}
    )


def test_owner_rulings_file_states_superseded_rules():
    """OWNER_RULINGS.md must exist at the repo root and name the
    superseded rules explicitly — the repository remembering the
    decisions is the fix for the owner having to repeat them."""
    text = (REPO_ROOT / "OWNER_RULINGS.md").read_text()
    lowered = text.lower()
    assert "superseded" in lowered
    assert "0.01" in text
    assert "15%" in text
    assert "fixed-lots" in lowered
    assert "score band" in lowered
    assert "min_override" in text
