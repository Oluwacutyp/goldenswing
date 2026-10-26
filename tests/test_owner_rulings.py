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

import pytest

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


def test_mt5_runner_has_no_legacy_sizer_call():
    """Wiring lock: the MT5 runner must not call the retired legacy
    sizer on any path, and must not import the dormant entry planner.
    The legacy sizer survives only in backtest/forensic code and
    paper simulation; the fence tests below prove the live paths."""
    source = (REPO_ROOT / "goldenswing/execution/mt5_runner.py").read_text()
    assert "size_position(" not in source
    assert "live.entry" not in source


def _callback_handler(tmp_path, *, fixed_lots):
    """Exness $12 paper executor as both executors, mirroring the
    existing paper harness — the handler's live (execute) path is
    what the fences below exercise."""
    from goldenswing.alerts import build_alert
    from goldenswing.execution.callbacks import ExecutionCallbackHandler
    from goldenswing.execution.exness_paper import ExnessPaperExecutor
    from goldenswing.execution.learning import TradeJournal
    from goldenswing.execution.risk import (
        RiskConfig,
        RiskGuards,
        gold_spec,
    )
    from goldenswing.golden.detector import GoldenSignal

    paper = ExnessPaperExecutor(balance=12.0)
    guards = RiskGuards(
        tmp_path / "risk.json",
        RiskConfig(max_drawdown_pct=0.0, max_consecutive_losses=0,
                   max_positions=1, max_total_risk_pct=0.0,
                   require_stops=True),
        kill_switch_path=tmp_path / "KILL")
    journal = TradeJournal(tmp_path / "journal.jsonl")
    handler = ExecutionCallbackHandler(
        executor=paper, paper_executor=paper, guards=guards,
        spec=gold_spec(), journal=journal, risk_pct=0.01,
        fixed_lots=fixed_lots)
    signal = GoldenSignal(
        index=5, timestamp="2026-10-10T05:00:00Z", direction=1,
        score=80.0, kind="continuation", factors=["trend aligned"],
        entry_zone=(1995.0, 2005.0), entry_ref=2000.0,
        invalidation=1990.0, targets=[2030.0], atr=10.0)
    alert = build_alert(signal, instrument="XAUUSD", mode="exness12")
    return handler, paper, alert


def test_callbacks_execute_ignores_fixed_lots(tmp_path):
    """A live (execute) order must never be sized with fixed lots,
    even when the handler was configured with them (paper knob).
    The knob is ignored (journal-noted) and the order trades on the
    dynamic rule — a stale paper setting must not kill a live trade,
    and must not size it either."""
    import json

    from goldenswing.alerts.callbacks import AlertAction

    handler, paper, alert = _callback_handler(tmp_path, fixed_lots=0.01)
    result = handler.handle(AlertAction.EXECUTE, alert)
    assert "EXECUTE" in result
    journal_text = (tmp_path / "journal.jsonl").read_text()
    assert "ignored" in journal_text and "fixed lots" in journal_text
    opens = [json.loads(line) for line in journal_text.splitlines()
             if json.loads(line).get("type") == "open"]
    assert opens and opens[0]["volume"] == pytest.approx(0.01)


def test_callbacks_execute_floors_below_minimum(tmp_path):
    """A live (execute) order on a $12 balance at a $10 stop risks
    $0.12 — raw 0.00012 lots. The retired sizer would return 0 lots
    and refuse; the dynamic rule floors to 0.01 and trades. (The
    paper executor holds the order pending until a bar pump fills
    it, so the proof is the journal open record, not positions().)"""
    import json

    from goldenswing.alerts.callbacks import AlertAction

    handler, paper, alert = _callback_handler(tmp_path, fixed_lots=None)
    result = handler.handle(AlertAction.EXECUTE, alert)
    assert "EXECUTE" in result
    journal_text = (tmp_path / "journal.jsonl").read_text()
    opens = [json.loads(line) for line in journal_text.splitlines()
             if json.loads(line).get("type") == "open"]
    assert opens and opens[0]["volume"] == pytest.approx(0.01)
