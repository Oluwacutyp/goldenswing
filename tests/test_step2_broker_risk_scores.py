"""Step 2 (2026-10-27): broker/risk lifecycle + score semantics.

Proven against the 2026-10-26 harvest, not asserted from the brief:

- The pool booked the STRUCTURAL stop while sizing, close R and the tick
  manager all measured on the BROKER stop (pool booked 2.28, broker
  exposure 3.27 on the first filled trade).  The pool must book the
  broker stop everywhere, and a widening is a NAMED exception.
- The OPEN record carried the sleeve-base placeholder as its score
  (60.0) while the signal record carried the judgment total (32.71).
  The OPEN record must carry the decision score and BOTH stops under
  their own names, with broker-distance risk_money.
- Score decay cancelled a valid judgment pending within 2 H1 closes:
  the decay floor defaulted to the legacy min_score (60.0) for a route
  that carries NO floor, and the H1 roster cannot even re-score a
  native playbook (natives are not in _get_playbooks), so every
  re-score read "gone".  No-floor routes skip floor-decay until
  step 4 restores a native-frame decay.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.executor import OrderResult
from goldenswing.execution.mt5_runner import (
    MT5Runner, MT5RunnerConfig, PendingOrderState)
from goldenswing.execution.playbook_registry import decay_exempt
from goldenswing.live.hierarchy import TimeframeStack


# ---------------------------------------------------------------- helpers
class _FakeSpec:
    """XAUUSD gold_spec shape: $0.01 tick worth $1.00 per lot."""

    tick_size = 0.01
    tick_value = 1.0

    def money_for_move(self, distance, volume):
        return abs(distance) / self.tick_size * self.tick_value * volume


class _Exec:
    def __init__(self):
        self.cancelled: list[int] = []

    def cancel_pending(self, ticket):
        self.cancelled.append(ticket)
        return OrderResult(True, ticket=ticket)

    def positions(self, symbol=None):
        return []

    def symbol_spec(self, symbol):
        return _FakeSpec()


def _h1_frame():
    return pd.DataFrame(
        {"open": [2000.0] * 60, "high": [2001.0] * 60,
         "low": [1999.0] * 60, "close": [2000.5] * 60,
         "volume": [100] * 60, "real_volume": [0] * 60,
         "spread": [30] * 60},
        index=pd.date_range("2026-01-01", periods=60, freq="h",
                            tz="UTC"))


def _runner(tmp_path):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, feed=None, executor=_Exec())
    runner._stack = TimeframeStack.from_frames({"H1": _h1_frame()})
    runner._playbooks = []
    return runner


def _notes(runner):
    return [r for r in runner.journal.records()
            if r.get("type") == "note"]


# ------------------------------------------------------- registry helper
def test_decay_exempt_only_for_no_floor_routes():
    assert decay_exempt("qm_m5") is True
    assert decay_exempt("qm_d1") is False
    assert decay_exempt("fvg_midline") is False
    assert decay_exempt("range_qm") is False


# ------------------------------------------------------------ decay skip
def test_decay_skips_no_floor_route_and_keeps_pending(tmp_path):
    runner = _runner(tmp_path)
    state = PendingOrderState(
        ticket=777, kind="limit", direction=1, level=2000.0,
        volume=0.01, sl=1990.0, tp=2030.0, alert_id="MT5-qm_m5-1",
        playbook="qm_m5", zone=None, invalidation=1990.0,
        placed_bar=58, expiry_bar=58 + 4, placed_score=32.71)
    runner._pending_orders[777] = state
    runner._check_pending_score_decay({})
    runner._check_pending_score_decay({})
    assert 777 in runner._pending_orders
    assert runner._pending_orders[777].decay_bars == 0
    skip_notes = [n for n in _notes(runner)
                  if "decay skipped" in n.get("text", "")]
    assert skip_notes, "exemption must be journalled, never silent"
    assert "no score floor" in skip_notes[0]["text"]


# ------------------------------------------------- fill-open record truth
def test_fill_open_carries_decision_score_and_broker_stop(tmp_path):
    runner = _runner(tmp_path)
    runner._send_alert = lambda *a, **k: None
    state = PendingOrderState(
        ticket=514994067, kind="limit", direction=1, level=4014.77,
        volume=0.01, sl=4011.50, tp=None, alert_id="MT5-qm_m5-10035",
        playbook="qm_m5", zone=None, invalidation=4012.485,
        placed_bar=100, expiry_bar=104,
        sig=SimpleNamespace(score=60.0, factors=[]),
        verdict=None, placed_score=32.71)
    pos = SimpleNamespace(ticket=514994067, entry_price=4014.77)
    runner._journal_fill_open(state, pos, 0.01, {}, partial=False)
    opens = [r for r in runner.journal.records()
             if r.get("type") == "open"]
    assert len(opens) == 1
    rec = opens[0]
    # Decision score, not the sleeve-base placeholder.
    assert rec["score"] == pytest.approx(32.71)
    # Both stops, each under its own name.
    assert rec["stop"] == pytest.approx(4012.485)
    assert rec["broker_sl"] == pytest.approx(4011.50)
    assert rec["broker_stop_distance"] == pytest.approx(3.27)
    # Risk on the broker distance (3.27), NOT the structural 2.285.
    assert rec["risk_money"] == pytest.approx(3.27)
    assert rec["risk_money"] != pytest.approx(2.285)


# ------------------------------------------------------- pool books broker
def test_pool_for_open_books_broker_stop(tmp_path):
    runner = _runner(tmp_path)
    runner.feed = SimpleNamespace(
        account_state=lambda: {"equity": 100000.0})
    pool = runner._pool_for_open(
        111, lots=0.01, entry=4014.77, stop=4011.50, direction=1,
        alert_id="MT5-x")
    assert pool is not None
    st = pool.state()
    assert st["consumed"] == pytest.approx(3.27)
    assert st["layers"]["MT5-x"]["stop"] == pytest.approx(4011.50)


def test_pool_widening_names_the_min_floor_exception(tmp_path):
    runner = _runner(tmp_path)
    runner.feed = SimpleNamespace(
        account_state=lambda: {"equity": 20.98})
    runner._sizing_detail["MT5-x"] = {"min_override": True}
    pool = runner._pool_for_open(
        111, lots=0.01, entry=4014.77, stop=4011.50, direction=1,
        alert_id="MT5-x")
    assert pool is not None
    widen = [n for n in _notes(runner) if "widened" in n.get("text", "")]
    assert widen, "widening must be journalled"
    assert "broker-minimum floor" in widen[0]["text"]
    assert "min_override" in widen[0]["text"]


def test_pool_widening_without_min_floor_names_the_exception(tmp_path):
    runner = _runner(tmp_path)
    runner.feed = SimpleNamespace(
        account_state=lambda: {"equity": 20.98})
    pool = runner._pool_for_open(
        111, lots=0.01, entry=4014.77, stop=4011.50, direction=1,
        alert_id="MT5-x")
    assert pool is not None
    widen = [n for n in _notes(runner) if "widened" in n.get("text", "")]
    assert widen
    assert "sized past the envelope" in widen[0]["text"]
