"""Step 5 (2026-10-27): stacking policy wired to reconciled broker state.

_try_layer used to decide on tick_manager (tracked) state alone and
never called the stacking policy at all.  Now the policy gates
sizing -> placement on BROKER state: the parent must be
broker-confirmed, the broker SL verified, the parent not losing,
the layer count and the shared budget respected on broker stops.

Fixtures mirror test_mt5_layering: a long base (entry 2000,
broker SL 1987.5, risk 12.5, 1.0 lot) held in tick_manager AND at
the (fake) broker.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.executor import (
    OrderRequest, OrderResult, Position)
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec


class _Exec:
    """Minimal broker: positions list is the truth."""

    def __init__(self, positions=None, fail_positions=False):
        self._positions = list(positions or [])
        self.fail_positions = fail_positions
        self.placed = []

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        if self.fail_positions:
            raise RuntimeError("broker unreachable")
        return [p for p in self._positions
                if symbol is None or p.symbol == symbol]

    def pending_orders(self, symbol=None):
        return []

    def cancel_pending(self, ticket):
        return OrderResult(True, ticket=ticket)

    def place_order(self, request):
        self.placed.append(request)
        return OrderResult(True, ticket=9999, price=request.entry_ref,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)


def _broker_pos(ticket=101, volume=1.0, entry=2000.0, sl=1987.5,
                direction=1, magic=260101, alert_id=""):
    return Position(ticket=ticket, symbol="XAUUSD", direction=direction,
                    volume=volume, entry_price=entry, sl=sl, tp=2030.0,
                    magic=magic, alert_id=alert_id)


def _runner(tmp_path, positions=None, fail_positions=False, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False, **cfg_kw)
    feed = SimpleNamespace(
        account_state=lambda: {"equity": 10000.0, "balance": 10000.0,
                               "free_margin": 10000.0},
        order_margin=lambda *, volume, price, direction:
            float(volume) * 100.0 * float(price) / 500.0)
    runner = MT5Runner(cfg, feed=feed,
                       executor=_Exec(positions, fail_positions))
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    runner.tick_manager.open(ticket=101, request=req, fill_price=2000.0,
                             entry_bar=6, broker_sl=1987.5)
    return runner


def _gate(runner, mid=2010.0, lots=0.08, stop0=1997.5,
          mode="same_setup", budget=2500.0, max_layers=3,
          trigger_key="MT5-fvg_midline-100", independent=True):
    pos = runner.tick_manager.position(101)
    layers = runner.tick_manager.layers(101)
    snapshot, fail = runner._layer_broker_state(101, pos)
    assert fail is None, f"snapshot failed: {fail}"
    return runner._layer_policy_gate(
        ticket=101, pos=pos, layers=layers, snapshot=snapshot,
        mid=mid, lots=lots, stop0=stop0, mode=mode,
        trigger_key=trigger_key, trigger_independent=independent,
        max_layers=max_layers, budget=budget, spec=gold_spec(),
        pb=SimpleNamespace(name="fvg_midline"),
        sig=SimpleNamespace(direction=1, entry_ref=2010.0, score=80.0,
                            index=100),
        stats={})


def _notes(runner):
    return "\n".join(
        r.get("text", "") for r in runner.journal.records()
        if r.get("type") == "note")


# ------------------------------------------------------------- gate cases
def test_gate_happy_path_same_setup(tmp_path):
    runner = _runner(tmp_path, positions=[_broker_pos()])
    assert _gate(runner) is True
    assert "stack.layer_eligible" in _notes(runner)


def test_gate_parent_not_at_broker(tmp_path):
    runner = _runner(tmp_path, positions=[])
    assert _gate(runner) is False
    assert "stack.parent_not_open" in _notes(runner)


def test_gate_broker_sl_unverified(tmp_path):
    runner = _runner(tmp_path, positions=[_broker_pos(sl=None)])
    assert _gate(runner) is False
    assert "stack.stop_unverified" in _notes(runner)


def test_gate_broker_volume_disagreement_fails_closed(tmp_path):
    # Broker shows a partial close the tracker has not seen yet.
    runner = _runner(tmp_path, positions=[_broker_pos(volume=0.5)])
    assert _gate(runner) is False
    assert "disagrees" in _notes(runner)


def test_gate_broker_read_failure_fails_closed(tmp_path):
    runner = _runner(tmp_path, positions=[_broker_pos()],
                     fail_positions=True)
    pos = runner.tick_manager.position(101)
    snapshot, fail = runner._layer_broker_state(101, pos)
    assert snapshot is None and fail is not None


def test_gate_no_add_while_losing_same_setup(tmp_path):
    runner = _runner(tmp_path, positions=[_broker_pos()])
    assert _gate(runner, mid=1990.0) is False  # -0.8R on broker stops
    assert "stack.averaging_down_refused" in _notes(runner)


def test_gate_pyramid_threshold(tmp_path):
    runner = _runner(tmp_path, positions=[_broker_pos()])
    # +0.4R: favourable, but below the pyramid 0.50R bar.
    assert _gate(runner, mid=2005.0, mode="pyramid") is False
    assert "stack.not_favourable_enough" in _notes(runner)
    assert _gate(runner, mid=2010.0, mode="pyramid") is True


def test_gate_shared_budget_exceeded(tmp_path):
    runner = _runner(tmp_path, positions=[_broker_pos()])
    # Existing $1250 on broker SL + proposed $100 > $1300 budget.
    assert _gate(runner, budget=1300.0) is False
    assert "stack.parent_budget_exceeded" in _notes(runner)


def test_gate_trigger_duplicate(tmp_path):
    runner = _runner(tmp_path, positions=[_broker_pos()])
    runner._layer_trigger_ids[101] = {"MT5-fvg_midline-100"}
    assert _gate(runner) is False
    assert "stack.trigger_duplicate" in _notes(runner)


def test_gate_child_id_stable(tmp_path):
    from goldenswing.live.stacking import child_entry_id
    assert (child_entry_id("setup-1", "same_setup", "trig-9")
            == child_entry_id("setup-1", "same_setup", "trig-9"))


# ------------------------------------------------------------- integration
def _h1_frame():
    return pd.DataFrame(
        {"open": [2000.0] * 60, "high": [2001.0] * 60,
         "low": [1999.0] * 60, "close": [2000.5] * 60,
         "volume": [100] * 60, "real_volume": [0] * 60,
         "spread": [30] * 60},
        index=pd.date_range("2026-01-01", periods=60, freq="h",
                            tz="UTC"))


def test_try_layer_skips_broker_closed_parent(tmp_path):
    # Broker closed the parent between reconciles; the tracker
    # still holds it.  No new risk may stack onto a ghost.
    runner = _runner(tmp_path, positions=[])
    runner._last_tick = {"mid": 2010.0, "bid": 2009.8, "ask": 2010.2}
    runner._stack = SimpleNamespace(frames={"H1": _h1_frame()})
    stats: dict = {}
    runner._try_layer(
        SimpleNamespace(name="fvg_midline"),
        SimpleNamespace(direction=1, score=80.0, entry_ref=2010.0,
                        invalidation=2000.0, targets=[2040.0],
                        atr=10.0, index=100),
        stats)
    assert stats.get("layers", 0) == 0
    assert len(runner.tick_manager.layers(101)) == 1
    assert "stack.parent_not_open" in _notes(runner)


def test_layer_fill_consumes_trigger(tmp_path):
    runner = _runner(tmp_path, positions=[_broker_pos()])
    runner._record_layer_fill(
        "fvg_midline",
        SimpleNamespace(direction=1, score=80.0, index=100),
        fill=2010.0, lots=0.08, stop0=1997.5, stats={})
    assert "MT5-fvg_midline-100" in runner._layer_trigger_ids[101]
