"""Step 6 (2026-10-27): Hot Scalp promotion path (flag-gated).

The promotion path translates a shadow candidate into the SHARED
order path (`_place_live_order`, so sizing, friction, margin,
market-closed, and news gates still apply) — but ONLY when the
operator has enabled `hot_scalp_live_enabled` on the host
(config-file/env only; /set cannot touch it).  Default OFF is
itself a test: a candidate with the flag off places nothing.

Claim fates (the V2 contract):
- success -> "order_reconciled", and only after the broker
  itself shows the new magic position;
- transient refusals (paused, spread widened, order-path refusal
  with nothing placed, holding in flat-only v1) -> back to
  "touched", re-triggerable;
- disagreement/ambiguity (broker shows an untracked magic
  position, broker lookup failed, fill without broker
  confirmation) -> claim LEFT claimed, fail closed.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.live.hierarchy import TimeframeStack
from goldenswing.live.hot_scalp import HotScalpConfig, HotScalpResearchEngine
from goldenswing.live.setup_store import SetupStore

NOW = "2026-10-27T00:50:30+00:00"


def _frame(tf_minutes, start, count, *, base=100.0):
    idx = pd.date_range(start, periods=count, freq=f"{tf_minutes}min")
    k = np.arange(count, dtype=float)
    close = base + 0.025 * np.sin(k / 2.0)
    open_ = close - 0.02 * np.cos(k)
    high = np.full(count, 103.0)
    low = np.full(count, 99.0)
    high = np.maximum(high, np.maximum(open_, close) + 0.1)
    low = np.minimum(low, np.minimum(open_, close) - 0.1)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": np.full(count, 100.0)},
                        index=idx)


class _MarketExec:
    """Fake broker: market fills synchronously at entry_ref."""

    def __init__(self, positions=None, fail_positions=False):
        self._positions = list(positions or [])
        self.fail_positions = fail_positions
        self.placed = []
        self._next_ticket = 777

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
        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions.append(Position(
            ticket=ticket, symbol=request.symbol,
            direction=request.direction, volume=request.volume,
            entry_price=float(request.entry_ref),
            sl=float(request.stop_loss), tp=request.take_profit,
            magic=request.magic, alert_id=request.alert_id))
        return OrderResult(True, ticket=ticket,
                           price=float(request.entry_ref),
                           volume=request.volume)


def _runner(tmp_path, *, live=True, positions=None, fail_positions=False,
            paused=False):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False, hot_scalp_live_enabled=live,
        paused=paused)
    feed = SimpleNamespace(
        account_state=lambda: {"equity": 10000.0, "balance": 10000.0,
                               "free_margin": 10000.0},
        order_margin=lambda *, volume, price, direction:
            float(volume) * 100.0 * float(price) / 500.0)
    runner = MT5Runner(cfg, feed=feed,
                       executor=_MarketExec(positions, fail_positions))
    # The market-closed guard is covered by step-3 tests; pin it
    # open here so promotion tests do not depend on the weekday.
    runner._market_open = lambda: (True, "test")
    runner._stack = TimeframeStack.from_frames({
        "H1": _frame(60, "2026-10-27T00:00:00Z", 60),
        "M5": _frame(5, "2026-10-27T00:00:00Z", 10),
        "M15": _frame(15, "2026-10-26T12:15:00Z", 50)})
    runner._hot_scalp_aux_frames = {
        "M1": _frame(1, "2026-10-27T00:00:00Z", 50)}
    store = SetupStore(tmp_path / "hot-scalp.sqlite")
    runner._hot_scalp = HotScalpResearchEngine(
        store, HotScalpConfig(min_m1_bars=100, min_m5_bars=100,
                              min_m15_bars=100),
        instance_id="process-A")
    return runner


def _candidate(runner, *, claimant="process-A", entry=100.30, stop=99.0,
               target=102.0, direction=1, spread=0.02, score=60.0):
    setup = {
        "setup_id": "hs-test-parent-1", "symbol": "XAUUSD",
        "strategy": "breakout_retest", "direction": direction,
        "formation_ts": "2026-10-27T00:40:00+00:00",
        "armed_ts": "2026-10-27T00:50:00+00:00",
        "expires_at": "2026-10-27T01:25:00+00:00",
        "level": 100.2, "zone_low": 100.0, "zone_high": 100.25,
        "stop": stop, "structure_ts": "2026-10-27T00:40:00+00:00",
        "context_direction": direction, "atr_m5": 0.5,
        "description": "promotion test setup",
    }
    store = runner._hot_scalp.store
    store.arm(setup)
    store.mark_touched(setup["setup_id"], NOW)
    claim = store.claim_trigger(setup["setup_id"], claimant, NOW)
    assert claim is not None, "test setup claim must succeed"
    return {
        "event": "candidate_shadow", "setup_id": setup["setup_id"],
        "parent_setup_id": setup["setup_id"],
        "strategy": "breakout_retest", "direction": direction,
        "entry_reference": entry, "structural_stop": stop,
        "target_reference": target, "spread": spread, "score": score,
        "trigger_ts": NOW, "claim": claim,
    }


def _tick(*, mid=100.30, bid=100.29, ask=100.31):
    return {"time": NOW, "mid": mid, "bid": bid, "ask": ask}


def _notes(runner):
    return "\n".join(
        r.get("text", "") for r in runner.journal.records()
        if r.get("type") == "note")


def _status(runner):
    # active() covers armed/touched only; claimed and terminal
    # rows need a direct read (test introspection, not engine API).
    import sqlite3
    con = sqlite3.connect(str(runner._hot_scalp.store.path))
    try:
        row = con.execute(
            "SELECT status FROM hot_scalp_setups WHERE setup_id=?",
            ("hs-test-parent-1",)).fetchone()
    finally:
        con.close()
    assert row is not None
    return row[0]


def test_flag_off_places_nothing_and_leaves_claim(tmp_path):
    runner = _runner(tmp_path, live=False)
    cand = _candidate(runner)
    assert runner._promote_hot_scalp_candidate(cand, _tick(), {}) is False
    assert runner.executor.placed == []
    assert _status(runner) == "claimed"
    assert "live authority OFF" in _notes(runner)


def test_happy_path_promotes_and_reconciles(tmp_path):
    runner = _runner(tmp_path, live=True)
    cand = _candidate(runner)
    assert runner._promote_hot_scalp_candidate(cand, _tick(), {}) is True
    assert len(runner.executor.placed) == 1
    placed = runner.executor.placed[0]
    assert placed.direction == 1 and placed.order_kind == "market"
    assert placed.magic == runner.cfg.magic
    assert len(runner.tick_manager.tickets()) == 1
    assert _status(runner) == "order_reconciled"
    assert "hot_scalp promoted" in _notes(runner)


def test_broker_disagreement_fails_closed_claim_left(tmp_path):
    mystery = Position(ticket=555, symbol="XAUUSD", direction=-1,
                       volume=0.5, entry_price=101.0, sl=102.0,
                       magic=260101, alert_id="MT5-something-else")
    runner = _runner(tmp_path, live=True, positions=[mystery])
    cand = _candidate(runner)
    assert runner._promote_hot_scalp_candidate(cand, _tick(), {}) is False
    assert runner.executor.placed == []
    assert _status(runner) == "claimed"


def test_broker_lookup_failure_leaves_claim(tmp_path):
    runner = _runner(tmp_path, live=True, fail_positions=True)
    cand = _candidate(runner)
    assert runner._promote_hot_scalp_candidate(cand, _tick(), {}) is False
    assert runner.executor.placed == []
    assert _status(runner) == "claimed"


def test_holding_refuses_flat_only_and_releases(tmp_path):
    runner = _runner(tmp_path, live=True)
    # Tracked position from another sleeve: promotion must not
    # stack onto it (flat-only v1).
    from goldenswing.execution.executor import OrderRequest
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    runner.tick_manager.open(ticket=101, request=req, fill_price=2000.0,
                             entry_bar=6, broker_sl=1987.5)
    cand = _candidate(runner)
    assert runner._promote_hot_scalp_candidate(cand, _tick(), {}) is False
    assert runner.executor.placed == []
    assert _status(runner) == "touched"
    assert "flat-only v1" in _notes(runner)


def test_wide_spread_at_promotion_refuses_and_releases(tmp_path):
    runner = _runner(tmp_path, live=True)
    cand = _candidate(runner)
    wide = _tick(mid=100.30, bid=98.0, ask=103.0)  # spread 5 > 0.2 x ATR~4
    assert runner._promote_hot_scalp_candidate(cand, wide, {}) is False
    assert runner.executor.placed == []
    assert _status(runner) == "touched"


def test_foreign_claim_refuses_without_touching_it(tmp_path):
    runner = _runner(tmp_path, live=True)
    cand = _candidate(runner, claimant="other-process")
    assert runner._promote_hot_scalp_candidate(cand, _tick(), {}) is False
    assert runner.executor.placed == []
    assert _status(runner) == "claimed"


def test_paused_refuses_and_releases(tmp_path):
    runner = _runner(tmp_path, live=True, paused=True)
    cand = _candidate(runner)
    assert runner._promote_hot_scalp_candidate(cand, _tick(), {}) is False
    assert runner.executor.placed == []
    assert _status(runner) == "touched"


def test_shadow_events_flow_into_promotion_on_the_live_tick(tmp_path):
    # The live-loop wiring shape: shadow returns events, and the
    # flag-gated loop promotes this tick's candidates.
    runner = _runner(tmp_path, live=True)
    store = runner._hot_scalp.store
    store.arm({
        "setup_id": "hs-test-parent-1", "symbol": "XAUUSD",
        "strategy": "breakout_retest", "direction": 1,
        "formation_ts": "2026-10-27T00:40:00+00:00",
        "armed_ts": "2026-10-27T00:50:00+00:00",
        "expires_at": "2026-10-27T01:25:00+00:00",
        "level": 100.2, "zone_low": 100.0, "zone_high": 100.25,
        "stop": 99.0, "structure_ts": "2026-10-27T00:40:00+00:00",
        "context_direction": 1, "atr_m5": 0.5,
        "description": "live-tick wiring test",
    })
    frames = {
        "M1": _frame(1, "2026-10-27T00:00:00Z", 50),
        "M5": _frame(5, "2026-10-27T00:00:00Z", 10),
        "M15": _frame(15, "2026-10-26T12:15:00Z", 50),
    }
    runner._stack = TimeframeStack.from_frames({
        "H1": _frame(60, "2026-10-27T00:00:00Z", 60),
        "M5": frames["M5"], "M15": frames["M15"]})
    runner._hot_scalp_aux_frames = {"M1": frames["M1"]}
    ticks = [
        {"time": "2026-10-27T00:50:10Z", "mid": 99.95,
         "bid": 99.94, "ask": 99.96},
        {"time": "2026-10-27T00:50:20Z", "mid": 100.15,
         "bid": 100.14, "ask": 100.16},
        {"time": "2026-10-27T00:50:30Z", "mid": 100.30,
         "bid": 100.29, "ask": 100.31},
    ]
    placed = 0
    for tick in ticks:
        events = runner._run_hot_scalp_shadow(tick)
        if runner.cfg.hot_scalp_live_enabled:
            for event in events or []:
                if event.get("event") == "candidate_shadow":
                    if runner._promote_hot_scalp_candidate(
                            event, tick, {}):
                        placed += 1
    assert placed == 1
    assert len(runner.executor.placed) == 1
    assert runner.executor.placed[0].order_kind == "market"
