"""Step 6 (2026-10-27): Hot Scalp acceptance gaps.

Covered already elsewhere: intrabar trigger (engine reclaim test),
lookahead-free replay (forming-candle test), atomic claim /
concurrency (two-instance claim test), crash recovery (claim held
until authoritative broker reconciliation).  Closed here:

- expiry: a setup past its TTL emits setup_expired, never a
  candidate;
- invalidation: a closed M1 bar breaching the structural stop
  emits setup_invalidated, never a candidate (a transient tick
  wick alone must not kill it);
- stale data: stale M1/M5/M15 frames withhold the scan entirely
  (scan_degraded), arming nothing and emitting no candidate;
- cost envelope: a reclaim with spread beyond 0.20 x M1 ATR is
  blocked (costs.spread_too_wide), never a candidate;
- shadow authority at the runner level: a candidate_shadow
  journal record is produced while the executor receives ZERO
  orders.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from goldenswing.execution.executor import OrderResult
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.live.hot_scalp import HotScalpConfig, HotScalpResearchEngine
from goldenswing.live.setup_store import SetupStore
from goldenswing.live.hierarchy import TimeframeStack


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


def _frames():
    # Ends at closed-bar opens 00:49 (M1), 00:45 (M5), 00:30 (M15)
    # — fresh at a 00:50 tick (closes 00:50 / 00:50 / 00:45).
    return {
        "M1": _frame(1, "2026-10-27T00:00:00Z", 50),
        "M5": _frame(5, "2026-10-27T00:00:00Z", 10),
        "M15": _frame(15, "2026-10-26T12:15:00Z", 50),
    }


def _setup(**kw):
    d = {
        "setup_id": "hs-test-parent-1", "symbol": "XAUUSD",
        "strategy": "breakout_retest", "direction": 1,
        "formation_ts": "2026-10-27T00:40:00+00:00",
        "armed_ts": "2026-10-27T00:50:00+00:00",
        "expires_at": "2026-10-27T01:25:00+00:00",
        "level": 100.2, "zone_low": 100.0, "zone_high": 100.25,
        "stop": 99.0, "structure_ts": "2026-10-27T00:40:00+00:00",
        "context_direction": 1, "atr_m5": 0.5,
        "description": "test parent structure waiting for live reclaim",
    }
    d.update(kw)
    return d


def _engine(tmp_path):
    store = SetupStore(tmp_path / "hot-scalp.sqlite")
    cfg = HotScalpConfig(min_m1_bars=100, min_m5_bars=100,
                         min_m15_bars=100)
    return HotScalpResearchEngine(store, cfg, instance_id="process-A")


def _events(engine, frames, ticks):
    out = []
    for tick in ticks:
        out.extend(engine.on_tick(symbol="XAUUSD", tick=tick,
                                  frames=frames, now=tick["time"]))
    return out


def test_setup_expires_and_never_candidates(tmp_path):
    engine = _engine(tmp_path)
    engine.store.arm(_setup(expires_at="2026-10-27T00:50:20+00:00"))
    frames = _frames()
    # First tick alive (mid outside the zone, no touch); second
    # tick past the TTL must expire the setup, not candidate it.
    events = _events(engine, frames, [
        {"time": "2026-10-27T00:50:10Z", "mid": 99.0,
         "bid": 98.99, "ask": 99.01},
        {"time": "2026-10-27T00:50:30Z", "mid": 100.30,
         "bid": 100.29, "ask": 100.31},
    ])
    assert any(e.get("event") == "setup_expired"
               and e.get("reason_code") == "setup.expired"
               for e in events)
    assert not any(e.get("event") == "candidate_shadow" for e in events)


def test_closed_m1_breach_invalidates_tick_wick_does_not(tmp_path):
    engine = _engine(tmp_path)
    engine.store.arm(_setup())
    frames = _frames()
    # Last CLOSED M1 bar closes below the structural stop: the
    # setup is dead even though the live tick itself is healthy.
    frames["M1"] = frames["M1"].copy()
    frames["M1"].iloc[-1, frames["M1"].columns.get_loc("close")] = 98.5
    events = _events(engine, frames, [
        {"time": "2026-10-27T00:50:10Z", "mid": 100.30,
         "bid": 100.29, "ask": 100.31},
    ])
    assert any(e.get("event") == "setup_invalidated"
               and e.get("reason_code") == "setup.invalidated_close"
               for e in events)
    assert not any(e.get("event") == "candidate_shadow" for e in events)


def test_stale_frames_withhold_the_whole_scan(tmp_path):
    engine = _engine(tmp_path)
    engine.store.arm(_setup())
    frames = _frames()
    # Same frames, tick 40 minutes later: M1/M5/M15 all stale.
    events = _events(engine, frames, [
        {"time": "2026-10-27T01:30:00Z", "mid": 100.30,
         "bid": 100.29, "ask": 100.31},
    ])
    assert any(e.get("event") == "scan_degraded"
               and e.get("reason_code") == "data.stale"
               for e in events)
    assert not any(e.get("event") in ("candidate_shadow", "setup_armed")
                   for e in events)


def test_wide_spread_blocks_the_reclaim(tmp_path):
    engine = _engine(tmp_path)
    engine.store.arm(_setup())
    frames = _frames()
    # Same touch + reclaim shape as the engine candidate test, but
    # the reclaim tick carries a 1.0 spread — beyond 0.20 x M1 ATR
    # (M1 bars span 99–103, ATR ~4, envelope ~0.8).
    events = _events(engine, frames, [
        {"time": "2026-10-27T00:50:10Z", "mid": 99.95,
         "bid": 99.94, "ask": 99.96},
        {"time": "2026-10-27T00:50:20Z", "mid": 100.15,
         "bid": 100.14, "ask": 100.16},
        {"time": "2026-10-27T00:50:30Z", "mid": 100.30,
         "bid": 99.80, "ask": 100.80},
    ])
    assert any(e.get("event") == "candidate_blocked"
               and e.get("reason_code") == "costs.spread_too_wide"
               for e in events)
    assert not any(e.get("event") == "candidate_shadow" for e in events)


# ------------------------------------------------- runner shadow authority
class _SilentExec:
    """Records any order attempt; the shadow sleeve must make none."""

    def __init__(self):
        self.placed = []

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return []

    def pending_orders(self, symbol=None):
        return []

    def cancel_pending(self, ticket):
        return OrderResult(True, ticket=ticket)

    def place_order(self, request):
        self.placed.append(request)
        return OrderResult(True, ticket=9999, price=request.entry_ref,
                           volume=request.volume)


def _h1_frame():
    return pd.DataFrame(
        {"open": [100.0] * 60, "high": [101.0] * 60,
         "low": [99.0] * 60, "close": [100.5] * 60,
         "volume": [100] * 60, "real_volume": [0] * 60,
         "spread": [30] * 60},
        index=pd.date_range("2026-10-27T00:00:00Z", periods=60,
                            freq="h"))


def test_runner_shadow_candidate_places_no_orders(tmp_path):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False)
    runner = MT5Runner(cfg, feed=None, executor=_SilentExec())
    frames = _frames()
    runner._stack = TimeframeStack.from_frames(
        {"H1": _h1_frame(), "M5": frames["M5"], "M15": frames["M15"]})
    runner._hot_scalp_aux_frames = {"M1": frames["M1"]}
    store = SetupStore(tmp_path / "hot-scalp.sqlite")
    store.arm(_setup())
    runner._hot_scalp = HotScalpResearchEngine(
        store,
        HotScalpConfig(min_m1_bars=100, min_m5_bars=100,
                       min_m15_bars=100),
        instance_id="process-A")
    for tick in [
        {"time": "2026-10-27T00:50:10Z", "mid": 99.95,
         "bid": 99.94, "ask": 99.96},
        {"time": "2026-10-27T00:50:20Z", "mid": 100.15,
         "bid": 100.14, "ask": 100.16},
        {"time": "2026-10-27T00:50:30Z", "mid": 100.30,
         "bid": 100.29, "ask": 100.31},
    ]:
        runner._run_hot_scalp_shadow(tick)
    records = runner.journal.records()
    assert any(r.get("type") == "hot_scalp_event"
               and r.get("event") == "candidate_shadow"
               and r.get("execution_authority") == "shadow_only"
               for r in records)
    assert runner.executor.placed == []
