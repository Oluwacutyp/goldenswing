"""Zone object, zone-arrival triggers, approach TTL, cost viability
and the unified candidate chain (Nov 3 missed-short audit build).

The audit's requirements, pinned:
- a zone is an analytical object (formation reason, bounds,
  freshness, touch state, reaction requirement, invalidation,
  expiry) — not two numbers;
- entries can trigger AT the zone on a qualifying reaction
  (Nina's shape), measured separately from cross/reclaim;
- a setup whose price is approaching its zone earns bounded TTL
  extensions instead of expiring just before price arrives;
- cost viability is estimated at arming against the friction
  invariant, without moving the gate;
- every lifecycle transition lands on ONE candidate chain with
  explicit authority and execution-status fields.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from goldenswing.live.hot_scalp import (
    HotScalpConfig,
    HotScalpResearchEngine,
    build_hot_scalp_setups,
)
from goldenswing.live.setup_store import SetupStore


def _frame(tf_minutes, start, count, *, base=100.0):
    idx = pd.date_range(start, periods=count, freq=f"{tf_minutes}min")
    k = np.arange(count, dtype=float)
    close = base + 0.025 * np.sin(k / 2.0)
    open_ = close - 0.02 * np.cos(k)
    high = np.maximum(np.full(count, 103.0),
                        np.maximum(open_, close) + 0.1)
    low = np.minimum(np.full(count, 99.0),
                       np.minimum(open_, close) - 0.1)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close,
                         "volume": np.full(count, 100.0)}, index=idx)


def _frames():
    return {
        "M1": _frame(1, "2026-10-27T00:00:00Z", 50),
        "M5": _frame(5, "2026-10-27T00:00:00Z", 10),
        "M15": _frame(15, "2026-10-26T12:15:00Z", 50),
    }


def _short_setup(**over):
    setup = {
        "setup_id": "hs-zone-short-1", "symbol": "XAUUSD",
        "strategy": "pullback_continuation", "direction": -1,
        "formation_ts": "2026-10-27T00:40:00+00:00",
        "armed_ts": "2026-10-27T00:50:00+00:00",
        "expires_at": "2026-10-27T01:25:00+00:00",
        "level": 100.2, "zone_low": 100.0, "zone_high": 100.25,
        "stop": 100.9, "structure_ts": "2026-10-27T00:40:00+00:00",
        "context_direction": -1, "atr_m5": 0.5,
        "description": "bearish pullback into the supply zone",
    }
    setup.update(over)
    return setup


def _engine(tmp_path, cfg=None):
    store = SetupStore(tmp_path / "hot-scalp.sqlite")
    engine = HotScalpResearchEngine(
        store, cfg or HotScalpConfig(
            min_m1_bars=100, min_m5_bars=100, min_m15_bars=100),
        instance_id="process-A")
    return store, engine


def _run_ticks(engine, ticks, frames=None):
    out = []
    for tick in ticks:
        out.extend(engine.on_tick(symbol="XAUUSD", tick=tick,
                                  frames=frames or _frames(),
                                  now=tick["time"]))
    return out


_ARRIVAL_TICKS = [
    {"time": "2026-10-27T00:50:10Z", "mid": 100.60,
     "bid": 100.59, "ask": 100.61},
    {"time": "2026-10-27T00:50:20Z", "mid": 100.15,
     "bid": 100.14, "ask": 100.16},
    {"time": "2026-10-27T00:50:30Z", "mid": 100.12,
     "bid": 100.11, "ask": 100.13},
]


def test_arrival_trigger_fires_inside_the_zone(tmp_path):
    store, engine = _engine(tmp_path)
    store.arm(_short_setup())
    out = _run_ticks(engine, _ARRIVAL_TICKS)
    candidates = [e for e in out if e.get("event") == "candidate_shadow"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["trigger_mode"] == "arrival"
    assert candidate["trigger_source"] == "live_tick_arrival"
    assert candidate["targets"]["tp1"] is not None
    assert candidate["targets"]["runner"] == "open"


def test_arrival_disabled_keeps_only_cross_reclaim(tmp_path):
    store, engine = _engine(tmp_path, HotScalpConfig(
        min_m1_bars=100, min_m5_bars=100, min_m15_bars=100,
        arrival_trigger_enabled=False))
    store.arm(_short_setup())
    out = _run_ticks(engine, _ARRIVAL_TICKS)
    assert not any(e.get("event") == "candidate_shadow" for e in out)


def test_approach_extension_saves_a_forming_setup(tmp_path):
    store, engine = _engine(tmp_path)
    store.arm(_short_setup(expires_at="2026-10-27T00:45:00+00:00"))
    out = _run_ticks(engine, [
        {"time": "2026-10-27T00:50:10Z", "mid": 100.60,
         "bid": 100.59, "ask": 100.61}])
    extended = [e for e in out if e.get("event") == "setup_extended"]
    assert len(extended) == 1
    assert extended[0]["extensions"] == 1
    rows = store.active()
    assert rows and rows[0]["payload"]["extensions"] == 1


def test_no_extension_when_price_is_far(tmp_path):
    store, engine = _engine(tmp_path)
    store.arm(_short_setup(expires_at="2026-10-27T00:45:00+00:00"))
    out = _run_ticks(engine, [
        {"time": "2026-10-27T00:50:10Z", "mid": 103.50,
         "bid": 103.49, "ask": 103.51}])
    assert any(e.get("event") == "setup_expired" for e in out)
    assert not any(e.get("event") == "setup_extended" for e in out)


def test_built_setups_carry_the_zone_object():
    from tests.test_hot_scalp_engine import _trend_frame

    frames = {
        "M1": _trend_frame(1, "2026-10-27T00:49:00Z"),
        "M5": _trend_frame(5, "2026-10-27T00:45:00Z"),
        "M15": _trend_frame(15, "2026-10-27T00:30:00Z"),
    }
    setups = build_hot_scalp_setups(
        "XAUUSD", frames, "2026-10-27T00:50:30Z", HotScalpConfig())
    assert setups
    for setup in setups:
        zone = setup["zone"]
        assert zone["low"] == setup["zone_low"]
        assert zone["high"] == setup["zone_high"]
        assert zone["formation"] == setup["description"]
        assert zone["invalidation"] == setup["stop"]
        assert zone["state"] == "armed"
        assert "arrival" in zone["reaction"]["modes"]
        assert zone["freshness_s"] >= 0


# ------------------------------------------------------------------
# unified candidate chain (runner side)
# ------------------------------------------------------------------

class _Journal:
    def __init__(self):
        self.records = []
        self.notes = []

    def append(self, record):
        self.records.append(record)

    def log_note(self, text):
        self.notes.append(str(text))


def _runner(tmp_path):
    from goldenswing.execution.mt5_runner import (
        MT5Runner,
        MT5RunnerConfig,
    )
    from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
    from mt5_fakes import FakeMT5

    feed = MT5Feed(FakeMT5(), MT5FeedConfig(symbol="XAUUSDm"))
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False, max_daily_trades=0,
        max_hourly_trades=0)
    runner = MT5Runner(cfg, feed=feed)
    runner.journal = _Journal()
    return runner


def test_chain_links_the_whole_lifecycle(tmp_path):
    runner = _runner(tmp_path)
    tick = {"time": "2026-11-03T03:45:00Z"}
    runner._hs_chain_from_event({
        "event": "setup_armed",
        "setup": {"setup_id": "hs-1", "parent_setup_id": "hp-1",
                  "strategy": "pullback_continuation", "direction": -1,
                  "zone": {"low": 3928.0, "high": 3930.0},
                  "stop": 3930.2, "expires_at": "x",
                  "cost_viability": {"viable": False}}}, tick)
    runner._hs_chain_from_event(
        {"event": "setup_touched", "setup_id": "hs-1"}, tick)
    runner._hs_chain_from_event({
        "event": "candidate_shadow", "setup_id": "hs-1",
        "parent_setup_id": "hp-1", "strategy": "pullback_continuation",
        "direction": -1, "entry_reference": 3928.96,
        "structural_stop": 3930.19, "target_reference": 3911.93,
        "spread": 0.24, "score": 82.7,
        "targets": {"tp1": 3911.93, "tp2": None, "runner": "open"}},
        tick)
    runner._hs_chain_from_event(
        {"event": "setup_expired", "setup_id": "hs-1"}, tick)
    chain = [r for r in runner.journal.records
             if r.get("type") == "candidate_chain"]
    assert [r["stage"] for r in chain] == [
        "armed", "zone_touched", "triggered", "expired"]
    assert all(r["chain_id"] == "hs-1" for r in chain)
    # the expired event carried no parent — the chain resolved it
    # from the armed record's mapping
    assert all(r["parent_id"] == "hp-1" for r in chain)
    assert all(r["authority"]["research_stream"] == "shadow_only"
               for r in chain)
    triggered = chain[2]
    assert triggered["targets"]["runner"] == "open"
    assert triggered["spread"] == 0.24


def test_chain_records_a_promotion_refusal(tmp_path):
    runner = _runner(tmp_path)
    runner.cfg.hot_scalp_live_enabled = False
    ok = runner._promote_hot_scalp_candidate(
        {"setup_id": "hs-9", "parent_setup_id": "hp-9"},
        {"time": "2026-11-03T03:45:00Z"}, {})
    assert ok is False
    chain = [r for r in runner.journal.records
             if r.get("type") == "candidate_chain"]
    assert len(chain) == 1
    assert chain[0]["stage"] == "promotion"
    assert chain[0]["execution_status"] == "refused"
    assert chain[0]["reason_code"] == "promotion.live_authority_off"
    assert chain[0]["authority"]["live_promotion"] is False
