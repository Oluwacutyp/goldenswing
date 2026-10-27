import numpy as np
import pandas as pd

from goldenswing.live.hot_scalp import HotScalpConfig, HotScalpResearchEngine
from goldenswing.live.setup_store import SetupStore


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


def test_hot_scalp_emits_one_shadow_candidate_after_live_touch_and_reclaim(tmp_path):
    store = SetupStore(tmp_path / "hot-scalp.sqlite")
    setup = {
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
    store.arm(setup)
    cfg = HotScalpConfig(min_m1_bars=100, min_m5_bars=100,
                         min_m15_bars=100)
    engine = HotScalpResearchEngine(store, cfg, instance_id="process-A")
    frames = {
        "M1": _frame(1, "2026-10-27T00:00:00Z", 50),
        "M5": _frame(5, "2026-10-27T00:00:00Z", 10),
        "M15": _frame(15, "2026-10-26T12:15:00Z", 50),
    }
    # Frames end at closed-bar opens 00:49 (M1), 00:45 (M5), and 00:30 (M15).
    ticks = [
        {"time": "2026-10-27T00:50:10Z", "mid": 99.95, "bid": 99.94, "ask": 99.96},
        {"time": "2026-10-27T00:50:20Z", "mid": 100.15, "bid": 100.14, "ask": 100.16},
        {"time": "2026-10-27T00:50:30Z", "mid": 100.30, "bid": 100.29, "ask": 100.31},
    ]
    output = []
    for tick in ticks:
        output.extend(engine.on_tick(symbol="XAUUSD", tick=tick,
                                     frames=frames, now=tick["time"]))
    candidates = [event for event in output if event.get("event") == "candidate_shadow"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["parent_setup_id"] == setup["setup_id"]
    assert candidate["trigger_source"] == "live_tick_reclaim"
    assert candidate["execution_authority"] == "shadow_only"
    assert 0 <= candidate["score"] <= 100
    assert candidate["score_is_calibrated_probability"] is False
    evidence_categories = {row["category"] for row in candidate["evidence"]}
    assert "momentum" in evidence_categories  # M1 EMA(9)/EMA(21), non-gating
    assert "timing" in evidence_categories    # session state is always visible
    assert candidate["session_context"]["session"]
    assert candidate["stacking_policy"]["max_layers"] == 3
    assert candidate["stacking_policy"]["shared_parent_risk_budget"] is True
    # A repeated scan/replayed trigger cannot emit a second candidate.
    output2 = engine.on_tick(symbol="XAUUSD", tick=ticks[-1], frames=frames,
                             now=ticks[-1]["time"])
    assert not any(event.get("event") == "candidate_shadow" for event in output2)


def _trend_frame(tf_minutes, end, count=120):
    idx = pd.date_range(end=end, periods=count, freq=f"{tf_minutes}min")
    k = np.arange(count, dtype=float)
    close = 100.0 + 0.025 * k + 0.18 * np.sin(k * 0.55)
    open_ = close - 0.04 * np.sign(np.cos(k * 0.55))
    high = np.maximum(open_, close) + 0.08
    low = np.minimum(open_, close) - 0.08
    if tf_minutes == 5:
        # A fresh bullish displacement/FVG against prior swing structure.
        high[-3] = close[-3] + 0.03
        open_[-2] = close[-2] - 0.02
        close[-2] = close[-2] - 0.03
        high[-2] = open_[-2] + 0.06
        low[-2] = close[-2] - 0.06
        open_[-1] = close[-2] + 0.20
        close[-1] = open_[-1] + 1.60
        high[-1] = close[-1] + 0.05
        low[-1] = open_[-1] - 0.05
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": np.full(count, 100.0)},
                        index=idx)


def test_strategy_families_arm_persistent_child_entries_under_one_parent():
    from goldenswing.live.hot_scalp import build_hot_scalp_setups

    frames = {
        "M1": _trend_frame(1, "2026-10-27T00:49:00Z"),
        "M5": _trend_frame(5, "2026-10-27T00:45:00Z"),
        "M15": _trend_frame(15, "2026-10-27T00:30:00Z"),
    }
    now = "2026-10-27T00:50:30Z"
    first = build_hot_scalp_setups("XAUUSD", frames, now, HotScalpConfig())
    second = build_hot_scalp_setups("XAUUSD", frames, "2026-10-27T00:50:45Z",
                                    HotScalpConfig())
    assert {row["strategy"] for row in first} >= {
        "pullback_continuation", "breakout_retest"}
    pullbacks = [row for row in first if row["strategy"] == "pullback_continuation"]
    assert len(pullbacks) >= 2
    assert len({row["parent_setup_id"] for row in pullbacks}) == 1
    assert len({row["setup_id"] for row in pullbacks}) == len(pullbacks)
    assert {row["setup_id"] for row in first} == {row["setup_id"] for row in second}
