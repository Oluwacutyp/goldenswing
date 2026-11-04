"""Hot Scalp continuation-target planning (owner approval
2026-11-04, workstream 2 — his PRIORITY item).

The disease, from the owner's live journal (Nov 4): the sleeve
armed good bullish zones, but at the touch its geometry priced
ONLY the nearest level found in the touch-time frame — usually a
micro-swing the travel to the zone had already eaten.  Exhibits:

- hs-63dc10ef (armed 01:55Z, FVG 3942.517-3947.592, stop
  3941.375078): touched, then blocked at 02:02:02Z with
  risk 5.585922 / reward 0.685 / net_rr 0.076383 against a
  legacy target of 3947.646 — while the creation structure's
  real objectives (the 3955.51 shelf, the 3964.50 zone) sat
  untouched above the zone.
- hs-ade06913 (armed 07:00Z, FVG 4020.418-4021.269, stop
  4019.056827): blocked at 07:04:00Z with risk 1.453173 /
  reward 0.403 / net_rr 0.096269 against a legacy target of
  4020.913 — INSIDE the zone being entered.  Its one true
  structural objective (4022.522, the 06:00 bar's extreme) still
  does not clear the unchanged 1.20R floor at the touch, so the
  new planner must block it too — honestly, with the table.

The fixtures replay the recorded touch geometry (mid, spread,
stop from the journal records; information at the time only).
The creation frames are reconstructed from journal-established
price facts (recorded zone touches at 3955.13/3955.51, recorded
zone level 3964.50, the 4022.522 reconciliation extreme).
"""
import numpy as np
import pandas as pd
import pytest

from goldenswing.live.hot_scalp import (
    HotScalpConfig, HotScalpResearchEngine, _plan_targets,
    _target_hierarchy, build_hot_scalp_setups)
from goldenswing.live.setup_store import SetupStore
from goldenswing.execution.mt5_runner import _hot_scalp_order_targets


# ---------------------------------------------------------------- frames

def _m5_frame(highs, end_open):
    idx = pd.date_range(end=end_open, periods=len(highs), freq="5min")
    high = np.array(highs, dtype=float)
    close = high - 0.6
    return pd.DataFrame(
        {"open": close - 0.1, "high": high, "low": close - 0.5,
         "close": close, "volume": np.full(len(highs), 100.0)},
        index=idx)


def _m1_frame(end_open, count, *, base, rng=1.5):
    idx = pd.date_range(end=end_open, periods=count, freq="1min")
    close = np.full(count, float(base))
    return pd.DataFrame(
        {"open": close.copy(), "high": close + rng / 2.0,
         "low": close - rng / 2.0, "close": close,
         "volume": np.full(count, 100.0)}, index=idx)


def _m15_frame(end_open, count, *, lo, hi):
    idx = pd.date_range(end=end_open, periods=count, freq="15min")
    close = np.linspace(lo, hi, count)
    return pd.DataFrame(
        {"open": close - 0.2, "high": close + 1.0, "low": close - 1.0,
         "close": close, "volume": np.full(count, 100.0)}, index=idx)


def _drive(engine, symbol, ticks, frames):
    out = []
    for tick in ticks:
        out.extend(engine.on_tick(symbol=symbol, tick=tick,
                                  frames=frames, now=tick["time"]))
    return out


# ---------------------------------------------------------------- planner units

def test_plan_tp1_exhausted_tp2_viable_plans_continuation():
    plan = _plan_targets(hierarchy_prices=[100.55, 104.0], direction=1,
                         entry_mid=100.45, spread=0.02, stop=99.0,
                         cfg=HotScalpConfig())
    # Attainable entry is the ask: 100.46.  TP1 has 0.09 gross
    # left — spent.  TP2 still pays 3.54 gross / ~2.4R net.
    assert plan["entry_attainable"] == pytest.approx(100.46)
    assert plan["legs"][0]["verdict"] in ("exhausted", "rejected")
    assert plan["legs"][1]["verdict"] == "selected"
    assert plan["plan_kind"] == "continuation"
    assert plan["planned_objective"] == 104.0
    assert plan["planned_targets"] == [104.0]


def test_plan_all_objectives_spent_blocks():
    plan = _plan_targets(hierarchy_prices=[100.3], direction=1,
                         entry_mid=100.45, spread=0.02, stop=99.0,
                         cfg=HotScalpConfig())
    assert plan["legs"][0]["verdict"] == "exhausted"
    assert plan["planned_objective"] is None
    assert plan["plan_kind"] == "blocked"
    assert plan["planned_targets"] == []


def test_plan_never_invents_a_target():
    for prices in ([], [100.5], [100.5, 101.0, 105.0]):
        plan = _plan_targets(hierarchy_prices=list(prices), direction=1,
                             entry_mid=100.4, spread=0.02, stop=99.0,
                             cfg=HotScalpConfig())
        assert [leg["price"] for leg in plan["legs"]] == list(prices)
        assert set(plan["planned_targets"]) <= set(prices)
    empty = _plan_targets(hierarchy_prices=[], direction=1,
                          entry_mid=100.4, spread=0.02, stop=99.0,
                          cfg=HotScalpConfig())
    assert empty["plan_kind"] == "no_target"


def test_plan_short_side_mirrors():
    plan = _plan_targets(hierarchy_prices=[99.45, 96.0], direction=-1,
                         entry_mid=99.55, spread=0.02, stop=101.0,
                         cfg=HotScalpConfig())
    # Attainable entry is the bid: 99.54.
    assert plan["entry_attainable"] == pytest.approx(99.54)
    assert plan["plan_kind"] == "continuation"
    assert plan["planned_objective"] == 96.0


def test_hierarchy_anchor_excludes_in_zone_swings():
    # Exhibit B's shape: a confirmed swing INSIDE the entry zone
    # (4020.913) must never become an objective; only the swing
    # beyond the zone edge (4022.522) is in the hierarchy.
    frame = _m5_frame(
        [4001, 4008, 4015, 4019, 4022.522, 4022.0, 4021.6, 4021.2,
         4020.9, 4021.0, 4020.8, 4020.95, 4020.7, 4020.6, 4020.85,
         4020.7], "2026-11-04T06:50:00Z")
    hierarchy = _target_hierarchy(frame, 1, 4020.418, 4021.269)
    assert [o["price"] for o in hierarchy] == [4022.522]
    assert hierarchy[0]["role"] == "tp1"


# ---------------------------------------------------------------- creation

def _trend_frame(tf_minutes, end, count=120):
    idx = pd.date_range(end=end, periods=count, freq=f"{tf_minutes}min")
    k = np.arange(count, dtype=float)
    close = 100.0 + 0.025 * k + 0.18 * np.sin(k * 0.55)
    open_ = close - 0.04 * np.sign(np.cos(k * 0.55))
    high = np.maximum(open_, close) + 0.08
    low = np.minimum(open_, close) - 0.08
    if tf_minutes == 5:
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
                         "close": close,
                         "volume": np.full(count, 100.0)}, index=idx)


def test_creation_attaches_hierarchy_only_when_flag_on():
    frames = {
        "M1": _trend_frame(1, "2026-10-27T00:49:00Z"),
        "M5": _trend_frame(5, "2026-10-27T00:45:00Z"),
        "M15": _trend_frame(15, "2026-10-27T00:30:00Z"),
    }
    now = "2026-10-27T00:50:30Z"
    on = build_hot_scalp_setups("XAUUSD", frames, now, HotScalpConfig())
    off = build_hot_scalp_setups(
        "XAUUSD", frames, now,
        HotScalpConfig(multi_target_planning_enabled=False))
    assert on and off
    for setup in on:
        assert "target_hierarchy" in setup
        for obj in setup["target_hierarchy"]:
            if setup["direction"] > 0:
                assert obj["price"] > setup["zone_high"]
            else:
                assert obj["price"] < setup["zone_low"]
    for setup in off:
        assert "target_hierarchy" not in setup
    # Identity is planning-independent: the flag must not change
    # which setups exist.
    assert {s["setup_id"] for s in on} == {s["setup_id"] for s in off}


# ---------------------------------------------------------------- Exhibit A

_A_SETUP = {
    "setup_id": "hs-63dc10ef330c904b42036f37ed9d",
    "parent_setup_id": "hp-17d0c51096bd9cbe8b15d4ea64ea",
    "symbol": "XAUUSDm", "strategy": "pullback_continuation",
    "direction": 1,
    "formation_ts": "2026-11-04T01:20:00+00:00",
    "armed_ts": "2026-11-04T01:55:03+00:00",
    "expires_at": "2026-11-04T02:05:00+00:00",
    "level": 3945.0545, "zone_low": 3942.517, "zone_high": 3947.592,
    "stop": 3941.375077798541,
    "structure_ts": "2026-11-04T01:15:00+00:00",
    "context_direction": 1, "atr_m5": 7.612814676392063,
    "description": "M15 HH + HL; M5 BOS and unfilled FVG "
                   "3942.51700-3947.59200",
}

# Creation frame (ends 01:20): the only confirmed swing highs
# above the zone are the recorded shelf 3955.51 and zone 3964.50.
_A_CREATION_HIGHS = [3938, 3944, 3952, 3960, 3962, 3964.50, 3961,
                     3958, 3955, 3953, 3951.5, 3952.5, 3955.51, 3951,
                     3946, 3944.5, 3943.8, 3944.2, 3943.5, 3944.0]
# Touch frame (ends 01:50): the approach carved a micro swing at
# 3947.646 — the legacy planner's target in the journal record.
_A_TOUCH_HIGHS = [3951.0, 3949.0, 3947.0, 3945.5, 3944.8, 3944.2,
                  3947.646, 3946.5, 3945.9, 3946.8, 3945.2, 3944.9]


def _exhibit_a_frames():
    return {
        "M1": _m1_frame("2026-11-04T01:57:00Z", 30, base=3945.0),
        "M5": _m5_frame(_A_TOUCH_HIGHS, "2026-11-04T01:50:00Z"),
        "M15": _m15_frame("2026-11-04T01:30:00Z", 40, lo=3930, hi=3944),
    }


_A_TICKS = [
    {"time": "2026-11-04T01:58:07+00:00", "mid": 3944.0,
     "bid": 3943.88, "ask": 3944.12},
    # The recorded touch geometry: mid = stop + 5.585922, spread 0.24.
    {"time": "2026-11-04T01:58:40+00:00", "mid": 3946.961,
     "bid": 3946.841, "ask": 3947.081},
]


def _exhibit_a_engine(tmp_path, cfg):
    store = SetupStore(tmp_path / "hs.sqlite")
    setup = dict(_A_SETUP)
    creation = _m5_frame(_A_CREATION_HIGHS, "2026-11-04T01:20:00Z")
    setup["target_hierarchy"] = _target_hierarchy(
        creation, 1, setup["zone_low"], setup["zone_high"])
    assert [o["price"] for o in setup["target_hierarchy"]] == \
        [3955.51, 3964.50]
    store.arm(setup)
    return HotScalpResearchEngine(store, cfg, instance_id="fixture-A")


def test_exhibit_a_continuation_objective_was_there_all_along(tmp_path):
    engine = _exhibit_a_engine(tmp_path, HotScalpConfig())
    events = _drive(engine, "XAUUSDm", _A_TICKS, _exhibit_a_frames())
    mine = [e for e in events
            if e.get("setup_id") == _A_SETUP["setup_id"]]
    candidates = [e for e in mine if e["event"] == "candidate_shadow"]
    assert len(candidates) == 1
    cand = candidates[0]
    plan = cand["target_plan"]
    # The legacy planner on the same touch reproduces the journal
    # block EXACTLY — the shadow comparison is the audit trail.
    legacy = cand["legacy_geometry"]
    assert legacy["target"] == pytest.approx(3947.646)
    assert legacy["risk"] == pytest.approx(5.585922201459198)
    assert legacy["reward"] == pytest.approx(0.6849999999999454)
    assert legacy["net_rr"] == pytest.approx(0.07638275703170794)
    assert legacy["verdict"] == "blocked"
    # The new planner prices the creation hierarchy instead.
    assert [leg["price"] for leg in plan["legs"]] == \
        [3955.51, 3964.50]
    assert plan["legs"][0]["verdict"] == "selected"
    assert plan["legs"][1]["verdict"] == "viable"
    assert plan["plan_kind"] == "standard"
    assert plan["hierarchy_source"] == "creation"
    assert cand["target_reference"] == pytest.approx(3955.51)
    assert cand["planned_targets"] == [3955.51, 3964.50]
    assert cand["net_reward_risk"] == pytest.approx(
        plan["legs"][0]["net_rr"])
    assert cand["net_reward_risk"] >= 1.20
    assert cand["targets"]["tp1"] == pytest.approx(3955.51)
    assert cand["targets"]["planned_objective"] == \
        pytest.approx(3955.51)
    # The invented-target guard: the legacy micro-swing appears in
    # NO leg of the plan.
    assert 3947.646 not in [leg["price"] for leg in plan["legs"]]


def test_exhibit_a_flag_off_is_byte_identical_legacy(tmp_path):
    cfg = HotScalpConfig(multi_target_planning_enabled=False)
    engine = _exhibit_a_engine(tmp_path, cfg)
    events = _drive(engine, "XAUUSDm", _A_TICKS, _exhibit_a_frames())
    mine = [e for e in events
            if e.get("setup_id") == _A_SETUP["setup_id"]]
    blocked = [e for e in mine if e["event"] == "candidate_blocked"]
    assert len(blocked) == 1
    event = blocked[0]
    # Exactly the journal record's shape — no plan fields leak in.
    assert set(event) == {"event", "setup_id", "reason_code",
                          "risk", "reward", "net_rr"}
    assert event["reason_code"] == "geometry.net_rr_below_min"
    assert event["risk"] == pytest.approx(5.585922201459198)
    assert event["reward"] == pytest.approx(0.6849999999999454)
    assert event["net_rr"] == pytest.approx(0.07638275703170794)
    assert not any(e["event"] == "candidate_shadow" for e in mine)


# ---------------------------------------------------------------- Exhibit B

_B_SETUP = {
    "setup_id": "hs-ade06913864859add85e251814e6",
    "parent_setup_id": "hp-0c882908882276d7643eec10397d",
    "symbol": "XAUUSDm", "strategy": "pullback_continuation",
    "direction": 1,
    "formation_ts": "2026-11-04T06:55:00+00:00",
    "armed_ts": "2026-11-04T07:00:00+00:00",
    "expires_at": "2026-11-04T07:40:00+00:00",
    "level": 4020.8435, "zone_low": 4020.418, "zone_high": 4021.269,
    "stop": 4019.0568265384536,
    "structure_ts": "2026-11-04T06:25:00+00:00",
    "context_direction": 1, "atr_m5": 9.07448974364392,
    "description": "M15 HH + HL; M5 BOS and unfilled FVG "
                   "4020.41800-4021.26900",
}

_B_CREATION_HIGHS = [4001, 4008, 4015, 4019, 4022.522, 4022.0,
                     4021.6, 4021.2, 4020.9, 4021.0, 4020.8, 4020.95,
                     4020.7, 4020.6, 4020.85, 4020.7]
_B_TOUCH_HIGHS = [4001, 4008, 4015, 4019, 4022.522, 4022.0, 4021.6,
                  4021.2, 4020.9, 4020.7, 4020.6, 4020.55, 4020.913,
                  4020.8, 4020.65, 4020.6, 4020.75, 4020.7, 4020.8,
                  4020.72]

_B_TICKS = [
    {"time": "2026-11-04T07:03:41+00:00", "mid": 4020.45,
     "bid": 4020.33, "ask": 4020.57},
    {"time": "2026-11-04T07:04:00+00:00", "mid": 4020.44,
     "bid": 4020.32, "ask": 4020.56},
    # The recorded touch geometry: mid = stop + 1.453173, spread 0.24.
    {"time": "2026-11-04T07:04:20+00:00", "mid": 4020.510,
     "bid": 4020.39, "ask": 4020.63},
]


def test_exhibit_b_blocks_honestly_with_the_full_table(tmp_path):
    store = SetupStore(tmp_path / "hs.sqlite")
    setup = dict(_B_SETUP)
    creation = _m5_frame(_B_CREATION_HIGHS, "2026-11-04T06:50:00Z")
    setup["target_hierarchy"] = _target_hierarchy(
        creation, 1, setup["zone_low"], setup["zone_high"])
    assert [o["price"] for o in setup["target_hierarchy"]] == \
        [4022.522]
    store.arm(setup)
    engine = HotScalpResearchEngine(store, HotScalpConfig(),
                                    instance_id="fixture-B")
    frames = {
        "M1": _m1_frame("2026-11-04T07:03:00Z", 30, base=4020.5),
        "M5": _m5_frame(_B_TOUCH_HIGHS, "2026-11-04T06:55:00Z"),
        "M15": _m15_frame("2026-11-04T06:45:00Z", 40, lo=3990, hi=4020),
    }
    events = _drive(engine, "XAUUSDm", _B_TICKS, frames)
    mine = [e for e in events
            if e.get("setup_id") == _B_SETUP["setup_id"]]
    blocked = [e for e in mine if e["event"] == "candidate_blocked"]
    assert len(blocked) == 1
    event = blocked[0]
    assert event["reason_code"] == "geometry.net_rr_below_min"
    # Headline numbers stay the journal's legacy ones.
    assert event["risk"] == pytest.approx(1.453173461546612)
    assert event["reward"] == pytest.approx(0.40299999999979264)
    assert event["net_rr"] == pytest.approx(0.09626893150725543)
    # ...and the plan table shows WHY, against the true hierarchy:
    # one real objective, priced honestly, still below the floor.
    plan = event["target_plan"]
    assert [leg["price"] for leg in plan["legs"]] == [4022.522]
    leg = plan["legs"][0]
    assert leg["verdict"] == "rejected"
    assert leg["remaining_reward"] == pytest.approx(1.892, abs=1e-3)
    assert leg["net_rr"] == pytest.approx(1.126, abs=1e-3)
    assert leg["net_rr"] < 1.20
    assert plan["planned_objective"] is None
    assert plan["plan_kind"] == "blocked"
    assert not any(e["event"] == "candidate_shadow" for e in mine)


# ---------------------------------------------------------------- engine behaviour

def _flat_frames():
    idx1 = pd.date_range("2026-10-27T00:00:00Z", periods=50, freq="1min")
    idx5 = pd.date_range("2026-10-27T00:00:00Z", periods=10, freq="5min")
    idx15 = pd.date_range("2026-10-26T12:15:00Z", periods=50,
                          freq="15min")

    def frame(idx, base):
        close = np.full(len(idx), base)
        return pd.DataFrame(
            {"open": close.copy(), "high": close + 0.15,
             "low": close - 0.15, "close": close,
             "volume": np.full(len(idx), 100.0)}, index=idx)

    return {"M1": frame(idx1, 100.1), "M5": frame(idx5, 100.1),
            "M15": frame(idx15, 100.1)}


def _arm_synthetic(store, hierarchy):
    setup = {
        "setup_id": "hs-synthetic-1", "symbol": "XAUUSD",
        "strategy": "breakout_retest", "direction": 1,
        "formation_ts": "2026-10-27T00:40:00+00:00",
        "armed_ts": "2026-10-27T00:50:00+00:00",
        "expires_at": "2026-10-27T01:25:00+00:00",
        "level": 100.2, "zone_low": 100.0, "zone_high": 100.25,
        "stop": 99.0, "structure_ts": "2026-10-27T00:40:00+00:00",
        "context_direction": 1, "atr_m5": 0.5,
        "description": "synthetic continuation fixture",
        "target_hierarchy": hierarchy,
    }
    store.arm(setup)
    return setup


def test_engine_continuation_plan_skips_exhausted_tp1(tmp_path):
    store = SetupStore(tmp_path / "hs.sqlite")
    _arm_synthetic(store, [
        {"rank": 1, "role": "tp1", "price": 100.55,
         "source": "m5_swing_high", "swing_ts": None},
        {"rank": 2, "role": "tp2", "price": 104.0,
         "source": "m5_swing_high", "swing_ts": None},
    ])
    cfg = HotScalpConfig(min_m1_bars=100, min_m5_bars=100,
                         min_m15_bars=100)
    engine = HotScalpResearchEngine(store, cfg, instance_id="proc-A")
    ticks = [
        {"time": "2026-10-27T00:50:10+00:00", "mid": 100.05,
         "bid": 100.04, "ask": 100.06},
        {"time": "2026-10-27T00:50:20+00:00", "mid": 100.45,
         "bid": 100.44, "ask": 100.46},
    ]
    events = _drive(engine, "XAUUSD", ticks, _flat_frames())
    candidates = [e for e in events
                  if e["event"] == "candidate_shadow"]
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["target_plan"]["plan_kind"] == "continuation"
    assert cand["target_reference"] == pytest.approx(104.0)
    assert cand["planned_targets"] == [104.0]
    assert cand["targets"]["tp1"] == pytest.approx(100.55)
    assert cand["targets"]["planned_objective"] == \
        pytest.approx(104.0)


def test_engine_all_objectives_spent_blocks_with_table(tmp_path):
    store = SetupStore(tmp_path / "hs.sqlite")
    _arm_synthetic(store, [
        {"rank": 1, "role": "tp1", "price": 100.3,
         "source": "m5_swing_high", "swing_ts": None},
    ])
    cfg = HotScalpConfig(min_m1_bars=100, min_m5_bars=100,
                         min_m15_bars=100)
    engine = HotScalpResearchEngine(store, cfg, instance_id="proc-A")
    ticks = [
        {"time": "2026-10-27T00:50:10+00:00", "mid": 100.05,
         "bid": 100.04, "ask": 100.06},
        {"time": "2026-10-27T00:50:20+00:00", "mid": 100.45,
         "bid": 100.44, "ask": 100.46},
    ]
    events = _drive(engine, "XAUUSD", ticks, _flat_frames())
    blocked = [e for e in events if e["event"] == "candidate_blocked"]
    assert len(blocked) == 1
    assert blocked[0]["reason_code"] == "geometry.net_rr_below_min"
    assert blocked[0]["target_plan"]["legs"][0]["verdict"] == \
        "exhausted"
    assert not any(e["event"] == "candidate_shadow" for e in events)


def test_engine_spread_envelope_still_blocks_before_planning(tmp_path):
    # The cost envelope is upstream of geometry and unchanged: a
    # widened spread blocks as costs.spread_too_wide and the
    # planner never gets to rescue the setup.
    store = SetupStore(tmp_path / "hs.sqlite")
    _arm_synthetic(store, [
        {"rank": 1, "role": "tp1", "price": 104.0,
         "source": "m5_swing_high", "swing_ts": None},
    ])
    cfg = HotScalpConfig(min_m1_bars=100, min_m5_bars=100,
                         min_m15_bars=100, max_spread_atr_m1=0.001)
    engine = HotScalpResearchEngine(store, cfg, instance_id="proc-A")
    ticks = [
        {"time": "2026-10-27T00:50:10+00:00", "mid": 100.05,
         "bid": 100.04, "ask": 100.06},
        {"time": "2026-10-27T00:50:20+00:00", "mid": 100.45,
         "bid": 100.44, "ask": 100.46},
    ]
    events = _drive(engine, "XAUUSD", ticks, _flat_frames())
    blocked = [e for e in events if e["event"] == "candidate_blocked"]
    assert len(blocked) == 1
    assert blocked[0]["reason_code"] == "costs.spread_too_wide"
    assert "target_plan" not in blocked[0]


# ---------------------------------------------------------------- promotion ladder

def test_order_targets_prefers_the_planned_ladder():
    cand = {"planned_targets": [3955.51, 3964.50],
            "targets": {"tp1": 3955.51, "tp2": 3964.50},
            "target_reference": 3955.51}
    assert _hot_scalp_order_targets(cand) == [3955.51, 3964.50]
    continuation = {"planned_targets": [104.0],
                    "targets": {"tp1": 100.55, "tp2": 104.0},
                    "target_reference": 104.0}
    assert _hot_scalp_order_targets(continuation) == [104.0]


def test_order_targets_legacy_selection_unchanged():
    staged = {"targets": {"tp1": 101.0, "tp2": 102.0},
              "target_reference": 101.0}
    assert _hot_scalp_order_targets(staged) == [101.0, 102.0]
    tp1_only = {"targets": {"tp1": 101.0, "tp2": None},
                "target_reference": 101.0}
    assert _hot_scalp_order_targets(tp1_only) == [101.0]
    bare = {"target_reference": 99.5}
    assert _hot_scalp_order_targets(bare) == [99.5]
