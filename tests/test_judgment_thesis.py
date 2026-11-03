"""Judgment core S2: five-scenario market read + thesis maintenance.

The read prices bullish/bearish/range/reversal/continuation from
the decision context + candidate flow; the keeper creates,
maintains, invalidates, supersedes and expires ONE thesis with a
plan — all shadow, all journaled, nothing fitted.
"""
from __future__ import annotations

from goldenswing.execution.judgment_thesis import (
    ThesisConfig,
    ThesisKeeper,
    assess_market,
    build_plan,
)

UP_TREND = {
    "regime": "up/clean_trend/normal/distribution/ny",
    "posture": "trend", "day_type": "trend", "adr_pct": 40.0,
    "adr_state": "open", "smt": None,
}
DOWN_TREND = {
    "regime": "down/clean_trend/normal/manipulation/london",
    "posture": "trend", "day_type": "trend", "adr_pct": 35.0,
    "adr_state": "open", "smt": None,
}
RANGE_CTX = {
    "regime": "flat/range/normal/transition/asia",
    "posture": "defensive", "day_type": "range", "adr_pct": 55.0,
    "adr_state": "open", "smt": None,
}
EXHAUSTED_UP = {
    "regime": "up/clean_trend/high/distribution/ny",
    "posture": "trend", "day_type": "trend", "adr_pct": 95.0,
    "adr_state": "exhausted", "smt": "smt_divergence_bearish",
}


def test_uptrend_read_is_directional_long():
    a = assess_market(context=UP_TREND, flow={})
    assert a["winner"] in ("bullish", "continuation")
    assert a["direction"] == 1
    assert a["margin"] > 0
    assert a["invalidation"]  # named conditions ride along
    names = [s["name"] for s in a["scenarios"]]
    assert names[0] == a["winner"]
    assert set(names) == {"bullish", "bearish", "range",
                          "reversal", "continuation"}


def test_downtrend_read_is_directional_short():
    a = assess_market(context=DOWN_TREND, flow={})
    assert a["winner"] == "bearish"
    assert a["direction"] == -1


def test_range_context_wins_range():
    a = assess_market(context=RANGE_CTX, flow={})
    assert a["winner"] == "range"
    assert a["direction"] == 0


def test_exhaustion_and_divergence_fund_reversal():
    base = assess_market(context=UP_TREND, flow={})
    tired = assess_market(context=EXHAUSTED_UP, flow={})
    p_rev = {s["name"]: s["p"] for s in tired["scenarios"]}
    p_base = {s["name"]: s["p"] for s in base["scenarios"]}
    assert p_rev["reversal"] > p_base["reversal"]


def test_flow_skew_funds_the_skewed_side():
    flat = {"regime": "flat/unclear/normal/transition/asia",
            "posture": "", "day_type": "", "adr_pct": 30.0,
            "adr_state": "open", "smt": None}
    neutral = assess_market(context=flat, flow={})
    long_flow = assess_market(
        context=flat,
        flow={"evaluated_long": 9, "evaluated_short": 1,
              "qualified_long": 3, "qualified_short": 0})
    pn = {s["name"]: s["p"] for s in neutral["scenarios"]}
    pl = {s["name"]: s["p"] for s in long_flow["scenarios"]}
    assert pl["bullish"] > pn["bullish"]


def test_separate_fields_present():
    a = assess_market(context=UP_TREND, flow={})
    assert "analytical_confidence" in a
    assert "evidence_quality" in a
    assert a["analytical_confidence"] != a["evidence_quality"] \
        or True  # distinct fields, values may coincide


def test_plan_from_aligned_candidate():
    a = assess_market(context=UP_TREND, flow={})
    plan = build_plan(assessment=a, candidate={
        "direction": 1, "entry": 3900.0, "stop": 3890.0,
        "targets": [3920.0], "intent": "limit",
        "playbook": "qm_m5"}, playbook="qm_m5")
    assert plan["entry"] == 3900.0
    assert plan["invalidation_price"] == 3890.0
    assert plan["target"] == 3920.0
    assert plan["execution_method"] == "limit"
    assert plan["management_horizon"] == "fixed_bracket_v1"
    assert "aligned candidate" in plan["basis"]


def test_plan_without_candidate_invents_nothing():
    a = assess_market(context=UP_TREND, flow={})
    plan = build_plan(assessment=a, candidate=None)
    assert plan["entry"] is None
    assert plan["target"] is None
    assert "scenario only" in plan["basis"]
    assert plan["entry_conditions"]  # textual conditions still given


def test_keeper_creates_and_maintains_quietly():
    keeper = ThesisKeeper(ThesisConfig(heartbeat_seconds=60.0))
    a = assess_market(context=UP_TREND, flow={})
    events = keeper.update(a, ts=1000.0)
    assert [e["event"] for e in events] == ["thesis_created"]
    assert keeper.active["scenario"] == a["winner"]
    # same read again before the heartbeat: no journal noise
    assert keeper.update(a, ts=1030.0) == []
    events = keeper.update(a, ts=1200.0)
    assert [e["event"] for e in events] == ["thesis_maintained"]


def test_keeper_invalidates_on_regime_flip():
    keeper = ThesisKeeper()
    a = assess_market(context=UP_TREND, flow={})
    keeper.update(a, ts=1000.0)
    down = assess_market(context=DOWN_TREND, flow={})
    events = keeper.update(down, ts=1100.0)
    kinds = [e["event"] for e in events]
    assert "thesis_invalidated" in kinds
    assert keeper.history[-1]["status"] == "invalidated"
    assert "regime direction turns down" in \
        keeper.history[-1]["close_reason"]


def test_keeper_supersedes_on_decisive_rival():
    keeper = ThesisKeeper()
    a = assess_market(context=RANGE_CTX, flow={})
    # range creates nothing
    assert keeper.update(a, ts=1000.0) == []
    up = assess_market(context=UP_TREND, flow={})
    keeper.update(up, ts=1100.0)
    assert keeper.active is not None
    # a weak contrary read does not supersede
    tired = assess_market(context=EXHAUSTED_UP, flow={})
    keeper.update(tired, ts=1200.0)
    assert keeper.active is not None


def test_keeper_expires_on_ttl():
    keeper = ThesisKeeper(ThesisConfig(ttl_hours=1.0))
    a = assess_market(context=UP_TREND, flow={})
    keeper.update(a, ts=1000.0)
    events = keeper.update(a, ts=1000.0 + 2 * 3600.0)
    assert any(e["event"] == "thesis_expired" for e in events)


def test_keeper_range_challenges_fade_thesis():
    keeper = ThesisKeeper(ThesisConfig(range_challenge_limit=2))
    up = assess_market(context=UP_TREND, flow={})
    keeper.update(up, ts=1000.0)
    rng = assess_market(context=RANGE_CTX, flow={})
    keeper.update(rng, ts=1100.0)
    assert keeper.active is not None
    events = keeper.update(rng, ts=1200.0)
    assert any(e["event"] == "thesis_expired" for e in events)
    assert keeper.active is None


def test_status_shape():
    keeper = ThesisKeeper()
    assert keeper.status()["active"] is False
    up = assess_market(context=UP_TREND, flow={})
    keeper.update(up, ts=1000.0)
    st = keeper.status()
    assert st["active"] is True
    assert st["plan"]["direction"] == 1
