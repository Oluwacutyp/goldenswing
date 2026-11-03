"""Forecast E3 — the forecast→judgment evidence feed.

Pins: stance from the ensemble's ranks (supporting /
contradictory / neutral / absent); the trust gate on E2's live
calibration (extreme ranks must resolve at >= 1.25x base over
>= 20 pooled resolutions before the vote weighs; anti-calibrated
feeds abstain and say so); freshness; the merge as ONE weighed
correlation group through the REAL judgment core; the funnel
evidence never mutated.
"""
from __future__ import annotations

from goldenswing.execution.forecast_evidence import (
    GROUP,
    forecast_block,
    horizon_for_tf,
    merge_into_evidence,
    trust_for,
)
from goldenswing.execution.judgment import evaluate_judgment


class _Ranks:
    def __init__(self, table, bar_time="2026-11-03 10:00:00"):
        self._table = table
        self.bar_time = bar_time

    def rank(self, horizon, direction):
        return float(self._table.get(horizon, {})
                     .get(direction, float("nan")))


def _book(*, base=0.30, bands=()):
    return {"horizons": {"24": {
        "up": {"base_rate": base, "n": 100, "bands": list(bands)},
        "dn": {"base_rate": base, "n": 100, "bands": []},
        "magnitude": {"base_rate": base, "n": 100, "bands": []}}}}


TRUSTED = _book(bands=({"band": "80-90", "n": 15, "rate": 0.40},
                       {"band": "90+", "n": 10, "rate": 0.50}))
ANTI = _book(bands=({"band": "80-90", "n": 15, "rate": 0.20},
                    {"band": "90+", "n": 10, "rate": 0.18}))


# ── stance ──────────────────────────────────────────────────────


def test_stance_supporting_contradictory_neutral_absent():
    hot = _Ranks({24: {"up": 92.0, "dn": 40.0}})
    b = forecast_block(ranks=hot, direction=1, horizon=24,
                       fresh=True)
    assert b["stance"] == "supporting"
    b = forecast_block(ranks=hot, direction=-1, horizon=24,
                       fresh=True)
    assert b["stance"] == "contradictory"
    flat = _Ranks({24: {"up": 60.0, "dn": 55.0}})
    b = forecast_block(ranks=flat, direction=1, horizon=24,
                       fresh=True)
    assert b["stance"] == "neutral"
    b = forecast_block(ranks=None, direction=1, horizon=24,
                       fresh=True)
    assert b["stance"] == "absent" and b["bar_time"] is None


def test_magnitude_alone_never_votes_a_direction():
    # both ranks extreme but balanced: weather, not evidence
    storm = _Ranks({24: {"up": 88.0, "dn": 85.0}})
    b = forecast_block(ranks=storm, direction=1, horizon=24,
                       fresh=True, book_view=TRUSTED)
    assert b["move_pressure"] == 88.0
    assert b["stance"] == "neutral" and b["weighed"] is False


# ── the trust gate ──────────────────────────────────────────────


def test_trust_earned_anti_and_unproven():
    assert trust_for(TRUSTED, 24, "up") == "trusted"
    assert trust_for(ANTI, 24, "up") == "anti_calibrated"
    thin = _book(bands=({"band": "90+", "n": 5, "rate": 0.9},))
    assert trust_for(thin, 24, "up") == "unproven"
    assert trust_for(None, 24, "up") == "unproven"


def test_weighed_only_when_supporting_fresh_and_trusted():
    hot = _Ranks({24: {"up": 92.0, "dn": 40.0}})
    b = forecast_block(ranks=hot, direction=1, horizon=24,
                       fresh=True, book_view=TRUSTED)
    assert b["weighed"] is True and b["text"]
    b = forecast_block(ranks=hot, direction=1, horizon=24,
                       fresh=False, book_view=TRUSTED)
    assert b["weighed"] is False          # stale: recorded only
    b = forecast_block(ranks=hot, direction=1, horizon=24,
                       fresh=True, book_view=None)
    assert b["weighed"] is False          # unproven: no weight
    assert b["stance"] == "supporting"    # ...but still journaled
    b = forecast_block(ranks=hot, direction=1, horizon=24,
                       fresh=True, book_view=ANTI)
    assert b["weighed"] is False
    assert b["trust"] == "anti_calibrated"


# ── the merge, through the real core ────────────────────────────


def _candidate():
    return {"direction": 1, "entry": 2000.0, "stop": 1990.0,
            "targets": [2030.0]}


def test_merge_adds_one_group_and_never_mutates_the_source():
    hot = _Ranks({24: {"up": 92.0, "dn": 40.0}})
    block = forecast_block(ranks=hot, direction=1, horizon=24,
                           fresh=True, book_view=TRUSTED)
    evidence = {"evidence_groups": {}, "supporting": [],
                "contradictory": [], "observed": {}, "inferred": {}}
    merged = merge_into_evidence(evidence, block)
    assert merged["evidence_groups"][GROUP] == [block["text"]]
    assert block["text"] in merged["supporting"]
    assert evidence["evidence_groups"] == {}   # source untouched

    plain = evaluate_judgment(candidate=_candidate(),
                              decision={}, evidence=evidence,
                              context={})
    fed = evaluate_judgment(candidate=_candidate(), decision={},
                            evidence=merged, context={})
    assert fed["evidence_balance"]["groups_for"] == [GROUP]
    assert fed["support_score"] > plain["support_score"]
    assert fed["expected_r"] > plain["expected_r"]


def test_unweighed_block_leaves_evidence_unchanged():
    hot = _Ranks({24: {"up": 92.0, "dn": 40.0}})
    block = forecast_block(ranks=hot, direction=1, horizon=24,
                           fresh=True, book_view=None)
    evidence = {"evidence_groups": {}, "supporting": []}
    merged = merge_into_evidence(evidence, block)
    assert GROUP not in (merged.get("evidence_groups") or {})


def test_horizon_follows_the_candidate_clock():
    assert horizon_for_tf("M5") == 12
    assert horizon_for_tf("H1") == 24
    assert horizon_for_tf("H4") == 72
    assert horizon_for_tf(None) == 24
