"""Phase 1, item 1: the MarketSituation is a governing input.

Pins:
- the contribution shape (pure helper): per-TF states read from the
  snapshot, ONE net direction vote for the whole situation (never one
  vote per TF), D1-opposed named, absent situation = None;
- the core reads it but it casts no vote: the intended TF's
  structure phase fills the structure read only when the bedrock
  label says nothing, the full contribution rides the market state
  so the record names it — and the support score, the opposition
  and the verdict are identical with an aligned, an opposed, or no
  situation (owner correction 2026-11-24: the alignment is evidence
  on the record, never a penalty);
- with no situation the verdict block is byte-identical to its
  pre-Phase-1 shape;
- the runner wires it: a funnel record names its situation
  contribution, the shadow verdict's market state carries the same
  block, and with the situation program off (or before the first
  situation) no block exists and the decision keeps its old shape.
"""
import json

import numpy as np
import pandas as pd
import pytest

from goldenswing.execution.judgment import evaluate_judgment
from goldenswing.execution.situation_input import (
    situation_contribution)
from goldenswing.live import situation

TF_FREQ = {"D1": "D", "H4": "4h", "H1": "h", "M30": "30min",
           "M15": "15min", "M5": "5min"}


# ------------------------------------------------------------------
# harness (frames + runner, after tests/test_situation.py)
# ------------------------------------------------------------------
def trend_frame(n=120, *, freq="h", start="2026-01-01", base=2000.0,
                drift=0.8, amp=6.0, period=12):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    closes = np.array([
        base + drift * i
        + amp * (1.0 - abs((i % period) / (period / 2.0) - 1.0))
        for i in range(n)], dtype=float)
    opens = closes - 0.3
    highs = closes + 0.7
    lows = closes - 0.7
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": 100.0}, index=idx)


def full_stack(*, drift=0.8):
    from goldenswing.live.hierarchy import TimeframeStack
    return TimeframeStack.from_frames(
        {tf: trend_frame(120, freq=TF_FREQ[tf], drift=drift)
         for tf in situation.SITUATION_TFS})


def _runner(tmp_path, **cfg_kw):
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False, **cfg_kw)
    return MT5Runner(cfg)


def _journal_records(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ------------------------------------------------------------------
# fixtures (candidate/decision/evidence, after
# tests/test_judgment_core.py)
# ------------------------------------------------------------------
def _candidate(**kw):
    base = {"direction": 1, "entry": 3900.0, "stop": 3890.0,
            "targets": [3920.0], "score": 72.0,
            "asymmetry_score": 72.0, "sleeve_base_score": 60.0,
            "score_components": None, "factors": [], "index": 5}
    base.update(kw)
    return base


def _evidence(*, supporting=(), contradictory=(), groups=None,
              observed=None, inferred=None, situation=None):
    ev = {
        "supporting": list(supporting),
        "contradictory": list(contradictory),
        "evidence_groups": groups or {},
        "independent_evidence_groups": sorted(groups or {}),
        "observed": observed or {
            "entry": 3900.0, "stop": 3890.0, "stop_dist": 10.0,
            "bias_d1": 1, "bias_h4": 1,
            "session_bucket": "overlap"},
        "inferred": inferred or {
            "score_total": 72.0, "score_flipped": False,
            "regime": "clean_trend", "posture": "TREND_PULLBACK",
            "journey_state": "with_journey", "room_r": 2.0},
    }
    if situation is not None:
        ev["situation"] = situation
    return ev


def _decision(**kw):
    base = {"result": "accepted", "reason_code": None,
            "order_result": "filled",
            "floor_result": {"floor": 55.0, "total": 72.0,
                             "path": "INTRADAY_CONTINUATION"}}
    base.update(kw)
    return base


_TREND_CTX = {"regime": "up/clean_trend/normal/accumulation/"
                        "overlap",
              "posture": "TREND_PULLBACK", "day_type": "trend",
              "adr_pct": 40.0, "adr_state": "open", "session": "overlap",
              "smt": None}

_NO_REGIME_CTX = dict(_TREND_CTX, regime=None)


def _snapshot(*, direction_word="up", phase="clean_trend",
              sit_id="sit-test"):
    return {"situation_id": sit_id,
            "per_tf": {tf: {
                "regime_label": f"{direction_word}/{phase}/normal/"
                                "accumulation/overlap",
                "structure_phase": phase,
                "direction_word": direction_word,
                "regime_confidence": 0.8,
                "volatility": {"atr_state": "normal"}}
                for tf in ("D1", "H4", "H1", "M15", "M5")}}


# ------------------------------------------------------------------
# pure helper
# ------------------------------------------------------------------
def test_contribution_shape_and_one_net_vote():
    contrib = situation_contribution(
        _snapshot(), intended_tf="H1", direction=1)
    assert contrib["situation_id"] == "sit-test"
    assert contrib["intended_tf"] == "H1"
    assert contrib["intended"]["direction_word"] == "up"
    assert contrib["intended"]["structure_phase"] == "clean_trend"
    assert set(contrib["per_tf"]) == {"D1", "H4", "H1", "M15", "M5"}
    # Five TFs agree — still ONE net vote, not five.
    assert contrib["alignment"]["per_tf"] == {
        tf: 1 for tf in ("D1", "H4", "H1", "M15", "M5")}
    assert contrib["alignment"]["net"] == 1
    assert contrib["alignment"]["d1_opposed"] is False


def test_contribution_d1_opposed():
    snap = _snapshot()
    snap["per_tf"]["D1"]["direction_word"] = "down"
    snap["per_tf"]["D1"]["regime_label"] = \
        "down/reversal/normal/distribution/overlap"
    contrib = situation_contribution(
        snap, intended_tf="H1", direction=1)
    assert contrib["alignment"]["per_tf"]["D1"] == -1
    assert contrib["alignment"]["d1_opposed"] is True
    # Four agree, one opposes: the net vote still reads +1.
    assert contrib["alignment"]["net"] == 1


def test_contribution_all_opposed():
    contrib = situation_contribution(
        _snapshot(direction_word="down", phase="reversal"),
        intended_tf="H1", direction=1)
    assert contrib["alignment"]["net"] == -1
    assert contrib["alignment"]["d1_opposed"] is True


def test_contribution_absent_situation_is_none():
    assert situation_contribution(
        None, intended_tf="H1", direction=1) is None
    assert situation_contribution(
        {}, intended_tf="H1", direction=1) is None
    assert situation_contribution(
        {"per_tf": {}}, intended_tf="H1", direction=1) is None
    assert situation_contribution(
        "not-a-snapshot", intended_tf="H1", direction=1) is None


def test_contribution_flat_abstains_and_no_candidate_reads_only():
    snap = _snapshot()
    for tf in snap["per_tf"]:
        snap["per_tf"][tf]["direction_word"] = "flat"
    contrib = situation_contribution(
        snap, intended_tf="H1", direction=1)
    assert contrib["alignment"]["per_tf"] == {}
    assert contrib["alignment"]["net"] == 0
    assert contrib["alignment"]["d1_opposed"] is False
    # No candidate: the states are still named, no votes cast.
    contrib = situation_contribution(
        _snapshot(), intended_tf="H1", direction=None)
    assert contrib["per_tf"]["D1"]["direction_word"] == "up"
    assert contrib["alignment"]["per_tf"] == {}
    assert contrib["alignment"]["net"] == 0


def test_contribution_missing_intended_tf_tolerated():
    contrib = situation_contribution(
        _snapshot(), intended_tf="M30", direction=1)
    assert contrib["intended"] == {}
    assert contrib["alignment"]["net"] == 1


# ------------------------------------------------------------------
# judgment core
# ------------------------------------------------------------------
def test_core_situation_casts_no_vote():
    base = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(), context=_TREND_CTX)
    up = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(situation=situation_contribution(
            _snapshot(), intended_tf="H1", direction=1)),
        context=_TREND_CTX)
    dn = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(situation=situation_contribution(
            _snapshot(direction_word="down", phase="reversal"),
            intended_tf="H1", direction=1)),
        context=_TREND_CTX)
    # No support weight, no opposition weight, no verdict change
    # — the situation is read into the picture and recorded,
    # never a penalty for fighting the daily picture.
    assert up["support_score"] == pytest.approx(
        base["support_score"])
    assert dn["support_score"] == pytest.approx(
        base["support_score"])
    assert up["opposition"] == pytest.approx(base["opposition"])
    assert dn["opposition"] == pytest.approx(base["opposition"])
    assert up["decision"] == base["decision"]
    assert dn["decision"] == base["decision"]
    assert up["market_state"]["situation"]["alignment"]["net"] == 1
    assert dn["market_state"]["situation"]["alignment"][
        "d1_opposed"] is True


def test_core_verdict_ignores_situation_alone():
    # Marginal candidate, no supporting groups: abstains bare —
    # and abstains with an aligned situation and with an opposed
    # one alike.  The situation alone never flips a verdict.
    base = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(), context=_TREND_CTX)
    assert base["decision"] == "abstain"
    up = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(situation=situation_contribution(
            _snapshot(), intended_tf="H1", direction=1)),
        context=_TREND_CTX)
    assert up["decision"] == "abstain"
    dn = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(situation=situation_contribution(
            _snapshot(direction_word="down", phase="reversal"),
            intended_tf="H1", direction=1)),
        context=_TREND_CTX)
    assert dn["decision"] == "abstain"


def test_core_structure_falls_back_to_situation_phase():
    # No bedrock regime AND no inferred regime: the structure
    # read is unknown bare.  (_structure_class falls back from
    # the bedrock label to inferred["regime"] before "unknown",
    # so the context regime alone is not enough to clear it.)
    no_regime_inferred = {
        "score_total": 72.0, "score_flipped": False,
        "regime": None, "posture": "TREND_PULLBACK",
        "journey_state": "with_journey", "room_r": 2.0}
    base = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(inferred=no_regime_inferred),
        context=_NO_REGIME_CTX)
    assert base["market_state"]["structure"] == "unknown"
    ranged = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(
            inferred=no_regime_inferred,
            situation=situation_contribution(
                _snapshot(phase="range"), intended_tf="H1",
                direction=1)),
        context=_NO_REGIME_CTX)
    assert ranged["market_state"]["structure"] == "range"
    # A bedrock read is never overridden by the situation.
    trended = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(situation=situation_contribution(
            _snapshot(phase="range"), intended_tf="H1",
            direction=1)),
        context=_TREND_CTX)
    assert trended["market_state"]["structure"] == "trend"


def test_core_empty_situation_is_byte_identical_to_absent():
    base = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(), context=_TREND_CTX)
    empty = evaluate_judgment(
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence(situation={}), context=_TREND_CTX)
    assert empty == base
    assert "situation" not in empty["market_state"]


# ------------------------------------------------------------------
# runner wiring
# ------------------------------------------------------------------
def test_runner_funnel_record_names_contribution(tmp_path):
    runner = _runner(tmp_path)
    stack = full_stack()
    runner._stack = stack
    runner._situation_on_closes(["H1"], {"time": stack.cursors["H1"]})
    rec = runner._emit_eval(
        playbook="probe", evaluation_clock="H1",
        intended_tf="H1", outcome="candidate_emitted",
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence())
    assert rec is not None
    contrib = rec["situation_contribution"]
    assert contrib["situation_id"] == rec["situation_id"]
    assert contrib["intended_tf"] == "H1"
    assert set(contrib["per_tf"]) == set(situation.SITUATION_TFS)
    # The crafted frames rise on every TF.
    assert contrib["alignment"]["net"] == 1
    assert contrib["alignment"]["d1_opposed"] is False
    # The shadow verdict weighed the same block.
    shadows = [r for r in _journal_records(tmp_path)
               if r["type"] == "judgment_shadow"]
    assert len(shadows) == 1
    assert shadows[0]["judgment"]["market_state"]["situation"] \
        == contrib
    assert shadows[0]["situation_id"] == rec["situation_id"]


def test_runner_no_contribution_before_first_situation(tmp_path):
    runner = _runner(tmp_path)
    rec = runner._emit_eval(
        playbook="probe", evaluation_clock="H1",
        intended_tf="H1", outcome="candidate_emitted",
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence())
    assert "situation_contribution" not in rec
    assert "situation_id" not in rec
    shadows = [r for r in _journal_records(tmp_path)
               if r["type"] == "judgment_shadow"]
    assert len(shadows) == 1
    assert "situation" not in shadows[0]["judgment"]["market_state"]


def test_runner_no_contribution_when_disabled(tmp_path):
    runner = _runner(tmp_path, situation_enabled=False)
    stack = full_stack()
    runner._stack = stack
    runner._situation_on_closes(["H1"], {"time": stack.cursors["H1"]})
    rec = runner._emit_eval(
        playbook="probe", evaluation_clock="H1",
        intended_tf="H1", outcome="candidate_emitted",
        candidate=_candidate(), decision=_decision(),
        evidence=_evidence())
    assert "situation_contribution" not in rec
    assert "situation_id" not in rec
    shadows = [r for r in _journal_records(tmp_path)
               if r["type"] == "judgment_shadow"]
    assert len(shadows) == 1
    assert "situation" not in shadows[0]["judgment"]["market_state"]
