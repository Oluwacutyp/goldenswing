"""Step 16 (owner rulings 2026-10-30 §3): direction.flip is
weighted evidence by default.

The blanket veto is gone: a structure flip (a close through the
last defended pivot on the score frame) zeroes the candidate's
structure share, stands as the funded opposing scenario, and is
journaled with its evidence + the pre-Step-16 counterfactual.
The ONLY surviving flip refusal is a NAMED, declared premise
invalidation (registry ``flip_premise``) — no playbook declares
one today, so the refusal tests declare one synthetically.
"""
from types import SimpleNamespace

import pytest

from goldenswing.execution import mt5_runner as mr
from goldenswing.live.asymmetry import ScoreContext, score_asymmetry
from goldenswing.live.multitf import find_swings

from test_confluence_rework import _frame, _wave
from test_funnel_m5 import (_evals, _fresh_tick, _m5_fixture,
                            _qm_runner)
from test_p3_trader import (_FakeScore, _StubPB, _drive_one,
                            _flat_h1, _runner, _sig,
                            _signal_records)


# ------------------------------------------------------- asymmetry

def _flip_frame():
    """The confluence suite's flip construction: a defended wave,
    then closes plunging through its last pivot."""
    closes = _wave(6, drop=1.5, rise=1.8, low_step=0.3)
    frame = _frame(closes, spread=0.2)
    last_low = float(frame["low"].min())
    plunge = closes + [last_low - 2.0, last_low - 2.6,
                       last_low - 2.4, last_low - 2.5]
    return _frame(plunge, spread=0.2)


def test_flip_evidence_rides_the_flipped_score():
    frame2 = _flip_frame()
    price = float(frame2["close"].iloc[-1])
    flipped = score_asymmetry(ScoreContext(
        direction=1, frame=frame2, atr=0.5, price=price))
    assert flipped.flipped is True
    assert flipped.direction == -1
    ev = flipped.flip_evidence
    assert ev is not None
    assert ev["direction"] == 1
    # The pivot is the frame's last defended (swing-low) pivot.
    highs, lows = find_swings(
        frame2["high"].to_numpy(dtype=float),
        frame2["low"].to_numpy(dtype=float), 3)
    assert ev["pivot"] == pytest.approx(float(lows[-1][1]))
    assert ev["breach_atr"] >= 0.1
    assert ev["opposite_total"] == pytest.approx(flipped.total)
    bd = flipped.breakdown()
    assert bd["flip_evidence"]["pivot"] == pytest.approx(
        ev["pivot"], abs=1e-3)


def test_allow_flip_false_prices_own_direction_with_evidence():
    frame2 = _flip_frame()
    price = float(frame2["close"].iloc[-1])
    own = score_asymmetry(ScoreContext(
        direction=1, frame=frame2, atr=0.5, price=price),
        allow_flip=False)
    assert own.flipped is False
    assert own.direction == 1
    assert own.component("structure") == 0.0
    struct = own.components["structure"]
    assert "structure_flip" in struct.flags
    assert "structure_flip_against" in struct.flags
    assert own.flip_evidence is not None
    assert own.flip_evidence["opposite_total"] is None


def test_no_flip_no_evidence():
    closes = _wave(6, drop=1.5, rise=1.8, low_step=0.3)
    frame = _frame(closes, spread=0.2)
    price = float(frame["close"].iloc[-1])
    for allow in (True, False):
        s = score_asymmetry(ScoreContext(
            direction=1, frame=frame, atr=0.5, price=price),
            allow_flip=allow)
        assert s.flipped is False
        assert s.flip_evidence is None
        assert "flip_evidence" not in s.breakdown()


# ------------------------------------------------------- score doubles

class _RepricedFlipScore(_FakeScore):
    """The production Step-16 shape: own-direction score with the
    structure share zeroed and the flip's evidence attached."""
    flipped = False
    flip_evidence = {"direction": 1, "pivot": 4016.53,
                     "breach_atr": 0.42, "opposite_total": 64.0,
                     "score_frame": "H1"}


# ------------------------------------------------------- roster route

def _roster_runner(tmp_path):
    runner = _runner(tmp_path, execute_playbooks=("fvg_midline",))
    h1 = _flat_h1()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._score_for = lambda **kw: _RepricedFlipScore(
        70.0, 0.0, 15.0)
    return runner, h1


def _zonepath_verdict():
    return {"bias_d1": 1, "bias_h4": 1, "regime": "trend",
            "direction": 1, "trigger": "tick-cross",
            "entry_clock": "M15", "amd_phase": "accumulation",
            "qm": {"status": "none"}, "volume": {},
            "style": "ZONE_PATH"}


def test_roster_flip_is_evidence_not_refusal(tmp_path):
    runner, h1 = _roster_runner(tmp_path)
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1,
               verdict=_zonepath_verdict())
    recs = _signal_records(runner)
    assert recs and recs[-1]["decision"] == "accepted"
    assert len(runner.executor.orders) == 1
    evals = [r for r in _evals(runner, "fvg_midline")
             if r["outcome"] == "candidate_emitted"]
    assert evals
    dec = evals[-1]["decision"]
    assert dec["reason_code"] != "direction.flip"
    block = dec["direction_flip"]
    assert block["stance"] == "evidence"
    assert block["pre_step16_rule"] == \
        "would have refused direction.flip"
    assert block["pivot"] == 4016.53
    assert block["opposite_total"] == 64.0
    assert block["own_total"] == 70.0
    scen = dec["competing_scenarios"]
    assert "funded by the weighed evidence" in scen[1]["status"]


def test_roster_flip_refuses_only_on_declared_premise(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        mr, "_flip_premise_for",
        lambda name: "the defended structure IS the setup")
    runner, h1 = _roster_runner(tmp_path)
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1,
               verdict=_zonepath_verdict())
    assert runner.executor.orders == []
    recs = _signal_records(runner)
    assert recs and recs[-1]["decision"] == "gated"
    assert "premise_invalidation[fvg_midline]" in recs[-1]["reason"]
    assert "the defended structure IS the setup" in \
        recs[-1]["reason"]
    assert "4016.53" in recs[-1]["reason"]
    evals = [r for r in _evals(runner, "fvg_midline")
             if r["outcome"] == "candidate_emitted"]
    dec = evals[-1]["decision"]
    assert dec["reason_code"] == "direction.flip"
    assert dec["direction_flip"]["stance"] == \
        "premise_invalidation"
    assert dec["direction_flip"]["premise"] == \
        "the defended structure IS the setup"


# ------------------------------------------------------- qm_m5 route

def _wire_qm(runner, m5):
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    return runner


def test_qm_flip_refuses_only_on_declared_premise(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        mr, "_flip_premise_for",
        lambda name: "qm premise declared for the test")
    runner = _wire_qm(_qm_runner(tmp_path), _m5_fixture())
    runner._score_for = lambda **kw: _RepricedFlipScore(
        41.0, 0.0, 15.0)
    runner._run_qm_m5(_m5_fixture(), {})
    evals = [r for r in _evals(runner, "qm_m5")
             if r["outcome"] == "candidate_emitted"]
    dec = evals[-1]["decision"]
    assert dec["result"] == "refused"
    assert dec["reason_code"] == "direction.flip"
    assert "premise_invalidation[qm_m5]" in dec["reason"]
    assert dec["direction_flip"]["stance"] == \
        "premise_invalidation"
    assert dec["direction_flip"]["premise"] == \
        "qm premise declared for the test"
    assert runner.executor.orders == []
