"""Phase 10 methodology: context-performance maps and hunting grounds.

A backtest's goal is a MAP of where the strategy performed well, not a
binary verdict.  These tests cover the measurement, the conservative
derivation rules, and the engine enforcement (session scope, hunting
ground, map bonuses).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from goldenswing.backtest.harness import Outcome
from goldenswing.confluence import ConfluenceEngine, map_bonuses
from goldenswing.contextmap import (HuntingGround, context_levels,
                                    derive_hunting_ground, measure_map)
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.risk import RiskGuards
from goldenswing.golden.detector import GoldenSignal
from goldenswing.playbooks.base import Candidate, EngineContext, Playbook


def _sig(index: int, direction: int = 1) -> GoldenSignal:
    return GoldenSignal(index=index, timestamp=None, direction=direction,
                        score=70.0, kind="sweep_reversal",
                        entry_ref=100.0, invalidation=99.0,
                        targets=[102.0], atr=1.0, factors=["f"])


def _oc(index: int, r: float, direction: int = 1) -> Outcome:
    return Outcome(signal=_sig(index, direction), entry_index=index + 1,
                   exit_index=index + 5, exit_reason="target",
                   r_net=r, hold_bars=4)


def _state(session="london", trend="up", daily_bias=1):
    return SimpleNamespace(session=session, killzone=False,
                           vol_regime="normal", trend=trend,
                           daily_bias=daily_bias, features=None,
                           timestamp=pd.Timestamp("2026-01-05 10:00"),
                           atr=1.0, close=100.0, active_fvgs=[])


# ------------------------------------------------------------ measure_map
def test_measure_map_splits_dimensions():
    """Outcomes are bucketed per dimension per level with correct stats."""
    states = [_state(session="asia", trend="up") for _ in range(10)]
    outs = [_oc(i, 0.5) for i in range(5)] + [_oc(i + 5, -0.5)
                                             for i in range(5)]
    m = measure_map(outs, states)
    asia = m["session"]["asia"]
    assert asia["n"] == 10
    assert asia["wins"] == 5
    assert abs(asia["avg_r"] - 0.0) < 1e-9
    assert abs(asia["total_r"] - 0.0) < 1e-9
    # halves split by signal order
    assert asia["half1_n"] == 5 and asia["half2_n"] == 5
    assert abs(asia["half1_avg_r"] - 0.5) < 1e-9
    assert abs(asia["half2_avg_r"] + 0.5) < 1e-9


def test_measure_map_skips_missing_state():
    states = [_state() for _ in range(3)]
    outs = [_oc(0, 0.5), _oc(99, 0.5)]  # bar 99 has no state
    m = measure_map(outs, states)
    assert m["session"]["london"]["n"] == 1


# --------------------------------------------------- derive_hunting_ground
def _cell(n, avg_r, h1=None, h2=None):
    return {"n": n, "avg_r": avg_r, "total_r": avg_r * n,
            "half1_n": n // 2, "half1_avg_r": h1 if h1 is not None else avg_r,
            "half2_n": n - n // 2,
            "half2_avg_r": h2 if h2 is not None else avg_r}


def test_derive_excludes_decisively_bad_cell():
    m = {"session": {
        "asia": _cell(30, -0.20, -0.15, -0.25),
        "london": _cell(30, 0.30),
    }}
    hg = derive_hunting_ground(m, overall_avg_r=0.10)
    assert ("session", "asia") in hg.excluded
    assert ("session", "london") not in hg.excluded


def test_derive_ignores_thin_cells():
    m = {"session": {"asia": _cell(10, -0.50, -0.5, -0.5)}}
    hg = derive_hunting_ground(m, overall_avg_r=0.10)
    assert not hg.excluded  # n < 25: measured, never gated


def test_derive_ignores_one_regime_artifact():
    m = {"session": {"asia": _cell(30, -0.20, 0.30, -0.70)}}
    hg = derive_hunting_ground(m, overall_avg_r=0.10)
    assert not hg.excluded  # h1 positive: not decisively bad


def test_derive_bonuses_proven_cells():
    m = {"session": {
        "asia": _cell(30, 0.60),    # edge +0.50 -> tier 2
        "london": _cell(30, 0.30),  # edge +0.20 -> tier 1
        "newyork": _cell(30, 0.15),  # edge +0.05 -> nothing
    }}
    hg = derive_hunting_ground(m, overall_avg_r=0.10)
    assert hg.bonuses[("session", "asia")] == 4.0
    assert hg.bonuses[("session", "london")] == 2.0
    assert ("session", "newyork") not in hg.bonuses


def test_hunting_ground_roundtrip():
    hg = HuntingGround(excluded={("trend", "down")},
                       bonuses={("session", "newyork"): 2.0})
    hg2 = HuntingGround.from_dict(hg.to_dict())
    assert hg2.excluded == hg.excluded
    assert hg2.bonuses == hg.bonuses


def test_context_levels_empty_without_state():
    assert context_levels([], 5, 1) == {}


# ------------------------------------------------------- engine enforcement
class _ScopedPB(Playbook):
    name = "london_only"
    live = True
    live_threshold = 0.0
    session_scope = "london"

    def __init__(self, cands):
        self._cands = cands

    def candidates(self, ctx):
        return self._cands


class _HGPB(Playbook):
    name = "hg"
    live = True
    live_threshold = 0.0
    hunting_ground = HuntingGround(
        excluded={("trend", "down")},
        bonuses={("session", "london"): 2.0})

    def __init__(self, cands):
        self._cands = cands

    def candidates(self, ctx):
        return self._cands


def _ctx_for(session="london", trend="up"):
    df = pd.DataFrame({"open": [100.0] * 10, "high": [101.0] * 10,
                       "low": [99.0] * 10, "close": [100.0] * 10})
    states = [_state(session=session, trend=trend) for _ in range(10)]
    return EngineContext(df=df, states=states, years=1.0)


def _run(pb, ctx):
    tmp = Path(tempfile.mkdtemp())
    eng = ConfluenceEngine([pb], RiskGuards(tmp / "r.json"),
                           TradeJournal(tmp / "j.jsonl"))
    return eng.run(ctx)


def _cand(index=5, direction=1, base=70.0):
    return Candidate(playbook="x", outcome=_oc(index, 0.5, direction),
                     base_score=base, thesis="t")


def test_session_scope_blocks_wrong_session():
    rep = _run(_ScopedPB([_cand()]), _ctx_for(session="asia"))
    assert rep.skips.get("context-violation") == 1
    assert rep.live_metrics["trades"] == 0


def test_session_scope_allows_right_session():
    rep = _run(_ScopedPB([_cand()]), _ctx_for(session="london"))
    assert "context-violation" not in rep.skips
    assert rep.live_metrics["trades"] == 1


def test_session_scope_any_fires_everywhere():
    class _Any(_ScopedPB):
        name = "any_pb"
        session_scope = "any"
    rep = _run(_Any([_cand()]), _ctx_for(session="off"))
    assert rep.live_metrics["trades"] == 1


def test_hunting_ground_exclusion_blocks_trade():
    rep = _run(_HGPB([_cand()]), _ctx_for(trend="down"))
    assert rep.skips.get("context-violation") == 1
    assert rep.live_metrics["trades"] == 0


def test_map_bonus_applies_in_proven_cell():
    pb = _HGPB([_cand(base=70.0)])
    ctx = _ctx_for(session="london", trend="up")
    tmp = Path(tempfile.mkdtemp())
    eng = ConfluenceEngine([pb], RiskGuards(tmp / "r.json"),
                           TradeJournal(tmp / "j.jsonl"))
    sc = eng.score(_cand(base=70.0), ctx, pb)
    assert sc.confluence == 72.0
    assert any("proven session=london" in b for b in sc.bonuses)


def test_map_bonus_absent_outside_proven_cell():
    pb = _HGPB([_cand(base=70.0)])
    ctx = _ctx_for(session="asia", trend="up")
    tmp = Path(tempfile.mkdtemp())
    eng = ConfluenceEngine([pb], RiskGuards(tmp / "r.json"),
                           TradeJournal(tmp / "j.jsonl"))
    sc = eng.score(_cand(base=70.0), ctx, pb)
    assert sc.confluence == 70.0


def test_check_context_tripwire_still_enforced():
    class _Strict(_ScopedPB):
        name = "strict"

        def check_live_signal(self, signal, ctx, context=None):
            return False, "always fails in test"

    rep = _run(_Strict([_cand()]), _ctx_for())
    assert rep.skips.get("context-violation") == 1
    assert rep.live_metrics["trades"] == 0
