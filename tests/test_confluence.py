"""Phase 9 confluence engine: unit tests.

The full anchor canary (+28.92% / 199 / PF 1.25) lives in
scripts/run_confluence_book.py as asserted integration — it needs the
whole 2y read+featurize and belongs in the runner, not the unit suite.
These tests cover the engine's logic: greedy equivalence, tie-breaks,
risk gating, journaling, and the playbook contract.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd

from goldenswing.backtest.harness import Outcome, build_book
from goldenswing.confluence import (ConfluenceConfig, ConfluenceEngine,
                                    context_bonuses)
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.risk import RiskGuards
from goldenswing.golden.detector import GoldenSignal
from goldenswing.playbooks.base import Candidate, EngineContext, Playbook


def _sig(index: int, direction: int = 1, score: float = 70.0) -> GoldenSignal:
    return GoldenSignal(index=index, timestamp=None, direction=direction,
                        score=score, kind="sweep_reversal",
                        entry_ref=100.0, invalidation=99.0,
                        targets=[102.0], atr=1.0, factors=["test factor"])


def _oc(index: int, r: float = 0.5, direction: int = 1,
        score: float = 70.0) -> Outcome:
    return Outcome(signal=_sig(index, direction, score),
                   entry_index=index + 1, exit_index=index + 5,
                   exit_reason="target", r_net=r, hold_bars=4)


def _cand(index: int, playbook: str = "sweep_reversal", r: float = 0.5,
          base: float = 70.0) -> Candidate:
    return Candidate(playbook=playbook, outcome=_oc(index, r),
                     base_score=base, thesis=f"{playbook} thesis")


class _LivePB(Playbook):
    name = "sweep_reversal"
    live = True
    live_threshold = 60.0

    def __init__(self, cands):
        self._cands = cands

    def candidates(self, ctx):
        return self._cands


def _engine(cands, risk_mode="shadow", halted=False):
    tmp = Path(tempfile.mkdtemp())
    risk = RiskGuards(tmp / "risk.json")
    if halted:
        # PREMISE CORRECTION (owner ruling 2026-10-23): three
        # -100 losses used to trip the 3% DAILY cap (the old
        # comment said consecutive-loss, but 3 < the limit of 6 —
        # it was the daily room that latched).  The daily cap is
        # revoked; trip the surviving account-level halt — max
        # drawdown — honestly instead.
        risk.note_balance(10_000.0)
        risk.record_outcome(-1_600.0, 8_400.0)  # 16% drawdown halt
        assert risk.halted
    journal = TradeJournal(tmp / "j.jsonl")
    eng = ConfluenceEngine([_LivePB(cands)], risk, journal,
                           risk_mode=risk_mode)
    df = pd.DataFrame({"open": [100.0] * 50, "high": [101.0] * 50,
                       "low": [99.0] * 50, "close": [100.0] * 50})
    ctx = EngineContext(df=df, states=[], years=1.0)
    return eng, ctx, journal


# ------------------------------------------------------------------ greedy
def test_greedy_matches_build_book():
    """The engine's one-position-at-a-time walk is the harness's
    build_book (the anchor canary depends on this identity)."""
    outs = [_oc(0, 0.5), _oc(2, -0.3), _oc(10, 1.0), _oc(12, 0.2)]
    assert ConfluenceEngine._greedy(outs, 60.0) == build_book(outs, 60.0)
    # overlapping signal is skipped by both
    assert [o.signal.index for o in ConfluenceEngine._greedy(outs, 60.0)] \
        == [0, 10]


def test_greedy_threshold():
    outs = [_oc(0, 0.5, score=55.0), _oc(10, 1.0, score=70.0)]
    assert [o.signal.index for o in ConfluenceEngine._greedy(outs, 60.0)] \
        == [10]


def test_same_bar_conflict_prefers_higher_confluence():
    """Two playbooks, one bar: the higher-confluence setup wins."""
    lo = _cand(10, playbook="a", base=60.0)
    hi = _cand(10, playbook="b", base=80.0)

    class PB(Playbook):
        live = True
        live_threshold = 0.0

        def __init__(self, name, cands):
            self.name = name
            self._cands = cands

        def candidates(self, ctx):
            return self._cands

    tmp = Path(tempfile.mkdtemp())
    eng = ConfluenceEngine([PB("a", [lo]), PB("b", [hi])],
                           RiskGuards(tmp / "r.json"),
                           TradeJournal(tmp / "j.jsonl"))
    df = pd.DataFrame({"open": [100.0] * 50, "high": [101.0] * 50,
                       "low": [99.0] * 50, "close": [100.0] * 50})
    ctx = EngineContext(df=df, states=[], years=1.0)
    rep = eng.run(ctx)
    assert len(rep.live_outcomes) == 1
    # both outcomes are identical payloads; the journal tells who won
    recs = eng.journal.records()
    opens = [r for r in recs if r["type"] == "open"]
    assert len(opens) == 1 and opens[0]["playbook"] == "b"


def test_no_double_position_across_playbooks():
    """One position at a time — a second playbook's signal while in a
    trade is skipped, never stacked."""
    c1 = _cand(10, playbook="a", base=70.0)   # exits bar 15
    c2 = _cand(12, playbook="b", base=90.0)   # fires while in trade

    class PB(Playbook):
        live = True
        live_threshold = 0.0

        def __init__(self, name, cands):
            self.name = name
            self._cands = cands

        def candidates(self, ctx):
            return self._cands

    tmp = Path(tempfile.mkdtemp())
    eng = ConfluenceEngine([PB("a", [c1]), PB("b", [c2])],
                           RiskGuards(tmp / "r.json"),
                           TradeJournal(tmp / "j.jsonl"))
    df = pd.DataFrame({"open": [100.0] * 50, "high": [101.0] * 50,
                       "low": [99.0] * 50, "close": [100.0] * 50})
    ctx = EngineContext(df=df, states=[], years=1.0)
    rep = eng.run(ctx)
    assert len(rep.live_outcomes) == 1
    assert rep.skips.get("position-lock", 0) == 1


# -------------------------------------------------------------------- risk
def test_risk_shadow_mode_never_alters_sequence():
    eng, ctx, _ = _engine([_cand(10), _cand(20)], risk_mode="shadow")
    rep = eng.run(ctx)
    assert len(rep.live_outcomes) == 2
    assert len(rep.risk_log) == 2
    assert all(r["mode"] == "shadow" for r in rep.risk_log)


def test_risk_enforce_blocks_when_halted():
    eng, ctx, journal = _engine([_cand(10)], risk_mode="enforce",
                                halted=True)
    rep = eng.run(ctx)
    assert len(rep.live_outcomes) == 0
    assert rep.skips.get("risk", 0) == 1
    recs = journal.records()
    assert any(r["type"] == "skip" and "risk" in r["playbook"]
               for r in recs)


# ------------------------------------------------------------------ journal
def test_journal_logs_full_thesis():
    eng, ctx, journal = _engine([_cand(10, base=70.0)])
    eng.run(ctx)
    recs = journal.records()
    opens = [r for r in recs if r["type"] == "open"]
    closes = [r for r in recs if r["type"] == "close"]
    assert len(opens) == 1 and len(closes) == 1
    assert "sweep_reversal thesis" in opens[0]["factors"]
    assert opens[0]["playbook"] == "sweep_reversal"
    assert closes[0]["r_net"] == 0.5


def test_shadow_candidates_logged_not_booked():
    class ShadowPB(Playbook):
        name = "fvg_midline"
        live = False

        def candidates(self, ctx):
            return [_cand(10, playbook="fvg_midline")]

    tmp = Path(tempfile.mkdtemp())
    eng = ConfluenceEngine([ShadowPB()], RiskGuards(tmp / "r.json"),
                           TradeJournal(tmp / "j.jsonl"))
    df = pd.DataFrame({"open": [100.0] * 50, "high": [101.0] * 50,
                       "low": [99.0] * 50, "close": [100.0] * 50})
    ctx = EngineContext(df=df, states=[], years=1.0)
    rep = eng.run(ctx)
    assert len(rep.live_outcomes) == 0
    assert rep.shadow["fvg_midline"]["n_candidates"] == 1
    notes = [r for r in eng.journal.records() if r["type"] == "note"]
    assert any("paper only" in r["text"] for r in notes)


# ------------------------------------------------------------------ bonuses
def test_context_bonuses_silent_without_features():
    """No FeatureSet -> no bonuses, no crash (sleeves on plain states)."""
    from goldenswing.reader.state import MarketState
    s = MarketState(index=5, timestamp=None, close=100.0, atr=1.0,
                    trend="range", bias=0)
    ctx = EngineContext(df=pd.DataFrame(), states=[s] * 6, years=1.0)
    total, notes = context_bonuses(_cand(5), ctx, ConfluenceConfig())
    assert total == 0.0 and notes == []


def test_playbook_contract():
    """Every shipped playbook honours the interface."""
    from goldenswing.playbooks import (FVGMidlinePlaybook,
                                       SweepReversalPlaybook,
                                       TrendlineRetestPlaybook)
    pbs = [SweepReversalPlaybook(), FVGMidlinePlaybook(),
           TrendlineRetestPlaybook()]
    # All three have a real incremental live_signal (the sleeves'
    # parity is proven in tests/test_paper_loop.py).  The anchor stays
    # shadow-only inside the paper loop via execute_playbooks, not via
    # this flag.
    assert all(pb.live for pb in pbs)
    assert all(isinstance(pb.live_signal([], 0), list) for pb in pbs)
    assert pbs[0].live_threshold == 60.0
    assert all(pb.name for pb in pbs)
    assert all(isinstance(pb.description, str) and pb.description
               for pb in pbs)


# ------------------------------------------------- GOD-CE trap bonus
def _trap_ctx(n: int = 20, boost_long_at: tuple = (),
              boost_short_at: tuple = ()):
    """EngineContext with a hand-made trap frame (no compute cost)."""
    df = pd.DataFrame({"open": [100.0] * n, "high": [101.0] * n,
                       "low": [99.0] * n, "close": [100.0] * n,
                       "volume": [1000.0] * n})
    tf = pd.DataFrame({"trap_boost_long": [i in boost_long_at
                                           for i in range(n)],
                       "trap_boost_short": [i in boost_short_at
                                            for i in range(n)]})
    return EngineContext(df=df, states=[], years=1.0, trap_features=tf)


def test_trap_bonus_fires_for_trendline_retest():
    """Direction-aligned trap boost at the signal bar earns the bonus —
    trendline_retest only (the playbook it was measured on)."""
    from goldenswing.confluence import trap_bonuses
    ctx = _trap_ctx(boost_long_at=(5,), boost_short_at=(7,))
    cfg = ConfluenceConfig()
    total, notes = trap_bonuses(5, 1, "trendline_retest", ctx, cfg)
    assert total == cfg.bonus_trap_boost == 4.0
    assert any("trap-boost" in n for n in notes)
    total, _ = trap_bonuses(7, -1, "trendline_retest", ctx, cfg)
    assert total == 4.0


def test_trap_bonus_direction_and_playbook_gated():
    """Wrong direction gets nothing; unmeasured playbooks get nothing."""
    from goldenswing.confluence import trap_bonuses
    ctx = _trap_ctx(boost_long_at=(5,))
    cfg = ConfluenceConfig()
    assert trap_bonuses(5, -1, "trendline_retest", ctx, cfg)[0] == 0.0
    assert trap_bonuses(5, 1, "sweep_reversal", ctx, cfg)[0] == 0.0
    assert trap_bonuses(5, 1, "fvg_midline", ctx, cfg)[0] == 0.0
    assert trap_bonuses(4, 1, "trendline_retest", ctx, cfg)[0] == 0.0


def test_trap_bonus_silent_without_frame():
    """No trap frame and the bonus path degrades silently.

    (ensure_trap_features would compute it from df — too heavy for a
    unit test — so this covers the explicit-None path via a stub.)"""
    from goldenswing.confluence import trap_bonuses
    import goldenswing.confluence as conf
    ctx = _trap_ctx()
    ctx.trap_features = None
    orig = conf.ensure_trap_features
    conf.ensure_trap_features = lambda c: (_ for _ in ()).throw(
        RuntimeError("no frame"))
    try:
        total, notes = trap_bonuses(5, 1, "trendline_retest",
                                    ctx, ConfluenceConfig())
    finally:
        conf.ensure_trap_features = orig
    assert total == 0.0 and notes == []


def test_trap_bonus_in_engine_score():
    """score() folds the trap bonus into confluence for trendline."""
    from goldenswing.confluence import ConfluenceEngine
    ctx = _trap_ctx(boost_long_at=(5,))
    cand = _cand(5, playbook="trendline_retest", base=50.0)
    eng, _, _ = _engine([])
    sc = eng.score(cand, ctx, _TrendPB())
    assert sc.confluence == 50.0 + 4.0
    assert any("trap-boost" in n for n in sc.bonuses)


class _TrendPB(Playbook):
    name = "trendline_retest"
    live = False
    live_threshold = 0.0

    def candidates(self, ctx):
        return []
