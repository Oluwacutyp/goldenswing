"""MT5 playbook expansion: promotion + new real-trader setups.

A14 (rework, owner addendum 2026-10-21): the shadow tier is
abolished on the MT5 path — every playbook the runner evaluates is
eligible to execute through the same decision flow (Asymmetry
Score floors, RiskPool, hard lines); the demo engine IS the
shadow.  ``execute_playbooks`` defaults to None (= all evaluated);
an explicit tuple remains the owner's narrowing override, and
``disabled_playbooks`` is the per-playbook kill switch.  The
promotion (sweep_reversal anchor + fvg_limit) and each new setup
get their tests here: signal fires on the right structure, does
not fire on chop, no repainting (batch candidates == live_signal
signal set).
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.playbooks.base import EngineContext


def _ctx(df):
    return EngineContext(df=df, states=[], years=2.0, instrument="XAUUSD")


class _FakeExecutor:
    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return []

    def account_state(self):
        return {"equity": 10000.0}


def _runner(tmp_path, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False,
        **cfg_kw)
    runner = MT5Runner(cfg, executor=_FakeExecutor())
    runner.feed = SimpleNamespace(
        account_state=lambda: {"equity": 10000.0, "balance": 10000.0,
                               "free_margin": 10000.0},
        poll_status=lambda: {"ok": True})
    return runner


def _df(rows, freq="h"):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * len(rows)},
        index=idx,
    )


def test_mt5_execute_set_defaults_to_all_evaluated(tmp_path):
    """A14: no validated-subset execute list any more — the default
    config executes EVERY playbook the runner evaluates (shadow was
    unconfirmable by construction; the demo engine is the shadow).
    An explicit tuple remains the owner's narrowing override."""
    from goldenswing.playbooks import (
        FVGLimitPlaybook,
        FVGMidlinePlaybook,
        InversionPlaybook,
        JudasSwingPlaybook,
        KillzoneOpenPlaybook,
        SweepReversalPlaybook,
        TrendlineRetestPlaybook,
    )

    # Default: execute_playbooks is None — "all evaluated".
    assert MT5RunnerConfig().execute_playbooks is None
    runner = _runner(tmp_path)
    evaluated = {pb.name for pb in runner._get_playbooks()}
    # FUNNEL §2/§3: the H1 roster is 16 — qm_m5 left for its
    # native M5 route (_run_qm_m5), like qm_d1 on the D1 route.
    # Both native routes stay execute-eligible by default.
    assert evaluated == {"sweep_reversal", "fvg_midline",
                         "trendline_retest", "fvg_limit",
                         "inversion", "killzone_open", "judas_swing",
                         "turtle_soup", "v_reversal",
                         "retracement_trap", "order_block",
                         "range_sweep_reclaim", "range_qm",
                         "range_mean_reversion",
                         "range_compression_breakout", "narrative"}
    # Evaluated set == execute set under the default config —
    # including the playbooks the old tier kept shadow
    # (turtle_soup, v_reversal, retracement_trap, order_block).
    assert all(runner._executes(name) for name in evaluated)
    assert runner._executes("qm_m5") and runner._executes("qm_d1")
    # An explicit tuple narrows execution to exactly its members;
    # evaluation is unaffected (the kill switch is separate).
    narrowed = _runner(tmp_path, execute_playbooks=("judas_swing",))
    assert narrowed._executes("judas_swing")
    assert not narrowed._executes("turtle_soup")
    assert not narrowed._executes("qm_m5")
    assert {pb.name for pb in narrowed._get_playbooks()} == evaluated
    # Playbook class facts the expansion originally pinned (these
    # describe the playbooks themselves, not the runner's execute
    # set): the promoted anchor + sleeves are live-capable…
    for cls in (SweepReversalPlaybook, FVGMidlinePlaybook,
                TrendlineRetestPlaybook, FVGLimitPlaybook,
                InversionPlaybook, KillzoneOpenPlaybook,
                JudasSwingPlaybook):
        assert cls.live is True
    # Range playbooks are live AND regime-scoped (never in trend).
    from goldenswing.playbooks.range_regime import (
        RangeCompressionBreakoutPlaybook, RangeMeanReversionPlaybook,
        RangeQMPlaybook, RangeSweepReclaimPlaybook)
    for cls in (RangeSweepReclaimPlaybook, RangeQMPlaybook,
                RangeMeanReversionPlaybook):
        assert cls.live is True
        assert cls.regime_scope == "range"
    # The compression-breakout playbook is the boundary playbook:
    # it trades the breakout moment itself (scope "both").
    assert RangeCompressionBreakoutPlaybook.live is True
    assert RangeCompressionBreakoutPlaybook.regime_scope == "both"
