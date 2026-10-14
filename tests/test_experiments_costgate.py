"""Tests for the GOD-CE cost-gate salvage (Workstream B).

- ``confluence.cost_gate_ok``: pine-faithful arithmetic, both legs,
  tp1 fallbacks, fail-closed behaviour (never raises).
- ``experiment_salvage.run_filter_experiment``: baseline vs filtered
  books, trap_df contract (positional alignment, column validation),
  cut-trade profile.
- ``experiment_costgate``: shares the harness keep_fn (two paths agree
  by construction), report builder renders.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from goldenswing.backtest.harness import Outcome
from goldenswing.confluence import ConfluenceConfig, cost_gate_ok
from goldenswing.golden.detector import GoldenSignal
from goldenswing.playbooks.base import Candidate

import experiment_costgate as CG
import experiment_salvage as SV


def _sig(**kw):
    base = dict(index=10, timestamp=None, direction=1, score=70.0,
                kind="sweep_reversal", invalidation=2990.0,
                targets=[3010.0], atr=5.0)
    base.update(kw)
    return GoldenSignal(**base)


def _cand(r_net: float, entry_index: int = 11, **sigkw) -> Candidate:
    sig = _sig(**sigkw)
    oc = Outcome(signal=sig, entry_index=entry_index, exit_index=20,
                 exit_reason="stop", r_net=r_net, hold_bars=9)
    return Candidate(playbook="sweep_reversal", outcome=oc,
                     base_score=70.0, thesis="t")


@pytest.fixture
def ctx():
    n = 40
    df = pd.DataFrame(
        {"open": np.full(n, 3000.0), "high": np.full(n, 3005.0),
         "low": np.full(n, 2995.0), "close": np.full(n, 3000.0),
         "volume": np.full(n, 100.0)})
    return SimpleNamespace(df=df, years=1.0)


# ------------------------------------------------------- cost_gate_ok
class TestCostGate:
    def test_pass_pine_arithmetic(self):
        # entry 3000, risk 10, cost_px = 3000*4/10000 = 1.2
        # leg1: 10 >= 2.4 ok; tp1 3010 -> tp1_r = 1.0;
        # leg2: 1.0 - 1.2/10 = 0.88 >= 0.6 ok
        ok, reason = cost_gate_ok(_sig(), 3000.0, ConfluenceConfig())
        assert ok and reason == ""

    def test_leg1_risk_too_small(self):
        sig = _sig(invalidation=2999.9)  # risk 0.1 < 2*1.2
        ok, reason = cost_gate_ok(sig, 3000.0, ConfluenceConfig())
        assert not ok and reason.startswith("risk-too-small")

    def test_leg2_net_tp1_r_too_low(self):
        # risk 100 clears leg1; nearest target 0.2R ahead -> net 0.188 < 0.6
        sig = _sig(invalidation=2900.0, targets=[3020.0])
        ok, reason = cost_gate_ok(sig, 3000.0, ConfluenceConfig())
        assert not ok and reason.startswith("net-tp1-r-too-low")

    def test_nearest_target_ahead_only(self):
        # targets behind entry are ignored; nearest ahead wins
        sig = _sig(invalidation=2990.0, targets=[2995.0, 3005.0, 3020.0])
        ok, reason = cost_gate_ok(sig, 3000.0, ConfluenceConfig())
        # tp1_r = 0.5 -> net 0.38 < 0.6 -> fail, naming nearest-target
        assert not ok and "nearest-target" in reason

    def test_short_direction(self):
        sig = _sig(direction=-1, invalidation=3010.0, targets=[2990.0])
        ok, _ = cost_gate_ok(sig, 3000.0, ConfluenceConfig())
        assert ok

    def test_fvg_fallback_2r_via_kind(self):
        # no targets ahead; sleeve_choch_fvg -> tp1_r = 2.0
        sig = _sig(kind="sleeve_choch_fvg", targets=[],
                   invalidation=2990.0)
        ok, reason = cost_gate_ok(sig, 3000.0, ConfluenceConfig())
        assert ok, reason  # 2.0 - 0.12 = 1.88 >= 0.6

    def test_fvg_fallback_2r_via_playbook_param(self):
        sig = _sig(kind="other", targets=[], invalidation=2990.0)
        ok, _ = cost_gate_ok(sig, 3000.0, ConfluenceConfig(),
                             playbook="fvg_midline")
        assert ok

    def test_default_fallback_1p5(self):
        # no targets, not fvg -> tp1_r = 1.5; net 1.5-0.12 = 1.38 ok
        sig = _sig(targets=[], invalidation=2990.0)
        ok, reason = cost_gate_ok(sig, 3000.0, ConfluenceConfig())
        assert ok, reason

    def test_default_fallback_can_fail_leg2(self):
        # tp1_r falls back to 1.5; with a strict min_net floor the
        # fallback itself is named in the failure reason.
        cfg = ConfluenceConfig(min_net_tp1_r=1.5)
        sig = _sig(targets=[], invalidation=2990.0)
        ok, reason = cost_gate_ok(sig, 3000.0, cfg)
        # net = 1.5 - 1.2/10 = 1.38 < 1.5
        assert not ok and "pine-tp1RR-default" in reason

    def test_fail_closed_no_entry_price(self):
        for bad in (None, float("nan"), -5.0, "x"):
            ok, reason = cost_gate_ok(_sig(), bad, ConfluenceConfig())
            assert not ok and reason.startswith("no-entry-price"), bad

    def test_fail_closed_no_invalidation(self):
        ok, reason = cost_gate_ok(_sig(invalidation=None), 3000.0,
                                  ConfluenceConfig())
        assert not ok and reason.startswith("no-invalidation")

    def test_fail_closed_risk_zero(self):
        ok, reason = cost_gate_ok(_sig(invalidation=3000.0), 3000.0,
                                  ConfluenceConfig())
        assert not ok and reason.startswith("risk<=0")

    def test_fail_closed_bad_direction(self):
        ok, reason = cost_gate_ok(_sig(direction=0), 3000.0,
                                  ConfluenceConfig())
        assert not ok and reason.startswith("no-direction")

    def test_never_raises_on_junk_targets(self):
        sig = _sig(targets=None)
        ok, _ = cost_gate_ok(sig, 3000.0, ConfluenceConfig())
        assert isinstance(ok, bool)
        sig = _sig(targets=[float("nan"), "junk", 3010.0])
        ok, _ = cost_gate_ok(sig, 3000.0, ConfluenceConfig())
        assert isinstance(ok, bool)

    def test_config_defaults_are_pine(self):
        cfg = ConfluenceConfig()
        assert cfg.cost_bps == 4.0 and cfg.min_net_tp1_r == 0.6


# ------------------------------------------------- run_filter_experiment
def _trap_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {c: [False] * n for c in SV.TRAP_COLUMNS})


class TestHarness:
    def test_baseline_vs_filtered_and_cut_profile(self, ctx):
        cands = []
        for i, r in enumerate([1.5, 0.5, -1.0, 2.0]):
            e = 11 + 10 * i
            sig = _sig(index=e - 1)
            oc = Outcome(signal=sig, entry_index=e, exit_index=e + 5,
                         exit_reason="stop", r_net=r, hold_bars=6)
            cands.append(Candidate(playbook="sweep_reversal", outcome=oc,
                                   base_score=70.0, thesis="t"))
        keep_fn = lambda cand, row: cand.outcome.r_net > 0  # noqa: E731
        res = SV.run_filter_experiment("pb", ctx, cands, keep_fn)
        assert res["n_candidates"] == 4
        assert res["n_kept"] == 3 and res["n_cut"] == 1
        assert res["baseline"]["trades"] == 4
        assert res["filtered"]["trades"] == 3
        cp = res["cut_profile"]
        assert cp["n"] == 1 and cp["avg_r"] == pytest.approx(-1.0)
        assert cp["buckets"]["<=-1R"] == 1

    def test_filtered_book_rebuilt_not_subtracted(self, ctx):
        # Overlapping trades: c1 (loser, occupies 11..20) blocks c2
        # (winner, 12..15) in the baseline. Cutting c1 re-times c2 INTO
        # the book — a "subtract the cut trade" approach would report
        # an empty book; the rebuilt book holds c2 (+1.0R).
        c1 = _cand(-2.0, entry_index=11)   # occupies 11..20
        oc2 = Outcome(signal=_sig(index=12), entry_index=12,
                      exit_index=15, exit_reason="target",
                      r_net=1.0, hold_bars=3)
        c2 = Candidate(playbook="sweep_reversal", outcome=oc2,
                       base_score=70.0, thesis="t")
        res = SV.run_filter_experiment(
            "pb", ctx, [c1, c2],
            lambda cand, row: cand is c2)  # noqa: E731
        assert res["baseline"]["trades"] == 1       # c1 only
        assert res["filtered"]["trades"] == 1       # c2, re-timed in
        assert res["filtered"]["avg_r"] == pytest.approx(1.0)

    def test_trap_row_positional(self, ctx):
        seen = []

        def keep(cand, row):
            seen.append((cand.entry_index,
                         None if row is None else bool(row["bait_long"])))
            return True

        tdf = _trap_df(40)
        tdf.loc[13, "bait_long"] = True
        SV.run_filter_experiment("pb", ctx, [_cand(1.0, entry_index=13)],
                                 keep, trap_df=tdf)
        assert seen == [(13, True)]

    def test_trap_none_when_no_df(self, ctx):
        seen = []
        SV.run_filter_experiment("pb", ctx, [_cand(1.0)],
                                 lambda c, r: seen.append(r) or True)  # noqa: E731
        assert seen == [None]

    def test_trap_df_length_mismatch_raises(self, ctx):
        with pytest.raises(ValueError, match="aligned to ctx.df by POSITION"):
            SV.run_filter_experiment("pb", ctx, [_cand(1.0)],
                                     lambda c, r: True,
                                     trap_df=_trap_df(39))

    def test_trap_df_missing_columns_raises(self, ctx):
        bad = _trap_df(40).drop(columns=["bait_long"])
        with pytest.raises(ValueError, match="missing contract columns"):
            SV.run_filter_experiment("pb", ctx, [_cand(1.0)],
                                     lambda c, r: True, trap_df=bad)

    def test_empty_cut_profile(self, ctx):
        res = SV.run_filter_experiment("pb", ctx, [_cand(1.0)],
                                       lambda c, r: True)
        assert res["cut_profile"]["n"] == 0
        assert res["filtered"]["trades"] == 1


# ------------------------------------------------- costgate experiment
class TestCostgateExperiment:
    def test_shared_keep_fn(self, ctx):
        # The experiment and the harness demo use the same keep_fn —
        # the two paths agree by construction.
        assert CG.costgate_keep_fn is SV.costgate_keep_fn

    def test_gate_with_reasons_fail_closed(self, ctx):
        cands = [_cand(1.0, entry_index=999)]  # out of range
        out = CG.gate_with_reasons(ctx, cands, ConfluenceConfig())
        assert out[0][1] is False
        assert out[0][2].startswith("no-entry-bar")

    def test_run_costgate_experiment_shape(self, ctx):
        cands = [_cand(1.0, entry_index=11),
                 _cand(-1.0, entry_index=12,
                       invalidation=2999.99)]  # leg-1 fail
        res = CG.run_costgate_experiment("pb", ctx, cands,
                                         ConfluenceConfig())
        assert res["n_cut"] == 1
        assert res["gate_reasons"] == {"risk-too-small": 1}
        assert res["cfg_used"] == {"cost_bps": 4.0, "min_net_tp1_r": 0.6}
        assert res["filtered"]["trades"] == 1

    def test_build_report_renders(self):
        m = {"trades": 10, "win_rate": 50.0, "pf": 1.5,
             "return_pct": 5.0, "max_dd_pct": 2.0, "sharpe": 1.0,
             "avg_r": 0.1, "avg_hold": 5.0}
        cp = {"n": 2, "win_rate": 0.0, "avg_r": -1.0, "pf": 0.0,
              "total_r": -2.0,
              "buckets": {"<=-1R": 2, "-1R..0R": 0, "0R..+1R": 0,
                          ">+1R": 0},
              "exit_reasons": {"stop": 2}}
        results = {n: {"n_candidates": 12, "n_kept": 10, "n_cut": 2,
                       "baseline": m, "filtered": m,
                       "cut_profile": cp, "gate_reasons": {"risk-too-small": 2}}
                   for n in ("sweep_reversal", "fvg_midline",
                             "trendline_retest")}
        md = CG.build_report(results, ConfluenceConfig())
        for n in ("sweep_reversal", "fvg_midline", "trendline_retest"):
            assert f"### {n}" in md
        assert "PINE DEFAULTS" in md
