"""Phase-3.5 validation helpers: the winner rule must implement the
pre-declared selection (higher book-@60 return; within 1.0pp, lower
MaxDD), and run_config must reproduce the Phase-3 ablation mechanics
on a small frame without error."""
import numpy as np
import pandas as pd

from goldenswing.backtest.validation import (COMBO_A, COMBO_B, ValResult,
                                             pick_winner, run_config)
from goldenswing.features import FeatureEngine
from goldenswing.reader.reader import MarketReader


def _vr(ret: float, dd: float) -> ValResult:
    return ValResult(name="x", n_signals=0,
                     book60={"return_pct": ret, "max_dd_pct": dd},
                     book50={}, shadow60={}, bands=[])


def test_winner_clear_return_gap():
    assert pick_winner(_vr(20.0, 12.0), _vr(25.0, 15.0)).book60[
        "return_pct"] == 25.0


def test_winner_close_returns_lower_dd_wins():
    w = pick_winner(_vr(20.0, 9.0), _vr(20.5, 14.0))
    assert w.book60["max_dd_pct"] == 9.0


def test_combos_are_the_phase3_recommendation():
    assert COMBO_A == frozenset({"relstr", "volume", "phases"})
    assert COMBO_B == COMBO_A | {"bollinger"}


def test_run_config_smoke(df_from_closes):
    rng = np.random.default_rng(7)
    closes = 2000.0 + np.cumsum(rng.normal(0, 3.0, 400))
    df = df_from_closes(list(closes))
    states = MarketReader().read(df)
    FeatureEngine().attach(df, states, None)
    res = run_config(df, states, COMBO_A, "smoke")
    assert res.name == "smoke"
    assert len(res.bands) == 4
    assert res.book60["trades"] >= 0
    base = run_config(df, MarketReader().read(df), None, "base")
    assert base.n_signals >= 0
