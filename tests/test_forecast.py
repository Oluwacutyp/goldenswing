"""Phase 6 — forecaster tests.

The three honesty mechanisms get direct tests:
- the label builder (known synthetic moves, same-bar adverse-wins,
  incomplete windows are NaN, never zero-filled);
- feature causality (rows built from a truncated series equal the
  full-series rows on the shared prefix);
- the walk-forward purge (no training row's label window may touch
  its fold's test block).
"""
import numpy as np
import pandas as pd
import pytest

from goldenswing.features import FeatureEngine
from goldenswing.forecast.features import build_feature_frame
from goldenswing.forecast.labels import build_labels
from goldenswing.forecast.walkforward import make_folds, run_walk_forward
from goldenswing.reader.reader import MarketReader

FLAT = (100.0, 100.3, 99.7, 100.0)


def _frame(special: dict[int, tuple], n: int = 100) -> pd.DataFrame:
    rows = [special.get(i, FLAT) for i in range(n)]
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * n}, index=idx)


def _labels(df, horizon=72):
    return build_labels(df, np.ones(len(df)), horizon=horizon)


def test_label_clean_up_move():
    # bar 10: +2.5 ATR high, only a 0.5 ATR dip -> clean big move up
    lab = _labels(_frame({10: (100.0, 102.5, 99.5, 100.0)}))
    assert lab["y_up"].iloc[0] == 1.0
    assert lab["y_dn"].iloc[0] == 0.0   # the rally is adverse for down


def test_label_adverse_first_kills_up():
    # bar 5 dips 1.5 ATR (adverse) before the bar-10 rally -> not clean
    lab = _labels(_frame({5: (100.0, 100.3, 98.5, 100.0),
                          10: (100.0, 102.5, 99.5, 100.0)}))
    assert lab["y_up"].iloc[0] == 0.0


def test_label_clean_down_move():
    lab = _labels(_frame({8: (100.0, 100.9, 97.5, 100.0)}))
    assert lab["y_dn"].iloc[0] == 1.0
    assert lab["y_up"].iloc[0] == 0.0


def test_label_same_bar_double_touch_is_adverse():
    lab = _labels(_frame({5: (100.0, 102.5, 98.5, 100.0)}))
    assert lab["y_up"].iloc[0] == 0.0
    assert lab["y_dn"].iloc[0] == 0.0


def test_label_quiet_market_is_zero_not_nan():
    lab = _labels(_frame({}))
    assert lab["y_up"].iloc[0] == 0.0
    assert lab["y_dn"].iloc[0] == 0.0


def test_label_incomplete_window_is_nan():
    lab = _labels(_frame({}), horizon=72)
    assert np.isnan(lab["y_up"].iloc[28])     # 28 + 72 > 99
    assert lab["y_up"].iloc[27] == 0.0        # 27 + 72 == 99: full window


def test_labels_mutually_exclusive_on_noise():
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, 400))
    rows = [(c, c + abs(rng.normal(0, 0.3)), c - abs(rng.normal(0, 0.3)), c)
            for c in close]
    lab = _labels(_frame({i: r for i, r in enumerate(rows)}, n=400),
                  horizon=72)
    both = (lab["y_up"] == 1.0) & (lab["y_dn"] == 1.0)
    assert not both.any()


def test_feature_rows_are_prefix_stable(df_from_closes):
    rng = np.random.default_rng(3)
    closes = 2000.0 + np.cumsum(rng.normal(0, 1.5, 420))
    df = df_from_closes(list(closes))

    def build(frame):
        states = MarketReader().read(frame)
        FeatureEngine().attach(frame, states, None)
        return build_feature_frame(states)

    full = build(df)
    trunc = build(df.iloc[:300])
    a = full.iloc[:300].to_numpy(dtype=float)
    b = trunc.to_numpy(dtype=float)
    assert full.columns.tolist() == trunc.columns.tolist()
    assert np.allclose(a, b, equal_nan=True)


def test_folds_purge_label_windows():
    n, horizon = 2000, 72
    known = np.ones(n, dtype=bool)
    folds = make_folds(n, known, horizon, initial_train=1000, step=250)
    assert folds, "expected folds"
    for f in folds:
        assert len(f.train_idx) and len(f.test_idx)
        # every training row's label window ends before the test block
        assert f.train_idx.max() + horizon < f.test_idx.min()
        assert not set(f.train_idx) & set(f.test_idx)


def test_walk_forward_smoke():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    lab = pd.DataFrame({
        "y_up": (rng.random(n) < 0.3).astype(float),
        "y_dn": (rng.random(n) < 0.3).astype(float)})
    res = run_walk_forward(X, lab, horizon=10, initial_train=300, step=100)
    assert len(res.probs) > 0
    for col in ("ens_up", "ens_dn"):
        p = res.probs[col].to_numpy()
        assert np.all((p >= 0.0) & (p <= 1.0))
    # first test bar is at/after the initial train boundary
    assert res.probs.index.min() >= 300
