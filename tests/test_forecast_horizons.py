"""Phase 6.5 — multi-horizon forecaster tests.

Covered honesty mechanisms:
- the horizon registry (the 72-bar row IS the Phase-6 definition);
- horizon-scaled label geometry (a move can be "big" on a half-day
  clock and nothing on a 1-day clock; a slow 3-ATR grind is a
  1-week event, invisible at 3 days, too small at 2 weeks);
- the purge scaling with the horizon;
- multi-horizon agreement (all clocks, same direction; ties never
  guessed);
- the rank API the alerts cite (percentiles, citation line, scorer
  over legacy and multi-horizon bundles).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from goldenswing.forecast.api import (ForecastTable, HorizonScores,
                                      MultiHorizonScorer, rank_series)
from goldenswing.forecast.features import build_feature_frame
from goldenswing.forecast.horizons import (AGREEMENT_HORIZONS,
                                           agreement_directions,
                                           cooldown_for,
                                           effective_samples)
from goldenswing.forecast.labels import (ADV_MULT, FAV_MULT, HORIZON,
                                         HORIZON_SPECS, build_labels,
                                         build_labels_for_horizon,
                                         spec_for)
from goldenswing.forecast.walkforward import make_folds
from goldenswing.reader.reader import MarketReader

FLAT = (100.0, 100.3, 99.7, 100.0)


def _frame(special: dict[int, tuple], n: int = 100) -> pd.DataFrame:
    rows = [special.get(i, FLAT) for i in range(n)]
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * n}, index=idx)


def _labels(df, horizon):
    return build_labels_for_horizon(df, np.ones(len(df)), horizon)


# ------------------------------------------------------------ registry
def test_registry_horizons_and_geometry():
    assert sorted(HORIZON_SPECS) == [12, 24, 72, 168, 360]
    expected = {12: (1.5, 0.75), 24: (2.0, 1.0), 72: (2.0, 1.0),
                168: (3.0, 1.5), 360: (4.0, 2.0)}
    for h, (fav, adv) in expected.items():
        spec = spec_for(h)
        assert (spec.fav_mult, spec.adv_mult) == (fav, adv)
        assert spec.fav_mult / spec.adv_mult == 2.0   # house 2:1


def test_72_spec_is_the_phase6_definition():
    spec = spec_for(72)
    assert (spec.horizon, spec.fav_mult, spec.adv_mult) \
        == (HORIZON, FAV_MULT, ADV_MULT)


def test_default_labels_equal_72_spec_labels():
    df = _frame({10: (100.0, 102.5, 99.5, 100.0),
                 30: (100.0, 100.9, 97.5, 100.0)})
    atr = np.ones(len(df))
    legacy = build_labels(df, atr)
    via_spec = build_labels(df, atr, horizon=72, fav_mult=2.0,
                            adv_mult=1.0)
    via_helper = _labels(df, 72)
    pd.testing.assert_frame_equal(legacy, via_spec)
    pd.testing.assert_frame_equal(legacy, via_helper)


# ------------------------------------------------------ label geometry
def test_pop_is_big_at_12h_but_not_at_1d():
    # +1.8 ATR pop, 0.4 ATR dip: clears 12h fav (1.5) cleanly,
    # never reaches 1d fav (2.0)
    df = _frame({5: (100.0, 101.8, 99.6, 100.0)})
    assert _labels(df, 12)["y_up"].iloc[0] == 1.0
    assert _labels(df, 24)["y_up"].iloc[0] == 0.0
    assert _labels(df, 12)["y_dn"].iloc[0] == 0.0


def _slow_rise(dip_bar: int | None = None) -> pd.DataFrame:
    rows = []
    for k in range(400):
        if k <= 80:
            rows.append(FLAT)
        elif k <= 110:
            c = 100.0 + 0.11 * (k - 80)
            rows.append((c, c + 0.05, c - 0.05, c))
        else:      # plateau at the top of the grind (~103.3)
            rows.append((103.3, 103.6, 103.0, 103.3))
    if dip_bar is not None:
        o, h, _l, c = rows[dip_bar]
        rows[dip_bar] = (o, h, 98.4, c)
    return _frame({i: r for i, r in enumerate(rows)}, n=400)


def test_slow_grind_is_a_1w_event_only():
    df = _slow_rise()
    assert _labels(df, 72)["y_up"].iloc[0] == 0.0    # window ends first
    assert _labels(df, 168)["y_up"].iloc[0] == 1.0   # +3 ATR by bar ~108
    assert _labels(df, 360)["y_up"].iloc[0] == 0.0   # never reaches +4


def test_adverse_dip_kills_the_1w_label():
    df = _slow_rise(dip_bar=40)      # 1.6 ATR dip >= 1.5 adverse @168
    assert _labels(df, 168)["y_up"].iloc[0] == 0.0


def test_incomplete_window_is_nan_at_360():
    lab = _labels(_frame({}, n=400), 360)
    assert np.isnan(lab["y_up"].iloc[50])        # 50 + 360 > 399
    assert not np.isnan(_labels(_frame({}, n=400), 168)
                         ["y_up"].iloc[50])


def test_labels_stay_mutually_exclusive_at_other_horizons():
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, 400))
    rows = {i: (c, c + abs(rng.normal(0, 0.3)),
                c - abs(rng.normal(0, 0.3)), c)
            for i, c in enumerate(close)}
    for h in (12, 168, 360):
        lab = _labels(_frame(rows, n=400), h)
        both = (lab["y_up"] == 1.0) & (lab["y_dn"] == 1.0)
        assert not both.any()


# ------------------------------------------------------------ walk-fwd
def test_purge_scales_with_horizon():
    n = 3000
    known = np.ones(n, dtype=bool)
    folds = make_folds(n, known, 360, initial_train=1500, step=250)
    assert folds
    for f in folds:
        assert f.train_idx.max() + 360 < f.test_idx.min()
        assert not set(f.train_idx) & set(f.test_idx)


def test_effective_samples_shrink_with_horizon():
    assert effective_samples(11426, 12) == 952
    assert effective_samples(11426, 72) == 158
    assert effective_samples(11426, 360) == 31


def test_cooldown_rule():
    assert cooldown_for(12) == 24
    assert cooldown_for(72) == 24
    assert cooldown_for(168) == 56
    assert cooldown_for(360) == 120


# ------------------------------------------------------------ agreement
def _probs(up: dict[int, float], dn: dict[int, float],
           n: int = 10) -> pd.DataFrame:
    pu = np.full(n, 0.3)
    pdn = np.full(n, 0.3)
    for t, v in up.items():
        pu[t] = v
    for t, v in dn.items():
        pdn[t] = v
    return pd.DataFrame({"ens_up": pu, "ens_dn": pdn},
                        index=pd.RangeIndex(n))


def test_agreement_requires_all_clocks_same_direction():
    by_h = {
        24: _probs({3: 0.70, 4: 0.80, 5: 0.30}, {5: 0.75}),
        72: _probs({3: 0.65, 4: 0.40, 5: 0.30}, {5: 0.70}),
        168: _probs({3: 0.60, 4: 0.90, 5: 0.30}, {5: 0.62}),
    }
    agree = agreement_directions(by_h)
    assert agree.get(3) == 1          # all three fire up
    assert 4 not in agree             # 72h clock does not fire up
    assert agree.get(5) == -1         # all three fire down
    assert set(agree) == {3, 5}


def test_agreement_tie_is_skipped_stronger_weak_link_wins():
    by_h = {
        24: _probs({7: 0.60, 8: 0.55}, {7: 0.60, 8: 0.70}),
        72: _probs({7: 0.60, 8: 0.55}, {7: 0.60, 8: 0.70}),
        168: _probs({7: 0.60, 8: 0.55}, {7: 0.60, 8: 0.70}),
    }
    agree = agreement_directions(by_h)
    assert 7 not in agree             # weakest links tie 0.60 vs 0.60
    assert agree.get(8) == -1         # down's weak link (0.70) wins


# ---------------------------------------------------------------- ranks
def test_rank_series_is_percentile_with_nan_passthrough():
    r = rank_series(np.array([1.0, 2.0, 3.0, np.nan, 0.5]))
    assert np.isnan(r[3])
    assert r[2] == 100.0              # max of the finite values
    assert r[3 - 3] > r[4]            # monotone in value
    assert np.all((r[np.isfinite(r)] >= 0.0)
                  & (r[np.isfinite(r)] <= 100.0))


def _table_with_ranks() -> ForecastTable:
    def hs(h, tag, rank_up_last):
        ru = np.full(20, 20.0)
        ru[-1] = rank_up_last
        return HorizonScores(
            horizon=h, tag=tag, prob_up=np.full(20, 0.4),
            prob_dn=np.full(20, 0.4), rank_up=ru,
            rank_dn=np.full(20, 10.0))
    return ForecastTable(scores={
        24: hs(24, "1d", 95.0), 72: hs(72, "3d", 97.0),
        168: hs(168, "1w", 50.0)})


def test_forecast_line_cites_qualifying_horizons():
    line = _table_with_ranks().line(1, 19)
    assert line is not None
    assert "@1d" in line and "@3d" in line
    assert "@1w" not in line          # rank 50: not a top-decile call


def test_forecast_line_is_none_when_nothing_qualifies():
    assert _table_with_ranks().line(-1, 19) is None


# ------------------------------------------------------- scorer / API
class _ConstModel:
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1.0 - self.p),
                                np.full(len(X), self.p)])


def _states(n: int = 420):
    rng = np.random.default_rng(3)
    closes = 2000.0 + np.cumsum(rng.normal(0, 1.5, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": closes - 0.05, "high": closes + 0.2,
         "low": closes - 0.2, "close": closes,
         "volume": [1000.0] * n}, index=idx)
    return MarketReader().read(df)


def _models(p: float) -> dict:
    return {"up": {"fake": _ConstModel(p)},
            "dn": {"fake": _ConstModel(1.0 - p)}}


def test_scorer_reads_legacy_bundle_as_72_only():
    states = _states()
    cols = list(build_feature_frame(states).columns)
    scorer = MultiHorizonScorer(
        {"models": _models(0.7), "features": cols}, states)
    table = scorer.table()
    assert table.horizons == (72,)
    assert np.allclose(table.scores[72].prob_up, 0.7)
    assert table.scores[72].rank_up[-1] == 100.0


def test_scorer_reads_multi_horizon_bundle():
    states = _states()
    cols = list(build_feature_frame(states).columns)
    bundle = {"features": cols,
              "horizons": {24: {"models": _models(0.6)},
                           72: {"models": _models(0.7)},
                           168: {"models": _models(0.8)}}}
    scorer = MultiHorizonScorer(bundle, states)
    table = scorer.table()
    assert table.horizons == (24, 72, 168)
    assert table.scores[24].tag == "1d"
    note = scorer.note_for(SimpleNamespace(direction=1,
                                           index=len(states) - 1))
    assert note is not None and note.text.startswith("Forecaster:")


def test_agreement_horizons_are_the_named_trio():
    assert AGREEMENT_HORIZONS == (24, 72, 168)
