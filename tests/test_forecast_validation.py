"""F3 acceptance battery — synthetic journals with KNOWN
properties.

The battery must measure what the journal says, with
chronological discipline: a perfectly calibrated generator
scores ~perfect reliability, a biased point estimator shows
its bias, true-80% intervals measure 0.80, thin journals
render "insufficient data" instead of numbers, and the
locked holdout is provably untouched by the evaluation
section (a record set where leakage would change a number).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from goldenswing.forecast.validation import (
    METHOD_F1,
    METHOD_F2,
    read_journal,
    render_markdown,
    validate,
)

BASE = datetime(2026, 1, 1, 0, 0, 0)
TS = "%Y-%m-%d %H:%M:%S"
CAL = {"state": "calibrated", "n": 500, "up": 0.6, "dn": 0.3,
       "neutral": 0.1, "large_up": None, "large_dn": None,
       "basis": "test"}
UNCAL = {"state": "uncalibrated", "reason": "test"}


def _ts(i: int) -> str:
    return (BASE + timedelta(hours=i)).strftime(TS)


def _forecast(i: int, *, horizon: int = 24,
              method: str = "band_conditional",
              close: float = 4000.0, direction="UP",
              point=None, interval=None, probs=None,
              shadow=None, disagreement: float = 0.1,
              deadband: float = 0.5, fav: float = 40.0,
              adv: float = 10.0) -> dict:
    ho = {
        "direction": direction, "point_price": point,
        "interval": interval, "expected_move_usd": 40.0,
        "expected_move_pct": 1.0, "fav_excursion_usd": fav,
        "adv_excursion_usd": adv,
        "disagreement": disagreement,
        "probabilities": probs if probs is not None else UNCAL,
        "method": method, "n": 100, "deadband_pct": deadband,
        "barriers": None, "scenario": None,
        "invalidation": None, "unavailable_reason": None,
    }
    if shadow is not None:
        ho["trained_shadow"] = shadow
    return {
        "type": "forecast", "cadence": "H1",
        "generated_at": _ts(i), "bar_time": _ts(i),
        "close": close,
        "horizons": {str(horizon): {"up": 80.0, "dn": 20.0}},
        "magnitude": {},
        "outlook": {
            "method": "f1-empirical-v1", "created": _ts(i),
            "horizons": {str(horizon): ho},
        },
    }


def _recon(i: int, *, horizon: int = 24, ret: float = 1.0,
           run_up: float = 40.0, run_dn: float = 10.0) -> dict:
    return {
        "type": "forecast_reconciliation", "bar_time": _ts(i),
        "cadence": "H1",
        "horizons": {str(horizon): {
            "y_up": 1 if ret > 0 else 0,
            "y_dn": 1 if ret < 0 else 0,
            "ret_pct": ret, "run_up": run_up,
            "run_dn": run_dn}},
    }


def _pair(i: int, **kw) -> list[dict]:
    ret = kw.pop("ret", 1.0)
    run_up = kw.pop("run_up", 40.0)
    run_dn = kw.pop("run_dn", 10.0)
    return [_forecast(i, **kw),
            _recon(i, ret=ret, run_up=run_up, run_dn=run_dn)]


def _journal(*pairs: list[dict]) -> list[dict]:
    out: list[dict] = []
    for p in pairs:
        out.extend(p)
    return out


def _cell(result: dict, horizon: int = 24,
          method: str = "f1-empirical") -> dict:
    return result["cells"][f"{horizon}|{method}"]


# ── the join ────────────────────────────────────────────────


def test_join_dedupes_first_wins_and_merges_recons():
    pair = _pair(0, point=4010.0)
    dup = _forecast(0, point=4999.0)  # same bar_time, later
    extra_recon = _recon(0)
    # a second horizon answered on the same bar, reconciled in
    # a separate append
    fc2 = _forecast(1, horizon=12, point=4005.0)
    fc2["outlook"]["horizons"]["24"] = \
        _forecast(1, point=4011.0)["outlook"]["horizons"]["24"]
    recs = _journal(pair, [dup, extra_recon, fc2,
                           _recon(1, horizon=12),
                           _recon(1, horizon=24)])
    result = validate(recs)
    cov = result["coverage"]
    assert cov["forecast_bars"] == 2
    assert cov["bars_reconciled"] == 2
    assert cov["scored_rows"] == 3
    # the duplicate's point (4999) must NOT be what was scored
    cell = _cell(result)
    assert cell["segments"]["pooled"]["point"]["mae_usd"] \
        == pytest.approx(29.5)  # |4010-4040| & |4011-4040| avg


def test_forecast_without_outlook_or_outcome_is_counted():
    bare = {"type": "forecast", "cadence": "H1",
            "bar_time": _ts(9), "close": 4000.0,
            "horizons": {}, "magnitude": {}}
    recs = _journal(_pair(0, point=4010.0)) + [bare,
                                               _forecast(3)]
    result = validate(recs)
    cov = result["coverage"]
    assert cov["forecast_bars"] == 3
    assert cov["bars_with_outlook"] == 2
    assert cov["scored_rows"] == 1
    assert cov["horizons_unresolved"] == 1


def test_unavailable_horizon_is_not_scored():
    recs = _journal(_pair(0, method="unavailable",
                          direction=None, point=None))
    result = validate(recs)
    assert result["coverage"]["scored_rows"] == 0
    assert result["overall"] == "insufficient_data"


# ── known-property generators ───────────────────────────────


def test_biased_point_estimator_shows_its_bias():
    recs: list[dict] = []
    for i in range(25):
        # realized endpoint is always close + 0.25% = 4010;
        # the estimator always says 4020 → +$10 bias per row
        recs += _pair(i, point=4020.0, ret=0.25,
                      interval={"low": 3990.0, "high": 4030.0,
                                "coverage": 0.80},
                      deadband=0.1)
    cell = _cell(validate(recs))
    p = cell["segments"]["pooled"]["point"]
    assert p["n"] == 25
    assert p["mae_usd"] == 10.0
    assert p["medae_usd"] == 10.0
    assert p["mae_pct"] == pytest.approx(0.25)
    assert p["naive_mae_pct"] == pytest.approx(0.25)
    assert p["skill"] == pytest.approx(1.0)
    iv = cell["segments"]["pooled"]["interval"]
    assert iv["coverage"] == 1.0
    d = cell["segments"]["pooled"]["direction"]
    assert d["hit_rate"] == 1.0  # UP call, ret +0.25 > 0.1


def test_perfectly_calibrated_probabilities_score_clean():
    recs: list[dict] = []
    # stated 0.6/0.3/0.1; realized exactly 60/30/10
    rets = [1.0] * 60 + [-1.0] * 30 + [0.0] * 10
    for i, ret in enumerate(rets):
        recs += _pair(i, probs=dict(CAL), ret=ret,
                      direction=None)
    cell = _cell(validate(recs))
    pr = cell["segments"]["pooled"]["probabilities"]
    assert pr["n"] == 100
    assert pr["ece"] == pytest.approx(0.0, abs=1e-9)
    # (60·.26 + 30·.86 + 10·1.26) / 100
    assert pr["brier"] == pytest.approx(0.54)
    for b in pr["bins"]:
        assert b["stated"] == pytest.approx(
            b["realized"], abs=1e-9)


def test_true_80_intervals_measure_080():
    recs: list[dict] = []
    # endpoints: 10 at −1%, 80 flat, 10 at +1%; the stated
    # interval ±0.5% catches exactly the flat 80
    rets = [-1.0] * 10 + [0.0] * 80 + [1.0] * 10
    for i, ret in enumerate(rets):
        recs += _pair(
            i, point=4000.0, ret=ret, direction="NEUTRAL",
            deadband=1.5,
            interval={"low": 3980.0, "high": 4020.0,
                      "coverage": 0.80})
    cell = _cell(validate(recs))
    iv = cell["segments"]["pooled"]["interval"]
    assert iv["n"] == 100
    assert iv["coverage"] == pytest.approx(0.80)
    d = cell["segments"]["pooled"]["direction"]
    assert d["calls"] == 0
    assert d["neutral_n"] == 100
    assert d["neutral_rate"] == 1.0


def test_neutral_abstention_quality_is_measured():
    recs: list[dict] = []
    # neutral calls with deadband 0.5: half the outcomes stay
    # inside the band (right to abstain), half break out
    rets = [0.1] * 10 + [2.0] * 5 + [-2.0] * 5
    for i, ret in enumerate(rets):
        recs += _pair(i, direction="NEUTRAL", ret=ret,
                      deadband=0.5)
    d = _cell(validate(recs))["segments"]["pooled"]["direction"]
    assert d["neutral_n"] == 20
    assert d["neutral_rate"] == pytest.approx(0.5)


def test_large_move_precision_and_recall():
    recs: list[dict] = []
    probs = dict(CAL)
    # 10 rows called at p=0.5: 6 realize the up barrier race
    for i in range(10):
        p = dict(probs, large_up=0.5, large_dn=None)
        recs += _pair(i, probs=p, ret=1.0 if i < 6 else -1.0)
    # 10 rows at p=0.2 (below the 0.40 call line): 1 realizes
    for j in range(10):
        p = dict(probs, large_up=0.2, large_dn=None)
        recs += _pair(10 + j, probs=p,
                      ret=1.0 if j == 0 else -1.0)
    lm = _cell(validate(recs))["segments"]["pooled"][
        "large_move"]["up"]
    assert lm["n"] == 20
    assert lm["mean_stated"] == pytest.approx(0.35)
    assert lm["realized_rate"] == pytest.approx(0.35)
    assert lm["calls"] == 10
    assert lm["precision"] == pytest.approx(0.6)
    assert lm["recall"] == pytest.approx(6 / 7, abs=1e-3)
    dn = _cell(validate(recs))["segments"]["pooled"][
        "large_move"]["dn"]
    assert dn["n"] == 0


# ── chronological discipline ────────────────────────────────


def test_holdout_is_locked_and_carries_the_verdict():
    recs: list[dict] = []
    for i in range(200):
        if i < 140:
            # eval: honest narrow intervals that cover, widths
            # cycling 1.8% / 2.0% / 2.2%
            w = 1.8 + (i % 3) * 0.2
            low = 4000.0 * (1 - w / 200)
            high = 4000.0 * (1 + w / 200)
        else:
            # holdout: wide intervals parked far above price —
            # pooled volatility edges would move if the
            # battery fitted them on the holdout too
            low, high = 4800.0, 5800.0
        ret = (-0.5, 0.0, 0.5)[i % 3]
        endpoint = 4000.0 * (1 + ret / 100)
        recs += _pair(i, point=endpoint, ret=ret,
                      interval={"low": low, "high": high,
                                "coverage": 0.80})
    result = validate(recs)
    # volatility edges fitted on the eval segment ONLY:
    # sorted eval widths → terciles land on 1.8 and 2.0
    assert result["vol_edges_pct"] == [1.8, 2.0]
    cell = _cell(result)
    assert cell["segments"]["eval"]["interval"]["coverage"] \
        == 1.0
    assert cell["segments"]["holdout"]["interval"]["coverage"] \
        == 0.0
    # pooled coverage (0.70) sits inside the ±0.10 band — if
    # the verdict read the pooled rows it would PASS; the
    # locked holdout is what is verdicted, and it fails
    assert cell["segments"]["pooled"]["interval"]["coverage"] \
        == pytest.approx(0.70)
    v = cell["verdict"]
    assert v["segment"] == "holdout"
    assert v["firm"] is True
    assert v["state"] == "outside_band"
    assert "coverage" in v["fails"]
    assert v["checks"]["coverage"]["value"] == 0.0


def test_segment_split_never_straddles_a_bar():
    recs: list[dict] = []
    for i in range(10):
        fc = _forecast(i, horizon=24, point=4010.0)
        fc["outlook"]["horizons"]["12"] = \
            _forecast(i, horizon=12,
                      point=4005.0)["outlook"]["horizons"]["12"]
        recs += [fc, _recon(i, horizon=24), _recon(i, horizon=12)]
    result = validate(recs)
    for method_key in ("f1-empirical",):
        c24 = _cell(result, 24, method_key)
        c12 = _cell(result, 12, method_key)
        # 10 bars → 7 eval / 3 holdout, both horizons together
        assert c24["segments"]["eval"]["n"] == 7
        assert c24["segments"]["holdout"]["n"] == 3
        assert c12["segments"]["eval"]["n"] == 7
        assert c12["segments"]["holdout"]["n"] == 3


def test_happy_path_cell_is_within_band_on_holdout():
    recs: list[dict] = []
    cycle = [1.0] * 6 + [-1.0] * 3 + [0.0]
    for i in range(200):
        ret = cycle[i % 10]
        direction = ("UP" if ret > 0 else
                     "DOWN" if ret < 0 else "NEUTRAL")
        point = 4000.0 * (1 + ret / 200)  # half the move
        if i % 20 < 17:
            interval = {"low": 3920.0, "high": 4080.0,
                        "coverage": 0.80}
        else:
            interval = {"low": 4120.0, "high": 4200.0,
                        "coverage": 0.80}
        recs += _pair(i, point=point, ret=ret,
                      direction=direction, probs=dict(CAL),
                      interval=interval, deadband=0.5)
    result = validate(recs)
    cell = _cell(result)
    v = cell["verdict"]
    assert v["state"] == "within_band"
    assert v["segment"] == "holdout"
    assert v["firm"] is True
    assert v["n"] == 60
    hold = cell["segments"]["holdout"]
    assert hold["interval"]["coverage"] == pytest.approx(0.85)
    assert hold["probabilities"]["ece"] == pytest.approx(0.0)
    assert hold["direction"]["hit_rate"] == 1.0
    assert hold["point"]["skill"] == pytest.approx(0.5)
    assert hold["point"]["mae_usd"] == pytest.approx(18.0)
    # DOWN rows' favourable side is run_dn (10) against a 40
    # estimate → fav MAE = 18 rows · 30 / 60
    assert hold["excursions"]["fav_mae_usd"] == \
        pytest.approx(9.0)
    report = render_markdown(result)
    assert "WITHIN BAND" in report
    assert "1d f1-empirical" in report


# ── honesty rules ───────────────────────────────────────────


def test_uncalibrated_probabilities_are_never_scored():
    recs: list[dict] = []
    for i in range(30):
        recs += _pair(i, probs={"state": "uncalibrated",
                                "up": 0.99, "dn": 0.005,
                                "neutral": 0.005},
                      ret=1.0)
    cell = _cell(validate(recs))
    pooled = cell["segments"]["pooled"]
    assert pooled["probabilities"]["n"] == 0
    assert pooled["probabilities"]["ece"] is None
    assert "ece" not in cell["verdict"]["checks"]


def test_shadow_series_scored_separately_never_in_verdict():
    shadow = {"state": "shadow", "up": 0.6, "dn": 0.3,
              "neutral": 0.1, "large_up": None,
              "large_dn": None, "deadband_pct": 0.5}
    recs: list[dict] = []
    rets = [1.0] * 60 + [-1.0] * 30 + [0.0] * 10
    for i, ret in enumerate(rets):
        recs += _pair(i, shadow=shadow, ret=ret)
    cell = _cell(validate(recs))
    pooled = cell["segments"]["pooled"]
    assert pooled["probabilities"]["n"] == 0
    sh = pooled["shadow_probabilities"]
    assert sh["n"] == 100
    assert sh["ece"] == pytest.approx(0.0, abs=1e-9)


def test_thin_journal_renders_insufficient_not_numbers():
    recs = _journal(*[_pair(i, point=4010.0) for i in range(5)])
    result = validate(recs)
    assert result["overall"] == "insufficient_data"
    v = _cell(result)["verdict"]
    assert v["state"] == "insufficient_data"
    assert v["n"] == 5
    report = render_markdown(result)
    assert "INSUFFICIENT DATA (n=5)" in report
    assert "not enough resolved forecast history" in report


def test_read_journal_counts_torn_lines(tmp_path):
    path = tmp_path / "journal.jsonl"
    good = json.dumps(_forecast(0))
    path.write_text(
        good + "\n" + '{"type": "forecast", "bar_t' + "\n"
        + "[1, 2]\n" + good + "\n", encoding="utf-8")
    records, malformed = read_journal(str(path))
    assert len(records) == 2
    assert malformed == 2


# ── disagreement + method comparison ────────────────────────


def _disagreement_journal(high_ratio: float) -> list[dict]:
    recs: list[dict] = []
    for i in range(40):
        high = i >= 20
        # realized endpoint always +1% (4040); low-disagreement
        # rows err 0.25%, high rows err 0.25% · ratio
        err_pct = 0.25 * (high_ratio if high else 1.0)
        point = 4040.0 - 4000.0 * err_pct / 100
        recs += _pair(i, point=point, ret=1.0,
                      disagreement=0.5 if high else 0.05)
    return recs


def test_disagreement_with_no_signal_is_flagged_decoration():
    dg = _cell(validate(_disagreement_journal(1.0)))[
        "segments"]["pooled"]["disagreement"]
    assert dg["reading"] == "no_measured_value"
    assert dg["mae_ratio"] == pytest.approx(1.0)


def test_disagreement_that_predicts_error_earns_its_place():
    dg = _cell(validate(_disagreement_journal(3.0)))[
        "segments"]["pooled"]["disagreement"]
    assert dg["reading"] == "earns_its_place"
    assert dg["mae_ratio"] == pytest.approx(3.0)


def test_head_to_head_picks_the_better_point_method():
    recs: list[dict] = []
    for i in range(40):
        # F1: point is the no-change close → skill 1.0
        recs += _pair(i, point=4000.0, ret=1.0)
        # F2 (same bars, second journal entry would dedupe —
        # so use different bars): point is exact → skill 0
        recs += _pair(1000 + i, method="f2-trained-v1",
                      point=4040.0, ret=1.0)
    result = validate(recs)
    assert {METHOD_F1, METHOD_F2} == {"f1-empirical",
                                      "f2-trained"}
    assert _cell(result, 24, "f2-trained")["segments"][
        "pooled"]["point"]["skill"] == pytest.approx(0.0)
    hh = result["head_to_head"]["24"]
    assert hh["contested"] is True
    assert hh["leader"] == "f2-trained"
    report = render_markdown(result)
    assert "**f2-trained** leads f1-empirical" in report


def test_regime_splits_use_journal_fields():
    recs: list[dict] = []
    for i in range(96):
        # bars at hours 0..23 four times: every session arm
        # clears the minimum cell size
        point = 4040.0
        recs += _pair(i, point=point, ret=1.0,
                      interval={"low": 3960.0 - i,
                                "high": 4040.0 + i,
                                "coverage": 0.80})
    splits = _cell(validate(recs))["splits"]
    sessions = splits["session"]
    assert set(sessions) == {"asia", "london", "new_york",
                             "off_hours"}
    assert sessions["asia"]["n"] == 32  # hours 0-7, four times
    assert "dominant_side" in splits


# ── the CLI ─────────────────────────────────────────────────


def test_cli_reports_on_thin_journal_and_fails_cleanly(
        tmp_path):
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "validate_forecast.py"
    journal = tmp_path / "j.jsonl"
    journal.write_text(
        "\n".join(json.dumps(r)
                  for r in _journal(_pair(0, point=4010.0)))
        + "\n", encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(script), "--journal",
         str(journal)], capture_output=True, text=True)
    assert run.returncode == 0
    assert "F3 acceptance battery" in run.stdout
    assert "INSUFFICIENT DATA" in run.stdout
    missing = subprocess.run(
        [sys.executable, str(script), "--journal",
         str(tmp_path / "nope.jsonl")],
        capture_output=True, text=True)
    assert missing.returncode == 2
