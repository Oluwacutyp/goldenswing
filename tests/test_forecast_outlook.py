"""Forecast F1 — the outlook block: numbers, not just ranks.

Pins: empirical estimation (band-conditional on the resolved
live sample ring, unconditional H1 fallback, thin-data honesty),
the deadband direction rule, earned probabilities (direction vs
large-move kept separate, E3's n>=20 precedent), realized-outcome
capture at resolution (resolve_record's pinned shape untouched),
journal round-trip, the card/status rendering (numbers lead,
ranks demoted to one auxiliary line, uncalibrated stated once,
horizon-split flag), disagreement from per-model outputs, and
the runner wiring (journal block always written; the card flag
gates only rendering).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goldenswing.forecast.calibration import (
    fresh_state,
    load_state,
    realized_outcomes,
    resolve_record,
    save_state,
)
from goldenswing.forecast.mt5 import ForecastRanks
from goldenswing.forecast.outlook import (
    INTERVAL_COVERAGE,
    MIN_BAND_N,
    MIN_PROB_N,
    build_outlook,
    contradiction_line,
    forward_outcomes,
    horizon_line,
    render_card,
    render_status_lines,
)
from goldenswing.reader.indicators import atr as atr_series

CLOSE = 4000.0


def _sample(horizon=24, up=85.0, dn=10.0, ret=0.8, run_up=60.0,
            run_dn=25.0, y_up=1, y_dn=0, bar="2026-10-20 00:00:00"):
    return {"bar_time": bar, "horizon": horizon, "up": up, "dn": dn,
            "ret_pct": ret, "run_up": run_up, "run_dn": run_dn,
            "y_up": y_up, "y_dn": y_dn}


def _band_samples(n=40, *, horizon=24, up=85.0, dn=10.0,
                  rets=(-0.4, 0.2, 0.8, 1.4, 2.0), run_up=60.0,
                  run_dn=25.0, y_up_hits=None, y_dn_hits=None):
    """A resolved band sample with a known distribution: ret
    cycles the given values (default mean 0.8, sd ~0.85); the
    barrier labels hit at the stated counts (default: large-up
    75%, large-dn 10%)."""
    if y_up_hits is None:
        y_up_hits = int(round(0.75 * n))
    if y_dn_hits is None:
        y_dn_hits = int(round(0.10 * n))
    return [_sample(
        horizon=horizon, up=up, dn=dn, ret=rets[i % len(rets)],
        run_up=run_up, run_dn=run_dn,
        y_up=1 if i < y_up_hits else 0,
        y_dn=1 if i < y_dn_hits else 0) for i in range(n)]


def _drift_frame(n=400, step=2.0, start=3000.0):
    idx = pd.date_range("2026-06-01", periods=n, freq="h", tz="UTC")
    close = start + step * np.arange(n)
    return pd.DataFrame(
        {"open": close - 1.0, "high": close + 4.0,
         "low": close - 4.0, "close": close}, index=idx)


# ── forward outcomes (the unconditional distribution) ───────────


def test_forward_outcomes_hand_computed():
    idx = pd.date_range("2026-06-01", periods=6, freq="h")
    df = pd.DataFrame(
        {"open": [100.0] * 6,
         "high": [101.0, 105.0, 103.0, 102.0, 104.0, 106.0],
         "low": [99.0, 98.0, 100.0, 101.0, 100.0, 102.0],
         "close": [100.0, 102.0, 101.0, 103.0, 102.0, 105.0]},
        index=idx)
    out = forward_outcomes(df, 2)
    assert len(out["ret_pct"]) == 4
    # bar 0: window bars 1-2 → end close 101 (+1%), max high 105,
    # min low 98
    assert out["ret_pct"][0] == pytest.approx(1.0)
    assert out["run_up"][0] == pytest.approx(5.0)
    assert out["run_dn"][0] == pytest.approx(2.0)
    # bar 3: window bars 4-5 → end 105 vs 103, max high 106, min 100
    assert out["ret_pct"][3] == pytest.approx((105 / 103 - 1) * 100)
    assert out["run_up"][3] == pytest.approx(3.0)
    assert out["run_dn"][3] == pytest.approx(3.0)


# ── estimation: band-conditional ────────────────────────────────


def test_band_conditional_outlook_numbers():
    outlook = build_outlook(
        ranks={"24": {"up": 85.0, "dn": 10.0}}, close=CLOSE,
        atr=25.0, samples=_band_samples(), h1=None,
        model_probs={24: {"up": [0.8, 0.6, 0.7],
                          "dn": [0.2, 0.3, 0.25]}},
        created="2026-11-03 17:00:00", data_cutoff="2026-11-03 16:00:00",
        model_version="forecast-bundle-v2")
    ho = outlook["horizons"]["24"]
    assert ho["method"] == "band_conditional"
    assert ho["n"] == 40
    assert ho["direction"] == "UP"
    assert ho["point_price"] == pytest.approx(CLOSE * 1.008, abs=0.6)
    assert ho["expected_move_pct"] == pytest.approx(0.8)
    assert ho["expected_move_usd"] == pytest.approx(32.0, abs=0.6)
    assert ho["fav_excursion_usd"] == pytest.approx(60.0)
    assert ho["adv_excursion_usd"] == pytest.approx(25.0)
    iv = ho["interval"]
    assert iv["coverage"] == INTERVAL_COVERAGE
    assert iv["low"] < ho["point_price"] < iv["high"]
    # disagreement = stdev of per-model leans [0.6, 0.3, 0.45]
    assert ho["disagreement"] == pytest.approx(
        float(np.std([0.6, 0.3, 0.45])), abs=1e-3)
    # barriers from the label geometry at ATR 25 (fav 2x / adv 1x)
    assert ho["barriers"]["fav"] == pytest.approx(CLOSE + 50.0)
    assert ho["invalidation"] == pytest.approx(CLOSE - 25.0)
    assert "barrier race" in ho["scenario"]
    # record-level meta
    assert outlook["method"] == "f1-empirical-v1"
    assert outlook["expiry"]["24"] == "2026-11-04 16:00:00"
    assert outlook["model_version"] == "forecast-bundle-v2"


def test_probabilities_earned_and_kept_separate():
    outlook = build_outlook(
        ranks={"24": {"up": 85.0, "dn": 10.0}}, close=CLOSE,
        atr=None, samples=_band_samples(), h1=None)
    probs = outlook["horizons"]["24"]["probabilities"]
    assert probs["state"] == "calibrated"
    assert probs["n"] == 40
    # endpoint-sign frequencies: deadband ~0.138 → up .8 / dn .2
    assert probs["up"] == pytest.approx(0.8)
    assert probs["dn"] == pytest.approx(0.2)
    assert probs["neutral"] == pytest.approx(0.0)
    # large-move frequencies are a DIFFERENT question (barrier race)
    assert probs["large_up"] == pytest.approx(0.75)
    assert probs["large_dn"] == pytest.approx(0.1)
    assert probs["large_up"] != probs["up"]


def test_probabilities_below_min_n_are_uncalibrated():
    few = _band_samples(n=MIN_PROB_N - 1)
    outlook = build_outlook(
        ranks={"24": {"up": 85.0, "dn": 10.0}}, close=CLOSE,
        atr=None, samples=few, h1=_drift_frame())
    probs = outlook["horizons"]["24"]["probabilities"]
    assert probs["state"] == "uncalibrated"
    assert str(MIN_PROB_N) in probs["reason"]


# ── estimation: fallback + thin-data honesty ────────────────────


def test_unconditional_fallback_when_band_thin():
    outlook = build_outlook(
        ranks={"12": {"up": 88.0, "dn": 5.0}}, close=CLOSE,
        atr=None, samples=_band_samples(n=3), h1=_drift_frame())
    ho = outlook["horizons"]["12"]
    assert ho["method"] == "unconditional_h1"
    assert ho["n"] >= 50
    assert ho["direction"] == "UP"  # +2/bar drift over 12 bars
    assert ho["probabilities"]["state"] == "uncalibrated"


def test_thin_everything_reports_unavailable_not_invented():
    outlook = build_outlook(
        ranks={"12": {"up": 88.0, "dn": 5.0}}, close=CLOSE,
        atr=None, samples=[], h1=_drift_frame(n=30))
    ho = outlook["horizons"]["12"]
    assert ho["direction"] is None
    assert ho["point_price"] is None
    assert ho["interval"] is None
    assert "thin history" in ho["unavailable_reason"]


def test_deadband_makes_flat_reads_neutral():
    flat = []
    for i in range(40):
        flat.append(_sample(ret=1.0 if i % 2 else -1.0,
                            y_up=0, y_dn=0))
    outlook = build_outlook(
        ranks={"24": {"up": 85.0, "dn": 10.0}}, close=CLOSE,
        atr=None, samples=flat, h1=None)
    ho = outlook["horizons"]["24"]
    assert ho["direction"] == "NEUTRAL"
    assert ho["point_price"] == pytest.approx(CLOSE, abs=0.6)
    assert ho["probabilities"]["neutral"] == pytest.approx(0.0)


def test_unscored_horizon_reports_unscored():
    outlook = build_outlook(
        ranks={"24": {"up": float("nan"), "dn": float("nan")}},
        close=CLOSE, atr=None, samples=_band_samples(),
        h1=_drift_frame())
    ho = outlook["horizons"]["24"]
    assert ho["direction"] is None
    assert "unscored" in ho["unavailable_reason"]


def test_wrong_band_samples_do_not_condition():
    # Samples resolved under a LOW band must not condition a
    # current HIGH-band read — it falls back, honestly.
    low_band = _band_samples(n=40, up=20.0, dn=15.0)
    outlook = build_outlook(
        ranks={"24": {"up": 85.0, "dn": 10.0}}, close=CLOSE,
        atr=None, samples=low_band, h1=_drift_frame())
    ho = outlook["horizons"]["24"]
    assert ho["method"] == "unconditional_h1"
    assert ho["probabilities"]["state"] == "uncalibrated"


# ── resolution extension ────────────────────────────────────────


def _calm_frame(post_bars: int):
    idx = pd.date_range("2026-10-01", periods=40 + post_bars,
                        freq="h")
    rows = [(2000.0, 2005.0, 1995.0, 2000.0)] * 40
    rows += [(2000.0, 2100.0, 1950.0, 2050.0)] * post_bars
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx)


def test_realized_outcomes_captured_per_horizon():
    df = _calm_frame(30)
    rec = {"bar_time": str(df.index[39]),
           "horizons": {"12": {"up": 90.0, "dn": 10.0},
                        "24": {"up": 90.0, "dn": 10.0}}}
    out = realized_outcomes(rec, df)
    assert out["24"] == {"ret_pct": 2.5, "run_up": 100.0,
                         "run_dn": 50.0}
    assert out["12"] == {"ret_pct": 2.5, "run_up": 100.0,
                         "run_dn": 50.0}
    # resolve_record's pinned shape is untouched by the extension
    res = resolve_record(rec, df, atr_series(df).to_numpy(dtype=float))
    assert set(res["24"].keys()) == {"y_up", "y_dn"}


def test_realized_outcomes_window_open_is_absent():
    df = _calm_frame(5)
    rec = {"bar_time": str(df.index[39]),
           "horizons": {"24": {"up": 90.0, "dn": 10.0}}}
    assert realized_outcomes(rec, df) == {}


def test_state_roundtrip_carries_samples(tmp_path):
    path = tmp_path / "forecast_calibration_state.json"
    state = fresh_state()
    state["samples"] = _band_samples(n=3)
    save_state(path, state)
    loaded, quarantined = load_state(path)
    assert not quarantined
    assert loaded["samples"] == state["samples"]


# ── rendering ───────────────────────────────────────────────────


def _two_horizon_outlook():
    samples = _band_samples()
    samples += _band_samples(
        horizon=12, up=5.0, dn=88.0, rets=(-0.9,), run_up=20.0,
        run_dn=55.0, y_up_hits=0, y_dn_hits=40)
    return build_outlook(
        ranks={"12": {"up": 5.0, "dn": 88.0},
               "24": {"up": 85.0, "dn": 10.0}},
        close=CLOSE, atr=25.0, samples=samples, h1=None,
        created="2026-11-03 17:00:00",
        data_cutoff="2026-11-03 16:00:00")


def test_card_leads_with_numbers_ranks_auxiliary():
    outlook = _two_horizon_outlook()
    card = render_card(
        close=CLOSE, bar_time="2026-11-03 16:00:00", outlook=outlook,
        ranks={"12": {"up": 5.0, "dn": 88.0},
               "24": {"up": 85.0, "dn": 10.0}},
        headline=("up", 24, 85.0))
    lines = card.splitlines()
    assert lines[0].startswith("🔮 FORECAST")
    assert "new crossing: big-move rank 85 UP @1d" in lines[1]
    assert lines[2].startswith("12h DOWN →")
    assert lines[3].startswith("1d UP →")
    assert "range" in lines[3] and "P up 80%" in lines[3]
    assert "horizons split" in card
    assert "1d UP vs 12h DOWN" in card
    aux = [ln for ln in lines if ln.startswith("ranks (auxiliary)")]
    assert len(aux) == 1 and "percentile ranks" in aux[0]


def test_card_states_uncalibrated_once():
    outlook = build_outlook(
        ranks={"12": {"up": 88.0, "dn": 5.0},
               "24": {"up": 85.0, "dn": 10.0}},
        close=CLOSE, atr=None, samples=[], h1=_drift_frame())
    card = render_card(
        close=CLOSE, bar_time="2026-11-03 16:00:00", outlook=outlook,
        ranks={"12": {"up": 88.0, "dn": 5.0},
               "24": {"up": 85.0, "dn": 10.0}})
    assert card.count("probabilities uncalibrated") == 1
    assert "P up" not in card


def test_status_lines_carry_numbers():
    lines = render_status_lines(_two_horizon_outlook())
    joined = "\n".join(lines)
    assert "1d UP →" in joined and "12h DOWN →" in joined
    assert "horizons split" in joined
    assert contradiction_line(_two_horizon_outlook()["horizons"])


def test_horizon_line_unavailable_is_honest():
    line = horizon_line("3d", {"direction": None,
                               "unavailable_reason": "thin history: x"})
    assert line == "3d outlook unavailable — thin history: x"


def test_outlook_block_is_json_clean():
    outlook = _two_horizon_outlook()
    record = {"type": "forecast", "bar_time": "2026-11-03 16:00:00",
              "close": CLOSE, "outlook": outlook}
    back = json.loads(json.dumps(record))
    assert back["outlook"]["horizons"]["24"]["point_price"] == \
        outlook["horizons"]["24"]["point_price"]


# ── runner wiring ───────────────────────────────────────────────


class _Journal:
    def __init__(self):
        self.records = []
        self.notes = []

    def append(self, record):
        self.records.append(record)

    def log_note(self, text):
        self.notes.append(text)


def _bare_runner(tmp_path, *, card_enabled=True, samples=None):
    from goldenswing.execution.mt5_runner import MT5Runner

    runner = MT5Runner.__new__(MT5Runner)
    runner.cfg = SimpleNamespace(
        forecast_calibration_enabled=True,
        forecast_outlook_card_enabled=card_enabled,
        journal_path=str(tmp_path / "journal.jsonl"))
    runner.journal = _Journal()
    state = fresh_state()
    state["samples"] = samples if samples is not None \
        else _band_samples()
    runner._forecast_calibration_state = state
    runner._forecast_last_outlook = None
    return runner


def test_runner_journals_outlook_and_renders_card(tmp_path):
    runner = _bare_runner(tmp_path)
    h1 = _drift_frame()
    record = {"bar_time": str(h1.index[-1]), "close": CLOSE,
              "horizons": {"24": {"up": 85.0, "dn": 10.0}},
              "magnitude": {}, "model_version": "forecast-bundle-v2",
              "feature_set": "mtf-v1"}
    ranks = ForecastRanks(
        bar_time=h1.index[-1], close=CLOSE,
        ranks={24: {"up": 85.0, "dn": 10.0}}, horizons=(24,),
        model_probs={24: {"up": [0.8, 0.6], "dn": [0.2, 0.3]}})
    text = runner._forecast_journal(
        record, {"H1": h1}, ranks, "H1", h1.index[-1],
        "legacy rank card")
    (journaled,) = runner.journal.records
    assert journaled["type"] == "forecast"
    ho = journaled["outlook"]["horizons"]["24"]
    assert ho["direction"] == "UP"
    assert ho["method"] == "band_conditional"
    assert ho["disagreement"] is not None
    assert runner._forecast_last_outlook is journaled["outlook"]
    assert text.startswith("🔮 FORECAST")
    assert "1d UP →" in text
    # registered for calibration alongside the journal write
    state = runner._forecast_calibration_state
    assert [p["bar_time"] for p in state["pending"]] == \
        [record["bar_time"]]


def test_runner_card_flag_off_keeps_legacy_text_block_still_written(
        tmp_path):
    runner = _bare_runner(tmp_path, card_enabled=False)
    h1 = _drift_frame()
    record = {"bar_time": str(h1.index[-1]), "close": CLOSE,
              "horizons": {"24": {"up": 85.0, "dn": 10.0}},
              "magnitude": {}, "model_version": "forecast-bundle-v2",
              "feature_set": "mtf-v1"}
    ranks = ForecastRanks(
        bar_time=h1.index[-1], close=CLOSE,
        ranks={24: {"up": 85.0, "dn": 10.0}}, horizons=(24,))
    text = runner._forecast_journal(
        record, {"H1": h1}, ranks, "H1", h1.index[-1],
        "legacy rank card")
    assert text == "legacy rank card"
    (journaled,) = runner.journal.records
    assert journaled["outlook"]["horizons"]["24"]["direction"] == "UP"


def test_sample_ring_floor_is_the_stated_one():
    # The constants are the contract: estimation needs a real
    # sample, probabilities follow the E3 precedent.
    assert MIN_BAND_N >= MIN_PROB_N == 20
