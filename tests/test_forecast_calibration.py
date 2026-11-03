"""Forecast E2 — large-move objective + live calibration.

Pins: the magnitude read derived from percentile ranks (never
dressed as probabilities); rank bands; resolution of journaled
forecasts against realized bars under the label barrier race
(up win / dn win / neither / window still open / bar unknown);
the calibration book's band rates and lift; magnitude
calibration; state persistence with corrupt quarantine.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from goldenswing.forecast.calibration import (
    CalibrationBook,
    band_for_rank,
    fresh_state,
    load_state,
    magnitude_block,
    resolve_record,
    save_state,
)
from goldenswing.reader.indicators import atr as atr_series


# ── magnitude + bands ───────────────────────────────────────────


def test_magnitude_block_reads_pressure_lean_conviction():
    mag = magnitude_block({"24": {"up": 90.0, "dn": 30.0},
                           "72": {"up": 20.0, "dn": 25.0}})
    assert mag["24"] == {"move_pressure": 90.0, "lean": 60.0,
                         "conviction": 60.0, "side": "up"}
    assert mag["72"]["side"] == "dn"
    assert mag["72"]["move_pressure"] == 25.0


def test_magnitude_block_unscored_stays_visible_not_quiet():
    mag = magnitude_block({"24": {"up": float("nan"),
                                  "dn": float("nan")}})
    assert mag["24"]["move_pressure"] is None
    assert mag["24"]["side"] is None


def test_rank_bands_follow_the_alert_threshold():
    assert band_for_rank(10.0) == "0-50"
    assert band_for_rank(79.9) == "65-80"
    assert band_for_rank(80.0) == "80-90"
    assert band_for_rank(90.0) == "90+"
    assert band_for_rank(100.0) == "90+"
    assert band_for_rank(float("nan")) is None


# ── resolution against realized bars ────────────────────────────


def _frame(post: list[tuple[float, float, float]], *, bars_pre=40):
    """H1 frame: a calm pre-period (range ~10 around 2000), the
    forecast bar at position bars_pre-1... actually at index
    position 39 (close 2000), then the crafted post bars."""
    idx = pd.date_range("2026-10-01", periods=bars_pre + len(post),
                        freq="h")
    rows = []
    for i in range(bars_pre):
        c = 2000.0 + (2.0 if i % 2 else -2.0)
        rows.append((c, c + 5.0, c - 5.0, c))
    for (o, h, l, c) in post:
        rows.append((o, h, l, c))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                      index=idx)
    return df


def _atr(df):
    return atr_series(df).to_numpy(dtype=float)


def _record(df, pos, horizons=("12", "24")):
    return {"bar_time": str(df.index[pos]),
            "horizons": {h: {"up": 90.0, "dn": 10.0}
                         for h in horizons},
            "magnitude": magnitude_block(
                {h: {"up": 90.0, "dn": 10.0} for h in horizons})}


def test_resolution_up_race_wins():
    pre = _frame([])
    a = float(_atr(pre)[39])
    ref = 2000.0
    post = [
        (ref, ref + 0.8 * a, ref - 0.2 * a, ref + 0.5 * a),
        (ref, ref + 1.6 * a, ref - 0.2 * a, ref + 1.2 * a),
        (ref, ref + 2.6 * a, ref - 0.2 * a, ref + 2.2 * a),
    ] + [(ref, ref + 2.2 * a, ref + 1.8 * a, ref + 2.0 * a)] * 30
    df = _frame(post)
    out = resolve_record(_record(df, 39), df, _atr(df))
    assert out["24"] == {"y_up": 1, "y_dn": 0}
    assert out["12"] == {"y_up": 1, "y_dn": 0}


def test_resolution_dn_race_wins_mirror():
    pre = _frame([])
    a = float(_atr(pre)[39])
    ref = 2000.0
    post = [
        (ref, ref + 0.2 * a, ref - 0.8 * a, ref - 0.5 * a),
        (ref, ref + 0.2 * a, ref - 1.6 * a, ref - 1.2 * a),
        (ref, ref + 0.2 * a, ref - 2.6 * a, ref - 2.2 * a),
    ] + [(ref, ref - 1.8 * a, ref - 2.2 * a, ref - 2.0 * a)] * 30
    df = _frame(post)
    out = resolve_record(_record(df, 39), df, _atr(df))
    assert out["24"] == {"y_up": 0, "y_dn": 1}


def test_resolution_neither_barrier_resolves():
    post = [(2000.0, 2002.0, 1998.0, 2000.0)] * 30
    df = _frame(post)
    out = resolve_record(_record(df, 39), df, _atr(df))
    assert out["24"] == {"y_up": 0, "y_dn": 0}


def test_resolution_window_still_open_is_absent_not_zero():
    post = [(2000.0, 2002.0, 1998.0, 2000.0)] * 5  # only 5 bars on
    df = _frame(post)
    out = resolve_record(_record(df, 39), df, _atr(df))
    assert out == {}          # nothing resolved, nothing invented


def test_resolution_unknown_bar_returns_none():
    df = _frame([(2000.0, 2002.0, 1998.0, 2000.0)] * 30)
    rec = _record(df, 39)
    rec["bar_time"] = "2027-01-01 00:00:00"
    assert resolve_record(rec, df, _atr(df)) is None


# ── the calibration book ────────────────────────────────────────


def test_book_bands_lift_and_magnitude():
    book = CalibrationBook()
    hot = {"horizons": {"24": {"up": 95.0, "dn": 5.0}},
           "magnitude": {"24": {"move_pressure": 95.0}}}
    cold = {"horizons": {"24": {"up": 20.0, "dn": 15.0}},
            "magnitude": {"24": {"move_pressure": 20.0}}}
    for _ in range(10):
        book.add(hot, {"24": {"y_up": 1, "y_dn": 0}})
    for _ in range(10):
        book.add(cold, {"24": {"y_up": 0, "y_dn": 0}})
    view = book.view()["horizons"]["24"]
    assert view["up"]["base_rate"] == 0.5
    bands = {b["band"]: b for b in view["up"]["bands"]}
    assert bands["90+"]["rate"] == 1.0
    assert bands["90+"]["lift"] == 2.0
    assert bands["0-50"]["rate"] == 0.0
    mag = {b["band"]: b for b in view["magnitude"]["bands"]}
    assert mag["90+"]["rate"] == 1.0      # moved every time
    assert view["magnitude"]["base_rate"] == 0.5


def test_book_state_roundtrip_and_quarantine(tmp_path):
    path = tmp_path / "forecast_calibration_state.json"
    state = fresh_state()
    state["resolved_total"] = 7
    save_state(path, state)
    loaded, quarantined = load_state(path)
    assert not quarantined and loaded["resolved_total"] == 7

    path.write_text("{torn json", encoding="utf-8")
    loaded, quarantined = load_state(path)
    assert quarantined
    assert loaded == fresh_state()
    assert (tmp_path / "forecast_calibration_state.corrupt").exists()
    # and the JSON the runner persists is the book's plain data
    book = CalibrationBook(loaded.get("book") or {})
    assert json.dumps(book.data) is not None
