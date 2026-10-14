"""Tests for failure-mode diagnosis (direction vs timing).

Net post-stop drift classifies: a stop followed by a sustained rally
in the signal direction is TIMING; a stop followed by price staying
adverse is DIRECTION; chop is NEITHER.  The classifier is symmetric
under the null -- a random walk after the stop has no preferred side.
"""
import numpy as np
import pandas as pd

from goldenswing.experiments.failure_modes import (
    FailureKind, classify, diagnose_trades, verdict,
)


def _df(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": np.full(n, 100.0),
        "high": np.full(n, 100.2),
        "low": np.full(n, 99.8),
        "close": np.full(n, 100.0),
    }, index=idx)


def _trade(strategy="s", signal_index=10, direction=1, entry=100.0,
           stop=99.0, exit_index=40, exit_reason="stop", r_net=-1.0,
           partial_taken=False) -> dict:
    risk = (entry - stop) * direction
    return {"strategy": strategy, "signal_index": signal_index,
            "direction": direction, "entry": entry, "stop": stop,
            "risk": risk, "exit_index": exit_index,
            "exit_reason": exit_reason, "r_net": r_net,
            "partial_taken": partial_taken}


def test_classify_timing():
    assert classify(1.5) == FailureKind.TIMING
    assert classify(1.0) == FailureKind.TIMING


def test_classify_direction():
    assert classify(-1.5) == FailureKind.DIRECTION
    assert classify(-1.0) == FailureKind.DIRECTION


def test_classify_neither_chop():
    assert classify(0.5) == FailureKind.NEITHER
    assert classify(-0.5) == FailureKind.NEITHER
    assert classify(0.0) == FailureKind.NEITHER


def test_classify_symmetric_boundaries():
    # symmetric thresholds: no built-in bias toward either verdict
    assert classify(0.99) == FailureKind.NEITHER
    assert classify(-0.99) == FailureKind.NEITHER


def _set_post(df, lo, hi, high=None, low=None, close=None):
    idx = df.index[lo:hi + 1]
    if high is not None:
        df.loc[idx, "high"] = high
    if low is not None:
        df.loc[idx, "low"] = low
    if close is not None:
        df.loc[idx, "close"] = close


def test_timing_long_stop_then_sustained_rally():
    df = _df()
    # stop at 99.0 on bar 40; price rallies and CLOSES at 100.5 by bar 64.
    _set_post(df, 41, 64, high=100.6, low=99.5, close=100.5)
    recs = diagnose_trades(df, [_trade(exit_index=40)])
    assert len(recs) == 1
    assert recs[0].kind == FailureKind.TIMING
    assert recs[0].drift_24 == abs((100.5 - 99.0) / 1.0)


def test_wick_recovery_without_close_is_not_timing():
    # wicks back 1.5R but closes flat: not a sustained rally -> NEITHER
    df = _df()
    _set_post(df, 41, 64, high=100.5, low=98.5, close=99.0)
    recs = diagnose_trades(df, [_trade(exit_index=40)])
    assert recs[0].kind == FailureKind.NEITHER
    assert recs[0].max_recovery_24 == abs((100.5 - 99.0) / 1.0)


def test_direction_long_stop_then_stays_adverse():
    df = _df()
    _set_post(df, 41, 64, high=98.9, low=97.5, close=98.0)
    recs = diagnose_trades(df, [_trade(exit_index=40)])
    assert len(recs) == 1
    assert recs[0].kind == FailureKind.DIRECTION


def test_neither_chop_after_stop():
    df = _df()
    _set_post(df, 41, 64, high=99.4, low=98.6, close=99.0)
    recs = diagnose_trades(df, [_trade(exit_index=40)])
    assert len(recs) == 1
    assert recs[0].kind == FailureKind.NEITHER


def test_short_direction_mirrored():
    df = _df()
    # short stopped at 101; price keeps rallying, closes 102.0.
    _set_post(df, 41, 64, high=102.1, low=101.0, close=102.0)
    t = _trade(direction=-1, entry=100.0, stop=101.0, exit_index=40)
    recs = diagnose_trades(df, [t])
    assert recs[0].kind == FailureKind.DIRECTION


def test_short_timing_mirrored():
    df = _df()
    # short stopped at 101; price collapses, closes 99.5 (favorable).
    _set_post(df, 41, 64, high=100.9, low=99.4, close=99.5)
    t = _trade(direction=-1, entry=100.0, stop=101.0, exit_index=40)
    recs = diagnose_trades(df, [t])
    assert recs[0].kind == FailureKind.TIMING


def test_insufficient_post_data():
    df = _df(n=45)
    recs = diagnose_trades(df, [_trade(exit_index=40)])
    assert recs[0].kind == FailureKind.INSUFFICIENT


def test_exclusions_not_diagnosed():
    df = _df()
    _set_post(df, 41, 64, high=100.6, low=99.5, close=100.5)
    trades = [
        _trade(exit_reason="timeout"),                 # not a stop
        _trade(exit_reason="stop", r_net=0.5),          # winner
        _trade(exit_reason="stop", partial_taken=True), # banked partial
        _trade(exit_reason="be"),                       # breakeven scratch
        _trade(exit_reason="trail"),                    # trailed stop
    ]
    assert diagnose_trades(df, trades) == []


def test_half_assignment():
    df = _df(n=120)  # half = 60
    _set_post(df, 41, 64, high=100.6, low=99.5, close=100.5)
    _set_post(df, 101, 119, high=100.6, low=99.5, close=100.5)
    early = _trade(signal_index=10, exit_index=40)
    late = _trade(signal_index=70, exit_index=100)
    recs = diagnose_trades(df, [early, late])
    by_sig = {r.signal_index: r for r in recs}
    assert by_sig[10].half == "H1"
    assert by_sig[70].half == "H2"


def test_verdict_thresholds():
    assert verdict({"direction": 7, "timing": 2, "neither": 1,
                    "insufficient": 0}) == "flip-candidate"
    assert verdict({"direction": 1, "timing": 8, "neither": 1,
                    "insufficient": 0}) == "refine-entry"
    assert verdict({"direction": 4, "timing": 4, "neither": 4,
                    "insufficient": 0}) == "stays-buried"
    assert verdict({"direction": 0, "timing": 0, "neither": 0,
                    "insufficient": 5}) == "no-data"


def test_session_tagged():
    df = _df()
    _set_post(df, 41, 64, high=100.6, low=99.5, close=100.5)
    recs = diagnose_trades(df, [_trade(exit_index=40, signal_index=10)])
    # 2024-01-01 10:00 UTC -> london session per sessions.py
    assert recs[0].session == "london"
