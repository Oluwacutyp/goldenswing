"""Tests for the HARD VETO rule (user-explicit, 2026-10-13).

Main bias TF = holding-period TF. Next higher TF = hard veto power.
Veto ONLY on STRONG disagreement: opposite 2-swing structure AND both
swing legs >= 0.25 x ATR14 on the veto TF (see check_hard_veto).
Flat / agreeing / weak-opposite veto TF = no veto.  Missing veto TF =
fail-open (no veto).
"""
import numpy as np
import pandas as pd
import pytest

from goldenswing.backtest.mtf12 import build_tf_frames
from goldenswing.live.multitf import (
    PRESETS, TimeframeConfig, check_hard_veto, check_vetoes, evaluate,
    evaluate_bias,
)
from goldenswing.backtest.exness12 import run_exness_backtest
from goldenswing.backtest.real_trader import run_real_trader_backtest


def mk_bars(rows, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx)


def smooth_trend_bars(direction: int, n: int = 60):
    """Smooth trend with a wiggle every 3rd bar so fractal swings form.

    direction +1: rising closes; -1: falling closes.  Small ranges so
    ATR stays small relative to the swing legs (strong structure).
    """
    rows = []
    prev = 200.0
    for i in range(n):
        close = 200.0 + direction * 1.5 * i + (2.0 if i % 3 == 2 else 0.0)
        rows.append((prev, close + 0.8, close - 0.8, close))
        prev = close
    return mk_bars(rows)


def weak_opposite_bars():
    """Opposite bias (-1) but dust-sized swing legs (0.1) against a
    ~22 ATR: marginal wiggle, NOT strong disagreement."""
    rows = []
    for k in range(6):
        rows.append((95.0, 100.0, 90.0, 95.0))
        d = 0.1 * k
        rows.append((85.0 - d, 110.0 - d, 80.0 - d, 85.0 - d))
    return mk_bars(rows)


def flat_bars(n: int = 30):
    return mk_bars([(100.0, 100.5, 99.5, 100.0)] * n)


# ---------------------------------------------------------------- veto unit
def test_veto_blocks_strong_opposite():
    veto = smooth_trend_bars(-1)
    assert evaluate_bias(veto, swing_n=1) == -1
    v = check_hard_veto(veto, +1, swing_n=1)
    assert v["veto_bias"] == -1
    assert v["strong"] is True
    assert v["vetoed"] is True
    assert v["leg_high"] >= v["threshold"]
    assert v["leg_low"] >= v["threshold"]


def test_veto_blocks_strong_opposite_short_side():
    veto = smooth_trend_bars(+1)
    assert evaluate_bias(veto, swing_n=1) == 1
    v = check_hard_veto(veto, -1, swing_n=1)
    assert v["vetoed"] is True


def test_veto_allows_flat():
    v = check_hard_veto(flat_bars(), +1, swing_n=1)
    assert v["veto_bias"] == 0
    assert v["vetoed"] is False


def test_veto_allows_agreeing():
    veto = smooth_trend_bars(+1)
    v = check_hard_veto(veto, +1, swing_n=1)
    assert v["veto_bias"] == +1
    assert v["vetoed"] is False


def test_veto_allows_weak_opposite():
    veto = weak_opposite_bars()
    assert evaluate_bias(veto, swing_n=1) == -1  # opposite bias, but...
    v = check_hard_veto(veto, +1, swing_n=1)
    assert v["strong"] is False
    assert v["vetoed"] is False
    assert v["leg_high"] < v["threshold"]


def test_veto_no_bias_no_veto():
    veto = smooth_trend_bars(-1)
    assert check_hard_veto(veto, 0, swing_n=1)["vetoed"] is False


def test_veto_too_short_is_not_veto():
    short = mk_bars([(100.0, 101.0, 99.0, 100.0)] * 3)
    assert check_hard_veto(short, +1, swing_n=1)["vetoed"] is False


# ------------------------------------------------------- presets + evaluate
def test_preset_veto_tfs():
    assert PRESETS["swing"].veto_tf == "W1"
    assert PRESETS["swing"].veto_tf2 is None
    assert PRESETS["intraday"].veto_tf == "D1"
    assert PRESETS["intraday"].veto_tf2 is None
    assert PRESETS["scalp"].veto_tf == "H4"
    assert PRESETS["scalp"].veto_tf2 == "D1"
    assert PRESETS["scalp_m1"].veto_tf == "H1"
    assert PRESETS["scalp_m1"].veto_tf2 is None


def _signal_frames(veto_bars=None, veto_tf="W1"):
    # Reuse the known-good signal setup from test_multitf.
    from tests.test_multitf import (
        bias_bull_bars, trigger_bull_bars, zone_support_bars,
    )
    frames = {"D1": bias_bull_bars(), "H4": zone_support_bars(),
              "H1": trigger_bull_bars()}
    if veto_bars is not None:
        frames[veto_tf] = veto_bars
    return frames


CFG_VETO = TimeframeConfig("D1", "H4", "H1", swing_n=1, veto_tf="W1")


def test_evaluate_veto_blocks_signal():
    res = evaluate(_signal_frames(smooth_trend_bars(-1)), CFG_VETO)
    assert res["signal"] is False
    assert res["reason"] == "hard_veto"
    assert res["veto_bias"] == -1


def test_evaluate_veto_allows_flat_and_agreeing():
    for veto in (flat_bars(), smooth_trend_bars(+1)):
        res = evaluate(_signal_frames(veto), CFG_VETO)
        assert res["signal"] is True, veto
        assert res["reason"] == ""


def test_evaluate_veto_allows_weak_opposite():
    res = evaluate(_signal_frames(weak_opposite_bars()), CFG_VETO)
    assert res["signal"] is True
    assert res["reason"] == ""


def test_evaluate_missing_veto_tf_is_fail_open():
    res = evaluate(_signal_frames(None), CFG_VETO)
    assert res["signal"] is True
    assert res["reason"] == ""


def test_evaluate_enable_veto_false_bypasses():
    res = evaluate(_signal_frames(smooth_trend_bars(-1)), CFG_VETO,
                   enable_veto=False)
    assert res["signal"] is True


def test_evaluate_second_veto_tf_blocks():
    cfg = TimeframeConfig("D1", "H4", "H1", swing_n=1, veto_tf="W1",
                          veto_tf2="VETO2")
    frames = _signal_frames(smooth_trend_bars(+1))  # first veto agrees
    frames["VETO2"] = smooth_trend_bars(-1)  # second veto strongly opposite
    res = evaluate(frames, cfg)
    assert res["signal"] is False
    assert res["reason"] == "hard_veto"
    assert res["veto_tf"] == "VETO2"
    # first veto alone agreeing + second missing = fail-open on second
    frames2 = _signal_frames(smooth_trend_bars(+1))
    assert evaluate(frames2, cfg)["signal"] is True


def test_check_vetoes_marks_missing_unavailable():
    out = check_vetoes({"D1": flat_bars()}, CFG_VETO, +1)
    assert out["vetoed"] is False
    assert out["vetoes"]["W1"]["available"] is False


# ------------------------------------------------------------- mtf12 wiring
def test_build_tf_frames_adds_veto_w1():
    idx = pd.date_range("2026-01-01", periods=400, freq="h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                       "close": 100.0, "volume": 1}, index=idx)
    frames = build_tf_frames({"H1": df}, PRESETS["swing"])
    assert "W1" in frames
    assert len(frames["W1"]) >= 1


# ------------------------------------------- real_trader / exness integration
def _trend_frames(up: bool = True, n: int = 600, seed: int = 7):
    rng = np.random.default_rng(seed)
    drift = 0.15 if up else -0.15
    closes = 2000.0 + np.cumsum(rng.normal(drift, 1.0, n))
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": closes - rng.normal(0, 0.3, n),
        "high": closes + np.abs(rng.normal(0, 0.6, n)),
        "low": closes - np.abs(rng.normal(0, 0.6, n)),
        "close": closes}, index=idx)
    return df


def _downtrend_veto_frame(n: int = 600):
    # Same index as _trend_frames, smooth strong downtrend: sine wiggle
    # (period 20, amplitude 8) over a -0.5/bar drift.  The wiggle must
    # exceed the drift across a swing_n=3 window, or no fractal swings
    # form at all (bias 0) — a +2 wiggle against -1.5/bar drift fails.
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    rows = []
    prev = 2000.0
    for i in range(n):
        close = 2000.0 - 0.5 * i + 8.0 * float(
            np.sin(2 * np.pi * i / 20))
        rows.append((prev, close + 0.8, close - 0.8, close))
        prev = close
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx)


def test_real_trader_strong_veto_blocks_all():
    up = _trend_frames(up=True)
    veto = _downtrend_veto_frame()
    frames = {"H4": up.copy(), "H1": up.copy(), "M15": up.copy(),
              "D1": veto}
    preset = PRESETS["intraday"]
    on = run_real_trader_backtest(frames, preset, enable_veto=True)
    assert on["veto_blocked"] > 0
    # vetoed bars had main bias +1 (longs); shorts agreed with the veto
    assert on["veto_blocked_by_direction"].get("long", 0) > 0
    # shadow counterfactual: blocked trades never leak into real trades
    assert all(t["veto_blocked"] is True for t in on["blocked_trades"])
    assert all(t["direction"] == 1 for t in on["blocked_trades"])
    for t in on["blocked_trades"]:
        assert t["stop_atr"] <= 2.5 + 1e-9
        assert t["target_R"] >= 1.5 - 1e-9
    # kept trades (if any) are all shorts — every long setup was vetoed
    assert all(t["direction"] == -1 for t in on["trades"])
    off = run_real_trader_backtest(frames, preset, enable_veto=False)
    assert off["veto_blocked"] == 0
    assert off["blocked_trades"] == []
    assert off["summary"]["total_trades"] >= 0


def test_real_trader_flat_veto_changes_nothing():
    up = _trend_frames(up=True)
    flat = pd.DataFrame({"open": 2000.0, "high": 2000.5, "low": 1999.5,
                         "close": 2000.0},
                        index=up.index)
    frames = {"H4": up.copy(), "H1": up.copy(), "M15": up.copy(),
              "D1": flat}
    preset = PRESETS["intraday"]
    on = run_real_trader_backtest(frames, preset, enable_veto=True)
    off = run_real_trader_backtest(frames, preset, enable_veto=False)
    assert on["veto_blocked"] == 0
    assert on["summary"]["total_trades"] == off["summary"]["total_trades"]


def test_exness_strong_veto_blocks_all():
    up = _trend_frames(up=True)
    veto = _downtrend_veto_frame()
    frames = {"H4": up.copy(), "H1": up.copy(), "M15": up.copy(),
              "D1": veto}
    preset = PRESETS["intraday"]
    on = run_exness_backtest(frames, preset, enable_veto=True)
    assert on["veto_blocked"] > 0
    assert on["veto_blocked_by_direction"].get("long", 0) > 0
    # shadow counterfactual never touches the real balance/blowups
    assert all(t["veto_blocked"] is True for t in on["blocked_trades"])
    assert all(t["direction"] == 1 for t in on["blocked_trades"])
    assert all(t["direction"] == -1 for t in on["trades"])
    assert on["blowups"] == sum(1 for t in on["trades"]
                                if t["is_blowup"])
    assert on["blocked_blowups"] == sum(
        1 for t in on["blocked_trades"] if t["is_blowup"])
    off = run_exness_backtest(frames, preset, enable_veto=False)
    assert off["veto_blocked"] == 0
    assert off["blocked_trades"] == []


def test_exness_flat_veto_changes_nothing():
    up = _trend_frames(up=True)
    flat = pd.DataFrame({"open": 2000.0, "high": 2000.5, "low": 1999.5,
                         "close": 2000.0},
                        index=up.index)
    frames = {"H4": up.copy(), "H1": up.copy(), "M15": up.copy(),
              "D1": flat}
    preset = PRESETS["intraday"]
    on = run_exness_backtest(frames, preset, enable_veto=True)
    off = run_exness_backtest(frames, preset, enable_veto=False)
    assert on["veto_blocked"] == 0
    assert on["summary"]["total_trades"] == off["summary"]["total_trades"]
