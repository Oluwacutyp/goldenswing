"""MacroTrend alignment: EMA200 bias, ladder, Golden Swing flag.

Every assertion comes from the code's actual behavior (verified
interactively before writing): EMA200 = last close vs EWMA
(span=200, adjust=False); <200 bars -> NEUTRAL; Golden Swing =
D1/W1/MN1 EMA biases all equal and nonzero.
"""
import numpy as np
import pandas as pd
import pytest

from goldenswing.live.macrotrend import (
    MacroTrendResult, compute_macrotrend, ema_bias, structure_bias)


def _frame(closes, start="2020-01-01", freq="D"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c + 1, "low": c - 1, "close": c,
         "volume": np.ones(len(c))},
        index=idx)


def _rising(n=250):
    return _frame(np.linspace(100, 200, n))


def _falling(n=250):
    return _frame(np.linspace(200, 100, n))


def _zigzag_up(n_waves=25):
    closes = []
    for k in range(n_waves):
        base = 100 + k * 4
        closes += [base, base + 2, base + 4, base + 3, base + 1,
                   base + 3, base + 5, base + 4, base + 2, base + 4]
    return _frame(closes)


def test_ema_bias_direction_and_flat():
    assert ema_bias(_rising()) == 1
    assert ema_bias(_falling()) == -1
    assert ema_bias(_frame(np.full(250, 150.0))) == 0


def test_ema_bias_short_frame_is_neutral():
    assert ema_bias(_rising(50)) == 0
    assert ema_bias(_rising(199)) == 0
    assert ema_bias(None) == 0
    assert ema_bias(_rising(0)) == 0


def test_structure_bias_zigzag():
    assert structure_bias(_zigzag_up()) == 1
    # Smooth trend has no confirmed swings -> flat structure.
    assert structure_bias(_rising()) == 0
    assert structure_bias(None) == 0


def test_golden_swing_all_agree():
    r = compute_macrotrend({"D1": _rising(), "W1": _rising(),
                            "MN1": _rising()})
    assert r.golden_swing is True
    assert r.alignment_level == 1.5
    assert r.alignment_label == "Golden_Swing"
    assert r.size_multiplier == 1.5
    assert r.conflict is False
    assert (r.d1_ema, r.w1_ema, r.mn1_ema) == (1, 1, 1)


def test_golden_swing_all_agree_bearish():
    r = compute_macrotrend({"D1": _falling(), "W1": _falling(),
                            "MN1": _falling()})
    assert r.golden_swing is True
    assert r.size_multiplier == 1.5
    assert (r.d1_ema, r.w1_ema, r.mn1_ema) == (-1, -1, -1)


def test_d1_w1_only_ladder():
    r = compute_macrotrend({"D1": _rising(), "W1": _rising(),
                            "MN1": _falling()})
    assert r.golden_swing is False
    assert r.alignment_level == 1.25
    assert r.alignment_label == "D1_W1"
    assert r.size_multiplier == 1.0


def test_conflict_halves_size():
    r = compute_macrotrend({"D1": _rising(), "W1": _falling(),
                            "MN1": _rising()})
    assert r.conflict is True
    assert r.golden_swing is False
    assert r.size_multiplier == 0.5


def test_missing_frame_is_not_golden():
    r = compute_macrotrend({"D1": _rising(), "W1": _rising()})
    assert r.golden_swing is False
    assert r.alignment_level == 1.25  # MN1 neutral: D1/W1 rung
    r2 = compute_macrotrend({})
    assert r2.alignment_level == 1.0
    assert r2.size_multiplier == 1.0
    assert r2.golden_swing is False


def test_structure_confirms_needs_all_three():
    zz = _zigzag_up()
    r = compute_macrotrend({"D1": zz, "W1": zz, "MN1": zz})
    assert r.structure_confirms is True
    assert r.golden_swing is True
    # Smooth frames: structure flat (0) disagrees with EMA (+1).
    r2 = compute_macrotrend({"D1": _rising(), "W1": _rising(),
                             "MN1": _rising()})
    assert r2.structure_confirms is False


def test_describe_contains_label_and_structure():
    r = compute_macrotrend({"D1": _zigzag_up(), "W1": _zigzag_up(),
                            "MN1": _zigzag_up()})
    text = r.describe()
    assert "Golden_Swing" in text
    assert "structure agrees" in text
    assert "D1 +1" in text


def test_result_is_frozen():
    r = compute_macrotrend({})
    assert isinstance(r, MacroTrendResult)
    with pytest.raises(Exception):
        r.golden_swing = True  # frozen dataclass: no mutation
