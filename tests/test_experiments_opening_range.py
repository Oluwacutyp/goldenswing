"""QQQ opening-range experiment: session logic, fills, gates, causality."""
import numpy as np
import pandas as pd
import pytest

from goldenswing.experiments.opening_range import (
    COST_PER_SHARE,
    FIXED_TP_R,
    STOP_BUFFER_OR,
    _daily_bias,
    _level_target,
    _pdh_pdl,
    find_retest_fill,
    has_fvg,
    opening_range,
    regular_sessions,
    scan_day,
    simulate_trade,
)


# ------------------------------------------------------------------ helpers
def _day(rows, date="2026-10-05", freq="5min"):
    """rows: list of (o,h,l,c) -> ET session frame with pos/ts columns."""
    idx = pd.date_range(f"{date} 09:30", periods=len(rows), freq=freq,
                        tz="America/New_York")
    df = pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * len(rows)},
        index=idx.tz_convert("UTC"),
    )
    df["pos"] = np.arange(len(df))
    df["ts"] = df.index
    return df


def _grid(n, o=100.0, h=101.0, l=99.0, c=100.0):
    return [(o, h, l, c)] * n


def _scan(rows, **kw):
    day = _day(rows)
    or_hi, or_lo, first_sig = opening_range(day, kw.get("grid", "5m"))
    kw.setdefault("prev_or_widths", [2.0] * 20)
    kw.setdefault("bias", None)
    kw.setdefault("use_gate", False)
    kw.setdefault("use_bias", False)
    return scan_day(day, or_hi, or_lo, or_hi - or_lo,
                    kw.get("grid", "5m"), first_sig, **kw)


# ------------------------------------------------------------------ OR/scan
def test_or_is_first_5m_bar():
    day = _day([(100, 102, 99, 101)] + _grid(10))
    hi, lo, first = opening_range(day, "5m")
    assert (hi, lo, first) == (102.0, 99.0, 1)


def test_or_m1_is_first_five_bars():
    rows = [(100, 101, 99, 100), (100, 103, 98, 101),
            (101, 102, 100, 101), (101, 101, 100, 100),
            (100, 100, 97, 98)] + _grid(10)
    hi, lo, first = opening_range(_day(rows, freq="1min"), "1m")
    assert (hi, lo, first) == (103.0, 97.0, 5)


def test_break_needs_close_through():
    # Wick through OR_hi (102) but close back inside -> no setup.
    rows = [(100, 102, 99, 101), (101, 103, 100, 101.5)] + _grid(10)
    assert _scan(rows) == []
    # Close through -> long setup.
    rows = [(100, 102, 99, 101), (101, 103, 100, 102.5)] + _grid(10)
    got = _scan(rows)
    assert len(got) == 1 and got[0].direction == 1
    assert got[0].break_pos == 1


def test_first_break_wins_and_mirrors_short():
    rows = [(100, 102, 99, 101), (101, 101, 98, 98.5)] + _grid(10)
    got = _scan(rows)
    assert len(got) == 1 and got[0].direction == -1


# ------------------------------------------------------------------ retest
def _setup(rows, **kw):
    got = _scan(rows, **kw)
    assert got, "expected a setup"
    return got[0]


def _full(rows):
    """Wrap day rows into a full frame (pos 0..n-1)."""
    return _day(rows)


def test_retest_fills_on_wick_touch():
    rows = [(100, 102, 99, 101),      # OR
            (101, 103, 100, 102.5),   # break close through 102
            (102.5, 103, 102.2, 102.8),
            (102.8, 103, 101.9, 102.6)]  # retest touch of 102
    s = _setup(rows)
    fl = find_retest_fill(_full(rows), s, use_fvg=False)
    assert fl is not None and fl.fill_pos == 3
    assert fl.entry == pytest.approx(102.0)


def test_retest_expires():
    rows = [(100, 102, 99, 101), (101, 103, 100, 102.5)] + \
        _grid(20, o=103, h=104, l=102.5, c=103)
    s = _setup(rows)
    assert find_retest_fill(_full(rows), s, use_fvg=False) is None


def test_gap_through_fill_and_stop_is_same_bar_stop():
    rows = [(100, 102, 99, 101),
            (101, 103, 100, 102.5),
            (102.5, 103, 100.5, 101.0)]  # opens 102.5, trades thru stop 101
    s = _setup(rows)
    fl = find_retest_fill(_full(rows), s, use_fvg=False)
    assert fl is not None and fl.same_bar_stop


def test_break_failure_close_back_through():
    rows = [(100, 102, 99, 101),
            (101, 103, 100, 102.5),
            (102.5, 102.8, 101.9, 101.5)]  # touches 102 then closes below
    s = _setup(rows)
    fl = find_retest_fill(_full(rows), s, use_fvg=False)
    assert fl is not None and fl.same_bar_stop


# ------------------------------------------------------------------ FVG
def test_has_fvg_three_candle_rule():
    high = np.array([100., 101., 102., 103.])
    low = np.array([99., 100., 102.5, 102.8])  # low[2] > high[0]: gap
    assert has_fvg(high, low, 3, 1)
    assert not has_fvg(high, low, 3, -1)
    low2 = np.array([99., 100., 99.5, 99.8])
    assert not has_fvg(high, low2, 3, 1)


def test_fvg_variant_passes_day_without_gap():
    rows = [(100, 102, 99, 101), (101, 102.5, 100, 102.2),
            (102.2, 102.8, 101.8, 102.5),
            (102.5, 103, 101.9, 102.6)]
    s = _setup(rows)
    # Tight overlap: no 3-candle gap anywhere in the window.
    assert find_retest_fill(_full(rows), s, use_fvg=True) is None
    fl = find_retest_fill(_full(rows), s, use_fvg=False)
    assert fl is not None and fl.fill_pos == 2


# ------------------------------------------------------------------ gates
def test_range_gate_skips_narrow_days():
    rows = [(100, 100.4, 99.8, 100.1),   # OR width 0.6
            (100.1, 101, 100, 100.8)] + _grid(10)
    got = _scan(rows, use_gate=True, prev_or_widths=[2.0] * 20)
    assert got and not got[0].gate_ok
    got = _scan(rows, use_gate=False, prev_or_widths=[2.0] * 20)
    assert got and got[0].gate_ok


def test_bias_filter_blocks_opposed_direction():
    rows = [(100, 102, 99, 101), (101, 103, 100, 102.5)] + _grid(10)
    got = _scan(rows, use_bias=True, bias=-1)
    assert got and not got[0].bias_ok
    got = _scan(rows, use_bias=True, bias=1)
    assert got and got[0].bias_ok


def test_daily_bias_uses_completed_bars_only():
    idx = pd.date_range("2026-09-01", periods=30, freq="1D", tz="UTC")
    closes = np.linspace(100, 130, 30)  # strong uptrend
    daily = pd.DataFrame(
        {"open": closes - 1, "high": closes + 1, "low": closes - 2,
         "close": closes, "volume": 1.0}, index=idx)
    assert _daily_bias(daily, idx[20].date()) == 1
    # Too-short history -> None (no lookahead fabrication).
    assert _daily_bias(daily.iloc[:10], idx[9].date()) is None


def test_pdh_pdl_from_prior_completed_day():
    idx = pd.date_range("2026-09-01", periods=5, freq="1D", tz="UTC")
    daily = pd.DataFrame(
        {"open": 100.0, "high": [110., 111., 112., 113., 114.],
         "low": [90., 91., 92., 93., 94.], "close": 100.0,
         "volume": 1.0}, index=idx)
    et_day = idx[3].tz_convert("America/New_York").date()
    pdh, pdl = _pdh_pdl(daily, et_day)
    assert (pdh, pdl) == (112.0, 92.0)


def test_level_target_skipped_when_no_room():
    s = _setup([(100, 102, 99, 101), (101, 103, 100, 102.5)] + _grid(10))
    # PDH only 0.5R above entry -> skipped.
    assert _level_target(s, 102.0, pdh=102.25, pdl=90.0) is None
    assert _level_target(s, 102.0, pdh=104.0, pdl=90.0) == 104.0


# ------------------------------------------------------------------ trade
def test_stop_wins_ties_and_cost_is_per_share():
    rows = [(100, 102, 99, 101),
            (101, 103, 100, 102.5),
            (102.5, 102.6, 101.9, 102.4),   # retest touch at 102
            (102.4, 105.5, 100.4, 103.0)]   # hits stop 101 AND 2R 105
    df = _full(rows)
    s = _setup(rows)
    fl = find_retest_fill(df, s, use_fvg=False)
    assert fl is not None and not fl.same_bar_stop
    risk = STOP_BUFFER_OR * s.or_width          # 0.5 * 3 = 1.5
    o = simulate_trade(df, len(df) - 1, fl, 102.0 + FIXED_TP_R * risk)
    assert o.exit_reason == "stop"
    assert o.r_net == pytest.approx(-1.0 - COST_PER_SHARE / risk)


def test_timeout_exits_at_session_close():
    rows = [(100, 102, 99, 101),
            (101, 103, 100, 102.5),
            (102.5, 102.6, 101.9, 102.4),
            (102.4, 102.8, 102.0, 102.6),
            (102.6, 103.0, 102.2, 102.9)]
    df = _full(rows)
    s = _setup(rows)
    fl = find_retest_fill(df, s, use_fvg=False)
    risk = STOP_BUFFER_OR * s.or_width
    o = simulate_trade(df, len(df) - 1, fl, 102.0 + FIXED_TP_R * risk)
    assert o.exit_reason == "timeout"
    assert o.exit_index == len(df) - 1


# ------------------------------------------------------------------ sessions
def test_regular_sessions_split_days_and_map_pos():
    idx = pd.DatetimeIndex(list(
        pd.date_range("2026-10-05 09:30", periods=4, freq="5min",
                      tz="America/New_York").tz_convert("UTC")) + list(
        pd.date_range("2026-10-06 09:30", periods=4, freq="5min",
                      tz="America/New_York").tz_convert("UTC")))
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                       "volume": 1.0}, index=idx)
    out = regular_sessions(df)
    assert len(out) == 2
    assert list(out[0][1]["pos"]) == [0, 1, 2, 3]
    assert list(out[1][1]["pos"]) == [4, 5, 6, 7]
    assert out[0][1]["ts"].iloc[0].tz_convert(
        "America/New_York").hour == 9
