"""Tests for the POI confirmation check (full confirmation family
at the POI) and its wiring.

Two wirings (since the 2026-10-15 trigger redesign):
- Hierarchy backtest: SHADOW / log-only — it annotates the trade
  book (``candle_confirmed`` stats) without changing decisions.
- Live runner: the family IS the trigger — a confirming shape at
  the POI FIRES (the pending is consumed, the signal path runs).
"""
import math

import pandas as pd
import pytest

from goldenswing.backtest.mtf_hierarchy import run_hierarchy_backtest
from goldenswing.entry.validator import (
    POIConfirmation, SHAPE_PRIORITY, confirm_at_poi)
from goldenswing.live.hierarchy import TimeframeStack

from test_mtf_hierarchy import _synthetic_frames
from test_mt5_runner import _runner


def _bars(rows, start="2026-01-01 00:00"):
    """OHLC frame from (open, high, low, close) rows, 15m apart."""
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx)


def _bars_vol(rows, start="2026-01-01 00:00"):
    """OHLCV frame from (open, high, low, close, volume) rows."""
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close", "volume"],
        index=idx)


# ---------------------------------------------------------------- unit
def test_long_rejection_confirms():
    # POI [100, 102]: wick dips to 99 (below bottom), closes 101.5
    # back above the edge, bullish, lower wick (2.5) >= body (0.5).
    frame = _bars([(101.0, 102.0, 99.0, 101.5)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert ann.confirmed and ann.shape == "rejection"
    assert ann.bar_index == frame.index[0]


def test_short_rejection_confirms():
    # POI [100, 102]: wick pokes to 103, closes 100.5 back below
    # the top edge, bearish, upper wick (2.0) >= body (0.5).
    frame = _bars([(101.0, 103.0, 100.0, 100.5)])
    ann = confirm_at_poi(frame, -1, 100.0, 102.0)
    assert ann.confirmed and ann.shape == "rejection"


def test_rejection_wrong_direction_does_not_confirm():
    # A bullish rejection bar does NOT confirm a short.
    frame = _bars([(101.0, 102.0, 99.0, 101.5)])
    assert confirm_at_poi(frame, -1, 100.0, 102.0).confirmed is False


def test_rejection_away_from_poi_does_not_confirm():
    # Same shape, but the POI is far above the bar.
    frame = _bars([(101.0, 102.0, 99.0, 101.5)])
    assert confirm_at_poi(frame, 1, 200.0, 202.0).confirmed is False


def test_rejection_close_not_back_does_not_confirm():
    # Wick penetrates but the close stays below the defended edge.
    frame = _bars([(101.0, 102.0, 99.0, 99.5)])
    assert confirm_at_poi(frame, 1, 100.0, 102.0).confirmed is False


def test_rejection_wick_smaller_than_body_does_not_confirm():
    # Penetrating bar, but lower wick 1.0 < body 2.0: a strong bar,
    # not a rejection; and no engulfing (no prior bar) -> none.
    frame = _bars([(99.0, 102.0, 98.0, 101.0)])
    assert confirm_at_poi(frame, 1, 100.0, 102.0).confirmed is False


def test_rejection_wick_exactly_on_edge_counts():
    # Low lands exactly on the defended edge: penetration is <=.
    frame = _bars([(101.0, 102.0, 100.0, 101.5)])
    assert confirm_at_poi(frame, 1, 100.0, 102.0).confirmed is True


def test_long_engulfing_confirms():
    # Prior bearish (100 -> 98); current (97.5 -> 101) covers it,
    # and overlaps the POI [100, 102].
    frame = _bars([(100.0, 100.5, 97.5, 98.0),
                   (97.5, 101.5, 97.0, 101.0)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert ann.confirmed and ann.shape == "engulfing"
    assert ann.bar_index == frame.index[1]


def test_short_engulfing_confirms():
    # Prior bullish (98 -> 100); current (100.5 -> 97) covers it.
    frame = _bars([(98.0, 100.5, 97.5, 100.0),
                   (100.5, 101.0, 96.5, 97.0)])
    ann = confirm_at_poi(frame, -1, 100.0, 102.0)
    assert ann.confirmed and ann.shape == "engulfing"


def test_engulfing_away_from_poi_does_not_confirm():
    frame = _bars([(100.0, 100.5, 97.5, 98.0),
                   (97.5, 101.5, 97.0, 101.0)])
    assert confirm_at_poi(frame, 1, 200.0, 202.0).confirmed is False


def test_engulfing_doji_prior_no_engulfing_but_sweep_reclaim():
    # Prior bar is a doji (open == close): no body to engulf — so
    # no engulfing.  But the full family sees more: bar 0 sweeps
    # below the POI bottom (99.5 < 100) and bar 1 reclaims it
    # (close 101.0 > 100, bullish) — a genuine sweep_reclaim.
    frame = _bars([(100.0, 100.5, 99.5, 100.0),
                   (99.0, 101.5, 98.5, 101.0)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert ann.confirmed is True
    assert "engulfing" not in ann.shapes
    assert "sweep_reclaim" in ann.shapes


def test_engulfing_equal_bodies_counts():
    # Current body exactly equals the prior body (inclusive cover).
    frame = _bars([(100.0, 100.5, 97.5, 98.0),
                   (98.0, 101.5, 97.5, 100.0)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert ann.confirmed and ann.shape == "engulfing"


def test_engulfing_partial_cover_no_engulfing_but_tweezer():
    # Current close (99.5) does not reach the prior open (100):
    # no engulfing.  But the two bars share an exact low (97.5),
    # first bearish then bullish, both at the POI — a genuine
    # tweezer bottom.
    frame = _bars([(100.0, 100.5, 97.5, 98.0),
                   (98.0, 101.5, 97.5, 99.5)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert ann.confirmed is True
    assert "engulfing" not in ann.shapes
    assert "tweezer_bottom" in ann.shapes


def test_doji_current_bar_never_confirms():
    frame = _bars([(101.0, 102.0, 99.0, 101.0),
                   (100.0, 100.5, 99.5, 100.0)])
    assert confirm_at_poi(frame, 1, 100.0, 102.0).confirmed is False
    assert confirm_at_poi(frame, -1, 100.0, 102.0).confirmed is False


def test_first_confirming_bar_wins():
    rows = [
        (101.0, 102.0, 100.5, 101.2),   # no: no penetration, no prior
        (101.0, 102.0, 99.0, 101.5),    # rejection
        (100.0, 100.5, 97.5, 98.0),     # prior for a later engulfing
        (97.5, 101.5, 97.0, 101.0),     # engulfing — but later
    ]
    frame = _bars(rows)
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert ann.confirmed and ann.shape == "rejection"
    assert ann.bar_index == frame.index[1]


def test_empty_frame_confirms_nothing():
    frame = _bars([])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert ann.confirmed is False and ann.shape == "none"


def test_nan_bar_skipped():
    rows = [(101.0, 102.0, 99.0, 101.5)]
    frame = _bars(rows)
    nan_row = pd.DataFrame(
        [(float("nan"),) * 4], columns=["open", "high", "low", "close"],
        index=[frame.index[0] - pd.Timedelta(minutes=15)])
    frame = pd.concat([nan_row, frame])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert ann.confirmed and ann.shape == "rejection"
    assert ann.bar_index == frame.index[1]


def test_bad_contract_raises():
    frame = _bars([(101.0, 102.0, 99.0, 101.5)])
    with pytest.raises(ValueError):
        confirm_at_poi(frame, 0, 100.0, 102.0)
    with pytest.raises(ValueError):
        confirm_at_poi(frame, 1, 102.0, 100.0)


def test_poi_confirmation_is_a_dataclass():
    ann = POIConfirmation(True, "rejection", "idx", "detail")
    assert ann.confirmed and ann.shape == "rejection"


# ------------------------------------------------------- shadow wiring
def test_backtest_trades_carry_candle_annotation():
    frames = _synthetic_frames()
    result = run_hierarchy_backtest(frames, swing_n=1, max_bars=120)
    assert "candle_confirmed" in result
    assert "candle_confirmed_trades" in result
    for trade in result["trades"]:
        assert "candle_confirmed" in trade
        assert trade["candle_shape"] in SHAPE_PRIORITY + ("none",)
        assert "candle_shapes" in trade
        assert all(s in SHAPE_PRIORITY for s in trade["candle_shapes"])
        if trade["candle_confirmed"]:
            assert trade["candle_shapes"]
            assert trade["candle_shape"] in trade["candle_shapes"]
    assert result["candle_confirmed_trades"] == sum(
        1 for t in result["trades"] if t["candle_confirmed"])


def test_runner_pending_fires_on_rejection_confirm(tmp_path):
    # Trigger redesign (2026-10-15): the confirmation family IS
    # the trigger — a confirming shape at the POI FIRES (the
    # pending is consumed and the signal path runs).  The old
    # annotate-without-deciding machinery is gone.
    runner = _runner(tmp_path)
    # M15 bars: gentle bars, then a long rejection bar at the POI.
    rows = [(100.5, 101.0, 100.0, 100.6),
            (100.6, 101.0, 100.1, 100.7),
            (100.7, 101.0, 100.2, 100.8),
            (100.8, 101.2, 100.3, 100.9),
            (100.9, 101.2, 100.4, 101.0),
            (101.0, 101.2, 100.5, 101.1),
            (101.1, 101.5, 99.0, 101.4)]  # rejection at POI [100, 102]
    frame = _bars(rows, start="2026-01-01 00:00")
    h1_open = frame.index[0]
    runner._stack = TimeframeStack.from_frames({"M15": frame})
    setup = {"direction": 1,
             "zone": {"bottom": 100.0, "top": 102.0, "kind": "support",
                      "level": 101.0},
             "buf": 0.1, "h4_highs": [103.0], "h4_lows": [99.0]}
    pend = {"armed": True, "trigger": "zone_touch", "direction": 1,
            "entry": 101.0, "zone": setup["zone"],
            "entry_clock": "M15", "h1_open": h1_open,
            "invalidation_level": 99.9, "setup": setup,
            "armed_bars": 0}
    runner._pending = pend
    fired = []
    runner._run_live_signal = lambda trig, tick, st: fired.append(trig)
    runner._evaluate_pending_entry(
        {"time": h1_open.to_pydatetime(), "mid": 101.0}, {})
    assert [f["trigger"] for f in fired] == ["rejection"]
    assert "rejection" in fired[0]["confirm_shapes"]
    assert runner._pending is None  # consumed by the confirmation


def test_runner_ignores_unarmed_pending(tmp_path):
    # Only ARMED touches await confirmation: a pending that is
    # not armed is ignored by the confirm shot, untouched.
    runner = _runner(tmp_path)
    frame = _bars([(101.0, 102.0, 99.0, 101.5)])
    runner._stack = TimeframeStack.from_frames({"M15": frame})
    pend = {"direction": 1, "h1_open": frame.index[0],
            "entry_clock": "M15", "trigger": "test"}
    runner._pending = pend
    runner._evaluate_pending_entry({}, {})
    assert runner._pending is pend


# ================================================================ full family
# --- pin bar ---
def test_pin_bar_long_confirms():
    # Wick 3.0 >= 2x body 0.5, >= 2/3 range 4.0; close in upper half.
    frame = _bars([(101.0, 102.0, 98.0, 101.5)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "pin_bar" in ann.shapes


def test_pin_bar_short_confirms():
    frame = _bars([(101.0, 104.0, 100.0, 100.5)])
    ann = confirm_at_poi(frame, -1, 100.0, 102.0)
    assert "pin_bar" in ann.shapes


def test_pin_bar_away_from_poi_does_not_confirm():
    frame = _bars([(101.0, 102.0, 98.0, 101.5)])
    assert confirm_at_poi(frame, 1, 200.0, 202.0).confirmed is False


def test_pin_bar_doji_never_confirms():
    # Body 0: a doji with a long wick is not a pin bar.
    frame = _bars([(100.0, 102.0, 98.0, 100.0)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "pin_bar" not in ann.shapes
    assert ann.confirmed is False


# --- morning / evening star ---
def test_morning_star_confirms():
    frame = _bars([(102.0, 102.5, 100.0, 100.5),
                   (100.5, 101.0, 100.0, 100.7),
                   (100.7, 102.0, 100.5, 102.0)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "morning_star" in ann.shapes
    assert ann.bar_indices["morning_star"] == frame.index[2]


def test_evening_star_confirms():
    frame = _bars([(98.0, 100.0, 97.5, 99.5),
                   (99.5, 100.0, 99.0, 99.3),
                   (99.3, 100.0, 98.0, 98.0)])
    ann = confirm_at_poi(frame, -1, 100.0, 102.0)
    assert "evening_star" in ann.shapes


def test_star_away_from_poi_does_not_confirm():
    frame = _bars([(102.0, 102.5, 100.0, 100.5),
                   (100.5, 101.0, 100.0, 100.7),
                   (100.7, 102.0, 100.5, 102.0)])
    assert confirm_at_poi(frame, 1, 200.0, 202.0).confirmed is False


def test_star_middle_bar_too_big_does_not_confirm():
    # Middle bar body 1.3 > 1/2 of the first bar's body 1.5? No:
    # 1.3 > 0.75, so no star.  (Lows stay >= POI bottom: no sweep,
    # third bar does not break the mother: no inside breakout.)
    frame = _bars([(102.0, 102.5, 100.0, 100.5),
                   (100.5, 102.0, 100.2, 101.8),
                   (101.8, 102.5, 100.5, 102.0)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "morning_star" not in ann.shapes
    assert ann.confirmed is False


# --- tweezer ---
def test_tweezer_bottom_near_equal_lows_confirms():
    frame = _bars([(100.0, 100.5, 97.5, 98.0),
                   (98.0, 101.5, 97.6, 99.5)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "tweezer_bottom" in ann.shapes


def test_tweezer_top_confirms():
    frame = _bars([(98.0, 102.0, 97.5, 100.0),
                   (100.0, 102.1, 98.5, 99.0)])
    ann = confirm_at_poi(frame, -1, 100.0, 102.0)
    assert "tweezer_top" in ann.shapes


def test_tweezer_unequal_extremes_does_not_confirm():
    # Lows differ by 1.0 > tolerance 0.3: not a tweezer, and the
    # second bar does not reclaim the POI: no sweep_reclaim either.
    frame = _bars([(100.0, 100.5, 97.5, 98.0),
                   (98.0, 101.5, 98.5, 99.5)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "tweezer_bottom" not in ann.shapes
    assert ann.confirmed is False


# --- inside-bar breakout ---
def test_inside_bar_breakout_long_confirms():
    frame = _bars([(100.0, 103.0, 99.0, 101.0),
                   (100.5, 102.0, 100.0, 101.5),
                   (101.5, 104.0, 101.0, 103.5)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "inside_bar_breakout" in ann.shapes
    assert ann.bar_indices["inside_bar_breakout"] == frame.index[2]


def test_inside_bar_breakout_short_confirms():
    frame = _bars([(102.0, 105.0, 99.0, 103.0),
                   (102.5, 104.0, 100.0, 102.0),
                   (102.0, 103.0, 98.0, 98.5)])
    ann = confirm_at_poi(frame, -1, 100.0, 102.0)
    assert "inside_bar_breakout" in ann.shapes


def test_inside_bar_no_breakout_does_not_confirm():
    # Third bar closes inside the mother (102.0 < 103): no breakout.
    # Lows stay >= POI bottom (no sweep); third bar is bearish
    # (no soldiers).
    frame = _bars([(100.5, 103.0, 100.2, 101.0),
                   (100.8, 102.0, 100.5, 101.5),
                   (101.5, 102.5, 101.0, 101.2)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "inside_bar_breakout" not in ann.shapes
    assert ann.confirmed is False


# --- three soldiers / three crows ---
def test_three_soldiers_confirms():
    frame = _bars([(100.0, 101.0, 99.5, 100.8),
                   (100.4, 101.5, 100.0, 101.2),
                   (100.9, 102.0, 100.5, 101.8)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "three_soldiers" in ann.shapes
    assert ann.bar_indices["three_soldiers"] == frame.index[2]


def test_three_crows_confirms():
    frame = _bars([(102.0, 102.5, 101.0, 101.2),
                   (101.6, 102.0, 100.5, 100.8),
                   (101.1, 101.5, 100.0, 100.2)])
    ann = confirm_at_poi(frame, -1, 100.0, 102.0)
    assert "three_crows" in ann.shapes


def test_soldiers_open_outside_prior_body_does_not_confirm():
    # Second bar opens at 101.5, outside the first bar's body
    # [100.0, 100.8]: no soldiers.  Lows stay >= POI bottom, so
    # no sweep_reclaim either.
    frame = _bars([(100.2, 101.0, 100.0, 100.8),
                   (101.5, 102.0, 101.0, 101.8),
                   (102.5, 103.0, 102.0, 102.8)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "three_soldiers" not in ann.shapes
    assert ann.confirmed is False


# --- BOS / CHoCH ---
def _bos_long_ctx():
    rows = [
        (100.0, 101.0, 99.0, 100.5),
        (100.5, 102.0, 100.0, 101.5),
        (101.5, 103.0, 101.0, 102.5),
        (102.5, 104.0, 102.0, 103.5),
        (103.5, 105.0, 103.0, 104.5),
        (104.5, 106.0, 104.0, 105.5),   # swing high 106
        (105.5, 105.5, 103.0, 104.0),
        (104.0, 104.5, 102.0, 103.0),
        (103.0, 103.5, 101.0, 102.0),
        (102.0, 102.5, 100.0, 101.0),
        (101.0, 101.5, 99.5, 100.5),
        (100.5, 101.0, 99.0, 100.0),
        (100.0, 107.0, 99.5, 106.5),    # break above 106
    ]
    return _bars(rows)


def test_bos_long_confirms_with_context():
    ctx = _bos_long_ctx()
    frame = ctx.iloc[9:]
    ann = confirm_at_poi(frame, 1, 100.0, 102.0, context=ctx)
    assert "bos" in ann.shapes
    assert "choch" not in ann.shapes  # only one swing high: no trend
    assert ann.bar_indices["bos"] == frame.index[-1]


def test_bos_short_confirms_with_context():
    rows = [
        (104.0, 105.0, 103.0, 103.5),
        (103.5, 104.0, 102.0, 102.5),
        (102.5, 103.0, 101.0, 101.5),
        (101.5, 102.0, 100.0, 100.5),
        (100.5, 101.0, 99.0, 99.5),
        (99.5, 100.0, 98.0, 98.5),     # swing low 98
        (98.5, 100.0, 99.0, 99.5),
        (99.5, 101.0, 100.0, 100.5),
        (100.5, 102.0, 101.0, 101.5),
        (101.5, 103.0, 101.0, 102.0),
        (102.0, 104.0, 101.5, 103.0),
        (103.0, 105.0, 102.5, 104.0),
        (104.0, 105.5, 97.0, 97.5),    # break below 98
    ]
    ctx = _bars(rows)
    frame = ctx.iloc[9:]
    ann = confirm_at_poi(frame, -1, 100.0, 102.0, context=ctx)
    assert "bos" in ann.shapes
    assert "choch" not in ann.shapes


def test_bos_no_break_does_not_confirm():
    ctx = _bos_long_ctx()
    frame = ctx.iloc[9:12]  # ends before the break bar
    ann = confirm_at_poi(frame, 1, 100.0, 102.0, context=ctx)
    assert "bos" not in ann.shapes
    assert ann.confirmed is False


def test_bos_needs_context_too_few_bars():
    # The 4-bar frame alone has no confirmable swing: no BOS
    # without the longer context (engulfing/sweep still fire).
    ctx = _bos_long_ctx()
    frame = ctx.iloc[9:]
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "bos" not in ann.shapes
    assert ann.confirmed is True  # other shapes still confirm


def test_choch_long_confirms_countertrend_break():
    rows = [
        (108.0, 110.0, 106.0, 107.0),
        (107.0, 109.0, 105.0, 106.0),
        (106.0, 108.0, 104.0, 105.0),
        (105.0, 111.0, 104.0, 110.0),   # swing high 111
        (110.0, 104.0, 102.0, 103.0),
        (103.0, 103.0, 101.0, 102.0),
        (102.0, 102.0, 100.0, 101.0),
        (101.0, 106.0, 100.5, 105.0),   # swing high 106 (lower)
        (105.0, 105.0, 103.0, 104.0),
        (104.0, 104.0, 102.0, 103.0),
        (103.0, 103.0, 101.0, 102.0),
        (102.0, 102.5, 99.0, 100.0),
        (100.0, 101.0, 98.5, 99.5),
        (99.5, 100.0, 98.0, 99.0),      # swing low 98
        (99.0, 100.0, 99.0, 99.5),
        (99.5, 101.0, 99.2, 100.5),
        (100.5, 102.0, 99.5, 101.5),
        (101.5, 108.0, 101.0, 107.0),   # break above 106, downtrend
    ]
    ctx = _bars(rows)
    frame = ctx.iloc[14:]
    ann = confirm_at_poi(frame, 1, 100.0, 102.0, context=ctx)
    assert "choch" in ann.shapes
    assert "bos" not in ann.shapes  # counter-trend break is CHoCH
    assert ann.bar_indices["choch"] == frame.index[-1]


def test_bos_uptrend_break_is_not_choch():
    rows = [
        (100.0, 101.0, 99.0, 100.2),
        (100.2, 102.0, 99.5, 101.0),
        (101.0, 103.0, 100.0, 102.0),
        (102.0, 104.0, 101.0, 103.0),   # swing high 104
        (103.0, 103.5, 101.5, 102.0),
        (102.0, 102.5, 100.5, 101.0),
        (101.0, 101.5, 99.5, 100.0),   # swing low 99.5
        (100.0, 105.0, 100.5, 104.0),  # swing high 105 (higher)
        (104.0, 104.5, 102.0, 103.0),
        (103.0, 103.5, 101.0, 102.0),
        (102.0, 102.5, 101.5, 101.5),
        (101.0, 101.5, 100.0, 100.5),  # swing low 100 (higher)
        (100.5, 102.0, 100.2, 101.5),
        (101.5, 103.0, 100.8, 102.5),
        (102.5, 107.0, 102.0, 106.5),  # break above 105, uptrend
    ]
    ctx = _bars(rows)
    frame = ctx.iloc[12:]
    ann = confirm_at_poi(frame, 1, 100.0, 102.0, context=ctx)
    assert "bos" in ann.shapes
    assert "choch" not in ann.shapes  # with-trend break is BOS


# --- failed retest ---
def test_failed_retest_long_confirms():
    frame = _bars([(101.0, 102.0, 99.5, 101.5),
                   (101.5, 103.0, 101.5, 102.5),
                   (102.5, 102.8, 100.5, 101.8)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "failed_retest" in ann.shapes
    assert ann.bar_indices["failed_retest"] == frame.index[2]


def test_failed_retest_short_confirms():
    frame = _bars([(101.0, 102.5, 100.5, 101.5),
                   (101.5, 102.0, 99.5, 99.8),
                   (99.8, 101.5, 99.0, 100.5)])
    ann = confirm_at_poi(frame, -1, 100.0, 102.0)
    assert "failed_retest" in ann.shapes


def test_failed_retest_through_close_kills_pattern():
    # Touch, leave, then a close THROUGH the far edge, then a
    # retest-shaped bar: the through killed the pattern, so the
    # last bar is a fresh touch, not a retest.
    frame = _bars([(101.0, 102.0, 99.5, 101.5),
                   (101.5, 103.0, 101.5, 102.5),
                   (102.5, 103.0, 99.0, 99.5),
                   (99.5, 101.0, 99.5, 101.0)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "failed_retest" not in ann.shapes


def test_failed_retest_rearms_after_through():
    # Touch, through (kills), touch again, leave, retest: confirms.
    frame = _bars([(101.0, 102.0, 99.5, 101.5),
                   (101.5, 103.0, 99.0, 99.5),
                   (99.5, 101.0, 99.8, 100.8),
                   (100.8, 103.0, 101.0, 102.5),
                   (102.5, 102.8, 100.5, 101.8)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "failed_retest" in ann.shapes
    assert ann.bar_indices["failed_retest"] == frame.index[4]


# --- HH+HL / LL+LH ---
def test_hh_hl_confirms():
    rows = [
        (100.0, 101.0, 99.0, 100.5),
        (100.5, 102.0, 100.0, 101.5),
        (101.5, 103.0, 101.0, 102.5),
        (102.5, 103.0, 100.5, 101.5),   # touch
        (101.5, 102.0, 100.0, 101.0),
        (101.0, 101.5, 99.5, 100.5),
        (100.5, 104.0, 100.0, 103.5),   # swing high 104
        (103.5, 103.0, 101.0, 102.0),
        (102.0, 102.5, 100.5, 101.5),
        (101.5, 101.0, 99.0, 100.0),    # swing low 99
        (100.0, 105.0, 99.5, 104.5),    # swing high 105 (higher)
        (104.5, 104.0, 102.0, 103.0),
        (103.0, 103.5, 101.0, 102.0),
        (102.0, 102.5, 100.5, 101.5),
        (101.5, 102.0, 100.0, 101.0),   # swing low 100 (higher)
        (101.0, 101.5, 100.2, 101.2),
        (101.2, 102.0, 100.5, 101.8),
        (101.8, 102.5, 101.0, 102.0),   # confirms the HL swing
    ]
    ctx = _bars(rows)
    frame = ctx.iloc[14:]
    ann = confirm_at_poi(frame, 1, 100.0, 102.0, context=ctx)
    assert "hh_hl" in ann.shapes
    assert ann.bar_indices["hh_hl"] == frame.index[-1]


def test_ll_lh_confirms():
    rows = [
        (102.0, 103.0, 101.0, 101.5),
        (101.5, 102.0, 100.5, 101.0),
        (101.0, 101.5, 99.5, 100.5),
        (100.5, 102.0, 99.0, 101.0),    # touch
        (101.0, 102.0, 100.5, 101.5),
        (101.5, 102.5, 101.0, 102.0),   # swing high 102.5
        (102.0, 102.0, 98.0, 98.5),     # swing low 98
        (98.5, 100.0, 99.0, 99.5),
        (99.5, 101.0, 100.0, 100.5),
        (100.5, 102.0, 101.0, 101.5),
        (101.5, 102.0, 97.0, 97.5),     # swing low 97 (lower)
        (97.5, 99.0, 98.0, 98.5),
        (98.5, 100.0, 99.0, 99.5),
        (99.5, 100.5, 100.0, 100.5),
        (100.5, 101.0, 99.5, 100.0),    # swing high 101 (lower)
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.8, 99.8, 100.2),
        (100.2, 100.9, 100.0, 100.5),   # confirms the LH swing
    ]
    ctx = _bars(rows)
    frame = ctx.iloc[14:]
    ann = confirm_at_poi(frame, -1, 100.0, 102.0, context=ctx)
    assert "ll_lh" in ann.shapes
    assert ann.bar_indices["ll_lh"] == frame.index[-1]


# --- displacement ---
def _flat_bars(n, a, b):
    """Alternating small bars (never inside each other)."""
    rows = [a, b] * ((n + 1) // 2)
    return rows[:n]


def test_displacement_long_confirms():
    rows = _flat_bars(15, (100.0, 101.0, 100.0, 100.5),
                      (100.5, 101.5, 100.2, 101.0))
    rows.append((100.5, 103.0, 100.2, 102.8))
    frame = _bars(rows)
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert ann.shapes == ["displacement"]


def test_displacement_short_confirms():
    rows = _flat_bars(15, (101.0, 102.0, 101.0, 101.5),
                      (100.8, 101.8, 100.5, 101.0))
    rows.append((101.5, 101.8, 99.0, 99.5))
    frame = _bars(rows)
    ann = confirm_at_poi(frame, -1, 100.0, 102.0)
    assert "displacement" in ann.shapes


def test_displacement_small_body_does_not_confirm():
    rows = _flat_bars(15, (100.0, 101.0, 100.0, 100.5),
                      (100.5, 101.5, 100.2, 101.0))
    # Body 0.4 << ATR ~1.3, though it closes beyond the POI top.
    rows.append((101.8, 102.5, 101.5, 102.2))
    frame = _bars(rows)
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "displacement" not in ann.shapes
    assert ann.confirmed is False


# --- volume spike ---
def test_volume_spike_confirms():
    rows = [(101.0, 102.0, 100.5, 101.5, 100.0)] * 6
    rows.append((101.5, 102.5, 100.5, 102.0, 250.0))
    frame = _bars_vol(rows)
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert ann.shapes == ["volume_spike"]


def test_volume_spike_needs_volume_column():
    rows = [(101.0, 102.0, 100.5, 101.5)] * 6
    rows.append((101.5, 102.5, 100.5, 102.0))
    frame = _bars(rows)
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "volume_spike" not in ann.shapes
    assert ann.confirmed is False


def test_volume_spike_needs_five_prior_bars():
    rows = [(101.0, 102.0, 100.5, 101.5, 100.0)] * 3
    rows.append((101.5, 102.5, 100.5, 102.0, 250.0))
    frame = _bars_vol(rows)
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "volume_spike" not in ann.shapes
    assert ann.confirmed is False


# --- sweep + reclaim ---
def test_sweep_reclaim_long_confirms():
    frame = _bars([(101.0, 102.0, 99.0, 99.5),
                   (99.5, 101.5, 99.2, 101.0)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "sweep_reclaim" in ann.shapes
    assert ann.bar_indices["sweep_reclaim"] == frame.index[1]


def test_sweep_reclaim_short_confirms():
    frame = _bars([(101.0, 103.0, 100.0, 102.5),
                   (102.5, 102.8, 100.5, 100.5)])
    ann = confirm_at_poi(frame, -1, 100.0, 102.0)
    assert "sweep_reclaim" in ann.shapes


def test_sweep_reclaim_too_late_does_not_confirm():
    # Sweep at bar 0.  Bars 1-5 never sweep again (lows >= 100)
    # and never reclaim (closes <= 100); the reclaim lands 6
    # bars after the sweep — outside the 5-bar window.  Bars are
    # shaped so no other family member fires either.
    frame = _bars([(101.0, 102.0, 99.0, 99.5),
                   (99.5, 101.0, 100.0, 99.8),
                   (99.8, 101.5, 100.1, 99.9),
                   (99.9, 101.2, 100.0, 99.7),
                   (99.8, 101.8, 100.2, 99.9),
                   (99.9, 101.4, 100.0, 99.5),
                   (100.0, 101.5, 100.5, 101.0)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "sweep_reclaim" not in ann.shapes
    assert ann.confirmed is False


# --- FVG tap + reaction ---
def test_fvg_reaction_long_confirms():
    frame = _bars([(100.0, 101.0, 99.5, 100.5),
                   (100.5, 101.5, 100.2, 101.0),
                   (101.0, 102.5, 101.8, 102.0),
                   (102.0, 102.5, 101.2, 102.2)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "fvg_reaction" in ann.shapes
    assert ann.bar_indices["fvg_reaction"] == frame.index[3]


def test_fvg_reaction_short_confirms():
    frame = _bars([(101.5, 102.5, 101.0, 101.8),
                   (101.8, 102.0, 101.2, 101.5),
                   (101.5, 100.8, 99.5, 100.0),
                   (100.0, 100.9, 99.5, 99.8)])
    ann = confirm_at_poi(frame, -1, 100.0, 102.0)
    assert "fvg_reaction" in ann.shapes


def test_fvg_gap_not_at_poi_does_not_confirm():
    frame = _bars([(100.0, 101.0, 99.5, 100.5),
                   (100.5, 101.5, 100.2, 101.0),
                   (101.0, 102.5, 101.8, 102.0),
                   (102.0, 102.5, 101.2, 102.2)])
    assert confirm_at_poi(frame, 1, 200.0, 202.0).confirmed is False


def test_fvg_stale_gap_does_not_confirm():
    # Gap [101.0, 101.8] forms at bars 0-2.  A bearish bridge bar
    # (close exactly on the gap bottom) avoids tapping it and
    # avoids forming a fresh gap; the 99 identical filler bars
    # stay above the gap and form no gaps; the tap lands 101 bars
    # after formation — past the 100-bar staleness cap.
    # (Fillers may trigger other shapes; only the FVG shape is
    # asserted absent here.)
    rows = [(100.0, 101.0, 99.5, 100.5),
            (100.5, 101.5, 100.2, 101.0),
            (101.0, 102.5, 101.8, 102.0),
            (102.0, 102.5, 101.2, 101.0)]
    rows += [(102.0, 102.5, 102.0, 102.2)] * 99
    rows.append((101.5, 102.5, 101.2, 102.2))
    frame = _bars(rows)
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert "fvg_reaction" not in ann.shapes


# --- result shape: all shapes, first shape, context dependence ---
def test_one_bar_can_confirm_via_several_shapes():
    frame = _bars([(101.0, 102.0, 98.0, 101.5)])
    ann = confirm_at_poi(frame, 1, 100.0, 102.0)
    assert ann.confirmed is True
    assert ann.shapes == ["rejection", "pin_bar"]
    assert ann.shape == "rejection"  # priority order breaks the tie
    assert ann.bar_indices["rejection"] == ann.bar_indices["pin_bar"]
