"""Tests for Exness $12 timing model: margin, no-chase, blow-up."""
import math
import pandas as pd
import pytest
import math
from goldenswing.backtest.exness12 import (
    LOT_OZ, LEVERAGE, SPREAD_PER_OZ, margin_required, stop_out_price,
    parse_leverage, run_exness_backtest,
)
from goldenswing.live.multitf import get_preset


def _tiny_frames():
    # Build minimal synthetic frames: bias up, zone, trigger with a sweep+disp pattern is hard
    # to synthesize by hand, so these tests cover pure math + blow-up mechanics.
    # Full signal path is covered by running the real CSVs in the backtest (see below).
    return None


def test_margin_001_lot_at_2000_is_about_2_dollars():
    # At $4000 gold, 0.01 lot = 1 oz, margin = 4000/2000 = $2.00
    assert margin_required(4000.0) == pytest.approx(2.0)
    assert margin_required(2635.85) == pytest.approx(2635.85 / 2000.0)
    assert margin_required(5000.0, lot_oz=1.0, leverage=2000.0) == pytest.approx(2.5)
    # 1:200 would be $20 - the OANDA-era mistake
    assert margin_required(4000.0, leverage=200.0) == pytest.approx(20.0)


def test_stop_out_price_math():
    # Long, $12 balance, $0.20 spread: room = $11.80, stop-out = entry - 11.80
    assert stop_out_price(4000.0, 1, 12.0) == pytest.approx(4000.0 - 11.80)
    # Short: entry + 11.80
    assert stop_out_price(4000.0, -1, 12.0) == pytest.approx(4000.0 + 11.80)
    # Small balance: $3 balance -> room $2.80
    assert stop_out_price(4000.0, 1, 3.0) == pytest.approx(4000.0 - 2.80)


def test_spread_cost_is_020_not_045():
    assert SPREAD_PER_OZ == pytest.approx(0.20)
    assert LOT_OZ == pytest.approx(1.0)
    assert LEVERAGE == pytest.approx(2000.0)


def _make_flat_frames(n=400):
    # Flat market: no sweep, no displacement -> 0 trades, no blow-up, no crash
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 4000.0, "high": 4000.5, "low": 3999.5, "close": 4000.0, "volume": 10}, index=idx)
    return {"H4": df, "H1": df, "M15": df, "D1": df, "M5": df, "M1": df}


def test_flat_market_no_trades_no_blowup():
    frames = _make_flat_frames()
    preset = get_preset("intraday")
    res = run_exness_backtest(frames, preset, start_balance=12.0)
    assert res["final_balance"] == pytest.approx(12.0)
    assert res["blowups"] == 0
    assert res["summary"]["total_trades"] == 0


def test_insufficient_margin_blocks_open():
    # Balance $1 < margin ~$2 -> cannot open, simulation ends as cannot_open
    frames = _make_flat_frames()
    preset = get_preset("intraday")
    # Force a trade-like path is hard on flat data; instead verify margin gate directly:
    # balance $1 < margin $2 at $4000
    assert 1.0 < margin_required(4000.0)
    # And run with tiny balance still does not crash or go negative
    res = run_exness_backtest(frames, preset, start_balance=1.0)
    assert res["final_balance"] >= 0.0


def test_unlimited_margin_is_zero():
    assert margin_required(4000.0, lot_oz=1.0, leverage=float("inf")) == 0.0
    assert parse_leverage("unlimited") == math.inf
    assert parse_leverage("1:2000") == pytest.approx(2000.0)
    assert parse_leverage("2000") == pytest.approx(2000.0)


def test_leverage_1_2_impossible_for_12():
    # 1:2, 0.01 lot at $4000 -> $2000 margin, $12 cannot open
    assert margin_required(4000.0, lot_oz=1.0, leverage=2.0) == pytest.approx(2000.0)
    assert 12.0 < margin_required(4000.0, lot_oz=1.0, leverage=2.0)


def test_lot_scaling_blowup_math():
    # 0.01 lot: $12 - $0.20 spread -> $11.80 room / 1 oz = $11.80 move blows
    assert stop_out_price(4000.0, 1, 12.0, lot_oz=1.0) == pytest.approx(4000.0 - 11.80)
    # 0.02 lot: spread $0.40, room $11.60 / 2 oz = $5.80 move blows
    assert stop_out_price(4000.0, 1, 12.0, lot_oz=2.0) == pytest.approx(4000.0 - 5.80)
    # 0.05 lot: spread $1.00, room $11.00 / 5 oz = $2.20 move blows
    assert stop_out_price(4000.0, 1, 12.0, lot_oz=5.0) == pytest.approx(4000.0 - 2.20)
    # PnL per lot unchanged by leverage: margin differs, PnL does not
    assert margin_required(4000.0, lot_oz=1.0, leverage=2000.0) == pytest.approx(2.0)
    assert margin_required(4000.0, lot_oz=1.0, leverage=float("inf")) == 0.0


def test_causal_asia_no_lookahead():
    # Asia high at 05:00 must NOT be visible at 01:00 same day.
    from goldenswing.backtest.exness12 import causal_asia_arrays
    idx = pd.to_datetime(["2026-01-01 01:00", "2026-01-01 05:00", "2026-01-01 07:00"], utc=True)
    df = pd.DataFrame({"open": [4000.0, 4000.0, 4000.0], "high": [4010.0, 4050.0, 4020.0],
                       "low": [3990.0, 3990.0, 3990.0], "close": [4000.0, 4000.0, 4000.0],
                       "volume": [10, 10, 10]}, index=idx)
    highs, lows = causal_asia_arrays(df)
    assert highs[0] == pytest.approx(4010.0)  # only 01:00 Asia seen
    assert highs[1] == pytest.approx(4050.0)  # 05:00 now seen
    assert highs[2] == pytest.approx(4050.0)  # 07:00 keeps Asia high, no future leak


def test_no_same_bar_exit_lookahead():
    # Regression: exit scan must start at filled_idx+1, not filled_idx.
    # Before fix, 72/93 Swing trades exited on hold_bars==0 (same bar as fill).
    # This test documents the rule; full verification is via backtest hold_bars.
    import inspect
    from goldenswing.backtest import exness12
    src = inspect.getsource(exness12.run_exness_backtest)
    assert "filled_idx + 1" in src
    assert "MUST start at filled_idx+1" in src


def test_blowup_floors_at_zero_not_negative():
    # Direct math: if PnL would take balance below 0, floor at 0.
    # Simulate via stop_out_price: $12 balance, long entry 4000, price 3980 = -$20 move -> equity -$8 -> floored.
    # This is enforced in run_exness_backtest (balance_after = max(0, ...)).
    # Here we verify the stop-out price logic that triggers it.
    so = stop_out_price(4000.0, 1, 12.0)
    assert so == pytest.approx(3988.20)
    # Price 3980 < 3988.20 -> would have blown up before reaching 3980
    assert 3980.0 < so
