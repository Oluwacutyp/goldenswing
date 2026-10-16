"""Tests for revived strategies (goldenswing/live/revival.py).

Every revival keeps: hard veto, sweep -> displacement -> retracement ->
LIMIT AT POI, gap skip, no-chase, Exness timing. Variants only change
the documented mechanic (POI / trigger / stop / entry wait).
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from goldenswing.backtest.exness12 import run_exness_backtest
from goldenswing.live.multitf import get_preset
from goldenswing.live.revival import (
    VARIANTS, body_stop_level, find_breaker, godce_allow, has_bodybreak,
    has_engulfing, marco_entry, ote_zone, retest_edge_limit,
)


# ------------------------------------------------------------------ OTE
def test_ote_zone_long_62_79_retracement():
    # Long leg: sweep low 4000 -> displacement high 4020. Span 20.
    # Zone: hi - 62% = 4007.6, hi - 79% = 4004.2, 70% limit = 4006.0.
    lo, hi, limit = ote_zone(4000.0, 4020.0, 1)
    assert lo == pytest.approx(4004.2)
    assert hi == pytest.approx(4007.6)
    assert limit == pytest.approx(4006.0)
    assert lo < limit < hi


def test_ote_zone_short_mirrored():
    lo, hi, limit = ote_zone(4020.0, 4000.0, -1)
    assert lo == pytest.approx(4012.4)
    assert hi == pytest.approx(4015.8)
    assert limit == pytest.approx(4014.0)


def test_ote_zone_zero_span_rejected():
    with pytest.raises(ValueError):
        ote_zone(4000.0, 4000.0, 1)


# ------------------------------------------------------------------ engulf
def test_has_engulfing_bullish():
    # Bar 1 bearish body 10..12, bar 2 bullish body 9..13 engulfs it.
    o = np.array([11.0, 9.0])
    h = np.array([13.0, 14.0])
    l = np.array([9.0, 8.0])
    c = np.array([10.0, 13.0])
    assert has_engulfing(o, h, l, c, 1, 1) is True
    assert has_engulfing(o, h, l, c, 1, -1) is False


def test_has_engulfing_no_engulf():
    o = np.array([11.0, 11.5])
    h = np.array([13.0, 13.0])
    l = np.array([9.0, 10.0])
    c = np.array([10.0, 12.0])  # body 11.5..12 does NOT engulf 10..11
    assert has_engulfing(o, h, l, c, 1, 1) is False


# ------------------------------------------------------------------ retest
def test_retest_edge_limit_long():
    limit, far = retest_edge_limit(4010.0, 4000.0, 1)
    assert limit == pytest.approx(4010.0)  # near edge = top
    assert far == pytest.approx(4000.0)    # far edge = bottom


def test_retest_edge_limit_short():
    limit, far = retest_edge_limit(4010.0, 4000.0, -1)
    assert limit == pytest.approx(4000.0)
    assert far == pytest.approx(4010.0)


# ------------------------------------------------------------------ breaker T3
def test_find_breaker_long_thesis_breaks_down():
    # Long thesis at zone 4000..4010. Bar 1 touches (low <= 4010),
    # bar 2 closes below 4000 -> breaker short at close.
    o = np.array([4005.0, 4005.0, 3995.0])
    h = np.array([4012.0, 4012.0, 4002.0])
    l = np.array([4001.0, 3998.0, 3990.0])
    c = np.array([4006.0, 4005.0, 3995.0])
    brk = find_breaker(o, h, l, c, 0, 2, 4010.0, 4000.0, 1)
    assert brk is not None
    assert brk["direction"] == -1
    assert brk["violation_idx"] == 2
    assert brk["entry"] == pytest.approx(3995.0)


def test_find_breaker_no_touch_no_breaker():
    o = np.array([4020.0, 4020.0, 4020.0])
    h = np.array([4025.0, 4025.0, 4025.0])
    l = np.array([4015.0, 4015.0, 4015.0])
    c = np.array([4020.0, 4020.0, 4020.0])
    assert find_breaker(o, h, l, c, 0, 2, 4010.0, 4000.0, 1) is None


def test_find_breaker_touch_but_no_close_through():
    o = np.array([4005.0, 4005.0, 4005.0])
    h = np.array([4012.0, 4012.0, 4012.0])
    l = np.array([4001.0, 3998.0, 4002.0])
    c = np.array([4006.0, 4005.0, 4005.0])  # never closes below 4000
    assert find_breaker(o, h, l, c, 0, 2, 4010.0, 4000.0, 1) is None


# ------------------------------------------------------------------ marco
def test_marco_entry_long():
    entry, stop = marco_entry(4001.0, 3990.0, 1, 10.0)
    assert entry == pytest.approx(4001.0)   # next-bar open
    assert stop == pytest.approx(3987.5)    # level - 0.25 ATR


def test_marco_entry_short():
    entry, stop = marco_entry(3999.0, 4010.0, -1, 10.0)
    assert entry == pytest.approx(3999.0)
    assert stop == pytest.approx(4012.5)


# ------------------------------------------------------------------ body stop / bodybreak
def test_body_stop_level_long_uses_body_not_wick():
    # Sweep bar: open 4005, close 3995 (body 3995..4005), wick low 3980.
    # Long body stop = body bottom 3995 - 0.15*10 = 3993.5, NOT 3980.
    o = np.array([4005.0, 4000.0])
    c = np.array([3995.0, 4002.0])
    stop = body_stop_level(o, c, 0, 1, 10.0)
    assert stop == pytest.approx(3993.5)


def test_has_bodybreak_long():
    # Sweep bar 0: body 3995..4005. Bar 1 closes 4006 > body top -> break.
    o = np.array([4005.0, 4000.0])
    c = np.array([3995.0, 4006.0])
    assert has_bodybreak(o, c, 0, 0, 1, 1) is True
    # Bar 1 close 4000 does NOT break through the body top.
    c2 = np.array([3995.0, 4000.0])
    assert has_bodybreak(o, c2, 0, 0, 1, 1) is False


# ------------------------------------------------------------------ GODCE filter
def test_godce_allow_bull_bias_allows_long_blocks_short():
    # Rising bias bars -> GODCE MOM mode, regime bull: long allowed,
    # short blocked. (Minimal-feat filter: only MOM mode blocks.)
    # Sawtooth + trend: confirmed rising swings (bias +1). A perfectly
    # linear series has NO confirmed swings (bias 0), so it must NOT
    # be used here — it would fall into MEAN mode, allowing both sides.
    idx = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
    _i = np.arange(60)
    close = 4000.0 + 0.8 * _i + 5.0 * np.sin(2 * np.pi * _i / 10)
    df = pd.DataFrame({
        "open": close - 1.0, "high": close + 2.0,
        "low": close - 2.0, "close": close, "volume": 10,
    }, index=idx)
    allowed_long, _ = godce_allow(1, df)
    allowed_short, _ = godce_allow(-1, df)
    assert allowed_long is True
    assert allowed_short is False


# ------------------------------------------------------------------ variant plumbing
def _flat_frames(n=400):
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 4000.0, "high": 4000.5, "low": 3999.5,
                       "close": 4000.0, "volume": 10}, index=idx)
    return {"H4": df, "H1": df, "M15": df, "D1": df, "M5": df, "M1": df}


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_flat_market_all_variants_no_trades_no_crash(variant):
    frames = _flat_frames()
    res = run_exness_backtest(frames, get_preset("intraday"),
                              start_balance=12.0, variant=variant)
    assert res["variant"] == variant
    assert res["summary"]["total_trades"] == 0
    assert res["final_balance"] == pytest.approx(12.0)
    assert res["blowups"] == 0


def test_flat_market_godce_filter_no_crash():
    frames = _flat_frames()
    res = run_exness_backtest(frames, get_preset("intraday"),
                              start_balance=12.0, godce_filter=True)
    assert res["godce_filter"] is True
    assert res["final_balance"] == pytest.approx(12.0)


def test_unknown_variant_rejected():
    frames = _flat_frames()
    with pytest.raises(ValueError):
        run_exness_backtest(frames, get_preset("intraday"),
                            start_balance=12.0, variant="nope")


def test_variant_source_keeps_no_same_bar_exit():
    # Exit scan must still start at filled_idx+1 for every variant.
    src = inspect.getsource(run_exness_backtest)
    assert "filled_idx + 1" in src


def test_runner_wires_fvg_limit_shadow_not_exec(tmp_path):
    from goldenswing.execution.runner import LiveRunner, RunnerConfig
    cfg = RunnerConfig(journal_path=tmp_path / "j.jsonl",
                       risk_state_path=tmp_path / "risk.json",
                       kill_switch_path=tmp_path / "KILL",
                       state_dir=tmp_path / "state")
    runner = LiveRunner(config=cfg, senders=[])
    names = [pb.name for pb in runner._get_playbooks()]
    assert "fvg_limit" in names
    # Auto-paper exec_set defaults must NOT include fvg_limit:
    # it stays shadow until forward paper proves it.
    import inspect as _inspect
    sig = _inspect.signature(runner.run_paper_loop)
    assert "fvg_limit" not in str(sig.parameters["execute_playbooks"].default)
