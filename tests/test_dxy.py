"""DXY correlation as live signal context — never a gate.

The owner's bot tracks XAUUSD vs DXYm correlation (50-bar Pearson on
returns) plus the Ghost Correlation Sentinel bands (strong inverse
< -0.8 rewards, toxic positive > -0.2 penalises).  Here: correlation
is measured the same way, the Sentinel bands become a bounded ±0.5
confluence nudge, and a dead DXY feed degrades to unavailable — the
gold path never depends on it.
"""
import numpy as np
import pandas as pd
import pytest

from goldenswing.live.dxy import (
    DXYState, DXYTracker, dxy_bonus, dxy_correlation)


def _frame(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="h",
                        tz="UTC")
    return pd.DataFrame({"close": closes}, index=idx)


def _oscillating(n: int, *, mirror: bool = False, drift: float = 0.0):
    """Closes from oscillating returns: gold and its mirror have
    returns of opposite sign every bar (corr ≈ -1); same phase
    (corr ≈ +1).  A linear trend would NOT do: returns of two
    linear trends drift the same way."""
    rets = np.sin(np.arange(n) / 3.0) * 0.01 + drift
    if mirror:
        rets = -rets
    return 100.0 * np.cumprod(1.0 + rets)


def test_anticorrelated_series_measure_near_minus_one():
    gold = _frame(_oscillating(60))
    dxy = _frame(_oscillating(60, mirror=True))
    corr, status = dxy_correlation(gold, dxy)
    assert corr == pytest.approx(-1.0, abs=0.05)
    assert "measured" in status


def test_positively_correlated_series_measure_near_plus_one():
    gold = _frame(_oscillating(60))
    dxy = _frame(_oscillating(60))
    corr, _ = dxy_correlation(gold, dxy)
    assert corr == pytest.approx(1.0, abs=0.05)


def test_insufficient_bars_returns_zero_with_reason():
    corr, status = dxy_correlation(_frame([100.0] * 10),
                                   _frame([100.0] * 10))
    assert corr == 0.0
    assert "insufficient" in status


def test_misaligned_indexes_join_on_timestamp():
    gold = _frame(_oscillating(60))
    dxy = _frame(_oscillating(60, mirror=True))
    dxy = dxy.iloc[5:]  # DXY starts 5 bars later
    corr, _ = dxy_correlation(gold, dxy)
    assert corr == pytest.approx(-1.0, abs=0.05)


def test_bonus_bands_are_the_sentinel_bands():
    assert dxy_bonus(-0.9)[0] == pytest.approx(0.5)
    assert dxy_bonus(-0.1)[0] == pytest.approx(-0.5)
    assert dxy_bonus(-0.5)[0] == 0.0
    assert dxy_bonus(None)[0] == 0.0
    assert "toxic positive" in dxy_bonus(-0.1)[1]


class _Feed:
    def __init__(self, dxy=None, exc=None):
        self._dxy = dxy
        self._exc = exc

    def bars_for(self, symbol, tf, count):
        if self._exc is not None:
            raise self._exc
        assert symbol == "DXYm" and tf == "H1"
        return self._dxy


def test_tracker_measures_correlation_and_bias():
    gold = _frame(_oscillating(300, drift=0.0005))
    dxy = _frame(_oscillating(300, mirror=True, drift=0.0005))
    tracker = DXYTracker(_Feed(dxy))
    state = tracker.refresh(gold)
    assert state.available is True
    assert state.corr == pytest.approx(-1.0, abs=0.05)
    assert state.bias == -1  # DXY drifting down, below its EMA200


def test_tracker_failure_degrades_to_unavailable():
    gold = _frame([100.0 + i * 0.5 for i in range(300)])
    dxy = _frame([110.0 - i * 0.05 for i in range(300)])
    tracker = DXYTracker(_Feed(dxy))
    tracker.refresh(gold)
    assert tracker.state.available is True
    tracker._feed = _Feed(exc=RuntimeError("terminal down"))
    state = tracker.refresh(gold)
    assert state.available is False
    assert "terminal down" in state.status


def test_tracker_disabled_never_fetches():
    calls = []

    class _CountingFeed(_Feed):
        def bars_for(self, symbol, tf, count):
            calls.append(symbol)
            return super().bars_for(symbol, tf, count)

    tracker = DXYTracker(_CountingFeed(_frame([100.0] * 300)),
                         enabled=False)
    state = tracker.refresh(_frame([100.0] * 300))
    assert state.available is False
    assert calls == []


def test_dxy_bonus_flows_into_live_score():
    from goldenswing.confluence import ConfluenceEngine
    from goldenswing.golden.detector import GoldenSignal
    from goldenswing.playbooks.base import EngineContext

    sig = GoldenSignal(
        index=100, timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        direction=1, score=70.0, kind="sweep_reversal", factors=[],
        entry_zone=(1.0, 2.0), entry_ref=1.5, invalidation=1.0,
        targets=[2.5], atr=1.0)
    ctx = EngineContext(df=pd.DataFrame({"close": [1.0] * 200}),
                        states=[], years=0.0)
    eng = ConfluenceEngine(playbooks=[], risk_guards=None, journal=None)
    stub = type("PB", (), {"name": "stub", "hunting_ground": type(
        "HG", (), {"bonuses": {}})()})()

    conf, notes = eng.score_live_signal(
        stub, sig, ctx, hierarchy_verdict={"dxy_corr": -0.9})
    assert conf == pytest.approx(70.5)
    assert any("DXY" in n for n in notes)

    conf, notes = eng.score_live_signal(
        stub, sig, ctx, hierarchy_verdict={"dxy_corr": None})
    assert conf == pytest.approx(70.0)
    assert not any("DXY" in n for n in notes)


def test_verdict_carries_dxy_corr_when_available(tmp_path):
    """Runner annotates the verdict from the tracker state; None when
    DXY is unavailable (scorer treats it as 0.0)."""
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    from goldenswing.features.engine import read_and_featurize
    from goldenswing.live.hierarchy import TimeframeStack

    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, feed=None, executor=None)
    h1 = pd.DataFrame(
        {"open": [2000.0] * 400, "high": [2001.0] * 400,
         "low": [1999.0] * 400, "close": [2000.5] * 400,
         "volume": [100] * 400, "real_volume": [0] * 400,
         "spread": [30] * 400},
        index=pd.date_range("2026-01-01", periods=400, freq="h",
                            tz="UTC"))
    runner._stack = TimeframeStack.from_frames({"H1": h1})
    states = read_and_featurize(h1, None)
    runner._dxy.state = DXYState(corr=-0.9, available=True,
                                 status="measured")
    verdict = runner._verdict_for_bar(h1, len(h1) - 1, states)
    assert verdict is not None
    assert verdict["dxy_corr"] == pytest.approx(-0.9)
    runner._dxy.state = DXYState(available=False,
                                 status="fetch failed")
    verdict = runner._verdict_for_bar(h1, len(h1) - 1, states)
    assert verdict["dxy_corr"] is None
