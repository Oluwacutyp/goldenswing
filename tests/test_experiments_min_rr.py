"""Tests for the minimum R:R qualification filter.

The filter is a pure function of the signal (causal by construction);
these tests pin the geometry, the boundary, the failure modes and the
monotonicity of the threshold sweep — all on synthetic signals so they
run in milliseconds.
"""
import pytest

from goldenswing.backtest.harness import ExitConfig
from goldenswing.experiments.min_rr import qualification, verdict_for
from goldenswing.golden.detector import GoldenSignal


def _sig(*, direction=1, entry_ref=2000.0, invalidation=1990.0,
         targets=(2020.0,), atr=10.0, score=65.0, index=100) -> GoldenSignal:
    return GoldenSignal(
        index=index, timestamp="t", direction=direction, score=score,
        kind="continuation", factors=[], entry_zone=(1998.0, 2002.0),
        entry_ref=entry_ref, invalidation=invalidation,
        targets=list(targets), atr=atr)


EXITS = ExitConfig()  # stop_buffer_atr=0.25, partial_min_r=1.0


def test_basic_pass():
    # long: entry 2000, inv 1990, atr 10 -> buffered stop 1987.5,
    # risk 12.5.  Pool at 2020: distance 20 >= 1R (12.5) -> rr = 1.6.
    sig = _sig()
    passes, rr = qualification(sig, 1.5, EXITS)
    assert passes is True
    assert rr == pytest.approx(1.6)


def test_boundary_passes_on_equality():
    # pool at 2025 -> distance 25, rr = 2.0 exactly -> passes at 2.0.
    sig = _sig(targets=(2025.0,))
    passes, rr = qualification(sig, 2.0, EXITS)
    assert rr == pytest.approx(2.0)
    assert passes is True
    passes2, _ = qualification(sig, 2.01, EXITS)
    assert passes2 is False


def test_too_close_pool_ignored_by_1r_rule():
    # pool at 2005 is only 5 < 1R (12.5) -> not a measurable target.
    # No other pool -> fails, rr None.
    sig = _sig(targets=(2005.0,))
    passes, rr = qualification(sig, 1.5, EXITS)
    assert passes is False
    assert rr is None


def test_no_pool_ahead_fails():
    # targets behind the entry only -> no measurable reward -> fail.
    sig = _sig(targets=(1995.0,))
    passes, rr = qualification(sig, 1.5, EXITS)
    assert passes is False
    assert rr is None
    # empty targets -> fail too.
    sig2 = _sig(targets=())
    passes2, rr2 = qualification(sig2, 1.5, EXITS)
    assert passes2 is False
    assert rr2 is None


def test_nonpositive_risk_fails():
    # invalidation on the wrong side of entry -> risk <= 0 -> fail.
    sig = _sig(entry_ref=2000.0, invalidation=2010.0)
    passes, rr = qualification(sig, 1.5, EXITS)
    assert passes is False
    assert rr is None


def test_short_direction_mirrored():
    # short: entry 2000, inv 2010, atr 10 -> buffered stop 2012.5,
    # risk 12.5.  Pool at 1975: distance 25 >= 1R -> rr = 2.0.
    sig = _sig(direction=-1, invalidation=2010.0, targets=(1975.0,))
    passes, rr = qualification(sig, 2.0, EXITS)
    assert passes is True
    assert rr == pytest.approx(2.0)
    # pool on the wrong side does not count.
    sig2 = _sig(direction=-1, invalidation=2010.0, targets=(2025.0,))
    passes2, rr2 = qualification(sig2, 1.5, EXITS)
    assert passes2 is False
    assert rr2 is None


def test_nearest_pool_used():
    # two pools: 2015 (rr=1.2) and 2030 (rr=2.4).  Nearest (>=1R) wins.
    sig = _sig(targets=(2030.0, 2015.0))
    passes, rr = qualification(sig, 1.5, EXITS)
    assert rr == pytest.approx(1.2)
    assert passes is False


def test_deterministic_and_pure():
    sig = _sig(targets=(2025.0,))
    assert qualification(sig, 2.0, EXITS) == qualification(sig, 2.0, EXITS)
    # same inputs, different object -> same decision (no hidden state).
    assert qualification(_sig(targets=(2025.0,)), 2.0, EXITS)[0] is True


def test_threshold_monotonicity():
    sigs = [_sig(targets=(t,), index=i)
            for i, t in enumerate((2013.0, 2020.0, 2025.0, 2035.0, 2050.0))]
    kept = {th: {s.index for s in sigs if qualification(s, th, EXITS)[0]}
            for th in (1.5, 2.0, 2.5, 3.0)}
    assert kept[3.0] <= kept[2.5] <= kept[2.0] <= kept[1.5]


def _book(**kw):
    base = {"trades": 100, "win_rate": 45.0, "pf": 1.20,
            "return_pct": 20.0, "max_dd_pct": 10.0, "sharpe": 0.80,
            "avg_r": 0.10, "avg_hold": 19.0}
    base.update(kw)
    return base


def test_verdict_adds():
    anchor = _book()
    res = type("R", (), {"book": _book(return_pct=22.5, sharpe=0.95)})()
    assert verdict_for(anchor, res) == "ADDS"


def test_verdict_adds_on_drawdown_improvement():
    anchor = _book()
    res = type("R", (), {
        "book": _book(return_pct=21.5, sharpe=0.80, max_dd_pct=8.5)})()
    assert verdict_for(anchor, res) == "ADDS"


def test_verdict_dilutes_even_with_higher_win_rate():
    # the rule: win rate up + return down = DILUTES, stated as such.
    anchor = _book()
    res = type("R", (), {
        "book": _book(return_pct=15.0, win_rate=60.0, sharpe=0.85)})()
    assert verdict_for(anchor, res) == "DILUTES"


def test_verdict_neutral():
    anchor = _book()
    res = type("R", (), {"book": _book(return_pct=20.5, sharpe=0.81)})()
    assert verdict_for(anchor, res) == "NEUTRAL"
