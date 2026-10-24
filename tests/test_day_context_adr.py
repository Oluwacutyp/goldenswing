"""ADR gauge guard pins (day_context burn-down).

The gauge-sanity guard at the end of ``compute_day_context``
disarms a broken/stub D1 sample (``adr_days = 0``) when the ADR is
implausibly small against the H1 ATR.  It used to import ATR from
``features.indicators`` — a module that never existed — behind a
bare ``except``, so the guard silently NEVER ran: a 0.5-point stub
ADR against a real day read 5549% consumed, state ``exhausted``,
and ``adr_days = 14`` (trusted).  The decision consumers all gate
on the disarm flag (exhaustion bank: ``adr_days >= 10``; S8 and
continuation scoring: ``_adr_gauge_valid`` requires
``adr_days >= 10``), so restoring the guard restores the whole
protection chain — pinned end to end here.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from goldenswing.live import day_context as dc
from goldenswing.live.multitf import atr14


def make_h1(seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-12 18:00", periods=6 * 24, freq="h",
                        tz="America/New_York")
    walk = np.cumsum(rng.normal(0, 2.0, len(idx))) + 4000.0
    high = walk + np.abs(rng.normal(0, 3.0, len(idx))) + 2.0
    low = walk - np.abs(rng.normal(0, 3.0, len(idx))) - 2.0
    return pd.DataFrame({"open": walk, "high": high, "low": low,
                         "close": walk + rng.normal(0, 1.0, len(idx))},
                        index=idx)


def make_d1(range_pts, n=15):
    idx = pd.date_range("2026-09-25", periods=n, freq="D", tz="UTC")
    base = 4000.0
    return pd.DataFrame({"open": base, "high": base + range_pts,
                         "low": base, "close": base + range_pts / 2},
                        index=idx)


def test_guard_import_is_the_engine_atr():
    # Regression pin: the guard must import ATR from a module that
    # exists.  multitf.atr14 is the engine's own ATR, used by the
    # runner, asymmetry and hierarchy alike.
    assert dc.atr14 is atr14


def test_stub_d1_disarmed():
    # The documented P3 regression case: 0.5-point stub ADR.
    ctx = dc.compute_day_context({"H1": make_h1(), "D1": make_d1(0.5)})
    assert ctx.adr == pytest.approx(0.5)
    assert ctx.adr_days == 0


def test_real_d1_kept():
    ctx = dc.compute_day_context({"H1": make_h1(), "D1": make_d1(40.0)})
    assert ctx.adr == pytest.approx(40.0)
    assert ctx.adr_days == 14
    assert ctx.adr_consumed_pct < 150.0


def test_guard_boundary_both_sides():
    # Guard fires strictly below half the H1 ATR, keeps at/above.
    h1 = make_h1()
    a = atr14(h1)
    assert a > 0
    below = dc.compute_day_context({"H1": h1, "D1": make_d1(0.4 * a)})
    assert below.adr_days == 0
    above = dc.compute_day_context({"H1": h1, "D1": make_d1(0.6 * a)})
    assert above.adr_days == 14


def test_no_d1_fallback_still_guarded():
    # No D1 frame: ADR from grouped H1 prior days, same guard.
    ctx = dc.compute_day_context({"H1": make_h1()})
    assert ctx.adr > 0
    assert ctx.adr_days >= 3  # prior completed days sampled


def test_disarm_propagates_to_decision_gate():
    # The runner's gauge-validity gate reads only _day_ctx, so it
    # is callable unbound against a bare namespace — no runner
    # build needed.  Stub -> refused, real -> accepted.
    from goldenswing.execution.mt5_runner import MT5Runner
    stub = dc.compute_day_context({"H1": make_h1(), "D1": make_d1(0.5)})
    real = dc.compute_day_context({"H1": make_h1(), "D1": make_d1(40.0)})
    assert MT5Runner._adr_gauge_valid(SimpleNamespace(_day_ctx=stub),
                                      8.0) is False
    assert MT5Runner._adr_gauge_valid(SimpleNamespace(_day_ctx=real),
                                      8.0) is True
