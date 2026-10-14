"""Manipulation-leg / liquidity-test filter tests.

The filter is entry selection, NOT a veto: it blocks only the chase
(entry beyond a level whose most recent test failed).  Retest entries
(at the level) and opposite-direction reversals always pass.  A
confirmed break (close beyond + displacement + next bar holds) lifts
the block.
"""
from __future__ import annotations

import pandas as pd

from goldenswing.live.manipulation import chase_blocked

# Winter Monday (2026-01-05): Asia 00:00-07:00 UTC.  14 flat Asia bars
# (00:00-06:00 opens need ATR history, so the series starts 2026-01-04
# 17:00 UTC in the off session).
OFF = [(100.0, 100.5, 99.5, 100.0)] * 7
ASIA = [(101.0, 102.0, 100.0, 101.0)] * 7  # Asia range 100-102


def _df(rows, start="2026-01-04 17:00"):
    idx = pd.date_range(start, periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * len(rows)},
        index=idx,
    )


def test_asia_sweep_fail_blocks_chase_long():
    # London bar sweeps above the Asia high (102) and closes back
    # inside -> manipulation.  A long chasing past 102 is blocked.
    df = _df(OFF + ASIA + [
        (101.0, 103.5, 101.0, 101.5),   # sweep wick above 102, close inside
        (101.5, 102.0, 101.0, 101.8),   # drift
    ])
    blocked, reason = chase_blocked(df, direction=1, entry_ref=103.0,
                                     atr_value=1.0)
    assert blocked is True
    assert "manipulation" in reason


def test_asia_sweep_fail_allows_retest_and_reversal():
    df = _df(OFF + ASIA + [
        (101.0, 103.5, 101.0, 101.5),   # sweep + close back inside
        (101.5, 102.0, 101.0, 101.8),
    ])
    # Retest entry AT the level: allowed (the post-manipulation entry).
    blocked, _ = chase_blocked(df, direction=1, entry_ref=102.0,
                               atr_value=1.0)
    assert blocked is False
    # Opposite direction (reversal short): allowed.
    blocked, _ = chase_blocked(df, direction=-1, entry_ref=99.0,
                               atr_value=1.0)
    assert blocked is False


def test_confirmed_break_lifts_the_block():
    # Sweep fails, then price breaks the level with displacement and
    # the next bar holds beyond it -> the break is real, chase allowed.
    df = _df(OFF + ASIA + [
        (101.0, 103.5, 101.0, 101.5),   # sweep + close back inside
        (101.5, 102.0, 101.0, 101.8),
        (101.8, 104.0, 101.5, 103.5),   # break: close beyond + body
        (103.5, 104.5, 103.0, 104.0),   # holds beyond
    ])
    blocked, _ = chase_blocked(df, direction=1, entry_ref=104.5,
                               atr_value=1.0)
    assert blocked is False


def test_swing_low_sweep_fail_blocks_chase_short():
    # A swing low is swept and fails -> a short chasing past it blocks.
    # Build a swing low at bar 20 (confirmed by bars 21-23), then a
    # sweep bar that wicks below it and closes back above.
    rows = [(100.0, 100.5, 99.5, 100.0)] * 14 + [
        (100.0, 100.5, 99.0, 99.5),     # 14
        (99.5, 100.0, 98.0, 98.5),      # 15
        (98.5, 99.0, 97.0, 97.5),       # 16: swing low 97 (confirms @19)
        (97.5, 98.5, 97.2, 98.0),       # 17
        (98.0, 99.0, 97.5, 98.5),       # 18
        (98.5, 99.5, 98.0, 99.0),       # 19: swing low confirmed
        (99.0, 99.5, 95.5, 98.0),       # 20: sweep wick below 97, close back
        (98.0, 98.5, 97.5, 98.2),       # 21
    ]
    df = _df(rows)
    blocked, reason = chase_blocked(df, direction=-1, entry_ref=96.0,
                                     atr_value=1.0)
    assert blocked is True
    assert "manipulation" in reason
    # A long (reversal) is allowed.
    blocked, _ = chase_blocked(df, direction=1, entry_ref=99.0,
                               atr_value=1.0)
    assert blocked is False


def test_no_test_no_block():
    # No sweep, no break: clean level, chase allowed.
    df = _df(OFF + ASIA + [
        (101.0, 101.5, 100.5, 101.2),
        (101.2, 101.8, 100.8, 101.5),
    ])
    blocked, _ = chase_blocked(df, direction=1, entry_ref=103.0,
                               atr_value=1.0)
    assert blocked is False


def test_stale_sweep_fail_ages_out():
    # Sweep failed 10 bars ago (> 8-bar lookback), nothing since:
    # the block has aged out.
    df = _df(OFF + ASIA + [
        (101.0, 103.5, 101.0, 101.5),   # sweep + close back inside
    ] + [(101.5, 102.0, 101.0, 101.5)] * 10)
    blocked, _ = chase_blocked(df, direction=1, entry_ref=103.0,
                               atr_value=1.0)
    assert blocked is False
