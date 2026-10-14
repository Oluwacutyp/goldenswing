"""Tests for the new MT5 playbooks: V-reversal, retracement trap,
inversion, order block, killzone open.

Each playbook gets: signal fires on the right structure, does not fire
on chop (or on the NEAR-MISS structure that belongs to a different
playbook), batch candidates == live_signal signal set (no drift), and
no repainting (a signal's facts are identical when future bars are
removed).  The execute-list / shadow assertions live in
test_mt5_playbook_expansion.py and are extended once validation lands.
"""
from __future__ import annotations

import pandas as pd
import pytest

from goldenswing.playbooks.base import EngineContext
from goldenswing.playbooks.inversion import InversionPlaybook
from goldenswing.playbooks.killzone_open import KillzoneOpenPlaybook
from goldenswing.playbooks.order_block import OrderBlockPlaybook
from goldenswing.playbooks.retracement_trap import RetracementTrapPlaybook
from goldenswing.playbooks.v_reversal import VReversalPlaybook

FLAT = [(100.0, 100.5, 99.5, 100.0)] * 20


def _df(rows, start="2026-01-01", freq="h"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * len(rows)},
        index=idx,
    )


def _ctx(df):
    return EngineContext(df=df, states=[], years=2.0, instrument="XAUUSD")


def _live_indices(pb, df):
    """live_signal driven bar-by-bar, like the runner does."""
    out = []
    for t in range(len(df)):
        for sig in pb.live_signal([], t, df):
            out.append(sig.index)
    return out


def _parity(pb_cls, df):
    """Batch candidates and incremental live_signal agree on the signal set."""
    batch = [c.signal_index for c in pb_cls().candidates(_ctx(df))]
    live = _live_indices(pb_cls(), df)
    assert batch == live, f"{pb_cls.__name__}: batch {batch} != live {live}"
    return batch


def _no_repaint(pb_cls, df, sig_index):
    """A signal's facts are identical when future bars are removed.

    Uses live_signal (which emits the signal at its bar regardless
    of what happens later) — candidates() drops unresolved trades,
    so it cannot prove no-repainting on a truncated frame.
    """
    full = {}
    pb = pb_cls()
    for t in range(len(df)):
        for sig in pb.live_signal([], t, df):
            full[sig.index] = sig
    assert sig_index in full
    trunc = df.iloc[:sig_index + 1]
    pb2 = pb_cls()
    short = {}
    for t in range(len(trunc)):
        for sig in pb2.live_signal([], t, trunc):
            short[sig.index] = sig
    assert sig_index in short
    a, b = full[sig_index], short[sig_index]
    assert a.entry_ref == b.entry_ref
    assert a.invalidation == b.invalidation
    assert a.targets == b.targets
    assert a.direction == b.direction


# ------------------------------------------------------------- V-reversal
V_BULLISH = FLAT + [
    (100.0, 100.5, 99.0, 99.0),     # 20: decline starts
    (99.0, 99.5, 98.0, 98.0),       # 21
    (98.0, 98.5, 97.0, 97.0),       # 22
    (97.0, 97.5, 96.0, 96.0),       # 23
    (96.0, 96.5, 95.0, 95.0),       # 24
    (95.0, 95.5, 90.0, 94.0),       # 25: SPIKE (fresh low, long wick)
    (94.0, 96.0, 93.5, 95.5),       # 26: RECLAIM (close > spike mid)
    (95.5, 97.0, 95.0, 96.5),       # 27
    (96.5, 98.0, 96.0, 97.5),       # 28
    (97.5, 99.0, 97.0, 98.5),       # 29
    (98.5, 100.0, 98.0, 99.5),      # 30: follow-through to the target
    (99.5, 102.0, 99.0, 101.0),     # 31
    (101.0, 104.0, 100.5, 103.0),   # 32
    (103.0, 105.5, 102.5, 105.0),   # 33: TARGET hit (104.35)
]


def test_v_reversal_bullish_fires_on_reclaim():
    df = _df(V_BULLISH)
    idx = _parity(VReversalPlaybook, df)
    assert idx == [26]
    sig = VReversalPlaybook().candidates(_ctx(df))[0].outcome.signal
    assert sig.direction == 1
    assert sig.entry_ref == pytest.approx(95.5)
    assert sig.invalidation == pytest.approx(90.0)  # spike low
    assert sig.stop if hasattr(sig, "stop") else True


def test_v_reversal_chop_never_fires():
    df = _df(FLAT + [(100.0, 100.5, 99.5, 100.0)] * 10)
    assert VReversalPlaybook().candidates(_ctx(df)) == []


def test_v_reversal_no_repaint():
    df = _df(V_BULLISH)
    _no_repaint(VReversalPlaybook, df, 26)


# ------------------------------------------------------- retracement trap
TRAP_BULLISH = FLAT + [
    (100.0, 100.5, 99.5, 100.0),    # 20
    (100.0, 100.0, 98.0, 98.5),     # 21
    (98.5, 99.0, 97.0, 97.5),       # 22
    (97.5, 98.0, 96.0, 96.5),       # 23: swing low 96 (confirms @26)
    (96.5, 97.5, 96.2, 97.0),       # 24
    (97.0, 98.0, 97.0, 97.5),       # 25
    (97.5, 98.5, 97.5, 98.0),       # 26: swing low confirmed
    (98.0, 98.5, 94.0, 94.5),       # 27: FALSE BREAK (close < 96)
    (94.5, 97.0, 94.0, 96.5),       # 28: RECLAIM (close back > 96)
    (96.5, 98.0, 96.0, 97.5),       # 29
    (97.5, 99.0, 97.0, 98.5),       # 30
    (98.5, 100.0, 98.0, 99.5),      # 31: follow-through to the target
    (99.5, 101.5, 99.0, 101.0),     # 32: TARGET hit (~100.7)
]


def test_retracement_trap_bullish_fires_on_reclaim():
    df = _df(TRAP_BULLISH)
    idx = _parity(RetracementTrapPlaybook, df)
    assert idx == [28]
    sig = RetracementTrapPlaybook().candidates(_ctx(df))[0].outcome.signal
    assert sig.direction == 1
    assert sig.entry_ref == pytest.approx(96.5)
    assert sig.invalidation == pytest.approx(94.0)  # false-break extreme


def test_retracement_trap_wick_only_is_not_a_trap():
    # A wick through the level that CLOSES back inside is a sweep,
    # not a trap: the trap playbook must stay silent.
    rows = TRAP_BULLISH[:27] + [
        (98.0, 98.5, 94.0, 97.0),   # 27: wick below 96, close inside
        (97.0, 98.0, 96.5, 97.5),   # 28
        (97.5, 99.0, 97.0, 98.5),   # 29
    ]
    df = _df(rows)
    assert RetracementTrapPlaybook().candidates(_ctx(df)) == []


def test_retracement_trap_no_repaint():
    df = _df(TRAP_BULLISH)
    _no_repaint(RetracementTrapPlaybook, df, 28)


# -------------------------------------------------------------- inversion
INV_BULLISH_BROKEN = FLAT + [
    (100.0, 100.0, 99.0, 99.5),     # 20: FVG bar 0 (high 100)
    (99.5, 101.0, 99.0, 100.0),     # 21: filler
    (100.0, 103.0, 101.0, 102.0),   # 22: FVG forms [100, 101]
    (102.0, 102.0, 98.0, 98.5),     # 23: INVERSION (close < 100)
    (98.5, 100.5, 98.0, 99.0),      # 24: RETEST (high >= 100, close < 101)
    (99.0, 99.5, 97.0, 97.5),       # 25
    (97.5, 98.0, 96.0, 96.5),       # 26
]


def test_inversion_bullish_fvg_broken_then_retest():
    df = _df(INV_BULLISH_BROKEN)
    idx = _parity(InversionPlaybook, df)
    assert idx == [24]
    sig = InversionPlaybook().candidates(_ctx(df))[0].outcome.signal
    assert sig.direction == -1  # broken support -> short the retest
    assert sig.entry_ref == pytest.approx(100.0)  # near edge (bottom)
    assert sig.invalidation == pytest.approx(101.0)  # reclaim = top


def test_inversion_holding_gap_never_inverts():
    # Gap holds (never closes through): no inversion, no signal.
    rows = FLAT + [
        (100.0, 100.0, 99.0, 99.5),
        (99.5, 101.0, 99.0, 100.0),
        (100.0, 103.0, 101.0, 102.0),   # FVG [100, 101]
        (102.0, 102.5, 100.5, 101.5),   # touches, holds
        (101.5, 102.0, 101.0, 101.5),
        (101.5, 103.0, 101.2, 102.5),
    ]
    df = _df(rows)
    assert InversionPlaybook().candidates(_ctx(df)) == []


def test_inversion_no_repaint():
    df = _df(INV_BULLISH_BROKEN)
    _no_repaint(InversionPlaybook, df, 24)


# ------------------------------------------------------------ order block
OB_BULLISH = FLAT + [
    (100.0, 101.0, 99.5, 100.5),    # 20
    (100.5, 102.0, 100.0, 101.5),   # 21
    (101.5, 103.0, 101.0, 102.5),   # 22
    (102.5, 104.0, 102.0, 103.5),   # 23: swing high 104 (confirms @26)
    (103.5, 103.5, 102.5, 103.0),   # 24 (high < 104: swing high stands)
    (103.0, 103.5, 102.0, 102.5),   # 25: DOWN bar -> OB candle [102, 103.5]
    (102.5, 103.0, 102.0, 102.5),   # 26: swing high confirmed
    (102.5, 106.0, 102.0, 105.5),   # 27: DISPLACEMENT (close > 104)
    (105.5, 105.5, 102.5, 104.0),   # 28: RETRACEMENT into OB
    (104.0, 105.0, 103.5, 104.5),   # 29
    (104.5, 106.0, 104.0, 105.5),   # 30
    (105.5, 107.0, 105.0, 106.5),   # 31: TARGET hit (~106.2)
]


def test_order_block_bullish_fires_on_retracement():
    df = _df(OB_BULLISH)
    idx = _parity(OrderBlockPlaybook, df)
    assert idx == [28]
    sig = OrderBlockPlaybook().candidates(_ctx(df))[0].outcome.signal
    assert sig.direction == 1
    assert sig.entry_ref == pytest.approx(103.5)  # OB near edge (top)
    assert sig.invalidation == pytest.approx(102.0)  # OB far edge (bottom)


def test_order_block_small_body_is_not_displacement():
    rows = OB_BULLISH[:27] + [
        (102.5, 103.5, 102.0, 103.0),   # 27: small body, no displacement
        (103.0, 103.5, 102.5, 103.0),   # 28: touches OB zone, no setup
        (103.0, 104.0, 102.5, 103.5),   # 29
    ]
    df = _df(rows)
    assert OrderBlockPlaybook().candidates(_ctx(df)) == []


def test_order_block_no_repaint():
    df = _df(OB_BULLISH)
    _no_repaint(OrderBlockPlaybook, df, 28)


# ---------------------------------------------------------- killzone open
# Winter Monday (2026-01-05): Asia 00:00-07:00 UTC, London from 07:00.
# The series starts Sunday 17:00 UTC (off session) so ATR14 is warm
# by the London open.  Indices: off 0-6, Asia 7-13, London open 14.
KZ_LONDON = ([(101.0, 102.0, 100.0, 101.0)] * 7 +   # 0-6: off session
             [(101.0, 102.0, 100.0, 101.0)] * 7 +   # 7-13: Asia range
             [
    (101.0, 103.5, 101.0, 103.0),   # 14: first London bar: BREAKOUT
    (103.0, 103.0, 101.8, 102.5),   # 15: RETEST (low touches 102, holds)
    (102.5, 104.0, 102.0, 103.5),   # 16: TARGET hit (~103.05)
    (103.5, 105.0, 103.0, 104.5),   # 17
])


def test_killzone_london_open_breakout_retest():
    df = _df(KZ_LONDON, start="2026-01-04 17:00")
    idx = _parity(KillzoneOpenPlaybook, df)
    assert idx == [15]
    sig = KillzoneOpenPlaybook().candidates(_ctx(df))[0].outcome.signal
    assert sig.direction == 1
    assert sig.entry_ref == pytest.approx(102.0)  # broken Asia high
    assert sig.invalidation == pytest.approx(101.8)  # pullback extreme


def test_killzone_no_retest_within_window():
    rows = ([(101.0, 102.0, 100.0, 101.0)] * 14 + [
        (101.0, 103.5, 101.0, 103.0),   # 14: breakout
        (103.0, 104.0, 103.0, 103.5),   # 15: no touch of 102
        (103.5, 104.5, 103.2, 104.0),   # 16
        (104.0, 105.0, 103.8, 104.5),   # 17
        (104.5, 105.5, 104.0, 105.0),   # 18
        (105.0, 106.0, 104.8, 105.5),   # 19
        (105.5, 106.5, 105.0, 106.0),   # 20: past the 6-bar window
        (106.0, 106.5, 105.5, 106.0),   # 21
    ])
    df = _df(rows, start="2026-01-04 17:00")
    assert KillzoneOpenPlaybook().candidates(_ctx(df)) == []


def test_killzone_wrong_session_rejected_by_gate():
    pb = KillzoneOpenPlaybook()
    ctx = _ctx(_df(KZ_LONDON, start="2026-01-04 17:00"))
    sig = pb.candidates(ctx)[0].outcome.signal
    assert sig.live_context["session"] == "london"
    tampered = dict(sig.live_context, session="asia")
    ok, _reason = pb.check_live_signal(sig, ctx, tampered)
    assert ok is False


def test_killzone_no_repaint():
    df = _df(KZ_LONDON, start="2026-01-04 17:00")
    _no_repaint(KillzoneOpenPlaybook, df, 15)
