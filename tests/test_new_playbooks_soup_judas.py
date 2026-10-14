"""Tests for Turtle Soup + Judas Swing playbooks.

Each playbook gets: signal fires on the right structure, does not fire
on chop / near-miss structures (close-beyond = real break, no trigger,
no confirmation, no retest within the armed window, retest that does
not hold), batch candidates == live_signal signal set (no drift), no
repainting, frozen levels survive a state round-trip, invalidation
exits, and the fade entry passes the manipulation filter (a chase
entry is still blocked).
"""
from __future__ import annotations

import pandas as pd
import pytest

from goldenswing.live.manipulation import chase_blocked
from goldenswing.playbooks.base import EngineContext
from goldenswing.playbooks.judas_swing import JudasSwingPlaybook
from goldenswing.playbooks.turtle_soup import TurtleSoupPlaybook

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
    out = []
    for t in range(len(df)):
        for sig in pb.live_signal([], t, df):
            out.append(sig.index)
    return out


def _parity(pb_cls, df):
    batch = [c.signal_index for c in pb_cls().candidates(_ctx(df))]
    live = _live_indices(pb_cls(), df)
    assert batch == live, f"{pb_cls.__name__}: batch {batch} != live {live}"
    return batch


def _no_repaint(pb_cls, df, sig_index):
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


def _state_roundtrip(pb_cls, df, split):
    """export/import mid-stream preserves the signal set."""
    uninterrupted = _live_indices(pb_cls(), df)
    pb = pb_cls()
    for t in range(split + 1):
        pb.live_signal([], t, df)
    pb2 = pb_cls()
    pb2.import_state(pb.export_state())
    resumed = []
    for t in range(split + 1, len(df)):
        for sig in pb2.live_signal([], t, df):
            resumed.append(sig.index)
    first = []
    pb3 = pb_cls()
    for t in range(split + 1):
        for sig in pb3.live_signal([], t, df):
            first.append(sig.index)
    assert first + resumed == uninterrupted


# ------------------------------------------------------------- turtle soup
# Bullish soup (long): 20-bar low 99.5 swept at bar 20 (low 98, close
# back inside at 100), trigger bar 21 (close 101 > sweep close),
# confirmation bar 22 (high 102 > sweep-bar high 100.5, leaves bullish
# FVG [100.5, 100.8]), holding retest bar 23 (low 100.6 touches the
# FVG, close 101 holds above the near edge 100.8), target bar 24.
SOUP_BULLISH = FLAT + [
    (100.0, 100.5, 98.0, 100.0),    # 20: SWEEP of the 20-bar low
    (100.0, 101.5, 99.0, 101.0),    # 21: TRIGGER (close > sweep close)
    (101.0, 102.0, 100.8, 101.5),   # 22: CONFIRM (high > sweep-bar high)
    (101.5, 101.5, 100.6, 101.0),   # 23: RETEST (holding) -> signal
    (101.0, 106.0, 104.0, 105.5),   # 24: TARGET hit (~105.46)
]


def test_turtle_soup_bullish_fires_on_holding_retest():
    df = _df(SOUP_BULLISH)
    idx = _parity(TurtleSoupPlaybook, df)
    assert idx == [23]
    sig = TurtleSoupPlaybook().candidates(_ctx(df))[0].outcome.signal
    assert sig.direction == 1
    assert sig.entry_ref == pytest.approx(100.8)  # FVG near edge (top)
    assert sig.invalidation == pytest.approx(100.5)  # FVG far edge
    assert sig.live_context["sweep_idx"] == 20
    assert sig.live_context["confirm_idx"] == 22
    assert sig.live_context["fvg_formed_idx"] == 22


def test_turtle_soup_chop_never_fires():
    df = _df(FLAT + [(100.0, 100.5, 99.5, 100.0)] * 10)
    assert TurtleSoupPlaybook().candidates(_ctx(df)) == []


def test_turtle_soup_close_beyond_is_a_real_break():
    # The sweep bar CLOSES beyond the extreme: real break, no setup.
    rows = FLAT + [
        (100.0, 100.5, 98.0, 99.0),    # 20: wick below, close below too
        (99.0, 100.0, 98.5, 99.5),     # 21
        (99.5, 101.0, 99.0, 100.5),    # 22
        (100.5, 101.5, 100.0, 101.0),  # 23
    ]
    df = _df(rows)
    assert TurtleSoupPlaybook().candidates(_ctx(df)) == []


def test_turtle_soup_no_trigger_expires():
    rows = FLAT + [
        (100.0, 100.5, 98.0, 100.0),   # 20: sweep
        (100.0, 100.5, 99.5, 100.0),   # 21-25: never close > sweep close
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),   # 26: past the 5-bar trigger window
    ]
    df = _df(rows)
    assert TurtleSoupPlaybook().candidates(_ctx(df)) == []


def test_turtle_soup_no_confirmation_expires():
    rows = FLAT + [
        (100.0, 100.5, 98.0, 100.0),   # 20: sweep
        (100.0, 101.5, 99.0, 101.0),   # 21: trigger
    ] + [(100.5, 100.5, 99.5, 100.5)] * 10  # 22-31: never through 100.5
    df = _df(rows)
    assert TurtleSoupPlaybook().candidates(_ctx(df)) == []


def test_turtle_soup_no_retest_within_armed_window():
    # Confirmed, FVG frozen, but price runs away and never retraces.
    rows = FLAT + [
        (100.0, 100.5, 98.0, 100.0),   # 20: sweep
        (100.0, 101.5, 99.0, 101.0),   # 21: trigger
        (101.0, 102.0, 100.8, 101.5),  # 22: confirm + FVG
    ]
    # Bars 23-34: steady rise, lows always above the FVG top (100.8).
    base = 102.0
    for i in range(12):
        rows.append((base + i, base + i + 1.0, base + i + 0.2,
                     base + i + 0.5))
    rows.append((114.0, 115.0, 114.2, 114.5))  # 35: past the armed window
    df = _df(rows)
    assert TurtleSoupPlaybook().candidates(_ctx(df)) == []


def test_turtle_soup_retest_that_does_not_hold_waits():
    # Bar 23 touches the FVG but closes back BELOW the near edge
    # (not a rejection): no signal. Bar 24 holds -> signal at 24.
    rows = FLAT + [
        (100.0, 100.5, 98.0, 100.0),   # 20: sweep
        (100.0, 101.5, 99.0, 101.0),   # 21: trigger
        (101.0, 102.0, 100.8, 101.5),  # 22: confirm + FVG
        (101.0, 101.5, 100.6, 100.2),  # 23: touch, close below near edge
        (100.2, 101.5, 100.6, 101.0),  # 24: touch + HOLD -> signal
        (101.0, 106.0, 104.0, 105.5),  # 25: target
    ]
    df = _df(rows)
    idx = _parity(TurtleSoupPlaybook, df)
    assert idx == [24]


def test_turtle_soup_invalidation_exit():
    rows = SOUP_BULLISH[:24] + [
        (101.0, 101.5, 99.0, 99.5),    # 24: close below the FVG bottom
    ]
    df = _df(rows)
    cands = TurtleSoupPlaybook().candidates(_ctx(df))
    assert len(cands) == 1
    assert cands[0].outcome.exit_reason == "invalidation"


def test_turtle_soup_no_repaint():
    df = _df(SOUP_BULLISH)
    _no_repaint(TurtleSoupPlaybook, df, 23)


def test_turtle_soup_state_roundtrip():
    df = _df(SOUP_BULLISH)
    _state_roundtrip(TurtleSoupPlaybook, df, split=22)


def test_turtle_soup_gate_rejects_bad_ordering():
    pb = TurtleSoupPlaybook()
    ctx = _ctx(_df(SOUP_BULLISH))
    sig = pb.candidates(ctx)[0].outcome.signal
    tampered = dict(sig.live_context, confirm_idx=99)
    ok, _reason = pb.check_live_signal(sig, ctx, tampered)
    assert ok is False


def test_turtle_soup_fade_entry_passes_manipulation_filter():
    df = _df(SOUP_BULLISH)
    sig = TurtleSoupPlaybook().candidates(_ctx(df))[0].outcome.signal
    closed = df.iloc[:sig.index + 1]
    blocked, _why = chase_blocked(closed, int(sig.direction),
                                  float(sig.entry_ref), float(sig.atr))
    assert blocked is False


# ------------------------------------------------------------ judas swing
# Winter Monday (2026-01-05): off 0-6, Asia 7-13 (range 100-102),
# London from 14.  Bearish Judas: bar 14 sweeps the Asia high
# (103.5, close back inside at 101.5), bar 15 triggers (close 100.5),
# bar 16 confirms (low 99 < sweep-bar low 101), bar 17 holding
# retests the Asia high (high 102.5, close 101 holds), bar 18 target.
JUDAS_BEARISH = (
    [(101.0, 102.0, 100.0, 101.0)] * 7 +   # 0-6: off session
    [(101.0, 102.0, 100.0, 101.0)] * 7 +   # 7-13: Asia range 100-102
    [
        (101.0, 103.5, 101.0, 101.5),   # 14: SWEEP (London, Asia high)
        (101.5, 101.5, 100.0, 100.5),   # 15: TRIGGER (close < sweep close)
        (100.5, 101.0, 99.0, 99.5),     # 16: CONFIRM (low < sweep-bar low)
        (99.5, 102.5, 99.0, 101.0),     # 17: RETEST (holding) -> signal
        (101.0, 101.5, 98.0, 99.0),     # 18: TARGET hit (99.0)
    ]
)


def test_judas_swing_bearish_fires_on_holding_retest():
    df = _df(JUDAS_BEARISH, start="2026-01-04 17:00")
    idx = _parity(JudasSwingPlaybook, df)
    assert idx == [17]
    sig = JudasSwingPlaybook().candidates(_ctx(df))[0].outcome.signal
    assert sig.direction == -1
    assert sig.entry_ref == pytest.approx(102.0)  # swept Asia level
    assert sig.invalidation == pytest.approx(103.5)  # sweep extreme
    assert sig.live_context["sweep_idx"] == 14
    assert sig.live_context["sweep_session"] == "london"
    assert sig.live_context["confirm_idx"] == 16


def test_judas_swing_chop_never_fires():
    rows = [(101.0, 102.0, 100.0, 101.0)] * 30
    df = _df(rows, start="2026-01-04 17:00")
    assert JudasSwingPlaybook().candidates(_ctx(df)) == []


def test_judas_swing_no_retest_within_armed_window():
    rows = JUDAS_BEARISH[:17] + [
        (99.5, 100.5, 98.5, 99.0),     # 17-28: drift down, never touch 102
        (99.0, 100.0, 98.0, 98.5),
        (98.5, 99.5, 97.5, 98.0),
        (98.0, 99.0, 97.0, 97.5),
        (97.5, 98.5, 96.5, 97.0),
        (97.0, 98.0, 96.0, 96.5),
        (96.5, 97.5, 95.5, 96.0),
        (96.0, 97.0, 95.0, 95.5),
        (95.5, 96.5, 94.5, 95.0),
        (95.0, 96.0, 94.0, 94.5),
        (94.5, 95.5, 93.5, 94.0),
        (94.0, 95.0, 93.0, 93.5),
        (93.5, 94.5, 92.5, 93.0),
    ]  # 29: past the 12-bar armed window (confirm 16 + 12)
    df = _df(rows, start="2026-01-04 17:00")
    assert JudasSwingPlaybook().candidates(_ctx(df)) == []


def test_judas_swing_retest_that_does_not_hold_waits():
    # Bar 17 touches the Asia high but closes ABOVE it (not a
    # rejection): no signal. Bar 18 holds -> signal at 18.
    rows = JUDAS_BEARISH[:17] + [
        (99.5, 103.0, 99.0, 102.8),    # 17: touch, close through the edge
        (102.8, 103.0, 100.0, 101.5),  # 18: touch + HOLD -> signal
        (101.5, 101.5, 98.0, 99.0),    # 19: target
    ]
    df = _df(rows, start="2026-01-04 17:00")
    idx = _parity(JudasSwingPlaybook, df)
    assert idx == [18]


def test_judas_swing_invalidation_exit():
    rows = JUDAS_BEARISH[:18] + [
        (101.0, 105.0, 101.0, 104.5),  # 18: close back beyond sweep extreme
    ]
    df = _df(rows, start="2026-01-04 17:00")
    cands = JudasSwingPlaybook().candidates(_ctx(df))
    assert len(cands) == 1
    assert cands[0].outcome.exit_reason == "invalidation"


def test_judas_swing_wrong_session_rejected_by_gate():
    pb = JudasSwingPlaybook()
    ctx = _ctx(_df(JUDAS_BEARISH, start="2026-01-04 17:00"))
    sig = pb.candidates(ctx)[0].outcome.signal
    assert sig.live_context["sweep_session"] == "london"
    tampered = dict(sig.live_context, sweep_session="asia")
    ok, _reason = pb.check_live_signal(sig, ctx, tampered)
    assert ok is False


def test_judas_swing_no_repaint():
    df = _df(JUDAS_BEARISH, start="2026-01-04 17:00")
    _no_repaint(JudasSwingPlaybook, df, 17)


def test_judas_swing_state_roundtrip():
    df = _df(JUDAS_BEARISH, start="2026-01-04 17:00")
    _state_roundtrip(JudasSwingPlaybook, df, split=16)


def test_judas_swing_fade_entry_passes_manipulation_filter():
    df = _df(JUDAS_BEARISH, start="2026-01-04 17:00")
    sig = JudasSwingPlaybook().candidates(_ctx(df))[0].outcome.signal
    closed = df.iloc[:sig.index + 1]
    blocked, _why = chase_blocked(closed, int(sig.direction),
                                  float(sig.entry_ref), float(sig.atr))
    assert blocked is False


# ------------------------------------------------- manipulation filter guard
def test_chase_entry_still_blocked():
    # Asia high 100.5 swept and failed (wick beyond, close back
    # inside): a long entry chasing past it is blocked.
    rows = [(100.0, 100.5, 99.5, 100.0)] * 7 + [
        (100.0, 102.0, 99.5, 100.0),   # sweep-fail of the Asia high
    ]
    df = _df(rows)  # starts 2026-01-01 00:00 UTC = Asia session
    blocked, _why = chase_blocked(df, 1, 103.0, 1.0)
    assert blocked is True
