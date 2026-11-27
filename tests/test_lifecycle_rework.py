"""GoldenSwing rework: armed-setup lifecycle (owner acceptance case 2).

The 25-27 Nov live statuses showed every fresh setup dying before
firing: a swing buy arms, waits on three confirmation bars, gets
two of three, never fires, times out — and another fresh setup arms
and does the same thing, on and on.  Three mechanisms killed them:

(a) the H1 close invalidated every armed POI pending after one H1
    bar (``h1_close_without_entry_confirm``);
(b) the armed TTL was 2 entry-clock bars while the confirmation
    family can need a 3rd (sweep + reclaim + confirm);
(c) a tracker-confirmed breakout was consumed BEFORE its fire ran,
    so one signal-path refusal spent it forever while /status kept
    showing it ARMED — and the retest was detected on H1 closes
    while price ran away from the edge.

The fixes (rework 2026-11-27): the pending lives until its own
entry-clock TTL (3 bars) or a genuine invalidation; the
confirmation window is anchored at the touch; a breakout is
consumed only when its fire produces an order (retrying on a new
entry-clock bar until its window expires); the retest is detected
on M15 closes (the tracker half is pinned in
tests/test_range_boundary.py).
"""
from __future__ import annotations

import pandas as pd
import test_mt5_tick_clock as tc
from test_range_boundary import (
    BoundaryTracker,
    _events,
    _range,
)
from test_range_boundary import (
    _bars as _rbars,
)
from test_range_boundary import (
    _frame as _rframe,
)

from goldenswing.live.hierarchy import check_live_trigger


def _breaker_bar(ts):
    """The M15 breaker bar (body 1.8 clears the displacement gate),
    on a fresh bar at ``ts``."""
    return pd.DataFrame(
        {"open": [1999.0], "high": [2000.8], "low": [1999.0],
         "close": [2000.8], "volume": [100], "real_volume": [0],
         "spread": [30]},
        index=pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC")]))


def test_screenshot_sequence_pending_fires_on_third_bar(tmp_path):
    """The 25-27 Nov sequence, fixed: arm, two entry closes with
    no confirmation (2/3), an H1 close passes, the 3rd entry bar
    confirms — the setup FIRES instead of dying.  The old code
    killed it twice over: the TTL (2) disarmed it at the 2nd close,
    and the H1 close invalidated it before the 3rd bar arrived."""
    runner = tc._runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    tc._eval(runner, tc._tick("2026-10-14 08:05", 2000.3), stats)
    assert runner._pending is not None
    assert runner._pending["armed_bars"] == 0
    # Entry close 1: no confirmation, still armed (1/3).
    flat1 = tc._quiet_bars(pd.date_range("2026-10-14 08:00", periods=1,
                                         freq="15min", tz="UTC"))
    runner.feed = tc._flat_m15_feed(flat1)
    runner._refresh_bars(tc._tick("2026-10-14 08:15", 2000.3), stats)
    assert runner._pending is not None
    assert runner._pending["armed_bars"] == 1
    # Entry close 2: no confirmation, still armed (2/3) — the old
    # TTL of 2 disarmed it right here.
    flat2 = tc._quiet_bars(pd.date_range("2026-10-14 08:15", periods=1,
                                         freq="15min", tz="UTC"))
    runner.feed = tc._flat_m15_feed(flat2)
    runner._refresh_bars(tc._tick("2026-10-14 08:30", 2000.3), stats)
    assert runner._pending is not None
    assert runner._pending["armed_bars"] == 2
    assert stats.get("invalidated", 0) == 0
    # An H1 close passes: the pending survives it — the old code
    # invalidated it here (h1_close_without_entry_confirm).
    runner._on_h1_close(tc._tick("2026-10-14 09:00", 2000.3), stats)
    assert runner._pending is not None
    assert stats.get("invalidated", 0) == 0
    # Entry close 3: the breaker confirms — the setup FIRES.
    fired = []
    runner._run_live_signal = lambda pend, tick, st: fired.append(
        pend["trigger"])
    runner.feed = tc._flat_m15_feed(_breaker_bar("2026-10-14 08:30"))
    runner._refresh_bars(tc._tick("2026-10-14 09:00", 2000.3), stats)
    assert fired == ["bos"]
    assert runner._pending is None  # consumed by the confirmation


def test_confirm_window_anchored_at_touch_not_h1_open(tmp_path):
    """The confirmation window starts at the entry bar containing
    the touch, not at the H1 bar's open: a shape that printed
    BEFORE the touch is not the touch's confirmation."""
    runner = tc._runner(tmp_path, m15_n=24, shift_hours=-2)
    setup = runner._live_setup
    frame = tc._frames(m15_n=24, shift_hours=-2)["M15"].copy()
    # Breaker on the 08:15 bar; quiet bars on 08:30 and 08:45.
    frame.loc[pd.Timestamp("2026-10-14 08:15", tz="UTC"),
              ["open", "high", "low", "close"]] = [
        1999.0, 2000.8, 1999.0, 2000.8]
    for ts in ("2026-10-14 08:30", "2026-10-14 08:45"):
        frame.loc[pd.Timestamp(ts, tz="UTC"),
                  ["open", "high", "low", "close"]] = [
            1997.75, 1997.80, 1997.70, 1997.75]
    # Touch at 08:05: the 08:15 breaker is inside the touch's
    # window — it confirms.
    trig = check_live_trigger(
        setup, entry_frame=frame, h1_open=pd.Timestamp(
            "2026-10-14 08:00", tz="UTC"),
        touch_ts=pd.Timestamp("2026-10-14 08:05", tz="UTC"),
        close=2000.3)
    assert trig is not None
    assert trig["trigger"] == "bos"
    # Touch at 08:45: the breaker printed half an hour before the
    # touch — it is not this touch's confirmation.
    trig_late = check_live_trigger(
        setup, entry_frame=frame, h1_open=pd.Timestamp(
            "2026-10-14 08:00", tz="UTC"),
        touch_ts=pd.Timestamp("2026-10-14 08:45", tz="UTC"),
        close=2000.3)
    assert trig_late is None


def _low_breakout_tracker():
    """BoundaryTracker driven to a confirmed SHORT breakout of the
    low edge (the 25 Nov ~16:00 UTC status: broke low edge 3502.7,
    retest pending, price far below the edge).  Range 4100-4200,
    ATR 10: 2 H1 closes below 4095 + M15 BOS down."""
    tr = BoundaryTracker(_range())
    h1 = _rframe(
        _rbars(6, close=4120.0)
        + [[4085.0, 4090.0, 4080.0, 4085.0, 1000.0],
           [4080.0, 4085.0, 4075.0, 4080.0, 1000.0]],
        freq="h")
    m15 = _rframe(
        _rbars(8, close=4110.0)
        + [[4095.0, 4100.0, 4090.0, 4095.0, 1000.0]]  # swing low
        + _rbars(3, close=4110.0)                       # confirms it
        + [[4085.0, 4090.0, 4080.0, 4085.0, 1000.0],
           [4080.0, 4085.0, 4075.0, 4080.0, 1000.0],
           [4075.0, 4080.0, 4070.0, 4075.0, 1000.0],
           [4070.0, 4075.0, 4065.0, 4070.0, 1000.0]])
    events = tr.update(h1, m15)
    assert _events(events, "breakout_confirmed")
    assert tr.regime == "breakout_trend"
    assert tr.breakout_direction == -1
    assert tr.breakout_edge == "low"
    assert tr.breakout_id == 1
    return tr


def _breakout_runner(tmp_path):
    """Runner with a tracker-confirmed SHORT breakout and the
    matching live setup (the 25 Nov 16:00 status shape)."""
    runner = tc._runner(tmp_path, m15_n=20, shift_hours=-2)
    tracker = _low_breakout_tracker()
    runner._boundary = tracker
    runner._live_setup = runner._breakout_setup(tracker)
    assert runner._live_setup["regime"] == "breakout_trend"
    assert runner._live_setup["direction"] == -1
    return runner, tracker


def _append_m15(runner, ts, close):
    """Advance the stack's M15 frame by one bar (the retry gate
    keys on the last closed entry bar)."""
    frame = runner._stack.frames["M15"]
    bar = pd.DataFrame(
        {"open": [close], "high": [close + 5.0], "low": [close - 5.0],
         "close": [close], "volume": [100], "real_volume": [0],
         "spread": [30]},
        index=pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC")]))
    runner._stack.frames["M15"] = pd.concat([frame, bar])


def test_breakout_packaging_failure_does_not_consume_id(tmp_path):
    """A breakout whose fire cannot even be packaged (risk <= 0:
    price sitting exactly on the stop) is NOT consumed — the old
    code marked the id before packaging, so the breakout could
    never fire again."""
    runner, tracker = _breakout_runner(tmp_path)
    stats: dict = {"bars": 0}
    # Mid == stop (4100 + buf 5 = 4105): risk 0, no packaging.
    tc._eval(runner, tc._tick("2026-10-14 08:05", 4105.0), stats)
    assert runner._breakout_fired_id == 0
    assert runner._breakout_fired_id != tracker.breakout_id


def test_breakout_refused_stays_fireable_then_fires(tmp_path):
    """The 25 Nov 16:00 status, fixed: the breakout is confirmed
    (2 closes + M15 BOS) and price is far below the edge, but the
    first fire produces no order — the breakout stays fireable,
    retries on a new entry bar (not every tick), and is consumed
    only when a fire produces an order."""
    runner, tracker = _breakout_runner(tmp_path)
    stats: dict = {"bars": 0}
    calls = []
    runner._run_live_signal = lambda pend, tick, st: calls.append(
        pend["trigger"])
    # First fire: price $50 below the edge (the screenshot's $47.6
    # shape), signal path produces no order.
    tc._eval(runner, tc._tick("2026-10-14 08:05", 4050.0), stats)
    assert calls == ["range_breakout"]
    assert runner._breakout_fired_id == 0  # NOT consumed
    assert "breakout fire refused" in tc._journal_text(tmp_path)
    # Same entry bar, another tick: no retry yet (the structure
    # only changes when a new entry bar prints).
    tc._eval(runner, tc._tick("2026-10-14 08:06", 4050.0), stats)
    assert len(calls) == 1
    # A new M15 bar prints: the fire retries.
    _append_m15(runner, "2026-10-14 08:00", 4050.0)
    tc._eval(runner, tc._tick("2026-10-14 08:16", 4050.0), stats)
    assert len(calls) == 2
    assert runner._breakout_fired_id == 0  # still not consumed
    # The next attempt produces an order (the stub simulates the
    # signal path placing one — the real path is proven by
    # test_runner_full_live_path_places_order): the breakout is
    # consumed now, and only now.
    runner._run_live_signal = lambda pend, tick, st: (
        runner._pending_orders.__setitem__("test-fill", object()),
        calls.append(pend["trigger"]))
    _append_m15(runner, "2026-10-14 08:15", 4050.0)
    tc._eval(runner, tc._tick("2026-10-14 08:31", 4050.0), stats)
    assert len(calls) == 3
    assert runner._breakout_fired_id == tracker.breakout_id == 1
    assert runner._fired_h1_bar == pd.Timestamp(
        "2026-10-14 08:00", tz="UTC")
    assert "trigger fired range_breakout" in tc._journal_text(tmp_path)
    # And it never fires twice: the id is spent.
    _append_m15(runner, "2026-10-14 08:30", 4050.0)
    tc._eval(runner, tc._tick("2026-10-14 08:46", 4050.0), stats)
    assert len(calls) == 3
