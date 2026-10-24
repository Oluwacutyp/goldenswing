"""MT5 engine: the H1 close is NOT the signal clock.

The trigger is sequential, not simultaneous (2026-10-15 redesign):
CONTAINMENT FIRST — the H1 reference level on the zone side must
nest inside the H4 POI zone (± buf) or the setup never arms.
Then ARMED on a tick whose price sits AT the H4 POI band, and
CONFIRMED when a POI confirmation-family shape fires on the
entry-TF bars composing the forming H1 bar (M15 in killzone, M30
elsewhere).  Disarm if price exits the zone (through the stop, or
ran away the other side) or the H1 close lapses the window.
Frames below are hand-built so the setup is armed: D1/H4 bias +1,
H4 support 2000.0, H1 ref_low 1998.5 nesting in the zone, and the
M15 breaker bar (10:30) confirms the entry.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.live.hierarchy import (TF_MINUTES, TimeframeStack,
                                        check_live_trigger,
                                        check_zone_touch, live_setup,
                                        live_setup_reason)

H1_HIGH = [1992, 1993, 1994, 1995, 1996, 1996, 1996, 1997, 1997, 1997,
           1997, 1997, 1998, 1997, 1997, 1996, 1996, 1996, 1997, 1997,
           1998, 1999, 2000, 2000, 2000, 2000, 2001, 2000, 2000, 2000.5]
H1_LOW = [1991, 1992, 1993, 1994, 1995, 1990, 1995, 1996, 1996, 1996,
          1996, 1996, 1997, 1996, 1996, 1995, 1995, 1995, 1996, 1992,
          1997, 1998, 1999, 1999, 1999, 1999, 1998.5, 1999, 1999, 1999.5]
H4_HIGH = [2001, 2001.5, 2002, 2002.5, 2003, 1998, 2002, 2002.5, 2003,
           2003, 2003, 2003, 2005, 2003, 2003, 2002.5, 2002.5, 2002.5,
           2003, 2003, 2003.5, 2004, 2004.5, 2004.5, 2004.5, 2004.5,
           2007, 2004, 2004, 2003.3]
H4_LOW = [1995, 1995.5, 1996, 1996.5, 1996.5, 1992, 1996, 1996.5, 1997,
          1997, 1997, 1997, 1999, 1997, 1997, 1996.5, 2001, 2001, 2001,
          2000, 2001, 2001, 2001.5, 2001.5, 2001.5, 2001.5, 2003, 2001,
          2001, 1997.3]
M15_HIGH = [1994, 1995, 1995.5, 1996, 1996, 1996, 1996.5, 1997, 1997,
            1997, 1998.5, 1998, 1998, 1998, 1998, 1998.5, 1998.5,
            1998.5, 2000, 1999, 1999.5, 1999.8, 2000.8, 2000.9]
M15_LOW = [1994, 1994, 1994, 1994, 1994, 1993, 1994, 1994, 1994, 1994,
           1994, 1995, 1995, 1995, 1994.5, 1995.5, 1995.5, 1995.5,
           1995.5, 1995.5, 1999, 1999.2, 1999.0, 2000.8]

FORMING_OPEN = pd.Timestamp("2026-10-14 10:00", tz="UTC")


def _bars(times, highs, lows, last_close=None):
    opens, closes = [], []
    for h, l in zip(highs, lows):
        mid = (h + l) / 2
        opens.append(mid - 0.1)
        closes.append(mid + 0.1)
    if last_close is not None:
        closes[-1] = last_close
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [100] * len(times), "real_volume": [0] * len(times),
         "spread": [30] * len(times)},
        index=pd.DatetimeIndex(times, tz="UTC"))


def _flat(times, level=2000.0):
    return _bars(times, [level + 0.5] * len(times),
                 [level - 0.5] * len(times))


def _frames(*, m15_n=24, shift_hours=0):
    """Closed-bars-only stack.  H1 ends 09:00 (forming bar 10:00);
    M15 is ``m15_n`` bars: 24 = full (breaker 10:30 present),
    23 = through the breaker, 20 = through 09:45 (no composing
    bars yet).  ``shift_hours`` moves every frame (e.g. -2 puts the
    forming bar at 08:00, inside the London killzone -> M15 entry
    clock; 0 leaves it at 10:00, outside killzone -> M30)."""
    shift = pd.Timedelta(hours=shift_hours)
    h1_idx = pd.date_range("2026-10-13 04:00", periods=30, freq="h",
                           tz="UTC") + shift
    h4_idx = pd.date_range("2026-10-09 12:00", periods=30, freq="4h",
                           tz="UTC") + shift
    d1_idx = pd.date_range("2026-09-14", periods=30, freq="D",
                           tz="UTC") + shift
    m15_idx = pd.date_range("2026-10-14 05:00", periods=24, freq="15min",
                            tz="UTC") + shift
    m30_idx = pd.date_range("2026-10-13 19:30", periods=30, freq="30min",
                            tz="UTC") + shift
    m5_idx = pd.date_range("2026-10-14 07:30", periods=30, freq="5min",
                           tz="UTC") + shift
    h1 = _bars(h1_idx, H1_HIGH, H1_LOW, last_close=2000.3)
    h4 = _bars(h4_idx, H4_HIGH, H4_LOW, last_close=2000.3)
    d1 = _bars(d1_idx, H4_HIGH, H4_LOW, last_close=2000.3)
    m15 = _bars(m15_idx, M15_HIGH, M15_LOW)
    # Breaker bar 22 (10:30 / 08:30 shifted): body 1.8 clears the
    # M15 displacement gate.
    m15.loc[m15_idx[22], ["open", "high", "low", "close"]] = [
        1999.0, 2000.8, 1999.0, 2000.8]
    m15.loc[m15_idx[23], ["open", "close"]] = [2000.8, 2000.9]
    m15 = m15.iloc[:m15_n]
    return {"D1": d1, "H4": h4, "H1": h1, "M15": m15,
            "M30": _flat(m30_idx), "M5": _flat(m5_idx)}


def _setup():
    setup = live_setup(_frames())
    assert setup is not None
    return setup


def _tick(when, mid):
    return {"time": pd.Timestamp(when, tz="UTC").to_pydatetime(),
            "mid": mid, "bid": mid, "ask": mid + 0.30,
            "tick_volume": 5, "real_volume": 0, "spread": 0.30}  # ask-bid, price units (real-feed convention)


class FakeExecutor:
    def __init__(self):
        self.orders: list[dict] = []
        self._ticket = 9000
        self._positions: list[Position] = []

    def connect(self, **kw):
        return True, "fake connected"

    def symbol_spec(self, symbol):
        from goldenswing.execution.risk import gold_spec
        return gold_spec()

    def positions(self, symbol=None):
        return [p for p in self._positions
                if symbol is None or p.symbol == symbol]

    def place_order(self, request):
        self._ticket += 1
        self.orders.append({"ticket": self._ticket,
                            "symbol": request.symbol,
                            "direction": request.direction,
                            "volume": request.volume,
                            "sl": request.stop_loss,
                            "tp": request.take_profit})
        self._positions.append(Position(
            ticket=self._ticket, symbol=request.symbol,
            direction=request.direction, volume=request.volume,
            entry_price=request.entry_ref, sl=request.stop_loss,
            tp=request.take_profit, magic=request.magic,
            alert_id=request.alert_id))
        return OrderResult(True, ticket=self._ticket,
                           price=request.entry_ref,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        self._positions = [p for p in self._positions
                           if p.ticket != ticket]
        return OrderResult(True, ticket=ticket)


def _runner(tmp_path, *, m15_n=20, shift_hours=0, executor=None,
            **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("stub",),
        # News pause OFF: the order path's news check reads the
        # REAL wall-clock calendar — these tests must not fail on
        # a real news day (e.g. PPI).  The pause itself is covered
        # by test_news_calendar.py.
        news_pause_enabled=False, **cfg_kw)
    runner = MT5Runner(cfg, feed=None, executor=executor or FakeExecutor())
    runner._stack = TimeframeStack.from_frames(
        _frames(m15_n=m15_n, shift_hours=shift_hours))
    for tf, frame in runner._stack.frames.items():
        runner._last_processed[tf] = (
            frame.index[-1] + pd.Timedelta(minutes=TF_MINUTES[tf]))
    runner._live_setup = live_setup(runner._stack)
    assert runner._live_setup is not None
    return runner


def _journal_text(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    return path.read_text() if path.exists() else ""


# ------------------------------------------------------------- live_setup
def _eval(runner, tick, stats):
    """Live-loop contract: _last_tick is the tick under evaluation.

    CLOCK ISOLATION (gs-clock pass): _on_tick sets _last_tick
    before evaluating, but direct _evaluate_live_tick callers
    left it None — the order path then fell back to the wall
    clock and the Friday-close gate refused orders on Friday
    evenings (2026-10-23).  Every direct evaluation goes
    through here so the engine clock is always the tick clock.
    """
    runner._last_tick = tick
    runner._evaluate_live_tick(tick, stats)


def test_live_setup_armed_from_closed_bars():
    setup = _setup()
    assert setup["direction"] == 1
    assert setup["bias_d1"] == 1 and setup["bias_h4"] == 1
    assert setup["zone"]["kind"] == "support"
    assert setup["zone"]["level"] == pytest.approx(2000.0)
    assert setup["ref_high"] == pytest.approx(2001.0)
    assert setup["ref_low"] == pytest.approx(1998.5)
    # CONTAINMENT: the H1 reference level nests inside the H4 zone.
    assert setup["ref_level"] == pytest.approx(1998.5)
    assert setup["zone"]["bottom"] - setup["buf"] <= setup["ref_level"] \
        <= setup["zone"]["top"] + setup["buf"]
    assert setup["atr_h1"] > 0 and setup["buf"] > 0


def test_live_setup_not_armed_when_level_does_not_nest():
    # H1's latest swing low sits far below the H4 support zone: the
    # levels live at different coordinates, so no tick could ever
    # satisfy both — the setup is invalid, not armed.
    saved = list(H1_LOW)
    try:
        H1_LOW[26] = 2000.0  # no dip: latest swing low is 1992
        frames = _frames()
        setup = live_setup(frames)
        assert setup is None
    finally:
        H1_LOW[:] = saved


def test_live_setup_not_armed_when_h4_disagrees_but_levels_do_not_nest():
    # H4 disagreement alone must NOT disarm (pullback context —
    # see test_live_pullback_signal).  This mirrored H4 still
    # yields None, but via CONTAINMENT: the H1 ref level sits far
    # from the mirrored H4 POI, so no tick could satisfy both.
    frames = _frames()
    # Mirror H4 around 2000: uptrend becomes downtrend (bias -1).
    h4 = frames["H4"]
    frames["H4"] = pd.DataFrame(
        {"open": 4000.0 - h4["close"], "high": 4000.0 - h4["low"],
         "low": 4000.0 - h4["high"], "close": 4000.0 - h4["open"],
         "volume": h4["volume"], "real_volume": h4["real_volume"],
         "spread": h4["spread"]}, index=h4.index)
    assert live_setup(frames) is None
    assert live_setup_reason(frames).startswith("containment")


# ------------------------------------------------------- check_zone_touch
def test_zone_touch_arms_only_inside_band():
    setup = _setup()
    zone = setup["zone"]
    assert check_zone_touch(setup, mid=2000.0) is True
    assert check_zone_touch(setup, mid=zone["bottom"]) is True
    assert check_zone_touch(setup, mid=zone["top"]) is True
    # Above the band (breakout, not a touch): no arm.
    assert check_zone_touch(setup, mid=zone["top"] + 1.0) is False
    # Below the band: no arm.
    assert check_zone_touch(setup, mid=zone["bottom"] - 1.0) is False


# ------------------------------------------------------- check_live_trigger
def test_trigger_fires_on_touch_plus_confirm():
    setup = _setup()
    m15 = _frames()["M15"]
    trig = check_live_trigger(
        setup, entry_frame=m15, h1_open=FORMING_OPEN, close=2000.5)
    assert trig is not None
    assert trig["trigger"] == "bos"
    assert trig["direction"] == 1
    assert trig["entry"] == pytest.approx(2000.5)
    # Stop sits just beyond the defended zone edge + buf.
    assert trig["stop"] == pytest.approx(
        setup["zone"]["bottom"] - setup["buf"], abs=1e-9)
    assert trig["risk"] == pytest.approx(
        2000.5 - trig["stop"], abs=1e-9)
    # Target is the 2R floor (nearest H4 pool is inside 2R).
    assert trig["target"] == pytest.approx(
        2000.5 + 2 * trig["risk"], abs=1e-6)
    # Invalidation level is the stop: through it, the touch failed.
    assert trig["level"] == pytest.approx(trig["stop"], abs=1e-9)
    assert trig["confirm_shapes"] == ["bos", "sweep_reclaim",
                                      "fvg_reaction"]


def test_trigger_does_not_fire_on_touch_alone():
    # Entry bars at the zone with NO confirming shape: touch alone
    # never fires.  Flat bars confirm nothing (no wick, no body,
    # no swing break).
    setup = _setup()
    m15 = _frames()["M15"]
    flat = _flat(m15.index, level=2000.0)
    assert check_live_trigger(
        setup, entry_frame=flat, h1_open=FORMING_OPEN,
        close=2000.0) is None


def test_trigger_does_not_fire_on_breakout_above_zone():
    # The old failure mode in reverse: a BOS bar ABOVE the zone is
    # a breakout, not a POI reaction.  The family itself says bos
    # here (its BOS detector does not require POI overlap), so this
    # test proves the trigger's AT-POI guard does the work.
    from goldenswing.entry.validator import confirm_at_poi
    setup = _setup()
    m15 = _frames()["M15"]
    above = m15.copy()
    above[["open", "high", "low", "close"]] += 10.0
    ann = confirm_at_poi(
        above[above.index >= FORMING_OPEN], 1,
        setup["zone"]["bottom"], setup["zone"]["top"], context=above)
    assert ann.confirmed and ann.shape == "bos"
    assert check_live_trigger(
        setup, entry_frame=above, h1_open=FORMING_OPEN,
        close=2010.5) is None


def test_trigger_window_is_composing_bars():
    setup = _setup()
    m15 = _frames()["M15"]
    # No composing bars yet (frame ends 09:45): no confirmation.
    assert check_live_trigger(
        setup, entry_frame=m15.iloc[:20], h1_open=FORMING_OPEN,
        close=2000.0) is None
    # Composing bars through 10:15 (breaker not closed yet): the
    # family still confirms on the early bars — the window is the
    # composing bars, not a fixed count.
    trig = check_live_trigger(
        setup, entry_frame=m15.iloc[:22], h1_open=FORMING_OPEN,
        close=2000.0)
    assert trig is not None
    assert trig["trigger"] == "bos"


# ------------------------------------------------------------- runner clock
def test_runner_touch_arms_on_tick_not_h1_close(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    # No H1 close, no refresh: a tick AT the zone arms the pending.
    _eval(runner, _tick("2026-10-14 10:05", 2000.3), stats)
    assert runner._pending is not None
    assert runner._pending["armed"] is True
    assert runner._pending["trigger"] == "zone_touch"
    assert runner._pending["direction"] == 1
    # Armed, not fired: no trigger count, no order, no fired line.
    assert stats.get("triggers", 0) == 0
    assert runner.executor.orders == []
    assert "trigger fired" not in _journal_text(tmp_path)
    # Next tick: still armed (no composing bars yet, no confirm).
    _eval(runner, _tick("2026-10-14 10:06", 2001.8), stats)
    assert runner._pending is not None
    assert runner._pending["armed"] is True
    assert stats.get("triggers", 0) == 0


def test_runner_entry_confirms_on_m15_close(tmp_path):
    # Shift -2h: forming bar 08:00, inside the London killzone, so the
    # session entry clock is M15 (at 10:00 it would be M30).
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    _eval(runner, _tick("2026-10-14 08:06", 2001.8), stats)
    assert runner._pending is not None
    assert runner._pending["armed"] is True
    assert runner._pending["entry_clock"] == "M15"
    fired = []

    def _record(pend, tick, st):
        fired.append(pend["trigger"])

    runner._run_live_signal = _record
    # M15 close at 08:45 (the 08:30 breaker bar closes): the feed
    # delivers the composing bars; the confirmation family fires.
    new_m15 = _frames(m15_n=24, shift_hours=-2)["M15"].iloc[20:23]  # 08:00, 08:15, 08:30

    class FakeFeed:
        def fetch_new_all(self, cursors):
            empty = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume",
                         "real_volume", "spread"])
            return {"M15": new_m15, "H1": empty, "M30": empty,
                    "M5": empty, "H4": empty, "D1": empty}

    runner.feed = FakeFeed()
    runner._refresh_bars(_tick("2026-10-14 08:45", 2000.5), stats)
    assert fired == ["bos"]  # signal path ran on the M15 close
    assert runner._pending is None  # consumed by the confirmation


def test_runner_entry_confirms_on_m15_close_at_h1_boundary(tmp_path):
    """An M15 close coinciding with the H1 close still confirms.

    The last composing M15 bar (08:45) closes exactly when its H1
    bar (08:00) does.  The confirmation family must get its shot
    BEFORE the H1-close expiry — the H1 close must not gate (and
    kill) the evaluation.  The stack iterates H1 before M15, so
    per-TF dict order would expire the pending first without the
    phased refresh.
    """
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    _eval(runner, _tick("2026-10-14 08:06", 2001.8), stats)
    assert runner._pending is not None
    assert runner._pending["armed"] is True
    assert runner._pending["entry_clock"] == "M15"
    fired = []
    runner._run_live_signal = lambda pend, tick, st: fired.append(
        pend["trigger"])
    # Boundary tick 09:00: the 08:00 H1 bar AND the 08:45 M15 bar
    # close together.  The feed delivers both.
    frames = _frames(m15_n=24, shift_hours=-2)
    new_m15 = frames["M15"].iloc[20:24]  # 08:00, 08:15, 08:30, 08:45
    new_h1 = frames["H1"].iloc[-1:].copy()
    new_h1.index = new_h1.index + pd.Timedelta(hours=1)  # the 08:00 bar

    class FakeFeed:
        def fetch_new_all(self, cursors):
            empty = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume",
                         "real_volume", "spread"])
            return {"M15": new_m15, "H1": new_h1, "M30": empty,
                    "M5": empty, "H4": empty, "D1": empty}

    runner.feed = FakeFeed()
    runner._refresh_bars(_tick("2026-10-14 09:00", 2000.5), stats)
    assert fired == ["bos"]  # entry confirmed, not expired
    assert runner._pending is None  # consumed by the confirmation
    assert stats.get("invalidated", 0) == 0
    assert "h1_close_without_entry_confirm" not in _journal_text(tmp_path)


def test_runner_second_touch_same_h1_bar_does_not_refire(tmp_path):
    """One trigger per H1-bar window: after a fire, a second touch
    in the SAME H1 bar must not re-arm / re-fire on the same
    composing bars.  Found in the full-engine backtest: the same
    H1 bar fired ``tweezer_top`` on every tick (dozens of fires per
    bar) because the arming-tick confirm shot re-fires on the
    already-closed composing bars.  The setup is fixed within an
    H1 bar, so a second fire there is necessarily a duplicate.
    The next H1 bar is a fresh window and re-arms normally.
    """
    runner = _runner(tmp_path, m15_n=23, shift_hours=-2)
    stats: dict = {"ticks": 0, "bars": 0, "signals": 0, "executed": 0,
                   "shadow": 0, "gated": 0, "hierarchy_blocked": 0,
                   "tick_actions": 0, "reconciled_closed": 0,
                   "triggers": 0, "invalidated": 0}
    fired = []
    runner._run_live_signal = lambda pend, tick, st: fired.append(
        pend["trigger"])
    # First touch fires on the already-closed breaker bar.
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    assert fired == ["bos"]
    assert stats["triggers"] == 1
    assert runner._pending is None  # consumed by the confirmation
    assert runner._fired_h1_bar == pd.Timestamp(
        "2026-10-14 08:00", tz="UTC")
    # Second touch, same H1 bar, still in the zone: no re-arm.
    _eval(runner, _tick("2026-10-14 08:06", 2000.3), stats)
    assert fired == ["bos"]  # no second fire
    assert stats["triggers"] == 1
    assert runner._pending is None
    # Next H1 bar: fresh window — the touch re-arms (the composing
    # bars for 09:00 haven't closed yet, so no confirm yet).
    _eval(runner, _tick("2026-10-14 09:05", 2000.3), stats)
    assert runner._pending is not None
    assert runner._pending["armed"] is True
    assert runner._pending["h1_open"] == pd.Timestamp(
        "2026-10-14 09:00", tz="UTC")


def test_runner_wick_beyond_level_does_not_invalidate(tmp_path):
    # Close Rule (owner, 2026-10-15): a WICK beyond the
    # invalidation level does NOT invalidate — that is the expected
    # stop-hunt.  Only a bar CLOSE beyond invalidates.  A tick
    # through the level is a wick (the old tick-level "reversal"
    # check is gone); so is an entry bar whose low sweeps beyond
    # the level but whose close returns above it.
    # (The wick bar closes above the invalidation level but below
    # the zone band — no overlap, so it cannot confirm either;
    # a bar closing back INSIDE the zone would be a sweep_reclaim
    # confirmation, which is a fire, not an invalidation.)
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    assert runner._pending is not None
    inv = runner._pending["invalidation_level"]
    # Tick wick through the level: still armed.
    _eval(runner, _tick("2026-10-14 08:06", 1997.0), stats)
    assert runner._pending is not None
    assert stats.get("invalidated", 0) == 0
    # Bar wick: low 1996.5 sweeps beyond the level (inv ≈ 1997.68)
    # but close 1997.75 returns above it (and below the zone
    # bottom ≈ 1997.83, so no overlap, no confirm).
    wick = _bars(pd.date_range("2026-10-14 08:00", periods=1,
                               freq="15min", tz="UTC"),
                 [1997.78], [1996.5], last_close=1997.75)
    assert wick["close"].iloc[0] > inv
    assert wick["high"].iloc[0] < runner._live_setup["zone"]["bottom"]
    runner.feed = _flat_m15_feed(wick)
    runner._refresh_bars(_tick("2026-10-14 08:15", 2000.3), stats)
    assert runner._pending is not None  # still armed after the wick
    assert runner._pending["armed_bars"] == 1  # one bar, no confirm
    assert stats.get("invalidated", 0) == 0
    assert "trigger invalidated" not in _journal_text(tmp_path)


def test_runner_close_beyond_level_invalidates_on_bar_close(tmp_path):
    # Close Rule: an entry-TF bar CLOSING beyond the invalidation
    # level invalidates instantly — zero hesitation, no waiting
    # for another bar, no hoping for a bounce.  Monitoring for
    # THAT setup terminates: a later re-touch on the same (failed)
    # zone does not re-arm.
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    assert runner._pending is not None
    inv = runner._pending["invalidation_level"]
    zone = runner._live_setup["zone"]
    # Bar closes at 1997.0, below the level (≈ 1997.68).
    bad = _bars(pd.date_range("2026-10-14 08:00", periods=1,
                              freq="15min", tz="UTC"),
                [1997.2], [1996.5], last_close=1997.0)
    assert bad["close"].iloc[0] < inv
    runner.feed = _flat_m15_feed(bad)
    runner._refresh_bars(_tick("2026-10-14 08:15", 2000.3), stats)
    assert runner._pending is None
    assert stats["invalidated"] == 1
    journal = _journal_text(tmp_path)
    assert "trigger invalidated (invalidation)" in journal
    assert "trigger fired" not in journal
    # The failed zone is terminated: re-touch does not re-arm.
    assert runner._invalidated_zone == (1, zone["bottom"], zone["top"])
    _eval(runner, _tick("2026-10-14 08:20", 2000.3), stats)
    assert runner._pending is None
    assert stats["invalidated"] == 1  # no second invalidation


def test_confirmed_setup_invalidated_before_entry_never_places(tmp_path):
    # CONFIRMED state (trigger fired, Golden Swing order staged
    # awaiting the Confirm tap — nothing placed yet) + an entry
    # bar closing beyond the signal's invalidation -> the staged
    # order is dropped: it NEVER places.  Both the payload and
    # its Confirm token die (a tap on the dead token reports
    # expired).
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    sig = SimpleNamespace(direction=1, entry_ref=2005.0,
                          invalidation=1990.0, score=80.0, index=29,
                          targets=[2040.0], atr=10.0)
    runner._order_confirms["key1"] = {
        "pb": SimpleNamespace(name="stub"), "sig": sig,
        "verdict": {"entry_clock": "M15"}, "entry_bar": 29}
    runner._confirm_tokens["tok1"] = {
        "action": "place_order", "arg": "key1", "created_at": 0.0}
    # Bar closes at 1985.0, below the signal's invalidation (1990).
    bad = _bars(pd.date_range("2026-10-14 08:00", periods=1,
                              freq="15min", tz="UTC"),
                [2001.0], [1980.0], last_close=1985.0)
    runner.feed = _flat_m15_feed(bad)
    runner._refresh_bars(_tick("2026-10-14 08:15", 2000.3), stats)
    assert runner._order_confirms == {}
    assert "tok1" not in runner._confirm_tokens
    assert runner.executor.orders == []  # nothing ever placed
    journal = _journal_text(tmp_path)
    assert "invalidated" in journal
    assert "no order placed" in journal
    ok, msg = runner.confirm_pending("tok1", {})
    assert ok is False
    assert "expired" in msg


def test_runner_left_zone_disarms_pending(tmp_path):
    # Price runs away above the zone without confirming: the moment
    # passed — disarm, don't wait forever on a stale arm.
    runner = _runner(tmp_path)
    stats: dict = {}
    _eval(runner, _tick("2026-10-14 10:05", 2000.3), stats)
    assert runner._pending is not None
    # Above zone top + buf (≈ 2002.32): left the zone.
    _eval(runner, _tick("2026-10-14 10:06", 2003.0), stats)
    assert runner._pending is None
    assert stats["invalidated"] == 1
    assert "trigger invalidated (left_zone)" in _journal_text(tmp_path)


def test_runner_confirm_after_exit_does_not_fire(tmp_path):
    # Arm, exit (disarm), THEN the confirming entry bars arrive on a
    # tick whose price is outside the zone: no re-arm, no fire.
    # Confirmation must arrive while price is still in/near the zone.
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    assert runner._pending is not None
    _eval(runner, _tick("2026-10-14 08:06", 2003.0), stats)
    assert runner._pending is None  # disarmed on exit
    fired = []
    runner._run_live_signal = lambda pend, tick, st: fired.append(
        pend["trigger"])
    new_m15 = _frames(m15_n=24, shift_hours=-2)["M15"].iloc[20:23]

    class FakeFeed:
        def fetch_new_all(self, cursors):
            empty = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume",
                         "real_volume", "spread"])
            return {"M15": new_m15, "H1": empty, "M30": empty,
                    "M5": empty, "H4": empty, "D1": empty}

    runner.feed = FakeFeed()
    # Entry-close tick OUTSIDE the zone: no re-arm, no fire.
    runner._refresh_bars(_tick("2026-10-14 08:45", 2003.0), stats)
    _eval(runner, _tick("2026-10-14 08:45", 2003.0), stats)
    assert fired == []
    assert runner._pending is None


def test_runner_confirm_after_window_lapse_does_not_fire(tmp_path):
    # Arm, H1 close lapses the window, THEN the confirming entry
    # bars arrive on a tick outside the zone: no fire.  A later
    # re-touch is a NEW setup, not this one.
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    assert runner._pending is not None
    runner._on_h1_close(_tick("2026-10-14 09:00", 2003.0), stats)
    assert runner._pending is None  # window lapsed
    assert stats["invalidated"] == 1
    fired = []
    runner._run_live_signal = lambda pend, tick, st: fired.append(
        pend["trigger"])
    new_m15 = _frames(m15_n=24, shift_hours=-2)["M15"].iloc[20:23]

    class FakeFeed:
        def fetch_new_all(self, cursors):
            empty = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume",
                         "real_volume", "spread"])
            return {"M15": new_m15, "H1": empty, "M30": empty,
                    "M5": empty, "H4": empty, "D1": empty}

    runner.feed = FakeFeed()
    runner._refresh_bars(_tick("2026-10-14 09:00", 2003.0), stats)
    _eval(runner, _tick("2026-10-14 09:00", 2003.0), stats)
    assert fired == []
    assert runner._pending is None


def _flat_m15_feed(bars):
    """FakeFeed delivering ``bars`` (M15) and nothing else."""
    class FakeFeed:
        def fetch_new_all(self, cursors):
            empty = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume",
                         "real_volume", "spread"])
            return {"M15": bars, "H1": empty, "M30": empty,
                    "M5": empty, "H4": empty, "D1": empty}
    return FakeFeed()


def _quiet_bars(times):
    """M15 bars that neither confirm nor invalidate: entirely in
    the gap between the invalidation level (≈1997.68) and the zone
    bottom (≈1997.83) — above invalidation (no Close Rule), below
    the zone (no overlap, no confirm).  For TTL tests: bars that
    close without confirmation and without failing the setup."""
    return _bars(times, [1997.80] * len(times), [1997.70] * len(times),
                 last_close=1997.75)


def test_runner_armed_setup_expires_after_ttl_without_confirmation(tmp_path):
    # Armed TTL (owner rule, 2026-10-15): two M15 closes without
    # confirmation (armed_ttl_bars = 2, mirroring score_decay_bars)
    # disarm the touch as stale — price lingered in the zone
    # without reacting (absorbed, edge decayed).  The quiet bars
    # close without confirmation and without failing the setup
    # (above invalidation, below the zone).
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    assert runner._pending is not None
    assert runner._pending["armed_bars"] == 0
    flat = _quiet_bars(pd.date_range("2026-10-14 08:00", periods=2,
                                     freq="15min", tz="UTC"))
    runner.feed = _flat_m15_feed(flat)
    runner._refresh_bars(_tick("2026-10-14 08:30", 2000.3), stats)
    assert runner._pending is None  # disarmed as stale
    assert stats["invalidated"] == 1
    journal = _journal_text(tmp_path)
    assert "trigger invalidated (stale)" in journal
    assert "trigger fired" not in journal
    # The dead zone is blocked until price exits and re-enters.
    zone = runner._live_setup["zone"]
    assert runner._stale_zone == (1, zone["bottom"], zone["top"])


def test_runner_confirm_after_ttl_expiry_does_not_fire(tmp_path):
    # Arm, expire as stale, THEN the confirming (breaker) bars
    # arrive on a tick outside the zone: no fire.  A confirmation
    # firing on a dead setup would trigger on stale context.
    # (The breaker bars are the same ones that confirm in
    # test_runner_entry_confirms_on_m15_close — they WOULD fire
    # if the pending were still armed.)
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    assert runner._pending is not None
    flat = _quiet_bars(pd.date_range("2026-10-14 08:00", periods=2,
                                     freq="15min", tz="UTC"))
    runner.feed = _flat_m15_feed(flat)
    runner._refresh_bars(_tick("2026-10-14 08:30", 2000.3), stats)
    assert runner._pending is None  # stale
    fired = []
    runner._run_live_signal = lambda pend, tick, st: fired.append(
        pend["trigger"])
    breaker = _frames(m15_n=24, shift_hours=-2)["M15"].iloc[22:23]
    runner.feed = _flat_m15_feed(breaker)
    runner._refresh_bars(_tick("2026-10-14 08:45", 2003.0), stats)
    _eval(runner, _tick("2026-10-14 08:45", 2003.0), stats)
    assert fired == []
    assert runner._pending is None


def test_runner_rearm_only_after_exit_and_reentry(tmp_path):
    # After a stale disarm, ticks still inside the zone do NOT
    # re-arm (or the TTL is meaningless).  Only exit + re-entry
    # lifts the block.
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    assert runner._pending is not None
    flat = _quiet_bars(pd.date_range("2026-10-14 08:00", periods=2,
                                     freq="15min", tz="UTC"))
    runner.feed = _flat_m15_feed(flat)
    runner._refresh_bars(_tick("2026-10-14 08:30", 2000.3), stats)
    assert runner._pending is None  # stale
    assert runner._stale_zone is not None
    # Still inside the zone: no re-arm.
    _eval(runner, _tick("2026-10-14 08:35", 2000.5), stats)
    assert runner._pending is None
    assert runner._stale_zone is not None
    # Exit the zone: the block lifts (no arm — price is outside).
    _eval(runner, _tick("2026-10-14 08:36", 2003.0), stats)
    assert runner._pending is None
    assert runner._stale_zone is None
    # Re-enter: arms again as a NEW touch.
    _eval(runner, _tick("2026-10-14 08:37", 2000.3), stats)
    assert runner._pending is not None
    assert runner._pending["armed"] is True
    assert runner._pending["armed_bars"] == 0
    journal = _journal_text(tmp_path)
    assert "trigger invalidated (stale)" in journal
    assert "trigger fired" not in journal


def test_runner_ttl_counts_one_bar_at_a_time(tmp_path):
    # One M15 close without confirmation: armed_bars = 1, still
    # armed.  The TTL binds at the SECOND close, not the first.
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    assert runner._pending is not None
    flat = _quiet_bars(pd.date_range("2026-10-14 08:00", periods=1,
                                     freq="15min", tz="UTC"))
    runner.feed = _flat_m15_feed(flat)
    runner._refresh_bars(_tick("2026-10-14 08:15", 2000.3), stats)
    assert runner._pending is not None  # still armed after 1 bar
    assert runner._pending["armed_bars"] == 1
    assert stats.get("invalidated", 0) == 0
    # The second close without confirmation expires it.
    flat2 = _quiet_bars(pd.date_range("2026-10-14 08:15", periods=1,
                                      freq="15min", tz="UTC"))
    runner.feed = _flat_m15_feed(flat2)
    runner._refresh_bars(_tick("2026-10-14 08:30", 2000.3), stats)
    assert runner._pending is None
    assert stats["invalidated"] == 1
    assert "trigger invalidated (stale)" in _journal_text(tmp_path)


def test_runner_h1_close_expires_pending_without_erasing(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    _eval(runner, _tick("2026-10-14 10:05", 2000.3), stats)
    assert runner._pending is not None
    assert runner._pending["armed"] is True
    runner._on_h1_close(_tick("2026-10-14 11:00", 2001.5), stats)
    assert runner._pending is None
    assert stats["invalidated"] == 1
    journal = _journal_text(tmp_path)
    assert "h1_close_without_entry_confirm" in journal
    assert "trigger fired" not in journal  # never fired: logged, never erased
    # The new forming bar starts with the boundary tick.
    assert runner._forming["open"] == pytest.approx(2001.5)
    assert runner._forming["open_time"] == pd.Timestamp(
        "2026-10-14 11:00", tz="UTC")
    # Volume starts at zero: the tick evaluator owns accumulation,
    # so the boundary tick is counted exactly once (there).
    assert runner._forming["volume"] == 0.0
    _eval(runner, _tick("2026-10-14 11:00", 2001.5), stats)
    assert runner._forming["volume"] == pytest.approx(5.0)


def test_runner_h1_close_does_not_gate_evaluation(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    # An H1 close with no touch: nothing arms, nothing breaks.
    runner._on_h1_close(_tick("2026-10-14 11:00", 2000.3), stats)
    assert runner._pending is None
    # Evaluation continues on the NEW bar: the touch arms mid-bar.
    _eval(runner, _tick("2026-10-14 11:05", 2000.3), stats)
    _eval(runner, _tick("2026-10-14 11:06", 2001.8), stats)
    assert runner._pending is not None
    assert runner._pending["armed"] is True
    assert runner._pending["trigger"] == "zone_touch"


def test_runner_forming_bar_tracks_while_holding(tmp_path):
    """The forming bar is market-data tracking: it keeps its
    intrabar high/low while a position is open.  Triggers DO form
    while holding (the layering pipeline) — but no NEW position
    opens: a confirming setup while holding becomes a layer, never
    a second position."""
    from goldenswing.execution.executor import OrderRequest
    runner = _runner(tmp_path)
    stats: dict = {}
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    runner.tick_manager.open(ticket=101, request=req, fill_price=2000.0,
                             entry_bar=6)
    _eval(runner, _tick("2026-10-14 10:05", 2000.3), stats)
    _eval(runner, _tick("2026-10-14 10:06", 2001.8), stats)
    # Forming bar tracked the excursion; the touch armed into the
    # layering pipeline — but no second position opened.
    assert runner._forming["high"] == pytest.approx(2001.8)
    assert runner._pending is not None
    assert runner._pending["armed"] is True
    assert runner._pending["trigger"] == "zone_touch"
    assert runner.tick_manager.tickets() == [101]
    assert runner.executor.orders == []
    # Position closes: the armed touch is still live and can
    # confirm into a fresh position.
    runner.tick_manager.remove(101)
    _eval(runner, _tick("2026-10-14 10:07", 2001.9), stats)
    assert runner._pending is not None
    assert runner._pending["armed"] is True
    assert runner._pending["trigger"] == "zone_touch"


def test_runner_bias_flip_invalidates_pending(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    _eval(runner, _tick("2026-10-14 10:05", 2000.3), stats)
    assert runner._pending is not None
    # D1/H4 close flips the bias: the pending premise is gone.
    runner._live_setup = {**runner._live_setup, "direction": -1}
    runner._check_pending_bias(stats)
    assert runner._pending is None
    assert stats["invalidated"] == 1
    assert "trigger invalidated (bias_flipped)" in _journal_text(tmp_path)


def test_runner_live_verdict_shape(tmp_path):
    runner = _runner(tmp_path, m15_n=24)
    h1_closed = runner._stack.frames["H1"]
    pend = {"trigger": "BOS_bull", "direction": 1, "entry": 2001.8,
            "stop": 1991.85, "target": 2021.7,
            "zone": runner._live_setup["zone"],
            "entry_clock": "M15", "h1_open": FORMING_OPEN}
    from goldenswing.features.engine import read_and_featurize
    h1 = h1_closed  # verdict needs states aligned to the closed frame
    states = read_and_featurize(h1, None)
    verdict = runner._live_verdict(
        pend, _tick("2026-10-14 10:46", 2001.9), h1_closed, states,
        len(h1) - 1)
    assert verdict["allowed"] is True
    assert verdict["blocked_leg"] == ""
    assert all(leg["passed"] for leg in verdict["legs"].values())
    assert verdict["direction"] == 1
    assert verdict["entry"] == pytest.approx(2001.9)  # confirm tick
    assert verdict["stop"] == pytest.approx(1991.85)
    assert verdict["risk"] == pytest.approx(abs(2001.9 - 1991.85))
    assert verdict["veto_enabled"] is False
    assert "volume" in verdict and "qm" in verdict
    assert verdict["entry_clock"] in ("M15", "M30")


def test_runner_full_live_path_places_order(tmp_path):
    """Touch (tick) + POI confirm (already-closed composing bars) ->
    the signal path runs and places the order: the whole live
    clock, end to end, with no H1 close involved.  Only ONE tick:
    the touch arms and the immediate confirm shot fires on the
    already-closed breaker bar — a second tick would re-arm into
    the layering pipeline, which is a different test.
    """
    runner = _runner(tmp_path, m15_n=23, shift_hours=-2)
    stats: dict = {"ticks": 0, "bars": 0, "signals": 0, "executed": 0,
                   "shadow": 0, "gated": 0, "hierarchy_blocked": 0,
                   "tick_actions": 0, "reconciled_closed": 0,
                   "triggers": 0, "invalidated": 0}
    # _place_live_order reads the account for sizing.
    runner.feed = SimpleNamespace(
        account_state=lambda: {"equity": 10000.0, "balance": 10000.0,
                               "free_margin": 10000.0},
        order_margin=lambda *, volume, price, direction:
            float(volume) * 100.0 * float(price) / 500.0)

    sig = SimpleNamespace(
        score=80.0, direction=1, index=len(runner._stack.frames["H1"]),
        entry_ref=2001.8, invalidation=1991.85, targets=[2021.7],
        atr=1.47, factors=[], kind="fvg_limit")

    class StubPlaybook:
        name = "stub"
        live = True
        session_scope = "any"
        hunting_ground = SimpleNamespace(excluded=[], bonuses={})

        def live_signal(self, states, t, h1):
            return [sig]

        def check_live_signal(self, signal, ctx):
            return True, ""

    runner._get_playbooks = lambda: [StubPlaybook()]
    # The touch tick: entry bars already closed -> confirm shot
    # fires immediately -> signal path -> order.
    # The signal's entry_ref (2001.8) is AHEAD of the confirm
    # price (2000.3), so setup-dependent order selection
    # (select_order_type) places a PENDING STOP at the level —
    # not a market fill.  ``executed`` counts fills only (market
    # fills + pending fills); a resting STOP increments
    # ``pending`` and fills only when the market reaches it.
    _eval(runner, _tick("2026-10-14 08:05", 2000.3), stats)
    assert stats["triggers"] == 1
    assert stats["signals"] == 1
    assert stats.get("pending") == 1
    assert stats["executed"] == 0  # resting STOP, not yet filled
    assert len(runner.executor.orders) == 1
    assert runner.executor.orders[0]["direction"] == 1
    assert runner._pending is None  # consumed by the confirmation
    journal = _journal_text(tmp_path)
    assert "trigger fired bos" in journal
    assert "mt5 pending: STOP" in journal
