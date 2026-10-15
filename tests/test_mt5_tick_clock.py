"""MT5 engine: the H1 close is NOT the signal clock.

The trigger (BOS/MSCH/CHoCH) fires on TICK-cross of the trigger level;
the entry leg evaluates on ENTRY-TF closes (M15 in killzone, M30
elsewhere); H1/H4/D1 closes only refresh their legs (bias / zones /
trigger levels) — they never gate evaluation.  The hierarchy legs and
gate formulas are unchanged (same functions, same configs) — only the
EVALUATION CLOCK changed.  Frames below are hand-built so the setup is
armed: D1/H4 bias +1, H4 support 2000.0, H1 ref_high 2001.0, and the
M15 breaker bar (10:30) confirms the entry.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.live.hierarchy import (TF_MINUTES, TimeframeStack,
                                        check_live_entry,
                                        check_live_trigger, live_setup)

H1_HIGH = [1992, 1993, 1994, 1995, 1996, 1996, 1996, 1997, 1997, 1997,
           1997, 1997, 1998, 1997, 1997, 1996, 1996, 1996, 1997, 1997,
           1998, 1999, 2000, 2000, 2000, 2000, 2001, 2000, 2000, 2000.5]
H1_LOW = [1991, 1992, 1993, 1994, 1995, 1990, 1995, 1996, 1996, 1996,
          1996, 1996, 1997, 1996, 1996, 1995, 1995, 1995, 1996, 1992,
          1997, 1998, 1999, 1999, 1999, 1999, 2000, 1999, 1999, 1999.5]
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
            "tick_volume": 5, "real_volume": 0, "spread": 30}


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
        execute_playbooks=("stub",), **cfg_kw)
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
def test_live_setup_armed_from_closed_bars():
    setup = _setup()
    assert setup["direction"] == 1
    assert setup["bias_d1"] == 1 and setup["bias_h4"] == 1
    assert setup["zone"]["kind"] == "support"
    assert setup["zone"]["level"] == pytest.approx(2000.0)
    assert setup["ref_high"] == pytest.approx(2001.0)
    assert setup["ref_low"] == pytest.approx(1992.0)
    assert setup["atr_h1"] > 0 and setup["buf"] > 0


def test_live_setup_not_armed_when_h4_disagrees():
    frames = _frames()
    # Mirror H4 around 2000: uptrend becomes downtrend (bias -1).
    h4 = frames["H4"]
    frames["H4"] = pd.DataFrame(
        {"open": 4000.0 - h4["close"], "high": 4000.0 - h4["low"],
         "low": 4000.0 - h4["high"], "close": 4000.0 - h4["open"],
         "volume": h4["volume"], "real_volume": h4["real_volume"],
         "spread": h4["spread"]}, index=h4.index)
    assert live_setup(frames) is None


# ------------------------------------------------------- check_live_trigger
def test_trigger_fires_intrabar_on_level_cross():
    setup = _setup()
    trig = check_live_trigger(
        setup, open_=2000.3, high=2001.8, low=2000.3, close=2001.8,
        closed_h1=_frames()["H1"])
    assert trig is not None
    assert trig["trigger"] == "BOS_bull"
    assert trig["direction"] == 1
    assert trig["entry"] == pytest.approx(2001.8)
    assert trig["stop"] == pytest.approx(1991.85, abs=0.05)
    assert trig["level"] == pytest.approx(2001.0)  # crossed level
    # Target is the 2R floor (nearest H4 pool is inside 2R).
    assert trig["target"] == pytest.approx(
        2001.8 + 2 * trig["risk"], abs=1e-6)


def test_trigger_does_not_fire_below_level_or_outside_zone():
    setup = _setup()
    h1 = _frames()["H1"]
    # Below the trigger level: no break.
    assert check_live_trigger(
        setup, open_=2000.3, high=2000.6, low=2000.3, close=2000.6,
        closed_h1=h1) is None
    # Past the level but outside the POI band (above zone top 2002.17).
    assert check_live_trigger(
        setup, open_=2000.3, high=2003.0, low=2000.3, close=2003.0,
        closed_h1=h1) is None


# -------------------------------------------------------- check_live_entry
def test_entry_evaluates_on_entry_tf_closes():
    m15 = _frames()["M15"]
    # Full frame: the 10:30 breaker confirms.
    assert check_live_entry(m15, 1, 3, since=FORMING_OPEN,
                            entry_clock="M15") == 1
    # No composing bars yet (frame ends 09:45): no confirmation.
    assert check_live_entry(m15.iloc[:20], 1, 3, since=FORMING_OPEN,
                            entry_clock="M15") is None
    # Composing bars through 10:15 (breaker not closed yet): no event.
    assert check_live_entry(m15.iloc[:22], 1, 3, since=FORMING_OPEN,
                            entry_clock="M15") is None


# ------------------------------------------------------------- runner clock
def test_runner_trigger_fires_on_tick_not_h1_close(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    # No H1 close, no refresh: ticks alone fire the trigger.
    runner._evaluate_live_tick(_tick("2026-10-14 10:05", 2000.3), stats)
    assert runner._pending is None  # forming bar opens at the zone
    runner._evaluate_live_tick(_tick("2026-10-14 10:06", 2001.8), stats)
    assert runner._pending is not None
    assert runner._pending["trigger"] == "BOS_bull"
    assert runner._pending["direction"] == 1
    assert stats["triggers"] == 1
    # Entry leg has no composing bars yet: pending waits, no order.
    assert runner.executor.orders == []
    assert "trigger fired BOS_bull" in _journal_text(tmp_path)


def test_runner_entry_confirms_on_m15_close(tmp_path):
    # Shift -2h: forming bar 08:00, inside the London killzone, so the
    # session entry clock is M15 (at 10:00 it would be M30).
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    runner._evaluate_live_tick(_tick("2026-10-14 08:05", 2000.3), stats)
    runner._evaluate_live_tick(_tick("2026-10-14 08:06", 2001.8), stats)
    assert runner._pending is not None
    assert runner._pending["entry_clock"] == "M15"
    fired = []

    def _record(pend, tick, st):
        fired.append(pend["trigger"])

    runner._run_live_signal = _record
    # M15 close at 08:45 (the 08:30 breaker bar closes): the feed
    # delivers the composing bars; the entry leg confirms.
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
    assert fired == ["BOS_bull"]  # signal path ran on the M15 close
    assert runner._pending is None  # consumed by the confirmation


def test_runner_entry_confirms_on_m15_close_at_h1_boundary(tmp_path):
    """An M15 close coinciding with the H1 close still confirms.

    The last composing M15 bar (08:45) closes exactly when its H1
    bar (08:00) does.  The entry leg must get its confirmation shot
    BEFORE the H1-close expiry — the H1 close must not gate (and
    kill) the evaluation.  The stack iterates H1 before M15, so
    per-TF dict order would expire the pending first without the
    phased refresh.
    """
    runner = _runner(tmp_path, m15_n=20, shift_hours=-2)
    stats: dict = {"bars": 0}
    runner._evaluate_live_tick(_tick("2026-10-14 08:05", 2000.3), stats)
    runner._evaluate_live_tick(_tick("2026-10-14 08:06", 2001.8), stats)
    assert runner._pending is not None
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
    assert fired == ["BOS_bull"]  # entry confirmed, not expired
    assert runner._pending is None  # consumed by the confirmation
    assert stats.get("invalidated", 0) == 0
    assert "h1_close_without_entry_confirm" not in _journal_text(tmp_path)


def test_runner_reversal_invalidates_pending_without_erasing(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    runner._evaluate_live_tick(_tick("2026-10-14 10:05", 2000.3), stats)
    runner._evaluate_live_tick(_tick("2026-10-14 10:06", 2001.8), stats)
    assert runner._pending is not None
    # Price back across the crossed level (2001.0): the break failed.
    runner._evaluate_live_tick(_tick("2026-10-14 10:07", 2000.9), stats)
    assert runner._pending is None
    assert stats["invalidated"] == 1
    journal = _journal_text(tmp_path)
    assert "trigger fired BOS_bull" in journal  # logged, never erased
    assert "trigger invalidated (reversal)" in journal


def test_runner_h1_close_expires_pending_without_erasing(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    runner._evaluate_live_tick(_tick("2026-10-14 10:05", 2000.3), stats)
    runner._evaluate_live_tick(_tick("2026-10-14 10:06", 2001.8), stats)
    assert runner._pending is not None
    runner._on_h1_close(_tick("2026-10-14 11:00", 2001.5), stats)
    assert runner._pending is None
    assert stats["invalidated"] == 1
    journal = _journal_text(tmp_path)
    assert "trigger fired BOS_bull" in journal
    assert "h1_close_without_entry_confirm" in journal
    # The new forming bar starts with the boundary tick.
    assert runner._forming["open"] == pytest.approx(2001.5)
    assert runner._forming["open_time"] == pd.Timestamp(
        "2026-10-14 11:00", tz="UTC")
    # Volume starts at zero: the tick evaluator owns accumulation,
    # so the boundary tick is counted exactly once (there).
    assert runner._forming["volume"] == 0.0
    runner._evaluate_live_tick(_tick("2026-10-14 11:00", 2001.5), stats)
    assert runner._forming["volume"] == pytest.approx(5.0)


def test_runner_h1_close_does_not_gate_evaluation(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    # An H1 close with no crossing: nothing fires, nothing breaks.
    runner._on_h1_close(_tick("2026-10-14 11:00", 2000.3), stats)
    assert runner._pending is None
    # Evaluation continues on the NEW bar: the trigger fires mid-bar.
    runner._evaluate_live_tick(_tick("2026-10-14 11:05", 2000.3), stats)
    runner._evaluate_live_tick(_tick("2026-10-14 11:06", 2001.8), stats)
    assert runner._pending is not None
    assert runner._pending["trigger"] == "BOS_bull"


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
    runner._evaluate_live_tick(_tick("2026-10-14 10:05", 2000.3), stats)
    runner._evaluate_live_tick(_tick("2026-10-14 10:06", 2001.8), stats)
    # Forming bar tracked the excursion; the trigger fired into the
    # layering pipeline — but no second position opened.
    assert runner._forming["high"] == pytest.approx(2001.8)
    assert runner._pending is not None
    assert runner._pending["trigger"] == "BOS_bull"
    assert runner.tick_manager.tickets() == [101]
    assert runner.executor.orders == []
    # Position closes: the pending setup is still live and can
    # confirm into a fresh position.
    runner.tick_manager.remove(101)
    runner._evaluate_live_tick(_tick("2026-10-14 10:07", 2001.9), stats)
    assert runner._pending is not None
    assert runner._pending["trigger"] == "BOS_bull"


def test_runner_bias_flip_invalidates_pending(tmp_path):
    runner = _runner(tmp_path)
    stats: dict = {}
    runner._evaluate_live_tick(_tick("2026-10-14 10:05", 2000.3), stats)
    runner._evaluate_live_tick(_tick("2026-10-14 10:06", 2001.8), stats)
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
    """Trigger (tick) + entry (already-closed composing bars) -> the
    signal path runs and places the order: the whole live clock,
    end to end, with no H1 close involved."""
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
    runner._evaluate_live_tick(_tick("2026-10-14 08:05", 2000.3), stats)
    # The trigger tick: entry bars already closed -> entry confirms
    # immediately -> signal path -> order.
    runner._evaluate_live_tick(_tick("2026-10-14 08:06", 2001.8), stats)
    assert stats["triggers"] == 1
    assert stats["signals"] == 1
    assert stats["executed"] == 1
    assert len(runner.executor.orders) == 1
    assert runner.executor.orders[0]["direction"] == 1
    assert runner._pending is None  # consumed by the confirmation
    journal = _journal_text(tmp_path)
    assert "trigger fired BOS_bull" in journal
