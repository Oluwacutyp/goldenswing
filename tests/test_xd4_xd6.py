"""XD-4 + XD-6 burn-down pins (cross-TF spec extension §A.3/XD rows).

XD-4 — management-clock inheritance: a position is managed on
its OWN clock's closes, and profile windows convert wall-time
-faithfully (the old table mapped H4/D1 to 1 bar-per-H1, so a
D1 swing's 8-trigger-bar window became 8 H1 bars = 8 hours).
XD-6 — forming-bar asymmetry: the live signal path's
generators read CLOSED bars only; the forming bar is labelled
reaction context, never a generator input (the old code
concatenated it into the playbook/feature/ATR frame).

Offline throughout: synthetic frames + fake executor + the
tick-clock harness shape (MT5Runner with feed=None).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.features.engine import read_and_featurize
from goldenswing.live.hierarchy import TimeframeStack
from goldenswing.live.management_profiles import (
    elapsed_h1_bars, mgmt_clock_due, profile_for)


# ------------------------------------------------------------ helpers
def _frame(rows, start, freq):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [100.0] * len(rows), "real_volume": [0.0] * len(rows),
         "spread": [30] * len(rows)}, index=idx)


def _frames():
    base = [(2000.0 + 0.1 * i, 2001.0 + 0.1 * i,
             1999.0 + 0.1 * i, 2000.5 + 0.1 * i) for i in range(60)]
    h1 = _frame(base, "2026-01-05 00:00", "h")
    m5 = _frame(base * 2, "2026-01-05 00:00", "5min")
    m15 = _frame(base, "2026-01-05 00:00", "15min")
    h4 = _frame(base[:30], "2026-01-05 00:00", "4h")
    d1 = _frame(base[:30], "2026-01-05 00:00", "D")
    return {"D1": d1, "H4": h4, "H1": h1, "M15": m15, "M5": m5}


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
        self.orders.append({"ticket": self._ticket})
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


def _runner(tmp_path):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("stub",),
        news_pause_enabled=False)
    runner = MT5Runner(cfg, feed=None, executor=FakeExecutor())
    runner._stack = TimeframeStack.from_frames(_frames())
    return runner


def _ctx(runner, clock, entry_mgmt, last_mgmt):
    return {
        "profile": profile_for(style="trend", clock_tf=clock),
        "progress_trimmed": False, "progress_trim_bar": None,
        "drift_trimmed": False, "preprint_key": None,
        "zone_banked": False, "adr_banked": False,
        "defended_stop": None, "entry_total": None,
        "opposing_prox": None, "style_key": "intraday",
        "invalidation": 1990.0, "entry_bar": 10,
        "clock_tf": clock, "management_clock": clock,
        "entry_mgmt_bar": entry_mgmt, "last_mgmt_bar": last_mgmt,
    }


class _Pos:
    def __init__(self):
        self.direction = 1
        self.best_r = 5.0  # progress NOT due: gating, not trims
        self.closed = False
        self.entry_price = 2000.0
        self.volume = 0.02


class _TickMgr:
    def __init__(self, pos):
        self._pos = pos

    def tickets(self):
        return [11]

    def position(self, ticket):
        return self._pos


# ------------------------------------------------------------ XD-4 pure
def test_xd4_profile_windows_wall_time_faithful():
    # Wrong-clock case (pre-fix table): D1/H4 mapped to 1.
    assert profile_for(style="trend", clock_tf="D1") \
        .progress_h1_bars == 192  # 8 D1 bars * 24 H1 bars
    assert profile_for(style="trend", clock_tf="H4") \
        .progress_h1_bars == 32   # 8 H4 bars * 4 H1 bars
    # Unchanged clocks (no behaviour change for them).
    assert profile_for(style="trend", clock_tf="H1") \
        .progress_h1_bars == 8
    assert profile_for(style="scalp", clock_tf="M5") \
        .progress_h1_bars == 1    # 12 M5 bars = 1 H1 bar
    assert profile_for(style="trend", clock_tf="M15") \
        .progress_h1_bars == 2    # 8 M15 bars = 2 H1 bars
    assert profile_for(style="trend", clock_tf="D1",
                       late_confirm=True).progress_h1_bars == 96


def test_xd4_elapsed_conversion_and_h1_identity():
    # H1: identical to the old cur_bar - entry_bar.
    for entry, now in [(10, 10), (10, 15), (0, 100)]:
        assert elapsed_h1_bars(clock_tf="H1", entry_mgmt_bar=entry,
                               current_mgmt_bar=now) == now - entry
    # M5: 144 M5 bars = 12 H1 bars; partial hours floor.
    assert elapsed_h1_bars(clock_tf="M5", entry_mgmt_bar=100,
                           current_mgmt_bar=244) == 12
    assert elapsed_h1_bars(clock_tf="M5", entry_mgmt_bar=100,
                           current_mgmt_bar=106) == 0
    # D1: 8 D1 bars = 192 H1 bars.
    assert elapsed_h1_bars(clock_tf="D1", entry_mgmt_bar=100,
                           current_mgmt_bar=108) == 192
    # Unknown clock falls back to H1 units, never a skip.
    assert elapsed_h1_bars(clock_tf="W1", entry_mgmt_bar=3,
                           current_mgmt_bar=9) == 6


def test_xd4_clock_due_gate():
    assert mgmt_clock_due(clock_tf="D1", last_mgmt_bar=None,
                          current_mgmt_bar=5) is True
    assert mgmt_clock_due(clock_tf="D1", last_mgmt_bar=5,
                          current_mgmt_bar=5) is False
    assert mgmt_clock_due(clock_tf="D1", last_mgmt_bar=5,
                          current_mgmt_bar=6) is True
    assert mgmt_clock_due(clock_tf="M5", last_mgmt_bar=100,
                          current_mgmt_bar=100) is False


# ------------------------------------------------------------ XD-4 runner
def test_xd4_registration_inherits_route_clock(tmp_path):
    runner = _runner(tmp_path)
    m5_len = len(runner._stack.frames["M5"])
    runner._signal_ctx = {"a1": {
        "playbook": "qm_m5", "clock_tf": "M5", "floor": 50.0,
        "entry_total": 60.0, "zone_d": None, "style_key": "scalp",
        "style_bucket": "MICRO_SCALP", "family": "scalp",
        "defended_stop": None}}
    ctx = runner._register_position_ctx(
        11, alert_id="a1", entry=2000.0, stop=1990.0, direction=1)
    assert ctx["management_clock"] == "M5"
    assert ctx["clock_tf"] == "M5"
    assert ctx["entry_mgmt_bar"] == m5_len
    assert ctx["last_mgmt_bar"] == m5_len - 1
    # Default (no parked clock): H1, as before.
    runner._signal_ctx = {"a2": {
        "playbook": "sweep_reversal", "floor": 50.0,
        "entry_total": 60.0, "zone_d": None,
        "defended_stop": None}}
    ctx2 = runner._register_position_ctx(
        12, alert_id="a2", entry=2000.0, stop=1990.0, direction=1)
    assert ctx2["management_clock"] == "H1"
    assert ctx2["entry_mgmt_bar"] == len(runner._stack.frames["H1"])


def test_xd4_management_gated_by_own_clock(tmp_path):
    runner = _runner(tmp_path)
    runner._last_tick = {"mid": 2000.0}
    runner.tick_manager = _TickMgr(_Pos())
    d1 = runner._stack.frames["D1"]
    d1_now = len(d1) - 1
    h1_now = len(runner._stack.frames["H1"]) - 1
    # (a) D1 position, no new D1 bar: the H1 pass SKIPS it.
    ctx = _ctx(runner, "D1", entry_mgmt=5, last_mgmt=d1_now)
    runner._position_ctx = {11: ctx}
    runner._manage_positions_bar_close({})
    assert ctx["last_mgmt_bar"] == d1_now  # untouched: skipped
    assert runner.executor.orders == []
    # (b) D1 position, one new D1 bar: the pass RUNS it.
    last = runner._stack.frames["D1"].iloc[[-1]].copy()
    last.index = last.index + pd.Timedelta(days=1)
    runner._stack.frames["D1"] = pd.concat(
        [runner._stack.frames["D1"], last])
    runner._manage_positions_bar_close({})
    assert ctx["last_mgmt_bar"] == d1_now + 1
    # (c) Clock filter: an H1 position is untouched by an M5 pass.
    ctx_h1 = _ctx(runner, "H1", entry_mgmt=5, last_mgmt=h1_now)
    runner._position_ctx = {11: ctx_h1}
    runner._manage_positions_bar_close({}, clocks=("M5",))
    assert ctx_h1["last_mgmt_bar"] == h1_now
    # (d) ...but an M5 position IS run by that same M5 pass.
    m5_now = len(runner._stack.frames["M5"]) - 1
    ctx_m5 = _ctx(runner, "M5", entry_mgmt=5, last_mgmt=m5_now - 1)
    runner._position_ctx = {11: ctx_m5}
    runner._manage_positions_bar_close({}, clocks=("M5",))
    assert ctx_m5["last_mgmt_bar"] == m5_now


def test_xd4_h1_position_pass_unchanged(tmp_path):
    # H1 positions: every H1 close is due, elapsed identical
    # to the old arithmetic (pure identity covered above; here
    # the pass itself runs and advances the watermark).
    runner = _runner(tmp_path)
    runner._last_tick = {"mid": 2000.0}
    runner.tick_manager = _TickMgr(_Pos())
    h1_now = len(runner._stack.frames["H1"]) - 1
    ctx = _ctx(runner, "H1", entry_mgmt=5, last_mgmt=h1_now - 1)
    runner._position_ctx = {11: ctx}
    runner._manage_positions_bar_close({})
    assert ctx["last_mgmt_bar"] == h1_now
    assert runner.executor.orders == []


# ------------------------------------------------------------ XD-6
def test_xd6_live_signal_generators_see_closed_bars_only(tmp_path):
    runner = _runner(tmp_path)
    h1_closed = runner._stack.frames["H1"]
    forming_open = h1_closed.index[-1] + pd.Timedelta(hours=1)
    runner._forming = {
        "open": 2100.0, "high": 2200.0, "low": 1900.0,
        "open_time": forming_open, "volume": 999.0,
        "real_volume": 0.0, "spread": 30}
    tick = {"time": forming_open.to_pydatetime(), "mid": 2150.0,
            "bid": 2150.0, "ask": 2150.3, "tick_volume": 5,
            "real_volume": 0, "spread": 0.30}
    pend = {"direction": 1, "stop": 1990.0, "trigger": "test-confirm",
            "entry_clock": "H1",
            "zone": {"level": 2000.0, "top": 2001.0,
                     "bottom": 1999.0},
            "target": 2010.0, "h1_open": h1_closed.index[-1],
            "setup": {}}
    captured = {}

    # Step 14: _run_live_signal also passes lifecycle_routing —
    # the capture accepts (and ignores) extra kwargs.
    def _capture(h1, t, states, years, verdict, stats, forming=None,
                 **_kw):
        captured["h1"] = h1
        captured["t"] = t
        captured["verdict"] = verdict
        captured["forming"] = forming

    runner._run_signal_path = _capture
    runner._run_live_signal(pend, tick, {})
    assert captured, "signal path was not reached"
    # Generators got the CLOSED frame: last bar is the last
    # closed bar, and the forming timestamp is nowhere in it.
    assert captured["h1"].index[-1] == h1_closed.index[-1]
    assert forming_open not in captured["h1"].index
    assert captured["t"] == len(h1_closed) - 1
    # ...while the forming bar is present, labelled, as
    # reaction context only.
    assert captured["forming"] is not None
    assert captured["forming"]["forming"] is True
    assert captured["forming"]["close"] == 2150.0
    assert captured["verdict"]["forming"]["forming"] is True
    journal = (tmp_path / "mt5_journal.jsonl").read_text() \
        if (tmp_path / "mt5_journal.jsonl").exists() else ""
    assert "XD-6" in journal


def test_xd6_old_path_was_contaminated_and_score_is_causal(tmp_path):
    runner = _runner(tmp_path)
    h1_closed = runner._stack.frames["H1"]
    forming_open = h1_closed.index[-1] + pd.Timedelta(hours=1)
    row = pd.DataFrame([{
        "open": 2100.0, "high": 2200.0, "low": 1900.0,
        "close": 2150.0, "volume": 999.0, "real_volume": 0.0,
        "spread": 30}],
        index=pd.DatetimeIndex([forming_open], tz="UTC"))
    # Mechanism proof: the OLD concatenated frame featurizes
    # to one MORE bar than the closed frame — the forming bar
    # entered generator inputs as if it were a closed bar.
    states_closed = read_and_featurize(h1_closed, None)
    states_old = read_and_featurize(pd.concat([h1_closed, row]),
                                    None)
    assert len(states_old) == len(states_closed) + 1
    # Score causality pin: the Asymmetry score reads the stack
    # frames (closed only), so a forming bar present in
    # runner._forming cannot move it.
    runner._forming = None
    s0 = runner._score_for(direction=1, verdict=None, price=2000.0,
                           zone_d=None, atr=1.0, style_key="intraday",
                           playbook="sweep_reversal")
    runner._forming = {
        "open": 2100.0, "high": 2200.0, "low": 1900.0,
        "open_time": forming_open, "volume": 999.0,
        "real_volume": 0.0, "spread": 30}
    s1 = runner._score_for(direction=1, verdict=None, price=2000.0,
                           zone_d=None, atr=1.0, style_key="intraday",
                           playbook="sweep_reversal")
    assert s0.total == s1.total
