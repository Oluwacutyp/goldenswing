"""Feature 4: trade-count caps + cooldowns + fingerprint dedup,
plus the Feature 5 pure time helpers (session multiplier, Friday
close) — wired in their own commit, tested here.
"""
from datetime import datetime, timezone

import pytest

from goldenswing.execution.entry_gates import (
    DEFAULT_SESSION_MULTIPLIERS, EntryGate, EntryGateConfig,
    is_friday_close, session_multiplier, signal_fingerprint)


# ------------------------------------------------------------ EntryGate
def _gate(**kw):
    now = [1_699_950_000.0]  # 2023-11-14 08:20:00 UTC (Tuesday)
    cfg = EntryGateConfig(**kw)
    return EntryGate(cfg, clock=lambda: now[0]), now


def test_caps_daily_and_hourly():
    gate, now = _gate(max_daily_trades=2, max_hourly_trades=1)
    assert gate.check_new_entry(now[0], "a")[0] is True
    gate.record_open(now[0], "a")
    # Hourly cap hit (1/hour).
    assert gate.check_new_entry(now[0], "b") == (False, "cap_hourly")
    # Next hour: hourly resets, daily still has room (1/2).
    now[0] += 3600
    assert gate.check_new_entry(now[0], "b")[0] is True
    gate.record_open(now[0], "b")
    # Next hour again: daily cap hit (2/2).
    now[0] += 3600
    assert gate.check_new_entry(now[0], "c") == (False, "cap_daily")
    # Next day: both reset.
    now[0] += 86400
    assert gate.check_new_entry(now[0], "c")[0] is True


def test_caps_zero_means_unlimited():
    gate, now = _gate()
    for i in range(50):
        assert gate.check_new_entry(now[0], f"fp{i}")[0] is True
        gate.record_open(now[0], f"fp{i}")
    assert gate.check_new_entry(now[0], "fp50")[0] is True


def test_cooldown_after_loss_not_after_win_by_default():
    gate, now = _gate(loss_cooldown_sec=300, win_cooldown_sec=0)
    gate.record_close(now[0], won=False)
    assert gate.check_new_entry(now[0] + 299, "a") == (False, "cooldown")
    assert gate.check_new_entry(now[0] + 300, "a")[0] is True
    gate.record_close(now[0] + 300, won=True)
    assert gate.check_new_entry(now[0] + 301, "b")[0] is True


def test_cooldown_after_win_when_configured():
    gate, now = _gate(loss_cooldown_sec=0, win_cooldown_sec=600)
    gate.record_close(now[0], won=True)
    assert gate.check_new_entry(now[0] + 599, "a") == (False, "cooldown")
    assert gate.check_new_entry(now[0] + 600, "a")[0] is True


def test_fingerprint_dedup_with_ttl():
    gate, now = _gate(fingerprint_ttl_sec=3600)
    gate.record_open(now[0], "fp")
    assert gate.check_new_entry(now[0] + 100, "fp") == (False, "duplicate")
    assert gate.check_new_entry(now[0] + 100, "other")[0] is True
    # After TTL the same fingerprint may fire again.
    assert gate.check_new_entry(now[0] + 3600, "fp")[0] is True


def test_gate_check_order_caps_before_cooldown_before_dedup():
    gate, now = _gate(max_daily_trades=1, loss_cooldown_sec=300,
                      fingerprint_ttl_sec=3600)
    gate.record_open(now[0], "fp")
    gate.record_close(now[0], won=False)
    # All three would refuse; the cap is reported first (stable
    # precedence: caps -> cooldown -> dedup).
    assert gate.check_new_entry(now[0] + 10, "fp") == (False, "cap_daily")


def test_signal_fingerprint_stable_and_distinct():
    a = signal_fingerprint("sweep_reversal", 1, 2005.0, 1990.0)
    assert a == signal_fingerprint("sweep_reversal", 1, 2005.0, 1990.0)
    assert a != signal_fingerprint("sweep_reversal", -1, 2005.0, 1990.0)
    assert a != signal_fingerprint("fvg_limit", 1, 2005.0, 1990.0)
    assert a != signal_fingerprint("sweep_reversal", 1, 2006.0, 1990.0)


# ------------------------------------------------- session / friday
def test_session_multiplier_uses_dst_aware_sessions():
    # Winter 2026-01-15: Asia 00-07Z, London 07-12Z, Overlap 12-16Z,
    # NY 16-21Z, Off 21-24Z (the module's documented winter buckets).
    assert session_multiplier(
        datetime(2026, 1, 15, 3, 0, tzinfo=timezone.utc)) == 0.8   # asia
    assert session_multiplier(
        datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc)) == 1.0   # london
    assert session_multiplier(
        datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)) == 1.2  # overlap
    assert session_multiplier(
        datetime(2026, 1, 15, 18, 0, tzinfo=timezone.utc)) == 1.1  # ny
    assert session_multiplier(
        datetime(2026, 1, 15, 22, 0, tzinfo=timezone.utc)) == 0.7  # off
    # Custom map overrides; missing session defaults to 1.0.
    assert session_multiplier(
        datetime(2026, 1, 15, 3, 0, tzinfo=timezone.utc),
        {"asia": 2.0}) == 2.0
    assert session_multiplier(
        datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc),
        {"asia": 2.0}) == 1.0


def test_is_friday_close():
    # 2026-10-16 is a Friday.
    assert is_friday_close(
        datetime(2026, 10, 16, 19, 59, tzinfo=timezone.utc)) is False
    assert is_friday_close(
        datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc)) is True
    assert is_friday_close(
        datetime(2026, 10, 16, 23, 0, tzinfo=timezone.utc)) is True
    # Thursday 20:00 is not Friday.
    assert is_friday_close(
        datetime(2026, 10, 15, 20, 0, tzinfo=timezone.utc)) is False
    # Naive datetimes are treated as UTC.
    assert is_friday_close(datetime(2026, 10, 16, 20, 0)) is True
    # Custom hour.
    assert is_friday_close(
        datetime(2026, 10, 16, 18, 0, tzinfo=timezone.utc),
        close_hour=18) is True


# ------------------------------------------------------- runner wiring
def test_runner_gate_refuses_duplicate_and_cooldown(tmp_path):
    from types import SimpleNamespace
    from goldenswing.execution.executor import OrderRequest
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    from goldenswing.execution.risk import gold_spec

    class FakeFeed:
        def account_state(self):
            return {"equity": 10000.0, "balance": 10000.0,
                    "free_margin": 10000.0}

    class FakeExecutor:
        def symbol_spec(self, symbol):
            return gold_spec()

        def place_order(self, req):
            return SimpleNamespace(ok=True, ticket=9, price=2005.0)

    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False, max_daily_trades=0,
        max_hourly_trades=0, loss_cooldown_sec=300,
        fingerprint_ttl_sec=3600)
    runner = MT5Runner(cfg, feed=FakeFeed(), executor=FakeExecutor())
    runner._market_open = lambda: (True, "open")
    runner._check_risk_guards = lambda *a: True
    runner._check_margin = lambda *a: True
    runner._alignment = SimpleNamespace(size_multiplier=1.0,
                                        alignment_label="D1_Only")
    runner._last_tick = {"mid": 2005.0, "bid": 2004.8, "ask": 2005.2}
    runner._stack = SimpleNamespace(frames={"H1": [1, 2, 3]})
    sig = SimpleNamespace(direction=1, entry_ref=2005.0,
                          invalidation=1990.0, atr=10.0,
                          targets=[2040.0], score=80.0, index=42,
                          factors=[])
    pb = SimpleNamespace(name="sweep_reversal")
    stats: dict = {}
    runner._place_live_order(pb, sig, stats)
    assert stats.get("executed") == 1
    # Same setup re-fires within TTL: dedup refuses.
    stats2: dict = {}
    runner._place_live_order(pb, sig, stats2)
    assert stats2.get("gate_duplicate") == 1
    assert stats2.get("executed", 0) == 0
    # A losing close starts the cooldown: a DIFFERENT setup is
    # refused for 300s.
    runner._record_close(9, 1995.0, "stop",
                         {"reconciled_closed": 0})
    other = SimpleNamespace(direction=1, entry_ref=2010.0,
                            invalidation=1995.0, atr=10.0,
                            targets=[2040.0], score=80.0, index=43,
                            factors=[])
    stats3: dict = {}
    runner._place_live_order(pb, other, stats3)
    assert stats3.get("gate_cooldown") == 1
    assert stats3.get("executed", 0) == 0


# -------------------------------------------- Feature 5: runner wiring
def _session_runner(tmp_path, **cfg_kw):
    from types import SimpleNamespace
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    from goldenswing.execution.risk import gold_spec

    class FakeFeed:
        def account_state(self):
            return {"equity": 10000.0, "balance": 10000.0,
                    "free_margin": 10000.0}

    class FakeExecutor:
        def __init__(self):
            self.placed: list = []
            self.closed: list = []

        def symbol_spec(self, symbol):
            return gold_spec()

        def place_order(self, req):
            self.placed.append(req)
            return SimpleNamespace(ok=True, ticket=9, price=2005.0)

        def close_position(self, ticket, volume=None):
            self.closed.append(ticket)
            return SimpleNamespace(ok=True, ticket=ticket)

    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False, entry_gates_enabled=False,
        **cfg_kw)
    executor = FakeExecutor()
    runner = MT5Runner(cfg, feed=FakeFeed(), executor=executor)
    runner._market_open = lambda: (True, "open")
    runner._check_risk_guards = lambda *a: True
    runner._check_margin = lambda *a: True
    runner._alignment = SimpleNamespace(size_multiplier=1.0,
                                        alignment_label="D1_Only")
    runner._stack = SimpleNamespace(frames={"H1": [1, 2, 3]})
    return runner, executor


def _sig(**kw):
    from types import SimpleNamespace
    base = dict(direction=1, entry_ref=2005.0, invalidation=1990.0,
                atr=10.0, targets=[2040.0], score=80.0, index=42,
                factors=[])
    base.update(kw)
    return SimpleNamespace(**base)


def test_friday_close_blocks_base_order_only(tmp_path):
    from types import SimpleNamespace
    # 2026-10-16 is a Friday; tick clock at 20:00 UTC.
    runner, _ = _session_runner(tmp_path)
    runner._last_tick = {"mid": 2005.0, "bid": 2004.8, "ask": 2005.2, "time": "2026-10-16 20:00:00"}
    pb = SimpleNamespace(name="sweep_reversal")
    stats: dict = {}
    runner._place_live_order(pb, _sig(), stats)
    assert stats.get("gate_friday") == 1
    assert stats.get("executed", 0) == 0
    # Thursday 20:00 UTC: not Friday, order flows.
    runner._last_tick = {"mid": 2005.0, "bid": 2004.8, "ask": 2005.2, "time": "2026-10-15 20:00:00"}
    stats2: dict = {}
    runner._place_live_order(pb, _sig(), stats2)
    assert stats2.get("gate_friday", 0) == 0
    assert stats2.get("executed") == 1
    # Friday 20:00, but a LAYER: flows (management, never blocked).
    runner._last_tick = {"mid": 2005.0, "bid": 2004.8, "ask": 2005.2, "time": "2026-10-16 20:00:00"}
    stats3: dict = {}
    runner._place_live_order(pb, _sig(entry_ref=2010.0), stats3,
                             is_layer=True, layer_lots=0.01)
    assert stats3.get("gate_friday", 0) == 0
    assert stats3.get("pending") == 1


def test_session_sizing_applies_overlap_multiplier(tmp_path):
    # Winter Thursday 14:00 UTC = overlap -> 1.2x.  Risk-% sizing:
    # equity 10000, risk 1% = $100; stop distance 17.5 -> 0.05 lots;
    # 1.2x -> 0.06 lots.
    runner, executor = _session_runner(tmp_path)
    runner._last_tick = {"mid": 2005.0, "bid": 2004.8, "ask": 2005.2, "time": "2026-01-15 14:00:00"}
    pb = __import__("types").SimpleNamespace(name="sweep_reversal")
    stats: dict = {}
    runner._place_live_order(pb, _sig(), stats)
    assert stats.get("executed") == 1
    assert executor.placed[0].volume == pytest.approx(0.06)


def test_friday_close_action_close_closes_positions_once(tmp_path):
    from goldenswing.execution.executor import OrderRequest
    runner, executor = _session_runner(
        tmp_path, friday_close_action="close")
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2005.0,
                       targets=[2040.0], atr=10.0,
                       alert_id="MT5-fvg_limit-42", bar_index=10,
                       magic=260101)
    runner.tick_manager.open(ticket=7, request=req, fill_price=2005.0,
                             entry_bar=10, broker_sl=1987.5)
    tick = {"mid": 2010.0, "time": "2026-10-16 20:00:00"}  # Friday
    stats: dict = {"reconciled_closed": 0}
    runner._maybe_friday_close(tick, stats)
    assert executor.closed == [7]
    assert runner.tick_manager.position(7) is None
    # Second tick the same Friday: no repeat.
    runner._maybe_friday_close(tick, stats)
    assert executor.closed == [7]


def test_friday_close_action_hold_is_noop(tmp_path):
    from goldenswing.execution.executor import OrderRequest
    runner, executor = _session_runner(tmp_path)  # default "hold"
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2005.0,
                       targets=[2040.0], atr=10.0,
                       alert_id="MT5-fvg_limit-42", bar_index=10,
                       magic=260101)
    runner.tick_manager.open(ticket=7, request=req, fill_price=2005.0,
                             entry_bar=10, broker_sl=1987.5)
    runner._maybe_friday_close({"mid": 2010.0,
                                "time": "2026-10-16 20:00:00"},
                               {"reconciled_closed": 0})
    assert executor.closed == []
    assert runner.tick_manager.position(7) is not None


def test_place_live_order_flags_scalp_playbooks(tmp_path):
    from types import SimpleNamespace
    runner, _ = _session_runner(
        tmp_path, scalp_playbooks=("sweep_reversal",))
    runner._last_tick = {"mid": 2005.0, "bid": 2004.8, "ask": 2005.2, "time": "2026-01-15 14:00:00"}
    pb = SimpleNamespace(name="sweep_reversal")
    stats: dict = {}
    runner._place_live_order(pb, _sig(), stats)
    assert stats.get("executed") == 1
    assert runner.tick_manager.position(9).is_scalp is True
    # A non-scalp playbook is not flagged.
    runner2, _ = _session_runner(
        tmp_path, scalp_playbooks=("sweep_reversal",))
    runner2._last_tick = {"mid": 2005.0, "bid": 2004.8, "ask": 2005.2, "time": "2026-01-15 14:00:00"}
    pb2 = SimpleNamespace(name="fvg_limit")
    stats2: dict = {}
    runner2._place_live_order(pb2, _sig(), stats2)
    assert stats2.get("executed") == 1
    assert runner2.tick_manager.position(9).is_scalp is False
