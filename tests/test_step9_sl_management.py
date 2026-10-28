"""Step 9 (2026-10-27): broker-aware, bounded SL moves.

Harvest-proven defect (Oct 27 journal, 58 "Invalid stops" across
4 tickets, 52 for ticket 516083401 alone): the tick manager
latched its profit-lock trigger on best excursion and re-emitted
the same stop every tick, while the runner resubmitted it to the
broker unchanged — even after price fell back below the proposed
stop (wrong side of the quote) and even after the broker reported
the position gone.  A failed modify was journalled "retry next
tick" and retried forever.

This fix, in three parts:
1. Side validity: a proposed SL must sit on the valid side of the
   live quote by the broker's stops/freeze margin (or one spread
   when the executor cannot report constraints).  Side-invalid
   levels stand down — noted once per window — and retry when
   price recovers.
2. Bounded retries: at most 3 submissions of the same level per
   side-valid window.  A side-invalid tick closes the window, so
   a level that becomes valid again gets a fresh budget (the
   10073 trade eventually succeeded on recovery — blocking
   forever would forfeit that).
3. Broker reconciliation: a "no position" report stands the ticket
   down for _reconcile (which books the broker_closed close) and
   is never retried.  Failure notes now carry attempted SL,
   broker SL, bid/ask, spread, margin, and retcode, so a rejection
   can actually be diagnosed.
Also fixed in the same pass: profit-lock "progress" was
direction-blind (abs()), so a long falling away from its target
counted as progress toward it and armed the lock underwater.
"""

from types import SimpleNamespace

from goldenswing.execution.executor import OrderRequest, OrderResult
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.execution.tick_manager import (
    TickManager, sl_level_valid)


class FakeSender:
    def __init__(self):
        self.texts = []

    def send_text(self, text, silent=False, reply_markup=None):
        self.texts.append(text)
        return SimpleNamespace(delivered=True, detail="ok")

    def send_photo(self, png, caption):
        self.texts.append(caption)
        return SimpleNamespace(delivered=True, detail="ok")


class FakeExecutor:
    """Counting executor; modify_sl behaviour set per test."""

    def __init__(self, *, modify=None, constraints=None):
        self.modify = modify  # None = success; else OrderResult
        self.constraints = constraints
        self.modify_calls = []

    def symbol_spec(self, symbol):
        return gold_spec()

    def sl_constraints(self, symbol):
        return self.constraints

    def modify_sl(self, ticket, sl, tp=None):
        self.modify_calls.append((ticket, float(sl)))
        if self.modify is not None:
            return self.modify
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        return OrderResult(True, ticket=ticket)

    def positions(self, symbol=None):
        return []


def _runner(tmp_path, executor):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, executor=executor)
    runner.senders = [FakeSender()]
    return runner


def _open_long(runner, ticket=7, entry=2005.0):
    """Long 0.01 @ 2005, raw invalidation 1990, ATR 10 -> buffered
    stop0 1987.5, risk 17.5.  Profit-lock level = 2005 +
    max(0.5*17.5, 0.75*10) = 2013.75."""
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=entry,
                       targets=[2040.0], atr=10.0,
                       alert_id="MT5-fvg_limit-42", bar_index=10,
                       magic=260101)
    runner.tick_manager.open(ticket=ticket, request=req, fill_price=entry,
                             entry_bar=10, broker_sl=1987.5)
    return req


def _notes(tmp_path):
    return (tmp_path / "mt5_journal.jsonl").read_text()


# ------------------------------------------------------------- pure validity
def test_sl_level_valid_edges():
    assert sl_level_valid(1, 100.0, 101.0, 101.2, 0.5)      # long, clear
    assert sl_level_valid(1, 100.5, 101.0, 101.2, 0.5)      # exactly margin
    assert not sl_level_valid(1, 100.6, 101.0, 101.2, 0.5)  # inside margin
    assert not sl_level_valid(1, 101.5, 101.0, 101.2, 0.5)  # wrong side
    assert sl_level_valid(-1, 102.0, 101.0, 101.2, 0.5)
    assert sl_level_valid(-1, 101.7, 101.0, 101.2, 0.5)     # exactly margin
    assert not sl_level_valid(-1, 101.6, 101.0, 101.2, 0.5)
    assert not sl_level_valid(-1, 100.5, 101.0, 101.2, 0.5)
    assert not sl_level_valid(0, 100.0, 101.0, 101.2, 0.5)


def test_failure_window_blocks_and_reopens():
    tm = TickManager()
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2005.0,
                       targets=[2040.0], atr=10.0,
                       alert_id="MT5-x", bar_index=10, magic=1)
    tm.open(ticket=7, request=req, fill_price=2005.0, entry_bar=10,
            broker_sl=1987.5)
    assert not tm.sl_blocked(7, "profit_lock", 2013.75)
    assert not tm.note_sl_failure(7, "profit_lock", 2013.75)
    assert not tm.note_sl_failure(7, "profit_lock", 2013.75)
    assert tm.note_sl_failure(7, "profit_lock", 2013.75)  # 3rd: blocked
    assert tm.sl_blocked(7, "profit_lock", 2013.75)
    tm.reset_sl_window(7, "profit_lock", 2013.75)
    assert not tm.sl_blocked(7, "profit_lock", 2013.75)
    assert tm.mark_broker_gone(7)
    assert not tm.mark_broker_gone(7)  # second call: already stood down
    assert tm.on_tick(7, 2023.0) == []  # gone positions emit nothing


# ------------------------------------------------------- runner: stand-down
def test_side_invalid_stands_down_once_then_retries(tmp_path):
    ex = FakeExecutor(modify=OrderResult(
        False, ticket=7, message="modify failed: Invalid stops",
        retcode=10016))
    runner = _runner(tmp_path, ex)
    _open_long(runner)
    stats = {"tick_actions": 0}
    # Arm at +1.03R (side-valid) -> 1 failed attempt.
    runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    assert len(ex.modify_calls) == 1
    # Price falls back below the lock level: stood down, noted once.
    for _ in range(3):
        runner._manage_tick(2000.0, stats, bid=2000.0, ask=2000.2)
    assert len(ex.modify_calls) == 1
    assert _notes(tmp_path).count("STOOD DOWN") == 1
    # Price recovers: fresh window, attempt resumes.
    runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    assert len(ex.modify_calls) == 2


def test_three_failures_block_until_window_reopens(tmp_path):
    ex = FakeExecutor(modify=OrderResult(
        False, ticket=7, message="modify failed: Invalid stops",
        retcode=10016))
    runner = _runner(tmp_path, ex)
    _open_long(runner)
    stats = {"tick_actions": 0}
    for _ in range(3):
        runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    assert len(ex.modify_calls) == 3
    # 4th side-valid tick: blocked, silent, no 4th submission.
    runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    assert len(ex.modify_calls) == 3
    assert "blocked after 3 rejections" in _notes(tmp_path)
    # Fallback closes the window; recovery re-opens it.
    runner._manage_tick(2000.0, stats, bid=2000.0, ask=2000.2)
    runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    assert len(ex.modify_calls) == 4


def test_failure_note_carries_diagnosis_fields(tmp_path):
    ex = FakeExecutor(modify=OrderResult(
        False, ticket=7, message="modify failed: Invalid stops",
        retcode=10016))
    runner = _runner(tmp_path, ex)
    _open_long(runner)
    runner._manage_tick(2023.0, {"tick_actions": 0},
                        bid=2023.0, ask=2023.2)
    note = [ln for ln in _notes(tmp_path).splitlines()
            if "SL move FAILED" in ln][0]
    for field in ("attempted 2013.75", "broker SL 1987.50",
                  "bid 2023.00", "ask 2023.20", "retcode 10016",
                  "Invalid stops"):
        assert field in note, note


def test_no_position_stands_down_without_retry(tmp_path):
    ex = FakeExecutor(modify=OrderResult(
        False, ticket=7, message="no position 7"))
    runner = _runner(tmp_path, ex)
    _open_long(runner)
    stats = {"tick_actions": 0}
    runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    assert len(ex.modify_calls) == 1
    assert "broker has no such position" in _notes(tmp_path)
    # Next ticks: nothing submitted, nothing re-noted.
    runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    assert len(ex.modify_calls) == 1
    assert _notes(tmp_path).count("broker has no such position") == 1
    # State stays for _reconcile to book the close.
    assert runner.tick_manager.position(7) is not None
    assert runner.tick_manager.on_tick(7, 2023.0) == []


def test_broker_margin_blocks_within_freeze_distance(tmp_path):
    ex = FakeExecutor(
        modify=OrderResult(False, ticket=7,
                           message="modify failed: Invalid stops",
                           retcode=10016),
        constraints={"stops_level_points": 100,
                     "freeze_level_points": 50, "point": 0.01})
    runner = _runner(tmp_path, ex)
    _open_long(runner)
    stats = {"tick_actions": 0}
    # Margin = max(100, 50) * 0.01 = 1.00.  Arm at 2023 (valid).
    runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    assert len(ex.modify_calls) == 1
    # Bid 2014.50: level 2013.75 is on the valid side but only 0.75
    # inside the 1.00 margin -> stood down, not submitted.
    runner._manage_tick(2014.5, stats, bid=2014.5, ask=2014.7)
    assert len(ex.modify_calls) == 1
    assert "STOOD DOWN" in _notes(tmp_path)
    # Bid 2015.00: 1.25 clear of the level -> submitted again.
    runner._manage_tick(2015.0, stats, bid=2015.0, ask=2015.2)
    assert len(ex.modify_calls) == 2


def test_underwater_long_does_not_arm_profit_lock(tmp_path):
    ex = FakeExecutor()
    runner = _runner(tmp_path, ex)
    _open_long(runner)
    # Adverse 18/35 = 0.51 "progress" under the old abs() rule;
    # favourable progress is -0.51, so no lock candidate fires.
    # (Price is through the broker stop, so the safety-net close
    # runs too — it needs the reconciled_closed stats key.)
    runner._manage_tick(1987.0, {"tick_actions": 0,
                                 "reconciled_closed": 0},
                        bid=1987.0, ask=1987.2)
    assert ex.modify_calls == []


def test_success_clears_failure_window(tmp_path):
    ex = FakeExecutor(modify=OrderResult(
        False, ticket=7, message="modify failed: Invalid stops",
        retcode=10016))
    runner = _runner(tmp_path, ex)
    _open_long(runner)
    stats = {"tick_actions": 0}
    runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    assert runner.tick_manager.sl_blocked(7, "profit_lock", 2013.75) \
        is False
    ex.modify = None  # broker accepts now
    runner._manage_tick(2023.0, stats, bid=2023.0, ask=2023.2)
    assert len(ex.modify_calls) == 3
    assert "SL -> 2013.75 (profit_lock)" in _notes(tmp_path)
