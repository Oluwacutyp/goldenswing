"""hot_scalp profit-lock defect — ticket 540274935 (2026-11-05).

The live journal, read tick by tick:

- 11:32:59 FILL hot_scalp +1 0.01 @ 4134.437, structural stop
  4131.469 (risk 2.968).  Pullback-continuation long into an M5
  FVG zone 4132.01-4135.883; the setup's targets: TP1 4136.181,
  planned objective 4138.671 (+1.43R), runner open.
- 11:34:08 the ONLY management action of the trade:
  "SL -> 4135.92 (profit_lock)" — entry + 0.50 x risk exactly,
  armed by the generic PROGRESS arm (price had travelled ~50%
  of the way to targets[0]; best excursion only ~0.71R).
- 11:34:19 (11 seconds later) price retested the zone top,
  filled through the lock at 4135.797 -> +0.46R.  The lock sat
  0.04 ABOVE the top edge of the very zone the trade was built
  on, and 0.26 BELOW the playbook's own TP1: the generic lock
  pre-empted the first objective and became the exit.
- The close note called it "SL hit (trail)" — the manager's
  action was profit_lock.  The label was inferred from price
  geometry (every modify syncs the trail level, so any moved
  stop read as "trail").

Fixes pinned here:

1. hot_scalp joins qm_m5 in
   ``profit_lock_progress_excluded_playbooks`` (the 2026-10-27
   "premature profit locking" precedent): travelling halfway
   to target no longer arms the lock; the best-R >= 1.0 arm
   still fires.  (Follow-up, same day: ticket 540791314 showed
   the best-R arm itself locking too early — hot_scalp's lock
   is now EARNED, and the touch arm is gone for hot_scalp;
   see tests/test_hot_scalp_earned_lock.py.  The pins below
   that exercise cap/label mechanics stamp the earning
   explicitly.)
2. Structure floor: a zone-born position's profit-lock level
   is capped a buffer (0.05R) INSIDE the entry zone's far edge
   (long: zone_high - buffer; short: zone_low + buffer), so a
   touch of the edge cannot eat the stop — price must
   penetrate the buffer.  A cap that no longer tightens the
   current stop suppresses the lock for that tick (recorded,
   journalled once by the runner); the lock stays pending.
   Zone-less positions behave byte-identically to before.
3. Honest close labels: the close note names the mechanism
   that LAST moved the stop (the tick manager's applied-modify
   record), falling back to plain "SL hit" — the placement
   stop — when no modify was ever applied.  Fixed-bracket
   closes (qm_m5) therefore always read as the placement stop.

Fixture arithmetic mirrors the journal exactly: ATR 1.5 ->
stop buffer 0.375 -> raw invalidation 4131.844 gives the
broker stop 4131.469 and risk 2.968; the lock distance is
max(0.5 x 2.968, 0.75 x 1.5) = 1.484 (the risk arm won, as in
the journal), so the raw lock level is 4135.921 — the level
that killed the live trade.
"""

from types import SimpleNamespace

import pytest

from goldenswing.execution.executor import OrderRequest, OrderResult
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.execution.tick_manager import TickAction, TickManager

ENTRY = 4134.437
STOP0 = 4131.469            # broker (buffered) stop; risk 2.968
RAW_INV = 4131.844          # STOP0 + 0.25 * ATR(1.5)
ATR = 1.5
TARGET = 4138.671           # targets[0]: the planned objective
ZONE = (4132.01, 4135.883)  # the M5 FVG zone the trade was built on
RAW_LOCK = 4135.921         # entry + 1.484 — the journal's lock
CAPPED_LOCK = 4135.7346     # zone_high - 0.05 * 2.968
PROGRESS_TICK = 4136.56     # progress 0.501 to target, best_r 0.715
RETEST_TICK = 4135.88       # the retest that ate the live trade
BEST_R_TICK = 4137.50       # best_r 1.032 — the surviving arm


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
    def __init__(self):
        self.modify_calls = []

    def symbol_spec(self, symbol):
        return gold_spec()

    def modify_sl(self, ticket, sl, tp=None):
        self.modify_calls.append((ticket, float(sl)))
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        return OrderResult(True, ticket=ticket)

    def positions(self, symbol=None):
        return []


def _runner(tmp_path, executor, sender=None):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, executor=executor)
    if sender is not None:
        runner.senders = [sender]
    return runner


def _request(direction=1, atr=ATR, targets=(TARGET,), stop_loss=RAW_INV):
    return OrderRequest(symbol="XAUUSD", direction=direction, volume=0.01,
                        stop_loss=stop_loss, entry_ref=ENTRY,
                        targets=list(targets), atr=atr,
                        alert_id="HS-540274935", bar_index=10, magic=1)


def _open(tm, *, playbook="hot_scalp", zone="journal", ticket=540274935,
          **req_kw):
    kwargs = {}
    if zone == "journal":
        kwargs["entry_zone"] = ZONE
    elif zone is not None:
        kwargs["entry_zone"] = zone
    return tm.open(ticket=ticket, request=_request(**req_kw),
                   fill_price=ENTRY, entry_bar=10, broker_sl=STOP0,
                   playbook=playbook, **kwargs)


def _locks(actions):
    return [a for a in actions if a.kind == "modify_sl"
            and a.reason == "profit_lock"]


def _notes(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    return path.read_text() if path.exists() else ""


# ------------------------------------------------------- journal replay
def test_journal_replay_progress_arm_never_locks():
    """The journal's path, replayed: the progress tick that
    armed the live lock now produces NOTHING for hot_scalp,
    and the retest tick that closed the live trade leaves the
    position untouched on its placement stop."""
    tm = TickManager()
    pos = _open(tm)
    assert pos.stop0 == pytest.approx(STOP0)
    assert pos.risk_price == pytest.approx(2.968)
    assert tm.on_tick(540274935, PROGRESS_TICK) == []
    assert tm.on_tick(540274935, RETEST_TICK) == []
    assert tm.effective_stop(540274935) == pytest.approx(STOP0)
    assert tm.profit_lock_state(540274935)["done"] is False
    assert tm.last_sl_reason(540274935) == ""


def test_journal_replay_control_playbook_locks_at_journal_level():
    """Same numbers, a playbook that keeps the progress arm:
    the lock fires at 4135.921 — the fixture reproduces the
    journal mechanism exactly, so the exclusion above is what
    saves the hot_scalp replay, not a changed fixture."""
    tm = TickManager()
    _open(tm, playbook="fvg_limit", zone=None)
    locks = _locks(tm.on_tick(540274935, PROGRESS_TICK))
    assert len(locks) == 1
    assert locks[0].level == pytest.approx(RAW_LOCK)


# ------------------------------------------------------------- fix 1
def test_hot_scalp_best_r_arm_still_fires():
    """SUPERSEDED by the earned-lock follow-up (ticket
    540791314, same morning): the +1R touch arm itself became
    the de-facto exit on the second trade, so for hot_scalp
    the touch no longer arms the lock at all — the lock must
    be EARNED (acceptance close beyond TP1, or the 2R
    backstop).  The full earned contract is pinned in
    tests/test_hot_scalp_earned_lock.py; this pin now records
    that the old arm is gone."""
    tm = TickManager()
    _open(tm, zone=None)
    assert _locks(tm.on_tick(540274935, BEST_R_TICK)) == []
    assert tm.pl_state(540274935)["earned"] is False
    assert tm.profit_lock_state(540274935)["done"] is False


def test_runner_hot_scalp_progress_tick_moves_no_stop(tmp_path):
    ex = FakeExecutor()
    runner = _runner(tmp_path, ex)
    _open(runner.tick_manager)
    runner._manage_tick(PROGRESS_TICK, {"tick_actions": 0},
                        bid=PROGRESS_TICK, ask=PROGRESS_TICK + 0.2)
    assert ex.modify_calls == []
    assert "profit_lock" not in _notes(tmp_path)


# ------------------------------------------------------------- fix 2
def test_zone_cap_long_floors_lock_inside_zone_edge():
    tm = TickManager()
    _open(tm)  # hot_scalp + the journal zone
    # Earned-lock follow-up: the cap is the subject here, so
    # the lock is earned explicitly (acceptance stamp) — the
    # +1R touch alone no longer arms hot_scalp.
    assert tm.mark_pl_earned(540274935, "acceptance: fixture")
    locks = _locks(tm.on_tick(540274935, BEST_R_TICK))
    assert len(locks) == 1
    assert locks[0].level == pytest.approx(CAPPED_LOCK)
    cap = tm.profit_lock_state(540274935)["zone_cap"]
    assert cap["raw_level"] == pytest.approx(RAW_LOCK)
    assert cap["level"] == pytest.approx(CAPPED_LOCK)
    assert cap["zone_high"] == pytest.approx(ZONE[1])
    assert cap["suppressed"] is False
    # A retest that only reaches the zone top (the live
    # trade's exit path) no longer touches the stop.
    tm.apply(540274935, locks[0])
    assert tm.on_tick(540274935, RETEST_TICK) == []
    assert tm.effective_stop(540274935) == pytest.approx(CAPPED_LOCK)


def test_zone_cap_short_mirrors_at_zone_low():
    """Short: entry 4134.437, stop0 4137.405 (raw invalidation
    4136.405 at ATR 4.0), lock distance max(1.484, 3.0) = 3.0
    -> raw lock 4131.437, below the zone's bottom edge; the
    cap lifts it to zone_low + buffer = 4132.1584."""
    tm = TickManager()
    tm.open(ticket=11, request=_request(direction=-1, atr=4.0,
                                        targets=(4130.203,),
                                        stop_loss=4136.405),
            fill_price=ENTRY, entry_bar=10, broker_sl=4137.405,
            playbook="hot_scalp", entry_zone=ZONE)
    assert tm.mark_pl_earned(11, "acceptance: fixture")
    locks = _locks(tm.on_tick(11, 4131.30))  # best_r 1.057
    assert len(locks) == 1
    assert locks[0].level == pytest.approx(4132.1584)
    cap = tm.profit_lock_state(11)["zone_cap"]
    assert cap["raw_level"] == pytest.approx(4131.437)
    assert cap["suppressed"] is False


def test_zone_cap_suppressed_when_it_cannot_tighten():
    """Zone (4131.0, 4131.60): the cap 4131.4516 sits BELOW the
    placement stop 4131.469 — the lock must not fire, must not
    be marked done, and the decision must be on record."""
    tm = TickManager()
    _open(tm, playbook="fvg_limit", zone=(4131.0, 4131.60))
    acts = tm.on_tick(540274935, BEST_R_TICK)
    assert _locks(acts) == []
    state = tm.profit_lock_state(540274935)
    assert state["done"] is False
    cap = state["zone_cap"]
    assert cap["raw_level"] == pytest.approx(RAW_LOCK)
    assert cap["level"] == pytest.approx(4131.4516)
    assert cap["suppressed"] is True
    # Still pending on the next tick — never silently consumed.
    assert _locks(tm.on_tick(540274935, BEST_R_TICK + 0.1)) == []
    assert tm.profit_lock_state(540274935)["done"] is False


def test_zoneless_and_degenerate_zones_behave_as_before():
    """No zone, a (0.0, 0.0) zone, and a malformed zone all
    leave the raw lock untouched and record no cap."""
    for zone in (None, (0.0, 0.0), (4135.883,), "junk"):
        tm = TickManager()
        _open(tm, zone=zone)
        assert tm.mark_pl_earned(540274935, "acceptance: fixture")
        locks = _locks(tm.on_tick(540274935, BEST_R_TICK))
        assert len(locks) == 1
        assert locks[0].level == pytest.approx(RAW_LOCK)
        assert tm.profit_lock_state(540274935)["zone_cap"] is None


def test_runner_capped_lock_journals_the_cap(tmp_path):
    ex = FakeExecutor()
    runner = _runner(tmp_path, ex)
    _open(runner.tick_manager, playbook="fvg_limit")
    runner._manage_tick(BEST_R_TICK, {"tick_actions": 0},
                        bid=BEST_R_TICK, ask=BEST_R_TICK + 0.2)
    assert len(ex.modify_calls) == 1
    assert ex.modify_calls[0][1] == pytest.approx(CAPPED_LOCK)
    notes = _notes(tmp_path)
    assert "SL -> 4135.73 (profit_lock)" in notes
    assert "entry-zone cap: raw lock 4135.92" in notes


def test_runner_suppressed_lock_journals_held_once(tmp_path):
    ex = FakeExecutor()
    runner = _runner(tmp_path, ex)
    _open(runner.tick_manager, playbook="fvg_limit",
          zone=(4131.0, 4131.60))
    for price in (BEST_R_TICK, BEST_R_TICK + 0.1):
        runner._manage_tick(price, {"tick_actions": 0},
                            bid=price, ask=price + 0.2)
    notes = _notes(tmp_path)
    assert notes.count("profit_lock HELD (entry-zone cap)") == 1
    assert "raw lock 4135.92 capped to 4131.45" in notes
    assert all(level != pytest.approx(RAW_LOCK)
               for _t, level in ex.modify_calls)


# ------------------------------------------------------------- fix 3
def test_close_label_names_profit_lock_not_trail(tmp_path):
    """The journal's close, relabelled honestly: the stop was
    moved by profit_lock, so the close names profit_lock —
    and the R is the journal's +0.46R."""
    sender = FakeSender()
    ex = FakeExecutor()
    runner = _runner(tmp_path, ex, sender)
    _open(runner.tick_manager)
    runner.tick_manager.apply(
        540274935, TickAction("modify_sl", level=RAW_LOCK,
                              reason="profit_lock"))
    assert runner.tick_manager.last_sl_reason(540274935) == "profit_lock"
    runner._record_close(540274935, 4135.797, "broker_closed",
                         {"reconciled_closed": 0})
    closes = [t for t in sender.texts if "MT5 CLOSE" in t]
    assert closes and "SL hit (profit_lock)" in closes[0]
    assert "+0.46R" in closes[0]
    assert not any("SL hit (trail)" in t for t in sender.texts)


def test_close_label_names_each_mechanism(tmp_path):
    for reason, level in (("trail", 4136.0), ("chandelier", 4136.2),
                          ("breakeven", ENTRY)):
        sender = FakeSender()
        runner = _runner(tmp_path, FakeExecutor(), sender)
        _open(runner.tick_manager, ticket=7)
        runner.tick_manager.apply(
            7, TickAction("modify_sl", level=level, reason=reason))
        runner._record_close(7, level, "broker_closed",
                             {"reconciled_closed": 0})
        assert any(f"SL hit ({reason})" in t for t in sender.texts), reason


def test_fixed_bracket_close_reads_placement_stop(tmp_path):
    """qm_m5 rides its placement bracket — no modify is ever
    applied, so its close can only read as the placement
    stop, never dressed in a mechanism's name."""
    sender = FakeSender()
    runner = _runner(tmp_path, FakeExecutor(), sender)
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2005.0,
                       targets=[2040.0], atr=10.0,
                       alert_id="MT5-qm_m5-1", bar_index=10, magic=260101)
    runner.tick_manager.open(ticket=7, request=req, fill_price=2005.0,
                             entry_bar=10, broker_sl=1987.5,
                             playbook="qm_m5")
    assert runner.tick_manager.last_sl_reason(7) == ""
    runner._record_close(7, 1987.5, "broker_closed",
                         {"reconciled_closed": 0})
    assert any("MT5 CLOSE" in t and "SL hit" in t for t in sender.texts)
    assert not any("SL hit (" in t for t in sender.texts)
