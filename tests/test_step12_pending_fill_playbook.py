"""Step 12c (2026-10-29): pending FULL fills must thread the playbook
into the tick manager — the step-10 exclusion depended on it.

Defect (proven from the live journal, build e58b402): step 10
disabled qm_m5's profit-lock progress arm via
``profit_lock_progress_excluded_playbooks``, keyed on the tick
state's playbook.  The market-fill path passes ``playbook=pb.name``
and the pending PARTIAL-fill branch passes ``playbook=state.playbook``
— but the pending FULL-fill branch of ``_record_pending_fill``
omitted it, so every full pending fill (the dominant qm_m5 path:
LIMIT pendings) tracked with playbook "" and the exclusion failed
open.  On 2026-10-29 the progress arm fired on all three qm_m5
trades; the tick record proves it to the second:

- 10395: progress level (mid 4028.56) first hit 13:42:56, lock sent
  13:42:56; the 1R level (4026.78) was never touched.  Exit +0.80R,
  target 4023.00 never reached.
- 10294: progress level first hit 05:30:36, lock sent 05:30:37; 1R
  level never touched.  Exit +0.53R.
- 10341: the 1R arm fired legitimately (1R level hit 09:17:45, lock
  09:17:46) — the control case.

These tests pin the threading at the runner level (the step-10
suite only exercised the manager with the playbook passed by hand,
which is how the omission slipped through).
"""

from types import SimpleNamespace

from goldenswing.execution.executor import OrderRequest, OrderResult
from goldenswing.execution.mt5_runner import (MT5Runner,
                                              MT5RunnerConfig,
                                              PendingOrderState)
from goldenswing.execution.risk import gold_spec


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
    def symbol_spec(self, symbol):
        return gold_spec()

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        return OrderResult(True, ticket=ticket)

    def positions(self, symbol=None):
        return []


def _runner(tmp_path):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, executor=FakeExecutor())
    runner.senders = [FakeSender()]
    return runner


def _pending_fill(runner, *, playbook, ticket=522691421):
    """The 10395 shape: qm_m5 SHORT LIMIT pending, full fill."""
    track = OrderRequest(
        symbol="XAUUSDm", direction=-1, volume=0.01,
        stop_loss=4039.614, entry_ref=4034.112,
        targets=[4023.002], atr=7.3,
        alert_id="MT5-qm_m5-10395", bar_index=0, magic=260101)
    state = PendingOrderState(
        ticket=ticket, kind="limit", direction=-1, level=4034.112,
        volume=0.01, sl=4041.44, tp=4023.002,
        alert_id="MT5-qm_m5-10395", playbook=playbook, zone=None,
        invalidation=4039.614, placed_bar=0, expiry_bar=99,
        premise="native", entry_clock="M5", track_request=track)
    runner._pending_orders[ticket] = state
    pos = SimpleNamespace(ticket=ticket, volume=0.01,
                          entry_price=4034.112)
    runner._record_pending_fill(state, pos, {})
    return ticket


def test_pending_full_fill_threads_playbook(tmp_path):
    runner = _runner(tmp_path)
    ticket = _pending_fill(runner, playbook="qm_m5")
    st = runner.tick_manager._states[ticket]
    assert st.playbook == "qm_m5"


def test_pending_filled_qm5_progress_arm_stays_off(tmp_path):
    runner = _runner(tmp_path)
    ticket = _pending_fill(runner, playbook="qm_m5")
    tm = runner.tick_manager
    # 10395's arming tick: mid 4028.40 = 51.6% of the way to the
    # 4023.00 target, best_r 0.78 (< 1R).  The progress arm must
    # NOT fire for a pending-filled qm_m5 position.
    actions = tm.on_tick(ticket, 4028.40)
    assert not any(a.kind == "modify_sl" for a in actions)
    # Step 13 (2026-10-29) supersedes the step-10 expectation that
    # the 1R arm still fires for qm_m5: the playbook now rides a
    # fixed bracket, so even a 1R excursion (mid 4026.50 = best_r
    # 1.04) moves NO stop.  Threading still decides this — the
    # fixed-bracket gate keys off the same threaded playbook this
    # test exists to prove (see the fvg_limit control below, whose
    # progress arm still fires, and test_step13's bracket-lifted
    # control, where the 1R arm returns).
    actions = tm.on_tick(ticket, 4026.50)
    assert not any(a.kind == "modify_sl" for a in actions)


def test_pending_filled_other_playbook_keeps_progress_arm(tmp_path):
    runner = _runner(tmp_path)
    ticket = _pending_fill(runner, playbook="fvg_limit")
    actions = runner.tick_manager.on_tick(ticket, 4028.40)
    locks = [a for a in actions
             if a.kind == "modify_sl" and a.reason == "profit_lock"]
    assert locks  # same shape, other playbook: the arm fires
