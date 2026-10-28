"""Step 10 (2026-10-27): disable qm_m5's generic early-lock progress arm.

Decision (forwarded 2026-10-27, "Decisions for GoldenSwing" §3):
for ordinary qm_m5 signals, reaching 50% of the way to a nearby
target is NOT sufficient evidence to tighten protection.  qm_m5's
close targets made that arm fire on small moves ("premature
profit locking").  Disabled until replay-grade evidence shows it
improves outcomes.

Scope, exactly as decided — nothing more:
- ONLY the progress arm (best_r < 1R, progress >= 50% to TP1)
  is disabled, and ONLY for playbook qm_m5.
- The best-R arm (best_r >= 1R) still fires for qm_m5.
- Breakeven, trail, Chandelier, and the initial protective stop
  are untouched, for every playbook.
- Other playbooks keep the progress arm; positions with unknown
  playbook (restart adoption) fail open to the old behaviour.

Plumbing: the runner passes the playbook at fill time
(market fills: pb.name; pending fills: state.playbook) into
tick_manager.open(playbook=...), stored on the tick state.  The
playbook is NEVER parsed out of the alert id for gating (that
helper's own comment says "follow-up text only, never a gate").

Numbers used here mirror the discipline harness: long @ 2005,
raw invalidation 1990 -> buffered stop0 1987.5, risk 17.5,
ATR 10 -> profit-lock level 2005 + max(0.5*17.5, 0.75*10)
= 2013.75.  Targets [2030] put 50% progress at 2017.5 with
best_r 0.71 (< 1), isolating the progress arm from the best-R
arm (targets [2040] would put 50% progress exactly at 1R).
"""

from types import SimpleNamespace

from goldenswing.execution.executor import OrderRequest, OrderResult
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.execution.tick_manager import TickManager


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


def _runner(tmp_path, executor):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, executor=executor)
    runner.senders = [FakeSender()]
    return runner


def _open(runner, *, playbook=None, targets=(2030.0,), ticket=7):
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2005.0,
                       targets=list(targets), atr=10.0,
                       alert_id="MT5-fvg_limit-42", bar_index=10,
                       magic=260101)
    kwargs = {}
    if playbook is not None:
        kwargs["playbook"] = playbook
    runner.tick_manager.open(ticket=ticket, request=req, fill_price=2005.0,
                             entry_bar=10, broker_sl=1987.5, **kwargs)
    return req


def _notes(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    return path.read_text() if path.exists() else ""


# ------------------------------------------------------------ manager-level
def test_progress_arm_disabled_for_qm5_only():
    tm = TickManager()
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2005.0,
                       targets=[2030.0], atr=10.0,
                       alert_id="MT5-qm_m5-1", bar_index=10, magic=1)
    tm.open(ticket=7, request=req, fill_price=2005.0, entry_bar=10,
            broker_sl=1987.5, playbook="qm_m5")
    # 50% to target, best_r 0.71: no profit_lock candidate.
    acts = tm.on_tick(7, 2017.5)
    assert [a for a in acts if a.kind == "modify_sl"
            and a.reason == "profit_lock"] == []
    # Same shape, other playbook: the arm fires.
    tm2 = TickManager()
    tm2.open(ticket=7, request=req, fill_price=2005.0, entry_bar=10,
             broker_sl=1987.5, playbook="fvg_limit")
    acts2 = tm2.on_tick(7, 2017.5)
    locks = [a for a in acts2 if a.kind == "modify_sl"
             and a.reason == "profit_lock"]
    assert len(locks) == 1 and locks[0].level == 2013.75


# ------------------------------------------------------------- runner-level
def test_qm5_progress_tick_moves_no_stop(tmp_path):
    ex = FakeExecutor()
    runner = _runner(tmp_path, ex)
    _open(runner, playbook="qm_m5")
    runner._manage_tick(2017.5, {"tick_actions": 0},
                        bid=2017.5, ask=2017.7)
    assert ex.modify_calls == []
    assert "profit_lock" not in _notes(tmp_path)


def test_qm5_best_r_arm_still_fires(tmp_path):
    ex = FakeExecutor()
    runner = _runner(tmp_path, ex)
    _open(runner, playbook="qm_m5", targets=(2040.0,))
    runner._manage_tick(2023.0, {"tick_actions": 0},
                        bid=2023.0, ask=2023.2)
    assert ex.modify_calls == [(7, 2013.75)]
    assert "SL -> 2013.75 (profit_lock)" in _notes(tmp_path)


def test_other_playbook_keeps_progress_arm(tmp_path):
    ex = FakeExecutor()
    runner = _runner(tmp_path, ex)
    _open(runner, playbook="fvg_limit")
    runner._manage_tick(2017.5, {"tick_actions": 0},
                        bid=2017.5, ask=2017.7)
    assert ex.modify_calls == [(7, 2013.75)]


def test_unknown_playbook_fails_open(tmp_path):
    ex = FakeExecutor()
    runner = _runner(tmp_path, ex)
    _open(runner)  # no playbook kwarg: restart-adoption shape
    runner._manage_tick(2017.5, {"tick_actions": 0},
                        bid=2017.5, ask=2017.7)
    assert ex.modify_calls == [(7, 2013.75)]
