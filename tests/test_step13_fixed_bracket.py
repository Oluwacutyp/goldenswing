"""Step 13 (2026-10-29): fixed-bracket exits for qm_m5 — the
owner's pick after the Oct 29 journal evidence.

Policy: for playbooks in ``TickConfig.fixed_bracket_playbooks``
(qm_m5), NO post-entry stop tightening fires — profit-lock,
breakeven, the R-trail and the Chandelier are all skipped, in the
tick manager AND on the runner paths that move stops outside the
manager (news pre-print breakeven, layer-fill rolls), with an
apply()-level refusal as defense in depth.  The position rides
its placement bracket (buffered structural SL + broker TP).
Closes (initial stop, panic, caps, layer stops, timeouts) and
partials are not stop moves and are untouched.

Evidence (Oct 29 journal, build e58b402): the progress-arm lock
closed 10395 at +0.80R against a 4023.00 target (+1.52R) and
10294 at +0.53R against a TP it later reached (+1.10R); the
trail banked 10341 at +1.05R after price missed TP by ~1pt.
The owner ordered a controlled demo test of the pure bracket
for qm_m5 only — the other 17 playbooks keep full management.

Shapes below reuse the journal trades: SHORT entry 4034.112,
structural stop 4039.614, broker stop0 4041.44 (ATR 7.3, risk
7.33), target 4023.002.
"""

from dataclasses import replace
from types import SimpleNamespace

from goldenswing.execution.executor import OrderRequest, OrderResult
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.execution.tick_manager import (ExitConfig, TickAction,
                                                TickConfig, TickManager)

ENTRY, STOP, BROKER_SL, TARGET, ATR = 4034.112, 4039.614, 4041.44, \
    4023.002, 7.3


def _manager(**cfg_kw):
    cfg = TickConfig(breakeven_mode="earned", breakeven_backstop_r=2.0,
                     **cfg_kw)
    return TickManager(exits=ExitConfig(), tick=cfg)


def _open(tm, playbook="qm_m5", ticket=1):
    req = OrderRequest(symbol="XAUUSDm", direction=-1, volume=0.01,
                       stop_loss=STOP, entry_ref=ENTRY,
                       targets=[TARGET], atr=ATR,
                       alert_id="MT5-qm_m5-10395", bar_index=0)
    pos = tm.open(ticket=ticket, request=req, fill_price=ENTRY,
                  entry_bar=0, broker_sl=BROKER_SL, playbook=playbook)
    pos.no_partial = True
    return pos


def test_qm5_no_stop_move_at_any_excursion():
    tm = _manager()
    _open(tm)
    # progress zone (0.52), 1R (4026.50), trail/chandelier zone
    # (1.6R -> 4022.38... still above TP), breakeven backstop (2R)
    for price in (4028.40, 4026.50, 4023.50, 4022.60):
        actions = tm.on_tick(1, price)
        assert not any(a.kind == "modify_sl" for a in actions), price
    assert tm.effective_stop(1) == BROKER_SL


def test_qm5_bracket_stop_still_exits():
    tm = _manager()
    _open(tm)
    tm.on_tick(1, 4026.50)  # excursion first (management would arm)
    actions = tm.on_tick(1, BROKER_SL + 0.01)  # back to the bracket
    assert any(a.kind == "stop_hit" for a in actions)


def test_other_playbook_still_managed():
    tm = _manager()
    _open(tm, playbook="fvg_limit")
    actions = tm.on_tick(1, 4028.40)  # progress zone
    assert any(a.kind == "modify_sl" and a.reason == "profit_lock"
               for a in actions)


def test_bracket_lifted_restores_step10_arms():
    tm = _manager(fixed_bracket_playbooks=())
    _open(tm)
    actions = tm.on_tick(1, 4028.40)  # progress arm stays OFF (step 10)
    assert not any(a.kind == "modify_sl" for a in actions)
    actions = tm.on_tick(1, 4026.50)  # 1R arm fires (step 10)
    locks = [a for a in actions if a.kind == "modify_sl"
             and a.reason == "profit_lock"]
    assert locks and abs(locks[0].level - 4028.62) < 0.05


def test_apply_refuses_modify_for_fixed_bracket():
    tm = _manager()
    _open(tm)
    tm.apply(1, TickAction("modify_sl", level=4028.62,
                           reason="profit_lock"))
    assert tm.effective_stop(1) == BROKER_SL
    assert tm.is_fixed_bracket(1)
    assert not tm.is_fixed_bracket(999)


# ------------------------------------------------------------ runner level

class FakeSender:
    def send_text(self, text, silent=False, reply_markup=None):
        return SimpleNamespace(delivered=True, detail="ok")

    def send_photo(self, png, caption):
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


def test_runner_never_modifies_qm5_stop(tmp_path):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    ex = FakeExecutor()
    runner = MT5Runner(cfg, executor=ex)
    runner.senders = [FakeSender()]
    req = OrderRequest(symbol="XAUUSDm", direction=-1, volume=0.01,
                       stop_loss=STOP, entry_ref=ENTRY,
                       targets=[TARGET], atr=ATR,
                       alert_id="MT5-qm_m5-10395", bar_index=0,
                       magic=260101)
    runner.tick_manager.open(ticket=7, request=req, fill_price=ENTRY,
                             entry_bar=0, broker_sl=BROKER_SL,
                             playbook="qm_m5")
    # 1.6R excursion tick through the full runner manage path
    runner._manage_tick(4022.60, {"tick_actions": 0},
                        bid=4022.60, ask=4022.84)
    assert ex.modify_calls == []
    notes = (tmp_path / "mt5_journal.jsonl").read_text()
    assert "SL ->" not in notes
    # config sanity: the runner's manager carries the default list
    assert runner.tick_manager.tick_cfg.fixed_bracket_playbooks == \
        ("qm_m5",)
    _ = replace  # imported for parity with step-10 isolation tests
