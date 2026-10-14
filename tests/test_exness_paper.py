"""Exness $12 paper loop: executor margin/stop-out math, fixed-lot
sizing, guard interaction, and side-by-side isolation.

The Exness loop must NEVER be confused with the live $10k OANDA loop:
separate executor (margin + stop-out), separate sizing (fixed 0.01),
separate journal/state (tested via the script's constants + runner
injection), and R-guards that cannot refuse a fixed-lot trade.
"""
import importlib.util
from pathlib import Path

import pytest

from goldenswing.alerts import build_alert
from goldenswing.alerts.callbacks import AlertAction
from goldenswing.execution.callbacks import ExecutionCallbackHandler
from goldenswing.execution.exness_paper import (
    COMMISSION_PER_TRADE,
    LEVERAGE,
    LOT_OZ,
    SPREAD_PER_OZ,
    START_BALANCE_12,
    ExnessPaperExecutor,
    margin_required,
    stop_out_price,
)
from goldenswing.execution.executor import OrderRequest
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.paper import PaperExecutor
from goldenswing.execution.risk import RiskConfig, RiskGuards, gold_spec
from goldenswing.execution.runner import LiveRunner, RunnerConfig
from goldenswing.golden.detector import GoldenSignal

REPO_ROOT = Path(__file__).resolve().parents[1]


def _order(**kw):
    base = dict(symbol="XAUUSD", direction=1, volume=0.01,
                stop_loss=3900.0, entry_ref=4000.0, targets=[],
                atr=0.0, alert_id="T1", bar_index=0)
    base.update(kw)
    return OrderRequest(**base)


def _signal(**kw):
    base = dict(index=5, timestamp="2026-10-10T05:00:00Z", direction=1,
                score=80.0, kind="continuation", factors=["trend aligned"],
                entry_zone=(1995.0, 2005.0), entry_ref=2000.0,
                invalidation=1990.0, targets=[2030.0], atr=10.0)
    base.update(kw)
    return GoldenSignal(**base)


def _bar(idx, o, h, l, c):
    return {"index": idx, "time": None, "open": o, "high": h,
            "low": l, "close": c}


# ------------------------------------------------------- pure-math parity
def test_margin_and_stopout_parity_with_exness12_backtest():
    """Same formulas as goldenswing.backtest.exness12, proven equal."""
    from goldenswing.backtest import exness12 as back

    for entry in (3900.0, 4000.0, 4180.5):
        for leverage in (2.0, 200.0, 2000.0, float("inf")):
            assert margin_required(entry, 1.0, leverage) == pytest.approx(
                back.margin_required(entry, 1.0, leverage))
        for direction in (1, -1):
            for balance in (12.0, 50.0, 100.0):
                assert stop_out_price(entry, direction, balance) == pytest.approx(
                    back.stop_out_price(entry, direction, balance))


def test_stop_out_price_formula():
    # Long: entry - (balance - spread). Short: entry + (balance - spread).
    assert stop_out_price(4000.0, 1, 12.0) == pytest.approx(3988.20)
    assert stop_out_price(4000.0, -1, 12.0) == pytest.approx(4011.80)


def test_margin_at_1_to_2_is_impossible_for_12():
    assert margin_required(4000.0, LOT_OZ, 2.0) == pytest.approx(2000.0)
    assert margin_required(4000.0, LOT_OZ, 2.0) > START_BALANCE_12
    assert margin_required(4000.0, LOT_OZ, LEVERAGE) == pytest.approx(2.0)


def test_exness_constants():
    assert START_BALANCE_12 == pytest.approx(12.0)
    assert LOT_OZ == pytest.approx(1.0)
    assert LEVERAGE == pytest.approx(2000.0)
    assert SPREAD_PER_OZ == pytest.approx(0.20)
    assert COMMISSION_PER_TRADE == pytest.approx(0.0)


# ------------------------------------------------------------ executor core
def test_spread_is_exness_not_oanda():
    assert ExnessPaperExecutor()._half_spread == pytest.approx(0.10)
    assert PaperExecutor()._half_spread == pytest.approx(0.225)


def test_place_order_refuses_when_margin_unaffordable():
    ex = ExnessPaperExecutor(balance=12.0, leverage=2.0)
    res = ex.place_order(_order())
    assert not res.ok and "insufficient margin" in res.message


def test_place_order_refuses_when_balance_cannot_cover_spread():
    ex = ExnessPaperExecutor(balance=0.10)
    res = ex.place_order(_order())
    assert not res.ok and "insufficient_for_spread" in res.message


def test_place_order_refuses_when_blown():
    ex = ExnessPaperExecutor(balance=0.0)
    res = ex.place_order(_order())
    assert not res.ok and "blown" in res.message


def test_place_order_accepts_when_margin_covered():
    ex = ExnessPaperExecutor(balance=12.0, leverage=2000.0)
    res = ex.place_order(_order())
    assert res.ok


def test_unlimited_leverage_margin_is_zero():
    ex = ExnessPaperExecutor(balance=12.0, leverage=float("inf"))
    assert ex.margin_for(4000.0, 0.01) == pytest.approx(0.0)
    assert ex.place_order(_order()).ok


def test_free_margin_subtracts_used_margin():
    ex = ExnessPaperExecutor(balance=12.0, leverage=2000.0)
    assert ex.place_order(_order()).ok
    ex.advance(_bar(1, 4000.0, 4004.0, 3996.0, 4001.0))
    state = ex.account_state()
    # margin at fill ~4000.10 * 1oz / 2000 = ~$2.00
    assert state.free_margin == pytest.approx(state.equity - 2.0, abs=0.05)


def test_stop_out_closes_at_equity_zero():
    ex = ExnessPaperExecutor(balance=12.0)
    assert ex.place_order(_order()).ok
    ex.advance(_bar(1, 4000.0, 4004.0, 3996.0, 4001.0))  # fill @ 4000.10
    assert len(ex.positions()) == 1
    # Stop-out price: 4000.10 - (12 - 0.10)/1 = 3988.20. Bar breaches it.
    ex.advance(_bar(2, 3990.0, 3995.0, 3980.0, 3985.0))
    assert ex.positions() == []
    assert ex.account_state().balance == pytest.approx(0.0)
    assert ex.closed and ex.closed[-1].reason == "stop_out_blowup"


def test_stop_out_beats_structural_stop_on_same_bar():
    """Matches exness12's conservative ordering: stop-out first."""
    ex = ExnessPaperExecutor(balance=12.0)
    # Structural stop 3995 is FAR above the stop-out price 3988.20;
    # one bar breaches both — the blow-up wins.
    assert ex.place_order(_order(stop_loss=3995.0)).ok
    ex.advance(_bar(1, 4000.0, 4004.0, 3996.0, 4001.0))
    ex.advance(_bar(2, 3990.0, 3995.0, 3980.0, 3985.0))
    assert ex.closed and ex.closed[-1].reason == "stop_out_blowup"
    assert ex.account_state().balance == pytest.approx(0.0)


def test_no_stop_out_when_price_stays_above():
    ex = ExnessPaperExecutor(balance=12.0)
    assert ex.place_order(_order()).ok
    ex.advance(_bar(1, 4000.0, 4004.0, 3996.0, 4001.0))
    ex.advance(_bar(2, 3995.0, 3998.0, 3990.0, 3996.0))  # low 3990 > 3988.20
    assert len(ex.positions()) == 1
    assert ex.account_state().balance == pytest.approx(12.0)


def test_pnl_is_one_dollar_per_dollar_move_minus_spread():
    ex = ExnessPaperExecutor(balance=12.0)
    assert ex.place_order(_order()).ok
    ex.advance(_bar(1, 4000.0, 4004.0, 3996.0, 4001.0))  # fill @ 4000.10
    assert ex.close_position(1).ok
    ex.advance(_bar(2, 4010.0, 4012.0, 4008.0, 4011.0))  # close @ 4009.90
    assert ex.closed and ex.closed[-1].reason == "manual"
    # $10 move - $0.20 spread = $9.80
    assert ex.closed[-1].pnl_money == pytest.approx(9.80)
    assert ex.account_state().balance == pytest.approx(21.80)


def test_gapped_stop_out_floors_at_zero():
    ex = ExnessPaperExecutor(balance=12.0)
    assert ex.place_order(_order()).ok
    ex.advance(_bar(1, 4000.0, 4004.0, 3996.0, 4001.0))
    # Gap open far below stop-out: close at the open, floored at $0.
    ex.advance(_bar(2, 3900.0, 3905.0, 3890.0, 3895.0))
    assert ex.account_state().balance == pytest.approx(0.0)
    assert ex.closed[-1].reason == "stop_out_blowup"


def test_export_import_round_trip_preserves_position():
    ex = ExnessPaperExecutor(balance=12.0)
    assert ex.place_order(_order()).ok
    ex.advance(_bar(1, 4000.0, 4004.0, 3996.0, 4001.0))
    ex2 = ExnessPaperExecutor(balance=999.0)  # proves import overwrites
    ex2.import_state(ex.export_state())
    assert ex2.account_state().balance == pytest.approx(12.0)
    assert len(ex2.positions()) == 1
    assert ex2.positions()[0].entry_price == pytest.approx(4000.10)
    ex2.shift_bar_indices(-1)  # rebase path must not crash
    assert len(ex2.positions()) == 1


# ------------------------------------------------------- fixed-lot sizing
def _exness_handler(tmp_path):
    paper = ExnessPaperExecutor(balance=12.0)
    guards = RiskGuards(tmp_path / "risk.json",
                        RiskConfig(daily_loss_pct=0.0, weekly_loss_pct=0.0,
                                   max_drawdown_pct=0.0,
                                   max_consecutive_losses=0,
                                   max_positions=1,
                                   max_total_risk_pct=0.0,
                                   require_stops=True),
                        kill_switch_path=tmp_path / "KILL")
    journal = TradeJournal(tmp_path / "journal.jsonl")
    handler = ExecutionCallbackHandler(
        executor=paper, paper_executor=paper, guards=guards,
        spec=gold_spec(), journal=journal, risk_pct=0.01,
        fixed_lots=0.01)
    return handler, paper, journal


def test_fixed_lots_ignores_risk_pct_and_forecaster_dial(tmp_path):
    handler, paper, journal = _exness_handler(tmp_path)
    alert = build_alert(_signal(), instrument="XAUUSD", mode="exness12")
    # A 2x forecaster dial and a huge risk_pct must NOT change the size.
    handler.register_context(alert.alert_id, phase="manipulation",
                             playbook="sweep_reversal",
                             forecast_size_mult=2.0)
    handler.risk_pct = 0.25
    msg = handler.handle(AlertAction.PAPER, alert)
    assert "0.01 lots" in msg
    paper.advance(_bar(6, 2000.0, 2004.0, 1996.0, 2001.0))
    assert len(paper.positions()) == 1
    assert paper.positions()[0].volume == pytest.approx(0.01)
    opens = [r for r in journal.records() if r["type"] == "open"]
    assert opens[0]["volume"] == pytest.approx(0.01)


def test_fixed_lots_execute_path_passes_exness_guards(tmp_path):
    """R-guards must not refuse a fixed-lot trade: exposure cap is 0
    (disabled) in the Exness config, so the $20 structural-stop risk
    money on a $12 account is allowed — margin is the constraint."""
    handler, paper, _journal = _exness_handler(tmp_path)
    alert = build_alert(_signal(), instrument="XAUUSD", mode="exness12")
    msg = handler.handle(AlertAction.EXECUTE, alert)
    assert "0.01 lots" in msg and "refused" not in msg


def test_exposure_cap_zero_disables_while_default_still_caps(tmp_path):
    # Full Exness config: every R-limit 0 (disabled) — only kill
    # switch / require_stops / max_positions still gate.
    disabled = RiskGuards(tmp_path / "r0.json",
                          RiskConfig(daily_loss_pct=0.0, weekly_loss_pct=0.0,
                                     max_drawdown_pct=0.0,
                                     max_consecutive_losses=0,
                                     max_positions=1,
                                     max_total_risk_pct=0.0),
                          kill_switch_path=tmp_path / "K0")
    assert disabled.check_new_trade(
        balance=12.0, open_risk_money=0.0, new_risk_money=1_000_000.0,
        open_count=0).allowed
    capped = RiskGuards(tmp_path / "r1.json", RiskConfig(),
                        kill_switch_path=tmp_path / "K1")
    assert not capped.check_new_trade(
        balance=12.0, open_risk_money=0.0, new_risk_money=1_000_000.0,
        open_count=0).allowed


# ------------------------------------------------------------- isolation
def _load_script():
    spec = importlib.util.spec_from_file_location(
        "paper_live_exness", REPO_ROOT / "scripts" / "paper_live_exness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_paths_are_separate_from_live_loop():
    mod = _load_script()
    assert mod.JOURNAL_PATH.parent.name == "exness12"
    assert mod.JOURNAL_PATH.name == "paper_journal.jsonl"
    assert mod.STATE_STORE_PATH.parent.name == "exness12"
    assert mod.RISK_STATE_PATH.parent.name == "exness12"
    assert mod.KILL_SWITCH_PATH.parent.name == "exness12"
    assert "paper_state" not in str(mod.JOURNAL_PATH)
    assert mod.JOURNAL_PATH != REPO_ROOT / "data" / "trade_journal.jsonl"
    assert mod.STATE_STORE_PATH != REPO_ROOT / "data" / "paper_state"
    assert mod.FIXED_LOTS == pytest.approx(0.01)
    assert "1:2000 is NOT guaranteed at CPI" in mod.ASSUMED_MODEL_NOTE


def test_runner_accepts_exness_executor_and_fixed_lots(tmp_path):
    cfg = RunnerConfig(
        journal_path=tmp_path / "exness_journal.jsonl",
        risk_state_path=tmp_path / "exness_risk.json",
        kill_switch_path=tmp_path / "KILL",
        mode="exness12",
        paper_balance=12.0,
        fixed_lots=0.01,
    )
    paper = ExnessPaperExecutor(balance=12.0)
    runner = LiveRunner(cfg, paper_executor=paper)
    assert runner.paper is paper
    assert runner.cfg.journal_path != REPO_ROOT / "data" / "trade_journal.jsonl"
    assert runner.cfg.fixed_lots == pytest.approx(0.01)
    # Default construction is unchanged: OANDA-model executor, no fixed lots.
    default = LiveRunner(RunnerConfig(
        journal_path=tmp_path / "live_journal.jsonl",
        risk_state_path=tmp_path / "live_risk.json",
        kill_switch_path=tmp_path / "KILL2"))
    assert isinstance(default.paper, PaperExecutor)
    assert not isinstance(default.paper, ExnessPaperExecutor)
    assert default.cfg.fixed_lots is None
