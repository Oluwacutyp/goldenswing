"""Step 12d (2026-10-29): broker-closed positions are priced from
the broker's closing DEAL, not the reconcile tick snapshot.

Evidence: on 2026-10-28 the FOMC long (ticket 520020067) hit its
TP at 4046.188 (+$5.88, TP badge in the owner's Exness app) but
the journal recorded exit 4042.89 / +0.50R / +$2.58 — the price
snapshot from the moment reconcile noticed the position gone.
Smaller errors on every other broker close (10294: journal
4040.731 vs deal 4041.074; 10395: 4028.26 vs 4028.622; 10341:
4029.225 vs 4029.081).  With deal prices, the close classifier
also names TP hits correctly instead of "broker_closed".
"""

import json
from types import SimpleNamespace

from goldenswing.execution.executor import OrderRequest, OrderResult
from goldenswing.execution.mt5 import MT5Executor
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec


def _deal(price, reason, entry=1, time=100, position_id=7):
    return SimpleNamespace(price=price, profit=1.0, reason=reason,
                           entry=entry, time=time,
                           position_id=position_id)


class FakeLib:
    def __init__(self, deals, *, position_raises=False):
        self._deals = deals
        self._position_raises = position_raises

    def history_deals_get(self, *args, **kwargs):
        if "position" in kwargs:
            if self._position_raises:
                raise RuntimeError("old build")
            return [d for d in self._deals
                    if d.position_id == kwargs["position"]]
        return list(self._deals)  # window form: runner filters


def test_closing_deal_picks_out_deal_and_maps_reason():
    lib = FakeLib([_deal(1990.0, 3, entry=0, time=90),
                   _deal(4046.19, 5, entry=1, time=100)])
    ex = MT5Executor(mt5_module=lib)
    deal = ex.closing_deal(7)
    assert deal["price"] == 4046.19 and deal["reason"] == "tp"


def test_closing_deal_window_fallback():
    lib = FakeLib([_deal(4028.62, 4, position_id=7),
                   _deal(1111.0, 4, position_id=8)],
                  position_raises=True)
    ex = MT5Executor(mt5_module=lib)
    deal = ex.closing_deal(7)
    assert deal["price"] == 4028.62 and deal["reason"] == "sl"


def test_closing_deal_none_when_no_out_deal():
    lib = FakeLib([_deal(1990.0, 3, entry=0)])
    assert MT5Executor(mt5_module=lib).closing_deal(7) is None


# ------------------------------------------------------------ runner level

class FakeSender:
    def send_text(self, text, silent=False, reply_markup=None):
        return SimpleNamespace(delivered=True, detail="ok")

    def send_photo(self, png, caption):
        return SimpleNamespace(delivered=True, detail="ok")


class FakeExecutor:
    def __init__(self, deal=None):
        self._deal = deal

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return []

    def closing_deal(self, ticket):
        return self._deal

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        return OrderResult(True, ticket=ticket)


def _runner_with_position(tmp_path, deal):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, executor=FakeExecutor(deal))
    runner.senders = [FakeSender()]
    req = OrderRequest(symbol="XAUUSDm", direction=1, volume=0.01,
                       stop_loss=4035.94, entry_ref=4040.51,
                       targets=[4046.19], atr=3.12,
                       alert_id="MT5-qm_m5-10163", bar_index=0,
                       magic=260101)
    runner.tick_manager.open(ticket=7, request=req, fill_price=4040.31,
                             entry_bar=0, broker_sl=4035.16,
                             playbook="qm_m5")
    return runner


def _close_record(tmp_path):
    for ln in (tmp_path / "mt5_journal.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r.get("type") == "close":
            return r
    raise AssertionError("no close record")


def test_reconcile_prices_close_from_deal(tmp_path):
    # The 10163 shape: TP deal at 4046.19, snapshot says 4042.89.
    runner = _runner_with_position(
        tmp_path, {"price": 4046.19, "profit": 5.88, "reason": "tp",
                   "time": 100})
    runner._reconcile(4042.89, {"reconciled_closed": 0})
    rec = _close_record(tmp_path)
    assert rec["exit"] == 4046.19
    assert rec["r_net"] > 1.1  # the true +1.14R, not +0.50R
    notes = (tmp_path / "mt5_journal.jsonl").read_text()
    assert "priced from the broker deal (tp @ 4046.19)" in notes
    assert "TP hit" in notes


def test_reconcile_falls_back_to_snapshot_without_deal(tmp_path):
    runner = _runner_with_position(tmp_path, None)
    runner._reconcile(4042.89, {"reconciled_closed": 0})
    rec = _close_record(tmp_path)
    assert rec["exit"] == 4042.89
