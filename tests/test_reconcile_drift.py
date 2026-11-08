"""§3.11 reconciliation audit — the modified state + provenance.

What already held (pinned elsewhere, re-verified this pass):
pending fills are detected by broker VOLUME, never assumed
cancelled (test_mt5_pending* / step-12 suites); broker-closed
positions are priced from the broker's closing DEAL
(test_step12d_deal_prices.py).

What this file pins — the gaps the audit found:

1. Close-price provenance was journaled only when the DEAL
   priced the exit.  Every fallback to the tick snapshot
   returned a None note, so a snapshot-priced close was
   indistinguishable in the journal and the misses the step-12d
   docstring promised to keep countable were not countable.
   Now every broker_closed close names its price source.

2. Modified state never reconciled: ``_reconcile`` compared
   only presence, so a broker-side SL move/removal, TP change
   or volume change on a TRACKED position was invisible — the
   tick manager managed, and the close classifier classified,
   against a stop the broker no longer held.  Drift is now
   journaled with both values (once per episode), the stop
   RECORD syncs to the broker (a ledger correction, never an
   order), and TP/volume drift is journaled without being
   auto-booked.  Resting pending orders get the same treatment.

3. The pending-fill open record still substituted ``sig.score``
   when ``placed_score`` was missing — the sleeve-base
   placeholder step 2 (2026-10-27) removed from that record.
   An unrecorded decision score now journals as 0 with a note,
   never as a substitute number.
"""
import json
from types import SimpleNamespace

from goldenswing.execution.executor import OrderRequest, OrderResult
from goldenswing.execution.mt5_runner import (MT5Runner,
                                              MT5RunnerConfig,
                                              PendingOrderState)
from goldenswing.execution.risk import gold_spec
from goldenswing.execution.tick_manager import TickAction


class FakeSender:
    def send_text(self, text, silent=False, reply_markup=None):
        return SimpleNamespace(delivered=True, detail="ok")

    def send_photo(self, png, caption):
        return SimpleNamespace(delivered=True, detail="ok")


class FakeExecutor:
    def __init__(self, *, deal=None, positions=(), pendings=()):
        self._deal = deal
        self._positions = list(positions)
        self._pendings = list(pendings)

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return list(self._positions)

    def pending_orders(self, symbol=None):
        return list(self._pendings)

    def closing_deal(self, ticket):
        return self._deal

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        return OrderResult(True, ticket=ticket)


class NoDealExecutor(FakeExecutor):
    """An executor with no closing-deal history at all."""
    closing_deal = None


def _runner(tmp_path, executor):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, executor=executor)
    runner.senders = [FakeSender()]
    return runner


def _open_tracked(runner, ticket=7):
    req = OrderRequest(symbol="XAUUSDm", direction=1, volume=0.01,
                       stop_loss=4035.94, entry_ref=4040.51,
                       targets=[4046.19], atr=3.12,
                       alert_id="MT5-qm_m5-10163", bar_index=0,
                       magic=260101)
    runner.tick_manager.open(ticket=ticket, request=req,
                             fill_price=4040.31, entry_bar=0,
                             broker_sl=4035.16, playbook="qm_m5")
    return req


def _broker_position(ticket=7, **over):
    pos = SimpleNamespace(
        ticket=ticket, magic=260101, alert_id="MT5-qm_m5-10163",
        symbol="XAUUSDm", direction=1, volume=0.01,
        entry_price=4040.31, sl=4035.16, tp=4046.19)
    for key, value in over.items():
        setattr(pos, key, value)
    return pos


def _journal_text(tmp_path):
    return (tmp_path / "mt5_journal.jsonl").read_text()


def _records(tmp_path, type_):
    return [json.loads(ln) for ln in _journal_text(tmp_path).splitlines()
            if json.loads(ln).get("type") == type_]


# ------------------------------------------------- provenance (gap 1)

def test_snapshot_fallback_close_journals_its_price_source(tmp_path):
    runner = _runner(tmp_path, FakeExecutor(deal=None))
    _open_tracked(runner)
    runner._reconcile(4042.89, {"reconciled_closed": 0})
    rec = _records(tmp_path, "close")[0]
    assert rec["exit"] == 4042.89  # snapshot price, as before
    # ...but now the journal SAYS the snapshot priced it.
    assert "priced from the tick snapshot" in _journal_text(tmp_path)


def test_executor_without_deal_history_journals_fallback(tmp_path):
    runner = _runner(tmp_path, NoDealExecutor())
    _open_tracked(runner)
    runner._reconcile(4042.89, {"reconciled_closed": 0})
    assert "priced from the tick snapshot" in _journal_text(tmp_path)


# ------------------------------------------------ modified state (gap 2)

def test_external_sl_drift_journaled_and_record_synced(tmp_path):
    ex = FakeExecutor(positions=[_broker_position(sl=4030.00)])
    runner = _runner(tmp_path, ex)
    _open_tracked(runner)
    stats = {"reconciled_closed": 0}
    runner._reconcile(4042.89, stats)
    text = _journal_text(tmp_path)
    assert "ticket 7 broker drift" in text
    assert "broker SL 4030.00 != tracked 4035.16" in text
    assert stats["reconcile_drift"] == 1
    # The stop RECORD follows the broker (ledger, not an order):
    tm = runner.tick_manager
    assert tm.effective_stop(7) == 4030.00
    assert tm.last_sl_reason(7) == "external"
    # Step 13 is untouched: the ENGINE still may not book a
    # modify for this fixed-bracket position.
    tm.apply(7, TickAction(kind="modify_sl", level=4038.0,
                           reason="trail"))
    assert tm.effective_stop(7) == 4030.00


def test_sl_drift_notes_once_per_episode(tmp_path):
    ex = FakeExecutor(positions=[_broker_position(sl=4030.00)])
    runner = _runner(tmp_path, ex)
    _open_tracked(runner)
    stats = {"reconciled_closed": 0}
    runner._reconcile(4042.89, stats)
    runner._reconcile(4042.89, stats)
    assert _journal_text(tmp_path).count("broker drift") == 1
    assert stats["reconcile_drift"] == 1


def test_missing_broker_stop_journaled(tmp_path):
    ex = FakeExecutor(positions=[_broker_position(sl=None)])
    runner = _runner(tmp_path, ex)
    _open_tracked(runner)
    runner._reconcile(4042.89, {"reconciled_closed": 0})
    text = _journal_text(tmp_path)
    assert "broker stop MISSING (tracked 4035.16)" in text
    assert runner.tick_manager._states[7].broker_sl is None


def test_volume_drift_journaled_not_booked(tmp_path):
    ex = FakeExecutor(positions=[_broker_position(volume=0.02)])
    runner = _runner(tmp_path, ex)
    _open_tracked(runner)
    stats = {"reconciled_closed": 0}
    runner._reconcile(4042.89, stats)
    runner._reconcile(4042.89, stats)
    text = _journal_text(tmp_path)
    assert "broker volume 0.02 != tracked 0.01" in text
    assert text.count("broker drift") == 1  # once per episode
    # Journaled, NOT auto-booked: tracked volume is unchanged.
    assert runner.tick_manager.position(7).volume == 0.01


def test_tp_drift_journaled_when_both_set(tmp_path):
    ex = FakeExecutor(positions=[_broker_position(tp=4050.00)])
    runner = _runner(tmp_path, ex)
    _open_tracked(runner)
    runner._reconcile(4042.89, {"reconciled_closed": 0})
    assert "broker TP 4050.00 != tracked first target 4046.19" \
        in _journal_text(tmp_path)


def test_pending_price_drift_journaled_once(tmp_path):
    order = SimpleNamespace(ticket=99, kind="limit", price=4036.00,
                            sl=4041.44, magic=260101)
    ex = FakeExecutor(pendings=[order])
    runner = _runner(tmp_path, ex)
    state = PendingOrderState(
        ticket=99, kind="limit", direction=-1, level=4034.112,
        volume=0.01, sl=4041.44, tp=4023.002,
        alert_id="MT5-qm_m5-10395", playbook="qm_m5", zone=None,
        invalidation=4039.614, placed_bar=0, expiry_bar=99,
        premise="native", entry_clock="M5", track_request=None)
    runner._pending_orders[99] = state
    stats = {"reconciled_closed": 0}
    runner._reconcile(4000.0, stats)
    runner._reconcile(4000.0, stats)
    text = _journal_text(tmp_path)
    assert "pending ticket 99 broker drift" in text
    assert "broker price 4036.00 != placed 4034.11" in text
    assert text.count("pending ticket 99 broker drift") == 1
    # Placed state kept — the drift is journaled, not smoothed in.
    assert runner._pending_orders[99].level == 4034.112


# --------------------------------------- pending-fill score (gap 3)

def _pending_fill(runner, *, placed_score, sig_score):
    track = OrderRequest(
        symbol="XAUUSDm", direction=-1, volume=0.01,
        stop_loss=4039.614, entry_ref=4034.112,
        targets=[4023.002], atr=7.3,
        alert_id="MT5-qm_m5-10395", bar_index=0, magic=260101)
    state = PendingOrderState(
        ticket=522691421, kind="limit", direction=-1,
        level=4034.112, volume=0.01, sl=4041.44, tp=4023.002,
        alert_id="MT5-qm_m5-10395", playbook="qm_m5", zone=None,
        invalidation=4039.614, placed_bar=0, expiry_bar=99,
        premise="native", entry_clock="M5", track_request=track,
        placed_score=placed_score,
        sig=SimpleNamespace(score=sig_score, factors=[]))
    runner._pending_orders[522691421] = state
    pos = SimpleNamespace(ticket=522691421, volume=0.01,
                          entry_price=4034.112)
    runner._record_pending_fill(state, pos, {})


def test_fill_without_placed_score_records_zero_not_substitute(tmp_path):
    runner = _runner(tmp_path, FakeExecutor())
    _pending_fill(runner, placed_score=0.0, sig_score=60.0)
    rec = _records(tmp_path, "open")[0]
    # NOT the sig's 60.0 placeholder: unrecorded means 0 + a note.
    assert rec["score"] == 0.0
    assert "no recorded decision score" in _journal_text(tmp_path)


def test_fill_with_placed_score_records_it_and_no_note(tmp_path):
    runner = _runner(tmp_path, FakeExecutor())
    _pending_fill(runner, placed_score=32.71, sig_score=60.0)
    rec = _records(tmp_path, "open")[0]
    assert rec["score"] == 32.71
    assert "no recorded decision score" not in _journal_text(tmp_path)
