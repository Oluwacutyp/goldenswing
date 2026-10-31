"""Step 17 acceptance tests — event-aware same-thesis
revalidation replacing the flat cooldown (owner rulings
2026-10-30 §5).

Journal evidence: the Oct-30 waterfall — the 14:45:33 hot_scalp
continuation candidate (same parent setup as the 14:41 fill,
target 3891.186, net R/R 7.04) was refused by the flat entry-gate
cooldown ~40s after the parent trade stopped out; price reached
~3882.  Under Step 17 that candidate — a NEW trigger under the
SAME parent setup — is admitted as a revalidation; a re-fire of
the SAME trigger, an unrelated thesis, and an identity-less
candidate stay refused, and caps/dedup are never lifted.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from goldenswing.execution.entry_gates import EntryGate, EntryGateConfig


def _gate(**kw):
    now = [1_000_000.0]
    gate = EntryGate(EntryGateConfig(**kw), clock=lambda: now[0])
    return gate, now


# ------------------------------------------------------------ gate units
def test_same_thesis_new_trigger_lifts_the_cooldown():
    gate, now = _gate(loss_cooldown_sec=300.0)
    gate.record_open(now[0], "fp-a")
    gate.record_close(now[0], won=False,
                      setup_key="S1", trigger_key="S1@T1")
    now[0] += 40.0  # the Oct-30 gap: ~40 s after the stop-out
    ok, reason = gate.check_new_entry(
        now[0], "fp-b", setup_key="S1", trigger_key="S1@T2")
    assert (ok, reason) == (True, "revalidated")


def test_same_trigger_stays_refused():
    gate, now = _gate(loss_cooldown_sec=300.0)
    gate.record_open(now[0], "fp-a")
    gate.record_close(now[0], won=False,
                      setup_key="S1", trigger_key="S1@T1")
    now[0] += 40.0
    # A re-fire of the very trigger whose trade just closed: the
    # cooldown's same-trigger suppression is NOT lifted.
    assert gate.check_new_entry(
        now[0], "fp-b", setup_key="S1",
        trigger_key="S1@T1") == (False, "cooldown")


def test_unrelated_or_identity_less_candidates_stay_refused():
    gate, now = _gate(loss_cooldown_sec=300.0)
    gate.record_open(now[0], "fp-a")
    gate.record_close(now[0], won=False,
                      setup_key="S1", trigger_key="S1@T1")
    now[0] += 40.0
    assert gate.check_new_entry(
        now[0], "fp-b", setup_key="S2",
        trigger_key="S2@T9") == (False, "cooldown")
    assert gate.check_new_entry(now[0], "fp-b") == (False, "cooldown")
    # A claim with a setup but NO trigger event identity is not
    # a revalidation event (ruling condition 4).
    assert gate.check_new_entry(
        now[0], "fp-b", setup_key="S1") == (False, "cooldown")


def test_dedup_still_refuses_after_a_lift():
    gate, now = _gate(loss_cooldown_sec=300.0)
    gate.record_open(now[0], "fp-a")
    gate.record_close(now[0], won=False,
                      setup_key="S1", trigger_key="S1@T1")
    now[0] += 40.0
    # Same levels re-firing = the same signal, whatever the
    # trigger claims: fingerprint dedup refuses it.
    assert gate.check_new_entry(
        now[0], "fp-a", setup_key="S1",
        trigger_key="S1@T2") == (False, "duplicate")


def test_caps_are_never_lifted():
    gate, now = _gate(loss_cooldown_sec=300.0, max_daily_trades=1)
    gate.record_open(now[0], "fp-a")
    gate.record_close(now[0], won=False,
                      setup_key="S1", trigger_key="S1@T1")
    now[0] += 40.0
    assert gate.check_new_entry(
        now[0], "fp-b", setup_key="S1",
        trigger_key="S1@T2") == (False, "cap_daily")


def test_expired_cooldown_is_an_ordinary_pass():
    gate, now = _gate(loss_cooldown_sec=300.0)
    gate.record_open(now[0], "fp-a")
    gate.record_close(now[0], won=False,
                      setup_key="S1", trigger_key="S1@T1")
    now[0] += 301.0
    assert gate.check_new_entry(
        now[0], "fp-b", setup_key="S1",
        trigger_key="S1@T2") == (True, "")


def test_unstamped_close_records_no_claimable_identity():
    gate, now = _gate(loss_cooldown_sec=300.0)
    gate.record_open(now[0], "fp-a")
    gate.record_close(now[0], won=False)  # legacy call shape
    now[0] += 40.0
    assert gate.check_new_entry(
        now[0], "fp-b", setup_key="S1",
        trigger_key="S1@T2") == (False, "cooldown")


# ------------------------------------------------------- runner wiring
class _FakeFeed:
    def account_state(self):
        return {"equity": 10000.0, "balance": 10000.0,
                "free_margin": 10000.0}


class _FakeExecutor:
    def __init__(self):
        self.orders = []
        self._ticket = 9000

    def symbol_spec(self, symbol):
        from goldenswing.execution.risk import gold_spec
        return gold_spec()

    def place_order(self, request):
        from goldenswing.execution.mt5 import OrderResult
        self._ticket += 1
        self.orders.append(request)
        return OrderResult(True, ticket=self._ticket,
                           price=request.entry_ref)

    def place_pending(self, request):
        return self.place_order(request)

    def positions(self):
        return []


def _runner(tmp_path):
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False, max_daily_trades=0,
        max_hourly_trades=0, loss_cooldown_sec=300,
        fingerprint_ttl_sec=3600)
    runner = MT5Runner(cfg, feed=_FakeFeed(), executor=_FakeExecutor())
    runner._market_open = lambda: (True, "open")
    runner._check_risk_guards = lambda *a: True
    runner._check_margin = lambda *a: True
    runner._alignment = SimpleNamespace(
        size_multiplier=1.0, alignment_label="D1_Only")
    runner._last_tick = {"mid": 2005.0, "bid": 2004.8, "ask": 2005.2,
                         "time": "2026-10-14 12:00:00"}  # pinned Wed
    runner._stack = SimpleNamespace(frames={"H1": [1, 2, 3]})
    return runner


def _sig(index, entry, inval, setup="", trigger=""):
    sig = SimpleNamespace(direction=1, entry_ref=entry,
                          invalidation=inval, atr=10.0,
                          targets=[2040.0], score=80.0, index=index,
                          factors=[])
    if setup:
        sig.setup_key = setup
    if trigger:
        sig.trigger_key = trigger
    return sig


def test_runner_full_loop_close_identity_feeds_the_gate(tmp_path):
    runner = _runner(tmp_path)
    pb = SimpleNamespace(name="sweep_reversal")
    stats: dict = {}
    runner._place_live_order(
        pb, _sig(42, 2005.0, 1990.0, "S1", "S1@T1"), stats)
    assert stats.get("executed") == 1
    ticket = runner.executor.orders[0].ticket \
        if hasattr(runner.executor.orders[0], "ticket") else 9001
    # The placed order's identity is on file under its alert_id.
    assert runner._open_identity, "open identity recorded"
    (sk, tk), = runner._open_identity.values()
    assert (sk, tk) == ("S1", "S1@T1")
    # Losing close: the gate learns WHICH setup closed.
    runner._record_close(ticket, 1995.0, "stop",
                         {"reconciled_closed": 0})
    assert runner.entry_gate._last_close_setup_key == "S1"
    assert runner.entry_gate._last_close_trigger_key == "S1@T1"
    assert runner._open_identity == {}  # consumed by the close


def test_runner_revalidation_admits_the_oct30_shape(tmp_path):
    runner = _runner(tmp_path)
    pb = SimpleNamespace(name="hot_scalp")
    stats: dict = {}
    runner._place_live_order(
        pb, _sig(42, 2005.0, 1990.0,
                 "hot_scalp:hs-1", "hot_scalp:hs-1@trig-1"), stats)
    assert stats.get("executed") == 1
    runner._record_close(9001, 1995.0, "stop",
                         {"reconciled_closed": 0})
    # ~40 s later (the gate reads the wall clock; the close just
    # happened, so we are inside the window by construction): a
    # NEW trigger under the SAME parent setup, new levels.
    stats2: dict = {}
    runner._place_live_order(
        pb, _sig(43, 2005.0, 1992.0,
                 "hot_scalp:hs-1", "hot_scalp:hs-1@trig-2"), stats2)
    assert stats2.get("gate_cooldown", 0) == 0
    assert stats2.get("gate_revalidated") == 1
    assert stats2.get("executed") == 1
    journal = (tmp_path / "j.jsonl").read_text()
    assert "cooldown lifted by same-thesis revalidation" in journal
    assert "hot_scalp:hs-1@trig-2" in journal


def test_runner_same_trigger_and_stranger_stay_refused(tmp_path):
    runner = _runner(tmp_path)
    pb = SimpleNamespace(name="hot_scalp")
    stats: dict = {}
    runner._place_live_order(
        pb, _sig(42, 2005.0, 1990.0,
                 "hot_scalp:hs-1", "hot_scalp:hs-1@trig-1"), stats)
    assert stats.get("executed") == 1
    runner._record_close(9001, 1995.0, "stop",
                         {"reconciled_closed": 0})
    # The SAME trigger re-presented (new levels): still refused.
    stats2: dict = {}
    runner._place_live_order(
        pb, _sig(43, 2005.0, 1992.0,
                 "hot_scalp:hs-1", "hot_scalp:hs-1@trig-1"), stats2)
    assert stats2.get("gate_cooldown") == 1
    assert stats2.get("executed", 0) == 0
    # An identity-less candidate: still refused.
    stats3: dict = {}
    runner._place_live_order(pb, _sig(44, 2005.0, 1993.0), stats3)
    assert stats3.get("gate_cooldown") == 1
    assert stats3.get("executed", 0) == 0


def test_existing_entry_gate_contract_unchanged():
    # The pre-Step-17 call shapes keep working positionally.
    gate, now = _gate(loss_cooldown_sec=300.0)
    assert gate.check_new_entry(now[0], "a") == (True, "")
    gate.record_open(now[0], "a")
    gate.record_close(now[0], True)
    assert gate.check_new_entry(now[0], "b") == (True, "")
