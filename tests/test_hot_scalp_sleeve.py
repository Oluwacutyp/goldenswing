"""Completion-03 (owner program 2026-11-05): the Hot Scalp
sleeve is multi-position.

Flat-only v1 (promotion.flat_only_holding) is superseded: a
triggered candidate may STACK onto held sleeve positions when

- it is independently justified — its own setup instance from
  the candidate chain; the same setup_id never opens twice;
- a stack on the SAME parent thesis revalidates at promotion
  time, read fresh from the setup store (thesis not invalidated,
  zone not consumed) — a stale trigger riding a dead thesis read
  is refused under its own reason;
- aggregate stop exposure — per parent thesis, and across the
  sleeve — plus the candidate's estimated exposure fits the
  envelope the RiskPool ledger budgets one position with
  (equity x risk_pct x layer_max_total_risk_mult; the sleeve
  scope scales it by the guards' position cap).

The shared order path's safeguards (RiskGuards count/risk,
friction, margin) are untouched and still bind downstream.

Harness note: ``runner._hs_parents`` is populated by chain
events in the live loop (``_hs_chain_from_event``) before
promotion runs; these tests call promotion directly, so they
record the same mapping explicitly after each open.
"""

import copy
import dataclasses
import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goldenswing.execution.executor import OrderRequest, OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.live.hierarchy import TimeframeStack
from goldenswing.live.hot_scalp import HotScalpConfig, HotScalpResearchEngine
from goldenswing.live.setup_store import SetupStore

NOW = "2026-10-27T00:50:30+00:00"


def _frame(tf_minutes, start, count, *, base=100.0):
    idx = pd.date_range(start, periods=count, freq=f"{tf_minutes}min")
    k = np.arange(count, dtype=float)
    close = base + 0.025 * np.sin(k / 2.0)
    open_ = close - 0.02 * np.cos(k)
    high = np.full(count, 103.0)
    low = np.full(count, 99.0)
    high = np.maximum(high, np.maximum(open_, close) + 0.1)
    low = np.minimum(low, np.minimum(open_, close) - 0.1)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": np.full(count, 100.0)},
                        index=idx)


class _MarketExec:
    """Fake broker: market fills synchronously at entry_ref."""

    def __init__(self):
        self._positions = []
        self.placed = []
        self._next_ticket = 777

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return [p for p in self._positions
                if symbol is None or p.symbol == symbol]

    def pending_orders(self, symbol=None):
        return []

    def cancel_pending(self, ticket):
        return OrderResult(True, ticket=ticket)

    def place_order(self, request):
        self.placed.append(request)
        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions.append(Position(
            ticket=ticket, symbol=request.symbol,
            direction=request.direction, volume=request.volume,
            entry_price=float(request.entry_ref),
            sl=float(request.stop_loss), tp=request.take_profit,
            magic=request.magic, alert_id=request.alert_id))
        return OrderResult(True, ticket=ticket,
                           price=float(request.entry_ref),
                           volume=request.volume)


def _runner(tmp_path, *, equity=1100.0):
    # PASS 5 (owner ruling 2026-11-08): this suite pins the
    # sleeve's stacking / risk mechanics, which operate on
    # placed orders — so its runners carry the owner's
    # explicit config promotion for the route.  The boundary
    # at the declared posture is pinned in
    # tests/test_route_authority.py.
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False, hot_scalp_live_enabled=True,
        judgment_route_modes={"hot_scalp_promotion":
                              "governing"})
    feed = SimpleNamespace(
        account_state=lambda: {"equity": equity, "balance": equity,
                               "free_margin": equity},
        order_margin=lambda *, volume, price, direction:
            float(volume) * 100.0 * float(price) / 500.0)
    runner = MT5Runner(cfg, feed=feed, executor=_MarketExec())
    runner._market_open = lambda: (True, "test")
    runner._stack = TimeframeStack.from_frames({
        "H1": _frame(60, "2026-10-27T00:00:00Z", 60),
        "M5": _frame(5, "2026-10-27T00:00:00Z", 10),
        "M15": _frame(15, "2026-10-26T12:15:00Z", 50)})
    runner._hot_scalp_aux_frames = {
        "M1": _frame(1, "2026-10-27T00:00:00Z", 50)}
    store = SetupStore(tmp_path / "hot-scalp.sqlite")
    runner._hot_scalp = HotScalpResearchEngine(
        store, HotScalpConfig(min_m1_bars=100, min_m5_bars=100,
                              min_m15_bars=100),
        instance_id="process-A")
    return runner


def _candidate(runner, *, setup_id, parent_id, entry=100.30, stop=90.30,
               target=115.0, zone=(100.0, 100.6), direction=1):
    """Arm + touch + claim one setup; return its candidate dict
    in the engine's shape."""
    setup = {
        "setup_id": setup_id, "parent_setup_id": parent_id,
        "symbol": "XAUUSD", "strategy": "breakout_retest",
        "direction": direction,
        "formation_ts": "2026-10-27T00:40:00+00:00",
        "armed_ts": "2026-10-27T00:50:00+00:00",
        "expires_at": "2026-10-27T01:25:00+00:00",
        "level": entry - 0.1, "zone_low": zone[0], "zone_high": zone[1],
        "stop": stop, "structure_ts": "2026-10-27T00:40:00+00:00",
        "context_direction": direction, "atr_m5": 0.5,
        "description": "sleeve test setup",
    }
    store = runner._hot_scalp.store
    store.arm(setup)
    store.mark_touched(setup_id, NOW)
    claim = store.claim_trigger(setup_id, "process-A", NOW)
    assert claim is not None, "test setup claim must succeed"
    return {
        "event": "candidate_shadow", "setup_id": setup_id,
        "parent_setup_id": parent_id, "strategy": "breakout_retest",
        "direction": direction, "entry_reference": entry,
        "structural_stop": stop, "target_reference": target,
        "spread": 0.02, "score": 60.0, "trigger_ts": NOW,
        "claim": claim,
    }


def _tick(*, mid=100.30):
    return {"time": NOW, "mid": mid, "bid": mid - 0.01, "ask": mid + 0.01}


def _promote(runner, cand):
    # Each promotion lands on its own bar, as in the live loop —
    # the signal index feeds the alert id, and two fills sharing
    # an alert id would collide in the step-17 identity map.
    counter = getattr(runner, "_bar_counter", None)
    if counter is None:
        counter = runner._bar_counter = itertools.count(100)
    runner._current_h1_bar = lambda: next(counter)
    # The live loop's chain events record the setup -> parent
    # mapping (_hs_chain_from_event) BEFORE promotion runs; the
    # direct-call harness records it at the same point.
    parents = getattr(runner, "_hs_parents", None)
    if parents is None:
        parents = runner._hs_parents = {}
    parents[cand["setup_id"]] = cand["parent_setup_id"]
    return runner._promote_hot_scalp_candidate(cand, _tick(), {})


def _status(runner, setup_id):
    row = runner._hot_scalp.store.get(setup_id)
    assert row is not None
    return row["status"]


def _chain_records(runner, *, stage=None, chain_id=None):
    out = []
    for r in runner.journal.records():
        if r.get("type") != "candidate_chain":
            continue
        if stage is not None and r.get("stage") != stage:
            continue
        if chain_id is not None and r.get("chain_id") != chain_id:
            continue
        out.append(r)
    return out


def _refusals(runner, setup_id):
    return [r for r in _chain_records(runner, stage="promotion",
                                      chain_id=setup_id)
            if r.get("execution_status") in ("refused", "withheld")]


def _notes(runner):
    return "\n".join(
        r.get("text", "") for r in runner.journal.records()
        if r.get("type") == "note")


# ------------------------------------------------------------ (a) stacking

def test_two_independent_setups_stack_and_manage_independently(tmp_path):
    runner = _runner(tmp_path, equity=10000.0)
    a = _candidate(runner, setup_id="hs-a", parent_id="hp-a",
                   entry=100.30, stop=90.30, target=115.0,
                   zone=(95.0, 106.0))
    b = _candidate(runner, setup_id="hs-b", parent_id="hp-b",
                   entry=100.30, stop=92.30, target=112.0,
                   zone=(99.9, 100.5))
    assert _promote(runner, a) is True
    assert _promote(runner, b) is True
    assert len(runner.executor.placed) == 2
    tickets = sorted(runner.tick_manager.tickets())
    assert len(tickets) == 2
    assert _status(runner, "hs-a") == "order_reconciled"
    assert _status(runner, "hs-b") == "order_reconciled"
    # Each position carries its OWN zone / targets / stop geometry.
    sleeve = {s["setup_id"]: s
              for s in runner._hot_scalp_sleeve_positions()}
    assert set(sleeve) == {"hs-a", "hs-b"}
    ta, tb = sleeve["hs-a"]["ticket"], sleeve["hs-b"]["ticket"]
    assert runner.tick_manager.entry_zone(ta) == (95.0, 106.0)
    assert runner.tick_manager.entry_zone(tb) == (99.9, 100.5)
    pa, pb = (runner.tick_manager.position(ta),
              runner.tick_manager.position(tb))
    assert pa.targets != pb.targets
    assert pa.invalidation == 90.30 and pb.invalidation == 92.30
    # Independent management: B at a favourable price earns no
    # profit lock (its lock is EARNED, and B earned nothing);
    # A's earned lock then fires for A only and leaves B's whole
    # state untouched.
    acts_b = runner.tick_manager.on_tick(tb, 111.0)
    assert [x for x in acts_b
            if x.kind == "modify_sl" and x.reason == "profit_lock"] == []
    snap_b = (dataclasses.asdict(pb),
              copy.deepcopy(
                  runner.tick_manager._states[tb].events),
              runner.tick_manager._states[tb].broker_sl,
              runner._pools[tb].state())
    assert runner.tick_manager.mark_pl_earned(
        ta, "acceptance: M5 close beyond TP1 (test)") is True
    locks = [x for x in runner.tick_manager.on_tick(ta, 112.45)
             if x.kind == "modify_sl" and x.reason == "profit_lock"]
    assert len(locks) == 1
    expected = pa.entry_price + max(0.5 * pa.risk_price,
                                    0.75 * pa.atr)
    assert locks[0].level == pytest.approx(expected, abs=0.02)
    runner.tick_manager.apply(ta, locks[0])
    assert runner.tick_manager._states[ta].broker_sl == \
        pytest.approx(expected, abs=0.02)
    assert (dataclasses.asdict(
                runner.tick_manager.position(tb)),
            runner.tick_manager._states[tb].events,
            runner.tick_manager._states[tb].broker_sl,
            runner._pools[tb].state()) == snap_b


def test_same_thesis_stack_within_budget_opens(tmp_path):
    runner = _runner(tmp_path, equity=1100.0)
    a1 = _candidate(runner, setup_id="hs-x1", parent_id="hp-x",
                    stop=90.30, zone=(100.0, 100.6))
    a2 = _candidate(runner, setup_id="hs-x2", parent_id="hp-x",
                    stop=91.30, zone=(99.8, 100.4))
    assert _promote(runner, a1) is True
    assert _promote(runner, a2) is True
    assert len(runner.tick_manager.tickets()) == 2
    # The parent record on the chain shows both children and
    # their aggregate after the second open.
    sleeve_records = _chain_records(runner, stage="sleeve",
                                    chain_id="hp-x")
    assert sleeve_records, "parent sleeve records must exist"
    last = sleeve_records[-1]
    assert last["execution_status"] == "position_opened"
    assert last["open_positions"] == 2
    assert set(last["child_setups"]) == {"hs-x1", "hs-x2"}
    assert last["aggregate_stop_exposure"] == pytest.approx(
        19.25, abs=1.0)


# ------------------------------------------------- (b) aggregate budget

def test_parent_risk_budget_refuses_third_same_thesis_stack(tmp_path):
    runner = _runner(tmp_path, equity=1100.0)
    a1 = _candidate(runner, setup_id="hs-x1", parent_id="hp-x",
                    stop=90.30, zone=(100.0, 100.6))
    a2 = _candidate(runner, setup_id="hs-x2", parent_id="hp-x",
                    stop=91.30, zone=(99.8, 100.4))
    a3 = _candidate(runner, setup_id="hs-x3", parent_id="hp-x",
                    stop=89.30, zone=(100.1, 100.7))
    assert _promote(runner, a1) is True
    assert _promote(runner, a2) is True
    assert _promote(runner, a3) is False
    assert len(runner.executor.placed) == 2
    refusals = _refusals(runner, "hs-x3")
    assert [r["reason_code"] for r in refusals] == \
        ["promotion.parent_risk_budget"]
    rec = refusals[0]
    # The journal carries authorized vs aggregate, distinguishable.
    assert rec["authorized_envelope"] == pytest.approx(22.0)
    assert rec["aggregate_exposure"] > rec["authorized_envelope"]
    assert rec["open_exposure"] == pytest.approx(19.25, abs=1.0)
    assert rec["candidate_estimated_exposure"] == pytest.approx(
        11.125, abs=0.5)
    assert _status(runner, "hs-x3") == "touched"  # transient: released


def test_sleeve_risk_budget_refuses_across_theses(tmp_path):
    runner = _runner(tmp_path, equity=1100.0)
    p1 = _candidate(runner, setup_id="hs-p1", parent_id="hp-1",
                    stop=75.30, zone=(100.0, 100.6))
    p2 = _candidate(runner, setup_id="hs-p2", parent_id="hp-2",
                    stop=74.30, zone=(100.0, 100.6))
    p3 = _candidate(runner, setup_id="hs-p3", parent_id="hp-3",
                    stop=90.30)
    p4 = _candidate(runner, setup_id="hs-p4", parent_id="hp-4",
                    stop=91.30)
    assert _promote(runner, p1) is True
    assert _promote(runner, p2) is True
    assert _promote(runner, p3) is True
    # Sleeve aggregate now ~60.4 of a 66 envelope; one more
    # estimated 10.1 breaches it — at the sleeve gate, under its
    # own reason (the order path's count cap would also bind,
    # but the accounting refusal is the one recorded).
    assert _promote(runner, p4) is False
    assert len(runner.executor.placed) == 3
    refusals = _refusals(runner, "hs-p4")
    assert [r["reason_code"] for r in refusals] == \
        ["promotion.sleeve_risk_budget"]
    assert refusals[0]["authorized_envelope"] == pytest.approx(66.0)
    assert refusals[0]["aggregate_exposure"] > 66.0
    assert _status(runner, "hs-p4") == "touched"


# ------------------------------------------- (c) same setup never twice

def test_same_setup_never_opens_twice(tmp_path):
    runner = _runner(tmp_path, equity=10000.0)
    a = _candidate(runner, setup_id="hs-a", parent_id="hp-a")
    assert _promote(runner, a) is True
    refire = dict(a)
    refire["claim"] = {"claimant": "process-A", "fencing_token": 999}
    assert runner._promote_hot_scalp_candidate(refire, _tick(), {}) \
        is False
    assert len(runner.executor.placed) == 1
    refusals = _refusals(runner, "hs-a")
    assert refusals[-1]["reason_code"] == \
        "promotion.setup_already_open"


# --------------------------------------- (d) same-thesis revalidation

def test_same_thesis_stack_refused_after_thesis_invalidation(tmp_path):
    runner = _runner(tmp_path, equity=10000.0)
    a = _candidate(runner, setup_id="hs-x1", parent_id="hp-x",
                   zone=(100.0, 100.6))
    b = _candidate(runner, setup_id="hs-x2", parent_id="hp-x",
                   zone=(99.8, 100.4))
    assert _promote(runner, a) is True
    # The structure breaks AFTER b's trigger: a later scan closes
    # through b's invalidation (terminate covers claimed rows).
    assert runner._hot_scalp.store.terminate(
        "hs-x2", "invalidated",
        "closed M1 candle breached structural invalidation", NOW)
    assert _promote(runner, b) is False
    assert len(runner.executor.placed) == 1
    refusals = _refusals(runner, "hs-x2")
    assert [r["reason_code"] for r in refusals] == \
        ["promotion.thesis_invalidated"]
    # The dead setup stays dead — the refusal must not resurrect
    # it to touched.
    assert _status(runner, "hs-x2") == "invalidated"


def test_same_thesis_stack_refused_when_zone_consumed(tmp_path):
    runner = _runner(tmp_path, equity=10000.0)
    a = _candidate(runner, setup_id="hs-x1", parent_id="hp-x",
                   zone=(100.0, 100.6))
    b = _candidate(runner, setup_id="hs-x2", parent_id="hp-x",
                   stop=91.30, zone=(100.0, 100.6))
    assert _promote(runner, a) is True
    assert _promote(runner, b) is False
    assert len(runner.executor.placed) == 1
    refusals = _refusals(runner, "hs-x2")
    assert [r["reason_code"] for r in refusals] == \
        ["promotion.zone_already_consumed"]
    assert _status(runner, "hs-x2") == "touched"


def test_each_stack_gets_its_own_fresh_judgment_evaluation(tmp_path):
    runner = _runner(tmp_path, equity=10000.0)
    a = _candidate(runner, setup_id="hs-x1", parent_id="hp-x",
                   stop=90.30, zone=(100.0, 100.6))
    b = _candidate(runner, setup_id="hs-x2", parent_id="hp-x",
                   stop=91.30, zone=(99.8, 100.4))
    assert _promote(runner, a) is True
    assert _promote(runner, b) is True
    evals = {}
    for r in runner.journal.records():
        if r.get("type") == "evaluation" and \
                r.get("playbook") == "hot_scalp":
            evals[(r.get("candidate") or {}).get("setup_id")] = r
    # The judgment stage re-ran per setup — the stack never rode
    # the open position's evaluation.
    assert set(evals) == {"hs-x1", "hs-x2"}


# --------------------------------------------- (e) global position cap

def test_global_max_positions_still_caps_the_sleeve(tmp_path):
    runner = _runner(tmp_path, equity=10000.0)
    for i, ticket in enumerate((101, 102, 103)):
        req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.5,
                           stop_loss=1990.0, entry_ref=2000.0,
                           targets=[2030.0], atr=10.0, bar_index=5)
        runner.tick_manager.open(ticket=ticket, request=req,
                                 fill_price=2000.0, entry_bar=6,
                                 broker_sl=1987.5)
    cand = _candidate(runner, setup_id="hs-a", parent_id="hp-a")
    assert _promote(runner, cand) is False
    assert runner.executor.placed == []
    refused = _chain_records(runner, stage="execution",
                             chain_id="hs-a")
    assert refused and refused[-1]["reason_code"] == \
        "execution.order_path_refused"
    assert "max positions reached (3/3)" in _notes(runner)
    assert _status(runner, "hs-a") == "touched"


# ------------------------------------------------- (f) close isolation

def test_close_of_one_position_leaves_the_other_byte_identical(tmp_path):
    runner = _runner(tmp_path, equity=10000.0)
    a = _candidate(runner, setup_id="hs-a", parent_id="hp-a",
                   entry=100.30, stop=90.30, target=115.0,
                   zone=(95.0, 106.0))
    b = _candidate(runner, setup_id="hs-b", parent_id="hp-b",
                   entry=100.30, stop=92.30, target=112.0,
                   zone=(99.9, 100.5))
    assert _promote(runner, a) is True
    assert _promote(runner, b) is True
    sleeve = {s["setup_id"]: s
              for s in runner._hot_scalp_sleeve_positions()}
    ta, tb = sleeve["hs-a"]["ticket"], sleeve["hs-b"]["ticket"]

    def snap(ticket):
        st = runner.tick_manager._states[ticket]
        return (dataclasses.asdict(st.pos),
                copy.deepcopy(st.events),
                st.broker_sl,
                runner._pools[ticket].state())

    before = snap(tb)
    # A's lifecycle runs to its stop: tick through the stop,
    # apply the stop-hit, then the runner books the close.
    acts = runner.tick_manager.on_tick(ta, 89.0)
    hits = [x for x in acts if x.kind == "stop_hit"]
    assert hits, "A must stop out at 89.0"
    for act in hits:
        runner.tick_manager.apply(ta, act)
    runner._record_close(ta, 89.0, "stop",
                         {"reconciled_closed": 0})
    assert snap(tb) == before
    # The parent record for hp-a tracks the close (child gone);
    # hp-b's sleeve thread gains no record from A's close.
    closed = [r for r in _chain_records(runner, stage="sleeve",
                                        chain_id="hp-a")
              if r.get("execution_status") == "position_closed"]
    assert closed and closed[-1]["closed_ticket"] == ta
    assert closed[-1]["child_tickets"] == []
    assert _chain_records(runner, stage="sleeve", chain_id="hp-b")[
        -1]["execution_status"] == "position_opened"
