"""Completion-02 (owner program 2026-11-05): connect the whole
playbook ecosystem to the shared judgment + execution
architecture — above all the two routes the old promotion gate
used to hold back (hot_scalp, the A6 resting limit /
arrival_zone).  Rewritten 2026-11-27: the promotion gate is
DELETED by owner order (2026-11-25/27 — everything live
directly, no shadow, no gating, no promotion).  What is pinned
here now:

- PER-ROUTE authority: every order route is live by
  declaration in the registry (LIVE_ORDER_ROUTES); no config
  step can promote or demote a route anymore — the
  judgment_route_modes field is inert.
- hot_scalp candidates terminate in funnel evaluation records
  + journaled judgment evaluations (type "judgment"): an
  abstention is RECORDED and REFUSES the attempt under
  judgment.abstain, exactly like a governed route's.
- arrival_zone placements get the same treatment at their
  decision point (candidate honestly constructed from the
  placement's own score/floor/geometry).
- the manifest's per-route authority table (states, governing
  now, accruing judgment-history counts).
- the pending_trigger verdict: a registered setup's
  order_result "pending_trigger" is a WAITING state with a
  journaled binder and journaled terminal exits (the
  2026-11-05 06:00 fvg_limit shape: opposite-direction fires
  decline by record, the instance stays armed, and validity
  lapse terminates it through upkeep — never a silent strand).
"""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_runner import (
    LIVE_ORDER_ROUTES,
    MT5Runner,
    MT5RunnerConfig,
)
from goldenswing.execution.risk import gold_spec
from goldenswing.live.hierarchy import TimeframeStack
from goldenswing.live.hot_scalp import (
    HotScalpConfig,
    HotScalpResearchEngine,
)
from goldenswing.live.setup_store import SetupStore
from goldenswing.live.zones import StructuralZone

from test_confluence_rework import _point3_frames
from test_confluence_rework import _runner as _az_runner
from test_p3_trader import _flat_h1
from test_step14_setup_lifecycle import _scan_runner, _sweep

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


def _up_m5_frame():
    """M5 frame with a real upward reaction through 100.30:
    closes rise 99.6 -> 101.6 on tight ranges, so the judgment
    measures strong displacement up, acceptance above the entry
    and follow-through held for a long candidate."""
    idx = pd.date_range("2026-10-27T00:00:00Z", periods=10,
                        freq="5min")
    close = np.linspace(99.6, 101.6, 10)
    return pd.DataFrame(
        {"open": close - 0.05, "high": close + 0.1,
         "low": close - 0.1, "close": close,
         "volume": np.full(10, 100.0)}, index=idx)


class _MarketExec:
    """Fake broker: market fills synchronously at entry_ref."""

    def __init__(self, positions=None, fail_positions=False):
        self._positions = list(positions or [])
        self.fail_positions = fail_positions
        self.placed = []
        self._next_ticket = 777

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        if self.fail_positions:
            raise RuntimeError("broker unreachable")
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


def _hs_runner(tmp_path, *, live=True, positions=None,
               fail_positions=False, paused=False):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False, hot_scalp_live_enabled=live,
        paused=paused)
    feed = SimpleNamespace(
        account_state=lambda: {"equity": 10000.0, "balance": 10000.0,
                               "free_margin": 10000.0},
        order_margin=lambda *, volume, price, direction:
            float(volume) * 100.0 * float(price) / 500.0)
    runner = MT5Runner(cfg, feed=feed,
                       executor=_MarketExec(positions, fail_positions))
    # The market-closed guard is covered by step-3 tests; pin it
    # open here so these tests do not depend on the weekday.
    runner._market_open = lambda: (True, "test")
    runner._stack = TimeframeStack.from_frames({
        "H1": _frame(60, "2026-10-27T00:00:00Z", 60),
        # The M5 frame is the entry clock the judgment reads:
        # a real upward leg through the entry (see _up_m5_frame).
        "M5": _up_m5_frame(),
        "M15": _frame(15, "2026-10-26T12:15:00Z", 50)})
    runner._hot_scalp_aux_frames = {
        "M1": _frame(1, "2026-10-27T00:00:00Z", 50)}
    store = SetupStore(tmp_path / "hot-scalp.sqlite")
    runner._hot_scalp = HotScalpResearchEngine(
        store, HotScalpConfig(min_m1_bars=100, min_m5_bars=100,
                              min_m15_bars=100),
        instance_id="process-A")
    return runner


def _hs_cand(runner, *, claimant="process-A", entry=100.30, stop=99.0,
             target=102.0, direction=1, spread=0.02, score=60.0):
    setup = {
        "setup_id": "hs-test-parent-1", "symbol": "XAUUSD",
        "strategy": "breakout_retest", "direction": direction,
        "formation_ts": "2026-10-27T00:40:00+00:00",
        "armed_ts": "2026-10-27T00:50:00+00:00",
        "expires_at": "2026-10-27T01:25:00+00:00",
        "level": 100.2, "zone_low": 100.0, "zone_high": 100.25,
        "stop": stop, "structure_ts": "2026-10-27T00:40:00+00:00",
        "context_direction": direction, "atr_m5": 0.5,
        "description": "completion-02 test setup",
    }
    store = runner._hot_scalp.store
    store.arm(setup)
    store.mark_touched(setup["setup_id"], NOW)
    claim = store.claim_trigger(setup["setup_id"], claimant, NOW)
    assert claim is not None, "test setup claim must succeed"
    return {
        "event": "candidate", "setup_id": setup["setup_id"],
        "parent_setup_id": setup["setup_id"],
        "strategy": "breakout_retest", "direction": direction,
        "entry_reference": entry, "structural_stop": stop,
        "target_reference": target, "spread": spread, "score": score,
        "trigger_ts": NOW, "claim": claim,
    }


def _hs_tick(*, mid=100.30, bid=100.29, ask=100.31):
    return {"time": NOW, "mid": mid, "bid": bid, "ask": ask}


def _hs_status(runner):
    con = sqlite3.connect(str(runner._hot_scalp.store.path))
    try:
        row = con.execute(
            "SELECT status FROM hot_scalp_setups WHERE setup_id=?",
            ("hs-test-parent-1",)).fetchone()
    finally:
        con.close()
    assert row is not None
    return row[0]


def _evals(runner, playbook):
    return [r for r in runner.journal.records()
            if r.get("type") == "evaluation"
            and r.get("playbook") == playbook]


def _judged(runner, playbook):
    return [r for r in runner.journal.records()
            if r.get("type") == "judgment"
            and r.get("playbook") == playbook]


def _chain(runner, stage):
    return [r for r in runner.journal.records()
            if r.get("type") == "candidate_chain"
            and r.get("stage") == stage]


# ------------------------------------------------ route states

def test_route_table_is_live_by_declaration(tmp_path):
    """Owner order 2026-11-25/27: every order route is live by
    declaration in the registry — no promotion, no shadow, no
    config step.  The runner's route table is exactly the
    registry's live set, every row live and governing now."""
    runner = _hs_runner(tmp_path)
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    assert set(table) == set(LIVE_ORDER_ROUTES)
    for route, row in table.items():
        assert row["authority"] == "live", route
        assert row["source"] == "live", route
        assert row["governing_now"] is True, route
    assert table["hot_scalp_promotion"]["authority"] == "live"
    assert table["resting_limit"]["authority"] == "live"


def test_route_config_step_is_dead(tmp_path):
    """judgment_route_modes is an inert field now: setting it —
    even to nonsense — changes no route's authority.  There is
    no config step that promotes or demotes a route anymore."""
    runner = _hs_runner(tmp_path)
    before = runner._judgment_route_table()
    runner.cfg.judgment_route_modes = {
        "hot_scalp_promotion": "governing",
        "qm_m5_native": "shadow_declared",
        "resting_limit": " SHADOW_DECLARED "}
    after = runner._judgment_route_table()
    assert before == after
    assert all(row["authority"] == "live" for row in after)


# ------------------------------------------------ hot_scalp

def test_hot_scalp_placement_journals_funnel_and_core(tmp_path):
    # Owner order 2026-11-25: the route governs by declaration,
    # so at default the candidate submits through the seam like
    # every live route — no verdict needed, no config step.
    runner = _hs_runner(tmp_path, live=True)
    cand = _hs_cand(runner)
    assert runner._place_hot_scalp_candidate(
        cand, _hs_tick(), {}) is True
    # The chain's own outcome is exactly as before the connection.
    assert len(runner.executor.placed) == 1
    assert _hs_status(runner) == "order_reconciled"
    # ONE funnel record, funnel-shaped, terminal = filled.
    evs = _evals(runner, "hot_scalp")
    assert len(evs) == 1
    rec = evs[0]
    assert rec["outcome"] == "candidate_emitted"
    assert rec["decision"]["result"] == "accepted"
    assert rec["decision"]["order_result"] == "filled"
    cand_block = rec["candidate"]
    assert cand_block["setup_id"] == "hs-test-parent-1"
    assert cand_block["direction"] == 1
    assert cand_block["entry"] == pytest.approx(100.30)
    assert cand_block["stop"] == pytest.approx(99.0)
    assert cand_block["targets"] == [102.0]
    # Phase-2 naming honesty: sleeve base score, no asymmetry.
    assert cand_block["asymmetry_score"] is None
    assert cand_block["sleeve_base_score"] == pytest.approx(60.0)
    # ONE judgment evaluation, route-stamped with the authority
    # the route actually rode in under (live).
    judged = _judged(runner, "hot_scalp")
    assert len(judged) == 1
    j = judged[0]["judgment"]
    assert j["decision"] == "take"
    assert j["mode"] == "active"
    assert j["authority"] == "live"
    assert j["authority_route"] == "hot_scalp_promotion"
    assert judged[0]["gate_order_result"] == "filled"
    # The chain's judgment stage carries the core summary too.
    chain_judged = _chain(runner, "judgment")
    assert len(chain_judged) == 1
    assert chain_judged[0]["core"]["decision"] == "take"
    assert chain_judged[0]["core"]["governing"] is True
    # Funnel counters reconcile the sleeve like any playbook.
    totals = runner.funnel_totals()
    assert totals["candidates"] >= 1 and totals["ordered"] >= 1


def test_hot_scalp_abstain_is_journaled_and_refuses(tmp_path):
    """Owner order 2026-11-25: the route governs by
    declaration, so the core abstains and the attempt is
    REFUSED under judgment.abstain — the claim resolves back
    to touched (a transient refusal — the lifecycle is
    untouched beyond this attempt).  The abstention is forced
    by structure, not a threshold: a SHORT candidate fights
    the measured upward reaction on the M5 entry clock, so
    the verdict abstains."""
    runner = _hs_runner(tmp_path, live=True)
    cand = _hs_cand(runner, direction=-1, entry=100.30,
                     stop=101.60, target=99.0)
    assert runner._place_hot_scalp_candidate(
        cand, _hs_tick(), {}) is False
    assert runner.executor.placed == []
    assert _hs_status(runner) == "touched"
    judged = _judged(runner, "hot_scalp")
    assert len(judged) == 1
    j = judged[0]["judgment"]
    assert j["decision"] == "abstain"
    assert j["authority"] == "live"
    assert _chain(runner, "judgment")[0]["core"]["decision"] \
        == "abstain"
    evs = _evals(runner, "hot_scalp")
    assert len(evs) == 1
    assert evs[0]["decision"]["reason_code"] \
        == "judgment.abstain"
    assert evs[0]["decision"]["order_result"] \
        == "not_attempted"


def test_hot_scalp_route_gate_refusal_also_journals(tmp_path):
    """A candidate refused by the route's OWN gates (here:
    paused) still terminates in a funnel record + a tap-journaled
    judgment evaluation — no refusal is invisible."""
    runner = _hs_runner(tmp_path, live=True, paused=True)
    cand = _hs_cand(runner)
    assert runner._place_hot_scalp_candidate(
        cand, _hs_tick(), {}) is False
    assert runner.executor.placed == []
    evs = _evals(runner, "hot_scalp")
    assert len(evs) == 1
    assert evs[0]["decision"]["result"] == "refused"
    assert evs[0]["decision"]["reason_code"] \
        == "hot_scalp.runner_paused"
    judged = _judged(runner, "hot_scalp")
    assert len(judged) == 1
    assert judged[0]["judgment"]["authority"] == "live"


def test_hot_scalp_shadow_mode_still_journals(tmp_path):
    """The global kill switch set to shadow: the placement path
    is untouched (the chain decides) and the verdict journals
    via the tap in shadow mode, route-stamped.  Shadow mode
    means orders proceed unjudged — never a gate."""
    runner = _hs_runner(tmp_path, live=True)
    runner.cfg.judgment_mode = "shadow"
    cand = _hs_cand(runner)
    assert runner._place_hot_scalp_candidate(
        cand, _hs_tick(), {}) is True
    assert len(runner.executor.placed) == 1
    judged = _judged(runner, "hot_scalp")
    assert len(judged) == 1
    j = judged[0]["judgment"]
    assert j["mode"] == "shadow"
    assert j["authority"] == "live"


# ------------------------------------------------ arrival_zone

def _az_drive(tmp_path, *, structure="up"):
    runner = _az_runner(tmp_path, resting_limit_enabled=True)
    frames = _point3_frames()
    if structure == "up":
        # The H1 entry clock must show the upward reaction the
        # long placement claims: the last closes rise through
        # the zone top (4122) — displacement up, acceptance
        # above the entry, follow-through held.  The point-3
        # fixture's own closes linger inside the band (a
        # down-leg's absorption); left alone, the long reads as
        # a fight and abstains — that is the abstain case below.
        h1 = frames["H1"].copy()
        h1.iloc[-4:, h1.columns.get_loc("close")] = [
            4118.0, 4124.0, 4128.0, 4132.0]
        h1.iloc[-4:, h1.columns.get_loc("high")] = [
            4119.2, 4125.2, 4129.2, 4133.2]
        h1.iloc[-4:, h1.columns.get_loc("low")] = [
            4116.8, 4122.8, 4126.8, 4130.8]
        frames["H1"] = h1
    runner._stack = SimpleNamespace(frames=frames)
    zone = StructuralZone(
        tf="H4", side="support", level=4110.0, top=4122.0,
        bottom=4098.0, score=85.0, tier="MAJOR", status="active",
        touches=0, atr=12.0, displacement_atr=2.5,
        departure_atr=2.5, swing_ts="2026-10-18T00:00:00+00:00")
    runner._structural_zones = [zone]
    runner._live_setup = {
        "direction": 1, "regime": "trend", "atr_h1": 12.0,
        "buf": 3.0, "major_zone": zone.to_dict(),
        "zone": {"top": 4122.0, "bottom": 4098.0,
                 "level": 4110.0},
        "bias_d1": 0, "bias_h4": 1,
    }
    tick = {"mid": 4122.0 + 3.6, "spread": 0.3,
            "time": pd.Timestamp("2026-10-21 13:00", tz="UTC")}
    runner._maybe_place_resting_limit(tick, {})
    return runner


def test_arrival_zone_placement_journals_funnel_and_core(
        tmp_path):
    # Owner order 2026-11-25: the route governs by declaration,
    # so at default the limit rests — no verdict, no config
    # step needed.
    runner = _az_drive(tmp_path, structure="up")
    # The placement itself is exactly as before the connection.
    assert len(runner._pending_orders) == 1
    state = next(iter(runner._pending_orders.values()))
    assert state.playbook == "arrival_zone"
    evs = _evals(runner, "arrival_zone")
    assert len(evs) == 1
    rec = evs[0]
    assert rec["decision"]["result"] == "accepted"
    assert rec["decision"]["order_result"] == "placed_pending"
    cand_block = rec["candidate"]
    assert cand_block["entry"] == pytest.approx(4122.0)
    # Structural invalidation: distal (4098) - buffer (3.0).
    assert cand_block["stop"] == pytest.approx(4095.0)
    assert cand_block["asymmetry_score"] is not None
    judged = _judged(runner, "arrival_zone")
    assert len(judged) == 1
    j = judged[0]["judgment"]
    assert j["authority"] == "live"
    assert j["authority_route"] == "resting_limit"
    assert j["mode"] == "active"


def test_arrival_zone_abstain_is_journaled_and_refuses(
        tmp_path):
    """Owner order 2026-11-25: the route governs by
    declaration, so the abstention REFUSES the placement
    under judgment.abstain — the approach's one funnel record
    terminates refused, and nothing rests.  The abstention is
    forced by structure: the point-3 H1 closes linger inside
    the band after a down-leg (absorption, no upward
    displacement), so the long fights the measured reaction."""
    runner = _az_drive(tmp_path, structure="down")
    assert runner._pending_orders == {}
    j = _judged(runner, "arrival_zone")[0]["judgment"]
    assert j["decision"] == "abstain"
    assert j["authority"] == "live"
    evs = _evals(runner, "arrival_zone")
    assert len(evs) == 1
    assert evs[0]["decision"]["reason_code"] \
        == "judgment.abstain"


# ------------------------------------------------ manifest

def test_manifest_route_authority_table(tmp_path):
    runner = _hs_runner(tmp_path, live=True)
    cand = _hs_cand(runner)
    # Owner order 2026-11-25: at the declared posture the
    # attempt PLACES — the route governs by declaration, and
    # the manifest census counts the run's one hot_scalp
    # judgment record under a live route.
    assert runner._place_hot_scalp_candidate(
        cand, _hs_tick(), {}) is True
    assert len(runner.executor.placed) == 1
    runner._journal_manifest()
    core = [r for r in runner.journal.records()
            if r.get("type") == "manifest"][-1][
                "doctrine"]["judgment_core"]
    # The authority routes are the registry's live set.
    assert core["authority_routes"] == sorted(LIVE_ORDER_ROUTES)
    # Nothing is not-authoritative anymore: every declared
    # route governs.
    assert core["not_authoritative"] == []
    table = {row["route"]: row
             for row in core["route_authority"]}
    # An exact route census: every live route, including the
    # five family routes (range, vol, trend, auction,
    # reversion, session ride their own route names).
    assert set(table) == set(LIVE_ORDER_ROUTES)
    assert table["hot_scalp_promotion"]["authority"] == "live"
    assert table["hot_scalp_promotion"]["governing_now"] is True
    assert table["resting_limit"]["authority"] == "live"
    assert table["resting_limit"]["governing_now"] is True
    # The run's accruing judgment history, counted.
    hs = table["hot_scalp_promotion"]
    assert hs["judgment_records_this_run"] == 1
    assert table["resting_limit"][
        "judgment_records_this_run"] == 0
    for route in ("range_h4_h1", "volatility_expansion",
                  "trend_momentum", "auction_value",
                  "mean_reversion", "session_range"):
        assert table[route]["authority"] == "live"
        assert table[route]["governing_now"] is True
        assert table[route][
            "judgment_records_this_run"] == 0
    for route in LIVE_ORDER_ROUTES:
        assert table[route]["authority"] == "live"
        assert table[route]["governing_now"] is True


# ------------------------------------------------ scan candidates

def test_sweep_candidates_are_core_evaluated(tmp_path):
    """Census pin for the range route's connection state: a
    scan candidate journals its funnel record with the live
    disposition AND receives a tap-journaled judgment
    evaluation — evaluated and registered live, never shadowed,
    never ordered from the scan itself."""
    runner, h1 = _scan_runner(tmp_path)
    _sweep(runner, h1, direction=1)
    scans = [r for r in _evals(runner, "fvg_midline")
             if r.get("outcome") == "candidate_emitted"]
    assert scans
    assert scans[-1]["decision"]["result"] == "registered"
    assert scans[-1]["decision"]["order_result"] \
        == "pending_trigger"
    jud = _judged(runner, "fvg_midline")
    assert jud
    assert jud[-1]["gate_result"] == "registered"


# ------------------------------------------------ pending_trigger

def test_pending_trigger_is_a_waiting_state_not_a_strand(
        tmp_path):
    """The 2026-11-05 06:00 fvg_limit shape, reproduced: a
    scan-registered LONG setup reads order_result
    "pending_trigger"; opposite-direction fires journal binder
    records declining it (direction_mismatch) and it stays
    armed; when its validity lapses, upkeep terminates it with
    a journaled expired transition.  Nothing is stranded: every
    state change is a record."""
    runner, h1 = _scan_runner(tmp_path)
    _sweep(runner, h1, direction=1)
    scans = [r for r in _evals(runner, "fvg_midline")
             if r.get("outcome") == "candidate_emitted"]
    assert scans and scans[-1]["decision"]["order_result"] \
        == "pending_trigger"
    store = runner._setup_store
    armed = store.active()
    assert len(armed) == 1
    setup = armed[0]
    assert setup.stage == "setup_armed"
    # Two opposite-direction fires: the binder journals its
    # selection both times and declines by record.
    for _ in range(2):
        runner._emit_trigger_binding(
            direction=-1, verdict={"entry": 2000.0}, h1=h1,
            t=len(h1) - 1, armed_all=store.active(),
            candidates={}, rules={})
    bindings = [r for r in runner.journal.records()
                if r.get("kind") == "trigger_binding"]
    assert len(bindings) == 2
    for b in bindings:
        assert b["reason_code"] \
            == "path.binding_direction_mismatch"
        inst = b["evidence"]["binding"]["instances"]
        assert inst[0]["disposition"] == "other_direction"
    # Still armed — waiting is the state, not a disappearance.
    assert store.get(setup.setup_id).stage == "setup_armed"
    # Validity lapse: a later frame's close runs upkeep, which
    # terminates the instance with a journaled transition.
    longer = _flat_h1(60, 2000.0)
    runner._upkeep_setups(longer)
    assert store.get(setup.setup_id).stage == "expired"
    events = [r for r in runner.journal.records()
              if r.get("kind") == "setup_event"
              and (r.get("evidence") or {}).get(
                  "lifecycle_stage") == "expired"]
    assert events
