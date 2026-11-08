"""MarketSituation Pass 6 — horizon management (SPEC §7, §10
Pass 6).

Pins:
- ManagementProfile carries the four §7.1 horizon fields,
  derived from each branch's own established progress pair.
- evaluate_management (STATE 6) gains the structural verdicts
  when given the profile + the situation's TFState — and is
  unchanged without them.
- Policy resolution by horizon (standard_scalp_v1 /
  standard_swing_v1 from established rule sets; intraday and
  qm_m5 byte-identical), with the registry's parameter claims
  pinned against the live TickConfig.
- Every scenario declares its management plan at proposal
  (management_block_for), the plan freezes into the book's
  management context at fill, and nothing outside rewrites
  the frozen copy.
- The book marks a filled scenario when its parent thesis is
  invalidated; the manager enacts the plan's thesis exit and
  journals it as a thesis exit, distinct from stop / target /
  time exits, with the plan stamps + plan step named.
- Management verdicts are journaled per position on its own
  management clock: a scalp position's validity expires while
  a swing position rides, same pass, same frames.
- Deviations from the plan (panic close, news pre-print
  securing) journal as safeguard_override records naming the
  safeguard.
- The per-horizon census separates horizons measurably and
  names a horizon indistinguishable from standard.
- qm_m5 tick management is byte-identical: the placement
  bracket only, partials aside — no stop ever moves.
- Hot Scalp's plan is the sleeve's (standard_scalp_v1, earned
  lock, sleeve target hierarchy), not the generic standard.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_runner import (
    MT5Runner, MT5RunnerConfig, management_block_for)
from goldenswing.execution.playbook_registry import (
    MANAGEMENT_POLICIES, POLICY_PARAMS, management_horizon_for,
    management_policy_for, policy_params_for)
from goldenswing.execution.scenario_book import ScenarioBook
from goldenswing.execution.setup_lifecycle import (
    STAGE_FILLED, TrackedSetup)
from goldenswing.execution.tick_manager import TickConfig, TickManager
from goldenswing.execution.executor import OrderRequest
from goldenswing.live.hierarchy import TimeframeStack
from goldenswing.live.management import (
    evaluate_management, plan_management)
from goldenswing.live.management_profiles import (
    build_management_plan, elapsed_h1_bars, management_census,
    profile_for, progress_due, structural_verdicts)


# ------------------------------------------------------------ profiles
@pytest.mark.parametrize("style,clock,trig,floor", [
    ("trend", "H1", 8, 0.5), ("trend", "D1", 8, 0.5),
    ("scalp", "M5", 12, 0.3), ("v_reversal", "H1", 6, 0.4),
    ("range", "H1", 6, 0.5),
])
def test_profile_carries_the_horizon_fields(style, clock, trig,
                                            floor):
    p = profile_for(style=style, clock_tf=clock) \
        if style != "range" else profile_for(
            style="intraday", philosophy="range", clock_tf=clock)
    # The stall definition IS the branch's progress pair, in
    # management-clock bars; the validity window is two full
    # progress windows; the exit clock is the profile's clock.
    assert p.stall_definition == {"by_bar": trig, "below_r": floor}
    assert p.validity_window_bars == 2 * trig
    assert p.exit_clock == clock
    assert p.progress_floor_r == floor
    j = p.journal()
    assert j["validity_window_bars"] == 2 * trig
    assert j["stall"] == {"by_bar": trig, "below_r": floor}
    assert j["exit_clock"] == clock


def test_stall_verdict_agrees_with_progress_due():
    # The management-clock stall form and the standing H1
    # progress form are the same rule (XD-4 conversion).
    for style, clock in (("trend", "H1"), ("trend", "D1"),
                         ("trend", "H4"), ("scalp", "M5")):
        p = profile_for(style=style, clock_tf=clock)
        for elapsed_mgmt in (0, 4, 8, 12, 16, 24, 40):
            for best_r in (0.1, 0.9):
                v = structural_verdicts(
                    p, direction=1, elapsed_mgmt_bars=elapsed_mgmt,
                    best_r=best_r)
                h1_form = progress_due(
                    p,
                    elapsed_h1_bars=elapsed_h1_bars(
                        clock_tf=clock, entry_mgmt_bar=0,
                        current_mgmt_bar=elapsed_mgmt),
                    best_r=best_r)
                assert v["stall"] == h1_form, \
                    (style, clock, elapsed_mgmt, best_r)


def test_structural_verdicts_from_the_situation_state():
    p = profile_for(style="trend", clock_tf="H1")
    choch_against = {"last_structure_event": {
        "kind": "choch", "direction": -1,
        "ts": "2026-11-08T03:00:00", "level": 3990.0,
        "against_bias": True}}
    v = structural_verdicts(p, direction=1, elapsed_mgmt_bars=3,
                            best_r=0.9, tf_state=choch_against)
    assert v["structure_break"] is True
    assert v["preserve"] is False
    bos_with = {"last_structure_event": {
        "kind": "bos", "direction": 1,
        "ts": "2026-11-08T03:00:00", "level": 4010.0,
        "against_bias": False}}
    v2 = structural_verdicts(p, direction=1, elapsed_mgmt_bars=3,
                             best_r=0.9, tf_state=bos_with)
    assert v2["structure_break"] is False
    assert v2["preserve"] is True
    # Validity: the trend/H1 window is 16 bars.
    v3 = structural_verdicts(p, direction=1, elapsed_mgmt_bars=16,
                             best_r=0.9)
    assert v3["validity_expired"] is True
    v4 = structural_verdicts(p, direction=1, elapsed_mgmt_bars=15,
                             best_r=0.9)
    assert v4["validity_expired"] is False


# ------------------------------------------------- STATE 6 extension
def _plan_dict(**kw):
    import dataclasses
    plan = plan_management(direction=1, entry_price=2000.0,
                           stop=1990.0, **kw)
    return dataclasses.asdict(plan)


def test_evaluate_management_legacy_shape_unchanged():
    plan = _plan_dict()
    action = evaluate_management(
        plan=plan, direction=1, entry_price=2000.0, stop=1990.0,
        current_price=2010.0, elapsed_h1=1, atr=10.0)
    # +1R: breakeven due, TP1 partial — the standing verdicts.
    assert action.breakeven is True
    assert action.partials == ["tp1"]
    # No profile supplied: no structural verdicts, ever.
    assert action.stall is False
    assert action.validity_expired is False
    assert action.structure_break is False
    assert action.preserve is False
    assert action.verdict_facts == []


def test_evaluate_management_with_profile_and_tf_state():
    plan = _plan_dict()
    profile = profile_for(style="trend", clock_tf="H1")
    tf_state = {"last_structure_event": {
        "kind": "choch", "direction": -1,
        "ts": "2026-11-08T03:00:00", "level": 1995.0,
        "against_bias": True}}
    action = evaluate_management(
        plan=plan, direction=1, entry_price=2000.0, stop=1990.0,
        current_price=2001.0, elapsed_h1=20, atr=10.0,
        profile=profile, tf_state=tf_state, best_r=0.1,
        elapsed_mgmt_bars=20)
    assert action.stall is True           # 20 >= 8, best 0.1 < 0.5
    assert action.validity_expired is True  # 20 >= 16
    assert action.structure_break is True
    assert action.preserve is False
    assert action.verdict_facts


# ------------------------------------------------------------ policies
def test_policy_resolution_by_horizon():
    assert management_policy_for("qm_m5") == ("fixed_bracket_v1", 1)
    assert management_policy_for("qm_d1") == ("standard_swing_v1", 1)
    assert management_policy_for("hot_scalp") == \
        ("standard_scalp_v1", 1)
    assert management_policy_for("sweep_reversal") == \
        ("standard_v1", 1)
    assert management_policy_for("range_qm") == ("standard_v1", 1)
    assert set(MANAGEMENT_POLICIES) == {
        "fixed_bracket_v1", "standard_v1", "standard_scalp_v1",
        "standard_swing_v1"}
    for name in ("qm_m5", "qm_d1", "hot_scalp", "sweep_reversal"):
        assert management_horizon_for(name) in (
            "scalp", "intraday", "swing")


def test_policy_params_match_the_live_tick_config():
    # The registry's claims about established rule sets are
    # pinned against the TickConfig that actually enforces
    # them — drift between the two fails here, not live.
    live = TickConfig().policy_parameters()
    assert live["fixed_bracket_playbooks"] == ["qm_m5"]
    assert POLICY_PARAMS["fixed_bracket_v1"]["profit_lock_mode"] \
        == "none"
    # The scalp policy's earned lock IS the live earned list.
    assert live["profit_lock_earned_playbooks"] == ["hot_scalp"]
    assert POLICY_PARAMS["standard_scalp_v1"]["profit_lock_mode"] \
        == "earned_acceptance"
    assert "hot_scalp" in \
        live["profit_lock_progress_excluded_playbooks"]
    # Inherited parameters are the live defaults, not copies.
    assert live["partial_tiers"] == [[1.0, 0.5], [2.0, 0.25],
                                     [3.0, 0.25]]
    assert live["tier_min_volume"] == 0.03
    assert policy_params_for("standard_swing_v1")[
        "partial_style_override"] == "value"
    assert policy_params_for("nope") == {}


# ------------------------------------------------------------ the plan
def _sig(**kw):
    base = dict(direction=1, entry_ref=2005.0, invalidation=1990.0,
                targets=[2040.0, 2060.0], atr=10.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_plan_declared_for_intraday():
    plan = management_block_for("sweep_reversal", _sig())[
        "management_plan"]
    assert plan["policy_id"] == "standard_v1"
    assert plan["horizon"] == "intraday"
    assert plan["management_tf"] == "H1"
    assert plan["time_stop"]["validity_window_bars"] == 16
    assert plan["time_stop"]["clock"] == "H1"
    assert plan["time_stop"]["progress"] == {"by_bar": 8,
                                             "below_r": 0.5}
    assert plan["invalidation"]["price"] == 1990.0
    assert plan["invalidation"]["thesis_exit"] == \
        "parent_thesis_invalidation"
    assert any("close beyond invalidation" in t
               for t in plan["invalidation"]["terms"])
    sched = plan["partial_schedule"]
    assert sched["source"] == "tick_tiers"
    assert sched["tiers"] == [[1.0, 0.5], [2.0, 0.25], [3.0, 0.25]]
    assert sched["tier_min_volume"] == 0.03
    assert sched["style"] == "ladder"
    assert sched["targets"] == [2040.0, 2060.0]
    assert plan["trail"] == {"method": "structure"}
    assert plan["profit_lock"] == {"mode": "standard"}


def test_plan_for_qm_m5_is_the_bracket_only():
    plan = management_block_for("qm_m5", _sig())["management_plan"]
    assert plan["policy_id"] == "fixed_bracket_v1"
    assert plan["profile"] is None
    assert plan["time_stop"] is None
    assert plan["trail"] == {"method": "none"}
    assert plan["profit_lock"] == {"mode": "none"}
    assert plan["invalidation"]["thesis_exit"] is None
    assert plan["partial_schedule"]["source"] == \
        "placement_bracket"


def test_plan_for_swing_reads_on_the_d1_clock():
    plan = management_block_for("qm_d1", _sig())["management_plan"]
    assert plan["policy_id"] == "standard_swing_v1"
    assert plan["management_tf"] == "D1"
    # Progress windows in D1 bars: validity = 16 D1 bars.
    assert plan["time_stop"]["validity_window_bars"] == 16
    assert plan["time_stop"]["clock"] == "D1"
    assert plan["time_stop"]["progress"] == {"by_bar": 8,
                                             "below_r": 0.5}
    # Partial style "value" (bank at value/zone), structure
    # trail — the swing policy's declared differences.
    assert plan["partial_schedule"]["style"] == "value"
    assert plan["trail"] == {"method": "structure"}


def test_hot_scalp_plan_is_the_sleeves_not_the_generic():
    sig = _sig(planned_targets=[2015.0, 2030.0])
    plan = management_block_for("hot_scalp", sig)["management_plan"]
    assert plan["policy_id"] == "standard_scalp_v1"
    assert plan["horizon"] == "scalp"
    assert plan["management_tf"] == "M5"
    # The sleeve's own target hierarchy is the schedule's
    # target source; the lock is earned-by-acceptance; the
    # profile is the scalp profile (validity 24 M5 bars).
    sched = plan["partial_schedule"]
    assert sched["source"] == "sleeve_target_hierarchy"
    assert sched["targets"] == [2015.0, 2030.0]
    assert plan["profit_lock"] == {"mode": "earned_acceptance"}
    assert plan["time_stop"]["validity_window_bars"] == 24
    # And it is NOT the generic standard plan.
    generic = management_block_for("sweep_reversal", _sig())[
        "management_plan"]
    assert plan["policy_id"] != generic["policy_id"]
    assert plan["profit_lock"] != generic["profit_lock"]


def test_build_management_plan_bracket_directly():
    plan = build_management_plan(
        policy_id="fixed_bracket_v1", policy_version=1,
        horizon="scalp", management_tf="M5", policy_params={},
        profile=None, invalidation_price=1990.0, targets=[2015.0])
    assert plan["time_stop"] is None
    assert plan["invalidation"]["thesis_exit"] is None


# ------------------------------------------------------------ the book
def _thesis(tid="thesis-1", **kw):
    base = {
        "id": tid, "created_ts": 1000.0, "updated_ts": 1000.0,
        "scenario": "bullish", "direction": 1,
        "winner_p": 0.40, "peak_p": 0.40, "state": "intact",
        "version": 1, "status": "active", "supersedes": None,
        "invalidation": [
            {"text": "regime direction turns down",
             "trigger": "regime_direction_down"}],
        "plan": {"direction": 1, "entry": 4005.0,
                 "invalidation_price": 3980.0, "target": 4050.0,
                 "management_horizon": "standard_v1",
                 "basis": "scenario only"},
        "children": [], "revisions": [],
        "standing": {"scenarios": {}, "winner": "bullish",
                     "margin": 0.1, "regime": None},
    }
    base.update(kw)
    return base


def _filled_book(tmp_path):
    """A book holding one ENTERED scenario (ticket 11) whose
    declared management carries a Pass 6 plan, parented to a
    live thesis mirror."""
    book = ScenarioBook()
    book.sync_thesis(_thesis(), ts=1)
    block = management_block_for(
        "sweep_reversal",
        SimpleNamespace(invalidation=3990.0, targets=[4020.0]))
    setup = TrackedSetup(
        setup_id="pb:1", parent_setup_id="pb:1",
        playbook="sweep_reversal", route="h1_roster", direction=1,
        entry_ref=4000.0, invalidation=3990.0, targets=[4020.0],
        management=block, formation_ts="2026-11-08T00:00:00",
        touch_ts="2026-11-08T01:00:00",
        created_ts="2026-11-08T01:00:00", trigger_tf="H1",
        validity_bars=6, expiry_ts="2026-11-08T07:00:00",
        thesis_id="thesis-1", ticket=11)
    book.propose_from_setup(setup, ts=2)
    book.sync_setup(setup, ts=3, stage=STAGE_FILLED)
    return book


def test_plan_freezes_at_fill_and_cannot_be_rewritten():
    book = _filled_book(None)
    frozen = book.get("pb:1")["management_context"]
    assert frozen["management_plan"]["policy_id"] == "standard_v1"
    assert frozen["management_plan"]["time_stop"][
        "validity_window_bars"] == 16
    # Mutate the scenario's declared management AFTER the
    # fill: the frozen context does not move.
    book.scenarios["pb:1"]["management"]["management_plan"][
        "policy_id"] = "tampered"
    again = book.get("pb:1")["management_context"]
    assert again["management_plan"]["policy_id"] == "standard_v1"


def test_thesis_invalidation_marks_the_filled_child():
    book = _filled_book(None)
    closed = _thesis(status="invalidated",
                     close_reason="regime direction turned down")
    records = book.sync_thesis(None, ts=4, closed_theses=[closed])
    child = book.get("pb:1")
    assert child["state"] == "entered"  # marked, not closed
    assert child["thesis_invalidated"]["thesis_id"] == "thesis-1"
    assert child["thesis_invalidated"]["detail"] == \
        "regime direction turned down"
    marks = [r for r in records
             if r["type"] == "scenario_thesis_invalidated"]
    assert len(marks) == 1
    assert marks[0]["scenario_id"] == "pb:1"
    assert marks[0]["ticket"] == 11
    # The manager's lookup finds the scenario by its ticket.
    found = book.scenario_for_ticket(11)
    assert found is not None
    assert found["scenario_id"] == "pb:1"
    assert book.scenario_for_ticket(999) is None


def test_thesis_expiry_does_not_mark():
    book = _filled_book(None)
    closed = _thesis(status="expired", close_reason="aged out")
    book.sync_thesis(None, ts=4, closed_theses=[closed])
    assert book.get("pb:1")["thesis_invalidated"] is None


# ------------------------------------------------------------ the runner
class FakeExecutor:
    def __init__(self):
        self.orders: list[dict] = []
        self._ticket = 9000
        self._positions: list[Position] = []

    def connect(self, **kw):
        return True, "fake connected"

    def symbol_spec(self, symbol):
        from goldenswing.execution.risk import gold_spec
        return gold_spec()

    def positions(self, symbol=None):
        return [p for p in self._positions
                if symbol is None or p.symbol == symbol]

    def place_order(self, request):
        self._ticket += 1
        self.orders.append({"ticket": self._ticket})
        self._positions.append(Position(
            ticket=self._ticket, symbol=request.symbol,
            direction=request.direction, volume=request.volume,
            entry_price=request.entry_ref, sl=request.stop_loss,
            tp=request.take_profit, magic=request.magic,
            alert_id=request.alert_id))
        return OrderResult(True, ticket=self._ticket,
                           price=request.entry_ref,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        self._positions = [p for p in self._positions
                           if p.ticket != ticket]
        return OrderResult(True, ticket=ticket)


def _frame(rows, start, freq):
    idx = pd.date_range(start, periods=len(rows), freq=freq,
                        tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [100.0] * len(rows),
         "real_volume": [0.0] * len(rows),
         "spread": [30] * len(rows)}, index=idx)


def _frames():
    base = [(2000.0 + 0.1 * i, 2001.0 + 0.1 * i,
             1999.0 + 0.1 * i, 2000.5 + 0.1 * i) for i in range(60)]
    return {"D1": _frame(base[:30], "2026-01-05 00:00", "D"),
            "H4": _frame(base[:30], "2026-01-05 00:00", "4h"),
            "H1": _frame(base, "2026-01-05 00:00", "h"),
            "M15": _frame(base, "2026-01-05 00:00", "15min"),
            "M5": _frame(base * 2, "2026-01-05 00:00", "5min")}


def _runner(tmp_path):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("stub",),
        news_pause_enabled=False)
    runner = MT5Runner(cfg, feed=None, executor=FakeExecutor())
    runner._stack = TimeframeStack.from_frames(_frames())
    runner._last_tick = {"mid": 2000.0}
    runner._send_alert = lambda *a, **k: None
    return runner


def _journal_records(runner):
    path = runner.cfg.journal_path
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _open(runner, ticket, *, playbook, stop=1990.0,
          targets=(2015.0,), volume=0.10, atr=10.0,
          broker_sl=1987.5):
    req = OrderRequest(symbol="XAUUSD", direction=1,
                       volume=volume, stop_loss=stop,
                       entry_ref=2000.0, targets=list(targets),
                       atr=atr, bar_index=5)
    pos = runner.tick_manager.open(
        ticket=ticket, request=req, fill_price=2000.0, entry_bar=6,
        broker_sl=broker_sl, playbook=playbook)
    runner.executor._positions.append(Position(
        ticket=ticket, symbol="XAUUSD", direction=1,
        volume=volume, entry_price=2000.0, sl=broker_sl,
        tp=targets[0] if targets else None, magic=runner.cfg.magic,
        alert_id=pos.alert_id))
    return pos


def _ctx(*, playbook, clock, entry_mgmt, last_mgmt, profile, plan):
    return {
        "profile": profile,
        "progress_trimmed": False, "progress_trim_bar": None,
        "drift_trimmed": False, "preprint_key": None,
        "zone_banked": False, "adr_banked": False,
        "defended_stop": None, "entry_total": None,
        "opposing_prox": None, "style_key": "intraday",
        "playbook": playbook,
        "invalidation": 1990.0, "entry_bar": 10,
        "clock_tf": clock, "management_clock": clock,
        "entry_mgmt_bar": entry_mgmt, "last_mgmt_bar": last_mgmt,
        "management_plan": plan,
    }


def test_verdicts_run_per_position_on_its_own_clock(tmp_path):
    runner = _runner(tmp_path)
    frames = runner._stack.frames
    m5_now = len(frames["M5"]) - 1
    d1_now = len(frames["D1"]) - 1
    # Scalp position: 30 M5 bars elapsed — past its plan's
    # 24-bar validity window.  Swing position: 1 D1 bar
    # elapsed — deep inside its 16-bar window.
    scalp_pos = _open(runner, 21, playbook="hot_scalp",
                      targets=(2015.0,))
    scalp_pos.best_r = 0.5
    swing_pos = _open(runner, 22, playbook="qm_d1",
                      targets=(2060.0,))
    swing_pos.best_r = 0.5
    scalp_plan = management_block_for(
        "hot_scalp", _sig())["management_plan"]
    swing_plan = management_block_for(
        "qm_d1", _sig())["management_plan"]
    runner._position_ctx = {
        21: _ctx(playbook="hot_scalp", clock="M5",
                 entry_mgmt=m5_now - 30, last_mgmt=m5_now - 1,
                 profile=profile_for(style="scalp", clock_tf="M5"),
                 plan=scalp_plan),
        22: _ctx(playbook="qm_d1", clock="D1",
                 entry_mgmt=d1_now - 1, last_mgmt=d1_now - 1,
                 profile=profile_for(style="trend", clock_tf="D1"),
                 plan=swing_plan),
    }
    runner._manage_positions_bar_close({})
    verdicts = {r["ticket"]: r for r in _journal_records(runner)
                if r.get("type") == "management_verdict"}
    assert set(verdicts) == {21, 22}
    # Same pass, same frames: the scalp's validity expired on
    # ITS clock; the swing rides on its own.
    assert verdicts[21]["validity_expired"] is True
    assert verdicts[21]["horizon"] == "scalp"
    assert verdicts[21]["management_tf"] == "M5"
    assert verdicts[21]["elapsed_mgmt_bars"] == 30
    assert verdicts[22]["validity_expired"] is False
    assert verdicts[22]["stall"] is False
    assert verdicts[22]["horizon"] == "swing"
    assert verdicts[22]["management_tf"] == "D1"
    assert verdicts[22]["elapsed_mgmt_bars"] == 1


def test_thesis_exit_is_enacted_and_journaled(tmp_path):
    runner = _runner(tmp_path)
    book = _filled_book(tmp_path)
    closed = _thesis(status="invalidated",
                     close_reason="regime direction turned down")
    book.sync_thesis(None, ts=4, closed_theses=[closed])
    runner._scenario_book = book
    pos = _open(runner, 11, playbook="sweep_reversal",
                stop=3990.0, targets=(4020.0,))
    pos.entry_price = 4000.0
    closes: list = []
    runner._record_close = lambda t, p, r, s: closes.append((t, r))
    frames = runner._stack.frames
    h1_now = len(frames["H1"]) - 1
    plan = management_block_for(
        "sweep_reversal", _sig())["management_plan"]
    runner._position_ctx = {
        11: _ctx(playbook="sweep_reversal", clock="H1",
                 entry_mgmt=h1_now - 2, last_mgmt=h1_now - 1,
                 profile=profile_for(style="trend", clock_tf="H1"),
                 plan=plan),
    }
    runner._manage_positions_bar_close({})
    # The position was closed by the manager on the thesis
    # evidence — and the close is named a thesis exit.
    assert runner.executor.positions() == []
    assert closes == [(11, "thesis_exit")]
    mgmt = [r for r in _journal_records(runner)
            if r.get("decision") == "management"
            and r.get("reason") == "thesis_exit"]
    assert len(mgmt) == 1
    ctx = mgmt[0]["context"]
    assert ctx["plan_step"] == "thesis_exit"
    assert ctx["management_policy_id"] == "standard_v1"
    assert ctx["management_horizon"] == "intraday"
    assert ctx["management_tf"] == "H1"


def test_management_records_carry_plan_stamps_and_steps(tmp_path):
    runner = _runner(tmp_path)
    pos = _open(runner, 31, playbook="sweep_reversal")
    plan = management_block_for(
        "sweep_reversal", _sig())["management_plan"]
    runner._position_ctx = {31: {"entry_total": 62.0,
                                 "management_plan": plan}}
    runner._log_management(31, pos, "progress_trim", "note")
    mgmt = [r for r in _journal_records(runner)
            if r.get("decision") == "management"]
    assert len(mgmt) == 1
    ctx = mgmt[0]["context"]
    assert ctx["management_policy_id"] == "standard_v1"
    assert ctx["management_horizon"] == "intraday"
    assert ctx["management_tf"] == "H1"
    assert ctx["plan_step"] == "stall_reduce"


def test_panic_close_journals_a_safeguard_override(tmp_path):
    runner = _runner(tmp_path)
    # Wide stop (100 pts) + ATR 10: a 50-point adverse move is
    # a panic (severity 50/30 > 1.5) BEFORE the stop is near.
    pos = _open(runner, 41, playbook="sweep_reversal",
                stop=1900.0, targets=(2100.0,), broker_sl=1897.5)
    runner._record_close = lambda t, p, r, s: None
    runner._last_tick = {"mid": 1950.0, "bid": 1949.8,
                         "ask": 1950.2}
    runner._manage_tick(1950.0, {"tick_actions": 0})
    overrides = [r for r in _journal_records(runner)
                 if r.get("type") == "safeguard_override"]
    assert len(overrides) == 1
    assert overrides[0]["safeguard"] == "panic_candle"
    assert overrides[0]["action"] == "close_position"
    assert overrides[0]["ticket"] == 41
    assert runner.executor.positions() == []


def test_preprint_securing_journals_a_safeguard_override(tmp_path):
    runner = _runner(tmp_path)
    runner.cfg.news_pause_enabled = True
    pos = _open(runner, 42, playbook="sweep_reversal")
    event = SimpleNamespace(
        title="NFP", time=datetime(2026, 11, 8, 13, 30,
                                   tzinfo=timezone.utc))
    runner.news = SimpleNamespace(
        next_event=lambda when: (event, 10.0),
        PREPRINT_LEAD_MIN=30)
    ctx = {"preprint_key": None, "style_key": "intraday",
           "invalidation": 1990.0}
    acted = runner._preprint_for_position(42, pos, ctx, 2000.0,
                                          10.0, {})
    assert acted is True
    overrides = [r for r in _journal_records(runner)
                 if r.get("type") == "safeguard_override"]
    assert len(overrides) == 1
    assert overrides[0]["safeguard"] == "news_preprint"
    assert overrides[0]["ticket"] == 42


# ------------------------------------------------------------ census
def test_census_separates_horizons_and_names_the_flat_one():
    def verdict(horizon, tf, window, stall=False):
        return {"type": "management_verdict", "horizon": horizon,
                "management_tf": tf,
                "validity_window_bars": window, "stall": stall,
                "validity_expired": False,
                "structure_break": False}
    records = [
        verdict("scalp", "M5", 24), verdict("scalp", "M5", 24),
        verdict("intraday", "H1", 16),
        verdict("swing", "D1", 16),
        {"type": "signal", "decision": "management",
         "reason": "thesis_exit",
         "context": {"management_horizon": "swing"}},
    ]
    census = management_census(records)
    by = census["by_horizon"]
    # Minutes vs hours vs days — the separation is in the data.
    assert by["scalp"]["validity_window_minutes"] == 120
    assert by["intraday"]["validity_window_minutes"] == 960
    assert by["swing"]["validity_window_minutes"] == 23040
    assert by["swing"]["thesis_exits"] == 1
    assert census["indistinguishable_from_standard"] == []
    # A scalp cell with standard's exact signature is named.
    flat = management_census([
        verdict("intraday", "H1", 16),
        {"type": "management_verdict", "horizon": "scalp",
         "management_tf": "H1", "validity_window_bars": 16,
         "stall": False, "validity_expired": False,
         "structure_break": False}])
    assert flat["indistinguishable_from_standard"] == ["scalp"]


# ------------------------------------------------- tick-level identity
def _tick_stream(tm, ticket, prices):
    out = []
    for price in prices:
        for action in tm.on_tick(ticket, price):
            out.append((action.kind, action.reason,
                        round(action.volume, 4), action.level))
            tm.apply(ticket, action)
    return out


def test_qm_m5_tick_management_is_byte_identical():
    tm = TickManager(tick=TickConfig())
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.10,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2015.0], atr=10.0, bar_index=5)
    tm.open(ticket=51, request=req, fill_price=2000.0, entry_bar=6,
            broker_sl=1987.5, playbook="qm_m5",
            management_horizon="scalp",
            management_policy_id="fixed_bracket_v1")
    assert tm.is_fixed_bracket(51) is True
    # +1R at 2012.5, +2R at 2025, +3R at 2037.5 (risk 12.5):
    # partials bank on the tiers; the stop NEVER moves (Step
    # 13's bracket is the whole management) — the exact action
    # stream the pre-pass manager produces.
    actions = _tick_stream(
        tm, 51, [2005.0, 2012.5, 2020.0, 2025.0, 2037.5, 1995.0])
    assert actions == [
        ("partial", "partial_tier_1", 0.05, 2012.5),
        ("partial", "partial_tier_2", 0.025, 2025.0),
        ("partial", "partial_tier_3", 0.025, 2037.5),
    ]
    assert all(kind != "modify_sl" for kind, *_ in actions)


def test_hot_scalp_lock_must_be_earned_unlike_standard():
    tm = TickManager(tick=TickConfig())
    for ticket, pb in ((52, "hot_scalp"), (53, "sweep_reversal")):
        req = OrderRequest(symbol="XAUUSD", direction=1,
                           volume=0.10, stop_loss=1990.0,
                           entry_ref=2000.0, targets=[2015.0],
                           atr=10.0, bar_index=5)
        tm.open(ticket=ticket, request=req, fill_price=2000.0,
                entry_bar=6, broker_sl=1987.5, playbook=pb)
    # +0.6R, halfway to TP1: the standard position's progress
    # arm locks (entry + 7.5); the Hot Scalp position's does
    # not — its lock is earned by acceptance or the 2R
    # backstop, per its own declared plan.
    hs = tm.on_tick(52, 2007.5)
    std = tm.on_tick(53, 2007.5)
    assert hs == []
    assert [(a.kind, a.reason, a.level) for a in std] == [
        ("modify_sl", "profit_lock", 2007.5)]
