"""Step 14 (owner rulings 2026-10-30 §1): persistent setup
lifecycle + H1 handoff.

The Oct 30 failure these tests pin shut: scan-detected roster
candidates were journaled "observed, never routed" and then
DISCARDED — the inversion shorts at 01:00/06:00 and the fvg_limit
short at 16:00 died as observations — and the H1 trigger that
fired at 03:30 left no downstream record at all.

Acceptance shape (journal evidence, per the ruling):
- a live-close scan candidate is REGISTERED with a stable setup
  ID + parent ID and journaled candidate_detected -> setup_armed;
- a warmup or shadow scan registers NOTHING;
- a same-direction trigger fire routes the PERSISTED candidate
  through the shared decision/order path (provenance shows the
  trigger path + the original touch; the decision record joins
  the chain by setup ID);
- an opposite-direction fire, an expired setup and an
  invalidated setup are never routed;
- EVERY fire produces an execution_evaluated disposition
  record — including a fire where nothing qualifies and a fire
  whose signal path raises (the 03:30 hole);
- setups survive a restart (new runner, same journal dir);
- a routed setup's pending fill / cancel resolves its
  order_submitted stage.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution import setup_lifecycle as slc
from goldenswing.execution.setup_lifecycle import (
    SetupLifecycleStore, TrackedSetup)

from test_p3_trader import (_FakeScore, _StubPB, _flat_h1, _frame,
                            _runner, _sig)


# ---------------------------------------------------------------- store

def _setup(setup_id="pb:2026-01-05 00:00:00:1:2000.00", **kw):
    base = dict(setup_id=setup_id, parent_setup_id=setup_id,
                playbook="fvg_midline", route="h1_roster",
                direction=1, entry_ref=2000.0, invalidation=1990.0,
                targets=[2030.0])
    base.update(kw)
    return TrackedSetup(**base)


def test_store_register_dedupe_and_terminal_lock(tmp_path):
    store = SetupLifecycleStore(tmp_path / "lc.json")
    s = _setup()
    assert store.register(s, ts="2026-01-05 00:00:00") is True
    # Re-detection of the same instance is a no-op.
    assert store.register(_setup(), ts="2026-01-05 01:00:00") \
        is False
    assert len(store.active()) == 1
    ev = store.transition(s.setup_id, slc.STAGE_ARMED,
                          ts="2026-01-05 00:01:00")
    assert ev["setup_id"] == s.setup_id
    assert ev["parent_setup_id"] == s.parent_setup_id
    store.transition(s.setup_id, slc.STAGE_REJECTED,
                     ts="2026-01-05 02:00:00", detail="refused")
    # A terminal setup never moves again.
    assert store.transition(s.setup_id, slc.STAGE_ARMED,
                            ts="2026-01-05 03:00:00") is None
    assert store.active() == []


def test_store_persistence_round_trip(tmp_path):
    path = tmp_path / "lc.json"
    store = SetupLifecycleStore(path)
    store.register(_setup(), ts="2026-01-05 00:00:00")
    store.transition("pb:2026-01-05 00:00:00:1:2000.00",
                     slc.STAGE_ARMED, ts="2026-01-05 00:01:00",
                     alert_id="MT5-fvg_midline-29")
    again = SetupLifecycleStore(path)
    s = again.get("pb:2026-01-05 00:00:00:1:2000.00")
    assert s is not None and s.stage == slc.STAGE_ARMED
    assert s.alert_id == "MT5-fvg_midline-29"
    assert [h["stage"] for h in s.history] == [
        slc.STAGE_DETECTED, slc.STAGE_ARMED]
    assert again.by_alert_id("MT5-fvg_midline-29") is s


def test_sanitize_context_drops_and_counts_non_json():
    clean, dropped = slc.sanitize_context(
        {"a": 1, "b": [1.0, "x"], "c": {"d": True},
         "frame": object(), "nested_bad": {"x": object()}})
    assert clean == {"a": 1, "b": [1.0, "x"], "c": {"d": True}}
    assert dropped == 2
    assert slc.sanitize_context(None) == ({}, 0)


# ---------------------------------------------------------------- runner

def _evals(runner, playbook=None, kind=None):
    out = [r for r in runner.journal.records()
           if r.get("type") == "evaluation"]
    if playbook is not None:
        out = [r for r in out if r.get("playbook") == playbook]
    if kind is not None:
        out = [r for r in out if r.get("kind") == kind]
    return out


def _setup_events(runner):
    return _evals(runner, kind="setup_event")


def _scan_runner(tmp_path, **kw):
    runner = _runner(tmp_path, **kw)
    h1 = _flat_h1(40, 2000.0)
    runner._stack = SimpleNamespace(
        frames={"H1": h1,
                "H4": _frame([(2000.0, 2001.0, 1999.0, 2000.5)]
                             * 40, freq="4h")})
    runner._live_setup = {"bias_d1": 1, "bias_h4": 1,
                          "regime": "trend",
                          "h4_context": "uptrend structure",
                          "size_mult": 1.0}
    tick_time = h1.index[-1] + pd.Timedelta(minutes=90)
    runner._last_tick = {"mid": 2000.0, "bid": 1999.9,
                         "ask": 2000.1, "spread": 0.2,
                         "time": tick_time}
    return runner, h1


def _sweep(runner, frame, *, source="live_close", shadow=False,
           direction=1):
    spec = SimpleNamespace(
        name="fvg_midline", route="h1_roster",
        playbook_class=lambda: _StubPB(
            "fvg_midline",
            _sig(direction=direction, index=len(frame) - 1)))
    runner._sweep_eval_one(
        spec, frame, clock="h1_close", intended_tf="H1",
        actual_tf="H1", source=source, shadow=shadow)
    return spec


def test_scan_candidate_registers_setup(tmp_path):
    runner, frame = _scan_runner(tmp_path)
    _sweep(runner, frame)
    store = runner._setup_store
    active = store.active()
    assert len(active) == 1
    s = active[0]
    assert s.playbook == "fvg_midline" and s.direction == 1
    assert s.stage == slc.STAGE_ARMED
    assert s.parent_setup_id == s.setup_id
    assert s.expiry_ts is not None
    stages = [r["trigger_stage"] for r in _setup_events(runner)]
    assert stages == [slc.STAGE_DETECTED, slc.STAGE_ARMED]
    ev = _setup_events(runner)[0]["evidence"]
    assert ev["setup_id"] == s.setup_id
    assert ev["parent_setup_id"] == s.parent_setup_id
    # The scan record itself is unchanged: still observed-only.
    rec = [r for r in _evals(runner, "fvg_midline",
                             kind="playbook_eval")
           if r["outcome"] == "candidate_emitted"][-1]
    assert rec["reason_code"] == "path.trigger_not_fired"


def test_warmup_and_shadow_scans_register_nothing(tmp_path):
    runner, frame = _scan_runner(tmp_path)
    _sweep(runner, frame, source="warmup_scan")
    _sweep(runner, frame, source="live_close", shadow=True)
    assert runner._setup_store.active() == []
    assert _setup_events(runner) == []


def test_rescan_same_instance_does_not_duplicate(tmp_path):
    runner, frame = _scan_runner(tmp_path)
    _sweep(runner, frame)
    _sweep(runner, frame)
    assert len(runner._setup_store.active()) == 1
    assert len(_setup_events(runner)) == 2  # detected + armed only


# ---------------------------------------------------------------- routing

def _fire_runner(tmp_path):
    runner, h1 = _scan_runner(
        tmp_path, execute_playbooks=("fvg_midline",))
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    runner._get_playbooks = lambda: [
        _StubPB("fvg_midline", _sig(index=0))]
    return runner, h1


def _verdict(direction=1):
    return {"allowed": True, "bias_d1": 1, "bias_h4": 1,
            "regime": "trend", "direction": direction,
            "zone": {"top": 2010.0, "bottom": 1990.0,
                     "level": 2000.0},
            "trigger": "bos", "entry": 2000.0, "stop": 1990.0,
            "target": 2030.0, "risk": 10.0, "entry_clock": "M30",
            "session": {}, "amd_phase": "accumulation",
            "qm": {"status": "none"}, "volume": {}}


def test_fire_routes_armed_setup_through_order_path(tmp_path):
    runner, h1 = _fire_runner(tmp_path)
    _sweep(runner, h1)
    setup = runner._setup_store.active()[0]
    routing: dict = {}
    stats = {"signals": 0, "gated": 0}
    runner._run_signal_path(h1, len(h1) - 1, [], 1.0, _verdict(1),
                            stats, lifecycle_routing=routing)
    assert routing["fvg_midline"]["status"] == "evaluated"
    terms = routing["fvg_midline"]["terminals"]
    assert terms and terms[0]["order_result"] in (
        "filled", "placed_pending")
    # The decision record joins the lifecycle chain by ID and its
    # candidate provenance shows the trigger path + eligibility
    # computed from the original live-close touch.
    recs = [r for r in _evals(runner, "fvg_midline")
            if (r.get("evidence") or {}).get("lifecycle_routed")]
    assert recs
    rec = recs[-1]
    assert rec["evidence"]["setup_id"] == setup.setup_id
    prov = rec["candidate"]["provenance"]
    assert prov["source"] == "trigger_path"
    assert prov["execution_eligible"] is True
    assert prov["instance_key"] == setup.setup_id
    # Lifecycle advances: fired -> evaluated -> submitted/filled.
    frags = runner._apply_lifecycle_routing(routing, 1)
    assert frags
    s = runner._setup_store.get(setup.setup_id)
    assert s.stage in (slc.STAGE_SUBMITTED, slc.STAGE_FILLED)
    stages = [h["stage"] for h in s.history]
    assert stages[:4] == [slc.STAGE_DETECTED, slc.STAGE_ARMED,
                          slc.STAGE_TRIGGER_FIRED,
                          slc.STAGE_EVALUATED]


def test_opposite_direction_fire_does_not_route(tmp_path):
    runner, h1 = _fire_runner(tmp_path)
    _sweep(runner, h1)
    routing: dict = {}
    stats = {"signals": 0, "gated": 0}
    runner._run_signal_path(h1, len(h1) - 1, [], 1.0, _verdict(-1),
                            stats, lifecycle_routing=routing)
    assert routing == {}
    assert runner._setup_store.active()[0].stage == slc.STAGE_ARMED
    assert not [r for r in _evals(runner, "fvg_midline")
                if (r.get("evidence") or {}).get("lifecycle_routed")]


def test_expired_setup_is_not_routed_and_upkeep_expires_it(
        tmp_path):
    runner, h1 = _fire_runner(tmp_path)
    spec = SimpleNamespace(
        name="fvg_midline", route="h1_roster", validity_bars=0,
        playbook_class=lambda: _StubPB(
            "fvg_midline", _sig(index=len(h1) - 1)))
    runner._sweep_eval_one(
        spec, h1, clock="h1_close", intended_tf="H1",
        actual_tf="H1", source="live_close", shadow=False)
    routing: dict = {}
    stats = {"signals": 0, "gated": 0}
    runner._run_signal_path(h1, len(h1) - 1, [], 1.0, _verdict(1),
                            stats, lifecycle_routing=routing)
    assert routing == {}
    runner._upkeep_setups(h1)
    s = runner._setup_store.get(
        runner._setup_store._setups and
        list(runner._setup_store._setups)[0])
    assert s.stage == slc.STAGE_EXPIRED
    assert [r["trigger_stage"] for r in _setup_events(runner)][-1] \
        == slc.STAGE_EXPIRED


def test_upkeep_invalidates_on_close_beyond_stop(tmp_path):
    runner, h1 = _scan_runner(tmp_path)
    _sweep(runner, h1)  # long, invalidation 1990
    bad = _frame([(1985.0, 1986.0, 1984.0, 1985.0)] * 3,
                 start=str(h1.index[-1] + pd.Timedelta(hours=1)))
    runner._upkeep_setups(bad)
    s = runner._setup_store.active()
    assert s == []
    stages = [r["trigger_stage"] for r in _setup_events(runner)]
    assert stages[-1] == slc.STAGE_INVALIDATED


# ------------------------------------------------------- disposition

def _pend(direction=-1):
    return {"direction": direction, "trigger": "bos",
            "confirm_shapes": ["bos", "failed_retest"],
            "entry": 4000.82, "stop": 4010.0, "target": 3990.0,
            "risk": 9.18, "level": 4010.0, "zone": None,
            "entry_clock": "M30",
            "h1_open": pd.Timestamp("2026-10-30 03:00:00"),
            "setup": {"bias_d1": -1, "bias_h4": -1,
                      "regime": "trend"}}


def _tick():
    return {"mid": 4000.5, "bid": 4000.3, "ask": 4000.7,
            "spread": 0.4, "time": "2026-10-30 03:30:00"}


def _dispositions(runner):
    return [r for r in _evals(runner, "__trigger__")
            if r.get("trigger_stage") == "execution_evaluated"]


def test_fire_with_no_qualifying_playbook_still_dispositions(
        tmp_path):
    """The 03:30 regression pin: a fired trigger whose signal
    path qualifies nothing must still leave an
    execution_evaluated record — the fire never vanishes."""
    runner, h1 = _scan_runner(tmp_path)
    runner._run_signal_path = lambda *a, **k: None
    runner._run_live_signal(_pend(), _tick(), {"signals": 0})
    dis = _dispositions(runner)
    assert dis and "no armed setups routed" in dis[-1]["reason"]


def test_fire_disposition_survives_signal_path_exception(tmp_path):
    runner, h1 = _scan_runner(tmp_path)

    def _boom(*a, **k):
        raise RuntimeError("path exploded")

    runner._run_signal_path = _boom
    with pytest.raises(RuntimeError):
        runner._run_live_signal(_pend(), _tick(), {"signals": 0})
    dis = _dispositions(runner)
    assert dis and dis[-1]["reason_code"] == "path.exception"


# ------------------------------------------------------- restart + hooks

def test_setups_survive_restart(tmp_path):
    runner, frame = _scan_runner(tmp_path)
    _sweep(runner, frame)
    setup_id = runner._setup_store.active()[0].setup_id
    runner2, _ = _scan_runner(tmp_path)
    s = runner2._setup_store.get(setup_id)
    assert s is not None and s.stage == slc.STAGE_ARMED


def test_cancel_pending_resolves_submitted_setup(tmp_path):
    runner, frame = _scan_runner(tmp_path)
    _sweep(runner, frame)
    s = runner._setup_store.active()[0]
    runner._transition_setup(s, slc.STAGE_TRIGGER_FIRED, "fire")
    runner._transition_setup(s, slc.STAGE_EVALUATED, "accepted")
    runner._transition_setup(
        s, slc.STAGE_SUBMITTED, "placed",
        alert_id="MT5-fvg_midline-39")
    state = SimpleNamespace(kind="limit", level=2000.0,
                            alert_id="MT5-fvg_midline-39")
    runner._pending_orders[777] = state
    stats: dict = {}
    assert runner._cancel_pending(
        777, "expired: no fill after 48 M5 bars", stats) is True
    assert runner._setup_store.get(s.setup_id).stage \
        == slc.STAGE_EXPIRED
    stages = [r["trigger_stage"] for r in _setup_events(runner)]
    assert stages[-1] == slc.STAGE_EXPIRED
