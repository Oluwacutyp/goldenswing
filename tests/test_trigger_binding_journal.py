"""Trigger-binding journal (WS4 residual #2 follow-up).

The Nov 3-4 episode trace: the window's 25 __trigger__
evaluations were 11 zone-touch armings + 8 left-zone
invalidations of the trigger machine's own pendings (records
that never reach the lifecycle binder by construction) and 3
SHORT fires whose execution dispositions read only "no armed
setups routed; armed setups remaining in direction: 0".  At
each fire the binder's same-direction group was empty — the
armed instances were inversion longs, the one earlier short
instance had expired the night before, and the next short
registered after the last fire — so non-binding was correct,
but proving it took a forensic pass over three record kinds.

What this file pins: every fire that reaches the lifecycle
routing selection journals ONE kind=trigger_binding record —
the bound instance(s) with the rule that picked each
(bracket / nearest / oldest / single), or the not-bound
reason from the closed taxonomy (no_armed_instances /
direction_mismatch / all_expired / shadow_only) with every
armed instance's disposition at fire time.  Observation-only:
the routing outcome itself is asserted unchanged everywhere.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from goldenswing.execution import setup_lifecycle as slc
from test_p3_trader import _FakeScore, _StubPB, _frame, _runner, _sig


# ------------------------------------------------------------ helpers

def _binding_records(runner):
    return [r for r in runner.journal.records()
            if r.get("kind") == "trigger_binding"]


def _flat_frame(price=2000.0, bars=40, end="2026-01-05 00:00"):
    idx = pd.date_range(end=end, periods=bars, freq="1h", tz="UTC")
    rows = [(price, price + 1.0, price - 1.0, price + 0.5)] * bars
    return pd.DataFrame(rows, index=idx,
                        columns=["open", "high", "low", "close"])


def _basic_runner(tmp_path, frame, **cfg_kw):
    runner = _runner(tmp_path, **cfg_kw)
    runner._stack = SimpleNamespace(
        frames={"H1": frame,
                "H4": _frame([(2000.0, 2001.0, 1999.0, 2000.5)]
                             * 40, freq="4h")})
    runner._live_setup = {"bias_d1": 1, "bias_h4": 1,
                          "regime": "trend",
                          "h4_context": "uptrend structure",
                          "size_mult": 1.0}
    tick_time = frame.index[-1] + pd.Timedelta(minutes=90)
    runner._last_tick = {"mid": 2000.0, "bid": 1999.9,
                         "ask": 2000.1, "spread": 0.2,
                         "time": tick_time}
    return runner


def _sweep(runner, frame, name="fvg_midline", *, shadow=False,
           direction=1, entry=2000.0, stop=1990.0,
           entry_zone=None, validity_bars=None):
    sig = _sig(direction=direction, entry=entry, stop=stop,
               index=len(frame) - 1)
    if entry_zone is not None:
        sig.entry_zone = entry_zone
    spec = SimpleNamespace(
        name=name, route="h1_roster",
        playbook_class=lambda: _StubPB(name, sig))
    if validity_bars is not None:
        spec.validity_bars = validity_bars
    runner._sweep_eval_one(
        spec, frame, clock="h1_close", intended_tf="H1",
        actual_tf="H1", source="live_close", shadow=shadow)
    return spec


def _verdict(direction=1, entry=2000.0):
    return {"allowed": True, "bias_d1": 1, "bias_h4": 1,
            "regime": "trend", "direction": direction,
            "zone": {"top": 2010.0, "bottom": 1990.0,
                     "level": 2000.0},
            "trigger": "bos", "entry": entry, "stop": 1990.0,
            "target": 2030.0, "risk": 10.0, "entry_clock": "M30",
            "session": {}, "amd_phase": "accumulation",
            "qm": {"status": "none"}, "volume": {}}


def _fire(runner, frame, verdict):
    routing: dict = {}
    stats = {"signals": 0, "gated": 0}
    runner._run_signal_path(frame, len(frame) - 1, [], 1.0,
                            verdict, stats,
                            lifecycle_routing=routing)
    return routing


def _binding(runner):
    recs = _binding_records(runner)
    assert len(recs) == 1
    return recs[0]["evidence"]["binding"], recs[0]


# --------------------------------------- the episode shape (the wound)

def test_episode_shape_fire_opposite_to_armed_longs(tmp_path):
    """Nov 4, 01:30Z rebuilt: two inversion LONGS armed (the
    20:00Z and 23:00Z touches); a SHORT fire reaches the
    binder and binds nothing — and the record now SAYS so,
    naming both instances and why."""
    frame_a = _flat_frame(3917.57, end="2026-11-04 01:00")
    runner = _basic_runner(tmp_path, frame_a,
                           execute_playbooks=("inversion",))
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    _sweep(runner, frame_a, name="inversion", direction=1,
           entry=3917.57, stop=3913.0,
           entry_zone=(3912.0, 3922.0), validity_bars=200)
    frame_b = _flat_frame(3927.03, end="2026-11-04 02:00")
    runner._last_tick = {
        "mid": 3927.03, "bid": 3926.93, "ask": 3927.13,
        "spread": 0.2,
        "time": frame_b.index[-1] + pd.Timedelta(minutes=90)}
    _sweep(runner, frame_b, name="inversion", direction=1,
           entry=3927.03, stop=3922.81,
           entry_zone=(3924.0, 3932.0), validity_bars=200)
    armed = runner._setup_store.active()
    assert len(armed) == 2
    runner._get_playbooks = lambda: [
        _StubPB("inversion", _sig(index=0))]
    # The fire: short, at the bearish zone (~3955) — the
    # episode's 01:30Z volume_spike.
    routing = _fire(runner, frame_b, _verdict(-1, entry=3955.26))
    # Nothing routed — the historical outcome, unchanged.
    assert routing == {}
    assert all(s.stage == slc.STAGE_ARMED
               for s in runner._setup_store.active())
    # ...and exactly one record explains it.
    binding, rec = _binding(runner)
    assert rec["outcome"] == "not_eligible"
    assert rec["reason_code"] == "path.binding_direction_mismatch"
    assert binding["decision"] == "not_bound"
    assert binding["reason"] == "direction_mismatch"
    assert binding["fire_direction"] == -1
    assert binding["bound"] == []
    disps = {i["setup_id"]: i["disposition"]
             for i in binding["instances"]}
    assert len(disps) == 2
    assert set(disps.values()) == {"other_direction"}
    assert {i["direction"] for i in binding["instances"]} == {1}


# ---------------------------------------------------------- bound fires

def test_bound_single_instance_names_rule_single(tmp_path):
    frame = _flat_frame(2000.0)
    runner = _basic_runner(tmp_path, frame,
                           execute_playbooks=("order_block",))
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    _sweep(runner, frame, name="order_block", direction=1,
           entry=2000.0, stop=1990.0,
           entry_zone=(1995.0, 2005.0), validity_bars=200)
    setup = runner._setup_store.active()[0]
    runner._get_playbooks = lambda: [
        _StubPB("order_block", _sig(index=0))]
    routing = _fire(runner, frame, _verdict(1, entry=2000.0))
    assert routing["order_block"]["setup_id"] == setup.setup_id
    binding, rec = _binding(runner)
    assert rec["outcome"] == "candidate_emitted"
    assert not rec["reason_code"]  # bound carries no reason code
    assert binding["decision"] == "bound"
    assert binding["bound"] == [{
        "setup_id": setup.setup_id, "playbook": "order_block",
        "rule": "single"}]
    assert binding["instances"] == [{
        "setup_id": setup.setup_id, "playbook": "order_block",
        "direction": 1, "expiry_ts": setup.expiry_ts,
        "disposition": "bound:single"}]


def _two_instance_runner(tmp_path):
    """Two armed order_block LONG instances: OLDER at entry
    2100 (zone 2095-2105), NEWER at 2000 (zone 1995-2005) —
    the WS3 binding fixture."""
    frame1 = _flat_frame(2100.0)
    runner = _basic_runner(
        tmp_path, frame1, execute_playbooks=("order_block",))
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    _sweep(runner, frame1, name="order_block", entry=2100.0,
           stop=2090.0, entry_zone=(2095.0, 2105.0),
           validity_bars=200)
    idx2 = pd.date_range(frame1.index[-1] + pd.Timedelta(hours=1),
                         periods=40, freq="1h", tz="UTC")
    frame2 = pd.DataFrame(
        [(2000.0, 2001.0, 1999.0, 2000.5)] * 40, index=idx2,
        columns=["open", "high", "low", "close"])
    runner._last_tick = {
        "mid": 2000.0, "bid": 1999.9, "ask": 2000.1,
        "spread": 0.2,
        "time": frame2.index[-1] + pd.Timedelta(minutes=90)}
    _sweep(runner, frame2, name="order_block", entry=2000.0,
           stop=1990.0, entry_zone=(1995.0, 2005.0))
    runner._get_playbooks = lambda: [
        _StubPB("order_block", _sig(index=0))]
    setups = runner._setup_store.active()
    assert len(setups) == 2
    older, newer = setups[0], setups[1]
    assert older.entry_ref == 2100.0 and newer.entry_ref == 2000.0
    return runner, frame2, older, newer


def test_bound_bracket_rule_and_not_selected_loser(tmp_path):
    runner, frame, older, newer = _two_instance_runner(tmp_path)
    routing = _fire(runner, frame, _verdict(1, entry=2001.0))
    assert routing["order_block"]["setup_id"] == newer.setup_id
    binding, _rec = _binding(runner)
    assert binding["bound"] == [{
        "setup_id": newer.setup_id, "playbook": "order_block",
        "rule": "bracket"}]
    disps = {i["setup_id"]: i["disposition"]
             for i in binding["instances"]}
    assert disps[newer.setup_id] == "bound:bracket"
    assert disps[older.setup_id] == "not_selected"


def test_bound_nearest_rule(tmp_path):
    runner, frame, older, newer = _two_instance_runner(tmp_path)
    # 2060 is outside both zones; nearer the older (2100).
    routing = _fire(runner, frame, _verdict(1, entry=2060.0))
    assert routing["order_block"]["setup_id"] == older.setup_id
    binding, _rec = _binding(runner)
    assert binding["bound"][0]["rule"] == "nearest"


def test_bound_tie_falls_back_to_oldest_rule(tmp_path):
    runner, frame, older, newer = _two_instance_runner(tmp_path)
    # 2050 is equidistant: the tie resolves to the oldest and
    # the record names the rule that actually decided it.
    routing = _fire(runner, frame, _verdict(1, entry=2050.0))
    assert routing["order_block"]["setup_id"] == older.setup_id
    binding, _rec = _binding(runner)
    assert binding["bound"][0]["rule"] == "oldest"


def test_one_record_per_fire_across_playbooks(tmp_path):
    frame = _flat_frame(2000.0)
    runner = _basic_runner(
        tmp_path, frame,
        execute_playbooks=("order_block", "fvg_midline"))
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    _sweep(runner, frame, name="order_block", direction=1,
           entry=2000.0, stop=1990.0, validity_bars=200)
    _sweep(runner, frame, name="fvg_midline", direction=1,
           entry=2000.0, stop=1990.0, validity_bars=200)
    runner._get_playbooks = lambda: [
        _StubPB("order_block", _sig(index=0)),
        _StubPB("fvg_midline", _sig(index=0))]
    routing = _fire(runner, frame, _verdict(1, entry=2000.0))
    assert set(routing) == {"order_block", "fvg_midline"}
    binding, _rec = _binding(runner)
    assert binding["decision"] == "bound"
    assert {b["playbook"] for b in binding["bound"]} == {
        "order_block", "fvg_midline"}


# --------------------------------------------------- not-bound taxonomy

def test_no_armed_instances(tmp_path):
    frame = _flat_frame(2000.0)
    runner = _basic_runner(tmp_path, frame)
    runner._get_playbooks = lambda: [
        _StubPB("order_block", _sig(index=0))]
    routing = _fire(runner, frame, _verdict(1, entry=2000.0))
    assert routing == {}
    binding, rec = _binding(runner)
    assert rec["reason_code"] == "path.binding_no_armed_instances"
    assert binding["reason"] == "no_armed_instances"
    assert binding["instances"] == []


def test_all_expired(tmp_path):
    frame_a = _flat_frame(2000.0, end="2026-01-05 00:00")
    runner = _basic_runner(tmp_path, frame_a,
                           execute_playbooks=("order_block",))
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    _sweep(runner, frame_a, name="order_block", direction=1,
           entry=2000.0, stop=1990.0, validity_bars=1)
    setup = runner._setup_store.active()[0]
    # Fire three bars after the touch: the instance's validity
    # (1 bar) has lapsed, though upkeep has not run — the
    # selection's own expiry check is what excludes it.
    frame_b = _flat_frame(2000.0, end="2026-01-05 03:00")
    runner._last_tick = {
        "mid": 2000.0, "bid": 1999.9, "ask": 2000.1,
        "spread": 0.2,
        "time": frame_b.index[-1] + pd.Timedelta(minutes=90)}
    runner._get_playbooks = lambda: [
        _StubPB("order_block", _sig(index=0))]
    routing = _fire(runner, frame_b, _verdict(1, entry=2000.0))
    assert routing == {}
    binding, rec = _binding(runner)
    assert rec["reason_code"] == "path.binding_all_expired"
    assert binding["reason"] == "all_expired"
    assert binding["instances"] == [{
        "setup_id": setup.setup_id, "playbook": "order_block",
        "direction": 1, "expiry_ts": setup.expiry_ts,
        "disposition": "expired"}]


def test_shadow_only(tmp_path):
    frame = _flat_frame(2000.0)
    runner = _basic_runner(tmp_path, frame,
                           execute_playbooks=("order_block",))
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    _sweep(runner, frame, name="order_block", direction=1,
           entry=2000.0, stop=1990.0, validity_bars=200,
           shadow=True)
    setup = runner._setup_store.active()[0]
    assert setup.shadow is True
    runner._get_playbooks = lambda: [
        _StubPB("order_block", _sig(index=0))]
    routing = _fire(runner, frame, _verdict(1, entry=2000.0))
    # Shadow is preserved, never routed — unchanged behavior.
    assert routing == {}
    assert runner._setup_store.get(setup.setup_id).stage == \
        slc.STAGE_ARMED
    binding, rec = _binding(runner)
    assert rec["reason_code"] == "path.binding_shadow_only"
    assert binding["reason"] == "shadow_only"
    assert binding["instances"][0]["disposition"] == "shadow"


# ------------------------------------------------------ reach discipline

def test_no_record_when_binder_not_reached(tmp_path):
    frame = _flat_frame(2000.0)
    runner = _basic_runner(tmp_path, frame,
                           execute_playbooks=("order_block",))
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    _sweep(runner, frame, name="order_block", direction=1,
           entry=2000.0, stop=1990.0, validity_bars=200)
    runner._get_playbooks = lambda: [
        _StubPB("order_block", _sig(index=0))]
    stats = {"signals": 0, "gated": 0}
    # No lifecycle_routing: the closed-bar path never binds,
    # and journals no binding record.
    runner._run_signal_path(frame, len(frame) - 1, [], 1.0,
                            _verdict(1, entry=2000.0), stats)
    assert _binding_records(runner) == []
    # Storeless runner with routing requested: no store, no
    # record, no crash.
    runner._setup_store = None
    routing = _fire(runner, frame, _verdict(1, entry=2000.0))
    assert routing == {}
    assert _binding_records(runner) == []
