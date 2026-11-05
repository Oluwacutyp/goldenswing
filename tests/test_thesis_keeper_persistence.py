"""Thesis-keeper persistence across restarts (WS4 episode-replay
residual #1, 2026-11-04).

The replay's first residual: journaled thesis assessments stop
at 16:00Z on Nov 3 and Nov 4 runs journal none, so the
successor thesis was still "active" at the 06:00Z scan only by
accident of re-creation — the ThesisKeeper's state did not
survive engine restarts, and a thesis invalidated before a
restart could wake up active after one.

This file pins the fix:
- keeper state is written on EVERY mutation (create, weaken,
  adopt, invalidate) to an atomic JSON store beside the
  journal — never a shutdown flush;
- a fresh keeper restored from that store is EXACTLY the old
  keeper: same live thesis (version, weakened state, children,
  revision chain), same history, same orphan set, same id
  sequence;
- an invalidated thesis restores as history, never as active,
  and the order boundary keeps cutting its orphans inside the
  window after a restart;
- a torn or semantically incomplete state REFUSES loudly
  (ThesisStateError naming the gap at keeper level; a journaled
  thesis_state_unreadable + a disabled thesis layer at runner
  level) — never a silent fresh-empty keeper;
- the runner journals the restoration (thesis_state_restored)
  with the real hydration timestamps — the record is available
  from when it was written, never backdated, so the replay
  harness's info-at-the-time discipline is untouched.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from goldenswing.execution.judgment_thesis import (
    ThesisKeeper,
    ThesisStateError,
    ThesisStateStore,
    assess_market,
)
from test_judgment_thesis_revision import (
    EP_CANDIDATE,
    EP_CTX,
    EP_FLOW_1300,
    EP_FLOW_1330,
    EP_SPIKE_CLOSE,
    _child_sig,
    _child_verdict,
    _create_thesis_via_cycle,
    _set_tick,
    _thesis_events,
    _thesis_runner,
)

UP_TREND = {
    "regime": "up/clean_trend/normal/distribution/ny",
    "posture": "trend", "day_type": "trend", "adr_pct": 40.0,
    "adr_state": "open", "smt": None,
}
DOWN_TREND = {
    "regime": "down/clean_trend/normal/manipulation/london",
    "posture": "trend", "day_type": "trend", "adr_pct": 35.0,
    "adr_state": "open", "smt": None,
}
RANGE_CTX = {
    "regime": "flat/range/normal/transition/asia",
    "posture": "defensive", "day_type": "range", "adr_pct": 55.0,
    "adr_state": "open", "smt": None,
}


def _ts(day: int, hour: int, minute: int = 0) -> float:
    return datetime(2026, 11, day, hour, minute,
                    tzinfo=timezone.utc).timestamp()


def _kinds(events) -> list:
    return [e["event"] for e in events]


def _bullish_keeper_with_child(store=None) -> ThesisKeeper:
    keeper = ThesisKeeper(store=store)
    up = assess_market(context=UP_TREND, flow={})
    cand = {"direction": 1, "entry": 3900.0, "stop": 3890.0,
            "targets": [3920.0], "playbook": "qm_m5"}
    events = keeper.update(up, ts=1000.0, candidate=cand,
                           playbook="qm_m5")
    assert _kinds(events) == ["thesis_created",
                              "thesis_child_registered"]
    return keeper


# ─── Write-as-it-happens + exact restoration ─────────────────

def test_state_written_on_mutation_and_restores_exactly(tmp_path):
    path = tmp_path / "judgment_thesis_state.json"
    store = ThesisStateStore(path)
    keeper = _bullish_keeper_with_child(store)
    # The creation cycle already wrote the state — no shutdown
    # flush is involved anywhere in this file.
    on_disk = json.loads(path.read_text())
    assert on_disk["kind"] == "judgment_thesis_state"
    assert on_disk["active"]["id"] == "thesis-1"
    assert on_disk["active"]["state"] == "intact"

    # Weaken it: the file moves with the mutation.
    rng = assess_market(context=RANGE_CTX, flow={})
    assert _kinds(keeper.update(rng, ts=1100.0)) == [
        "thesis_weakened"]
    on_disk = json.loads(path.read_text())
    assert on_disk["active"]["state"] == "weakened"
    assert on_disk["active"]["version"] == 2

    # A direct adoption (the WS3 scan path) persists too.
    ev = keeper.adopt_child(
        {"direction": 1, "entry": 3905.0, "stop": 3895.0,
         "targets": [3925.0], "playbook": "fvg_limit"},
        "fvg_limit", ts=1150.0)
    assert ev is not None and ev["event"] == \
        "thesis_child_registered"
    on_disk = json.loads(path.read_text())
    assert len(on_disk["active"]["children"]) == 2

    # Drop the instance (the "restart") and restore.
    restored = ThesisKeeper()
    summary = restored.restore(store.load())
    assert restored.snapshot() == keeper.snapshot()
    assert restored.status() == keeper.status()
    assert summary["active_thesis"]["id"] == "thesis-1"
    assert summary["active_thesis"]["state"] == "weakened"
    assert summary["active_thesis"]["version"] == 2
    assert summary["active_thesis"]["children"] == 2
    assert summary["closed_total"] == 0
    assert summary["theses"] == [
        {"id": "thesis-1", "status": "active"}]

    # The weakened state and the child links are live, not
    # archival: the gate still parents both children as
    # children of the weakened thesis, at its restored version.
    gate = restored.gate_check(playbook="qm_m5", direction=1,
                               entry=3900.0, atr=10.0, ts=1200.0)
    assert gate["verdict"] == "child"
    assert gate["state"] == "weakened"
    assert gate["version"] == 2
    link = restored.child_link(playbook="fvg_limit", direction=1,
                               entry=3905.0, stop=3895.0)
    assert link == {"thesis_id": "thesis-1", "thesis_version": 2}
    kinds = [r["kind"] for r in restored.active["revisions"]]
    assert kinds == ["weakened"]

    # The id sequence survives: when thesis-1 is invalidated,
    # the successor founded in the same update mints thesis-2,
    # never a reused thesis-1.
    down = assess_market(context=DOWN_TREND, flow={})
    events = restored.update(down, ts=1300.0)
    assert _kinds(events) == ["thesis_invalidated",
                              "thesis_orphaned",
                              "thesis_created"]
    assert events[-1]["thesis_id"] == "thesis-2"


def test_invalidated_thesis_never_wakes_active(tmp_path):
    path = tmp_path / "judgment_thesis_state.json"
    store = ThesisStateStore(path)
    keeper = ThesisKeeper(store=store)
    assessment = assess_market(context=EP_CTX,
                               flow=dict(EP_FLOW_1300))
    events = keeper.update(
        assessment, ts=_ts(3, 13), candidate=dict(EP_CANDIDATE),
        playbook="fvg_limit")
    assert _kinds(events) == ["thesis_created",
                              "thesis_child_registered"]
    # The episode's own invalidation: the 13:00-bar close
    # through the thesis's stated invalidation price.  (A
    # range read carries the price fact, so no successor is
    # founded in the same update — the desk stands thesisless,
    # which is what this test restores.)
    flat = assess_market(context=RANGE_CTX, flow={})
    events = keeper.update(
        flat, ts=_ts(3, 15), price=EP_SPIKE_CLOSE)
    assert _kinds(events) == ["thesis_invalidated",
                              "thesis_orphaned"]

    restored = ThesisKeeper()
    summary = restored.restore(store.load())
    # History, not active — the whole point of the fix.
    assert restored.active is None
    assert restored.status() == {"active": False,
                                 "closed_total": 1}
    assert summary["active_thesis"] is None
    assert summary["theses"] == [
        {"id": "thesis-1", "status": "invalidated"}]
    assert summary["orphan_thesis_id"] == "thesis-1"
    assert summary["orphan_close_status"] == "invalidated"
    assert summary["orphans_recognized"] == 1

    # The order boundary still cuts the dead thesis's child
    # inside the orphan window — after the restart — and lets
    # it go once the window has passed.
    gate = restored.gate_check(
        playbook="fvg_limit", direction=-1, entry=3930.786,
        atr=14.0, ts=_ts(3, 15) + 1800.0)
    assert gate["verdict"] == "orphaned"
    assert gate["close_status"] == "invalidated"
    gate = restored.gate_check(
        playbook="fvg_limit", direction=-1, entry=3930.786,
        atr=14.0, ts=_ts(3, 15) + 3601.0)
    assert gate["verdict"] == "independent"


def test_restart_mid_window_episode_shape(tmp_path):
    """The Nov 3 -> Nov 4 shape from the replay residual:
    assessments at 13:00 / 13:30 / 16:00, then the journal goes
    silent (restarts).  With persistence, the 06:00Z scan next
    day meets the SAME thesis — maintained, not re-created —
    with its original creation time driving the TTL."""
    path = tmp_path / "judgment_thesis_state.json"
    store = ThesisStateStore(path)
    keeper = ThesisKeeper(store=store)
    a1300 = assess_market(context=EP_CTX,
                          flow=dict(EP_FLOW_1300))
    events = keeper.update(
        a1300, ts=_ts(3, 13), candidate=dict(EP_CANDIDATE),
        playbook="fvg_limit")
    assert _kinds(events) == ["thesis_created",
                              "thesis_child_registered"]
    a1330 = assess_market(context=EP_CTX,
                          flow=dict(EP_FLOW_1330))
    assert _kinds(keeper.update(a1330, ts=_ts(3, 13, 30))) == [
        "thesis_maintained"]
    assert _kinds(keeper.update(a1330, ts=_ts(3, 16))) == [
        "thesis_maintained"]

    # The restart: a brand-new keeper over the same store.
    restored = ThesisKeeper()
    summary = restored.restore(store.load())
    assert summary["active_thesis"]["id"] == "thesis-1"

    # Nov 4, 06:00Z: the desk's next read maintains thesis-1.
    # No thesis_created — the run does not re-found the thesis
    # it already holds — and the creation time is Nov 3 13:00,
    # so the 24h horizon is measured from the real birth.
    events = restored.update(a1330, ts=_ts(4, 6))
    assert _kinds(events) == ["thesis_maintained"]
    assert restored.active["id"] == "thesis-1"
    assert restored.active["created_ts"] == _ts(3, 13)


# ─── Corruption honesty ──────────────────────────────────────

def test_torn_state_files_refuse_with_named_gaps(tmp_path):
    path = tmp_path / "judgment_thesis_state.json"
    store = ThesisStateStore(path)

    path.write_text('{"kind": "judgment_thesis_state", "ver')
    with pytest.raises(ThesisStateError, match="torn"):
        store.load()

    path.write_text(json.dumps(["not", "a", "state"]))
    with pytest.raises(ThesisStateError, match="not a state"):
        store.load()

    path.write_text(json.dumps({"kind": "something_else",
                                "version": 1}))
    with pytest.raises(ThesisStateError, match="kind"):
        store.load()

    keeper = _bullish_keeper_with_child()
    state = keeper.snapshot()
    del state["history"]
    path.write_text(json.dumps(state))
    with pytest.raises(ThesisStateError, match="history"):
        store.load()


def test_semantically_incomplete_state_refuses(tmp_path):
    keeper = _bullish_keeper_with_child()
    good = keeper.snapshot()

    broken = json.loads(json.dumps(good))
    del broken["active"]["children"]
    with pytest.raises(ThesisStateError, match="children"):
        ThesisKeeper().restore(broken)

    # Close the thesis so history is populated, then lie about
    # the closed status: an "active" record inside history is
    # exactly the wake-up-active corruption, and it refuses.
    down = assess_market(context=DOWN_TREND, flow={})
    keeper.update(down, ts=2000.0)
    closed = keeper.snapshot()
    assert closed["history"][0]["status"] == "invalidated"
    closed["history"][0]["status"] = "active"
    with pytest.raises(ThesisStateError, match="status"):
        ThesisKeeper().restore(closed)

    orphan_broken = json.loads(json.dumps(keeper.snapshot()))
    orphan_broken["orphan_close"] = {}
    with pytest.raises(ThesisStateError, match="orphan_close"):
        ThesisKeeper().restore(orphan_broken)


# ─── Runner level: restore, journal, refuse ──────────────────

def test_runner_restart_restores_thesis_and_journals_it(
        tmp_path):
    runner1 = _thesis_runner(tmp_path)
    _create_thesis_via_cycle(runner1)
    state_path = tmp_path / "judgment_thesis_state.json"
    assert state_path.exists()

    # A new run over the same journal home: the FIRST thesis-
    # layer touch here is the order boundary itself, and it
    # must already see the restored thesis — the child flows
    # under its live parent, exactly as before the restart.
    runner2 = _thesis_runner(tmp_path)
    _set_tick(runner2, 3930.786)
    stats: dict = {}
    runner2._place_live_order(
        SimpleNamespace(name="fvg_limit"), _child_sig(), stats,
        verdict=_child_verdict(), entry_bar=29)
    assert runner2.executor.sent_kinds == ["market"]
    assert not any(k.startswith("thesis_cut") for k in stats)

    events2 = [r for r in _thesis_events(tmp_path)
               if r.get("run_id") == runner2._run_id]
    restored = [r for r in events2
                if r["event"] == "thesis_state_restored"]
    assert len(restored) == 1
    rec = restored[0]
    assert rec["active_thesis"]["id"] == "thesis-1"
    assert rec["active_thesis"]["scenario"] == "bearish"
    assert rec["theses"] == [
        {"id": "thesis-1", "status": "active"}]
    # The restoration record carries its REAL timestamps (the
    # hydration moment), never the thesis's original times —
    # the replay harness admits records by availability, and
    # this one was not available before it was written.
    assert float(rec["ts"]) == rec["cycle_ts"]
    assert rec["cycle_ts"] > _ts(4, 6)

    # The following cycle maintains the restored thesis; it
    # does not re-found it, and the restoration is not
    # journaled twice.
    _create_thesis_via_cycle(runner2)
    events2 = [r for r in _thesis_events(tmp_path)
               if r.get("run_id") == runner2._run_id]
    kinds2 = [r["event"] for r in events2]
    assert kinds2.count("thesis_state_restored") == 1
    assert "thesis_created" not in kinds2
    keeper2 = runner2._thesis_keeper
    assert keeper2 is not None
    assert keeper2.active["id"] == "thesis-1"


def test_runner_without_state_file_starts_fresh_quietly(
        tmp_path):
    runner = _thesis_runner(tmp_path)
    _create_thesis_via_cycle(runner)
    kinds = [r["event"] for r in _thesis_events(tmp_path)]
    assert kinds == ["thesis_created", "thesis_child_registered"]
    # ...and the run LEAVES a state file for its successor.
    assert (tmp_path / "judgment_thesis_state.json").exists()


def test_runner_corrupt_state_disables_layer_loudly(tmp_path):
    (tmp_path / "judgment_thesis_state.json").write_text(
        '{"kind": "judgment_thesis_state", "ver')
    runner = _thesis_runner(tmp_path)
    _create_thesis_via_cycle(runner)
    _create_thesis_via_cycle(runner)
    events = _thesis_events(tmp_path)
    refused = [r for r in events
               if r["event"] == "thesis_state_unreadable"]
    # Journaled exactly once, with the specific gap — and no
    # thesis was founded on guessed state in either cycle.
    assert len(refused) == 1
    assert "torn" in refused[0]["error"]
    assert not any(r["event"] == "thesis_created"
                   for r in events)
    assert getattr(runner, "_thesis_keeper", None) is None

    # Other layers are unaffected: an independent order still
    # reaches the (fake) broker — the refusal is scoped to the
    # thesis layer, exactly as specified.
    _set_tick(runner, 3930.786)
    stats: dict = {}
    runner._place_live_order(
        SimpleNamespace(name="fvg_limit"), _child_sig(), stats,
        verdict=_child_verdict(), entry_bar=29)
    assert runner.executor.sent_kinds == ["market"]
