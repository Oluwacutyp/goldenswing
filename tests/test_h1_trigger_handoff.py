"""Workstream 3 (owner approval 2026-11-04): H1 TRIGGER HANDOFF.

The door between the scheduled scans and the streaming trigger
path.  Before WS3, the 2026-11-04 06:00:00Z H1 close scan
emitted the impulse bar's only candidate — order_block SHORT,
entry 3972.506, stop 3999.328, score 60 — as
``observed / not_attempted`` (path.trigger_not_fired) and the
setup then had no traceable chain: discovery and execution
lived in two rooms with no door.

What this file pins (flag ``h1_trigger_handoff_enabled``,
default ON):

- the episode fixture, journal-exact: the scan candidate is
  PRESERVED as a setup instance (stable identity, thesis link
  field carried), the scan record carries the instance's
  registered disposition with the legacy disposition journaled
  alongside, and the journal's subsequent evidence (the 06:00
  H1 close at 4027.01, beyond the invalidation; NO h1_trigger
  fire exists for this setup in the journal window) ends the
  instance in a named terminal state — invalidated — with
  every stage journaled;
- thesis linkage through the WS1 child registration: an active
  thesis's adopted child stamps the instance at birth, the
  adoption event is journaled exactly once, and every setup
  event carries the parent thesis id;
- dedupe: a re-scan of the same setup lands on the same
  instance;
- trigger binding: a fire routes the CORRESPONDING armed
  instance (entry-zone bracket, then nearest entry, then
  oldest), not merely the oldest;
- a fired trigger cannot vanish: a raising evaluation lands
  as journaled lifecycle transitions and the instance stays
  live (retry-safe re-arm);
- shadow (§G) preservation: the instance is preserved, its
  scan disposition vocabulary is unchanged, a same-direction
  fire journals a trigger_observed disposition WITHOUT
  routing it, and upkeep ends it journaled;
- flag OFF: the scan evaluation record is the legacy record,
  byte for byte in its stable fields.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from goldenswing.execution import setup_lifecycle as slc
from goldenswing.execution.judgment_thesis import (
    ThesisConfig,
    ThesisKeeper,
    assess_market,
)
from test_p3_trader import _FakeScore, _StubPB, _frame, _runner, _sig

EPISODE_SETUP_ID = ("order_block:2026-11-04 05:00:00+00:00:"
                    "-1:3972.51")


# ------------------------------------------------------------ helpers

def _evals(runner, playbook=None, kind=None):
    out = [r for r in runner.journal.records()
           if r.get("type") == "evaluation"]
    if playbook is not None:
        out = [r for r in out if r.get("playbook") == playbook]
    if kind is not None:
        out = [r for r in out if r.get("kind") == kind]
    return out


def _setup_events(runner, playbook=None):
    return _evals(runner, playbook, kind="setup_event")


def _scan_records(runner, playbook):
    return [r for r in _evals(runner, playbook,
                              kind="playbook_eval")
            if r["outcome"] == "candidate_emitted"]


def _episode_frame():
    """H1 bars as held at the 2026-11-04 06:00:00Z close scan:
    the last (evaluated) bar is the 05:00 impulse bar."""
    idx = pd.date_range(end="2026-11-04 05:00:00", periods=35,
                        freq="1h", tz="UTC")
    rows = [(3950.0, 3952.0, 3948.0, 3951.0)] * 34 + [
        (3951.0, 3997.0, 3949.0, 3996.93)]
    return pd.DataFrame(rows, index=idx,
                        columns=["open", "high", "low", "close"])


def _episode_runner(tmp_path, **cfg_kw):
    runner = _runner(tmp_path, **cfg_kw)
    frame = _episode_frame()
    runner._stack = SimpleNamespace(
        frames={"H1": frame,
                "H4": _frame([(3950.0, 3952.0, 3948.0, 3951.0)]
                             * 40, freq="4h")})
    runner._live_setup = {"bias_d1": -1, "bias_h4": -1,
                          "regime": "trend",
                          "h4_context": "downtrend structure",
                          "size_mult": 1.0}
    runner._last_tick = {"mid": 3996.0, "bid": 3995.9,
                         "ask": 3996.1, "spread": 0.2,
                         "time": pd.Timestamp(
                             "2026-11-04 06:00:00", tz="UTC")}
    return runner, frame


def _episode_sig(frame):
    sig = _sig(direction=-1, entry=3972.506, stop=3999.328,
               index=len(frame) - 1)
    sig.score = 60.0
    sig.targets = [3924.985866071428]
    return sig


def _episode_spec(sig):
    return SimpleNamespace(
        name="order_block", route="h1_roster",
        playbook_class=lambda: _StubPB("order_block", sig))


def _episode_sweep(runner, frame):
    runner._sweep_eval_one(
        _episode_spec(_episode_sig(frame)), frame,
        clock="h1_close", intended_tf="H1", actual_tf="H1",
        source="live_close", shadow=False)


def _flat_frame(price=2000.0, bars=40):
    idx = pd.date_range("2026-01-05", periods=bars, freq="1h",
                        tz="UTC")
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
           route="h1_roster", clock="h1_close", intended_tf="H1",
           entry_zone=None, validity_bars=None):
    sig = _sig(direction=direction, entry=entry, stop=stop,
               index=len(frame) - 1)
    if entry_zone is not None:
        sig.entry_zone = entry_zone
    spec = SimpleNamespace(
        name=name, route=route,
        playbook_class=lambda: _StubPB(name, sig))
    if validity_bars is not None:
        spec.validity_bars = validity_bars
    runner._sweep_eval_one(
        spec, frame, clock=clock, intended_tf=intended_tf,
        actual_tf=intended_tf, source="live_close", shadow=shadow)
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


# ------------------------------------------------- episode fixture

def test_episode_order_block_preserved_and_terminated(tmp_path):
    """The 2026-11-04 06:00:00Z scan candidate (journal-exact):
    order_block SHORT, entry 3972.506, stop 3999.328, score 60,
    target 3924.985866071428, evaluated on the 05:00 impulse
    bar's close.  The journal for run d70e3ce3 carries NO
    judgment_thesis records in the window (the keeper is
    per-run and no thesis was active) — the instance's thesis
    link is therefore honestly empty, with the field present
    on every record.  No h1_trigger fire for this setup exists
    in the journal's subsequent window either; the honest
    terminal is the close-rule invalidation when the 06:00 H1
    bar closed 4027.01, beyond the 3999.328 invalidation."""
    runner, frame = _episode_runner(tmp_path)
    _episode_sweep(runner, frame)

    store = runner._setup_store
    active = store.active()
    assert len(active) == 1
    s = active[0]
    # Identity + facts, exactly as journaled.
    assert s.setup_id == EPISODE_SETUP_ID
    assert s.parent_setup_id == EPISODE_SETUP_ID
    assert s.playbook == "order_block" and s.direction == -1
    assert s.entry_ref == 3972.506
    assert s.invalidation == 3999.328
    assert s.targets == [3924.985866071428]
    assert s.score == 60.0
    assert s.touch_ts == "2026-11-04 05:00:00+00:00"
    assert s.expiry_ts == "2026-11-04 11:00:00+00:00"
    assert s.stage == slc.STAGE_ARMED
    assert s.shadow is False
    # No live thesis adopted this candidate (the journal shows
    # none for the run) — the link is empty, not invented.
    assert s.thesis_id is None

    # Every lifecycle link journaled with the setup id and the
    # thesis-link field present.
    events = _setup_events(runner, "order_block")
    assert [r["trigger_stage"] for r in events] == [
        slc.STAGE_DETECTED, slc.STAGE_ARMED]
    for r in events:
        ev = r["evidence"]
        assert ev["setup_id"] == EPISODE_SETUP_ID
        assert "thesis_id" in ev and ev["thesis_id"] is None
        assert ev["shadow"] is False

    # The scan record carries the instance's real disposition;
    # the legacy observed disposition rides alongside.
    rec = _scan_records(runner, "order_block")[-1]
    assert rec["reason_code"] == "path.setup_registered"
    d = rec["decision"]
    assert d["result"] == "registered"
    assert d["order_result"] == "pending_trigger"
    assert d["execution_eligible"] is False
    assert d["setup_id"] == EPISODE_SETUP_ID
    assert d["lifecycle_stage"] == slc.STAGE_ARMED
    assert d["thesis_id"] is None
    assert d["legacy_disposition"] == {
        "result": "observed", "order_result": "not_attempted",
        "reason_code": "path.trigger_not_fired"}
    assert rec["assessment"]["verdict"] == "watch"
    assert rec["evidence"]["setup_id"] == EPISODE_SETUP_ID
    assert rec["evidence"]["thesis_id"] is None
    assert "registered as setup" in \
        d["competing_scenarios"][0]["status"]

    # No order authority was exercised by the scan.
    assert not [r for r in _setup_events(runner, "order_block")
                if r["trigger_stage"] in (slc.STAGE_TRIGGER_FIRED,
                                          slc.STAGE_SUBMITTED,
                                          slc.STAGE_FILLED)]

    # Subsequent journal evidence: the 06:00 H1 bar closed
    # 4027.01 — beyond the short's invalidation.  Upkeep ends
    # the instance in the named terminal state, journaled with
    # the journal's own wording.
    idx2 = pd.date_range(end="2026-11-04 06:00:00", periods=3,
                         freq="1h", tz="UTC")
    h1b = pd.DataFrame(
        [(3996.0, 3998.0, 3994.0, 3997.0),
         (3997.0, 4010.0, 3995.0, 4008.0),
         (4008.0, 4028.0, 4005.0, 4027.01)],
        index=idx2, columns=["open", "high", "low", "close"])
    runner._upkeep_setups(h1b)
    s = store.get(EPISODE_SETUP_ID)
    assert s.stage == slc.STAGE_INVALIDATED
    assert [h["stage"] for h in s.history] == [
        slc.STAGE_DETECTED, slc.STAGE_ARMED, slc.STAGE_INVALIDATED]
    term = [r for r in _setup_events(runner, "order_block")
            if r["trigger_stage"] == slc.STAGE_INVALIDATED]
    assert len(term) == 1
    assert term[0]["reason"] == (
        "close rule: H1 close 4027.01 beyond invalidation "
        "3999.33")
    assert term[0]["evidence"]["setup_id"] == EPISODE_SETUP_ID
    assert store.active() == []


# ------------------------------------------------------- thesis link

_DOWN_TREND = {
    "regime": "down/clean_trend/normal/manipulation/london",
    "posture": "trend", "day_type": "trend", "adr_pct": 35.0,
    "adr_state": "open", "smt": None,
}


def test_scan_instance_stamped_with_parent_thesis(tmp_path):
    """WS1 linkage: with a live bearish thesis on the keeper,
    the scan's order_block candidate is adopted as its child
    through the SAME registration the judgment cycle uses; the
    instance is stamped at birth, the adoption event is
    journaled exactly once, and every setup event + the scan
    record carry the parent thesis id."""
    keeper = ThesisKeeper(ThesisConfig())
    keeper.update(assess_market(context=_DOWN_TREND, flow={}),
                  ts=1762300000.0, candidate=None,
                  playbook="seed", price=3972.0)
    assert keeper.active is not None
    thesis_id = keeper.active["id"]

    runner, frame = _episode_runner(tmp_path)
    runner._thesis_keeper = keeper
    _episode_sweep(runner, frame)

    s = runner._setup_store.active()[0]
    assert s.thesis_id == thesis_id
    assert s.thesis_version == 1
    # The child identity is the WS1 identity: playbook,
    # direction, entry, stop.
    children = keeper.active["children"]
    assert len(children) == 1
    child = children[0]
    assert child["playbook"] == "order_block"
    assert child["direction"] == -1
    assert child["entry"] == 3972.506
    assert child["stop"] == 3999.328
    assert child["thesis_id"] == thesis_id
    # Adoption journaled exactly once (the cycle inside the
    # scan record found the child already registered).
    adopted = [r for r in runner.journal.records()
               if r.get("type") == "judgment_thesis"
               and r.get("event") == "thesis_child_registered"]
    assert len(adopted) == 1
    assert adopted[0]["thesis_id"] == thesis_id
    # The whole chain carries the link.
    for r in _setup_events(runner, "order_block"):
        assert r["evidence"]["thesis_id"] == thesis_id
    rec = _scan_records(runner, "order_block")[-1]
    assert rec["decision"]["thesis_id"] == thesis_id
    assert rec["evidence"]["thesis_id"] == thesis_id


# ------------------------------------------------------------- dedupe

def test_rescan_dedupes_onto_same_instance_flag_on(tmp_path):
    runner, frame = _episode_runner(tmp_path)
    _episode_sweep(runner, frame)
    _episode_sweep(runner, frame)
    active = runner._setup_store.active()
    assert len(active) == 1
    assert active[0].setup_id == EPISODE_SETUP_ID
    # detected + armed journaled once — the re-scan is a
    # refresh of the same instance, not a second birth.
    assert len(_setup_events(runner, "order_block")) == 2
    recs = _scan_records(runner, "order_block")
    assert len(recs) == 2
    assert all(r["decision"]["setup_id"] == EPISODE_SETUP_ID
               for r in recs)
    assert all(r["reason_code"] == "path.setup_registered"
               for r in recs)


# ---------------------------------------------------- trigger binding

def _two_instance_runner(tmp_path):
    """Two armed order_block LONG instances: the OLDER at entry
    2100 (zone 2095–2105), the NEWER at entry 2000 (zone
    1995–2005)."""
    frame1 = _flat_frame(2100.0)
    runner = _basic_runner(
        tmp_path, frame1, execute_playbooks=("order_block",))
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    # The older instance carries a long validity window so it
    # is still armed at the second frame's close (the binding
    # question is correspondence, not expiry).
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


def test_fire_binds_corresponding_instance_not_oldest(tmp_path):
    runner, frame, older, newer = _two_instance_runner(tmp_path)
    routing: dict = {}
    stats = {"signals": 0, "gated": 0}
    # The fire's own entry is 2000 — inside the NEWER
    # instance's zone, far from the older one's.  Pre-handoff
    # binding routed the oldest; WS3 binds the corresponding
    # instance.
    runner._run_signal_path(frame, len(frame) - 1, [], 1.0,
                            _verdict(1, entry=2000.0), stats,
                            lifecycle_routing=routing)
    assert routing["order_block"]["setup_id"] == newer.setup_id
    runner._apply_lifecycle_routing(routing, 1)
    # The routed instance reached a terminal disposition; the
    # next fire binds the remaining armed instance.
    assert runner._setup_store.get(newer.setup_id).stage in (
        slc.STAGE_SUBMITTED, slc.STAGE_FILLED, slc.STAGE_REJECTED)
    routing2: dict = {}
    runner._run_signal_path(frame, len(frame) - 1, [], 1.0,
                            _verdict(1, entry=2000.0), stats,
                            lifecycle_routing=routing2)
    assert routing2["order_block"]["setup_id"] == older.setup_id


def test_select_lifecycle_candidate_rules(tmp_path):
    runner, _frame_h1 = _episode_runner(tmp_path)
    older = SimpleNamespace(entry_ref=2100.0,
                            entry_zone=[2095.0, 2105.0])
    newer = SimpleNamespace(entry_ref=2000.0,
                            entry_zone=[1995.0, 2005.0])
    sel = runner._select_lifecycle_candidate
    # Bracket wins over age.
    assert sel([older, newer], {"entry": 2001.0}) is newer
    # No bracket: nearest entry wins.
    assert sel([older, newer], {"entry": 2060.0}) is older
    # Tie on distance: oldest wins.
    assert sel([older, newer], {"entry": 2050.0}) is older
    # No usable fire price: oldest (the pre-handoff rule).
    assert sel([older, newer], {}) is older
    assert sel([older, newer], {"entry": None}) is older
    assert sel([older, newer], {"entry": float("nan")}) is older


# ------------------------------------------- fired trigger never lost

def test_fired_trigger_cannot_vanish_on_exception(tmp_path):
    frame = _flat_frame(2000.0)
    runner = _basic_runner(tmp_path, frame)
    _sweep(runner, frame)
    setup = runner._setup_store.active()[0]
    runner._get_playbooks = lambda: [
        _StubPB("fvg_midline", _sig(index=0))]

    def _boom(*a, **k):
        raise RuntimeError("judgment exploded")

    runner._run_one_playbook = _boom
    routing: dict = {}
    stats = {"signals": 0, "gated": 0}
    runner._run_signal_path(frame, len(frame) - 1, [], 1.0,
                            _verdict(1), stats,
                            lifecycle_routing=routing)
    # The failure is a journaled disposition, not a silence.
    assert routing["fvg_midline"]["status"] == "exception"
    assert routing["fvg_midline"]["setup_id"] == setup.setup_id
    frags = runner._apply_lifecycle_routing(routing, 1)
    assert frags
    s = runner._setup_store.get(setup.setup_id)
    # Retry-safe: the instance is live again (re-armed), with
    # the full fired chain in its history — nothing vanished.
    assert s.stage == slc.STAGE_ARMED
    assert [h["stage"] for h in s.history] == [
        slc.STAGE_DETECTED, slc.STAGE_ARMED,
        slc.STAGE_TRIGGER_FIRED, slc.STAGE_EVALUATED,
        slc.STAGE_ARMED]
    stages = [r["trigger_stage"]
              for r in _setup_events(runner, "fvg_midline")]
    assert stages == [h["stage"] for h in s.history]


# ------------------------------------------------- shadow preservation

def test_shadow_instance_preserved_never_routed(tmp_path):
    frame = _flat_frame(2000.0)
    runner = _basic_runner(tmp_path, frame)
    _sweep(runner, frame, name="range_qm", shadow=True,
           route="range_h4_h1", clock="h4_generation",
           intended_tf="H4")
    active = runner._setup_store.active()
    assert len(active) == 1
    s = active[0]
    assert s.shadow is True and s.stage == slc.STAGE_ARMED

    # The scan record keeps the honest shadow disposition —
    # vocabulary unchanged — but now names the instance.
    rec = _scan_records(runner, "range_qm")[-1]
    assert rec["reason_code"] == "path.shadow_route"
    assert rec["decision"]["result"] == "shadow"
    assert rec["decision"]["order_result"] == "not_attempted"
    assert rec["decision"]["setup_id"] == s.setup_id
    assert rec["evidence"]["setup_id"] == s.setup_id

    # A same-direction fire does NOT route it (no execution
    # intent, ever) — but the trigger-stage disposition is
    # journaled on the instance instead of the setup vanishing.
    runner._get_playbooks = lambda: [
        _StubPB("range_qm", _sig(index=0))]
    routing: dict = {}
    stats = {"signals": 0, "gated": 0}
    runner._run_signal_path(frame, len(frame) - 1, [], 1.0,
                            _verdict(1), stats,
                            lifecycle_routing=routing)
    assert "range_qm" not in routing
    observed = [r for r in _setup_events(runner, "range_qm")
                if r["trigger_stage"] ==
                slc.STAGE_TRIGGER_OBSERVED]
    assert len(observed) == 1
    assert observed[0]["evidence"]["setup_id"] == s.setup_id
    assert observed[0]["evidence"]["shadow"] is True
    assert "never routed" in observed[0]["reason"]
    # Still armed afterwards — observation is not a transition.
    assert runner._setup_store.get(s.setup_id).stage == \
        slc.STAGE_ARMED

    # Upkeep ends it with a journaled terminal state.
    bad = _frame([(1985.0, 1986.0, 1984.0, 1985.0)] * 3,
                 start=str(frame.index[-1]
                           + pd.Timedelta(hours=1)))
    runner._upkeep_setups(bad)
    s2 = runner._setup_store.get(s.setup_id)
    assert s2.stage == slc.STAGE_INVALIDATED
    assert [h["stage"] for h in s2.history] == [
        slc.STAGE_DETECTED, slc.STAGE_ARMED, slc.STAGE_INVALIDATED]


# ------------------------------------------- registration failure

def test_registration_failure_falls_back_to_legacy(tmp_path):
    """Dependency failure is explicit, never a guessed state:
    if the store registration raises, the failure is noted and
    the scan record carries the honest legacy disposition (the
    candidate was NOT preserved, so it must not claim to be
    registered)."""
    runner, frame = _episode_runner(tmp_path)

    def _raise(*a, **k):
        raise OSError("store disk gone")

    runner._setup_store.register = _raise
    _episode_sweep(runner, frame)
    rec = _scan_records(runner, "order_block")[-1]
    assert rec["reason_code"] == "path.trigger_not_fired"
    assert rec["decision"]["result"] == "observed"
    assert "setup_id" not in rec["decision"]
    notes = [r for r in runner.journal.records()
             if r.get("type") == "note"
             and "handoff registration failed"
             in str(r.get("text") or r.get("note") or "")]
    assert notes


# ---------------------------------------------------------- flag off

def test_flag_off_scan_record_is_legacy_byte_identical(tmp_path):
    runner, frame = _episode_runner(
        tmp_path, h1_trigger_handoff_enabled=False)
    _episode_sweep(runner, frame)
    # Registration still happens (the §1 machinery predates the
    # flag); the RECORDS are the legacy ones.
    assert len(runner._setup_store.active()) == 1
    rec = _scan_records(runner, "order_block")[-1]
    assert rec["reason_code"] == "path.trigger_not_fired"
    assert rec["reason"] == (
        "scheduled scan found a candidate on the closing bar; "
        "this scan route holds no order authority and does not "
        "run the h1_trigger streaming evaluation — recorded as "
        "observed, never routed from the scan (see Trigger "
        "decision)")
    d = rec["decision"]
    assert d["result"] == "observed"
    assert d["order_result"] == "not_attempted"
    assert d["execution_eligible"] is False
    assert "setup_id" not in d
    assert "lifecycle_stage" not in d
    assert "legacy_disposition" not in d
    assert "setup_id" not in rec["evidence"]
    assert "observed, not routed" in \
        d["competing_scenarios"][0]["status"]
    # Setup events carry no WS3 keys either.
    for r in _setup_events(runner, "order_block"):
        assert "thesis_id" not in r["evidence"]
        assert "shadow" not in r["evidence"]


# ------------------------------------------------------ store stamps

def test_store_thesis_stamp_shadow_filter_and_old_file(tmp_path):
    path = tmp_path / "lc.json"
    store = slc.SetupLifecycleStore(path)
    s = slc.TrackedSetup(
        setup_id="pb:2026-01-05 00:00:00:1:2000.00",
        parent_setup_id="pb:2026-01-05 00:00:00:1:2000.00",
        playbook="pb", route="h1_roster", direction=1,
        entry_ref=2000.0, invalidation=1990.0)
    assert store.register(s, ts="2026-01-05 00:00:00") is True
    assert store.stamp_thesis(s.setup_id, "thesis-7", 3) is True
    assert store.get(s.setup_id).thesis_id == "thesis-7"
    assert store.get(s.setup_id).thesis_version == 3
    assert store.stamp_thesis("nope", "thesis-7", 3) is False
    # Shadow filter: None = all, True/False select.
    sh = slc.TrackedSetup(
        setup_id="pb2:2026-01-05 00:00:00:1:2000.00",
        parent_setup_id="pb2:2026-01-05 00:00:00:1:2000.00",
        playbook="pb2", route="range_h4_h1", direction=1,
        entry_ref=2000.0, invalidation=1990.0, shadow=True)
    assert store.register(sh, ts="2026-01-05 00:00:00") is True
    assert len(store.active()) == 2
    assert [x.setup_id for x in store.active(shadow=True)] == \
        [sh.setup_id]
    assert [x.setup_id for x in store.active(shadow=False)] == \
        [s.setup_id]
    # A pre-WS3 store file (no thesis/shadow keys) loads with
    # honest defaults.
    raw = json.loads(path.read_text())
    for rec in raw["setups"]:
        rec.pop("thesis_id", None)
        rec.pop("thesis_version", None)
        rec.pop("shadow", None)
    path.write_text(json.dumps(raw))
    again = slc.SetupLifecycleStore(path)
    s2 = again.get(s.setup_id)
    assert s2.thesis_id is None and s2.thesis_version is None
    assert s2.shadow is False
