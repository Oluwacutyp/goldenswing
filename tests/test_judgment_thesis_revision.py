"""Judgment core S2 REVISION + demo activation (owner approval
and deployment ruling, 2026-11-04 — missed-move workstream 1).

The Nov 3 episode: the core created a bearish thesis at 13:00Z
(winner_p 0.6536, plan seeded from an fvg_limit candidate,
invalidation price 3944.42), maintained it at 13:30Z (0.7568) —
then journaled nothing thesis-wise ever again while price ran
through its own invalidation price to 4046.  The keeper never
saw price, erosion short of a named trigger was invisible, and
no version of the thesis was ever revised.

This file pins the fix:
- weakening is a journaled REVISION (new version, exact
  evidence), distinct from structural invalidation;
- the thesis's stated invalidation terms — named triggers AND
  its plan invalidation price — are evaluated as data;
- replacement preserves the version chain (supersedes pointer);
- children adopted under a live thesis are journaled as
  orphaned when it closes, and the shared order boundary cuts
  a mid-flight orphan while letting a live child through to a
  (fake) broker outcome — the full trace, on journal records.

The episode fixture is built from the journal's OWN records:
the recorded assessment inputs of 2026-11-03 13:00Z / 13:30Z
reproduce the journaled probabilities exactly through
assess_market (asserted), the fvg_limit plan values are the
journal's, and the invalidating price (3960.977) is the
journal's 13:00-bar close from its forecast records.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from goldenswing.execution.judgment_thesis import (
    ThesisConfig,
    ThesisKeeper,
    assess_market,
)
from goldenswing.execution.mt5_runner import MT5Runner
from goldenswing.golden.detector import GoldenSignal
from test_mt5_order_selection import (
    FakeExecutor,
    FakeFeed,
    _frames,
    _runner as _base_runner,
    _verdict,
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

# ─── The episode's recorded inputs (journal, 2026-11-03) ─────
EP_CTX = {
    "regime": "down/clean_trend/normal/distribution/overlap",
    "posture": "trend_day", "day_type": "neutral",
    "adr_pct": 119.3, "adr_state": "extended", "smt": None,
}
EP_FLOW_1300 = {"evaluated_long": 0, "evaluated_short": 2,
                "qualified_long": 0, "qualified_short": 1}
EP_FLOW_1330 = {"evaluated_long": 0, "evaluated_short": 6,
                "qualified_long": 0, "qualified_short": 1}
EP_CANDIDATE = {
    "direction": -1, "entry": 3930.786,
    "stop": 3944.4220892857143, "targets": [3910.3318660714285],
    "playbook": "fvg_limit", "intent": "limit",
}
EP_SPIKE_CLOSE = 3960.977  # the journal's 13:00-bar close


def _ts(day: int, hour: int, minute: int = 0) -> float:
    return datetime(2026, 11, day, hour, minute,
                    tzinfo=timezone.utc).timestamp()


def _kinds(events) -> list:
    return [e["event"] for e in events]


# ─── Weakening: revision, not invalidation ───────────────────

def test_weakening_is_a_journaled_revision_and_thesis_stands():
    keeper = ThesisKeeper()
    up = assess_market(context=UP_TREND, flow={})
    cand = {"direction": 1, "entry": 3900.0, "stop": 3890.0,
            "targets": [3920.0], "playbook": "qm_m5"}
    events = keeper.update(up, ts=1000.0, candidate=cand,
                           playbook="qm_m5")
    assert _kinds(events) == ["thesis_created",
                              "thesis_child_registered"]
    peak = keeper.active["peak_p"]

    rng = assess_market(context=RANGE_CTX, flow={})
    events = keeper.update(rng, ts=1100.0)
    assert _kinds(events) == ["thesis_weakened"]
    ev = events[0]
    assert ev["thesis_id"] == "thesis-1"
    assert ev["version"] == 2
    assert ev["supersedes_version"] == 1
    # the exact evidence rides the record: before/after probs
    assert ev["evidence"]["p_after"] < ev["evidence"]["p_before"]
    assert ev["evidence"]["peak_p"] == peak
    assert ev["evidence"]["scenarios_after"]["range"] > \
        ev["evidence"]["scenarios_before"]["range"]
    assert "eroded" in ev["reason"]
    # the thesis STANDS — weakened, still parenting its child
    assert keeper.active is not None
    assert keeper.active["state"] == "weakened"
    assert keeper.active["version"] == 2
    gate = keeper.gate_check(playbook="qm_m5", direction=1,
                             entry=3900.0, atr=10.0, ts=1150.0)
    assert gate["verdict"] == "child"
    assert gate["state"] == "weakened"
    # a further eroded read does not re-journal the weakening
    assert keeper.update(rng, ts=1200.0) == []
    # recovery to the peak band clears the state, on the record
    events = keeper.update(up, ts=1300.0 + 3600.0)
    assert keeper.active["state"] == "intact"
    kinds = [r["kind"] for r in keeper.active["revisions"]]
    assert kinds == ["weakened", "recovered"]


def test_weakening_threshold_is_stated_data_not_vibes():
    keeper = ThesisKeeper(ThesisConfig(weaken_drop=0.30))
    up = assess_market(context=UP_TREND, flow={})
    keeper.update(up, ts=1000.0)
    rng = assess_market(context=RANGE_CTX, flow={})
    # the range read erodes bullish by ~0.285 < 0.30: quiet
    assert keeper.update(rng, ts=1100.0) == []
    assert keeper.active["state"] == "intact"
    assert keeper.active["version"] == 1


def test_weakened_thesis_still_dies_on_structural_break():
    keeper = ThesisKeeper()
    up = assess_market(context=UP_TREND, flow={})
    keeper.update(up, ts=1000.0)
    rng = assess_market(context=RANGE_CTX, flow={})
    keeper.update(rng, ts=1100.0)
    assert keeper.active["state"] == "weakened"
    down = assess_market(context=DOWN_TREND, flow={})
    events = keeper.update(down, ts=1200.0)
    assert "thesis_invalidated" in _kinds(events)
    assert keeper.history[-1]["status"] == "invalidated"
    # version history survived the close
    assert [r["kind"] for r in
            keeper.history[-1]["revisions"]] == ["weakened"]
    # (the killing read itself still prices bearish, so a fresh
    # scenario-only thesis re-forms — the legacy fall-through;
    # thesis-1 is dead either way and parents nothing more)


# ─── Price: the stated invalidation terms, as data ───────────

def test_price_cross_invalidates_long_and_short():
    short_keeper = ThesisKeeper()
    down = assess_market(context=DOWN_TREND, flow={})
    cand = {"direction": -1, "entry": 3900.0, "stop": 3910.0,
            "targets": [3880.0], "playbook": "fvg_limit"}
    short_keeper.update(down, ts=1000.0, candidate=cand,
                        playbook="fvg_limit", price=3900.0)
    # approaching is not crossing
    assert short_keeper.update(
        down, ts=1100.0, price=3909.99) == []
    events = short_keeper.update(down, ts=1200.0, price=3910.0)
    assert "thesis_invalidated" in _kinds(events)
    inv = next(e for e in events
               if e["event"] == "thesis_invalidated")
    assert "invalidation price" in inv["reason"]
    assert inv["evidence"]["price"] == 3910.0
    assert inv["evidence"]["invalidation_price"] == 3910.0

    long_keeper = ThesisKeeper()
    up = assess_market(context=UP_TREND, flow={})
    lcand = {"direction": 1, "entry": 3900.0, "stop": 3890.0,
             "targets": [3920.0], "playbook": "qm_m5"}
    long_keeper.update(up, ts=1000.0, candidate=lcand,
                       playbook="qm_m5", price=3900.0)
    events = long_keeper.update(up, ts=1100.0, price=3889.0)
    assert "thesis_invalidated" in _kinds(events)


def test_price_never_fires_without_a_plan_price_or_a_price():
    keeper = ThesisKeeper()
    up = assess_market(context=UP_TREND, flow={})
    keeper.update(up, ts=1000.0)  # scenario-only plan: no price
    assert keeper.update(up, ts=1100.0, price=1.0) == []
    assert keeper.active is not None
    # and with a plan price, an absent price changes nothing
    keeper2 = ThesisKeeper()
    cand = {"direction": 1, "entry": 3900.0, "stop": 3890.0,
            "targets": [3920.0], "playbook": "qm_m5"}
    keeper2.update(up, ts=1000.0, candidate=cand, playbook="qm_m5")
    assert keeper2.update(up, ts=1100.0) == []
    assert keeper2.active is not None


# ─── Replacement: the version chain ──────────────────────────

def test_replacement_preserves_lineage_and_orphans_children():
    # A decisive rival that wins WITHOUT firing a named
    # invalidation trigger replaces the thesis (a regime flip
    # to the opposite direction would instead fire the bearish
    # thesis's own "regime direction turns up" trigger — that
    # path is invalidation, tested elsewhere).  Here reversal
    # wins on exhaustion + manipulation + SMT divergence while
    # the regime label still reads down.
    keeper = ThesisKeeper(ThesisConfig(min_margin=0.05))
    down = assess_market(
        context=DOWN_TREND,
        flow={"evaluated_long": 3, "evaluated_short": 0,
              "qualified_long": 1, "qualified_short": 0})
    assert down["winner"] == "bearish"
    cand = {"direction": -1, "entry": 3900.0, "stop": 3910.0,
            "targets": [3880.0], "playbook": "fvg_limit"}
    keeper.update(down, ts=1000.0, candidate=cand,
                  playbook="fvg_limit")
    rival = assess_market(
        context={"regime": "down/unclear/spike/manipulation/"
                           "london",
                 "posture": "trend", "day_type": "trend",
                 "adr_pct": 95.0, "adr_state": "exhausted",
                 "smt": "smt_divergence_bearish"}, flow={})
    assert rival["winner"] == "reversal"
    events = keeper.update(rival, ts=1100.0)
    kinds = _kinds(events)
    # replacement, orphans journaled, and the new thesis
    # created in the same cycle
    assert kinds == ["thesis_superseded", "thesis_orphaned",
                     "thesis_created"]
    created = events[-1]
    assert created["supersedes"] == "thesis-1"
    assert keeper.active["id"] == "thesis-2"
    assert keeper.active["supersedes"] == "thesis-1"
    assert keeper.active["version"] == 1
    old = keeper.history[-1]
    assert old["id"] == "thesis-1"
    assert old["status"] == "superseded"
    orphan = next(e for e in events
                  if e["event"] == "thesis_orphaned")
    assert orphan["children"][0]["playbook"] == "fvg_limit"
    assert orphan["close_status"] == "superseded"
    # the dead thesis's child is cut at the gate; the new
    # (reversal, long) thesis's own direction flows
    gate = keeper.gate_check(playbook="fvg_limit", direction=-1,
                             entry=3900.0, atr=10.0, ts=1200.0)
    assert gate["verdict"] == "orphaned"
    assert gate["close_status"] == "superseded"
    gate = keeper.gate_check(playbook="qm_m5", direction=1,
                             entry=3950.0, atr=10.0, ts=1200.0)
    assert gate["verdict"] == "child"
    assert gate["thesis_id"] == "thesis-2"


def test_orphan_window_lapses_and_gate_abstains():
    keeper = ThesisKeeper()
    down = assess_market(context=DOWN_TREND, flow={})
    cand = {"direction": -1, "entry": 3900.0, "stop": 3910.0,
            "targets": [3880.0], "playbook": "fvg_limit"}
    keeper.update(down, ts=1000.0, candidate=cand,
                  playbook="fvg_limit", price=3900.0)
    keeper.update(down, ts=1100.0, price=3911.0)  # invalidated
    gate = keeper.gate_check(playbook="fvg_limit", direction=-1,
                             entry=3900.0, atr=10.0, ts=1200.0)
    assert gate["verdict"] == "orphaned"
    # an hour later the armed window has passed: a re-emitted
    # setup is the desk's own fresh candidate again
    gate = keeper.gate_check(playbook="fvg_limit", direction=-1,
                             entry=3900.0, atr=10.0,
                             ts=1100.0 + 3601.0)
    # no longer the cut — it falls through to the live thesis's
    # alignment (the re-formed thesis-2), never "orphaned"
    assert gate["verdict"] == "child"
    assert gate["thesis_id"] == "thesis-2"
    # a different level was never this thesis's child: it is
    # unregistered flow aligned with the re-formed live thesis
    gate = keeper.gate_check(playbook="fvg_limit", direction=-1,
                             entry=3800.0, atr=10.0, ts=1200.0)
    assert gate["verdict"] == "child"
    assert gate["registered"] is False
    assert gate["thesis_id"] == "thesis-2"


# ─── Happy path: the regression pin ──────────────────────────

def test_happy_path_event_sequence_unchanged():
    keeper = ThesisKeeper(ThesisConfig(heartbeat_seconds=60.0))
    up = assess_market(context=UP_TREND, flow={})
    assert _kinds(keeper.update(up, ts=1000.0)) == \
        ["thesis_created"]
    assert keeper.update(up, ts=1030.0) == []
    assert _kinds(keeper.update(up, ts=1200.0)) == \
        ["thesis_maintained"]
    events = keeper.update(up, ts=1000.0 + 25 * 3600.0)
    # expiry, then the still-directional read re-forms a fresh
    # thesis in the same cycle (the legacy fall-through) — the
    # expired thesis itself shows no revision machinery at all
    assert _kinds(events) == ["thesis_expired", "thesis_created"]
    thesis = keeper.history[-1]
    assert thesis["status"] == "expired"
    assert thesis["version"] == 1
    assert thesis["revisions"] == []
    assert thesis["children"] == []


# ─── The episode fixture (journal records, 2026-11-03) ───────

def test_episode_fixture_bearish_thesis_revised_on_its_own_terms():
    # Fidelity: the journal's recorded inputs reproduce its
    # journaled assessments exactly.
    a1 = assess_market(context=EP_CTX, flow=EP_FLOW_1300)
    assert a1["winner"] == "bearish"
    assert a1["winner_p"] == 0.6536
    a2 = assess_market(context=EP_CTX, flow=EP_FLOW_1330)
    assert a2["winner_p"] == 0.7568

    keeper = ThesisKeeper()
    t0 = _ts(3, 13, 0)
    events = keeper.update(a1, ts=t0, candidate=EP_CANDIDATE,
                            playbook="fvg_limit", price=3930.786)
    assert _kinds(events) == ["thesis_created",
                              "thesis_child_registered"]
    thesis = keeper.active
    assert thesis["plan"]["basis"] == \
        "aligned candidate fvg_limit"
    assert thesis["plan"]["invalidation_price"] == \
        3944.4220892857143
    gate = keeper.gate_check(playbook="fvg_limit", direction=-1,
                             entry=3930.786, atr=14.0, ts=t0)
    assert gate["verdict"] == "child"

    # 13:30Z: maintained, exactly as the journal recorded.
    events = keeper.update(a2, ts=_ts(3, 13, 30), price=3932.0)
    assert _kinds(events) == ["thesis_maintained"]
    assert events[0]["winner_p"] == 0.7568

    # 14:00Z: the 13:00 bar closes at 3960.977 — through the
    # thesis's own stated invalidation price (3944.42), while
    # the regime read on record has NOT flipped.  The live core
    # journaled nothing here; the revised keeper must revise.
    events = keeper.update(a2, ts=_ts(3, 14, 0),
                           price=EP_SPIKE_CLOSE)
    assert _kinds(events) == ["thesis_invalidated",
                              "thesis_orphaned",
                              "thesis_created"]
    inv = events[0]
    assert "invalidation price" in inv["reason"]
    assert "3944.42" in inv["reason"]
    assert inv["evidence"]["price"] == EP_SPIKE_CLOSE
    assert inv["evidence"]["invalidation_price"] == \
        3944.4220892857143
    orphan = events[1]
    assert orphan["children"][0]["playbook"] == "fvg_limit"
    assert orphan["children"][0]["entry"] == 3930.786
    dead = keeper.history[-1]
    assert dead["id"] == "thesis-1"
    assert dead["status"] == "invalidated"
    # The killing read still prices bearish, so a fresh thesis
    # re-forms (legacy fall-through) — but it inherits NOTHING
    # from the dead one: scenario-only plan, no invalidation
    # price, no children, no supersedes link.
    assert keeper.active["id"] == "thesis-2"
    assert keeper.active["plan"]["invalidation_price"] is None
    assert keeper.active["children"] == []
    assert keeper.active["supersedes"] is None

    # The dead thesis generates no actionable child afterwards:
    # its registered fvg child is cut at the gate — the orphan
    # match outranks mere alignment with the re-formed thesis.
    gate = keeper.gate_check(playbook="fvg_limit", direction=-1,
                             entry=3930.786, atr=14.0,
                             ts=_ts(3, 14, 5))
    assert gate["verdict"] == "orphaned"
    assert gate["close_status"] == "invalidated"
    assert gate["thesis_id"] == "thesis-1"

    # Nov 4 06:00Z: the journal's shadow state read regime flat
    # / structure compression.  Even the re-formed bearish
    # thesis has collapsed by now — the flat read WEAKENS it on
    # the record — and thesis-1's history is untouched: the
    # order_block short the desk produced that bar is nobody's
    # registered child, and never thesis-1's.
    flat = assess_market(
        context={"regime": "flat/compression/normal/"
                           "distribution/ny",
                 "posture": "spike_stand_aside",
                 "day_type": "neutral", "adr_pct": 150.0,
                 "adr_state": "extended", "smt": None}, flow={})
    assert flat["winner"] == "range"
    events = keeper.update(flat, ts=_ts(4, 6, 0), price=3997.0)
    assert _kinds(events) == ["thesis_weakened"]
    assert keeper.active["id"] == "thesis-2"
    assert keeper.active["state"] == "weakened"
    assert len(keeper.history[0]["children"]) == 1
    gate = keeper.gate_check(playbook="order_block", direction=-1,
                             entry=3972.506, atr=14.0,
                             ts=_ts(4, 6, 0))
    assert gate["registered"] is False
    assert gate["thesis_id"] == "thesis-2"


# ─── Runner traces: thesis -> child -> order boundary ────────

def _thesis_runner(tmp_path, **cfg_kw):
    runner = _base_runner(tmp_path, **cfg_kw)
    runner._decision_context = lambda extra=None: dict(EP_CTX)
    return runner


def _set_tick(runner, price: float) -> None:
    import pandas as pd

    runner._last_tick = {
        "time": pd.Timestamp("2026-11-04 12:00",
                             tz="UTC").to_pydatetime(),
        "mid": price, "bid": price, "ask": price + 0.3,
        "tick_volume": 1, "real_volume": 0, "spread": 30}


def _journal_records(tmp_path) -> list:
    path = tmp_path / "mt5_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)
            for line in path.read_text().splitlines() if line]


def _thesis_events(tmp_path) -> list:
    return [r for r in _journal_records(tmp_path)
            if r.get("type") == "judgment_thesis"]


def _child_sig(**kw) -> GoldenSignal:
    base = dict(index=29, timestamp=None, direction=-1,
                score=80.0, kind="fvg_limit", factors=[],
                entry_zone=(3925.0, 3936.0), entry_ref=3930.786,
                invalidation=3944.42, targets=[3910.33], atr=14.0)
    base.update(kw)
    return GoldenSignal(**base)


def _child_verdict(**kw) -> dict:
    base = _verdict(direction=-1, bias_d1=-1, bias_h4=-1,
                     zone={"bottom": 3925.0, "top": 3936.0,
                           "kind": "resistance",
                           "level": 3930.786},
                     entry=3930.786, stop=3944.42,
                     target=3910.33, risk=13.63,
                     trigger="FVG_bear")
    base.update(kw)
    return base


def _create_thesis_via_cycle(runner) -> None:
    _set_tick(runner, 3930.786)
    runner._judgment_thesis_cycle(
        candidate=dict(EP_CANDIDATE),
        decision={"result": "qualified"},
        playbook="fvg_limit", ts="2026-11-03 13:00:00+00:00")


def test_trace_child_flows_to_fake_broker_outcome(tmp_path):
    runner = _thesis_runner(tmp_path)
    _create_thesis_via_cycle(runner)
    events = _thesis_events(tmp_path)
    kinds = [r["event"] for r in events]
    assert kinds == ["thesis_created", "thesis_child_registered"]
    child_ev = events[1]
    assert child_ev["thesis_id"] == "thesis-1"
    assert child_ev["child"]["entry"] == 3930.786

    # The child reaches the shared order boundary under its
    # live thesis and flows: market fill at the level, a real
    # (fake) broker outcome — position opened, journal shows it.
    _set_tick(runner, 3930.786)
    stats: dict = {}
    runner._place_live_order(
        SimpleNamespace(name="fvg_limit"), _child_sig(), stats,
        verdict=_child_verdict(), entry_bar=29)
    assert runner.executor.sent_kinds == ["market"]
    positions = runner.executor.positions()
    assert len(positions) == 1
    assert positions[0].direction == -1
    assert not any(k.startswith("thesis_cut") for k in stats)


def test_trace_duplicate_child_submission_still_prevented(
        tmp_path):
    runner = _thesis_runner(tmp_path)
    _create_thesis_via_cycle(runner)
    # Price away from the level: the child rests as a pending.
    _set_tick(runner, 3900.0)
    stats: dict = {}
    runner._place_live_order(
        SimpleNamespace(name="fvg_limit"), _child_sig(), stats,
        verdict=_child_verdict(), entry_bar=29)
    runner._place_live_order(
        SimpleNamespace(name="fvg_limit"), _child_sig(), stats,
        verdict=_child_verdict(), entry_bar=29)
    # The pre-existing duplicate machinery (Step-15 twin guard /
    # one-pending rule) still caps it at ONE submission with
    # the thesis gate active.
    assert len(runner.executor.sent_kinds) == 1
    assert len(runner._pending_orders) == 1


def test_trace_child_cut_mid_flight_by_invalidation(tmp_path):
    runner = _thesis_runner(tmp_path)
    _create_thesis_via_cycle(runner)
    # The thesis's own invalidation price is crossed while the
    # child is still mid-flight (registered, not yet placed):
    # the next cycle revises the thesis on the journal.
    _set_tick(runner, EP_SPIKE_CLOSE)
    runner._judgment_thesis_cycle(
        candidate={"direction": -1, "entry": 3945.5,
                   "stop": 3960.0, "targets": [3930.0],
                   "playbook": "order_block"},
        decision={"result": "qualified"},
        playbook="order_block", ts="2026-11-03 14:00:00+00:00")
    kinds = [r["event"] for r in _thesis_events(tmp_path)]
    assert "thesis_invalidated" in kinds
    assert "thesis_orphaned" in kinds

    # The mid-flight child now reaches the order boundary —
    # and is cut there: no broker submission, the cut is on
    # the journal and the stats, with the parent named.
    _set_tick(runner, 3930.786)
    stats: dict = {}
    runner._place_live_order(
        SimpleNamespace(name="fvg_limit"), _child_sig(), stats,
        verdict=_child_verdict(), entry_bar=29)
    assert runner.executor.sent_kinds == []
    assert runner.executor.positions() == []
    assert stats.get("thesis_cut::invalidated") == 1
    text = (tmp_path / "mt5_journal.jsonl").read_text()
    assert "child setup cut at the order boundary" in text
    assert "thesis-1" in text


def test_activation_flag_off_restores_journal_only(tmp_path):
    runner = _thesis_runner(
        tmp_path, judgment_thesis_active_enabled=False)
    _create_thesis_via_cycle(runner)
    _set_tick(runner, EP_SPIKE_CLOSE)
    runner._judgment_thesis_cycle(
        candidate={"direction": -1, "entry": 3945.5,
                   "stop": 3960.0, "targets": [3930.0],
                   "playbook": "order_block"},
        decision={"result": "qualified"},
        playbook="order_block", ts="2026-11-03 14:00:00+00:00")
    # the lifecycle still journals its revision...
    kinds = [r["event"] for r in _thesis_events(tmp_path)]
    assert "thesis_orphaned" in kinds
    # ...but with the activation gate off, the boundary does
    # not cut: the desk's own path decides, as before the ruling
    _set_tick(runner, 3930.786)
    stats: dict = {}
    runner._place_live_order(
        SimpleNamespace(name="fvg_limit"), _child_sig(), stats,
        verdict=_child_verdict(), entry_bar=29)
    assert runner.executor.sent_kinds == ["market"]
    assert not any(k.startswith("thesis_cut") for k in stats)
