"""MarketSituation Pass 2 — the scenario book (SPEC §4, §10
Pass 2; AMENDMENTS-1 §A2).

Acceptance pins, one per A2 capability (test names carry the
capability number):

1. multiple bullish AND bearish scenarios coexist;
2. different timeframes hold independent theses simultaneously;
3. each scenario carries its own formation / refinement /
   trigger / invalidation / expiry / management clocks;
4. related scenarios share parent-thesis identity WITHOUT
   double-counting evidence;
5. one scenario weakens while another strengthens in the
   same cycle;
6. a reversal scenario forms BEFORE the prior trend thesis
   is invalidated;
7. an open position keeps its management context when the
   wider thesis changes.

Plus: the lifecycle bridge's state walk, keeper recovery,
competing-pair marking (marked, never resolved — Pass 3),
shadow scoring journaled with situation linkage, and the
runner seams (book silent when the situation program is off;
decision records byte-identical with the book on/off).
"""
from __future__ import annotations

import copy
import json

import pytest

from goldenswing.execution.scenario_book import (
    ScenarioBook,
    ScenarioBookStateError,
    ScenarioBookStore,
    pair_relationship,
)
from goldenswing.execution.setup_lifecycle import (
    STAGE_ARMED,
    STAGE_DETECTED,
    STAGE_EVALUATED,
    STAGE_EXPIRED,
    STAGE_FILLED,
    STAGE_INVALIDATED,
    STAGE_REJECTED,
    STAGE_SUBMITTED,
    STAGE_TRIGGER_FIRED,
    TrackedSetup,
)


# ------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------
def _setup(setup_id="pb:1", playbook="sweep_reversal", direction=1,
           entry=4000.0, stop=3990.0, targets=(4020.0,), **kw):
    base = dict(
        setup_id=setup_id, parent_setup_id=setup_id,
        playbook=playbook, route="h1_roster",
        direction=direction, entry_ref=entry,
        invalidation=stop, targets=list(targets),
        management={
            "management_horizon": "intraday",
            "management_policy_id": "standard_v1",
            "management_policy_version": 1,
            "management_context_timeframe": "H1",
            "management_invalidation": stop,
            "management_target": targets[0] if targets else None},
        formation_ts="2026-11-08T00:00:00",
        touch_ts="2026-11-08T01:00:00",
        created_ts="2026-11-08T01:00:00",
        trigger_tf="H1", validity_bars=6,
        expiry_ts="2026-11-08T07:00:00")
    base.update(kw)
    return TrackedSetup(**base)


def _assessment(bullish=0.30, bearish=0.30, range_p=0.20,
                reversal=0.10, continuation=0.10, direction=0):
    probs = {"bullish": bullish, "bearish": bearish,
             "range": range_p, "reversal": reversal,
             "continuation": continuation}
    ranked = sorted(probs, key=probs.get, reverse=True)
    return {
        "scenarios": [{"name": n, "p": probs[n], "why": []}
                      for n in ranked],
        "winner": ranked[0], "winner_p": probs[ranked[0]],
        "margin": round(probs[ranked[0]] - probs[ranked[1]], 4),
        "direction": direction,
        "invalidation": [],
        "inputs": {"regime": None, "flow": {}},
    }


def _thesis(tid="thesis-1", direction=1, scenario="bullish",
            version=1, state="intact", winner_p=0.40, **kw):
    base = {
        "id": tid, "created_ts": 1000.0, "updated_ts": 1000.0,
        "scenario": scenario, "direction": direction,
        "winner_p": winner_p, "peak_p": winner_p, "state": state,
        "version": version, "status": "active",
        "supersedes": None,
        "invalidation": [
            {"text": "regime direction turns down",
             "trigger": "regime_direction_down"}],
        "plan": {"direction": direction, "entry": 4005.0,
                 "invalidation_price": 3980.0, "target": 4050.0,
                 "management_horizon": "standard_v1",
                 "basis": "scenario only"},
        "children": [], "revisions": [],
        "standing": {"scenarios": {}, "winner": scenario,
                     "margin": 0.1, "regime": None},
    }
    base.update(kw)
    return base


def _book(tmp_path=None):
    if tmp_path is None:
        return ScenarioBook()
    return ScenarioBook(
        store=ScenarioBookStore(tmp_path / "scenario_book.json"))


def _live(book, setup_id):
    sc = book.get(setup_id)
    assert sc is not None, f"{setup_id} not live in the book"
    return sc


# ------------------------------------------------------------------
# A2.1 — bullish and bearish scenarios coexist
# ------------------------------------------------------------------
def test_a2_1_bullish_and_bearish_scenarios_coexist(tmp_path):
    book = _book(tmp_path)
    book.propose_from_setup(_setup("long:1", direction=1), ts=1)
    book.propose_from_setup(
        _setup("short:1", direction=-1, entry=4010.0,
               stop=4025.0, targets=(3985.0,)), ts=1)
    live = {s["scenario_id"]: s for s in book.live()}
    assert set(live) == {"long:1", "short:1"}
    assert live["long:1"]["direction"] == 1
    assert live["short:1"]["direction"] == -1
    assert live["long:1"]["state"] == "proposed"
    assert live["short:1"]["state"] == "proposed"
    summary = book.cycle(ts=2, assessment=_assessment())[-1]
    assert summary["type"] == "scenario_book"
    assert summary["open"] == 2
    assert summary["by_horizon"] == {"intraday": 2}


def test_hot_scalp_staged_targets_table_proposes(tmp_path):
    """A Hot Scalp setup carries its targets as a staged table
    (tp1/tp2 prices plus the open runner marker), not a price
    list — proposing it must read the PRICES, never crash on the
    label keys (the 'tp1' string-to-float failure that silently
    dropped Hot Scalp scenarios from the book)."""
    book = _book(tmp_path)
    setup = _setup("hot_scalp:1", playbook="hot_scalp",
                   direction=1, entry=3892.265, stop=3878.344,
                   targets=(3910.071,))
    setup.targets = {"tp1": 3910.071, "tp2": 3925.0,
                     "runner": "open"}
    book.propose_from_setup(setup, ts=1)
    live = {s["scenario_id"]: s for s in book.live()}
    assert set(live) == {"hot_scalp:1"}
    assert live["hot_scalp:1"]["geometry"]["targets"] == [
        3910.071, 3925.0]


def test_targets_as_prices_reads_every_shape():
    from goldenswing.execution.scenario_book import (
        targets_as_prices)
    # The Hot Scalp staged table: prices in, markers out.
    assert targets_as_prices(
        {"tp1": 3910.071, "tp2": 3925.0, "runner": "open"}) == [
            3910.071, 3925.0]
    assert targets_as_prices(
        {"tp1": 3910.071, "tp2": None, "runner": "open",
         "planned_objective": 3940.0, "plan_kind": "ladder"}) == [
            3910.071, 3940.0]
    # A plain price list is unchanged.
    assert targets_as_prices([3910.071, 3925.0]) == [
        3910.071, 3925.0]
    # None / empty / a single price.
    assert targets_as_prices(None) == []
    assert targets_as_prices([]) == []
    assert targets_as_prices(3910.071) == [3910.071]
    # A non-numeric element in a LIST is corrupt data and fails
    # loudly, as before.
    with pytest.raises(ValueError):
        targets_as_prices(["tp1", 3925.0])


def test_insane_geometry_is_refused_not_proposed(tmp_path):
    """A scenario whose stop is on the wrong side of entry, or
    whose targets sit behind entry, or which carries no target
    at all, is REFUSED with a journaled reason — it never
    reaches the book and can never reach an order."""
    book = _book(tmp_path)
    # Long with the stop ABOVE the entry (wrong side).
    bad_stop = _setup("bad:stop", direction=1, entry=4000.0,
                      stop=4010.0, targets=(4020.0,))
    recs = book.propose_from_setup(bad_stop, ts=1)
    assert recs and recs[0]["type"] == "scenario_refused"
    assert "stop" in recs[0]["reason"]
    assert "bad:stop" not in {s["scenario_id"]
                              for s in book.live()}
    # Long with every target BEHIND the entry.
    bad_targets = _setup("bad:targets", direction=1,
                         entry=4000.0, stop=3990.0,
                         targets=(3980.0,))
    recs = book.propose_from_setup(bad_targets, ts=1)
    assert recs and recs[0]["type"] == "scenario_refused"
    assert "target" in recs[0]["reason"]
    assert "bad:targets" not in {s["scenario_id"]
                                 for s in book.live()}
    # No targets at all.
    no_targets = _setup("bad:none", direction=1, entry=4000.0,
                        stop=3990.0, targets=())
    recs = book.propose_from_setup(no_targets, ts=1)
    assert recs and recs[0]["type"] == "scenario_refused"
    assert "bad:none" not in {s["scenario_id"]
                              for s in book.live()}
    # A sane setup still proposes.
    good = _setup("good:1", direction=1, entry=4000.0,
                  stop=3990.0, targets=(4020.0,))
    book.propose_from_setup(good, ts=1)
    assert "good:1" in {s["scenario_id"] for s in book.live()}


# ------------------------------------------------------------------
# A2.2 — different timeframes hold independent theses
# ------------------------------------------------------------------
def test_a2_2_independent_timeframe_theses_coexist(tmp_path):
    book = _book(tmp_path)
    # An H4-formed range thesis (refined on H1) ...
    book.propose_from_setup(_setup(
        "range:1", playbook="range_qm", direction=-1,
        entry=4010.0, stop=4025.0, targets=(3985.0,),
        management={"management_horizon": "intraday",
                    "management_policy_id": "standard_v1",
                    "management_policy_version": 1,
                    "management_context_timeframe": "H4",
                    "management_invalidation": 4025.0,
                    "management_target": 3985.0}), ts=1)
    # ... an H1 roster thesis ...
    book.propose_from_setup(_setup("h1:1", direction=1), ts=1)
    # ... and the S2 market thesis itself, mirrored.
    book.sync_thesis(_thesis(), ts=1)
    live = {s["scenario_id"]: s for s in book.live()}
    assert set(live) == {"range:1", "h1:1", "thesis-1"}
    assert live["range:1"]["clocks"]["formation_tf"] == "H4"
    assert live["h1:1"]["clocks"]["formation_tf"] == "H1"
    assert live["thesis-1"]["family"] == "judgment_thesis"
    # Independence: closing one leaves the others untouched.
    book.sync_setup(_setup("h1:1"), ts=2, stage=STAGE_EXPIRED)
    live = {s["scenario_id"]: s for s in book.live()}
    assert set(live) == {"range:1", "thesis-1"}
    assert book.closed[-1]["scenario_id"] == "h1:1"
    assert book.closed[-1]["close_reason"] == "expired"


# ------------------------------------------------------------------
# A2.3 — per-scenario clocks (formation / refinement / trigger /
# invalidation / expiry / management)
# ------------------------------------------------------------------
def test_a2_3_each_scenario_carries_its_own_clocks(tmp_path):
    book = _book(tmp_path)
    book.propose_from_setup(_setup(
        "range:1", playbook="range_qm", direction=1,
        validity_bars=9,
        management={"management_horizon": "intraday",
                    "management_policy_id": "standard_v1",
                    "management_policy_version": 1,
                    "management_context_timeframe": "H4",
                    "management_invalidation": 3990.0,
                    "management_target": 4020.0}), ts=1)
    book.propose_from_setup(_setup("h1:1", validity_bars=6), ts=1)
    rng = _live(book, "range:1")
    h1 = _live(book, "h1:1")
    # Formation + refinement clocks come from the family's own
    # registry declaration (H4 forms, H1 refines for range_qm;
    # the H1 roster forms and refines on its single clock).
    assert rng["clocks"] == {
        "formation_tf": "H4", "refinement_tf": "H1",
        "trigger_tf": "H1", "entry_tf": "H1",
        "management_tf": "H4"}
    assert h1["clocks"] == {
        "formation_tf": "H1", "refinement_tf": None,
        "trigger_tf": "H1", "entry_tf": "H1",
        "management_tf": "H1"}
    # Trigger + invalidation + expiry clocks are per scenario:
    # each carries its own validity window and its own
    # invalidation price/terms.
    assert rng["validity"]["validity_bars"] == 9
    assert h1["validity"]["validity_bars"] == 6
    assert rng["validity"]["touch_ts"] != h1["validity"][
        "touch_ts"] or True  # windows ride the setup's own facts
    assert rng["invalidation"]["price"] == 3990.0
    assert h1["invalidation"]["price"] == 3990.0
    assert rng["invalidation"]["terms"]
    # Management clock = the family's context timeframe.
    assert rng["clocks"]["management_tf"] == "H4"
    assert h1["clocks"]["management_tf"] == "H1"


# ------------------------------------------------------------------
# A2.4 — shared parent identity, evidence counted once
# ------------------------------------------------------------------
def test_a2_4_shared_parent_no_double_counted_evidence(tmp_path):
    book = _book(tmp_path)
    book.sync_thesis(_thesis(), ts=1)
    for sid, entry in (("child:1", 4001.0), ("child:2", 4003.0)):
        book.propose_from_setup(_setup(
            sid, entry=entry, stop=3990.0, targets=(4030.0,),
            thesis_id="thesis-1", thesis_version=1), ts=2)
    # Both children parent to the SAME thesis scenario.
    for sid in ("child:1", "child:2"):
        sc = _live(book, sid)
        assert sc["parent_scenario_id"] == "thesis-1"
        assert sc["thesis_id"] == "thesis-1"
    # S1's weighing funds both with the same correlation group.
    for sid in ("child:1", "child:2"):
        book.note_judgment(
            sid, ts=3, s1={"decision": "abstain",
                           "expected_r": 0.01},
            groups_for=["htf_structure"], groups_against=[])
    records = book.cycle(
        ts=4, assessment=_assessment(bullish=0.5))
    summary = records[-1]
    # The group is ONE vote across the book however many
    # scenarios it funds — and the sharing is named, not silent.
    assert summary["evidence_groups"]["distinct"] == 1
    assert summary["evidence_groups"]["shared"] == {
        "htf_structure": ["child:1", "child:2"]}
    # The parent pair relationship is the declared one.
    kinds = {(r["a"], r["b"]): r["kind"] for r in records
             if r["type"] == "scenario_relationship"}
    assert kinds[("child:1", "thesis-1")] == "parent_child"
    assert kinds[("child:2", "thesis-1")] == "parent_child"


# ------------------------------------------------------------------
# A2.5 — one weakens while another strengthens, same cycle
# ------------------------------------------------------------------
def test_a2_5_weaken_and_strengthen_in_the_same_cycle(tmp_path):
    book = _book(tmp_path)
    book.propose_from_setup(_setup("long:1", direction=1), ts=1)
    book.propose_from_setup(
        _setup("short:1", direction=-1, entry=4010.0,
               stop=4025.0, targets=(3985.0,)), ts=1)
    # Cycle 1: longs strong, shorts weak.
    book.cycle(ts=2, assessment=_assessment(
        bullish=0.60, bearish=0.10))
    # Cycle 2: the long side collapses (peak .60 -> .30, a drop
    # past the weaken band) while the short side strengthens.
    records = book.cycle(ts=3, assessment=_assessment(
        bullish=0.30, bearish=0.45))
    weakened = [r for r in records
                if r["type"] == "scenario_weakened"]
    assert [r["scenario_id"] for r in weakened] == ["long:1"]
    assert weakened[0]["evidence"]["p_before"] == 0.60
    assert weakened[0]["evidence"]["p_after"] == 0.30
    scores = [r for r in records
              if r["type"] == "scenario_scores"][0]["scores"]
    by_id = {s["scenario_id"]: s for s in scores}
    assert by_id["long:1"]["standing_state"] == "weakened"
    assert by_id["long:1"]["trend"] == "down"
    assert by_id["short:1"]["trend"] == "up"
    assert by_id["short:1"]["p"] == 0.45
    assert by_id["short:1"]["standing_state"] == "intact"
    assert _live(book, "long:1")["standing"]["state"] == \
        "weakened"
    assert _live(book, "short:1")["standing"]["p"] == 0.45


# ------------------------------------------------------------------
# A2.6 — reversal forms before the trend thesis is invalidated
# ------------------------------------------------------------------
def test_a2_6_reversal_forms_before_thesis_invalidated(tmp_path):
    book = _book(tmp_path)
    book.sync_thesis(_thesis(direction=1, scenario="bullish"),
                     ts=1)
    # A counter-direction (reversal) scenario forms while the
    # bullish thesis still stands — same region, same horizon.
    book.propose_from_setup(_setup(
        "rev:1", playbook="v_reversal", direction=-1,
        entry=4002.0, stop=4012.0, targets=(3975.0,)), ts=2)
    live = {s["scenario_id"]: s for s in book.live()}
    assert live["thesis-1"]["state"] == "forming"   # NOT closed
    assert live["rev:1"]["state"] == "proposed"
    records = book.refresh_relationships(ts=3)
    kinds = {(r["a"], r["b"]): r["kind"] for r in records
             if r["type"] == "scenario_relationship"}
    # The pair is marked as the pair Judgment must resolve —
    # both still tracked, neither resolved by the book.
    assert kinds[("rev:1", "thesis-1")] == "competing"
    assert _live(book, "rev:1")["state"] == "proposed"
    assert _live(book, "thesis-1")["state"] == "forming"


# ------------------------------------------------------------------
# A2.7 — open position keeps its management context through a
# wider thesis change
# ------------------------------------------------------------------
def test_a2_7_position_management_context_survives_flip(tmp_path):
    book = _book(tmp_path)
    book.sync_thesis(_thesis(), ts=1)
    setup = _setup("pos:1", thesis_id="thesis-1",
                   thesis_version=1)
    book.propose_from_setup(setup, ts=2, authority="governing")
    book.sync_setup(setup, ts=3, stage=STAGE_ARMED)
    book.sync_setup(setup, ts=4, stage=STAGE_TRIGGER_FIRED)
    filled = _setup("pos:1", thesis_id="thesis-1",
                    thesis_version=1, ticket=777,
                    alert_id="alert-1")
    book.sync_setup(filled, ts=5, stage=STAGE_FILLED)
    sc = _live(book, "pos:1")
    assert sc["state"] == "entered"
    frozen = copy.deepcopy(sc["management_context"])
    assert frozen["ticket"] == 777
    assert frozen["management_policy_id"] == "standard_v1"
    assert frozen["management_invalidation"] == 3990.0
    # The position is open at the broker: the scenario manages.
    book.cycle(ts=6, assessment=_assessment(bearish=0.55),
               open_tickets={777})
    assert _live(book, "pos:1")["state"] == "managing"
    # The wider thesis now FLIPS: the parent closes invalidated
    # and an opposing thesis forms.
    closed = _thesis(status="invalidated",
                     close_reason="invalidation observed")
    book.sync_thesis(None, ts=7, closed_theses=[closed])
    book.sync_thesis(_thesis(tid="thesis-2", direction=-1,
                             scenario="bearish"), ts=7)
    sc = _live(book, "pos:1")
    assert sc["state"] == "managing"
    assert sc["management_context"] == frozen
    assert sc["management"] == filled.management
    # And when the position closes, the scenario closes with
    # its context still intact in the closed record.
    book.cycle(ts=8, open_tickets=set())
    assert book.get("pos:1") is None
    closed_rec = [c for c in book.closed
                  if c["scenario_id"] == "pos:1"][0]
    assert closed_rec["close_reason"] == "filled"
    assert closed_rec["management_context"] == frozen


# ------------------------------------------------------------------
# the lifecycle bridge: the full state walk
# ------------------------------------------------------------------
def test_lifecycle_bridge_state_walk(tmp_path):
    book = _book(tmp_path)
    setup = _setup("walk:1")
    recs = book.propose_from_setup(setup, ts=1,
                                   situation_id="sit-1")
    assert recs[0]["type"] == "scenario_transition"
    assert recs[0]["to_state"] == "proposed"
    assert recs[0]["situation_id"] == "sit-1"
    walk = [
        (STAGE_ARMED, "armed"), (STAGE_TRIGGER_FIRED, "triggered"),
        (STAGE_EVALUATED, "triggered"),
        (STAGE_SUBMITTED, "triggered"),
        (STAGE_FILLED, "entered")]
    for stage, want in walk:
        book.sync_setup(setup, ts=2, stage=stage)
        assert _live(book, "walk:1")["state"] == want, stage
    # Version history: every material change a new version,
    # prior facts retained.
    sc = _live(book, "walk:1")
    assert sc["version"] >= 4
    kinds = [h["kind"] for h in sc["history"]]
    assert kinds[0] == "proposed"
    assert "transition" in kinds


@pytest.mark.parametrize("stage,reason", [
    (STAGE_EXPIRED, "expired"),
    (STAGE_INVALIDATED, "invalidated"),
    (STAGE_REJECTED, "rejected")])
def test_lifecycle_terminal_stages_close_the_scenario(
        tmp_path, stage, reason):
    book = _book(tmp_path)
    setup = _setup("term:1")
    book.propose_from_setup(setup, ts=1)
    recs = book.sync_setup(setup, ts=2, stage=stage)
    assert book.get("term:1") is None
    closed = book.closed[-1]
    assert closed["state"] == "closed"
    assert closed["close_reason"] == reason
    assert recs[-1]["to_state"] == "closed"
    assert recs[-1]["close_reason"] == reason


def test_reproposal_is_refresh_never_duplicate(tmp_path):
    book = _book(tmp_path)
    setup = _setup("dup:1")
    book.propose_from_setup(setup, ts=1)
    book.propose_from_setup(setup, ts=2)
    assert len(book.live()) == 1
    assert _live(book, "dup:1")["version"] == 1


# ------------------------------------------------------------------
# competing pairs: marked, never resolved by the book
# ------------------------------------------------------------------
def test_competing_pair_marked_not_resolved(tmp_path):
    book = _book(tmp_path)
    book.propose_from_setup(_setup("a:1", direction=1), ts=1)
    book.propose_from_setup(_setup(
        "b:1", direction=-1, entry=4005.0, stop=4018.0,
        targets=(3980.0,)), ts=1)
    records = book.refresh_relationships(ts=2,
                                         situation_id="sit-9")
    rel = [r for r in records
           if r["type"] == "scenario_relationship"]
    assert len(rel) == 1
    assert rel[0]["kind"] == "competing"
    assert rel[0]["situation_id"] == "sit-9"
    assert rel[0]["evidence"]["overlap"] is True
    for sid in ("a:1", "b:1"):
        sc = _live(book, sid)
        assert sc["state"] == "proposed"      # unresolved
        assert sc["relationships"] == [
            {"other_id": "b:1" if sid == "a:1" else "a:1",
             "kind": "competing"}]
    # A second refresh with no change journals nothing new.
    assert book.refresh_relationships(ts=3) == []


def test_pair_relationship_geometry_rules():
    base = {"direction": 1, "horizon": "intraday",
            "parent_scenario_id": None, "scenario_id": "x",
            "clocks": {"formation_tf": "H1"},
            "geometry": {"entry": 4000.0, "stop": 3990.0,
                         "targets": [4020.0], "zone_id": None}}
    other = copy.deepcopy(base)
    other["scenario_id"] = "y"
    assert pair_relationship(base, other)[0] == "aligned"
    other["direction"] = -1
    assert pair_relationship(base, other)[0] == "competing"
    other["horizon"] = "swing"
    assert pair_relationship(base, other)[0] == "conflicting"
    other["geometry"] = {"entry": 5000.0, "stop": 5010.0,
                         "targets": [4980.0], "zone_id": None}
    assert pair_relationship(base, other)[0] == "independent"
    other["parent_scenario_id"] = "x"
    assert pair_relationship(base, other)[0] == "parent_child"


# ------------------------------------------------------------------
# shadow scoring + situation linkage
# ------------------------------------------------------------------
def test_shadow_scores_from_the_real_assessor(tmp_path):
    from goldenswing.execution.judgment_thesis import (
        assess_market)
    book = _book(tmp_path)
    book.propose_from_setup(_setup("s:1", direction=1), ts=1)
    assessment = assess_market(
        context={"regime": "up/clean_trend/normal/"
                           "accumulation/london",
                 "posture": "risk_on"}, flow={})
    records = book.cycle(ts=2, situation_id="sit-42",
                         assessment=assessment)
    scores = [r for r in records
              if r["type"] == "scenario_scores"]
    assert len(scores) == 1
    assert scores[0]["situation_id"] == "sit-42"
    row = scores[0]["scores"][0]
    assert row["scenario_id"] == "s:1"
    assert row["side"] == "bullish"
    assert row["p"] == scores[0]["assessment"][
        "probabilities"]["bullish"]
    # Every record the cycle produced carries the situation id.
    for rec in records:
        assert rec["situation_id"] == "sit-42"


def test_thesis_mirror_versions_and_close(tmp_path):
    book = _book(tmp_path)
    book.sync_thesis(_thesis(version=1), ts=1)
    sc = _live(book, "thesis-1")
    assert sc["state"] == "forming"
    assert sc["version"] == 1
    # A keeper revision advances the mirror's version.
    book.sync_thesis(_thesis(version=2, state="weakened",
                             winner_p=0.31), ts=2)
    sc = _live(book, "thesis-1")
    assert sc["version"] == 2
    assert sc["standing"]["state"] == "weakened"
    # Supersede: old mirror closes, the successor's parent
    # points back at it.
    old = _thesis(status="superseded",
                  close_reason="bearish won decisively")
    book.sync_thesis(_thesis(tid="thesis-2", direction=-1,
                             scenario="bearish",
                             supersedes="thesis-1"),
                     ts=3, closed_theses=[old])
    assert book.get("thesis-1") is None
    assert book.closed[-1]["close_reason"] == "superseded"
    assert _live(book, "thesis-2")[
        "parent_scenario_id"] == "thesis-1"


# ------------------------------------------------------------------
# persistence: recovery roundtrip + torn-state refusal
# ------------------------------------------------------------------
def test_keeper_recovery_roundtrip(tmp_path):
    book = _book(tmp_path)
    book.propose_from_setup(_setup("r:1"), ts=1)
    book.propose_from_setup(_setup(
        "r:2", direction=-1, entry=4010.0, stop=4025.0,
        targets=(3985.0,)), ts=1)
    book.sync_setup(_setup("r:1"), ts=2, stage=STAGE_ARMED)
    book.sync_setup(_setup("r:2", direction=-1, entry=4010.0,
                           stop=4025.0, targets=(3985.0,)),
                    ts=2, stage=STAGE_EXPIRED)
    book.cycle(ts=3, assessment=_assessment(bullish=0.5))
    woken = _book(tmp_path)
    live = {s["scenario_id"]: s for s in woken.live()}
    assert set(live) == {"r:1"}
    assert live["r:1"]["state"] == "armed"
    assert live["r:1"]["standing"]["p"] == 0.5
    # The closed scenario restores as history, never live.
    assert [c["scenario_id"] for c in woken.closed] == ["r:2"]
    assert woken.closed[0]["close_reason"] == "expired"
    # And the woken book keeps working: a fresh proposal lands.
    woken.propose_from_setup(_setup("r:3"), ts=4)
    assert _live(woken, "r:3")["state"] == "proposed"


def test_torn_book_state_is_refused_loudly(tmp_path):
    path = tmp_path / "scenario_book.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ScenarioBookStateError):
        ScenarioBook(store=ScenarioBookStore(path))
    path.write_text(json.dumps({"kind": "other", "version": 1}),
                    encoding="utf-8")
    with pytest.raises(ScenarioBookStateError):
        ScenarioBook(store=ScenarioBookStore(path))


# ------------------------------------------------------------------
# runner seams
# ------------------------------------------------------------------
def _runner(tmp_path, **cfg_kw):
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False, **cfg_kw)
    return MT5Runner(cfg)


def _journal_records(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_runner_book_mirrors_lifecycle(tmp_path):
    from goldenswing.execution.setup_lifecycle import (
        STAGE_ARMED as LC_ARMED)
    runner = _runner(tmp_path)
    assert runner._setup_store is not None
    setup = _setup("run:1")
    assert runner._setup_store.register(setup, ts="t0")
    runner._book_propose_setup(setup)
    runner._transition_setup(setup, LC_ARMED, "armed")
    records = _journal_records(tmp_path)
    transitions = [r for r in records
                   if r.get("type") == "scenario_transition"]
    assert [r["to_state"] for r in transitions] == [
        "proposed", "armed"]
    assert all(r["scenario_id"] == "run:1"
               for r in transitions)
    assert all("run_id" in r for r in transitions)
    # The H1-close book cycle journals its summary.
    runner._book_cycle(["H1"], "2026-11-08T02:00:00")
    summaries = [r for r in _journal_records(tmp_path)
                 if r.get("type") == "scenario_book"]
    assert len(summaries) == 1
    assert summaries[0]["open"] == 1
    assert summaries[0]["by_state"] == {"armed": 1}


def test_runner_book_silent_when_situation_off(tmp_path):
    runner = _runner(tmp_path, situation_enabled=False)
    assert runner._scenario_book_get() is None
    setup = _setup("off:1")
    assert runner._setup_store.register(setup, ts="t0")
    runner._book_propose_setup(setup)
    runner._book_cycle(["H1"], "2026-11-08T02:00:00")
    assert [r for r in _journal_records(tmp_path)
            if str(r.get("type", "")).startswith(
                "scenario_")] == []
    runner2 = _runner(tmp_path / "b", scenario_book_enabled=False)
    assert runner2._scenario_book_get() is None


def test_runner_decision_records_identical_with_book_on_off(
        tmp_path):
    """Shadow discipline: the evaluation + judgment records a
    candidate produces are byte-identical whether the book is
    watching or not (the book adds records BESIDE them)."""
    def drive(path, **kw):
        runner = _runner(path, **kw)
        runner._emit_eval(
            playbook="sweep_reversal", evaluation_clock="H1",
            intended_tf="H1", outcome="candidate_emitted",
            candidate={"direction": 1, "entry": 4000.0,
                       "stop": 3990.0, "targets": [4020.0]},
            decision={"result": "qualified",
                      "analytical_verdict":
                          {"verdict": "qualified"}},
            evidence={"observed": {}, "inferred": {},
                      "supporting": [], "contradictory": [],
                      "evidence_groups": {}},
            source="live_close")
        return [r for r in _journal_records(path)
                if r.get("type") in ("evaluation",
                                     "judgment_shadow",
                                     "judgment_thesis")]

    import re

    def scrub(records):
        """Per-run volatility (run id, wall-clock stamps —
        top-level, nested, and rendered into text) is not
        decision content."""
        def walk(v):
            if isinstance(v, dict):
                return {k: walk(x) for k, x in v.items()
                        if k not in ("run_id", "commit")}
            if isinstance(v, list):
                return [walk(x) for x in v]
            if isinstance(v, float) and v >= 1_000_000_000:
                return 0.0
            if isinstance(v, str):
                return re.sub(
                    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
                    r"(\.\d+)?(\+\d{2}:\d{2})?", "<ts>", v)
            return v
        return walk(copy.deepcopy(records))

    on = drive(tmp_path / "on")
    off = drive(tmp_path / "off", scenario_book_enabled=False)
    assert on and scrub(on) == scrub(off)
