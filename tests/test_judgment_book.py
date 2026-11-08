"""MarketSituation Pass 3 — Judgment over the book (SPEC §5.2
+ §10 Pass 3; AMENDMENTS-1 §A6 coverage gaps).

What is pinned here:

* POPULATION (Pass 2's named remainder, completed): the book
  mirrors the synchronous native candidates (qm_m5 / qm_d1)
  and the Hot Scalp sleeve's setups, so the §10 audit holds
  FUNNEL-WIDE.  The audit's identity rule: a candidate_emitted
  record's identity is its evidence.setup_id (lifecycle), else
  its evidence.provenance.instance_key (native + scan), else
  its candidate.setup_id (Hot Scalp promotion).  Scenario
  identities are the book's live + closed scenario ids,
  EXCLUDING the family "judgment_thesis" scenarios (the S2
  thesis mirror is a thesis, not a candidate — it is audited
  by the thesis tests).  trigger_event / trigger_binding
  records carry no candidate identity (they are events ABOUT
  setups) and are outside the audit by construction.  An
  identity may hold two book lives across a restart boundary
  of accounting (a warmup residue close, then the live
  registration) — the audit is over identity SETS: every
  candidate identity has a scenario, every scenario has a
  candidate identity, and no identity is ever live twice.

* JUDGMENT (shadow): per H1 book cycle, competing pairs are
  resolved on the assessor's own numbers (§5.2 rules 2–4) and
  lone triggered scenarios are judged on their S1 verdicts;
  every verdict journals a book_judgment record citing the
  situation, the relationship kinds, and the scenario VERSIONS
  judged.  Governed routes keep their legacy authority —
  nothing here is read by a decision path.

* COVERAGE (A6): a measured situation configuration (a ready
  timeframe's bias+phase, or an engine relationship kind) that
  persists across COVERAGE_WINDOW_CYCLES consecutive book
  cycles with no live scenario proposed into it journals ONE
  coverage_gap record; a covered configuration never gaps;
  the cycle summary names the strongest active gap.

* SHADOW DISCIPLINE: decision / evaluation / judgment_shadow
  records are byte-identical with the book-judgment cycle on
  vs off; scenario_book_enabled=False silences the whole
  apparatus.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from goldenswing.execution import judgment_book as jb
from goldenswing.execution.scenario_book import (
    ScenarioBook, ScenarioBookStore)

from test_p3_trader import _FakeScore, _StubEngine, _runner, _sig
from test_funnel_m5 import _flat_h1, _fresh_tick, _m5_fixture
from test_h1_trigger_handoff import (
    _episode_frame, _episode_sig, _episode_spec)
from test_hot_scalp_sleeve import (
    _candidate as _hs_candidate, _promote as _hs_promote,
    _runner as _hs_runner, _tick as _hs_tick)


# ------------------------------------------------------------ units

def _book(tmp_path):
    store = ScenarioBookStore(tmp_path / "scenario_book.json")
    return ScenarioBook(store=store)


def _pair_book(tmp_path, *, long_er=0.30, short_er=0.10,
               long_conf=0.60, short_conf=0.40,
               long_groups=("g1",), short_groups=("g1",),
               long_decision="take", short_decision="take",
               long_standing=None, short_standing=None):
    """A competing long/short pair, both triggered, both with
    S1 verdicts attached — the §5.2 rule-2 fixture."""
    book = _book(tmp_path)
    base = dict(family="qm_m5", route="qm_m5_native",
                horizon="scalp",
                clocks={"formation_tf": "M5",
                        "trigger_tf": "M5", "entry_tf": "M5",
                        "management_tf": "M5"},
                invalidation_terms=["native lifecycle"],
                authority="governing")
    book.propose_scenario(
        "PL", ts="T0", direction=1,
        geometry={"entry": 100.0, "stop": 99.0,
                  "targets": [102.0]}, **base)
    book.propose_scenario(
        "PS", ts="T0", direction=-1,
        geometry={"entry": 100.0, "stop": 101.0,
                  "targets": [98.0]}, **base)
    for sid in ("PL", "PS"):
        book.sync_scenario(sid, ts="T1", to_state="triggered",
                           stage="candidate_emitted",
                           detail="accepted")
    book.note_judgment(
        "PL", ts="T1",
        s1={"decision": long_decision, "expected_r": long_er,
            "confidence": long_conf, "reason": "long case"},
        groups_for=list(long_groups), groups_against=[])
    book.note_judgment(
        "PS", ts="T1",
        s1={"decision": short_decision, "expected_r": short_er,
            "confidence": short_conf, "reason": "short case"},
        groups_for=list(short_groups), groups_against=[])
    if long_standing is not None:
        book.scenarios["PL"]["standing"] = long_standing
    if short_standing is not None:
        book.scenarios["PS"]["standing"] = short_standing
    book.refresh_relationships(ts="T2", situation_id="sit-1")
    return book


def _judgments(records):
    return [r for r in records if r["type"] == "book_judgment"]


def test_pair_take_long(tmp_path):
    book = _pair_book(tmp_path)
    records, extras = jb.evaluate_book(
        book, ts="T3", situation_id="sit-9", situation=None,
        tracker=jb.CoverageTracker())
    (rec,) = _judgments(records)
    assert rec["scope"] == "competing_pair"
    assert rec["verdict"] == "take"
    assert rec["take_scenario_id"] == "PL"
    assert rec["verdicts"] == {"PL": "take", "PS": "monitor"}
    assert rec["loser_reason"] == "competing_pair_lost"
    assert rec["situation_id"] == "sit-9"
    assert rec["pair"] == ["PL", "PS"]
    # The versions judged are cited — a verdict never floats
    # free of the exact scenario states it read.
    assert rec["inputs"]["PL"]["version"] == \
        book.get("PL")["version"]
    assert rec["inputs"]["PL"]["expected_r"] == 0.30
    assert rec["thresholds"]["min_expected_r"] == 0.05
    # Rule 3: the shared group is cited as ONE vote.
    assert any("one vote per group" in r and "g1" in r
               for r in rec["reasons"])
    assert extras["judgment"]["pairs_taken"] == 1
    assert extras["judgment"]["pairs_abstained"] == 0


def test_pair_take_short(tmp_path):
    book = _pair_book(tmp_path, long_er=0.08, short_er=0.42,
                      long_conf=0.30, short_conf=0.70)
    records, _ = jb.evaluate_book(
        book, ts="T3", situation_id="sit-9", situation=None,
        tracker=jb.CoverageTracker())
    (rec,) = _judgments(records)
    assert rec["verdict"] == "take"
    assert rec["take_scenario_id"] == "PS"
    assert rec["verdicts"]["PL"] == "monitor"


def test_pair_epsilon_tie_abstains_both(tmp_path):
    # |Δ| within the pair epsilon: two readings of one market
    # too close to separate — §5.2 rule 4 abstains BOTH, sides
    # stay live, closest stays visible.
    book = _pair_book(tmp_path, long_er=0.30, short_er=0.29)
    records, extras = jb.evaluate_book(
        book, ts="T3", situation_id="sit-9", situation=None,
        tracker=jb.CoverageTracker())
    (rec,) = _judgments(records)
    assert rec["verdict"] == "abstain"
    assert rec["take_scenario_id"] is None
    assert rec["closest_scenario_id"] == "PL"
    assert rec["verdicts"] == {"PL": "abstain", "PS": "abstain"}
    assert any("epsilon" in r for r in rec["reasons"])
    assert extras["judgment"]["pairs_abstained"] == 1


def test_pair_winner_below_bars_abstains_with_closest(tmp_path):
    # The winner must clear the desk's stated bars on its OWN
    # S1 numbers; a sub-bar winner abstains the pair WITH the
    # closest named (the owner rulings' journal-the-closest
    # discipline, carried to the pair read).
    book = _pair_book(tmp_path, long_er=0.04, short_er=0.01,
                      long_conf=0.50, short_conf=0.50)
    records, _ = jb.evaluate_book(
        book, ts="T3", situation_id="sit-9", situation=None,
        tracker=jb.CoverageTracker())
    (rec,) = _judgments(records)
    assert rec["verdict"] == "abstain"
    assert rec["closest_scenario_id"] == "PL"
    assert any("does not clear the book's bars" in r
               for r in rec["reasons"])


def test_pair_unscored_side_abstains_named(tmp_path):
    book = _pair_book(tmp_path)
    book.scenarios["PS"]["s1"] = None  # never evaluated by S1
    records, _ = jb.evaluate_book(
        book, ts="T3", situation_id="sit-9", situation=None,
        tracker=jb.CoverageTracker())
    (rec,) = _judgments(records)
    assert rec["verdict"] == "abstain"
    assert any("unscored" in r and "PS" in r
               for r in rec["reasons"])


def test_pair_assessor_standing_cited(tmp_path):
    standing = {"p": 0.31, "peak_p": 0.55, "state": "weakened"}
    book = _pair_book(tmp_path, long_standing=standing)
    records, _ = jb.evaluate_book(
        book, ts="T3", situation_id="sit-9", situation=None,
        tracker=jb.CoverageTracker())
    (rec,) = _judgments(records)
    assert any("assessor standing" in r and "weakened" in r
               for r in rec["reasons"])
    assert rec["inputs"]["PL"]["standing_state"] == "weakened"


def _lone_book(tmp_path, *, s1):
    book = _book(tmp_path)
    book.propose_scenario(
        "LONE", ts="T0", family="qm_d1", route="qm_d1_native",
        direction=-1, horizon="swing",
        clocks={"formation_tf": "D1", "trigger_tf": "D1",
                "entry_tf": "D1", "management_tf": "D1"},
        geometry={"entry": 100.0, "stop": 102.0,
                  "targets": [95.0]},
        invalidation_terms=["native lifecycle"],
        authority="governing")
    book.sync_scenario("LONE", ts="T1", to_state="triggered",
                       stage="candidate_emitted",
                       detail="accepted")
    if s1 is not None:
        book.note_judgment("LONE", ts="T1", s1=s1,
                           groups_for=["solo"], groups_against=[])
    return book


def test_lone_triggered_take_carries_s1_reason(tmp_path):
    book = _lone_book(tmp_path, s1={
        "decision": "take", "expected_r": 0.22,
        "confidence": 0.5, "reason": "clean geometry"})
    records, extras = jb.evaluate_book(
        book, ts="T2", situation_id="sit-2", situation=None,
        tracker=jb.CoverageTracker())
    (rec,) = _judgments(records)
    assert rec["scope"] == "lone_triggered"
    assert rec["verdict"] == "take"
    assert rec["take_scenario_id"] == "LONE"
    assert any("clean geometry" in r for r in rec["reasons"])
    assert extras["judgment"]["lone_taken"] == 1


def test_lone_triggered_abstain_names_closest(tmp_path):
    book = _lone_book(tmp_path, s1={
        "decision": "abstain", "expected_r": 0.03,
        "confidence": 0.10, "reason": "below the bar"})
    records, extras = jb.evaluate_book(
        book, ts="T2", situation_id="sit-2", situation=None,
        tracker=jb.CoverageTracker())
    (rec,) = _judgments(records)
    assert rec["verdict"] == "abstain"
    assert rec["closest_scenario_id"] == "LONE"
    assert any("below the bar" in r for r in rec["reasons"])
    assert extras["judgment"]["lone_abstained"] == 1


def test_lone_triggered_without_s1_abstains_absent(tmp_path):
    book = _lone_book(tmp_path, s1=None)
    records, _ = jb.evaluate_book(
        book, ts="T2", situation_id="sit-2", situation=None,
        tracker=jb.CoverageTracker())
    (rec,) = _judgments(records)
    assert rec["verdict"] == "abstain"
    assert any("absent stays absent" in r
               for r in rec["reasons"])


def test_triggered_side_of_pair_is_not_judged_lone(tmp_path):
    # A triggered scenario INSIDE a competing pair is judged
    # by the pair read only — never double-judged as lone.
    book = _pair_book(tmp_path)
    records, extras = jb.evaluate_book(
        book, ts="T3", situation_id="sit-9", situation=None,
        tracker=jb.CoverageTracker())
    assert extras["judgment"]["lone_triggered"] == 0
    assert len(_judgments(records)) == 1


# ------------------------------------------------------- coverage

def _situation(*, tfs=None, rels=None):
    return {"situation_id": "sit-x", "per_tf": tfs or {},
            "relationships": rels or []}


def _tf_state(tf, *, bias=1, phase="expansion",
              readiness="ready"):
    return {"tf": tf, "readiness": readiness, "bias": bias,
            "structure_phase": phase,
            "direction_word": "up" if bias > 0 else "down"}


def test_coverage_gap_fires_once_at_window(tmp_path):
    book = _book(tmp_path)
    tracker = jb.CoverageTracker()
    sit = _situation(tfs={"D1": _tf_state("D1")})
    fired = []
    extras = {}
    for i in range(5):
        records, extras = jb.evaluate_book(
            book, ts=f"T{i}", situation_id=f"s{i}",
            situation=sit, tracker=tracker)
        fired += [r for r in records
                  if r["type"] == "coverage_gap"]
    assert len(fired) == 1
    gap = fired[0]
    assert gap["config_key"] == "tf:D1:bias=+1:phase=expansion"
    assert gap["first_seen_ts"] == "T0"
    assert gap["first_situation_id"] == "s0"
    assert gap["cycles"] == jb.COVERAGE_WINDOW_CYCLES
    assert gap["window"] == jb.COVERAGE_WINDOW_CYCLES
    # The summary channel names the strongest active gap.
    assert extras["coverage_gap"]["config_key"] == \
        gap["config_key"]


def test_coverage_covered_configuration_never_gaps(tmp_path):
    book = _book(tmp_path)
    # A live scenario formed on D1 in the bias direction
    # covers the D1 configuration.
    book.propose_scenario(
        "COV", ts="T0", family="order_block",
        route="h1_roster", direction=1, horizon="swing",
        clocks={"formation_tf": "D1", "trigger_tf": "H1",
                "entry_tf": "H1", "management_tf": "H1"},
        geometry={"entry": 100.0, "stop": 99.0,
                  "targets": [103.0]},
        invalidation_terms=["x"], authority="shadow")
    tracker = jb.CoverageTracker()
    sit = _situation(tfs={"D1": _tf_state("D1")})
    for i in range(5):
        records, extras = jb.evaluate_book(
            book, ts=f"T{i}", situation_id=f"s{i}",
            situation=sit, tracker=tracker)
        assert not [r for r in records
                    if r["type"] == "coverage_gap"]
        assert "coverage_gap" not in extras


def test_coverage_counterdirection_scenario_does_not_cover(
        tmp_path):
    # A SHORT scenario into a +1 measured bias is trading
    # AGAINST the configuration, not into it — the gap stands.
    book = _book(tmp_path)
    book.propose_scenario(
        "AGAINST", ts="T0", family="order_block",
        route="h1_roster", direction=-1, horizon="swing",
        clocks={"formation_tf": "D1", "trigger_tf": "H1",
                "entry_tf": "H1", "management_tf": "H1"},
        geometry={"entry": 100.0, "stop": 101.0,
                  "targets": [97.0]},
        invalidation_terms=["x"], authority="shadow")
    tracker = jb.CoverageTracker()
    sit = _situation(tfs={"D1": _tf_state("D1")})
    fired = []
    for i in range(3):
        records, _ = jb.evaluate_book(
            book, ts=f"T{i}", situation_id=f"s{i}",
            situation=sit, tracker=tracker)
        fired += [r for r in records
                  if r["type"] == "coverage_gap"]
    assert len(fired) == 1


def test_coverage_break_resets_the_streak(tmp_path):
    book = _book(tmp_path)
    tracker = jb.CoverageTracker()
    sit = _situation(tfs={"D1": _tf_state("D1")})
    empty = _situation()
    fired = []
    for i, s in enumerate([sit, sit, empty, sit, sit]):
        records, _ = jb.evaluate_book(
            book, ts=f"T{i}", situation_id=f"s{i}",
            situation=s, tracker=tracker)
        fired += [r for r in records
                  if r["type"] == "coverage_gap"]
    assert fired == []  # never 3 consecutive cycles


def test_coverage_unready_timeframe_never_configures(tmp_path):
    book = _book(tmp_path)
    tracker = jb.CoverageTracker()
    sit = _situation(tfs={
        "D1": _tf_state("D1", readiness="collecting")})
    for i in range(4):
        records, _ = jb.evaluate_book(
            book, ts=f"T{i}", situation_id=f"s{i}",
            situation=sit, tracker=tracker)
        assert not [r for r in records
                    if r["type"] == "coverage_gap"]


def test_coverage_relationship_gap_outranks_timeframe(tmp_path):
    # Both configurations persist uncovered for the window;
    # the summary's strongest is the RELATIONSHIP (a
    # cross-timeframe read is the more structural absence).
    book = _book(tmp_path)
    tracker = jb.CoverageTracker()
    sit = _situation(
        tfs={"D1": _tf_state("D1")},
        rels=[{"htf": "D1", "ltf": "H1", "kind": "correcting",
               "source": "engine", "reason": "ltf against"}])
    extras = {}
    for i in range(3):
        _, extras = jb.evaluate_book(
            book, ts=f"T{i}", situation_id=f"s{i}",
            situation=sit, tracker=tracker)
    assert extras["coverage_gap"]["config_key"] == \
        "rel:D1->H1:correcting"


def test_coverage_independent_relationship_never_gaps(
        tmp_path):
    book = _book(tmp_path)
    tracker = jb.CoverageTracker()
    sit = _situation(rels=[
        {"htf": "D1", "ltf": "H1", "kind": "independent",
         "source": "engine", "reason": "no shared structure"}])
    for i in range(4):
        records, _ = jb.evaluate_book(
            book, ts=f"T{i}", situation_id=f"s{i}",
            situation=sit, tracker=tracker)
        assert not [r for r in records
                    if r["type"] == "coverage_gap"]


# ------------------------------------------- book mirror units

def test_native_mirror_lifecycle(tmp_path):
    book = _book(tmp_path)
    kw = dict(family="qm_m5", route="qm_m5_native",
              direction=1, horizon="scalp",
              clocks={"formation_tf": "M5", "trigger_tf": "M5",
                      "entry_tf": "M5", "management_tf": "M5"},
              invalidation_terms=["native"],
              authority="governing")
    recs = book.propose_scenario(
        "K1", ts="T0", geometry={"entry": 10.0, "stop": 9.0,
                                 "targets": [12.0]}, **kw)
    assert [r["type"] for r in recs] == ["scenario_transition"]
    assert book.get("K1")["state"] == "proposed"
    # Re-proposal is idempotent — the same candidate is the
    # same scenario, never a duplicate.
    assert book.propose_scenario(
        "K1", ts="T1", geometry={"entry": 10.0, "stop": 9.0,
                                 "targets": [12.0]}, **kw) == []
    assert len(book.scenarios) == 1
    # The funnel tap's S1 verdict lands on the live scenario.
    book.note_judgment("K1", ts="T1", s1={"decision": "take"},
                       groups_for=["g"], groups_against=[])
    assert book.get("K1")["s1"]["decision"] == "take"
    # Trigger, then the broker ticket enters it with the
    # management context frozen.
    book.sync_scenario("K1", ts="T2", to_state="triggered",
                       stage="candidate_emitted",
                       detail="accepted")
    book.scenarios["K1"]["management"] = {"playbook": "qm_m5"}
    recs = book.note_position("K1", 777, ts="T3")
    assert book.get("K1")["state"] == "entered"
    assert book.get("K1")["ticket"] == 777
    frozen = book.get("K1")["management_context"]
    assert frozen is not None and frozen["ticket"] == 777
    assert recs and recs[0]["to_state"] == "entered"
    # Re-noting the same ticket is a no-op.
    assert book.note_position("K1", 777, ts="T4") == []


def test_mirror_close_is_terminal_and_sticky(tmp_path):
    book = _book(tmp_path)
    kw = dict(family="qm_d1", route="qm_d1_native",
              direction=-1, horizon="swing",
              clocks={"formation_tf": "D1", "trigger_tf": "D1",
                      "entry_tf": "D1", "management_tf": "D1"},
              invalidation_terms=["native"], authority="shadow")
    book.propose_scenario(
        "K2", ts="T0", geometry={"entry": 10.0, "stop": 11.0,
                                 "targets": []}, **kw)
    recs = book.sync_scenario(
        "K2", ts="T1", close_reason="rejected",
        stage="candidate_disposed", detail="veto: hard lines")
    assert recs[0]["close_reason"] == "rejected"
    assert book.get("K2") is None
    # A closed scenario never re-opens through the mirror.
    assert book.propose_scenario(
        "K2", ts="T2", geometry={"entry": 10.0, "stop": 11.0,
                                 "targets": []}, **kw) == []
    assert book.get("K2") is None


def test_summary_extras_merge_and_default_shape(tmp_path):
    book = _pair_book(tmp_path)
    # Default cycle: the Pass 2 summary shape, unchanged.
    records = book.cycle(ts="T4", situation_id="sit-9",
                         assessment=None, open_tickets=None)
    summary = records[-1]
    assert summary["type"] == "scenario_book"
    assert "judgment" not in summary
    assert "coverage_gap" not in summary
    # With extras: judgment counts + strongest gap ride it.
    _, extras = jb.evaluate_book(
        book, ts="T5", situation_id="sit-9", situation=None,
        tracker=jb.CoverageTracker())
    extras["coverage_gap"] = {
        "config_key": "tf:D1:bias=+1:phase=expansion",
        "label": "D1 measured state", "first_seen_ts": "T0",
        "first_situation_id": "s0", "cycles": 3}
    records = book.cycle(ts="T5", situation_id="sit-9",
                         assessment=None, open_tickets=None,
                         summary_extras=extras)
    summary = records[-1]
    assert summary["judgment"]["pairs"] == 1
    assert summary["judgment"]["pairs_taken"] == 1
    assert summary["coverage_gap"]["config_key"].startswith(
        "tf:D1")


# --------------------------------------- runner: the audit

def _audit_sets(runner):
    """The §10 population audit's two identity sets (see the
    module docstring for the identity rule)."""
    funnel: set = set()
    for r in runner.journal.records():
        if r.get("type") == "evaluation" \
                and r.get("outcome") == "candidate_emitted":
            ev = r.get("evidence") or {}
            ident = ev.get("setup_id")
            if not ident:
                prov = ev.get("provenance") or {}
                ident = prov.get("instance_key")
            if not ident:
                cand = r.get("candidate") or {}
                ident = cand.get("setup_id")
            assert ident, (
                "candidate_emitted record without a resolvable "
                f"identity (playbook={r.get('playbook')})")
            funnel.add(str(ident))
        if r.get("type") == "hot_scalp_event" \
                and r.get("event") == "setup_armed":
            live_path = r.get("live_path") or {}
            if live_path.get("chain_id"):
                funnel.add(str(live_path["chain_id"]))
    book = runner._scenario_book_get()
    assert book is not None
    scenarios = {sid for sid, sc in book.scenarios.items()
                 if sc.get("family") != "judgment_thesis"}
    scenarios |= {c["scenario_id"] for c in book.closed
                  if c.get("family") != "judgment_thesis"}
    return funnel, scenarios


def test_population_audit_funnel_wide(tmp_path):
    """Drive all three candidate populations on ONE runner —
    roster sweep (live + warmup), native qm_m5 (live fill),
    Hot Scalp (arm → trigger → promote/fill) — and pin the
    §10 audit: candidate identities == scenario identities,
    exactly."""
    runner = _hs_runner(tmp_path)
    # --- Hot Scalp leg: arm, trigger, promote.
    cand = _hs_candidate(runner, setup_id="hs-audit-1",
                         parent_id="hp-audit-1")
    row = runner._hot_scalp.store.get("hs-audit-1")
    real_engine = runner._hot_scalp
    armed_event = {"event": "setup_armed", "setup": dict(row)}
    runner._hot_scalp = SimpleNamespace(
        on_tick=lambda **kw: [armed_event],
        store=real_engine.store)
    runner._run_hot_scalp_shadow(_hs_tick())
    runner._hot_scalp = SimpleNamespace(
        on_tick=lambda **kw: [cand], store=real_engine.store)
    runner._run_hot_scalp_shadow(_hs_tick())
    runner._hot_scalp = real_engine
    _hs_promote(runner, cand)
    # A book cycle while the real tick manager still holds the
    # sleeve's position: its ticket resolves onto the scenario.
    runner._book_cycle(["H1"], asof="2026-10-27T01:30:00+00:00")
    # --- Roster leg: one warmup candidate (residue mirror),
    # one live candidate (handoff registration mirror).
    runner._live_setup = {"bias_d1": -1, "bias_h4": -1,
                          "regime": "trend",
                          "h4_context": "downtrend structure",
                          "size_mult": 1.0}
    import pandas as pd
    runner._last_tick = {
        "mid": 3996.0, "bid": 3995.9, "ask": 3996.1,
        "spread": 0.2,
        "time": pd.Timestamp("2026-11-04 06:00:00", tz="UTC")}
    frame = _episode_frame()
    warm_sig = _episode_sig(frame)
    warm_sig.entry_ref = 3900.0  # a distinct instance key
    runner._sweep_eval_one(
        _episode_spec(warm_sig), frame, clock="h1_close",
        intended_tf="H1", actual_tf="H1", source="warmup",
        shadow=True)
    runner._sweep_eval_one(
        _episode_spec(_episode_sig(frame)), frame,
        clock="h1_close", intended_tf="H1", actual_tf="H1",
        source="live_close", shadow=False)
    # --- Native leg: a live qm_m5 touch through the shared
    # order path (the funnel fixture's proven fill drive).
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0,
                                                 15.0)
    runner._get_engine = lambda: _StubEngine()
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": _flat_h1(30, 95.0), "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    runner._run_qm_m5(m5, {})
    # --- One book cycle: tickets resolve, judgment runs.
    runner._book_cycle(["H1"], asof="2026-10-27T03:00:00+00:00")

    funnel, scenarios = _audit_sets(runner)
    assert funnel, "the drive must emit candidates"
    assert "hs-audit-1" in funnel
    assert funnel == scenarios, (
        f"candidates without scenarios: "
        f"{sorted(funnel - scenarios)}; scenarios without "
        f"candidates: {sorted(scenarios - funnel)}")
    book = runner._scenario_book_get()
    # The sleeve scenario rode the whole life: armed →
    # triggered → entered (its promotion filled, and the
    # cycle's ticket resolution found it).
    hs = book.get("hs-audit-1")
    assert hs is not None
    # Entered at the first cycle, then carried into management
    # by the cycle's own position reconciliation.
    assert hs["state"] in ("entered", "managing")
    assert hs["ticket"] is not None
    assert hs["clocks"]["entry_tf"] == "M1"
    # The native scenario carries its S1 verdict (the funnel
    # tap found it live — proposed BEFORE the order path ran).
    qm = [sc for sc in list(book.scenarios.values())
          + book.closed if sc.get("family") == "qm_m5"]
    assert qm and qm[0]["s1"] is not None
    # The cycle's summary carries the judgment account.
    summaries = [r for r in runner.journal.records()
                 if r.get("type") == "scenario_book"]
    assert summaries and "judgment" in summaries[-1]


def test_book_judgment_through_runner_cycle(tmp_path):
    # Two mirror-native scenarios in competition, judged by
    # the runner's own H1 book cycle.
    runner = _runner(tmp_path)
    book = runner._scenario_book_get()
    assert book is not None
    for sid, direction, er in (("RJ-L", 1, 0.35),
                               ("RJ-S", -1, 0.12)):
        sig = _sig(direction=direction, entry=100.0,
                   stop=99.0 if direction > 0 else 101.0,
                   index=0)
        sig.targets = [102.0] if direction > 0 else [98.0]
        runner._book_mirror_native(
            "qm_m5", "qm_m5_native", key=sid, sig=sig, prov={},
            disposition="triggered", detail="accepted")
        book.note_judgment(
            sid, ts="T1",
            s1={"decision": "take", "expected_r": er,
                "confidence": 0.5, "reason": "case"},
            groups_for=["g"], groups_against=[])
    judged_version = book.get("RJ-L")["version"]
    runner._book_cycle(["H1"], asof="2026-11-04T07:00:00+00:00")
    recs = [r for r in runner.journal.records()
            if r.get("type") == "book_judgment"]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["verdict"] == "take"
    assert rec["take_scenario_id"] == "RJ-L"
    assert "situation_id" in rec
    # The version judged is the version that stood when the
    # cycle read the book (the assessor may move it after).
    assert rec["inputs"]["RJ-L"]["version"] == judged_version
    summaries = [r for r in runner.journal.records()
                 if r.get("type") == "scenario_book"]
    assert summaries[-1]["judgment"]["pairs_taken"] == 1


# --------------------------------------- shadow discipline

def _scrub(records):
    """The Pass 2 comparison scrub: run ids / commits /
    wall-clock fields out, record substance compared."""
    cleaned = []
    for rec in records:
        rec = copy.deepcopy(rec)
        rec.pop("run_id", None)
        rec.pop("commit", None)
        for key in ("timestamp", "ts", "evaluated_at",
                    "recorded_at"):
            if isinstance(rec.get(key), str):
                rec[key] = "<ts>"

        def _norm(value):
            if isinstance(value, dict):
                return {k: _norm(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_norm(v) for v in value]
            if isinstance(value, float) and abs(value) >= 1e9:
                return 0.0
            return value

        cleaned.append(_norm(rec))
    return cleaned


def _decision_records(runner):
    return [r for r in runner.journal.records()
            if r.get("type") in ("evaluation", "decision",
                                 "judgment_shadow")]


def _drive_episode(runner):
    import pandas as pd
    runner._live_setup = {"bias_d1": -1, "bias_h4": -1,
                          "regime": "trend",
                          "h4_context": "downtrend structure",
                          "size_mult": 1.0}
    runner._last_tick = {
        "mid": 3996.0, "bid": 3995.9, "ask": 3996.1,
        "spread": 0.2,
        "time": pd.Timestamp("2026-11-04 06:00:00", tz="UTC")}
    frame = _episode_frame()
    runner._sweep_eval_one(
        _episode_spec(_episode_sig(frame)), frame,
        clock="h1_close", intended_tf="H1", actual_tf="H1",
        source="live_close", shadow=False)
    runner._book_cycle(["H1"], asof="2026-11-04T07:00:00+00:00")


def test_decision_records_byte_identical_judgment_on_off(
        tmp_path):
    on = _runner(tmp_path / "on")
    _drive_episode(on)
    off = _runner(tmp_path / "off",
                  judgment_book_enabled=False)
    _drive_episode(off)
    assert _scrub(_decision_records(on)) == \
        _scrub(_decision_records(off))
    # …and the off runner really was off: no judgment records,
    # no judgment keys on the summary — while its book still
    # mirrored the population.
    assert not [r for r in off.journal.records()
                if r.get("type") in ("book_judgment",
                                     "coverage_gap")]
    off_summaries = [r for r in off.journal.records()
                     if r.get("type") == "scenario_book"]
    assert off_summaries
    assert "judgment" not in off_summaries[-1]
    assert off._scenario_book_get().scenarios
    on_summaries = [r for r in on.journal.records()
                    if r.get("type") == "scenario_book"]
    assert "judgment" in on_summaries[-1]


def test_scenario_book_disabled_total_silence(tmp_path):
    runner = _runner(tmp_path, scenario_book_enabled=False)
    assert runner._scenario_book_get() is None
    _drive_episode(runner)
    noisy = [r for r in runner.journal.records()
             if r.get("type") in (
                 "scenario_transition", "scenario_scores",
                 "scenario_weakened", "scenario_relationship",
                 "scenario_book", "book_judgment",
                 "coverage_gap")]
    assert noisy == []
