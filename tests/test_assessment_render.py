"""Visible market-reading milestone (owner, 2026-10-23).

The owner reads "signals" as a structured MARKET ASSESSMENT he
can challenge — structure read, itemized evidence, competing
scenarios, verdict + plain-language reason — not a bare BUY/SELL
or a score.  These tests pin:

(a) the pure renderer (goldenswing.execution.assessment_render):
    every verdict word renders; absent structure/scenarios/
    markers render as EXPLICIT not-produced lines, never invented
    prose; evidence is itemized with detector + source-bar ts,
    supporting vs contradictory in separate groups, correlated
    items tagged with their group; a record with no assessment
    block renders the absence itself;
(b) SHADOW records (§G range routes) are labelled SHADOW and
    can never render as a signal;
(c) the journal: every evaluation record the runner emits now
    carries ``assessment_text`` (human-readable evidence);
(d) the /assess bot command (latest per playbook / one playbook
    / all / unknown / empty journal);
(e) the outbound surface: a LIVE signal/watch terminal verdict
    announces its rendered assessment through the existing
    _send_alert path; warmup / shadow / refused evaluations
    never announce;
(f) an end-to-end scripted-frames drive: native M5 route, one
    H1 roster playbook and one §G range playbook each produce
    rendered assessments in the journal.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from goldenswing.alerts.commands import CommandContext
from goldenswing.execution.assessment_render import (
    is_shadow_record, latest_evaluations, render_assessment)
from goldenswing.execution.funnel import assessment_block
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_commands import MT5_COMMANDS

from test_funnel_m5 import _fresh_tick, _m5_fixture, _qm_runner
from test_p3_trader import _flat_h1, _frame
from test_phase2_fixes import _wire


# ------------------------------------------------------- records

def _signal_record() -> dict:
    """A qm_m5-shaped accepted record (mirrors the Phase 2
    sample journal, trimmed to the fields the renderer reads)."""
    evidence = {
        "route": "qm_m5_native",
        "observed": {"qm_level": 96.0, "head": 100.0,
                     "neckline": 90.0, "bias_d1": 0, "bias_h4": 0,
                     "session_bucket": "asia"},
        "inferred": {"regime": "trend", "journey_state": "open",
                     "alignment": "D1_Only"},
        "supporting": ["displacement_break: the neckline break "
                       "displaced"],
        "contradictory": ["counter-D1 context: macro bias is "
                          "against the short"],
        "evidence_groups": {
            "qm_pattern_event": [
                "displacement_break: the neckline break displaced"],
            "htf_context": [
                "counter-D1 context: macro bias is against "
                "the short"]},
        "independent_evidence_groups": ["qm_pattern_event",
                                        "htf_context"],
        "provenance": {"evaluated_bar_ts":
                       "2026-01-01 02:35:00+00:00",
                       "evaluation_ts":
                       "2026-01-01 02:42:00+00:00",
                       "source": "live_close"},
    }
    decision = {
        "result": "accepted", "reason_code": None,
        "reason": "weighed judgment: score 70 on M5 evidence",
        "execution_eligible": True, "order_result": "placed_pending",
        "analytical_verdict": {"verdict": "qualified",
                               "reason_code": None},
        "execution": {"eligible": True,
                      "outcome": "placed_pending",
                      "reason_code": None},
        "competing_scenarios": [
            {"scenario": "take the -1 QM touch at 96.00",
             "status": "selected"},
            {"scenario": "take the +1 side (the flip side)",
             "status": "not funded by the weighed evidence"}],
        "floor_authority": "NO score floor (judgment route)",
        "selected_action": "submit to the shared order path",
    }
    candidate = {
        "direction": -1, "entry": 96.0, "stop": 100.34,
        "targets": [90.0], "score": 70.0, "asymmetry_score": 70.0,
        "sleeve_base_score": 60.0,
        "provenance": evidence["provenance"]}
    return {
        "type": "evaluation", "kind": "playbook_eval",
        "ts": "2026-01-01T02:42:00+00:00",
        "playbook": "qm_m5", "intended_tf": "M5",
        "actual_input_tf": "M5", "evaluation_clock": "m5_close",
        "last_bar_ts": "2026-01-01 02:35:00+00:00",
        "outcome": "candidate_emitted", "reason_code": None,
        "evaluation_source": "live_close",
        "candidate": candidate, "decision": decision,
        "evidence": evidence,
        "assessment": assessment_block(
            outcome="candidate_emitted", decision=decision,
            evidence=evidence, candidate=candidate,
            last_bar_ts="2026-01-01 02:35:00+00:00",
            data_freshness_s=120.0),
    }


def _no_setup_record() -> dict:
    return {
        "type": "evaluation", "kind": "playbook_eval",
        "ts": "2026-01-01T03:00:00+00:00",
        "playbook": "fvg_midline", "intended_tf": "H1",
        "actual_input_tf": "H1", "evaluation_clock": "h1_close",
        "last_bar_ts": "2026-01-01 02:00:00+00:00",
        "outcome": "no_setup",
        "reason": "scheduled scan on the closing bar — no signal",
        "evaluation_source": "live_close",
        "candidate": None, "decision": None,
        "evidence": {"route": "h1_roster", "shadow": False},
        "assessment": assessment_block(
            outcome="no_setup",
            evidence={"route": "h1_roster"},
            reason="scheduled scan on the closing bar — no signal",
            last_bar_ts="2026-01-01 02:00:00+00:00"),
    }


def _shadow_record() -> dict:
    rec = _no_setup_record()
    rec["playbook"] = "range_qm"
    rec["evaluation_clock"] = "h4_generation"
    rec["intended_tf"] = "H4"
    rec["outcome"] = "candidate_emitted"
    rec["reason_code"] = "path.shadow_route"
    rec["reason"] = "§G shadow evaluation route: journaled for " \
                    "grading, NO order authority"
    rec["candidate"] = {"direction": 1, "entry": 2000.0,
                        "stop": 1990.0, "targets": [2020.0],
                        "score": 55.0}
    rec["decision"] = {"result": "shadow",
                       "reason_code": "path.shadow_route",
                       "reason": rec["reason"],
                       "execution_eligible": False,
                       "order_result": "not_attempted"}
    rec["evidence"] = {"route": "range_h4_h1", "shadow": True,
                       "d1_bias_context": 1}
    rec["assessment"] = assessment_block(
        outcome="candidate_emitted", decision=rec["decision"],
        evidence=rec["evidence"],
        reason_code="path.shadow_route", reason=rec["reason"])
    return rec


# ------------------------------------------------- (a) renderer

def test_render_signal_full_shape():
    text = render_assessment(_signal_record(), symbol="XAUUSD")
    assert "🧭 MARKET ASSESSMENT — XAUUSD" in text
    assert "qm_m5 · M5 · m5_close · route qm_m5_native" in text
    assert "bar 2026-01-01 02:35:00+00:00" in text
    assert "VERDICT: SIGNAL" in text
    assert "Reason: weighed judgment: score 70" in text
    # Candidate geometry: entry / invalidation / targets.
    assert "Candidate: SHORT" in text
    assert "entry 96.00" in text
    assert "invalidation (stop) 100.34" in text
    assert "targets 90.00" in text
    # Truthful score naming: the asymmetry total is the headline;
    # the sleeve base is labelled a placeholder, never a floor.
    assert "asymmetry score 70" in text
    assert "never a floor" in text
    # Two verdict layers, distinct.
    assert "analytical verdict: qualified" in text
    assert "execution: eligible — placed_pending" in text
    # Structure + HTF from the record's own evidence.
    assert "regime: trend" in text
    assert "bias_d1: 0" in text
    assert "Levels: QM level 96.00" in text
    # Scenarios verbatim.
    assert "take the -1 QM touch at 96.00 — selected" in text


def test_render_evidence_itemized_grouped_split_by_stance():
    text = render_assessment(_signal_record(), symbol="XAUUSD")
    sup = text.split("Supporting evidence:")[1] \
              .split("Contradictory evidence:")[0]
    con = text.split("Contradictory evidence:")[1] \
              .split("Competing scenarios")[0]
    # Supporting item: text + detector + source-bar timestamp +
    # correlation-group tag.
    assert "displacement_break: the neckline break displaced" in sup
    assert "qm_m5_native, bar 2026-01-01 02:35:00+00:00" in sup
    assert "[qm_pattern_event]" in sup
    # The contradictory item is in the CONTRADICTORY group only.
    assert "counter-D1 context" in con
    assert "counter-D1 context" not in sup
    assert "[htf_context]" in con
    # Groups are named as independent-evidence groups (one event
    # per group, not stacked confirmations).
    assert "Independent evidence groups" in text
    assert "not" in text and "stacked confirmations" in text


@pytest.mark.parametrize("verdict, fragment", [
    ("watch", "VERDICT: WATCH"),
    ("no-setup", "VERDICT: NO-SETUP"),
    ("rejected", "VERDICT: REJECTED"),
    ("expired", "VERDICT: EXPIRED"),
])
def test_render_each_verdict_word(verdict, fragment):
    rec = _no_setup_record()
    rec["assessment"] = dict(rec["assessment"], verdict=verdict)
    text = render_assessment(rec, symbol="XAUUSD")
    assert fragment in text


def test_render_absent_sections_are_explicit_never_invented():
    text = render_assessment(_no_setup_record(), symbol="XAUUSD")
    assert "VERDICT: NO-SETUP" in text
    # The roster route produces no structure read / scenarios:
    # the render says so, in words — and invents nothing.
    assert "Market structure: not produced" in text
    assert "HTF context: not produced" in text
    assert "Competing scenarios: not produced" in text
    assert "Supporting evidence:" in text
    assert "➖ none recorded" in text
    # Census-backed capability markers render as explicit lines.
    # Census-wiring pass premise correction: the taxonomy and
    # premium/discount are now INTEGRATED as recorded evidence
    # on the live path (legacy SMC partially); this hand-built
    # record carries no census snapshot, so the markers state
    # the wiring, not the retired absent/not_integrated facts.
    assert "named_candlestick_taxonomy: integrated" in text
    assert "premium_discount: integrated" in text
    assert "legacy_smc_block: partially_integrated" in text


def test_render_record_without_assessment_block_is_honest():
    rec = _signal_record()
    rec["assessment"] = None
    text = render_assessment(rec, symbol="XAUUSD")
    assert "NOT PRODUCED" in text
    assert "no assessment block" in text
    # The decision facts the record carries still show...
    assert "Decision on record: accepted" in text
    # ...but no market read is fabricated.
    assert "Market structure:" not in text
    assert "VERDICT:" not in text


def test_render_empty_record_never_invents_prose():
    text = render_assessment({}, symbol=None)
    assert "NOT PRODUCED" in text
    assert "instrument not recorded" in text
    assert "regime" not in text
    assert "VERDICT:" not in text


# ---------------------------------------------------- (b) shadow

def test_shadow_record_is_labelled_and_never_a_signal():
    rec = _shadow_record()
    assert is_shadow_record(rec)
    text = render_assessment(rec, symbol="XAUUSD")
    assert "SHADOW EVALUATION" in text
    assert "NO order authority" in text
    assert "VERDICT: SIGNAL" not in text
    assert "VERDICT: WATCH" in text


def test_shadow_verdict_signal_is_forced_down_to_watch():
    rec = _shadow_record()
    rec["assessment"] = dict(rec["assessment"], verdict="signal")
    text = render_assessment(rec, symbol="XAUUSD")
    assert "VERDICT: SIGNAL" not in text
    assert "VERDICT: WATCH" in text


def test_live_record_is_not_shadow():
    assert not is_shadow_record(_signal_record())
    assert not is_shadow_record(_no_setup_record())


# ------------------------------------------- (d) /assess command

def _journal_with_records(tmp_path) -> TradeJournal:
    # Journal order is append order; latest_evaluations takes
    # the LAST record per playbook.  The older qm_m5 watch is
    # superseded by the later qm_m5 signal.
    older = _signal_record()
    older["ts"] = "2025-12-31T00:00:00+00:00"
    older["assessment"] = dict(older["assessment"],
                               verdict="watch")
    j = TradeJournal(tmp_path / "mt5_journal.jsonl")
    j.append(older)
    j.append(_no_setup_record())
    j.append(_signal_record())
    j.append(_shadow_record())
    return j


def _ctx(journal, tmp_path):
    return CommandContext(journal=journal, journal_path=journal.path,
                          reports_dir=tmp_path / "reports")


def test_assess_registered():
    assert "/assess" in MT5_COMMANDS


def test_assess_default_shows_newest_plus_index(tmp_path):
    j = _journal_with_records(tmp_path)
    res = MT5_COMMANDS["/assess"][1](_ctx(j, tmp_path), [])
    assert "MARKET ASSESSMENT" in res.text
    # Index lists every playbook with its latest verdict; the
    # shadow playbook is marked SHADOW in the index too.
    assert "fvg_midline: NO-SETUP" in res.text
    assert "qm_m5: SIGNAL" in res.text
    assert "range_qm: WATCH (SHADOW)" in res.text


def test_assess_one_playbook(tmp_path):
    j = _journal_with_records(tmp_path)
    res = MT5_COMMANDS["/assess"][1](_ctx(j, tmp_path), ["qm_m5"])
    assert "VERDICT: SIGNAL" in res.text
    assert "displacement_break" in res.text


def test_assess_shadow_playbook_labelled(tmp_path):
    j = _journal_with_records(tmp_path)
    res = MT5_COMMANDS["/assess"][1](_ctx(j, tmp_path), ["range_qm"])
    assert "SHADOW EVALUATION" in res.text
    assert "VERDICT: SIGNAL" not in res.text


def test_assess_all_roster(tmp_path):
    j = _journal_with_records(tmp_path)
    res = MT5_COMMANDS["/assess"][1](_ctx(j, tmp_path), ["all"])
    # One assessment per playbook, as text or (past the message
    # length cap, the /recap pattern) as a document.
    if res.document is not None:
        text = res.document.read_text()
    else:
        text = res.text
    assert text.count("MARKET ASSESSMENT") == 3
    assert "VERDICT: SIGNAL" in text
    assert "SHADOW EVALUATION" in text


def test_assess_unknown_playbook_is_honest(tmp_path):
    j = _journal_with_records(tmp_path)
    res = MT5_COMMANDS["/assess"][1](_ctx(j, tmp_path), ["nope"])
    assert "no evaluation journaled for 'nope'" in res.text
    assert "qm_m5" in res.text  # it names what HAS been assessed


def test_assess_empty_journal(tmp_path):
    j = TradeJournal(tmp_path / "mt5_journal.jsonl")
    res = MT5_COMMANDS["/assess"][1](_ctx(j, tmp_path), [])
    assert "no playbook evaluations journaled yet" in res.text


def test_latest_evaluations_skips_trigger_events():
    recs = [_no_setup_record(),
            {"type": "evaluation", "kind": "trigger_event",
             "playbook": "__trigger__"},
            _signal_record()]
    latest = latest_evaluations(recs)
    assert set(latest) == {"fvg_midline", "qm_m5"}


# --------------------------- (c)+(e)+(f) runner end-to-end drive

def _drive_roster_and_range(tmp_path):
    """One scripted-frames drive: native M5 (live), the H1
    roster close sweep and the §G range family — the same shape
    as the Phase 2 roster-matrix drive."""
    runner = _wire(_qm_runner(tmp_path), _m5_fixture())
    h1 = _flat_h1(60, 95.0)
    h4 = _frame([(95.0, 95.5, 94.5, 95.0)] * 60, freq="4h")
    d1 = _frame([(95.0, 95.5, 94.5, 95.0)] * 40, freq="1d")
    m5 = _m5_fixture()
    runner._stack = SimpleNamespace(
        frames={"H1": h1, "H4": h4, "D1": d1, "M5": m5})
    runner._last_tick = _fresh_tick(m5)
    stats: dict = {}
    runner._run_qm_m5(m5, stats, source="live_close")
    runner._run_roster_close_evals(stats, source="live_close")
    runner._run_range_family("h4_generation", stats,
                             source="live_close")
    runner._run_range_family("h1_refinement", stats,
                             source="live_close")
    return runner


def _evals(runner, playbook=None):
    out = [r for r in runner.journal.records()
           if r.get("type") == "evaluation"]
    if playbook is not None:
        out = [r for r in out if r.get("playbook") == playbook]
    return out


def test_journal_records_carry_rendered_assessment_text(tmp_path):
    runner = _drive_roster_and_range(tmp_path)
    recs = _evals(runner)
    assert recs, "the drive must journal evaluations"
    for rec in recs:
        text = rec.get("assessment_text")
        assert text, f"{rec.get('playbook')} record lacks " \
                     "assessment_text"
        assert "MARKET ASSESSMENT" in text
        assert "VERDICT:" in text
        # The stored text is exactly the renderer's output for
        # the record as emitted (journal and bot can never
        # diverge).  The journal stamps ``ts`` at append time —
        # after the render — so the comparison nulls it out;
        # the renderer then uses the same provenance evaluation
        # timestamp the stored render used.
        assert text == render_assessment(
            {**rec, "ts": None}, symbol=runner.cfg.symbol)


def test_e2e_native_m5_signal_assessment(tmp_path):
    runner = _drive_roster_and_range(tmp_path)
    cands = [r for r in _evals(runner, "qm_m5")
             if r["outcome"] == "candidate_emitted"]
    assert cands, "the M5 fixture must produce a QM candidate"
    text = cands[-1]["assessment_text"]
    assert "VERDICT: SIGNAL" in text
    assert "qm_m5 · M5 · m5_close" in text
    assert "displacement_break" in text
    # Census-wiring pass premise correction: this live record
    # now carries the wired taxonomy read — the marker flipped
    # from the pre-wiring "absent" census fact to "recorded".
    assert "named_candlestick_taxonomy: recorded" in text


def test_e2e_h1_roster_no_setup_assessment(tmp_path):
    runner = _drive_roster_and_range(tmp_path)
    recs = _evals(runner, "fvg_midline")
    assert recs, "the roster sweep must evaluate fvg_midline"
    text = recs[-1]["assessment_text"]
    assert "VERDICT: NO-SETUP" in text
    assert "fvg_midline · H1" in text


def test_e2e_range_shadow_assessment_labelled(tmp_path):
    runner = _drive_roster_and_range(tmp_path)
    recs = _evals(runner, "range_qm")
    assert recs, "the §G family must evaluate range_qm"
    for rec in recs:
        assert is_shadow_record(rec)
        assert "SHADOW EVALUATION" in rec["assessment_text"]
        assert "VERDICT: SIGNAL" not in rec["assessment_text"]


def test_live_signal_announces_its_assessment(tmp_path):
    runner = _wire(_qm_runner(tmp_path), _m5_fixture())
    sent: list[str] = []
    runner._send_alert = lambda text, **kw: sent.append(text)
    runner._run_qm_m5(_m5_fixture(), {}, source="live_close")
    assessments = [t for t in sent if "MARKET ASSESSMENT" in t]
    assert assessments, "a live signal verdict must announce " \
                        "its rendered assessment"
    assert "VERDICT: SIGNAL" in assessments[-1]


def test_warmup_signal_never_announces(tmp_path):
    runner = _wire(_qm_runner(tmp_path), _m5_fixture())
    sent: list[str] = []
    runner._send_alert = lambda text, **kw: sent.append(text)
    runner._run_qm_m5(_m5_fixture(), {}, source="warmup_scan")
    assert [t for t in sent if "MARKET ASSESSMENT" in t] == []


def test_announce_guards_refusals_and_shadow(tmp_path):
    runner = _wire(_qm_runner(tmp_path), _m5_fixture())
    sent: list[str] = []
    runner._send_alert = lambda text, **kw: sent.append(text)
    refused = _signal_record()
    refused["assessment"] = dict(refused["assessment"],
                                 verdict="rejected")
    runner._announce_assessment(refused)
    shadow = _shadow_record()
    shadow["evaluation_source"] = "live_close"
    runner._announce_assessment(shadow)
    runner._announce_assessment(None)
    runner._announce_assessment({})
    assert sent == []
