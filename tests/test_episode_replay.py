"""Workstream 4 (owner approval 2026-11-04): EPISODE REPLAY as
full-system acceptance.

The battery re-runs recorded history through the CURRENT
engine code (WS1 + WS2 + WS3) using only information available
at the time, and grades the chain against the live journal.
These tests pin, on journal-exact fixtures
(goldenswing/backtest/episodes/):

- the TIMESTAMP LAW: the Nov 3 17:00Z bar's forecast was
  journaled in restart catch-up at 18:51:56Z — invisible to
  any decision at bar time, consumable only from its
  forecast_available_ts; as-of-bar and operationally-available
  results are separate facts;
- WS1: fed the journal's own assessment stream, the current
  ThesisKeeper invalidates the bearish thesis at 15:00Z on the
  thesis's OWN invalidation price and orphans its child — the
  named revision the live journal (three per-run keepers)
  never produced;
- WS2: the episode's two journal-exact Hot Scalp exhibits —
  the blocked long whose continuation objective was there all
  along becomes a candidate (TP1 net R/R 1.456), and the one
  whose only objective prices 1.126 stays blocked (the 1.20
  floor was not lowered, no target invented);
- WS3: both of the window's scan candidates — live
  disposition observed / not_attempted, then silence — are
  preserved as tracked instances and end in named terminals
  (inversion long: expired; order_block short: invalidated by
  the close rule), with zero orders placed by the scan path;
- regime coverage: the Oct 29 trend day (4/4 accepted shorts
  filled, +3.8687R) and the Oct 28 range day (5/6 signals
  gated, sleeve filled nothing) — every journaled chain ends
  in a named disposition under the current funnel taxonomy.
"""
from __future__ import annotations

import json

import pytest

from goldenswing.backtest.episode_replay import (
    HindsightError, Tape, forecast_leg, forecast_records,
    handoff_leg, hot_scalp_leg, load_episode, run_all,
    render_markdown, signal_chain_leg, thesis_leg)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return run_all(tmp_path_factory.mktemp("episode-replay"))


def _episode(results, name):
    return next(r for r in results if r["name"] == name)


# ------------------------------------------------------ timestamp law

def test_forecast_four_timestamps_separated():
    ep = load_episode("centerpiece")
    recs = forecast_records(ep)
    center = next(r for r in recs
                  if r["forecast_bar_ts"].startswith("2026-11-03T17:00")
                  and r["cadence"] == "D1")
    assert center["forecast_bar_ts"] == "2026-11-03T17:00:00+00:00"
    assert center["forecast_input_cutoff_ts"] == \
        "2026-11-03T18:00:00+00:00"
    # generated_at rides verbatim from the journal record (the
    # D1 cadence anchor), never silently replaced by bar time.
    assert center["forecast_generated_ts"] == \
        "2026-11-02 00:00:00+00:00"
    assert center["forecast_available_ts"] == \
        "2026-11-03T18:51:56.817346+00:00"
    assert center["forecast_bar_ts"] < \
        center["forecast_input_cutoff_ts"] < \
        center["forecast_available_ts"]
    assert center["horizons"]["24"]["up"] == pytest.approx(85.1485, abs=1e-3)


def test_forecast_invisible_until_available():
    ep = load_episode("centerpiece")
    _, facts = forecast_leg(ep)
    # Journaled in restart catch-up: at bar time and through
    # the whole 17:00 hour, NO decision could have consumed it.
    assert facts["visible_at_bar_time"] is False
    assert facts["visible_at_1852"] is True
    assert facts["visible_at_scan_0600"] is True
    # As-of-bar (hindsight grading) kept separate from
    # operationally-available.
    assert facts["as_of_bar"]["realized_12h_run_up"] == 95.35
    assert facts["operational"]["hours_available_before_0600_scan"] == \
        pytest.approx(11.13, abs=0.01)


def test_tape_refuses_hindsight():
    rec = {"available_ts": "2026-11-03T18:51:56+00:00", "x": 1}
    tape = Tape([rec])
    assert tape.view("2026-11-03T17:59:00+00:00") == []
    assert tape.view("2026-11-03T18:52:00+00:00") == [rec]
    with pytest.raises(HindsightError):
        tape.require(rec, "2026-11-03T18:00:00+00:00")


# ------------------------------------------------------------ WS1 leg

def test_thesis_leg_invalidates_on_own_terms(tmp_path):
    ep = load_episode("centerpiece")
    _, facts = thesis_leg(ep)
    events = facts["events"]
    kinds = [e["event"] for e in events]
    assert kinds == ["thesis_created", "thesis_child_registered",
                     "thesis_maintained", "thesis_invalidated",
                     "thesis_orphaned", "thesis_created",
                     "thesis_maintained"]
    inv = next(e for e in events if e["event"] == "thesis_invalidated")
    # 15:00Z — the journaled price fact 3944.595 crossed the
    # thesis's own invalidation price (the fvg_limit child's
    # stop, 3944.4220892857143), in the same cycle whose
    # assessment carried the erosion (peak 0.7568 -> 0.5337).
    assert inv["record_ts"] == "2026-11-03 15:00:00+00:00"
    assert "invalidation price 3944.42" in inv["reason"]
    # The live journal's keeper(s) produced NO revision event
    # of any kind across the whole journal.
    live = [e["event"] for e in facts["live_events"]]
    assert live == ["thesis_created", "thesis_maintained",
                    "thesis_created", "thesis_created"]
    # Gate probes at the two scan moments of Nov 4: the long is
    # no child of the bearish thesis (independent); the short
    # is adopted by the successor thesis (child) — the gate is
    # not what saves it; WS3's close rule is (handoff leg).
    gates = {g["playbook"]: g["verdict"]["verdict"]
             for g in facts["gates"]}
    assert gates == {"inversion": "independent",
                     "order_block": "child"}


# ------------------------------------------------------------ WS3 leg

def test_handoff_leg_every_candidate_named_terminal(tmp_path):
    ep = load_episode("centerpiece")
    _, facts = handoff_leg(ep, tmp_path)
    finals = facts["finals"]
    assert len(finals) == 2
    by_playbook = {f["playbook"]: f for f in finals.values()}
    ob = by_playbook["order_block"]
    assert ob["stage"] == "invalidated"
    assert ob["history"] == ["candidate_detected", "setup_armed",
                             "invalidated"]
    assert ob["entry"] == 3972.506 and ob["invalidation"] == 3999.328
    inv = by_playbook["inversion"]
    assert inv["stage"] == "expired"
    assert inv["history"] == ["candidate_detected", "setup_armed",
                              "expired"]
    assert all(f["terminal"] for f in finals.values())
    # Registration is not authority: the scan path placed no
    # orders in replay, exactly as designed.
    assert facts["orders_placed"] == 0


# ------------------------------------------------------------ WS2 leg

def test_hot_scalp_leg_exhibits(tmp_path):
    ep = load_episode("centerpiece")
    _, facts = hot_scalp_leg(ep, tmp_path)
    ex = {e["id"]: e for e in facts["exhibits"]}
    a = ex["A"]
    # Live: blocked at net_rr 0.0764 against a spent
    # micro-swing.  Current engine: the creation hierarchy is
    # priced instead — TP1 selected at 1.456 net R/R.
    assert a["event"] == "candidate_shadow"
    assert a["hierarchy"] == [3955.51, 3964.50]
    assert a["legs"][0]["price"] == 3955.51
    assert a["legs"][0]["verdict"] == "selected"
    assert a["legs"][0]["net_rr"] == pytest.approx(1.4562, abs=1e-3)
    assert a["legs"][1]["verdict"] == "viable"
    assert a["legacy"]["net_rr"] == pytest.approx(0.0764, abs=1e-3)
    assert a["legacy"]["verdict"] == "blocked"
    b = ex["B"]
    # The honest block: one true objective, 1.126 < 1.20.
    assert b["event"] == "candidate_blocked"
    assert b["plan_kind"] == "blocked"
    assert b["legs"] == [{"price": 4022.522, "verdict": "rejected",
                          "net_rr": pytest.approx(1.1264, abs=1e-3)}]
    assert b["headline_net_rr"] == pytest.approx(0.0963, abs=1e-3)
    # The window census behind the exhibits (journal facts).
    assert facts["geometry_blocked_by_direction"] == {"1": 78, "-1": 9}
    assert facts["geometry_blocked_total"] == 121


# ------------------------------------------------------- regime chains

def test_trend_episode_chains():
    ep = load_episode("trend_oct29")
    _, facts = signal_chain_leg(ep)
    assert facts["filled"] == 4
    assert facts["rejected"] == 6
    assert facts["accepted_unfilled"] == 0
    # The day the engine handled: four accepted shorts, all
    # positive — journal ground truth, summed exactly.
    assert facts["filled_r_sum"] == pytest.approx(3.8687, abs=1e-3)
    for chain in facts["chains"]:
        assert chain["terminal"] in ("filled", "rejected",
                                     "accepted_unfilled")
        if chain["terminal"] == "rejected":
            assert chain["taxonomy"]["code"]
            assert chain["taxonomy"]["category"] is not None


def test_range_episode_chains_and_census(results):
    ep = load_episode("range_oct28")
    _, facts = signal_chain_leg(ep)
    assert facts["filled"] == 1
    assert facts["filled_r_sum"] == pytest.approx(0.5021, abs=1e-3)
    assert facts["rejected"] == 5
    res = _episode(results, "range_oct28")
    det = res["legs"]["detection"]
    # Two-sided arming, nothing filled: the rotation signature.
    assert det["hot_scalp_armed_by_direction"] == {"1": 64, "-1": 37}
    # Detection was not silent — 30 candidates were emitted
    # (24 of them H1 roster scans) and every one was refused
    # downstream; none became a fill.
    assert det["eval_census"]["candidate_emitted"] == 30
    assert det["eval_census"]["no_setup"] == 690


def test_centerpiece_chains_accounted(results):
    res = _episode(results, "centerpiece")
    s = res["legs"]["signals"]
    # 2 filled (the 12:20 spike-stopped short -1.0001R and the
    # night-dip short +1.2609R), 3 accepted-but-unfilled,
    # 5 refused below the floor.
    assert s["filled"] == 2
    assert s["filled_r_sum"] == pytest.approx(0.2608, abs=1e-3)
    assert s["accepted_unfilled"] == 3
    assert s["rejected"] == 5


# ------------------------------------------------------------ battery

def test_battery_verdicts_and_residuals(results):
    center = _episode(results, "centerpiece")
    v = center["verdicts"]
    assert v["WS1 thesis revision"]["verdict"] == "DEMONSTRATED"
    assert v["WS2 Hot Scalp targets"]["verdict"] == "DEMONSTRATED"
    assert v["WS3 H1 handoff"]["verdict"] == "DEMONSTRATED"
    assert v["Timestamp law / forecast availability"]["verdict"] == \
        "DEMONSTRATED"
    # Honest residuals are findings, not decorations: the
    # assessment-stream discontinuity and the unexercised
    # weakening path are named in the battery's own output.
    text = " ".join(center["residuals"])
    assert "Assessment-stream discontinuity" in text
    assert "Weakening is not exercised" in text
    for res in results:
        assert res["stage_table"]
        assert res["residuals"]
    json.dumps(results, default=str)  # report payload serializes


def test_report_renders_the_law_and_verdicts(results):
    md = render_markdown(results)
    for needle in ("forecast_bar_ts", "forecast_input_cutoff_ts",
                   "forecast_generated_ts", "forecast_available_ts",
                   "WS1 thesis revision", "WS2 Hot Scalp targets",
                   "WS3 H1 handoff", "Honest residuals", "95.35"):
        assert needle in md
