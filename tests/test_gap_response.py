"""Gap response: a coverage gap names the missing behaviour,
proposes a candidate detector, and the detector is evaluated
on chronological data — with the outcome journaled beside
the gap.  These tests pin the proposal rules, the evaluation
outcomes (including the no-lookahead rule), the episode
semantics, the deliberate-gap refusal, and the cycle
integration."""
from __future__ import annotations

from goldenswing.execution import coverage_report as cr
from goldenswing.execution import gap_response as gr


def _rec(ts, tf, bias, phase, atr, low, high, rels=None):
    """One journal-shaped situation record."""
    return {
        "ts": ts, "type": "situation", "tf": tf,
        "state": {
            "tf": tf, "bias": bias, "structure_phase": phase,
            "volatility": {"atr": atr},
            "swings": {
                "confirmed_low": (
                    {"price": low, "ts": ts} if low is not None
                    else None),
                "confirmed_high": (
                    {"price": high, "ts": ts} if high is not None
                    else None),
            },
        },
        "relationships": rels,
    }


def _gap(key, config, classification="unwatched",
         first_seen="2026-11-10 10:00:00+00:00"):
    return {"config_key": key, "configuration": config,
            "classification": classification,
            "classification_basis": "test",
            "first_seen_ts": first_seen,
            "label": config.get("label") or key}


_TF_CONFIG = {"key": "tf:M15:bias=-1:phase=clean_trend",
              "kind": "tf", "tf": "M15", "bias": -1,
              "phase": "clean_trend",
              "label": "M15 measured state: bias -1, "
                       "structure phase clean_trend"}


# ------------------------------------------------------------ proposals

def test_propose_trend_continuation_down_prices_from_state():
    state = {"swings": {
        "confirmed_low": {"price": 4000.0, "ts": "t"},
        "confirmed_high": {"price": 4020.0, "ts": "t"}},
        "volatility": {"atr": 10.0}}
    det = gr.propose_detector(_TF_CONFIG, state)
    assert det["detector_key"] == "trend_continuation:M15:bias=-1"
    assert det["direction"] == -1
    assert det["target_price"] == 3990.0  # 4000 - 1.0 ATR
    assert det["invalidation_price"] == 4025.0  # 4020 + 0.5 ATR


def test_propose_trend_continuation_up_mirrors():
    config = {"key": "tf:H4:bias=+1:phase=clean_trend",
              "kind": "tf", "tf": "H4", "bias": 1,
              "phase": "clean_trend"}
    state = {"swings": {
        "confirmed_low": {"price": 4000.0, "ts": "t"},
        "confirmed_high": {"price": 4020.0, "ts": "t"}},
        "volatility": {"atr": 10.0}}
    det = gr.propose_detector(config, state)
    assert det["direction"] == 1
    assert det["target_price"] == 4030.0  # 4020 + 1.0 ATR
    assert det["invalidation_price"] == 3995.0  # 4000 - 0.5 ATR


def test_propose_without_state_states_rule_prices_pending():
    det = gr.propose_detector(_TF_CONFIG)
    assert det["detector_key"] == "trend_continuation:M15:bias=-1"
    assert det["target_price"] is None
    assert "1.0 ATR" in det["rule"]


def test_propose_correction_completion_direction_is_per_episode():
    config = {"key": "rel:H4->M15:correcting",
              "kind": "relationship", "htf": "H4", "ltf": "M15",
              "rel_kind": "correcting"}
    # No fixed direction at proposal: the direction is the
    # HTF bias as of each trigger, read at evaluation time.
    det = gr.propose_detector(config)
    assert det["detector_key"] == "correction_completion:H4->M15"
    assert det["direction"] is None
    assert det["direction_source"] == "htf_bias"
    assert det["target_price"] is None  # no trigger state yet


def test_propose_unknown_class_is_none():
    assert gr.propose_detector(
        {"kind": "tf", "tf": "M15", "bias": 0,
         "phase": "range"}) is None
    assert gr.propose_detector(
        {"kind": "relationship", "htf": "H4", "ltf": "M15",
         "rel_kind": "aligned"}) is None


# ------------------------------------------------------------ evaluation

def _trend_stream():
    """One M15 bias=-1 clean_trend episode (trigger 10:00),
    favourable swing to 3985 at 12:00, then a range record
    and a late record so the 6h window completes."""
    return [
        _rec("2026-11-10 09:00:00+00:00", "M15", 0, "range",
             10.0, 4005.0, 4025.0),
        _rec("2026-11-10 10:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 4000.0, 4020.0),
        _rec("2026-11-10 11:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 4000.0, 4020.0),
        _rec("2026-11-10 12:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 3985.0, 4020.0),
        _rec("2026-11-10 14:00:00+00:00", "M15", 0, "range",
             10.0, 3985.0, 4020.0),
        _rec("2026-11-10 20:00:00+00:00", "M15", 0, "range",
             10.0, 3985.0, 4020.0),
    ]


def test_evaluate_hit_first_threshold_wins():
    det = gr.propose_detector(_TF_CONFIG)
    ev = gr.evaluate_detector(det, _trend_stream())
    assert ev["episodes"] == 1  # 3 consecutive trigger records
    assert ev["evaluated"] == 1
    assert ev["hits"] == 1
    assert ev["hit_rate"] == 1.0
    assert ev["mean_favourable_atr"] == 1.5
    row = ev["episodes_detail"][0]
    assert row["outcome"] == "hit"
    assert row["target_first_ts"] == "2026-11-10 12:00:00+00:00"


def test_evaluate_invalidation_first_is_not_a_hit():
    stream = [
        _rec("2026-11-10 09:00:00+00:00", "M15", 0, "range",
             10.0, 4005.0, 4025.0),
        _rec("2026-11-10 10:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 4000.0, 4020.0),
        # adverse swing first: high 4026 crosses 4025 at 11:00
        _rec("2026-11-10 11:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 4000.0, 4026.0),
        # favourable swing later: low 3985 at 12:00
        _rec("2026-11-10 12:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 3985.0, 4026.0),
        _rec("2026-11-10 20:00:00+00:00", "M15", 0, "range",
             10.0, 3985.0, 4026.0),
    ]
    det = gr.propose_detector(_TF_CONFIG)
    ev = gr.evaluate_detector(det, stream)
    assert ev["evaluated"] == 1
    assert ev["invalidations"] == 1
    assert ev["hits"] == 0
    assert ev["episodes_detail"][0]["outcome"] == "invalidation"


def test_evaluate_neither_when_no_threshold_reached():
    stream = [
        _rec("2026-11-10 09:00:00+00:00", "M15", 0, "range",
             10.0, 4005.0, 4025.0),
        _rec("2026-11-10 10:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 4000.0, 4020.0),
        _rec("2026-11-10 12:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 3995.0, 4022.0),
        _rec("2026-11-10 20:00:00+00:00", "M15", 0, "range",
             10.0, 3995.0, 4022.0),
    ]
    det = gr.propose_detector(_TF_CONFIG)
    ev = gr.evaluate_detector(det, stream)
    assert ev["evaluated"] == 1
    assert ev["neither"] == 1
    assert ev["mean_favourable_atr"] == 0.5
    assert ev["mean_adverse_atr"] == 0.2


def test_evaluate_no_lookahead_swings_before_trigger_ignored():
    stream = [
        # favourable low 3985 exists BEFORE the trigger
        _rec("2026-11-10 09:00:00+00:00", "M15", 0, "range",
             10.0, 3985.0, 4025.0),
        _rec("2026-11-10 10:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 4000.0, 4020.0),
        _rec("2026-11-10 12:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 4000.0, 4020.0),
        _rec("2026-11-10 20:00:00+00:00", "M15", 0, "range",
             10.0, 4000.0, 4020.0),
    ]
    det = gr.propose_detector(_TF_CONFIG)
    ev = gr.evaluate_detector(det, stream)
    assert ev["evaluated"] == 1
    assert ev["hits"] == 0
    assert ev["neither"] == 1


def test_evaluate_incomplete_window_excluded_not_missed():
    stream = _trend_stream()[:3]  # stream ends 11:00, window ends 16:00
    det = gr.propose_detector(_TF_CONFIG)
    ev = gr.evaluate_detector(det, stream)
    assert ev["episodes"] == 1
    assert ev["evaluated"] == 0
    assert ev["excluded_incomplete"] == 1
    assert ev["hits"] == 0
    assert ev["hit_rate"] is None


def test_evaluate_no_reference_excluded():
    stream = [
        _rec("2026-11-10 09:00:00+00:00", "M15", 0, "range",
             10.0, 4005.0, 4025.0),
        _rec("2026-11-10 10:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, None, None),
        _rec("2026-11-10 20:00:00+00:00", "M15", 0, "range",
             10.0, 4005.0, 4025.0),
    ]
    det = gr.propose_detector(_TF_CONFIG)
    ev = gr.evaluate_detector(det, stream)
    assert ev["excluded_no_reference"] == 1
    assert ev["evaluated"] == 0


def test_evaluate_correction_completion_uses_htf_bias_per_episode():
    rel = [{"htf": "H4", "ltf": "M15", "kind": "correcting",
            "source": "engine"}]
    stream = [
        # H4 bias -1 as of the trigger; M15 correcting.
        _rec("2026-11-10 09:00:00+00:00", "H4", -1,
             "clean_trend", 40.0, 4000.0, 4100.0),
        _rec("2026-11-10 10:00:00+00:00", "M15", 1,
             "clean_trend", 10.0, 4000.0, 4020.0, rel),
        _rec("2026-11-10 12:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 3985.0, 4020.0, rel),
        _rec("2026-11-10 20:00:00+00:00", "M15", 0, "range",
             10.0, 3985.0, 4020.0),
    ]
    config = {"key": "rel:H4->M15:correcting",
              "kind": "relationship", "htf": "H4", "ltf": "M15",
              "rel_kind": "correcting"}
    det = gr.propose_detector(config)
    ev = gr.evaluate_detector(det, stream)
    assert ev["episodes"] == 1
    assert ev["evaluated"] == 1
    assert ev["hits"] == 1
    assert ev["episodes_detail"][0]["direction"] == -1


def test_episodes_rearm_after_absence():
    stream = _trend_stream() + [
        _rec("2026-11-11 10:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 3980.0, 4010.0),
        _rec("2026-11-11 20:00:00+00:00", "M15", 0, "range",
             10.0, 3980.0, 4010.0),
    ]
    det = gr.propose_detector(_TF_CONFIG)
    ev = gr.evaluate_detector(det, stream)
    assert ev["episodes"] == 2
    assert ev["evaluated"] == 2


def test_other_timeframes_do_not_break_episode():
    stream = [
        _rec("2026-11-10 10:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 4000.0, 4020.0),
        _rec("2026-11-10 10:05:00+00:00", "M5", 1,
             "clean_trend", 3.0, 4010.0, 4022.0),
        _rec("2026-11-10 11:00:00+00:00", "M15", -1,
             "clean_trend", 10.0, 3990.0, 4020.0),
        _rec("2026-11-10 20:00:00+00:00", "M15", 0, "range",
             10.0, 3990.0, 4020.0),
    ]
    det = gr.propose_detector(_TF_CONFIG)
    ev = gr.evaluate_detector(det, stream)
    assert ev["episodes"] == 1
    assert ev["hits"] == 1


# ------------------------------------------------------------ responses

def test_respond_unwatched_evaluates_when_stream_supplied():
    resp = gr.respond_to_gap(
        _gap(_TF_CONFIG["key"], _TF_CONFIG), _trend_stream())
    assert resp["status"] == "evaluated"
    assert resp["evaluation"]["hits"] == 1


def test_respond_without_stream_is_proposal_only():
    resp = gr.respond_to_gap(
        _gap(_TF_CONFIG["key"], _TF_CONFIG), None)
    assert resp["status"] == "proposed"
    assert resp["detector"]["detector_key"] == \
        "trend_continuation:M15:bias=-1"
    assert resp["evaluation"] is None


def test_respond_deliberate_gap_gets_no_detector():
    resp = gr.respond_to_gap(
        _gap(_TF_CONFIG["key"], _TF_CONFIG,
             classification="deliberate_data"), _trend_stream())
    assert resp["status"] == "standing_aside"
    assert resp["detector"] is None
    assert resp["evaluation"] is None


def test_respond_untemplated_class_names_itself():
    range_config = {"key": "tf:M15:bias=+0:phase=range",
                    "kind": "tf", "tf": "M15", "bias": 0,
                    "phase": "range"}
    resp = gr.respond_to_gap(
        _gap(range_config["key"], range_config), _trend_stream())
    assert resp["status"] == "untemplated"
    assert "range" in resp["reason"]


def test_cycle_responses_cache_and_regrow():
    state = gr.new_state()
    gaps = [_gap(_TF_CONFIG["key"], _TF_CONFIG)]
    short = _trend_stream()[:3]  # episode incomplete
    out = gr.cycle_responses(state, open_gaps=gaps,
                             situations=short)
    assert out[_TF_CONFIG["key"]]["evaluation"][
        "excluded_incomplete"] == 1
    # Same stream again: kept, not recomputed.
    out2 = gr.cycle_responses(state, open_gaps=gaps,
                              situations=short)
    assert out2[_TF_CONFIG["key"]] is out[_TF_CONFIG["key"]]
    # Longer stream: the fixed rule runs again, the episode
    # is now evaluable.
    out3 = gr.cycle_responses(state, open_gaps=gaps,
                              situations=_trend_stream())
    assert out3[_TF_CONFIG["key"]]["evaluation"]["hits"] == 1


def test_cycle_responses_classification_change_recomputes():
    state = gr.new_state()
    gaps = [_gap(_TF_CONFIG["key"], _TF_CONFIG)]
    out = gr.cycle_responses(state, open_gaps=gaps,
                             situations=_trend_stream())
    assert out[_TF_CONFIG["key"]]["status"] == "evaluated"
    gaps_now_deliberate = [_gap(
        _TF_CONFIG["key"], _TF_CONFIG,
        classification="deliberate_data")]
    out2 = gr.cycle_responses(
        state, open_gaps=gaps_now_deliberate,
        situations=_trend_stream())
    assert out2[_TF_CONFIG["key"]]["status"] == "standing_aside"


# ------------------------------------------------------------ cycle wiring

def _coverage_gap_record():
    return {"type": "coverage_gap",
            "config_key": _TF_CONFIG["key"],
            "configuration": dict(_TF_CONFIG),
            "label": _TF_CONFIG["label"],
            "first_seen_ts": "2026-11-10 10:00:00+00:00",
            "ts": "2026-11-10 10:00:00+00:00", "cycles": 3}


def test_cycle_journals_response_and_second_cycle_is_quiet():
    state = cr.new_state()
    rec = cr.cycle(state=state, records=[_coverage_gap_record()],
                   now="2026-11-10 10:00:00+00:00")
    assert rec is not None
    resp = rec["gap_responses"][_TF_CONFIG["key"]]
    assert resp["status"] == "proposed"  # no stream supplied
    rec2 = cr.cycle(state=state, records=[],
                    now="2026-11-10 10:00:00+00:00")
    assert rec2 is None  # nothing changed: no re-journal


def test_cycle_upgrades_proposal_when_stream_arrives():
    state = cr.new_state()
    cr.cycle(state=state, records=[_coverage_gap_record()],
             now="2026-11-10 10:00:00+00:00")
    rec = cr.cycle(state=state, records=[],
                   now="2026-11-10 21:00:00+00:00",
                   situations=_trend_stream())
    assert rec is not None
    resp = rec["gap_responses"][_TF_CONFIG["key"]]
    assert resp["status"] == "evaluated"
    assert resp["evaluation"]["hits"] == 1
    lines = cr.render_lines(state)
    assert any("trend_continuation:M15:bias=-1" in line
               and "1/1 hit" in line for line in lines)


def test_evaluate_records_with_stream_answers_gaps():
    snap = cr.evaluate_records(
        [_coverage_gap_record()], situations=_trend_stream(),
        now="2026-11-10 21:00:00+00:00")
    resp = snap["gap_responses"][_TF_CONFIG["key"]]
    assert resp["status"] == "evaluated"
    assert resp["evaluation"]["episodes"] == 1
