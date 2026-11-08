"""MarketSituation Pass 4 — measured priors (SPEC §5.3 +
AMENDMENTS-1 §A3).

Pins:
* shrinkage: eff_n -> 0 reproduces the stated constants;
  large effective n lets the measured rates dominate; the
  blend is on EFFECTIVE n, never raw n.
* the §A3 universe: taken + abstained + shadow-evaluated
  candidates all measure; an executed-only rate is impossible
  by construction (the fixture's abstained/shadow outcomes
  move the rate away from the executed-only value).
* correlation: a cluster of pairs inside one situation is
  ONE effective observation (eff_n < raw n, rates follow the
  episode-collapsed weight).
* sub-cells: below the effective-n floor the provider pools
  upward to the base cell; at/above it, resolution uses the
  sub-cell.
* the gate: holdout loss (or a thin holdout) holds the FIXED
  priors; a holdout win puts MEASURED priors in force, and
  the cycle's verdict record cites the calibration version
  it judged.
* purity: evaluate_judgment without priors_for is
  byte-identical to the pristine (b0f4adb) output; a stated
  provider changes no number, only adds the trace; a measured
  provider re-prices.
* the runner seam: the standing calibration state folds
  journal windows on the S3 cadence, journals deployment
  records, and _judgment_evaluate stamps the priors trace.
"""
from __future__ import annotations

import json

from goldenswing.execution import judgment_calibration as cal
from goldenswing.execution.judgment import evaluate_judgment
from goldenswing.execution.judgment_validation import compare_priors

from test_p3_trader import _runner


# ------------------------------------------------------------ fixtures
def _shadow(ts, *, playbook="qm_m5", tf="M5", decision="take",
            mode="active", authority=None, executed=True,
            structure="trend", regdir="up", vol="normal",
            scen=(0.50, 0.30, 0.20), situation=None):
    judgment = {
        "decision": decision, "expected_r": 0.4, "confidence": 0.6,
        "mode": mode,
        "market_state": {"structure": structure,
                         "regime_direction": regdir,
                         "volatility": vol},
        "scenarios": [
            {"name": "continuation", "p": scen[0], "payoff_r": 1.5},
            {"name": "reversal", "p": scen[1], "payoff_r": -1.0},
            {"name": "chop", "p": scen[2], "payoff_r": -0.1}],
    }
    if authority is not None:
        judgment["authority"] = authority
    row = {
        "type": "judgment_shadow", "ts": ts, "playbook": playbook,
        "intended_tf": tf,
        "gate_result": None,
        "gate_order_result": "filled" if executed else None,
        "gate_reason_code": None,
        "judgment": judgment,
    }
    if situation is not None:
        row["situation_id"] = situation
    return row


def _recon(ts, *, playbook="qm_m5", r=1.2, direction=1):
    return {
        "type": "judgment_reconciliation", "judged_ts": ts,
        "playbook": playbook, "direction": direction,
        "decision": "take", "horizon_h": 4, "entry": 4000.0,
        "exit_price": 4000.0 + r, "realized_r": r, "move": r,
        "move_atr": 0.5, "hit": r > 0,
    }


def _ts(i: int) -> str:
    return f"2026-10-{1 + i // 24:02d} {i % 24:02d}:00:00"


def _dataset(n, *, outcome_r, **shadow_kw):
    records = []
    for i in range(n):
        ts = _ts(i)
        r = outcome_r(i) if callable(outcome_r) else outcome_r
        records.append(_shadow(ts, **shadow_kw))
        records.append(_recon(
            ts, r=r, playbook=shadow_kw.get("playbook", "qm_m5")))
    return records


# ------------------------------------------------------------ shrinkage
def test_shrinkage_tiny_evidence_is_effectively_stated():
    pairs = cal.build_calibration_pairs(_dataset(1, outcome_r=1.2))
    body = cal.calibrate(pairs)
    cell = body["cells"]["trend|with_bias"]
    assert cell["eff_n"] == 1.0
    # (1 * measured + 100 * stated) / 101 per class.
    assert cell["priors"]["continuation"] == round(
        (1.0 * 1.0 + 100 * 0.42) / 101, 6)
    assert abs(cell["priors"]["continuation"] - 0.42) < 0.01


def test_shrinkage_large_n_lets_measured_dominate():
    pairs = cal.build_calibration_pairs(
        _dataset(400, outcome_r=lambda i: 1.2 if i % 4 else -1.1))
    cell = cal.calibrate(pairs)["cells"]["trend|with_bias"]
    assert cell["eff_n"] == 400.0
    assert cell["measured"]["continuation"] == 0.75
    assert cell["priors"]["continuation"] == round(
        (400 * 0.75 + 100 * 0.42) / 500, 6)
    assert cell["priors"]["continuation"] > 0.65


def test_no_pairs_no_cells_and_provider_is_stated():
    body = cal.calibrate([])
    assert body["cells"] == {}
    provider = cal.PriorsProvider(
        calibration=body, verdict={"verdict": "measured_wins"},
        mode="calibrated")
    got = provider.for_cell("trend", "with_bias")
    assert got["source"] == "stated"
    assert got["continuation"] == 0.42
    assert got["reversal"] == 0.38
    assert got["chop"] == 0.20
    assert got["n"] == 0.0


# ------------------------------------------------------------ §A3 universe
def test_universe_includes_abstained_and_shadow():
    records = []
    # 10 executed takes, all reversal outcomes.
    for i in range(10):
        ts = _ts(i)
        records += [_shadow(ts, decision="take", executed=True),
                    _recon(ts, r=-1.1)]
    # 30 abstained / shadow-evaluated, all continuation.
    for i in range(10, 40):
        ts = _ts(i)
        records += [
            _shadow(ts, decision="abstain", executed=False,
                    mode="shadow"),
            _recon(ts, r=1.2)]
    pairs = cal.build_calibration_pairs(records)
    body = cal.calibrate(pairs)
    universe = body["universe"]
    assert universe["n_classed"] == 40
    assert universe["verdict_take"] == 10
    assert universe["verdict_abstain"] == 30
    assert universe["gate_executed"] == 10
    assert universe["shadow_only"] == 30
    cell = body["cells"]["trend|with_bias"]
    # The universe rate: 30 of 40 continuations.  An
    # executed-only rate would be 0.0 — the two must differ,
    # and the calibration must carry the universe value.
    assert cell["measured"]["continuation"] == 0.75
    assert cell["measured"]["continuation"] != 0.0


# ------------------------------------------------------------ §A3 correlation
def test_correlated_cluster_counts_once():
    records = []
    # Five pairs inside ONE situation, all continuation.
    for i in range(5):
        ts = _ts(i)
        records += [_shadow(ts, situation="sit-A"),
                    _recon(ts, r=1.2)]
    # Five pairs in five distinct situations, all reversal.
    for i in range(5, 10):
        ts = _ts(i)
        records += [_shadow(ts, situation=f"sit-{i}"),
                    _recon(ts, r=-1.1)]
    pairs = cal.build_calibration_pairs(records)
    cell = cal.calibrate(pairs)["cells"]["trend|with_bias"]
    assert cell["n"] == 10
    assert cell["eff_n"] == 6.0
    assert cell["eff_n"] < cell["n"]
    # Episode-collapsed: the cluster is one continuation vote
    # against five reversal votes — not five against five.
    assert cell["measured"]["continuation"] == round(1 / 6, 6)
    # And the blend follows effective n, not raw n.
    assert cell["priors"]["continuation"] == round(
        (6 * (1 / 6) + 100 * 0.42) / 106, 6)


# ------------------------------------------------------------ §A3 sub-cells
def _measured_provider(pairs, **kw):
    body = cal.calibrate(pairs)
    return cal.PriorsProvider(
        calibration=body, verdict={"verdict": "measured_wins"},
        mode="calibrated", version=7)


def test_subcell_below_floor_pools_upward():
    # 10 effective in the family sub-cell, 60 in the base cell.
    records = _dataset(10, outcome_r=1.2, playbook="rare_pb")
    records += _dataset(50, outcome_r=-1.1, playbook="other_pb")
    # renumber the second batch's timestamps to stay unique
    for j, row in enumerate(records):
        if j >= 20:
            row["ts"] = _ts(100 + j)
            row["judged_ts"] = _ts(100 + j)
    pairs = cal.build_calibration_pairs(records)
    body = cal.calibrate(pairs)
    sub = body["sub_cells"]["trend|with_bias@family=rare_pb"]
    assert sub["eff_n"] == 10.0
    assert sub["cleared"] is False
    provider = _measured_provider(pairs)
    got = provider.for_cell("trend", "with_bias",
                            playbook="rare_pb", intended_tf="M5")
    assert got["cell"] == "trend|with_bias"
    assert got["source"] == "measured"


def test_subcell_above_floor_resolves():
    records = _dataset(30, outcome_r=1.2, playbook="busy_pb")
    records += _dataset(30, outcome_r=-1.1, playbook="other_pb")
    for j, row in enumerate(records):
        if j >= 60:
            row["ts"] = _ts(100 + j)
            row["judged_ts"] = _ts(100 + j)
    pairs = cal.build_calibration_pairs(records)
    body = cal.calibrate(pairs)
    sub = body["sub_cells"]["trend|with_bias@family=busy_pb"]
    assert sub["eff_n"] == 30.0
    assert sub["cleared"] is True
    provider = _measured_provider(pairs)
    got = provider.for_cell("trend", "with_bias",
                            playbook="busy_pb", intended_tf="M5")
    assert got["cell"] == "trend|with_bias@family=busy_pb"
    assert got["n"] == 30.0
    # The sub-cell's own measured rate (all continuation),
    # not the pooled base rate (half).
    assert got["continuation"] == round(
        (30 * 1.0 + 100 * 0.42) / 130, 6)


# ------------------------------------------------------------ the gate
def _gate_dataset(holdout_r, n_eval=70, n_holdout=30):
    records = []
    total = n_eval + n_holdout
    for i in range(total):
        ts = f"2026-10-{1 + i // 24:02d} {i % 24:02d}:00:00" \
            if i < 96 else f"2026-11-{1 + (i - 96) // 24:02d} " \
                           f"{i % 24:02d}:00:00"
        r = 1.2 if i < n_eval else holdout_r
        records.append(_shadow(ts))
        records.append(_recon(ts, r=r))
    return records


def test_holdout_loss_holds_fixed_priors():
    # Eval segment: continuations (measured learns to love
    # continuation).  Holdout: reversals — the measured side
    # prices them worse than the stated baseline.
    verdict = compare_priors(_gate_dataset(holdout_r=-1.1))
    assert verdict["verdict"] == "fixed_holds"
    assert verdict["holdout_n"] >= 20
    assert verdict["measured"]["log_loss"] \
        > verdict["stated"]["log_loss"]
    provider = cal.PriorsProvider(
        calibration=cal.calibrate(
            cal.build_calibration_pairs(_gate_dataset(-1.1))),
        verdict=verdict, mode="calibrated")
    assert provider.in_force == "stated"
    assert provider.for_cell("trend", "with_bias")["source"] \
        == "stated"


def test_holdout_win_puts_measured_in_force():
    verdict = compare_priors(_gate_dataset(holdout_r=1.2))
    assert verdict["verdict"] == "measured_wins"
    assert verdict["measured"]["log_loss"] \
        < verdict["stated"]["log_loss"]


def test_thin_holdout_is_insufficient():
    verdict = compare_priors(_dataset(30, outcome_r=1.2))
    assert verdict["verdict"] == "insufficient_data"


def test_cycle_journals_records_and_verdict_cites_version():
    state = cal.new_state()
    records = _gate_dataset(holdout_r=1.2)
    out = cal.cycle(state, records, now=__import__("datetime")
                    .datetime(2026, 11, 5), bar="2026-11-05 00:00")
    types = [r["type"] for r in out]
    assert "judgment_priors" in types
    assert "judgment_priors_verdict" in types
    calibration = next(r for r in out
                       if r["type"] == "judgment_priors")
    verdict = next(r for r in out
                   if r["type"] == "judgment_priors_verdict")
    assert verdict["verdict"] == "measured_wins"
    assert verdict["calibration_version"] == calibration["version"]
    provider = cal.provider_from_state(state, mode="calibrated")
    assert provider.in_force == "measured"
    got = provider.for_cell("trend", "with_bias")
    assert got["source"] == "measured"
    assert got["version"] == calibration["version"]
    # A quiet second cycle journals nothing new.
    out2 = cal.cycle(state, [], now=__import__("datetime")
                     .datetime(2026, 11, 5, 1), bar="2026-11-05 01:00")
    assert out2 == []


def test_cycle_dedupes_repeated_windows():
    state = cal.new_state()
    records = _dataset(6, outcome_r=1.2)
    from datetime import datetime
    cal.cycle(state, records, now=datetime(2026, 10, 2),
              bar="2026-10-02 00:00")
    n_pairs = len(state["pairs"])
    cal.cycle(state, records, now=datetime(2026, 10, 2, 1),
              bar="2026-10-02 01:00")
    assert len(state["pairs"]) == n_pairs == 6


# ------------------------------------------------------------ assessor purity
_CAND = {"direction": 1, "entry": 2000.0, "stop": 1990.0,
         "targets": [2020.0], "asymmetry_score": 72.0}
_DEC = {"floor_result": {"total": 72.0, "floor": 50.0}}
_EV = {"observed": {"bias_d1": 1, "bias_h4": 1, "stop_dist": 10.0,
                    "session_bucket": "london"},
       "inferred": {"regime": "clean_trend", "posture": "aligned",
                    "score_total": 72.0, "room_r": 2.0},
       "evidence_groups": {"htf_structure": ["bos up"],
                           "zone_context": ["at demand"]},
       "supporting": ["bos up", "at demand"], "contradictory": []}
_CTX = {"regime": "up/clean_trend/normal/expansion/london",
        "adr_pct": 40.0, "adr_state": "fresh", "posture": "aligned"}

#: The pristine (b0f4adb) output for the fixture above —
#: generated from the pre-Pass-4 tree; the no-provider call
#: must reproduce it exactly.
_PRISTINE = {
    "confidence": 0.5781, "decision": "take",
    "evidence_balance": {"against": 0.0, "groups_against": [],
                         "groups_for": ["htf_structure",
                                        "zone_context"],
                         "support": 2.5},
    "expected_r": 1.8562,
    "market_state": {"adr_pct": 40.0, "adr_state": "fresh",
                     "bias_alignment": 2, "day_type": None,
                     "journey_state": None, "phase": "expansion",
                     "posture": "aligned",
                     "regime_direction": "up", "room_r": 2.0,
                     "session": "london", "smt": None,
                     "structure": "trend", "volatility": "normal"},
    "mode": "shadow", "opposition": 0.0,
    "reason": "continuation leads (p=0.94 vs reversal 0.02); "
              "expected +1.86R at 2.00R to the first target on "
              "2.5 vs 0.0 weighed evidence",
    "reason_category": 4,
    "scenarios": [
        {"name": "continuation", "p": 0.9401, "payoff_r": 2.0},
        {"name": "reversal", "p": 0.0201, "payoff_r": -1.0},
        {"name": "chop", "p": 0.0397, "payoff_r": -0.1}],
    "score": {"edge": 0.44, "floor": 50.0, "total": 72.0},
    "support_score": 4.16,
    "thresholds": {"min_confidence": 0.15, "min_expected_r": 0.05},
}


def test_no_provider_is_byte_identical_to_pristine():
    out = evaluate_judgment(candidate=_CAND, decision=_DEC,
                            evidence=_EV, context=_CTX)
    assert out == _PRISTINE
    assert "priors" not in out


def test_stated_provider_changes_no_number_only_trace():
    provider = cal.PriorsProvider(mode="stated")
    out = evaluate_judgment(
        candidate=_CAND, decision=_DEC, evidence=_EV,
        context=_CTX, priors_for=provider.for_cell)
    trace = out.pop("priors")
    assert out == _PRISTINE
    assert trace == {"source": "stated", "version": "stated",
                     "cell": "trend|with_bias", "n": 0.0}


def test_measured_provider_reprices_and_traces():
    body = cal.calibrate(cal.build_calibration_pairs(
        _dataset(300, outcome_r=1.2)))
    provider = cal.PriorsProvider(
        calibration=body, verdict={"verdict": "measured_wins"},
        mode="calibrated", version=3)
    out = evaluate_judgment(
        candidate=_CAND, decision=_DEC, evidence=_EV,
        context=_CTX, priors_for=provider.for_cell)
    assert out["priors"]["source"] == "measured"
    assert out["priors"]["version"] == 3
    assert out["priors"]["cell"].startswith("trend|with_bias")
    cont = next(s for s in out["scenarios"]
                if s["name"] == "continuation")
    assert cont["p"] != _PRISTINE["scenarios"][0]["p"]


def test_rel_dir_derivation_agrees_with_calibration():
    seen = {}

    def probe(*, structure, rel_dir, volatility):
        seen.update(structure=structure, rel_dir=rel_dir,
                    volatility=volatility)
        return {**cal.STATED_PRIORS, "source": "stated",
                "version": "stated", "cell": "x", "n": 0.0}

    for word, direction in (("up", 1), ("up", -1), ("down", -1),
                            ("down", 1), ("flat", 1)):
        ctx = dict(_CTX)
        ctx["regime"] = f"{word}/clean_trend/normal/expansion/london"
        cand = dict(_CAND, direction=direction)
        evaluate_judgment(candidate=cand, decision=_DEC,
                          evidence=_EV, context=ctx,
                          priors_for=probe)
        assert seen["rel_dir"] == cal.rel_direction(word, direction)
        assert seen["structure"] == "trend"
        assert seen["volatility"] == "normal"


# ------------------------------------------------------------ runner seam
def _journal_rows(runner):
    path = runner.cfg.journal_path
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


def test_runner_default_trace_is_stated(tmp_path):
    runner = _runner(tmp_path)
    bundle = runner._judgment_evaluate(
        candidate=_CAND, decision=_DEC, evidence=_EV,
        intended_tf="M5", eval_now="2026-10-14 12:00:00")
    trace = bundle["judgment"]["priors"]
    assert trace["source"] == "stated"
    assert trace["version"] == "stated"
    assert trace["n"] == 0.0


def test_runner_first_phase_journals_deployment_records(tmp_path):
    runner = _runner(tmp_path)
    runner._priors_phase([], "2026-10-14 13:00:00")
    types = [r["type"] for r in _journal_rows(runner)]
    assert "judgment_priors" in types
    assert "judgment_priors_verdict" in types


def test_runner_measured_in_force_after_winning_windows(tmp_path):
    runner = _runner(tmp_path, judgment_priors="calibrated")
    # The desk's decision context is an input, not the seam
    # under test: pin it to the fixture's so the evaluated
    # candidate lands in the measured cell.
    runner._decision_context = lambda: dict(_CTX)
    records = _gate_dataset(holdout_r=1.2)
    runner._priors_phase(records, "2026-11-05 00:00:00")
    provider = runner._priors_provider()
    assert provider is not None
    assert provider.in_force == "measured"
    bundle = runner._judgment_evaluate(
        candidate=_CAND, decision=_DEC, evidence=_EV,
        intended_tf="M5", eval_now="2026-11-05 01:00:00")
    assert bundle["judgment"]["priors"]["source"] == "measured"
    # And with the config gate shut, the same state prices
    # stated — the config is the owner's kill switch.
    runner.cfg.judgment_priors = "stated"
    bundle = runner._judgment_evaluate(
        candidate=_CAND, decision=_DEC, evidence=_EV,
        intended_tf="M5", eval_now="2026-11-05 02:00:00")
    assert bundle["judgment"]["priors"]["source"] == "stated"
