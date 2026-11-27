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
* the verdict no longer takes priors (owner order
  2026-11-27 — the formula core is RETIRED): the
  calibration module measures priors from the journal
  for the validation reports, which read it directly.
  Only the config default (calibrated) is pinned here.
"""
from __future__ import annotations

import json

from goldenswing.execution import judgment_calibration as cal
from goldenswing.execution.judgment_validation import compare_priors


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


# ------------------------------------------------------------ runner seam
# The verdict no longer takes priors (owner order
# 2026-11-27 — the formula core is RETIRED): the
# calibration module measures priors from the journal
# for the validation reports, which read it directly.
# Only the config default is pinned here.


def test_calibrated_is_the_default_priors_mode():
    """E3, earned: the full live journal (2026-11-14, 108
    calibration pairs, 30 on the locked holdout) returns
    measured_wins — measured log-loss 1.3732 vs stated
    1.3796 — so the Core prices the calibrated priors by
    default. The fallback is automatic: no verdict or a loss,
    and the stated constants price (pinned by the test
    above). The config stays the owner's kill switch."""
    from goldenswing.execution.mt5_runner import MT5RunnerConfig
    assert MT5RunnerConfig().judgment_priors == "calibrated"
