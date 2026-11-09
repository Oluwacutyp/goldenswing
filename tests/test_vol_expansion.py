"""MarketSituation Pass 7a — the volatility-expansion family.

Pins, in the GS conventions (fixtures judged by what was
observable AT THE TIME — every detector assertion runs the
causal read at the bar itself, and one test proves the read
at bar j is identical on the frame truncated at j):

- the measurement core (live/vol_expansion.py): a coil is a
  Bollinger squeeze RUN with ATR contracting, measured on
  the formation frame; the state machine resolves
  coil -> trigger -> acceptance (escape) / rejection
  (failed expansion) / expiry (drift, no hypothesis);
- parity: the detector's ATR and atr_ratio at a bar equal
  live/situation.build_tf_state's on identical bars,
  exactly (the situation engine's own construction);
- the playbooks: signals only with multi-timeframe
  corroboration (a threaded situation's OTHER timeframe
  contracting); geometry (entry at the acceptance /
  rejection close, stop inside the coil / beyond the
  extreme, targets ahead of the entry); the arm read;
- the route contract: both routes DECLARED shadow (Pass 5
  machinery), governing in no mode, no order path exists
  for the partition, undeclared names still fail closed;
- the Pass 6 management shape: the declared block runs the
  intraday standard plan on the FORMATION clock (the
  signal's own declaration), the book takes the instance's
  clocks, and the plan freezes at entry;
- the runner wiring: scheduled scans journal evaluations
  (situation-stamped), route-stamped judgment_shadow
  verdicts, arm records, and book scenarios — and place no
  orders;
- the promotion evaluation (execution/vol_promotion.py):
  the §G evaluator over the family's own route-stamped
  history; promotion_ready is a recommendation, binds
  nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import goldenswing.execution.mt5_runner as mr
from goldenswing.execution import range_promotion, vol_promotion
from goldenswing.execution.mt5_runner import (
    JUDGMENT_ROUTE_PLAYBOOK, management_block_for)
from goldenswing.execution.playbook_registry import (
    FLOOR_AUTHORITY, NATIVE_SPECS, PLAYBOOK_REGISTRY, ROSTER_SPECS,
    TREND_SPECS, VOL_SPECS, ROUTE_VOLATILITY_EXPANSION)
from goldenswing.execution.scenario_book import ScenarioBook
from goldenswing.execution.setup_lifecycle import TrackedSetup
from goldenswing.live import vol_expansion as ve
from goldenswing.live.situation import build_tf_state
from goldenswing.playbooks.vol_expansion import (
    VolExpansionEscapePlaybook, VolFailedExpansionPlaybook)

from test_step14_setup_lifecycle import _scan_runner


# ------------------------------------------------------------ fixtures

def _frame(bars, freq="1h"):
    idx = pd.date_range("2026-01-05 00:00", periods=len(bars),
                        freq=freq)
    return pd.DataFrame(
        bars, columns=["open", "high", "low", "close"], index=idx)


def _baseline(n=300, center=2000.0, amp=8.0, seed=7):
    """Volatile weather: wide ranges, bandwidth high — the
    ground a coil compresses FROM."""
    rng = np.random.default_rng(seed)
    bars, price = [], center
    for i in range(n):
        wave = amp * np.sin(2 * np.pi * i / 24.0)
        o = price
        c = center + wave + rng.normal(0, 0.8)
        h = max(o, c) + abs(rng.normal(0, 1.2)) + 0.6
        l = min(o, c) - abs(rng.normal(0, 1.2)) - 0.6
        bars.append((o, h, l, c))
        price = c
    return bars


def _coil(n=30, center=2000.0, seed=11):
    """The coil: 30 bars of ~1-wide ranges around 2000 after
    the baseline's ~8-wide weather (bandwidth percentile and
    ATR ratio both collapse — measured in the tests, never
    assumed)."""
    rng = np.random.default_rng(seed)
    bars = []
    for _ in range(n):
        c = center + rng.normal(0, 0.15)
        o = center + rng.normal(0, 0.15)
        bars.append((o, max(o, c) + 0.24, min(o, c) - 0.24, c))
    return bars


_BASE = _baseline()
_COIL = _coil()
TAIL = _baseline(40, center=2012.0, amp=3.0, seed=21)
COIL_END = len(_BASE) + len(_COIL) - 1          # 329
TRIG = COIL_END + 1                             # 330
# Displacement trigger: body-dominated close beyond the
# coil high (~2000.47) + 0.25 ATR.
_TRIGGER = (2000.4, 2009.6, 1999.8, 2009.0)
# Acceptance: the next bar's whole range holds beyond the
# coil (low 2008.6 > coil high) — entry at its close.
_ACCEPT = (2009.0, 2010.4, 2008.6, 2009.8)
# Rejection: a close back INSIDE the coil (1999.8).
_REJECT = (2008.0, 2008.8, 1999.0, 1999.8)
# Rejection in the coil's upper half (mid still ahead).
_REJECT_HIGH = (2008.0, 2008.8, 1999.9, 2000.2)


def _escape_frame():
    return _frame(_BASE + _COIL + [_TRIGGER, _ACCEPT] + TAIL)


def _failed_frame(reject=_REJECT):
    return _frame(_BASE + _COIL + [_TRIGGER, reject] + TAIL)


def _situation(sid="sit-vol-1", other="contracting"):
    """A situation snapshot in the keeper's schema — the
    corroboration reads per_tf[*].volatility.atr_state."""
    return {"situation_id": sid, "per_tf": {
        "H1": {"volatility": {"atr_state": "contracting"},
               "structure_phase": "range"},
        "H4": {"volatility": {"atr_state": other}},
        "M15": {"volatility": {"atr_state": "normal"}}}}


def _escape_pb(situation=_situation()):
    pb = VolExpansionEscapePlaybook()
    pb.set_formation_tf("H1")
    pb.set_situation(situation)
    return pb


# ------------------------------------------------- detector: the coil

def test_coil_is_a_squeeze_run_with_atr_contracting():
    prep = ve.PreparedFrame(_escape_frame())
    coil = ve.coil_ending_at(prep, COIL_END)
    assert coil is not None
    assert coil["end"] == COIL_END and coil["bars"] >= 5
    # Both organs agree at the coil's end (measured, not
    # assumed): bandwidth deep in squeeze territory, ATR
    # contracting by the situation engine's own threshold.
    assert coil["bw_percentile"] < 20.0
    assert coil["atr_ratio"] < 0.75
    assert prep.atr_state(COIL_END) == "contracting"
    # Geometry is the run's own envelope.
    assert coil["high"] == pytest.approx(2000.475, abs=0.01)
    assert coil["low"] == pytest.approx(1999.472, abs=0.01)
    assert coil["width"] == pytest.approx(
        coil["high"] - coil["low"])


def test_detector_parity_with_situation_engine():
    """On identical bars the detector's ATR / atr_ratio /
    atr_state equal build_tf_state's EXACTLY — the ratio is
    the situation engine's own construction (Wilder ATR
    over the atr_series trailing-100 median)."""
    df = _escape_frame()
    prep = ve.PreparedFrame(df)
    st = build_tf_state("H1", df.iloc[:COIL_END + 1],
                        asof=df.index[COIL_END])
    vol = st["volatility"]
    assert vol["atr"] == pytest.approx(prep.atr(COIL_END))
    assert vol["atr_ratio"] == pytest.approx(
        prep.atr_ratio(COIL_END))
    assert vol["atr_state"] == "contracting"
    assert prep.atr_state(COIL_END) == vol["atr_state"]


def test_no_coil_no_sequence():
    """A violent bar in uncompressed weather is not an
    expansion: no coil, no sequence, no signal."""
    bars = _BASE + [(2000.4, 2012.0, 1999.8, 2011.0)] + TAIL
    df = _frame(bars)
    prep = ve.PreparedFrame(df)
    assert ve.sequence_at(prep, len(_BASE)) is None
    pb = _escape_pb()
    assert pb.live_signal([], len(df) - 1, df) == []


# ------------------------------------------- detector: the state machine

def test_escape_sequence_and_geometry():
    df = _escape_frame()
    prep = ve.PreparedFrame(df)
    at_trigger = ve.sequence_at(prep, TRIG)
    assert at_trigger["phase"] == "triggered"
    assert at_trigger["direction"] == 1
    assert at_trigger["trigger_bar"] == TRIG
    seq = ve.sequence_at(prep, TRIG + 1)
    assert seq["phase"] == "accepted"
    assert seq["event_bar"] == TRIG + 1
    geo = ve.escape_geometry(prep, seq)
    assert geo["entry"] == pytest.approx(2009.8)  # accept close
    # Stop just INSIDE the coil edge (the declared
    # invalidation), target the measured move from the
    # entry actually transacted.
    assert geo["stop"] < geo["edge"]
    assert geo["stop"] == pytest.approx(
        geo["edge"] - 0.25 * seq["coil"]["atr"])
    assert geo["targets"] == [pytest.approx(
        geo["entry"] + seq["coil"]["width"])]


def test_failed_expansion_sequence_and_geometry():
    df = _failed_frame()
    prep = ve.PreparedFrame(df)
    seq = ve.sequence_at(prep, TRIG + 1)
    assert seq["phase"] == "rejected"
    assert seq["direction"] == 1          # the ESCAPE was up
    assert seq["event_bar"] == TRIG + 1
    geo = ve.failed_geometry(prep, seq)
    assert geo["entry"] == pytest.approx(1999.8)
    # Stop beyond the escape's excursion extreme + buffer.
    assert geo["stop"] > geo["extreme"]
    assert geo["stop"] == pytest.approx(
        geo["extreme"] + 0.25 * seq["coil"]["atr"])
    # The rejection closed in the coil's lower half: the mid
    # is BEHIND a short entry and is dropped; the opposite
    # edge (ahead by construction) remains.
    assert geo["targets"] == [pytest.approx(seq["coil"]["low"])]


def test_failed_expansion_upper_half_rejection_keeps_both_targets():
    df = _failed_frame(_REJECT_HIGH)
    prep = ve.PreparedFrame(df)
    seq = ve.sequence_at(prep, TRIG + 1)
    assert seq["phase"] == "rejected"
    geo = ve.failed_geometry(prep, seq)
    assert geo["targets"] == [pytest.approx(seq["coil"]["mid"]),
                              pytest.approx(seq["coil"]["low"])]


def test_trigger_without_resolution_expires_as_drift():
    """Probes that dip inside (low through the edge) but
    CLOSE beyond it neither accept (range not held) nor
    reject (no close inside): the window runs out and the
    sequence is drift — no hypothesis, no signal."""
    probes = [(2009.0, 2009.6, 1999.6, 2008.8),
              (2008.8, 2010.0, 1999.9, 2009.4),
              (2009.4, 2010.6, 2000.1, 2010.0)]
    df = _frame(_BASE + _COIL + [_TRIGGER] + probes + TAIL)
    prep = ve.PreparedFrame(df)
    assert ve.sequence_at(prep, TRIG + 1)["phase"] == "triggered"
    assert ve.sequence_at(prep, TRIG + 3)["phase"] == "triggered"
    assert ve.sequence_at(prep, TRIG + 4)["phase"] == "expired"
    pb = _escape_pb()
    assert pb.live_signal([], len(df) - 1, df) == []
    pb2 = VolFailedExpansionPlaybook()
    pb2.set_formation_tf("H1")
    pb2.set_situation(_situation())
    assert pb2.live_signal([], len(df) - 1, df) == []


def test_close_through_opposite_edge_is_not_a_rejection():
    """A close punching through the coil's far edge is an
    escape the other way (or chaos), never the failed
    expansion's rejection close."""
    through = (2008.0, 2008.8, 1998.2, 1998.6)   # close < coil low
    df = _frame(_BASE + _COIL + [_TRIGGER, through] + TAIL)
    prep = ve.PreparedFrame(df)
    seq = ve.sequence_at(prep, TRIG + 1)
    assert seq is None or seq["phase"] != "rejected"


def test_reads_are_causal_no_hindsight():
    """sequence_at(j) on the full frame equals the read on
    the frame truncated at j — judged only by what was
    observable at the time."""
    df = _escape_frame()
    prep_full = ve.PreparedFrame(df)
    for j in (COIL_END, TRIG, TRIG + 1, TRIG + 3):
        prep_cut = ve.PreparedFrame(df.iloc[:j + 1])
        a, b = ve.sequence_at(prep_full, j), \
            ve.sequence_at(prep_cut, j)
        assert (a is None) == (b is None), j
        if a is not None:
            assert a["phase"] == b["phase"], j
            assert a["event_bar"] == b["event_bar"], j
            assert a["coil"]["end"] == b["coil"]["end"], j


def test_down_escape_mirrors():
    """The same sequence inverted: trigger below the coil,
    acceptance holding below, entry short at the acceptance
    close, stop just inside the low edge."""
    def inv(bar):
        o, h, l, c = bar
        return (4000.0 - o, 4000.0 - l, 4000.0 - h, 4000.0 - c)
    bars = ([inv(b) for b in _BASE] + [inv(b) for b in _COIL]
            + [inv(_TRIGGER), inv(_ACCEPT)]
            + [inv(b) for b in TAIL])
    df = _frame(bars)
    prep = ve.PreparedFrame(df)
    seq = ve.sequence_at(prep, TRIG + 1)
    assert seq["phase"] == "accepted" and seq["direction"] == -1
    geo = ve.escape_geometry(prep, seq)
    assert geo["entry"] == pytest.approx(4000.0 - 2009.8)
    assert geo["stop"] > geo["edge"]      # just inside, above
    assert geo["targets"][0] < geo["entry"]


# ------------------------------------------------------- playbook layer

def test_escape_signal_shape_and_context():
    df = _escape_frame()
    pb = _escape_pb()
    sigs = pb.live_signal([], len(df) - 1, df)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.index == TRIG + 1 and sig.direction == 1
    assert sig.entry_ref == pytest.approx(2009.8)
    assert sig.invalidation < sig.entry_ref
    ctx = sig.live_context
    assert ctx["family"] == "volatility_expansion"
    assert ctx["hypothesis"] == "expansion_escape"
    assert ctx["formation_tf"] == "H1"
    assert ctx["entry_tf"] == "M15"
    assert ctx["management_context_timeframe"] == "H1"
    assert ctx["scenario_clocks"]["management_tf"] == "H1"
    assert ctx["scenario_clocks"]["trigger_tf"] == "H1"
    assert ctx["corroboration"]["source"] == "situation"
    assert ctx["corroboration"]["by_tf"] == "H4"
    # §6.2's phase input rides as recorded context.
    assert ctx["corroboration"]["formation_phase"] == "range"
    assert ctx["situation_id"] == "sit-vol-1"
    assert ctx["coil"]["bars"] >= 5
    assert any("back inside" in t
               for t in ctx["invalidation_terms"])


def test_failed_signal_shape_and_context():
    df = _failed_frame()
    pb = VolFailedExpansionPlaybook()
    pb.set_formation_tf("H1")
    pb.set_situation(_situation())
    sigs = pb.live_signal([], len(df) - 1, df)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.direction == -1                     # fade the escape
    assert sig.entry_ref == pytest.approx(1999.8)  # rejection close
    assert sig.invalidation > 2009.6               # beyond extreme
    ctx = sig.live_context
    assert ctx["hypothesis"] == "failed_expansion"
    assert ctx["rejection"]["escape_direction"] == 1


def test_uncorroborated_situation_blocks_both_hypotheses():
    """The situation is threaded but NO other timeframe is
    contracting: the coil is a detection, not a hypothesis —
    no signal from either playbook."""
    df = _escape_frame()
    flat = _situation(other="normal")
    pb = _escape_pb(situation=flat)
    assert pb.live_signal([], len(df) - 1, df) == []
    df_f = _failed_frame()
    pb2 = VolFailedExpansionPlaybook()
    pb2.set_formation_tf("H1")
    pb2.set_situation(flat)
    assert pb2.live_signal([], len(df_f) - 1, df_f) == []


def test_real_situation_states_corroborate():
    """The corroborating state comes from the situation
    engine itself: build_tf_state over the neighbour frame
    (truncated at the coil end, as a keeper snapshot as of
    that bar would read) contracts — and the playbook
    accepts it as corroboration."""
    df = _escape_frame()
    neighbour = build_tf_state("H4", df.iloc[:COIL_END + 1],
                               asof=df.index[COIL_END])
    assert neighbour["volatility"]["atr_state"] == "contracting"
    snap = {"situation_id": "sit-real", "per_tf": {
        "H4": neighbour,
        "M15": {"volatility": {"atr_state": "normal"}}}}
    pb = _escape_pb(situation=snap)
    assert len(pb.live_signal([], len(df) - 1, df)) == 1


def test_arm_read_reports_coil_then_trigger():
    pb = _escape_pb()
    df = _escape_frame()
    coil_read = pb.pending_state(df.iloc[:COIL_END + 1])
    assert coil_read["phase"] == "coil"
    assert coil_read["corroborated"] is True
    trig_read = pb.pending_state(df.iloc[:TRIG + 1])
    assert trig_read["phase"] == "triggered"
    assert trig_read["direction"] == 1
    assert trig_read["trigger"]["close"] == pytest.approx(2009.0)
    # Long after resolution: nothing pending.
    assert pb.pending_state(df) is None


# --------------------------------------------------- registry + routes

def test_registry_partition_and_posture():
    assert [s.name for s in VOL_SPECS] == [
        "vol_expansion_escape", "vol_failed_expansion"]
    for spec in VOL_SPECS:
        assert spec.route == ROUTE_VOLATILITY_EXPANSION
        assert spec.shadow_evaluation is True
        assert spec.entry_thesis == "market"
        assert spec.management_horizon == "intraday"
        # NOT on the trigger path: the roster is untouched.
        assert spec not in ROSTER_SPECS
        assert spec.playbook_class.live is False
    assert {s.name for s in ROSTER_SPECS} | {
        s.name for s in VOL_SPECS} | {
        s.name for s in TREND_SPECS} | {
        s.name for s in NATIVE_SPECS} == {
        s.name for s in PLAYBOOK_REGISTRY}
    assert "no order authority" in FLOOR_AUTHORITY[
        ROUTE_VOLATILITY_EXPANSION]


def test_routes_declared_shadow_and_never_governing(tmp_path):
    runner, _ = _scan_runner(tmp_path)
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    for name in ("vol_expansion_escape", "vol_failed_expansion"):
        assert runner._judgment_route_state(name) == \
            "shadow_declared"
        assert runner._judgment_route_source(name) == "declared"
        assert table[name]["governing_now"] is False
        assert JUDGMENT_ROUTE_PLAYBOOK[name] == name
    # The runner's tables agree with the family's own map.
    assert set(vol_promotion.VOL_JUDGMENT_ROUTES) == \
        set(mr.VOL_JUDGMENT_ROUTE_NAMES)
    # Promotion exists ONLY as the owner's separate decision:
    # an explicit config step on a DECLARED route (Pass 5's
    # path; the family's pre-registered evaluation is the
    # evidence it is decided on) — this pass grants nothing,
    # and even a promoted route has no order path
    # (test_family_has_no_order_path).
    runner.cfg.judgment_route_modes = {
        "vol_expansion_escape": "governing"}
    assert runner._judgment_route_state(
        "vol_expansion_escape") == "governing"
    assert runner._judgment_route_source(
        "vol_expansion_escape") == "config"
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    assert table["vol_expansion_escape"]["governing_now"] is True
    assert table["vol_failed_expansion"]["governing_now"] is False
    # A typo'd config VALUE promotes nothing either.
    runner.cfg.judgment_route_modes = {
        "vol_expansion_escape": "governingg"}
    assert runner._judgment_route_state(
        "vol_expansion_escape") == "shadow_declared"


def test_family_has_no_order_path():
    """No seam maps a vol playbook to an order route: the
    route_promotion order table (built from the trigger-path
    roster) names neither, so no candidate of this family
    can reach an order through any existing route."""
    from goldenswing.execution.route_promotion import (
        ORDER_ROUTE_BY_PLAYBOOK)
    for name in ("vol_expansion_escape", "vol_failed_expansion"):
        assert name not in ORDER_ROUTE_BY_PLAYBOOK


def test_undeclared_names_still_fail_closed(tmp_path):
    """Pass 5's invariant, family-adjacent: a route name no
    authority source declares resolves shadow/undeclared
    and is REFUSED at the judgment seam in every mode, with
    the journaled alarm (the seam is where fail-closed
    binds; a config entry is itself an owner authority
    source, so the no-config case is the pinned one — the
    shipped invariant in tests/test_route_authority.py)."""
    from test_route_authority import (
        _candidate, _decision, _evidence, _jc_runner)
    runner, _ = _scan_runner(tmp_path)
    assert runner._judgment_route_state(
        "vol_expansion_typo") == "shadow_declared"
    assert runner._judgment_route_source(
        "vol_expansion_typo") == "undeclared"
    for mode in ("active", "shadow"):
        jc = _jc_runner(tmp_path / mode, judgment_mode=mode)
        _bundle, refusal = jc._judgment_authority(
            candidate=_candidate(), decision=_decision(),
            evidence=_evidence(), intended_tf="H1",
            eval_now=jc._evaluation_now(),
            route="vol_expansion_typo")
        assert refusal is not None
        assert "undeclared" in refusal
        alarms = [r for r in jc.journal.records()
                  if r["type"] == "undeclared_route"]
        assert len(alarms) == 1
        assert alarms[0]["route"] == "vol_expansion_typo"


# ------------------------------------------------------- management (P6)

def test_management_block_runs_standard_plan_on_formation_clock():
    df = _escape_frame()
    sig = _escape_pb().live_signal([], len(df) - 1, df)[0]
    block = management_block_for("vol_expansion_escape", sig)
    assert block["management_horizon"] == "intraday"
    assert block["management_policy_id"] == "standard_v1"
    # The signal's declared clock wins: managed on H1, the
    # clock the thesis formed on.
    assert block["management_context_timeframe"] == "H1"
    assert block["management_invalidation"] == sig.invalidation
    assert block["management_target"] == sig.targets[0]
    # The standard plan's thesis exit is armed — the return
    # inside the coil is the thesis exit (Pass 6 machinery).
    from goldenswing.execution.playbook_registry import (
        policy_params_for)
    params = policy_params_for(block["management_policy_id"])
    assert params["thesis_exit"] == "parent_thesis_invalidation"


def test_management_block_formation_clock_is_per_instance():
    """The same playbook formed on H4 is managed on H4 —
    the instance's declaration, not the registry's native
    clock (H1)."""
    sig = SimpleNamespace(
        targets=[2010.0], invalidation=2000.0,
        live_context={"management_context_timeframe": "H4"})
    block = management_block_for("vol_expansion_escape", sig)
    assert block["management_context_timeframe"] == "H4"
    # Controls: signals WITHOUT the declaration resolve
    # exactly as before (native clock / roster defaults).
    plain = SimpleNamespace(targets=[1.0], invalidation=0.5,
                            live_context={})
    assert management_block_for(
        "vol_expansion_escape", plain
    )["management_context_timeframe"] == "H1"
    assert management_block_for(
        "sweep_reversal", plain
    )["management_context_timeframe"] == "H1"
    assert management_block_for(
        "range_qm", plain
    )["management_context_timeframe"] == "H4"


def test_book_takes_instance_clocks_and_freezes_plan():
    sig = _escape_pb().live_signal([], len(_escape_frame()) - 1,
                                   _escape_frame())[0]
    clocks = {"formation_tf": "H4", "refinement_tf": None,
              "trigger_tf": "H4", "entry_tf": "H1",
              "management_tf": "H4"}
    setup = TrackedSetup(
        setup_id="vol:1", parent_setup_id="vol:1",
        playbook="vol_expansion_escape",
        route=ROUTE_VOLATILITY_EXPANSION, direction=1,
        entry_ref=sig.entry_ref, invalidation=sig.invalidation,
        targets=list(sig.targets),
        management=management_block_for(
            "vol_expansion_escape", sig),
        formation_ts="2026-01-10T00:00:00",
        touch_ts="2026-01-10T01:00:00",
        created_ts="2026-01-10T01:00:00",
        trigger_tf="H1", validity_bars=6,
        expiry_ts="2026-01-10T07:00:00",
        live_context={"scenario_clocks": dict(clocks),
                      "family": "volatility_expansion",
                      "invalidation_terms":
                          sig.live_context["invalidation_terms"]})
    book = ScenarioBook()
    book.propose_from_setup(setup, ts="2026-01-10T01:00:00")
    sc = book.scenarios["vol:1"]
    assert sc["clocks"] == clocks
    # Freeze at fill (Pass 6): the frozen context carries
    # the BUILT plan — standard_v1 running on the declared
    # formation clock, with the family's own invalidation
    # terms — and later edits to the scenario's declared
    # management never rewrite it.
    from goldenswing.execution.setup_lifecycle import (
        STAGE_FILLED)
    book.sync_setup(setup, ts="2026-01-10T02:00:00",
                    stage=STAGE_FILLED)
    frozen = book.scenarios["vol:1"]["management_context"]
    assert frozen["management_plan"]["policy_id"] == \
        "standard_v1"
    assert frozen["management_plan"]["management_tf"] == "H1"
    # The plan's declared terms are the standing pair (the
    # price invalidation + the thesis exit) — the same
    # shape every family's block declares; the family's own
    # named terms ride on the signal context (pinned in
    # test_escape_signal_shape_and_context).
    terms = frozen["management_plan"]["invalidation"]["terms"]
    assert any("close beyond invalidation" in t for t in terms)
    assert any("parent thesis invalidated" in t for t in terms)
    book.scenarios["vol:1"]["management"]["management_plan"][
        "policy_id"] = "tampered"
    assert book.scenarios["vol:1"]["management_context"][
        "management_plan"]["policy_id"] == "standard_v1"


def test_book_clocks_default_without_declaration():
    """A setup carrying no scenario_clocks declaration gets
    the standing defaults (the hook is inert for every
    other family)."""
    setup = TrackedSetup(
        setup_id="ctl:1", parent_setup_id="ctl:1",
        playbook="sweep_reversal", route="h1_roster",
        direction=1, entry_ref=2000.0, invalidation=1990.0,
        targets=[2020.0],
        management={"management_horizon": "intraday",
                    "management_policy_id": "standard_v1",
                    "management_policy_version": 1,
                    "management_context_timeframe": "H1",
                    "management_invalidation": 1990.0,
                    "management_target": 2020.0},
        formation_ts="2026-01-10T00:00:00",
        touch_ts="2026-01-10T01:00:00",
        created_ts="2026-01-10T01:00:00",
        trigger_tf="H1", validity_bars=6,
        expiry_ts="2026-01-10T07:00:00", live_context={})
    book = ScenarioBook()
    book.propose_from_setup(setup, ts="2026-01-10T01:00:00")
    clocks = book.scenarios["ctl:1"]["clocks"]
    assert clocks["formation_tf"] == "H1"
    assert clocks["management_tf"] == "H1"


# --------------------------------------------------------- runner wiring

def _family_runner(tmp_path, frame):
    runner, _ = _scan_runner(tmp_path)
    runner._stack = SimpleNamespace(frames={
        "H1": frame,
        "M15": _frame(_baseline(120, seed=5), freq="15min"),
        "M5": _frame(_baseline(120, seed=6), freq="5min")})
    runner._situation_keeper = SimpleNamespace(
        latest=lambda: _situation(),
        latest_id=lambda: "sit-vol-1")
    return runner


def test_runner_scan_journals_evaluations_verdicts_and_book(tmp_path):
    # The frame ends AT the acceptance bar: a scheduled
    # sweep honors signals on the closing bar (the fresh-
    # instance design), never stale history.
    frame = _escape_frame().iloc[:TRIG + 2]
    runner = _family_runner(tmp_path, frame)
    runner._run_vol_expansion_family(("H1",), {},
                                     source="live_close")
    records = runner.journal.records()
    evals = [r for r in records if r["type"] == "evaluation"
             and r.get("kind") == "playbook_eval"
             and r["playbook"] == "vol_expansion_escape"]
    assert evals and all(
        r["evaluation_clock"] == "vol_expansion_h1" for r in evals)
    assert all(r["situation_id"] == "sit-vol-1" for r in evals)
    emitted = [r for r in evals if r["outcome"] == "candidate_emitted"]
    assert len(emitted) == 1
    # §6.2's day-fuel input rides on the family's records
    # as context (the fixture runner holds no day context:
    # the key is present, the value honestly None).
    assert "adr_state_context" in emitted[0]["evidence"]
    assert "never consulted" in emitted[0]["evidence"]["adr_role"]
    # The §G stamp: the scan's passive verdict is journaled
    # as a judgment_shadow identifiable as THIS route's
    # history (the route is the evaluate call's stamp; it
    # lands on the verdict, not the evaluation record).
    shadows = [r for r in records if r["type"] == "judgment_shadow"
               and (r.get("judgment") or {}).get(
                   "authority_route") == "vol_expansion_escape"]
    assert len(shadows) == 1
    assert shadows[0]["judgment"]["authority"] == "shadow_declared"
    # The scenario entered the book as a shadow proposal
    # with the instance's clocks.
    setups = runner._setup_store.active()
    assert len(setups) == 1
    book = runner._scenario_book_get()
    sc = book.scenarios[setups[0].setup_id]
    assert sc["family"] == "vol_expansion_escape"
    assert sc["authority"] == "shadow"
    assert sc["clocks"]["formation_tf"] == "H1"
    assert sc["clocks"]["management_tf"] == "H1"
    assert sc["clocks"]["entry_tf"] == "M15"
    # No order of any kind was placed.
    assert not any("order" in str(r.get("type")) for r in records)


def test_runner_journals_arm_transitions(tmp_path):
    trig_frame = _escape_frame().iloc[:TRIG + 1]
    runner = _family_runner(tmp_path, trig_frame)
    runner._run_vol_expansion_family(("H1",), {},
                                     source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "vol_expansion_arm"]
    assert len(arms) == 1
    arm = arms[0]
    assert arm["family"] == "volatility_expansion"
    assert arm["phase"] == "triggered" and arm["direction"] == 1
    assert arm["tf"] == "H1"
    assert arm["situation_id"] == "sit-vol-1"
    assert arm["coil"]["bars"] >= 5
    # Re-running the same scan journals no duplicate arm.
    runner._run_vol_expansion_family(("H1",), {},
                                     source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "vol_expansion_arm"]
    assert len(arms) == 1


def test_runner_family_clocks_cover_formation_tfs(tmp_path):
    """Every formation clock is evaluated and accountable:
    frames the stack doesn't carry land as not_eligible
    records on their own vol_expansion_* clocks."""
    runner = _family_runner(tmp_path, _escape_frame())
    runner._run_vol_expansion_family(("H4", "H1", "M15"), {},
                                     source="live_close")
    clocks = {(r["playbook"], r["evaluation_clock"])
              for r in runner.journal.records()
              if r["type"] == "evaluation"}
    for name in ("vol_expansion_escape", "vol_failed_expansion"):
        assert (name, "vol_expansion_h4") in clocks
        assert (name, "vol_expansion_h1") in clocks
        assert (name, "vol_expansion_m15") in clocks


# ------------------------------------------------- promotion evaluation

def _arm_records(route, take_rs, abstain_rs, *, start=0):
    """Route-stamped shadow verdicts + judgment
    reconciliations in the evaluator's own record shapes —
    the §G suite's fabricators, re-pointed at this family's
    playbooks (one shadow + one +4h reconciliation per
    outcome, paired inside the evaluator)."""
    from test_range_promotion_evaluation import _arm as _rp_arm
    return (_rp_arm(route, "take", take_rs, start=start)
            + _rp_arm(route, "abstain", abstain_rs,
                      start=start + 1000))


def test_promotion_evaluation_insufficient_then_ready_binds_nothing(
        tmp_path):
    route = "vol_expansion_escape"
    state = vol_promotion.new_state()
    few = _arm_records(route, [1.0] * 6, [0.0] * 6)
    rec = vol_promotion.cycle(state=state, records=few, now=None,
                              bar="2026-11-03 00:00")
    assert rec is not None and rec["type"] == "vol_promotion_evaluation"
    row = rec["routes"][route]
    assert row["verdict"] == range_promotion.VERDICT_INSUFFICIENT
    many = _arm_records(route, [1.2] * 22, [-0.3] * 22,
                        start=5000)
    rec2 = vol_promotion.cycle(state=state, records=many, now=None,
                               bar="2026-11-04 00:00")
    assert rec2 is not None
    row = rec2["routes"][route]
    assert row["verdict"] == range_promotion.VERDICT_READY
    assert row["take"]["median_r"] == pytest.approx(1.2)
    # binds=False made visible: the verdict speaks as a
    # RECOMMENDATION, never as an effect.
    assert "RECOMMENDATION" in row["verdict_reason"]
    text = "\n".join(vol_promotion.render_lines(state))
    assert "PROMOTION READY (recommendation only)" in text
    # The verdict changed nothing about the route's state:
    # still declared shadow, still governing nowhere.
    runner, _ = _scan_runner(tmp_path)
    assert runner._judgment_route_state(route) == "shadow_declared"


def test_promotion_host_phase_journals_and_persists(tmp_path):
    """The runner host (the §G arrangement): one self-audit
    window in -> a dated vol_promotion_evaluation record in
    the journal, state persisted under its own key in the
    audit state file, and /status renders the block."""
    runner, _ = _scan_runner(tmp_path)
    records = _arm_records("vol_expansion_escape",
                           [1.0] * 3, [0.0] * 3)
    runner._vol_promotion_phase(
        records, "2026-11-03 00:00",
        window={"records": len(records)})
    out = [r for r in runner.journal.records()
           if r["type"] == "vol_promotion_evaluation"]
    assert len(out) == 1
    assert out[0]["routes"]["vol_expansion_escape"][
        "verdict"] == range_promotion.VERDICT_INSUFFICIENT
    assert "vol_promotion" in runner._audit_state()
    assert "vol promotion (Pass 7a)" in runner.vol_promotion_line()


def test_promotion_evaluation_one_shot_ignores_other_routes():
    records = _arm_records("vol_failed_expansion",
                           [1.0] * 22, [0.0] * 22)
    snap = vol_promotion.evaluate_records(records)
    assert snap["routes"]["vol_failed_expansion"]["verdict"] == \
        range_promotion.VERDICT_READY
    assert snap["routes"]["vol_expansion_escape"]["verdict"] == \
        range_promotion.VERDICT_INSUFFICIENT
    # The range family's history never leaks in either.
    snap2 = vol_promotion.evaluate_records(
        _arm_records("range_qm", [1.0] * 22, [0.0] * 22))
    assert all(row["verdict"] ==
               range_promotion.VERDICT_INSUFFICIENT
               for row in snap2["routes"].values())
