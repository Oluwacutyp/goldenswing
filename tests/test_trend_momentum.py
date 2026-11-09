"""MarketSituation Pass 7b — the trend / momentum family.

Pins, in the GS conventions (fixtures judged by what was
observable AT THE TIME — every detector assertion runs the
causal read at the bar itself, and one test proves the read
at (j, m) is identical on the frames truncated at (j, m)):

- the measurement core (live/trend_momentum.py): an
  impulse is a formation close beyond a confirmed swing
  whose follow-through at its extreme CLOSE read sustained;
  the pullback retraces inside the leg's band without a
  formation close beyond the impulse origin (that close
  INVALIDATES); the turn is the pullback's minor-structure
  break on the trigger clock; momentum is follow-through
  quality (sustained / decaying / absorbed); acceleration
  is velocity through the 80th percentile of its own
  trailing distribution with closes accepting beyond the
  last confirmed swing;
- the playbooks: the trend state is a RELATIONSHIP fact
  read from the situation (formation bias + trend phase +
  the formation->trigger pair kind — pullback accepts
  aligned / correcting / nested, acceleration aligned
  only); absorbed momentum and a sustained counter-move
  are abstentions, never entries; the frame-structure
  fallback corroborates only when the canonical organs
  themselves state the trend;
- the route contract: both routes DECLARED shadow
  (Pass 5 machinery), governing in no mode, no order
  path exists for the partition, undeclared names still
  fail closed;
- the Pass 6 management shape: the instance declares its
  horizon (intraday on H4 / swing on D1) and the standard
  policy FOR THAT HORIZON runs on the formation clock;
  the book takes the instance's clocks + horizon and the
  plan freezes at entry; qm_m5's fixed bracket is never
  horizon-overridden;
- the runner wiring: scheduled scans journal evaluations
  (situation-stamped), route-stamped judgment_shadow
  verdicts, arm records, and book scenarios — and place
  no orders;
- the promotion evaluation (execution/trend_promotion.py):
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
from goldenswing.execution import range_promotion, trend_promotion
from goldenswing.execution.mt5_runner import (
    JUDGMENT_ROUTE_PLAYBOOK, management_block_for)
from goldenswing.execution.playbook_registry import (
    FLOOR_AUTHORITY, NATIVE_SPECS, PLAYBOOK_REGISTRY, ROSTER_SPECS,
    AUCTION_SPECS, REVERSION_SPECS, SESSION_SPECS, TREND_SPECS,
    VOL_SPECS,
    ROUTE_TREND_MOMENTUM)
from goldenswing.execution.scenario_book import ScenarioBook
from goldenswing.execution.setup_lifecycle import TrackedSetup
from goldenswing.live import trend_momentum as tm
from goldenswing.live.situation import build_tf_state
from goldenswing.playbooks.trend_momentum import (
    TrendImpulsePullbackPlaybook, TrendMomentumAccelerationPlaybook)

from test_step14_setup_lifecycle import _scan_runner


# ------------------------------------------------------------ fixtures
#
# Deterministic bars from scripted closes: the wick is a
# fixed function of the bar's own move, and the turn
# segments carry explicit (high, low) overrides — a minor
# swing peak must be decisive against the next bar, which
# OPENS at the peak close (fixture geometry, stated once).

def _unit_frame(closes, freq, *, overrides=None):
    bars = []
    prev = closes[0]
    for i, c in enumerate(closes):
        o = prev
        move = abs(c - o)
        h = max(o, c) + 0.18 + 0.06 * move
        l = min(o, c) - 0.18 - 0.06 * move
        if overrides and i in overrides:
            h, l = overrides[i]
        bars.append((o, h, l, c))
        prev = c
    idx = pd.date_range("2026-01-05 00:00", periods=len(bars),
                        freq=freq)
    return pd.DataFrame(bars, columns=["open", "high", "low",
                                       "close"], index=idx)


def _resample(df, rule):
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last"}).dropna()


_CHOP = [2000.0 + 2.5 * np.sin(2 * np.pi * i / 16.0)
         for i in range(96)]
_DIP = [1998.0, 1996.0, 1994.0, 1992.5, 1993.5, 1995.0]
_IMPULSE = [1998.0, 2001.0, 2004.0, 2007.0, 2010.0, 2013.0,
            2016.0, 2019.0, 2021.0, 2023.0, 2024.0, 2025.0,
            2025.5, 2026.0]
_PULLBACK = [2024.0, 2022.5, 2021.0, 2019.5, 2018.0, 2016.5,
             2015.0, 2014.0, 2013.0, 2012.5]
# The turn segment starts at H1 index 126; its minor peak
# is bar 127.
_TURN_OV = {127: (2017.4, 2015.6), 128: (2016.2, 2014.9),
            129: (2015.4, 2014.1)}
_TURN_CLOSE_BAR = 130          # the turn's break close


def _happy_closes():
    return (_CHOP + _DIP + _IMPULSE + _PULLBACK
            + [2014.5, 2016.5, 2015.8, 2015.0, 2018.5, 2019.0])


def _happy_frames():
    h1 = _unit_frame(_happy_closes(), "1h", overrides=_TURN_OV)
    return h1, _resample(h1, "4h")


def _invalidated_frames():
    closes = (_CHOP + _DIP + _IMPULSE
              + [2022.0, 2019.0, 2016.0, 2013.0, 2010.0, 2007.0,
                 2004.0, 2001.0, 1998.0, 1995.0, 1993.0, 1991.0])
    h1 = _unit_frame(closes, "1h")
    return h1, _resample(h1, "4h")


def _counter_frames():
    """The pullback as a counter-impulse: ~0.80 of the leg
    retraced in two body-dominated formation bars, then a
    textbook turn shape (which must NOT be taken)."""
    closes = (_CHOP + _DIP + _IMPULSE
              + [2021.0, 2017.0, 2013.0, 2009.0, 2006.5, 2005.0,
                 2004.0, 2003.5]
              + [2005.0, 2008.5, 2006.8, 2006.0, 2010.5, 2011.0])
    ov = {125: (2009.6, 2004.2), 126: (2008.2, 2005.9),
          127: (2006.4, 2005.2)}
    h1 = _unit_frame(closes, "1h", overrides=ov)
    return h1, _resample(h1, "4h")


def _absorbed_frames():
    """A shallow fast pullback, then nine formation bars of
    tight churn at the low: by the turn, the trailing
    momentum window is all post-impulse chop — follow-
    through absorbed.  The turn shape itself is valid."""
    band = [2018.2 + 0.4 * np.sin(2 * np.pi * i / 8.0)
            for i in range(40)]
    closes = (_CHOP + _DIP + _IMPULSE
              + [2023.0, 2020.5, 2019.0, 2017.8] + band
              + [2019.5, 2021.0, 2019.6, 2019.0, 2022.5, 2023.0])
    ov = {161: (2021.9, 2019.9), 162: (2020.4, 2018.8),
          163: (2019.5, 2018.2)}
    h1 = _unit_frame(closes, "1h", overrides=ov)
    return h1, _resample(h1, "4h")


def _accel_frames():
    """100 formation bars of chop, a weak steady rise, a
    small dip, then rising-increment closes: velocity
    breaks out of its own flat distribution with a fresh
    confirmed-swing break in the final segment."""
    closes = []
    for i in range(400):
        closes.append(2000.0 + 2.5 * np.sin(2 * np.pi * i / 64.0))
    p = closes[-1]
    for _ in range(64):
        p += 0.5
        closes.append(p)
    for _ in range(12):
        p -= 0.8
        closes.append(p)
    for step in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.0, 4.5, 4.5,
                 5.0, 5.0, 5.5):
        p += step
        closes.append(p)
    h1 = _unit_frame(closes, "1h")
    return h1, _resample(h1, "4h")


def _staircase_frames():
    """An ESTABLISHED trend the canonical organs themselves
    state (nine rising swing cycles, then the family's
    impulse / pullback / turn on top): the fixture for the
    frame-structure fallback — evaluate_bias reads HH + HL
    and classify_regime reads clean_trend on these bars."""
    closes = [2000.0]
    p = 2000.0
    for _cycle in range(9):
        for _ in range(24):
            p += 0.55
            closes.append(p)
        for _ in range(12):
            p -= 0.55
            closes.append(p)
    closes += [2062.0, 2065.0, 2068.0, 2071.0, 2074.0, 2077.0,
               2080.0, 2083.0, 2085.0, 2087.0, 2088.0, 2089.0,
               2089.5, 2090.0]
    closes += [2088.0, 2086.5, 2085.0, 2083.5, 2082.0, 2080.5,
               2079.0, 2078.0, 2077.0, 2076.5]
    base = len(closes)
    closes += [2078.5, 2080.5, 2079.8, 2079.0, 2082.5, 2083.0]
    ov = {base + 1: (2081.4, 2079.6), base + 2: (2080.2, 2078.9),
          base + 3: (2079.4, 2078.1)}
    h1 = _unit_frame(closes, "1h", overrides=ov)
    return h1, _resample(h1, "4h")


def _d1_frames():
    """The happy path in H4 units under a D1 formation
    clock (the swing variant).  Incommensurate chop
    periods: the plain 16-unit sine sampled every 6 units
    ties D1 closes, and tied closes form no swings."""
    chop = [2000.0 + 3.0 * np.sin(2 * np.pi * i / 26.0)
            + 0.7 * np.sin(2 * np.pi * i / 7.0)
            for i in range(96)]
    closes = (chop + _DIP + _IMPULSE + _PULLBACK
              + [2014.5, 2016.5, 2015.8, 2015.0, 2018.5, 2019.0])
    units = _unit_frame(closes, "4h", overrides=_TURN_OV)
    return units, _resample(units, "24h")


def _situation(sid="sit-trend-1", *, pair="aligned",
               htf_pair="aligned", bias=1, phase="clean_trend"):
    """A situation snapshot in the keeper's schema — the
    trend-state read consumes per_tf[*].bias /
    structure_phase and the engine relationships' kinds."""
    return {"situation_id": sid, "per_tf": {
        "D1": {"bias": bias, "structure_phase": phase},
        "H4": {"bias": bias, "structure_phase": phase},
        "H1": {"bias": bias, "structure_phase": phase}},
        "relationships": [
            {"htf": "D1", "ltf": "H4", "kind": htf_pair,
             "source": "engine"},
            {"htf": "H4", "ltf": "H1", "kind": pair,
             "source": "engine"}]}


def _pullback_pb(formation, tf="H4", situation=_situation()):
    pb = TrendImpulsePullbackPlaybook()
    pb.set_formation(formation, tf)
    pb.set_situation(situation)
    return pb


def _seq(h1, h4, j, m):
    prep_f = tm.PreparedFrame(h4)
    prep_t = tm.PreparedFrame(h1, swing_n=tm.TRIGGER_SWING_N)
    return tm.sequence_at(prep_f, prep_t, j, m,
                          formation_tf="H4", trigger_tf="H1")


# ------------------------------------------------- detector: the sequence

def test_impulse_pullback_sequence_and_geometry():
    h1, h4 = _happy_frames()
    seq = _seq(h1, h4, 31, _TURN_CLOSE_BAR)
    assert seq["phase"] == "triggered"
    assert seq["direction"] == 1
    imp = seq["impulse"]
    # The impulse: a close beyond the chop's last confirmed
    # swing high, origin the last confirmed swing low known
    # at the break, follow-through sustained at the leg's
    # extreme close.
    assert imp["break_level"] == pytest.approx(2002.7, abs=0.6)
    assert imp["origin_price"] == pytest.approx(1997.3, abs=0.6)
    assert imp["momentum_at_extreme"]["state"] == "sustained"
    pb = seq["pullback"]
    assert pb["depth_frac"] == pytest.approx(0.48, abs=0.03)
    assert pb["counter_sustained"] is False
    assert seq["turn"]["close"] == pytest.approx(2018.5)
    geo = tm.pullback_geometry(tm.PreparedFrame(h4), seq,
                               seq["turn"], at_bar=31)
    assert geo["entry"] == pytest.approx(2018.5)
    # Stop beyond the defended swing (the pullback extreme)
    # by the desk's standing buffer — structural, not fixed.
    assert geo["defended_swing"] == \
        pytest.approx(pb["extreme_price"])
    assert geo["stop"] < geo["defended_swing"]
    assert geo["stop"] == pytest.approx(
        geo["defended_swing"] - 0.25 * geo["atr"])
    # Targets: the impulse extreme, then its measured
    # projection — both ahead of the entry.
    assert geo["targets"][0] == \
        pytest.approx(imp["extreme_price"])
    assert geo["targets"][1] == pytest.approx(
        imp["extreme_price"] + imp["leg"])
    assert all(t > geo["entry"] for t in geo["targets"])


def test_momentum_reads_sustained_then_absorbed():
    """Momentum is follow-through QUALITY: sustained at the
    impulse's extreme on the happy path; absorbed after the
    post-impulse churn on the absorbed path (same turn
    shape, opposite verdict)."""
    h1, h4 = _happy_frames()
    mom = tm.PreparedFrame(h4).momentum_at(31, 1)
    assert mom["state"] == "sustained"
    assert mom["net_atr"] >= 1.0 and mom["efficiency"] >= 0.30
    h1a, h4a = _absorbed_frames()
    seq = _seq(h1a, h4a, 40, 164)
    assert seq["phase"] == "triggered"
    mom_a = tm.PreparedFrame(h4a).momentum_at(40, 1)
    assert mom_a["state"] == "absorbed"


def test_close_beyond_origin_invalidates():
    """A formation CLOSE beyond the impulse origin is the
    trend claim's death — the sequence reads invalidated
    with the killing bar named, however the path got there."""
    h1, h4 = _invalidated_frames()
    seq = _seq(h1, h4, 31, 127)
    assert seq["phase"] == "invalidated"
    assert seq["break_bar"] == 31
    assert float(h4["close"].iloc[31]) < \
        seq["impulse"]["origin_price"]


def test_reads_are_causal_no_hindsight():
    """sequence_at(j, m) on the full frames equals the read
    on the frames truncated at (j, m) — judged only by what
    was observable at the time."""
    h1, h4 = _happy_frames()
    for (j, m) in ((30, 125), (31, _TURN_CLOSE_BAR)):
        a = _seq(h1, h4, j, m)
        b = _seq(h1.iloc[:m + 1], h4.iloc[:j + 1], j, m)
        assert (a is None) == (b is None), (j, m)
        if a is not None:
            assert a["phase"] == b["phase"], (j, m)
            assert a["impulse"]["origin_price"] == \
                b["impulse"]["origin_price"], (j, m)
            assert a["impulse"]["break_bar"] == \
                b["impulse"]["break_bar"], (j, m)
            if a["pullback"] is not None:
                assert a["pullback"]["depth_frac"] == \
                    b["pullback"]["depth_frac"], (j, m)


def test_acceleration_core_read():
    h1, h4 = _accel_frames()
    prep = tm.PreparedFrame(h4)
    # Before the final segment's break: no acceleration.
    assert tm.acceleration_at(prep, 118) is None
    assert tm.acceleration_at(prep, 119) is None
    acc = tm.acceleration_at(prep, 120)
    assert acc is not None and acc["direction"] == 1
    assert acc["velocity_percentile"] >= \
        tm.VELOCITY_PERCENTILE
    assert acc["velocity"] > 0
    # Entry at the confirming formation close; stop back
    # beyond the broken swing; target the leg's measured
    # projection.
    assert acc["entry"] == pytest.approx(float(
        h4["close"].iloc[120]))
    assert acc["stop"] < acc["broken_swing"]["price"] \
        < acc["entry"]
    assert acc["targets"] == [pytest.approx(
        acc["entry"] + acc["leg"])]
    assert acc["momentum"]["state"] == "sustained"


# ------------------------------------------------------- playbook layer

def test_pullback_signal_shape_and_context():
    h1, h4 = _happy_frames()
    pb = _pullback_pb(h4)
    sigs = pb.live_signal([], len(h1) - 1, h1)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.index == _TURN_CLOSE_BAR and sig.direction == 1
    assert sig.entry_ref == pytest.approx(2018.5)
    assert sig.invalidation < 2012.2      # beyond defended swing
    assert sig.targets[0] == pytest.approx(2026.3, abs=0.3)
    ctx = sig.live_context
    assert ctx["family"] == "trend_momentum"
    assert ctx["hypothesis"] == "impulse_pullback"
    assert ctx["formation_tf"] == "H4"
    assert ctx["trigger_tf"] == "H1"
    assert ctx["entry_tf"] == "H1"
    assert ctx["management_context_timeframe"] == "H4"
    assert ctx["management_horizon"] == "intraday"
    assert ctx["scenario_clocks"] == {
        "formation_tf": "H4", "refinement_tf": None,
        "trigger_tf": "H1", "entry_tf": "H1",
        "management_tf": "H4"}
    assert ctx["trend_state"]["source"] == "situation"
    assert ctx["trend_state"]["pair_kind"] == "aligned"
    assert ctx["trend_state"]["htf_pair_kind"] == "aligned"
    assert ctx["situation_id"] == "sit-trend-1"
    assert ctx["impulse"]["momentum_at_extreme"] == "sustained"
    assert ctx["pullback"]["counter_sustained"] is False
    assert ctx["momentum"]["state"] == "sustained"
    assert any("impulse origin" in t
               for t in ctx["invalidation_terms"])


def test_correcting_pair_is_the_pullbacks_home_ground():
    """SPEC §6.1: the correcting pair IS the pullback state
    at the pair level — it corroborates (an aligned-only
    gate would refuse the family's own home ground)."""
    h1, h4 = _happy_frames()
    pb = _pullback_pb(h4, situation=_situation(pair="correcting"))
    assert len(pb.live_signal([], len(h1) - 1, h1)) == 1


def test_situation_gates_block_the_hypothesis():
    """The trend state is a relationship fact: a conflicting
    pair, an opposing formation bias, or a non-trend phase
    each refuse the entry — the sequence is a detection,
    not a hypothesis, without the wider state's say."""
    h1, h4 = _happy_frames()
    for snap in (_situation(pair="conflicting"),
                 _situation(bias=-1),
                 _situation(phase="range")):
        pb = _pullback_pb(h4, situation=snap)
        assert pb.live_signal([], len(h1) - 1, h1) == []


def test_counter_sustained_pullback_is_an_abstention():
    """The turn shape fires at the core — and the playbook
    refuses it: the pullback displaced AGAINST the trend
    (a counter-impulse).  Entering here trades the other
    side's follow-through, not a pullback."""
    h1, h4 = _counter_frames()
    seq = _seq(h1, h4, 31, 128)
    assert seq["phase"] == "triggered"
    assert seq["pullback"]["counter_sustained"] is True
    pb = _pullback_pb(h4)
    assert pb.live_signal([], len(h1) - 1, h1) == []


def test_absorbed_momentum_is_an_abstention():
    h1, h4 = _absorbed_frames()
    pb = _pullback_pb(h4)
    assert pb.live_signal([], len(h1) - 1, h1) == []


def test_invalidated_sequence_produces_no_signal():
    h1, h4 = _invalidated_frames()
    pb = _pullback_pb(h4)
    assert pb.live_signal([], len(h1) - 1, h1) == []


def test_frame_structure_fallback_needs_a_stated_trend():
    """No snapshot threaded (the research path): the
    formation frame's own canonical organs must state the
    trend.  On the happy fixture they do NOT (one impulse
    out of a range, its base a lower low — bias 0) and the
    playbook honestly stays silent; on the established
    staircase they DO (HH + HL, clean_trend) and the same
    sequence corroborates with source frame_structure."""
    h1, h4 = _happy_frames()
    pb = TrendImpulsePullbackPlaybook()
    pb.set_formation(h4, "H4")
    assert pb.live_signal([], len(h1) - 1, h1) == []
    h1s, h4s = _staircase_frames()
    pb2 = TrendImpulsePullbackPlaybook()
    pb2.set_formation(h4s, "H4")
    sigs = pb2.live_signal([], len(h1s) - 1, h1s)
    assert len(sigs) == 1
    ctx = sigs[0].live_context
    assert ctx["trend_state"]["source"] == "frame_structure"
    assert ctx["trend_state"]["formation_phase"] == "clean_trend"
    assert ctx["situation_id"] is None


def test_engine_states_do_not_corroborate_the_happy_fixture():
    """The integration truth: a snapshot assembled from the
    situation engine's OWN reads on the happy fixture
    (bias 0, no trend phase at the turn bar) does not
    corroborate — the family defers to the engine's actual
    reads, never to the fixture's intent."""
    h1, h4 = _happy_frames()
    state = build_tf_state("H4", h4.iloc[:32],
                           asof=h4.index[31])
    assert state["bias"] == 0
    snap = {"situation_id": "sit-real",
            "per_tf": {"H4": state},
            "relationships": [
                {"htf": "H4", "ltf": "H1", "kind": "aligned",
                 "source": "engine"}]}
    pb = _pullback_pb(h4, situation=snap)
    assert pb.live_signal([], len(h1) - 1, h1) == []


def test_acceleration_signal_shape_and_context():
    h1, h4 = _accel_frames()
    pb = TrendMomentumAccelerationPlaybook()
    pb.set_formation(h4, "H4")
    pb.set_situation(_situation())
    sigs = pb.live_signal([], len(h1) - 1, h1)
    # Two qualifying RUNS (the trend-birth velocity break
    # and the established-trend acceleration); each fires
    # once, on its first bar — the run at formation bar 121
    # is the same event as bar 120, not a second entry.
    assert [s.index for s in sigs] == [403, 483]
    sig = sigs[-1]
    assert sig.direction == 1
    assert sig.entry_ref == pytest.approx(float(
        h4["close"].iloc[120]))
    ctx = sig.live_context
    assert ctx["family"] == "trend_momentum"
    assert ctx["hypothesis"] == "momentum_acceleration"
    assert ctx["trend_state"]["pair_kind"] == "aligned"
    assert ctx["acceleration"]["velocity_percentile"] >= \
        tm.VELOCITY_PERCENTILE
    assert ctx["acceleration"]["broken_swing"] == \
        pytest.approx(2034.7, abs=0.6)
    assert sig.invalidation < ctx["acceleration"]["broken_swing"]
    assert sig.targets[0] > sig.entry_ref


def test_acceleration_requires_the_aligned_pair():
    """Continuation, not breakout: a merely correcting pair
    (the pullback hypothesis's ground) does not corroborate
    acceleration — participation must be live (aligned)."""
    h1, h4 = _accel_frames()
    pb = TrendMomentumAccelerationPlaybook()
    pb.set_formation(h4, "H4")
    pb.set_situation(_situation(pair="correcting"))
    assert pb.live_signal([], len(h1) - 1, h1) == []


def test_d1_formation_declares_the_swing_horizon():
    units, d1 = _d1_frames()
    pb = TrendImpulsePullbackPlaybook()
    pb.set_formation(d1, "D1")
    pb.set_situation(_situation())
    sigs = pb.live_signal([], len(units) - 1, units)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.entry_ref == pytest.approx(2018.5)
    ctx = sig.live_context
    assert ctx["formation_tf"] == "D1"
    assert ctx["trigger_tf"] == "H4"
    assert ctx["management_horizon"] == "swing"
    assert ctx["management_context_timeframe"] == "D1"
    assert ctx["scenario_clocks"]["management_tf"] == "D1"
    assert ctx["trend_state"]["pair_kind"] == "aligned"


def test_unthreaded_playbook_produces_nothing():
    """The family is a two-clock detector: without a
    threaded formation frame, candidates() honestly
    produces nothing (never a one-clock guess)."""
    pb = TrendImpulsePullbackPlaybook()
    ctx = SimpleNamespace(df=_happy_frames()[0])
    assert pb.candidates(ctx) == []


def test_arm_read_reports_pullback_then_invalidation():
    h1, h4 = _happy_frames()
    pb = _pullback_pb(h4)
    read = pb.pending_state(h1.iloc[:126])
    assert read["phase"] == "pullback"
    assert read["direction"] == 1
    assert read["corroborated"] is True
    assert read["impulse"]["origin_price"] == \
        pytest.approx(1997.3, abs=0.6)
    assert read["identity"]
    h1i, h4i = _invalidated_frames()
    pbi = _pullback_pb(h4i)
    dead = pbi.pending_state(h1i)
    assert dead["phase"] == "invalidated"
    assert dead["break_ts"] is not None


# --------------------------------------------------- registry + routes

def test_registry_partition_and_posture():
    assert [s.name for s in TREND_SPECS] == [
        "trend_impulse_pullback", "trend_momentum_acceleration"]
    for spec in TREND_SPECS:
        assert spec.route == ROUTE_TREND_MOMENTUM
        assert spec.shadow_evaluation is True
        assert spec.entry_thesis == "market"
        assert spec.management_horizon == "intraday"
        # NOT on the trigger path: the roster is untouched.
        assert spec not in ROSTER_SPECS
        assert spec.playbook_class.live is False
    assert {s.name for s in ROSTER_SPECS} | {
        s.name for s in NATIVE_SPECS} | {
        s.name for s in VOL_SPECS} | {
        s.name for s in TREND_SPECS} | {
        s.name for s in AUCTION_SPECS} | {
        s.name for s in REVERSION_SPECS} | {
        s.name for s in SESSION_SPECS} == {
        s.name for s in PLAYBOOK_REGISTRY}
    assert "no order authority" in FLOOR_AUTHORITY[
        ROUTE_TREND_MOMENTUM]


def test_routes_declared_shadow_and_never_governing(tmp_path):
    runner, _ = _scan_runner(tmp_path)
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    for name in ("trend_impulse_pullback",
                 "trend_momentum_acceleration"):
        assert runner._judgment_route_state(name) == \
            "shadow_declared"
        assert runner._judgment_route_source(name) == "declared"
        assert table[name]["governing_now"] is False
        assert JUDGMENT_ROUTE_PLAYBOOK[name] == name
    # The runner's tables agree with the family's own map.
    assert set(trend_promotion.TREND_JUDGMENT_ROUTES) == \
        set(mr.TREND_JUDGMENT_ROUTE_NAMES)
    # Promotion exists ONLY as the owner's separate decision:
    # an explicit config step on a DECLARED route (Pass 5's
    # path; the family's pre-registered evaluation is the
    # evidence it is decided on) — this pass grants nothing,
    # and even a promoted route has no order path
    # (test_family_has_no_order_path).
    runner.cfg.judgment_route_modes = {
        "trend_impulse_pullback": "governing"}
    assert runner._judgment_route_state(
        "trend_impulse_pullback") == "governing"
    assert runner._judgment_route_source(
        "trend_impulse_pullback") == "config"
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    assert table["trend_impulse_pullback"]["governing_now"] is True
    assert table["trend_momentum_acceleration"][
        "governing_now"] is False
    # A typo'd config VALUE promotes nothing either.
    runner.cfg.judgment_route_modes = {
        "trend_impulse_pullback": "governingg"}
    assert runner._judgment_route_state(
        "trend_impulse_pullback") == "shadow_declared"


def test_family_has_no_order_path():
    """No seam maps a trend playbook to an order route: the
    route_promotion order table (built from the trigger-path
    roster) names neither, so no candidate of this family
    can reach an order through any existing route."""
    from goldenswing.execution.route_promotion import (
        ORDER_ROUTE_BY_PLAYBOOK)
    for name in ("trend_impulse_pullback",
                 "trend_momentum_acceleration"):
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
        "trend_momentum_typo") == "shadow_declared"
    assert runner._judgment_route_source(
        "trend_momentum_typo") == "undeclared"
    for mode in ("active", "shadow"):
        jc = _jc_runner(tmp_path / mode, judgment_mode=mode)
        _bundle, refusal = jc._judgment_authority(
            candidate=_candidate(), decision=_decision(),
            evidence=_evidence(), intended_tf="H1",
            eval_now=jc._evaluation_now(),
            route="trend_momentum_typo")
        assert refusal is not None
        assert "undeclared" in refusal
        alarms = [r for r in jc.journal.records()
                  if r["type"] == "undeclared_route"]
        assert len(alarms) == 1
        assert alarms[0]["route"] == "trend_momentum_typo"


# ------------------------------------------------------- management (P6)

def test_management_block_intraday_standard_on_formation_clock():
    h1, h4 = _happy_frames()
    sig = _pullback_pb(h4).live_signal([], len(h1) - 1, h1)[0]
    block = management_block_for("trend_impulse_pullback", sig)
    assert block["management_horizon"] == "intraday"
    assert block["management_policy_id"] == "standard_v1"
    # The signal's declared clock wins: managed on H4, the
    # clock the thesis formed on (not the H1 entry clock).
    assert block["management_context_timeframe"] == "H4"
    assert block["management_invalidation"] == sig.invalidation
    assert block["management_target"] == sig.targets[0]
    # The standard plan's thesis exit is armed — a momentum-
    # decay / structure verdict reaches the position through
    # this machinery (Pass 6), never around it.
    from goldenswing.execution.playbook_registry import (
        policy_params_for)
    params = policy_params_for(block["management_policy_id"])
    assert params["thesis_exit"] == "parent_thesis_invalidation"


def test_management_block_swing_instance_gets_swing_policy():
    """The instance's declared horizon resolves the policy
    through the registry's own horizon rule: a D1-formed
    instance is managed as a swing (standard_swing_v1 on
    D1), an H4-formed one as intraday — same playbook."""
    d1_sig = SimpleNamespace(
        targets=[2060.0], invalidation=1990.0,
        live_context={"management_context_timeframe": "D1",
                      "management_horizon": "swing"})
    block = management_block_for("trend_impulse_pullback", d1_sig)
    assert block["management_horizon"] == "swing"
    assert block["management_policy_id"] == "standard_swing_v1"
    assert block["management_context_timeframe"] == "D1"
    # Controls: signals WITHOUT the declarations resolve
    # exactly as before (registry defaults / native clocks).
    plain = SimpleNamespace(targets=[1.0], invalidation=0.5,
                            live_context={})
    assert management_block_for(
        "trend_impulse_pullback", plain
    )["management_policy_id"] == "standard_v1"
    assert management_block_for(
        "trend_impulse_pullback", plain
    )["management_context_timeframe"] == "H4"
    assert management_block_for(
        "sweep_reversal", plain
    )["management_context_timeframe"] == "H1"
    assert management_block_for(
        "range_qm", plain
    )["management_context_timeframe"] == "H4"


def test_qm_m5_fixed_bracket_never_horizon_overridden():
    """Step 13 is closed: even a (hypothetical) horizon-
    carrying context cannot move qm_m5 off its fixed
    bracket — the name resolution wins by name."""
    carrier = SimpleNamespace(
        targets=[2010.0], invalidation=2000.0,
        live_context={"management_horizon": "swing"})
    plain = SimpleNamespace(targets=[2010.0],
                            invalidation=2000.0, live_context={})
    block = management_block_for("qm_m5", carrier)
    control = management_block_for("qm_m5", plain)
    assert block["management_policy_id"] == "fixed_bracket_v1"
    assert block["management_horizon"] == \
        control["management_horizon"]


def _book_setup(sig, *, playbook="trend_impulse_pullback",
                route=ROUTE_TREND_MOMENTUM, setup_id="trend:1"):
    return TrackedSetup(
        setup_id=setup_id, parent_setup_id=setup_id,
        playbook=playbook, route=route, direction=1,
        entry_ref=sig.entry_ref, invalidation=sig.invalidation,
        targets=list(sig.targets),
        management=management_block_for(playbook, sig),
        formation_ts="2026-01-10T00:00:00",
        touch_ts="2026-01-10T01:00:00",
        created_ts="2026-01-10T01:00:00",
        trigger_tf=sig.live_context["trigger_tf"],
        validity_bars=6, expiry_ts="2026-01-10T07:00:00",
        live_context={"scenario_clocks":
                          dict(sig.live_context["scenario_clocks"]),
                      "management_horizon":
                          sig.live_context["management_horizon"],
                      "family": "trend_momentum",
                      "invalidation_terms":
                          sig.live_context["invalidation_terms"]})


def test_book_takes_instance_clocks_horizon_and_freezes_plan():
    h1, h4 = _happy_frames()
    sig = _pullback_pb(h4).live_signal([], len(h1) - 1, h1)[0]
    setup = _book_setup(sig)
    book = ScenarioBook()
    book.propose_from_setup(setup, ts="2026-01-10T01:00:00")
    sc = book.scenarios["trend:1"]
    assert sc["clocks"] == sig.live_context["scenario_clocks"]
    assert sc["horizon"] == "intraday"
    # Freeze at fill (Pass 6): the frozen context carries
    # the BUILT plan — standard_v1 running on the declared
    # formation clock — and later edits to the scenario's
    # declared management never rewrite it.
    from goldenswing.execution.setup_lifecycle import (
        STAGE_FILLED)
    book.sync_setup(setup, ts="2026-01-10T02:00:00",
                    stage=STAGE_FILLED)
    frozen = book.scenarios["trend:1"]["management_context"]
    assert frozen["management_plan"]["policy_id"] == \
        "standard_v1"
    assert frozen["management_plan"]["management_tf"] == "H4"
    terms = frozen["management_plan"]["invalidation"]["terms"]
    assert any("close beyond invalidation" in t for t in terms)
    assert any("parent thesis invalidated" in t for t in terms)
    book.scenarios["trend:1"]["management"]["management_plan"][
        "policy_id"] = "tampered"
    assert book.scenarios["trend:1"]["management_context"][
        "management_plan"]["policy_id"] == "standard_v1"


def test_book_swing_instance_freezes_the_swing_plan():
    units, d1 = _d1_frames()
    pb = TrendImpulsePullbackPlaybook()
    pb.set_formation(d1, "D1")
    pb.set_situation(_situation())
    sig = pb.live_signal([], len(units) - 1, units)[0]
    setup = _book_setup(sig, setup_id="trend:2")
    book = ScenarioBook()
    book.propose_from_setup(setup, ts="2026-01-10T01:00:00")
    sc = book.scenarios["trend:2"]
    assert sc["horizon"] == "swing"
    assert sc["clocks"]["management_tf"] == "D1"
    from goldenswing.execution.setup_lifecycle import (
        STAGE_FILLED)
    book.sync_setup(setup, ts="2026-01-10T02:00:00",
                    stage=STAGE_FILLED)
    frozen = book.scenarios["trend:2"]["management_context"]
    assert frozen["management_plan"]["policy_id"] == \
        "standard_swing_v1"
    assert frozen["management_plan"]["management_tf"] == "D1"


def test_book_horizon_defaults_without_declaration():
    """A setup carrying no horizon declaration gets the
    registry default (the hook is inert for every other
    family)."""
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
    assert book.scenarios["ctl:1"]["horizon"] == "intraday"


# --------------------------------------------------------- runner wiring

def _family_runner(tmp_path, h1_frame, h4_frame):
    runner, _ = _scan_runner(tmp_path)
    runner._stack = SimpleNamespace(frames={
        "H4": h4_frame, "H1": h1_frame,
        "D1": _unit_frame([2000.0] * 40, "24h")})
    runner._situation_keeper = SimpleNamespace(
        latest=lambda: _situation(),
        latest_id=lambda: "sit-trend-1")
    return runner


def test_runner_scan_journals_evaluations_verdicts_and_book(tmp_path):
    # The frames end AT the turn bar: a scheduled sweep
    # honors signals on the closing bar (the fresh-instance
    # design), never stale history.
    h1, h4 = _happy_frames()
    runner = _family_runner(tmp_path, h1.iloc[:131],
                            h4.iloc[:32])
    runner._run_trend_momentum_family(("H4",), {},
                                      source="live_close")
    records = runner.journal.records()
    evals = [r for r in records if r["type"] == "evaluation"
             and r.get("kind") == "playbook_eval"
             and r["playbook"] == "trend_impulse_pullback"]
    assert evals and all(
        r["evaluation_clock"] == "trend_momentum_h4"
        for r in evals)
    assert all(r["situation_id"] == "sit-trend-1" for r in evals)
    emitted = [r for r in evals if r["outcome"] == "candidate_emitted"]
    assert len(emitted) == 1
    # The family's context evidence rides on the record:
    # family identity, and the D1 read as CONTEXT only
    # (never a veto — §G precedent, stated on the record).
    assert emitted[0]["evidence"]["family"] == "trend_momentum"
    assert "never" in emitted[0]["evidence"]["d1_role"]
    # The §G stamp: the scan's passive verdict is journaled
    # as a judgment_shadow identifiable as THIS route's
    # history (the route is the evaluate call's stamp; it
    # lands on the verdict, not the evaluation record).
    shadows = [r for r in records if r["type"] == "judgment_shadow"
               and (r.get("judgment") or {}).get(
                   "authority_route") == "trend_impulse_pullback"]
    assert len(shadows) == 1
    assert shadows[0]["judgment"]["authority"] == "shadow_declared"
    # The scenario entered the book as a shadow proposal
    # with the instance's clocks + declared horizon.
    setups = runner._setup_store.active()
    assert len(setups) == 1
    book = runner._scenario_book_get()
    sc = book.scenarios[setups[0].setup_id]
    assert sc["family"] == "trend_impulse_pullback"
    assert sc["authority"] == "shadow"
    assert sc["horizon"] == "intraday"
    assert sc["clocks"]["formation_tf"] == "H4"
    assert sc["clocks"]["management_tf"] == "H4"
    assert sc["clocks"]["entry_tf"] == "H1"
    # No order of any kind was placed.
    assert not any("order" in str(r.get("type")) for r in records)


def test_runner_journals_arm_transitions(tmp_path):
    h1, h4 = _happy_frames()
    runner = _family_runner(tmp_path, h1.iloc[:126],
                            h4.iloc[:31])
    runner._run_trend_momentum_family(("H4",), {},
                                      source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "trend_momentum_arm"]
    assert len(arms) == 1
    arm = arms[0]
    assert arm["family"] == "trend_momentum"
    assert arm["hypothesis"] == "impulse_pullback"
    assert arm["phase"] == "pullback" and arm["direction"] == 1
    assert arm["formation_tf"] == "H4"
    assert arm["trigger_tf"] == "H1"
    assert arm["situation_id"] == "sit-trend-1"
    assert arm["impulse"]["origin_price"] == \
        pytest.approx(1997.3, abs=0.6)
    # Re-running the same scan journals no duplicate arm.
    runner._run_trend_momentum_family(("H4",), {},
                                      source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "trend_momentum_arm"]
    assert len(arms) == 1


def test_runner_journals_invalidation_arms(tmp_path):
    """A sequence's structural death is journaled as an arm
    record in its own right (Pass 9 grades families on
    their deaths as well as their entries)."""
    h1, h4 = _invalidated_frames()
    runner = _family_runner(tmp_path, h1, h4)
    runner._run_trend_momentum_family(("H4",), {},
                                      source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "trend_momentum_arm"
            and r["phase"] == "invalidated"]
    assert len(arms) == 1
    assert arms[0]["break_ts"] is not None


def test_runner_family_clocks_cover_formation_tfs(tmp_path):
    """Every formation clock is evaluated and accountable:
    both playbooks land evaluation records on the D1
    (swing) and H4 (intraday) clocks alike."""
    h1, h4 = _happy_frames()
    runner = _family_runner(tmp_path, h1, h4)
    runner._run_trend_momentum_family(("D1", "H4"), {},
                                      source="live_close")
    clocks = {(r["playbook"], r["evaluation_clock"])
              for r in runner.journal.records()
              if r["type"] == "evaluation"}
    for name in ("trend_impulse_pullback",
                 "trend_momentum_acceleration"):
        assert (name, "trend_momentum_d1") in clocks
        assert (name, "trend_momentum_h4") in clocks


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
    route = "trend_impulse_pullback"
    state = trend_promotion.new_state()
    few = _arm_records(route, [1.0] * 6, [0.0] * 6)
    rec = trend_promotion.cycle(state=state, records=few, now=None,
                                bar="2026-11-03 00:00")
    assert rec is not None and \
        rec["type"] == "trend_promotion_evaluation"
    row = rec["routes"][route]
    assert row["verdict"] == range_promotion.VERDICT_INSUFFICIENT
    many = _arm_records(route, [1.2] * 22, [-0.3] * 22,
                        start=5000)
    rec2 = trend_promotion.cycle(state=state, records=many,
                                 now=None, bar="2026-11-04 00:00")
    assert rec2 is not None
    row = rec2["routes"][route]
    assert row["verdict"] == range_promotion.VERDICT_READY
    assert row["take"]["median_r"] == pytest.approx(1.2)
    # binds=False made visible: the verdict speaks as a
    # RECOMMENDATION, never as an effect.
    assert "RECOMMENDATION" in row["verdict_reason"]
    text = "\n".join(trend_promotion.render_lines(state))
    assert "PROMOTION READY (recommendation only)" in text
    # The verdict changed nothing about the route's state:
    # still declared shadow, still governing nowhere.
    runner, _ = _scan_runner(tmp_path)
    assert runner._judgment_route_state(route) == "shadow_declared"


def test_promotion_host_phase_journals_and_persists(tmp_path):
    """The runner host (the §G arrangement): one self-audit
    window in -> a dated trend_promotion_evaluation record
    in the journal, state persisted under its own key in
    the audit state file, and /status renders the block."""
    runner, _ = _scan_runner(tmp_path)
    records = _arm_records("trend_impulse_pullback",
                           [1.0] * 3, [0.0] * 3)
    runner._trend_promotion_phase(
        records, "2026-11-03 00:00",
        window={"records": len(records)})
    out = [r for r in runner.journal.records()
           if r["type"] == "trend_promotion_evaluation"]
    assert len(out) == 1
    assert out[0]["routes"]["trend_impulse_pullback"][
        "verdict"] == range_promotion.VERDICT_INSUFFICIENT
    assert "trend_promotion" in runner._audit_state()
    assert "trend promotion (Pass 7b)" in \
        runner.trend_promotion_line()


def test_promotion_evaluation_one_shot_ignores_other_routes():
    records = _arm_records("trend_momentum_acceleration",
                           [1.0] * 22, [0.0] * 22)
    snap = trend_promotion.evaluate_records(records)
    assert snap["routes"]["trend_momentum_acceleration"][
        "verdict"] == range_promotion.VERDICT_READY
    assert snap["routes"]["trend_impulse_pullback"][
        "verdict"] == range_promotion.VERDICT_INSUFFICIENT
    # The other families' histories never leak in either.
    for other in ("range_qm", "vol_expansion_escape"):
        snap2 = trend_promotion.evaluate_records(
            _arm_records(other, [1.0] * 22, [0.0] * 22))
        assert all(row["verdict"] ==
                   range_promotion.VERDICT_INSUFFICIENT
                   for row in snap2["routes"].values())
