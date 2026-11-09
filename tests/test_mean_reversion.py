"""MarketSituation Pass 7d (SPEC §6.4) — the mean
reversion family: exhaustion of extensions AWAY from
developing value.

The family's ground, distinct from 7c's auction family
(7c trades acceptance / rotation of probes AT value
edges; 7d trades exhaustion of moves that left value):

- the core read (live/mean_reversion.py): a causal
  episode machine over the day's H1 bars, built on 7c's
  developing-value layer (the shared layer — POC /
  edges / migration, tick-volume basis NAMED, never
  dressed) and 7b's momentum measures.  An episode
  opens when a formation close sits >= 2.0 ATR beyond
  the ANCHORED value edge (the developing state at the
  last close inside value, fixed for the episode's
  life); the stall confirms when closes stop making
  progress beyond the extreme, momentum is no longer
  sustained, and value did NOT migrate toward the
  extreme; a formation close through the defended swing
  breaks the continuation; a close beyond the extreme
  after stall / break KILLS the episode (invalidated —
  7c-style acceptance arriving is this family's death,
  journaled with its kind); a close back inside the
  reference edge closes it; an extension that never
  stalls inside its window expires;
- two hypotheses, separately declared: the extension
  fade (entry on the trigger clock's first minor-swing
  turn back toward value after the stall; targets POC
  then the opposite edge — the rotation, staged; stop
  beyond the extreme) and the failed continuation
  (entry at the first trigger close after the defended
  swing broke, while the fade still has its rotation);
- the regime gate is the family's honesty: an extension
  inside a CLEAN TREND / BREAKOUT (the situation's
  formation-TF structure read) is REFUSED — the family
  abstains and journals why — unless the reads name the
  post-exhaustion state (momentum absorbed at the
  decision bar and >= 4 formation bars without close
  progress past the extreme);
- the route contract: both routes DECLARED shadow
  (Pass 5 machinery), governing in no mode, no order
  path exists for the partition, undeclared names still
  fail closed;
- the Pass 6 management shape: horizon PER INSTANCE
  (intraday on M15 triggers, scalp on M5), the declared
  horizon resolving its policy on the formation clock
  (H1); the book takes the instance's clocks and the
  plan freezes at entry; qm_m5's fixed bracket is never
  horizon-overridden;
- the runner wiring: scheduled trigger-clock scans
  journal evaluations (situation-stamped, family +
  volume-basis evidence with the basis NAMED),
  route-stamped judgment_shadow verdicts, arm records
  (corroborated transitions, invalidations AND regime
  refusals), and book scenarios — and place no orders;
- the situation wiring: location carries extension_atr,
  the signed distance of price from the developing POC
  in H1 ATR units, from the same state the family
  reads;
- the promotion evaluation
  (execution/reversion_promotion.py): the shared
  evaluator over the family's own route-stamped history;
  promotion_ready is a recommendation, binds nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from test_step14_setup_lifecycle import _scan_runner

import goldenswing.execution.mt5_runner as mt5_runner
from goldenswing.execution import range_promotion, reversion_promotion
from goldenswing.execution.mt5_runner import (
    JUDGMENT_ROUTE_PLAYBOOK,
    management_block_for,
)
from goldenswing.execution.playbook_registry import (
    AUCTION_SPECS,
    FLOOR_AUTHORITY,
    NATIVE_SPECS,
    PLAYBOOK_REGISTRY,
    REVERSION_SPECS,
    ROSTER_SPECS,
    ROUTE_MEAN_REVERSION,
    TREND_SPECS,
    VOL_SPECS,
)
from goldenswing.execution.scenario_book import ScenarioBook
from goldenswing.execution.setup_lifecycle import TrackedSetup
from goldenswing.live import developing_value as dv
from goldenswing.live import mean_reversion as mrev
from goldenswing.live.multitf import atr14
from goldenswing.playbooks.mean_reversion import (
    MeanReversionExtensionFadePlaybook,
    MeanReversionFailedContinuationPlaybook,
)

# ------------------------------------------------------------ fixtures
#
# Deterministic H1 bars from scripted closes + volumes, on
# a single UTC day (2026-01-05 00:00 UTC = one NY trading
# day under day_context.trading_day_key).  The value phase
# is 11 balanced bars around 2000 on steady volume: the
# developing profile is measurable from bar 9 and its
# edges sit just outside the oscillation (VAH ~2001.6,
# VAL ~1998.7 — read from the data in the tests, never
# assumed).  Bar highs/lows carry a deterministic jitter:
# the engine's fractal swing rule treats ties as
# never-swings, so symmetric fixtures would silently have
# no swings at all.  M15 / M5 frames are interpolated
# from the H1 closes, with the decisive tail segments
# overridden explicitly.

_BASE = [2000, 2002, 1998, 2001, 1999, 2003, 1997, 2000,
         2002, 1998, 2000]
_BASE_V = [100] * 11


def _h1_frame(closes, vols, *, ranges=None):
    bars, prev = [], closes[0]
    for i, c in enumerate(closes):
        o = prev
        if ranges and i in ranges:
            h, l = ranges[i]
        else:
            h = max(o, c) + 0.5 + 0.07 * (i % 3)
            l = min(o, c) - 0.35 - 0.06 * (i % 4)
        bars.append((o, h, l, c, vols[i]))
        prev = c
    idx = pd.date_range("2026-01-05 00:00", periods=len(bars),
                        freq="1h")
    return pd.DataFrame(
        bars, columns=["open", "high", "low", "close", "volume"],
        index=idx)


def _interp_from_h1(h1, steps, freq, *, extra_bars=0):
    closes, prev = [], float(h1["close"].iloc[0])
    for c in h1["close"]:
        for k in range(steps):
            closes.append(prev + (float(c) - prev)
                          * (k + 1) / steps)
        prev = float(c)
    for _ in range(extra_bars):
        closes.append(prev)
    bars, p = [], closes[0]
    for i, c in enumerate(closes):
        bars.append((p, max(p, c) + 0.3 + 0.04 * (i % 2),
                     min(p, c) - 0.3 - 0.05 * (i % 3), c, 25.0))
        p = c
    idx = pd.date_range(h1.index[0], periods=len(bars), freq=freq)
    return pd.DataFrame(
        bars, columns=["open", "high", "low", "close", "volume"],
        index=idx)


def _m15_from_h1(h1, *, extra_bars=0):
    return _interp_from_h1(h1, 4, "15min", extra_bars=extra_bars)


def _m5_from_h1(h1, *, extra_bars=0):
    return _interp_from_h1(h1, 12, "5min", extra_bars=extra_bars)


def _set_tail(frame, start, closes):
    p = float(frame["close"].iloc[start - 1])
    for i, c in enumerate(closes):
        frame.iloc[start + i] = (
            p, max(p, c) + 0.3 + 0.04 * (i % 2),
            min(p, c) - 0.3 - 0.05 * (i % 3), c, 25.0)
        p = c
    return frame


def _situation(sid="sit-reversion-1", *, phase="range", bias=0):
    """The situation snapshot in live/situation.py's
    schema: the formation TF's structure read for the
    regime gate, and the developing state (day-matched,
    basis named) for arm corroboration."""
    return {"situation_id": sid,
            "per_tf": {
                "H1": {"bias": bias, "structure_phase": phase}},
            "location": {
                "value": {
                    "poc": 1999.9, "vah": 2001.6, "val": 1998.7,
                    "developing": True,
                    "developing_state": {
                        "day_key": "2026-01-05", "bars": 23,
                        "volume_basis": "tick_volume"}}},
            "relationships": []}


_CHURN_A = [2012.9, 2013.1, 2012.3, 2012.8, 2012.1, 2012.5,
            2011.9, 2012.3]
_RNG_A = {15 + i: (c + 0.25, c - 0.25)
          for i, c in enumerate(_CHURN_A)}


def _stall_h1():
    """Climb to a 2013.5 extreme close (bar 14), then eight
    low-volume churn bars: value does not follow, momentum
    decays — the stall confirms at bar 21."""
    return _h1_frame(
        _BASE + [2004, 2007, 2010.5, 2013.5] + _CHURN_A,
        _BASE_V + [100] * 4 + [5] * 8, ranges=_RNG_A)


def _stall_m15(h1):
    """The stall day on M15, ending on the turn bar: the
    tail breaks the last confirmed minor swing low."""
    m15 = _m15_from_h1(h1, extra_bars=8)
    return _set_tail(m15, 92, [2011.8, 2011.2, 2011.7, 2011.3,
                               2010.6, 2010.2, 2009.8, 2009.4])


def _exhausted_h1():
    """Same extension, but the churn is flat then
    body-dominated DOWN bars: momentum is ABSORBED by the
    last formation bar — the post-exhaustion shape."""
    churn = [2013.4, 2013.3, 2013.45, 2013.35, 2012.2, 2011.0,
             2011.6, 2010.4]
    rng = {15: (2013.6, 2013.0), 16: (2013.55, 2012.95),
           17: (2013.65, 2013.05), 18: (2013.6, 2013.0),
           19: (2013.3, 2012.05), 20: (2012.1, 2010.85),
           21: (2011.75, 2010.9), 22: (2011.5, 2010.25)}
    return _h1_frame(
        _BASE + [2004, 2007, 2010.5, 2013.5] + churn,
        _BASE_V + [100] * 4 + [5] * 8, ranges=rng)


def _exhausted_m15(h1):
    m15 = _m15_from_h1(h1, extra_bars=8)
    return _set_tail(m15, 92, [2010.0, 2009.4, 2009.9, 2009.5,
                               2008.8, 2008.4, 2008.0, 2007.6])


def _migrating_h1():
    """The stall day's geometry on HEAVY churn volume: the
    developing POC itself migrates up to the extreme —
    value accepted the extension, so no stall ever
    confirms (7c's acceptance fact, seen from 7d)."""
    return _h1_frame(
        _BASE + [2004, 2007, 2010.5, 2013.5] + _CHURN_A,
        _BASE_V + [100] * 4 + [300] * 8, ranges=_RNG_A)


def _fc_h1():
    """Steep climb, high plateau, a dip whose low is the
    defended swing (2013.5, confirmed at bar 20), a higher
    extreme (2025, bar 20), then a formation close through
    the defended swing at bar 21: the continuation died."""
    closes = _BASE + [2010, 2016, 2019, 2018, 2019.5, 2018.5,
                      2014.5, 2020, 2023, 2025, 2012.0, 2010.5]
    ranges = {11: (2010.4, 2004.0), 12: (2016.4, 2011.0),
              13: (2019.4, 2016.5), 14: (2019.0, 2015.5),
              15: (2020.0, 2016.0), 16: (2019.5, 2015.8),
              17: (2017.0, 2013.5), 18: (2020.4, 2015.0),
              19: (2023.4, 2018.0), 20: (2025.4, 2020.0),
              21: (2018.0, 2011.0)}
    return _h1_frame(closes, [100] * 23, ranges=ranges)


def _invalidated_h1():
    """The stall confirms at bar 21; bar 22 closes beyond
    the extreme close on heavy volume — value followed
    beyond the extreme: acceptance, the fade's death."""
    return _h1_frame(
        _BASE + [2004, 2007, 2010.5, 2013.5] + _CHURN_A[:6]
        + [2011.9, 2016.0],
        _BASE_V + [100] * 4 + [5] * 6 + [5, 300],
        ranges={15 + i: (c + 0.25, c - 0.25)
                for i, c in enumerate(_CHURN_A[:6])})


def _mirror_h1(h1):
    """Price-mirror a frame around 4000 (up extension ->
    down extension) for the symmetry pins."""
    out = h1.copy()
    out["close"] = 4000.0 - h1["close"]
    out["open"] = 4000.0 - h1["open"]
    out["high"] = 4000.0 - h1["low"]
    out["low"] = 4000.0 - h1["high"]
    return out


def _fade_pb(h1, situation=None, trigger_tf=None):
    pb = MeanReversionExtensionFadePlaybook()
    pb.set_formation(h1, "H1")
    if trigger_tf:
        pb.set_formation_tf(trigger_tf)
    pb.set_situation(situation)
    return pb


def _fc_pb(h1, situation=None):
    pb = MeanReversionFailedContinuationPlaybook()
    pb.set_formation(h1, "H1")
    pb.set_situation(situation)
    return pb


# ------------------------------------------------ episode machine (core)

def test_episode_opens_beyond_anchored_edge_and_stalls():
    h1 = _stall_h1()
    prep = mrev.PreparedReversion(h1)
    anchor = prep.value_at(10)
    ep13 = mrev.reversion_episode_at(prep, 13)
    assert ep13["phase"] == mrev.PHASE_EXTENDED
    assert ep13["direction"] == 1
    # The reference is the ANCHORED state — the developing
    # value at the last close inside it — not the drifting
    # current state.
    assert ep13["ref"]["vah"] == pytest.approx(anchor["vah"])
    assert ep13["ref"]["poc"] == pytest.approx(anchor["poc"])
    assert ep13["extension_atr_first"] >= \
        mrev.EXTENSION_MIN_ATR
    ep = mrev.reversion_episode_at(prep, 22)
    assert ep["phase"] == mrev.PHASE_STALLED
    assert ep["extreme_close"] == pytest.approx(2013.5)
    assert ep["stall_bar"] == 21
    assert ep["momentum_at_stall"]["state"] == "decaying"
    # The value read at the stall rides along, basis named.
    assert ep["value"]["volume_basis"] == "tick_volume"
    assert ep["extension_atr"] > 0


def test_episode_never_stalls_while_value_migrates():
    """Heavy churn at the extreme: the developing POC
    migrates up to price — the auction accepted the
    extension.  The episode stays extended (no stall, no
    fade): this family's abstention on 7c's fact."""
    h1 = _migrating_h1()
    prep = mrev.PreparedReversion(h1)
    ep = mrev.reversion_episode_at(prep, 22)
    assert ep is not None and ep["phase"] == mrev.PHASE_EXTENDED
    assert dv.migrated_toward(prep.value_at(22),
                              ep["extreme_state"], 1)


def test_episode_invalidated_by_acceptance_beyond_extreme():
    h1 = _invalidated_h1()
    prep = mrev.PreparedReversion(h1)
    assert mrev.reversion_episode_at(prep, 21)["phase"] == \
        mrev.PHASE_STALLED
    ep = mrev.reversion_episode_at(prep, 22)
    assert ep["phase"] == mrev.PHASE_INVALIDATED
    assert ep["death_kind"] == "acceptance"
    assert ep["death_bar"] == 22


def test_episode_breaks_through_defended_swing():
    h1 = _fc_h1()
    prep = mrev.PreparedReversion(h1)
    ep = mrev.reversion_episode_at(prep, 22)
    assert ep["phase"] == mrev.PHASE_BROKE
    assert ep["break_bar"] == 21
    assert ep["defended_swing"] == pytest.approx(2013.5)
    assert ep["extreme_close"] == pytest.approx(2025.0)


def test_episode_closes_back_inside_value():
    h1 = _h1_frame(_BASE + [2004, 2007, 2010.5, 2013.5,
                            2012.5, 2000.5],
                   _BASE_V + [100] * 6)
    prep = mrev.PreparedReversion(h1)
    assert mrev.reversion_episode_at(prep, 15)["phase"] == \
        mrev.PHASE_EXTENDED
    ep = mrev.reversion_episode_at(prep, 16)
    assert ep["phase"] == mrev.PHASE_CLOSED


def test_episode_expires_when_extension_outlives_window():
    """A jump extension that keeps making extremes for the
    full window without stalling is a trend repricing:
    the episode expires on the clock (first bar + 12)."""
    base10 = [2000, 2002, 1998, 2001, 1999, 2003, 1997,
              2000, 2002, 2000]
    h1 = _h1_frame(base10 + [2013, 2014.5, 2016, 2017.5, 2019,
                             2020.5, 2022, 2023.5, 2025, 2026.5,
                             2028, 2029.5, 2031],
                   [100] * 23)
    prep = mrev.PreparedReversion(h1)
    assert mrev.reversion_episode_at(prep, 21)["phase"] == \
        mrev.PHASE_EXTENDED
    ep = mrev.reversion_episode_at(prep, 22)
    assert ep["phase"] == mrev.PHASE_EXPIRED
    assert ep["first_bar"] == 10


def test_no_episode_on_a_balanced_day():
    h1 = _h1_frame(_BASE + _BASE[:12],
                   _BASE_V + [100] * 12)
    prep = mrev.PreparedReversion(h1)
    assert mrev.reversion_episode_at(prep, 22) is None


def test_episode_reads_are_causal():
    """The episode at bar j is a function of bars <= j:
    identical on the frame truncated at j."""
    h1 = _stall_h1()
    full = mrev.PreparedReversion(h1)
    for j in (13, 17, 21, 22):
        trunc = mrev.PreparedReversion(h1.iloc[:j + 1])
        assert mrev.reversion_episode_at(trunc, j) == \
            mrev.reversion_episode_at(full, j)


def test_down_extension_mirrors_up():
    h1 = _mirror_h1(_stall_h1())
    prep = mrev.PreparedReversion(h1)
    ep = mrev.reversion_episode_at(prep, 22)
    assert ep["phase"] == mrev.PHASE_STALLED
    assert ep["direction"] == -1
    assert ep["extreme_close"] == pytest.approx(4000 - 2013.5)


# ------------------------------------------------------- extension fade

def test_fade_signal_shape_and_context():
    h1 = _stall_h1()
    m15 = _stall_m15(h1)
    pb = _fade_pb(h1)
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.direction == -1
    assert sig.entry_ref == pytest.approx(
        float(m15["close"].iloc[sig.index]))
    prep = mrev.PreparedReversion(h1)
    ep = mrev.reversion_episode_at(prep, 22)
    # Staged rotation: POC first, opposite edge second —
    # both read from the anchored reference, not assumed.
    assert sig.targets == [pytest.approx(ep["ref"]["poc"]),
                           pytest.approx(ep["ref"]["val"])]
    # Stop beyond the extension extreme (+ buffer).
    atr_f = prep.atr(ep["stall_bar"])
    assert sig.invalidation == pytest.approx(
        ep["extreme_price"] + mrev.STOP_BUFFER_ATR * atr_f)
    assert sig.invalidation > ep["extreme_close"]
    ctx = sig.live_context
    assert ctx["family"] == "mean_reversion"
    assert ctx["hypothesis"] == "extension_fade"
    assert ctx["volume_basis"] == "tick_volume"
    assert ctx["management_horizon"] == "intraday"
    assert ctx["management_context_timeframe"] == "H1"
    assert ctx["scenario_clocks"] == {
        "formation_tf": "H1", "refinement_tf": None,
        "trigger_tf": "M15", "entry_tf": "M15",
        "management_tf": "H1"}
    assert ctx["regime_gate"]["verdict"] == "pass"
    assert ctx["extension"]["extreme_close"] == \
        pytest.approx(2013.5)
    assert any("thesis exit" in t
               for t in ctx["invalidation_terms"])


def test_fade_m5_instance_declares_scalp_horizon():
    h1 = _stall_h1()
    m5 = _m5_from_h1(h1, extra_bars=24)
    m5 = _set_tail(m5, 276, [2011.9, 2011.4, 2010.9, 2011.5,
                             2011.0, 2010.3, 2009.8, 2009.3,
                             2008.9, 2008.5, 2008.1, 2007.7]
                   + [2007.3] * 12)
    pb = _fade_pb(h1, trigger_tf="M5")
    sigs = pb.live_signal([], len(m5) - 1, m5)
    assert len(sigs) == 1
    assert sigs[0].live_context["management_horizon"] == "scalp"
    assert sigs[0].live_context["scenario_clocks"][
        "trigger_tf"] == "M5"


def test_fade_abstains_while_value_still_migrating():
    h1 = _migrating_h1()
    m15 = _stall_m15(h1)
    pb = _fade_pb(h1)
    assert pb.live_signal([], len(m15) - 1, m15) == []


def test_fade_abstains_before_the_stall():
    """Mid-churn (formation bar 20, momentum still
    sustained): no stall, no turn, no signal."""
    h1 = _stall_h1()
    m15 = _m15_from_h1(h1).iloc[:84]
    pb = _fade_pb(h1)
    assert pb.live_signal([], len(m15) - 1, m15) == []


def test_fade_down_extension_is_long_to_poc_then_vah():
    h1 = _mirror_h1(_stall_h1())
    m15 = _m15_from_h1(h1, extra_bars=8)
    m15 = _set_tail(m15, 92, [1988.2, 1988.8, 1988.3, 1988.7,
                              1989.4, 1989.8, 1990.2, 1990.6])
    pb = _fade_pb(h1)
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.direction == 1
    prep = mrev.PreparedReversion(h1)
    ep = mrev.reversion_episode_at(prep, 22)
    assert sig.targets == [pytest.approx(ep["ref"]["poc"]),
                           pytest.approx(ep["ref"]["vah"])]
    assert all(t > sig.entry_ref for t in sig.targets)
    assert sig.invalidation < ep["extreme_close"]


# ---------------------------------------------------------- regime gate

def test_gate_refuses_fade_inside_clean_trend():
    """The family's honesty: the same stall that fires in
    rotation is REFUSED when the formation TF reads a
    clean trend — and the refusal is a named state, not a
    silent absence."""
    h1 = _stall_h1()
    m15 = _stall_m15(h1)
    pb = _fade_pb(h1, _situation(phase="clean_trend", bias=1))
    assert pb.live_signal([], len(m15) - 1, m15) == []
    pending = pb.pending_state(m15)
    assert pending["phase"] == "regime_refused"
    assert pending["episode_phase"] == mrev.PHASE_STALLED
    gate = pending["gate"]
    assert gate["verdict"] == "refused"
    assert gate["basis"] == "clean_trend"
    assert gate["source"] == "situation"
    assert gate["structure_phase"] == "clean_trend"


def test_gate_refuses_inside_breakout_too():
    h1 = _stall_h1()
    m15 = _stall_m15(h1)
    pb = _fade_pb(h1, _situation(phase="breakout", bias=1))
    assert pb.live_signal([], len(m15) - 1, m15) == []
    assert pb.pending_state(m15)["phase"] == "regime_refused"


def test_gate_post_exhaustion_override_fires():
    """Formation TF reads clean_trend, but the extension
    itself is the named post-exhaustion state (momentum
    absorbed at the decision bar, >= 4 bars without close
    progress): the gate passes BY ITS OVERRIDE, named."""
    h1 = _exhausted_h1()
    m15 = _exhausted_m15(h1)
    pb = _fade_pb(h1, _situation(phase="clean_trend", bias=1))
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    gate = sigs[0].live_context["regime_gate"]
    assert gate["verdict"] == "pass"
    assert gate["basis"] == "post_exhaustion"
    assert gate["momentum_now"] == "absorbed"
    assert gate["no_progress_bars"] >= \
        mrev.POST_EXHAUSTION_MIN_BARS


def test_gate_passes_in_rotational_situation():
    h1 = _stall_h1()
    m15 = _stall_m15(h1)
    pb = _fade_pb(h1, _situation(phase="range", bias=0))
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    gate = sigs[0].live_context["regime_gate"]
    assert gate["basis"] == "rotational_state"
    assert gate["source"] == "situation"


# -------------------------------------------------- failed continuation

def test_failed_continuation_signal_shape():
    h1 = _fc_h1()
    m15 = _m15_from_h1(h1)
    pb = _fc_pb(h1)
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    sig = sigs[0]
    # First M15 close after the break bar (21) closes.
    assert sig.index == 88
    assert sig.direction == -1
    prep = mrev.PreparedReversion(h1)
    ep = mrev.reversion_episode_at(prep, 22)
    assert sig.targets == [pytest.approx(ep["ref"]["poc"]),
                           pytest.approx(ep["ref"]["val"])]
    assert sig.invalidation > ep["extreme_close"]
    ctx = sig.live_context
    assert ctx["hypothesis"] == "failed_continuation"
    assert ctx["regime_gate"]["basis"] == "structure_break"


def test_failed_continuation_silent_without_a_break():
    h1 = _stall_h1()
    m15 = _stall_m15(h1)
    pb = _fc_pb(h1)
    assert pb.live_signal([], len(m15) - 1, m15) == []


# ------------------------------------------------------------- registry

def test_registry_partition_covers_reversion():
    for spec in REVERSION_SPECS:
        assert spec in PLAYBOOK_REGISTRY
        assert spec not in ROSTER_SPECS
        assert spec.playbook_class.live is False
        assert spec.shadow_evaluation is True
    assert {s.name for s in ROSTER_SPECS} | {
        s.name for s in NATIVE_SPECS} | {
        s.name for s in VOL_SPECS} | {
        s.name for s in TREND_SPECS} | {
        s.name for s in AUCTION_SPECS} | {
        s.name for s in REVERSION_SPECS} == {
        s.name for s in PLAYBOOK_REGISTRY}
    assert len(PLAYBOOK_REGISTRY) == 27
    assert "no order authority" in FLOOR_AUTHORITY[
        ROUTE_MEAN_REVERSION]


def test_routes_declared_shadow_and_never_governing(tmp_path):
    runner, _ = _scan_runner(tmp_path)
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    for name in ("mean_reversion_extension_fade",
                 "mean_reversion_failed_continuation"):
        assert runner._judgment_route_state(name) == \
            "shadow_declared"
        assert runner._judgment_route_source(name) == "declared"
        assert table[name]["governing_now"] is False
        assert JUDGMENT_ROUTE_PLAYBOOK[name] == name
    assert set(reversion_promotion.REVERSION_JUDGMENT_ROUTES) == \
        set(mt5_runner.REVERSION_JUDGMENT_ROUTE_NAMES)


def test_family_has_no_order_path():
    from goldenswing.execution.route_promotion import (
        ORDER_ROUTE_BY_PLAYBOOK)
    for name in ("mean_reversion_extension_fade",
                 "mean_reversion_failed_continuation"):
        assert name not in ORDER_ROUTE_BY_PLAYBOOK


def test_undeclared_names_still_fail_closed(tmp_path):
    from test_route_authority import (
        _candidate, _decision, _evidence, _jc_runner)
    runner, _ = _scan_runner(tmp_path)
    assert runner._judgment_route_state(
        "mean_reversion_typo") == "shadow_declared"
    assert runner._judgment_route_source(
        "mean_reversion_typo") == "undeclared"
    for mode in ("active", "shadow"):
        jc = _jc_runner(tmp_path / mode, judgment_mode=mode)
        _bundle, refusal = jc._judgment_authority(
            candidate=_candidate(), decision=_decision(),
            evidence=_evidence(), intended_tf="H1",
            eval_now=jc._evaluation_now(),
            route="mean_reversion_typo")
        assert refusal is not None
        assert "undeclared" in refusal
        alarms = [r for r in jc.journal.records()
                  if r["type"] == "undeclared_route"]
        assert len(alarms) == 1
        assert alarms[0]["route"] == "mean_reversion_typo"


# ------------------------------------------------------- management (P6)

def test_management_block_intraday_standard_on_formation_clock():
    h1 = _stall_h1()
    m15 = _stall_m15(h1)
    sig = _fade_pb(h1).live_signal([], len(m15) - 1, m15)[0]
    block = management_block_for("mean_reversion_extension_fade",
                                 sig)
    assert block["management_horizon"] == "intraday"
    assert block["management_policy_id"] == "standard_v1"
    assert block["management_context_timeframe"] == "H1"
    assert block["management_invalidation"] == sig.invalidation
    assert block["management_target"] == sig.targets[0]
    from goldenswing.execution.playbook_registry import (
        policy_params_for)
    params = policy_params_for(block["management_policy_id"])
    assert params["thesis_exit"] == "parent_thesis_invalidation"


def test_management_block_scalp_for_m5_instance():
    h1 = _stall_h1()
    m5 = _m5_from_h1(h1, extra_bars=24)
    m5 = _set_tail(m5, 276, [2011.9, 2011.4, 2010.9, 2011.5,
                             2011.0, 2010.3, 2009.8, 2009.3,
                             2008.9, 2008.5, 2008.1, 2007.7]
                   + [2007.3] * 12)
    sig = _fade_pb(h1, trigger_tf="M5").live_signal(
        [], len(m5) - 1, m5)[0]
    block = management_block_for("mean_reversion_extension_fade",
                                 sig)
    assert block["management_horizon"] == "scalp"
    assert block["management_policy_id"] == "standard_scalp_v1"
    assert block["management_context_timeframe"] == "H1"


def test_qm_m5_fixed_bracket_never_horizon_overridden():
    """Step 13 is closed: even a (hypothetical) horizon-
    carrying context cannot move qm_m5 off its fixed
    bracket — the name resolution wins by name (the 7b
    guard, re-pinned beside this family's wiring)."""
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


def _book_setup(sig, *, playbook="mean_reversion_extension_fade",
                setup_id="reversion:1"):
    return TrackedSetup(
        setup_id=setup_id, parent_setup_id=setup_id,
        playbook=playbook, route=ROUTE_MEAN_REVERSION,
        direction=int(sig.direction),
        entry_ref=sig.entry_ref, invalidation=sig.invalidation,
        targets=list(sig.targets),
        management=management_block_for(playbook, sig),
        formation_ts="2026-01-05T14:00:00",
        touch_ts="2026-01-05T22:45:00",
        created_ts="2026-01-05T22:45:00",
        trigger_tf=sig.live_context["trigger_tf"],
        validity_bars=6, expiry_ts="2026-01-06T04:00:00",
        live_context={"scenario_clocks":
                          dict(sig.live_context["scenario_clocks"]),
                      "family": "mean_reversion",
                      "invalidation_terms":
                          sig.live_context["invalidation_terms"]})


def test_book_takes_instance_clocks_and_freezes_plan():
    h1 = _stall_h1()
    m15 = _stall_m15(h1)
    sig = _fade_pb(h1).live_signal([], len(m15) - 1, m15)[0]
    setup = _book_setup(sig)
    book = ScenarioBook()
    book.propose_from_setup(setup, ts="2026-01-05T22:45:00")
    sc = book.scenarios["reversion:1"]
    assert sc["clocks"] == sig.live_context["scenario_clocks"]
    assert sc["horizon"] == "intraday"
    from goldenswing.execution.setup_lifecycle import STAGE_FILLED
    book.sync_setup(setup, ts="2026-01-05T23:00:00",
                    stage=STAGE_FILLED)
    frozen = book.scenarios["reversion:1"]["management_context"]
    assert frozen["management_plan"]["policy_id"] == "standard_v1"
    assert frozen["management_plan"]["management_tf"] == "H1"
    terms = frozen["management_plan"]["invalidation"]["terms"]
    assert any("close beyond invalidation" in t for t in terms)
    assert any("parent thesis invalidated" in t for t in terms)


# --------------------------------------------------------- runner wiring

def _family_runner(tmp_path, h1_frame, m15_frame, *,
                   situation=None, m5_frame=None):
    runner, _ = _scan_runner(tmp_path)
    frames = {"H1": h1_frame, "M15": m15_frame,
              "H4": _h1_frame([2000.0] * 40, [100] * 40),
              "D1": _h1_frame([2000.0] * 40, [100] * 40)}
    if m5_frame is not None:
        frames["M5"] = m5_frame
    runner._stack = SimpleNamespace(frames=frames)
    snap = situation if situation is not None else _situation()
    runner._situation_keeper = SimpleNamespace(
        latest=lambda: snap,
        latest_id=lambda: snap["situation_id"])
    return runner


def test_runner_scan_journals_evaluations_verdicts_and_book(
        tmp_path):
    # The M15 frame ends AT the turn bar: a scheduled sweep
    # honors signals on the closing bar (the fresh-instance
    # design), never stale history.
    h1 = _stall_h1()
    m15_full = _stall_m15(h1)
    turn_idx = _fade_pb(h1).live_signal(
        [], len(m15_full) - 1, m15_full)[0].index
    m15 = m15_full.iloc[:turn_idx + 1]
    runner = _family_runner(tmp_path, h1, m15,
                            m5_frame=_m5_from_h1(h1))
    runner._run_mean_reversion_family(("M15", "M5"), {},
                                      source="live_close")
    records = runner.journal.records()
    evals = [r for r in records if r["type"] == "evaluation"
             and r.get("kind") == "playbook_eval"
             and r["playbook"] == "mean_reversion_extension_fade"]
    assert evals and all(
        r["situation_id"] == "sit-reversion-1" for r in evals)
    m15_evals = [r for r in evals
                 if r["evaluation_clock"] == "mean_reversion_m15"]
    emitted = [r for r in m15_evals
               if r["outcome"] == "candidate_emitted"]
    assert len(emitted) == 1
    # The family's context evidence rides on the record:
    # family identity, the D1 read as CONTEXT only, and
    # the volume basis NAMED (tick volume, never dressed).
    assert emitted[0]["evidence"]["family"] == "mean_reversion"
    assert "never" in emitted[0]["evidence"]["d1_role"]
    assert emitted[0]["evidence"]["volume_basis"] == "tick_volume"
    assert "tick-volume" in \
        emitted[0]["evidence"]["volume_basis_role"]
    # The §G stamp: the scan's passive verdict is journaled
    # as a judgment_shadow identifiable as THIS route's
    # history.
    shadows = [r for r in records if r["type"] == "judgment_shadow"
               and (r.get("judgment") or {}).get(
                   "authority_route")
               == "mean_reversion_extension_fade"]
    assert len(shadows) == 1
    assert shadows[0]["judgment"]["authority"] == "shadow_declared"
    # The scenario entered the book as a shadow proposal
    # with the family's clocks.
    setups = runner._setup_store.active()
    assert len(setups) == 1
    book = runner._scenario_book_get()
    sc = book.scenarios[setups[0].setup_id]
    assert sc["family"] == "mean_reversion_extension_fade"
    assert sc["authority"] == "shadow"
    assert sc["horizon"] == "intraday"
    assert sc["clocks"]["formation_tf"] == "H1"
    assert sc["clocks"]["management_tf"] == "H1"
    assert sc["clocks"]["entry_tf"] == "M15"
    # No order of any kind was placed.
    assert not any("order" in str(r.get("type")) for r in records)


def test_runner_family_clocks_cover_all_playbooks(tmp_path):
    """Every family playbook is evaluated and accountable
    on both of the family's trigger clocks."""
    h1 = _stall_h1()
    runner = _family_runner(tmp_path, h1, _stall_m15(h1),
                            m5_frame=_m5_from_h1(h1))
    runner._run_mean_reversion_family(("M15", "M5"), {},
                                      source="live_close")
    clocks = {(r["playbook"], r["evaluation_clock"])
              for r in runner.journal.records()
              if r["type"] == "evaluation"}
    for name in ("mean_reversion_extension_fade",
                 "mean_reversion_failed_continuation"):
        assert (name, "mean_reversion_m15") in clocks
        assert (name, "mean_reversion_m5") in clocks


def test_runner_journals_arm_transitions(tmp_path):
    h1 = _stall_h1()
    runner = _family_runner(tmp_path, h1, _stall_m15(h1))
    runner._run_mean_reversion_family(("M15",), {},
                                      source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "mean_reversion_arm"]
    assert len(arms) == 1
    arm = arms[0]
    assert arm["family"] == "mean_reversion"
    assert arm["phase"] == "stalled" and arm["direction"] == 1
    assert arm["formation_tf"] == "H1"
    assert arm["trigger_tf"] == "M15"
    assert arm["situation_id"] == "sit-reversion-1"
    # The arm record names the basis: tick volume.
    assert arm["volume_basis"] == "tick_volume"
    assert arm["value"]["volume_basis"] == "tick_volume"
    assert arm["regime_gate"]["verdict"] == "pass"
    # Re-running the same scan journals no duplicate arm.
    runner._run_mean_reversion_family(("M15",), {},
                                      source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "mean_reversion_arm"]
    assert len(arms) == 1


def test_runner_journals_regime_refused_arm(tmp_path):
    """The gate's refusal is itself an arm record — Pass 9
    grades what the family refused and why."""
    h1 = _stall_h1()
    runner = _family_runner(
        tmp_path, h1, _stall_m15(h1),
        situation=_situation(phase="clean_trend", bias=1))
    runner._run_mean_reversion_family(("M15",), {},
                                      source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "mean_reversion_arm"]
    assert len(arms) == 1
    assert arms[0]["phase"] == "regime_refused"
    assert arms[0]["regime_gate"]["basis"] == "clean_trend"
    assert arms[0]["regime_gate"]["verdict"] == "refused"


def test_runner_journals_invalidated_arm(tmp_path):
    h1 = _invalidated_h1()
    runner = _family_runner(tmp_path, h1, _m15_from_h1(h1))
    runner._run_mean_reversion_family(("M15",), {},
                                      source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "mean_reversion_arm"]
    assert [a["phase"] for a in arms] == ["invalidated"]
    assert arms[0]["episode"]["death_kind"] == "acceptance"
    assert arms[0]["episode"]["death_ts"] is not None


def test_situation_location_carries_extension_atr(tmp_path):
    """The §6.4 read on the situation itself: the signed
    distance of price from the DEVELOPING POC in H1 ATR
    units, computed from the same state the family reads."""
    runner, _ = _scan_runner(tmp_path)
    h1 = _stall_h1()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    price = float(h1["close"].iloc[-1])
    blocks = runner._situation_blocks(
        {"mid": price, "spread": 0.3}, h1.index[-1], price)
    location = blocks["location"]
    state = dv.developing_state(h1, len(h1) - 1)
    expected = (price - state["poc"]) / atr14(h1)
    assert location["extension_atr"] == pytest.approx(expected)
    assert location["extension_atr"] > 0
    assert location["value"]["developing_state"][
        "volume_basis"] == "tick_volume"


# ------------------------------------------------- promotion evaluation

def _arm_records(route, take_rs, abstain_rs, *, start=0):
    """Route-stamped shadow verdicts + judgment
    reconciliations in the evaluator's own record shapes —
    the §G suite's fabricators, re-pointed at this family's
    playbooks."""
    from test_range_promotion_evaluation import _arm as _rp_arm
    return (_rp_arm(route, "take", take_rs, start=start)
            + _rp_arm(route, "abstain", abstain_rs,
                      start=start + 1000))


def test_promotion_evaluation_insufficient_then_ready_binds_nothing(
        tmp_path):
    route = "mean_reversion_extension_fade"
    state = reversion_promotion.new_state()
    few = _arm_records(route, [1.0] * 6, [0.0] * 6)
    rec = reversion_promotion.cycle(
        state=state, records=few, now=None,
        bar="2026-11-03 00:00")
    assert rec is not None and \
        rec["type"] == "reversion_promotion_evaluation"
    row = rec["routes"][route]
    assert row["verdict"] == range_promotion.VERDICT_INSUFFICIENT
    many = _arm_records(route, [1.2] * 22, [-0.3] * 22,
                        start=5000)
    rec2 = reversion_promotion.cycle(
        state=state, records=many, now=None,
        bar="2026-11-04 00:00")
    assert rec2 is not None
    row = rec2["routes"][route]
    assert row["verdict"] == range_promotion.VERDICT_READY
    assert row["take"]["median_r"] == pytest.approx(1.2)
    # binds=False made visible: the verdict speaks as a
    # RECOMMENDATION, never as an effect.
    assert "RECOMMENDATION" in row["verdict_reason"]
    text = "\n".join(reversion_promotion.render_lines(state))
    assert "PROMOTION READY (recommendation only)" in text
    runner, _ = _scan_runner(tmp_path)
    assert runner._judgment_route_state(route) == "shadow_declared"


def test_promotion_host_phase_journals_and_persists(tmp_path):
    """The runner host (the §G arrangement): one self-audit
    window in -> a dated reversion_promotion_evaluation
    record in the journal, state persisted under its own
    key in the audit state file, and /status renders the
    block."""
    runner, _ = _scan_runner(tmp_path)
    records = _arm_records("mean_reversion_failed_continuation",
                           [1.0] * 3, [0.0] * 3)
    runner._reversion_promotion_phase(
        records, "2026-11-03 00:00",
        window={"records": len(records)})
    out = [r for r in runner.journal.records()
           if r["type"] == "reversion_promotion_evaluation"]
    assert len(out) == 1
    assert out[0]["routes"]["mean_reversion_failed_continuation"][
        "verdict"] == range_promotion.VERDICT_INSUFFICIENT
    assert "reversion_promotion" in runner._audit_state()
    assert "mean reversion promotion (Pass 7d)" in \
        runner.reversion_promotion_line()


def test_promotion_evaluation_one_shot_ignores_other_routes():
    records = _arm_records("mean_reversion_failed_continuation",
                           [1.0] * 22, [0.0] * 22)
    snap = reversion_promotion.evaluate_records(records)
    assert snap["routes"]["mean_reversion_failed_continuation"][
        "verdict"] == range_promotion.VERDICT_READY
    assert snap["routes"]["mean_reversion_extension_fade"][
        "verdict"] == range_promotion.VERDICT_INSUFFICIENT
    # The other families' histories never leak in either.
    for other in ("range_qm", "vol_expansion_escape",
                  "trend_impulse_pullback",
                  "auction_value_acceptance"):
        snap2 = reversion_promotion.evaluate_records(
            _arm_records(other, [1.0] * 22, [0.0] * 22))
        assert all(row["verdict"] ==
                   range_promotion.VERDICT_INSUFFICIENT
                   for row in snap2["routes"].values())
