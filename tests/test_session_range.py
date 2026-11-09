"""MarketSituation Pass 7e (SPEC §6.5) — the session /
opening range family: the day's opening structures.

The family's ground, distinct from every sibling:

- the references are the day's own FROZEN session facts,
  read from the engine's existing session organs —
  live/day_context.py's NY trading day + Initial Balance
  (the day's first H1 bar, frozen when the second bar
  prints), live/session_timeframes.py's DST-aware
  session partition, and the prior day's value area
  (live/volume_profile.py — a TICK-VOLUME profile, named
  on every record that carries it).  Nothing here
  invents a second clock;
- four hypotheses, separately declared: the opening-
  range breakout (acceptance = >= 2 closes beyond the
  frozen edge WITH follow-through — never the first
  poke), the opening-range failure (a probe that closes
  back inside before acceptance → staged rotation to
  the mid, then the opposite edge), the opening-drive
  reversal (the day opened outside prior value and
  London swept the overnight extreme and rejected), and
  the Asia-range sweep (London's raid of the day's first
  value on a day WITHOUT the drive premise — the two
  premises partition the same event, never double-fire);
- time honesty is declared and enforced: entries exist
  only in the day's tradable sessions (London / overlap
  / New York, by the entry bar's own session label); a
  resolution that lands in the off hours is journaled
  as the fact it is and refused as an entry; a day that
  forms no range abstains with its reason on the arm
  record; a range that never resolves abstains as
  past_cutoff when the entry window closes;
- the route contract: all four routes DECLARED shadow
  (Pass 5 machinery), governing in no mode, no order
  path exists for the partition, undeclared names still
  fail closed;
- the Pass 6 management shape: the family's horizon is
  CONSTANT (intraday — every sequence is a session
  object), resolved by the registry default on the
  formation clock (H1); the book takes the signal's
  declared clocks and the plan freezes at entry;
  qm_m5's fixed bracket is never horizon-overridden;
- the runner wiring: scheduled formation-clock scans
  journal evaluations (situation-stamped, family +
  volume-basis evidence with the basis NAMED),
  route-stamped judgment_shadow verdicts, arm records
  (corroborated transitions, invalidations AND the
  standing abstentions), and book scenarios — and place
  no orders;
- the situation wiring: location carries opening_range,
  the frozen IB as the situation's own measurement,
  from the same day object the family reads;
- the promotion evaluation
  (execution/session_promotion.py): the shared evaluator
  over the family's own route-stamped history;
  promotion_ready is a recommendation, binds nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from test_step14_setup_lifecycle import _scan_runner

import goldenswing.execution.mt5_runner as mt5_runner
from goldenswing.execution import range_promotion, session_promotion
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
    ROUTE_SESSION_RANGE,
    SESSION_SPECS,
    TREND_SPECS,
    VOL_SPECS,
)
from goldenswing.execution.scenario_book import ScenarioBook
from goldenswing.execution.setup_lifecycle import TrackedSetup
from goldenswing.live import session_range as sr
from goldenswing.live.multitf import atr14
from goldenswing.playbooks.session_range import (
    SessionAsiaSweepPlaybook,
    SessionDriveReversalPlaybook,
    SessionOpeningBreakoutPlaybook,
    SessionOpeningFailurePlaybook,
)

# ------------------------------------------------------------ fixtures
#
# Deterministic H1 bars on ONE NY trading day: the frame
# starts Sunday 2026-01-04 23:00 UTC, so bar 0 is the
# trading day's first bar (its Initial Balance) and the
# day key is 2026-01-05.  Session layout of the 24 bars
# (winter, session_of): 0 off · 1–7 asia · 8–12 london ·
# 13–16 overlap · 17–21 newyork · 22–23 off.  M15 frames
# are interpolated from the H1 closes (4 bars per H1
# bar, so the first M15 bar after formation bar k is
# index (k+1)*4); the drive fixture's weekend gap is
# preserved by interpolating each day separately —
# MT5 frames are timestamp-true, and the family's
# cross-clock reads are timestamp reads.

DAY_START = "2026-01-04 23:00"


def _bar(o, h, l, c, v=100.0):
    return (o, h, l, c, v)


def _frame(bars, start=DAY_START):
    idx = pd.date_range(start, periods=len(bars), freq="1h")
    return pd.DataFrame(
        bars, columns=["open", "high", "low", "close", "volume"],
        index=idx)


def _interp(h1, steps=4):
    closes, prev = [], float(h1["close"].iloc[0])
    for c in h1["close"]:
        for k in range(steps):
            closes.append(prev + (float(c) - prev)
                          * (k + 1) / steps)
        prev = float(c)
    bars, p = [], closes[0]
    for i, c in enumerate(closes):
        bars.append((p, max(p, c) + 0.3 + 0.04 * (i % 2),
                     min(p, c) - 0.3 - 0.05 * (i % 3), c, 25.0))
        p = c
    idx = pd.date_range(h1.index[0], periods=len(bars),
                        freq="15min")
    return pd.DataFrame(
        bars, columns=["open", "high", "low", "close", "volume"],
        index=idx)


_IB_A = _bar(2000, 2002, 1998, 2001)   # IB [1998, 2002], mid 2000


def _asia_quiet():
    out, prev = [], 2001.0
    for c in [2000.5, 1999.5, 2000.0, 2001.0, 1999.0, 2000.0,
              2000.5]:
        out.append(_bar(prev, max(prev, c) + 0.5,
                        min(prev, c) - 0.4, c))
        prev = c
    return out


def _accept_h1():
    """Probe at London bar 8 (close 2003 beyond the IB
    high), accepted at bar 9 (second close beyond, body-
    dominated): the breakout's ground."""
    return _frame([_IB_A] + _asia_quiet() + [
        _bar(2000.5, 2003.4, 2000.2, 2003.0),
        _bar(2003.0, 2003.8, 2002.9, 2003.6),
        _bar(2003.6, 2004.6, 2003.2, 2004.2),
        _bar(2004.2, 2005.0, 2003.8, 2004.6),
    ] + [_bar(2004.6, 2005.2, 2004.0, 2004.8)] * 12)


def _reject_h1():
    """Probe at bar 8, close back inside at bar 9: the
    failure's ground."""
    return _frame([_IB_A] + _asia_quiet() + [
        _bar(2000.5, 2003.4, 2000.2, 2003.0),
        _bar(2002.5, 2003.2, 2000.5, 2001.0),
        _bar(2001.0, 2001.6, 1999.8, 2000.4),
    ] + [_bar(2000.4, 2001.2, 1999.6, 2000.2)] * 13)


def _poke_h1():
    """A WEAK probe (close barely beyond, no displacement)
    and a weaker second close beyond (no follow-through):
    never accepted; the close back inside at bar 10
    rejects it."""
    return _frame([_IB_A] + _asia_quiet() + [
        _bar(2001.8, 2002.6, 2001.5, 2002.3),
        _bar(2002.3, 2002.7, 2002.0, 2002.2),
        _bar(2002.2, 2002.5, 2000.9, 2001.2),
        _bar(2001.2, 2001.6, 2000.2, 2000.6),
    ] + [_bar(2000.6, 2001.2, 2000.0, 2000.8)] * 12)


def _sweep_h1():
    """Wide IB contains the whole day; London's first bar
    raids the Asia high (extreme 2004.5 > 2003) and closes
    back inside the Asia range."""
    asia, prev = [], 2000.0
    for c in [2001, 2002.5, 2001.5, 1999, 1998, 1999.5, 2000.5]:
        asia.append(_bar(prev, max(prev, c) + 0.5,
                         min(prev, c) - 0.5, c))
        prev = c
    return _frame([_bar(2000, 2004, 1996, 2000)] + asia + [
        _bar(2002.0, 2004.5, 2001.5, 2002.0),
        _bar(2002.0, 2002.6, 2000.8, 2001.2),
    ] + [_bar(2001.2, 2001.8, 2000.2, 2000.8)] * 14)


def _sweep_accepted_h1():
    """London CLOSES beyond the Asia edge: acceptance
    territory — the raid read dies for that side."""
    asia, prev = [], 2000.0
    for c in [2001, 2002.5, 2001.5, 1999, 1998, 1999.5, 2000.5]:
        asia.append(_bar(prev, max(prev, c) + 0.5,
                         min(prev, c) - 0.5, c))
        prev = c
    return _frame([_bar(2000, 2004, 1996, 2000)] + asia + [
        _bar(2002.0, 2004.2, 2001.8, 2003.8),
        _bar(2003.8, 2004.4, 2002.6, 2002.4),
    ] + [_bar(2002.4, 2003.0, 2001.6, 2002.2)] * 14)


def _drive_frames():
    """A balanced prior day (day key 2026-01-03, value
    ~[1989.6, 1990.7] — read from the data in the tests,
    never assumed), then a day that OPENS at 2000, far
    above prior value, whose first London bar sweeps the
    Asia high (2002.5 > 2001) and closes back inside."""
    prior, prev = [], 1990.0
    for c in [1990, 1991, 1989.5, 1990.5, 1989, 1991.5, 1990,
              1989.8, 1990.8, 1991.2, 1989.6, 1990.4, 1990,
              1991, 1989.4, 1990.6, 1990.2, 1989.7, 1991.1,
              1990.3, 1989.9, 1990.7, 1990.1, 1990.0]:
        prior.append(_bar(prev, max(prev, c) + 0.4,
                          min(prev, c) - 0.4, c))
        prev = c
    f_prior = _frame(prior, start="2026-01-02 23:00")
    asia, prev = [], 2000.0
    for c in [2000.4, 2000.8, 2000.2, 2000.6, 2000.1, 2000.5,
              2000.3]:
        asia.append(_bar(prev, max(prev, c) + 0.2,
                         min(prev, c) - 0.2, c))
        prev = c
    day = _frame([_bar(2000, 2001, 1998.5, 2000)] + asia + [
        _bar(2000.3, 2002.5, 2000.0, 2000.4),
        _bar(2000.4, 2000.9, 1999.6, 1999.9),
    ] + [_bar(1999.9, 2000.5, 1999.2, 1999.7)] * 14)
    h1 = pd.concat([f_prior, day])
    m15 = pd.concat([_interp(f_prior), _interp(day)])
    return h1, m15


def _flat_h1():
    return _frame([_bar(2000, 2000, 2000, 2000, 0.0)] * 24)


def _quiet_h1():
    """A range that forms and never resolves: the
    past-cutoff ground.  (The quiet bars' highs stay
    under the Asia high 2001.5 — a poke above it would
    be an accidental sweep, a different day.)"""
    return _frame([_IB_A] + _asia_quiet()
                  + [_bar(2000.5, 2001.4, 1999.4, 2000.4)] * 16)


def _offhours_h1():
    """Acceptance lands at bar 22 — the off hours."""
    return _frame([_IB_A] + _asia_quiet()
                  + [_bar(2000.5, 2001.4, 1999.4, 2000.4)] * 13
                  + [_bar(2000.4, 2003.4, 2000.2, 2003.0),
                     _bar(2003.0, 2003.8, 2002.9, 2003.6),
                     _bar(2003.6, 2004.2, 2003.2, 2003.9)])


def _late_ny_h1():
    """Acceptance lands at bar 20 — still New York, still
    a legal entry."""
    return _frame([_IB_A] + _asia_quiet()
                  + [_bar(2000.5, 2001.4, 1999.4, 2000.4)] * 11
                  + [_bar(2000.4, 2003.4, 2000.2, 2003.0),
                     _bar(2003.0, 2003.8, 2002.9, 2003.6),
                     _bar(2003.6, 2004.4, 2003.2, 2004.0),
                     _bar(2004.0, 2004.6, 2003.6, 2004.2),
                     _bar(2004.2, 2004.8, 2003.8, 2004.4)])


def _situation(sid="sit-session-1", *, ib_high=2002.0,
               ib_low=1998.0, zones_above=None,
               zones_below=None):
    """The situation snapshot in live/situation.py's
    schema: day levels matching fixture A's IB (arm
    corroboration), location zones (the breakout's
    second target), the day's opening type as recorded
    evidence."""
    return {"situation_id": sid,
            "location": {
                "day_levels": {
                    "ib_high": ib_high, "ib_low": ib_low,
                    "overnight_high": 2001.5,
                    "overnight_low": 1998.6},
                "zones_above": zones_above or [],
                "zones_below": zones_below or []},
            "regime": {"day": {"opening_type": "open_drive"}},
            "relationships": []}


def _pb(cls, h1, *, situation=None):
    pb = cls()
    pb.set_formation(h1, "H1")
    pb.set_formation_tf("M15")
    if situation is not None:
        pb.set_situation(situation)
    return pb


def _day(h1, j):
    return sr.session_day_at(sr.PreparedSession(h1), j)


# ------------------------------------------------------------ core read

def test_range_is_the_frozen_ib():
    h1 = _accept_h1()
    early = _day(h1, 9)
    late = _day(h1, 23)
    assert early["ib"] == late["ib"]
    assert early["ib"]["high"] == 2002.0
    assert early["ib"]["low"] == 1998.0
    assert early["ib"]["width"] == 4.0
    assert early["ib"]["mid"] == 2000.0
    # Later bars traded to 2005.2 — the frozen range did
    # not widen by a tick.
    assert float(h1["high"].max()) > 2005.0


def test_first_close_beyond_is_a_probe_not_acceptance():
    h1 = _accept_h1()
    probing = _day(h1, 8)["breakout"]
    assert probing["phase"] == sr.PHASE_PROBING
    assert probing["outside_closes"] == 1
    accepted = _day(h1, 9)["breakout"]
    assert accepted["phase"] == sr.PHASE_ACCEPTED
    assert accepted["outside_closes"] == 2
    assert accepted["displacement"] is True
    assert accepted["resolution_bar"] == 9


def test_acceptance_needs_follow_through_not_just_closes():
    h1 = _poke_h1()
    # Two closes beyond the edge, neither displacement
    # nor progress: still only a probe.
    assert _day(h1, 9)["breakout"]["phase"] == sr.PHASE_PROBING
    # The close back inside then rejects it.
    rejected = _day(h1, 10)["breakout"]
    assert rejected["phase"] == sr.PHASE_REJECTED
    assert rejected["resolution_bar"] == 10


def test_rejection_reads_failed_auction():
    h1 = _reject_h1()
    seq = _day(h1, 9)["breakout"]
    assert seq["phase"] == sr.PHASE_REJECTED
    assert seq["direction"] == 1
    assert seq["extreme_price"] == pytest.approx(2003.4)


def test_opposite_probe_answers_after_a_rejection():
    """The traverse-then-accept shape: an up probe is
    rejected, price rotates through the range, and the
    DOWN probe is accepted — the day's second question,
    asked only after a rejection."""
    bars = [_IB_A] + _asia_quiet() + [
        _bar(2000.5, 2003.4, 2000.2, 2003.0),
        _bar(2002.5, 2003.2, 2000.5, 2001.0),
        _bar(2001.0, 2001.4, 1999.9, 2000.6),
        _bar(2000.6, 2001.0, 1999.7, 2000.1),
        _bar(2000.1, 2000.5, 1999.2, 1999.6),
        _bar(1999.6, 1999.9, 1996.8, 1997.2),
        _bar(1997.2, 1997.6, 1996.2, 1996.6),
    ] + [_bar(1996.6, 1997.2, 1995.8, 1996.2)] * 9
    h1 = _frame(bars)
    seq = _day(h1, 14)["breakout"]
    assert seq["phase"] == sr.PHASE_ACCEPTED
    assert seq["direction"] == -1


def test_asia_sweep_reads_the_raid():
    h1 = _sweep_h1()
    day = _day(h1, 9)
    assert day["asia"]["high"] == pytest.approx(2003.0)
    sweep = day["sweep"]
    assert sweep["phase"] == sr.PHASE_SWEPT
    assert sweep["direction"] == 1
    assert sweep["extreme_price"] == pytest.approx(2004.5)
    # The wide IB never probed: no breakout sequence.
    assert day["breakout"] is None


def test_london_close_beyond_asia_edge_is_not_a_sweep():
    h1 = _sweep_accepted_h1()
    assert _day(h1, 10)["sweep"] is None
    assert _day(h1, 23)["sweep"] is None


def test_drive_premise_and_rejection_read():
    h1, _m15 = _drive_frames()
    day = _day(h1, 33)
    prior = day["prior_value"]
    assert prior is not None
    assert prior["volume_basis"] == "tick_volume"
    assert day["opened_outside_value"] == 1
    assert day["day_open"] > prior["vah"]
    drive = day["drive"]
    assert drive["phase"] == sr.PHASE_DRIVE_REJECTED
    assert drive["direction"] == 1
    assert drive["extreme_price"] == pytest.approx(2002.5)
    # The core reads the sweep too — the PARTITION lives
    # in the playbooks (the drive premise owns the day).
    assert day["sweep"]["phase"] == sr.PHASE_SWEPT


def test_flat_day_forms_no_range():
    day = _day(_flat_h1(), 10)
    assert day["ib"] is None
    assert day["no_range"]["reason"] == "flat_open"
    assert day["breakout"] is None
    assert day["sweep"] is None


def test_past_cutoff_read():
    h1 = _quiet_h1()
    assert _day(h1, 20)["past_cutoff"] is False
    day = _day(h1, 23)
    assert day["past_cutoff"] is True
    assert day["last_session"] == "off"
    assert day["breakout"] is None


def test_day_reads_are_causal():
    """The day read at bar j is a function of bars <= j:
    identical on the frame truncated at j — the frozen
    range, the sequences, the premise, all of it."""
    for builder in (_accept_h1, _reject_h1, _poke_h1,
                    _sweep_h1, _quiet_h1):
        h1 = builder()
        full = sr.PreparedSession(h1)
        for j in (5, 8, 9, 12, 20, 23):
            trunc = sr.PreparedSession(h1.iloc[:j + 1])
            assert sr.session_day_at(trunc, j) == \
                sr.session_day_at(full, j)


# ------------------------------------------------------- breakout

def test_breakout_signal_shape_and_context():
    h1 = _accept_h1()
    m15 = _interp(h1)
    pb = _pb(SessionOpeningBreakoutPlaybook, h1)
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    sig = sigs[0]
    # First trigger close after the accepting formation
    # close (bar 9 -> M15 index 40).
    assert sig.index == 40
    assert sig.direction == 1
    assert sig.entry_ref == pytest.approx(
        float(m15["close"].iloc[40]))
    # Measured extension: the IB height projected from
    # the broken edge — and nothing else unthreaded.
    assert sig.targets == [pytest.approx(2006.0)]
    # Stop back inside the range, beyond the broken edge.
    atr_f = sr.PreparedSession(h1).atr(9)
    assert sig.invalidation == pytest.approx(
        2002.0 - sr.STOP_BUFFER_ATR * atr_f)
    assert sig.invalidation < 2002.0
    ctx = sig.live_context
    assert ctx["family"] == "session_range"
    assert ctx["hypothesis"] == "opening_range_breakout"
    assert ctx["formation_tf"] == "H1"
    assert ctx["trigger_tf"] == "M15"
    # The horizon is CONSTANT for this family: no
    # per-instance declaration rides the signal.
    assert "management_horizon" not in ctx
    assert ctx["management_context_timeframe"] == "H1"
    assert ctx["scenario_clocks"] == {
        "formation_tf": "H1", "refinement_tf": None,
        "trigger_tf": "M15", "entry_tf": "M15",
        "management_tf": "H1"}
    assert ctx["entry_session"] == "london"
    assert ctx["day"]["range"]["high"] == 2002.0
    assert any("thesis exit" in t
               for t in ctx["invalidation_terms"])


def test_breakout_second_target_is_the_situation_zone():
    h1 = _accept_h1()
    m15 = _interp(h1)
    sit = _situation(zones_above=[{"bottom": 2008.0,
                                   "top": 2009.5}])
    pb = _pb(SessionOpeningBreakoutPlaybook, h1, situation=sit)
    sig = pb.live_signal([], len(m15) - 1, m15)[0]
    assert sig.targets == [pytest.approx(2006.0),
                           pytest.approx(2008.0)]
    assert sig.live_context["situation_id"] == "sit-session-1"
    assert sig.live_context["opening_type"] == "open_drive"


def test_breakout_silent_until_acceptance():
    """The scan that ends between the probe and the
    acceptance emits nothing — the first poke is not a
    breakout, at the playbook level too."""
    h1 = _accept_h1()
    m15 = _interp(h1).iloc[:40]   # ends at the probe's hour
    pb = _pb(SessionOpeningBreakoutPlaybook, h1)
    assert pb.live_signal([], len(m15) - 1, m15) == []


# ------------------------------------------------------- failure

def test_failure_signal_shape():
    h1 = _reject_h1()
    m15 = _interp(h1)
    pb = _pb(SessionOpeningFailurePlaybook, h1)
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.index == 40
    assert sig.direction == -1
    # Staged rotation: IB mid first, opposite edge second.
    assert sig.targets == [pytest.approx(2000.0),
                           pytest.approx(1998.0)]
    # Stop beyond the probe extreme (+ buffer).
    atr_f = sr.PreparedSession(h1).atr(9)
    assert sig.invalidation == pytest.approx(
        2003.4 + sr.STOP_BUFFER_ATR * atr_f)
    assert sig.live_context["hypothesis"] == \
        "opening_range_failure"
    # The breakout playbook is silent on the same day:
    # one sequence, one resolution, one owner.
    other = _pb(SessionOpeningBreakoutPlaybook, h1)
    assert other.live_signal([], len(m15) - 1, m15) == []


def test_poke_family_reads_failure_never_breakout():
    h1 = _poke_h1()
    m15 = _interp(h1)
    breakout = _pb(SessionOpeningBreakoutPlaybook, h1)
    assert breakout.live_signal([], len(m15) - 1, m15) == []
    failure = _pb(SessionOpeningFailurePlaybook, h1)
    sigs = failure.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    assert sigs[0].index == 44   # after the bar-10 rejection


# ------------------------------------------------------- sweep + drive

def test_sweep_signal_shape():
    h1 = _sweep_h1()
    m15 = _interp(h1)
    pb = _pb(SessionAsiaSweepPlaybook, h1)
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.index == 36   # first trigger after bar 8
    assert sig.direction == -1
    day = _day(h1, 9)
    assert sig.targets == [pytest.approx(day["asia"]["mid"]),
                           pytest.approx(day["asia"]["low"])]
    assert sig.live_context["hypothesis"] == "asia_range_play"


def test_drive_signal_shape_and_premise_partition():
    h1, m15 = _drive_frames()
    pb = _pb(SessionDriveReversalPlaybook, h1)
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.direction == -1
    day = _day(h1, 33)
    # Staged fade: IB mid first, then the prior value
    # edge the drive left (read from the data).
    assert sig.targets == [pytest.approx(1999.75),
                           pytest.approx(
                               day["prior_value"]["vah"])]
    assert sig.live_context["hypothesis"] == \
        "opening_drive_reversal"
    assert sig.live_context["day"]["prior_value"][
        "volume_basis"] == "tick_volume"
    # The premise partition: the sweep playbook abstains
    # on a drive day — one event, one owner.
    sweep = _pb(SessionAsiaSweepPlaybook, h1)
    assert sweep.live_signal([], len(m15) - 1, m15) == []


# ------------------------------------------------------- time honesty

def test_cutoff_refuses_the_offhours_acceptance():
    """The acceptance at bar 22 is a FACT (the arm read
    says accepted) — and the family refuses it as an
    entry: a 21:00 breakout is a different bet from the
    morning shape, never silently traded as one."""
    h1 = _offhours_h1()
    m15 = _interp(h1)
    pb = _pb(SessionOpeningBreakoutPlaybook, h1)
    assert pb.live_signal([], len(m15) - 1, m15) == []
    pending = pb.pending_state(m15)
    assert pending["phase"] == "accepted"


def test_late_newyork_acceptance_still_fires():
    h1 = _late_ny_h1()
    m15 = _interp(h1)
    pb = _pb(SessionOpeningBreakoutPlaybook, h1)
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    assert sigs[0].live_context["entry_session"] == "newyork"


def test_flat_day_every_playbook_silent():
    h1 = _flat_h1()
    m15 = _interp(h1)
    for cls in (SessionOpeningBreakoutPlaybook,
                SessionOpeningFailurePlaybook,
                SessionDriveReversalPlaybook,
                SessionAsiaSweepPlaybook):
        pb = _pb(cls, h1)
        assert pb.live_signal([], len(m15) - 1, m15) == []
    pending = _pb(SessionOpeningBreakoutPlaybook,
                  h1).pending_state(m15)
    assert pending["phase"] == "no_range"
    assert pending["day"]["no_range"]["reason"] == "flat_open"


# ------------------------------------------------------------ registry

def test_registry_partition_covers_session():
    for spec in SESSION_SPECS:
        assert spec in PLAYBOOK_REGISTRY
        assert spec not in ROSTER_SPECS
        assert spec.playbook_class.live is False
        assert spec.shadow_evaluation is True
    assert {s.name for s in ROSTER_SPECS} | {
        s.name for s in NATIVE_SPECS} | {
        s.name for s in VOL_SPECS} | {
        s.name for s in TREND_SPECS} | {
        s.name for s in AUCTION_SPECS} | {
        s.name for s in REVERSION_SPECS} | {
        s.name for s in SESSION_SPECS} == {
        s.name for s in PLAYBOOK_REGISTRY}
    assert len(PLAYBOOK_REGISTRY) == 31
    assert "no order authority" in FLOOR_AUTHORITY[
        ROUTE_SESSION_RANGE]


def test_routes_declared_shadow_and_never_governing(tmp_path):
    runner, _ = _scan_runner(tmp_path)
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    for name in ("session_opening_breakout",
                 "session_opening_failure",
                 "session_drive_reversal",
                 "session_asia_sweep"):
        assert runner._judgment_route_state(name) == \
            "shadow_declared"
        assert runner._judgment_route_source(name) == "declared"
        assert table[name]["governing_now"] is False
        assert JUDGMENT_ROUTE_PLAYBOOK[name] == name
    assert set(session_promotion.SESSION_JUDGMENT_ROUTES) == \
        set(mt5_runner.SESSION_JUDGMENT_ROUTE_NAMES)


def test_family_has_no_order_path():
    from goldenswing.execution.route_promotion import (
        ORDER_ROUTE_BY_PLAYBOOK)
    for name in ("session_opening_breakout",
                 "session_opening_failure",
                 "session_drive_reversal",
                 "session_asia_sweep"):
        assert name not in ORDER_ROUTE_BY_PLAYBOOK


def test_undeclared_names_still_fail_closed(tmp_path):
    from test_route_authority import (
        _candidate, _decision, _evidence, _jc_runner)
    runner, _ = _scan_runner(tmp_path)
    assert runner._judgment_route_state(
        "session_typo") == "shadow_declared"
    assert runner._judgment_route_source(
        "session_typo") == "undeclared"
    for mode in ("active", "shadow"):
        jc = _jc_runner(tmp_path / mode, judgment_mode=mode)
        _bundle, refusal = jc._judgment_authority(
            candidate=_candidate(), decision=_decision(),
            evidence=_evidence(), intended_tf="H1",
            eval_now=jc._evaluation_now(),
            route="session_typo")
        assert refusal is not None
        assert "undeclared" in refusal
        alarms = [r for r in jc.journal.records()
                  if r["type"] == "undeclared_route"]
        assert len(alarms) == 1
        assert alarms[0]["route"] == "session_typo"


# ------------------------------------------------------- management (P6)

def test_management_block_intraday_standard_on_formation_clock():
    h1 = _accept_h1()
    m15 = _interp(h1)
    sig = _pb(SessionOpeningBreakoutPlaybook,
              h1).live_signal([], len(m15) - 1, m15)[0]
    block = management_block_for("session_opening_breakout",
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


def _book_setup(sig, *, playbook="session_opening_breakout",
                setup_id="session:1"):
    return TrackedSetup(
        setup_id=setup_id, parent_setup_id=setup_id,
        playbook=playbook, route=ROUTE_SESSION_RANGE,
        direction=int(sig.direction),
        entry_ref=sig.entry_ref, invalidation=sig.invalidation,
        targets=list(sig.targets),
        management=management_block_for(playbook, sig),
        formation_ts="2026-01-05T08:00:00",
        touch_ts="2026-01-05T09:00:00",
        created_ts="2026-01-05T09:00:00",
        trigger_tf=sig.live_context["trigger_tf"],
        validity_bars=6, expiry_ts="2026-01-06T04:00:00",
        live_context={"scenario_clocks":
                          dict(sig.live_context["scenario_clocks"]),
                      "family": "session_range",
                      "invalidation_terms":
                          sig.live_context["invalidation_terms"]})


def test_book_takes_family_clocks_and_freezes_plan():
    h1 = _accept_h1()
    m15 = _interp(h1)
    sig = _pb(SessionOpeningBreakoutPlaybook,
              h1).live_signal([], len(m15) - 1, m15)[0]
    setup = _book_setup(sig)
    book = ScenarioBook()
    book.propose_from_setup(setup, ts="2026-01-05T09:00:00")
    sc = book.scenarios["session:1"]
    assert sc["clocks"] == sig.live_context["scenario_clocks"]
    assert sc["horizon"] == "intraday"
    from goldenswing.execution.setup_lifecycle import STAGE_FILLED
    book.sync_setup(setup, ts="2026-01-05T09:15:00",
                    stage=STAGE_FILLED)
    frozen = book.scenarios["session:1"]["management_context"]
    assert frozen["management_plan"]["policy_id"] == "standard_v1"
    assert frozen["management_plan"]["management_tf"] == "H1"
    terms = frozen["management_plan"]["invalidation"]["terms"]
    assert any("close beyond invalidation" in t for t in terms)
    assert any("parent thesis invalidated" in t for t in terms)


# --------------------------------------------------------- runner wiring

def _family_runner(tmp_path, h1_frame, m15_frame, *,
                   situation=None):
    runner, _ = _scan_runner(tmp_path)
    frames = {"H1": h1_frame, "M15": m15_frame,
              "H4": _frame([_bar(2000, 2001, 1999, 2000)] * 40,
                           start="2026-01-01 00:00"),
              "D1": _frame([_bar(2000, 2001, 1999, 2000)] * 40,
                           start="2025-12-01 00:00")}
    runner._stack = SimpleNamespace(frames=frames)
    snap = situation if situation is not None else _situation()
    runner._situation_keeper = SimpleNamespace(
        latest=lambda: snap,
        latest_id=lambda: snap["situation_id"])
    return runner


def test_runner_scan_journals_evaluations_verdicts_and_book(
        tmp_path):
    # The M15 frame ends AT the entry bar: a scheduled
    # sweep honors signals on the closing bar (the
    # fresh-instance design), never stale history.
    h1 = _accept_h1()
    m15_full = _interp(h1)
    entry_idx = _pb(SessionOpeningBreakoutPlaybook,
                    h1).live_signal(
        [], len(m15_full) - 1, m15_full)[0].index
    m15 = m15_full.iloc[:entry_idx + 1]
    sit = _situation(zones_above=[{"bottom": 2008.0,
                                   "top": 2009.5}])
    runner = _family_runner(tmp_path, h1, m15, situation=sit)
    runner._run_session_range_family(("H1",), {},
                                     source="live_close")
    records = runner.journal.records()
    evals = [r for r in records if r["type"] == "evaluation"
             and r.get("kind") == "playbook_eval"
             and r["playbook"] == "session_opening_breakout"]
    assert evals and all(
        r["situation_id"] == "sit-session-1" for r in evals)
    emitted = [r for r in evals
               if r["evaluation_clock"] == "session_range_h1"
               and r["outcome"] == "candidate_emitted"]
    assert len(emitted) == 1
    # The family's context evidence rides on the record:
    # family identity, the D1 read as CONTEXT only, and
    # the volume basis NAMED (tick volume, never dressed).
    assert emitted[0]["evidence"]["family"] == "session_range"
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
               == "session_opening_breakout"]
    assert len(shadows) == 1
    assert shadows[0]["judgment"]["authority"] == "shadow_declared"
    # The scenario entered the book as a shadow proposal
    # with the family's clocks and BOTH targets (the
    # measured extension, then the situation zone).
    setups = runner._setup_store.active()
    assert len(setups) == 1
    assert setups[0].targets == [pytest.approx(2006.0),
                                 pytest.approx(2008.0)]
    book = runner._scenario_book_get()
    sc = book.scenarios[setups[0].setup_id]
    assert sc["family"] == "session_opening_breakout"
    assert sc["authority"] == "shadow"
    assert sc["horizon"] == "intraday"
    assert sc["clocks"]["formation_tf"] == "H1"
    assert sc["clocks"]["management_tf"] == "H1"
    assert sc["clocks"]["entry_tf"] == "M15"
    # No order of any kind was placed.
    assert not any("order" in str(r.get("type")) for r in records)


def test_runner_family_clocks_cover_all_playbooks(tmp_path):
    """Every family playbook is evaluated and accountable
    on the family's formation clock."""
    h1 = _accept_h1()
    runner = _family_runner(tmp_path, h1, _interp(h1))
    runner._run_session_range_family(("H1",), {},
                                     source="live_close")
    clocks = {(r["playbook"], r["evaluation_clock"])
              for r in runner.journal.records()
              if r["type"] == "evaluation"}
    for name in ("session_opening_breakout",
                 "session_opening_failure",
                 "session_drive_reversal",
                 "session_asia_sweep"):
        assert (name, "session_range_h1") in clocks


def test_runner_journals_accepted_arm_and_dedupes(tmp_path):
    h1 = _accept_h1()
    runner = _family_runner(tmp_path, h1, _interp(h1))
    runner._run_session_range_family(("H1",), {},
                                     source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "session_range_arm"]
    assert len(arms) == 1
    arm = arms[0]
    assert arm["family"] == "session_range"
    assert arm["phase"] == "accepted"
    assert arm["seq_kind"] == "breakout"
    assert arm["direction"] == 1
    assert arm["formation_tf"] == "H1"
    assert arm["trigger_tf"] == "M15"
    assert arm["situation_id"] == "sit-session-1"
    assert arm["volume_basis"] == "tick_volume"
    assert arm["day"]["range"]["high"] == 2002.0
    assert arm["day"]["range"]["low"] == 1998.0
    # Re-running the same scan journals no duplicate arm.
    runner._run_session_range_family(("H1",), {},
                                     source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "session_range_arm"]
    assert len(arms) == 1


def test_runner_journals_no_range_arm(tmp_path):
    """The no-range day abstains ON THE RECORD — Pass 9
    grades refusals too."""
    h1 = _flat_h1()
    runner = _family_runner(tmp_path, h1, _interp(h1))
    runner._run_session_range_family(("H1",), {},
                                     source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "session_range_arm"]
    assert len(arms) == 1
    assert arms[0]["phase"] == "no_range"
    assert arms[0]["day"]["no_range"]["reason"] == "flat_open"


def test_runner_journals_past_cutoff_arm(tmp_path):
    """A range that never resolves abstains as
    past_cutoff when the entry window closes."""
    h1 = _quiet_h1()
    runner = _family_runner(tmp_path, h1, _interp(h1))
    runner._run_session_range_family(("H1",), {},
                                     source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "session_range_arm"]
    assert len(arms) == 1
    assert arms[0]["phase"] == "past_cutoff"
    assert arms[0]["day"]["range"]["high"] == 2002.0


def test_situation_location_carries_opening_range(tmp_path):
    """The §6.5 read on the situation itself: the frozen
    IB (high / low / mid + width in H1 ATR units) from
    the same day object the family reads."""
    runner, _ = _scan_runner(tmp_path)
    h1 = _accept_h1()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    # The day object is a runner attribute (the scenario
    # refresh maintains it); build it the same way here —
    # on the engine's tz-aware frame shape (MT5 frames
    # are tz-aware UTC; the day object's gap read
    # converts to NY time).
    from goldenswing.live.day_context import (
        compute_day_context)
    h1_aware = _accept_h1()
    h1_aware.index = h1_aware.index.tz_localize("UTC")
    runner._day_ctx = compute_day_context({"H1": h1_aware})
    price = float(h1["close"].iloc[-1])
    blocks = runner._situation_blocks(
        {"mid": price, "spread": 0.3}, h1.index[-1], price)
    opening = blocks["location"]["opening_range"]
    assert opening["high"] == 2002.0
    assert opening["low"] == 1998.0
    assert opening["mid"] == 2000.0
    assert opening["width_atr"] == pytest.approx(
        4.0 / atr14(h1))
    assert blocks["location"]["day_levels"]["ib_high"] == 2002.0


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
    route = "session_opening_breakout"
    state = session_promotion.new_state()
    few = _arm_records(route, [1.0] * 6, [0.0] * 6)
    rec = session_promotion.cycle(
        state=state, records=few, now=None,
        bar="2026-11-03 00:00")
    assert rec is not None and \
        rec["type"] == "session_promotion_evaluation"
    row = rec["routes"][route]
    assert row["verdict"] == range_promotion.VERDICT_INSUFFICIENT
    many = _arm_records(route, [1.2] * 22, [-0.3] * 22,
                        start=5000)
    rec2 = session_promotion.cycle(
        state=state, records=many, now=None,
        bar="2026-11-04 00:00")
    assert rec2 is not None
    row = rec2["routes"][route]
    assert row["verdict"] == range_promotion.VERDICT_READY
    assert row["take"]["median_r"] == pytest.approx(1.2)
    # binds=False made visible: the verdict speaks as a
    # RECOMMENDATION, never as an effect.
    assert "RECOMMENDATION" in row["verdict_reason"]
    text = "\n".join(session_promotion.render_lines(state))
    assert "PROMOTION READY (recommendation only)" in text
    runner, _ = _scan_runner(tmp_path)
    assert runner._judgment_route_state(route) == "shadow_declared"


def test_promotion_host_phase_journals_and_persists(tmp_path):
    """The runner host (the §G arrangement): one self-audit
    window in -> a dated session_promotion_evaluation
    record in the journal, state persisted under its own
    key in the audit state file, and /status renders the
    block."""
    runner, _ = _scan_runner(tmp_path)
    records = _arm_records("session_asia_sweep",
                           [1.0] * 3, [0.0] * 3)
    runner._session_promotion_phase(
        records, "2026-11-03 00:00",
        window={"records": len(records)})
    out = [r for r in runner.journal.records()
           if r["type"] == "session_promotion_evaluation"]
    assert len(out) == 1
    assert out[0]["routes"]["session_asia_sweep"][
        "verdict"] == range_promotion.VERDICT_INSUFFICIENT
    assert "session_promotion" in runner._audit_state()
    assert "session range promotion (Pass 7e)" in \
        runner.session_promotion_line()


def test_promotion_evaluation_one_shot_ignores_other_routes():
    records = _arm_records("session_asia_sweep",
                           [1.0] * 22, [0.0] * 22)
    snap = session_promotion.evaluate_records(records)
    assert snap["routes"]["session_asia_sweep"][
        "verdict"] == range_promotion.VERDICT_READY
    assert snap["routes"]["session_opening_breakout"][
        "verdict"] == range_promotion.VERDICT_INSUFFICIENT
    # The other families' histories never leak in either.
    for other in ("range_qm", "vol_expansion_escape",
                  "trend_impulse_pullback",
                  "auction_value_acceptance",
                  "mean_reversion_extension_fade"):
        snap2 = session_promotion.evaluate_records(
            _arm_records(other, [1.0] * 22, [0.0] * 22))
        assert all(row["verdict"] ==
                   range_promotion.VERDICT_INSUFFICIENT
                   for row in snap2["routes"].values())
