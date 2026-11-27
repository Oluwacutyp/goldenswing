"""MarketSituation Pass 7c — the auction / value family.

Pins, in the GS conventions (fixtures judged by what was
observable AT THE TIME — every detector assertion runs the
causal read at the bar itself, and one test proves the read
at (j, m) is identical on the frames truncated at (j, m)):

- the developing-value extension (live/developing_value.py):
  the §6.3 shared value layer — POC / VAH / VAL as they
  BUILD over the trading day, causal per bar, exactly
  volume_profile's construction (parity pinned at the day's
  last bar), migration read against the same day's previous
  bar with a stated dead-band, node structure (HVN shelves,
  LVNs flanked by shelves), and the volume basis NAMED on
  every state (tick volume — never dressed as traded
  volume).  Absent stays absent: too few bars / flat /
  no volume name their status and carry no levels;
- the measurement core (live/auction_value.py): a probe is
  a close beyond the pre-probe value after a close inside
  it (a wick alone forms nothing); acceptance = >= 2
  closes holding beyond the probed edge + developing value
  migrating toward the probe; rejection = a close back
  inside before acceptance; neither inside the window =
  expired; an accepted sequence whose re-pricing does not
  hold, and a rejected sequence the auction re-accepts,
  both read INVALIDATED;
- the playbooks: acceptance enters on the M15 retest that
  holds outside the probed edge (a prior M15 close back
  inside voids it; targets are levels beyond the entry —
  a consumed projection is an abstention); rejection
  enters at the first M15 close after the rejection,
  rotating to POC then the opposite edge; LVN traversal
  enters on the crossing close into a flanked thin node,
  once per contiguous visit to the thin region;
- the route contract: all three routes DECLARED shadow
  (Pass 5 machinery), governing in no mode, no order path
  exists for the partition, undeclared names still fail
  closed;
- the Pass 6 management shape: intraday horizon,
  standard_v1 running on the formation clock (H1); the
  book takes the instance's clocks and the plan freezes
  at entry; qm_m5's fixed bracket is never
  horizon-overridden;
- the runner wiring: scheduled scans journal evaluations
  (situation-stamped, family + volume-basis evidence),
  route-stamped judgment verdicts, arm records
  (corroborated transitions AND invalidations, basis
  named), and book scenarios — and place no orders;
- the situation wiring: location.value carries the
  developing_state built by the same extension;
- the evaluation (execution/evaluation.py): the shared
  S4-pairing measurement — route-stamped judgment
  verdicts + their +4h reconciliations, paired per arm;
  the verdict is a measurement only, never an action.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from test_step14_setup_lifecycle import _scan_runner

import goldenswing.execution.mt5_runner as mr
from goldenswing.execution import evaluation as _eval
from test_evaluation import _arm_records
from goldenswing.execution.mt5_runner import (
    management_block_for,
)
from goldenswing.execution.playbook_registry import (
    AUCTION_SPECS,
    FLOOR_AUTHORITY,
    NATIVE_SPECS,
    PLAYBOOK_REGISTRY,
    REVERSION_SPECS,
    ROSTER_SPECS,
    ROUTE_AUCTION_VALUE,
    SESSION_SPECS,
    TREND_SPECS,
    VOL_SPECS,
)
from goldenswing.execution.playbook_registry import order_route_for
from goldenswing.execution.scenario_book import ScenarioBook
from goldenswing.execution.setup_lifecycle import TrackedSetup
from goldenswing.live import auction_value as av
from goldenswing.live import developing_value as dv
from goldenswing.live.volume_profile import volume_profile
from goldenswing.playbooks.auction_value import (
    AuctionLVNTraversalPlaybook,
    AuctionValueAcceptancePlaybook,
    AuctionValueRejectionPlaybook,
)

# ------------------------------------------------------------ fixtures
#
# Deterministic H1 bars from scripted closes + volumes, on a
# single UTC day (2026-01-05 00:00 UTC = one NY trading day
# under day_context.trading_day_key).  The value phase is 11
# balanced bars around 2000 on steady volume: the developing
# profile is measurable from bar 9 and its edges sit just
# outside the oscillation (VAH ~2001.3, VAL ~1998.7 — read
# from the data in the tests, never assumed).  M15 frames
# are interpolated from the H1 closes (4 bars per hour), with
# the decisive segments overridden explicitly.

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
            h, l = max(o, c) + 0.4, min(o, c) - 0.4
        bars.append((o, h, l, c, vols[i]))
        prev = c
    idx = pd.date_range("2026-01-05 00:00", periods=len(bars),
                        freq="1h")
    return pd.DataFrame(bars, columns=["open", "high", "low",
                                       "close", "volume"],
                        index=idx)


def _m15_from_h1(h1, *, extra_bars=0):
    closes, prev = [], float(h1["close"].iloc[0])
    for c in h1["close"]:
        for k in range(4):
            closes.append(prev + (float(c) - prev) * (k + 1) / 4.0)
        prev = float(c)
    for _ in range(extra_bars):
        closes.append(prev)
    bars, p = [], closes[0]
    for c in closes:
        bars.append((p, max(p, c) + 0.3, min(p, c) - 0.3, c, 25.0))
        p = c
    idx = pd.date_range(h1.index[0], periods=len(bars),
                        freq="15min")
    return pd.DataFrame(bars, columns=["open", "high", "low",
                                       "close", "volume"],
                        index=idx)


def _acceptance_frames():
    """Probe on heavy volume at bar 12, second outside close
    at 13 (acceptance), retest on M15 bars 56-58 (the bar at
    index 58 comes back to the probed edge and holds)."""
    h1 = _h1_frame(
        _BASE + [2001, 2009, 2011, 2010, 2012],
        _BASE_V + [100, 300, 300, 150, 150],
        ranges={12: (2010.0, 2001.5), 13: (2012.0, 2008.0)})
    m15 = _m15_from_h1(h1)
    m15.iloc[56] = (2009.5, 2010.0, 2008.0, 2009.0, 25.0)
    m15.iloc[57] = (2009.0, 2009.2, 2006.0, 2007.0, 25.0)
    m15.iloc[58] = (2007.0, 2007.5, 2001.0, 2003.5, 25.0)
    return h1, m15


def _rejection_frames():
    """Probe at bar 12 on ordinary volume, close back inside
    at 13 (rejection); the first M15 close after it (index
    56) sits inside value."""
    h1 = _h1_frame(
        _BASE + [2001, 2009, 2001, 2000],
        _BASE_V + [100, 100, 100, 100],
        ranges={12: (2010.0, 2001.5)})
    m15 = _m15_from_h1(h1)
    m15.iloc[56] = (2000.8, 2001.2, 2000.1, 2000.5, 25.0)
    return h1, m15


def _lvn_frames():
    """Two volume shelves (closes ~1997 and ~2008.5 on heavy
    volume) joined by a thin pass-through: the developing
    profile carries LVNs between the shelves.  The M15 tail
    then descends from the high shelf and crosses into the
    thin region (bar 64)."""
    closes = [1997, 1998.5, 1996.5, 1998, 1997.5, 1996.8,
              2002, 2004.5,
              2008, 2009.5, 2007.5, 2009, 2008.2, 2007.8,
              2008.5, 2009.2]
    vols = [150] * 6 + [15, 15] + [150] * 8
    h1 = _h1_frame(closes, vols)
    m15 = _m15_from_h1(h1, extra_bars=4)
    m15.iloc[64] = (2008.5, 2008.8, 2004.8, 2005.0, 25.0)
    m15.iloc[65] = (2005.0, 2005.2, 2002.8, 2003.0, 25.0)
    return h1, m15


def _situation(sid="sit-auction-1", *, zones_above=(),
               zones_below=()):
    """A situation snapshot in the keeper's schema — the
    family consumes location.value (with the developing
    state the Pass 7c situation wiring carries) and the
    zones; the per_tf block rides along unused by this
    family, as the schema carries it."""
    return {"situation_id": sid,
            "per_tf": {
                "H1": {"bias": 1, "structure_phase": "range"}},
            "location": {
                "value": {
                    "poc": 2000.13, "vah": 2009.75,
                    "val": 1998.2, "developing": True,
                    "developing_state": {
                        "day_key": "2026-01-05", "bars": 14,
                        "volume_basis": "tick_volume"}},
                "zones_above": list(zones_above),
                "zones_below": list(zones_below)},
            "relationships": []}


def _acceptance_pb(h1, situation=None):
    pb = AuctionValueAcceptancePlaybook()
    pb.set_formation(h1, "H1")
    pb.set_situation(situation)
    return pb


# ------------------------------------------- developing value

def test_developing_value_parity_with_completed_profile():
    """The extension IS volume_profile's construction,
    sliced causally: at the day's last bar the developing
    state equals the completed profile of the same bars."""
    h1, _ = _acceptance_frames()
    day = h1.iloc[:14]
    state = dv.developing_state(day, len(day) - 1)
    vp = volume_profile(day)
    assert state["status"] == "ok"
    assert state["poc"] == vp["poc"]
    assert state["vah"] == vp["vah"]
    assert state["val"] == vp["val"]


def test_developing_value_absent_stays_absent():
    h1, _ = _acceptance_frames()
    state = dv.developing_state(h1, 5)  # 6 bars < MIN_BARS
    assert state["status"] == "insufficient bars"
    assert state["poc"] is None and state["vah"] is None
    flat = _h1_frame([2000.0] * 12, [100] * 12,
                     ranges={i: (2000.0, 2000.0)
                             for i in range(12)})
    state = dv.developing_state(flat, 11)
    assert state["status"] == "flat range"
    assert state["poc"] is None


def test_developing_value_names_its_basis():
    h1, _ = _acceptance_frames()
    state = dv.developing_state(h1, 13)
    assert state["volume_basis"] == "tick_volume"
    assert dv.VOLUME_BASIS == "tick_volume"


def test_developing_value_migration_read_from_data():
    """Migration is asserted FROM the computed states:
    across the acceptance fixture's probe bars the probed
    edge (VAH) ends higher than the pre-probe reference,
    and migrated_toward agrees; across the light-volume
    probe (no re-pricing) it does not."""
    h1, _ = _acceptance_frames()
    series = dv.DevelopingSeries(h1)
    ref = series.at(11)
    now = series.at(13)
    assert now["vah"] > ref["vah"]
    assert dv.migrated_toward(now, ref, 1) is True
    light = _h1_frame(_BASE + [2001, 2009, 2010],
                      _BASE_V + [100, 10, 10],
                      ranges={12: (2010.0, 2001.5)})
    series2 = dv.DevelopingSeries(light)
    assert dv.migrated_toward(series2.at(13),
                              series2.at(11), 1) is False


def test_developing_value_nodes_are_flanked_shelves():
    h1, _ = _lvn_frames()
    state = dv.developing_state(h1, len(h1) - 1)
    assert state["status"] == "ok"
    assert len(state["hvns"]) >= 2
    assert state["lvns"], "the thin pass-through must read as LVNs"
    for lvn in state["lvns"]:
        assert lvn["below_hvn"] < lvn["price"] < lvn["above_hvn"]
        assert lvn["low"] < lvn["price"] < lvn["high"]


def test_developing_value_reads_are_causal():
    h1, _ = _acceptance_frames()
    full = dv.DevelopingSeries(h1).at(13)
    trunc = dv.DevelopingSeries(h1.iloc[:14]).at(13)
    for key in ("poc", "vah", "val", "day_key", "bars"):
        assert full[key] == trunc[key]


# ------------------------------------------------ core sequence

def test_probe_and_acceptance_sequence():
    h1, _ = _acceptance_frames()
    prep = av.PreparedAuction(h1)
    seq = av.auction_sequence_at(prep, 12)
    assert seq["phase"] == "probing"
    assert seq["direction"] == 1 and seq["outside_closes"] == 1
    seq = av.auction_sequence_at(prep, 13)
    assert seq["phase"] == "accepted"
    # The acceptance edge is the PRE-PROBE value's edge —
    # read from the data (the state at bar 11), not a
    # hardcoded level.
    ref = dv.DevelopingSeries(h1).at(11)
    assert seq["accept_edge"] == pytest.approx(ref["vah"])
    assert seq["ref"]["poc"] == pytest.approx(ref["poc"])


def test_rejection_sequence():
    h1, _ = _rejection_frames()
    prep = av.PreparedAuction(h1)
    seq = av.auction_sequence_at(prep, 13)
    assert seq["phase"] == "rejected"
    assert seq["direction"] == 1
    assert seq["extreme_price"] == pytest.approx(2010.0)


def test_wick_only_probe_forms_no_sequence():
    """A probe that goes nowhere: the bar's RANGE crosses
    the edge but its CLOSE stays inside value — no probe,
    no hypothesis (acceptance/rejection are close reads,
    never wick reads)."""
    h1 = _h1_frame(_BASE + [2001, 2000.5, 2001],
                   _BASE_V + [100, 100, 100],
                   ranges={12: (2011.0, 2000.0)})
    prep = av.PreparedAuction(h1)
    assert av.auction_sequence_at(prep, 13) is None
    for cls in (AuctionValueAcceptancePlaybook,
                AuctionValueRejectionPlaybook):
        pb = cls()
        pb.set_formation(h1, "H1")
        m15 = _m15_from_h1(h1)
        assert pb.live_signal([], len(m15) - 1, m15) == []


def test_unresolved_probe_expires():
    h1 = _h1_frame(_BASE + [2001, 2009, 2010, 2010.5, 2011],
                   _BASE_V + [100, 10, 10, 10, 10],
                   ranges={12: (2010.0, 2001.5)})
    prep = av.PreparedAuction(h1)
    assert av.auction_sequence_at(prep, 14)["phase"] == "probing"
    seq = av.auction_sequence_at(prep, 15)
    assert seq["phase"] == "expired"


def test_accepted_sequence_invalidates_when_reprcing_fails():
    h1 = _h1_frame(
        _BASE + [2001, 2009, 2011, 2008, 2003, 2001],
        _BASE_V + [100, 300, 300, 80, 80, 80],
        ranges={12: (2010.0, 2001.5), 13: (2012.0, 2008.0)})
    prep = av.PreparedAuction(h1)
    assert av.auction_sequence_at(prep, 13)["phase"] == "accepted"
    seq = av.auction_sequence_at(prep, 16)
    assert seq["phase"] == "invalidated"
    assert seq["death_bar"] == 16


def test_rejected_sequence_invalidates_on_reacceptance():
    h1 = _h1_frame(
        _BASE + [2001, 2009, 2001, 2000, 2011],
        _BASE_V + [100, 100, 100, 100, 100],
        ranges={12: (2010.0, 2001.5)})
    prep = av.PreparedAuction(h1)
    assert av.auction_sequence_at(prep, 13)["phase"] == "rejected"
    seq = av.auction_sequence_at(prep, 15)
    assert seq["phase"] == "invalidated"
    assert seq["death_bar"] == 15


def test_down_probe_accepts_symmetrically():
    h1 = _h1_frame(
        _BASE + [1999, 1991, 1989, 1990],
        _BASE_V + [100, 300, 300, 150],
        ranges={12: (1998.5, 1990.0), 13: (1992.0, 1988.0)})
    prep = av.PreparedAuction(h1)
    seq = av.auction_sequence_at(prep, 13)
    assert seq["phase"] == "accepted" and seq["direction"] == -1
    ref = dv.DevelopingSeries(h1).at(11)
    assert seq["accept_edge"] == pytest.approx(ref["val"])


def test_sequence_reads_are_causal():
    h1, _m15 = _acceptance_frames()
    prep_full = av.PreparedAuction(h1)
    prep_trunc = av.PreparedAuction(h1.iloc[:14])
    a = av.auction_sequence_at(prep_full, 13)
    b = av.auction_sequence_at(prep_trunc, 13)
    assert a["phase"] == b["phase"] == "accepted"
    assert a["accept_edge"] == b["accept_edge"]
    assert a["identity"] == b["identity"]


# ------------------------------------------------------- playbooks

def test_acceptance_signal_shape_and_context():
    h1, m15 = _acceptance_frames()
    pb = _acceptance_pb(h1, _situation())
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.index == 58 and sig.direction == 1
    assert sig.entry_ref == pytest.approx(2003.5)
    ref = dv.DevelopingSeries(h1).at(11)
    atr = av.PreparedAuction(h1).atr(13)
    # Stop beyond the probed edge by the desk's buffer.
    assert sig.invalidation == pytest.approx(
        ref["vah"] - av.STOP_BUFFER_ATR * atr)
    # Every target is a level BEYOND the entry.
    assert all(t > sig.entry_ref for t in sig.targets)
    ctx = sig.live_context
    assert ctx["family"] == "auction_value"
    assert ctx["hypothesis"] == "value_acceptance"
    assert ctx["volume_basis"] == "tick_volume"
    assert ctx["value"]["volume_basis"] == "tick_volume"
    assert ctx["situation_id"] == "sit-auction-1"
    assert ctx["scenario_clocks"] == {
        "formation_tf": "H1", "refinement_tf": None,
        "trigger_tf": "M15", "entry_tf": "M15",
        "management_tf": "H1"}
    assert ctx["probe"]["ref"]["vah"] == pytest.approx(
        ref["vah"])
    assert any("thesis exit" in t
               for t in ctx["invalidation_terms"])


def test_acceptance_targets_next_situation_zone():
    h1, m15 = _acceptance_frames()
    zones = [{"bottom": 2006.0}]
    pb = _acceptance_pb(h1, _situation(zones_above=zones))
    sig = pb.live_signal([], len(m15) - 1, m15)[0]
    assert sig.live_context["target_source"] == "situation_zone"
    assert sig.targets[0] == pytest.approx(2006.0)


def test_acceptance_retest_voided_by_close_back_inside():
    """An M15 close back inside the probed edge before the
    retest hold voids the entry read entirely."""
    h1, m15 = _acceptance_frames()
    m15.iloc[57] = (2009.0, 2009.2, 2000.0, 2000.5, 25.0)
    pb = _acceptance_pb(h1)
    assert pb.live_signal([], len(m15) - 1, m15) == []


def test_acceptance_without_retest_is_no_signal():
    """Acceptance alone never enters: with no M15 bar
    returning to the edge, there is no signal (the entry
    IS the retest hold, not the acceptance close)."""
    h1, m15 = _acceptance_frames()
    m15.iloc[58] = (2007.0, 2007.5, 2005.0, 2006.5, 25.0)
    pb = _acceptance_pb(h1)
    assert pb.live_signal([], len(m15) - 1, m15) == []


def test_rejection_signal_shape_and_context():
    h1, m15 = _rejection_frames()
    pb = AuctionValueRejectionPlaybook()
    pb.set_formation(h1, "H1")
    pb.set_situation(_situation())
    sigs = pb.live_signal([], len(m15) - 1, m15)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.index == 56 and sig.direction == -1
    assert sig.entry_ref == pytest.approx(2000.5)
    state = dv.DevelopingSeries(h1).at(13)
    ref = dv.DevelopingSeries(h1).at(11)
    # Rotation: POC first, then the opposite (far) edge.
    assert sig.targets[0] == pytest.approx(state["poc"])
    assert sig.targets[1] == pytest.approx(ref["val"])
    # Stop beyond the probe's extreme.
    assert sig.invalidation > 2010.0
    ctx = sig.live_context
    assert ctx["hypothesis"] == "value_rejection_rotation"
    # The completion rate rides as a stated hypothesis,
    # never an assumption baked into the geometry.
    assert "never assumed" in ctx["rotation"]["completion_rate"]


def test_rejection_and_acceptance_never_double_fire():
    """One sequence machine, two resolutions: on the
    acceptance fixture the rejection playbook is silent,
    and vice versa."""
    h1, m15 = _acceptance_frames()
    rej = AuctionValueRejectionPlaybook()
    rej.set_formation(h1, "H1")
    assert rej.live_signal([], len(m15) - 1, m15) == []
    h1r, m15r = _rejection_frames()
    acc = _acceptance_pb(h1r)
    assert acc.live_signal([], len(m15r) - 1, m15r) == []


def test_lvn_traversal_signal():
    h1, m15 = _lvn_frames()
    pb = AuctionLVNTraversalPlaybook()
    pb.set_formation(h1, "H1")
    sigs = pb.live_signal([], len(m15) - 1, m15)
    # Exactly one entry for the whole descent through the
    # thin region (adjacent LVN bins are one region).
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.index == 64 and sig.direction == -1
    assert sig.entry_ref == pytest.approx(2005.0)
    state = dv.DevelopingSeries(h1).at(len(h1) - 1)
    lvn = sig.live_context["lvn"]
    # Target the far shelf; stop beyond the near shelf.
    assert sig.targets == [pytest.approx(lvn["far_hvn"])]
    assert lvn["far_hvn"] == pytest.approx(
        min(state["hvns"]))
    assert sig.invalidation > lvn["near_hvn"]
    assert sig.live_context["hypothesis"] == "lvn_traversal"


def test_lvn_jump_over_span_is_never_chased():
    """A close that jumps the whole thin region and lands
    at the far shelf never entered a span: no traversal."""
    h1, m15 = _lvn_frames()
    m15.iloc[64] = (2008.5, 2008.8, 1995.5, 1996.5, 25.0)
    pb = AuctionLVNTraversalPlaybook()
    pb.set_formation(h1, "H1")
    frame = m15.iloc[:65]
    assert pb.live_signal([], len(frame) - 1, frame) == []


def test_unthreaded_playbook_produces_nothing():
    _h1, m15 = _acceptance_frames()
    pb = AuctionValueAcceptancePlaybook()
    assert pb.live_signal([], len(m15) - 1, m15) == []
    ctx = SimpleNamespace(df=m15)
    assert pb.candidates(ctx) == []


def test_arm_read_reports_probing_then_accepted():
    h1, m15 = _acceptance_frames()
    pb = _acceptance_pb(h1, _situation())
    read = pb.pending_state(m15.iloc[:52])   # through the probe close
    assert read["phase"] == "probing"
    assert read["direction"] == 1
    assert read["corroborated"] is True
    assert read["value"]["volume_basis"] == "tick_volume"
    assert read["identity"]
    read = pb.pending_state(m15.iloc[:59])   # through the retest bar
    assert read["phase"] == "accepted"


# --------------------------------------------------- registry + routes

def test_registry_partition_and_posture():
    assert [s.name for s in AUCTION_SPECS] == [
        "auction_value_acceptance", "auction_value_rejection",
        "auction_lvn_traversal"]
    for spec in AUCTION_SPECS:
        assert spec.route == ROUTE_AUCTION_VALUE
        assert spec.trigger_kind == "m15_trigger_fire"
        assert spec.entry_thesis == "market"
        assert spec.management_horizon == "intraday"
        assert spec.chart_studies == ("volume_profile",)
        # Not in the roster tuple: the family partition is
        # separate (though on the trigger path since
        # origination, owner order 2026-11-25).
        assert spec not in ROSTER_SPECS
        assert spec.playbook_class.live is True
    assert {s.name for s in ROSTER_SPECS} | {
        s.name for s in NATIVE_SPECS} | {
        s.name for s in VOL_SPECS} | {
        s.name for s in TREND_SPECS} | {
        s.name for s in AUCTION_SPECS} | {
        s.name for s in REVERSION_SPECS} | {
        s.name for s in SESSION_SPECS} == {
        s.name for s in PLAYBOOK_REGISTRY}
    assert "never the H1 roster's floor" in \
        FLOOR_AUTHORITY[ROUTE_AUCTION_VALUE]


def test_routes_declared_governing(tmp_path):
    """Every route governs (owner order 2026-11-25/27 — the
    promotion gate is deleted, no route is shadowed or
    demoted): the family's playbooks ride their own route,
    live, and the manifest's per-route table says so."""
    runner, _ = _scan_runner(tmp_path)
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    for name in ("auction_value_acceptance",
                 "auction_value_rejection",
                 "auction_lvn_traversal"):
        route = order_route_for(name)
        assert route == ROUTE_AUCTION_VALUE
        assert table[route]["authority"] == "live"
        assert table[route]["source"] == "live"
        assert table[route]["governing_now"] is True
    # The runner's tables agree with the registry: the
    # family's playbooks are the registry's own specs.
    assert {s.name for s in AUCTION_SPECS} == \
        set(mr.AUCTION_JUDGMENT_ROUTE_NAMES)


def test_family_attributed_to_own_route():
    """The registry maps each family playbook to its OWN
    family's route — never the H1 roster's — so a violation
    attributed to the playbook lands on that route, and the
    route governs by declaration (owner order
    2026-11-25/27)."""
    for name in ("auction_value_acceptance",
                 "auction_value_rejection",
                 "auction_lvn_traversal"):
        assert order_route_for(name) == ROUTE_AUCTION_VALUE


def test_management_block_intraday_standard_on_formation_clock():
    h1, m15 = _acceptance_frames()
    sig = _acceptance_pb(h1).live_signal([], len(m15) - 1, m15)[0]
    block = management_block_for("auction_value_acceptance", sig)
    assert block["management_horizon"] == "intraday"
    assert block["management_policy_id"] == "standard_v1"
    # Managed on H1, the clock the thesis formed on (not
    # the M15 entry clock).
    assert block["management_context_timeframe"] == "H1"
    assert block["management_invalidation"] == sig.invalidation
    assert block["management_target"] == sig.targets[0]
    from goldenswing.execution.playbook_registry import policy_params_for
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


def _book_setup(sig, *, playbook="auction_value_acceptance",
                setup_id="auction:1"):
    return TrackedSetup(
        setup_id=setup_id, parent_setup_id=setup_id,
        playbook=playbook, route=ROUTE_AUCTION_VALUE,
        direction=int(sig.direction),
        entry_ref=sig.entry_ref, invalidation=sig.invalidation,
        targets=list(sig.targets),
        management=management_block_for(playbook, sig),
        formation_ts="2026-01-05T13:00:00",
        touch_ts="2026-01-05T14:30:00",
        created_ts="2026-01-05T14:30:00",
        trigger_tf=sig.live_context["trigger_tf"],
        validity_bars=6, expiry_ts="2026-01-05T20:00:00",
        live_context={"scenario_clocks":
                          dict(sig.live_context["scenario_clocks"]),
                      "family": "auction_value",
                      "invalidation_terms":
                          sig.live_context["invalidation_terms"]})


def test_book_takes_instance_clocks_and_freezes_plan():
    h1, m15 = _acceptance_frames()
    sig = _acceptance_pb(h1).live_signal([], len(m15) - 1, m15)[0]
    setup = _book_setup(sig)
    book = ScenarioBook()
    book.propose_from_setup(setup, ts="2026-01-05T14:30:00")
    sc = book.scenarios["auction:1"]
    assert sc["clocks"] == sig.live_context["scenario_clocks"]
    assert sc["horizon"] == "intraday"
    from goldenswing.execution.setup_lifecycle import STAGE_FILLED
    book.sync_setup(setup, ts="2026-01-05T15:00:00",
                    stage=STAGE_FILLED)
    frozen = book.scenarios["auction:1"]["management_context"]
    assert frozen["management_plan"]["policy_id"] == \
        "standard_v1"
    assert frozen["management_plan"]["management_tf"] == "H1"
    terms = frozen["management_plan"]["invalidation"]["terms"]
    assert any("close beyond invalidation" in t for t in terms)
    assert any("parent thesis invalidated" in t for t in terms)


# --------------------------------------------------------- runner wiring

def _family_runner(tmp_path, h1_frame, m15_frame):
    runner, _ = _scan_runner(tmp_path)
    runner._stack = SimpleNamespace(frames={
        "H1": h1_frame, "M15": m15_frame,
        "H4": _h1_frame([2000.0] * 40, [100] * 40),
        "D1": _h1_frame([2000.0] * 40, [100] * 40)})
    runner._situation_keeper = SimpleNamespace(
        latest=lambda: _situation(),
        latest_id=lambda: "sit-auction-1")
    return runner


def test_runner_scan_journals_evaluations_verdicts_and_book(
        tmp_path):
    # The frames end AT the retest bar: a scheduled sweep
    # honors signals on the closing bar (the fresh-instance
    # design), never stale history.
    h1, m15 = _acceptance_frames()
    runner = _family_runner(tmp_path, h1, m15.iloc[:59])
    runner._run_auction_value_family(("H1",), {},
                                     source="live_close")
    records = runner.journal.records()
    evals = [r for r in records if r["type"] == "evaluation"
             and r.get("kind") == "playbook_eval"
             and r["playbook"] == "auction_value_acceptance"]
    assert evals and all(
        r["evaluation_clock"] == "auction_value_h1"
        for r in evals)
    assert all(r["situation_id"] == "sit-auction-1"
               for r in evals)
    emitted = [r for r in evals
               if r["outcome"] == "candidate_emitted"]
    assert len(emitted) == 1
    # The family's context evidence rides on the record:
    # family identity, the D1 read as CONTEXT only, and
    # the volume basis NAMED (tick volume, never dressed).
    assert emitted[0]["evidence"]["family"] == "auction_value"
    assert "never" in emitted[0]["evidence"]["d1_role"]
    assert emitted[0]["evidence"]["volume_basis"] == \
        "tick_volume"
    assert "tick-volume" in \
        emitted[0]["evidence"]["volume_basis_role"]
    # The route stamp: the scan's passive verdict is journaled
    # as a judgment record identifiable as THIS route's
    # history.
    judged = [r for r in records if r["type"] == "judgment"
               and (r.get("judgment") or {}).get(
                   "authority_route") == "auction_value_acceptance"]
    assert len(judged) == 1
    assert judged[0]["judgment"]["authority"] == "live"
    # The scenario entered the book as a governing proposal
    # (origination) with the family's clocks.
    setups = runner._setup_store.active()
    assert len(setups) == 1
    book = runner._scenario_book_get()
    sc = book.scenarios[setups[0].setup_id]
    assert sc["family"] == "auction_value_acceptance"
    assert sc["authority"] == "live"
    assert sc["horizon"] == "intraday"
    assert sc["clocks"]["formation_tf"] == "H1"
    assert sc["clocks"]["management_tf"] == "H1"
    assert sc["clocks"]["entry_tf"] == "M15"
    # No order of any kind was placed.
    assert not any("order" in str(r.get("type")) for r in records)


def test_runner_journals_arm_transitions(tmp_path):
    h1, m15 = _acceptance_frames()
    runner = _family_runner(tmp_path, h1, m15.iloc[:52])
    runner._run_auction_value_family(("H1",), {},
                                     source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "auction_value_arm"]
    assert len(arms) == 1
    arm = arms[0]
    assert arm["family"] == "auction_value"
    assert arm["phase"] == "probing" and arm["direction"] == 1
    assert arm["formation_tf"] == "H1"
    assert arm["trigger_tf"] == "M15"
    assert arm["situation_id"] == "sit-auction-1"
    # The arm record names the basis: tick volume.
    assert arm["volume_basis"] == "tick_volume"
    assert arm["value"]["volume_basis"] == "tick_volume"
    # Re-running the same scan journals no duplicate arm.
    runner._run_auction_value_family(("H1",), {},
                                     source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "auction_value_arm"]
    assert len(arms) == 1


def test_runner_journals_accepted_and_invalidated_arms(tmp_path):
    h1, m15 = _acceptance_frames()
    runner = _family_runner(tmp_path, h1, m15.iloc[:59])
    runner._run_auction_value_family(("H1",), {},
                                     source="live_close")
    arms = [r for r in runner.journal.records()
            if r["type"] == "auction_value_arm"]
    assert [a["phase"] for a in arms] == ["accepted"]
    # A dead sequence journals its death as an arm record
    # in its own right (Pass 9 grades families on their
    # deaths as well as their entries).
    h1d = _h1_frame(
        _BASE + [2001, 2009, 2011, 2008, 2003, 2001],
        _BASE_V + [100, 300, 300, 80, 80, 80],
        ranges={12: (2010.0, 2001.5), 13: (2012.0, 2008.0)})
    m15d = _m15_from_h1(h1d)
    runner2 = _family_runner(tmp_path / "dead", h1d, m15d)
    runner2._run_auction_value_family(("H1",), {},
                                      source="live_close")
    deaths = [r for r in runner2.journal.records()
              if r["type"] == "auction_value_arm"
              and r["phase"] == "invalidated"]
    assert len(deaths) == 1
    assert deaths[0]["death_ts"] is not None


def test_runner_family_clocks_cover_all_playbooks(tmp_path):
    """Every family playbook is evaluated and accountable
    on the family's clock."""
    h1, m15 = _acceptance_frames()
    runner = _family_runner(tmp_path, h1, m15)
    runner._run_auction_value_family(("H1",), {},
                                     source="live_close")
    clocks = {(r["playbook"], r["evaluation_clock"])
              for r in runner.journal.records()
              if r["type"] == "evaluation"}
    for name in ("auction_value_acceptance",
                 "auction_value_rejection",
                 "auction_lvn_traversal"):
        assert (name, "auction_value_h1") in clocks


def test_situation_value_block_carries_developing_state(tmp_path):
    """The §6.3 extension on the situation itself: the
    runner's location.value carries developing_state,
    built by the same module the family reads — with the
    basis named."""
    runner, _ = _scan_runner(tmp_path)
    h1, _ = _acceptance_frames()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    blocks = runner._situation_blocks(
        {"mid": 2003.5, "spread": 0.3}, h1.index[-1], 2003.5)
    value = blocks["location"]["value"]
    assert value["developing"] is True
    dev = value["developing_state"]
    assert dev["volume_basis"] == "tick_volume"
    assert dev["day_key"] == "2026-01-05"
    state = dv.developing_state(h1, len(h1) - 1)
    assert dev["poc"] == pytest.approx(state["poc"])
    assert dev["migration"] is not None


# ------------------------------------------------- evaluation (S4 pairing)

# The measurement the deleted auction_promotion module shared,
# now in execution/evaluation.py: route-stamped judgment
# verdicts + their +4h reconciliations, paired per arm.
# The verdict is a measurement only — it grants, promotes,
# demotes and changes nothing.  The host cycle that journals
# it lives in book_evaluation.py (test_book_evaluation.py);
# there is no per-family phase anymore.

def _routes_map():
    return {s.name: s.name for s in AUCTION_SPECS}


def _playbooks():
    return tuple(s.name for s in AUCTION_SPECS)


def test_evaluation_insufficient_then_ready(tmp_path):
    route = "auction_value_acceptance"
    state = _eval.new_state()
    few = _arm_records(route, [1.0] * 6, [0.0] * 6)
    snap = _eval.accumulate(
        state, few, now=None, routes_map=_routes_map(),
        playbooks=_playbooks(), family="auction")
    row = snap["routes"][route]
    assert row["verdict"] == _eval.VERDICT_INSUFFICIENT
    many = _arm_records(route, [1.2] * 22, [-0.3] * 22,
                        start=5000)
    snap2 = _eval.accumulate(
        state, many, now=None, routes_map=_routes_map(),
        playbooks=_playbooks(), family="auction")
    row = snap2["routes"][route]
    assert row["verdict"] == _eval.VERDICT_READY
    assert row["take"]["median_r"] == pytest.approx(1.2)
    assert "a measurement, never an action" in row["verdict_reason"]
    # The verdict changed nothing about the route's state:
    # still live, still governing.
    runner, _ = _scan_runner(tmp_path)
    table = {r["route"]: r for r in runner._judgment_route_table()}
    assert table[order_route_for(route)]["governing_now"] is True


def test_evaluation_one_shot_ignores_other_routes():
    records = _arm_records("auction_lvn_traversal",
                           [1.0] * 22, [0.0] * 22)
    snap = _eval.evaluate_records(
        records, routes_map=_routes_map(),
        playbooks=_playbooks())
    assert snap["routes"]["auction_lvn_traversal"]["verdict"] == \
        _eval.VERDICT_READY
    assert snap["routes"]["auction_value_acceptance"]["verdict"] == \
        _eval.VERDICT_INSUFFICIENT
    # Another family's history never leaks in either.
    snap2 = _eval.evaluate_records(
        _arm_records("range_qm", [1.0] * 22, [0.0] * 22),
        routes_map=_routes_map(), playbooks=_playbooks())
    assert all(row["verdict"] ==
               _eval.VERDICT_INSUFFICIENT
               for row in snap2["routes"].values())
