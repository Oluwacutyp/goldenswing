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
  route-stamped judgment_shadow verdicts, arm records
  (corroborated transitions AND invalidations, basis
  named), and book scenarios — and place no orders;
- the situation wiring: location.value carries the
  developing_state built by the same extension;
- the promotion evaluation (execution/auction_promotion.py):
  the §G evaluator over the family's own route-stamped
  history; promotion_ready is a recommendation, binds
  nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from test_step14_setup_lifecycle import _scan_runner

import goldenswing.execution.mt5_runner as mr
from goldenswing.execution import auction_promotion, range_promotion
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
    ROUTE_AUCTION_VALUE,
    SESSION_SPECS,
    TREND_SPECS,
    VOL_SPECS,
)
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
        assert spec.shadow_evaluation is True
        assert spec.entry_thesis == "market"
        assert spec.management_horizon == "intraday"
        assert spec.chart_studies == ("volume_profile",)
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
        ROUTE_AUCTION_VALUE]


def test_routes_declared_shadow_and_never_governing(tmp_path):
    runner, _ = _scan_runner(tmp_path)
    table = {row["route"]: row
             for row in runner._judgment_route_table()}
    for name in ("auction_value_acceptance",
                 "auction_value_rejection",
                 "auction_lvn_traversal"):
        assert runner._judgment_route_state(name) == \
            "shadow_declared"
        assert runner._judgment_route_source(name) == "declared"
        assert table[name]["governing_now"] is False
        assert JUDGMENT_ROUTE_PLAYBOOK[name] == name
    # The runner's tables agree with the family's own map.
    assert set(auction_promotion.AUCTION_JUDGMENT_ROUTES) == \
        set(mr.AUCTION_JUDGMENT_ROUTE_NAMES)


def test_family_has_no_order_path():
    """No seam maps an auction playbook to an order route:
    the route_promotion order table (built from the
    trigger-path roster) names none of them, so no
    candidate of this family can reach an order through
    any existing route."""
    from goldenswing.execution.route_promotion import ORDER_ROUTE_BY_PLAYBOOK
    for name in ("auction_value_acceptance",
                 "auction_value_rejection",
                 "auction_lvn_traversal"):
        assert name not in ORDER_ROUTE_BY_PLAYBOOK


def test_undeclared_names_still_fail_closed(tmp_path):
    """Pass 5's invariant, family-adjacent: a route name no
    authority source declares resolves shadow/undeclared
    and is REFUSED at the judgment seam in every mode, with
    the journaled alarm."""
    from test_route_authority import _candidate, _decision, _evidence, _jc_runner
    runner, _ = _scan_runner(tmp_path)
    assert runner._judgment_route_state(
        "auction_value_typo") == "shadow_declared"
    assert runner._judgment_route_source(
        "auction_value_typo") == "undeclared"
    for mode in ("active", "shadow"):
        jc = _jc_runner(tmp_path / mode, judgment_mode=mode)
        _bundle, refusal = jc._judgment_authority(
            candidate=_candidate(), decision=_decision(),
            evidence=_evidence(), intended_tf="H1",
            eval_now=jc._evaluation_now(),
            route="auction_value_typo")
        assert refusal is not None
        assert "undeclared" in refusal
        alarms = [r for r in jc.journal.records()
                  if r["type"] == "undeclared_route"]
        assert len(alarms) == 1
        assert alarms[0]["route"] == "auction_value_typo"


# ------------------------------------------------------- management (P6)

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
    # The §G stamp: the scan's passive verdict is journaled
    # as a judgment_shadow identifiable as THIS route's
    # history.
    shadows = [r for r in records if r["type"] == "judgment_shadow"
               and (r.get("judgment") or {}).get(
                   "authority_route") == "auction_value_acceptance"]
    assert len(shadows) == 1
    assert shadows[0]["judgment"]["authority"] == \
        "shadow_declared"
    # The scenario entered the book as a shadow proposal
    # with the family's clocks.
    setups = runner._setup_store.active()
    assert len(setups) == 1
    book = runner._scenario_book_get()
    sc = book.scenarios[setups[0].setup_id]
    assert sc["family"] == "auction_value_acceptance"
    assert sc["authority"] == "shadow"
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
    route = "auction_value_acceptance"
    state = auction_promotion.new_state()
    few = _arm_records(route, [1.0] * 6, [0.0] * 6)
    rec = auction_promotion.cycle(
        state=state, records=few, now=None,
        bar="2026-11-03 00:00")
    assert rec is not None and \
        rec["type"] == "auction_promotion_evaluation"
    row = rec["routes"][route]
    assert row["verdict"] == range_promotion.VERDICT_INSUFFICIENT
    many = _arm_records(route, [1.2] * 22, [-0.3] * 22,
                        start=5000)
    rec2 = auction_promotion.cycle(
        state=state, records=many, now=None,
        bar="2026-11-04 00:00")
    assert rec2 is not None
    row = rec2["routes"][route]
    assert row["verdict"] == range_promotion.VERDICT_READY
    assert row["take"]["median_r"] == pytest.approx(1.2)
    # binds=False made visible: the verdict speaks as a
    # RECOMMENDATION, never as an effect.
    assert "RECOMMENDATION" in row["verdict_reason"]
    text = "\n".join(auction_promotion.render_lines(state))
    assert "PROMOTION READY (recommendation only)" in text
    # The verdict changed nothing about the route's state:
    # still declared shadow, still governing nowhere.
    runner, _ = _scan_runner(tmp_path)
    assert runner._judgment_route_state(route) == \
        "shadow_declared"


def test_promotion_host_phase_journals_and_persists(tmp_path):
    """The runner host (the §G arrangement): one self-audit
    window in -> a dated auction_promotion_evaluation
    record in the journal, state persisted under its own
    key in the audit state file, and /status renders the
    block."""
    runner, _ = _scan_runner(tmp_path)
    records = _arm_records("auction_value_rejection",
                           [1.0] * 3, [0.0] * 3)
    runner._auction_promotion_phase(
        records, "2026-11-03 00:00",
        window={"records": len(records)})
    out = [r for r in runner.journal.records()
           if r["type"] == "auction_promotion_evaluation"]
    assert len(out) == 1
    assert out[0]["routes"]["auction_value_rejection"][
        "verdict"] == range_promotion.VERDICT_INSUFFICIENT
    assert "auction_promotion" in runner._audit_state()
    assert "auction promotion (Pass 7c)" in \
        runner.auction_promotion_line()


def test_promotion_evaluation_one_shot_ignores_other_routes():
    records = _arm_records("auction_lvn_traversal",
                           [1.0] * 22, [0.0] * 22)
    snap = auction_promotion.evaluate_records(records)
    assert snap["routes"]["auction_lvn_traversal"][
        "verdict"] == range_promotion.VERDICT_READY
    assert snap["routes"]["auction_value_acceptance"][
        "verdict"] == range_promotion.VERDICT_INSUFFICIENT
    # The other families' histories never leak in either.
    for other in ("range_qm", "vol_expansion_escape",
                  "trend_impulse_pullback"):
        snap2 = auction_promotion.evaluate_records(
            _arm_records(other, [1.0] * 22, [0.0] * 22))
        assert all(row["verdict"] ==
                   range_promotion.VERDICT_INSUFFICIENT
                   for row in snap2["routes"].values())
