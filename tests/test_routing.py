"""Phase-5 routing: phase grouping, playbook classification, matrix
building, muting derivation, and the detector's routing cap.

The muting rule is the contract: a (phase, playbook) cell is muted
ONLY when the measured shadow book says it loses (avg R < 0) on at
least MUTE_MIN_N trades.  Nothing else may be muted.
"""
import pytest

from goldenswing.backtest.harness import Outcome
from goldenswing.features.state import FeatureSet, PhaseState
from goldenswing.golden.detector import (DetectorConfig,
                                         GoldenSwingDetector, playbook_of)
from goldenswing.modes.routing import (MUTE_MIN_N, CellStats, RouteTable,
                                       build_matrix, derive_route_table,
                                       phase_group)
from goldenswing.reader.state import (LiquidityPool, MarketState,
                                      OrderBlock, StructureEvent)


# ---------------------------------------------------------------- helpers
def _state(phase="trend_up", event=None, event_age=None, **kw) -> MarketState:
    base = dict(
        index=100, timestamp="t", close=2000.0, atr=10.0,
        trend="up", bias=1,
        pools_above=[LiquidityPool(price=2060.0, side="bsl",
                                   strength=1, created_at=50)],
        active_obs=[OrderBlock(top=2005.0, bottom=1995.0, direction=1,
                               created_at=90, candle_index=89)],
        bull_disp_3=1.3, momentum=1.0, vol_regime="normal",
        eq_position=0.3, zone="discount",
        session="london", killzone=True,
        daily_trend="up", daily_bias=1,
        last_structure_event=event, structure_event_age=event_age,
        features=FeatureSet(index=100, phase=PhaseState(phase=phase)),
    )
    base.update(kw)
    return MarketState(**base)


def _outcome(state: MarketState, kind: str, r: float) -> Outcome:
    from goldenswing.golden.detector import GoldenSignal
    sig = GoldenSignal(index=state.index, timestamp="t", direction=1,
                       score=70.0, kind=kind)
    return Outcome(signal=sig, entry_index=state.index + 1,
                   exit_index=state.index + 5, exit_reason="target",
                   r_net=r, hold_bars=5)


# ---------------------------------------------------------------- grouping
@pytest.mark.parametrize("raw,group", [
    ("accumulation", "accumulation"), ("distribution", "distribution"),
    ("consolidation", "consolidation"), ("manipulation", "manipulation"),
    ("expansion_up", "expansion"), ("expansion_down", "expansion"),
    ("trend_up", "trend"), ("trend_down", "trend"),
    ("unknown", "unknown"), (None, "unknown"), ("nonsense", "unknown"),
])
def test_phase_group(raw, group):
    assert phase_group(raw) == group


# ------------------------------------------------------------- playbooks
def test_playbook_sweep_is_its_own():
    assert playbook_of(_state(), "sweep_reversal") == "sweep_reversal"


def test_playbook_fresh_bos_continuation_is_breakout():
    ev = StructureEvent(index=98, kind="BOS", direction=1, level=1990.0)
    s = _state(event=ev, event_age=2)
    assert playbook_of(s, "continuation") == "breakout"


def test_playbook_stale_bos_is_trend_continuation():
    ev = StructureEvent(index=90, kind="BOS", direction=1, level=1990.0)
    s = _state(event=ev, event_age=9)
    assert playbook_of(s, "continuation") == "trend_continuation"


def test_playbook_choch_is_not_a_breakout():
    ev = StructureEvent(index=98, kind="CHOCH", direction=1, level=1990.0)
    s = _state(event=ev, event_age=1)
    assert playbook_of(s, "continuation") == "trend_continuation"


def test_playbook_no_event_is_trend_continuation():
    assert playbook_of(_state(), "continuation") == "trend_continuation"


# ---------------------------------------------------------------- matrix
def test_build_matrix_aggregates_by_phase_and_playbook():
    s_trend = _state(phase="trend_up")
    s_acc = _state(phase="accumulation")
    states = {100: s_trend, 101: s_acc}
    outs = [
        _outcome(s_trend, "continuation", 1.0),
        _outcome(s_trend, "continuation", -0.5),
        _outcome(s_trend, "continuation", 0.5),
        _outcome(s_acc, "sweep_reversal", -1.0),
    ]
    # states are indexed by position in build_matrix — pass a list
    state_list = [None] * 102
    state_list[100] = s_trend
    state_list[101] = s_acc
    outs[3].signal.index = 101
    cells = build_matrix(outs, state_list)
    c = cells[("trend", "trend_continuation")]
    assert c.n == 3
    assert c.win_rate == pytest.approx(200.0 / 3.0)
    assert c.avg_r == pytest.approx(1.0 / 3.0)
    sw = cells[("accumulation", "sweep_reversal")]
    assert sw.n == 1 and sw.avg_r == pytest.approx(-1.0)


# ---------------------------------------------------------------- muting
def test_muting_rule_exact():
    cells = {
        ("trend", "trend_continuation"): CellStats(
            n=MUTE_MIN_N, win_rate=40.0, avg_r=-0.10, total_r=-3.0),
        ("trend", "breakout"): CellStats(       # loser but too small
            n=MUTE_MIN_N - 1, win_rate=40.0, avg_r=-0.50, total_r=-14.5),
        ("expansion", "sweep_reversal"): CellStats(  # winner, big n
            n=120, win_rate=55.0, avg_r=0.20, total_r=24.0),
        ("consolidation", "breakout"): CellStats(    # exactly break-even
            n=80, win_rate=50.0, avg_r=0.0, total_r=0.0),
    }
    table = derive_route_table(cells)
    assert table.muted == frozenset({("trend", "trend_continuation")})
    assert table.cap == 59.0


def test_route_table_cap_uses_grouped_phase():
    table = RouteTable(muted=frozenset({("expansion", "breakout")}))
    assert table.cap_for("expansion_down", "breakout") == 59.0
    assert table.cap_for("expansion_up", "breakout") == 59.0
    assert table.cap_for("expansion_up", "trend_continuation") is None
    assert table.cap_for("trend_up", "breakout") is None
    assert table.cap_for(None, "breakout") is None


# ------------------------------------------------------- detector wiring
P5 = dict(require_sweep_confirmation=True, daily_veto=True,
          feature_families=frozenset({"phases"}))


def test_detector_routing_cap_mutes_and_marks():
    s = _state(phase="trend_up")
    base = GoldenSwingDetector(DetectorConfig(**P5)).evaluate(s)
    assert base is not None and base.score > 59.0
    assert base.kind == "continuation"

    table = RouteTable(muted=frozenset({("trend", "trend_continuation")}))
    routed = GoldenSwingDetector(
        DetectorConfig(route_table=table, **P5)).evaluate(s)
    assert routed is not None
    assert routed.score == 59.0
    assert any("phase routing" in f for f in routed.factors)


def test_detector_routing_leaves_other_cells_alone():
    s = _state(phase="trend_up")
    base = GoldenSwingDetector(DetectorConfig(**P5)).evaluate(s)
    table = RouteTable(muted=frozenset({("trend", "breakout"),
                                        ("accumulation", "sweep_reversal")}))
    routed = GoldenSwingDetector(
        DetectorConfig(route_table=table, **P5)).evaluate(s)
    assert routed.score == base.score


def test_detector_no_table_reproduces_unrouted():
    s = _state(phase="trend_up")
    a = GoldenSwingDetector(DetectorConfig(**P5)).evaluate(s)
    b = GoldenSwingDetector(
        DetectorConfig(route_table=None, **P5)).evaluate(s)
    assert a.score == b.score and a.factors == b.factors
