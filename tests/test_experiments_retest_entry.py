"""JeaFx retest-entry sleeve: setup, fill, target, causality."""
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import make_df

from goldenswing.experiments.retest_entry import (
    BOS_FRESH_BARS,
    ORDER_EXPIRY_BARS,
    RetestFill,
    RetestSetup,
    find_fill,
    scan_setups,
    select_pool_target,
    simulate_trade,
)
from goldenswing.features.state import SDZone, ZonesState
from goldenswing.reader.state import (LiquidityPool, MarketState,
                                      StructureEvent, SweepEvent)


# ------------------------------------------------------------------ helpers
def _zone(direction=1, top=110.0, bottom=100.0, created_at=90, touches=0):
    return SDZone(top=top, bottom=bottom, direction=direction,
                  created_at=created_at, base_bars=3, departure=1.5,
                  touches=touches)


def _zones_state(direction=1, top=110.0, bottom=100.0, created_at=90,
                 touches=0):
    z = _zone(direction, top, bottom, created_at, touches)
    if direction == 1:
        return ZonesState(nearest_demand=z)
    return ZonesState(nearest_supply=z)


def _state(t, bos_age=2, direction=1, bos_level=118.0, sweep_age=5,
           sweep_index=None, zones=None, atr=10.0, close=116.0,
           daily_bias=None, pools=(), kind="BOS"):
    ev = None if kind is None else StructureEvent(
        index=t - bos_age, kind=kind, direction=direction, level=bos_level)
    sw = None if sweep_index is None else SweepEvent(
        index=sweep_index, level=120.0, extreme=121.0, direction=-direction)
    return MarketState(
        index=t, timestamp=None, close=close, atr=atr, trend="up", bias=1,
        daily_bias=daily_bias if daily_bias is not None else direction,
        last_structure_event=ev,
        structure_event_age=bos_age if kind else None,
        last_sweep=sw, sweep_age=sweep_age if sw else None,
        pools_above=list(pools) if direction == 1 else [],
        pools_below=list(pools) if direction == -1 else [],
        features=SimpleNamespace(zones=zones) if zones is not None else None)


def _flat_states(n, t_setup=100, **kw):
    """n states, all blank except a configured setup at bar t_setup."""
    states = [_state(i, kind=None, zones=_zones_state()) for i in range(n)]
    states[t_setup] = _state(t_setup, **kw)
    return states


def _quiet_df(n, price=116.0, start="2024-01-01"):
    rows = [(price, price + 1, price - 1, price)] * n
    return make_df(rows, start=start)


def _setup_state(t=100, **kw):
    """A passing long setup: zone 100..110 (mid 105), BoS @98 breaking
    118, sweep @95, daily bias +1, atr 10, close 116."""
    return _state(t, zones=_zones_state(created_at=90), sweep_index=t - 5,
                  **kw)


# -------------------------------------------------------------------- scan
def test_setup_found_when_full_sequence_present():
    df = _quiet_df(120)
    states = _flat_states(120, t_setup=100, zones=_zones_state(created_at=90),
                          sweep_index=95)
    setups = scan_setups(df, states)
    assert len(setups) == 1
    s = setups[0]
    assert s.direction == 1 and s.mid == pytest.approx(105.0)
    # stop = zone bottom (100) - 0.25 * ATR(10)
    assert s.stop == pytest.approx(97.5)
    assert s.risk == pytest.approx(7.5)
    assert s.sweep_index == 95 and s.bos_index == 98


def test_setup_rejected_when_bos_stale():
    df = _quiet_df(150)
    states = _flat_states(150, t_setup=140, bos_age=BOS_FRESH_BARS + 1,
                          zones=_zones_state(created_at=130),
                          sweep_index=130)
    assert scan_setups(df, states) == []


def test_setup_rejected_when_no_sweep():
    df = _quiet_df(120)
    states = _flat_states(120, t_setup=100,
                          zones=_zones_state(created_at=90),
                          sweep_index=None)
    assert scan_setups(df, states) == []


def test_setup_rejected_when_sweep_after_bos():
    df = _quiet_df(120)
    # sweep @99 is AFTER the BoS @98: the failed raid must come first
    states = _flat_states(120, t_setup=100, bos_age=2,
                          zones=_zones_state(created_at=90),
                          sweep_index=99)
    assert scan_setups(df, states) == []


def test_setup_rejected_when_zone_tested():
    df = _quiet_df(120)
    states = _flat_states(120, t_setup=100,
                          zones=_zones_state(created_at=90, touches=1),
                          sweep_index=95)
    assert scan_setups(df, states) == []


def test_setup_rejected_when_zone_created_after_bos():
    # a zone drawn after the break is not the level it departed from
    df = _quiet_df(120)
    states = _flat_states(120, t_setup=100, bos_age=2,
                          zones=_zones_state(created_at=99),
                          sweep_index=95)
    assert scan_setups(df, states) == []


def test_setup_rejected_when_bos_level_not_beyond_zone():
    # BoS broke 105, which is INSIDE the zone (100..110): not a push
    # away from demand.
    df = _quiet_df(120)
    states = _flat_states(120, t_setup=100, bos_level=105.0,
                          zones=_zones_state(created_at=90),
                          sweep_index=95)
    assert scan_setups(df, states) == []


def test_setup_rejected_when_daily_bias_opposes():
    df = _quiet_df(120)
    states = _flat_states(120, t_setup=100,
                          zones=_zones_state(created_at=90),
                          sweep_index=95, daily_bias=-1)
    assert scan_setups(df, states) == []


def test_setup_rejected_when_stop_too_wide():
    df = _quiet_df(120)
    # zone 100..110, mid 105, stop 97.5 -> risk 7.5; atr 1 -> 7.5 ATR
    states = _flat_states(120, t_setup=100,
                          zones=_zones_state(created_at=90),
                          sweep_index=95, atr=1.0)
    assert scan_setups(df, states) == []
    # NOTE: the lower bound (risk < 0.25 ATR) is unreachable by
    # construction: stop = far edge - 0.25*ATR and mid >= far edge for
    # longs (mirrored for shorts), so risk >= 0.25*ATR always.


def test_setup_requires_price_above_midline_long():
    df = _quiet_df(120, price=104.0)       # close 104 < mid 105
    states = _flat_states(120, t_setup=100,
                          zones=_zones_state(created_at=90),
                          sweep_index=95)
    assert scan_setups(df, states) == []


def test_setup_requires_features_attached():
    df = _quiet_df(120)
    states = [_state(i, kind=None) for i in range(120)]
    states[100] = _state(100, zones=None, sweep_index=95)
    assert scan_setups(df, states) == []


def test_short_setup_mirrors():
    rows = [(84, 85, 83, 84)] * 120
    df = make_df(rows)
    # supply zone 100..110 (mid 105); short stop = top + 0.25*10 = 112.5
    states = _flat_states(120, t_setup=100, direction=-1, bos_level=96.0,
                          zones=_zones_state(direction=-1, top=110.0,
                                             bottom=100.0, created_at=90),
                          sweep_index=95, close=84.0)
    setups = scan_setups(df, states)
    assert len(setups) == 1
    s = setups[0]
    assert s.direction == -1 and s.mid == pytest.approx(105.0)
    assert s.stop == pytest.approx(112.5)
    assert s.risk == pytest.approx(7.5)


# -------------------------------------------------------------------- fill
def _setup(t=100, direction=1, mid=105.0, stop=97.5, atr=10.0,
           entry_mode="mid"):
    risk = (mid - stop) * direction
    return RetestSetup(index=t, direction=direction, mid=mid, stop=stop,
                       risk=risk, atr=atr, entry_mode=entry_mode,
                       zone_top=110.0, zone_bottom=100.0,
                       zone_created_at=98, bos_index=98, bos_level=118.0,
                       sweep_index=95)


def test_fill_on_midline_touch():
    rows = [(116, 117, 115, 116)] * 101 + [(106, 107, 104, 106)] + \
        [(106, 107, 105, 106)] * 30
    df = make_df(rows)
    fl = find_fill(df, _setup())
    assert fl is not None and fl.fill_index == 101
    assert fl.entry == pytest.approx(105.0) and not fl.same_bar_stop


def test_fill_adverse_on_gap_through_midline():
    rows = [(116, 117, 115, 116)] * 101 + [(103, 104, 102, 103)] + \
        [(103, 104, 102, 103)] * 30
    df = make_df(rows)
    fl = find_fill(df, _setup())
    assert fl is not None and fl.entry == pytest.approx(103.0)


def test_fill_same_bar_stop_when_bar_trades_through_both():
    # bar to 96 wicks through mid 105 AND stop 97.5 -> filled then
    # stopped the same bar.
    rows = [(116, 117, 115, 116)] * 101 + [(106, 107, 96, 102)] + \
        [(102, 103, 101, 102)] * 30
    df = make_df(rows)
    fl = find_fill(df, _setup())
    assert fl is not None and fl.same_bar_stop
    out = simulate_trade(df, fl, target=130.0)
    assert out.exit_reason == "stop" and out.hold_bars == 0
    assert out.r_net == pytest.approx(-1.0 - 0.45 / 7.5)


def test_fill_same_bar_stop_when_close_kills_zone():
    # close below the zone bottom (100) kills the level; the bar touched
    # the midline on its way through -> filled then stopped.
    rows = [(116, 117, 115, 116)] * 101 + [(106, 107, 104, 99.0)] + \
        [(99, 100, 98, 99)] * 30
    df = make_df(rows)
    fl = find_fill(df, _setup())
    assert fl is not None and fl.same_bar_stop


def test_fill_expires_with_no_touch():
    df = _quiet_df(200)
    assert find_fill(df, _setup()) is None


# ------------------------------------------------------------------ target
def _pool(price, created_at):
    return LiquidityPool(price=price, side="bsl", strength=1,
                         created_at=created_at)


def test_pool_target_needs_one_r_room_and_picks_nearest():
    rows = [(116, 117, 115, 116)] * 101 + [(106, 107, 104, 106)] + \
        [(120, 121, 119, 120)] * 80
    df = make_df(rows)
    fl = find_fill(df, _setup())          # entry 105, risk 7.5
    assert fl is not None
    # near pool 110 < 105+7.5 -> skipped; far pool 120 taken
    near, far = _pool(110.0, 99), _pool(120.0, 99)
    states = [_state(i, kind=None) for i in range(len(df))]
    states[fl.fill_index] = _state(fl.fill_index, kind=None,
                                   pools=(near, far))
    assert select_pool_target(states, fl) == pytest.approx(120.0)


def test_pool_target_excludes_future_pools():
    rows = [(116, 117, 115, 116)] * 101 + [(106, 107, 104, 106)] + \
        [(120, 121, 119, 120)] * 80
    df = make_df(rows)
    fl = find_fill(df, _setup())
    assert fl is not None
    future = _pool(120.0, fl.fill_index + 5)
    states = [_state(i, kind=None) for i in range(len(df))]
    states[fl.fill_index] = _state(fl.fill_index, kind=None,
                                   pools=(future,))
    assert select_pool_target(states, fl) is None


# --------------------------------------------------------------- simulate
def test_stop_wins_tie_and_cost_applies():
    rows = ([(116, 117, 115, 116)] * 101 + [(106, 107, 104, 106)] +
            [(106, 140, 96, 100)] + [(100, 101, 99, 100)] * 80)
    df = make_df(rows)
    fl = find_fill(df, _setup())
    out = simulate_trade(df, fl, target=130.0)
    assert out.exit_reason == "stop"
    assert out.r_net == pytest.approx(-1.0 - 0.45 / 7.5)


def test_target_hit_books_profit():
    rows = ([(116, 117, 115, 116)] * 101 + [(106, 107, 104, 106)] +
            [(106, 121, 105, 120)] + [(120, 121, 119, 120)] * 80)
    df = make_df(rows)
    fl = find_fill(df, _setup())
    out = simulate_trade(df, fl, target=120.0)
    assert out.exit_reason == "target"
    assert out.r_net == pytest.approx((120.0 - 105.0) / 7.5 - 0.45 / 7.5)


def test_timeout_closes_at_horizon():
    df = _quiet_df(400)
    fl = find_fill(df, _setup())
    assert fl is None                    # never touches: no trade
    fl = RetestFill(_setup(), 101, 105.0)
    out = simulate_trade(df, fl, target=200.0)
    assert out.exit_reason == "timeout"
    assert out.exit_index == 101 + 72


# ------------------------------------------------------ edge entry mode
def test_edge_entry_uses_zone_top_for_longs():
    df = _quiet_df(120)
    states = _flat_states(120, t_setup=100,
                          zones=_zones_state(created_at=90),
                          sweep_index=95)
    setups = scan_setups(df, states, entry="edge")
    assert len(setups) == 1
    s = setups[0]
    assert s.entry_mode == "edge"
    assert s.mid == pytest.approx(110.0)      # zone top = limit price
    assert s.stop == pytest.approx(97.5)      # zone bottom - 0.25*10
    assert s.risk == pytest.approx(12.5)


def test_edge_entry_uses_zone_bottom_for_shorts():
    rows = [(84, 85, 83, 84)] * 120
    df = make_df(rows)
    states = _flat_states(120, t_setup=100, direction=-1, bos_level=96.0,
                          zones=_zones_state(direction=-1, top=110.0,
                                             bottom=100.0, created_at=90),
                          sweep_index=95, close=84.0)
    setups = scan_setups(df, states, entry="edge")
    assert len(setups) == 1
    s = setups[0]
    assert s.mid == pytest.approx(100.0)      # zone bottom = limit price
    assert s.stop == pytest.approx(112.5)     # zone top + 0.25*10
    assert s.risk == pytest.approx(12.5)


def test_edge_fill_on_edge_touch_without_midline_touch():
    # bar wicks to 109 (touches zone top 110? no: 109 < 110, no fill)...
    # bar wicks to 110 exactly -> fills at the edge, far above the
    # midline (105) and the stop (97.5).
    rows = [(116, 117, 115, 116)] * 101 + [(116, 117, 110, 114)] + \
        [(114, 115, 113, 114)] * 30
    df = make_df(rows)
    fl = find_fill(df, _setup(mid=110.0, stop=97.5, entry_mode="edge"))
    assert fl is not None and fl.fill_index == 101
    assert fl.entry == pytest.approx(110.0) and not fl.same_bar_stop


def test_edge_entry_wider_risk_than_mid():
    # same zone: edge risk = full zone height + buffer, mid risk = half
    # height + buffer.
    df = _quiet_df(120)
    states = _flat_states(120, t_setup=100,
                          zones=_zones_state(created_at=90),
                          sweep_index=95)
    mid = scan_setups(df, states, entry="mid")[0]
    edge = scan_setups(df, states, entry="edge")[0]
    assert edge.risk == pytest.approx(mid.risk + 5.0)  # half zone height


# --------------------------------------------------------------- causality
def test_no_lookahead_on_real_data():
    """Every setup/fill/target on real data uses only bars <= decision,
    for both entry modes."""
    from goldenswing.backtest.data import load_gold_h1
    from goldenswing.features.engine import read_and_featurize
    df = load_gold_h1()
    states = read_and_featurize(df)
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    for entry in ("mid", "edge"):
        setups = scan_setups(df, states, entry=entry)
        assert len(setups) > 0, f"expected setups ({entry}) on 2y of gold"
        for s in setups:
            assert s.entry_mode == entry
            assert s.zone_created_at <= s.index
            assert s.bos_index <= s.index
            assert s.sweep_index < s.bos_index
            # pullback geometry: price on the favourable side of the
            # level at the signal bar
            if s.direction == 1:
                assert df["close"].to_numpy()[s.index] > s.mid
            else:
                assert df["close"].to_numpy()[s.index] < s.mid
        for s in setups[:50]:
            fl = find_fill(df, s)
            if fl is None:
                continue
            assert fl.fill_index > s.index
            assert fl.fill_index <= s.index + ORDER_EXPIRY_BARS
            for p in (states[fl.fill_index].pools_above
                      if s.direction == 1
                      else states[fl.fill_index].pools_below):
                assert p.created_at <= fl.fill_index
