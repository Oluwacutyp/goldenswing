"""CHoCH -> FVG-midline -> opposing-OB sleeve: setup, fill, target, causality."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_df

from goldenswing.experiments.choch_fvg import (
    CHOCH_FRESH_BARS,
    ORDER_EXPIRY_BARS,
    find_fill,
    scan_setups,
    select_ob_target,
    simulate_trade,
    _fvg_alive,
    _obs_active_at,
)
from goldenswing.reader.state import (FairValueGap, MarketState, OrderBlock,
                                      StructureEvent)


# ------------------------------------------------------------------ helpers
def _fvg(direction=1, top=110.0, bottom=100.0, created_at=90):
    return FairValueGap(top=top, bottom=bottom, direction=direction,
                        created_at=created_at)


def _ob(direction=-1, top=130.0, bottom=125.0, created_at=95):
    return OrderBlock(top=top, bottom=bottom, direction=direction,
                      created_at=created_at, candle_index=94)


def _state(t, choch_age=2, direction=1, fvgs=(), obs=(), atr=10.0,
           close=112.0, kind="CHOCH"):
    ev = None if kind is None else StructureEvent(
        index=t - choch_age, kind=kind, direction=direction,
        level=95.0 if direction == 1 else 130.0)
    return MarketState(
        index=t, timestamp=None, close=close, atr=atr, trend="up", bias=1,
        last_structure_event=ev,
        structure_event_age=choch_age if kind else None,
        active_fvgs=list(fvgs), active_obs=list(obs))


def _flat_states(n, t_setup=100, **kw):
    """n states, all blank except a configured setup at bar t_setup."""
    states = [_state(i, kind=None) for i in range(n)]
    states[t_setup] = _state(t_setup, **kw)
    return states


def _quiet_df(n, price=112.0, start="2024-01-01"):
    rows = [(price, price + 1, price - 1, price)] * n
    return make_df(rows, start=start)


# -------------------------------------------------------------------- scan
def test_setup_found_when_fresh_choch_and_clean_fvg():
    df = _quiet_df(120)
    g = _fvg(created_at=98, top=110.0, bottom=100.0)   # mid 105 < 112
    states = _flat_states(120, choch_age=2, fvgs=(g,))
    setups = scan_setups(df, states)
    assert len(setups) == 1
    s = setups[0]
    assert s.direction == 1 and s.mid == pytest.approx(105.0)
    # stop = broken swing low (95) - 0.25 * ATR(10)
    assert s.stop == pytest.approx(92.5)
    assert s.risk == pytest.approx(12.5)


def test_setup_rejected_when_choch_stale():
    df = _quiet_df(150)
    g = _fvg(created_at=120, top=110.0, bottom=100.0)
    states = _flat_states(150, t_setup=140, choch_age=CHOCH_FRESH_BARS + 1,
                          fvgs=(g,))
    assert scan_setups(df, states) == []


def test_setup_rejected_when_midline_already_touched():
    # a wick through the midline since creation disqualifies the FVG
    rows = [(112, 113, 111, 112)] * 98 + [(112, 113, 103, 112)] + \
        [(112, 113, 111, 112)] * 21
    df = make_df(rows)
    g = _fvg(created_at=98, top=110.0, bottom=100.0)   # mid 105, touched@98
    states = _flat_states(120, choch_age=2, fvgs=(g,))
    assert scan_setups(df, states) == []


def test_setup_rejected_when_fvg_too_old_for_choch():
    df = _quiet_df(120)
    g = _fvg(created_at=80, top=110.0, bottom=100.0)  # before choch window
    states = _flat_states(120, choch_age=2, fvgs=(g,))  # choch @98
    assert scan_setups(df, states) == []


def test_setup_rejected_when_stop_too_tight_or_too_wide():
    df = _quiet_df(120)
    # risk 12.5 vs ATR 10 -> fine; now shrink ATR so risk > 4 ATR
    g = _fvg(created_at=98, top=110.0, bottom=100.0)
    states = _flat_states(120, choch_age=2, fvgs=(g,), atr=2.0)
    assert scan_setups(df, states) == []          # 12.5 > 4*2
    # midline just above the buffered stop: risk 0.5 < 0.25 * ATR(10)
    g2 = _fvg(created_at=98, top=94.0, bottom=92.0)   # mid 93
    states = _flat_states(120, choch_age=2, fvgs=(g2,))
    assert scan_setups(df, states) == []


def test_setup_requires_price_above_midline_long():
    df = _quiet_df(120, price=104.0)              # close 104 < mid 105
    g = _fvg(created_at=98, top=110.0, bottom=100.0)
    states = _flat_states(120, choch_age=2, fvgs=(g,))
    assert scan_setups(df, states) == []


def test_short_setup_mirrors():
    rows = [(88, 89, 87, 88)] * 120
    df = make_df(rows)
    g = _fvg(direction=-1, top=100.0, bottom=90.0, created_at=98)  # mid 95
    states = _flat_states(120, choch_age=2, direction=-1, fvgs=(g,),
                          close=88.0)
    setups = scan_setups(df, states)
    assert len(setups) == 1
    s = setups[0]
    assert s.direction == -1 and s.mid == pytest.approx(95.0)
    assert s.stop == pytest.approx(132.5)          # 130 + 0.25*10
    assert s.risk == pytest.approx(37.5)


# -------------------------------------------------------------------- fill
def _setup(t=100, direction=1, mid=105.0, stop=92.5, atr=10.0):
    from goldenswing.experiments.choch_fvg import SleeveSetup
    risk = (mid - stop) * direction
    return SleeveSetup(index=t, direction=direction, mid=mid, stop=stop,
                       risk=risk, atr=atr,
                       fvg_top=110.0, fvg_bottom=100.0, fvg_created_at=98,
                       choch_index=98, choch_level=95.0)


def test_fill_on_midline_touch():
    rows = [(112, 113, 111, 112)] * 101 + [(106, 107, 104, 106)] + \
        [(106, 107, 105, 106)] * 30
    df = make_df(rows)
    fl = find_fill(df, _setup())
    assert fl is not None and fl.fill_index == 101
    assert fl.entry == pytest.approx(105.0) and not fl.same_bar_stop


def test_fill_adverse_on_gap_through_midline():
    rows = [(112, 113, 111, 112)] * 101 + [(103, 104, 102, 103)] + \
        [(103, 104, 102, 103)] * 30
    df = make_df(rows)
    fl = find_fill(df, _setup())
    assert fl is not None and fl.entry == pytest.approx(103.0)


def test_fill_same_bar_stop_when_bar_trades_through_both():
    # stop (101) sits INSIDE the FVG (bottom 100): a bar to 100.5 fills at
    # the midline and stops the same bar without killing the FVG.
    rows = [(112, 113, 111, 112)] * 101 + [(106, 107, 100.5, 102)] + \
        [(102, 103, 101, 102)] * 30
    df = make_df(rows)
    s = _setup(stop=101.0)
    fl = find_fill(df, s)
    assert fl is not None and fl.same_bar_stop
    out = simulate_trade(df, fl, target=130.0)
    assert out.exit_reason == "stop" and out.hold_bars == 0
    assert out.r_net == pytest.approx(-1.0 - 0.45 / 4.0)


def test_fill_expires_with_no_touch():
    df = _quiet_df(200)
    assert find_fill(df, _setup()) is None


def test_fvg_death_cancels_before_fill():
    # bar 101 wicks to 99: below the FVG bottom (100) -> FVG dead.
    # death is checked BEFORE the fill, so no trade (not a fill).
    rows = [(112, 113, 111, 112)] * 101 + [(106, 107, 99, 106)] + \
        [(106, 107, 105, 106)] * 30
    df = make_df(rows)
    assert find_fill(df, _setup()) is None


def test_fvg_alive_rule():
    low = np.array([101.0] * 10 + [99.0])
    high = np.array([110.0] * 11)
    assert _fvg_alive(low, high, 110.0, 100.0, 1, 5, 9)
    assert not _fvg_alive(low, high, 110.0, 100.0, 1, 5, 10)
    # a wick exactly to the bottom edge mitigates (reader: low <= bottom)
    low3 = np.array([101.0] * 10 + [100.0])
    assert not _fvg_alive(low3, high, 110.0, 100.0, 1, 5, 10)
    # creation bar itself never mitigates
    low2 = np.array([99.0] * 11)
    assert _fvg_alive(low2, high, 110.0, 100.0, 1, 10, 10)


# ------------------------------------------------------------------ target
def test_ob_target_needs_one_r_room_and_picks_nearest():
    rows = [(112, 113, 111, 112)] * 101 + [(106, 107, 104, 106)] + \
        [(120, 121, 119, 120)] * 80
    df = make_df(rows)
    fl = find_fill(df, _setup())          # entry 105, risk 12.5
    assert fl is not None
    # near OB: bottom 112 < 105+12.5 -> skipped; far OB bottom 125 taken
    near = _ob(direction=-1, top=115.0, bottom=112.0, created_at=99)
    far = _ob(direction=-1, top=128.0, bottom=125.0, created_at=99)
    states = [_state(i, kind=None, obs=(near, far)) for i in range(len(df))]
    assert select_ob_target(df, states, fl) == pytest.approx(125.0)


def test_ob_target_excludes_future_obs():
    rows = [(112, 113, 111, 112)] * 101 + [(106, 107, 104, 106)] + \
        [(120, 121, 119, 120)] * 80
    df = make_df(rows)
    fl = find_fill(df, _setup())
    assert fl is not None
    future = _ob(direction=-1, top=128.0, bottom=125.0,
                 created_at=fl.fill_index + 5)
    states = [_state(i, kind=None, obs=(future,)) for i in range(len(df))]
    assert select_ob_target(df, states, fl) is None


def test_ob_target_excludes_mitigated_obs():
    # bearish OB dies when a close prints above its top.  The kill bar
    # (100) lands BEFORE the fill bar (101), so the OB is dead at entry.
    rows = ([(112, 113, 111, 112)] * 100 + [(128, 130, 128, 129.5)] +
            [(106, 107, 104, 106)] + [(120, 121, 119, 120)] * 80)
    df = make_df(rows)
    fl = find_fill(df, _setup())
    assert fl is not None and fl.fill_index == 101
    dead = _ob(direction=-1, top=128.0, bottom=125.0, created_at=99)
    states = [_state(i, kind=None, obs=(dead,)) for i in range(len(df))]
    assert select_ob_target(df, states, fl) is None


def test_obs_active_at_union_recovers_capped_obs():
    # state snapshots only carry their own obs list; the union must find
    # an OB that is absent from the fill bar's snapshot.
    df = _quiet_df(120)
    close = df["close"].to_numpy(dtype=float)
    old = _ob(direction=-1, top=128.0, bottom=125.0, created_at=50)
    states = [_state(i, kind=None) for i in range(120)]
    states[60] = _state(60, kind=None, obs=(old,))
    live = _obs_active_at(close, states, 110)
    assert any(o.created_at == 50 for o in live)


# --------------------------------------------------------------- simulate
def test_stop_wins_tie_and_cost_applies():
    rows = ([(112, 113, 111, 112)] * 101 + [(106, 107, 104, 106)] +
            [(106, 140, 90, 100)] + [(100, 101, 99, 100)] * 80)
    df = make_df(rows)
    fl = find_fill(df, _setup())
    out = simulate_trade(df, fl, target=130.0)
    assert out.exit_reason == "stop"
    # stopped at 92.5 from 105: -1R gross, minus cost
    assert out.r_net == pytest.approx(-1.0 - 0.45 / 12.5)


def test_target_hit_books_profit():
    rows = ([(112, 113, 111, 112)] * 101 + [(106, 107, 104, 106)] +
            [(106, 131, 105, 130)] + [(130, 131, 129, 130)] * 80)
    df = make_df(rows)
    fl = find_fill(df, _setup())
    out = simulate_trade(df, fl, target=125.0)
    assert out.exit_reason == "target"
    assert out.r_net == pytest.approx((125.0 - 105.0) / 12.5 - 0.45 / 12.5)


def test_timeout_closes_at_horizon():
    df = _quiet_df(400)
    fl = find_fill(df, _setup())
    # force a fill on bar 101 by hand for the walk test
    from goldenswing.experiments.choch_fvg import SleeveFill
    fl = SleeveFill(_setup(), 101, 105.0)
    out = simulate_trade(df, fl, target=200.0)
    assert out.exit_reason == "timeout"
    assert out.exit_index == 101 + 72


# --------------------------------------------------------------- causality
def test_no_lookahead_on_real_data():
    """Every setup/fill/target on real data uses only bars <= decision."""
    from goldenswing.backtest.data import load_gold_h1
    from goldenswing.reader.reader import MarketReader
    df = load_gold_h1()
    states = MarketReader().read(df)
    setups = scan_setups(df, states)
    assert len(setups) > 0, "expected setups on 2y of gold"
    for s in setups:
        assert s.fvg_created_at <= s.index
        assert s.choch_index <= s.index
        # midline untouched on bars created..index (the scan's own rule)
        low = df["low"].to_numpy(dtype=float)
        seg = low[s.fvg_created_at: s.index + 1]
        if s.direction == 1:
            assert bool((seg > s.mid).all())
    for s in setups[:50]:
        fl = find_fill(df, s)
        if fl is None:
            continue
        assert fl.fill_index > s.index
        assert fl.fill_index <= s.index + ORDER_EXPIRY_BARS
        close = df["close"].to_numpy(dtype=float)
        for ob in _obs_active_at(close, states, fl.fill_index):
            assert ob.created_at <= fl.fill_index
