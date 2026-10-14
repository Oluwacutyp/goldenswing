"""Causality: truncating the series must not change earlier states."""
import numpy as np
import pandas as pd

from goldenswing.reader import MarketReader
from conftest import make_df


def _random_walk(n=300, seed=7):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1.0, n).cumsum()
    closes = 2000.0 + steps * 3.0
    rows = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        rows.append((float(o), float(max(o, c) + abs(rng.normal(0, 0.8))),
                     float(min(o, c) - abs(rng.normal(0, 0.8))), float(c)))
    return make_df(rows, start="2024-03-01")


def _digest(s):
    ev = s.last_structure_event
    sw = s.last_sweep
    return (
        s.trend,
        round(s.atr, 8),
        round(s.eq_position, 8),
        round(s.momentum, 8),
        (ev.index, ev.kind, ev.direction) if ev else None,
        (sw.index, sw.direction, round(sw.level, 8)) if sw else None,
        s.sweep_age,
        tuple(sorted((round(g.top, 8), round(g.bottom, 8), g.direction)
                     for g in s.active_fvgs)),
        tuple(sorted((round(o.top, 8), round(o.bottom, 8), o.direction)
                     for o in s.active_obs)),
        tuple(round(p.price, 8) for p in s.pools_above),
        tuple(round(p.price, 8) for p in s.pools_below),
        s.session,
        s.killzone,
        s.vol_regime,
        s.daily_bias,
        s.daily_trend,
        round(s.daily_swing_high, 8) if s.daily_swing_high is not None else None,
        round(s.daily_swing_low, 8) if s.daily_swing_low is not None else None,
        s.bull_disp_age,
        s.bear_disp_age,
        s.bull_disp_fvg_age,
        s.bear_disp_fvg_age,
        s.trend_since,
    )


def test_truncation_does_not_change_past_states():
    df = _random_walk()
    cut = 180
    full = MarketReader().read(df)
    trunc = MarketReader().read(df.iloc[:cut])
    assert len(full) == len(df)
    assert len(trunc) == cut
    for t in [0, 5, 25, 60, 120, cut - 1]:
        assert _digest(full[t]) == _digest(trunc[t]), f"state diverged at bar {t}"
