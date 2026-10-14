"""Multi-timeframe context: H4 and D1 read with the SAME MarketReader,
mapped causally onto the H1 bars.

An HTF bar's reader state is only *known* when that bar closes, so an
H1 bar starting at ``ts`` uses the last HTF state whose bar end is
``<= ts + 1h`` (the H1 bar's own close — the moment its features are
evaluated and any signal would fire).  Completed bars only: a
still-forming H4/D1 bar never contributes.  HTF key levels are the
most recent confirmed HTF swing high/low from the mapped state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..reader.reader import MarketReader
from .state import MTFState

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum"}
_SPANS = {"4h": pd.Timedelta(hours=4), "1D": pd.Timedelta(days=1)}


def _htf_states(df: pd.DataFrame, rule: str):
    """(known_at array, trend list, bias list, swing-high list,
    swing-low list) for one resampled timeframe."""
    sub = df.resample(rule, origin="start_day").agg(_AGG).dropna(
        subset=["open", "high", "low", "close"])
    if len(sub) == 0:
        return None
    states = MarketReader().read(sub)
    known_at = (sub.index + _SPANS[rule]).to_numpy()
    trends = [s.trend for s in states]
    biases = [s.bias for s in states]
    shs = [s.recent_swing_highs[-1].price if s.recent_swing_highs else None
           for s in states]
    sls = [s.recent_swing_lows[-1].price if s.recent_swing_lows else None
           for s in states]
    return known_at, trends, biases, shs, sls


def _map(h1_index: pd.DatetimeIndex, packed) -> list[tuple]:
    n = len(h1_index)
    if packed is None:
        return [("range", 0, None, None)] * n
    known_at, trends, biases, shs, sls = packed
    keys = (h1_index + pd.Timedelta(hours=1)).to_numpy()
    pos = np.searchsorted(known_at, keys, side="right") - 1
    out = []
    for p in pos:
        if p < 0:
            out.append(("range", 0, None, None))
        else:
            out.append((trends[p], biases[p], shs[p], sls[p]))
    return out


def compute(df: pd.DataFrame, states, cfg) -> list[MTFState]:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    df = df.copy()
    df.index = idx
    h4 = _map(idx, _htf_states(df, "4h"))
    d1 = _map(idx, _htf_states(df, "1D"))
    out: list[MTFState] = []
    for t, s in enumerate(states):
        h4t, h4b, h4sh, h4sl = h4[t]
        d1t, d1b, d1sh, d1sl = d1[t]
        out.append(MTFState(
            h4_trend=h4t, h4_bias=h4b, d1_trend=d1t, d1_bias=d1b,
            h4_swing_high=h4sh, h4_swing_low=h4sl,
            d1_swing_high=d1sh, d1_swing_low=d1sl,
            alignment=float(np.mean([d1b, h4b, s.bias])),
        ))
    return out
