"""Study mathematics for the chart renderers (SPEC §8.4).

The matplotlib and TradingView renderers both draw declared
studies from the signal-model registry; the calculations live
here, once, so the two mediums cannot disagree about what a
study IS.  A study is computed on ITS OWN clock's frame (the
frame named by its resolved ``clock_tf``) and aligned onto a
panel by as-of join: each panel bar takes the study's last value
at or before its own timestamp — never a value from the future,
never a re-computation on the wrong clock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def asof_align(src_index, values, panel_index) -> np.ndarray:
    """Last source value at or before each panel timestamp.

    NaN where the panel bar predates the source series."""
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0 or len(panel_index) == 0:
        return np.full(len(panel_index), np.nan)
    try:
        src_ns = np.asarray(src_index.asi8, dtype=np.int64)
        panel_ns = np.asarray(panel_index.asi8, dtype=np.int64)
    except Exception:
        return np.full(len(panel_index), np.nan)
    pos = np.searchsorted(src_ns, panel_ns, side="right") - 1
    out = np.full(len(panel_index), np.nan)
    ok = pos >= 0
    out[ok] = vals[pos[ok]]
    return out


def _sma(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window, min_periods=1).mean()


def study_series(frame, kind: str, params: dict | None = None
                 ) -> dict | None:
    """Full-frame component series for one price/sub study kind.

    Returns {component: np.ndarray} aligned to ``frame``'s index,
    or None when the frame cannot support the study (too short,
    no volume for VWAP).  Kinds: ``bollinger`` (mid/upper/lower),
    ``atr_band`` (upper/lower), ``vwap`` (vwap), ``ao`` (ao),
    ``ema_pair`` (fast/slow)."""
    if frame is None or len(frame) == 0 or "close" not in frame:
        return None
    params = params or {}
    close = frame["close"].astype(float)
    if kind == "bollinger":
        period = int(params.get("period", 20))
        mult = float(params.get("mult", 2.0))
        mid = _sma(close, period)
        sd = close.rolling(period, min_periods=1).std(ddof=0)
        return {"mid": mid.to_numpy(dtype=float),
                "upper": (mid + mult * sd).to_numpy(dtype=float),
                "lower": (mid - mult * sd).to_numpy(dtype=float)}
    if kind == "atr_band":
        period = int(params.get("period", 14))
        mult = float(params.get("mult", 1.0))
        if "high" not in frame or "low" not in frame:
            return None
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        prev = close.shift(1)
        tr = pd.concat([(high - low), (high - prev).abs(),
                        (low - prev).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False,
                     min_periods=1).mean()
        return {"upper": (close + mult * atr).to_numpy(dtype=float),
                "lower": (close - mult * atr).to_numpy(dtype=float)}
    if kind == "vwap":
        vol_col = next((c for c in ("tick_volume", "volume",
                                    "real_volume")
                        if c in frame), None)
        if vol_col is None or "high" not in frame \
                or "low" not in frame:
            return None
        vol = frame[vol_col].astype(float)
        typical = (frame["high"].astype(float)
                   + frame["low"].astype(float) + close) / 3.0
        cum_vol = vol.cumsum()
        vwap = (typical * vol).cumsum() / cum_vol.where(
            cum_vol > 0)
        return {"vwap": vwap.to_numpy(dtype=float)}
    if kind == "ao":
        if "high" not in frame or "low" not in frame:
            return None
        median = (frame["high"].astype(float)
                  + frame["low"].astype(float)) / 2.0
        ao = _sma(median, 5) - _sma(median, 34)
        return {"ao": ao.to_numpy(dtype=float)}
    if kind == "ema_pair":
        fast = close.ewm(span=int(params.get("fast", 9)),
                         adjust=False).mean()
        slow = close.ewm(span=int(params.get("slow", 21)),
                         adjust=False).mean()
        return {"fast": fast.to_numpy(dtype=float),
                "slow": slow.to_numpy(dtype=float)}
    return None


def study_levels(frame, kind: str) -> dict | None:
    """Horizontal level studies: {name: price} or None.

    ``volume_profile`` reads the developing value area of the
    frame (causal, per-bar — the Pass 7c organ): POC / VAH / VAL
    of the frame's final state, volume basis tick_volume as the
    organ itself names it."""
    if frame is None or len(frame) == 0:
        return None
    if kind == "volume_profile":
        try:
            from ..live.developing_value import current_developing
            state = current_developing(frame)
        except Exception:
            return None
        if not state:
            return None
        out = {}
        for name, key in (("POC", "poc"), ("VAH", "vah"),
                          ("VAL", "val")):
            value = state.get(key)
            if value is not None:
                out[name] = float(value)
        return out or None
    return None
