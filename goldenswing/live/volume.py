"""Volume features from MT5 bars — the old bots' usage, not new logic.

The owner's previous bots used bar ``volume`` (MT5 ``tick_volume``)
three ways, and this module implements exactly those three:

- ``volume_ratio`` — last bar's volume / 20-bar mean volume
  (``rolling(20).mean().iloc[-1]``, mean INCLUDING the last bar,
  the way the old bots computed it).  ``last < avg`` read as weak
  confirmation; ``last > 2 * avg`` read as a volume surge.
- ``vol_z`` — ``(last - mean20) / std20`` of the same 20-bar window.
- ``vwap`` / ``obv`` — volume-weighted average price and on-balance
  volume over the window (``cumsum(close*volume)/cumsum(volume)`` and
  ``cumsum(sign(close.diff) * volume)``).

Volume is CONFIRMATION CONTEXT, never a gate: the features ride on
the hierarchy verdict as ``verdict["volume"]``, and
:func:`volume_bonus` turns the ratio into the old bots' bounded
confirmation score (``clip((ratio - 1) / 2, -0.5, 0.5)``) for the
confluence scorer.  Nothing here blocks a signal — hard veto stays
OFF, and a missing/zero volume column degrades to "no volume
information", never to a fake block.

``real_volume`` (MT5's actual traded volume) is reported alongside
when the terminal provides it — for most metals symbols it is 0,
and ``real_volume_available`` says so plainly instead of letting a
caller mistake tick volume for real volume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Baseline window the old bots used for the volume mean / z-score.
VOLUME_BASELINE_BARS = 20

#: Ratio at/above which the old bots read a volume surge.
VOLUME_SURGE_RATIO = 2.0


def volume_features(frame: pd.DataFrame, *,
                    window: int = VOLUME_BASELINE_BARS) -> dict:
    """Volume features of the LAST (closed) bar of ``frame``.

    ``frame`` needs a ``volume`` column (MT5 ``tick_volume``) and may
    carry ``real_volume``.  Returns a JSON-safe dict; when the frame
    is too short or has no usable volume, ``volume_ratio`` is None
    and ``status`` says why — never a fabricated number.
    """
    out: dict = {
        "volume": None, "real_volume": None,
        "real_volume_available": False,
        "volume_mean20": None, "volume_ratio": None, "vol_z": None,
        "vwap": None, "obv": None,
        "volume_confirms": False, "volume_surge": False,
        "status": "no_volume",
    }
    if frame is None or len(frame) == 0 or "volume" not in frame.columns:
        return out
    sub = frame.tail(window)
    volumes = sub["volume"].to_numpy(dtype=float)
    volumes = volumes[np.isfinite(volumes)]
    if len(volumes) == 0 or float(volumes[-1]) <= 0:
        return out
    last = float(volumes[-1])
    mean = float(np.mean(volumes))
    std = float(np.std(volumes)) if len(volumes) > 1 else 0.0
    out["volume"] = last
    out["volume_mean20"] = mean
    out["volume_ratio"] = last / mean if mean > 0 else None
    out["vol_z"] = (last - mean) / std if std > 0 else 0.0
    closes = sub["close"].to_numpy(dtype=float)
    if len(closes) == len(volumes) and float(np.sum(volumes)) > 0:
        out["vwap"] = float(
            np.sum(closes * volumes) / np.sum(volumes))
        diffs = np.diff(closes, prepend=closes[0])
        out["obv"] = float(np.sum(np.sign(diffs) * volumes))
    if "real_volume" in sub.columns:
        real = sub["real_volume"].to_numpy(dtype=float)
        real = real[np.isfinite(real)]
        if len(real) and float(np.max(real)) > 0:
            out["real_volume_available"] = True
            out["real_volume"] = float(real[-1])
    ratio = out["volume_ratio"]
    if ratio is not None:
        out["volume_confirms"] = ratio >= 1.0
        out["volume_surge"] = ratio >= VOLUME_SURGE_RATIO
        out["status"] = "ok"
    return out


def volume_bonus(volume: dict | None) -> tuple[float, str]:
    """Old bots' bounded volume confirmation score.

    ``clip((volume_ratio - 1) / 2, -0.5, 0.5)`` — at-average volume
    contributes 0, double-average volume contributes +0.5, zero
    volume contributes -0.5.  Returns ``(bonus, note)``; missing
    volume returns ``(0.0, "")`` — no information, no penalty.
    """
    if not volume:
        return 0.0, ""
    ratio = volume.get("volume_ratio")
    if ratio is None:
        return 0.0, ""
    bonus = float(np.clip((float(ratio) - 1.0) / 2.0, -0.5, 0.5))
    return bonus, f"volume_ratio={ratio:.2f}"
