"""MacroTrend: multi-timeframe alignment (the owner's Golden Swing).

The owner's bot (myxauusdbot v17) defines Golden Swing as D1 + W1 +
MN1 EMA200 biases all agreeing: alignment 1.5x, 1.5x size, and a
Telegram confirmation flow. This module ports that definition and
goes beyond it in two ways:

1. Structure agreement is computed too (the existing swing-based
   ``evaluate_bias`` per TF) and reported alongside the EMA bias, so
   a Golden Swing whose structure disagrees is visible instead of
   silently "max confidence".
2. Alignment here is a SIZING + CONFIRMATION layer, never a
   hierarchy gate leg. It cannot block a trade — the hierarchy
   decides that. It only scales the size of a trade the hierarchy
   already allowed, and asks for a tap on Golden Swing setups.

EMA bias: last close vs EMA200 (EWMA, span=200, adjust=False —
the owner's formula). Fewer than 200 bars -> NEUTRAL (0): a
200-period average on 50 bars is not a bias, it is noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .multitf import evaluate_bias

EMA_PERIOD = 200

#: Alignment ladder, exactly the owner's: D1-only 1.0x, D1+W1 1.25x,
#: D1+W1+MN1 (Golden Swing) 1.5x.
ALIGN_D1_ONLY = 1.0
ALIGN_D1_W1 = 1.25
ALIGN_GOLDEN_SWING = 1.5

#: Size multipliers: Golden Swing 1.5x, D1/W1 conflict 0.5x, else 1.0x.
SIZE_GOLDEN_SWING = 1.5
SIZE_CONFLICT = 0.5
SIZE_NEUTRAL = 1.0

MACRO_TFS = ("D1", "W1", "MN1")


@dataclass(frozen=True)
class MacroTrendResult:
    """Alignment snapshot.  Biases are +1 / 0 / -1 (bull / flat / bear)."""

    d1_ema: int = 0
    w1_ema: int = 0
    mn1_ema: int = 0
    d1_structure: int = 0
    w1_structure: int = 0
    mn1_structure: int = 0
    alignment_level: float = ALIGN_D1_ONLY
    alignment_label: str = "D1_Only"
    golden_swing: bool = False
    conflict: bool = False
    structure_confirms: bool = False
    size_multiplier: float = SIZE_NEUTRAL

    def describe(self) -> str:
        """One line for /status + journal, e.g. ``D1 +1 / W1 +1 / MN1 +1
        -> Golden Swing 1.5x (structure agrees)``."""
        sign = lambda b: f"{b:+d}"
        bits = (f"D1 {sign(self.d1_ema)} / W1 {sign(self.w1_ema)} "
                f"/ MN1 {sign(self.mn1_ema)} -> {self.alignment_label} "
                f"{self.alignment_level:g}x")
        if self.conflict:
            bits += " (D1/W1 conflict)"
        bits += (" (structure agrees)" if self.structure_confirms
                 else " (structure disagrees)")
        return bits


def ema_bias(frame: pd.DataFrame | None, period: int = EMA_PERIOD) -> int:
    """EMA bias of one frame: +1 close above EMA, -1 below, 0 flat /
    missing / too short.  Uses the frame's last CLOSED bar only —
    callers pass closed bars (the feed's ``_closed_only`` contract)."""
    if frame is None or len(frame) < period:
        return 0
    try:
        ema = frame["close"].ewm(span=period, adjust=False).mean().iloc[-1]
        last = float(frame["close"].iloc[-1])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0
    if last > ema:
        return 1
    if last < ema:
        return -1
    return 0


def structure_bias(frame: pd.DataFrame | None) -> int:
    """Swing-structure bias of one frame via the existing
    ``evaluate_bias`` (last two confirmed swings).  Missing / short
    frame -> 0."""
    if frame is None or len(frame) == 0:
        return 0
    try:
        return int(evaluate_bias(frame))
    except Exception:
        return 0


def compute_macrotrend(frames: dict | None) -> MacroTrendResult:
    """Alignment from a ``{tf: frame}`` mapping (closed bars only).

    Golden Swing = D1, W1, MN1 EMA biases all equal and nonzero
    (the owner's definition — structure does NOT change the flag,
    it is reported separately in ``structure_confirms``).
    """
    frames = frames or {}
    d1 = ema_bias(frames.get("D1"))
    w1 = ema_bias(frames.get("W1"))
    mn1 = ema_bias(frames.get("MN1"))
    d1_s = structure_bias(frames.get("D1"))
    w1_s = structure_bias(frames.get("W1"))
    mn1_s = structure_bias(frames.get("MN1"))

    conflict = d1 != 0 and w1 != 0 and d1 != w1
    if d1 != 0 and d1 == w1 == mn1:
        level, label, golden = (ALIGN_GOLDEN_SWING, "Golden_Swing", True)
    elif d1 != 0 and d1 == w1:
        level, label, golden = (ALIGN_D1_W1, "D1_W1", False)
    else:
        level, label, golden = (ALIGN_D1_ONLY, "D1_Only", False)
    if golden:
        mult = SIZE_GOLDEN_SWING
    elif conflict:
        mult = SIZE_CONFLICT
    else:
        mult = SIZE_NEUTRAL
    confirms = (d1 != 0 and w1 != 0 and mn1 != 0
                and d1_s == d1 and w1_s == w1 and mn1_s == mn1)
    return MacroTrendResult(
        d1_ema=d1, w1_ema=w1, mn1_ema=mn1,
        d1_structure=d1_s, w1_structure=w1_s, mn1_structure=mn1_s,
        alignment_level=level, alignment_label=label,
        golden_swing=golden, conflict=conflict,
        structure_confirms=confirms, size_multiplier=mult)
