"""Unified Multi-Style Architecture: one fractal engine, all
horizons.

The SAME state machine — HTF Zone -> Sweep -> Displacement ->
CHoCH -> FVG — applies on ALL timeframes.  Context changes the
horizon, NOT the system.  There are no separate swing / scalp
engines, no hardcoded style per playbook.

Style is DERIVED from (HTF zone TF, trigger TF, target
distance), never hardcoded:

- Macro Swing: HTF zone = MN1/W1, trigger = D1/H4, target =
  opposite W1 liquidity pool, holding days-weeks, SL = sweep
  extreme (wide), size = risk / wide SL (smaller lots),
  TP = W1 opposite zone / 3R+.  Log style=MACRO_SWING.
- Intraday Trend: HTF zone = D1/H4, trigger = H1/M15, target =
  H4/D1 opposite zone / liquidity, holding hours, session
  killzone London/NY open preferred, SL = H4 sweep extreme,
  size = risk / SL, TP = 2-3R.  Log style=INTRADAY_TREND.
- Micro Scalp: HTF zone = H1/M15 range edge / M15 FVG,
  trigger = M5/M1 nested CHoCH + sub-FVG, target = next
  M15/H1 liquidity pool / 1-2R, holding minutes-hour, SL =
  M5 sub-CHoCH extreme (tight), size = risk / tight SL
  (larger lots, capped by margin), TP = 1-2R / next pool.
  Log style=MICRO_SCALP.

Concurrency: if multiple styles fire at the same time (e.g.
macro swing long + micro scalp short inside the same range),
they are SEPARATE setups with separate SL/TP/size — never
conflicting.  Log both, manage separately, never cancel one
because the other exists.  Position caps: max 1 per style
per zone, max 3 total open, margin shield gates all.

``derive_style`` is the pure derivation (zone TF decides the
base style; trigger TF + target distance are recorded in the
log, not used to override).  ``StyleBook`` is the pure
position-cap rule: the runner consults it before opening a
setup, one setup per (style, zone), three setups total.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TF_RANK = {"MN1": 0, "W1": 1, "D1": 2, "H4": 3, "H1": 4,
           "M15": 5, "M5": 6, "M1": 7}

STYLES = ("MACRO_SWING", "INTRADAY_TREND", "MICRO_SCALP")

MAX_TOTAL_OPEN = 3


def derive_style(zone_tf: str, trigger_tf: str,
                 target_atr: float | None = None) -> str:
    """Derive the style from the HTF zone TF (+ trigger TF and
    target distance for the log).  Zone TF decides: MN1/W1 ->
    MACRO_SWING, D1/H4 -> INTRADAY_TREND, H1/M15/M5/M1 ->
    MICRO_SCALP.  Unknown zone TF -> INTRADAY_TREND (the
    engine's middle horizon), never a crash."""
    if zone_tf in ("MN1", "W1"):
        return "MACRO_SWING"
    if zone_tf in ("D1", "H4"):
        return "INTRADAY_TREND"
    if zone_tf in ("H1", "M15", "M5", "M1"):
        return "MICRO_SCALP"
    return "INTRADAY_TREND"


@dataclass
class StyleBook:
    """Position-cap book: max 1 per style per zone, max 3
    total open.  Pure rule — the runner owns the positions
    and consults this before opening a setup."""

    open_setups: set = field(default_factory=set)  # (style, zone_id)

    def can_open(self, style: str, zone_id: str) -> tuple[bool, str]:
        if (style, zone_id) in self.open_setups:
            return False, (f"already open: {style} on {zone_id} "
                           "(1 per style per zone)")
        if len(self.open_setups) >= MAX_TOTAL_OPEN:
            return False, (f"position cap: {MAX_TOTAL_OPEN} total "
                           "open")
        return True, ""

    def open(self, style: str, zone_id: str) -> tuple[bool, str]:
        ok, reason = self.can_open(style, zone_id)
        if not ok:
            return False, reason
        self.open_setups.add((style, zone_id))
        return True, ""

    def close(self, style: str, zone_id: str) -> None:
        self.open_setups.discard((style, zone_id))
