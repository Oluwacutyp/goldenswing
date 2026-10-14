"""Mode configurations: one market read, three execution modes.

Every mode runs the SAME MarketReader, the SAME FeatureEngine and the
SAME detector (Phase-2 flags + Combo A families).  A mode changes only:

- the bar timeframe (which also rescales every bar-count parameter of
  the reader/features — 48 bars of AMD base is 48h on H1, 12h on M15,
  4h on M5; that rescaling is inherent to "same read, faster clock"
  and is documented, not hidden);
- the entry filters (killzones, setup kinds, displacement strength);
- the exit style and horizon;
- nothing about costs: $0.45/oz round trip in every mode, so the
  cost-as-R damage at small timeframes is visible, not assumed away.

Each mode's harness run is independent: its own data frame, its own
reader pass, its own detector instance.  No state is shared between
modes.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..backtest.ablation import P2_FLAGS
from ..backtest.validation import COMBO_A
from ..golden.detector import DetectorConfig

#: Score a signal needs to be traded, in every mode (the Combo A book
#: was built at this gate).
GOLDEN_THRESHOLD = 60.0


@dataclass(frozen=True)
class ModeConfig:
    name: str
    interval: str                 # yfinance interval: "1h" | "15m" | "5m"
    period: str                   # yfinance period: "2y" | "60d"
    threshold: float = GOLDEN_THRESHOLD
    killzone_only: bool = False
    allowed_kinds: frozenset | None = None  # None = every setup kind
    min_displacement: float = 0.0  # min |disp_3| (ATR bodies) in trade dir
    exit_style: str = "swing_v2"   # "swing_v2" | "nearest_pool"
    horizon: int = 72              # max hold, bars of the mode timeframe
    stop_buffer_atr: float = 0.25  # for the nearest_pool simulator
    cost_per_oz: float = 0.45      # round trip, same as swing — on purpose

    def detector_config(self, route_table: object = None) -> DetectorConfig:
        """The one detector every mode shares (Combo A + Phase-2
        flags), optionally carrying a Phase-5 route table."""
        return DetectorConfig(
            emit_floor=35.0, golden_threshold=self.threshold,
            feature_families=COMBO_A, route_table=route_table,
            **P2_FLAGS)


#: Swing — the system as validated in Phase 3.5: H1 signals, v2 swing
#: exits (buffered structural stop, partial at first pool, trail),
#: 72-bar (~3 day) horizon.
SWING = ModeConfig(
    name="swing", interval="1h", period="2y",
    exit_style="swing_v2", horizon=72)

#: Scalp — M15 through the same read.  Entries only inside the
#: London/NY killzones (the hours where M15 gold actually moves);
#: target is the nearest mapped liquidity pool, max hold 8 bars (2h).
SCALP = ModeConfig(
    name="scalp", interval="15m", period="60d",
    killzone_only=True, exit_style="nearest_pool", horizon=8)

#: Fast scalp — M5, momentum continuation ONLY: trend-aligned
#: displacement follow-through (no sweep reversals, no counter-trend —
#: the daily veto is already in the shared detector).  Requires a
#: displacement-strength bar in the trade direction (|disp_3| >= 1.0
#: ATR), killzones only, max hold 6 bars (30 min).
FAST_SCALP = ModeConfig(
    name="fast_scalp", interval="5m", period="60d",
    killzone_only=True, allowed_kinds=frozenset({"continuation"}),
    min_displacement=1.0, exit_style="nearest_pool", horizon=6)

MODES = (SWING, SCALP, FAST_SCALP)
