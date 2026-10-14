"""Core data structures for the market-reading layer.

Every timestamp in this module is a *bar index* in the source frame, and every
event records the bar at which it became *known* (``confirmed_at`` / creation
bar).  A :class:`MarketState` for bar ``t`` is built exclusively from events
known at or before ``t`` — that is what makes the reader causal.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SwingPoint:
    """A confirmed fractal swing."""

    price: float
    index: int          # bar where the swing extreme sits
    confirmed_at: int   # bar where the fractal became knowable (index + n)
    kind: str           # "high" | "low"


@dataclass
class FlipLevel:
    """A broken S/R level — support that broke (now resistance, direction=-1)
    or resistance that broke (now support, direction=+1). The polarity flip
    is a classic reversal setup: price retests the broken level from the
    other side and gets rejected.
    """
    price: float
    broken_at: int       # bar where the break closed beyond the level
    direction: int       # -1: broken support -> resistance | +1: broken resistance -> support
    swing_index: int     # bar of the original swing point
    active: bool = True


@dataclass
class LiquidityPool:
    """Resting stops around an obvious level (buy-side above highs,
    sell-side below lows)."""

    price: float
    side: str           # "bsl" (buy-side, above price action) | "ssl"
    strength: int       # number of swing points clustered at the level
    created_at: int
    swept_at: int | None = None
    cleared_at: int | None = None

    @property
    def active(self) -> bool:
        return self.swept_at is None and self.cleared_at is None


@dataclass(frozen=True)
class SweepEvent:
    """A liquidity sweep: wick beyond a level, close back on the other side.

    ``direction`` is the *trade* direction the sweep sets up: sweeping
    sell-side liquidity below the lows grabs stops and fuels a move UP, so
    ``direction == +1``; sweeping buy-side above the highs gives ``-1``.
    """

    index: int
    level: float
    extreme: float      # wick extreme that took the liquidity
    direction: int      # +1 bullish (swept lows) | -1 bearish (swept highs)


@dataclass(frozen=True)
class StructureEvent:
    """BOS (break of structure, trend continuation) or CHoCH (change of
    character, trend flip)."""

    index: int
    kind: str           # "BOS" | "CHOCH"
    direction: int      # +1 up break | -1 down break
    level: float        # the swing price that was broken


@dataclass
class FairValueGap:
    """3-candle imbalance.  Bullish: low[t] > high[t-2], gap is
    (high[t-2], low[t]).  Known at the close of candle t."""

    top: float
    bottom: float
    direction: int      # +1 bullish | -1 bearish
    created_at: int
    mitigated_at: int | None = None

    @property
    def active(self) -> bool:
        return self.mitigated_at is None

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.top + self.bottom)


@dataclass
class OrderBlock:
    """The last opposite candle before a displacement leg that broke
    structure.  Zone is the candle's [low, high]."""

    top: float
    bottom: float
    direction: int      # +1 bullish (demand) | -1 bearish (supply)
    created_at: int
    candle_index: int
    mitigated_at: int | None = None

    @property
    def active(self) -> bool:
        return self.mitigated_at is None


@dataclass
class MarketState:
    """Everything the system knows about the market at one bar.

    Built only from bars <= ``index``.  Lists are capped snapshots of the
    most relevant active structures.
    """

    index: int
    timestamp: object           # pandas Timestamp of the bar
    close: float
    atr: float

    # structure
    trend: str                  # "up" | "down" | "range"
    bias: int                   # +1 / -1 / 0 derived from trend
    recent_swing_highs: list[SwingPoint] = field(default_factory=list)
    recent_swing_lows: list[SwingPoint] = field(default_factory=list)
    last_structure_event: StructureEvent | None = None
    structure_event_age: int | None = None

    # liquidity
    pools_above: list[LiquidityPool] = field(default_factory=list)  # nearest first
    pools_below: list[LiquidityPool] = field(default_factory=list)  # nearest first
    last_sweep: SweepEvent | None = None
    sweep_age: int | None = None

    # ICT levels
    active_fvgs: list[FairValueGap] = field(default_factory=list)
    active_obs: list[OrderBlock] = field(default_factory=list)

    # dealing range / premium-discount
    range_high: float = float("nan")
    range_low: float = float("nan")
    eq_position: float = 0.5    # 0 = range low, 1 = range high
    zone: str = "equilibrium"  # "premium" | "discount" | "equilibrium"

    # regime
    atr_percentile: float = 50.0    # 0..100 over rolling window
    vol_regime: str = "normal"      # "compressed" | "normal" | "expanded"
    momentum: float = 0.0           # ATR-normalised ROC(10)
    body_ratio: float = 0.0         # signed body / ATR of current bar
    bull_disp_3: float = 0.0        # strongest bullish body/ATR in last 3 bars
    bear_disp_3: float = 0.0        # strongest bearish body/ATR in last 3 bars
    # bars since the last displacement-strength candle (body >= 1.0x ATR)
    bull_disp_age: int | None = None
    bear_disp_age: int | None = None
    # bars since a displacement FVG was created (gap whose creating candle
    # had body >= 1.0x ATR)
    bull_disp_fvg_age: int | None = None
    bear_disp_fvg_age: int | None = None
    trend_since: int | None = None  # bar the current trend state began

    # higher timeframe (D1, completed days only — see MarketReader)
    daily_trend: str = "range"      # "up" | "down" | "range"
    daily_bias: int = 0             # +1 / -1 / 0
    daily_swing_high: float | None = None   # most recent confirmed D1 swing
    daily_swing_low: float | None = None

    # EMA stack alignment (9/21/50): +1 bullish stack (9>21>50),
    # -1 bearish stack (9<21<50), 0 mixed. The user's pullback-bot
    # filter — signals trade with the stack, never against it.
    ema_stack: int = 0

    # S/R flip levels: broken supports/resistances awaiting a retest.
    # A broken support (direction=-1) is a short setup when price
    # rallies back into it; a broken resistance (direction=+1) is a
    # long setup when price pulls back into it.
    active_flips: list = field(default_factory=list)

    # sessions
    session: str = "off"            # asia|london|overlap|newyork|off
    killzone: bool = False
    session_high: float = float("nan")
    session_low: float = float("nan")

    # Phase-3 feature layer (goldenswing.features.FeatureSet), attached
    # post-read by the FeatureEngine.  Typed as object so this module
    # never imports the features package (no import cycle).  None means
    # "features not computed" and the detector must behave exactly as
    # Phase 2 in that case.
    features: object = None

    def nearest_fvg(self, direction: int) -> FairValueGap | None:
        """Nearest active FVG in ``direction`` whose zone is at/near price."""
        best: FairValueGap | None = None
        best_dist = float("inf")
        for g in self.active_fvgs:
            if g.direction != direction:
                continue
            if self.close > g.top:
                dist = self.close - g.top
            elif self.close < g.bottom:
                dist = g.bottom - self.close
            else:
                dist = 0.0
            if dist < best_dist:
                best, best_dist = g, dist
        return best

    def nearest_ob(self, direction: int) -> OrderBlock | None:
        best: OrderBlock | None = None
        best_dist = float("inf")
        for ob in self.active_obs:
            if ob.direction != direction:
                continue
            if self.close > ob.top:
                dist = self.close - ob.top
            elif self.close < ob.bottom:
                dist = ob.bottom - self.close
            else:
                dist = 0.0
            if dist < best_dist:
                best, best_dist = ob, dist
        return best
