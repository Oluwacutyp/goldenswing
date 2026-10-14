"""Per-bar feature state: one dataclass per family, bundled in a
:class:`FeatureSet` aligned 1:1 with the reader's bars.

Every field is a *state known at the bar's close* — see common.py for the
causality contract each family honours.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .divergence import Divergence, DivergenceState


# --------------------------------------------------------------- Bollinger
@dataclass(frozen=True)
class BollingerState:
    mid: float = float("nan")
    upper: float = float("nan")
    lower: float = float("nan")
    pct_b: float = float("nan")        # (close-lower)/(upper-lower); 0.5 = mid
    bandwidth: float = float("nan")    # (upper-lower)/mid
    bw_percentile: float = 50.0        # trailing percentile of bandwidth
    squeeze: bool = False              # bandwidth percentile < 20
    walk: int = 0                      # +1 riding the upper band, -1 lower
    rejection: int = 0                 # +1 bullish rejection at lower band,
                                       # -1 bearish rejection at upper band


# -------------------------------------------------------------------- Fib
@dataclass(frozen=True)
class FibState:
    range_high: float = float("nan")
    range_low: float = float("nan")
    leg_direction: int = 0             # +1 last completed leg was up
                                       # (low->high), -1 down, 0 unknown
    levels: tuple = ()                 # ((fraction, price), ...) ascending frac
    nearest_frac: float = float("nan")
    nearest_dist_atr: float = float("nan")
    at_fib: bool = False               # within 0.15 ATR of a level
    ote_low: float = float("nan")      # OTE zone (0.62-0.79 retracement)
    ote_high: float = float("nan")
    in_ote_long: bool = False          # price in the OTE zone for a long
    in_ote_short: bool = False         # (after a down leg)


# --------------------------------------------------------------- Ichimoku
@dataclass(frozen=True)
class IchimokuState:
    tenkan: float = float("nan")
    kijun: float = float("nan")
    cloud_top: float = float("nan")    # cloud values KNOWN at this bar
    cloud_bottom: float = float("nan") # (computed 26 bars ago, causal)
    cloud_bullish: bool = False        # Senkou A above Senkou B (known)
    position: str = "unknown"          # above | below | inside | unknown
    tk_cross: int = 0                  # +1 bullish TK cross this bar
    cloud_twist: int = 0               # +1 twist to bullish this bar
    chikou: int = 0                    # +1 chikou above price 26 bars ago
    # Phase 4: the D1 cloud, mapped causally (last COMPLETED daily bar's
    # known cloud).  ``d1_position`` is the current close vs that cloud.
    d1_position: str = "unknown"       # above | below | inside | unknown
    d1_cloud_bullish: bool = False


# --------------------------------------------------------- relative strength
@dataclass(frozen=True)
class PeerRS:
    available: bool = False
    ratio: float = float("nan")
    trend: int = 0                     # +1 ratio rising, -1 falling, 0 flat
    percentile: float = 50.0           # trailing percentile of the ratio


@dataclass(frozen=True)
class RSState:
    silver: PeerRS = field(default_factory=PeerRS)
    dollar: PeerRS = field(default_factory=PeerRS)
    score: float = float("nan")        # mean of available peer trends
    available: bool = False            # at least one peer usable

    @property
    def any_available(self) -> bool:
        return self.available


# -------------------------------------------------------- classical patterns
@dataclass(frozen=True)
class PatternInstance:
    kind: str          # head_shoulders | inv_head_shoulders |
                       # double_top | double_bottom
    direction: int     # trade direction the pattern implies on completion
    status: str        # "active" (formed, neckline intact) | "broken"
    formed_at: int     # bar the final shoulder/top confirmed
    neckline: float
    extreme: float     # head / double-top extreme price
    invalidation: float
    target: float      # measured-move target


@dataclass(frozen=True)
class PatternsState:
    active: tuple = ()                 # tuple[PatternInstance, ...]
    event: PatternInstance | None = None   # formed or broke on THIS bar


# ---------------------------------------------------------- supply / demand
@dataclass(frozen=True)
class SDZone:
    top: float
    bottom: float
    direction: int     # +1 demand (drop-base-rally) | -1 supply
    created_at: int    # departure bar — the bar the zone became known
    base_bars: int
    departure: float   # departure candle body in ATR units (strength)
    touches: int = 0

    @property
    def freshness(self) -> str:
        return "fresh" if self.touches == 0 else "tested"


@dataclass(frozen=True)
class ZonesState:
    nearest_supply: SDZone | None = None
    nearest_demand: SDZone | None = None
    supply_dist_atr: float = float("nan")   # close to supply zone edge
    demand_dist_atr: float = float("nan")
    in_supply: bool = False
    in_demand: bool = False
    # full live zone list (Phase 4.5 — the AOC family clusters these).
    # Adding a field with a default keeps every existing constructor
    # working; the detector never reads it unless aoc is enabled.
    active: tuple = ()                      # tuple[SDZone, ...] live


# ------------------------------------------------------------------- phases
@dataclass(frozen=True)
class PhaseState:
    phase: str = "unknown"   # accumulation | distribution | consolidation
                             # | manipulation | expansion_up |
                             # expansion_down | trend_up | trend_down
    phase_age: int = 0
    range_high: float = float("nan")   # base range while in a base phase
    range_low: float = float("nan")
    event: str | None = None  # spring | upthrust | manipulation_ssl |
                              # manipulation_bsl | manipulation_fallback |
                              # phase_exit_up | phase_exit_down | base_start
    event_direction: int = 0  # trade direction the event sets up
    # Provenance (AMD fallback work): how the current phase was entered.
    via: str = ""             # "" | "direct" (witnessed) | "fallback"
                              # (BOS + sweep + FVG inferred)
    confidence: float = 0.0   # 0..1 prior for the phase label; direct
                              # manipulation 0.80, fallback 0.55, base
                              # 0.60, expansion/trend 0.70, unknown 0.0
    evidence: tuple = ()      # human-readable facts behind the entry
                              # event; () on continuation bars


# ------------------------------------------------------------------- volume
@dataclass(frozen=True)
class VolumeState:
    available: bool = False
    rvol: float = float("nan")         # volume / trailing-20 mean volume
    climax: bool = False               # rvol >= climax_rvol
    climax_reversal: int = 0           # +1 bullish climax reversal at lows,
                                       # -1 bearish at highs, 0 none
    dry_up: bool = False               # rvol <= dry_rvol
    vwap: float = float("nan")         # session (UTC day) VWAP, causal


# --------------------------------------------------------------- footprints
@dataclass(frozen=True)
class FootprintsState:
    tags: tuple = ()                   # tags firing on THIS bar
    evidence: tuple = ()               # human-readable evidence strings
    institutional_dir: int = 0         # bank-entry signature direction
    institutional_age: int | None = None
    cta_trend: int = 0                 # CTA-character trend state
    retail_chase: int = 0              # chase candle direction this bar
    retail_trap: int = 0               # trap direction (fade the chase)


# ---------------------------------------------------------------------- MTF
@dataclass(frozen=True)
class MTFState:
    h4_trend: str = "range"
    h4_bias: int = 0
    d1_trend: str = "range"
    d1_bias: int = 0
    h4_swing_high: float | None = None
    h4_swing_low: float | None = None
    d1_swing_high: float | None = None
    d1_swing_low: float | None = None
    alignment: float = 0.0             # mean of (d1, h4, h1) biases, -1..1

    def aligned_count(self, direction: int) -> int:
        return sum(1 for b in (self.d1_bias, self.h4_bias) if b == direction)

    def opposed_count(self, direction: int) -> int:
        return sum(1 for b in (self.d1_bias, self.h4_bias)
                   if b == -direction and b != 0)


# -------------------------------------------------------------- trendlines
@dataclass(frozen=True)
class Trendline:
    """One swing trendline and its lifecycle state at this bar.

    ``direction`` +1 = support line through HIGHER swing lows (rising),
    -1 = resistance through LOWER swing highs (falling).  The line's
    price at bar ``t`` is ``price_at(t)``; anchors are the two confirmed
    swings that defined it.  ``state``: FORMING (2 anchors, no retest
    yet) | ACTIVE (>= 1 retest touch — the "3rd contact validates"
    rule) | BROKEN (closed through on this bar, terminal) | EXPIRED
    (aged out / drifted away, terminal)."""

    direction: int
    anchor1_index: int
    anchor1_price: float
    anchor2_index: int
    anchor2_price: float
    slope: float               # price per bar
    created_at: int            # bar the line became known (2nd anchor's
                               # confirmation bar)
    touches: int = 0           # retest touches AFTER creation
    last_touch_at: int | None = None
    state: str = "FORMING"

    def price_at(self, t: int) -> float:
        return self.anchor1_price + self.slope * (t - self.anchor1_index)


@dataclass(frozen=True)
class TrendlineEvent:
    kind: str                  # "touch" | "bounce" | "break"
    direction: int             # the line's direction (+1 support)
    line_price: float          # line price at the event bar
    touches: int               # touch count after this event


@dataclass(frozen=True)
class TrendlinesState:
    active: tuple = ()                    # tuple[Trendline, ...] live
    events: tuple = ()                    # tuple[TrendlineEvent, ...]
                                          # fired on THIS bar


# ---------------------------------------------------- area of concentration
@dataclass(frozen=True)
class AOCMember:
    """One level inside an area of concentration."""
    family: str                # pool | sd_zone | ob | fvg | trendline |
                               # fib | daily | eq
    price: float               # the level's price at this bar
    weight: float              # family prior weight
    recency: float             # 0.5..1.0 age decay (see features/aoc.py)


@dataclass(frozen=True)
class AOCZone:
    """A cluster of levels from >= 2 independent families sitting
    within ~0.35 ATR of each other — the comment-section "area of
    concentration", a trader's pivot zone.

    ``families`` are DISTINCT family names (>= 2 for an AOC,
    >= 3 for a MAJOR pivot zone).  ``score`` = sum of
    weight x recency over members — concentration, not size."""
    center: float               # weight-weighted mean member price
    top: float                  # highest member price
    bottom: float               # lowest member price
    width: float                # top - bottom (price units)
    families: tuple             # distinct member families, sorted
    score: float                # concentration score
    members: tuple              # tuple[AOCMember, ...]


@dataclass(frozen=True)
class AOCState:
    zones: tuple = ()           # tuple[AOCZone, ...] top-8 by score
    above: AOCZone | None = None   # nearest AOC with center > close
    below: AOCZone | None = None   # nearest AOC with center < close
    above_dist_atr: float = float("nan")
    below_dist_atr: float = float("nan")

    def major_at(self, price: float, tol: float) -> AOCZone | None:
        """The MAJOR (>= 3 families) pivot zone containing ``price``
        within ``tol``, or None.  This is the patterns gate's
        "major level" test (Phase 4.5 repair)."""
        best: AOCZone | None = None
        best_score = -1.0
        for z in self.zones:
            if len(z.families) < 3:
                continue
            if z.bottom - tol <= price <= z.top + tol \
                    or abs(z.center - price) <= tol:
                if z.score > best_score:
                    best, best_score = z, z.score
        return best


# ------------------------------------------------------------------- bundle
@dataclass(frozen=True)
class FeatureSet:
    """All feature families for one bar."""

    index: int
    bollinger: BollingerState = field(default_factory=BollingerState)
    fib: FibState = field(default_factory=FibState)
    ichimoku: IchimokuState = field(default_factory=IchimokuState)
    relstr: RSState = field(default_factory=RSState)
    patterns: PatternsState = field(default_factory=PatternsState)
    zones: ZonesState = field(default_factory=ZonesState)
    phase: PhaseState = field(default_factory=PhaseState)
    volume: VolumeState = field(default_factory=VolumeState)
    footprints: FootprintsState = field(default_factory=FootprintsState)
    mtf: MTFState = field(default_factory=MTFState)
    trendlines: TrendlinesState = field(default_factory=TrendlinesState)
    aoc: AOCState = field(default_factory=AOCState)
    divergence: DivergenceState = field(default_factory=DivergenceState)


#: The family names the detector / ablation use.  Phase 3 scored the
#: first ten flat (context-free).  Phase 4 adds ``trendlines`` (an
#: event family, contextual by construction) and the ``*_ctx`` gated
#: re-designs of the Phase-3 diluters — see features/gating.py.
#: Phase 4.5 adds ``aoc`` (area of concentration — level clustering,
#: gated like the others; also feeds the repaired patterns major-level
#: test) — see features/aoc.py.  ``divergence`` (RSI divergence, mined
#: from practitioner video 2026-10-11) — see features/divergence.py.
FAMILIES = (
    "bollinger", "fib", "ichimoku", "relstr", "patterns", "zones",
    "phases", "footprints", "volume", "mtf",
    "trendlines",
    "fib_ctx", "zones_ctx", "ichimoku_ctx", "patterns_ctx",
    "footprints_ctx", "mtf_ctx",
    "aoc",
    "divergence",
)
