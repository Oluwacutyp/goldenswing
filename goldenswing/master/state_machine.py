"""Master State Machine — the unified execution pipeline.

Four tiers, strictly hierarchical. Upstream tiers CONSTRAIN downstream ones;
they don't vote. A high Tier-3 score in a Tier-1-forbidden direction is not
a trade — it's a setup that never gets evaluated.

  Tier 1: Directional Regime Gate ......... D1 trend (200 EMA). Bullish,
      bearish, or neutral. Neutral = FLAT, nothing downstream runs.
      (Empirical: D1-aligned FVGs respect 86.4% vs 81.6% counter-trend.
      Volatility-expansion veto NOT supported by data — expansion is rare
      and doesn't degrade zone respect. Tier 1 is directional, not volatility.)
  Tier 2: Structural Coordinate Map ........ 4H FVG zones (streaming tracker)
      + D1 structural levels. Only zones in the Tier-1 direction survive.
      (Empirical: 4H FVGs respect ~85% standalone; Daily locational alignment
      adds nothing — 83.8% vs 85.3%. No alignment bonus.)
  Tier 3: Confluence & Scoring ............. Independence-aware 4-pillar
      scoring (structure, fib, VP, FVG). Same-bar merges, within-category
      dedup, proximity gate. Score -> risk tier mapping.
  Tier 4: Deterministic Risk Manager ........ 0.75-1.0 ATR stop buffers (HTF),
      3% portfolio heat cap, limit-at-touch entry, close-through invalidation.

Data flow is one direction: T1 -> T2 -> T3 -> T4. No feedback, no voting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Regime(Enum):
    BULLISH = "bullish"      # D1 above 200 EMA: longs only
    BEARISH = "bearish"      # D1 below 200 EMA: shorts only
    NEUTRAL = "neutral"      # no D1 conviction: FLAT


class Decision(Enum):
    FLAT = "flat"            # no trade
    LONG = "long"
    SHORT = "short"


@dataclass
class Tier1Output:
    regime: Regime
    reason: str


@dataclass
class Tier2Output:
    zones: list              # FVGZone objects in the regime direction
    regime: Regime


@dataclass
class Tier3Output:
    zone: object             # the selected FVGZone
    score: int               # independent confluence factors
    factors: list[str]
    direction: int


@dataclass
class Tier4Output:
    decision: Decision
    entry: float             # limit price
    stop: float
    risk_pct: float
    invalidation: str
    reason: str


# Risk tier mapping: confluence score -> risk percentage.
# Kelly-grounded: H4 backtest (n=226, p=0.398, b=1.85) gives full Kelly 7.3%
# — mathematically optimal, practically insane (50%+ drawdowns routine).
# Quarter-Kelly = 1.8%. Our tiers sit at/below quarter-Kelly: score 3-4
# (1.5-2.0%) ≈ quarter-Kelly, scores 1-2 are more conservative.
# Kelly inputs MUST come from the Tier-1-filtered population, not the raw
# book — the directional gate changes p and b. Recompute per segment when
# n supports it; never hand-tune.
RISK_BY_SCORE = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0}

#: Stop buffer in ATR multiples for HTF (empirical: 0.75-1.0 optimal).
STOP_BUFFER_ATR = 0.75


class MasterStateMachine:
    """The unified pipeline. One method, four tiers, no voting."""

    def __init__(self, tracker, max_heat_pct: float = 3.0):
        self.tracker = tracker          # FVGTracker (Tier 2)
        self.max_heat_pct = max_heat_pct
        self.current_heat_pct = 0.0

    # -- Tier 1 ---------------------------------------------------------
    def tier1_regime(self, d1_close: float, d1_ema200: float,
                     d1_bar_time=None, now=None) -> Tier1Output:
        """Directional gate. The boss. Everything downstream obeys.

        Defense-in-depth: if bar timestamps are supplied, a stale D1 feed
        forces NEUTRAL (no permission) — fail closed on stale data. The
        pipeline's data-health gate is the primary defense; this catches any
        caller that bypasses it.
        """
        if d1_bar_time is not None:
            from datetime import datetime, timezone, timedelta
            now = now or datetime.now(timezone.utc)
            if now - d1_bar_time > timedelta(hours=30):
                return Tier1Output(Regime.NEUTRAL, "stale D1 feed: no permission")
        if d1_ema200 == 0:
            return Tier1Output(Regime.NEUTRAL, "no D1 EMA200")
        if d1_close > d1_ema200:
            return Tier1Output(Regime.BULLISH, "D1 above 200 EMA")
        if d1_close < d1_ema200:
            return Tier1Output(Regime.BEARISH, "D1 below 200 EMA")
        return Tier1Output(Regime.NEUTRAL, "D1 at EMA200")

    # -- Tier 2 ---------------------------------------------------------
    def tier2_zones(self, t1: Tier1Output) -> Tier2Output:
        """Structural map filtered by regime direction. Counter-trend zones
        are discarded here — they never reach scoring."""
        if t1.regime == Regime.NEUTRAL:
            return Tier2Output([], t1.regime)
        want = 1 if t1.regime == Regime.BULLISH else -1
        zones = self.tracker.active(direction=want)
        return Tier2Output(zones, t1.regime)

    # -- Tier 3 ---------------------------------------------------------
    def tier3_score(self, t2: Tier2Output, score_fn) -> Tier3Output | None:
        """Score each surviving zone; pick the best. score_fn is the
        confluence engine (injected — Tier 3 doesn't own data)."""
        best = None
        for z in t2.zones:
            s = score_fn(z)
            if best is None or s.score > best.score:
                best = s
        return best

    # -- Tier 4 ---------------------------------------------------------
    # POSITION LIFECYCLE CONTRACT (design invariant — do not "improve"):
    #   1. Sizing is entry-time immutable. The risk multiplier (1.0 / 0.5 / 0.0)
    #      prices NEW entries only. A position opened at 0.5% during probation
    #      stays 0.5% after graduation to NOMINAL; a full-size position is not
    #      trimmed when entering DEGRADED. Mid-trade resizing would corrupt
    #      R-multiple accounting, override the trade's own stop plan, and
    #      create a feedback loop (graduation -> add size -> inflated metrics).
    #   2. HALT stops new entries; it never liquidates. Open positions run
    #      their original stops/targets. Forced liquidation would realize
    #      losses at the worst moment and replace trade-level risk control
    #      (the stop) with portfolio-level panic.
    #   3. The heat cap is absolute (3%) and prospective: it blocks new
    #      entries that would breach it. State transitions never forcibly
    #      reduce current_heat_pct; over-cap heat decays naturally as
    #      positions close.
    # In short: the risk multiplier is an ENTRY GATE, not a portfolio
    # thermostat.
    def tier4_execute(self, t3: Tier3Output | None, atr: float,
                      current_price: float) -> Tier4Output:
        """Deterministic risk: score -> size, ATR -> stop, heat -> veto."""
        if t3 is None or t3.score < 1:
            return Tier4Output(Decision.FLAT, 0, 0, 0, "",
                               "no qualifying zone")
        risk_pct = RISK_BY_SCORE.get(t3.score, 0.0)
        if risk_pct <= 0:
            return Tier4Output(Decision.FLAT, 0, 0, 0, "",
                               f"score {t3.score} below risk threshold")
        if self.current_heat_pct + risk_pct > self.max_heat_pct:
            return Tier4Output(Decision.FLAT, 0, 0, 0, "",
                               "portfolio heat cap")

        z = t3.zone
        buf = STOP_BUFFER_ATR * atr
        if t3.direction == 1:
            entry = z.top            # limit at near edge
            stop = z.bottom - buf
            decision = Decision.LONG
            invalidation = f"4H close below {z.bottom:.2f}"
        else:
            entry = z.bottom
            stop = z.top + buf
            decision = Decision.SHORT
            invalidation = f"4H close above {z.top:.2f}"
        return Tier4Output(decision, entry, stop, risk_pct, invalidation,
                           f"score {t3.score}: {', '.join(t3.factors)}")

    # -- Full pipeline --------------------------------------------------
    def evaluate(self, d1_close: float, d1_ema200: float, atr: float,
                 current_price: float, score_fn) -> Tier4Output:
        """One call: T1 gates T2 gates T3 gates T4. Returns the decision."""
        t1 = self.tier1_regime(d1_close, d1_ema200)
        if t1.regime == Regime.NEUTRAL:
            return Tier4Output(Decision.FLAT, 0, 0, 0, "",
                               f"Tier 1 veto: {t1.reason}")
        t2 = self.tier2_zones(t1)
        t3 = self.tier3_score(t2, score_fn)
        return self.tier4_execute(t3, atr, current_price)
