"""Regime-aware trading system.

The complete trader, not a pattern collection:
1. READ the regime (D1 bias + H1 stack + H1 trend)
2. ROUTE to the right playbooks (only what works in this regime)
3. VALIDATE entries with candles (the entry system)
4. MANAGE risk (existing harness)

A pattern without regime context is a coin flip. The regime tells you
which patterns to trade, which direction, and when to stand aside.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..reader.state import MarketState


class Regime(Enum):
    """Market regime from HTF + H1 alignment."""
    STRONG_BULL = "strong_bull"    # D1 bull + H1 bull stack: longs, aggressive
    BULL_LEAN = "bull_lean"        # leaning bull: longs, selective
    STRONG_BEAR = "strong_bear"    # D1 bear + H1 bear stack: shorts, aggressive
    BEAR_LEAN = "bear_lean"        # leaning bear: shorts, selective
    CHOP = "chop"                  # no edge: stand aside


@dataclass
class RegimeConfig:
    """Which playbooks are enabled per regime, and direction bias."""
    allowed_directions: set[int]   # {1} longs only, {-1} shorts only, {1,-1} both, {} none
    allowed_kinds: set[str] | None  # None = all, else whitelist
    min_score: float               # regime-adjusted threshold
    aggressive: bool               # position sizing modifier


# Regime → config: the routing table.
# Strong trends: trade with the trend, aggressive.
# Leans: trade with the lean, selective (higher threshold).
# Chop: stand aside.
REGIME_ROUTING: dict[Regime, RegimeConfig] = {
    Regime.STRONG_BULL: RegimeConfig(
        allowed_directions={1},
        allowed_kinds=None,  # all long playbooks
        min_score=60.0,
        aggressive=True,
    ),
    Regime.BULL_LEAN: RegimeConfig(
        allowed_directions={1},
        allowed_kinds={"sweep_reversal", "continuation", "sr_flip"},
        min_score=65.0,  # more selective
        aggressive=False,
    ),
    Regime.STRONG_BEAR: RegimeConfig(
        allowed_directions={-1},
        allowed_kinds=None,  # all short playbooks
        min_score=60.0,
        aggressive=True,
    ),
    Regime.BEAR_LEAN: RegimeConfig(
        allowed_directions={-1},
        allowed_kinds={"sweep_reversal", "continuation", "sr_flip"},
        min_score=65.0,
        aggressive=False,
    ),
    Regime.CHOP: RegimeConfig(
        allowed_directions=set(),  # stand aside
        allowed_kinds=set(),
        min_score=100.0,  # impossible
        aggressive=False,
    ),
}


def classify_regime(s: MarketState) -> Regime:
    """Classify the market regime from HTF bias and H1 structure.

    Strong trend needs alignment: D1 bias + H1 EMA stack + H1 trend
    all pointing the same way. Anything less is a lean or chop.
    """
    d1 = s.daily_bias  # 1 bull, -1 bear, 0 neutral
    stack = s.ema_stack  # 1 bull stack, -1 bear stack, 0 mixed
    trend = s.trend  # "up", "down", "range"

    # strong bull: everything aligned
    if d1 == 1 and stack == 1 and trend == "up":
        return Regime.STRONG_BULL
    # strong bear: everything aligned
    if d1 == -1 and stack == -1 and trend == "down":
        return Regime.STRONG_BEAR

    # bull lean: D1 bull + (stack or trend bullish)
    if d1 == 1 and (stack == 1 or trend == "up"):
        return Regime.BULL_LEAN
    # bear lean: D1 bear + (stack or trend bearish)
    if d1 == -1 and (stack == -1 or trend == "down"):
        return Regime.BEAR_LEAN

    # H1-only trends without D1 support are leans
    if stack == 1 and trend == "up":
        return Regime.BULL_LEAN
    if stack == -1 and trend == "down":
        return Regime.BEAR_LEAN

    return Regime.CHOP


def route_signal(s: MarketState, direction: int, kind: str,
                 score: float) -> tuple[bool, str]:
    """Check if a signal is allowed in the current regime.

    Returns (allowed, reason).
    """
    regime = classify_regime(s)
    cfg = REGIME_ROUTING[regime]

    if direction not in cfg.allowed_directions:
        return False, f"regime {regime.value} blocks direction {direction}"
    if cfg.allowed_kinds is not None and kind not in cfg.allowed_kinds:
        return False, f"regime {regime.value} blocks kind {kind}"
    if score < cfg.min_score:
        return False, f"score {score:.0f} < regime min {cfg.min_score:.0f}"

    return True, f"regime {regime.value} allows"


def hierarchical_filter(signals: list, states: list[MarketState],
                       min_score: float = 65.0) -> list:
    """Version 1 strict hierarchy as post-filter (proven +7.6%).

    D1 assigns direction: only signals matching daily_bias pass.
    H1 stack must align with direction.
    Score must meet threshold.

    This is the simple, proven implementation. The detector generates
    signals normally; this enforces the hierarchy on results.
    """
    out = []
    for sig in signals:
        s = states[sig.index]
        # D1 boss: assigns direction (0 = sit out)
        bias = s.daily_bias
        if bias == 0:
            continue
        # Signal must match D1 direction
        if sig.direction != bias:
            continue
        # H1 stack must align
        if s.ema_stack != bias:
            continue
        # Score threshold
        if sig.score < min_score:
            continue
        out.append(sig)
    return out
