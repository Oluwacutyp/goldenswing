"""GOD Conviction Engine — parameter set.

Faithful port of the Pine inputs from GOD-CE-SLIM.pine (itself logic-identical
to the full GOD Conviction Engine). Standard risk profile. ASOC self-optimization
is deliberately OFF by default (in-sample adaptive tuning gets its own experiment).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GodceParams:
    # ── Core engine ──
    fastLen: int = 9
    medLen: int = 21
    slowLen: int = 50
    trendLen: int = 200
    hmaLen: int = 50
    atrLen: int = 14
    adxLen: int = 14
    adxMin: int = 18
    zLen: int = 20
    volLen: int = 20
    useVolImb: bool = True
    useVolConf: bool = False
    useKillzone: bool = False
    # ── Pattern ──
    useIchimoku: bool = True
    useHs: bool = True
    hsAtrTol: int = 2
    # ── Macro ──
    useMacro: bool = True
    # ── Elastic / regime ──
    useElastic: bool = True
    minElastic: int = 5
    maxElastic: int = 55
    regTrendTh: float = 0.20
    regimeLen: int = 20
    # ── Liquidity / sessions ──
    swingLen: int = 10
    eqTolerance: float = 0.30
    useWeekOpen: bool = True
    # ── SMC zones ──
    maxOBs: int = 5
    maxFVGs: int = 5
    fvgMinPct: int = 20
    zoneMaxAge: int = 120
    eqTol: float = 0.30  # EQH/EQL ATR tolerance
    qGravity: float = 0.50
    qCorePct: float = 0.35
    useLiqHier: bool = True
    maxLiqTier: int = 4
    # ── Oscillators ──
    macdFast: int = 12
    macdSlow: int = 26
    macdSig: int = 9
    rsiLen: int = 14
    rsiOBLvl: int = 70
    rsiOSLvl: int = 30
    # ── Wyckoff / dark pool / ASOC ──
    useWyckoff: bool = True
    useVolDiv: bool = True
    useDarkPool: bool = True
    useASOC: bool = False          # deliberately OFF in v1 (own experiment later)
    asocWindow: int = 20
    optStep: float = 1.0
    useFactorGate: bool = True
    minFactors: int = 4
    # ── Risk & Hydra ──
    baseThresh: float = 70.0       # Standard profile
    slMult: float = 1.0
    trailATR: float = 1.2
    tp1RR: float = 1.5
    tp2RR: float = 3.0
    usePVS: bool = True
    accBalance: float = 10000.0
    riskPct: float = 1.0
    tickValue: float = 1.0
    useHydra: bool = True
    moveBEat50: bool = True
    useProfitLock: bool = True
    beOffsetR: float = 0.10
    useSpikeShield: bool = True
    useCircuitBreaker: bool = True
    maxDailyLossR: float = 3.0
    maxTradesDay: int = 4
    # ── Router ──
    useRouter: bool = True
    modeDwell: int = 2
    spikeAtrMult: float = 2.2
    # ── Trap intelligence ──
    useTrapEngine: bool = True
    useBeyondFvg: bool = True
    useInvFvg: bool = True
    blockSusBreak: bool = True
    trapMemoryBars: int = 12
    brkLookback: int = 6
    trapVolMult: float = 1.15
    maxInvFvgs: int = 4
    costBps: float = 4.0
    useCostGate: bool = True
    minNetRR: float = 0.6
    # ── Backtest-only knobs (not pine inputs) ──
    warmupBars: int = 600          # no trading before this many bars
    fillAtClose: bool = True       # pine fills at confirmed bar close


DEFAULT = GodceParams()

# Regime codes (pine regimeStr → int)
REGIME_RANGING = 0
REGIME_TREND_BULL = 1
REGIME_TREND_BEAR = 2
REGIME_VOLATILE = 3
REGIME_CHOPPY = 4
REGIME_BREAKOUT = 5
REGIME_REVERSAL = 6

REGIME_NAMES = {
    REGIME_RANGING: "RANGING",
    REGIME_TREND_BULL: "TRENDING BULL",
    REGIME_TREND_BEAR: "TRENDING BEAR",
    REGIME_VOLATILE: "VOLATILE",
    REGIME_CHOPPY: "CHOPPY",
    REGIME_BREAKOUT: "BREAKOUT",
    REGIME_REVERSAL: "REVERSAL",
}

# Router mode codes (pine M_* constants)
M_FLAT = 0
M_MOM = 1
M_MEAN = 2
M_POST = 3
M_BRK = 4
M_LEG = 5

MODE_NAMES = {
    M_FLAT: "FLAT",
    M_MOM: "MOMENTUM",
    M_MEAN: "MEAN-REV",
    M_POST: "POST-SPIKE",
    M_BRK: "BREAKOUT",
    M_LEG: "LEGACY",
}
