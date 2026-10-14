"""GOD Conviction Engine — 26-factor confluence scoring.

Vectorized port of Section 18 of GOD-CE-SLIM.pine: buyScore/sellScore
(0–100), distinct-factor counts, and the dynamic threshold stack
(Standard risk profile, ASOC off).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import (
    GodceParams,
    REGIME_BREAKOUT,
    REGIME_CHOPPY,
    REGIME_REVERSAL,
    REGIME_TREND_BEAR,
    REGIME_TREND_BULL,
)


def compute_scores(feat: pd.DataFrame, P: GodceParams) -> pd.DataFrame:
    f = feat
    idx = f.index
    buy = pd.Series(0.0, index=idx)
    sell = pd.Series(0.0, index=idx)
    B = lambda c: c.fillna(False).to_numpy(dtype=bool)  # noqa: E731
    F = lambda c: c.fillna(0.0).to_numpy(dtype=float)  # noqa: E731

    # 1. macro
    mb, mbe, mv = B(f["macroBullish"]), B(f["macroBearish"]), B(f["macroValid"])
    buy += np.where(mv & mb, 20.0, np.where(mv & mbe, -25.0, 0.0))
    sell += np.where(mv & mbe, 20.0, np.where(mv & mb, -25.0, 0.0))
    # 2. trend stack & HMA
    buy += np.where(B(f["bullStack"]), 12.0, 0.0) + np.where(F(f["close"]) > F(f["hmaBase"]), 6.0, 0.0)
    sell += np.where(B(f["bearStack"]), 12.0, 0.0) + np.where(F(f["close"]) < F(f["hmaBase"]), 6.0, 0.0)
    # 3. elastic EMA
    buy += np.where(F(f["close"]) > F(f["emaElastic"]), 6.0, 0.0)
    sell += np.where(~(F(f["close"]) > F(f["emaElastic"])), 6.0, 0.0)
    # 4. multi-TF bias
    buy += np.where(B(f["tf3Bull"]), 8.0, 0.0) + np.where(B(f["tf2Bull"]), 4.0, 0.0) + np.where(B(f["tf1Bull"]), 3.0, 0.0)
    sell += np.where(B(f["tf3Bear"]), 8.0, 0.0) + np.where(B(f["tf2Bear"]), 4.0, 0.0) + np.where(B(f["tf1Bear"]), 3.0, 0.0)
    # 5. sweeps (symSweepW = 1.0 for gold)
    vbs, vss, vi = B(f["validBuySweep"]), B(f["validSellSweep"]), B(f["volImbalance"])
    buy += np.where(vbs & vi, 25.0, np.where(vbs, 18.0, 0.0))
    sell += np.where(vss & vi, 25.0, np.where(vss, 18.0, 0.0))
    # 6. OB gravity
    b_hit, b_core, b_grav = B(f["bObHit"]), B(f["bObCoreTap"]), F(f["bObZoneGravity"])
    r_hit, r_core, r_grav = B(f["rObHit"]), B(f["rObCoreTap"]), F(f["rObZoneGravity"])
    buy += np.where(b_hit | (b_grav > 0.4),
                    np.minimum(22.0, np.where(b_hit, np.where(b_core, 18.0, 12.0), 8.0) + b_grav * 10.0), 0.0)
    sell += np.where(r_hit | (r_grav > 0.4),
                     np.minimum(22.0, np.where(r_hit, np.where(r_core, 18.0, 12.0), 8.0) + r_grav * 10.0), 0.0)
    # 6b. beyond-FVG
    bfg = F(f["bfgState"])
    rfg = F(f["rfgState"])
    buy += np.where(bfg == 2, np.where(B(f["bfgFull"]), 15.0, 11.0), 0.0)
    sell += np.where(rfg == 2, np.where(B(f["rfgFull"]), 15.0, 11.0), 0.0)
    buy += np.where(B(f["baitLong"]), -8.0, 0.0)
    sell += np.where(B(f["baitShort"]), -8.0, 0.0)
    # 6c. trap fade
    bta, bte = B(f["bearTrapActive"]), B(f["bullTrapActive"])
    buy += np.where(bta, np.where(B(f["bearTrapWasEng"]), 15.0, 10.0), 0.0)
    sell += np.where(bte, np.where(B(f["bullTrapWasEng"]), 15.0, 10.0), 0.0)
    buy += np.where(B(f["ifvgDemTest"]), 6.0, 0.0)
    sell += np.where(B(f["ifvgSupTest"]), 6.0, 0.0)
    # 7. breakers
    rbs, sbr = B(f["rbsHit"]), B(f["sbrHit"])
    buy += np.where(rbs & B(f["inDiscount"]), 10.0, np.where(rbs, 6.0, 0.0))
    sell += np.where(sbr & B(f["inPremium"]), 10.0, np.where(sbr, 6.0, 0.0))
    # 8. premium/discount
    buy += np.where(B(f["inDiscount"]), 8.0, 0.0)
    sell += np.where(B(f["inPremium"]), 8.0, 0.0)
    # 9. ADX
    buy += np.where(B(f["adxOk"]), 6.0, 0.0)
    sell += np.where(B(f["adxOk"]), 6.0, 0.0)
    # 10. MACD
    buy += np.where(B(f["macdAbove"]), 4.0, 0.0) + np.where(B(f["macdCross"]) | B(f["bullDiv"]), 4.0, 0.0)
    sell += np.where(B(f["macdBelow"]), 4.0, 0.0) + np.where(B(f["macdDeadX"]) | B(f["bearDiv"]), 4.0, 0.0)
    # 11. RSI
    buy += np.where(B(f["rsiOS"]) & B(f["inDiscount"]), 8.0, 0.0) + np.where(B(f["rsiBullDiv"]), 4.0, 0.0)
    sell += np.where(B(f["rsiOB"]) & B(f["inPremium"]), 8.0, 0.0) + np.where(B(f["rsiBearDiv"]), 4.0, 0.0)
    # 12. dark pool proximity
    dpb, dps = F(f["dpBuyLevel"]), F(f["dpSellLevel"])
    buy += np.where(~np.isnan(dpb) & (np.abs(F(f["close"]) - dpb) < F(f["atr"]) * 0.6), 6.0, 0.0)
    sell += np.where(~np.isnan(dps) & (np.abs(F(f["close"]) - dps) < F(f["atr"]) * 0.6), 6.0, 0.0)
    # 13. structure
    buy += np.where(B(f["bullStruct"]), 8.0, 0.0)
    sell += np.where(B(f["bearStruct"]), 8.0, 0.0)
    # 14. VWAP
    buy += np.where(B(f["aboveVwap"]), 5.0, 0.0)
    sell += np.where(~B(f["aboveVwap"]), 5.0, 0.0)
    # 15. wyckoff / vol div
    buy += np.where(B(f["volBullDiv"]) & B(f["inDiscount"]), 4.0, 0.0) + np.where(B(f["wyckSpring"]), 4.0, 0.0)
    sell += np.where(B(f["volBearDiv"]) & B(f["inPremium"]), 4.0, 0.0) + np.where(B(f["wyckUTAD"]), 4.0, 0.0)
    # 16. z-score
    buy += np.where(F(f["zScore"]) <= -1.8, 10.0, 0.0)
    sell += np.where(F(f["zScore"]) >= 1.8, 10.0, 0.0)
    # 17. OHLC confluence (symOhlcW = 2.0 for gold → 10 pts)
    obuf = F(f["atr"]) * 0.4
    buy += np.where(~np.isnan(F(f["pdLow"])) & (np.abs(F(f["low"]) - F(f["pdLow"])) < obuf), 10.0, 0.0)
    sell += np.where(~np.isnan(F(f["pdHigh"])) & (np.abs(F(f["high"]) - F(f["pdHigh"])) < obuf), 10.0, 0.0)
    buy += np.where(~np.isnan(F(f["pwLow"])) & (np.abs(F(f["low"]) - F(f["pwLow"])) < obuf), 5.0, 0.0)
    sell += np.where(~np.isnan(F(f["pwHigh"])) & (np.abs(F(f["high"]) - F(f["pwHigh"])) < obuf), 5.0, 0.0)
    piv = ~np.isnan(F(f["dailyPivot"])) & (np.abs(F(f["close"]) - F(f["dailyPivot"])) < obuf)
    buy += np.where(piv, 6.0, 0.0)
    sell += np.where(piv, 6.0, 0.0)
    # 18. footprint
    buy += np.where(B(f["aggressiveBuy"]) & B(f["inDiscount"]), 5.0, 0.0)
    sell += np.where(B(f["aggressiveSell"]) & B(f["inPremium"]), 5.0, 0.0)
    buy += np.where(B(f["absorption"]) & B(f["atBuyZone"]), 3.0, 0.0)
    sell += np.where(B(f["absorption"]) & B(f["atSellZone"]), 3.0, 0.0)
    # 19. sentiment contrarian
    buy += np.where((F(f["sentiment"]) < -0.4) & B(f["atBuyZone"]), 6.0, 0.0)
    sell += np.where((F(f["sentiment"]) > 0.4) & B(f["atSellZone"]), 6.0, 0.0)
    # 20. candle at zone
    buy += np.where((B(f["bullPin"]) | B(f["bullEngulf"]) | B(f["ibBullBreak"])) & B(f["atBuyZone"]), 5.0, 0.0)
    sell += np.where((B(f["bearPin"]) | B(f["bearEngulf"]) | B(f["ibBearBreak"])) & B(f["atSellZone"]), 5.0, 0.0)
    # 21. EQ pools
    buy += np.where(B(f["nearEQL"]) & B(f["inDiscount"]), 4.0, 0.0)
    sell += np.where(B(f["nearEQH"]) & B(f["inPremium"]), 4.0, 0.0)
    # 22. weekly open
    wo = ~np.isnan(F(f["weekOpen"])) & (np.abs(F(f["close"]) - F(f["weekOpen"])) < F(f["atr"]) * 0.4)
    buy += np.where(wo, 3.0, 0.0)
    sell += np.where(wo, 3.0, 0.0)
    # 23. value band
    buy += np.where(B(f["belowValue"]) & B(f["inDiscount"]), 2.0, 0.0)
    sell += np.where(B(f["aboveValue"]) & B(f["inPremium"]), 2.0, 0.0)
    # 24. daily kumo
    dkt = F(f["dayKumoTop"])
    dkb = F(f["dayKumoBot"])
    buy += np.where(~np.isnan(dkt) & (F(f["close"]) > dkt), 3.0, 0.0)
    sell += np.where(~np.isnan(dkb) & (F(f["close"]) < dkb), 3.0, 0.0)
    buy += np.where(~np.isnan(dkt) & (np.abs(F(f["close"]) - dkt) <= F(f["atr"]) * 0.3) & (F(f["close"]) >= dkt), 2.0, 0.0)
    sell += np.where(~np.isnan(dkb) & (np.abs(F(f["close"]) - dkb) <= F(f["atr"]) * 0.3) & (F(f["close"]) <= dkb), 2.0, 0.0)
    # 25. TK/KJ cross
    buy += np.where(B(f["tkKjGold"]), 3.0, 0.0)
    sell += np.where(B(f["tkKjDeath"]), 3.0, 0.0)
    # 26. H&S breaks (pine: 8 pts only within 5 bars of the break)
    pos = np.arange(len(idx))
    hsT3 = (F(f["hsT_state"]) == 3) & ((pos - F(f["hsT_broke"])) <= 5)
    hsB3 = (F(f["hsB_state"]) == 3) & ((pos - F(f["hsB_broke"])) <= 5)
    sell += np.where(hsT3, 8.0, 0.0)
    buy += np.where(hsB3, 8.0, 0.0)
    sell += np.where(F(f["hsD_Tst"]) == 3, 6.0, 0.0)
    buy += np.where(F(f["hsD_Bst"]) == 3, 6.0, 0.0)

    buy = buy.clip(0.0, 100.0)
    sell = sell.clip(0.0, 100.0)

    # ── distinct factor counts ──
    bf = pd.Series(0, index=idx, dtype=int)
    sf = pd.Series(0, index=idx, dtype=int)

    def _bi(mask):
        m = mask if isinstance(mask, pd.Series) else pd.Series(np.asarray(mask), index=idx)
        return m.fillna(False).astype(bool).astype(int)

    def add_b(mask):
        nonlocal bf
        bf = bf + _bi(mask)

    def add_s(mask):
        nonlocal sf
        sf = sf + _bi(mask)

    add_b(mv & mb)
    add_s(mv & mbe)
    add_b(f["bullStack"])
    add_s(f["bearStack"])
    add_b(f["close"] > f["hmaBase"])
    add_s(f["close"] < f["hmaBase"])
    add_b(f["close"] > f["emaElastic"])
    add_s(f["close"] < f["emaElastic"])
    add_b(f["tf3Bull"])
    add_b(f["tf2Bull"])
    add_b(f["tf1Bull"])
    add_s(f["tf3Bear"])
    add_s(f["tf2Bear"])
    add_s(f["tf1Bear"])
    add_b(f["validBuySweep"])
    add_s(f["validSellSweep"])
    add_b(f["bObHit"] | (f["bObZoneGravity"] > 0.4))
    add_s(f["rObHit"] | (f["rObZoneGravity"] > 0.4))
    add_b(f["bFvgHit"] & ~f["baitLong"].fillna(False))
    add_s(f["rFvgHit"] & ~f["baitShort"].fillna(False))
    add_b(f["bearTrapActive"] | f["ifvgDemTest"] | (f["bfgState"] == 2))
    add_s(f["bullTrapActive"] | f["ifvgSupTest"] | (f["rfgState"] == 2))
    add_b(f["rbsHit"])
    add_s(f["sbrHit"])
    add_b(f["inDiscount"])
    add_s(f["inPremium"])
    add_b(f["adxOk"])
    add_s(f["adxOk"])
    add_b(f["macdAbove"])
    add_s(f["macdBelow"])
    add_b(f["macdCross"] | f["bullDiv"])
    add_s(f["macdDeadX"] | f["bearDiv"])
    add_b(f["rsiOS"] & f["inDiscount"])
    add_s(f["rsiOB"] & f["inPremium"])
    add_b(f["rsiBullDiv"])
    add_s(f["rsiBearDiv"])
    add_b(~f["dpBuyLevel"].isna() & (np.abs(f["close"] - f["dpBuyLevel"]) < f["atr"] * 0.6))
    add_s(~f["dpSellLevel"].isna() & (np.abs(f["close"] - f["dpSellLevel"]) < f["atr"] * 0.6))
    add_b(f["bullStruct"])
    add_s(f["bearStruct"])
    add_b(f["aboveVwap"])
    add_s(~f["aboveVwap"].fillna(True))
    add_b((f["volBullDiv"] & f["inDiscount"]) | f["wyckSpring"])
    add_s((f["volBearDiv"] & f["inPremium"]) | f["wyckUTAD"])
    add_b(f["zScore"] <= -1.8)
    add_s(f["zScore"] >= 1.8)
    add_b((~f["pdLow"].isna() & (np.abs(f["low"] - f["pdLow"]) < obuf))
          | (~f["pwLow"].isna() & (np.abs(f["low"] - f["pwLow"]) < obuf))
          | (~f["dailyPivot"].isna() & (np.abs(f["close"] - f["dailyPivot"]) < obuf)))
    add_s((~f["pdHigh"].isna() & (np.abs(f["high"] - f["pdHigh"]) < obuf))
          | (~f["pwHigh"].isna() & (np.abs(f["high"] - f["pwHigh"]) < obuf))
          | (~f["dailyPivot"].isna() & (np.abs(f["close"] - f["dailyPivot"]) < obuf)))
    add_b(((f["aggressiveBuy"] & f["inDiscount"]) | (f["absorption"] & f["atBuyZone"])))
    add_s(((f["aggressiveSell"] & f["inPremium"]) | (f["absorption"] & f["atSellZone"])))
    add_b((f["sentiment"] < -0.4) & f["atBuyZone"])
    add_s((f["sentiment"] > 0.4) & f["atSellZone"])
    add_b(((f["bullPin"] | f["bullEngulf"] | f["ibBullBreak"]) & f["atBuyZone"]))
    add_s(((f["bearPin"] | f["bearEngulf"] | f["ibBearBreak"]) & f["atSellZone"]))
    add_b(f["nearEQL"] & f["inDiscount"])
    add_s(f["nearEQH"] & f["inPremium"])
    add_b(wo)
    add_s(wo)
    add_b(f["belowValue"] & f["inDiscount"])
    add_s(f["aboveValue"] & f["inPremium"])
    add_b(~f["dayKumoTop"].isna() & (f["close"] > f["dayKumoTop"]))
    add_s(~f["dayKumoBot"].isna() & (f["close"] < f["dayKumoBot"]))
    add_b(f["tkKjGold"])
    add_s(f["tkKjDeath"])
    add_b(hsB3)
    add_s(hsT3)
    add_b(f["hsD_Bst"] == 3)
    add_s(f["hsD_Tst"] == 3)

    # ── thresholds ──
    regime = f["regime"].fillna(0).to_numpy(dtype=int)
    adj = np.select(
        [(regime == REGIME_TREND_BULL) | (regime == REGIME_TREND_BEAR),
         regime == REGIME_CHOPPY,
         regime == REGIME_BREAKOUT,
         regime == REGIME_REVERSAL],
        [-5.0, 8.0, -3.0, 2.0], default=0.0)
    dyn = np.clip(P.baseThresh + adj, 55.0, 92.0)
    bias = f["biasScore"].fillna(0).to_numpy()
    buy_th = dyn.copy()
    sell_th = dyn.copy()
    buy_th = np.where(bias == -3, np.minimum(buy_th + 8.0, 95.0), buy_th)
    sell_th = np.where(bias == 3, np.minimum(sell_th + 8.0, 95.0), sell_th)
    choppy = (regime == REGIME_CHOPPY) | B(f["lowVol"])
    buy_th = np.where(choppy, np.minimum(buy_th + 5.0, 95.0), buy_th)
    sell_th = np.where(choppy, np.minimum(sell_th + 5.0, 95.0), sell_th)
    volat = B(f["regimeVolatile"])
    buy_th = np.where(volat, np.maximum(buy_th - 5.0, 50.0), buy_th)
    sell_th = np.where(volat, np.maximum(sell_th - 5.0, 50.0), sell_th)

    return pd.DataFrame({
        "buyScore": buy, "sellScore": sell,
        "buyFactors": bf, "sellFactors": sf,
        "buyThresh": buy_th, "sellThresh": sell_th,
        "dynThreshold": dyn,
    }, index=idx)
