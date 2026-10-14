"""GOD Conviction Engine — strategy router.

Port of the pine's STRATEGY ROUTER block: candidate-mode computation
(vectorized), 2-bar dwell state machine, per-mode playbook rules
(f_modeRules) and per-mode setup agreement (f_modeSignal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import (
    M_BRK,
    M_FLAT,
    M_LEG,
    M_MEAN,
    M_MOM,
    M_POST,
    GodceParams,
)


def compute_candidate_mode(feat: pd.DataFrame, P: GodceParams) -> pd.Series:
    """Vectorized candMode per the pine's if/else-if chain."""
    f = feat
    B = lambda c: c.fillna(False)  # noqa: E731
    base_ok = B(f["volOk"]) & B(f["sessionOk"]) & ~B(f["cautionMode"])
    # note: canTrade == not cautionMode for Standard profile
    squeeze_adx = B(f["squeeze30"]) & B(f["adxUp"])
    trending = f["regime"].isin([1, 2])  # TRENDING BULL/BEAR codes
    conds = [
        ~base_ok,
        B(f["atrRatio"]) > P.spikeAtrMult,
        B(f["vBotTrig"]) | B(f["vTopTrig"]) | B(f["vBotBreak"]) | B(f["vTopBreak"]),
        B(f["patBreakLong"]) | B(f["patBreakShort"]),
        B(f["asiaBreakLong"]) | B(f["asiaBreakShort"]),
        squeeze_adx,  # f_modeViable(M_BRK) == True with ASOC off
        trending & B(f["adxOk"]),
    ]
    choices = [M_FLAT, M_POST, M_POST, M_BRK, M_BRK, M_BRK, M_MOM]
    return pd.Series(np.select(conds, choices, default=M_MEAN), index=f.index)


def apply_dwell(cand: pd.Series, P: GodceParams) -> pd.Series:
    """2-bar dwell: candidate must persist modeDwell confirmed bars."""
    n = len(cand)
    cv = cand.to_numpy(dtype=int)
    mode = np.zeros(n, dtype=int)
    m = 0
    mcand = 0
    hold = 0
    for i in range(n):
        hold = hold + 1 if cv[i] == mcand else 1
        if cv[i] != mcand:
            mcand = cv[i]
        if hold >= P.modeDwell and mcand != m:
            m = mcand
        mode[i] = m
    return pd.Series(mode, index=cand.index)


def mode_rules(mode: int, t1: float, t2: float, regime_bull: bool, regime_bear: bool):
    """f_modeRules: [allowLong, allowShort, allowCounter, tp1R, tp2R, tp3R,
    slCapAtr, runnerAtr, threshOff]."""
    aL = aS = ctr = True
    f1, f2, f3, cap, run, off = t1, t2, t2 + 1.5, 2.0, 1.2, 0.0
    if mode == M_MOM:
        ctr = False
        aL = regime_bull
        aS = regime_bear
    elif mode == M_MEAN:
        f1, f2, f3, cap, run, off = min(f1, 0.75), min(f2, 1.5), min(f2, 1.5) + 1.0, 1.5, 0.8, -5.0
    elif mode == M_POST:
        f1, f2, f3, cap, run, off = min(f1, 0.5), min(f2, 1.0), min(f2, 1.0) + 0.75, 1.25, 1.0, -5.0
    elif mode == M_BRK:
        f1, f2, f3, cap, run, off = min(f1, 1.0), min(f2, 2.0), min(f2, 2.0) + 1.0, 1.5, 1.0, -10.0
    return aL, aS, ctr, f1, f2, f3, cap, run, off



def compute_mode_signals(feat: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per-mode setup agreement (f_modeSignal for all modes)."""
    f = feat
    B = lambda c: c.fillna(False)  # noqa: E731
    atr = f["atr"]
    out = pd.DataFrame(index=f.index)
    val = f["vwapVal"]  # intraday
    low, high = f["low"], f["high"]
    out["momLongOk"] = ((low.shift(1) <= val + atr * 0.75)
                        | (low.shift(2) <= val + atr * 0.75)
                        | (low.shift(3) <= val + atr * 0.75)).fillna(False)
    out["momShortOk"] = ((high.shift(1) >= val - atr * 0.75)
                         | (high.shift(2) >= val - atr * 0.75)
                         | (high.shift(3) >= val - atr * 0.75)).fillna(False)
    out["meanLongOk"] = (B(f["inDiscount"]) & (B(f["validBuySweep"]) | B(f["atBuyZone"]) | B(f["rsiOS"]))).fillna(False)
    out["meanShortOk"] = (B(f["inPremium"]) & (B(f["validSellSweep"]) | B(f["atSellZone"]) | B(f["rsiOB"]))).fillna(False)
    out["postLongOk"] = (B(f["vBotTrig"]) | B(f["vBotBreak"])).fillna(False)
    out["postShortOk"] = (B(f["vTopTrig"]) | B(f["vTopBreak"])).fillna(False)
    hh30 = f["high"].rolling(30).max().shift(1)
    ll30 = f["low"].rolling(30).min().shift(1)
    out["brkLongOk"] = (((f["close"] > hh30) & (f["volume"] > f["volSma"])) | B(f["patBreakLong"]) | B(f["asiaBreakLong"])).fillna(False)
    out["brkShortOk"] = (((f["close"] < ll30) & (f["volume"] > f["volSma"])) | B(f["patBreakShort"]) | B(f["asiaBreakShort"])).fillna(False)
    return out


def mode_signal_for(mode: int, sig: pd.DataFrame, i: int) -> tuple[bool, bool]:
    """f_modeSignal for one bar, selecting the precomputed agreement row."""
    if mode == M_MOM:
        return bool(sig["momLongOk"].iat[i]), bool(sig["momShortOk"].iat[i])
    if mode == M_MEAN:
        return bool(sig["meanLongOk"].iat[i]), bool(sig["meanShortOk"].iat[i])
    if mode == M_POST:
        return bool(sig["postLongOk"].iat[i]), bool(sig["postShortOk"].iat[i])
    if mode == M_BRK:
        return bool(sig["brkLongOk"].iat[i]), bool(sig["brkShortOk"].iat[i])
    return True, True
