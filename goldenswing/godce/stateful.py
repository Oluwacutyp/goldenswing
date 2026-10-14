"""GOD Conviction Engine — stateful bar-loop components.

Ports the Pine ``var``-based state machines: market-structure tracker,
Head & Shoulders detectors, the SMC zone engine (OB/FVG/breakers/iFVG),
and the trap-intelligence layer (SFP, fake breakouts, beyond-FVG).
All loops are O(n) over numpy arrays.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .params import GodceParams

_NAN = np.nan


def structure_features(
    h: pd.Series, l: pd.Series, c: pd.Series, o: pd.Series,
    ph: pd.Series, pl: pd.Series, atr: pd.Series,
    eq_tol: float, asian_high: pd.Series, asian_low: pd.Series,
) -> pd.DataFrame:
    """Market-structure tracker: last pivots, BOS/CHoCH, swing labels,
    EQH/EQL pools, liquidity sweeps."""
    n = len(h)
    H, L, C, O = h.to_numpy(), l.to_numpy(), c.to_numpy(), o.to_numpy()
    PH, PL, ATR = ph.to_numpy(), pl.to_numpy(), atr.to_numpy()
    AH, AL = asian_high.to_numpy(), asian_low.to_numpy()

    cols_bool = ["bullBOS", "bearBOS", "bullCHoCH", "bearCHoCH", "isHH", "isHL",
                 "isLH", "isLL", "bullStruct", "bearStruct", "isEQL", "isEQH",
                 "nearEQL", "nearEQH", "bslSweep", "sslSweep",
                 "validBuySweep", "validSellSweep"]
    cols_f = ["lastPH", "lastPL", "lastPH2", "lastPL2", "eqlLevel", "eqhLevel"]
    out_b = {k: np.zeros(n, dtype=bool) for k in cols_bool}
    out_f = {k: np.full(n, _NAN) for k in cols_f}

    lastPH = lastPH2 = lastPL = lastPL2 = _NAN
    struct_state = 0
    for i in range(n):
        if not np.isnan(PH[i]):
            lastPH2, lastPH = lastPH, PH[i]
        if not np.isnan(PL[i]):
            lastPL2, lastPL = lastPL, PL[i]
        a = ATR[i]
        a_ok = not np.isnan(a) and a > 0
        bull_bos = (not np.isnan(lastPL2) and not np.isnan(lastPH)
                    and C[i] > lastPH and a_ok and (C[i] - O[i]) > a * 0.3)
        bear_bos = (not np.isnan(lastPH2) and not np.isnan(lastPL)
                    and C[i] < lastPL and a_ok and (O[i] - C[i]) > a * 0.3)
        bull_choch = (not np.isnan(lastPL2) and not np.isnan(lastPH2)
                      and C[i] > lastPH2 and a_ok and (C[i] - O[i]) > a * 0.3)
        bear_choch = (not np.isnan(lastPH2) and not np.isnan(lastPL2)
                      and C[i] < lastPL2 and a_ok and (O[i] - C[i]) > a * 0.3)
        if bull_bos or bull_choch:
            struct_state = 1
        if bear_bos or bear_choch:
            struct_state = -1
        is_hh = not np.isnan(lastPH) and not np.isnan(lastPH2) and lastPH > lastPH2
        is_hl = not np.isnan(lastPL) and not np.isnan(lastPL2) and lastPL > lastPL2
        is_lh = not np.isnan(lastPH) and not np.isnan(lastPH2) and lastPH < lastPH2
        is_ll = not np.isnan(lastPL) and not np.isnan(lastPL2) and lastPL < lastPL2
        is_eql = (not np.isnan(lastPL) and not np.isnan(lastPL2) and a_ok
                  and abs(lastPL - lastPL2) <= a * eq_tol)
        is_eqh = (not np.isnan(lastPH) and not np.isnan(lastPH2) and a_ok
                  and abs(lastPH - lastPH2) <= a * eq_tol)
        eql = (lastPL + lastPL2) / 2 if is_eql else _NAN
        eqh = (lastPH + lastPH2) / 2 if is_eqh else _NAN
        bsl = (not np.isnan(lastPH) and H[i] > lastPH and C[i] < lastPH and C[i] < O[i])
        ssl = (not np.isnan(lastPL) and L[i] < lastPL and C[i] > lastPL and C[i] > O[i])
        sah = (not np.isnan(AH[i]) and H[i] > AH[i] and C[i] < AH[i] and C[i] < O[i])
        sal = (not np.isnan(AL[i]) and L[i] < AL[i] and C[i] > AL[i] and C[i] > O[i])

        out_b["bullBOS"][i] = bull_bos
        out_b["bearBOS"][i] = bear_bos
        out_b["bullCHoCH"][i] = bull_choch
        out_b["bearCHoCH"][i] = bear_choch
        out_b["isHH"][i] = is_hh
        out_b["isHL"][i] = is_hl
        out_b["isLH"][i] = is_lh
        out_b["isLL"][i] = is_ll
        out_b["bullStruct"][i] = struct_state == 1
        out_b["bearStruct"][i] = struct_state == -1
        out_b["isEQL"][i] = is_eql
        out_b["isEQH"][i] = is_eqh
        out_b["nearEQL"][i] = is_eql and a_ok and abs(C[i] - eql) <= a * 0.5
        out_b["nearEQH"][i] = is_eqh and a_ok and abs(C[i] - eqh) <= a * 0.5
        out_b["bslSweep"][i] = bsl
        out_b["sslSweep"][i] = ssl
        out_b["validBuySweep"][i] = ssl or sal
        out_b["validSellSweep"][i] = bsl or sah
        out_f["lastPH"][i] = lastPH
        out_f["lastPL"][i] = lastPL
        out_f["lastPH2"][i] = lastPH2
        out_f["lastPL2"][i] = lastPL2
        out_f["eqlLevel"][i] = eql
        out_f["eqhLevel"][i] = eqh

    data = {**{k: v for k, v in out_b.items()}, **{k: v for k, v in out_f.items()}}
    return pd.DataFrame(data, index=h.index)


def asian_range(h: pd.Series, l: pd.Series, in_asian: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Progressive Asian-range high/low (00–07 UTC), held after the session."""
    day = pd.Series(h.index.date, index=h.index)
    ah = h.where(in_asian).groupby(day).cummax().ffill()
    al = l.where(in_asian).groupby(day).cummin().ffill()
    return ah, al


# ── Head & Shoulders ─────────────────────────────────────────────────

def _hs_detect(h: np.ndarray, l: np.ndarray, c: np.ndarray,
               swing_len: int, hs_atr_tol: float, atr: np.ndarray,
               top: bool) -> dict[str, np.ndarray]:
    """Pivot-confirmed H&S state machine. ``top=True`` → bearish H&S top,
    ``top=False`` → bullish inverse H&S. Returns per-bar state, live
    neckline value and break-bar index. Faithful port of f_hs_top/f_hs_bottom."""
    n = len(c)
    state = np.zeros(n, dtype=int)
    nl = np.full(n, _NAN)
    broke = np.full(n, -1, dtype=int)

    # pivot series (confirmed, positioned at confirmation bar)
    w = 2 * swing_len + 1
    px = h if top else l
    s = pd.Series(px)
    if top:
        is_p = (s == s.rolling(w, center=True, min_periods=w).max()).to_numpy()
    else:
        is_p = (s == s.rolling(w, center=True, min_periods=w).min()).to_numpy()

    phV: deque = deque(maxlen=4)  # (value, bar_pos)
    st = 0
    LS = HEAD = RS = _NAN
    LSb = HEADb = RSb = 0
    NL1 = _NAN
    NL1b = 0
    NL2 = _NAN
    NL2b = 0
    brokeBar = -1

    for i in range(n):
        a = atr[i]
        a_ok = not np.isnan(a) and a > 0
        # register newly confirmed pivot (pivot bar = i - swing_len)
        if is_p[i] and not np.isnan(px[i]):
            phV.appendleft((float(px[i]), i - swing_len))

        # invalidation & expiry
        reset = False
        if st == 1:
            if (is_p[i] and top and px[i] > HEAD) or (is_p[i] and not top and px[i] < HEAD):
                reset = True
            elif i - HEADb > 150:
                reset = True
        if st == 2:
            if (top and c[i] > HEAD) or (not top and c[i] < HEAD):
                reset = True
            elif i - RSb > 120:
                reset = True
        if st == 3 and i - brokeBar > 10:
            reset = True
        if reset:
            st = 0
            LS = HEAD = RS = NL1 = NL2 = _NAN
            brokeBar = -1

        # 0 → 1 : head forms beyond prior swing (left shoulder)
        if st == 0 and len(phV) >= 2 and a_ok:
            headC, headB = phV[0]
            lsC, lsB = phV[1]
            if (top and headC > lsC + a * 0.5) or (not top and headC < lsC - a * 0.5):
                st = 1
                LS, LSb = lsC, lsB
                HEAD, HEADb = headC, headB
                kA = i - HEADb
                kB = i - LSb
                if top:
                    best, bestB = l[HEADb], HEADb
                    for k in range(kA + 1, kB):
                        if 0 <= i - k < n and l[i - k] < best:
                            best, bestB = l[i - k], i - k
                else:
                    best, bestB = h[HEADb], HEADb
                    for k in range(kA + 1, kB):
                        if 0 <= i - k < n and h[i - k] > best:
                            best, bestB = h[i - k], i - k
                NL1, NL1b = best, bestB

        # 1 → 2 : right shoulder near left shoulder
        if st == 1 and is_p[i] and a_ok:
            pv = float(px[i])
            cond = (top and pv < HEAD - a * 0.5) or (not top and pv > HEAD + a * 0.5)
            if cond and abs(pv - LS) <= a * hs_atr_tol:
                st = 2
                RS, RSb = pv, i - swing_len
                kA2 = i - RSb
                kB2 = i - HEADb
                if top:
                    best, bestB = l[RSb], RSb
                    for k in range(kA2 + 1, kB2):
                        if 0 <= i - k < n and l[i - k] < best:
                            best, bestB = l[i - k], i - k
                else:
                    best, bestB = h[RSb], RSb
                    for k in range(kA2 + 1, kB2):
                        if 0 <= i - k < n and h[i - k] > best:
                            best, bestB = h[i - k], i - k
                NL2, NL2b = best, bestB

        # 2 / 3 : live neckline + break watch
        if st in (2, 3) and not np.isnan(NL1) and not np.isnan(NL2):
            denom = max(NL2b - NL1b, 1)
            slope = (NL2 - NL1) / denom
            nl_now = NL2 + slope * (i - NL2b)
            nl[i] = nl_now
            if st == 2 and ((top and c[i] < nl_now) or (not top and c[i] > nl_now)):
                st = 3
                brokeBar = i
        state[i] = st
        broke[i] = brokeBar if st == 3 else -1

    return {"state": state, "nl": nl, "broke": broke}


def hs_features(h: pd.Series, l: pd.Series, c: pd.Series,
                swing_len: int, hs_atr_tol: float, atr: pd.Series) -> pd.DataFrame:
    """H&S top + bottom detectors on one timeframe."""
    T = _hs_detect(h.to_numpy(), l.to_numpy(), c.to_numpy(),
                   swing_len, hs_atr_tol, atr.to_numpy(), top=True)
    B = _hs_detect(h.to_numpy(), l.to_numpy(), c.to_numpy(),
                   swing_len, hs_atr_tol, atr.to_numpy(), top=False)
    return pd.DataFrame({
        "hsT_state": T["state"], "hsT_nl": T["nl"], "hsT_broke": T["broke"],
        "hsB_state": B["state"], "hsB_nl": B["nl"], "hsB_broke": B["broke"],
    }, index=h.index)


# ── SMC zone engine + trap intelligence ──────────────────────────────

def zones_traps_features(df: pd.DataFrame, feat: pd.DataFrame, P: GodceParams) -> pd.DataFrame:
    """OB/FVG/breaker zone engine with hit detection, touch decay, quantum
    gravity, inverted-FVG flips, plus the trap layer: SFP, fake breakouts,
    beyond-FVG state machines, inside-gap bait flags, iFVG tests."""
    n = len(df)
    O = df["open"].to_numpy()
    H = df["high"].to_numpy()
    L = df["low"].to_numpy()
    C = df["close"].to_numpy()
    V = df["volume"].to_numpy()

    def col(name):
        return feat[name].to_numpy()

    volSma, ATR = col("volSma"), col("atr")
    PH, PL = col("ph"), col("pl")
    lastPH, lastPL = col("lastPH"), col("lastPL")
    isEQH, isEQL = col("isEQH"), col("isEQL")
    eqhLevel, eqlLevel = col("eqhLevel"), col("eqlLevel")
    AH, AL = col("asianHigh"), col("asianLow")
    pdH, pdL, pwH, pwL = col("pdHigh"), col("pdLow"), col("pwHigh"), col("pwLow")
    htfBullImp, htfBearImp = col("htfBullImp"), col("htfBearImp")
    htfObH, htfObL, isNewHtf = col("htfObH"), col("htfObL"), col("isNewHtfBar")
    regAtrMult = 1.2  # H1 intraday

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-P.qGravity * x))

    # zone stores
    bOb, rOb = [], []          # dicts: top,bot,bar,ok,str,touches
    bFvg, rFvg = [], []
    ifvgSup, ifvgDem = [], []
    rbs, sbr = [], []          # dicts: top,bot,ok,state

    def add_fvg(store, top, bot, bidx):
        if len(store) >= P.maxFVGs:
            store.pop(0)
        store.append({"top": top, "bot": bot, "bar": bidx, "ok": True, "str": 70.0})

    # output arrays
    B = lambda: np.zeros(n, dtype=bool)
    F = lambda: np.full(n, _NAN)
    out_b = {k: B() for k in [
        "bObHit", "bObCoreTap", "rObHit", "rObCoreTap", "bFvgHit", "rFvgHit",
        "rbsHit", "sbrHit", "bullTrapNow", "bearTrapNow",
        "bullTrapActive", "bearTrapActive", "bullTrapWasEng", "bearTrapWasEng",
        "fakeBrkUp", "fakeBrkDn",
        "brkUpSus", "brkDnSus", "bfgSweepEv", "bfgReclaimEv", "bfgFull",
        "rfgSweepEv", "rfgReclaimEv", "rfgFull",
        "insideBullFvg", "insideBearFvg",
        "baitLong", "baitShort", "ifvgSupTest", "ifvgDemTest",
        "ifvgFlipSupEv", "ifvgFlipDemEv"]}
    out_f = {k: F() for k in [
        "bObZoneGravity", "rObZoneGravity", "bfgState", "rfgState",
        "ifvgSupLvl", "ifvgDemLvl"]}
    out_f["bfgState"][:] = 0
    out_f["rfgState"][:] = 0

    # trap state
    bullTrapBar = _NAN
    bullTrapWasEng = False
    bearTrapBar = _NAN
    bearTrapWasEng = False
    brkUpBar, brkUpLvl = _NAN, _NAN
    brkDnBar, brkDnLvl = _NAN, _NAN
    bfgState, bfgBot, bfgTop, bfgBar = 0, _NAN, _NAN, _NAN
    bfgReclaimFull = False
    rfgState, rfgTop, rfgBot, rfgBar = 0, _NAN, _NAN, _NAN
    rfgReclaimFull = False
    dpBuyLevel = _NAN   # dark-pool levels (detection kept, chart lines dropped)
    dpSellLevel = _NAN
    dpB = np.full(n, _NAN)
    dpS = np.full(n, _NAN)

    for i in range(n):
        a = ATR[i]
        a_ok = not np.isnan(a) and a > 0
        vs = volSma[i]
        v_ok = not np.isnan(vs) and vs > 0
        # local liquidity-tier classifier (needs this bar's levels)
        def liq_tier_local(level, close, _a=a, _pdH=pdH[i], _pdL=pdL[i],
                           _pwH=pwH[i], _pwL=pwL[i]):
            if not P.useLiqHier or np.isnan(_a) or _a <= 0:
                return 0
            tier = 0
            if abs(close - level) / _a < 0.3:
                tier = 1
            if ((not np.isnan(_pdH) and abs(level - _pdH) < _a * 0.5)
                    or (not np.isnan(_pdL) and abs(level - _pdL) < _a * 0.5)):
                tier = 2
            if ((not np.isnan(_pwH) and abs(level - _pwH) < _a * 0.8)
                    or (not np.isnan(_pwL) and abs(level - _pwL) < _a * 0.8)):
                tier = 3
            if abs(level - round(level / 50.0) * 50.0) < 0.05 or abs(level - round(level / 1000.0) * 1000.0) < 0.05:
                tier = 4
            return min(tier, P.maxLiqTier)

        # ── impulse + HTF OB creation ──
        imp_th = a * regAtrMult * 0.85 * 1.2 if a_ok else _NAN  # symImpulse=0.85 (gold)
        bull_imp = (a_ok and i >= 1 and (C[i] - O[i]) > imp_th
                    and C[i] > H[i - 1] and C[i - 1] < O[i - 1])
        bear_imp = (a_ok and i >= 1 and (O[i] - C[i]) > imp_th
                    and C[i] < L[i - 1] and C[i - 1] > O[i - 1])
        if htfBullImp[i] and isNewHtf[i] and not np.isnan(htfObH[i]):
            if len(bOb) >= P.maxOBs:
                bOb.pop(0)
            bOb.append({"top": htfObH[i], "bot": htfObL[i], "bar": i, "ok": True,
                        "str": 100.0, "touches": 0,
                        "tier": liq_tier_local(htfObH[i], C[i])})
        elif bull_imp:
            if len(bOb) >= P.maxOBs:
                bOb.pop(0)
            bstr = min((C[i] - O[i]) / a * 25.0 + 50.0, 100.0)
            bOb.append({"top": H[i - 1], "bot": L[i - 1], "bar": i - 1, "ok": True,
                        "str": bstr, "touches": 0,
                        "tier": liq_tier_local(H[i - 1], C[i])})
        if htfBearImp[i] and isNewHtf[i] and not np.isnan(htfObH[i]):
            if len(rOb) >= P.maxOBs:
                rOb.pop(0)
            rOb.append({"top": htfObH[i], "bot": htfObL[i], "bar": i, "ok": True,
                        "str": 100.0, "touches": 0,
                        "tier": liq_tier_local(htfObH[i], C[i])})
        elif bear_imp:
            if len(rOb) >= P.maxOBs:
                rOb.pop(0)
            rstr = min((O[i] - C[i]) / a * 25.0 + 50.0, 100.0)
            rOb.append({"top": H[i - 1], "bot": L[i - 1], "bar": i - 1, "ok": True,
                        "str": rstr, "touches": 0,
                        "tier": liq_tier_local(H[i - 1], C[i])})

        # ── breaker creation ──
        if not np.isnan(PH[i]) and C[i] > PH[i] and a_ok:
            if len(rbs) >= 8:
                rbs.pop(0)
            rbs.append({"top": PH[i], "bot": PH[i] - a * 0.5, "ok": True, "state": 0})
        if not np.isnan(PL[i]) and C[i] < PL[i] and a_ok:
            if len(sbr) >= 8:
                sbr.pop(0)
            sbr.append({"top": PL[i] + a * 0.5, "bot": PL[i], "ok": True, "state": 0})

        # ── FVG detection ──
        fvg_min = a * (P.fvgMinPct / 100.0) * 0.75 if a_ok else _NAN  # symFvgScl=0.75 (gold)
        if a_ok and i >= 2:
            if L[i] > H[i - 2] and (L[i] - H[i - 2]) >= fvg_min:
                add_fvg(bFvg, L[i], H[i - 2], i - 1)
            if H[i] < L[i - 2] and (L[i - 2] - H[i]) >= fvg_min:
                add_fvg(rFvg, L[i - 2], H[i], i - 1)

        # ── OB hit / gravity ──
        b_hit = b_core = False
        b_grav = 0.0
        for z in bOb:
            if not z["ok"]:
                continue
            t, b_, cb = z["top"], z["bot"], z["bar"]
            if L[i] <= t and C[i] >= b_:
                b_hit = True
                hz = t - b_
                if hz > 0:
                    core_b = b_ + hz * (1 - P.qCorePct) / 2
                    core_t = b_ + hz * (1 + P.qCorePct) / 2
                    if core_b <= C[i] <= core_t:
                        b_core = True
                z["touches"] += 1
                if z["touches"] >= 3:
                    z["str"] *= 0.7
            if a_ok:
                b_grav += sigmoid(-abs(C[i] - t) / a) * (z["str"] / 100.0)
            if C[i] < b_ or (i - cb) > P.zoneMaxAge:
                z["ok"] = False
        r_hit = r_core = False
        r_grav = 0.0
        for z in rOb:
            if not z["ok"]:
                continue
            t, b_, cb = z["top"], z["bot"], z["bar"]
            if H[i] >= b_ and C[i] <= t:
                r_hit = True
                hz = t - b_
                if hz > 0:
                    core_b = b_ + hz * (1 - P.qCorePct) / 2
                    core_t = b_ + hz * (1 + P.qCorePct) / 2
                    if core_b <= C[i] <= core_t:
                        r_core = True
                z["touches"] += 1
                if z["touches"] >= 3:
                    z["str"] *= 0.7
            if a_ok:
                b_grav += sigmoid(-abs(C[i] - b_) / a) * (z["str"] / 100.0)
            if C[i] > t or (i - cb) > P.zoneMaxAge:
                z["ok"] = False
        out_b["bObHit"][i] = b_hit
        out_b["bObCoreTap"][i] = b_core
        out_b["rObHit"][i] = r_hit
        out_b["rObCoreTap"][i] = r_core
        out_f["bObZoneGravity"][i] = b_grav
        out_f["rObZoneGravity"][i] = r_grav

        # ── FVG hit + iFVG flips ──
        bf_hit = False
        flip_sup = False
        for z in bFvg:
            if not z["ok"]:
                continue
            t, b_, cb = z["top"], z["bot"], z["bar"]
            if L[i] <= t and C[i] >= b_:
                bf_hit = True
            failed = C[i] < b_
            if failed or (i - cb) > P.zoneMaxAge:
                z["ok"] = False
                if P.useInvFvg and failed:
                    flip_sup = True
                    if len(ifvgSup) >= P.maxInvFvgs:
                        ifvgSup.pop(0)
                    ifvgSup.append({"top": t, "bot": b_, "bar": i, "ok": True})
        rf_hit = False
        flip_dem = False
        for z in rFvg:
            if not z["ok"]:
                continue
            t, b_, cb = z["top"], z["bot"], z["bar"]
            if H[i] >= b_ and C[i] <= t:
                rf_hit = True
            failed = C[i] > t
            if failed or (i - cb) > P.zoneMaxAge:
                z["ok"] = False
                if P.useInvFvg and failed:
                    flip_dem = True
                    if len(ifvgDem) >= P.maxInvFvgs:
                        ifvgDem.pop(0)
                    ifvgDem.append({"top": t, "bot": b_, "bar": i, "ok": True})
        out_b["bFvgHit"][i] = bf_hit
        out_b["rFvgHit"][i] = rf_hit
        out_b["ifvgFlipSupEv"][i] = flip_sup
        out_b["ifvgFlipDemEv"][i] = flip_dem

        # ── breaker hits ──
        rbs_hit = sbr_hit = False
        for z in rbs:
            if not z["ok"]:
                continue
            if L[i] <= z["top"] and C[i] >= z["bot"]:
                rbs_hit = True
                z["state"] = 1
            if C[i] < z["bot"] or (i - 10) > P.zoneMaxAge:
                z["ok"] = False
        for z in sbr:
            if not z["ok"]:
                continue
            if H[i] >= z["bot"] and C[i] <= z["top"]:
                sbr_hit = True
                z["state"] = 1
            if C[i] > z["top"] or (i - 10) > P.zoneMaxAge:
                z["ok"] = False
        out_b["rbsHit"][i] = rbs_hit
        out_b["sbrHit"][i] = sbr_hit

        # ── dark pool levels ──
        dz = feat["deltaZ"].to_numpy()[i]
        if P.useDarkPool and not np.isnan(dz):
            if dz > 1.8:
                dpBuyLevel = C[i]
            if dz < -1.8:
                dpSellLevel = C[i]
        dpB[i] = dpBuyLevel
        dpS[i] = dpSellLevel

        # ── 1) swing failure patterns ──
        bull_trap = (P.useTrapEngine and not np.isnan(lastPH[i]) and H[i] > lastPH[i]
                     and C[i] < lastPH[i] and C[i] < O[i] and v_ok and V[i] > vs * P.trapVolMult)
        bear_trap = (P.useTrapEngine and not np.isnan(lastPL[i]) and L[i] < lastPL[i]
                     and C[i] > lastPL[i] and C[i] > O[i] and v_ok and V[i] > vs * P.trapVolMult)
        prox = a * 0.35 if a_ok else _NAN
        bull_eng = bull_trap and a_ok and (
            (isEQH[i] and abs(lastPH[i] - eqhLevel[i]) <= prox)
            or (not np.isnan(AH[i]) and abs(H[i] - AH[i]) <= prox)
            or (not np.isnan(pdH[i]) and abs(H[i] - pdH[i]) <= prox)
            or (not np.isnan(pwH[i]) and abs(H[i] - pwH[i]) <= prox))
        bear_eng = bear_trap and a_ok and (
            (isEQL[i] and abs(lastPL[i] - eqlLevel[i]) <= prox)
            or (not np.isnan(AL[i]) and abs(L[i] - AL[i]) <= prox)
            or (not np.isnan(pdL[i]) and abs(L[i] - pdL[i]) <= prox)
            or (not np.isnan(pwL[i]) and abs(L[i] - pwL[i]) <= prox))
        if bull_trap:
            bullTrapBar, bullTrapWasEng = i, bool(bull_eng)
        if bear_trap:
            bearTrapBar, bearTrapWasEng = i, bool(bear_eng)
        bull_active = not np.isnan(bullTrapBar) and i - bullTrapBar <= P.trapMemoryBars
        bear_active = not np.isnan(bearTrapBar) and i - bearTrapBar <= P.trapMemoryBars
        out_b["bullTrapNow"][i] = bull_trap
        out_b["bearTrapNow"][i] = bear_trap
        out_b["bullTrapActive"][i] = bull_active
        out_b["bearTrapActive"][i] = bear_active

        # ── 2) fake-breakout machine ──
        brk_up = (P.useTrapEngine and not np.isnan(lastPH[i]) and C[i] > lastPH[i]
                  and i >= 1 and C[i - 1] <= lastPH[i])
        brk_dn = (P.useTrapEngine and not np.isnan(lastPL[i]) and C[i] < lastPL[i]
                  and i >= 1 and C[i - 1] >= lastPL[i])
        brk_up_sus = brk_up and v_ok and V[i] < vs
        brk_dn_sus = brk_dn and v_ok and V[i] < vs
        if brk_up:
            brkUpBar, brkUpLvl = i, lastPH[i]
        if brk_dn:
            brkDnBar, brkDnLvl = i, lastPL[i]
        f_up = f_dn = False
        if not np.isnan(brkUpBar) and i - brkUpBar <= P.brkLookback and C[i] < (brkUpLvl if not np.isnan(brkUpLvl) else C[i]):
            f_up = True
            brkUpBar = _NAN
            bullTrapBar, bullTrapWasEng = i, bool(bullTrapWasEng or bull_eng)
            bull_active = True
        if not np.isnan(brkDnBar) and i - brkDnBar <= P.brkLookback and C[i] > (brkDnLvl if not np.isnan(brkDnLvl) else C[i]):
            f_dn = True
            brkDnBar = _NAN
            bearTrapBar, bearTrapWasEng = i, bool(bearTrapWasEng or bear_eng)
            bear_active = True
        if not np.isnan(brkUpBar) and i - brkUpBar > P.brkLookback:
            brkUpBar = _NAN
        if not np.isnan(brkDnBar) and i - brkDnBar > P.brkLookback:
            brkDnBar = _NAN
        out_b["fakeBrkUp"][i] = f_up
        out_b["fakeBrkDn"][i] = f_dn
        out_b["brkUpSus"][i] = brk_up_sus
        out_b["brkDnSus"][i] = brk_dn_sus
        out_b["bullTrapActive"][i] = bull_active
        out_b["bearTrapActive"][i] = bear_active
        out_b["bullTrapWasEng"][i] = bool(bullTrapWasEng) and bull_active
        out_b["bearTrapWasEng"][i] = bool(bearTrapWasEng) and bear_active

        # ── 3) beyond-FVG machines ──
        b_sweep = b_reclaim = r_sweep = r_reclaim = False
        if P.useBeyondFvg:
            for z in bFvg:
                t, b_, cb = z["top"], z["bot"], z["bar"]
                if i - cb > P.zoneMaxAge:
                    continue
                if bfgState == 0 and L[i] < b_ and C[i] > b_:
                    bfgState, bfgBot, bfgTop, bfgBar = 2, b_, t, i
                    bfgReclaimFull = C[i] > t
                    b_sweep = b_reclaim = True
                elif bfgState == 0 and C[i] < b_:
                    bfgState, bfgBot, bfgTop, bfgBar = 1, b_, t, i
                    bfgReclaimFull = False
                    b_sweep = True
                elif bfgState == 1 and C[i] > bfgBot:
                    bfgState, bfgBar = 2, i
                    bfgReclaimFull = C[i] > bfgTop
                    b_reclaim = True
            for z in rFvg:
                t, b_, cb = z["top"], z["bot"], z["bar"]
                if i - cb > P.zoneMaxAge:
                    continue
                if rfgState == 0 and H[i] > t and C[i] < t:
                    rfgState, rfgTop, rfgBot, rfgBar = 2, t, b_, i
                    rfgReclaimFull = C[i] < b_
                    r_sweep = r_reclaim = True
                elif rfgState == 0 and C[i] > t:
                    rfgState, rfgTop, rfgBot, rfgBar = 1, t, b_, i
                    rfgReclaimFull = False
                    r_sweep = True
                elif rfgState == 1 and C[i] < rfgTop:
                    rfgState, rfgBar = 2, i
                    rfgReclaimFull = C[i] < rfgBot
                    r_reclaim = True
        if bfgState > 0 and not np.isnan(bfgBar) and i - bfgBar > P.trapMemoryBars:
            bfgState = 0
        if rfgState > 0 and not np.isnan(rfgBar) and i - rfgBar > P.trapMemoryBars:
            rfgState = 0
        out_f["bfgState"][i] = bfgState
        out_f["rfgState"][i] = rfgState
        out_b["bfgFull"][i] = bool(bfgReclaimFull) and bfgState == 2
        out_b["rfgFull"][i] = bool(rfgReclaimFull) and rfgState == 2
        out_b["bfgSweepEv"][i] = b_sweep
        out_b["bfgReclaimEv"][i] = b_reclaim
        out_b["rfgSweepEv"][i] = r_sweep
        out_b["rfgReclaimEv"][i] = r_reclaim

        # ── 4) inside-gap bait ──
        in_bull = in_bear = False
        if P.useBeyondFvg:
            for z in bFvg:
                if z["ok"] and z["bot"] <= C[i] <= z["top"]:
                    in_bull = True
                    break
            for z in rFvg:
                if z["ok"] and z["bot"] <= C[i] <= z["top"]:
                    in_bear = True
                    break
        out_b["insideBullFvg"][i] = in_bull
        out_b["insideBearFvg"][i] = in_bear
        out_b["baitLong"][i] = P.useBeyondFvg and in_bull and bfgState != 2
        out_b["baitShort"][i] = P.useBeyondFvg and in_bear and rfgState != 2

        # ── 5) iFVG tests ──
        sup_test = False
        sup_lvl = _NAN
        if P.useInvFvg:
            for z in ifvgSup:
                if not z["ok"]:
                    continue
                t, b_, cb = z["top"], z["bot"], z["bar"]
                if H[i] >= b_ and C[i] <= t and C[i] <= O[i]:
                    sup_test, sup_lvl = True, t
                if C[i] > t or i - cb > P.zoneMaxAge:
                    z["ok"] = False
        dem_test = False
        dem_lvl = _NAN
        if P.useInvFvg:
            for z in ifvgDem:
                if not z["ok"]:
                    continue
                t, b_, cb = z["top"], z["bot"], z["bar"]
                if L[i] <= t and C[i] >= b_ and C[i] >= O[i]:
                    dem_test, dem_lvl = True, b_
                if C[i] < b_ or i - cb > P.zoneMaxAge:
                    z["ok"] = False
        out_b["ifvgSupTest"][i] = sup_test
        out_b["ifvgDemTest"][i] = dem_test
        out_f["ifvgSupLvl"][i] = sup_lvl
        out_f["ifvgDemLvl"][i] = dem_lvl

    data = {}
    for k, v in out_b.items():
        data[k] = pd.Series(v, index=df.index)
    for k, v in out_f.items():
        data[k] = pd.Series(v, index=df.index)
    data["atBuyZone"] = data["bObHit"] | data["bFvgHit"] | data["rbsHit"]
    data["atSellZone"] = data["rObHit"] | data["rFvgHit"] | data["sbrHit"]
    data["dpBuyLevel"] = pd.Series(dpB, index=df.index)
    data["dpSellLevel"] = pd.Series(dpS, index=df.index)
    return pd.DataFrame(data, index=df.index)


# ── session sense + line geometry + mode-signal agreement ────────────

def session_sense(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Instrument-aware clock (gold). Vectorized port of SESSION SENSE."""
    wd = index.weekday
    hr = index.hour
    is_btc = False
    is_weekend = is_btc and ((wd == 6) | (wd == 5))  # kept for schema; False for gold
    is_gold_weekday = (wd >= 0) & (wd <= 4)
    return pd.DataFrame({
        "isWeekend": is_weekend,
        "isGoldWeekday": is_gold_weekday,
        "isAsia": is_gold_weekday & (hr < 6),
        "isLondon": is_gold_weekday & (hr >= 7) & (hr < 12),
        "isGoldActive": is_gold_weekday & (hr >= 7) & (hr < 17),
    }, index=index)


def geometry_features(df: pd.DataFrame, feat: pd.DataFrame, P: GodceParams) -> pd.DataFrame:
    """Trendline least-squares geometry, pattern classification, Asia-range
    breaks, squeeze detection. Faithful port of the LINE GEOMETRY block."""
    n = len(df)
    H, L, C = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    V = df["volume"].to_numpy()
    ATR = feat["atr"].to_numpy()
    volSma = feat["volSma"].to_numpy()
    isAsia = feat["isAsia"].to_numpy() if "isAsia" in feat else np.zeros(n, dtype=bool)
    isLondon = feat["isLondon"].to_numpy() if "isLondon" in feat else np.zeros(n, dtype=bool)
    isGoldWeekday = feat["isGoldWeekday"].to_numpy() if "isGoldWeekday" in feat else np.ones(n, dtype=bool)
    day = df.index.day.to_numpy()

    # confirmed 3,3 pivots
    w = 7
    hs = pd.Series(H)
    ls = pd.Series(L)
    piv_hi = (hs == hs.rolling(w, center=True, min_periods=w).max()).to_numpy()
    piv_lo = (ls == ls.rolling(w, center=True, min_periods=w).min()).to_numpy()

    def fit_line(Pv, Bv, cnt):
        # least squares over last cnt pivots; x = bars since oldest
        P_ = Pv[-cnt:]
        B_ = Bv[-cnt:]
        x0 = B_[0]
        x = B_ - x0
        sx, sy = x.sum(), P_.sum()
        sxx, sxy = (x * x).sum(), (x * P_).sum()
        denom = cnt * sxx - sx * sx
        slope = (cnt * sxy - sx * sy) / denom if denom > 0 else 0.0
        b0 = (sy - slope * sx) / cnt
        resid = np.abs(P_ - (slope * x + b0)).mean()
        return slope, b0, resid, x0

    geoPlP: deque = deque(maxlen=8)
    geoPlB: deque = deque(maxlen=8)
    geoPhP: deque = deque(maxlen=8)
    geoPhB: deque = deque(maxlen=8)

    supSlope = np.full(n, _NAN)
    supNow = np.full(n, _NAN)
    resSlope = np.full(n, _NAN)
    resNow = np.full(n, _NAN)
    supClean = np.zeros(n, dtype=bool)
    resClean = np.zeros(n, dtype=bool)
    pattern = np.zeros(n, dtype=int)

    asiaHi = _NAN
    asiaLo = _NAN
    asiaDay = -1
    asia_hi = np.full(n, _NAN)
    asia_lo = np.full(n, _NAN)

    for i in range(n):
        a = ATR[i]
        # pivot collection (confirmed at i, pivot bar = i-3)
        if piv_hi[i] and not np.isnan(H[i]):
            geoPhP.append(H[i])
            geoPhB.append(i - 3)
        if piv_lo[i] and not np.isnan(L[i]):
            geoPlP.append(L[i])
            geoPlB.append(i - 3)
        hasSup = len(geoPlP) >= 3
        hasRes = len(geoPhP) >= 3
        sS = sN = rS = rN = _NAN
        sCl = rCl = False
        if hasSup and not np.isnan(a):
            cnt = min(4, len(geoPlP))
            sS, b0, r0, x0 = fit_line(np.array(geoPlP), np.array(geoPlB), cnt)
            sN = sS * (i - x0) + b0
            sCl = r0 < a * 0.75
        if hasRes and not np.isnan(a):
            cnt = min(4, len(geoPhP))
            rS, b0, r0, x0 = fit_line(np.array(geoPhP), np.array(geoPhB), cnt)
            rN = rS * (i - x0) + b0
            rCl = r0 < a * 0.75
        supSlope[i], supNow[i] = sS, sN
        resSlope[i], resNow[i] = rS, rN
        supClean[i], resClean[i] = sCl, rCl

        pat = 0
        if hasSup and hasRes and sCl and rCl and not np.isnan(a):
            gap = rN - sN
            gslope = rS - sS
            apex = gap / -gslope if gslope < 0 and gap > 0 else _NAN
            if gap > a * 0.5:
                sRise, sFall = sS > a * 0.02, sS < -a * 0.02
                rRise, rFall = rS > a * 0.02, rS < -a * 0.02
                if not (sRise or sFall) and not (rRise or rFall):
                    pat = 6  # RANGE
                elif gslope < -a * 0.01 and not np.isnan(apex) and 5 <= apex <= 150:
                    if not (rRise or rFall) and sRise:
                        pat = 4
                    elif not (sRise or sFall) and rFall:
                        pat = 5
                    elif rFall and sRise:
                        pat = 1
                    elif rFall and sFall:
                        pat = 2
                    elif rRise and sRise:
                        pat = 3
        pattern[i] = pat

        # asia range break tracking (session-sense day)
        if day[i] != asiaDay:
            asiaHi, asiaLo, asiaDay = _NAN, _NAN, day[i]
        if isAsia[i]:
            asiaHi = H[i] if np.isnan(asiaHi) else max(asiaHi, H[i])
            asiaLo = L[i] if np.isnan(asiaLo) else min(asiaLo, L[i])
        asia_hi[i], asia_lo[i] = asiaHi, asiaLo

    out = pd.DataFrame(index=df.index)
    out["supSlope"] = supSlope
    out["supNow"] = supNow
    out["resSlope"] = resSlope
    out["resNow"] = resNow
    out["supClean"] = supClean
    out["resClean"] = resClean
    out["supTrend"] = supClean & (supSlope > feat["atr"].to_numpy() * 0.05)
    out["supTest"] = (~np.isnan(supNow)) & (L <= supNow) & (C > supNow)
    out["resTrendUp"] = resClean & (resSlope > feat["atr"].to_numpy() * 0.05)
    out["resTest"] = (~np.isnan(resNow)) & (H >= resNow) & (C < resNow)
    out["pattern"] = pattern
    vs = volSma
    v_ok = ~np.isnan(vs) & (vs > 0)
    out["patBreakLong"] = (pattern != 0) & (~np.isnan(resNow)) & (C > resNow) & \
        (np.concatenate([[False], C[:-1] < resNow[:-1] - np.where(np.isnan(resSlope[:-1]), 0, resSlope[:-1])])) & \
        v_ok & (V > np.where(np.isnan(vs), np.inf, vs))
    out["patBreakShort"] = (pattern != 0) & (~np.isnan(supNow)) & (C < supNow) & \
        (np.concatenate([[False], C[:-1] > supNow[:-1] - np.where(np.isnan(supSlope[:-1]), 0, supSlope[:-1])])) & \
        v_ok & (V > np.where(np.isnan(vs), np.inf, vs))
    asia_valid = isGoldWeekday & (~np.isnan(asia_hi)) & (asia_hi > asia_lo)
    prev_c = np.concatenate([[np.nan], C[:-1]])
    out["asiaBreakLong"] = asia_valid & isLondon & (C > asia_hi) & (prev_c <= asia_hi) & v_ok & (V > np.where(np.isnan(vs), np.inf, vs))
    out["asiaBreakShort"] = asia_valid & isLondon & (C < asia_lo) & (prev_c >= asia_lo) & v_ok & (V > np.where(np.isnan(vs), np.inf, vs))
    range30 = (pd.Series(H).rolling(30).max() - pd.Series(L).rolling(30).min()).to_numpy()
    out["squeeze30"] = range30 < feat["atr"].to_numpy() * 6.0
    adx = feat["adxLine"].to_numpy()
    out["adxUp"] = (~np.isnan(adx)) & (adx > np.concatenate([[np.nan] * 3, adx[:-3]]))
    return out
