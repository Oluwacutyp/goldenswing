"""GOD Conviction Engine — event-driven trade engine.

Port of Sections 19–20 of GOD-CE-SLIM.pine: the per-bar sequence
(conviction → entries → conviction-invalidation → Hydra management →
exit ladder → accounting), the broker-cost / lure / bait gates, and the
trader-discipline circuit breaker.

Fill convention is faithful to the pine: entries at the confirmed bar
close; the exit ladder is evaluated on the same bar's high/low with the
pine's if/else-if priority.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import (
    M_FLAT,
    M_LEG,
    M_POST,
    MODE_NAMES,
    REGIME_NAMES,
    REGIME_TREND_BEAR,
    REGIME_TREND_BULL,
    GodceParams,
)
from .router import mode_rules, mode_signal_for


def conviction(row: dict, trade_state: int) -> float:
    """f_conviction — live directional confluence 0..100."""
    g = row.get
    is_long = trade_state == 1
    c = 50.0
    if g("isGoldActive"):
        c += 3.0
    regime = g("regime")
    if is_long:
        if regime == REGIME_TREND_BULL:
            c += 12.0
        if regime == REGIME_TREND_BEAR:
            c -= 12.0
        c += 8.0 if g("close") > g("vwapVal") else -5.0
    else:
        if regime == REGIME_TREND_BEAR:
            c += 12.0
        if regime == REGIME_TREND_BULL:
            c -= 12.0
        c += 8.0 if g("close") < g("vwapVal") else -5.0
    adx = g("adxLine") or 0.0
    adx_prev3 = g("adxLine_m3") or 0.0
    if g("adxOk") and adx > adx_prev3:
        c += 8.0
    if is_long:
        c += 8.0 if g("emaF") > g("emaS") else -8.0
    else:
        c += 8.0 if g("emaF") < g("emaS") else -8.0
    if is_long:
        if g("macroBullish"):
            c += 8.0
        if g("macroBearish"):
            c -= 8.0
    else:
        if g("macroBearish"):
            c += 8.0
        if g("macroBullish"):
            c -= 8.0
    if is_long:
        if g("validBuySweep") or g("vBotTrig"):
            c += 8.0
        if g("validSellSweep"):
            c -= 10.0
    else:
        if g("validSellSweep") or g("vTopTrig"):
            c += 8.0
        if g("validBuySweep"):
            c -= 10.0
    if is_long:
        if (g("hsB_state") or 0) >= 2:
            c += 8.0
        if (g("hsT_state") or 0) >= 2:
            c -= 8.0
    else:
        if (g("hsT_state") or 0) >= 2:
            c += 8.0
        if (g("hsB_state") or 0) >= 2:
            c -= 8.0
    if is_long:
        if g("rsiOS"):
            c += 5.0
        if g("rsiOB"):
            c -= 5.0
    else:
        if g("rsiOB"):
            c += 5.0
        if g("rsiOS"):
            c -= 5.0
    atr = g("atr") or 0.0
    close, high, low = g("close"), g("high"), g("low")
    if is_long:
        if g("supTrend") and g("supTest"):
            c += 8.0
        if g("patBreakLong"):
            c += 8.0
        if g("asiaBreakLong"):
            c += 8.0
        if g("supTrend") and close < (g("supNow") or 0):
            c -= 10.0
        if g("resClean") and (g("resSlope") or 0) < 0 and high >= (g("resNow") or 0) > 0 and close < (g("resNow") or 0):
            c -= 8.0
        if g("pattern") == 2:
            c += 5.0
        if g("pattern") == 3:
            c -= 5.0
    else:
        if g("resTrendUp") and g("resTest"):
            c += 8.0
        if g("patBreakShort"):
            c += 8.0
        if g("asiaBreakShort"):
            c += 8.0
        if g("resTrendUp") and close > (g("resNow") or 0):
            c -= 10.0
        if g("supClean") and (g("supSlope") or 0) > 0 and low <= (g("supNow") or 0) and close > (g("supNow") or 0):
            c -= 8.0
        if g("pattern") == 3:
            c += 5.0
        if g("pattern") == 2:
            c -= 5.0
    rnd = round(close / 100.0) * 100.0
    if abs(close - rnd) < atr * 0.5 and atr > 0:
        if is_long and close < rnd:
            c -= 5.0
        if is_long and close > rnd:
            c += 5.0
        if not is_long and close > rnd:
            c -= 5.0
        if not is_long and close < rnd:
            c += 5.0
    pdH, pdL, pwH, pwL = g("pdHigh"), g("pdLow"), g("pwHigh"), g("pwLow")
    if pdH is not None and pdL is not None and not np.isnan(pdH) and not np.isnan(pdL):
        if is_long and close > pdH:
            c += 5.0
        if is_long and close < pdL:
            c -= 5.0
        if not is_long and close < pdL:
            c += 5.0
        if not is_long and close > pdH:
            c -= 5.0
    if pwH is not None and pwL is not None and not np.isnan(pwH) and not np.isnan(pwL):
        if is_long and close > pwH:
            c += 5.0
        if is_long and close < pwL:
            c -= 5.0
        if not is_long and close < pwL:
            c += 5.0
        if not is_long and close > pwH:
            c -= 5.0
    return max(0.0, min(100.0, c))


# ── event loop ───────────────────────────────────────────────────────

def _session_tag(hour: int) -> str:
    if 0 <= hour < 7:
        return "asian"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 17:
        return "newyork"
    return "off"


def run_engine(df: pd.DataFrame, feat: pd.DataFrame, sig: pd.DataFrame,
               P: GodceParams) -> pd.DataFrame:
    """Run the GOD-CE event loop. Returns one row per closed trade."""
    n = len(df)
    idx = df.index
    O = df["open"].to_numpy()
    H = df["high"].to_numpy()
    L = df["low"].to_numpy()
    C = df["close"].to_numpy()
    hour = idx.hour.to_numpy()
    dow = idx.weekday.to_numpy()

    F = feat

    def row_dict(i):
        d = {c: F[c].iat[i] for c in F.columns if c in (
            "regime", "isGoldActive", "close", "high", "low", "vwapVal",
            "hmaBase", "adxLine", "adxOk", "emaF", "emaS", "macroBullish",
            "macroBearish", "validBuySweep", "validSellSweep", "vBotTrig",
            "vTopTrig", "hsB_state", "hsT_state", "rsiOS", "rsiOB",
            "supTrend", "supTest", "patBreakLong", "asiaBreakLong",
            "supNow", "resClean", "resSlope", "resNow", "pattern",
            "resTrendUp", "resTest", "supSlope", "supClean",
            "patBreakShort", "asiaBreakShort", "pdHigh", "pdLow",
            "pwHigh", "pwLow", "atr")}
        d["adxLine_m3"] = F["adxLine"].iat[i - 3] if i >= 3 else np.nan
        return d

    mode = F["mode"].to_numpy(dtype=int)
    buy_score = F["buyScore"].to_numpy()
    sell_score = F["sellScore"].to_numpy()
    buy_f = F["buyFactors"].to_numpy()
    sell_f = F["sellFactors"].to_numpy()
    buy_th = F["buyThresh"].to_numpy()
    sell_th = F["sellThresh"].to_numpy()
    regime = F["regime"].fillna(0).to_numpy(dtype=int)
    atr = F["atr"].to_numpy()
    recent_lo = F["recentSwingLow"].to_numpy()
    recent_hi = F["recentSwingHigh"].to_numpy()
    atr_ratio = F["atrRatio"].to_numpy()
    vol_ok = F["volOk"].fillna(True).to_numpy(dtype=bool)
    sess_ok = F["sessionOk"].fillna(True).to_numpy(dtype=bool)
    caution = F["cautionMode"].fillna(False).to_numpy(dtype=bool)
    reg_vol = F["regimeVolatile"].fillna(False).to_numpy(dtype=bool)
    bait_l = F["baitLong"].fillna(False).to_numpy(dtype=bool)
    bait_s = F["baitShort"].fillna(False).to_numpy(dtype=bool)
    brk_up_sus = F["brkUpSus"].fillna(False).to_numpy(dtype=bool)
    brk_dn_sus = F["brkDnSus"].fillna(False).to_numpy(dtype=bool)
    bear_trap_active = F["bearTrapActive"].fillna(False).to_numpy(dtype=bool)
    bull_trap_active = F["bullTrapActive"].fillna(False).to_numpy(dtype=bool)
    bias = F["biasScore"].fillna(0).to_numpy()

    trades = []
    # position state
    ts = 0
    entry_px = sl_px = tp1_px = tp2_px = tp3_px = risk_r = np.nan
    tp1_done = be_active = conv_soft = False
    trade_bars = hydra_phase = conv_low_bars = 0
    max_profit_r = peak_conv = 0.0
    pos_run_atr = 1.2
    entry_mode = 0
    entry_i = 0
    entry_buy_score = entry_sell_score = 0.0
    # discipline
    day_r = 0.0
    trades_today = 0
    cur_day = None
    risk_dollars = P.accBalance * P.riskPct / 100.0

    for i in range(n):
        if i < P.warmupBars:
            continue
        # discipline reset on UTC day change
        d = idx[i].date()
        if d != cur_day:
            cur_day = d
            day_r = 0.0
            trades_today = 0
        cb_halt = P.useCircuitBreaker and (day_r <= -abs(P.maxDailyLossR)
                                           or trades_today >= P.maxTradesDay)

        rd = row_dict(i)
        conv_now = conviction(rd, ts if ts != 0 else (1 if buy_score[i] >= sell_score[i] else -1))
        # pine: convNow = tradeState==1 ? f(true) : tradeState==-1 ? f(false) : f(buyScore>=sellScore)
        # with trade_state 0 → is_long = (buyScore >= sellScore); our call above matches.

        m = mode[i]
        aL, aS, mctr, mt1, mt2, mt3, mcap, mrun, moff = mode_rules(
            m if P.useRouter else M_LEG, P.tp1RR, P.tp2RR,
            regime[i] == REGIME_TREND_BULL, regime[i] == REGIME_TREND_BEAR)
        mth_b = min(max(buy_th[i] + moff, 50.0), 95.0)
        mth_s = min(max(sell_th[i] + moff, 50.0), 95.0)
        dyn_sl_buf = P.slMult * 1.5 if reg_vol[i] else P.slMult
        spike_blocked = P.useSpikeShield and (atr_ratio[i] > P.spikeAtrMult)
        can_trade = not caution[i]
        play_l, play_s = mode_signal_for(m, sig, i)
        spike_ok2 = (P.useRouter and m == M_POST) or not spike_blocked

        # ── entries ──
        if ts == 0 and vol_ok[i] and sess_ok[i] and can_trade and not cb_halt:
            mode_open = (m > M_FLAT) if P.useRouter else True
            buy_ok = (aL and mode_open and play_l and spike_ok2
                      and buy_score[i] >= mth_b
                      and (not P.useFactorGate or buy_f[i] >= P.minFactors))
            sell_ok = (aS and mode_open and play_s and spike_ok2
                       and sell_score[i] >= mth_s
                       and (not P.useFactorGate or sell_f[i] >= P.minFactors))
            if not mctr:
                # pine: isCounterBiasBuy = overallBias=="BEARISH" (biasScore<=-2)
                #       and buyScore >= buyThresh
                if bias[i] <= -2 and buy_score[i] >= buy_th[i]:
                    buy_ok = False
                if bias[i] >= 2 and sell_score[i] >= sell_th[i]:
                    sell_ok = False
            # cost gate
            cost_px = C[i] * P.costBps / 10000.0
            a_i = atr[i]
            p_sl_l = max(recent_lo[i] - a_i * dyn_sl_buf, C[i] - a_i * mcap) if not np.isnan(a_i) else np.nan
            p_risk_l = max(C[i] - p_sl_l, 0.0) if not np.isnan(p_sl_l) else 0.0
            p_sl_s = min(recent_hi[i] + a_i * dyn_sl_buf, C[i] + a_i * mcap) if not np.isnan(a_i) else np.nan
            p_risk_s = max(p_sl_s - C[i], 0.0) if not np.isnan(p_sl_s) else 0.0
            cost_ok_l = (p_risk_l >= cost_px * 2.0
                         and P.tp1RR - cost_px / max(p_risk_l, 1e-9) >= P.minNetRR) if P.useCostGate else True
            cost_ok_s = (p_risk_s >= cost_px * 2.0
                         and P.tp1RR - cost_px / max(p_risk_s, 1e-9) >= P.minNetRR) if P.useCostGate else True
            # lure gate
            lure_l = P.useTrapEngine and P.blockSusBreak and brk_up_sus[i] and not bear_trap_active[i]
            lure_s = P.useTrapEngine and P.blockSusBreak and brk_dn_sus[i] and not bull_trap_active[i]
            if bait_l[i] or lure_l or not cost_ok_l:
                buy_ok = False
            if bait_s[i] or lure_s or not cost_ok_s:
                sell_ok = False
            go_l = buy_ok and (not sell_ok or buy_score[i] >= sell_score[i] + 5.0)
            go_s = sell_ok and (not buy_ok or sell_score[i] >= buy_score[i] + 5.0)

            if go_l or go_s:
                is_l = go_l
                ts = 1 if is_l else -1
                entry_px = C[i]
                if is_l:
                    sl_px = max(recent_lo[i] - a_i * dyn_sl_buf, C[i] - a_i * mcap)
                    risk_r = entry_px - sl_px
                else:
                    sl_px = min(recent_hi[i] + a_i * dyn_sl_buf, C[i] + a_i * mcap)
                    risk_r = sl_px - entry_px
                if not (risk_r > 0):
                    ts = 0
                    continue
                tp1_px = entry_px + risk_r * mt1 if is_l else entry_px - risk_r * mt1
                tp2_px = entry_px + risk_r * mt2 if is_l else entry_px - risk_r * mt2
                tp3_px = (entry_px + risk_r * mt3 if is_l else entry_px - risk_r * mt3) if P.usePVS else np.nan
                pos_run_atr = mrun
                peak_conv = conv_now
                conv_low_bars = 0
                conv_soft = False
                tp1_done = be_active = False
                trade_bars = hydra_phase = 0
                max_profit_r = 0.0
                entry_mode = m
                entry_i = i
                entry_buy_score, entry_sell_score = buy_score[i], sell_score[i]
                trades_today += 1

        # ── conviction invalidation + management ──
        if ts != 0:
            is_l = ts == 1
            exited = False
            exit_px = np.nan
            exit_reason = ""

            if conv_now > peak_conv:
                peak_conv = conv_now
            conv_low_bars = conv_low_bars + 1 if conv_now < 20.0 else 0
            if conv_now < 40.0:
                soft_t = entry_px - 0.5 * risk_r if is_l else entry_px + 0.5 * risk_r
                if (is_l and soft_t > sl_px) or (not is_l and soft_t < sl_px):
                    sl_px = soft_t
                    conv_soft = True
            if peak_conv - conv_now >= 20.0 and ((is_l and C[i] > entry_px) or (not is_l and C[i] < entry_px)):
                mid = (entry_px + C[i]) / 2
                if (is_l and mid > sl_px) or (not is_l and mid < sl_px):
                    sl_px = mid
            if conv_low_bars >= 2:
                exited, exit_px, exit_reason = True, C[i], "conviction"

            if not exited:
                trade_bars += 1
                cur_r = ((C[i] - entry_px) if is_l else (entry_px - C[i])) / max(risk_r, 1e-9)
                max_profit_r = max(max_profit_r, cur_r)
                a_i = atr[i]
                # hydra
                if P.useHydra and risk_r > 0 and not np.isnan(a_i):
                    if not tp1_done:
                        if cur_r >= 0.3:
                            tp_ = C[i] - a_i * P.trailATR if is_l else C[i] + a_i * P.trailATR
                            if (is_l and tp_ > sl_px) or (not is_l and tp_ < sl_px):
                                sl_px = tp_
                                hydra_phase = 1
                    elif not be_active:
                        be_active = True
                        sl_px = entry_px
                        hydra_phase = 2
                    else:
                        if cur_r >= 1.5:
                            runner = (C[i] - a_i * P.slMult * pos_run_atr if is_l
                                      else C[i] + a_i * P.slMult * pos_run_atr)
                            if (is_l and runner > sl_px) or (not is_l and runner < sl_px):
                                sl_px = runner
                                hydra_phase = 3
                            if P.useProfitLock and max_profit_r >= 1.5:
                                lock = (entry_px + (max_profit_r - 0.5) * risk_r if is_l
                                        else entry_px - (max_profit_r - 0.5) * risk_r)
                                if (is_l and lock > sl_px) or (not is_l and lock < sl_px):
                                    sl_px = lock
                        elif P.moveBEat50 and cur_r >= 0.5 and ((is_l and sl_px < entry_px) or (not is_l and sl_px > entry_px)):
                            sl_px = entry_px
                            hydra_phase = 2
                # intelligent BE pre-TP1
                pre_tp1 = ((C[i] - entry_px) >= risk_r * 0.75) if is_l else ((entry_px - C[i]) >= risk_r * 0.75)
                if not tp1_done and pre_tp1 and conv_now >= 60.0 and ((is_l and sl_px < entry_px) or (not is_l and sl_px > entry_px)):
                    sl_px = entry_px + (0.1 * risk_r if is_l else -0.1 * risk_r)
                    be_active = True
                # TP1
                tp1_hit = (H[i] >= tp1_px) if is_l else (L[i] <= tp1_px)
                if not tp1_done and tp1_hit:
                    tp1_done = True
                    be_active = True
                    if conv_now >= 70.0:
                        sl_px = entry_px + (0.25 * risk_r if is_l else -0.25 * risk_r)
                    elif conv_now >= 50.0:
                        sl_px = entry_px + (P.beOffsetR * risk_r if is_l else -P.beOffsetR * risk_r)
                    else:
                        sl_px = entry_px
                # exit ladder
                if P.usePVS:
                    tp3_hit = (H[i] >= tp3_px) if is_l else (L[i] <= tp3_px)
                    sl_hit = (L[i] <= sl_px) if is_l else (H[i] >= sl_px)
                    tp2_hit = (H[i] >= tp2_px) if is_l else (L[i] <= tp2_px)
                    if tp3_hit and not np.isnan(tp3_px):
                        exited, exit_px, exit_reason = True, tp3_px, "TP3"
                    elif sl_hit:
                        exited, exit_px, exit_reason = True, sl_px, "BE" if be_active else "SL"
                    elif tp2_hit:
                        pass  # runner on
                else:
                    tp2_hit = (H[i] >= tp2_px) if is_l else (L[i] <= tp2_px)
                    sl_hit = (L[i] <= sl_px) if is_l else (H[i] >= sl_px)
                    if tp2_hit:
                        exited, exit_px, exit_reason = True, tp2_px, "TP2"
                    elif sl_hit:
                        exited, exit_px, exit_reason = True, sl_px, "BE" if be_active else "SL"

            if exited:
                gross_r = ((exit_px - entry_px) if is_l else (entry_px - exit_px)) / risk_r
                cost_r = (C[entry_i] * P.costBps / 10000.0) / risk_r
                net_r = gross_r - cost_r
                day_r += net_r
                trades.append({
                    "entry_time": idx[entry_i], "exit_time": idx[i],
                    "side": "long" if is_l else "short",
                    "entry": entry_px, "sl": sl_px, "exit": exit_px,
                    "exit_reason": exit_reason,
                    "gross_R": gross_r, "cost_R": cost_r, "net_R": net_r,
                    "pnl_usd": net_r * risk_dollars,
                    "mode": MODE_NAMES.get(entry_mode, str(entry_mode)),
                    "regime": REGIME_NAMES.get(regime[entry_i], "?"),
                    "session": _session_tag(hour[entry_i]),
                    "dow": int(dow[entry_i]),
                    "buyScore": entry_buy_score, "sellScore": entry_sell_score,
                    "bars_held": i - entry_i, "max_R": max_profit_r,
                })
                ts = 0

    return pd.DataFrame(trades)
