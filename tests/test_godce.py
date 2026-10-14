"""Tests for goldenswing/godce/ — the GOD Conviction Engine port."""
import numpy as np
import pandas as pd
import pytest

from goldenswing.godce import GodceParams
from goldenswing.godce import indicators as I
from goldenswing.godce import stateful as S
from goldenswing.godce import scoring as SC
from goldenswing.godce import router as R
from goldenswing.godce.engine import conviction, run_engine
from goldenswing.godce.params import (
    M_BRK, M_FLAT, M_MEAN, M_MOM, M_POST, REGIME_TREND_BULL,
)


def _ohlc(n=300, seed=7, trend=0.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.001, n)
    close = 2600 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    vol = np.abs(rng.normal(1000, 200, n))
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _dxy(n, idx):
    rng = np.random.default_rng(3)
    c = 103 + np.cumsum(rng.normal(0, 0.0005, n))
    return pd.DataFrame(
        {"open": c, "high": c * 1.0002, "low": c * 0.9998, "close": c,
         "volume": np.ones(n)},
        index=idx,
    )


def _htf(df):
    return {
        "h4": I.resample_ohlc(df, "4h"),
        "d1": I.resample_ohlc(df, "1D"),
        "w1": I.resample_ohlc(df, "1W"),
    }


# ── primitives ───────────────────────────────────────────────────────

def test_ema_matches_pandas_ewm():
    s = pd.Series(np.arange(1.0, 51.0))
    got = I.ema(s, 10).to_numpy()
    want = s.ewm(span=10, adjust=False).mean().to_numpy()
    assert np.allclose(got[10:], want[10:], rtol=1e-9)


def test_atr_positive_and_finite():
    df = _ohlc()
    a = I.atr(df.high, df.low, df.close, 14)
    assert (a.dropna() > 0).all()


def test_pivothigh_confirmed_no_repaint():
    # pivot at bar 10 (the peak), confirmed at bar 15
    h = pd.Series([1.0] * 20)
    h.iloc[10] = 5.0
    ph = I.pivothigh(h, 5, 5)
    assert ph.notna().sum() == 1
    assert ph.index[ph.notna()][0] == 15  # positioned at confirmation bar


def test_htf_closed_uses_last_closed_bar():
    # no lookahead: value at intraday bar t comes from the HTF bar that
    # closed strictly before t
    idx = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    h4 = pd.Series([1.0, 2.0], index=pd.to_datetime(
        ["2025-01-01 00:00", "2025-01-01 04:00"], utc=True))
    out = I.htf_closed(h4, "4h", idx)
    # bars 00:00-03:59 see NaN (first H4 bar not yet closed); 04:00-07:59 see 1.0
    assert out.iloc[:4].isna().all()
    assert (out.iloc[4:] == 1.0).all()


# ── structure ────────────────────────────────────────────────────────

def test_structure_bos_marks_bullish_state():
    # zigzag uptrend with real swings: higher highs broken to the upside
    n = 80
    t = np.arange(n)
    base = 100 + 0.15 * t + 3.0 * np.sin(t * 2 * np.pi / 20)
    df = _ohlc(n)
    df["close"] = base
    df["high"] = base + 0.4
    df["low"] = base - 0.4
    df["open"] = base - 0.6  # body 0.6 > 0.3*ATR so BOS can trigger
    P = GodceParams()
    feat = I.compute_features(df, _dxy(n, df.index), _htf(df), P)
    st = S.structure_features(df.high, df.low, df.close, df.open,
                              feat["ph"], feat["pl"], feat["atr"],
                              P.eqTol, feat["asianHigh"], feat["asianLow"])
    assert st["bullStruct"].any()
    assert st["bullBOS"].any() or st["isHH"].any()


# ── scoring ──────────────────────────────────────────────────────────

def test_scoring_clamped_and_factor_gate_counts():
    P = GodceParams()
    n = 400
    df = _ohlc(n)
    feat = I.compute_features(df, _dxy(n, df.index), _htf(df), P)
    for c in ["open", "high", "low", "close", "volume"]:
        feat[c] = df[c]
    # wire the columns scoring needs that come from later stages
    for c in ["bObHit", "bObCoreTap", "rObHit", "rObCoreTap", "bFvgHit",
              "rFvgHit", "rbsHit", "sbrHit", "atBuyZone", "atSellZone",
              "baitLong", "baitShort", "bearTrapActive", "bullTrapActive",
              "bearTrapWasEng", "bullTrapWasEng", "ifvgDemTest", "ifvgSupTest",
              "vBotTrig", "vTopTrig", "hsT_state", "hsB_state", "hsD_Tst",
              "hsD_Bst", "patBreakLong", "patBreakShort", "sentiment",
              "bullPin", "bullEngulf", "ibBullBreak", "bearPin", "bearEngulf",
              "ibBearBreak", "nearEQL", "nearEQH", "belowValue", "aboveValue",
              "tkKjGold", "tkKjDeath", "inDiscount", "inPremium",
              "validBuySweep", "validSellSweep", "bullStruct", "bearStruct",
              "aboveVwap", "volBullDiv", "volBearDiv", "wyckSpring", "wyckUTAD",
              "rsiOS", "rsiOB", "rsiBullDiv", "rsiBearDiv", "macdAbove",
              "macdBelow", "macdCross", "macdDeadX", "bullDiv", "bearDiv",
              "bullStack", "bearStack", "tf1Bull", "tf2Bull", "tf3Bull",
              "tf1Bear", "tf2Bear", "tf3Bear", "adxOk", "lowVol",
              "regimeVolatile", "dayKumoTop", "dayKumoBot", "weekOpen",
              "dailyPivot", "pdHigh", "pdLow", "pwHigh", "pwLow",
              "aggressiveBuy", "aggressiveSell", "absorption",
              "dpBuyLevel", "dpSellLevel", "macroBullish", "macroBearish",
              "macroValid", "volImbalance", "hmaBase", "emaElastic",
              "bObZoneGravity", "rObZoneGravity", "bfgState", "rfgState",
              "bfgFull", "rfgFull", "hsT_broke", "hsB_broke",
              "biasScore", "regime"]:
        if c not in feat:
            feat[c] = False if c[0].islower() and "Level" not in c and c not in (
                "sentiment", "bObZoneGravity", "rObZoneGravity", "bfgState",
                "rfgState", "hsT_broke", "hsB_broke", "biasScore", "regime",
                "hmaBase", "emaElastic", "dayKumoTop", "dayKumoBot",
                "weekOpen", "dailyPivot", "pdHigh", "pdLow", "pwHigh", "pwLow",
                "dpBuyLevel", "dpSellLevel") else np.nan
    feat["regime"] = feat["regime"].fillna(0) if "regime" in feat else 0
    out = SC.compute_scores(feat, P)
    assert ((out["buyScore"] >= 0) & (out["buyScore"] <= 100)).all()
    assert ((out["sellScore"] >= 0) & (out["sellScore"] <= 100)).all()
    assert (out["buyFactors"] >= 0).all()
    # perfect bull alignment should score high
    assert out["buyScore"].max() > 50


# ── router ───────────────────────────────────────────────────────────

def test_router_dwell_requires_two_bars():
    cand = pd.Series([2, 2, 1, 1, 1, 2, 2, 2])
    mode = R.apply_dwell(cand, GodceParams(modeDwell=2))
    # switches to 2 only after 2 consecutive bars of candidate 2
    assert mode.iloc[0] == 0  # hold=1 < 2, stays FLAT
    assert mode.iloc[1] == 2  # hold=2 → switch
    assert mode.iloc[2] == 2  # single blip of 1 not enough (hold=1)
    assert mode.iloc[3] == 1  # two bars of 1 → switch
    assert mode.tolist() == [0, 2, 2, 1, 1, 1, 2, 2]


def test_mode_rules_momentum_blocks_counter():
    aL, aS, ctr, *_ = R.mode_rules(M_MOM, 1.5, 3.0, True, False)
    assert ctr is False and aL is True and aS is False
    aL, aS, ctr, f1, f2, f3, cap, run, off = R.mode_rules(M_MEAN, 1.5, 3.0, True, False)
    assert (f1, f2, f3, off) == (0.75, 1.5, 2.5, -5.0)


# ── engine ───────────────────────────────────────────────────────────

def _minimal_feat(df, P, **over):
    n = len(df)
    base = {
        "mode": M_MEAN, "buyScore": 95.0, "sellScore": 10.0,
        "buyFactors": 10, "sellFactors": 0,
        "buyThresh": 70.0, "sellThresh": 70.0,
        "regime": REGIME_TREND_BULL, "atr": 2.0,
        "recentSwingLow": df["low"].rolling(5).min(),
        "recentSwingHigh": df["high"].rolling(5).max(),
        "atrRatio": 1.0, "volOk": True, "sessionOk": True,
        "cautionMode": False, "regimeVolatile": False,
        "baitLong": False, "baitShort": False,
        "brkUpSus": False, "brkDnSus": False,
        "bearTrapActive": False, "bullTrapActive": False,
        "biasScore": 0.0,
    }
    base.update(over)
    feat = pd.DataFrame(base, index=df.index)
    # signal frame the router needs
    sig = pd.DataFrame({
        "momLongOk": True, "momShortOk": True,
        "meanLongOk": True, "meanShortOk": True,
        "postLongOk": False, "postShortOk": False,
        "brkLongOk": False, "brkShortOk": False,
    }, index=df.index)
    # conviction() needs these columns
    for c in ["isGoldActive", "close", "high", "low", "vwapVal", "hmaBase",
              "adxLine", "adxOk", "emaF", "emaS", "macroBullish",
              "macroBearish", "validBuySweep", "validSellSweep", "vBotTrig",
              "vTopTrig", "hsB_state", "hsT_state", "rsiOS", "rsiOB",
              "supTrend", "supTest", "patBreakLong", "asiaBreakLong",
              "supNow", "resClean", "resSlope", "resNow", "pattern",
              "resTrendUp", "resTest", "supSlope", "supClean",
              "patBreakShort", "asiaBreakShort", "pdHigh", "pdLow",
              "pwHigh", "pwLow"]:
        if c not in feat:
            feat[c] = df[c] if c in df else 0.0
    feat["vwapVal"] = df["close"]
    feat["adxLine"] = 20.0
    return feat, sig


def test_engine_enters_and_hits_tp3_runner():
    # steady grind up: entry, TP1→BE, runner to TP3
    n = 40
    close = 2600 + np.arange(n) * 3.0
    df = _ohlc(n)
    df["close"] = close
    df["open"] = close - 0.5
    df["high"] = close + 1.0
    df["low"] = close - 1.0
    P = GodceParams(warmupBars=5, usePVS=True)
    feat, sig = _minimal_feat(df, P)
    trades = run_engine(df, feat, sig, P)
    assert len(trades) >= 1
    t0 = trades.iloc[0]
    assert t0["side"] == "long"
    assert t0["exit_reason"] == "TP3"
    assert t0["net_R"] > 2.0


def test_engine_cost_gate_blocks_unprofitable():
    n = 40
    df = _ohlc(n)
    P = GodceParams(warmupBars=5, useCostGate=True, costBps=400)  # absurd cost
    feat, sig = _minimal_feat(df, P)
    trades = run_engine(df, feat, sig, P)
    assert len(trades) == 0


def test_engine_circuit_breaker_halts_after_3R_day():
    # grind UP but force SHORT entries → repeated losses trip the breaker
    n = 60
    close = 2600 + np.arange(n) * 2.0
    df = _ohlc(n)
    df.index = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    df["close"] = close
    df["open"] = close - 0.5
    df["high"] = close + 1.0
    df["low"] = close - 1.0

    def run(**kw):
        P = GodceParams(warmupBars=5, **kw)
        feat, sig = _minimal_feat(
            df, P, buyScore=10.0, sellScore=95.0, sellFactors=10,
            regime=2, biasScore=0.0)
        return run_engine(df, feat, sig, P)

    no_halt = run(maxDailyLossR=100.0, maxTradesDay=100)
    halted = run(maxDailyLossR=0.01, maxTradesDay=100)
    assert len(no_halt) > 0
    assert len(halted) < len(no_halt)
    # halted run: every trade after the first belongs to a day that never
    # went below -0.01R — with one day total, at most the pre-halt trades
    assert len(halted) <= 2


def test_conviction_bounded_and_trend_aware():
    P = GodceParams()
    row = {c: 0.0 for c in ["adxLine", "adxLine_m3", "atr", "close", "high",
                            "low", "vwapVal"]}
    row.update({"regime": REGIME_TREND_BULL, "isGoldActive": True,
                "close": 100.0, "vwapVal": 99.0, "adxOk": True,
                "adxLine": 30.0, "adxLine_m3": 25.0, "emaF": 101.0,
                "emaS": 99.0, "atr": 1.0, "high": 100.5, "low": 99.5,
                "pattern": 0})
    c_long = conviction(row, 1)
    c_short = conviction(row, -1)
    assert 0.0 <= c_long <= 100.0 and 0.0 <= c_short <= 100.0
    assert c_long > c_short  # with-trend long scores higher than counter short
