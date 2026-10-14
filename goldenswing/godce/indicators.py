"""GOD Conviction Engine — vectorized indicator layer.

Faithful port of the Pine series logic from GOD-CE-SLIM.pine. Every Pine
``series`` becomes a DataFrame column; ``x[1]`` becomes ``.shift(1)``.
Stateful constructs (structure trackers, H&S, zones, traps) live in
:mod:`goldenswing.godce.stateful`.

Deliberate deviation from the pine: HTF values obtained via ``request.security``
use the last CLOSED HTF bar (``.shift(1)`` before forward-fill). The pine used
developing (repainting) HTF values; for a no-lookahead backtest the closed-bar
convention is the honest one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import (
    GodceParams,
    REGIME_BREAKOUT,
    REGIME_CHOPPY,
    REGIME_RANGING,
    REGIME_REVERSAL,
    REGIME_TREND_BEAR,
    REGIME_TREND_BULL,
    REGIME_VOLATILE,
)
from .stateful import asian_range


# ── primitive ports ──────────────────────────────────────────────────

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def tr(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    return rma(tr(h, l, c), n)


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: float(np.dot(x, w)) / w.sum(), raw=True)


def hma(s: pd.Series, n: int) -> pd.Series:
    half = n // 2
    sqrt_n = int(round(np.sqrt(n)))
    return wma(2 * wma(s, half) - wma(s, n), sqrt_n)


def dmi(h: pd.Series, l: pd.Series, c: pd.Series, dilen: int, adxlen: int):
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    tr_ = tr(h, l, c)
    plus_di = 100 * rma(plus_dm, dilen) / rma(tr_, dilen).replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, dilen) / rma(tr_, dilen).replace(0, np.nan)
    denom = (plus_di + minus_di).replace(0, np.nan)
    adx = rma((100 * (plus_di - minus_di).abs() / denom).fillna(0.0), adxlen)
    return plus_di, minus_di, adx


def rsi(c: pd.Series, n: int) -> pd.Series:
    chg = c.diff()
    up = chg.clip(lower=0.0)
    down = -chg.clip(upper=0.0)
    rs = rma(up, n) / rma(down, n).replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def macd(c: pd.Series, fast: int, slow: int, sig: int):
    ml = ema(c, fast) - ema(c, slow)
    sl = ema(ml, sig)
    return ml, sl, ml - sl


def pivothigh(h: pd.Series, left: int, right: int) -> pd.Series:
    w = left + right + 1
    is_ph = h == h.rolling(w, center=True, min_periods=w).max()
    return h.where(is_ph).shift(right)


def pivotlow(l: pd.Series, left: int, right: int) -> pd.Series:
    w = left + right + 1
    is_pl = l == l.rolling(w, center=True, min_periods=w).min()
    return l.where(is_pl).shift(right)


def crossed_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossed_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def barssince(cond: pd.Series) -> pd.Series:
    c = cond.fillna(False).to_numpy(dtype=bool)
    idx = np.arange(len(c))
    last = np.maximum.accumulate(np.where(c, idx, -1))
    out = np.where(last >= 0, idx - last, np.nan)
    return pd.Series(out, index=cond.index)


def linreg_slope(s: pd.Series, length: int) -> pd.Series:
    """Per-bar slope of the least-squares line over the last ``length`` bars.

    Equals ``ta.linreg(s, length, 0) - ta.linreg(s, length, 1)``.
    """
    n = len(s)
    pos = np.arange(n, dtype=float)
    L = float(length)
    sy = s.rolling(length).sum()
    sjy = (s * pos).rolling(length).sum()
    start = pos - L + 1.0
    sum_x = L * (L - 1.0) / 2.0
    sum_x2 = (L - 1.0) * L * (2.0 * L - 1.0) / 6.0
    sum_xy = sjy - start * sy
    denom = L * sum_x2 - sum_x ** 2
    return (L * sum_xy - sum_x * sy) / denom


def session_vwap(price: pd.Series, volume: pd.Series, day: pd.Series) -> pd.Series:
    pv = price * volume
    return pv.groupby(day).cumsum() / volume.groupby(day).cumsum().replace(0, np.nan)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )


def htf_closed(series: pd.Series, rule: str, index: pd.DatetimeIndex) -> pd.Series:
    """HTF series at last CLOSED bar, forward-filled to ``index`` (no lookahead)."""
    return series.shift(1).reindex(index, method="ffill")


# ── main feature builder ─────────────────────────────────────────────

def compute_features(
    df: pd.DataFrame,
    dxy: pd.DataFrame,
    htf: dict[str, pd.DataFrame],
    P: GodceParams,
) -> pd.DataFrame:
    """Compute all vectorized GOD-CE features. Stateful parts are merged in
    by :func:`goldenswing.godce.backtest.run` — this returns the pure
    vectorized block."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    idx = df.index
    f: dict[str, pd.Series | pd.DataFrame] = {}

    # ── sessions (UTC) ──
    utc_hour = idx.hour
    # NFP: first Friday of the month — done RIGHT (pine's dayofweek convention was suspect)
    is_nfp = (idx.weekday == 4) & (idx.day <= 7)
    f["inAsian"] = (utc_hour >= 0) & (utc_hour < 7)
    f["inLondon"] = (utc_hour >= 7) & (utc_hour < 12)
    f["inLondonKZ"] = (utc_hour >= 7) & (utc_hour < 10)
    f["inNY"] = (utc_hour >= 12) & (utc_hour < 17)
    f["inNYKZ"] = (utc_hour >= 12) & (utc_hour < 15)
    f["inNYClose"] = (utc_hour >= 17) & (utc_hour < 20)
    f["inKillzone"] = f["inLondonKZ"] | f["inNYKZ"]
    f["sessionOk"] = True  # useKillzone=False
    f["inLondonFix"] = utc_hour == 15  # gold
    f["isNfpDay"] = is_nfp  # gold
    f["cautionMode"] = f["isNfpDay"] | f["inLondonFix"]
    f["asianHigh"], f["asianLow"] = asian_range(h, l, pd.Series(f["inAsian"], index=idx))

    # ── core engine ──
    f["atr"] = atr(h, l, c, P.atrLen)
    f["atrAvg"] = f["atr"].rolling(20).mean()
    f["atrRatio"] = f["atr"] / f["atrAvg"].replace(0, np.nan)
    f["lowVol"] = f["atrRatio"] < 0.70
    f["highVol"] = f["atrRatio"] > 1.50
    f["emaF"] = ema(c, P.fastLen)
    f["emaM"] = ema(c, P.medLen)
    f["emaS"] = ema(c, P.slowLen)
    f["emaT"] = ema(c, P.trendLen)
    f["hmaBase"] = hma(c, P.hmaLen)
    _, _, adx_line = dmi(h, l, c, P.adxLen, P.adxLen)
    f["adxLine"] = adx_line
    f["adxOk"] = adx_line > P.adxMin
    f["volSma"] = v.rolling(P.volLen).mean()
    f["volImbalance"] = v > f["volSma"] * 2.0 if P.useVolImb else v > f["volSma"] * 1.2
    f["volStrong"] = v > f["volSma"] * 1.2
    f["volOk"] = True  # useVolConf=False
    f["zScore"] = (c - c.rolling(P.zLen).mean()) / c.rolling(P.zLen).std(ddof=0).replace(0, np.nan)
    f["bullStack"] = (f["emaF"] > f["emaM"]) & (f["emaM"] > f["emaS"]) & (c > f["emaT"]) & (c > f["hmaBase"])
    f["bearStack"] = (f["emaF"] < f["emaM"]) & (f["emaM"] < f["emaS"]) & (c < f["emaT"]) & (c < f["hmaBase"])

    # ── regime ──
    reg_slope_len = max(P.regimeLen // 2, 5)
    bb_mid = c.rolling(P.regimeLen).mean()
    bb_dev = c.rolling(P.regimeLen).std(ddof=0)
    bb_width = 2 * bb_dev / bb_mid.replace(0, np.nan)
    hurst = np.log(atr(h, l, c, 10) / atr(h, l, c, 20).replace(0, 1e-9)) / np.log(2)
    hurst_norm = hurst.add(1.0).div(2.0).clip(0.0, 1.0)
    f["hurstNorm"] = hurst_norm
    slope = linreg_slope(c, reg_slope_len)
    slope_norm = slope / f["atr"].replace(0, np.nan)
    high_adx = adx_line > 25
    low_adx = adx_line < 18
    wide_bb = bb_width > bb_width.rolling(P.regimeLen).mean() * 1.3
    narrow_bb = bb_width < bb_width.rolling(P.regimeLen).mean() * 0.7
    price_above = c > f["emaS"]
    price_below = c < f["emaS"]
    conds = [
        high_adx & price_above & (slope_norm > P.regTrendTh),
        high_adx & price_below & (slope_norm < -P.regTrendTh),
        f["highVol"] & wide_bb & ~high_adx,
        f["lowVol"] & narrow_bb & low_adx,
        f["highVol"] & high_adx & (price_above | price_below) & ((slope_norm > 0.3) | (slope_norm < -0.3)),
        (crossed_above(c, f["emaS"]) | crossed_above(c, f["emaT"])) & (slope_norm.abs() < 0.1),
    ]
    f["regime"] = pd.Series(
        np.select(conds,
                  [REGIME_TREND_BULL, REGIME_TREND_BEAR, REGIME_VOLATILE,
                   REGIME_CHOPPY, REGIME_BREAKOUT, REGIME_REVERSAL],
                  default=REGIME_RANGING),
        index=idx, dtype=int,
    )
    f["regimeVolatile"] = f["regime"].isin([REGIME_VOLATILE, REGIME_BREAKOUT])

    # ── elastic EMA (Hurst-driven recursion) ──
    el_len = np.where(
        hurst_norm > 0.66, P.minElastic,
        np.where(hurst_norm > 0.34, (P.minElastic + P.maxElastic) // 2, P.maxElastic),
    ).astype(float)
    f["elasticLen"] = pd.Series(el_len, index=idx)
    ee = np.full(len(c), np.nan)
    cv = c.to_numpy()
    prev = np.nan
    for i in range(len(c)):
        L = el_len[i]
        if np.isnan(L):
            continue
        alpha = 2.0 / (L + 1.0)
        prev = cv[i] if np.isnan(prev) else alpha * cv[i] + (1 - alpha) * prev
        ee[i] = prev
    f["emaElastic"] = pd.Series(ee, index=idx)

    # ── PD array & institutional levels ──
    f["rangeH"] = h.rolling(50).max()
    f["rangeL"] = l.rolling(50).min()
    f["rangeEQ"] = (f["rangeH"] + f["rangeL"]) / 2
    f["inPremium"] = c > f["rangeEQ"]
    f["inDiscount"] = c < f["rangeEQ"]
    d1 = htf["d1"]
    f["pdHigh"] = htf_closed(d1["high"], "1D", idx)
    f["pdLow"] = htf_closed(d1["low"], "1D", idx)
    f["pdClose"] = htf_closed(d1["close"], "1D", idx)
    w1 = htf["w1"]
    f["pwHigh"] = htf_closed(w1["high"], "1W", idx)
    f["pwLow"] = htf_closed(w1["low"], "1W", idx)
    f["weekOpen"] = htf_closed(w1["open"], "1W", idx)
    f["dailyPivot"] = (f["pdHigh"] + f["pdLow"] + f["pdClose"]) / 3
    f["recentSwingLow"] = l.rolling(5).min()
    f["recentSwingHigh"] = h.rolling(5).max()

    # ── macro (DXY, inverse for gold) ──
    dx = dxy["close"].reindex(idx, method="ffill")
    dx_e50, dx_e200 = ema(dx, 50), ema(dx, 200)
    macro_valid = dx.notna() & dx_e50.notna()
    f["macroValid"] = macro_valid & P.useMacro
    # inverse: DXY down = gold bullish
    f["macroBullish"] = f["macroValid"] & (dx < dx_e50) & (dx_e50 < dx_e200)
    f["macroBearish"] = f["macroValid"] & (dx > dx_e50) & (dx_e50 > dx_e200)

    # ── multi-timeframe bias ──
    bE50, bE200 = ema(c, 50), ema(c, 200)
    f["tf1Bull"] = (c > bE50) & (bE50 > bE200)
    f["tf1Bear"] = (c < bE50) & (bE50 < bE200)
    h4 = htf["h4"]
    h4_e50, h4_e200, h4_c = ema(h4["close"], 50), ema(h4["close"], 200), h4["close"]
    f["tf2Bull"] = htf_closed((h4_c > h4_e50) & (h4_e50 > h4_e200), "4h", idx).fillna(False)
    f["tf2Bear"] = htf_closed((h4_c < h4_e50) & (h4_e50 < h4_e200), "4h", idx).fillna(False)
    d_e50, d_e200, d_c = ema(d1["close"], 50), ema(d1["close"], 200), d1["close"]
    f["tf3Bull"] = htf_closed((d_c > d_e50) & (d_e50 > d_e200), "1D", idx).fillna(False)
    f["tf3Bear"] = htf_closed((d_c < d_e50) & (d_e50 < d_e200), "1D", idx).fillna(False)
    f["biasScore"] = (
        f["tf1Bull"].astype(int) + f["tf2Bull"].astype(int) + f["tf3Bull"].astype(int)
        - f["tf1Bear"].astype(int) - f["tf2Bear"].astype(int) - f["tf3Bear"].astype(int)
    )
    bs = f["biasScore"]
    f["overallBias"] = pd.Series(np.select([bs >= 2, bs <= -2], [1, -1], default=0), index=idx)

    # ── HTF order-block feed (4h impulse) ──
    h4_atr = rma(tr(h4["high"], h4["low"], h4["close"]), P.atrLen)
    h4_bull = (h4["close"] - h4["open"]) > h4_atr * 1.5
    h4_bull = h4_bull & (h4["close"] > h4["high"].shift(1)) & (h4["close"].shift(1) < h4["open"].shift(1))
    h4_bear = (h4["open"] - h4["close"]) > h4_atr * 1.5
    h4_bear = h4_bear & (h4["close"] < h4["low"].shift(1)) & (h4["close"].shift(1) > h4["open"].shift(1))
    f["htfBullImp"] = htf_closed(h4_bull, "4h", idx).fillna(False)
    f["htfBearImp"] = htf_closed(h4_bear, "4h", idx).fillna(False)
    f["htfObH"] = htf_closed(h4["high"].shift(1), "4h", idx)
    f["htfObL"] = htf_closed(h4["low"].shift(1), "4h", idx)
    h4_period = h4.index.floor("4h").to_numpy()
    is_new = pd.Series(np.concatenate([[True], h4_period[1:] != h4_period[:-1]]), index=h4.index)
    f["isNewHtfBar"] = is_new.reindex(idx, method="ffill").fillna(False)

    # ── raw pivots (confirmed) ──
    f["ph"] = pivothigh(h, P.swingLen, P.swingLen)
    f["pl"] = pivotlow(l, P.swingLen, P.swingLen)

    # ── oscillators ──
    ml, sl_, hist = macd(c, P.macdFast, P.macdSlow, P.macdSig)
    f["macdAbove"] = ml > sl_
    f["macdBelow"] = ml < sl_
    f["macdCross"] = crossed_above(ml, sl_)
    f["macdDeadX"] = crossed_below(ml, sl_)
    f["macdHistUp"] = hist > hist.shift(1)
    f["bullDiv"] = (l.shift(1) < l.shift(5)) & (ml.shift(1) > ml.shift(5)) & (l.shift(1) == l.rolling(5).min().shift(1))
    f["bearDiv"] = (h.shift(1) > h.shift(5)) & (ml.shift(1) < ml.shift(5)) & (h.shift(1) == h.rolling(5).max().shift(1))
    f["rsiVal"] = rsi(c, P.rsiLen)
    f["rsiOB"] = f["rsiVal"] >= P.rsiOBLvl
    f["rsiOS"] = f["rsiVal"] <= P.rsiOSLvl
    f["rsiBullDiv"] = (l.shift(1) < l.shift(5)) & (f["rsiVal"].shift(1) > f["rsiVal"].shift(5)) & (l.shift(1) == l.rolling(5).min().shift(1))
    f["rsiBearDiv"] = (h.shift(1) > h.shift(5)) & (f["rsiVal"].shift(1) < f["rsiVal"].shift(5)) & (h.shift(1) == h.rolling(5).max().shift(1))
    day = pd.Series(idx.date, index=idx)
    f["vwapVal"] = session_vwap((h + l + c) / 3, v, day)
    f["aboveVwap"] = c > f["vwapVal"]
    f["vwapHigh"] = session_vwap(h, v, day)
    f["vwapLow"] = session_vwap(l, v, day)
    f["aboveValue"] = c > f["vwapHigh"]
    f["belowValue"] = c < f["vwapLow"]

    # ── Ichimoku ──
    def _ichi(hh, ll):
        tk = (hh.rolling(9).max() + ll.rolling(9).min()) / 2
        kj = (hh.rolling(26).max() + ll.rolling(26).min()) / 2
        spa = (tk + kj) / 2
        spb = (hh.rolling(52).max() + ll.rolling(52).min()) / 2
        return tk, kj, spa, spb

    tk, kj, spa, spb = _ichi(h, l)
    f["ichiTk"] = tk
    f["ichiKj"] = kj
    f["tkKjGold"] = crossed_above(tk, kj)
    f["tkKjDeath"] = crossed_below(tk, kj)
    dtk, dkj, dspA, dspB = _ichi(d1["high"], d1["low"])
    f["dayKumoTop"] = htf_closed(pd.Series(np.maximum(dspA.shift(26), dspB.shift(26)), index=d1.index), "1D", idx)
    f["dayKumoBot"] = htf_closed(pd.Series(np.minimum(dspA.shift(26), dspB.shift(26)), index=d1.index), "1D", idx)

    # ── footprint / dark pool inputs ──
    delta = (c - o) * v / (h - l).replace(0, np.nan)
    cumd = delta.cumsum()
    davg = cumd.rolling(50).mean()
    f["deltaZ"] = (cumd - davg) / cumd.rolling(50).std(ddof=0).replace(0, np.nan)
    f["aggressiveBuy"] = (delta > 0) & (v > f["volSma"] * 1.5)
    f["aggressiveSell"] = (delta < 0) & (v > f["volSma"] * 1.5)
    f["absorption"] = (c - o).abs() < f["atr"] * 0.2
    f["absorption"] = f["absorption"] & (v > f["volSma"] * 1.3)

    # ── candles ──
    body = (c - o).abs()
    uw = h - pd.concat([c, o], axis=1).max(axis=1)
    lw = pd.concat([c, o], axis=1).min(axis=1) - l
    rng = h - l
    f["bullPin"] = (rng > f["atr"] * 0.3) & (lw > body * 2.0) & (lw > uw * 2.0)
    f["bearPin"] = (rng > f["atr"] * 0.3) & (uw > body * 2.0) & (uw > lw * 2.0)
    f["bullEngulf"] = (c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1)) & (body > (c.shift(1) - o.shift(1)).abs())
    f["bearEngulf"] = (c < o) & (c.shift(1) > o.shift(1)) & (c < o.shift(1)) & (o > c.shift(1)) & (body > (c.shift(1) - o.shift(1)).abs())
    inside = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))
    f["ibBullBreak"] = inside.shift(1) & (c > h.shift(1))
    f["ibBearBreak"] = inside.shift(1) & (c < l.shift(1))

    # ── V-reversal detector inputs ──
    f["volC"] = v / f["volSma"].replace(0, np.nan)
    f["vPreHigh"] = h.rolling(15).max()
    f["vPreLow"] = l.rolling(15).min()
    f["vSpikeBot"] = l.rolling(12).min()
    f["vSpikeTop"] = h.rolling(12).max()

    return pd.DataFrame(f, index=idx)


# ── V-reversal triggers + sentiment (need barssince / structure) ─────

def v_reversal_triggers_df(df: pd.DataFrame, feat: pd.DataFrame, P: GodceParams) -> pd.DataFrame:
    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    atr = feat["atr"]
    volC = feat["volC"]
    vLook, vSpikeA, vWin = 12, 2.5, 3
    vPreHigh = h.rolling(vLook + vWin).max()
    vPreLow = l.rolling(vLook + vWin).min()
    vSpikeBot = l.rolling(vLook).min()
    vSpikeTop = h.rolling(vLook).max()
    vDrop = vPreHigh - vSpikeBot
    vPump = vSpikeTop - vPreLow
    barsBot = barssince(l <= vSpikeBot)
    barsTop = barssince(h >= vSpikeTop)
    out = pd.DataFrame(index=df.index)
    out["vBotTrig"] = (vDrop >= vSpikeA * atr) & (barsBot >= 1) & (barsBot <= vWin) & \
        (c - vSpikeBot >= 0.5 * vDrop) & (c > o) & (volC >= 1.2)
    out["vTopTrig"] = (vPump >= vSpikeA * atr) & (barsTop >= 1) & (barsTop <= vWin) & \
        (vSpikeTop - c >= 0.5 * vPump) & (c < o) & (volC >= 1.2)
    out["vBotBreak"] = (vDrop >= vSpikeA * atr) & (barsBot >= 1) & (barsBot <= vWin) & \
        (c > vPreHigh) & (volC >= 1.2)
    out["vTopBreak"] = (vPump >= vSpikeA * atr) & (barsTop >= 1) & (barsTop <= vWin) & \
        (c < vPreLow) & (volC >= 1.2)
    return out.fillna(False)


def add_sentiment(feat: pd.DataFrame, struct: pd.DataFrame) -> pd.Series:
    """Contrarian sentiment vector (-1..+1), needs structure booleans."""
    s = pd.Series(0.0, index=feat.index)
    s = s + np.where(feat["rsiVal"] > 50, 0.2, np.where(feat["rsiVal"] < 50, -0.2, 0.0))
    s = s + np.where(feat["macdAbove"], 0.2, np.where(feat["macdBelow"], -0.2, 0.0))
    s = s + np.where((feat["adxLine"] > 25) & (feat["close"] > feat["emaS"]), 0.3,
             np.where((feat["adxLine"] > 25) & (feat["close"] < feat["emaS"]), -0.3, 0.0))
    s = s + np.where(struct["bullStruct"], 0.3, np.where(struct["bearStruct"], -0.3, 0.0))
    return s.clip(-1.0, 1.0)


# ── Wyckoff + volume divergence (need structure's lastPL2) ──────────

def wyckoff_divergence(df: pd.DataFrame, feat: pd.DataFrame, P: GodceParams) -> pd.DataFrame:
    """Volume divergence + Wyckoff spring/UTAD. Vectorized port of the pine's
    volBullDiv/volBearDiv/wyckSpring/wyckUTAD (needs lastPL2 from structure)."""
    l, h, c = df["low"], df["high"], df["close"]
    v = df["volume"]
    out = pd.DataFrame(index=df.index)
    out["volBullDiv"] = (P.useVolDiv & (l.shift(1) < l.shift(5)) & (v.shift(1) < v.shift(5))
                         & (l.shift(1) == l.rolling(5).min().shift(1))).fillna(False)
    out["volBearDiv"] = (P.useVolDiv & (h.shift(1) > h.shift(5)) & (v.shift(1) < v.shift(5))
                         & (h.shift(1) == h.rolling(5).max().shift(1))).fillna(False)
    rangeH = h.rolling(50).max()
    rangeL = l.rolling(50).min()
    rangeEQ = (rangeH + rangeL) * 0.5
    out["wyckSpring"] = (P.useWyckoff & feat["lastPL2"].notna()
                         & (l < feat["lastPL2"]) & (c > feat["lastPL2"])
                         & (v > feat["volSma"] * 1.3) & feat["inDiscount"]).fillna(False)
    out["wyckUTAD"] = (P.useWyckoff & (h > rangeH) & (c < rangeEQ)
                       & (v < feat["volSma"] * 0.8) & feat["inPremium"]).fillna(False)
    return out
