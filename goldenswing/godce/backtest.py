"""GOD Conviction Engine — backtest orchestration.

Data loading (GC H1 + DXY H1, tz-aligned to UTC), HTF resampling, the full
feature pipeline, walk-forward folds, metrics and the MAP breakdowns
(session / regime / router mode / weekday / side).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .indicators import (
    compute_features,
    htf_closed,
    resample_ohlc,
    v_reversal_triggers_df,
    add_sentiment,
    wyckoff_divergence,
)
from .params import MODE_NAMES, REGIME_NAMES, GodceParams
from .router import apply_dwell, compute_candidate_mode, compute_mode_signals
from .scoring import compute_scores
from .stateful import (
    geometry_features,
    hs_features,
    session_sense,
    structure_features,
    zones_traps_features,
)
from .engine import run_engine


def load_ohlc(path: str, tz: str) -> pd.DataFrame:
    """Load a CSV with time,open,high,low,close,volume → tz-aware UTC index."""
    df = pd.read_csv(path)
    tcol = next(c for c in df.columns if c.lower() in ("time", "timestamp", "date", "datetime"))
    ts = pd.to_datetime(df[tcol], utc=True)  # mixed -04:00/-05:00 offsets → UTC
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df.index = ts.dt.tz_convert("UTC")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def build_features(df: pd.DataFrame, dxy: pd.DataFrame, P: GodceParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full GOD-CE feature pipeline. Returns (df, feat)."""
    htf = {"h4": resample_ohlc(df, "4h"), "d1": resample_ohlc(df, "1D"),
           "w1": resample_ohlc(df, "1W")}
    feat = compute_features(df, dxy, htf, P)
    feat = pd.concat([df[["open", "high", "low", "close", "volume"]], feat], axis=1)
    feat = pd.concat([feat, session_sense(df.index)], axis=1)

    # killzone / session / volume gates (ports of SESSION SENSE + CORE ENGINE)
    hr = df.index.hour.to_numpy()
    in_kz = ((hr >= 7) & (hr < 10)) | ((hr >= 12) & (hr < 15))
    feat["sessionOk"] = (~P.useKillzone) | in_kz
    feat["volOk"] = (~P.useVolConf) | (df["volume"] > feat["volSma"] * 1.2)
    feat["volOk"] = feat["volOk"].fillna(True)

    struct = structure_features(
        df["high"], df["low"], df["close"], df["open"],
        feat["ph"], feat["pl"], feat["atr"],
        P.eqTol, feat["asianHigh"], feat["asianLow"])
    feat = pd.concat([feat, struct], axis=1)

    zones = zones_traps_features(df, feat, P)
    feat = pd.concat([feat, zones], axis=1)

    hs = hs_features(df["high"], df["low"], df["close"],
                     P.swingLen, P.hsAtrTol, feat["atr"])
    feat = pd.concat([feat, hs.rename(columns=lambda c: "hs_" + c)], axis=1)
    # daily H&S via closed-bar convention
    d = resample_ohlc(df, "1D")
    d_atr = d["high"].sub(d["low"]).rolling(P.atrLen).mean()  # simple daily ATR
    hs_d = hs_features(d["high"], d["low"], d["close"],
                       P.swingLen, P.hsAtrTol, d_atr)
    feat["hsD_Tst"] = htf_closed(hs_d["hsT_state"], "1D", df.index).fillna(0)
    feat["hsD_Bst"] = htf_closed(hs_d["hsB_state"], "1D", df.index).fillna(0)

    # legacy aliases expected by scoring (hsT_state etc. without prefix)
    feat["hsT_state"] = feat["hs_hsT_state"]
    feat["hsT_broke"] = feat["hs_hsT_broke"]
    feat["hsB_state"] = feat["hs_hsB_state"]
    feat["hsB_broke"] = feat["hs_hsB_broke"]

    geo = geometry_features(df, feat, P)
    feat = pd.concat([feat, geo], axis=1)

    vrev = v_reversal_triggers_df(df, feat, P)
    feat = pd.concat([feat, vrev], axis=1)

    wd = wyckoff_divergence(df, feat, P)
    feat = pd.concat([feat, wd], axis=1)

    feat["sentiment"] = add_sentiment(feat, struct)

    scores = compute_scores(feat, P)
    feat = pd.concat([feat, scores], axis=1)

    sig = compute_mode_signals(feat)
    feat["candMode"] = compute_candidate_mode(feat, P)
    feat["mode"] = apply_dwell(feat["candMode"], P)
    return df, feat, sig


def run_backtest(gold_path: str, dxy_path: str, P: GodceParams | None = None):
    """Load data, build features, run the engine. Returns (trades, df, feat)."""
    P = P or GodceParams()
    df = load_ohlc(gold_path, "America/New_York")
    dxy = load_ohlc(dxy_path, "UTC")
    df, feat, sig = build_features(df, dxy, P)
    trades = run_engine(df, feat, sig, P)
    return trades, df, feat


# ── metrics + MAP ────────────────────────────────────────────────────

def summarize(trades: pd.DataFrame, risk_dollars: float = 100.0) -> dict:
    if trades.empty:
        return {"n": 0}
    t = trades.copy()
    wins = t[t["net_R"] > 0]
    ret_pct = t["pnl_usd"].sum() / 10000.0 * 100.0
    avg_r = t["net_R"].mean()
    win_rate = len(wins) / len(t)
    gross_win = wins["net_R"].sum()
    gross_loss = -t[t["net_R"] <= 0]["net_R"].sum()
    expectancy = avg_r
    # max drawdown in R on cumulative net_R
    cum = t["net_R"].cumsum()
    dd = (cum.cummax() - cum).max()
    return {
        "n": len(t),
        "win_rate": win_rate,
        "avg_R": avg_r,
        "expectancy_R": expectancy,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "total_R": t["net_R"].sum(),
        "total_pnl_usd": t["pnl_usd"].sum(),
        "ret_pct_on_10k": ret_pct,
        "max_dd_R": dd,
        "by_reason": t.groupby("exit_reason")["net_R"].agg(["count", "sum", "mean"]).to_dict(),
    }


def map_breakdown(trades: pd.DataFrame) -> dict:
    """The MAP: performance by session, regime, router mode, weekday, side."""
    if trades.empty:
        return {}
    t = trades.copy()
    out = {}
    for key in ("session", "regime", "mode", "dow", "side", "exit_reason"):
        g = t.groupby(key)["net_R"].agg(["count", "sum", "mean"])
        g["win_rate"] = t.groupby(key).apply(lambda x: (x["net_R"] > 0).mean(), include_groups=False)
        out[key] = g.round(3).to_dict(orient="index")
    return out


def walk_forward(gold_path: str, dxy_path: str, P: GodceParams | None = None,
                 train_months: int = 12, test_months: int = 3):
    """Expanding walk-forward: train expands by test_months each fold.
    (GOD-CE v1 has no tunable params in v1, so 'train' is a formality —
    features are computed on the full history with closed-bar HTF, and
    trades are evaluated only on the test window. No OOS tuning.

    The engine runs on each test window with warmupBars=0: features are
    already stabilized (computed on full history), so no warmup is needed.)"""
    P = P or GodceParams()
    df = load_ohlc(gold_path, "America/New_York")
    dxy = load_ohlc(dxy_path, "UTC")
    df, feat, sig = build_features(df, dxy, P)
    P_wf = dataclasses.replace(P, warmupBars=0)

    start = df.index.min() + pd.DateOffset(months=train_months)
    end = df.index.max()
    folds = []
    fold_start = start
    k = 0
    while fold_start < end:
        fold_end = fold_start + pd.DateOffset(months=test_months)
        mask = (df.index >= fold_start) & (df.index < fold_end)
        sub = df[mask]
        if len(sub) < 100:
            break
        trades = run_engine(df[mask], feat[mask], sig[mask], P_wf)
        trades["fold"] = k
        folds.append(trades)
        k += 1
        fold_start = fold_end
    all_trades = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()
    return all_trades, df, feat
