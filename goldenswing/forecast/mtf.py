"""Genuine multi-timeframe forecast core.

The forecast is built from the engine's ACTUAL timeframe stack —
M5, M15, H1, H4, D1 closed bars from MT5 — not from one H1 frame
with higher timeframes resampled out of it. H1 bars cannot
reconstruct M5/M15 candle detail, and resampled H4/D1 context is
not the engine's real H4/D1 frames. Every feature below is
trailing-only (computed from bars at or before the row's time),
aligned to H1 row times by as-of merge, so no feature ever reads
a future bar. Labels still count H1 bars (horizons are defined in
H1 bars), but their inputs are multi-timeframe.

Bundle format v2 ("forecast-bundle-v2", feature_set "mtf-v1").
Ranks are percentiles of a fresh score against the TRAINING rows'
score distribution (reference quantiles stored in the bundle), so
a single fresh row can be scored without a history frame.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import HORIZONS, build_labels_for_horizon, spec_for
from .walkforward import make_models

BUNDLE_VERSION = "forecast-bundle-v2"
FEATURE_SET = "mtf-v1"

# Timeframes the forecast reads, in stack-frame key form.
MTF_TFS = ("M5", "M15", "H1", "H4", "D1")

# Per-timeframe trailing windows (in that TF's own bars). The
# stacked warmup is 63 bars: ATR needs 14 bars, and the trailing
# ATR percentile needs 50 valid ATR values after that (14 + 50 - 1).
# So D1 context needs 63 days of history before a row is usable.
# Rows before every TF's warmup are dropped.
_TF_WINDOWS = {"pct": 50, "range": 20, "ema_fast": 9, "ema_slow": 21}

# Retrain trigger (build default, tunable — not an owner ruling):
# 500 new H1 bars ≈ 3 weeks of hourly data. Tied to the forecast
# itself: the longest horizon is 360 bars (15 days), so retraining
# only after 500 genuinely new observations means the model has
# seen substantially new data relative to what it predicts over;
# the cost (minutes of background fitting) is paid roughly monthly.
MIN_NEW_BARS_RETRAIN = 500

# Minimum usable MTF rows before a training thread even starts —
# the same floor the trainer itself enforces. This is deliberately a
# USABLE-row floor, not a raw-H1-bar floor: the old 2012 raw-bar
# number assumed every TF's window starts cold (63 days of D1 warmup
# ≈ 1512 H1 bars + 500 usable), but the live stack's D1/H4 frames
# run years deep, so only H1's own 63-bar warmup binds there and a
# 1999-bar H1 stack already yields ~1900 usable rows. Measuring
# usable rows keeps the statistical floor intact without refusing a
# trainable stack.
MIN_USABLE_ROWS_TRAIN = 500


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Trailing average true range (Wilder-style rolling mean)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _tf_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Trailing-only features for one timeframe's closed bars.

    Every column uses bars at or before its own row: returns,
    ATR ratios, EMA trend/position, trailing ATR percentile, and
    position inside the trailing range. Nothing centered, nothing
    ranked over the full series — appending future bars cannot
    change already-computed rows (proven by test, not asserted).
    """
    df = df.sort_index()
    close, open_ = df["close"], df["open"]
    atr = _atr(df)
    atr_n = atr / close.replace(0, np.nan)
    ema_fast = close.ewm(span=_TF_WINDOWS["ema_fast"], adjust=False).mean()
    ema_slow = close.ewm(span=_TF_WINDOWS["ema_slow"], adjust=False).mean()
    w = _TF_WINDOWS["pct"]
    r = _TF_WINDOWS["range"]
    hi, lo = df["high"].rolling(r).max(), df["low"].rolling(r).min()
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_ret1"] = close.pct_change(1)
    out[f"{prefix}_ret5"] = close.pct_change(5)
    out[f"{prefix}_ret20"] = close.pct_change(20)
    out[f"{prefix}_atr_n"] = atr_n
    out[f"{prefix}_atr_pct"] = atr_n.rolling(w).rank(pct=True)
    out[f"{prefix}_trend"] = (ema_fast - ema_slow) / atr.replace(0, np.nan)
    out[f"{prefix}_pos"] = (close - ema_slow) / atr.replace(0, np.nan)
    out[f"{prefix}_range_pos"] = (close - lo) / (hi - lo).replace(0, np.nan)
    out[f"{prefix}_body"] = (close - open_) / atr.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def build_mtf_frame(frames: dict) -> pd.DataFrame:
    """One feature row per H1 bar, from the actual stack frames.

    ``frames`` maps TF -> closed-bar OHLC DataFrame (UTC
    DatetimeIndex, sorted). H1 sets the row times; every other TF
    contributes its last closed bar at or before each H1 time
    (as-of merge — never a future bar). A TF with no frame is
    skipped (its columns simply don't exist); rows missing any
    present TF's warmup are dropped. Returns the usable rows only.
    """
    h1 = frames.get("H1")
    if h1 is None or len(h1) == 0:
        raise ValueError("cannot build MTF frame: no H1 frame")
    if not isinstance(h1.index, pd.DatetimeIndex):
        raise ValueError("cannot build MTF frame: H1 needs a DatetimeIndex")
    base = pd.DataFrame(index=h1.sort_index().index)
    blocks = []
    for tf in MTF_TFS:
        frame = frames.get(tf)
        if frame is None or len(frame) == 0:
            continue
        feats = _tf_features(frame, tf.lower())
        merged = pd.merge_asof(
            base, feats, left_index=True, right_index=True,
            direction="backward")
        blocks.append(merged)
    if not blocks:
        raise ValueError("cannot build MTF frame: no usable TF frames")
    X = pd.concat(blocks, axis=1)
    # Cross-timeframe agreement: share of available TFs trending up.
    trend_cols = [c for c in X.columns if c.endswith("_trend")]
    if trend_cols:
        X["x_trend_agree"] = (X[trend_cols] > 0).mean(axis=1)
    return X.dropna()


def _purge_mask(n: int, cut: int, horizon: int) -> np.ndarray:
    """Chronological train mask with a horizon-aware purge.

    A bar at index i carries a label window over bars (i, i+h].
    The bar is train-eligible only when its whole window ends
    BEFORE the holdout starts (i + h < cut) — otherwise its label
    reads holdout-period outcomes. Without this purge, training
    rows near the split leak the holdout into the reported metrics.
    """
    idx = np.arange(n)
    return (idx < cut) & (idx + horizon < cut)


def _validate_chronological(X: pd.DataFrame, h1: pd.DataFrame, atr,
                            *, holdout_frac: float = 0.2) -> dict:
    """Chronological holdout validation, per horizon, on the MTF
    frame's rows. Train rows are the first (1 - holdout_frac) of
    usable rows, purged by horizon (see _purge_mask); holdout rows
    are the last holdout_frac with known labels. AUCs are None when
    the holdout has too few labelled bars or a single class — the
    longest horizon on short history legitimately yields None.
    Validation failure records missing AUCs; it never fails
    training, because shipped models don't depend on validation."""
    from sklearn.metrics import roc_auc_score

    n = len(X)
    cut = int(n * (1.0 - holdout_frac))
    Xv = X.to_numpy(dtype=float)
    out: dict[int, dict] = {}
    for h in HORIZONS:
        labels = build_labels_for_horizon(h1, atr, h)
        # Labels align to H1 bars; reindex to the MTF frame's rows.
        y_up = labels["y_up"].reindex(X.index)
        y_dn = labels["y_dn"].reindex(X.index)
        known = y_up.notna().to_numpy()
        train_mask = known & _purge_mask(n, cut, h)
        hold_mask = known & (np.arange(n) >= cut)
        entry: dict = {
            "n_train": int(train_mask.sum()),
            "n_holdout": int(hold_mask.sum()),
            "auc_up": None, "auc_dn": None,
        }
        if train_mask.sum() >= 50 and hold_mask.sum() >= 10:
            try:
                val = _fit_on_rows(Xv, y_up, y_dn, h, train_mask)
                for dname, y in (("up", y_up), ("dn", y_dn)):
                    yy = y.to_numpy(dtype=float)[hold_mask]
                    if len(np.unique(yy)) < 2:
                        continue
                    ps = [np.asarray(m.predict_proba(Xv[hold_mask]),
                                     dtype=float)[:, 1]
                          for m in val["models"][dname].values()]
                    p = np.mean(ps, axis=0) if ps else np.full(
                        hold_mask.sum(), np.nan)
                    entry[f"auc_{dname}"] = round(
                        float(roc_auc_score(yy, p)), 4)
            except Exception:  # noqa: BLE001 — recorded as missing AUCs
                pass
        out[h] = entry
    return out


def _fit_on_rows(Xv, y_up, y_dn, horizon: int, rows) -> dict:
    """Fit one horizon's models on explicit MTF-frame rows."""
    spec = spec_for(horizon)
    models: dict[str, dict] = {}
    base_rates: dict[str, float] = {}
    for dname, y in (("up", y_up), ("dn", y_dn)):
        yy = y.to_numpy(dtype=float)[rows]
        models[dname] = {}
        for name, model in make_models(0).items():
            model.fit(Xv[rows], yy)
            models[dname][name] = model
        base_rates[dname] = float(yy.mean())
    return {"models": models, "base_rates": base_rates,
            "spec": asdict(spec), "labelled_bars": int(rows.sum())}


def _validate_bundle(bundle) -> tuple[bool, str]:
    """Full bundle validation. A dict with a 'features' key is NOT
    enough: every model, horizon, and schema entry must be present
    and well-formed, or the bundle is not READY. Old (v1) bundles
    fail here on version — and are retrained automatically."""
    if not isinstance(bundle, dict):
        return False, "bundle is not a dict"
    if bundle.get("version") != BUNDLE_VERSION:
        return False, (f"bundle version {bundle.get('version')!r} "
                       f"!= {BUNDLE_VERSION!r}")
    if bundle.get("symbol") != "XAUUSD":
        return False, "bundle symbol is not XAUUSD"
    feats = bundle.get("features")
    if not isinstance(feats, list) or not feats:
        return False, "bundle has no feature list"
    if bundle.get("feature_set") != FEATURE_SET:
        return False, "bundle is not an MTF feature set"
    models = bundle.get("models")
    if not isinstance(models, dict) or not models.get("up") \
            or not models.get("dn"):
        return False, "bundle lacks up/dn models"
    horizons = bundle.get("horizons")
    if not isinstance(horizons, dict):
        return False, "bundle lacks horizons"
    for h in HORIZONS:
        entry = horizons.get(h) or horizons.get(str(h))
        if not isinstance(entry, dict):
            return False, f"bundle lacks horizon {h}"
        mh = entry.get("models")
        if not isinstance(mh, dict) or not mh.get("up") or not mh.get("dn"):
            return False, f"bundle horizon {h} lacks up/dn models"
    try:
        pd.Timestamp(bundle.get("trained_through"))
    except Exception:  # noqa: BLE001 — unparseable stamp is invalid
        return False, "bundle trained_through unparseable"
    if not bundle.get("trained_at"):
        return False, "bundle lacks trained_at"
    quantiles = bundle.get("ref_quantiles")
    if not isinstance(quantiles, dict):
        return False, "bundle lacks reference quantiles"
    for h in HORIZONS:
        entry = quantiles.get(h) or quantiles.get(str(h))
        if not isinstance(entry, dict) or not entry.get("up") \
                or not entry.get("dn"):
            return False, f"bundle lacks reference quantiles for {h}"
    return True, "bundle valid"


def _reference_quantiles(Xv, horizons: dict) -> dict:
    """Per-horizon/direction score distribution of the shipped
    models on their own training rows (101 percentiles each).
    Fresh-row ranks are percentiles against this distribution."""
    quantiles: dict[int, dict] = {}
    for h, entry in horizons.items():
        quantiles[h] = {}
        for dname in ("up", "dn"):
            ps = [np.asarray(m.predict_proba(Xv), dtype=float)[:, 1]
                  for m in entry["models"][dname].values()]
            p = np.mean(ps, axis=0) if ps else np.full(len(Xv), np.nan)
            quantiles[h][dname] = np.nanpercentile(
                p, np.linspace(0, 100, 101)).tolist()
    return quantiles


def _rank_from_quantiles(score: float, quantiles) -> float:
    """Percentile rank of one fresh score against 101 reference
    percentiles (0-100). NaN score -> NaN rank."""
    if score != score:  # NaN
        return float("nan")
    q = np.asarray(quantiles, dtype=float)
    return float(100.0 * (q <= score).mean())


def train_from_frames(frames: dict, out_path: str | Path) -> dict:
    """Train a v2 MTF bundle from the engine's own stack frames and
    save it atomically. Raises ValueError on failed data-quality
    gates — the caller records the precise reason and the old bundle
    (if any) stays untouched. Returns a metadata summary."""
    import joblib

    h1 = frames.get("H1")
    if h1 is None or len(h1) == 0:
        raise ValueError("cannot train: no H1 frame")
    if not isinstance(h1.index, pd.DatetimeIndex):
        raise ValueError("cannot train: H1 frame needs a DatetimeIndex")
    for col in ("open", "high", "low", "close"):
        if col not in h1.columns:
            raise ValueError(f"cannot train: H1 frame lacks '{col}'")
    X = build_mtf_frame(frames)
    if len(X) < MIN_USABLE_ROWS_TRAIN:
        raise ValueError(
            f"cannot train: too few usable MTF rows "
            f"({len(X)} < {MIN_USABLE_ROWS_TRAIN})")
    h1 = h1.sort_index()
    atr = _atr(h1).to_numpy(dtype=float)
    # Labels align to H1 bars; reindex to the MTF frame's rows.
    keep = [c for c in X.columns
            if X[c].notna().sum() > 0 and X[c].dropna().nunique() >= 2]
    if not keep:
        raise ValueError("no varying features in frame — cannot train")
    Xv = X[keep].to_numpy(dtype=float)
    horizons: dict[int, dict] = {}
    for h in HORIZONS:
        labels = build_labels_for_horizon(h1, atr, h)
        y_up = labels["y_up"].reindex(X.index)
        y_dn = labels["y_dn"].reindex(X.index)
        known = y_up.notna().to_numpy()
        horizons[h] = _fit_on_rows(Xv, y_up, y_dn, h, known)
        horizons[h]["definition"] = (
            f"big move = favourable excursion >= "
            f"{spec_for(h).fav_mult:g}x ATR before adverse >= "
            f"{spec_for(h).adv_mult:g}x ATR within "
            f"{spec_for(h).horizon} H1 bars")
    try:
        validation = _validate_chronological(X[keep], h1, atr)
    except Exception as exc:  # noqa: BLE001 — recorded, not fatal
        validation = {"error": f"validation failed: {exc}"}
    legacy = horizons[72]
    bundle = {
        "version": BUNDLE_VERSION,
        "feature_set": FEATURE_SET,
        "symbol": "XAUUSD",
        "models": legacy["models"],
        "features": keep,
        "base_rates": legacy["base_rates"],
        "train_bars": int(len(X)),
        "train_start": str(X.index[0]),
        "trained_through": str(X.index[-1]),
        "trained_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "definition": legacy["definition"],
        "horizons": horizons,
        "validation": validation,
        "ref_quantiles": _reference_quantiles(Xv, horizons),
        "train_frame": "mt5_stack",
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic swap: a concurrent scorer (mtime-cached bundle reload)
    # must never read a half-written bundle. On failure the old
    # bundle stays untouched.
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    joblib.dump(bundle, tmp_path)
    os.replace(tmp_path, out_path)
    return {
        "saved": str(out_path),
        "version": BUNDLE_VERSION,
        "feature_set": FEATURE_SET,
        "symbol": "XAUUSD",
        "trained_at": bundle["trained_at"],
        "train_bars": bundle["train_bars"],
        "trained_through": bundle["trained_through"],
        "horizons": {h: horizons[h]["labelled_bars"] for h in HORIZONS},
        "base_rates": legacy["base_rates"],
        "validation": validation,
        "dropped_constant_features": [c for c in X.columns if c not in keep],
    }


def score_last_row(frames: dict, bundle: dict) -> tuple[dict | None, str]:
    """Score the last H1 row's MTF features against a validated
    bundle. Returns (ForecastRanks-like dict, note) — ranks are
    percentiles against the training distribution. Returns
    (None, reason) when scoring is impossible."""
    from .mt5 import ForecastRanks  # local import: mt5 owns the rank type

    try:
        X = build_mtf_frame(frames)
    except Exception as exc:  # noqa: BLE001 — scoring honesty
        return None, f"cannot build MTF frame: {exc}"
    if len(X) == 0:
        return None, "no usable MTF rows"
    h1 = frames["H1"].sort_index()
    row = X.iloc[[-1]]
    keep = list(bundle["features"])
    Xrow = row.reindex(columns=keep)
    ranks: dict[int, dict[str, float]] = {}
    for h in sorted(bundle["horizons"], key=lambda k: int(k)):
        h = int(h)
        entry = bundle["horizons"][h] \
            if h in bundle["horizons"] else bundle["horizons"][str(h)]
        ranks[h] = {}
        for dname in ("up", "dn"):
            ps = [np.asarray(m.predict_proba(Xrow.to_numpy(dtype=float)),
                             dtype=float)[:, 1]
                  for m in entry["models"][dname].values()]
            p = float(np.mean(ps)) if ps else float("nan")
            q = bundle["ref_quantiles"][h][dname] \
                if h in bundle["ref_quantiles"] \
                else bundle["ref_quantiles"][str(h)][dname]
            ranks[h][dname] = _rank_from_quantiles(p, q)
    return ForecastRanks(
        bar_time=h1.index[-1],
        close=float(h1["close"].iloc[-1]),
        ranks=ranks,
        horizons=tuple(sorted(ranks)),
    ), "forecast scored"


def needs_training(bundle_path: str | Path, h1) -> tuple[bool, str]:
    """Does the engine need to (re)train? No bundle, an unloadable
    bundle, or a bundle failing full validation all mean yes.
    Otherwise yes only when MIN_NEW_BARS_RETRAIN H1 bars arrived
    since trained_through. Never raises on a bad bundle — a bad
    bundle is a reason to train, not a crash."""
    import joblib

    path = Path(bundle_path)
    if not path.exists():
        return True, f"no forecaster bundle at {path}"
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 — bad bundle -> retrain
        return True, f"bundle unloadable ({exc}) — retraining"
    valid, reason = _validate_bundle(bundle)
    if not valid:
        return True, f"bundle invalid ({reason}) — retraining"
    through = pd.Timestamp(bundle["trained_through"])
    idx = h1.sort_index().index
    # A naive stamp (old bundle, hand-edited stamp) must not crash
    # the comparison against a tz-aware stack: read it under the
    # engine's UTC convention, and vice versa for a naive stack.
    if idx.tz is not None and through.tzinfo is None:
        through = through.tz_localize("UTC")
    elif idx.tz is None and through.tzinfo is not None:
        through = through.tz_localize(None)
    new_bars = int((idx > through).sum())
    if new_bars >= MIN_NEW_BARS_RETRAIN:
        return True, f"{new_bars} new H1 bars since {through}"
    return False, f"bundle current ({new_bars} new H1 bars < {MIN_NEW_BARS_RETRAIN})"
