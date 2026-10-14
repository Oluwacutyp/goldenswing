"""Forecaster evaluation: discrimination, alert quality, calibration.

Everything is computed on the out-of-fold rows only — the same rows for
every model, the ensemble, and the detector-score baseline, so the
comparison is apples-to-apples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

THRESHOLDS = (0.5, 0.6, 0.7)


def auc_table(probs: pd.DataFrame, labels: pd.DataFrame,
              det_signed: pd.Series) -> pd.DataFrame:
    """AUC per model / ensemble / detector-score baseline, per
    direction.  ``det_signed`` is the detector's signed score series
    (score x direction) aligned positionally with ``probs``."""
    rows = []
    idx = probs.index
    for dname, ycol in (("up", "y_up"), ("dn", "y_dn")):
        y = labels[ycol].loc[idx].to_numpy(dtype=float)
        det_pred = det_signed.loc[idx].to_numpy(dtype=float)
        det_pred = np.clip(det_pred, 0.0, None) if dname == "up" \
            else np.clip(-det_pred, 0.0, None)
        preds = {m: probs[f"{m}_{dname}"].to_numpy(dtype=float)
                 for m in ("gbm", "logreg", "extratrees")}
        preds["ensemble"] = probs[f"ens_{dname}"].to_numpy(dtype=float)
        preds["detector_score"] = det_pred
        for name, p in preds.items():
            ok = ~np.isnan(p)
            auc = roc_auc_score(y[ok], p[ok]) if len(np.unique(y[ok])) > 1 \
                else float("nan")
            rows.append({"direction": dname, "model": name,
                         "auc": float(auc), "n": int(ok.sum())})
    return pd.DataFrame(rows)


def alert_quality(probs: pd.DataFrame, labels: pd.DataFrame,
                  column: str, ycol: str,
                  thresholds: tuple[float, ...] = THRESHOLDS
                  ) -> pd.DataFrame:
    """Precision / recall / alert count at probability thresholds."""
    idx = probs.index
    y = labels[ycol].loc[idx].to_numpy(dtype=float)
    p = probs[column].to_numpy(dtype=float)
    rows = []
    for th in thresholds:
        fired = p >= th
        rows.append({
            "threshold": th, "alerts": int(fired.sum()),
            "alert_rate_pct": float(100.0 * fired.mean()),
            "precision": float(precision_score(y, fired, zero_division=0)),
            "recall": float(recall_score(y, fired, zero_division=0)),
            "base_rate": float(y.mean()),
        })
    return pd.DataFrame(rows)


def calibration(probs: pd.DataFrame, labels: pd.DataFrame,
                column: str, ycol: str, bins: int = 10) -> pd.DataFrame:
    """Reliability by predicted-probability decile: mean predicted vs
    empirical event rate per bin."""
    idx = probs.index
    d = pd.DataFrame({"p": probs[column].to_numpy(dtype=float),
                      "y": labels[ycol].loc[idx].to_numpy(dtype=float)})
    try:
        d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    except ValueError:
        d["bin"] = pd.cut(d["p"], bins)
    g = d.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), empirical=("y", "mean"))
    return g.reset_index(drop=True)
