"""Walk-forward training with purged, embargoed folds.

Time-series discipline:

- Expanding window: fold ``k`` tests on bars
  ``[start_k, start_k + step)`` and trains on bars before ``start_k``.
- PURGE / EMBARGO: a training row ``t`` is admitted only if its whole
  label window lies before the test block: ``t + horizon < start_k``.
  A row whose 72-bar outcome window overlaps the test block would leak
  test-period price action into training — it is dropped, always.
- Test rows are admitted only when their label is known (full window
  inside the data).
- Models are retrained from scratch at every fold boundary.  No random
  shuffling anywhere: order is time, always.

Models (per direction, up and down trained separately):

1. ``gbm`` — sklearn HistGradientBoosting (handles NaN natively).
2. ``logreg`` — median-imputed, standardised, class-weighted logistic
   regression (the calibrated linear baseline).
3. ``extratrees`` — ExtraTrees ensemble (bagged trees, different bias
   from boosting).

The ensemble probability is the simple mean of the three.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INITIAL_TRAIN = 5000   # bars before the first test block
STEP = 500             # test block size / retrain cadence

MODEL_NAMES = ("gbm", "logreg", "extratrees")


def make_models(seed: int = 0) -> dict:
    """Fresh model instances (one set per direction per fold)."""
    from sklearn.ensemble import (ExtraTreesClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=seed)),
    ])
    extratrees = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=seed)),
    ])
    return {"gbm": gbm, "logreg": logreg, "extratrees": extratrees}


@dataclass
class Fold:
    k: int
    train_idx: np.ndarray    # purged training rows (positions)
    test_idx: np.ndarray     # test rows (positions, labels known)


def make_folds(n: int, label_known: np.ndarray, horizon: int,
               initial_train: int = INITIAL_TRAIN,
               step: int = STEP) -> list[Fold]:
    """Expanding-window folds with the embargo purge applied."""
    folds: list[Fold] = []
    start = initial_train
    k = 0
    while start < n:
        test = np.arange(start, min(start + step, n))
        test = test[label_known[test]]
        # purge: training row t admitted iff t + horizon < start
        # (its label is fully determined before the test block opens)
        train = np.arange(0, start - horizon)
        train = train[label_known[train]]
        if len(test) and len(train):
            folds.append(Fold(k=k, train_idx=train, test_idx=test))
            k += 1
        start += step
    return folds


@dataclass
class WalkForwardResult:
    """Out-of-fold probabilities, one row per predicted bar."""
    probs: pd.DataFrame      # index = bar position; columns:
                             # {model}_{up,dn}, ens_up, ens_dn
    folds: list[Fold] = field(default_factory=list)

    @property
    def index(self) -> pd.Index:
        return self.probs.index


def run_walk_forward(X: pd.DataFrame, labels: pd.DataFrame,
                     horizon: int, initial_train: int = INITIAL_TRAIN,
                     step: int = STEP, seed: int = 0) -> WalkForwardResult:
    """Train per fold / per direction / per model; collect OOF probs.

    ``labels`` columns ``y_up`` / ``y_dn`` are float 0/1/NaN aligned
    positionally with ``X``.
    """
    n = len(X)
    y = {"up": labels["y_up"].to_numpy(dtype=float),
         "dn": labels["y_dn"].to_numpy(dtype=float)}
    label_known = ~np.isnan(y["up"]) & ~np.isnan(y["dn"])
    folds = make_folds(n, label_known, horizon, initial_train, step)

    cols = [f"{m}_{d}" for m in MODEL_NAMES for d in ("up", "dn")] \
        + ["ens_up", "ens_dn"]
    out = pd.DataFrame(np.nan, index=pd.RangeIndex(n), columns=cols)

    Xv = X.to_numpy(dtype=float)
    for fold in folds:
        tr, te = fold.train_idx, fold.test_idx
        for dname in ("up", "dn"):
            ytr = y[dname][tr]
            if len(np.unique(ytr)) < 2:
                continue
            preds = []
            for mname, model in make_models(seed).items():
                model.fit(Xv[tr], ytr)
                p = model.predict_proba(Xv[te])[:, 1]
                out.loc[te, f"{mname}_{dname}"] = p
                preds.append(p)
            if preds:
                out.loc[te, f"ens_{dname}"] = np.mean(preds, axis=0)
    probs = out.dropna(subset=["ens_up", "ens_dn"])
    return WalkForwardResult(probs=probs, folds=folds)
