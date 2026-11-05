"""Forecast F2 — trained estimators under the outlook.

F1 (forecast/outlook.py) estimates the outlook from EMPIRICAL
distributions: resolved live forecasts in the current rank band,
or the horizon's unconditional forward distribution over the H1
frame.  Honest, but coarse — it conditions on two numbers (the
dominant rank and its band) and ignores the rest of the state.

F2 trains real estimators on the same multi-timeframe state the
ensemble already computes (``mtf.build_mtf_frame`` — no parallel
feature stack), per horizon (12/24/72/168/360 H1 bars):

- a **direction classifier** (3-class: dn / neutral / up, the
  classes split by the F1 deadband convention — 0.15 × the
  training sample's forward-return stdev);
- **barrier classifiers** for the house large-move labels
  (``labels.build_labels_for_horizon``: y_up / y_dn), so the
  trained large-move probabilities stay SEPARATE from direction,
  exactly as F1 keeps them;
- **quantile regressors** for the forward return at the F1
  interval quantiles (0.10 / 0.50 / 0.90 at 80% coverage) — the
  point estimate is the trained median;
- **excursion regressors** (mean) for the forward run-up /
  run-down as % of close, quoted direction-relative like F1.

**Estimator choice (environment evidence).**  sklearn 1.9.1 is
installed; lightgbm is NOT.  The models are
``HistGradientBoosting*`` throughout — the house ``gbm`` of
``walkforward.make_models`` is the same family (handles NaN
natively, fast on the bounded training window), hyperparameters
mirroring it.  No new dependency enters the engine.

**Calibration.**  Raw classifier scores are mapped through
**isotonic regression** (sklearn's, out-of-bounds clipped),
fitted per class on WALK-FORWARD OUT-OF-FOLD predictions:
expanding chronological folds with the house purge (a training
row is admitted only when its whole label window ends before
the test block opens — ``walkforward.make_folds``).  Per-class
isotonic maps break the sum-to-one, so mapped probabilities are
renormalized — stated here, not hidden.

**Walk-forward discipline.**  Training reads only bars at or
before an explicit cutoff (``through``): every frame is sliced
to the cutoff BEFORE any feature or label is computed, and a
row's label exists only when its full forward window lies
inside the sliced history — a future bar cannot influence the
artifact, by construction (pinned by test, not asserted).

**Graduation — the honesty core.**  A trained horizon answers
the outlook ONLY when its stated probabilities have earned it
in the LIVE probability book (:class:`ProbabilityBook`, kept
beside the rank book in the calibration state): at least
``GRAD_MIN_N`` resolved trained-source statements with live
expected calibration error at or below ``GRAD_MAX_ECE``.  Until
then the estimator runs SHADOW: its probabilities are journaled
on the record (``trained_shadow``) and resolved into the book —
the same shadow-first, earn-trust-by-measurement path as E3 —
while F1's empirical method keeps answering.  The switch is
per horizon, and the record names the source on every horizon.
Nothing blends silently.

**Cadence.**  Retraining triggers (``needs_retrain``): no
artifact, an invalid artifact, ``RETRAIN_NEW_BARS`` new H1 bars
since the artifact's training cutoff, or an artifact older than
``RETRAIN_MAX_AGE_DAYS``.  The runner evaluates the trigger on
its calibration cycle and trains in a background thread; the
artifact swap is atomic (tmp + replace) and a failed train
leaves the previous artifact live — the record always carries
the artifact's trained-at stamp and training window, so a stale
model never claims freshness.

Pure core + one service class; the runner owns cadence,
journaling and the book's persistence calls.  No LLM anywhere,
no order path reads any of this — forecasts stay research.
"""
from __future__ import annotations

import datetime
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labels import HORIZONS, build_labels_for_horizon, spec_for
from .mtf import _atr, build_mtf_frame
from .outlook import (DEADBAND_SIGMA, INTERVAL_COVERAGE,
                      TRAINED_METHOD)

#: Artifact format version (the per-horizon method stamp is
#: outlook.TRAINED_METHOD, the same string).
ARTIFACT_VERSION = TRAINED_METHOD
#: Usable MTF rows before training starts (the ensemble's floor).
MIN_TRAIN_ROWS = 500
#: Labeled rows a horizon needs before it gets an estimator.
MIN_HORIZON_ROWS = 150
#: Bound on the fit: the most recent usable rows only.
MAX_TRAIN_ROWS = 12000
#: Out-of-fold (score, outcome) pairs an isotonic fit needs.
MIN_OOF_PAIRS = 100
#: Graduation gates on the LIVE book (trained source only).
GRAD_MIN_N = 60
GRAD_MAX_ECE = 0.10
#: Retrain cadence (mirrors the ensemble's 500-bar trigger).
RETRAIN_NEW_BARS = 500
RETRAIN_MAX_AGE_DAYS = 21
#: Probability-book bins over stated probability.
PROB_BINS = 10

#: Direction classes, classifier index order.
CLASSES = ("dn", "neutral", "up")
_CLASS_IDX = {name: i for i, name in enumerate(CLASSES)}

_Q_LO = (1.0 - INTERVAL_COVERAGE) / 2.0
_Q_HI = 1.0 - _Q_LO


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _r(value: float | None, nd: int = 2) -> float | None:
    return None if value is None else round(float(value), nd)


# ── models (house gbm family; sklearn present, lightgbm not) ──


def _clf(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        random_state=seed)


def _qreg(alpha: float, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="quantile", quantile=alpha,
        max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        random_state=seed)


def _mreg(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        random_state=seed)


# ── training data ───────────────────────────────────────────────


def _forward_arrays(h1: pd.DataFrame, horizon: int):
    """Per-bar forward outcomes, position-aligned (NaN where the
    window is incomplete or non-finite): endpoint return %,
    run-up and run-down in price units.  The same primitives as
    outlook.forward_outcomes, kept aligned instead of filtered
    so training rows can be indexed by bar position."""
    n = len(h1)
    ret = np.full(n, np.nan)
    run_up = np.full(n, np.nan)
    run_dn = np.full(n, np.nan)
    if n <= horizon or horizon <= 0:
        return ret, run_up, run_dn
    close = h1["close"].to_numpy(dtype=float)
    high = h1["high"].to_numpy(dtype=float)
    low = h1["low"].to_numpy(dtype=float)
    rmax = pd.Series(high).rolling(horizon).max().to_numpy()
    rmin = pd.Series(low).rolling(horizon).min().to_numpy()
    base = close[:-horizon]
    ret[:-horizon] = (close[horizon:] / base - 1.0) * 100.0
    run_up[:-horizon] = rmax[horizon:] - base
    run_dn[:-horizon] = base - rmin[horizon:]
    ok = np.isfinite(ret) & np.isfinite(run_up) & np.isfinite(run_dn)
    ret[~ok] = np.nan
    run_up[~ok] = np.nan
    run_dn[~ok] = np.nan
    return ret, run_up, run_dn


def _slice_frames(frames: dict, through) -> dict:
    """Every frame cut to bars at or before ``through`` — the
    cutoff is applied before ANY feature or label exists, so
    nothing downstream can read past it."""
    if through is None:
        return dict(frames)
    ts = pd.Timestamp(through)
    out = {}
    for tf, frame in (frames or {}).items():
        if frame is None or len(frame) == 0:
            out[tf] = frame
            continue
        out[tf] = frame.sort_index().loc[:ts]
    return out


def _oof_probs(factory, Xv: np.ndarray, y: np.ndarray,
               known: np.ndarray, horizon: int,
               seed: int) -> np.ndarray:
    """Walk-forward out-of-fold class probabilities for one
    target: expanding chronological folds, house purge via
    walkforward.make_folds.  Returns an (n_rows, n_classes)
    array, NaN where a row was never out-of-fold predicted."""
    from .walkforward import make_folds

    n = len(Xv)
    out = np.full((n, len(CLASSES)), np.nan)
    if factory == "binary":
        out = np.full((n, 2), np.nan)
    folds = make_folds(n, known, horizon,
                       initial_train=max(200, n // 2),
                       step=max(100, n // 8))
    for fold in folds[:6]:  # bounded: the calibration fit's tail
        tr, te = fold.train_idx, fold.test_idx
        ytr = y[tr]
        if len(np.unique(ytr[~np.isnan(ytr)])) < 2:
            continue
        model = _clf(seed)
        model.fit(Xv[tr], ytr)
        proba = np.asarray(model.predict_proba(Xv[te]),
                           dtype=float)
        for j, cls in enumerate(model.classes_):
            out[np.ix_(te, [int(cls)])] = proba[:, [j]]
    return out


def _fit_isotonic(scores: np.ndarray, outcomes: np.ndarray):
    """Isotonic map (score -> probability) or None when the
    out-of-fold pairs cannot carry one: too few pairs, or a
    single realized class (a map fitted on one class is a
    constant wearing a lab coat)."""
    ok = np.isfinite(scores) & np.isfinite(outcomes)
    if int(ok.sum()) < MIN_OOF_PAIRS:
        return None
    if len(np.unique(outcomes[ok])) < 2:
        return None
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores[ok], outcomes[ok])
    return iso


def _apply_isotonic(iso, raw: float) -> float:
    return float(iso.predict([float(raw)])[0])


def _horizon_entry(X: pd.DataFrame, Xv: np.ndarray,
                   h1: pd.DataFrame, atr: np.ndarray,
                   horizon: int, seed: int) -> dict | None:
    """Train one horizon's full estimator set, or None when its
    labeled rows fall below the floor (the artifact records the
    horizon as absent — F1 keeps answering it)."""
    pos = h1.index.get_indexer(X.index)
    close = h1["close"].to_numpy(dtype=float)
    ret_all, run_up_all, run_dn_all = _forward_arrays(h1, horizon)
    ret = ret_all[pos]
    labels = build_labels_for_horizon(h1, atr, horizon)
    y_up = labels["y_up"].reindex(X.index).to_numpy(dtype=float)
    y_dn = labels["y_dn"].reindex(X.index).to_numpy(dtype=float)
    known = np.isfinite(ret) & np.isfinite(y_up) & np.isfinite(y_dn)
    if int(known.sum()) < MIN_HORIZON_ROWS:
        return None
    sd = float(np.std(ret[known]))
    if not math.isfinite(sd) or sd <= 0:
        return None
    deadband = DEADBAND_SIGMA * sd
    cls = np.where(ret > deadband, _CLASS_IDX["up"],
                   np.where(ret < -deadband, _CLASS_IDX["dn"],
                            _CLASS_IDX["neutral"])).astype(float)
    cls[~known] = np.nan

    entry: dict[str, Any] = {
        "deadband_pct": round(deadband, 4),
        "n_train": int(known.sum()),
        "oof_n": 0,
        "base_rates": {
            name: round(float(np.mean(cls[known] == idx)), 4)
            for name, idx in _CLASS_IDX.items()},
    }

    # ── direction: final fit + OOF isotonic per class ──
    dir_model = _clf(seed)
    dir_model.fit(Xv[known], cls[known])
    entry["dir_model"] = dir_model
    oof = _oof_probs("multi", Xv, cls, known, horizon, seed)
    oof_rows = np.isfinite(oof[:, 0])
    entry["oof_n"] = int(oof_rows.sum())
    calibrators: dict[str, Any] = {}
    for name, idx in _CLASS_IDX.items():
        outcomes = (cls == idx).astype(float)
        outcomes[~known] = np.nan
        calibrators[name] = _fit_isotonic(oof[:, idx], outcomes)
    entry["dir_calibrators"] = calibrators

    # ── barrier (large-move) classifiers ──
    for side, y in (("up", y_up), ("dn", y_dn)):
        yk = np.isfinite(y)
        model = _clf(seed)
        model.fit(Xv[yk], y[yk])
        entry[f"{side}_model"] = model
        yb = np.where(yk, y, np.nan)
        oof_b = _oof_probs("binary", Xv, yb, yk, horizon, seed)
        entry[f"{side}_calibrator"] = _fit_isotonic(
            oof_b[:, 1], yb)

    # ── quantiles of the forward return + excursion means ──
    entry["quantiles"] = {}
    for alpha in (_Q_LO, 0.5, _Q_HI):
        model = _qreg(alpha, seed)
        model.fit(Xv[known], ret[known])
        entry["quantiles"][alpha] = model
    entry["exc_models"] = {}
    ref = close[pos]
    for side, arr in (("up", run_up_all), ("dn", run_dn_all)):
        pct = arr[pos] / np.where(ref > 0, ref, np.nan) * 100.0
        ok = np.isfinite(pct)
        model = _mreg(seed)
        model.fit(Xv[ok], pct[ok])
        entry["exc_models"][side] = model
    return entry


def train_trained(frames: dict, out_path: str | Path, *,
                  through=None, seed: int = 0) -> dict:
    """Train the F2 artifact from the stack frames and save it
    atomically.  ``through`` is the walk-forward cutoff: frames
    are sliced to it before anything is computed.  Raises
    ValueError on failed data-quality gates — the caller records
    the reason and the previous artifact stays untouched."""
    import joblib

    frames = _slice_frames(frames, through)
    h1 = frames.get("H1")
    if h1 is None or len(h1) == 0:
        raise ValueError("cannot train F2: no H1 frame")
    if not isinstance(h1.index, pd.DatetimeIndex):
        raise ValueError("cannot train F2: H1 needs a DatetimeIndex")
    X = build_mtf_frame(frames)
    if len(X) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"cannot train F2: too few usable MTF rows "
            f"({len(X)} < {MIN_TRAIN_ROWS})")
    if len(X) > MAX_TRAIN_ROWS:
        X = X.iloc[-MAX_TRAIN_ROWS:]
    h1 = h1.sort_index()
    atr = _atr(h1).to_numpy(dtype=float)
    Xv = X.to_numpy(dtype=float)
    horizons: dict[int, dict] = {}
    skipped: list[int] = []
    for h in HORIZONS:
        entry = _horizon_entry(X, Xv, h1, atr, h, seed)
        if entry is None:
            skipped.append(h)
        else:
            horizons[h] = entry
    if not horizons:
        raise ValueError(
            "cannot train F2: no horizon cleared its labeled-row "
            f"floor ({MIN_HORIZON_ROWS})")
    artifact = {
        "version": ARTIFACT_VERSION,
        "feature_set": "mtf-v1",
        "symbol": "XAUUSD",
        "features": list(X.columns),
        "train_start": str(X.index[0]),
        "trained_through": str(X.index[-1]),
        "trained_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "train_rows": int(len(X)),
        "horizons": horizons,
        "skipped_horizons": skipped,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    joblib.dump(artifact, tmp_path)
    os.replace(tmp_path, out_path)
    return {
        "saved": str(out_path),
        "version": ARTIFACT_VERSION,
        "trained_at": artifact["trained_at"],
        "trained_through": artifact["trained_through"],
        "train_rows": artifact["train_rows"],
        "horizons": sorted(horizons),
        "skipped_horizons": skipped,
    }


def _validate_artifact(artifact) -> tuple[bool, str]:
    if not isinstance(artifact, dict):
        return False, "artifact is not a dict"
    if artifact.get("version") != ARTIFACT_VERSION:
        return False, (f"artifact version "
                       f"{artifact.get('version')!r} != "
                       f"{ARTIFACT_VERSION!r}")
    feats = artifact.get("features")
    if not isinstance(feats, list) or not feats:
        return False, "artifact has no feature list"
    horizons = artifact.get("horizons")
    if not isinstance(horizons, dict) or not horizons:
        return False, "artifact has no horizons"
    for h, entry in horizons.items():
        if not isinstance(entry, dict) \
                or entry.get("dir_model") is None:
            return False, f"artifact horizon {h} lacks a model"
    try:
        pd.Timestamp(artifact.get("trained_through"))
        pd.Timestamp(artifact.get("trained_at"))
    except Exception:  # noqa: BLE001 — unparseable stamp is invalid
        return False, "artifact stamps unparseable"
    return True, "artifact valid"


def retrain_accrual(artifact: dict | None, h1_index,
                    now: datetime.datetime) -> dict:
    """The F2 retrain accrual as DATA (completion-04): how many
    new H1 bars have accrued against ``RETRAIN_NEW_BARS`` since
    the artifact's training cutoff, and how old the artifact is
    against ``RETRAIN_MAX_AGE_DAYS``.  ``needs_retrain`` is this
    computation's verdict; the status surface and the runtime
    manifest read the same numbers, so "how close is the next
    retrain" is never a reason string someone has to parse."""
    acc: dict[str, Any] = {
        "new_bars": None, "threshold": RETRAIN_NEW_BARS,
        "age_days": None, "max_age_days": RETRAIN_MAX_AGE_DAYS,
        "trained_through": None, "due": True, "cause": None,
        "reason": ""}
    if not artifact:
        acc.update(cause="no_artifact",
                   reason="no trained artifact")
        return acc
    valid, reason = _validate_artifact(artifact)
    if not valid:
        acc.update(cause="invalid",
                   reason=f"artifact invalid ({reason}) — retraining")
        return acc
    through = pd.Timestamp(artifact["trained_through"])
    idx = pd.DatetimeIndex(h1_index)
    if idx.tz is not None and through.tzinfo is None:
        through = through.tz_localize("UTC")
    elif idx.tz is None and through.tzinfo is not None:
        through = through.tz_localize(None)
    new_bars = int((idx > through).sum())
    trained_at = pd.Timestamp(artifact["trained_at"])
    age_days = (pd.Timestamp(now) - trained_at).total_seconds() \
        / 86400.0
    acc.update(new_bars=new_bars, age_days=round(age_days, 1),
               trained_through=str(artifact["trained_through"]))
    if new_bars >= RETRAIN_NEW_BARS:
        acc.update(due=True, cause="new_bars",
                   reason=f"{new_bars} new H1 bars since {through}")
    elif age_days >= RETRAIN_MAX_AGE_DAYS:
        acc.update(due=True, cause="age",
                   reason=f"artifact is {age_days:.0f} days old")
    else:
        acc.update(due=False, cause=None,
                   reason=(f"artifact current ({new_bars} new H1 "
                           f"bars, {age_days:.1f} days old)"))
    return acc


def needs_retrain(artifact: dict | None, h1_index,
                  now: datetime.datetime) -> tuple[bool, str]:
    """The F2 retrain trigger: no (valid) artifact, enough new
    H1 bars since the training cutoff, or an old artifact."""
    acc = retrain_accrual(artifact, h1_index, now)
    return bool(acc["due"]), str(acc["reason"])


# ── the LIVE probability book ───────────────────────────────────


def direction_class(ret_pct: float | None,
                    deadband_pct: float | None) -> str | None:
    """The realized direction class of a resolved forecast:
    endpoint return against the deadband the forecast itself
    stated (F1 convention).  None when the return is unknown —
    an unmeasurable resolution is never bucketed."""
    ret = _f(ret_pct)
    if ret is None:
        return None
    db = abs(float(deadband_pct)) if deadband_pct is not None \
        else 0.0
    if ret > db:
        return "up"
    if ret < -db:
        return "dn"
    return "neutral"


class ProbabilityBook:
    """Live calibration of STATED probabilities, by source.

    Every journaled forecast whose outlook stated probabilities
    (the F1 empirical frequencies, the trained estimator's
    rendered probabilities, or the trained estimator's SHADOW
    probabilities) is registered by the runner and folded here
    when it resolves: for each direction class, the stated
    probability is binned by decile against the realized class.
    Sources are kept apart — an empirical track record can never
    graduate the trained estimator; graduation reads the
    "trained" source only.

    Plain-dict backed (JSON-serializable) so the runner persists
    it inside the calibration state verbatim, like the rank
    book."""

    def __init__(self, data: dict | None = None) -> None:
        self.data: dict[str, Any] = data or {"horizons": {}}

    def add(self, horizon: int, source: str, probs: dict,
            realized: str | None) -> bool:
        """Fold one resolved stated forecast.  Returns False
        (folds nothing) when the source or outcome is unusable —
        a statement without a measurable outcome is not data."""
        if source not in ("trained", "empirical") \
                or realized not in CLASSES:
            return False
        slot = self.data["horizons"].setdefault(
            str(int(horizon)), {}).setdefault(
                source, {"n": 0, "cells": {}})
        folded = False
        for cls in CLASSES:
            p = _f((probs or {}).get(cls))
            if p is None:
                continue
            p = min(1.0, max(0.0, p))
            b = min(PROB_BINS - 1, int(p * PROB_BINS))
            cell = slot["cells"].setdefault(
                f"{cls}|{b}", {"n": 0, "hits": 0, "sum": 0.0})
            cell["n"] += 1
            cell["sum"] = round(cell["sum"] + p, 4)
            cell["hits"] += 1 if realized == cls else 0
            folded = True
        if folded:
            slot["n"] += 1
        return folded

    def _source_view(self, slot: dict) -> dict:
        cells = (slot or {}).get("cells") or {}
        total_n = 0
        err = 0.0
        for cell in cells.values():
            n = int(cell.get("n") or 0)
            if not n:
                continue
            stated = float(cell.get("sum") or 0.0) / n
            realized = float(cell.get("hits") or 0) / n
            err += abs(stated - realized) * n
            total_n += n
        return {
            "n": int((slot or {}).get("n") or 0),
            "ece": (round(err / total_n, 4) if total_n else None),
            "cell_n": total_n,
        }

    def view(self) -> dict[str, Any]:
        horizons: dict[str, Any] = {}
        for h, sources in (self.data.get("horizons") or {}).items():
            horizons[str(h)] = {
                source: self._source_view(slot)
                for source, slot in (sources or {}).items()}
        return {"horizons": horizons}

    def graduated(self, horizon: int) -> tuple[bool, dict]:
        """The graduation gate for one horizon, trained source
        only: enough live resolutions AND a live calibration
        error at or under the bound.  Returns (ok, {n, ece,
        gates}) — the numbers travel with the verdict so the
        record can say WHY a horizon did or did not switch."""
        view = self.view()["horizons"].get(str(int(horizon))) \
            or {}
        trained = view.get("trained") or {}
        n = int(trained.get("n") or 0)
        ece = trained.get("ece")
        ok = n >= GRAD_MIN_N and ece is not None \
            and ece <= GRAD_MAX_ECE
        return ok, {"n": n, "ece": ece,
                    "min_n": GRAD_MIN_N, "max_ece": GRAD_MAX_ECE}


def stated_probabilities(outlook: dict | None) -> dict:
    """The stated probabilities a journaled outlook carries, per
    horizon, for registration into the probability book:
    ``{str(h): {"source", "up", "dn", "neutral",
    "deadband_pct"}}``.  The trained source wins when present
    (rendered after graduation, else the shadow block — shadow
    statements are exactly what the book learns from); the
    empirical source is F1's calibrated band frequencies.  A
    horizon that stated nothing registers nothing."""
    out: dict[str, Any] = {}
    if not isinstance(outlook, dict):
        return out
    for key, ho in (outlook.get("horizons") or {}).items():
        if not isinstance(ho, dict):
            continue
        probs = ho.get("probabilities") or {}
        shadow = ho.get("trained_shadow")
        entry = None
        if ho.get("method") == TRAINED_METHOD \
                and probs.get("state") == "calibrated":
            entry = {"source": "trained",
                     "up": probs.get("up"), "dn": probs.get("dn"),
                     "neutral": probs.get("neutral"),
                     "deadband_pct": ho.get("deadband_pct")}
        elif isinstance(shadow, dict):
            entry = {"source": "trained",
                     "up": shadow.get("up"),
                     "dn": shadow.get("dn"),
                     "neutral": shadow.get("neutral"),
                     "deadband_pct": shadow.get("deadband_pct")}
        elif probs.get("state") == "calibrated":
            entry = {"source": "empirical",
                     "up": probs.get("up"), "dn": probs.get("dn"),
                     "neutral": probs.get("neutral"),
                     "deadband_pct": ho.get("deadband_pct")}
        if entry is not None and entry["up"] is not None:
            out[str(key)] = entry
    return out


def render_prob_book_lines(book_view: dict,
                           states: dict) -> list[str]:
    """The /status probability-book section: per horizon, the
    trained estimator's standing (graduated / shadow / no
    artifact) with its live n and calibration error, and the
    empirical source's standing beside it.  Numbers first, the
    gates stated once."""
    horizons = (book_view or {}).get("horizons") or {}
    keys = sorted(set(horizons) | {str(h) for h in states},
                  key=lambda k: int(k))
    lines = ["  probability book (F2 — stated vs realized):"]
    any_line = False
    from .outlook import tag_for

    for key in keys:
        state = states.get(int(key), states.get(key, ""))
        entry = horizons.get(key) or {}
        trained = entry.get("trained") or {}
        empirical = entry.get("empirical") or {}
        parts = []
        # "no artifact" with an empty book says nothing — the
        # section exists only when there is something to report.
        if state and (state != "no artifact"
                      or trained.get("n") or empirical.get("n")):
            parts.append(state)
        if trained.get("n"):
            parts.append(
                f"trained n={trained['n']} · ECE "
                f"{trained['ece'] if trained['ece'] is not None else '—'}")
        if empirical.get("n"):
            parts.append(
                f"empirical n={empirical['n']} · ECE "
                f"{empirical['ece'] if empirical['ece'] is not None else '—'}")
        if not parts:
            continue
        lines.append(f"    {tag_for(int(key))}: "
                     + " · ".join(parts))
        any_line = True
    if not any_line:
        return []
    lines.append(
        f"    graduation gate: trained n≥{GRAD_MIN_N} and "
        f"ECE≤{GRAD_MAX_ECE} on live resolutions")
    return lines


# ── prediction ──────────────────────────────────────────────────


class TrainedOutlook:
    """Runner-side F2 service: loads the trained artifact
    (mtime-cached, like the ensemble bundle) and answers
    per-horizon estimates for the outlook.  A missing or invalid
    artifact answers nothing — F1 stands, and the status surface
    says why."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._artifact: dict | None = None
        self._mtime: float | None = None
        self._note = "not loaded"

    def artifact(self) -> tuple[dict | None, str]:
        import joblib

        if not self.path.exists():
            self._artifact, self._mtime = None, None
            self._note = f"no trained artifact at {self.path}"
            return None, self._note
        mtime = self.path.stat().st_mtime
        if self._artifact is not None and self._mtime == mtime:
            return self._artifact, "artifact cached"
        try:
            artifact = joblib.load(self.path)
        except Exception as exc:  # noqa: BLE001 — loader honesty
            self._artifact, self._mtime = None, None
            self._note = f"artifact unloadable: {exc}"
            return None, self._note
        valid, reason = _validate_artifact(artifact)
        if not valid:
            self._artifact, self._mtime = None, None
            self._note = f"artifact invalid: {reason}"
            return None, self._note
        self._artifact, self._mtime = artifact, mtime
        self._note = "artifact loaded"
        return artifact, self._note

    def _calibrated_probs(self, entry: dict, row: np.ndarray):
        """(probs | None, shadow | None, large) for one horizon:
        the isotonic-mapped, renormalized direction probabilities
        plus the barrier (large-move) probabilities.  probs is
        None when no isotonic fit exists — no honest probability
        can be stated, in render OR in shadow."""
        model = entry["dir_model"]
        proba = np.asarray(model.predict_proba(row), dtype=float)[0]
        raw = {name: 0.0 for name in CLASSES}
        for j, cls in enumerate(model.classes_):
            raw[CLASSES[int(cls)]] = float(proba[j])
        calibrators = entry.get("dir_calibrators") or {}
        if any(calibrators.get(name) is None for name in CLASSES):
            return None, None, {}
        mapped = {name: _apply_isotonic(calibrators[name],
                                        raw[name])
                  for name in CLASSES}
        total = sum(mapped.values())
        if total <= 0:
            return None, None, {}
        probs = {name: round(mapped[name] / total, 4)
                 for name in CLASSES}
        large: dict[str, float | None] = {}
        for side in ("up", "dn"):
            cal = entry.get(f"{side}_calibrator")
            if cal is None:
                large[side] = None
                continue
            bm = entry[f"{side}_model"]
            bp = np.asarray(bm.predict_proba(row),
                            dtype=float)[0]
            raw_b = 0.0
            for j, cls in enumerate(bm.classes_):
                if int(cls) == 1:
                    raw_b = float(bp[j])
            large[side] = round(_apply_isotonic(cal, raw_b), 4)
        shadow = {
            "state": "shadow",
            "up": probs["up"], "dn": probs["dn"],
            "neutral": probs["neutral"],
            "large_up": large["up"], "large_dn": large["dn"],
            "deadband_pct": entry.get("deadband_pct"),
            "trained_at": None, "train_window": None,
        }
        return probs, shadow, large

    def predict(self, frames: dict, *, close: float | None,
                atr: float | None,
                book: ProbabilityBook | None = None
                ) -> dict[int, dict]:
        """Per-horizon trained estimates for one scored bar:
        ``{h: {"fields": {...}, "shadow": {...} | None,
        "meta": {...}}}``.  ``fields`` mirrors the F1 horizon
        block's estimation keys exactly (the outlook substitutes
        them whole when the horizon has graduated — never a
        field-level blend).  Empty when no artifact can answer."""
        artifact, _note = self.artifact()
        if artifact is None or close is None:
            return {}
        try:
            X = build_mtf_frame(frames)
        except Exception:  # noqa: BLE001 — F1 stands
            return {}
        if len(X) == 0:
            return {}
        row = X.iloc[[-1]].reindex(
            columns=list(artifact["features"])).to_numpy(dtype=float)
        book_view = book.view()["horizons"] if book is not None \
            else {}
        window = {"start": artifact.get("train_start"),
                  "through": artifact.get("trained_through"),
                  "rows": artifact.get("train_rows")}
        out: dict[int, dict] = {}
        horizons = artifact.get("horizons") or {}
        for key in sorted(horizons, key=lambda k: int(k)):
            h = int(key)
            entry = horizons[key] \
                if key in horizons else horizons.get(h)
            if entry is None:
                continue
            got = self._calibrated_probs(entry, row)
            if got[0] is None:
                probs, shadow, large = None, None, {}
            else:
                probs, shadow, large = got
            if shadow is not None:
                shadow["trained_at"] = artifact.get("trained_at")
                shadow["train_window"] = window
            quantiles = entry["quantiles"]
            qs = sorted(float(quantiles[a].predict(row)[0])
                        for a in (_Q_LO, 0.5, _Q_HI))
            q_lo, med, q_hi = qs
            deadband = float(entry.get("deadband_pct") or 0.0)
            if med > deadband:
                direction = "UP"
            elif med < -deadband:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"
            eff_dir = direction.lower() \
                if direction in ("UP", "DOWN") else "up"
            exc = {side: max(0.0, float(
                entry["exc_models"][side].predict(row)[0]))
                for side in ("up", "dn")}
            fav_src = exc["up"] if eff_dir == "up" else exc["dn"]
            adv_src = exc["dn"] if eff_dir == "up" else exc["up"]
            point = close * (1.0 + med / 100.0)
            fields: dict[str, Any] = {
                "direction": direction,
                "point_price": _r(point, 1),
                "interval": {
                    "low": _r(close * (1.0 + q_lo / 100.0), 1),
                    "high": _r(close * (1.0 + q_hi / 100.0), 1),
                    "coverage": INTERVAL_COVERAGE},
                "expected_move_usd": _r(point - close, 1),
                "expected_move_pct": _r(med, 2),
                "fav_excursion_usd": _r(fav_src / 100.0 * close, 1),
                "adv_excursion_usd": _r(adv_src / 100.0 * close, 1),
                "method": TRAINED_METHOD,
                "n": int(entry.get("n_train") or 0),
                "deadband_pct": _r(deadband, 3),
                "barriers": None, "scenario": None,
                "invalidation": None,
                "unavailable_reason": None,
                "trained_at": artifact.get("trained_at"),
                "train_window": window,
            }
            if probs is not None:
                tview = (book_view.get(str(h)) or {}).get(
                    "trained") or {}
                live_n = int(tview.get("n") or 0)
                basis = (
                    f"trained estimator {ARTIFACT_VERSION}, "
                    f"isotonic-calibrated on "
                    f"{int(entry.get('oof_n') or 0)} out-of-fold "
                    f"rows; live book n={live_n}")
                if tview.get("ece") is not None:
                    basis += f", ECE {tview['ece']}"
                fields["probabilities"] = {
                    "state": "calibrated", "n": live_n,
                    "up": probs["up"], "dn": probs["dn"],
                    "neutral": probs["neutral"],
                    "large_up": large.get("up"),
                    "large_dn": large.get("dn"),
                    "basis": basis}
            else:
                fields["probabilities"] = {
                    "state": "uncalibrated",
                    "reason": ("trained estimator has no "
                               "out-of-fold calibration fit")}
            if atr is not None and atr > 0:
                try:
                    spec = spec_for(h)
                except KeyError:
                    spec = None
                if spec is not None:
                    d = 1.0 if eff_dir == "up" else -1.0
                    fav_price = close + d * spec.fav_mult * atr
                    adv_price = close - d * spec.adv_mult * atr
                    fields["barriers"] = {
                        "fav": _r(fav_price, 1),
                        "adv": _r(adv_price, 1), "side": eff_dir,
                        "fav_mult": spec.fav_mult,
                        "adv_mult": spec.adv_mult}
                    fields["invalidation"] = _r(adv_price, 1)
                    fields["scenario"] = (
                        f"{spec.tag} barrier race: favourable "
                        f"{fav_price:,.1f} ({spec.fav_mult:g}x ATR) "
                        f"reached before adverse {adv_price:,.1f} "
                        f"({spec.adv_mult:g}x ATR) within {h} "
                        f"H1 bars")
            out[h] = {
                "fields": fields, "shadow": shadow,
                "meta": {"version": ARTIFACT_VERSION,
                         "trained_at": artifact.get("trained_at"),
                         "trained_through":
                             artifact.get("trained_through"),
                         "train_window": window}}
        return out
