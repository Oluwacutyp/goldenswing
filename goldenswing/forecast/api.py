"""Multi-horizon forecaster API (Phase 6.5).

The deployment-side counterpart of horizons.py: given a trained
multi-horizon bundle (``scripts/train_forecast.py``) and market
states, score every bar at every horizon the bundle carries and
express each score as a RANK — the percentile of that bar's ensemble
score within the scored history, per horizon per direction.  Ranks,
not probabilities: Phase 6 established the raw outputs are
uncalibrated ranking scores, and the rank is the honest way to cite
them ("big-move rank high @1d and @3d").

Bundle formats accepted:

- Phase-6.5 bundle: ``{"features": [...], "horizons": {horizon:
  {"models": {"up": {...}, "dn": {...}}, ...}}, "models": <72-bar
  legacy entry>, ...}``
- Phase-6 legacy bundle: only the top-level ``"models"`` /
  ``"features"`` — treated as the single 72-bar horizon, so old
  artefacts keep working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.data import REPO_ROOT
from .labels import spec_for

DEFAULT_BUNDLE = REPO_ROOT / "data" / "forecast_models.joblib"


def rank_series(values: np.ndarray) -> np.ndarray:
    """Percentile rank (0-100) of each value within the finite
    values of the series itself — the same definition the live
    runner uses for its single-horizon note (share of history at or
    below the value).  NaN in, NaN out."""
    values = np.asarray(values, dtype=float)
    finite = np.sort(values[np.isfinite(values)])
    out = np.full(len(values), np.nan)
    if len(finite) < 2:
        return out
    idx = np.searchsorted(finite, values[np.isfinite(values)],
                          side="right")
    out[np.isfinite(values)] = idx / len(finite) * 100.0
    return out


@dataclass
class HorizonScores:
    """Per-bar ensemble scores + ranks for one horizon."""
    horizon: int
    tag: str
    prob_up: np.ndarray
    prob_dn: np.ndarray
    rank_up: np.ndarray
    rank_dn: np.ndarray

    def rank(self, direction: int, t: int) -> float:
        series = self.rank_up if direction == 1 else self.rank_dn
        return float(series[t]) if 0 <= t < len(series) else float("nan")


@dataclass
class ForecastTable:
    """All horizons' scores for a scored history, bar-aligned."""
    scores: dict[int, HorizonScores]

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted(self.scores))

    def ranks_at(self, direction: int, t: int) -> list[tuple[int, float]]:
        """(horizon, rank) pairs at bar ``t``, best rank first."""
        pairs = [(h, self.scores[h].rank(direction, t))
                 for h in self.horizons]
        pairs = [(h, r) for h, r in pairs if np.isfinite(r)]
        return sorted(pairs, key=lambda p: -p[1])

    def line(self, direction: int, t: int,
             top_pct: float = 10.0) -> str | None:
        """The alert citation: horizons whose rank for ``direction``
        at bar ``t`` sits in the top ``top_pct`` of their history,
        e.g. ``big-move rank high @1d (top 8%), @3d (top 4%)``.
        None when no horizon qualifies — silence over decoration."""
        hits = [(h, r) for h, r in self.ranks_at(direction, t)
                if r >= 100.0 - top_pct]
        if not hits:
            return None
        hits.sort()                      # cite short -> long horizon
        parts = [f"@{self.scores[h].tag} "
                 f"(top {max(1.0, 100.0 - r):.0f}%)" for h, r in hits]
        return "Forecaster: big-move rank high " + ", ".join(parts)


class MultiHorizonScorer:
    """Scores states with a saved bundle at every horizon it
    carries.  Construction takes the bundle dict directly (tests
    inject fake models); :meth:`try_load` mirrors
    ``ForecasterScorer.try_load`` — every failure returns
    (None, reason), never an exception."""

    def __init__(self, bundle: dict, states):
        from .features import build_feature_frame
        X = build_feature_frame(states)
        cols = list(bundle["features"])
        for c in cols:
            if c not in X.columns:
                X[c] = np.nan
        Xv = X[cols].to_numpy(dtype=float)

        entries: dict[int, dict] = {}
        if isinstance(bundle.get("horizons"), dict):
            for key, entry in bundle["horizons"].items():
                entries[int(key)] = entry
        if not entries and "models" in bundle:
            entries[72] = {"models": bundle["models"]}   # legacy bundle

        self.table_scores: dict[int, HorizonScores] = {}
        for h, entry in sorted(entries.items()):
            probs: dict[str, np.ndarray] = {}
            for dname in ("up", "dn"):
                models = entry["models"][dname]
                ps = [np.asarray(m.predict_proba(Xv), dtype=float)[:, 1]
                      for m in models.values()]
                probs[dname] = np.mean(ps, axis=0) if ps else \
                    np.full(len(Xv), np.nan)
            try:
                tag = spec_for(h).tag
            except KeyError:
                tag = f"{h}b"
            self.table_scores[h] = HorizonScores(
                horizon=h, tag=tag,
                prob_up=probs["up"], prob_dn=probs["dn"],
                rank_up=rank_series(probs["up"]),
                rank_dn=rank_series(probs["dn"]))
        self._table = ForecastTable(scores=self.table_scores)

    @classmethod
    def try_load(cls, path, states
                 ) -> tuple["MultiHorizonScorer | None", str]:
        if path is None:
            return None, "no forecaster bundle configured"
        path = Path(path)
        if not path.exists():
            return None, f"no forecaster bundle at {path}"
        try:
            import joblib
        except ImportError:
            return None, ("bundle exists but joblib/scikit-learn are "
                          "not installed on this host")
        try:
            bundle = joblib.load(path)
            if not isinstance(bundle, dict) \
                    or "features" not in bundle \
                    or ("models" not in bundle
                        and "horizons" not in bundle):
                return None, f"bundle at {path} has an unexpected shape"
            return cls(bundle, states), \
                f"multi-horizon bundle loaded ({path})"
        except Exception as exc:  # unpickle needs sklearn; say so
            return None, f"forecaster bundle could not be loaded: {exc}"

    def table(self) -> ForecastTable:
        return self._table

    def note_for(self, signal):
        """A charts.ForecastNote citing the horizons for one signal:
        the top-decile citation when any horizon qualifies, else the
        single best horizon — always a true statement about ranks."""
        from ..charts import ForecastNote
        line = self._table.line(signal.direction, signal.index)
        if line is None:
            pairs = self._table.ranks_at(signal.direction, signal.index)
            if not pairs:
                return None
            h, r = pairs[0]
            line = (f"Forecaster: big-move rank best "
                    f"@{self._table.scores[h].tag} "
                    f"(top {max(1.0, 100.0 - r):.0f}%)")
        return ForecastNote(text=line)


def forecast(df: pd.DataFrame, peers: dict | None = None,
             bundle_path: Path | None = DEFAULT_BUNDLE
             ) -> ForecastTable | None:
    """Score ``df`` at every horizon the saved bundle carries.

    Returns per-horizon rank scores (see :class:`ForecastTable`), or
    None when no usable bundle exists on this host — callers that
    need the reason should use ``MultiHorizonScorer.try_load``."""
    from ..features import FeatureEngine
    from ..reader.reader import MarketReader
    states = MarketReader().read(df)
    FeatureEngine().attach(df, states, peers)
    scorer, _reason = MultiHorizonScorer.try_load(bundle_path, states)
    return scorer.table() if scorer is not None else None
