"""MT5 forecast service: the engine's own data -> forecast alerts.

Owner rulings §3: the ensemble forecasts big moves (horizons 1 day and
above) and the engine ALERTS those forecasts. It NEVER trades, never
sizes, never gates, never vetoes an order. Forecast output and order
decisions are separate by construction.

- Training runs on the engine's own MT5 history (the H1 frame from the
  runner's bar stack), on install / first run, retraining as history
  grows. No external frame, no shipped bundle.
- Loading is honest: no bundle -> an "untrained" note, never a crash.
- Outputs are RANKS (uncalibrated scores), never probabilities, and
  every alert says so and says it is not a trade signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import FeatureEngine
from ..reader.reader import MarketReader
from .api import MultiHorizonScorer
from .features import build_feature_frame
from .labels import HORIZONS, build_labels_for_horizon, spec_for
from .walkforward import make_models

# Alert gate (build default, tunable — not an owner ruling): a forecast
# alerts when a 1d or 3d big-move rank reaches this percentile.
DEFAULT_THRESHOLD = 80.0
# Horizons that GATE an alert. Longer horizons are displayed, not gating.
ALERT_HORIZONS = (24, 72)


@dataclass
class ForecastRanks:
    """Ranks at the last bar of a scored frame, per horizon."""

    bar_time: object
    close: float
    ranks: dict[int, dict[str, float]]  # horizon -> {"up":, "dn":}
    horizons: tuple[int, ...] = ()

    def rank(self, horizon: int, direction: str) -> float:
        return float(self.ranks.get(horizon, {}).get(direction, float("nan")))


def build_states(df: pd.DataFrame, peers: dict | None = None):
    """MarketReader states + attached features for one OHLC frame."""
    df = df.sort_index()
    states = MarketReader().read(df)
    FeatureEngine().attach(df, states, peers)
    return states


def _ranks_from_scorer(scorer, df: pd.DataFrame) -> ForecastRanks:
    """Ranks at the last bar from a constructed scorer."""
    t = len(df) - 1
    ranks: dict[int, dict[str, float]] = {}
    for h in sorted(scorer.table_scores):
        up = scorer.table_scores[h].rank(1, t)
        dn = scorer.table_scores[h].rank(-1, t)
        ranks[h] = {"up": float(up), "dn": float(dn)}
    return ForecastRanks(
        bar_time=df.index[-1],
        close=float(df["close"].iloc[-1]),
        ranks=ranks,
        horizons=tuple(sorted(ranks)),
    )


def score_last_bar_bundle(
    df: pd.DataFrame, bundle: dict, *, peers: dict | None = None
) -> tuple[ForecastRanks | None, str]:
    """Score the last bar of ``df`` against an already-loaded bundle."""
    df = df.sort_index()
    if len(df) < 50:
        return None, f"too few bars to score ({len(df)} < 50)"
    states = build_states(df, peers)
    if not states:
        return None, "no market states from frame"
    scorer = MultiHorizonScorer(bundle, states)
    return _ranks_from_scorer(scorer, df), "forecast scored"


def score_last_bar(
    df: pd.DataFrame, bundle_path: str | Path, *, peers: dict | None = None
) -> tuple[ForecastRanks | None, str]:
    """Score the last bar of ``df`` against the bundle at ``bundle_path``.

    Returns (ranks, note). ``ranks`` is None whenever scoring is
    impossible (too few bars, no bundle, unloadable bundle) — the note
    says why, and the caller must surface it, never crash on it.
    """
    df = df.sort_index()
    if len(df) < 50:
        return None, f"too few bars to score ({len(df)} < 50)"
    states = build_states(df, peers)
    if not states:
        return None, "no market states from frame"
    scorer, why = MultiHorizonScorer.try_load(bundle_path, states)
    if scorer is None:
        return None, why
    return _ranks_from_scorer(scorer, df), f"forecast scored ({why})"


def notable(
    ranks: ForecastRanks, *, threshold: float = DEFAULT_THRESHOLD
) -> tuple[str, int, float] | None:
    """Strongest gating-horizon forecast at/above ``threshold``.

    Returns (direction, horizon, rank) or None. Only 1d/3d gate an
    alert; longer horizons are displayed by format_forecast_alert but
    never trigger one on their own.
    """
    best: tuple[str, int, float] | None = None
    for h in ALERT_HORIZONS:
        for direction in ("up", "dn"):
            r = ranks.rank(h, direction)
            if np.isfinite(r) and r >= threshold:
                if best is None or r > best[2]:
                    best = (direction, h, float(r))
    return best


def _horizon_label(h: int) -> str:
    return {12: "12h", 24: "1d", 72: "3d", 168: "7d", 360: "15d"}.get(h, f"{h}b")


def format_forecast_alert(ranks: ForecastRanks) -> str:
    """Alert text. Labeled a forecast, never a signal, ranks only."""
    parts = []
    for h in ranks.horizons:
        up, dn = ranks.rank(h, "up"), ranks.rank(h, "dn")
        parts.append(f"{_horizon_label(h)} up {up:.0f} / dn {dn:.0f}")
    top = notable(ranks)
    headline = (
        f"big-move rank {top[2]:.0f} {top[0].upper()} "
        f"@{_horizon_label(top[1])}" if top else "no gating rank above threshold"
    )
    return (
        "🔮 FORECAST — not a trade signal, not an order\n"
        f"XAUUSD {ranks.close:,.1f} @ {ranks.bar_time} — {headline}\n"
        + " · ".join(parts)
        + "\nRanks, not probabilities: uncalibrated ensemble scores."
    )


def _fit_entry(df: pd.DataFrame, Xv, atr, horizon: int) -> dict:
    """One horizon's ensemble, fitted on the full frame — the same
    shape as scripts/train_forecast.py, driven by the caller's frame
    (the engine's own MT5 history) instead of an external feed."""
    spec = spec_for(horizon)
    labels = build_labels_for_horizon(df, atr, horizon)
    known = labels["y_up"].notna().to_numpy()
    models: dict[str, dict] = {}
    base_rates: dict[str, float] = {}
    for dname, col in (("up", "y_up"), ("dn", "y_dn")):
        y = labels[col].to_numpy(dtype=float)[known]
        models[dname] = {}
        for name, model in make_models(0).items():
            model.fit(Xv[known], y)
            models[dname][name] = model
        base_rates[dname] = float(y.mean())
    return {
        "models": models,
        "base_rates": base_rates,
        "spec": asdict(spec),
        "labelled_bars": int(known.sum()),
        "definition": (f"big move = favourable excursion >= "
                       f"{spec.fav_mult:g}x ATR before adverse >= "
                       f"{spec.adv_mult:g}x ATR within "
                       f"{spec.horizon} H1 bars"),
    }


def train_from_frame(
    df: pd.DataFrame, out_path: str | Path, *, peers: dict | None = None
) -> dict:
    """Train a bundle from the engine's own frame and save it.

    Returns a summary (horizons, labelled bars, trained_through).
    The bundle is a host artifact (data/ is git-ignored): it is
    produced on the machine that runs the engine, never shipped.
    """
    import joblib

    df = df.sort_index()
    states = build_states(df, peers)
    atr = np.array([s.atr for s in states], dtype=float)
    X = build_feature_frame(states)
    # HistGradientBoosting bins each column at fit time and crashes on
    # a column with fewer than 2 distinct values. On MT5 history there
    # are no peers, so the relative-strength columns are constant by
    # construction — drop constant/all-NaN columns and record the kept
    # set in the bundle, so scoring selects exactly these columns.
    keep = [c for c in X.columns
            if X[c].notna().sum() > 0 and X[c].dropna().nunique() >= 2]
    if not keep:
        raise ValueError("no varying features in frame — cannot train")
    Xv = X[keep].to_numpy(dtype=float)
    horizons = {h: _fit_entry(df, Xv, atr, h) for h in HORIZONS}
    legacy = horizons[72]
    last_known_bar = int(np.where(
        build_labels_for_horizon(df, atr, 72)["y_up"].notna()
        .to_numpy())[0][-1])
    bundle = {
        "models": legacy["models"],
        "features": keep,
        "base_rates": legacy["base_rates"],
        "trained_through": str(df.index[last_known_bar]),
        "definition": legacy["definition"],
        "horizons": horizons,
        "train_frame": "mt5_history",
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return {
        "saved": str(out_path),
        "trained_through": bundle["trained_through"],
        "horizons": {h: horizons[h]["labelled_bars"] for h in HORIZONS},
        "base_rates": legacy["base_rates"],
        "dropped_constant_features": [c for c in X.columns if c not in keep],
    }


@dataclass
class MT5Forecaster:
    """Runner-side forecast service: score H1 closes, alert notables.

    The runner calls :meth:`maybe_alert` on each closed H1 bar with
    the stack's H1 frame. At most one alert per closed bar, only when
    notable. Never touches orders — the runner's order paths do not
    read this object at all.

    The bundle dict is cached and reloaded only when its file mtime
    changes (a retrain), so scoring a new bar does not re-read a
    multi-hundred-MB file. The scorer itself is rebuilt per bar —
    it is constructed from the states, which grow every bar.
    """

    bundle_path: str | Path
    threshold: float = DEFAULT_THRESHOLD
    last_alert_bar: object = None
    _bundle: dict | None = field(default=None, repr=False)
    _bundle_mtime: float | None = field(default=None, repr=False)

    def _load_bundle(self) -> tuple[dict | None, str]:
        import joblib

        path = Path(self.bundle_path)
        if not path.exists():
            return None, f"no forecaster bundle at {path}"
        mtime = path.stat().st_mtime
        if self._bundle is not None and self._bundle_mtime == mtime:
            return self._bundle, "bundle cached"
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 — loader honesty
            return None, f"bundle unloadable: {exc}"
        if not isinstance(bundle, dict) or "features" not in bundle:
            return None, f"bundle at {path} is not a forecast bundle"
        self._bundle, self._bundle_mtime = bundle, mtime
        return bundle, "bundle loaded"

    def maybe_alert(
        self, df: pd.DataFrame, *, peers: dict | None = None
    ) -> tuple[str | None, str]:
        """(alert_text | None, note). ``note`` explains every None."""
        bundle, note = self._load_bundle()
        if bundle is None:
            return None, note
        ranks, note = score_last_bar_bundle(df, bundle, peers=peers)
        if ranks is None:
            return None, note
        if self.last_alert_bar is not None and ranks.bar_time == self.last_alert_bar:
            return None, "already alerted for this bar"
        top = notable(ranks, threshold=self.threshold)
        if top is None:
            return None, "no gating rank above threshold"
        self.last_alert_bar = ranks.bar_time
        return format_forecast_alert(ranks), "forecast alert"
