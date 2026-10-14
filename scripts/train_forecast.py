#!/usr/bin/env python3
"""Train the FINAL forecaster ensembles on all labelled data and
save them for live/paper use.

This is the deployment artefact companion to scripts/forecast.py
(Phase 6, the honest 72-bar *evaluation*) and
scripts/forecast_horizons.py (Phase 6.5, the per-horizon
evaluation): the walk-forward runs there are the honest grading;
this script fits the same three models per direction on the full
sample — once per registered horizon — and persists them (joblib)
together with the feature column order, so an alerting loop can
score the latest bar at every horizon without retraining.

Saved bundle (data/forecast_models.joblib — data/ is git-ignored):
    {"models": <72-bar entry, the Phase-6 legacy top level>,
     "features": [column names], "base_rates": {...},
     "trained_through": <last bar timestamp>,
     "horizons": {horizon: {"models": {"up": {...}, "dn": {...}},
                            "base_rates": {...},
                            "spec": {...}, "labelled_bars": int}}}

The top-level "models"/"base_rates" remain the 72-bar ensemble so
the Phase-6/8 consumers (execution ForecasterScorer) work
unchanged; "horizons" is what forecast.api.MultiHorizonScorer reads.

Usage: python3 scripts/train_forecast.py
"""
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.backtest.data import load_gold_h1, load_peers  # noqa: E402
from goldenswing.features import FeatureEngine  # noqa: E402
from goldenswing.forecast.features import build_feature_frame  # noqa: E402
from goldenswing.forecast.labels import (HORIZONS,  # noqa: E402
                                        build_labels_for_horizon,
                                        spec_for)
from goldenswing.forecast.walkforward import make_models  # noqa: E402
from goldenswing.reader.reader import MarketReader  # noqa: E402


def _fit_entry(df, Xv, atr, horizon: int) -> dict:
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


def main() -> None:
    df = load_gold_h1()
    peers = load_peers()
    states = MarketReader().read(df)
    FeatureEngine().attach(df, states, peers)
    atr = np.array([s.atr for s in states], dtype=float)

    X = build_feature_frame(states)
    Xv = X.to_numpy(dtype=float)

    horizons = {h: _fit_entry(df, Xv, atr, h) for h in HORIZONS}
    legacy = horizons[72]
    last_known_bar = int(np.where(
        build_labels_for_horizon(df, atr, 72)["y_up"].notna()
        .to_numpy())[0][-1])

    bundle = {
        "models": legacy["models"],
        "features": list(X.columns),
        "base_rates": legacy["base_rates"],
        "trained_through": str(df.index[last_known_bar]),
        "definition": legacy["definition"],
        "horizons": horizons,
    }
    out = Path(__file__).resolve().parents[1] / "data" / "forecast_models.joblib"
    joblib.dump(bundle, out)
    for h in HORIZONS:
        e = horizons[h]
        print(f"h={h:>3}: trained on {e['labelled_bars']} labelled "
              f"bars; base rates up {e['base_rates']['up']:.3f} / "
              f"dn {e['base_rates']['dn']:.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
