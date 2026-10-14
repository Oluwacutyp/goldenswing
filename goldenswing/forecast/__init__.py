"""Ensemble ML forecaster (Phase 6 + 6.5).

Predicts, at each H1 bar, whether price will make a *big move*
within a horizon window: favourable excursion reaching >= fav x ATR
before adverse excursion reaches >= adv x ATR (see labels.py).
Phase 6 built this for the 72-bar (3-day) horizon; Phase 6.5 —
after the owner's "not limited to 3 days" — generalises it to every
registered horizon (12 / 24 / 72 / 168 / 360 bars, each with its own
barrier geometry and purge).  Three diverse learners (gradient
boosting, logistic regression, extra-trees) are trained walk-forward
with purged/embargoed folds — never on a random shuffle — and graded
on discrimination, calibration, and trading tests against the plain
Combo-A detector book.

Modules:
- ``labels`` — horizon-general big-move labels + HORIZON_SPECS.
- ``walkforward`` — purged expanding-window walk-forward engine.
- ``evaluate`` / ``trading`` — grading + gate/standalone/null tests.
- ``horizons`` — per-horizon evaluation + agreement test (6.5).
- ``api`` — deployment-side per-horizon rank scores for alerts.
- ``run`` — the Phase-6 single-horizon (72-bar) run -> FORECAST.md.
"""
