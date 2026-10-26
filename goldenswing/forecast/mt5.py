"""MT5 forecast service: the engine's own stack frames -> forecast alerts.

Owner rulings §3: the ensemble forecasts big moves (horizons 1 day and
above) and the engine ALERTS those forecasts. It NEVER trades, never
sizes, never gates, never vetoes an order. Forecast output and order
decisions are separate by construction.

The forecast is genuinely multi-timeframe (see forecast/mtf.py): its
features come from the engine's actual M5/M15/H1/H4/D1 closed bars,
as-of aligned, trailing-only — never from one H1 frame with higher
timeframes resampled out of it. The H1-only trainer is retired; it
lives in git history, not in this path.

Lifecycle: the engine trains itself (warm start when untrained or
invalid, D1 closes when 500+ new H1 bars arrived), in the background,
atomically — nobody runs a command, nobody tells it. /forecast_train
exists as the operator's explicit lever, not the only way a fresh
install gets a model. Readiness is full bundle validation, never a
single-key sniff test. Ranks are percentiles against the training
distribution, never probabilities, and every alert says so.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .mtf import (
    BUNDLE_VERSION,
    MIN_H1_BARS_TRAIN,
    needs_training,
    score_last_row,
    train_from_frames,
    _validate_bundle,
)

# Alert gate (build default, tunable — not an owner ruling): a forecast
# alerts when a 1d or 3d big-move rank reaches this percentile.
DEFAULT_THRESHOLD = 80.0
# Horizons that GATE an alert. Longer horizons are displayed, not gating.
ALERT_HORIZONS = (24, 72)


@dataclass
class ForecastRanks:
    """Ranks at the last scored row, per horizon."""

    bar_time: object
    close: float
    ranks: dict[int, dict[str, float]]  # horizon -> {"up":, "dn":}
    horizons: tuple[int, ...] = ()

    def rank(self, horizon: int, direction: str) -> float:
        return float(self.ranks.get(horizon, {}).get(direction, float("nan")))


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


def format_forecast_alert(ranks: ForecastRanks,
                          *, headline: tuple[str, int, float] | None = None) -> str:
    """Alert text. Labeled a forecast, never a signal, ranks only.

    ``headline`` is the crossing that fired the alert; when omitted
    it falls back to the strongest gating rank (display callers).
    """
    parts = []
    for h in ranks.horizons:
        up, dn = ranks.rank(h, "up"), ranks.rank(h, "dn")
        parts.append(f"{_horizon_label(h)} up {up:.0f} / dn {dn:.0f}")
    top = headline if headline is not None else notable(ranks)
    headline_txt = (
        f"big-move rank {top[2]:.0f} {top[0].upper()} "
        f"@{_horizon_label(top[1])}" if top else "no gating rank above threshold"
    )
    return (
        "🔮 FORECAST — not a trade signal, not an order\n"
        f"XAUUSD {ranks.close:,.1f} @ {ranks.bar_time} — {headline_txt}\n"
        + " · ".join(parts)
        + "\nRanks, not probabilities: uncalibrated ensemble scores."
    )


@dataclass
class MT5Forecaster:
    """Runner-side forecast service: score daily closes, alert crossings.

    The runner calls :meth:`maybe_alert` on each closed D1 bar with
    the stack's frames (the frames are the model's input; the D1
    close is the cadence — a days-ahead forecast is daily news, not
    hourly noise). Alerts fire ONLY when a gating rank crosses from
    below to at/above threshold: a rank sitting above threshold does
    not re-alert, and dropping below re-arms the crossing. Never
    touches orders — the runner's order paths do not read this object
    at all.

    The bundle dict is cached and reloaded only when its file mtime
    changes (a retrain), so scoring a new bar does not re-read a
    multi-hundred-MB file.
    """

    bundle_path: str | Path
    threshold: float = DEFAULT_THRESHOLD
    _above: dict[tuple[str, int], bool] = field(
        default_factory=dict, repr=False)
    _bundle: dict | None = field(default=None, repr=False)
    _bundle_mtime: float | None = field(default=None, repr=False)
    _train_result: tuple[bool, str] | None = field(
        default=None, repr=False)
    _training: bool = field(default=False, repr=False)
    _last_evaluation: tuple[object, "ForecastRanks"] | None = field(
        default=None, repr=False)
    _last_error: str | None = field(default=None, repr=False)

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
        valid, reason = _validate_bundle(bundle)
        if not valid:
            return None, f"bundle invalid: {reason}"
        self._bundle, self._bundle_mtime = bundle, mtime
        return bundle, "bundle loaded"

    def maybe_train(
        self, frames: dict
    ) -> tuple[threading.Thread | None, str]:
        """Start background training if the bundle needs it.

        Returns (thread, note). ``thread`` is None when no training is
        needed, one is already running, or history is too short for
        MTF training (under MIN_H1_BARS_TRAIN raw H1 bars: 63 days of
        D1 warmup + 500 usable rows). The outcome lands in
        ``_train_result`` for the runner to journal on the tick
        thread; the worker thread never touches the journal. A failed
        train leaves the old bundle untouched (atomic swap) and
        records the failure — the next D1 close retries. Nothing here
        needs an owner command: the engine knows it needs training
        because this method is called on every D1 close and at warm
        start; /forecast_train is the operator's explicit lever.
        """
        if self._training:
            return None, "training already running"
        h1 = frames.get("H1") if isinstance(frames, dict) else None
        if h1 is None or len(h1) == 0:
            return None, "no H1 frame — cannot train"
        if len(h1) < MIN_H1_BARS_TRAIN:
            return None, (
                f"too few H1 bars for MTF training "
                f"({len(h1)} < {MIN_H1_BARS_TRAIN})")
        need, reason = needs_training(self.bundle_path, h1)
        if not need:
            return None, f"bundle current: {reason}"
        self._training = True
        self._train_result = None
        frames_copy = dict(frames)

        def _run() -> None:
            try:
                summary = train_from_frames(
                    frames_copy, self.bundle_path)
                self._train_result = (
                    True,
                    f"trained through {summary['trained_through']}")
            except Exception as exc:  # noqa: BLE001 — recorded, retried
                self._train_result = (False, f"training failed: {exc}")
            finally:
                self._training = False

        thread = threading.Thread(
            target=_run, name="forecast-train", daemon=True)
        thread.start()
        return thread, f"training started: {reason}"

    def _decide(self, ranks: ForecastRanks) -> tuple[str | None, str]:
        """Crossing-only alert decision for one scored close."""
        crossed: list[tuple[str, int, float]] = []
        for h in ALERT_HORIZONS:
            for direction in ("up", "dn"):
                r = ranks.rank(h, direction)
                is_above = bool(np.isfinite(r) and r >= self.threshold)
                was_above = self._above.get((direction, h), False)
                self._above[(direction, h)] = is_above
                if is_above and not was_above:
                    crossed.append((direction, h, float(r)))
        if not crossed:
            return None, "no new threshold crossing"
        top = max(crossed, key=lambda c: c[2])
        return format_forecast_alert(ranks, headline=top), "forecast alert"

    def maybe_alert(
        self, frames: dict
    ) -> tuple[str | None, str]:
        """(alert_text | None, note). ``note`` explains every None.
        Successful scorings update the last evaluation (for /status);
        failures record the last error."""
        bundle, note = self._load_bundle()
        if bundle is None:
            self._last_error = note
            return None, note
        ranks, note = score_last_row(frames, bundle)
        if ranks is None:
            self._last_error = note
            return None, note
        self._last_error = None
        self._last_evaluation = (ranks.bar_time, ranks)
        return self._decide(ranks)

    def status_dict(self, *, h1_bars: int | None = None) -> dict:
        """Real runtime forecast status for /status. Every value is
        read from the live bundle / live state — nothing hard-coded.
        ``ready`` is True only when a fully validated bundle is
        present; an untrained, unloadable, corrupt, or old-format
        bundle reports ready False with the precise reason."""
        bundle, note = self._load_bundle()
        ready = bundle is not None
        meta = bundle if isinstance(bundle, dict) else {}
        last_eval = None
        if self._last_evaluation is not None:
            bar_time, ranks = self._last_evaluation
            last_eval = {
                "bar_time": str(bar_time),
                "ranks": {h: ranks.ranks[h] for h in ranks.horizons},
            }
        return {
            "ready": ready,
            "reason": note,
            "bundle_path": str(self.bundle_path),
            "model_loaded": ready,
            "version": meta.get("version"),
            "feature_set": meta.get("feature_set"),
            "symbol": meta.get("symbol"),
            "trained_at": meta.get("trained_at"),
            "trained_through": meta.get("trained_through"),
            "train_bars": meta.get("train_bars"),
            "h1_bars": h1_bars,
            "labelled_bars": {
                h: (meta.get("horizons", {}).get(h)
                    or meta.get("horizons", {}).get(str(h), {})).get(
                        "labelled_bars")
                for h in (12, 24, 72, 168, 360)} if ready else None,
            "validation": meta.get("validation"),
            "last_evaluation": last_eval,
            "last_error": self._last_error,
            "training_running": self._training,
            "last_train_result": self._train_result,
            "alerts_enabled": ready,
            "threshold": self.threshold,
        }
