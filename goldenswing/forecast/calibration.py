"""Forecast E2 — the large-move objective + LIVE calibration.

E1 journaled the ensemble's forecasts (per-horizon up/dn
PERCENTILE RANKS — a rank of 90 means the model's big-move score
for that direction sits at the 90th percentile of its reference
distribution; it is not a probability and nothing in here ever
treats it as one).  E2 adds the two things E1's records were
missing:

1. **The magnitude read.**  Direction is only half the forecast
   question; the other half is "is a LARGE move coming at all,
   whichever way it breaks?"  Derived per horizon from the same
   ranks: ``move_pressure`` (the stronger direction's rank),
   ``lean`` (up − dn), ``conviction`` (|lean|).  A horizon with
   both ranks low is a forecast of QUIET; one with either rank
   extreme is a forecast of motion.  Journaled as its own block
   so downstream consumers (the judgment evidence feed, E3)
   read magnitude and direction as separate facts.

2. **Live calibration.**  Offline walk-forward calibration
   (evaluate.calibration) describes the TRAINING distribution.
   This book describes the LIVE one: every journaled forecast is
   resolved against the bars that actually followed, under the
   label geometry it was trained on (labels.build_labels — the
   barrier race: fav × ATR reached before adv × ATR), and folded
   into rank bands.  A calibrated ranker shows empirical
   resolution rates that CLIMB with the band and a top band whose
   rate is a multiple of the base rate (lift).  The magnitude
   read calibrates the same way: how often did a clean big move
   (either direction) actually resolve, per move-pressure band?

Pure core: the runner owns cadence, journaling and persistence
calls; everything here is a function of its inputs.

F1 (2026-11-03) extends resolution: :func:`realized_outcomes`
captures the realized endpoint return and excursions per horizon
alongside the barrier-race outcome, and the state gains a bounded
``samples`` ring (one entry per resolved forecast-horizon) — the
conditioning pool the outlook (forecast/outlook.py) estimates
from, and the raw material F3 scores the outlook against.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labels import build_labels_for_horizon, spec_for

#: Percentile-rank bands for calibration.  Edges follow the
#: alerting threshold (80) so the band the desk actually acts on
#: is measured on its own.
RANK_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 50.0, "0-50"),
    (50.0, 65.0, "50-65"),
    (65.0, 80.0, "65-80"),
    (80.0, 90.0, "80-90"),
    (90.0, 101.0, "90+"),
)


def band_for_rank(rank: float) -> str | None:
    """The calibration band a percentile rank falls in
    (None for non-finite ranks — an unscored direction is not
    evidence and is never bucketed)."""
    if rank is None or not math.isfinite(float(rank)):
        return None
    r = float(rank)
    for lo, hi, name in RANK_BANDS:
        if lo <= r < hi:
            return name
    return None


def magnitude_block(horizons: dict) -> dict[str, dict[str, Any]]:
    """The large-move read, per horizon, from up/dn ranks.

    ``horizons`` is the journaled shape ``{str(h): {"up": r,
    "dn": r}}``.  Non-finite ranks stay visible as None — a
    horizon the ensemble could not score is reported unscored,
    never as quiet."""
    out: dict[str, dict[str, Any]] = {}
    for key, pair in (horizons or {}).items():
        up = (pair or {}).get("up")
        dn = (pair or {}).get("dn")
        up_f = float(up) if up is not None else float("nan")
        dn_f = float(dn) if dn is not None else float("nan")
        if math.isfinite(up_f) and math.isfinite(dn_f):
            lean = up_f - dn_f
            out[str(key)] = {
                "move_pressure": max(up_f, dn_f),
                "lean": round(lean, 2),
                "conviction": round(abs(lean), 2),
                "side": "up" if lean > 0 else (
                    "dn" if lean < 0 else "flat"),
            }
        elif math.isfinite(up_f) or math.isfinite(dn_f):
            only = up_f if math.isfinite(up_f) else dn_f
            out[str(key)] = {
                "move_pressure": only,
                "lean": None, "conviction": None,
                "side": "up" if math.isfinite(up_f) else "dn",
            }
        else:
            out[str(key)] = {
                "move_pressure": None, "lean": None,
                "conviction": None, "side": None,
            }
    return out


def _bar_position(record: dict, h1: pd.DataFrame) -> int | None:
    """The forecast bar's position in the frame (None when the
    bar is not in the frame at all).  Shared by the label
    resolution and the F1 realized-outcome capture so both read
    the same windows."""
    bar_time = record.get("bar_time")
    if bar_time is None:
        return None
    ts = pd.Timestamp(bar_time)
    index = h1.index
    if ts not in index:
        return None
    return int(len(index) - 1 - index[::-1].get_loc(ts)) \
        if index.duplicated().any() else int(index.get_loc(ts))


def resolve_record(record: dict, h1: pd.DataFrame,
                   atr_values: np.ndarray) -> dict | None:
    """Resolve a journaled forecast against realized H1 bars.

    Returns None when the forecast's bar is not in the frame at
    all (the caller decides whether it fell out of the live
    window).  Otherwise returns ``{horizon: {"y_up": 0/1,
    "y_dn": 0/1}}`` for every horizon whose full label window
    has elapsed, plus ``{"void": [horizons]}`` for horizons whose
    label is unknowable (non-finite ATR at the forecast bar —
    the training pipeline calls those unknown too, never 0).
    Horizons still inside their window are simply absent."""
    pos = _bar_position(record, h1)
    if pos is None:
        return None
    out: dict[str, Any] = {}
    void: list[int] = []
    for key in (record.get("horizons") or {}):
        try:
            horizon = int(key)
            spec_for(horizon)
        except (ValueError, KeyError):
            continue  # unregistered geometry: never labelled ad hoc
        if pos + horizon >= len(h1):
            continue  # window still open
        start = max(0, pos - 2)
        end = pos + horizon + 1
        frame = h1.iloc[start:end]
        atr_slice = atr_values[start:end]
        labels = build_labels_for_horizon(frame, atr_slice, horizon)
        row = labels.iloc[pos - start]
        if not np.isfinite(row["y_up"]) \
                or not np.isfinite(row["y_dn"]):
            void.append(horizon)
            continue
        out[str(horizon)] = {"y_up": int(row["y_up"]),
                             "y_dn": int(row["y_dn"])}
    if void:
        out["void"] = void
    return out


#: Cap on the resolved-sample ring in the calibration state —
#: the F1 outlook's conditioning pool.  Bounded so the state
#: file stays small; the oldest samples age out first.
SAMPLE_RING_CAP = 1500


def realized_outcomes(record: dict, h1: pd.DataFrame) -> dict:
    """F1: realized endpoint return + excursions per horizon for
    a journaled forecast — the resolution side of the outlook.

    Same windows :func:`resolve_record` labels.  Returns
    ``{str(h): {"ret_pct": float, "run_up": float, "run_dn":
    float}}`` for every registered horizon whose full window has
    elapsed: ``ret_pct`` is the endpoint return % (close[t+h] vs
    close[t]); ``run_up`` = max high in the window − close[t]
    and ``run_dn`` = close[t] − min low in the window — the two
    direction-neutral excursion primitives from which F3 derives
    direction-relative favourable/adverse and scores point
    error, interval coverage and excursion quality against the
    journaled outlook.  Horizons still inside their window are
    absent; an unknown bar returns {}."""
    pos = _bar_position(record, h1)
    if pos is None:
        return {}
    close = h1["close"].to_numpy(dtype=float)
    high = h1["high"].to_numpy(dtype=float)
    low = h1["low"].to_numpy(dtype=float)
    ref = close[pos]
    if not np.isfinite(ref) or ref <= 0:
        return {}
    out: dict[str, Any] = {}
    for key in (record.get("horizons") or {}):
        try:
            horizon = int(key)
            spec_for(horizon)
        except (ValueError, KeyError):
            continue  # unregistered geometry: never measured ad hoc
        if pos + horizon >= len(h1):
            continue  # window still open
        window_hi = high[pos + 1: pos + horizon + 1]
        window_lo = low[pos + 1: pos + horizon + 1]
        end_close = close[pos + horizon]
        if not (np.isfinite(end_close)
                and np.isfinite(window_hi).all()
                and np.isfinite(window_lo).all()):
            continue
        out[str(horizon)] = {
            "ret_pct": round(
                float((end_close / ref - 1.0) * 100.0), 4),
            "run_up": round(float(window_hi.max() - ref), 2),
            "run_dn": round(float(ref - window_lo.min()), 2),
        }
    return out


class CalibrationBook:
    """Rank-band calibration over resolved live forecasts.

    Plain-dict backed (JSON-serializable) so the runner can
    persist it inside its state file verbatim."""

    def __init__(self, data: dict | None = None) -> None:
        self.data: dict[str, Any] = data or {
            "dir": {}, "mag": {}, "base_dir": {}, "base_mag": {}}

    def add(self, record: dict, resolution: dict) -> int:
        """Fold one forecast's newly resolved horizons in.
        Returns how many horizons were folded."""
        folded = 0
        horizons = record.get("horizons") or {}
        magnitude = record.get("magnitude") or {}
        for key, outcome in (resolution or {}).items():
            if key == "void" or not isinstance(outcome, dict):
                continue
            pair = horizons.get(key) or horizons.get(int(key)) or {}
            for direction, ykey in (("up", "y_up"), ("dn", "y_dn")):
                rank = pair.get(direction)
                band = band_for_rank(rank) \
                    if rank is not None else None
                y = int(outcome[ykey])
                base = self.data["base_dir"].setdefault(
                    f"{key}|{direction}", {"n": 0, "hits": 0})
                base["n"] += 1
                base["hits"] += y
                if band is not None:
                    slot = self.data["dir"].setdefault(
                        f"{key}|{direction}|{band}",
                        {"n": 0, "sum_rank": 0.0, "hits": 0})
                    slot["n"] += 1
                    slot["sum_rank"] = round(
                        slot["sum_rank"] + float(rank), 2)
                    slot["hits"] += y
            moved = 1 if (outcome["y_up"] or outcome["y_dn"]) else 0
            mag = (magnitude.get(key)
                   or magnitude.get(int(key)) or {})
            pressure = mag.get("move_pressure")
            if pressure is None:
                ranks = [pair.get("up"), pair.get("dn")]
                ranks = [float(r) for r in ranks if r is not None
                         and math.isfinite(float(r))]
                pressure = max(ranks) if ranks else None
            base_m = self.data["base_mag"].setdefault(
                str(key), {"n": 0, "moves": 0})
            base_m["n"] += 1
            base_m["moves"] += moved
            band = band_for_rank(pressure) \
                if pressure is not None else None
            if band is not None:
                slot = self.data["mag"].setdefault(
                    f"{key}|{band}",
                    {"n": 0, "sum_rank": 0.0, "moves": 0})
                slot["n"] += 1
                slot["sum_rank"] = round(
                    slot["sum_rank"] + float(pressure), 2)
                slot["moves"] += moved
            folded += 1
        return folded

    def view(self) -> dict[str, Any]:
        """The readable calibration: per horizon, per direction,
        band rates against the base rate (lift), and the same for
        the magnitude read."""
        def _bands(store: dict, prefix: str, hit_key: str,
                   base: dict | None) -> list[dict[str, Any]]:
            rows = []
            base_rate = None
            if base and base.get("n"):
                base_rate = base["hits" if hit_key == "hits"
                                 else "moves"] / base["n"]
            for _lo, _hi, name in RANK_BANDS:
                slot = store.get(f"{prefix}|{name}")
                if not slot or not slot.get("n"):
                    continue
                rate = slot[hit_key] / slot["n"]
                rows.append({
                    "band": name, "n": slot["n"],
                    "mean_rank": round(
                        slot["sum_rank"] / slot["n"], 1),
                    "rate": round(rate, 4),
                    "lift": (round(rate / base_rate, 2)
                             if base_rate else None),
                })
            return rows

        horizons: dict[str, Any] = {}
        keys = set()
        for store in (self.data["dir"], self.data["mag"]):
            for k in store:
                keys.add(k.split("|")[0])
        for h in sorted(keys, key=lambda x: int(x)):
            entry: dict[str, Any] = {}
            for direction in ("up", "dn"):
                base = self.data["base_dir"].get(f"{h}|{direction}")
                entry[direction] = {
                    "base_rate": (round(base["hits"] / base["n"], 4)
                                  if base and base.get("n") else None),
                    "n": (base or {}).get("n", 0),
                    "bands": _bands(self.data["dir"],
                                    f"{h}|{direction}", "hits", base),
                }
            base_m = self.data["base_mag"].get(h)
            entry["magnitude"] = {
                "base_rate": (round(base_m["moves"] / base_m["n"], 4)
                              if base_m and base_m.get("n") else None),
                "n": (base_m or {}).get("n", 0),
                "bands": _bands(self.data["mag"], h, "moves", base_m),
            }
            horizons[h] = entry
        return {"horizons": horizons}


def fresh_state() -> dict[str, Any]:
    return {"pending": [], "book": {}, "resolved_total": 0,
            "since_summary": 0, "last_summary_day": None,
            "last_cycle_bar": None, "dropped": 0, "void": 0,
            # F1: resolved-sample ring — one entry per resolved
            # (forecast, horizon) with its journaled ranks and
            # realized outcomes; the outlook's conditioning pool.
            "samples": [],
            # F2: the LIVE probability book (forecast/trained.py
            # ProbabilityBook data) — stated probabilities by
            # source, resolved against realized direction classes;
            # graduation reads the trained source from here.
            "prob_book": {}}


def load_state(path: Path) -> tuple[dict[str, Any], bool]:
    """(state, quarantined).  Missing file = fresh state.
    Unreadable/corrupt = the file is renamed aside and a fresh
    state returned — calibration history is evidence, and a torn
    evidence file is quarantined, never silently overwritten."""
    state = fresh_state()
    if not path.exists():
        return state, False
    try:
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError("state is not an object")
        for key, value in loaded.items():
            state[key] = value
        return state, False
    except Exception:  # noqa: BLE001 - quarantine path
        try:
            path.rename(path.with_suffix(".corrupt"))
        except Exception:  # noqa: BLE001
            pass
        return state, True


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Atomic write (tmp + replace), the audit-state pattern."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, separators=(",", ":")))
    tmp.replace(path)
