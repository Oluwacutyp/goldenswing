"""Forecast F3 — the acceptance battery.

F1 (forecast/outlook.py) put numbers on every journaled
forecast; F2 (forecast/trained.py) put trained estimators under
them, with probabilities that render as calibrated only when the
live book's graduation gates earn it.  F3 is the measurement
stage: **how good is each method, per horizon, on the record it
actually left?**  It answers from the journal alone —

- every ``forecast`` record carries its outlook block (the
  statement as made: direction, point price, interval, expected
  excursions, disagreement, and the probabilities it stated at
  the time);
- every ``forecast_reconciliation`` record carries what
  actually happened for that ``bar_time`` (barrier-race labels
  ``y_up`` / ``y_dn`` plus the F1 realized outcomes: endpoint
  return ``ret_pct`` and excursions ``run_up`` / ``run_dn``).

The two join on ``bar_time`` — the same key the runner's
calibration registration dedupes on (first journaled forecast
for a bar is the statement of record; reconciliation appends
for a bar merge, first value per horizon wins, exactly as the
cycle never re-resolves a horizon).

**What is scored, per horizon × method.**  The ANSWERING method
is the horizon's own ``method`` stamp: ``f2-trained-v1`` when a
graduated trained estimator answered, ``band_conditional`` /
``unconditional_h1`` (reported together as ``f1-empirical``)
when F1 did.  Against the realized outcomes:

- point error — MAE / median AE of ``point_price`` vs the
  realized endpoint price, in USD and %, plus the **no-change
  naive** on the same rows (point = the forecast's own close):
  the skill ratio against it is the verdict's point bar — a
  forecast that cannot beat assuming no move has not earned its
  number;
- interval quality — empirical coverage of the stated interval
  against its own stated nominal coverage, and mean width;
- direction — hit rate of UP/DOWN calls where the realized
  direction class uses the deadband THE FORECAST STATED
  (``trained.direction_class`` — the same convention the live
  probability book resolves under); NEUTRAL calls are scored
  separately as abstention quality (how often staying out was
  right — the realized class really was neutral);
- probability calibration — reliability by decile bin, ECE and
  multiclass Brier, over statements whose probabilities were
  stated as ``calibrated`` AT THE TIME (rendered block state).
  Never retroactively: a horizon that stated ``uncalibrated``
  contributes nothing here, whatever it later became.  The
  trained estimator's pre-graduation SHADOW statements are
  measured as their own clearly-named series (they are evidence
  for the graduation question) but they were stated as
  ``shadow``, not ``calibrated``, so they never enter the
  calibrated verdicts;
- large-move calls — stated ``large_up`` / ``large_dn``
  probabilities vs the realized barrier races: mean stated vs
  realized rate, Brier, and precision/recall at the stated call
  line (``LARGE_CALL_P``);
- excursion quality — MAE of the favourable/adverse excursion
  estimates vs the realized direction-relative runs;
- the disagreement field, on trial — the cell's rows split at
  their own disagreement median: if high-disagreement forecasts
  are not measurably worse (point error), the field is flagged
  as decoration for that cell instead of assumed useful;
- regime splits — the headline metrics (point MAE %, direction
  hit rate, coverage) split by session (bar-clock hours), by
  volatility band (interval-width terciles), and by the
  ensemble's dominant side.  Tercile edges are fixed on the
  EVALUATION segment only and applied unchanged to the
  holdout — the holdout does not even choose its own bins.

**Protocol (S4 discipline, non-negotiable).**  Scored bars are
walked in bar-time order.  The final ``HOLDOUT_FRAC`` of scored
bars is a LOCKED HOLDOUT: scored once, with the same stated
bands, never fitted on, never threshold-chosen on — this module
contains no fitting at all; every band below is a stated
constant with its reason, fixed in code before any journal was
read.  The evaluation segment is additionally shown in
``WF_WINDOWS`` chronological walk-forward windows so drift is
visible rather than averaged away.  Every metric carries its
n; a metric under ``REPORT_MIN_N`` renders as
``insufficient data (n=…)`` — never a number dressed up.

**Verdicts.**  Per horizon × method, from the holdout segment
when its n is verdict-grade (``VERDICT_MIN_N``), else from all
scored rows and marked provisional.  A cell is WITHIN BAND when
every applicable stated bar passes: interval coverage within
``COVERAGE_TOL`` of nominal, ECE at or under ``ECE_MAX`` (the
same bar F2's graduation uses), direction hit rate above the
chance line, point skill better than the naive.  Components
with insufficient n neither pass nor fail — they are named as
unmeasured.  The battery NEVER promotes anything: no order,
sizing, risk or graduation path reads this module.  It reports;
promotion stays a separate owner decision.

Pure core: functions of the journal records handed in.  No LLM
anywhere in computation or narration; rendering is templates
over computed fields, exactly as S4 renders.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .outlook import INTERVAL_COVERAGE, TRAINED_METHOD, tag_for
from .trained import direction_class

#: Method names as the report speaks of them.
METHOD_F1 = "f1-empirical"
METHOD_F2 = "f2-trained"
SHADOW_SERIES = "f2-trained-shadow"
_F1_METHODS = ("band_conditional", "unconditional_h1")

#: Chronology protocol (S4 precedent: HOLDOUT_FRAC = 0.30, and
#: the evaluation segment shown in chronological windows so a
#: method that decays is seen decaying).
HOLDOUT_FRAC = 0.30
WF_WINDOWS = 3

#: Evidence floors.  REPORT_MIN_N follows the house trust
#: precedent (E3's TRUST_MIN_N = 20, F1's MIN_PROB_N = 20: below
#: twenty resolved statements a frequency is an anecdote).
#: VERDICT_MIN_N follows S4's BAR_MIN_PAIRS = 60 and F2's
#: GRAD_MIN_N = 60 (the n at which this system lets a number
#: carry a decision-grade claim).
REPORT_MIN_N = 20
VERDICT_MIN_N = 60

#: Verdict bands — stated constants, fixed before any journal
#: was read, each with its reason:
#: - coverage: at verdict-grade n=60 the binomial standard
#:   error of an honest 80% interval is ~0.052, so ±0.10 is
#:   ~±2σ — inside the band is statistically consistent with
#:   the stated coverage; outside is miscalibration, not noise.
COVERAGE_TOL = 0.10
#: - ECE: the same 0.10 bar F2's live graduation gate uses
#:   (GRAD_MAX_ECE) — the battery holds a method to the
#:   standard the system already states for itself.
ECE_MAX = 0.10
#: - direction: the chance line.  A directional forecaster
#:   must beat a coin on the calls it chose to make.
DIRECTION_MIN_HIT = 0.50
#: - point: skill ratio vs the no-change naive on the same
#:   rows must be under 1.0 — beat doing nothing.
POINT_SKILL_MAX = 1.00
#: - large-move call line: the barrier race is a minority
#:   event by construction (the favourable barrier stands
#:   1.5–4× ATR away; realized base rates sit well under 50%),
#:   so the canonical 0.5 line would grade almost every honest
#:   statement as "never called".  0.40 is where a stated
#:   probability commits to "this race is live".  The Brier
#:   score beside it grades the probabilities themselves,
#:   threshold-free.
LARGE_CALL_P = 0.40

#: Reliability bins over stated probability (trained.py's
#: PROB_BINS precedent).
PROB_BINS = 10
#: Regime splits: a cell needs this many scored rows before
#: its arms mean anything, and each arm needs its own floor.
SPLIT_MIN_CELL = 30
SPLIT_MIN_ARM = 10
#: Disagreement diagnostic: each side of the median split
#: needs at least this many rows before the ratio is read,
#: and "measurably worse" means a 10% worse point error —
#: inside ±10% the field carries no measured signal.
DISAGREE_MIN_ARM = 15
DISAGREE_WORSE_RATIO = 1.10

#: Session bands on the journal's own bar clock (the bars are
#: H1; named as clock bands, not claimed exchange sessions).
SESSIONS: tuple[tuple[int, int, str], ...] = (
    (0, 8, "asia"), (8, 13, "london"), (13, 21, "new_york"),
    (21, 24, "off_hours"))


def read_journal(path: str | Path) -> tuple[list[dict], int]:
    """Tolerant JSONL read: returns (records, malformed_count).
    Mirrors the S4 reader exactly: a torn final line (the
    runner was mid-append at a crash) is counted, never fatal —
    the journal is evidence, and evidence with a torn edge
    still testifies."""
    records: list[dict] = []
    malformed = 0
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if isinstance(row, dict):
            records.append(row)
        else:
            malformed += 1
    return records, malformed


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _r(value: float | None, nd: int = 4) -> float | None:
    return None if value is None else round(float(value), nd)


def _mean(values: list[float]) -> float | None:
    return _r(sum(values) / len(values)) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2:
        return _r(s[mid])
    return _r((s[mid - 1] + s[mid]) / 2.0)


def _session_of(bar_time: str) -> str | None:
    try:
        import pandas as pd

        hour = int(pd.Timestamp(bar_time).hour)
    except Exception:  # noqa: BLE001 — unparseable stamp: no split
        return None
    for lo, hi, name in SESSIONS:
        if lo <= hour < hi:
            return name
    return None


def _method_of(ho: dict) -> str | None:
    """The answering method a horizon block names: the trained
    stamp, the F1 estimation methods (one family — the report
    grades F1's empirical estimation as one method, its two
    sample sources being F1's own internal fallback), or None
    for a horizon that never answered."""
    method = str(ho.get("method") or "")
    if method == TRAINED_METHOD:
        return METHOD_F2
    if method in _F1_METHODS:
        return METHOD_F1
    return None


# ── the join: journal records -> scored rows ────────────────────


def build_scored_rows(records: list[dict]) -> tuple[list[dict],
                                                    dict]:
    """Join forecasts to their reconciliations on bar_time and
    flatten to one scored row per (forecast, horizon) whose
    outlook answered and whose outcome was realized.

    Returns (rows, coverage) — coverage counts the journal's
    shape honestly (forecasts seen, without outlook, bars never
    reconciled, horizons not yet resolved), in S4's
    pairs/pending spirit: pending is reported, never scored."""
    forecasts: dict[str, dict] = {}
    order: list[str] = []
    n_forecast = 0
    n_with_outlook = 0
    for row in records:
        if row.get("type") != "forecast":
            continue
        n_forecast += 1
        bar = str(row.get("bar_time") or "")
        if not bar or bar in forecasts:
            continue  # registration dedupe precedent: first wins
        forecasts[bar] = row
        order.append(bar)
        if isinstance(row.get("outlook"), dict):
            n_with_outlook += 1
    reconciled: dict[str, dict[str, Any]] = {}
    for row in records:
        if row.get("type") != "forecast_reconciliation":
            continue
        bar = str(row.get("bar_time") or "")
        if not bar:
            continue
        slot = reconciled.setdefault(bar, {})
        for key, outcome in (row.get("horizons") or {}).items():
            if isinstance(outcome, dict) \
                    and str(key) not in slot:
                slot[str(key)] = outcome
    rows: list[dict] = []
    n_unresolved = 0
    for bar in order:
        record = forecasts[bar]
        outlook = record.get("outlook")
        if not isinstance(outlook, dict):
            continue
        outcomes = reconciled.get(bar) or {}
        close = _f(record.get("close"))
        ranks = record.get("horizons") or {}
        for key, ho in (outlook.get("horizons") or {}).items():
            if not isinstance(ho, dict):
                continue
            method = _method_of(ho)
            outcome = outcomes.get(str(key)) or {}
            ret = _f(outcome.get("ret_pct"))
            if method is None or ret is None or close is None \
                    or close <= 0:
                if method is not None:
                    n_unresolved += 1
                continue
            horizon = int(key)
            pair = ranks.get(str(key)) or ranks.get(horizon) \
                or {}
            up_r, dn_r = _f(pair.get("up")), _f(pair.get("dn"))
            dom = None
            if up_r is not None and dn_r is not None:
                dom = "up" if up_r >= dn_r else "dn"
            barriers = ho.get("barriers") or {}
            eff = None
            if ho.get("direction") in ("UP", "DOWN"):
                eff = str(ho["direction"]).lower()
            elif barriers.get("side") in ("up", "dn"):
                eff = str(barriers["side"])
            elif dom is not None:
                eff = dom
            run_up, run_dn = _f(outcome.get("run_up")), \
                _f(outcome.get("run_dn"))
            fav = adv = None
            if run_up is not None and run_dn is not None \
                    and eff is not None:
                fav = run_up if eff == "up" else run_dn
                adv = run_dn if eff == "up" else run_up
            interval = ho.get("interval") or {}
            probs = ho.get("probabilities") or {}
            shadow = ho.get("trained_shadow")
            rows.append({
                "bar_time": bar,
                "generated_at": str(record.get("generated_at")
                                    or ""),
                "cadence": record.get("cadence"),
                "horizon": horizon,
                "method": method,
                "close": close,
                "direction": ho.get("direction"),
                "point_price": _f(ho.get("point_price")),
                "interval_low": _f(interval.get("low")),
                "interval_high": _f(interval.get("high")),
                "interval_coverage": _f(interval.get("coverage"))
                or INTERVAL_COVERAGE,
                "fav_est": _f(ho.get("fav_excursion_usd")),
                "adv_est": _f(ho.get("adv_excursion_usd")),
                "disagreement": _f(ho.get("disagreement")),
                "deadband_pct": _f(ho.get("deadband_pct")),
                "probs": (probs if probs.get("state")
                          == "calibrated" else None),
                "shadow": (shadow if isinstance(shadow, dict)
                           and shadow.get("up") is not None
                           else None),
                "dom_side": dom,
                "session": _session_of(bar),
                "ret_pct": ret,
                "endpoint": close * (1.0 + ret / 100.0),
                "run_up": run_up, "run_dn": run_dn,
                "fav_real": fav, "adv_real": adv,
                "y_up": outcome.get("y_up"),
                "y_dn": outcome.get("y_dn"),
            })
    rows.sort(key=lambda r: (r["bar_time"], r["horizon"]))
    coverage = {
        "forecast_records": n_forecast,
        "forecast_bars": len(forecasts),
        "bars_with_outlook": n_with_outlook,
        "bars_reconciled": sum(
            1 for bar in order if bar in reconciled),
        "scored_rows": len(rows),
        "horizons_unresolved": n_unresolved,
    }
    return rows, coverage


def _split_segments(rows: list[dict]) -> tuple[dict[str,
                                                    list[dict]],
                                               str | None]:
    """Chronological split of scored rows at BAR granularity
    (a bar's horizons never straddle the line): the last
    HOLDOUT_FRAC of scored bars is the locked holdout.
    Returns ({segment: rows}, boundary_bar_time)."""
    bars = sorted({r["bar_time"] for r in rows})
    split = int(len(bars) * (1.0 - HOLDOUT_FRAC))
    if split:
        boundary = bars[split - 1]
        return ({
            "eval": [r for r in rows if r["bar_time"] <= boundary],
            "holdout": [r for r in rows
                        if r["bar_time"] > boundary],
        }, boundary)
    return ({"eval": [], "holdout": list(rows)}, None)


def _vol_edges(eval_rows: list[dict]) -> tuple[float,
                                               float] | None:
    """Interval-width tercile edges from the EVALUATION segment
    only (pooled across cells — a descriptive split needs the
    numbers a per-cell split cannot carry).  Applied unchanged
    to both segments; None when the eval segment is too thin to
    fix honest edges."""
    widths = sorted(
        (r["interval_high"] - r["interval_low"]) / r["close"]
        * 100.0
        for r in eval_rows
        if r["interval_low"] is not None
        and r["interval_high"] is not None)
    if len(widths) < SPLIT_MIN_CELL:
        return None
    lo = widths[int(len(widths) / 3.0)]
    hi = widths[int(2.0 * len(widths) / 3.0)]
    if hi <= lo:
        return None
    return (round(lo, 4), round(hi, 4))


# ── metric families over a row set ──────────────────────────────


def _point_metrics(rows: list[dict]) -> dict[str, Any]:
    errs = [r for r in rows if r["point_price"] is not None]
    if not errs:
        return {"n": 0, "mae_usd": None, "medae_usd": None,
                "mae_pct": None, "medae_pct": None,
                "naive_mae_pct": None, "skill": None}
    usd = [abs(r["point_price"] - r["endpoint"]) for r in errs]
    pct = [u / r["close"] * 100.0 for u, r in zip(usd, errs)]
    naive = [abs(r["ret_pct"]) for r in errs]
    mae_pct, naive_pct = _mean(pct), _mean(naive)
    return {
        "n": len(errs),
        "mae_usd": _mean(usd), "medae_usd": _median(usd),
        "mae_pct": mae_pct, "medae_pct": _median(pct),
        "naive_mae_pct": naive_pct,
        "skill": (_r(mae_pct / naive_pct, 4)
                  if mae_pct is not None and naive_pct else None),
    }


def _interval_metrics(rows: list[dict]) -> dict[str, Any]:
    iv = [r for r in rows if r["interval_low"] is not None
          and r["interval_high"] is not None]
    if not iv:
        return {"n": 0, "coverage": None, "mean_width_pct": None,
                "target": None}
    covered = sum(1 for r in iv
                  if r["interval_low"] <= r["endpoint"]
                  <= r["interval_high"])
    widths = [(r["interval_high"] - r["interval_low"])
              / r["close"] * 100.0 for r in iv]
    return {"n": len(iv),
            "coverage": _r(covered / len(iv)),
            "mean_width_pct": _mean(widths),
            "target": iv[0]["interval_coverage"]}


def _direction_metrics(rows: list[dict]) -> dict[str, Any]:
    calls = hits = 0
    neutral = neutral_right = 0
    for r in rows:
        realized = direction_class(r["ret_pct"],
                                   r["deadband_pct"])
        if r["direction"] in ("UP", "DOWN"):
            calls += 1
            want = "up" if r["direction"] == "UP" else "dn"
            hits += 1 if realized == want else 0
        elif r["direction"] == "NEUTRAL":
            neutral += 1
            neutral_right += 1 if realized == "neutral" else 0
    return {
        "calls": calls,
        "hits": hits,
        "hit_rate": (_r(hits / calls) if calls else None),
        "neutral_n": neutral,
        "neutral_right": neutral_right,
        "neutral_rate": (_r(neutral_right / neutral)
                         if neutral else None),
    }


def _prob_metrics(stated: list[tuple[dict, str | None,
                                     float | None]]) -> dict:
    """Reliability / ECE / Brier over stated probability
    triples: (probs, realized_class, deadband_unused).  Each
    statement contributes its three class probabilities; bins
    are deciles of the stated probability, pooled over classes
    for the ECE (the ProbabilityBook's own convention)."""
    cells: dict[tuple[str, int], dict[str, float]] = {}
    briers: list[float] = []
    n = 0
    for probs, realized, _db in stated:
        vals = {cls: _f(probs.get(cls))
                for cls in ("up", "dn", "neutral")}
        if realized is None or any(v is None
                                   for v in vals.values()):
            continue
        n += 1
        briers.append(sum(
            (float(vals[cls]) - (1.0 if realized == cls else 0.0))
            ** 2 for cls in vals))
        for cls, p in vals.items():
            b = min(PROB_BINS - 1, int(float(p) * PROB_BINS))
            cell = cells.setdefault((cls, b),
                                    {"n": 0, "sum": 0.0, "hits": 0})
            cell["n"] += 1
            cell["sum"] += float(p)
            cell["hits"] += 1 if realized == cls else 0
    total = sum(int(c["n"]) for c in cells.values())
    ece = None
    if total:
        ece = _r(sum(abs(c["sum"] / c["n"] - c["hits"] / c["n"])
                     * c["n"] for c in cells.values()) / total)
    bins = [{
        "class": cls, "bin": f"{b / PROB_BINS:.1f}-"
        f"{(b + 1) / PROB_BINS:.1f}", "n": int(c["n"]),
        "stated": _r(c["sum"] / c["n"]),
        "realized": _r(c["hits"] / c["n"]),
    } for (cls, b), c in sorted(cells.items()) if c["n"]]
    return {"n": n, "ece": ece, "brier": _mean(briers),
            "bins": bins}


def _rendered_probs(rows: list[dict]) -> list:
    return [(r["probs"],
             direction_class(r["ret_pct"], r["deadband_pct"]),
             r["deadband_pct"])
            for r in rows if r["probs"] is not None]


def _shadow_probs(rows: list[dict]) -> list:
    return [(r["shadow"],
             direction_class(r["ret_pct"],
                             _f((r["shadow"] or {})
                                .get("deadband_pct"))),
             None)
            for r in rows if r["shadow"] is not None]


def _large_metrics(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side, pkey, ykey in (("up", "large_up", "y_up"),
                             ("dn", "large_dn", "y_dn")):
        stated = [(float(r["probs"][pkey]),
                   1 if r[ykey] else 0)
                  for r in rows
                  if r["probs"] is not None
                  and r["probs"].get(pkey) is not None
                  and r[ykey] is not None]
        if not stated:
            out[side] = {"n": 0, "mean_stated": None,
                         "realized_rate": None, "brier": None,
                         "calls": 0, "precision": None,
                         "recall": None}
            continue
        calls = [(p, y) for p, y in stated if p >= LARGE_CALL_P]
        tp = sum(1 for _p, y in calls if y)
        fp = len(calls) - tp
        fn = sum(1 for p, y in stated
                 if p < LARGE_CALL_P and y)
        out[side] = {
            "n": len(stated),
            "mean_stated": _mean([p for p, _y in stated]),
            "realized_rate": _r(
                sum(y for _p, y in stated) / len(stated)),
            "brier": _mean([(p - y) ** 2 for p, y in stated]),
            "calls": len(calls),
            "precision": (_r(tp / (tp + fp)) if tp + fp else None),
            "recall": (_r(tp / (tp + fn)) if tp + fn else None),
        }
    return out


def _excursion_metrics(rows: list[dict]) -> dict[str, Any]:
    both = [r for r in rows if r["fav_est"] is not None
            and r["fav_real"] is not None]
    if not both:
        return {"n": 0, "fav_mae_usd": None, "adv_mae_usd": None}
    return {
        "n": len(both),
        "fav_mae_usd": _mean(
            [abs(r["fav_est"] - r["fav_real"]) for r in both]),
        "adv_mae_usd": _mean(
            [abs(r["adv_est"] - r["adv_real"]) for r in both
             if r["adv_est"] is not None
             and r["adv_real"] is not None]),
    }


def _disagreement_metrics(rows: list[dict]) -> dict[str, Any]:
    """The disagreement field on trial: split the cell's rows
    at their own disagreement median.  The field earns its
    place only when the high-disagreement half is measurably
    WORSE on point error (ratio at or over
    DISAGREE_WORSE_RATIO); near-parity flags it as decoration;
    an inverted ratio says the field misleads."""
    sub = [r for r in rows if r["disagreement"] is not None
           and r["point_price"] is not None]
    base = {"n": len(sub), "median": None, "high": None,
            "low": None, "mae_ratio": None, "reading": None}
    if len(sub) < 2 * DISAGREE_MIN_ARM:
        return base
    med = _median([r["disagreement"] for r in sub])
    high = [r for r in sub if r["disagreement"] > med]
    low = [r for r in sub if r["disagreement"] <= med]
    if len(high) < DISAGREE_MIN_ARM or len(low) < DISAGREE_MIN_ARM:
        base["median"] = med
        return base

    def arm(rs: list[dict]) -> dict[str, Any]:
        d = _direction_metrics(rs)
        return {"n": len(rs),
                "mae_pct": _point_metrics(rs)["mae_pct"],
                "hit_rate": d["hit_rate"], "calls": d["calls"]}

    hi, lo = arm(high), arm(low)
    ratio = None
    if hi["mae_pct"] is not None and lo["mae_pct"]:
        ratio = _r(hi["mae_pct"] / lo["mae_pct"], 4)
    reading = None
    if ratio is not None:
        if ratio >= DISAGREE_WORSE_RATIO:
            reading = "earns_its_place"
        elif ratio <= 1.0 / DISAGREE_WORSE_RATIO:
            reading = "inverted"
        else:
            reading = "no_measured_value"
    return {"n": len(sub), "median": med, "high": hi, "low": lo,
            "mae_ratio": ratio, "reading": reading}


def _score_rows(rows: list[dict]) -> dict[str, Any]:
    """Every metric family over one row set (a segment of one
    horizon × method cell, or a regime arm)."""
    return {
        "n": len(rows),
        "point": _point_metrics(rows),
        "interval": _interval_metrics(rows),
        "direction": _direction_metrics(rows),
        "probabilities": _prob_metrics(_rendered_probs(rows)),
        "shadow_probabilities": _prob_metrics(
            _shadow_probs(rows)),
        "large_move": _large_metrics(rows),
        "excursions": _excursion_metrics(rows),
        "disagreement": _disagreement_metrics(rows),
    }


def _wf_windows(rows: list[dict]) -> list[dict[str, Any]]:
    """The evaluation segment in chronological walk-forward
    windows: point error per window, so drift shows."""
    if not rows:
        return []
    out = []
    for i in range(WF_WINDOWS):
        chunk = rows[i * len(rows) // WF_WINDOWS:
                     (i + 1) * len(rows) // WF_WINDOWS]
        if chunk:
            out.append({"window": i + 1, "n": len(chunk),
                        "mae_pct":
                        _point_metrics(chunk)["mae_pct"]})
    return out


def _headline(rows: list[dict]) -> dict[str, Any]:
    """The regime-split headline: the three numbers every arm
    is compared on."""
    p = _point_metrics(rows)
    d = _direction_metrics(rows)
    iv = _interval_metrics(rows)
    return {"n": len(rows), "mae_pct": p["mae_pct"],
            "dir_calls": d["calls"], "dir_hit_rate": d["hit_rate"],
            "interval_n": iv["n"], "coverage": iv["coverage"]}


def _splits(rows: list[dict],
            edges: tuple[float, float] | None) -> dict:
    """Regime splits for one cell (pooled segments — a split is
    a descriptive cut, stated as such; verdicts never read
    splits).  Arms under SPLIT_MIN_ARM are omitted, not
    dressed up."""
    if len(rows) < SPLIT_MIN_CELL:
        return {}

    def arm_map(keyfn) -> dict[str, Any]:
        arms: dict[str, list[dict]] = {}
        for r in rows:
            k = keyfn(r)
            if k is not None:
                arms.setdefault(k, []).append(r)
        return {k: _headline(v) for k, v in sorted(arms.items())
                if len(v) >= SPLIT_MIN_ARM}

    def vol_of(r: dict) -> str | None:
        if edges is None or r["interval_low"] is None \
                or r["interval_high"] is None:
            return None
        w = (r["interval_high"] - r["interval_low"]) \
            / r["close"] * 100.0
        return ("calm" if w < edges[0]
                else "mid" if w < edges[1] else "storm")

    return {
        "session": arm_map(lambda r: r["session"]),
        "volatility": arm_map(vol_of),
        "dominant_side": arm_map(
            lambda r: (f"{r['dom_side']}-dominant"
                       if r["dom_side"] else None)),
    }


# ── verdicts ────────────────────────────────────────────────────


def _verdict(scored: dict[str, dict]) -> dict[str, Any]:
    """One cell's verdict, from the holdout when the holdout is
    verdict-grade, else from all scored rows (provisional —
    stated, never hidden).  Components with insufficient n are
    unmeasured: they neither pass nor fail."""
    hold = scored["holdout"]
    pooled = scored["pooled"]
    if hold["point"]["n"] >= VERDICT_MIN_N:
        seg_name, seg = "holdout", hold
        firm = True
    else:
        seg_name, seg = "pooled", pooled
        firm = False
    checks: dict[str, Any] = {}
    fails: list[str] = []
    iv, pr, di, po = seg["interval"], seg["probabilities"], \
        seg["direction"], seg["point"]
    if iv["n"] >= REPORT_MIN_N and iv["coverage"] is not None:
        ok = abs(iv["coverage"] - (iv["target"]
                                   or INTERVAL_COVERAGE)) \
            <= COVERAGE_TOL
        checks["coverage"] = {"ok": ok, "value": iv["coverage"],
                              "n": iv["n"]}
        if not ok:
            fails.append("coverage")
    if pr["n"] >= REPORT_MIN_N and pr["ece"] is not None:
        ok = pr["ece"] <= ECE_MAX
        checks["ece"] = {"ok": ok, "value": pr["ece"],
                         "n": pr["n"]}
        if not ok:
            fails.append("ece")
    if di["calls"] >= REPORT_MIN_N and di["hit_rate"] is not None:
        ok = di["hit_rate"] > DIRECTION_MIN_HIT
        checks["direction"] = {"ok": ok, "value": di["hit_rate"],
                               "n": di["calls"]}
        if not ok:
            fails.append("direction")
    if po["n"] >= REPORT_MIN_N and po["skill"] is not None:
        ok = po["skill"] < POINT_SKILL_MAX
        checks["point_skill"] = {"ok": ok, "value": po["skill"],
                                 "n": po["n"]}
        if not ok:
            fails.append("point_skill")
    if po["n"] < REPORT_MIN_N or not checks:
        state = "insufficient_data"
    elif fails:
        state = "outside_band"
    else:
        state = "within_band"
    return {"state": state, "segment": seg_name, "firm": firm,
            "n": po["n"], "checks": checks, "fails": fails,
            "mae_usd": po["mae_usd"], "coverage": iv["coverage"],
            "hit_rate": di["hit_rate"], "ece": pr["ece"]}


def _head_to_head(cells: dict[str, dict]) -> dict[str, Any]:
    """Per horizon: which method leads, where both have
    evidence.  The primary metric order is stated:
    (1) point skill vs the naive, (2) distance of coverage
    from its stated target, (3) ECE — each lower-is-better,
    evidence weight (n) breaking a full tie."""
    out: dict[str, Any] = {}
    horizons = sorted({c["horizon"] for c in cells.values()})
    for h in horizons:
        cands = {}
        for cell in cells.values():
            if cell["horizon"] != h:
                continue
            p = cell["segments"]["pooled"]
            if p["point"]["n"] >= REPORT_MIN_N:
                cands[cell["method"]] = p
        if len(cands) < 2:
            out[str(h)] = {
                "leader": (next(iter(cands)) if cands else None),
                "contested": False,
                "note": ("only method with evidence"
                         if cands else
                         "no method has verdict-grade evidence"),
            }
            continue

        def key(item: tuple[str, dict]) -> tuple:
            _m, p = item
            skill = p["point"]["skill"]
            cov = p["interval"]["coverage"]
            target = p["interval"]["target"] or INTERVAL_COVERAGE
            return (
                skill if skill is not None else math.inf,
                abs(cov - target) if cov is not None else math.inf,
                p["probabilities"]["ece"]
                if p["probabilities"]["ece"] is not None
                else math.inf,
                -p["point"]["n"])

        ranked = sorted(cands.items(), key=key)
        out[str(h)] = {"leader": ranked[0][0], "contested": True,
                       "runner_up": ranked[1][0],
                       "note": ("by point skill, then coverage "
                                "distance, then ECE, over all "
                                "scored rows")}
    return out


def validate(records: list[dict], *,
             malformed: int = 0) -> dict[str, Any]:
    """The full battery over a journal's records."""
    rows, coverage = build_scored_rows(records)
    segments, boundary = _split_segments(rows)
    edges = _vol_edges(segments["eval"])
    cells: dict[str, dict[str, Any]] = {}
    for horizon in sorted({r["horizon"] for r in rows}):
        for method in (METHOD_F1, METHOD_F2):
            mine = [r for r in rows if r["horizon"] == horizon
                    and r["method"] == method]
            if not mine:
                continue
            seg_rows = {
                "eval": [r for r in segments["eval"]
                         if r["horizon"] == horizon
                         and r["method"] == method],
                "holdout": [r for r in segments["holdout"]
                            if r["horizon"] == horizon
                            and r["method"] == method],
                "pooled": mine,
            }
            scored = {name: _score_rows(rs)
                      for name, rs in seg_rows.items()}
            cells[f"{horizon}|{method}"] = {
                "horizon": horizon, "method": method,
                "segments": scored,
                "wf_windows": _wf_windows(seg_rows["eval"]),
                "splits": _splits(mine, edges),
                "verdict": _verdict(scored),
            }
    verdicts = [c["verdict"]["state"] for c in cells.values()]
    overall = ("report_ready"
               if any(v in ("within_band", "outside_band")
                      for v in verdicts)
               else "insufficient_data")
    return {
        "journal_records": len(records),
        "malformed_lines": malformed,
        "coverage": coverage,
        "holdout_boundary": boundary,
        "vol_edges_pct": (list(edges) if edges else None),
        "cells": cells,
        "head_to_head": _head_to_head(cells),
        "overall": overall,
        "bands": {
            "holdout_frac": HOLDOUT_FRAC,
            "report_min_n": REPORT_MIN_N,
            "verdict_min_n": VERDICT_MIN_N,
            "coverage_tol": COVERAGE_TOL,
            "ece_max": ECE_MAX,
            "direction_min_hit": DIRECTION_MIN_HIT,
            "point_skill_max": POINT_SKILL_MAX,
            "large_call_p": LARGE_CALL_P,
        },
        "note": (
            "Measurement only — this battery promotes nothing "
            "and no order, sizing, risk or graduation path "
            "reads it. Probabilities are scored only where "
            "they were stated as calibrated at the time; the "
            "trained shadow series is measured separately and "
            "never enters a verdict. The holdout (final "
            f"{int(HOLDOUT_FRAC * 100)}% of scored bars) was "
            "scored once with the same stated bands; every "
            "band is a constant fixed in code, and volatility "
            "split edges were fixed on the evaluation segment "
            "only."),
    }


# ── rendering (plain language first, S4 style) ──────────────────


def _usd(value: Any) -> str:
    return "—" if value is None else f"${float(value):,.2f}"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.0f}%"


def _num(value: Any, nd: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{nd}f}"


def _verdict_line(cell: dict) -> str:
    v = cell["verdict"]
    tag = tag_for(cell["horizon"])
    seg = v["segment"] + ("" if v["firm"] else ", provisional")
    if v["state"] == "insufficient_data":
        return (f"- {tag} {cell['method']}: — INSUFFICIENT "
                f"DATA (n={v['n']})")
    parts = [f"point MAE {_usd(v['mae_usd'])} (n={v['n']})"]
    if v["coverage"] is not None:
        parts.append(f"coverage {_num(v['coverage'])}")
    if v["hit_rate"] is not None:
        parts.append(f"direction {_pct(v['hit_rate'])}")
    if v["ece"] is not None:
        parts.append(f"ECE {_num(v['ece'])}")
    body = " · ".join(parts)
    if v["state"] == "within_band":
        return (f"- {tag} {cell['method']}: {body} — "
                f"WITHIN BAND ({seg})")
    detail = ", ".join(
        f"{name} {_num(v['checks'][name]['value'])}"
        for name in v["fails"])
    return (f"- {tag} {cell['method']}: {body} — "
            f"OUTSIDE BAND: {detail} ({seg})")


def _cell_detail(cell: dict) -> list[str]:
    tag = tag_for(cell["horizon"])
    lines = [f"### {tag} — {cell['method']}"]
    for name in ("eval", "holdout"):
        s = cell["segments"][name]
        p, iv, d = s["point"], s["interval"], s["direction"]
        pr, sh = s["probabilities"], s["shadow_probabilities"]
        ex, dg = s["excursions"], s["disagreement"]
        lines.append(
            f"- {name}: n={s['n']} · point MAE "
            f"{_usd(p['mae_usd'])} ({_num(p['mae_pct'])}% vs "
            f"naive {_num(p['naive_mae_pct'])}%, skill "
            f"{_num(p['skill'])}) · coverage "
            f"{_num(iv['coverage'])} (n={iv['n']}, width "
            f"{_num(iv['mean_width_pct'])}%) · direction "
            f"{_pct(d['hit_rate'])} on {d['calls']} calls, "
            f"neutral right {_pct(d['neutral_rate'])} "
            f"(n={d['neutral_n']})")
        lines.append(
            f"  probabilities (stated calibrated): n={pr['n']} "
            f"· ECE {_num(pr['ece'])} · Brier "
            f"{_num(pr['brier'])} · shadow series: n={sh['n']} "
            f"· ECE {_num(sh['ece'])} · Brier {_num(sh['brier'])}")
        lm = s["large_move"]
        for side in ("up", "dn"):
            m = lm[side]
            lines.append(
                f"  large-{side}: n={m['n']} · stated "
                f"{_num(m['mean_stated'])} vs realized "
                f"{_num(m['realized_rate'])} · Brier "
                f"{_num(m['brier'])} · calls {m['calls']} at "
                f"p≥{LARGE_CALL_P} → precision "
                f"{_num(m['precision'])}, recall "
                f"{_num(m['recall'])}")
        lines.append(
            f"  excursions: fav MAE {_usd(ex['fav_mae_usd'])} "
            f"· adv MAE {_usd(ex['adv_mae_usd'])} (n={ex['n']}) "
            f"· disagreement reading: "
            f"{dg['reading'] or 'unmeasured'} "
            f"(high/low MAE ratio {_num(dg['mae_ratio'])}, "
            f"n={dg['n']})")
    wf = cell["wf_windows"]
    if wf:
        lines.append(
            "  walk-forward (eval): " + " · ".join(
                f"W{w['window']} MAE {_num(w['mae_pct'])}% "
                f"(n={w['n']})" for w in wf))
    return lines


def _split_lines(cell: dict) -> list[str]:
    if not cell["splits"]:
        return []
    tag = tag_for(cell["horizon"])
    lines = [f"### {tag} — {cell['method']} (pooled segments)"]
    for kind, arms in cell["splits"].items():
        if not arms:
            continue
        parts = []
        for name, a in arms.items():
            parts.append(
                f"{name}: MAE {_num(a['mae_pct'])}% · dir "
                f"{_pct(a['dir_hit_rate'])} "
                f"({a['dir_calls']} calls) · coverage "
                f"{_num(a['coverage'])} (n={a['n']})")
        lines.append(f"- {kind}: " + " | ".join(parts))
    return lines


def render_markdown(result: dict[str, Any]) -> str:
    """Plain language first (the owner's rule), tables after."""
    cov = result["coverage"]
    cells = result["cells"]
    lines = ["# Forecast — F3 acceptance battery", ""]
    if result["overall"] == "insufficient_data":
        lines.append(
            "There is not enough resolved forecast history "
            "yet for any horizon × method cell to carry a "
            "verdict — the journal needs more reconciled "
            "forecasts before any number here would be honest. "
            "What exists is counted below, unscored where it "
            "must be.")
    else:
        within = sum(1 for c in cells.values()
                     if c["verdict"]["state"] == "within_band")
        outside = sum(1 for c in cells.values()
                      if c["verdict"]["state"] == "outside_band")
        lines.append(
            f"Measured over {cov['scored_rows']} scored "
            f"forecast-horizons from {cov['bars_reconciled']} "
            f"reconciled bars: {within} cell(s) within their "
            f"stated bands, {outside} outside. Head-to-head, "
            "by point skill then coverage distance then ECE:")
        for h, hh in sorted(result["head_to_head"].items(),
                            key=lambda kv: int(kv[0])):
            tag = tag_for(int(h))
            if hh["leader"] is None:
                lines.append(f"- {tag}: no method has evidence "
                             "yet.")
            elif hh["contested"]:
                lines.append(
                    f"- {tag}: **{hh['leader']}** leads "
                    f"{hh['runner_up']}.")
            else:
                lines.append(
                    f"- {tag}: only {hh['leader']} has "
                    "evidence.")
    lines += [
        "",
        f"Journal: {result['journal_records']} records "
        f"({result['malformed_lines']} torn lines skipped) · "
        f"{cov['forecast_records']} forecast records, "
        f"{cov['bars_with_outlook']} bars with outlooks, "
        f"{cov['horizons_unresolved']} answered horizons "
        "without a realized outcome (pending or unmeasurable).",
        "",
        "## Verdicts (per horizon × method)",
        "",
    ]
    if cells:
        for key in sorted(cells, key=lambda k: (int(k.split("|")[0]),
                                                k)):
            lines.append(_verdict_line(cells[key]))
    else:
        lines.append("- no scored cells — see the counts above.")
    lines += ["", "## Detail", ""]
    for key in sorted(cells, key=lambda k: (int(k.split("|")[0]),
                                            k)):
        lines += _cell_detail(cells[key])
        lines.append("")
    split_blocks = [b for key in sorted(cells)
                    for b in [_split_lines(cells[key])] if b]
    if split_blocks:
        lines += ["## Regime splits", ""]
        for block in split_blocks:
            lines += block
            lines.append("")
    lines.append(f"*{result['note']}*")
    return "\n".join(lines)
