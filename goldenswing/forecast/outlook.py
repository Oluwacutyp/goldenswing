"""Forecast F1 — the outlook: numbers, not just ranks.

E1–E3 journaled the ensemble's per-horizon PERCENTILE RANKS, the
large-move magnitude read, a live rank-band calibration book, and
a judgment evidence feed.  What none of them ever produced was
the forecast itself — the owner's correction (2026-11-03):

    "what sort of forecast doesn't have an exact number or
    direction?"

F1 attaches an OUTLOOK BLOCK to every journaled forecast record.
Per horizon it carries: direction (expected-return sign against a
stated deadband), a point price, a prediction interval with its
nominal coverage stated, the expected move ($ and %), expected
favourable/adverse excursions, ensemble disagreement, and — only
where the resolved live history has earned them — calibrated
probabilities.  At record level: created ts, data cutoff, expiry
per horizon, model/ensemble version, and the estimation method,
named on the record with its sample sizes.

**Estimation (v1 method, ``f1-empirical-v1``).**  Empirical
conditional distributions, no trained regressor (that is F2):

- The conditioning state is the ensemble's own: the DOMINANT
  direction (higher of the up/dn ranks) and its E2 rank band
  (``calibration.RANK_BANDS`` — the same bands the calibration
  book uses).  The conditional sample is the resolved live
  forecasts in the state ring whose dominant direction and band
  match the current read.
- When that band sample is thin (< ``MIN_BAND_N``), the horizon
  falls back to its UNCONDITIONAL forward distribution over the
  H1 frame at scoring time (every bar's realized forward return /
  excursions) — method named ``unconditional_h1`` with its n.
- When even that is thin, the horizon's fields are absent and the
  block says why.  No fabricated precision anywhere.

Point = the sample MEAN of forward returns (stated estimator).
Interval = the sample's central quantiles at
``INTERVAL_COVERAGE``.  Excursions are quoted relative to the
outlook direction (the dominant side when direction is NEUTRAL).

**Probabilities are earned, separately.**  Direction
probabilities (up/dn/neutral by endpoint sign against the same
deadband) and large-move probabilities per direction (the house
barrier-race labels, kept SEPARATE from direction — a 46% chance
of finishing up is not an 18% chance of a large up move) come
from the band-conditional resolved sample only, and only when
that sample clears ``MIN_PROB_N`` (20 — the E3 trust precedent,
``forecast_evidence.TRUST_MIN_N``).  Below it the block reads
``uncalibrated`` with the count that failed.  The percentile
ranks still exist — demoted to one auxiliary line on the card.

F2 replaces the estimator behind this same block shape; F3
scores point error, interval coverage and excursion quality from
the journal (the resolution side — realized ret/excursions per
horizon — lands in ``calibration.realized_outcomes``).

Pure core: everything here is a function of its inputs.  No LLM,
no order path reads this block — forecasts stay research/shadow.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .calibration import band_for_rank
from .labels import spec_for

#: Method name + version stamped on every outlook block.
OUTLOOK_METHOD = "f1-empirical-v1"
#: F2 method stamp (forecast/trained.py): the per-horizon method
#: of a horizon answered by a GRADUATED trained estimator, and
#: the block-level method when at least one horizon is.  The
#: block stamp never claims more than the horizons under it —
#: each horizon names its own source.
TRAINED_METHOD = "f2-trained-v1"
#: Nominal coverage of the prediction interval (central quantiles).
INTERVAL_COVERAGE = 0.80
#: Floors for the empirical samples.  MIN_PROB_N follows the E3
#: trust precedent (forecast_evidence.TRUST_MIN_N = 20 resolved
#: samples before a live frequency may be quoted as a probability).
MIN_BAND_N = 30
MIN_PROB_N = 20
MIN_UNCOND_N = 50
#: Direction deadband: |expected return| within this fraction of
#: the estimation sample's return stdev is NEUTRAL, not a call.
DEADBAND_SIGMA = 0.15

#: Card/status horizon tags — the alert vocabulary (mt5's
#: _horizon_label: 168 bars is "7d", 360 is "15d").
HORIZON_TAGS: dict[int, str] = {12: "12h", 24: "1d", 72: "3d",
                                168: "7d", 360: "15d"}


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


def tag_for(horizon: int) -> str:
    return HORIZON_TAGS.get(int(horizon), f"{int(horizon)}b")


# ── realized forward outcomes over a frame ─────────────────────


def forward_outcomes(h1: pd.DataFrame, horizon: int) -> dict:
    """Realized forward outcomes for a forecast made at each bar.

    For bar ``t`` (full window required): ``ret_pct`` = endpoint
    return % over bars t+1..t+horizon; ``run_up`` = max high in
    the window − close[t] (the most an up move ran); ``run_dn`` =
    close[t] − min low in the window (the mirror).  Returns
    aligned float arrays over the bars that have a full, finite
    window — the horizon's unconditional empirical distribution.
    """
    empty = {"ret_pct": np.zeros(0), "run_up": np.zeros(0),
             "run_dn": np.zeros(0)}
    if h1 is None or horizon <= 0 or len(h1) <= horizon:
        return empty
    close = h1["close"].to_numpy(dtype=float)
    high = h1["high"].to_numpy(dtype=float)
    low = h1["low"].to_numpy(dtype=float)
    rmax = pd.Series(high).rolling(horizon).max().to_numpy()
    rmin = pd.Series(low).rolling(horizon).min().to_numpy()
    base = close[:-horizon]
    ret = (close[horizon:] / base - 1.0) * 100.0
    run_up = rmax[horizon:] - base
    run_dn = base - rmin[horizon:]
    ok = np.isfinite(ret) & np.isfinite(run_up) & np.isfinite(run_dn)
    return {"ret_pct": ret[ok], "run_up": run_up[ok],
            "run_dn": run_dn[ok]}


# ── conditioning ────────────────────────────────────────────────


def _dominant(up: float | None, dn: float | None
              ) -> tuple[str | None, float | None]:
    """The ensemble's dominant side at one horizon: the higher
    rank and its direction.  (None, None) when either side is
    unscored — a one-sided read has no state to condition on."""
    if up is None or dn is None:
        return None, None
    return ("up", up) if up >= dn else ("dn", dn)


def _band_pool(samples: list, horizon: int, dom_dir: str,
               dom_band: str) -> list[dict]:
    """Resolved live forecasts whose dominant direction + rank
    band match the current read at this horizon — the conditional
    sample.  A sample's own dominant side is recomputed from its
    journaled ranks, so the conditioning is exactly the state's."""
    pool: list[dict] = []
    for s in samples or []:
        try:
            if int(s.get("horizon")) != int(horizon):
                continue
        except (TypeError, ValueError):
            continue
        d, r = _dominant(_f(s.get("up")), _f(s.get("dn")))
        if d == dom_dir and r is not None \
                and band_for_rank(r) == dom_band:
            pool.append(s)
    return pool


def _pool_arrays(pool: list[dict]) -> dict:
    def col(name: str) -> np.ndarray:
        vals = [_f(s.get(name)) for s in pool]
        return np.array([v for v in vals if v is not None],
                        dtype=float)
    return {"ret_pct": col("ret_pct"), "run_up": col("run_up"),
            "run_dn": col("run_dn")}


def _disagreement(model_probs: Any, horizon: int) -> float | None:
    """Spread across the ensemble's per-model outputs at one
    horizon: the stdev of per-model leans (p_up − p_dn).  None
    when per-model outputs were not carried (aggregate only) —
    reported absent, never synthesized."""
    if not isinstance(model_probs, dict):
        return None
    entry = model_probs.get(horizon) \
        or model_probs.get(str(horizon))
    if not isinstance(entry, dict):
        return None
    ups, dns = entry.get("up"), entry.get("dn")
    if not ups or not dns or len(ups) != len(dns) or len(ups) < 2:
        return None
    leans = [float(u) - float(d) for u, d in zip(ups, dns)]
    if not all(math.isfinite(x) for x in leans):
        return None
    return round(float(np.std(leans)), 4)


# ── the outlook block ───────────────────────────────────────────


def _apply_trained(out: dict, entry: dict | None) -> None:
    """F2: fold one horizon's trained estimate into its F1 block.

    GRADUATED (the live probability book has earned it — see
    forecast/trained.py): the trained fields replace the
    estimation keys WHOLE.  Nothing is field-blended: direction,
    point, interval, excursions, probabilities and method all
    come from the one source, and the block says which.  The
    ensemble disagreement (a fact about the ranker, not about
    either estimator) is preserved.

    NOT graduated: F1's fields stand untouched and the trained
    probabilities ride as ``trained_shadow`` — journaled, resolved
    into the probability book, rendered nowhere — until the book
    graduates the horizon."""
    if not isinstance(entry, dict):
        return
    meta = entry.get("meta") or {}
    out["trained"] = {
        "version": meta.get("version"),
        "trained_at": meta.get("trained_at"),
        "trained_through": meta.get("trained_through"),
        "graduated": bool(entry.get("graduated")),
    }
    fields = entry.get("fields")
    if entry.get("graduated") and isinstance(fields, dict):
        disagreement = out.get("disagreement")
        for key, value in fields.items():
            out[key] = value
        out["disagreement"] = disagreement
    elif isinstance(entry.get("shadow"), dict):
        out["trained_shadow"] = entry["shadow"]


def horizon_outlook(*, horizon: int, up_rank: float | None,
                    dn_rank: float | None, close: float | None,
                    atr: float | None, samples: list,
                    h1: pd.DataFrame | None,
                    disagreement: float | None,
                    trained_entry: dict | None = None) -> dict:
    """One horizon's outlook: F1 empirical estimation, with the
    F2 trained estimate folded in per _apply_trained when the
    runner supplies one."""
    out = _horizon_outlook_f1(
        horizon=horizon, up_rank=up_rank, dn_rank=dn_rank,
        close=close, atr=atr, samples=samples, h1=h1,
        disagreement=disagreement)
    _apply_trained(out, trained_entry)
    return out


def _horizon_outlook_f1(*, horizon: int, up_rank: float | None,
                        dn_rank: float | None, close: float | None,
                        atr: float | None, samples: list,
                        h1: pd.DataFrame | None,
                        disagreement: float | None) -> dict:
    """F1 estimation proper.  Every estimated field is None with
    ``unavailable_reason`` set when the data cannot carry it."""
    out: dict[str, Any] = {
        "direction": None, "point_price": None, "interval": None,
        "expected_move_usd": None, "expected_move_pct": None,
        "fav_excursion_usd": None, "adv_excursion_usd": None,
        "disagreement": disagreement,
        "probabilities": {"state": "uncalibrated",
                          "reason": "no resolved forecasts yet"},
        "method": "unavailable", "n": 0, "deadband_pct": None,
        "barriers": None, "scenario": None, "invalidation": None,
        "unavailable_reason": None,
    }
    if close is None:
        out["unavailable_reason"] = "no scored close"
        return out
    dom_dir, dom_rank = _dominant(up_rank, dn_rank)
    if dom_dir is None:
        # Unscored stays unscored (the E2 rule): never dressed as
        # an unconditional outlook the ensemble never made.
        out["unavailable_reason"] = \
            "horizon unscored — no ensemble ranks"
        out["probabilities"] = {
            "state": "uncalibrated", "reason": "horizon unscored"}
        return out
    dom_band = band_for_rank(dom_rank) if dom_rank is not None \
        else None
    cond_pool = _band_pool(samples, horizon, dom_dir, dom_band) \
        if dom_dir and dom_band else []

    # ── estimation sample: band-conditional, else unconditional ──
    est: dict | None = None
    method = "unavailable"
    if len(cond_pool) >= MIN_BAND_N:
        est, method = _pool_arrays(cond_pool), "band_conditional"
    else:
        uncond = forward_outcomes(h1, horizon) \
            if h1 is not None else {"ret_pct": np.zeros(0)}
        if len(uncond["ret_pct"]) >= MIN_UNCOND_N:
            est, method = uncond, "unconditional_h1"
    if est is None or len(est["ret_pct"]) == 0:
        out["unavailable_reason"] = (
            f"thin history: band sample {len(cond_pool)} "
            f"(< {MIN_BAND_N}) and unconditional sample below "
            f"{MIN_UNCOND_N}")
        out["probabilities"] = _probabilities(
            cond_pool, deadband=None)
        return out

    ret = est["ret_pct"]
    sd = float(np.std(ret))
    deadband = DEADBAND_SIGMA * sd
    mean_ret = float(np.mean(ret))
    if mean_ret > deadband:
        direction = "UP"
    elif mean_ret < -deadband:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"
    eff_dir = direction.lower() if direction in ("UP", "DOWN") \
        else (dom_dir or "up")
    lo_q = (1.0 - INTERVAL_COVERAGE) / 2.0
    q_lo, q_hi = np.quantile(ret, [lo_q, 1.0 - lo_q])
    fav_src = est["run_up"] if eff_dir == "up" else est["run_dn"]
    adv_src = est["run_dn"] if eff_dir == "up" else est["run_up"]
    point = close * (1.0 + mean_ret / 100.0)
    out.update({
        "direction": direction,
        "point_price": _r(point, 1),
        "interval": {"low": _r(close * (1.0 + float(q_lo) / 100.0), 1),
                     "high": _r(close * (1.0 + float(q_hi) / 100.0), 1),
                     "coverage": INTERVAL_COVERAGE},
        "expected_move_usd": _r(point - close, 1),
        "expected_move_pct": _r(mean_ret, 2),
        "fav_excursion_usd": _r(float(np.mean(fav_src)), 1),
        "adv_excursion_usd": _r(float(np.mean(adv_src)), 1),
        "method": method, "n": int(len(ret)),
        "deadband_pct": _r(deadband, 3),
        "unavailable_reason": None,
    })
    out["probabilities"] = _probabilities(cond_pool, deadband)

    # ── scenario + invalidation: the label machinery's barriers ──
    if atr is not None and atr > 0:
        try:
            spec = spec_for(horizon)
        except KeyError:
            spec = None
        if spec is not None:
            d = 1.0 if eff_dir == "up" else -1.0
            fav_price = close + d * spec.fav_mult * atr
            adv_price = close - d * spec.adv_mult * atr
            out["barriers"] = {
                "fav": _r(fav_price, 1), "adv": _r(adv_price, 1),
                "side": eff_dir,
                "fav_mult": spec.fav_mult, "adv_mult": spec.adv_mult}
            out["invalidation"] = _r(adv_price, 1)
            out["scenario"] = (
                f"{spec.tag} barrier race: favourable "
                f"{fav_price:,.1f} ({spec.fav_mult:g}x ATR) reached "
                f"before adverse {adv_price:,.1f} "
                f"({spec.adv_mult:g}x ATR) within {horizon} H1 bars")
    return out


def _probabilities(cond_pool: list[dict],
                   deadband: float | None) -> dict:
    """Calibrated probabilities from the band-conditional resolved
    sample — or the honest uncalibrated state.  Direction
    (endpoint sign vs the deadband) and large-move (barrier-race
    labels) are computed and reported SEPARATELY."""
    if len(cond_pool) < MIN_PROB_N:
        return {"state": "uncalibrated",
                "reason": (f"band sample {len(cond_pool)} "
                           f"< {MIN_PROB_N} resolved forecasts")}
    ret = np.array([v for v in
                    (_f(s.get("ret_pct")) for s in cond_pool)
                    if v is not None], dtype=float)
    if len(ret) < MIN_PROB_N:
        return {"state": "uncalibrated",
                "reason": "band sample lacks realized returns"}
    db = deadband if deadband is not None else \
        DEADBAND_SIGMA * float(np.std(ret))
    up = float(np.mean(ret > db))
    dn = float(np.mean(ret < -db))
    neutral = float(np.mean(np.abs(ret) <= db))
    y_up = [v for v in (_f(s.get("y_up")) for s in cond_pool)
            if v is not None]
    y_dn = [v for v in (_f(s.get("y_dn")) for s in cond_pool)
            if v is not None]
    return {
        "state": "calibrated", "n": int(len(ret)),
        "up": _r(up, 4), "dn": _r(dn, 4), "neutral": _r(neutral, 4),
        "large_up": (_r(float(np.mean(y_up)), 4) if len(y_up) >= MIN_PROB_N else None),
        "large_dn": (_r(float(np.mean(y_dn)), 4) if len(y_dn) >= MIN_PROB_N else None),
        "basis": (f"empirical frequencies of {len(ret)} resolved "
                  f"forecasts in the current rank band"),
    }


def build_outlook(*, ranks: dict, close: float | None,
                  atr: float | None, samples: list,
                  h1: pd.DataFrame | None,
                  model_probs: Any = None,
                  created: str | None = None,
                  data_cutoff: str | None = None,
                  model_version: Any = None,
                  feature_set: Any = None,
                  trained: dict | None = None) -> dict:
    """The whole outlook block for one journaled forecast record.

    ``ranks`` is the record's journaled shape ``{str(h): {"up",
    "dn"}}``; ``samples`` is the calibration state's resolved-
    sample ring; ``model_probs`` the forecaster's per-model raw
    probabilities for the scored row (disagreement only).
    ``trained`` (F2, optional) is the trained estimator's
    per-horizon output ``{"version", "trained_at",
    "trained_through", "horizons": {str(h): entry}}`` — see
    _apply_trained for how an entry folds in.  Without it the
    block is exactly the F1 block."""
    horizons: dict[str, Any] = {}
    expiry: dict[str, Any] = {}
    cutoff = None
    if data_cutoff:
        try:
            cutoff = pd.Timestamp(data_cutoff)
        except Exception:  # noqa: BLE001 — expiry just omits
            cutoff = None
    trained_horizons = (trained or {}).get("horizons") or {}
    for key in sorted((ranks or {}).keys(), key=lambda k: int(k)):
        h = int(key)
        pair = ranks.get(key) or {}
        entry = trained_horizons.get(str(h)) \
            or trained_horizons.get(h)
        horizons[str(h)] = horizon_outlook(
            horizon=h, up_rank=_f(pair.get("up")),
            dn_rank=_f(pair.get("dn")), close=close, atr=atr,
            samples=samples, h1=h1,
            disagreement=_disagreement(model_probs, h),
            trained_entry=entry)
        expiry[str(h)] = (str(cutoff + pd.Timedelta(hours=h))
                          if cutoff is not None else None)
    graduated = [key for key, ho in horizons.items()
                 if ho.get("method") == TRAINED_METHOD]
    block: dict[str, Any] = {
        "method": TRAINED_METHOD if graduated else OUTLOOK_METHOD,
        "created": created,
        "data_cutoff": data_cutoff,
        "model_version": model_version,
        "feature_set": feature_set,
        "interval_coverage": INTERVAL_COVERAGE,
        "point_estimator": (
            "trained median (0.5 quantile) on graduated horizons; "
            "mean of the estimation sample elsewhere"
            if graduated else "mean of the estimation sample"),
        "expiry": expiry,
        "horizons": horizons,
    }
    if trained:
        block["trained"] = {
            "version": trained.get("version"),
            "trained_at": trained.get("trained_at"),
            "trained_through": trained.get("trained_through"),
            "graduated_horizons": graduated,
        }
    return block


# ── rendering (deterministic text; no LLM anywhere) ─────────────


def horizon_line(tag: str, ho: dict) -> str:
    """One horizon's outlook in plain trader language, numbers
    first.  An unavailable horizon says so and why."""
    if not ho or ho.get("direction") is None:
        return (f"{tag} outlook unavailable — "
                f"{ho.get('unavailable_reason') if ho else 'no data'}")
    iv = ho.get("interval") or {}
    line = (
        f"{tag} {ho['direction']} → {ho['point_price']:,.1f} · "
        f"range {iv.get('low'):,.1f}–{iv.get('high'):,.1f} "
        f"({int(round(float(iv.get('coverage') or 0) * 100))}%) · "
        f"move {ho['expected_move_usd']:+,.1f} "
        f"({ho['expected_move_pct']:+.2f}%) · "
        f"fav +{ho['fav_excursion_usd']:,.0f} / "
        f"adv −{ho['adv_excursion_usd']:,.0f}")
    probs = ho.get("probabilities") or {}
    if probs.get("state") == "calibrated":
        line += (f" · P up {probs['up']:.0%} · dn {probs['dn']:.0%} "
                 f"· flat {probs['neutral']:.0%}")
        if probs.get("large_up") is not None:
            line += (f" · large-up {probs['large_up']:.0%} / "
                     f"large-dn {probs['large_dn']:.0%}")
        line += f" (n={probs['n']})"
    return line


def contradiction_line(horizons: dict) -> str | None:
    """Detection + naming only (narration is F2): when horizons
    materially disagree — at least one UP and one DOWN call beyond
    their deadbands — name the two most opposed clocks."""
    ups: list[tuple[str, float]] = []
    dns: list[tuple[str, float]] = []
    for key, ho in (horizons or {}).items():
        if not ho or ho.get("expected_move_pct") is None:
            continue
        tag = tag_for(int(key))
        if ho.get("direction") == "UP":
            ups.append((tag, float(ho["expected_move_pct"])))
        elif ho.get("direction") == "DOWN":
            dns.append((tag, float(ho["expected_move_pct"])))
    if not ups or not dns:
        return None
    up_tag = max(ups, key=lambda p: p[1])[0]
    dn_tag = min(dns, key=lambda p: p[1])[0]
    return (f"⚠ horizons split: {up_tag} UP vs {dn_tag} DOWN — "
            f"the clocks disagree across horizons")


def contradiction_narration(horizons: dict) -> str | None:
    """F2: the split, explained — deterministically, from the
    computed fields only (no LLM, no invented cause).  Which
    horizons oppose and by how much, whether the split is a
    near-vs-far clock split, where ensemble disagreement peaks,
    and which side carries the heavier expected move.  None when
    the horizons do not materially disagree (the same test as
    contradiction_line)."""
    rows: list[tuple[int, str, str, float, float | None]] = []
    for key, ho in (horizons or {}).items():
        if not ho or ho.get("expected_move_pct") is None:
            continue
        if ho.get("direction") not in ("UP", "DOWN"):
            continue
        rows.append((int(key), tag_for(int(key)),
                     str(ho["direction"]),
                     float(ho["expected_move_pct"]),
                     _f(ho.get("disagreement"))))
    ups = [r for r in rows if r[2] == "UP"]
    dns = [r for r in rows if r[2] == "DOWN"]
    if not ups or not dns:
        return None
    up_top = max(ups, key=lambda r: r[3])
    dn_top = min(dns, key=lambda r: r[3])
    parts = [
        f"the split in numbers: {up_top[1]} expects "
        f"{up_top[3]:+.2f}% while {dn_top[1]} expects "
        f"{dn_top[3]:+.2f}%"]
    near = [r for r in rows if r[0] <= 24]
    far = [r for r in rows if r[0] >= 168]
    if near and far:
        near_mean = sum(r[3] for r in near) / len(near)
        far_mean = sum(r[3] for r in far) / len(far)
        if (near_mean > 0) != (far_mean > 0):
            parts.append(
                f"it is a clock split — near horizons "
                f"({'/'.join(r[1] for r in near)}) average "
                f"{near_mean:+.2f}% against far horizons "
                f"({'/'.join(r[1] for r in far)}) at "
                f"{far_mean:+.2f}%, so the short clocks and the "
                f"long clocks are reading different moves, not "
                f"the same move at different sizes")
    heavier = up_top if abs(up_top[3]) >= abs(dn_top[3]) else dn_top
    parts.append(
        f"the heavier read is {heavier[1]} {heavier[2]} at "
        f"{heavier[3]:+.2f}% expected")
    spread = [(r[1], r[4]) for r in rows if r[4] is not None]
    if spread:
        tag, val = max(spread, key=lambda p: p[1])
        parts.append(
            f"ensemble disagreement peaks at {tag} ({val:.2f}) — "
            f"the models argue most on that clock")
    return "split read: " + "; ".join(parts) + "."


def _ranks_aux_line(ranks: dict) -> str:
    parts = []
    for key in sorted((ranks or {}).keys(), key=lambda k: int(k)):
        pair = ranks.get(key) or {}
        up, dn = _f(pair.get("up")), _f(pair.get("dn"))
        if up is None or dn is None:
            continue
        parts.append(f"{tag_for(int(key))} up {up:.0f} / dn {dn:.0f}")
    return ("ranks (auxiliary): " + " · ".join(parts)
            + " — percentile ranks, not probabilities")


def render_card(*, close: float | None, bar_time: Any,
                outlook: dict, ranks: dict,
                headline: tuple | None = None) -> str:
    """The 🔮 forecast card: numbers lead, ranks demoted to one
    auxiliary line, uncalibrated stated once where it applies."""
    lines = ["🔮 FORECAST — not a trade signal, not an order"]
    head = f"XAUUSD {close:,.1f} @ {bar_time}" if close is not None \
        else f"XAUUSD @ {bar_time}"
    if headline is not None:
        head += (f" — new crossing: big-move rank {headline[2]:.0f} "
                 f"{str(headline[0]).upper()} "
                 f"@{tag_for(int(headline[1]))}")
    lines.append(head)
    horizons = (outlook or {}).get("horizons") or {}
    any_uncalibrated = False
    for key in sorted(horizons.keys(), key=lambda k: int(k)):
        ho = horizons[key]
        lines.append(horizon_line(tag_for(int(key)), ho))
        probs = ho.get("probabilities") or {}
        if probs.get("state") != "calibrated":
            any_uncalibrated = True
    split = contradiction_line(horizons)
    if split:
        lines.append(split)
        story = contradiction_narration(horizons)
        if story:
            lines.append(story)
    lines.append(_ranks_aux_line(ranks))
    if any_uncalibrated:
        lines.append(
            "probabilities uncalibrated where no P line shows — "
            "the resolved history in that rank band is still thin")
    method = (outlook or {}).get("method") or OUTLOOK_METHOD
    if method == OUTLOOK_METHOD:
        lines.append(
            f"outlook: {OUTLOOK_METHOD} — empirical distributions, "
            f"intervals at {INTERVAL_COVERAGE:.0%} nominal coverage")
    else:
        lines.append(
            f"outlook: {method} — trained estimators on graduated "
            f"horizons (each horizon names its method), empirical "
            f"elsewhere, intervals at {INTERVAL_COVERAGE:.0%} "
            f"nominal coverage")
    return "\n".join(lines)


def render_status_lines(outlook: dict) -> list[str]:
    """The /status forecast section's outlook block: the same
    numbers, one line per horizon, indented to the section."""
    if not outlook:
        return []
    lines = ["  outlook (F1 — numbers, not ranks):"]
    horizons = outlook.get("horizons") or {}
    for key in sorted(horizons.keys(), key=lambda k: int(k)):
        lines.append("    " + horizon_line(tag_for(int(key)),
                                           horizons[key]))
    split = contradiction_line(horizons)
    if split:
        lines.append("    " + split)
        story = contradiction_narration(horizons)
        if story:
            lines.append("    " + story)
    return lines
