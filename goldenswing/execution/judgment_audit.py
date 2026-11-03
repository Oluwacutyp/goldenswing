"""Judgment Core S3 — outcome reconciliation and self-audit.

S1 judges candidates in shadow; S2 maintains a market thesis in
shadow.  Neither used to answer for itself: a judgment was
journaled and never compared with what the market actually did,
so calibration was decorative and a whole class of failure —
sitting out a tradeable move — was invisible (2026-11-03 London:
gold fell through the session while eighteen assessments produced
one WATCH and no candidate; the judgment audit must catch that
shape itself).

This module is the pure core: forward-outcome measurement,
calibration bucketing, and missed-opportunity detection.  The
runner owns the state and the journal; everything here is a
function of its inputs.  Shadow posture throughout: the audit
measures the judgment core, it never steers it.
"""
from __future__ import annotations

#: Forward horizons (hours) at which a shadow judgment is reconciled.
HORIZONS_H: tuple[int, ...] = (1, 4)

#: A closed H1 bar whose body is at least this many ATRs is a
#: material move — the kind a trading desk exists to catch.
MISSED_MOVE_ATR = 1.2

#: Confidence bands for calibration.  A bucket is honest only when
#: it carries its count beside its hit rate.
CONFIDENCE_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))


def forward_outcome(direction: int, entry_price: float,
                    exit_price: float, atr: float) -> dict:
    """What actually happened after a judgment, in price and ATR.

    ``hit`` is directional only (price moved the judged way); the
    magnitudes ride along so a hit by a whisker and a hit by a
    session move are never averaged into the same story."""
    move = (float(exit_price) - float(entry_price)) * int(direction)
    return {
        "move": round(move, 5),
        "move_atr": (round(move / atr, 4)
                     if atr and atr > 0 else None),
        "hit": move > 0,
    }


def band_for(confidence: float) -> str:
    for lo, hi in CONFIDENCE_BANDS:
        if lo <= confidence < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return "1.00+"


def calibration_add(buckets: dict, *, confidence: float,
                    expected_r: float | None,
                    realized_r: float) -> dict:
    """Fold one reconciled judgment into its confidence bucket.

    realized_r is the forward move expressed in R of the judged
    trade's own stop distance (the runner computes it; a judgment
    without geometry reconciles direction-only and is counted in a
    separate tally by the caller, never silently dropped)."""
    band = band_for(float(confidence))
    slot = buckets.setdefault(band, {
        "n": 0, "hits": 0, "sum_expected_r": 0.0,
        "sum_realized_r": 0.0, "expected_n": 0})
    slot["n"] += 1
    if realized_r > 0:
        slot["hits"] += 1
    slot["sum_realized_r"] = round(
        slot["sum_realized_r"] + float(realized_r), 4)
    if expected_r is not None:
        slot["expected_n"] += 1
        slot["sum_expected_r"] = round(
            slot["sum_expected_r"] + float(expected_r), 4)
    return buckets


def calibration_view(buckets: dict) -> dict:
    """The readable form: per band, count, hit rate, and the gap
    between what the core expected and what it got."""
    out = {}
    for band, slot in sorted(buckets.items()):
        n = int(slot.get("n") or 0)
        if not n:
            continue
        en = int(slot.get("expected_n") or 0)
        out[band] = {
            "n": n,
            "hit_rate": round(slot["hits"] / n, 4),
            "mean_expected_r": (round(slot["sum_expected_r"] / en, 4)
                                if en else None),
            "mean_realized_r": round(slot["sum_realized_r"] / n, 4),
        }
    return out


def missed_opportunity(*, bar_open: float, bar_close: float,
                       atr: float, evaluated: bool) -> dict | None:
    """A material H1 move the desk never even evaluated.

    Returns the audit record body, or None when the bar was quiet
    or the desk did evaluate something during it (an abstention
    after evaluation is a decision; no evaluation at all is the
    failure this detector exists for)."""
    if evaluated or not atr or atr <= 0:
        return None
    move = float(bar_close) - float(bar_open)
    if abs(move) < MISSED_MOVE_ATR * atr:
        return None
    return {
        "direction": 1 if move > 0 else -1,
        "move": round(move, 5),
        "move_atr": round(abs(move) / atr, 4),
        "reason_code": "audit.no_evaluation_on_material_bar",
    }
