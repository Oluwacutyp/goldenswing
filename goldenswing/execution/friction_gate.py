"""Entry friction gate (owner ruling 6, 2026-10-23).

A HARD entry refusal at the risk/execution boundary: when the
measured spread exceeds 10% of the proposed order's stop distance
(``spread / stop_distance > 0.10``), the entry is refused.  The
refusal is a fact about transaction cost relative to the risk
being taken — it sits OUTSIDE any judgment or scoring (no score,
floor, regime or bias reading can override or be overridden by
it), and it does NOT kill the setup: the candidate stays alive
for reevaluation when friction normalizes (the runner wires that
per route — the armed plan is never invalidated by this gate,
and the native M5 route holds its dedupe cursor back).

This is a SEPARATE mechanism from:
- the hard-lines cost gate (``cost_gate`` in the confluence
  engine: cost in bps vs a minimum net TP1 R — a setup-quality
  check inside gate_live_signal / hard_line_check), and
- RiskPool's friction RESERVE (a sizing reserve inside the risk
  pool — capital allocation, not an entry refusal).
Different questions, different names, different reason codes:
this gate's code is ``risk.friction`` (funnel taxonomy category
8 — hard risk veto).

Units: the spread may arrive as a PRICE distance (the live tick
contract: ``tick["spread"] == ask - bid``; see feed/mt5.py) or in
POINTS (MT5 bar convention: the bar ``spread`` column is points,
feed/mt5.py again).  Points are normalized through the
instrument's ``point`` size (SymbolSpec.point — XAUUSDm: 0.01)
before any comparison: comparing points against a price-distance
stop without normalization is the classic 100x unit error this
gate exists to never make.

Fail closed: missing, non-finite or non-positive stop distance,
or a spread that cannot be measured from the supplied inputs,
refuses the entry (no crash, no order) — an unmeasurable cost is
not a free pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# The owner's bound: refusal is STRICTLY above 0.10 — a ratio of
# exactly 0.10 passes.
FRICTION_RATIO_LIMIT = 0.10

# Dedicated reason code (funnel taxonomy category 8).
FRICTION_REASON_CODE = "risk.friction"


@dataclass(frozen=True)
class FrictionAssessment:
    """One gate reading.  ``refused`` is the verdict; the measured
    fields are what gets journaled — a refusal without its
    measurements is not auditable."""

    refused: bool
    spread_price: float | None = None
    stop_distance: float | None = None
    ratio: float | None = None
    detail: str = ""

    @property
    def reason_code(self) -> str | None:
        return FRICTION_REASON_CODE if self.refused else None


def _finite(value) -> float | None:
    """float(value) when it is a real, finite number, else None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def assess_entry_friction(*, entry, stop, spread_price=None,
                          spread_points=None, point=None
                          ) -> FrictionAssessment:
    """Assess one proposed entry against the friction bound.

    entry / stop: the proposed order's entry level and defended
    stop (price units).  spread_price: spread as a price distance
    (preferred — the live tick contract).  spread_points + point:
    the points-denominated alternative (bar convention), used only
    when no price spread is supplied; ``point`` is the
    instrument's point size (e.g. 0.01 for XAUUSDm).

    Never raises: any input defect is a refusal with the defect
    named in ``detail`` (fail closed).
    """
    entry_f = _finite(entry)
    stop_f = _finite(stop)
    if entry_f is None or stop_f is None:
        return FrictionAssessment(
            refused=True,
            detail="friction gate: entry/stop not measurable "
                   f"(entry={entry!r}, stop={stop!r}) — fail "
                   "closed, no order")
    stop_distance = abs(entry_f - stop_f)
    if stop_distance <= 0:
        return FrictionAssessment(
            refused=True, stop_distance=stop_distance,
            detail="friction gate: stop distance is zero "
                   f"(entry {entry_f}, stop {stop_f}) — no risk "
                   "distance to measure friction against, no order")
    spread = _finite(spread_price)
    spread_src = "price"
    if spread is None:
        pts = _finite(spread_points)
        pt = _finite(point)
        if pts is not None and pt is not None and pt > 0:
            spread = pts * pt
            spread_src = f"points {pts:g} x point {pt:g}"
        else:
            return FrictionAssessment(
                refused=True, stop_distance=stop_distance,
                detail="friction gate: spread not measurable "
                       f"(spread_price={spread_price!r}, "
                       f"spread_points={spread_points!r}, "
                       f"point={point!r}) — fail closed, no order")
    if spread < 0:
        return FrictionAssessment(
            refused=True, stop_distance=stop_distance,
            spread_price=spread,
            detail=f"friction gate: negative spread ({spread}) "
                   "— crossed/invalid quote, no order")
    ratio = spread / stop_distance
    if ratio > FRICTION_RATIO_LIMIT:
        return FrictionAssessment(
            refused=True, spread_price=spread,
            stop_distance=stop_distance, ratio=ratio,
            detail=f"friction gate: spread {spread:.2f} "
                   f"({spread_src}) / stop distance "
                   f"{stop_distance:.2f} = {ratio:.3f} > "
                   f"{FRICTION_RATIO_LIMIT:.2f} — entry refused, "
                   "setup stays alive for reevaluation")
    return FrictionAssessment(
        refused=False, spread_price=spread,
        stop_distance=stop_distance, ratio=ratio,
        detail=f"friction ok: spread {spread:.2f} ({spread_src}) "
               f"/ stop {stop_distance:.2f} = {ratio:.3f} <= "
               f"{FRICTION_RATIO_LIMIT:.2f}")
