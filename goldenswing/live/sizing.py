"""Risk-based sizing + the RiskPool ledger (spec §4, amendment A13).

Owner-designed (session 2026-10-21): sizes are computed from risk,
never hardcoded lots.  Owner ruling 2026-10-23: dynamic risk
sizing is the SOLE sizing path on the MT5 engine — the fixed-lots
config mode that A13 kept as an override is revoked and removed;
this module's ``size_for_risk`` is the production sizing core.

Hard rules implemented here:
- **Decimal everywhere.**  Every Decimal is constructed via
  ``Decimal(str(x))`` — ``Decimal(0.1)`` imports the float's binary
  error exactly and would silently corrupt step arithmetic.
- **Step quantization by integer step-count**: steps = (raw / step)
  floored ROUND_FLOOR; lots = steps * step.  Correct for steps of
  0.05 / 0.25, not only 0.01 (quantize-to-exponent is not).
- Clamp to VOLUME_MAX; respect VOLUME_LIMIT against already-open
  volume on the symbol.
- Floored result below VOLUME_MIN = **DECLINE** (a verdict with its
  numbers, journaled by the caller) — never round up: rounding up
  spends risk nobody authorized.
- Margin is checked by the caller before submit (it needs the live
  account); this module reports the requirement inputs.

RiskPool — the per-position risk ledger:
- Budget in Decimal risk-money (the position's authorized risk).
- Live consumption = sum over open layers of the layer's ACCOUNTED
  risk.  A layer's raw risk = lots * (entry - stop distance in
  ticks) * tick_value, floored at 0 per layer: a layer whose stop
  has trailed into profit contributes 0 — locked profit never
  mints budget for other layers.
- Stop -> breakeven refunds the layer's consumption MINUS a
  friction reserve: accounted = new_risk + FRICTION_FRAC *
  (accounted - new_risk).  FRICTION_FRAC = 5%: a gold stop-out
  through a moved stop pays the spread (~$0.20-0.35 on Exness
  Standard) plus gap slippage; against typical stop distances of
  $5-25 that is 1-5% of the amount at risk, so 5% is the
  conservative top of the band — the pool never overstates
  headroom by more than a twentieth of one layer's risk.
- Consumption is computed from REALIZED lots (post-quantization),
  so the sub-step remainder of every layer's authorization stays
  in the pool as headroom for the next layer: total quantization
  drag is < 1 step per POSITION, not per layer (residual carry).
- The position cap is a hard ceiling: ``authorize`` refuses any
  layer that would push consumption over budget.

This module is pure: no broker calls, no journal writes.  The
runner owns submission and journaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal

#: Friction reserve fraction kept from every stop-tightening refund.
FRICTION_FRAC = Decimal("0.05")


def D(x) -> Decimal:
    """The ONLY Decimal constructor in the sizing path: via str(),
    so a float input contributes its decimal repr, not its binary
    expansion."""
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


@dataclass(frozen=True)
class VolumeConstraints:
    """Broker volume rules for one symbol (MT5 SYMBOL_VOLUME_*)."""

    min_volume: Decimal
    max_volume: Decimal
    step: Decimal
    limit: Decimal | None = None     # VOLUME_LIMIT (0/None = none)

    @classmethod
    def from_spec(cls, spec, *, limit=None) -> "VolumeConstraints":
        return cls(min_volume=D(spec.volume_min),
                   max_volume=D(spec.volume_max),
                   step=D(spec.volume_step),
                   limit=D(limit) if limit else None)


def quantize_down(raw: Decimal, step: Decimal) -> Decimal:
    """Floor ``raw`` to a whole number of steps (integer step-count
    arithmetic — exact in Decimal for any step)."""
    if step <= 0:
        raise ValueError(f"volume step must be positive, got {step}")
    steps = (raw / step).to_integral_value(rounding=ROUND_FLOOR)
    return steps * step


@dataclass
class SizingResult:
    verdict: str            # "ok" | "decline_below_min" |
    # "decline_volume_limit" | "decline_nonpositive"
    lots: Decimal | None    # quantized lots when verdict == "ok"
    raw_lots: Decimal       # pre-quantization size
    risk_money: Decimal     # authorized risk for this layer
    realized_risk: Decimal  # risk the quantized lots actually carry
    drag: Decimal           # authorized - realized (>= 0)

    @property
    def ok(self) -> bool:
        return self.verdict == "ok" and self.lots is not None


def size_for_risk(risk_money, stop_distance, spec, *,
                  constraints: VolumeConstraints | None = None,
                  open_volume=Decimal("0")) -> SizingResult:
    """Lots for one layer: risk_money / (stop ticks * tick value),
    step-quantized down, limit- and min-checked.

    ``spec`` is live/risk.py's SymbolSpec (floats) — converted via
    D() at the boundary; all arithmetic after that is Decimal.
    """
    cons = constraints or VolumeConstraints.from_spec(spec)
    risk_money = D(risk_money)
    stop_distance = D(stop_distance)
    tick_size = D(spec.tick_size)
    tick_value = D(spec.tick_value)
    zero = Decimal("0")
    if risk_money <= 0 or stop_distance <= 0 or tick_size <= 0 \
            or tick_value <= 0:
        return SizingResult("decline_nonpositive", None, zero,
                            risk_money, zero, zero)
    ticks = stop_distance / tick_size
    raw = risk_money / (ticks * tick_value)
    lots = quantize_down(raw, cons.step)
    if lots > cons.max_volume:
        lots = quantize_down(cons.max_volume, cons.step)
    if cons.limit is not None and cons.limit > 0:
        allowed = cons.limit - D(open_volume)
        if lots > allowed:
            lots = quantize_down(allowed, cons.step) \
                if allowed > 0 else zero
            if lots < cons.min_volume:
                realized = lots * ticks * tick_value
                return SizingResult("decline_volume_limit", None, raw,
                                    risk_money, realized,
                                    risk_money - realized)
    if lots < cons.min_volume:
        realized = lots * ticks * tick_value
        return SizingResult("decline_below_min", None, raw, risk_money,
                            realized, risk_money - realized)
    realized = lots * ticks * tick_value
    return SizingResult("ok", lots, raw, risk_money, realized,
                        risk_money - realized)


# ------------------------------------------------------------- RiskPool

@dataclass
class _PoolLayer:
    lots: Decimal
    entry: Decimal
    stop: Decimal
    direction: int
    tick_size: Decimal
    tick_value: Decimal
    accounted: Decimal      # consumption this layer currently books

    def raw_risk(self) -> Decimal:
        dist = (self.entry - self.stop) * D(self.direction)
        if dist <= 0:
            return Decimal("0")
        return self.lots * (dist / self.tick_size) * self.tick_value


class RiskPool:
    """Per-position risk ledger (see module docstring)."""

    def __init__(self, budget, *,
                 friction_frac: Decimal = FRICTION_FRAC) -> None:
        self.budget = D(budget)
        self.friction_frac = D(friction_frac)
        self._layers: dict[str, _PoolLayer] = {}

    # ------------------------------------------------ reads
    def consumption(self) -> Decimal:
        total = Decimal("0")
        for layer in self._layers.values():
            total += layer.accounted
        return max(Decimal("0"), total)

    def headroom(self) -> Decimal:
        return max(Decimal("0"), self.budget - self.consumption())

    def layer_ids(self) -> list[str]:
        return list(self._layers)

    def state(self) -> dict:
        """Status/journal shape (floats at the boundary only)."""
        return {"budget": float(self.budget),
                "consumed": float(self.consumption()),
                "headroom": float(self.headroom()),
                "layers": {k: {"lots": float(v.lots),
                               "entry": float(v.entry),
                               "stop": float(v.stop),
                               "accounted": float(v.accounted)}
                           for k, v in self._layers.items()}}

    # ------------------------------------------------ writes
    def authorize(self, layer_id: str, lots, entry, stop, direction,
                  spec) -> bool:
        """Book a layer if its realized risk fits the ceiling.
        Returns False (books nothing) when the position cap would
        be exceeded — the cap is a hard line (spec §4.4)."""
        layer = _PoolLayer(lots=D(lots), entry=D(entry), stop=D(stop),
                           direction=int(direction),
                           tick_size=D(spec.tick_size),
                           tick_value=D(spec.tick_value),
                           accounted=Decimal("0"))
        layer.accounted = layer.raw_risk()
        if self.consumption() + layer.accounted > self.budget:
            return False
        self._layers[str(layer_id)] = layer
        return True

    def move_stop(self, layer_id: str, new_stop) -> Decimal:
        """Roll a layer's stop; returns the consumption refunded
        (net of the friction reserve).  Tightening only — a stop
        that would WIDEN the layer's risk raises."""
        layer = self._layers[str(layer_id)]
        old = layer.accounted
        old_stop = layer.stop
        layer.stop = D(new_stop)
        new_risk = layer.raw_risk()
        if new_risk > old:
            layer.stop = old_stop
            raise ValueError(
                f"stop move widens risk for layer {layer_id}: "
                f"{old} -> {new_risk}")
        layer.accounted = new_risk + self.friction_frac * (old - new_risk)
        return old - layer.accounted

    def remove(self, layer_id: str) -> Decimal:
        """Close out a layer; returns the consumption released."""
        layer = self._layers.pop(str(layer_id), None)
        return layer.accounted if layer is not None else Decimal("0")

    def resize_stop_distance(self, layer_id: str) -> Decimal:
        layer = self._layers[str(layer_id)]
        return abs(layer.entry - layer.stop)
