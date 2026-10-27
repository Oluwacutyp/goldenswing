"""Risk-aware stacking policy for Hot Scalp research.

This module decides whether a *proposed additional layer* is structurally
eligible. It does not place orders. Risk values supplied here must be based on
normalized broker stops (not a tighter internal structural stop), and unknown
risk fails closed.

Supported modes:
- pyramid: add only after the parent position has moved favourably by a
  configured amount and a fresh, independent trigger appears.
- same_setup: permit a second/third independently triggered entry under the
  same parent thesis, but never add while the parent is losing.

Both modes share the parent setup's total risk budget. A child trigger ID is
idempotent and distinct from the parent ID; repeated processing cannot create a
new identity for the same trigger.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any


@dataclass(frozen=True)
class StackingConfig:
    max_layers: int = 3
    pyramid_min_favourable_r: float = 0.50
    same_setup_min_parent_r: float = 0.0

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_layers) <= 3:
            raise ValueError("max_layers must be between 1 and 3")
        if self.pyramid_min_favourable_r < 0:
            raise ValueError("pyramid threshold must be non-negative")
        if self.same_setup_min_parent_r < 0:
            raise ValueError("same-setup threshold must not permit adding into a loss")


def child_entry_id(parent_setup_id: str, mode: str,
                   trigger_id: str) -> str:
    """Stable child identity; never depends on scan time or process ID."""
    if not str(parent_setup_id).strip() or not str(trigger_id).strip():
        raise ValueError("parent_setup_id and trigger_id are required")
    if mode not in {"pyramid", "same_setup"}:
        raise ValueError("mode must be 'pyramid' or 'same_setup'")
    digest = sha256(f"{parent_setup_id}|{mode}|{trigger_id}".encode()).hexdigest()[:28]
    return f"hl-{digest}"


def evaluate_layer_add(*, parent_setup_id: str, mode: str,
                       trigger_id: str, trigger_direction: int,
                       parent_direction: int, parent_open: bool,
                       parent_thesis_valid: bool, trigger_fresh: bool,
                       trigger_independent: bool, trigger_already_used: bool,
                       parent_unrealized_r: float | None,
                       active_layers: int, max_layers: int,
                       risk_budget_money: float | None,
                       existing_worst_case_risk_money: float | None,
                       proposed_worst_case_risk_money: float | None,
                       broker_stop_verified: bool,
                       config: StackingConfig | None = None) -> dict[str, Any]:
    """Return a reason-coded layer decision; never authorizes an order.

    `existing_worst_case_risk_money` and `proposed_worst_case_risk_money`
    must use broker-normalized stop prices and correct contract/volume specs.
    `risk_budget_money` is the parent setup's shared total risk budget.
    """
    cfg = config or StackingConfig(max_layers=max_layers)
    mode = str(mode).strip().lower()
    stable_child_id = child_entry_id(parent_setup_id, mode, trigger_id)

    def no(code: str, reason: str) -> dict[str, Any]:
        return {"eligible": False, "reason_code": code, "reason": reason,
                "parent_setup_id": parent_setup_id,
                "child_entry_id": stable_child_id, "mode": mode,
                "execution_authority": "research_policy_only"}

    if mode not in {"pyramid", "same_setup"}:
        return no("stack.mode_unknown", "unsupported stacking mode")
    if not parent_open:
        return no("stack.parent_not_open", "additional layers require an open parent position")
    if not parent_thesis_valid:
        return no("stack.parent_invalid", "parent setup is invalid or expired")
    if parent_direction not in (-1, 1) or trigger_direction != parent_direction:
        return no("stack.direction_mismatch", "layer direction does not match parent thesis")
    if not trigger_fresh:
        return no("stack.trigger_stale", "layer trigger is not fresh")
    if not trigger_independent:
        return no("stack.correlated_trigger", "trigger duplicates evidence already used by the parent")
    if trigger_already_used:
        return no("stack.trigger_duplicate", "this child trigger has already been consumed")
    if active_layers >= min(int(max_layers), cfg.max_layers):
        return no("stack.layer_limit", "configured maximum layer count reached")
    if active_layers < 1:
        return no("stack.layer_state_invalid", "parent layer count is inconsistent")
    if not broker_stop_verified:
        return no("stack.stop_unverified", "broker-normalized protective stop is not verified")
    if parent_unrealized_r is None or not math.isfinite(float(parent_unrealized_r)):
        return no("stack.parent_pnl_unknown", "parent mark-to-market in R is unavailable")
    threshold = (cfg.pyramid_min_favourable_r if mode == "pyramid"
                 else cfg.same_setup_min_parent_r)
    if float(parent_unrealized_r) < threshold:
        code = "stack.not_favourable_enough" if mode == "pyramid" else "stack.averaging_down_refused"
        reason = (f"pyramid requires parent >= {threshold:.2f}R favourable progress"
                  if mode == "pyramid" else
                  "same-setup adds are refused while the parent position is losing")
        return no(code, reason)

    risk_values = (risk_budget_money, existing_worst_case_risk_money,
                   proposed_worst_case_risk_money)
    if any(v is None or not math.isfinite(float(v)) for v in risk_values):
        return no("stack.risk_unknown", "shared budget or normalized stop-risk is unknown")
    budget, existing, proposed = (float(v) for v in risk_values)
    if budget <= 0 or existing < 0 or proposed <= 0:
        return no("stack.risk_invalid", "risk budget must be positive and layer risks valid")
    combined = existing + proposed
    if combined > budget + max(1e-8, budget * 1e-9):
        return no("stack.parent_budget_exceeded",
                  f"combined worst-case risk {combined:.2f} exceeds shared budget {budget:.2f}")

    return {
        "eligible": True, "reason_code": "stack.layer_eligible",
        "reason": "fresh independent same-direction trigger fits parent risk budget",
        "parent_setup_id": parent_setup_id, "child_entry_id": stable_child_id,
        "mode": mode, "active_layers_before": int(active_layers),
        "active_layers_after": int(active_layers) + 1,
        "max_layers": min(int(max_layers), cfg.max_layers),
        "existing_worst_case_risk_money": round(existing, 2),
        "proposed_worst_case_risk_money": round(proposed, 2),
        "combined_worst_case_risk_money": round(combined, 2),
        "shared_risk_budget_money": round(budget, 2),
        "execution_authority": "research_policy_only",
    }
