"""Auditable, correlation-aware evidence aggregation for live strategies.

This module is deliberately a *measurement* layer, not an execution gate.
Evidence is timestamped and attributable; absent information is neutral; and
multiple labels derived from the same market event can contribute at most once.
The initial weights are caller-supplied research weights and must be validated
out of sample before they are promoted to execution authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping
import math

VALID_STANCES = frozenset({"supports", "contradicts", "neutral"})

# One common evidence budget shared by the existing-playbook audit and the
# separate Hot Scalp research sleeve. These are provisional research weights,
# not calibrated probabilities and never an execution gate.
BASE_CATEGORY_WEIGHTS = {
    "higher_context": 15.0,
    "structure": 18.0,
    "imbalance": 10.0,
    "location": 9.0,
    "reaction": 22.0,
    "activity": 5.0,
    "timing": 5.0,
    "execution_cost": 10.0,
    "momentum": 6.0,
}


@dataclass(frozen=True)
class EvidenceItem:
    key: str
    label: str
    source: str
    observed_at: str
    timeframe: str
    category: str
    correlation_group: str
    stance: str
    strength: float = 1.0
    available: bool = True
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("evidence key must be non-empty")
        if self.stance not in VALID_STANCES:
            raise ValueError(f"invalid evidence stance: {self.stance}")
        if not self.category.strip():
            raise ValueError("evidence category must be non-empty")
        if not math.isfinite(float(self.strength)):
            raise ValueError("evidence strength must be finite")
        object.__setattr__(self, "strength", max(0.0, min(1.0,
                                                          float(self.strength))))

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "source": self.source,
            "observed_at": self.observed_at, "timeframe": self.timeframe,
            "category": self.category,
            "correlation_group": self.correlation_group or self.key,
            "stance": self.stance, "strength": round(self.strength, 4),
            "available": bool(self.available), "detail": self.detail,
        }


@dataclass
class EvidenceScore:
    score: float
    support_points: float
    contradiction_points: float
    net_points: float
    category_points: dict[str, float] = field(default_factory=dict)
    counted: list[dict] = field(default_factory=list)
    neutral_or_missing: list[dict] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    method: str = "correlation_capped_research_v1"

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "support_points": round(self.support_points, 2),
            "contradiction_points": round(self.contradiction_points, 2),
            "net_points": round(self.net_points, 2),
            "category_points": {k: round(v, 2)
                                for k, v in self.category_points.items()},
            "counted": list(self.counted),
            "neutral_or_missing": list(self.neutral_or_missing),
            "duplicates": list(self.duplicates), "method": self.method,
        }


def score_evidence(items: Iterable[EvidenceItem],
                   category_weights: Mapping[str, float]) -> EvidenceScore:
    """Score an evidence set on a neutral-50 scale, capped to [0, 100].

    ``category_weights`` is a fixed budget whose sum should be 100. Each
    category can contribute at most its weight, regardless of how many
    correlated detectors fire. A missing/unknown item contributes zero; it
    never becomes support by default. Within each correlation group, only the
    strongest item is counted. Equal-strength opposing readings cancel to
    neutral rather than choosing a convenient side.

    This score is intended for ranking and inspection only until weights are
    calibrated on chronological, cost-aware held-out data.
    """
    weights = {str(k): float(v) for k, v in category_weights.items()}
    if not weights or any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("category weights must be finite, non-negative and non-empty")
    if not math.isclose(sum(weights.values()), 100.0, abs_tol=1e-6):
        raise ValueError("category weights must sum to 100")

    unique: dict[str, EvidenceItem] = {}
    duplicates: list[str] = []
    for item in items:
        if item.key in unique:
            duplicates.append(item.key)
            continue
        unique[item.key] = item

    valid: list[tuple[EvidenceItem, float]] = []
    neutral: list[dict] = []
    for item in unique.values():
        cap = weights.get(item.category, 0.0)
        if not item.available or item.stance == "neutral" or cap <= 0:
            neutral.append({**item.to_dict(), "why_neutral": (
                "unavailable" if not item.available else
                "neutral stance" if item.stance == "neutral" else
                "category has no scoring budget")})
            continue
        sign = 1.0 if item.stance == "supports" else -1.0
        valid.append((item, sign * cap * item.strength))

    groups: dict[str, list[tuple[EvidenceItem, float]]] = {}
    for item, pts in valid:
        group = item.correlation_group.strip() or item.key
        groups.setdefault(group, []).append((item, pts))

    # One contribution per independent event. Opposing ties cancel, and the
    # choice is deterministic so two processes serialize identical evidence.
    picked: list[tuple[EvidenceItem, float]] = []
    for group, candidates in sorted(groups.items()):
        max_abs = max(abs(p) for _, p in candidates)
        strongest = [(i, p) for i, p in candidates
                     if math.isclose(abs(p), max_abs, abs_tol=1e-9)]
        signs = {1 if p > 0 else -1 for _, p in strongest}
        if len(signs) > 1:
            neutral.append({"correlation_group": group,
                            "keys": sorted(i.key for i, _ in strongest),
                            "why_neutral": "equal-strength conflicting evidence in one event"})
            continue
        picked.append(sorted(strongest, key=lambda pair: pair[0].key)[0])

    raw_by_cat: dict[str, float] = {k: 0.0 for k in weights}
    selected: list[dict] = []
    for item, pts in picked:
        raw_by_cat[item.category] = raw_by_cat.get(item.category, 0.0) + pts
        selected.append({**item.to_dict(), "points_before_category_cap": round(pts, 3)})
    category_points = {cat: max(-weights[cat], min(weights[cat], val))
                       for cat, val in raw_by_cat.items()}
    support = sum(max(0.0, val) for val in category_points.values())
    contradiction = sum(max(0.0, -val) for val in category_points.values())
    net = sum(category_points.values())
    # Fixed total budget makes absent evidence neutral rather than inflating
    # the score through a smaller denominator.
    score = max(0.0, min(100.0, 50.0 + net / 2.0))
    return EvidenceScore(score, support, contradiction, net,
                         category_points, selected, neutral, duplicates)
