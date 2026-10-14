"""Confluence Engine v2 — 4-pillar scoring with independence guarantees.

Pillars:
1. Structure (sr_flip, swing high/low, order block)
2. Fibonacci (50%, 61.8%, 78.6% retracement)
3. Volume Profile (POC, HVN, VAH/VAL)
4. FVG (Fair Value Gap — imbalance zones)

Scoring rule (the answer to redundant weighting):
- WITHIN a category: overlapping factors at the same location are the SAME
  story told twice. Cluster by proximity (dynamic ATR threshold), keep one.
- ACROSS categories: factors from DIFFERENT mechanisms (FVG displacement vs
  VP accumulation vs structural swing) are INDEPENDENT evidence. An FVG
  overlapping a VP POC is genuine confluence — count both.
- Exception: if two factors across categories share the SAME source bar,
  they're the same event described twice (e.g. a swing high that is also an
  FVG anchor). Merge into one factor, count once.

A zone needs at least 2 independent factors to be high-quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Factor:
    """A single confluence factor with provenance metadata."""
    category: str       # structure | fib | vp | fvg
    price: float        # the level price
    anchor_idx: int     # source bar index (-1 if multi-bar, e.g. VP)
    label: str          # e.g. "sr_flip", "0.618", "POC", "bullish_fvg"


@dataclass
class ConfluenceZone:
    """A price zone with independence-aware confluence scoring."""
    top: float
    bottom: float
    mid: float
    score: int                    # independent factors, 0-4
    factors: list[str] = field(default_factory=list)
    # e.g. ["structure:sr_flip", "fvg:bullish"]
    merged: list[str] = field(default_factory=list)
    # e.g. ["structure:swing_high merged into fvg:bearish (same bar)"]
    direction: int = 0            # +1 bullish zone, -1 bearish zone


def cluster_threshold(atr: float, atr_ratio: float) -> float:
    """Dynamic within-category clustering threshold.

    Wide ATR (expansion) → wider clusters (0.8 ATR).
    Tight ATR (compression) → tighter clusters (0.3 ATR).
    Normal → 0.5 ATR.
    """
    if atr_ratio > 1.5:
        return 0.8 * atr
    if atr_ratio < 0.7:
        return 0.3 * atr
    return 0.5 * atr


def dedupe_within_category(factors: list[Factor], atr: float,
                           atr_ratio: float = 1.0) -> list[Factor]:
    """Cluster same-category factors by proximity. Keep the freshest per cluster.

    Two FVGs at the same price = one story. Two VP levels far apart = two stories.
    """
    if not factors:
        return []
    thresh = cluster_threshold(atr, atr_ratio)
    sorted_f = sorted(factors, key=lambda f: f.price)
    clusters: list[list[Factor]] = [[sorted_f[0]]]
    for f in sorted_f[1:]:
        if abs(f.price - clusters[-1][-1].price) <= thresh:
            clusters[-1].append(f)
        else:
            clusters.append([f])
    # Keep freshest anchor per cluster (highest anchor_idx)
    return [max(c, key=lambda f: f.anchor_idx) for c in clusters]


def dedupe_cross_category(factors: list[Factor]) -> tuple[list[Factor], list[str]]:
    """Merge factors that share the same source bar.

    Same bar = same market event = one factor, not two.
    Factors with anchor_idx == -1 (multi-bar, e.g. VP) never merge.
    Returns (surviving factors, merge notes).
    """
    by_bar: dict[int, list[Factor]] = {}
    for f in factors:
        if f.anchor_idx >= 0:
            by_bar.setdefault(f.anchor_idx, []).append(f)
    merged_notes = []
    drop = set()
    # Priority when merging: fvg > structure > fib (most specific mechanism first)
    priority = {"fvg": 0, "structure": 1, "fib": 2, "vp": 3}
    for bar, group in by_bar.items():
        if len(group) > 1:
            group_sorted = sorted(group, key=lambda f: priority.get(f.category, 9))
            keeper = group_sorted[0]
            for other in group_sorted[1:]:
                drop.add(id(other))
                merged_notes.append(
                    f"{other.category}:{other.label} merged into "
                    f"{keeper.category}:{keeper.label} (same bar {bar})"
                )
                keeper.label = f"{keeper.label}+{other.label}"
    survivors = [f for f in factors if id(f) not in drop]
    return survivors, merged_notes


def score_confluence(
    price: float,
    structure: list[Factor] | None = None,
    fib: list[Factor] | None = None,
    vp: list[Factor] | None = None,
    fvg: list[Factor] | None = None,
    atr: float = 0.0,
    atr_ratio: float = 1.0,
    tol_atr: float = 0.3,
) -> ConfluenceZone:
    """Score confluence AT a price level.

    Only factors within tol_atr of `price` count — a swing high 200 points
    away is not confluence with this zone. This is the proximity gate the
    legacy score_zone had; the Factor API must not drop it.

    Each category's factors are deduplicated within-category first.
    Cross-category factors survive unless they share a source bar.
    """
    tol = tol_atr * atr if atr > 0 else 0
    all_factors: list[Factor] = []
    for cat_factors in (structure, fib, vp, fvg):
        if cat_factors:
            # Proximity gate FIRST: only factors near the evaluation price
            near = [f for f in cat_factors if abs(f.price - price) <= tol]
            all_factors.extend(
                dedupe_within_category(near, atr, atr_ratio)
            )

    survivors, merged_notes = dedupe_cross_category(all_factors)

    if not survivors:
        return ConfluenceZone(top=price + tol, bottom=price - tol,
                              mid=price, score=0)

    mid = price  # zone is centered on the evaluation price

    # Net direction: majority of directional factors
    dir_votes = sum(
        1 for f in survivors
        if (f.category == "fvg" and "bullish" in f.label)
        or (f.category == "structure" and "support" in f.label)
    ) - sum(
        1 for f in survivors
        if (f.category == "fvg" and "bearish" in f.label)
        or (f.category == "structure" and "resistance" in f.label)
    )
    direction = 1 if dir_votes > 0 else (-1 if dir_votes < 0 else 0)

    return ConfluenceZone(
        top=mid + tol,
        bottom=mid - tol,
        mid=mid,
        score=len(survivors),
        factors=[f"{f.category}:{f.label}" for f in survivors],
        merged=merged_notes,
        direction=direction,
    )


def is_high_quality(zone: ConfluenceZone, min_factors: int = 2) -> bool:
    """Check if a zone meets the minimum confluence threshold."""
    return zone.score >= min_factors


# ---------------------------------------------------------------------------
# Legacy single-call API (kept for backward compat with existing scripts)
# ---------------------------------------------------------------------------

def score_zone(price: float, atr: float,
               structure_levels: list[tuple[float, str]] | None = None,
               fib_levels: list[tuple[float, float]] | None = None,
               vp_levels: list[str] | None = None,
               tol_atr: float = 0.3) -> ConfluenceZone:
    """Legacy scoring without provenance metadata.

    NOTE: without anchor indices, cross-category same-bar merging cannot
    apply. Prefer score_confluence with Factor objects for live use.
    """
    factors = []
    tol = tol_atr * atr if atr > 0 else 0

    if structure_levels:
        for lp, label in structure_levels:
            if abs(lp - price) <= tol:
                factors.append(f"structure:{label}")
                break

    if fib_levels:
        for lp, ratio in fib_levels:
            if abs(lp - price) <= tol:
                factors.append(f"fib:{ratio}")
                break

    if vp_levels:
        for vpl in vp_levels:
            factors.append(f"vp:{vpl}")
            break

    return ConfluenceZone(
        top=price + tol,
        bottom=price - tol,
        mid=price,
        score=len(factors),
        factors=factors,
    )
