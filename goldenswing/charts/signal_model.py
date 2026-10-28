"""Shared signal-view truth: the ONE model behind the PNG chart and
the HTML page.

Both renderers take the same inputs — the decision score recorded
for the signal (never the sleeve-base placeholder that lives on the
signal object), the structural invalidation AND the broker stop when
one is known — and both derive the same panel roles, the same
effective R, and the same label de-collision from this module. The
renderers differ only in medium (pixels vs DOM); they never differ
in what the trade was.

Honesty rules, enforced here so neither renderer can drift:
- a missing decision score renders as "Score unavailable", never as
  a placeholder number;
- a missing broker stop renders as "pending at placement", never as
  the structural stop relabeled;
- a panel's role comes from the placing route + entry clock, never
  from a hardcoded "H1 is the trigger" label.
"""

from __future__ import annotations

_NATIVE_PLAYBOOKS = ("qm_m5", "qm_d1")


def panel_role(tf: str, *, playbook: str, entry_clock: str) -> str:
    """Role of one timeframe panel for this signal's route.

    The entry clock is the entry panel. On native routes (qm_m5 /
    qm_d1) H1 is context, not the trigger — the trigger lives on the
    entry clock. On H1-roster routes H1 is the trigger. H4/D1 are
    structure on every route; anything else is context.
    """
    if tf == entry_clock:
        return "entry"
    if playbook in _NATIVE_PLAYBOOKS:
        return "structure" if tf in ("H4", "D1") else "context"
    if tf == "H1":
        return "trigger"
    if tf in ("H4", "D1"):
        return "structure"
    return "context"


def score_text(decision_score: float | None, *,
               fallback: float | None = None) -> str:
    """Score line for a page: the decision score, or the honest gap."""
    v = decision_score if decision_score is not None else fallback
    if v is None:
        return "Score unavailable"
    return f"Score {v:.0f}"


def effective_r(*, entry: float, broker_stop: float | None,
                targets) -> float | None:
    """Reward/risk to the first target on the BROKER stop.

    None when no broker stop is known yet or no target exists — the
    effective R cannot be stated before the broker stop exists, and
    the structural distance is not a substitute for it.
    """
    if broker_stop is None or not targets:
        return None
    risk = abs(entry - broker_stop)
    if risk <= 0:
        return None
    return abs(float(targets[0]) - entry) / risk


def decollide_label_ys(values_px: list[float], *, min_gap: float,
                       top: float | None = None,
                       bottom: float | None = None) -> list[float]:
    """Push stacked label positions apart to at least ``min_gap`` px.

    Positions are pixel coordinates on one axis (either direction
    works — the function only preserves order and enforces gaps).
    Returns positions in the ORIGINAL order. A forward pass pushes
    each label above the previous one; if ``top`` is given and the
    stack overflows it, a backward pass pulls the stack down from
    the top; if ``bottom`` is then violated, a final forward pass
    re-anchors at the bottom (the stack only overflows both bounds
    when more labels than space exist, which the callers avoid by
    capping targets).
    """
    n = len(values_px)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values_px[i])
    adj = [float(v) for v in values_px]
    for k in range(1, n):
        prev, cur = order[k - 1], order[k]
        if adj[cur] < adj[prev] + min_gap:
            adj[cur] = adj[prev] + min_gap
    if top is not None and adj[order[-1]] > top:
        adj[order[-1]] = top
        for k in range(n - 2, -1, -1):
            cur, nxt = order[k], order[k + 1]
            if adj[cur] > adj[nxt] - min_gap:
                adj[cur] = adj[nxt] - min_gap
    if bottom is not None and adj[order[0]] < bottom:
        adj[order[0]] = bottom
        for k in range(1, n):
            prev, cur = order[k - 1], order[k]
            if adj[cur] < adj[prev] + min_gap:
                adj[cur] = adj[prev] + min_gap
    return adj
