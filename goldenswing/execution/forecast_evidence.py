"""Forecast → Judgment evidence feed (E3).

The ensemble's live read enters the judgment core the way every
other evidence family does: as ONE correlation group
(``forecast_ensemble``) with one weighed vote, counted by
``judgment._evidence_balance`` under the group's stated weight.
Forecasts are evidence (owner forecast doctrine, 2026-11-02) —
never sizing, never a gate, never a veto: this module can only
add or withhold a vote, and the core's hard safeguards live
outside the evidence path entirely.

Three honesty rules, mirroring the evidence_map doctrine:

1. **Absent stays absent.**  No bundle, no scored bar, a stale
   bar, or unscored ranks → no group, no vote, and the journaled
   block says which.  Nothing is synthesized to fill the gap.
2. **Recorded before weighed.**  The feed block is journaled on
   every ``judgment_shadow`` record verbatim, whether or not it
   voted.  Until the ensemble's live calibration (E2's book)
   has EARNED trust, the block is recorded with no decision
   weight — the census doctrine applied to a decision organ.
3. **Trust is measured, not assumed.**  The vote weighs only
   when the live calibration book shows the ensemble's extreme
   ranks (bands 80-90 and 90+, pooled, at the feed horizon)
   resolving at >= ``TRUST_LIFT`` × the base rate over at least
   ``TRUST_MIN_N`` resolutions for the candidate's direction.
   If those extreme ranks resolve WORSE than base
   (lift < ``ANTI_LIFT``), the feed is anti-calibrated: it
   abstains and the block says so — a forecaster that is wrong
   at its most confident gets no vote until it re-earns one.

Stance (stated, deterministic): at the feed horizon, if the
candidate-direction rank is >= 65 and leads the opposite rank
by >= 10, the ensemble supports; mirrored for contradiction;
anything else is neutral (recorded, no vote).  Magnitude alone
never votes a direction — a big move expected the OTHER way is
the contradiction branch, and a big move with no lean is
weather, not evidence.
"""
from __future__ import annotations

import math
from typing import Any

GROUP = "forecast_ensemble"

STANCE_MIN_RANK = 65.0
STANCE_MIN_LEAD = 10.0
TRUST_MIN_N = 20
TRUST_LIFT = 1.25
ANTI_LIFT = 0.90
TRUST_BANDS = ("80-90", "90+")

#: Candidate clock → feed horizon (H1 bars).  Shorter clocks
#: read the shortest forecast window; swing clocks read deeper.
HORIZON_BY_TF = {"M5": 12, "M15": 12, "M30": 24, "H1": 24,
                 "H4": 72, "D1": 72}
DEFAULT_HORIZON = 24


def horizon_for_tf(intended_tf: str | None) -> int:
    return HORIZON_BY_TF.get(str(intended_tf or "").upper(),
                             DEFAULT_HORIZON)


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def trust_state(book_view: dict | None, horizon: int,
                direction_word: str) -> dict:
    """The E3 trust evaluation as DATA (completion-04) for one
    horizon + direction: the verdict ``trust_for`` returns, the
    pooled top-band evidence behind it (n per trust band, the
    pooled rate, the base rate, the lift), and the gates.  One
    computation, two readers — the judgment feed's per-candidate
    verdict and the status surface's standing answer to "why
    isn't the forecast weighing in"."""
    state: dict[str, Any] = {
        "state": "unproven", "horizon": int(horizon),
        "direction": direction_word, "base_rate": None,
        "n_pool": 0, "pool_rate": None, "lift": None,
        "bands": [], "min_n": TRUST_MIN_N,
        "trust_lift": TRUST_LIFT, "anti_lift": ANTI_LIFT}
    if not book_view:
        return state
    entry = ((book_view.get("horizons") or {})
             .get(str(horizon)) or {})
    side = entry.get(direction_word) or {}
    base_rate = side.get("base_rate")
    state["base_rate"] = base_rate
    n_pool = 0
    hits_pool = 0.0
    for band in side.get("bands") or []:
        if band.get("band") in TRUST_BANDS:
            n = int(band.get("n") or 0)
            n_pool += n
            hits_pool += float(band.get("rate") or 0.0) * n
            state["bands"].append({
                "band": band.get("band"), "n": n,
                "rate": band.get("rate"),
                "lift": band.get("lift")})
    state["n_pool"] = n_pool
    if base_rate is None or not base_rate or not n_pool:
        return state
    state["pool_rate"] = round(hits_pool / n_pool, 4)
    lift = (hits_pool / n_pool) / base_rate
    state["lift"] = round(lift, 4)
    if n_pool < TRUST_MIN_N:
        return state
    # The verdict compares the RAW lift (the rounded value is
    # display only) — trust_for's historical boundary behavior
    # is unchanged by this refactor.
    if lift >= TRUST_LIFT:
        state["state"] = "trusted"
    elif lift < ANTI_LIFT:
        state["state"] = "anti_calibrated"
    return state


def trust_for(book_view: dict | None, horizon: int,
              direction_word: str) -> str:
    """'trusted' | 'anti_calibrated' | 'unproven' from the E2
    calibration book's view for one horizon + direction."""
    return str(trust_state(book_view, horizon,
                           direction_word)["state"])


def render_trust_lines(book_view: dict | None) -> list[str]:
    """The /status E3 trust section (completion-04): per horizon
    and direction with any book data, the verdict with its
    evidence — pooled top-band resolutions against the gate,
    the lift against the trust/anti lines, and the per-band n.
    A horizon whose bands are still 0–2 deep reads exactly that
    here, instead of the answer living inside individual shadow
    records.  Empty when the book has nothing at all."""
    if not book_view:
        return []
    horizons = book_view.get("horizons") or {}
    lines: list[str] = []
    for key in sorted(horizons, key=lambda k: int(k)):
        for word in ("up", "dn"):
            st = trust_state(book_view, int(key), word)
            side = (horizons.get(key) or {}).get(word) or {}
            if not st["bands"] and not side.get("n"):
                continue
            band_bits = " · ".join(
                f"{b['band']} n={b['n']}" for b in st["bands"])
            lift_bit = (f"lift {st['lift']:.2f} "
                        f"(trust ≥{TRUST_LIFT}, anti <{ANTI_LIFT})"
                        if st["lift"] is not None
                        else "lift — (no base rate)")
            lines.append(
                f"    {key}h {word}: {st['state']} — top-band "
                f"{st['n_pool']}/{TRUST_MIN_N} resolutions · "
                f"{lift_bit}"
                + (f" · {band_bits}" if band_bits else ""))
    if not lines:
        return []
    return ["  forecast trust (E3 — weighed only when trusted):",
            *lines]


def forecast_block(*, ranks: Any, direction: int,
                   horizon: int, fresh: bool,
                   book_view: dict | None = None) -> dict:
    """The journaled feed block for one candidate.

    ``ranks`` is a ForecastRanks (``.rank(h, word)`` /
    ``.bar_time``) or None.  The block is always returned —
    absence is a stated fact inside it, never a missing key."""
    block: dict[str, Any] = {
        "group": GROUP, "horizon": horizon,
        "bar_time": None, "dir_rank": None, "opp_rank": None,
        "move_pressure": None, "lean": None,
        "stance": "absent", "trust": "unproven",
        "weighed": False, "text": None,
        "fresh": bool(fresh),
    }
    if ranks is None or direction not in (1, -1):
        return block
    word = "up" if direction == 1 else "dn"
    opp = "dn" if direction == 1 else "up"
    dir_rank = _f(ranks.rank(horizon, word))
    opp_rank = _f(ranks.rank(horizon, opp))
    block["bar_time"] = str(getattr(ranks, "bar_time", "") or "") \
        or None
    block["dir_rank"] = dir_rank
    block["opp_rank"] = opp_rank
    if dir_rank is None and opp_rank is None:
        return block
    vals = [v for v in (dir_rank, opp_rank) if v is not None]
    block["move_pressure"] = max(vals)
    if dir_rank is not None and opp_rank is not None:
        block["lean"] = round(dir_rank - opp_rank, 2)
    stance = "neutral"
    if dir_rank is not None and opp_rank is not None:
        if dir_rank >= STANCE_MIN_RANK \
                and dir_rank - opp_rank >= STANCE_MIN_LEAD:
            stance = "supporting"
        elif opp_rank >= STANCE_MIN_RANK \
                and opp_rank - dir_rank >= STANCE_MIN_LEAD:
            stance = "contradictory"
    block["stance"] = stance
    block["trust"] = trust_for(book_view, horizon, word)
    if stance in ("supporting", "contradictory") and fresh \
            and block["trust"] == "trusted":
        block["weighed"] = True
        block["text"] = (
            f"forecast {word}@{horizon}h rank {dir_rank:.0f} "
            f"vs {opp} {opp_rank:.0f} "
            f"(pressure {block['move_pressure']:.0f}, "
            f"trust {block['trust']})")
    return block


def merge_into_evidence(evidence: dict | None,
                        block: dict) -> dict:
    """A copy of ``evidence`` with the forecast group merged in
    when the block weighed.  Unweighed blocks (recorded-only,
    neutral, stale, absent) return the evidence unchanged — the
    funnel record's own evidence is never mutated either way."""
    evidence = dict(evidence or {})
    if not block.get("weighed") or not block.get("text"):
        return evidence
    text = block["text"]
    groups = dict(evidence.get("evidence_groups") or {})
    groups[GROUP] = [text]
    evidence["evidence_groups"] = groups
    side = "supporting" if block["stance"] == "supporting" \
        else "contradictory"
    items = list(evidence.get(side) or [])
    if text not in items:
        items.append(text)
    evidence[side] = items
    return evidence
