"""MarketSituation as a governing input (Phase 1, item 1).

Pass 0 built the situation as a measurement and stamped only its id
onto funnel records — no decision path read it.  Phase 1 promotes it
to an input of the governing decision: the runner's judgment
evaluation reads the situation's per-timeframe states into its
market-state picture, and the funnel record names what the situation
contributed.  The situation casts NO vote on the verdict (owner
correction 2026-11-24): its direction alignment is evidence on the
record, never a penalty for a setup that fights the daily picture.

This module is the pure shape of that input — no runner, no keeper,
no I/O — so the core and the tests can build it from a snapshot dict
alone:

- ``situation_contribution(snapshot, intended_tf, direction)`` reads
  the snapshot's ``per_tf`` states (the live/situation.py schema:
  ``regime_label`` / ``structure_phase`` / ``direction_word`` /
  ``regime_confidence`` / ``volatility.atr_state`` per TF) plus the
  direction alignment of those states against the candidate's
  direction.  None when no snapshot exists (or it carries no TF
  states at all) — an absent situation is an absent input, never
  an assumed one.
- The alignment is ONE net vote for the whole situation (the sign
  of the sum of the per-TF direction votes), never one vote per TF:
  the per-TF directions are one measurement of the same market, and
  counting each TF separately would stack five confirmations of one
  read.  ``d1_opposed`` names the D1 direction opposing the
  candidate.  Both ride the record as evidence — the core reads the
  block into its market-state picture and casts no vote from it.
- ``direction_word`` is the bedrock vocabulary (live/regime.py):
  "up" / "down" / "flat".  "flat" (and anything absent or unknown)
  abstains — it is not a vote for either side.
"""
from __future__ import annotations

#: Direction words the situation engine emits (live/regime.py),
#: mapped to the judgment core's +/-1 direction vocabulary.
_DIRECTION_VOTES = {"up": 1, "down": -1}


def _direction_vote(word) -> int | None:
    """+1 / -1 for a direction word, None when it abstains
    ("flat", absent, or anything unrecognised)."""
    if isinstance(word, str):
        return _DIRECTION_VOTES.get(word)
    return None


def situation_contribution(snapshot, *, intended_tf,
                           direction) -> dict | None:
    """The situation's contribution to one evaluation.

    ``snapshot`` is the MarketSituation snapshot dict
    (``per_tf`` + ``situation_id``); ``intended_tf`` the
    evaluation's timeframe; ``direction`` the candidate's
    direction (+1 / -1) or None when the evaluation carries no
    candidate.  Returns None when there is no snapshot (or no
    TF states in it) — the caller then attaches nothing and the
    record stays byte-identical to its pre-Phase-1 shape.
    """
    if not isinstance(snapshot, dict):
        return None
    per_tf = snapshot.get("per_tf") or {}
    if not isinstance(per_tf, dict) or not per_tf:
        return None
    reads: dict[str, dict] = {}
    votes: dict[str, int] = {}
    for tf in sorted(per_tf):
        state = per_tf.get(tf) or {}
        if not isinstance(state, dict):
            continue
        vol = state.get("volatility") or {}
        if not isinstance(vol, dict):
            vol = {}
        reads[tf] = {
            "regime_label": state.get("regime_label"),
            "structure_phase": state.get("structure_phase"),
            "direction_word": state.get("direction_word"),
            "regime_confidence": state.get("regime_confidence"),
            "volatility_state": vol.get("atr_state"),
        }
        if direction in (1, -1):
            vote = _direction_vote(state.get("direction_word"))
            if vote is not None:
                votes[tf] = 1 if vote == direction else -1
    if not reads:
        return None
    total = sum(votes.values())
    net = 0 if total == 0 else (1 if total > 0 else -1)
    return {
        "situation_id": snapshot.get("situation_id"),
        "intended_tf": intended_tf,
        "intended": reads.get(intended_tf) or {},
        "per_tf": reads,
        "alignment": {
            "per_tf": votes,
            "net": net,
            "d1_opposed": votes.get("D1") == -1,
        },
    }
